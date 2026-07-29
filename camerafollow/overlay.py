"""Director view: draw what the tracker is thinking onto a frame.

Not decoration -- this is the tuning instrument. Nearly every "it moves badly"
problem is legible here in a few seconds:

* box flickering between people      -> subject policy (stickiness, min_height_frac)
* reticle far from the subject       -> framing (target_y, subject_anchor)
* dead-zone box constantly straddled -> deadzone too small, or lead_time wrong
* velocity bar pinned at the limit   -> max_velocity too low for the room
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Box, FramingError, Rect

_WHITE = (255, 255, 255)
_GREY = (150, 150, 150)
_GREEN = (80, 220, 120)
_AMBER = (60, 190, 250)
_RED = (70, 70, 235)
_BLUE = (235, 180, 80)
_ZONE = (120, 200, 120)
_ZONE_BAD = (90, 90, 220)


def draw_director(
    frame,
    *,
    tracks,
    subject,
    view: Rect,
    error: FramingError | None,
    predicted: Box | None,
    deadzone: tuple[float, float],
    stats: dict,
    target_point: tuple[float, float] | None,
    zones=None,
    key_hints: bool = False,
):
    """Return a copy of ``frame`` annotated with the full tracker state."""
    img = frame.copy()

    if zones is not None and zones.active:
        _draw_zones(img, zones)

    # every tracked person
    for t in tracks:
        if not t.confirmed:
            continue
        is_subject = subject is not None and t.id == subject.id
        colour = _GREEN if is_subject else _GREY
        thickness = 2 if is_subject else 1
        x1, y1, x2, y2 = t.box.as_int()
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
        tag = f"#{t.id}"
        if t.misses:
            tag += f" (coasting {t.misses})"
        cv2.putText(img, tag, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        if is_subject:
            cv2.circle(img, (int(t.box.cx), int(t.box.y2)), 4, colour, -1, cv2.LINE_AA)

    # where we believe the subject will be when the motor gets there
    if predicted is not None:
        px1, py1, px2, py2 = predicted.as_int()
        cv2.rectangle(img, (px1, py1), (px2, py2), _AMBER, 1, cv2.LINE_AA)
        cv2.putText(img, "predicted", (px1, py2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _AMBER, 1)

    # the output window (meaningful for digital PTZ)
    vx, vy = int(view.x), int(view.y)
    vw, vh = int(view.w), int(view.h)
    cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), _BLUE, 2)

    # framing target + dead zone, drawn in view space
    if target_point is not None:
        tx, ty = int(target_point[0]), int(target_point[1])
        cv2.drawMarker(img, (tx, ty), _WHITE, cv2.MARKER_CROSS, 22, 1)
        dzx = int(deadzone[0] * view.w / 2)
        dzy = int(deadzone[1] * view.h / 2)
        cv2.rectangle(img, (tx - dzx, ty - dzy), (tx + dzx, ty + dzy), _WHITE, 1)

    _draw_hud(img, stats, error, key_hints)
    return img


def _draw_zones(img, zones) -> None:
    """Show the configured tracking area, so a mis-typed zone is obvious."""
    include, excludes = zones.polygons(img.shape)
    if include is not None:
        cv2.polylines(img, [include], True, _ZONE, 2, cv2.LINE_AA)
        cv2.putText(
            img,
            "tracking zone",
            tuple(include[0] + np.array([6, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _ZONE,
            1,
            cv2.LINE_AA,
        )
    for poly in excludes:
        cv2.polylines(img, [poly], True, _ZONE_BAD, 2, cv2.LINE_AA)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], _ZONE_BAD)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)


def _draw_hud(img, stats: dict, error: FramingError | None, key_hints: bool) -> None:
    h, w = img.shape[:2]
    lines = [
        f"control {stats.get('control_fps', 0):5.1f} fps   detect {stats.get('detect_fps', 0):5.1f} fps",
        f"subject {stats.get('subject') or '-'}    people {stats.get('people', 0)}",
        f"pan {stats.get('pan', 0.0):+.2f}   tilt {stats.get('tilt', 0.0):+.2f}   zoom {stats.get('zoom', 0.0):+.2f}",
    ]
    if error is not None:
        lines.append(f"err  x {error.x:+.2f}   y {error.y:+.2f}")
    if stats.get("locked"):
        lines.append("MANUAL LOCK")
    if stats.get("connected") is False:
        lines.append("SOURCE DISCONNECTED")

    pad = 8
    box_h = 18 * len(lines) + pad * 2
    box_w = 330
    overlay = img[0:box_h, 0:box_w].copy()
    cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img[0:box_h, 0:box_w], 0.65, 0, img[0:box_h, 0:box_w])
    for i, line in enumerate(lines):
        colour = _RED if line.startswith(("MANUAL", "SOURCE")) else _WHITE
        cv2.putText(
            img,
            line,
            (pad, pad + 14 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )

    # Only for the local preview window; the same frame is served to browsers,
    # where these keys do nothing.
    if key_hints:
        hint = "q quit   l lock   u unlock   h home   o overlay"
        cv2.putText(img, hint, (pad, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)
