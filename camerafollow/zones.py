"""Tracking zones: where in the scene a subject is allowed to be.

Without this, every person the detector sees is a candidate -- an audience
member standing up, someone crossing a corridor behind a lectern, a coach on
the sideline, a passer-by at a window. Zones are how commercial systems solve
it, and they are the difference between a tracker that works in a real room and
one that only works in an empty one.

Coordinates are normalised ``0..1`` against the **source frame**, so a zone
describes a physical part of the scene and stays correct no matter how the
output is cropped or what resolution the camera is set to.

Two shapes are accepted, because a rectangle covers most cases and a polygon
covers the rest::

    zone:
      include: [0.1, 0.35, 0.8, 0.65]                 # x, y, w, h
      exclude:
        - [[0.0, 0.0], [0.3, 0.0], [0.3, 0.4], [0.0, 0.4]]   # polygon

The test point defaults to the subject's **feet**, not their centre. Whether a
person is "on stage" or "on the court" is decided by where they are standing,
and a centre point wrongly excludes anyone tall near a zone edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import Box

Polygon = list[list[float]]


@dataclass
class ZoneConfig:
    #: Subject must be inside this. ``None`` means the whole frame.
    include: Sequence | None = None
    #: Subject is rejected if inside any of these. Applied after ``include``.
    exclude: list = field(default_factory=list)
    #: Which point of the subject box is tested: "feet", "center", or "head".
    anchor: str = "feet"


class Zones:
    """Compiled zone geometry, cached per frame size."""

    def __init__(self, cfg: ZoneConfig | None = None) -> None:
        self.cfg = cfg or ZoneConfig()
        self._include_norm = _as_polygon(self.cfg.include) if self.cfg.include else None
        self._exclude_norm = [_as_polygon(p) for p in (self.cfg.exclude or [])]
        self._cache_shape: tuple[int, int] | None = None
        self._include_px: np.ndarray | None = None
        self._exclude_px: list[np.ndarray] = []

        anchor = (self.cfg.anchor or "feet").lower()
        if anchor not in ("feet", "center", "centre", "head"):
            raise ValueError(
                f"unknown zone anchor {self.cfg.anchor!r} (expected feet, center, or head)"
            )
        self.anchor = "center" if anchor == "centre" else anchor

    @property
    def active(self) -> bool:
        return self._include_norm is not None or bool(self._exclude_norm)

    def _compile(self, frame_shape) -> None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if self._cache_shape == (h, w):
            return
        scale = np.array([w, h], dtype=np.float64)
        self._include_px = (
            (np.asarray(self._include_norm, dtype=np.float64) * scale).astype(np.int32)
            if self._include_norm is not None
            else None
        )
        self._exclude_px = [
            (np.asarray(p, dtype=np.float64) * scale).astype(np.int32) for p in self._exclude_norm
        ]
        self._cache_shape = (h, w)

    def anchor_point(self, box: Box) -> tuple[float, float]:
        if self.anchor == "feet":
            return (box.cx, box.y2)
        if self.anchor == "head":
            return (box.cx, box.y1)
        return (box.cx, box.cy)

    def contains(self, box: Box, frame_shape) -> bool:
        """Is this subject inside the tracking area?"""
        if not self.active:
            return True
        self._compile(frame_shape)
        point = self.anchor_point(box)

        if self._include_px is not None:
            if cv2.pointPolygonTest(self._include_px, point, False) < 0:
                return False
        for poly in self._exclude_px:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return False
        return True

    def polygons(self, frame_shape):
        """``(include, excludes)`` in pixel coordinates, for drawing."""
        if not self.active:
            return None, []
        self._compile(frame_shape)
        return self._include_px, self._exclude_px


def _as_polygon(spec) -> Polygon:
    """Accept either ``[x, y, w, h]`` or a list of ``[x, y]`` points."""
    pts = list(spec)
    if len(pts) == 4 and all(isinstance(v, (int, float)) for v in pts):
        x, y, w, h = (float(v) for v in pts)
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    if len(pts) < 3:
        raise ValueError(
            f"a zone polygon needs at least 3 points, or 4 numbers for [x, y, w, h]; got {spec!r}"
        )
    out: Polygon = []
    for p in pts:
        if len(p) != 2:
            raise ValueError(f"zone point must be [x, y], got {p!r}")
        out.append([float(p[0]), float(p[1])])
    return out
