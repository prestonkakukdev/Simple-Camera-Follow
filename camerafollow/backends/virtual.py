"""Virtual (digital) PTZ -- auto-framing with no motors at all.

A static 4K camera, cropped to a moving 1080p window, gives you real
auto-tracking with zero moving parts. For a stage this is often *better* than a
motorised head: silent, instant, no backlash, nothing to fail mid-service, and
you can run several independent virtual cameras off one sensor.

It is also how you should develop. Everything in the pipeline above this file is
identical whether the output is a crop or a motor command, so you can tune all
your framing and control gains against a recorded service on a laptop, then move
the tuned config to hardware unchanged.

The window is driven with the same normalised velocities a motor gets, and it
reports its crop through ``view_rect`` so framing errors are measured against
what the audience sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from ..types import Rect
from .base import BaseBackend


@dataclass
class VirtualConfig:
    #: Output resolution. None => source resolution divided by ``zoom``.
    out_width: int | None = 1920
    out_height: int | None = 1080
    #: Starting crop factor. 2.0 means the window is half the sensor wide.
    zoom: float = 2.0
    min_zoom: float = 1.0
    max_zoom: float = 4.0
    #: Full-speed pan travel in view-widths per second.
    pan_rate: float = 0.9
    tilt_rate: float = 0.7
    #: Full-speed zoom in octaves per second.
    zoom_rate: float = 0.35


class VirtualPTZ(BaseBackend):
    name = "virtual"

    def __init__(self, cfg: VirtualConfig | None = None) -> None:
        self.cfg = cfg or VirtualConfig()
        self.zoom = _clamp(self.cfg.zoom, self.cfg.min_zoom, self.cfg.max_zoom)
        # Crop centre in normalised sensor coordinates.
        self.cx = 0.5
        self.cy = 0.5
        self._frame_shape: tuple[int, int] | None = None

    def move(self, pan: float, tilt: float, zoom: float, dt: float) -> None:
        cfg = self.cfg
        if zoom:
            self.zoom = _clamp(
                self.zoom * (2.0 ** (zoom * cfg.zoom_rate * dt)),
                cfg.min_zoom,
                cfg.max_zoom,
            )

        view_w = 1.0 / self.zoom
        view_h = 1.0 / self.zoom
        # Rates are in view-widths/sec, so panning speed scales with the shot --
        # a tight shot moves slowly across the sensor, exactly like a real head.
        self.cx += pan * cfg.pan_rate * view_w * dt
        self.cy -= tilt * cfg.tilt_rate * view_h * dt

        half_w = view_w / 2
        half_h = view_h / 2
        self.cx = _clamp(self.cx, half_w, 1.0 - half_w)
        self.cy = _clamp(self.cy, half_h, 1.0 - half_h)

    def stop(self) -> None:
        return

    def view_rect(self, frame_shape) -> Rect:
        h, w = frame_shape[0], frame_shape[1]
        vw = w / self.zoom
        vh = h / self.zoom
        x = _clamp(self.cx * w - vw / 2, 0.0, max(0.0, w - vw))
        y = _clamp(self.cy * h - vh / 2, 0.0, max(0.0, h - vh))
        return Rect(x, y, vw, vh)

    def render(self, frame):
        r = self.view_rect(frame.shape)
        x1, y1 = int(r.x), int(r.y)
        x2, y2 = int(r.x + r.w), int(r.y + r.h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame
        ow = self.cfg.out_width
        oh = self.cfg.out_height
        if ow and oh and (crop.shape[1] != ow or crop.shape[0] != oh):
            interp = cv2.INTER_AREA if crop.shape[1] > ow else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (ow, oh), interpolation=interp)
        return crop


def _clamp(v: float, lo: float, hi: float) -> float:
    if hi < lo:
        return lo
    return lo if v < lo else hi if v > hi else v
