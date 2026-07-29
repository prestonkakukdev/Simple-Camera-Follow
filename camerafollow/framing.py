"""Cinematographic framing rules -> a normalised error signal.

Three rules do almost all the work of a competent operator:

* **Headroom** -- the subject's face sits near the upper third, not the middle.
  Framing on the box centre is the single most common giveaway of an amateur
  auto-tracker: it puts the chin in the middle of frame and a metre of ceiling
  above the head.
* **Lead room** -- when the subject moves, leave space *ahead* of them. A walking
  preacher framed dead-centre looks like the camera is chasing; framed with
  lead room it looks like the camera anticipated.
* **Size** -- optional zoom to hold a constant shot size as they move towards or
  away from the lens.

Output is normalised so ``1.0`` == "half a view away", which makes the
controller gains independent of resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Box, FramingError, Rect


@dataclass
class FramingConfig:
    #: Where on the subject we aim. 0.0 = top of box, 1.0 = bottom.
    #: ~0.15 lands on the face for a full/medium body box.
    subject_anchor: float = 0.15
    #: Where in the view that anchor should sit vertically (0 = top).
    target_y: float = 0.32
    #: Where in the view the subject should sit horizontally when still.
    target_x: float = 0.50
    #: Lead room strength. Fraction of a half-view shifted at full speed.
    lead_gain: float = 0.22
    #: Speed (view-widths/sec) at which lead room saturates.
    lead_saturation: float = 0.55
    #: Seconds of lead-room smoothing, so a direction change eases across.
    lead_smoothing: float = 0.45
    #: Desired subject height as a fraction of view height (zoom target).
    target_height_frac: float = 0.62
    zoom_enabled: bool = False
    #: Dead band on shot size, as a fraction. Prevents zoom "hunting".
    zoom_tolerance: float = 0.12


class Composer:
    def __init__(self, cfg: FramingConfig | None = None) -> None:
        self.cfg = cfg or FramingConfig()
        self._lead = 0.0

    def reset(self) -> None:
        self._lead = 0.0

    def compute(
        self,
        box: Box,
        view: Rect,
        velocity: tuple[float, float],
        dt: float,
    ) -> FramingError:
        cfg = self.cfg
        half_w = max(1.0, view.w / 2)
        half_h = max(1.0, view.h / 2)

        # --- lead room -----------------------------------------------------
        vx_norm = velocity[0] / max(1.0, view.w)  # view-widths per second
        lead_target = _clamp(vx_norm / cfg.lead_saturation, -1.0, 1.0) * cfg.lead_gain
        alpha = 1.0 - _exp_decay(dt, cfg.lead_smoothing)
        self._lead += (lead_target - self._lead) * alpha

        # Moving right => leave space on the right => sit left of centre.
        target_x_frac = cfg.target_x - self._lead
        target_x_frac = _clamp(target_x_frac, 0.15, 0.85)

        # --- anchor points -------------------------------------------------
        anchor_x = box.cx
        anchor_y = box.y1 + cfg.subject_anchor * box.h

        goal_x = view.x + target_x_frac * view.w
        goal_y = view.y + cfg.target_y * view.h

        err_x = (anchor_x - goal_x) / half_w
        err_y = (anchor_y - goal_y) / half_h

        # --- shot size -----------------------------------------------------
        err_z = 0.0
        if cfg.zoom_enabled:
            frac = box.h / max(1.0, view.h)
            ratio = frac / max(1e-3, cfg.target_height_frac)
            if abs(ratio - 1.0) > cfg.zoom_tolerance:
                # Positive error => subject too small => zoom in.
                err_z = _clamp(1.0 - ratio, -1.0, 1.0)

        return FramingError(x=err_x, y=err_y, zoom=err_z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _exp_decay(dt: float, tau: float) -> float:
    """``exp(-dt/tau)`` guarded against tau == 0."""
    import math

    if tau <= 1e-6:
        return 0.0
    return math.exp(-dt / tau)
