"""Motion control: framing error -> a motor velocity that looks human.

The difference between "robot tracking a person" and "operator on a fluid head"
lives entirely in this file. Four elements, in order of how much they matter:

1. **Dead zone.** Below a threshold the camera simply does not move. Real
   operators do not micro-correct, and a tracker without a dead zone breathes
   and jitters constantly. Hysteresis widens the zone once stopped, so the
   camera settles instead of hovering at the boundary.
2. **Soft entry (smoothstep).** Velocity ramps in from zero at the dead-zone
   edge rather than stepping. Without it every move starts with a visible jolt.
3. **Slew limiting.** A hard cap on acceleration gives the ease-in/ease-out of a
   fluid head. This is the single most "expensive-looking" parameter.
4. **Lead / derivative term.** Adds velocity proportional to how fast the error
   is growing, so the camera matches a walking subject's speed instead of
   forever trailing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AxisConfig:
    #: No movement while |error| is under this (fraction of a half-view).
    deadzone: float = 0.06
    #: Dead zone multiplier once the axis has stopped. >1 == stickier at rest.
    deadzone_hysteresis: float = 1.8
    #: Width of the smoothstep ramp just outside the dead zone.
    soft_zone: float = 0.10
    #: Proportional gain: output velocity per unit of error.
    kp: float = 1.5
    #: Damping / lead on the error rate. Tracks constant-speed motion.
    kd: float = 0.35
    #: Low-pass time constant on the error rate, in seconds.
    #: Each new detection steps the Kalman state, and dividing that step by dt
    #: spikes a raw derivative -- which shows up as visible chatter at the
    #: moment the subject changes direction. Filter it. Raise if you still see
    #: twitching on reversals; lower if the camera feels sluggish to react.
    kd_tau: float = 0.08
    #: Max |velocity| in normalised units (1.0 == backend's full speed).
    max_velocity: float = 0.85
    #: Max change in velocity per second. Lower == more cinematic.
    max_acceleration: float = 2.2
    #: Extra ramp-down when the subject is lost, in seconds.
    stop_ramp: float = 0.35
    invert: bool = False


class AxisController:
    """One independent axis (pan, tilt, or zoom)."""

    def __init__(self, cfg: AxisConfig | None = None) -> None:
        self.cfg = cfg or AxisConfig()
        self.velocity = 0.0
        self._prev_error: float | None = None
        self._derr = 0.0
        self._moving = False

    def reset(self) -> None:
        self.velocity = 0.0
        self._prev_error = None
        self._derr = 0.0
        self._moving = False

    def step(self, error: float, dt: float) -> float:
        cfg = self.cfg
        dt = max(1e-4, dt)

        # 1. dead zone with hysteresis
        dz = cfg.deadzone if self._moving else cfg.deadzone * cfg.deadzone_hysteresis
        mag = abs(error)

        # Keep the filtered error rate updated even inside the dead zone, so a
        # move that starts on the far side begins with a settled derivative
        # rather than a spike.
        self._update_derivative(error, dt)

        if mag <= dz:
            self._moving = False
            return self._slew(0.0, dt)

        self._moving = True

        # 2. soft entry so motion begins from zero, not from kp*deadzone
        over = mag - dz
        ramp = _smoothstep(min(1.0, over / max(1e-6, cfg.soft_zone)))
        command = math.copysign(cfg.kp * over * ramp, error)

        # 4. lead term on the *filtered* error rate
        command += cfg.kd * self._derr * ramp

        command = _clamp(command, -cfg.max_velocity, cfg.max_velocity)
        return self._slew(command, dt)

    def _update_derivative(self, error: float, dt: float) -> None:
        if self._prev_error is None:
            self._prev_error = error
            return
        raw = (error - self._prev_error) / dt
        self._prev_error = error
        alpha = 1.0 - math.exp(-dt / max(1e-4, self.cfg.kd_tau))
        self._derr += (raw - self._derr) * alpha

    def coast_to_stop(self, dt: float) -> float:
        """Called when there is no subject: bleed off speed, never cut."""
        self._moving = False
        self._prev_error = None
        self._derr = 0.0
        decay = math.exp(-dt / max(1e-3, self.cfg.stop_ramp))
        target = self.velocity * decay
        # Exponential decay never actually reaches zero. Snap the tail, or the
        # head creeps forever at a speed too small to see but large enough to
        # keep the motor energised.
        if abs(target) < 0.01:
            target = 0.0
        return self._slew(target, dt)

    # 3. slew limiting
    def _slew(self, target: float, dt: float) -> float:
        cfg = self.cfg
        max_delta = cfg.max_acceleration * dt
        delta = _clamp(target - self.velocity, -max_delta, max_delta)
        self.velocity += delta
        if abs(self.velocity) < 1e-4:
            self.velocity = 0.0
        return -self.velocity if cfg.invert else self.velocity


def _smoothstep(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
