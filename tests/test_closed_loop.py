"""Closed-loop simulation: does the whole stack actually hold a shot?

A synthetic subject walks across a 4K sensor. Detections are fed in at a
realistic (slower) rate with pixel noise, exactly as a real detector would, and
the full chain -- tracker, subject policy, framing, controllers, digital PTZ --
runs against it.

These assertions are the ones that matter. Every unit test above can pass while
the camera still looks terrible; this one fails if it does.
"""

import math
import random
from collections import deque

from camerafollow.backends.virtual import VirtualConfig, VirtualPTZ
from camerafollow.control import AxisConfig, AxisController
from camerafollow.framing import Composer, FramingConfig
from camerafollow.subject import SubjectPolicy, SubjectSelector
from camerafollow.tracker import MultiTracker
from camerafollow.types import Box, Detection

SENSOR = (2160, 3840, 3)
CONTROL_HZ = 60.0
DETECT_HZ = 20.0
DT = 1.0 / CONTROL_HZ


class Rig:
    """Assembles the same components the real pipeline does.

    ``actuator_delay`` models the gap between issuing a command and the head
    actually moving -- network round trip, motor spin-up, stream buffering.
    Real rigs have 80-250ms of it; that is precisely what ``lead`` exists to
    cancel, so a simulation without it cannot show whether ``lead`` works.
    """

    def __init__(self, *, actuator_delay: float = 0.0, **framing):
        self.tracker = MultiTracker(min_hits=2, max_age=3.0)
        self.selector = SubjectSelector(SubjectPolicy(min_height_frac=0.05))
        self.composer = Composer(FramingConfig(**framing))
        self.pan = AxisController(AxisConfig(deadzone=0.05, kp=1.6, kd=0.35))
        self.tilt = AxisController(AxisConfig(deadzone=0.07, kp=1.2, max_velocity=0.6))
        self.ptz = VirtualPTZ(VirtualConfig(zoom=2.0, pan_rate=1.2, tilt_rate=0.8))
        self.lead = 0.10
        self._pipe = deque([(0.0, 0.0)] * max(0, round(actuator_delay * CONTROL_HZ)))
        self.history: list[tuple[float, float, float]] = []  # (err_x, err_y, pan_v)
        #: Framing error measured against the subject's *true* position, which
        #: is what an audience sees. The controller never gets to look at this.
        self.true_error: list[tuple[float, float]] = []

    def _actuate(self, pan: float, tilt: float) -> None:
        if self._pipe:
            self._pipe.append((pan, tilt))
            pan, tilt = self._pipe.popleft()
        self.ptz.move(pan, tilt, 0.0, DT)

    def step(self, detection: Detection | None, truth: Box | None = None):
        self.tracker.predict(DT)
        if detection is not None:
            self.tracker.update([detection])
        view = self.ptz.view_rect(SENSOR)
        subject = self.selector.select(self.tracker, view)

        if subject is None:
            self._actuate(self.pan.coast_to_stop(DT), self.tilt.coast_to_stop(DT))
            return

        predicted = subject.box_ahead(self.lead)
        err = self.composer.compute(predicted, view, subject.velocity, DT)
        pan = self.pan.step(err.x, DT)
        tilt = self.tilt.step(-err.y, DT)
        self._actuate(pan, tilt)
        self.history.append((err.x, err.y, pan))

        if truth is not None:
            cfg = self.composer.cfg
            v = self.ptz.view_rect(SENSOR)
            tx = v.x + cfg.target_x * v.w
            ty = v.y + cfg.target_y * v.h
            self.true_error.append(
                (
                    (truth.cx - tx) / (v.w / 2),
                    ((truth.y1 + cfg.subject_anchor * truth.h) - ty) / (v.h / 2),
                )
            )


def _walk(rig: Rig, path, seconds: float, *, noise=6.0, seed=1234):
    """Drive the rig for ``seconds`` with ``path(t) -> (cx, cy, h)``."""
    rng = random.Random(seed)
    detect_period = 1.0 / DETECT_HZ
    next_detect = 0.0
    steps = int(seconds * CONTROL_HZ)
    for i in range(steps):
        t = i * DT
        cx, cy, h = path(t)
        truth = Box.from_cxcywh(cx, cy, h / 3, h)
        det = None
        if t >= next_detect:
            det = Detection(
                Box.from_cxcywh(
                    cx + rng.gauss(0, noise),
                    cy + rng.gauss(0, noise),
                    h / 3 + rng.gauss(0, noise / 2),
                    h + rng.gauss(0, noise),
                ),
                0.9,
            )
            next_detect += detect_period
        rig.step(det, truth)


def _final_errors(rig, last_seconds=2.0):
    n = int(last_seconds * CONTROL_HZ)
    return rig.history[-n:]


def test_acquires_a_standing_subject():
    """Off-centre at start, correctly framed within a couple of seconds."""
    rig = Rig()
    _walk(rig, lambda t: (2600.0, 1100.0, 900.0), seconds=6.0)
    errs = _final_errors(rig)
    assert max(abs(e[0]) for e in errs) < 0.12
    assert max(abs(e[1]) for e in errs) < 0.14


def test_holds_a_walking_subject():
    """A preacher crossing the stage.

    Kept inside the crop window's reach: at 2x zoom on a 3840-wide sensor the
    view centre can only travel 960..2880, and past that the error necessarily
    grows because there is no more sensor to pan into.
    """
    rig = Rig()
    _walk(rig, lambda t: (1500.0 + 140.0 * t, 1100.0, 900.0), seconds=8.0)
    errs = _final_errors(rig)
    # steady-state trailing error stays inside the frame's comfortable zone
    assert max(abs(e[0]) for e in errs) < 0.25


def test_does_not_oscillate_on_a_pacing_subject():
    """Back-and-forth pacing must not put the camera into a hunting loop."""
    rig = Rig()
    _walk(rig, lambda t: (1900.0 + 500.0 * math.sin(t * 0.8), 1100.0, 900.0), seconds=14.0)
    pans = [h[2] for h in _final_errors(rig, 6.0)]
    # count zero-crossings of pan velocity; a hunting loop produces many more
    crossings = sum(1 for a, b in zip(pans, pans[1:], strict=False) if a * b < 0)
    assert crossings <= 12


def test_motion_is_smooth_no_frame_to_frame_jerk():
    """The acceleration limit must hold end to end, not just in the unit test."""
    rig = Rig()
    _walk(rig, lambda t: (1200.0 + 300.0 * t, 1100.0 + 60.0 * math.sin(t), 900.0), seconds=8.0)
    pans = [h[2] for h in rig.history]
    max_jerk = max(abs(b - a) for a, b in zip(pans, pans[1:], strict=False))
    assert max_jerk <= rig.pan.cfg.max_acceleration * DT + 1e-6


def test_camera_is_still_when_the_subject_is_still():
    """The dead zone must actually hold: a stationary speaker gets a locked-off shot."""
    rig = Rig()
    _walk(rig, lambda t: (1920.0, 1080.0, 900.0), seconds=10.0, noise=8.0)
    pans = [h[2] for h in _final_errors(rig, 4.0)]
    assert max(abs(p) for p in pans) < 0.02


def test_coasts_through_an_occlusion_without_a_snap():
    """Subject disappears for half a second; the shot must not lurch."""
    rig = Rig()

    def path(t):
        return (1600.0 + 130.0 * t, 1100.0, 900.0)

    rng = random.Random(7)
    steps = int(9.0 * CONTROL_HZ)
    next_detect = 0.0
    for i in range(steps):
        t = i * DT
        occluded = 4.0 <= t < 4.6
        det = None
        if t >= next_detect and not occluded:
            cx, cy, h = path(t)
            det = Detection(Box.from_cxcywh(cx + rng.gauss(0, 5), cy, h / 3, h), 0.9)
            next_detect += 1.0 / DETECT_HZ
        elif t >= next_detect:
            next_detect += 1.0 / DETECT_HZ
        cx, cy, h = path(t)
        rig.step(det, Box.from_cxcywh(cx, cy, h / 3, h))

    pans = [h[2] for h in rig.history]
    assert (
        max(abs(b - a) for a, b in zip(pans, pans[1:], strict=False))
        <= rig.pan.cfg.max_acceleration * DT + 1e-6
    )
    assert max(abs(e[0]) for e in _final_errors(rig, 2.0)) < 0.3


def _mean_true_error(rig, last_seconds=3.0):
    n = int(last_seconds * CONTROL_HZ)
    tail = rig.true_error[-n:]
    return sum(abs(e[0]) for e in tail) / len(tail)


def test_lead_prediction_cancels_actuator_latency():
    """The core claim of the whole design, measured against ground truth.

    Given a rig with 120ms of actuator delay, aiming ``lead_time`` ahead should
    visibly beat aiming at where the subject currently is. Error here is
    measured against the subject's true position -- what the audience sees --
    not against the prediction the controller was working from.
    """

    def path(t):
        return (1500.0 + 150.0 * t, 1100.0, 900.0)

    with_lead = Rig(actuator_delay=0.12)
    with_lead.lead = 0.12
    _walk(with_lead, path, seconds=9.0)

    without = Rig(actuator_delay=0.12)
    without.lead = 0.0
    _walk(without, path, seconds=9.0)

    assert _mean_true_error(with_lead) < _mean_true_error(without) * 0.8


def test_lead_prediction_does_not_overshoot_a_still_subject():
    """Latency compensation must cost nothing when the subject isn't moving."""
    rig = Rig(actuator_delay=0.12)
    rig.lead = 0.15
    _walk(rig, lambda t: (2100.0, 1100.0, 900.0), seconds=8.0)
    assert _mean_true_error(rig) < 0.1
