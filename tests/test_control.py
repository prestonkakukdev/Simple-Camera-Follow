"""The controller is what the audience actually sees, so pin its behaviour."""

from camerafollow.control import AxisConfig, AxisController


def _run(ctrl, error, seconds, dt=1 / 60):
    out = []
    for _ in range(int(seconds / dt)):
        out.append(ctrl.step(error, dt))
    return out


def test_deadzone_holds_still():
    ctrl = AxisController(AxisConfig(deadzone=0.06))
    assert all(v == 0.0 for v in _run(ctrl, 0.03, 0.5))


def test_moves_outside_deadzone():
    ctrl = AxisController(AxisConfig(deadzone=0.06))
    assert _run(ctrl, 0.5, 1.0)[-1] > 0.1


def test_direction_follows_error_sign():
    ctrl = AxisController(AxisConfig())
    assert _run(ctrl, -0.5, 1.0)[-1] < 0


def test_velocity_ramps_in_rather_than_stepping():
    """No jolt on the first frame of a move -- this is the 'looks cheap' bug."""
    ctrl = AxisController(AxisConfig(max_acceleration=2.0))
    first = ctrl.step(1.0, 1 / 60)
    assert abs(first) <= 2.0 / 60 + 1e-9


def test_acceleration_is_limited_every_frame():
    ctrl = AxisController(AxisConfig(max_acceleration=1.5))
    dt = 1 / 60
    prev = 0.0
    for _ in range(200):
        v = ctrl.step(1.0, dt)
        assert abs(v - prev) <= 1.5 * dt + 1e-9
        prev = v


def test_velocity_is_clamped():
    ctrl = AxisController(AxisConfig(max_velocity=0.4, kp=10.0))
    assert max(abs(v) for v in _run(ctrl, 1.0, 2.0)) <= 0.4 + 1e-9


def test_hysteresis_prevents_chatter_at_the_boundary():
    """Sitting just inside the widened dead zone must not restart the motor."""
    cfg = AxisConfig(deadzone=0.05, deadzone_hysteresis=2.0)
    ctrl = AxisController(cfg)
    _run(ctrl, 0.07, 1.0)  # 0.07 < 0.05*2, so a stopped axis stays stopped
    assert ctrl.velocity == 0.0


def test_coast_to_stop_decays_smoothly():
    ctrl = AxisController(AxisConfig(max_acceleration=3.0))
    _run(ctrl, 1.0, 1.0)
    assert ctrl.velocity > 0.1
    dt = 1 / 60
    prev = ctrl.velocity
    for _ in range(120):
        v = ctrl.coast_to_stop(dt)
        assert abs(v - prev) <= 3.0 * dt + 1e-9
        prev = v
    assert ctrl.velocity == 0.0


def test_invert_flips_output_only():
    a = AxisController(AxisConfig())
    b = AxisController(AxisConfig(invert=True))
    va = _run(a, 0.5, 1.0)[-1]
    vb = _run(b, 0.5, 1.0)[-1]
    assert va == -vb != 0.0
