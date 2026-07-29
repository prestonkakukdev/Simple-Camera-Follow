"""Framing rules and the digital-PTZ backend."""

import pytest

from camerafollow.backends import build_backend
from camerafollow.backends.virtual import VirtualConfig, VirtualPTZ
from camerafollow.framing import Composer, FramingConfig
from camerafollow.types import Box, Rect

VIEW = Rect(0, 0, 1920, 1080)
DT = 1 / 60


def _standing(cx=960.0, cy=540.0, h=600.0):
    return Box.from_cxcywh(cx, cy, h / 3, h)


def test_centred_subject_with_headroom_is_already_framed():
    cfg = FramingConfig(subject_anchor=0.15, target_y=0.32)
    box = _standing()
    # place the anchor exactly on the goal
    anchor_offset = cfg.subject_anchor * box.h
    box = Box.from_cxcywh(960, cfg.target_y * 1080 + box.h / 2 - anchor_offset, 200, 600)
    err = Composer(cfg).compute(box, VIEW, (0.0, 0.0), DT)
    assert err.x == pytest.approx(0.0, abs=1e-6)
    assert err.y == pytest.approx(0.0, abs=1e-6)


def test_subject_right_of_centre_gives_positive_x_error():
    err = Composer().compute(_standing(cx=1500), VIEW, (0.0, 0.0), DT)
    assert err.x > 0


def test_framing_targets_the_face_not_the_body_centre():
    """Framing on the box centre is the classic amateur-tracker giveaway."""
    cfg = FramingConfig(subject_anchor=0.15, target_y=0.32)
    body_centred = Box.from_cxcywh(960, 0.32 * 1080, 200, 600)
    err = Composer(cfg).compute(body_centred, VIEW, (0.0, 0.0), DT)
    # the anchor sits above the box centre, so this "centred" box reads as too high
    assert err.y < 0


def test_lead_room_leaves_space_ahead_of_travel():
    cfg = FramingConfig(lead_gain=0.25, lead_smoothing=0.0)
    comp = Composer(cfg)
    box = _standing()
    still = comp.compute(box, VIEW, (0.0, 0.0), DT).x

    comp.reset()
    moving_right = comp.compute(box, VIEW, (700.0, 0.0), DT).x
    # moving right => the subject should sit left of centre => larger +x error
    assert moving_right > still

    comp.reset()
    moving_left = comp.compute(box, VIEW, (-700.0, 0.0), DT).x
    assert moving_left < still


def test_lead_room_is_smoothed_across_a_direction_change():
    cfg = FramingConfig(lead_gain=0.25, lead_smoothing=0.5)
    comp = Composer(cfg)
    box = _standing()
    for _ in range(60):
        comp.compute(box, VIEW, (700.0, 0.0), DT)
    settled = comp.compute(box, VIEW, (700.0, 0.0), DT).x
    flipped = comp.compute(box, VIEW, (-700.0, 0.0), DT).x
    assert abs(flipped - settled) < 0.05  # eases across, never snaps


def test_zoom_error_signs():
    cfg = FramingConfig(zoom_enabled=True, target_height_frac=0.6, zoom_tolerance=0.05)
    comp = Composer(cfg)
    small = comp.compute(Box.from_cxcywh(960, 540, 100, 300), VIEW, (0, 0), DT)
    assert small.zoom > 0  # too small => zoom in
    comp.reset()
    big = comp.compute(Box.from_cxcywh(960, 540, 300, 1000), VIEW, (0, 0), DT)
    assert big.zoom < 0


def test_zoom_is_silent_inside_its_tolerance():
    cfg = FramingConfig(zoom_enabled=True, target_height_frac=0.6, zoom_tolerance=0.2)
    err = Composer(cfg).compute(Box.from_cxcywh(960, 540, 200, 620), VIEW, (0, 0), DT)
    assert err.zoom == 0.0


# -- virtual PTZ ------------------------------------------------------------
def test_virtual_view_rect_matches_zoom():
    ptz = VirtualPTZ(VirtualConfig(zoom=2.0))
    r = ptz.view_rect((1080, 1920, 3))
    assert r.w == pytest.approx(960)
    assert r.h == pytest.approx(540)


def test_virtual_pan_moves_the_window_and_clamps_at_the_sensor_edge():
    ptz = VirtualPTZ(VirtualConfig(zoom=2.0, pan_rate=1.0))
    start = ptz.view_rect((1080, 1920, 3)).x
    for _ in range(30):
        ptz.move(1.0, 0.0, 0.0, DT)
    assert ptz.view_rect((1080, 1920, 3)).x > start
    for _ in range(600):
        ptz.move(1.0, 0.0, 0.0, DT)
    r = ptz.view_rect((1080, 1920, 3))
    assert r.x + r.w <= 1920 + 1e-6  # never reads past the sensor


def test_virtual_tilt_up_moves_the_window_up():
    ptz = VirtualPTZ(VirtualConfig(zoom=2.0, tilt_rate=1.0))
    start = ptz.view_rect((1080, 1920, 3)).y
    for _ in range(30):
        ptz.move(0.0, 1.0, 0.0, DT)
    assert ptz.view_rect((1080, 1920, 3)).y < start


def test_virtual_zoom_is_bounded():
    ptz = VirtualPTZ(VirtualConfig(zoom=2.0, min_zoom=1.5, max_zoom=2.5, zoom_rate=2.0))
    for _ in range(600):
        ptz.move(0.0, 0.0, 1.0, DT)
    assert ptz.zoom <= 2.5 + 1e-9
    for _ in range(1200):
        ptz.move(0.0, 0.0, -1.0, DT)
    assert ptz.zoom >= 1.5 - 1e-9


def test_null_backend_is_selectable_for_dry_runs():
    b = build_backend("null")
    b.move(0.5, -0.2, 0.0, DT)
    assert b.last == (0.5, -0.2, 0.0)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend("hovercraft")
