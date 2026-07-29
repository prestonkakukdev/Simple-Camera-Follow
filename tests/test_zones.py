"""Tracking zones."""

import pytest

from camerafollow.subject import SubjectPolicy, SubjectSelector
from camerafollow.tracker import MultiTracker
from camerafollow.types import Box, Detection, Rect
from camerafollow.zones import ZoneConfig, Zones

SHAPE = (1080, 1920, 3)
VIEW = Rect(0, 0, 1920, 1080)


def _person(cx, feet_y, h=600.0):
    return Box(cx - h / 6, feet_y - h, cx + h / 6, feet_y)


def test_no_zone_accepts_everything():
    z = Zones()
    assert not z.active
    assert z.contains(_person(10, 50), SHAPE)


def test_rect_shorthand_expands_to_a_polygon():
    z = Zones(ZoneConfig(include=[0.25, 0.5, 0.5, 0.4]))
    assert z.contains(_person(960, 800), SHAPE)  # feet at 0.74 y, 0.5 x
    assert not z.contains(_person(200, 800), SHAPE)  # too far left
    assert not z.contains(_person(960, 400), SHAPE)  # feet above the zone


def test_polygon_zone():
    z = Zones(ZoneConfig(include=[[0.4, 0.4], [0.9, 0.4], [0.9, 0.9], [0.4, 0.9]]))
    assert z.contains(_person(1200, 900), SHAPE)
    assert not z.contains(_person(300, 900), SHAPE)


def test_exclude_overrides_include():
    z = Zones(
        ZoneConfig(
            include=[0.0, 0.0, 1.0, 1.0],
            exclude=[[0.0, 0.0, 0.3, 1.0]],
        )
    )
    assert z.contains(_person(1500, 900), SHAPE)
    assert not z.contains(_person(300, 900), SHAPE)


def test_anchor_choice_changes_the_verdict():
    """Whether someone is 'in the area' depends on which point you test."""
    box = _person(960, 700, h=600)  # head at y=100, feet at y=700
    top_half = ZoneConfig(include=[0.0, 0.0, 1.0, 0.5])
    assert Zones(ZoneConfig(**{**top_half.__dict__, "anchor": "head"})).contains(box, SHAPE)
    assert not Zones(ZoneConfig(**{**top_half.__dict__, "anchor": "feet"})).contains(box, SHAPE)


def test_feet_is_the_default_anchor():
    assert Zones().anchor == "feet"


def test_bad_anchor_is_rejected():
    with pytest.raises(ValueError, match="unknown zone anchor"):
        Zones(ZoneConfig(anchor="elbow"))


def test_too_few_points_is_rejected():
    with pytest.raises(ValueError, match="at least 3 points"):
        Zones(ZoneConfig(include=[[0.1, 0.1], [0.5, 0.5]]))


def test_zone_geometry_is_resolution_independent():
    """Normalised coords mean a zone survives a resolution change."""
    z = Zones(ZoneConfig(include=[0.25, 0.25, 0.5, 0.5]))
    for shape in ((1080, 1920, 3), (2160, 3840, 3), (720, 1280, 3)):
        h, w = shape[0], shape[1]
        inside = Box(w * 0.45, h * 0.3, w * 0.55, h * 0.6)
        outside = Box(w * 0.02, h * 0.3, w * 0.12, h * 0.6)
        assert z.contains(inside, shape), shape
        assert not z.contains(outside, shape), shape


# -- integration with subject selection -------------------------------------
def _settle(tr, dets, frames=8):
    for _ in range(frames):
        tr.predict(1 / 20)
        tr.update(dets)


def test_selector_ignores_people_outside_the_zone():
    """The whole point: a nearer, more central bystander must not win."""
    tr = MultiTracker(min_hits=2)
    on_stage = Detection(_person(1500, 900, h=500), 0.9)
    bystander = Detection(_person(800, 1050, h=900), 0.95)  # bigger AND more central
    _settle(tr, [on_stage, bystander])

    open_sel = SubjectSelector(SubjectPolicy(min_height_frac=0.05))
    assert open_sel.select(tr, VIEW, SHAPE).box.h == 900  # bystander wins outright

    zoned = SubjectSelector(
        SubjectPolicy(min_height_frac=0.05),
        Zones(ZoneConfig(include=[0.5, 0.0, 0.5, 1.0])),  # right half only
    )
    picked = zoned.select(tr, VIEW, SHAPE)
    assert picked is not None and picked.box.h == 500  # the on-stage subject


def test_tracks_outside_the_zone_keep_their_identity():
    """Stepping out of the zone must not renumber someone.

    Zones gate *eligibility*, not tracking. If they filtered detections the
    subject would get a new ID every time they crossed the line, and every
    re-entry would look like a stranger.
    """
    tr = MultiTracker(min_hits=2)
    zoned = SubjectSelector(
        SubjectPolicy(min_height_frac=0.05, hold_seconds=0.0),
        Zones(ZoneConfig(include=[0.5, 0.0, 0.5, 1.0])),
    )
    _settle(tr, [Detection(_person(1400, 900), 0.9)])
    first = zoned.select(tr, VIEW, SHAPE)
    assert first is not None
    original_id = first.id

    _settle(tr, [Detection(_person(300, 900), 0.9)])  # walks out of the zone
    assert zoned.select(tr, VIEW, SHAPE) is None
    assert tr.by_id(original_id) is not None  # still tracked

    _settle(tr, [Detection(_person(1400, 900), 0.9)])  # walks back in
    back = zoned.select(tr, VIEW, SHAPE)
    assert back is not None and back.id == original_id
