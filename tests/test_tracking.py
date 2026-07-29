"""Kalman, tracker, and subject-selection behaviour."""

import pytest

from camerafollow.kalman import BoxKalman
from camerafollow.subject import SubjectPolicy, SubjectSelector
from camerafollow.tracker import MultiTracker
from camerafollow.types import Box, Detection, Rect

VIEW = Rect(0, 0, 1920, 1080)


def _person(cx, cy=540, w=200, h=600, score=0.9):
    return Detection(Box.from_cxcywh(cx, cy, w, h), score)


# -- kalman -----------------------------------------------------------------
def test_kalman_converges_on_a_static_box():
    box = Box.from_cxcywh(800, 500, 200, 600)
    kf = BoxKalman(Box.from_cxcywh(600, 500, 200, 600))
    for _ in range(40):
        kf.predict(1 / 30)
        kf.update(box)
    assert kf.box.cx == pytest.approx(800, abs=15)


def test_kalman_estimates_velocity():
    kf = BoxKalman(Box.from_cxcywh(400, 500, 200, 600))
    x = 400.0
    for _ in range(60):
        kf.predict(1 / 30)
        x += 300 / 30  # 300 px/sec
        kf.update(Box.from_cxcywh(x, 500, 200, 600))
    vx, _ = kf.velocity
    assert vx == pytest.approx(300, rel=0.25)


def test_prediction_leads_a_moving_subject():
    """The whole latency-hiding trick, in one assertion."""
    kf = BoxKalman(Box.from_cxcywh(400, 500, 200, 600))
    x = 400.0
    for _ in range(60):
        kf.predict(1 / 30)
        x += 10
        kf.update(Box.from_cxcywh(x, 500, 200, 600))
    ahead = kf.box_ahead(0.2)
    assert ahead.cx > kf.box.cx + 30


def test_stale_measurements_are_trusted_less():
    fresh = BoxKalman(Box.from_cxcywh(400, 500, 200, 600))
    stale = BoxKalman(Box.from_cxcywh(400, 500, 200, 600))
    jump = Box.from_cxcywh(900, 500, 200, 600)
    fresh.predict(1 / 30)
    stale.predict(1 / 30)
    # give both a velocity, since staleness is weighted by how fast the subject
    # is moving -- a stationary subject's box doesn't go stale
    for kf in (fresh, stale):
        kf.x[4] = 400.0
    fresh.update(jump, staleness=0.0)
    stale.update(jump, staleness=0.3)
    assert stale.box.cx < fresh.box.cx


# -- tracker ----------------------------------------------------------------
def test_ids_persist_across_frames():
    tr = MultiTracker(min_hits=2)
    x = 500.0
    for _ in range(10):
        tr.predict(1 / 30)
        tr.update([_person(x)])
        x += 8
    ids = {t.id for t in tr.tracks}
    assert ids == {1}
    assert tr.tracks[0].confirmed


def test_two_people_get_distinct_stable_ids():
    tr = MultiTracker(min_hits=2)
    for _ in range(10):
        tr.predict(1 / 30)
        tr.update([_person(400), _person(1400)])
    assert {t.id for t in tr.tracks} == {1, 2}


def test_track_survives_a_short_occlusion():
    tr = MultiTracker(min_hits=2, max_age=1.0)
    for _ in range(10):
        tr.predict(1 / 30)
        tr.update([_person(500)])
    for _ in range(20):  # subject walks behind a pillar
        tr.predict(1 / 30)
        tr.update([])
    assert len(tr.tracks) == 1
    tr.predict(1 / 30)
    tr.update([_person(500)])
    assert tr.tracks[0].id == 1  # same person, same ID


def test_track_is_dropped_after_a_long_absence():
    tr = MultiTracker(min_hits=2, max_age=0.4)
    for _ in range(10):
        tr.predict(1 / 30)
        tr.update([_person(500)])
    for _ in range(15):
        tr.predict(1 / 30)
        tr.update([])
    assert tr.tracks == []


# -- subject selection ------------------------------------------------------
def _settle(tr, dets, frames=8):
    for _ in range(frames):
        tr.predict(1 / 30)
        tr.update(dets)


def test_selects_the_dominant_person():
    tr = MultiTracker(min_hits=2)
    _settle(tr, [_person(500, h=300), _person(1200, h=700)])
    sel = SubjectSelector()
    subject = sel.select(tr, VIEW)
    assert subject is not None and subject.box.h == 700


def test_stickiness_prevents_the_shot_flipping():
    """A bystander who becomes marginally larger must not steal the shot."""
    tr = MultiTracker(min_hits=2)
    _settle(tr, [_person(500, h=620), _person(1200, h=560)])
    sel = SubjectSelector(SubjectPolicy(stickiness=1.2))
    first = sel.select(tr, VIEW)
    assert first is not None
    locked = first.id
    # the other person steps forward and is now slightly bigger
    _settle(tr, [_person(500, h=600), _person(1200, h=660)])
    assert sel.select(tr, VIEW).id == locked


def test_manual_lock_overrides_scoring():
    tr = MultiTracker(min_hits=2)
    _settle(tr, [_person(500, h=300), _person(1200, h=800)])
    small = min(tr.tracks, key=lambda t: t.box.h)
    sel = SubjectSelector()
    sel.lock(small.id)
    assert sel.select(tr, VIEW).id == small.id
    sel.unlock()
    assert sel.select(tr, VIEW).id != small.id


def test_background_people_are_ignored():
    tr = MultiTracker(min_hits=2)
    _settle(tr, [_person(500, h=60)])  # < min_height_frac of a 1080 view
    sel = SubjectSelector(SubjectPolicy(min_height_frac=0.10, hold_seconds=0.0))
    assert sel.select(tr, VIEW) is None


def test_no_people_means_no_subject():
    sel = SubjectSelector(SubjectPolicy(hold_seconds=0.0))
    assert sel.select(MultiTracker(), VIEW) is None


def test_hold_seconds_is_not_silently_capped_by_track_lifetime():
    """Regression: `max_age` must not cull a locked track mid-hold.

    These two settings live in different layers and are easy to drift apart.
    When they do, `hold_seconds` quietly stops meaning what it says and the
    camera abandons a subject earlier than configured.
    """
    tr = MultiTracker(min_hits=2, max_age=3.0)
    sel = SubjectSelector(SubjectPolicy(hold_seconds=2.5))
    _settle(tr, [_person(960)])
    sel.select(tr, VIEW)
    assert sel.locked_id is not None

    # 2.4s of occlusion, one detection cycle at a time
    for _ in range(48):
        tr.predict(1 / 20)
        tr.update([])
    assert tr.by_id(1) is not None, "track culled before hold_seconds elapsed"
    assert tr.by_id(1).age_seconds < 3.0


def test_lock_survives_occlusion_immediately_after_acquisition():
    """Regression: `_last_seen` used to start at 0.0, comparing against the epoch."""
    tr = MultiTracker(min_hits=2, max_age=3.0)
    sel = SubjectSelector(SubjectPolicy(hold_seconds=2.5))
    _settle(tr, [_person(960)])
    sel.select(tr, VIEW)
    tr.predict(1 / 20)
    tr.update([])  # occluded on the very next cycle
    assert sel.select(tr, VIEW) is not None
