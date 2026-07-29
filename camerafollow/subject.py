"""Subject selection: deciding *who* the camera is following.

This is the layer that separates a demo from something you can point at a
service. Detection and tracking are commodity; picking the preacher out of a
worship team and *not changing your mind* is the part that makes or breaks it.

Policy: score every confirmed track, but give the currently-locked subject a
large bonus (hysteresis) so a momentarily larger bystander can never steal the
shot. A lock survives ``hold_seconds`` of the subject being invisible before
the camera is allowed to re-target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .tracker import MultiTracker, Track
from .types import Rect
from .zones import Zones


@dataclass
class SubjectPolicy:
    #: Weight on subject size. Higher => prefers whoever is closest to camera.
    size_weight: float = 1.0
    #: Weight on proximity to the centre of the current view.
    center_weight: float = 0.6
    #: Weight on detection confidence.
    score_weight: float = 0.3
    #: Score bonus for the incumbent. This is the anti-flip-flop term.
    stickiness: float = 1.2
    #: Seconds the lock survives with the subject unseen (walks behind a pillar).
    hold_seconds: float = 2.5
    #: Ignore boxes smaller than this fraction of view height (background people).
    min_height_frac: float = 0.10
    #: Ignore boxes taller than this fraction (someone walking past the lens).
    max_height_frac: float = 1.60


class SubjectSelector:
    def __init__(self, policy: SubjectPolicy | None = None, zones: Zones | None = None) -> None:
        self.policy = policy or SubjectPolicy()
        #: Optional geometric restriction on where a subject may be. Applied to
        #: eligibility only -- tracks outside the zone still exist and keep
        #: their IDs, so someone stepping out and back does not get renumbered.
        self.zones = zones or Zones()
        self.locked_id: int | None = None
        self.manual = False
        self._frame_shape = None
        # Must start at "now". Starting at 0.0 makes the first hold-window
        # check compare against the epoch, so a subject occluded immediately
        # after acquisition loses its lock on the very next cycle.
        self._last_seen: float = time.perf_counter()

    # -- manual override ----------------------------------------------------
    def lock(self, track_id: int) -> None:
        """Pin the camera to one track and stop auto-selecting."""
        self.locked_id = track_id
        self.manual = True
        self._last_seen = time.perf_counter()

    def unlock(self) -> None:
        self.manual = False
        self.locked_id = None

    # -- automatic selection ------------------------------------------------
    def select(self, tracker: MultiTracker, view: Rect, frame_shape=None) -> Track | None:
        now = time.perf_counter()
        self._frame_shape = frame_shape
        current = tracker.by_id(self.locked_id) if self.locked_id is not None else None

        if current is not None and current.misses == 0:
            self._last_seen = now

        # A manual lock is absolute: hold it until the track dies outright.
        if self.manual:
            if current is None:
                self.manual = False
                self.locked_id = None
            else:
                return current

        # Keep coasting on the Kalman prediction through a brief occlusion.
        if current is not None and (now - self._last_seen) < self.policy.hold_seconds:
            candidates = list(self._eligible(tracker, view))
            if current in candidates or current.misses > 0:
                best = self._best(candidates, view, incumbent=current)
                if best is not None and best.id != current.id:
                    self.locked_id = best.id
                    return best
                return current

        best = self._best(list(self._eligible(tracker, view)), view, incumbent=current)
        if best is None:
            if current is not None and (now - self._last_seen) < self.policy.hold_seconds:
                return current
            self.locked_id = None
            return None

        self.locked_id = best.id
        self._last_seen = now
        return best

    # -- internals ----------------------------------------------------------
    def _eligible(self, tracker: MultiTracker, view: Rect):
        p = self.policy
        shape = getattr(self, "_frame_shape", None)
        for t in tracker.tracks:
            if not t.confirmed or t.misses > 0:
                continue
            frac = t.box.h / max(1.0, view.h)
            if frac < p.min_height_frac or frac > p.max_height_frac:
                continue
            if shape is not None and not self.zones.contains(t.box, shape):
                continue
            yield t

    def _best(self, candidates: list[Track], view: Rect, incumbent: Track | None) -> Track | None:
        if not candidates:
            return None
        p = self.policy
        best: Track | None = None
        best_score = float("-inf")
        for t in candidates:
            size = min(1.0, t.box.h / max(1.0, view.h))
            dx = (t.box.cx - view.cx) / max(1.0, view.w / 2)
            dy = (t.box.cy - view.cy) / max(1.0, view.h / 2)
            centrality = 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5)
            s = p.size_weight * size + p.center_weight * centrality + p.score_weight * t.score
            if incumbent is not None and t.id == incumbent.id:
                s += p.stickiness
            if s > best_score:
                best_score = s
                best = t
        return best
