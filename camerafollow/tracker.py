"""Multi-object tracker: stable IDs for every person in frame.

Greedy IoU association over Kalman-predicted boxes (the SORT recipe). Deliberately
simple and dependency-light -- the job here is only to keep an ID attached to a
body across occlusions and missed detections, so the subject-selection layer
above has something continuous to lock onto.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .kalman import BoxKalman
from .types import Box, Detection


@dataclass
class Track:
    id: int
    kf: BoxKalman
    score: float = 0.0
    hits: int = 1
    age: int = 0
    #: Consecutive detection cycles with no match. 0 == seen right now.
    misses: int = 0
    #: Seconds since this track was last corrected by a detection. Time, not
    #: frames: detection rate varies with load, and a frame-counted lifetime
    #: silently means something different on every machine.
    age_seconds: float = 0.0
    confirmed: bool = False
    # Coarse colour signature of the torso, used to re-attach an ID after an
    # occlusion. Optional -- stays None when appearance matching is disabled.
    appearance: object | None = None

    @property
    def box(self) -> Box:
        return self.kf.box

    @property
    def velocity(self) -> tuple[float, float]:
        return self.kf.velocity

    def box_ahead(self, lead: float) -> Box:
        return self.kf.box_ahead(lead)


class MultiTracker:
    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        max_age: float = 2.0,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    def predict(self, dt: float) -> None:
        for t in self.tracks:
            t.kf.predict(dt)
            t.age += 1
            t.age_seconds += dt

    def update(self, detections: Sequence[Detection], *, staleness: float = 0.0) -> None:
        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for track_idx, det_idx in matches:
            t = self.tracks[track_idx]
            d = detections[det_idx]
            t.kf.update(d.box, staleness=staleness, trust=max(0.05, min(1.0, d.score)))
            t.score = d.score
            t.hits += 1
            t.misses = 0
            t.age_seconds = 0.0
            if t.hits >= self.min_hits:
                t.confirmed = True

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].misses += 1

        for det_idx in unmatched_dets:
            d = detections[det_idx]
            self.tracks.append(Track(id=self._next_id, kf=BoxKalman(d.box), score=d.score))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age_seconds <= self.max_age]

    def _associate(
        self, detections: Sequence[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        pairs: list[tuple[float, int, int]] = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                iou = t.box.iou(d.box)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        matches: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_t]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def visible(self) -> Iterable[Track]:
        """Tracks that are confirmed and currently being observed."""
        return (t for t in self.tracks if t.confirmed and t.misses == 0)

    def by_id(self, track_id: int) -> Track | None:
        for t in self.tracks:
            if t.id == track_id:
                return t
        return None
