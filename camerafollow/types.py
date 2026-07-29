"""Core value types shared across the pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates of the *full source frame*."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h)

    @staticmethod
    def from_cxcywh(cx: float, cy: float, w: float, h: float) -> Box:
        w = max(1.0, w)
        h = max(1.0, h)
        return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def iou(self, other: Box) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_int(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class Detection:
    """One detected object from a Detector."""

    box: Box
    score: float
    label: str = "person"


@dataclass(frozen=True)
class DetectionResult:
    """A batch of detections, tagged with the capture time of their frame.

    ``frame_time`` is a ``time.perf_counter()`` stamp taken when the frame was
    grabbed, *not* when inference finished. The tracker needs this to correct
    for the age of an asynchronous detection.
    """

    detections: Sequence[Detection] = field(default_factory=tuple)
    frame_time: float = 0.0
    latency: float = 0.0


@dataclass(frozen=True)
class Rect:
    """The region of the source frame that is actually being output.

    For a motorised rig this is always the whole frame. For the virtual
    (digital) PTZ backend it is the current crop window, and framing errors
    must be measured against it rather than against the sensor.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @staticmethod
    def full(frame_shape: tuple[int, int, int] | tuple[int, int]) -> Rect:
        h, w = frame_shape[0], frame_shape[1]
        return Rect(0.0, 0.0, float(w), float(h))


@dataclass
class FramingError:
    """Normalised framing error. ``+-1.0`` means "half a view away"."""

    x: float
    y: float
    zoom: float = 0.0
