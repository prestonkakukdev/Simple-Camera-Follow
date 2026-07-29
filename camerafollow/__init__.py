"""camerafollow -- an open-source AI camera operator.

Detects people in a video feed, locks onto a subject, and drives a pan/tilt head
(or a digital crop window) to keep them beautifully framed, with no human at the
camera.

    from camerafollow import AppConfig, FollowPipeline

    cfg = AppConfig.load("configs/virtual_4k.yaml")
    FollowPipeline(cfg).run()

The pipeline is deliberately made of small, swappable parts -- detector,
tracker, subject policy, framing rules, controller, output backend -- so a new
camera, a new motor, or a new model is one class, not a fork.
"""

from .config import AppConfig, BackendConfig, RuntimeConfig, TrackerConfig
from .control import AxisConfig, AxisController
from .detectors import DetectorConfig, build_detector
from .framing import Composer, FramingConfig
from .kalman import BoxKalman
from .outputs import OutputConfig, build_sinks
from .server import ServerConfig
from .sources import SourceConfig, VideoSource
from .subject import SubjectPolicy, SubjectSelector
from .tracker import MultiTracker, Track
from .types import Box, Detection, FramingError, Rect
from .zones import ZoneConfig, Zones

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "AxisConfig",
    "AxisController",
    "BackendConfig",
    "Box",
    "BoxKalman",
    "Composer",
    "Detection",
    "DetectorConfig",
    "FollowPipeline",
    "FramingConfig",
    "FramingError",
    "MultiTracker",
    "OutputConfig",
    "Rect",
    "RuntimeConfig",
    "ServerConfig",
    "SourceConfig",
    "SubjectPolicy",
    "SubjectSelector",
    "Track",
    "TrackerConfig",
    "VideoSource",
    "ZoneConfig",
    "Zones",
    "build_detector",
    "build_sinks",
    "__version__",
]


def __getattr__(name: str):
    # Imported lazily: pulling in the pipeline drags in the display stack, which
    # is unwanted on a headless box that only wants the control primitives.
    if name == "FollowPipeline":
        from .pipeline import FollowPipeline

        return FollowPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
