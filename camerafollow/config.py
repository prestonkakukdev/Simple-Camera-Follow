"""Configuration: one YAML file describes an entire rig.

Every tunable lives here so a deployment is a config file, not a fork. The same
binary runs a 4K digital-PTZ auto-frame on a laptop and a motorised VISCA head
in a sanctuary; only the YAML differs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from .control import AxisConfig
from .detectors import DetectorConfig
from .framing import FramingConfig
from .outputs import OutputConfig
from .server import ServerConfig
from .sources import SourceConfig
from .subject import SubjectPolicy
from .zones import ZoneConfig


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.25
    #: Seconds a track survives unseen before it is dropped. Must be >=
    #: subject.hold_seconds, or the tracker culls a locked subject before the
    #: selector's hold window expires and the configured hold never happens.
    max_age: float = 3.0
    #: Detections needed before a track is trusted. Suppresses one-frame ghosts.
    min_hits: int = 3


@dataclass
class BackendConfig:
    kind: str = "virtual"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    #: Seconds ahead the framing stage aims. Set this to your measured
    #: end-to-end latency (detector + link + motor spin-up). See README.
    lead_time: float = 0.12
    #: Cap on detector rate. None => as fast as the hardware manages.
    detect_rate: float | None = 20.0
    #: Cap on the control loop. Should comfortably exceed the detector rate.
    control_rate: float = 60.0
    show_preview: bool = True
    #: Draw boxes, IDs, the target reticle and the dead zone on the preview.
    show_overlay: bool = True
    #: Seconds with no subject before the head returns to its home position.
    return_home_after: float | None = None
    #: Seconds the source may be disconnected before the head parks itself.
    park_after_disconnect: float = 2.0


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    subject: SubjectPolicy = field(default_factory=SubjectPolicy)
    framing: FramingConfig = field(default_factory=FramingConfig)
    pan: AxisConfig = field(default_factory=AxisConfig)
    tilt: AxisConfig = field(default_factory=lambda: AxisConfig(kp=1.2, max_velocity=0.6))
    zoom: AxisConfig = field(
        default_factory=lambda: AxisConfig(
            deadzone=0.15, kp=0.8, kd=0.0, max_velocity=0.35, max_acceleration=0.6
        )
    )
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # -- io -----------------------------------------------------------------
    @staticmethod
    def load(path: str | Path) -> AppConfig:
        raw = _read_mapping(Path(path))
        return AppConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> AppConfig:
        return _build(AppConfig, raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def apply_overrides(self, overrides: Mapping[str, Any]) -> AppConfig:
        """Apply dotted-key overrides, e.g. ``{"pan.kp": 2.0}``."""
        for dotted, value in overrides.items():
            target: Any = self
            parts = dotted.split(".")
            for part in parts[:-1]:
                if not hasattr(target, part):
                    raise KeyError(f"unknown config section {dotted!r}")
                target = getattr(target, part)
            leaf = parts[-1]
            if isinstance(target, dict):
                target[leaf] = value
            elif hasattr(target, leaf):
                setattr(target, leaf, _coerce(type(getattr(target, leaf)), value))
            else:
                raise KeyError(f"unknown config key {dotted!r}")
        return self


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    import json

    return json.loads(text)


def _build(cls, raw: Mapping[str, Any]):
    if not isinstance(raw, Mapping):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(raw).__name__}")
    # Silently ignoring an unknown key is the worst possible behaviour here:
    # you lose an afternoon wondering why `deadzone` did nothing, when you
    # actually typed `dead_zone`. Unknown keys are a hard error.
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(
            f"unknown key(s) in {cls.__name__}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    # `from __future__ import annotations` makes f.type a string, so resolve
    # the real types rather than guessing from field names.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        hint = hints.get(name)
        if is_dataclass(hint) and isinstance(value, Mapping):
            kwargs[name] = _build(hint, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce(target_type, value):
    if value is None or target_type is type(None):
        return value
    try:
        if target_type is bool and isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if target_type in (int, float, str, bool):
            return target_type(value)
    except (TypeError, ValueError):
        pass
    return value
