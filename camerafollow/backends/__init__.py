"""Output backends -- where the motion command actually goes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BaseBackend, PTZBackend

__all__ = ["BaseBackend", "PTZBackend", "build_backend"]


def build_backend(kind: str, options: Mapping[str, Any] | None = None) -> BaseBackend:
    """Construct a backend by name.

    ``virtual``       digital PTZ, crops a moving window out of a wide frame
    ``visca``         any VISCA PTZ head, over IP or serial
    ``serial``        DIY motorised head running the bundled firmware
    ``null``          accepts and discards commands (dry runs, CI)
    """
    opts = dict(options or {})
    kind = kind.lower()

    if kind in ("virtual", "digital", "crop"):
        from .virtual import VirtualConfig, VirtualPTZ

        return VirtualPTZ(VirtualConfig(**opts))

    if kind in ("visca", "ptz"):
        from .visca import ViscaBackend, ViscaConfig

        return ViscaBackend(ViscaConfig(**opts))

    if kind in ("serial", "arduino", "diy"):
        from .serial_head import SerialHead, SerialHeadConfig

        return SerialHead(SerialHeadConfig(**opts))

    if kind in ("null", "none", "dry"):
        return NullBackend()

    raise ValueError(f"unknown backend {kind!r} (expected virtual, visca, serial, or null)")


class NullBackend(BaseBackend):
    name = "null"

    def __init__(self) -> None:
        self.last = (0.0, 0.0, 0.0)

    def move(self, pan: float, tilt: float, zoom: float, dt: float) -> None:
        self.last = (pan, tilt, zoom)

    def stop(self) -> None:
        self.last = (0.0, 0.0, 0.0)
