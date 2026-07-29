"""The head/motor abstraction.

Everything above this line is hardware-agnostic. To support a new rig you
implement one class with ``move()`` and, if it changes what part of the sensor
is being shown, ``view_rect()``.

Velocities are normalised to ``[-1, 1]``:
  * pan  -- positive = camera turns right (subject appears to move left)
  * tilt -- positive = camera turns up
  * zoom -- positive = zoom in (tele)

Backends are responsible for mapping that onto their own speed scale, and for
failing safe: if the controller stops calling ``move()``, the rig must stop.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Rect


@runtime_checkable
class PTZBackend(Protocol):
    def move(self, pan: float, tilt: float, zoom: float, dt: float) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class BaseBackend:
    """Convenience base with sane defaults for the optional hooks."""

    name = "base"

    def move(self, pan: float, tilt: float, zoom: float, dt: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.move(0.0, 0.0, 0.0, 0.0)

    def view_rect(self, frame_shape) -> Rect:
        """What part of the source frame is being output.

        Full frame for anything that physically moves the camera. Overridden by
        digital-PTZ backends, where framing error must be measured against the
        crop window and not the sensor.
        """
        return Rect.full(frame_shape)

    def render(self, frame):
        """Transform the source frame into the program output. Identity by default."""
        return frame

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
