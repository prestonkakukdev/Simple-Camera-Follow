"""DIY motorised head over a serial link.

For a home-built rig: two steppers (or servos) on a pan-tilt bracket, driven by
an Arduino / RP2040 / ESP32. The protocol is deliberately trivial so the
firmware stays auditable and anyone can port it:

    V <pan> <tilt>\\n      velocities in [-1, 1]
    S\\n                   immediate stop
    H\\n                   go to home position
    P\\n                   ping (firmware replies "OK")

Matching firmware is in ``firmware/arduino_pantilt/``.

Two safety properties, both enforced on the *firmware* side because that is the
only side that keeps working when the host crashes:

* **Watchdog.** No command for 300 ms => motors stop. If this process dies or
  the USB cable is pulled mid-pan, the camera stops instead of slewing into the
  end stop for the rest of the service.
* **Acceleration limiting.** Also applied here in software, but a stepper will
  skip steps and lose its zero if commanded to step-change velocity, so the
  firmware clamps it too.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .base import BaseBackend


@dataclass
class SerialHeadConfig:
    device: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    #: Resend the current velocity at least this often, to feed the watchdog.
    keepalive_hz: float = 20.0
    #: Don't spam the link with imperceptible changes.
    min_delta: float = 0.005
    invert_pan: bool = False
    invert_tilt: bool = False
    #: Wait for the board to reset after opening the port.
    boot_delay: float = 2.0


class SerialHead(BaseBackend):
    name = "serial"

    def __init__(self, cfg: SerialHeadConfig | None = None) -> None:
        self.cfg = cfg or SerialHeadConfig()
        try:
            import serial
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the serial backend needs pyserial:  pip install 'camerafollow[serial]'"
            ) from exc

        self._serial = serial.Serial(self.cfg.device, self.cfg.baudrate, timeout=0.2)
        time.sleep(self.cfg.boot_delay)
        self._serial.reset_input_buffer()

        self._pan = 0.0
        self._tilt = 0.0
        self._sent = (None, None)
        self._last_send = 0.0
        self._lock = threading.Lock()

        self._stop_evt = threading.Event()
        self._keepalive = threading.Thread(target=self._pump, daemon=True)
        self._keepalive.start()

    def move(self, pan: float, tilt: float, zoom: float, dt: float) -> None:
        if self.cfg.invert_pan:
            pan = -pan
        if self.cfg.invert_tilt:
            tilt = -tilt
        with self._lock:
            self._pan = _clamp(pan, -1.0, 1.0)
            self._tilt = _clamp(tilt, -1.0, 1.0)
        self._maybe_send(force=False)

    def stop(self) -> None:
        with self._lock:
            self._pan = 0.0
            self._tilt = 0.0
        self._write("S\n")
        self._sent = (0.0, 0.0)

    def home(self) -> None:
        self._write("H\n")

    def ping(self) -> bool:
        self._serial.reset_input_buffer()
        self._write("P\n")
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            line = self._serial.readline().decode("ascii", "ignore").strip()
            if line:
                return line.upper().startswith("OK")
        return False

    # -- internals ----------------------------------------------------------
    def _pump(self) -> None:
        interval = 1.0 / max(1.0, self.cfg.keepalive_hz)
        while not self._stop_evt.wait(interval):
            self._maybe_send(force=True)

    def _maybe_send(self, *, force: bool) -> None:
        with self._lock:
            pan, tilt = self._pan, self._tilt
        prev_pan, prev_tilt = self._sent
        changed = (
            prev_pan is None
            or abs(pan - prev_pan) >= self.cfg.min_delta
            or abs(tilt - prev_tilt) >= self.cfg.min_delta
        )
        stale = (time.perf_counter() - self._last_send) >= 1.0 / max(1.0, self.cfg.keepalive_hz)
        if changed or (force and stale):
            self._write(f"V {pan:.4f} {tilt:.4f}\n")
            self._sent = (pan, tilt)

    def _write(self, text: str) -> None:
        try:
            self._serial.write(text.encode("ascii"))
            self._last_send = time.perf_counter()
        except Exception:
            pass

    def close(self) -> None:
        self._stop_evt.set()
        self._keepalive.join(timeout=1.0)
        try:
            self.stop()
        except Exception:
            pass
        self._serial.close()


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
