"""VISCA head control, over IP (UDP) or serial.

VISCA is the lingua franca of broadcast PTZ. Sony, Panasonic, Canon, PTZOptics,
BirdDog, AVer, Marshall, Lumens, and essentially every cheap Chinese PTZ head
speak it. If your rig is a commodity PTZ camera, this is your backend -- and
note that you then need no capture card and no motor of your own: the camera
supplies both the picture (over RTSP/NDI/SDI) and the motion.

Two transports:
  * ``visca-ip``  -- UDP 52381, the modern default.
  * ``visca-serial`` -- RS-232/RS-485 at 9600 or 38400 baud, for older heads.

Pan/tilt drive:  ``8x 01 06 01 VV WW XX YY FF``
    VV pan speed 01-18, WW tilt speed 01-14
    XX  01 left / 02 right / 03 stop
    YY  01 up   / 02 down  / 03 stop
Zoom variable:   ``8x 01 04 07 2p`` tele, ``3p`` wide, ``00`` stop  (p = 0-7)
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

from .base import BaseBackend


@dataclass
class ViscaConfig:
    #: "ip" or "serial".
    transport: str = "ip"
    host: str = "192.168.1.100"
    port: int = 52381
    #: Serial only.
    device: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    #: VISCA camera address, 1-7.
    address: int = 1
    max_pan_speed: int = 0x18
    max_tilt_speed: int = 0x14
    max_zoom_speed: int = 7
    #: Below this the axis is commanded to stop rather than crawl. VISCA speed 1
    #: is often still too fast to be useful, so we gate rather than round to it.
    min_command: float = 0.02
    invert_pan: bool = False
    invert_tilt: bool = False


class ViscaBackend(BaseBackend):
    name = "visca"

    def __init__(self, cfg: ViscaConfig | None = None) -> None:
        self.cfg = cfg or ViscaConfig()
        self._seq = 0
        self._sock: socket.socket | None = None
        self._serial = None
        self._last: tuple[int, int, int, int] | None = None
        self._last_zoom: int | None = None

        if self.cfg.transport.lower() == "ip":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(0.2)
        else:
            try:
                import serial
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "serial VISCA needs pyserial:  pip install 'camerafollow[serial]'"
                ) from exc
            self._serial = serial.Serial(self.cfg.device, self.cfg.baudrate, timeout=0.2)

    # -- transport ----------------------------------------------------------
    def _send(self, payload: bytes) -> None:
        if self._sock is not None:
            header = struct.pack(">HHI", 0x0100, len(payload), self._seq)
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            self._sock.sendto(header + payload, (self.cfg.host, self.cfg.port))
        elif self._serial is not None:
            self._serial.write(payload)

    @property
    def _addr(self) -> int:
        return 0x80 | (self.cfg.address & 0x07)

    # -- commands -----------------------------------------------------------
    def move(self, pan: float, tilt: float, zoom: float, dt: float) -> None:
        cfg = self.cfg
        if cfg.invert_pan:
            pan = -pan
        if cfg.invert_tilt:
            tilt = -tilt

        pan_speed, pan_dir = _axis(pan, cfg.max_pan_speed, cfg.min_command, 0x01, 0x02)
        tilt_speed, tilt_dir = _axis(tilt, cfg.max_tilt_speed, cfg.min_command, 0x02, 0x01)

        state = (pan_speed, tilt_speed, pan_dir, tilt_dir)
        if state != self._last:
            self._send(
                bytes(
                    [self._addr, 0x01, 0x06, 0x01, pan_speed, tilt_speed, pan_dir, tilt_dir, 0xFF]
                )
            )
            self._last = state

        z = _zoom_byte(zoom, cfg.max_zoom_speed, cfg.min_command)
        if z != self._last_zoom:
            self._send(bytes([self._addr, 0x01, 0x04, 0x07, z, 0xFF]))
            self._last_zoom = z

    def stop(self) -> None:
        self._send(bytes([self._addr, 0x01, 0x06, 0x01, 0x01, 0x01, 0x03, 0x03, 0xFF]))
        self._send(bytes([self._addr, 0x01, 0x04, 0x07, 0x00, 0xFF]))
        self._last = None
        self._last_zoom = None

    def home(self) -> None:
        self._send(bytes([self._addr, 0x01, 0x06, 0x04, 0xFF]))

    def recall_preset(self, index: int) -> None:
        self._send(bytes([self._addr, 0x01, 0x04, 0x3F, 0x02, index & 0x7F, 0xFF]))

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        if self._sock is not None:
            self._sock.close()
        if self._serial is not None:
            self._serial.close()


def _axis(value: float, max_speed: int, min_command: float, neg_dir: int, pos_dir: int):
    """Map a normalised velocity onto (speed byte, direction byte)."""
    if abs(value) < min_command:
        return 0x01, 0x03  # speed is ignored when direction is "stop"
    speed = int(round(abs(value) * max_speed))
    speed = max(1, min(max_speed, speed))
    return speed, (pos_dir if value > 0 else neg_dir)


def _zoom_byte(value: float, max_speed: int, min_command: float) -> int:
    if abs(value) < min_command:
        return 0x00
    p = max(0, min(max_speed, int(round(abs(value) * max_speed))))
    return (0x20 | p) if value > 0 else (0x30 | p)
