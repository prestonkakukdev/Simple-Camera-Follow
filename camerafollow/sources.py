"""Video input.

One OpenCV-backed source covers every ingest path that matters:

* **HDMI capture card** (Elgato Cam Link, Magewell, cheap MS2109 dongles) --
  these enumerate as a plain UVC device, so ``source: 0`` just works.
* **USB / CSI camera** -- same path.
* **Network stream** -- ``rtsp://``, ``http://``, ``srt://``, or an SDP file.
  Most IP/PTZ cameras and NDI-to-RTSP bridges land here.
* **Video file** -- for development and for tuning gains offline against a
  recording of the real thing. Do this before you touch hardware.

Two properties matter for unattended operation:

**Only the newest frame is kept.** Never queue frames in a control loop: a
backlog becomes control latency, and control latency becomes overshoot.

**Disconnects are survivable.** A yanked HDMI cable, a camera reboot, or a
network blip must not end the session. The reader thread reconnects with
exponential backoff and the pipeline holds still until the picture returns. A
*frozen* stream is caught too -- an RTSP feed that stops delivering without
raising an error is a common and otherwise invisible failure.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2


@dataclass
class SourceConfig:
    #: Device index (``0``), file path, or stream URL.
    source: str | int = 0
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    #: Force a backend API, e.g. "avfoundation", "v4l2", "dshow", "ffmpeg".
    backend: str | None = None
    #: MJPG is worth forcing on USB capture cards; YUY2 caps you at ~5 fps at 1080p.
    fourcc: str | None = "MJPG"
    #: Loop video files (dev convenience).
    loop: bool = True
    #: Reopen the source automatically after a failure.
    reconnect: bool = True
    reconnect_delay: float = 1.0
    reconnect_max_delay: float = 15.0
    #: Treat the source as dead if no new frame arrives for this long. Catches
    #: streams that freeze without reporting an error. None disables the check.
    stall_timeout: float | None = 5.0


_BACKENDS = {
    "any": cv2.CAP_ANY,
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
}


class VideoSource:
    def __init__(self, cfg: SourceConfig | None = None) -> None:
        self.cfg = cfg or SourceConfig()
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._is_file = False
        self._finished = False
        self.connected = False
        self.reconnects = 0
        self.last_error: str | None = None

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> VideoSource:
        """Open the source. Raises if the *first* connection fails.

        Later failures are handled by reconnection -- but a source that was
        never reachable is a configuration problem, and the user needs to hear
        about it immediately rather than watch silent retries.
        """
        src = self._resolved_source()
        self._is_file = isinstance(src, str) and "://" not in src
        self._cap = self._open_capture()
        if self._cap is None:
            raise RuntimeError(
                f"could not open video source {self.cfg.source!r}. "
                "Run `camerafollow devices` to list capture devices, and check that no "
                "other application (OBS, Zoom, Teams) already has the device open."
            )

        self.connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        if self.cfg.stall_timeout is not None and not self._is_file:
            self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
            self._watchdog_thread.start()

        deadline = time.perf_counter() + 5.0
        while self._frame is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        if self._frame is None:
            self.close()
            raise RuntimeError(f"no frames from source {self.cfg.source!r} after 5s")
        return self

    def _resolved_source(self):
        src = self.cfg.source
        if isinstance(src, str) and src.isdigit():
            return int(src)
        return src

    def _open_capture(self) -> cv2.VideoCapture | None:
        cfg = self.cfg
        api = _BACKENDS.get((cfg.backend or "any").lower(), cv2.CAP_ANY)
        try:
            cap = cv2.VideoCapture(self._resolved_source(), api)
        except Exception as exc:
            self.last_error = str(exc)
            return None
        if not cap.isOpened():
            cap.release()
            return None

        if not self._is_file:
            if cfg.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            if cfg.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            if cfg.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            if cfg.fps:
                cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # -- reader thread ------------------------------------------------------
    def _reader(self) -> None:
        cfg = self.cfg
        interval = 1.0 / cfg.fps if (self._is_file and cfg.fps) else 0.0
        delay = cfg.reconnect_delay

        while not self._stop.is_set():
            if self._cap is None:
                if not self._reconnect_allowed():
                    return
                self._cap = self._open_capture()
                if self._cap is None:
                    self._sleep(delay)
                    delay = min(delay * 2, cfg.reconnect_max_delay)
                    continue
                self.connected = True
                self.reconnects += 1
                delay = cfg.reconnect_delay

            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.perf_counter()
                if interval:
                    self._sleep(interval)
                continue

            # A file that ran out is finished, not broken.
            if self._is_file:
                if cfg.loop and self._cap is not None:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._finished = True
                self.connected = False
                return

            self.last_error = "read failed"
            self._drop_capture()
            if not self._reconnect_allowed():
                self._finished = True
                return
            self._sleep(delay)
            delay = min(delay * 2, cfg.reconnect_max_delay)

    def _watchdog(self) -> None:
        """Kill a capture that has stopped delivering.

        Needed as a separate thread because the failure mode it catches is
        ``cap.read()`` *blocking* -- an RTSP peer that goes away over TCP can
        hang the reader for minutes. Releasing the capture from here unblocks
        it, and the reader loop then reconnects normally. A frozen feed is
        otherwise invisible: no error, no exception, just a camera that quietly
        stops following anything.
        """
        timeout = self.cfg.stall_timeout
        if timeout is None:
            return
        while not self._stop.wait(min(1.0, timeout / 2)):
            if self._is_file or not self.connected:
                continue
            if self.age > timeout:
                self.last_error = f"no frame for {timeout:.0f}s"
                self._drop_capture()

    def _reconnect_allowed(self) -> bool:
        return self.cfg.reconnect and not self._stop.is_set()

    def _drop_capture(self) -> None:
        self.connected = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(seconds)

    # -- consumer API -------------------------------------------------------
    def read(self):
        """Return ``(frame, capture_time)``.

        Keeps returning the last good frame while disconnected, so callers can
        stay in their loop; check :attr:`connected` to decide whether to act on
        it. Returns ``(None, 0.0)`` only before the first frame ever arrives.
        """
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame, self._frame_time

    @property
    def running(self) -> bool:
        """False once the source is permanently done (file ended, or gave up)."""
        return not self._finished and not self._stop.is_set()

    @property
    def age(self) -> float:
        """Seconds since the last frame arrived."""
        with self._lock:
            if self._frame_time == 0.0:
                return float("inf")
            return time.perf_counter() - self._frame_time

    def close(self) -> None:
        self._stop.set()
        for t in (self._thread, self._watchdog_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._thread = None
        self._watchdog_thread = None
        self._drop_capture()

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def list_devices(limit: int = 8) -> list[int]:
    """Probe device indices so users can find their capture card."""
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
