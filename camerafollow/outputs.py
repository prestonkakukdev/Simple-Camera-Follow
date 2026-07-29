"""Where the framed picture goes.

Without this module the digital-PTZ mode is a demo: it can auto-frame beautifully
and there is no way to get the result into OBS, Zoom, a switcher, or a stream.
Three sinks cover essentially every destination:

* ``VirtualCameraSink`` -- the framed output appears as a webcam. Zoom, Teams,
  Meet, OBS, Discord and every browser pick it up with no configuration. This is
  the one most people want, and it is what makes a static camera plus this
  software a complete replacement for an operator.
* ``FFmpegSink`` -- pipe to ffmpeg, which reaches RTMP (YouTube, Twitch), SRT,
  RTSP, HLS, UDP multicast, NDI via OBS, or any file format. One subprocess, no
  Python dependency beyond an ffmpeg binary on PATH.
* ``FileSink`` -- plain recording via OpenCV, for review and offline tuning.

Every sink runs its writer on a background thread with a tiny drop-oldest queue.
Encoding is slow and bursty; a sink that blocks the control loop would turn a
network hiccup into visible camera stutter. Dropping a frame is always better
than delaying a motor command.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Protocol

import cv2
import numpy as np


@dataclass
class OutputConfig:
    #: Record the framed output to a file.
    file: str | None = None
    #: Publish as a system webcam. Needs the `camera` extra (pyvirtualcam), plus
    #: OBS installed on Windows/macOS or v4l2loopback on Linux.
    virtual_camera: bool = False
    #: Give the virtual camera a specific device (Linux: /dev/video10).
    virtual_camera_device: str | None = None
    #: Any ffmpeg destination: rtmp://, srt://, udp://, rtsp://, or a file path.
    stream: str | None = None
    #: Container. Inferred from the URL scheme when left unset.
    stream_format: str | None = None
    stream_bitrate: str = "6M"
    #: x264 preset. "veryfast" is the right default for live; "medium" if you
    #: are writing a file and have CPU to spare.
    stream_preset: str = "veryfast"
    stream_audio: str | None = None  # optional ffmpeg input for audio, e.g. a device
    ffmpeg_path: str = "ffmpeg"
    #: Extra ffmpeg arguments inserted before the destination.
    ffmpeg_args: list[str] = field(default_factory=list)
    #: Frame rate declared to sinks. None => inherit the source rate.
    #: This must match reality: a 60fps source written with 30fps metadata
    #: plays back at double speed.
    fps: int | None = None


class Sink(Protocol):
    name: str

    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


def build_sinks(cfg: OutputConfig) -> list[Sink]:
    sinks: list[Sink] = []
    fps = cfg.fps or 30
    if cfg.file:
        sinks.append(FileSink(cfg.file, fps=fps))
    if cfg.virtual_camera:
        sinks.append(VirtualCameraSink(fps=fps, device=cfg.virtual_camera_device))
    if cfg.stream:
        sinks.append(FFmpegSink(cfg))
    return sinks


class _ThreadedSink:
    """Base: hands frames to a worker thread, dropping the oldest under load.

    ``depth`` sets how much backlog is tolerated, and the right value differs by
    destination. A live output wants a shallow queue: showing the present
    matters more than showing every frame, and a deep queue just adds latency.
    A recording wants a deep one: dropped frames are lost forever, and a brief
    disk stall should be absorbed rather than punched through the file.
    """

    name = "sink"

    def __init__(self, depth: int = 2) -> None:
        self._queue: Queue[np.ndarray] = Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self.dropped = 0
        self.failed: str | None = None

    def write(self, frame: np.ndarray) -> None:
        if self.failed is not None:
            return
        if not self._started:
            self._started = True
            self._thread.start()
        try:
            self._queue.put_nowait(frame)
        except Full:
            # Drop the oldest rather than the newest: a live output should show
            # the present, not work through a backlog of the past.
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except Full:
                self.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._consume(frame)
            except Exception as exc:
                self.failed = f"{type(exc).__name__}: {exc}"
                return

    def _consume(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=2.0)
        self._teardown()

    def _teardown(self) -> None:
        pass


class FileSink(_ThreadedSink):
    name = "file"

    def __init__(self, path: str, *, fps: int = 30, fourcc: str = "mp4v") -> None:
        # ~8 seconds of slack at 30fps, so a slow disk or a spinning-up encoder
        # costs latency in the file rather than missing frames.
        super().__init__(depth=240)
        self.path = path
        self.fps = fps
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None

    def _consume(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open {self.path} for writing")
        self._writer.write(frame)

    def _teardown(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class VirtualCameraSink(_ThreadedSink):
    """Publishes the framed output as a system webcam."""

    name = "virtual-camera"

    def __init__(self, *, fps: int = 30, device: str | None = None) -> None:
        super().__init__(depth=2)
        self.fps = fps
        self.device = device
        self._cam = None

    def _consume(self, frame: np.ndarray) -> None:
        if self._cam is None:
            try:
                import pyvirtualcam
                from pyvirtualcam import PixelFormat
            except ImportError as exc:
                raise RuntimeError(
                    "virtual camera needs pyvirtualcam:  pip install 'camerafollow[camera]'\n"
                    "  Windows/macOS: install OBS Studio (it provides the virtual camera driver)\n"
                    "  Linux: sudo modprobe v4l2loopback"
                ) from exc
            h, w = frame.shape[:2]
            kwargs = {"width": w, "height": h, "fps": self.fps, "fmt": PixelFormat.BGR}
            if self.device:
                kwargs["device"] = self.device
            self._cam = pyvirtualcam.Camera(**kwargs)
        # No sleep_until_next_frame here: the source dictates our timing, and
        # sleeping would couple output pacing to the control loop.
        self._cam.send(frame)

    def _teardown(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    @property
    def device_name(self) -> str | None:
        return getattr(self._cam, "device", None)


class FFmpegSink(_ThreadedSink):
    """Pipes raw frames to ffmpeg. Reaches RTMP, SRT, RTSP, HLS, UDP, or a file."""

    name = "ffmpeg"

    #: URL scheme -> container format ffmpeg should mux into.
    _FORMATS = {
        "rtmp": "flv",
        "rtmps": "flv",
        "srt": "mpegts",
        "udp": "mpegts",
        "rtp": "rtp_mpegts",
        "rtsp": "rtsp",
    }

    def __init__(self, cfg: OutputConfig) -> None:
        # Live by default; a file destination gets recording semantics.
        live = "://" in (cfg.stream or "")
        super().__init__(depth=4 if live else 240)
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None

    def _format_for(self, dest: str) -> str | None:
        if self.cfg.stream_format:
            return self.cfg.stream_format
        scheme = dest.split("://", 1)[0].lower() if "://" in dest else ""
        return self._FORMATS.get(scheme)

    def _build_command(self, width: int, height: int) -> list[str]:
        cfg = self.cfg
        fps = cfg.fps or 30
        dest = cfg.stream or ""
        cmd = [
            cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(cfg.fps),
            "-i",
            "-",
        ]
        if cfg.stream_audio:
            cmd += ["-f", "lavfi", "-i", cfg.stream_audio, "-c:a", "aac", "-b:a", "128k"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            cfg.stream_preset,
            # zerolatency stops x264 buffering frames for lookahead, which is
            # what makes a stream lag seconds behind the room.
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            cfg.stream_bitrate,
            "-maxrate",
            cfg.stream_bitrate,
            "-bufsize",
            cfg.stream_bitrate,
            # Keyframe every two seconds: what most CDNs and switchers expect.
            "-g",
            str(max(1, fps * 2)),
        ]
        cmd += list(cfg.ffmpeg_args)
        fmt = self._format_for(dest)
        if fmt:
            cmd += ["-f", fmt]
        cmd += [dest]
        return cmd

    def _consume(self, frame: np.ndarray) -> None:
        if self._proc is None:
            if shutil.which(self.cfg.ffmpeg_path) is None:
                raise RuntimeError(
                    f"{self.cfg.ffmpeg_path!r} not found on PATH. Install ffmpeg "
                    "(macOS: brew install ffmpeg, Debian/Ubuntu: apt install ffmpeg, "
                    "Windows: winget install ffmpeg)."
                )
            h, w = frame.shape[:2]
            self._proc = subprocess.Popen(
                self._build_command(w, h),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited with code {self._proc.returncode}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def _teardown(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=5.0)
        except Exception:
            self._proc.kill()
        self._proc = None
