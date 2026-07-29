"""Video source: reconnection and lifecycle.

Uses generated video files rather than hardware, so these run anywhere.
"""

import time

import cv2
import numpy as np
import pytest

from camerafollow.sources import SourceConfig, VideoSource


@pytest.fixture
def clip(tmp_path):
    def make(name="clip.mp4", frames=30, w=160, h=120):
        path = tmp_path / name
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for i in range(frames):
            f = np.full((h, w, 3), i * 3 % 255, np.uint8)
            vw.write(f)
        vw.release()
        return path

    return make


def test_reads_frames_from_a_file(clip):
    # loop=True, or the reader races to EOF and disconnects before we look
    src = VideoSource(SourceConfig(source=str(clip()), loop=True, fps=30, fourcc=None))
    with src:
        frame, ts = src.read()
        assert frame is not None and ts > 0
        assert src.connected
        assert frame.shape[:2] == (120, 160)


def test_missing_source_fails_immediately_with_guidance(tmp_path):
    """A source that never worked is a config error, not something to retry."""
    src = VideoSource(SourceConfig(source=str(tmp_path / "nope.mp4"), fourcc=None))
    with pytest.raises(RuntimeError) as exc:
        src.open()
    assert "could not open" in str(exc.value)
    assert "camerafollow devices" in str(exc.value)


def test_file_end_finishes_rather_than_reconnecting(clip):
    src = VideoSource(SourceConfig(source=str(clip(frames=10)), loop=False, fps=60, fourcc=None))
    with src:
        deadline = time.perf_counter() + 5
        while src.running and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert not src.running
        assert src.reconnects == 0


def test_looping_keeps_the_source_running(clip):
    src = VideoSource(SourceConfig(source=str(clip(frames=5)), loop=True, fps=60, fourcc=None))
    with src:
        time.sleep(0.5)
        assert src.running and src.connected


def test_only_the_newest_frame_is_kept(clip):
    """Queued frames become control latency, which becomes overshoot."""
    src = VideoSource(SourceConfig(source=str(clip(frames=200)), loop=True, fourcc=None))
    with src:
        time.sleep(0.3)
        assert src.age < 0.5


def test_read_before_any_frame_returns_none():
    src = VideoSource(SourceConfig(source=0))
    assert src.read() == (None, 0.0)
    assert src.age == float("inf")


def test_close_is_idempotent(clip):
    src = VideoSource(SourceConfig(source=str(clip()), fourcc=None)).open()
    src.close()
    src.close()
    assert not src.connected


class _FlakyCapture:
    """A capture that delivers `good` frames, then fails forever."""

    def __init__(self, good=5):
        self.good = good
        self.released = False

    def read(self):
        if self.good > 0:
            self.good -= 1
            return True, np.zeros((10, 10, 3), np.uint8)
        return False, None

    def isOpened(self):
        return True

    def release(self):
        self.released = True

    def set(self, *a):
        return True


def test_source_reconnects_after_a_failure(monkeypatch):
    """A yanked cable or a camera reboot must not end the session."""
    opened = []

    def fake_open(self):
        cap = _FlakyCapture(good=5)
        opened.append(cap)
        return cap

    monkeypatch.setattr(VideoSource, "_open_capture", fake_open)
    src = VideoSource(
        SourceConfig(
            source="rtsp://fake/stream",
            reconnect=True,
            reconnect_delay=0.05,
            reconnect_max_delay=0.1,
            stall_timeout=None,
        )
    )
    with src:
        deadline = time.perf_counter() + 3
        while src.reconnects < 2 and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert src.reconnects >= 2, "source never reconnected"
        assert src.running, "pipeline would have exited"
        assert all(c.released for c in opened[:-1]), "leaked a capture handle"


def test_reconnect_disabled_finishes_instead_of_retrying(monkeypatch):
    monkeypatch.setattr(VideoSource, "_open_capture", lambda self: _FlakyCapture(good=3))
    src = VideoSource(
        SourceConfig(source="rtsp://fake/stream", reconnect=False, stall_timeout=None)
    )
    with src:
        deadline = time.perf_counter() + 3
        while src.running and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert not src.running
        assert src.reconnects == 0


def test_last_good_frame_is_served_while_disconnected(monkeypatch):
    """Callers stay in their loop; `connected` tells them not to act on it."""
    monkeypatch.setattr(VideoSource, "_open_capture", lambda self: _FlakyCapture(good=2))
    src = VideoSource(
        SourceConfig(source="rtsp://fake/stream", reconnect=False, stall_timeout=None)
    )
    with src:
        time.sleep(0.4)
        frame, ts = src.read()
        assert frame is not None and ts > 0  # still readable
        assert not src.connected  # but explicitly stale
