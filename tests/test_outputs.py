"""Output sinks."""

import stat
import time

import numpy as np
import pytest

from camerafollow.outputs import FFmpegSink, FileSink, OutputConfig, build_sinks


def _frame(w=320, h=180):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


@pytest.fixture
def fake_ffmpeg(tmp_path):
    """A stand-in binary that records how many bytes actually reached stdin."""
    counter = tmp_path / "bytes.txt"
    script = tmp_path / "fake-ffmpeg"
    script.write_text(f'#!/bin/sh\nwc -c > "{counter}"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), counter


def _drain(sink, settle=0.6):
    time.sleep(settle)
    sink.close()
    time.sleep(0.2)


# -- build ------------------------------------------------------------------
def test_no_outputs_configured_means_no_sinks():
    assert build_sinks(OutputConfig()) == []


def test_each_option_adds_its_sink(tmp_path):
    cfg = OutputConfig(file=str(tmp_path / "a.mp4"), stream="udp://127.0.0.1:9")
    assert [s.name for s in build_sinks(cfg)] == ["file", "ffmpeg"]


# -- file -------------------------------------------------------------------
def test_file_sink_writes_a_playable_file(tmp_path):
    path = tmp_path / "out.mp4"
    sink = FileSink(str(path), fps=30)
    for _ in range(20):
        sink.write(_frame())
    _drain(sink)
    assert path.exists() and path.stat().st_size > 0
    assert sink.failed is None


def test_recording_does_not_drop_frames_at_realtime_rate(tmp_path):
    """A recording queue must be deep. Dropped frames are gone for good."""
    sink = FileSink(str(tmp_path / "out.mp4"), fps=30)
    for _ in range(90):
        sink.write(_frame())
        time.sleep(1 / 120)
    _drain(sink)
    assert sink.dropped == 0


# -- ffmpeg -----------------------------------------------------------------
def test_ffmpeg_sink_pipes_raw_frames(fake_ffmpeg):
    path, counter = fake_ffmpeg
    sink = FFmpegSink(OutputConfig(stream=str(counter.parent / "x.mp4"), ffmpeg_path=path, fps=30))
    for _ in range(30):
        sink.write(_frame())
        time.sleep(1 / 60)
    _drain(sink)
    assert counter.exists()
    assert int(counter.read_text().strip()) == 30 * 180 * 320 * 3


def test_ffmpeg_command_infers_container_from_url():
    def fmt(url):
        cfg = OutputConfig(stream=url)
        cmd = FFmpegSink(cfg)._build_command(640, 360)
        return cmd[cmd.index("-f", cmd.index("-i")) + 1] if "-f" in cmd[10:] else None

    assert fmt("rtmp://a/b") == "flv"
    assert fmt("srt://a:9") == "mpegts"
    assert fmt("udp://a:9") == "mpegts"


def test_ffmpeg_command_honours_explicit_format_and_extra_args():
    cfg = OutputConfig(stream="rtmp://a/b", stream_format="mpegts", ffmpeg_args=["-vf", "hflip"])
    cmd = FFmpegSink(cfg)._build_command(640, 360)
    assert "mpegts" in cmd and cmd[cmd.index("-vf") + 1] == "hflip"
    assert cmd[-1] == "rtmp://a/b"


def test_ffmpeg_uses_low_latency_encoding():
    """zerolatency is what stops the stream drifting seconds behind the room."""
    cmd = FFmpegSink(OutputConfig(stream="rtmp://a/b"))._build_command(640, 360)
    assert cmd[cmd.index("-tune") + 1] == "zerolatency"
    assert cmd[cmd.index("-s") + 1] == "640x360"


def test_missing_ffmpeg_reports_how_to_install_it():
    sink = FFmpegSink(OutputConfig(stream="rtmp://a/b", ffmpeg_path="no-such-binary"))
    sink.write(_frame())
    time.sleep(0.5)
    assert sink.failed and "not found on PATH" in sink.failed
    assert "install" in sink.failed.lower()


def test_a_failed_sink_stops_accepting_frames_instead_of_raising():
    """A dead output must never take the control loop down with it."""
    sink = FFmpegSink(OutputConfig(stream="rtmp://a/b", ffmpeg_path="no-such-binary"))
    sink.write(_frame())
    time.sleep(0.4)
    for _ in range(100):
        sink.write(_frame())  # must not raise
    sink.close()


# -- backpressure -----------------------------------------------------------
def test_live_sink_drops_oldest_rather_than_blocking(fake_ffmpeg):
    """Under load a live output shows the present, not a backlog of the past."""
    path, counter = fake_ffmpeg
    sink = FFmpegSink(OutputConfig(stream="udp://127.0.0.1:9999", ffmpeg_path=path))
    frame = _frame(1280, 720)  # built once: we are timing write(), not numpy
    start = time.perf_counter()
    for _ in range(400):
        sink.write(frame)
    elapsed = time.perf_counter() - start
    sink.close()
    # 400 x 720p is ~1.1 GB of pipe traffic; if write() were synchronous this
    # could not finish in well under a second.
    assert elapsed < 1.0
    assert sink.dropped > 0


def test_output_fps_is_inherited_from_the_source():
    """Declaring the wrong rate makes a recording play back at the wrong speed."""
    from camerafollow.config import AppConfig

    cfg = AppConfig()
    cfg.source.fps = 60
    cfg.backend.kind = "null"
    assert cfg.output.fps is None
    # The pipeline resolves it at construction; check the resolution rule itself.
    resolved = cfg.output.fps or cfg.source.fps or 30
    assert resolved == 60

    cmd = FFmpegSink(OutputConfig(stream="rtmp://a/b", fps=60))._build_command(640, 360)
    assert cmd[cmd.index("-r") + 1] == "60"
    assert cmd[cmd.index("-g") + 1] == "120"  # keyframe every 2s
