"""Command line entry point."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .config import AppConfig


def main(argv: list[str] | None = None) -> int:
    # Line-buffer stdout. Python block-buffers when piped, so a headless rig
    # under systemd or `>> log` would show nothing until the buffer filled --
    # which looks exactly like a hang.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - exotic streams
        pass

    parser = argparse.ArgumentParser(
        prog="camerafollow",
        description="Open-source AI camera operator: detect, track, and follow a subject.",
        epilog="Docs and examples: https://github.com/prestonkakukdev/camerafollow",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the follow pipeline")
    run.add_argument("-c", "--config", help="path to a YAML config file")
    run.add_argument("-s", "--source", help="device index, file path, or stream URL")
    run.add_argument("-b", "--backend", help="virtual, visca, serial, or null")
    run.add_argument("-d", "--detector", help="yolo or hog")
    run.add_argument("--record", metavar="PATH", help="record the framed output to a file")
    run.add_argument(
        "--stream", metavar="URL", help="publish via ffmpeg (rtmp://, srt://, udp://…)"
    )
    run.add_argument(
        "--virtual-camera",
        action="store_true",
        help="publish as a webcam for Zoom/OBS/Meet/Teams",
    )
    run.add_argument(
        "--serve",
        nargs="?",
        const="127.0.0.1:8477",
        metavar="HOST:PORT",
        help="start the JSON control API for scripting and live tuning (default 127.0.0.1:8477)",
    )
    run.add_argument("--no-preview", action="store_true", help="run headless")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config value, e.g. --set pan.kp=2.0",
    )

    sub.add_parser("devices", help="list capture devices (find your HDMI card)")
    sub.add_parser("doctor", help="check this machine is set up correctly")

    init = sub.add_parser("init-config", help="write a starter config file")
    init.add_argument("path", nargs="?", default="camerafollow.yaml")

    ping = sub.add_parser("ping", help="check a DIY serial head is responding")
    ping.add_argument("device", help="serial device, e.g. /dev/ttyUSB0 or COM3")
    ping.add_argument("--baudrate", type=int, default=115200)

    args = parser.parse_args(argv)

    if args.command == "devices":
        return _devices()
    if args.command == "doctor":
        return _doctor()
    if args.command == "init-config":
        return _init_config(Path(args.path))
    if args.command == "ping":
        return _ping(args.device, args.baudrate)
    return _run(args)


def _run(args) -> int:
    cfg = AppConfig.load(args.config) if args.config else AppConfig()

    if args.source is not None:
        cfg.source.source = args.source
    if args.backend is not None:
        cfg.backend.kind = args.backend
    if args.detector is not None:
        cfg.detector.kind = args.detector
    if args.record is not None:
        cfg.output.file = args.record
    if args.stream is not None:
        cfg.output.stream = args.stream
    if args.virtual_camera:
        cfg.output.virtual_camera = True
    if args.no_preview:
        cfg.runtime.show_preview = False
    if args.serve is not None:
        cfg.server.enabled = True
        host, _, port = args.serve.rpartition(":")
        if host:
            cfg.server.host = host
        if port.isdigit():
            cfg.server.port = int(port)

    if args.set:
        try:
            cfg.apply_overrides(dict(_parse_kv(item) for item in args.set))
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    from .pipeline import FollowPipeline

    pipeline = FollowPipeline(cfg)
    outs = [s.name for s in pipeline.sinks] or ["preview only"]
    print(
        f"source={cfg.source.source}  detector={cfg.detector.kind}  "
        f"backend={cfg.backend.kind}  lead={cfg.runtime.lead_time * 1000:.0f}ms"
    )
    print(f"output: {', '.join(outs)}")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nstopping")
    return 0


def _parse_kv(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), _parse_scalar(raw.strip())


def _parse_scalar(raw: str) -> object:
    try:
        import yaml

        return yaml.safe_load(raw)
    except Exception:
        return raw


def _devices() -> int:
    from .sources import list_devices

    found = list_devices()
    if not found:
        print("no capture devices found.")
        print("HDMI capture cards enumerate as plain UVC devices; check the cable and")
        print("that no other application (OBS, Zoom, Teams) has the device open.")
        return 1
    print("available device indices:")
    for i in found:
        print(f"  {i}   ->  camerafollow run --source {i}")
    return 0


def _init_config(path: Path) -> int:
    if path.exists():
        print(f"refusing to overwrite existing {path}", file=sys.stderr)
        return 1
    AppConfig().save(path)
    print(f"wrote {path}")
    print("edit it, then:  camerafollow run -c", path)
    return 0


def _ping(device: str, baudrate: int) -> int:
    from .backends.serial_head import SerialHead, SerialHeadConfig

    head = SerialHead(SerialHeadConfig(device=device, baudrate=baudrate))
    try:
        ok = head.ping()
    finally:
        head.close()
    print("head responded" if ok else "no response from head")
    return 0 if ok else 1


# ---------------------------------------------------------------- doctor ----
_OK, _WARN, _BAD = "  ok  ", " warn ", " FAIL "


def _doctor() -> int:
    """Report everything a bug report needs. Run this first when something breaks."""
    print("camerafollow doctor\n")
    problems = 0

    print("core")
    py = sys.version_info
    _line(_OK if py >= (3, 10) else _BAD, "python", f"{py.major}.{py.minor}.{py.micro}")
    problems += py < (3, 10)
    for mod, label in (("numpy", "numpy"), ("cv2", "opencv"), ("yaml", "pyyaml")):
        problems += _report_import(mod, label, required=True)

    print("\ndetector")
    if _report_import("ultralytics", "ultralytics") == 0:
        try:
            import torch

            dev = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
            _line(_OK if dev != "cpu" else _WARN, "compute", dev)
            if dev == "cpu":
                print("         cpu inference works but caps your detector rate")
        except ImportError:
            _line(_BAD, "torch", "missing (ultralytics needs it)")
            problems += 1
    else:
        print("         install with:  pip install 'camerafollow[yolo]'")
        print("         or run --detector hog (no download, lower quality)")

    print("\noutputs")
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            v = subprocess.run(
                [ff, "-version"], capture_output=True, text=True, timeout=5
            ).stdout.splitlines()[0]
            _line(_OK, "ffmpeg", v.split(" Copyright")[0])
        except Exception:
            _line(_OK, "ffmpeg", ff)
    else:
        _line(_WARN, "ffmpeg", "not on PATH -- --stream unavailable")
        print("         macOS: brew install ffmpeg | Debian: apt install ffmpeg")
    if _report_import("pyvirtualcam", "virtual camera") != 0:
        print("         install with:  pip install 'camerafollow[camera]'")
        print("         plus OBS (Windows/macOS) or v4l2loopback (Linux)")

    print("\nmotor control")
    if _report_import("serial", "pyserial") != 0:
        print("         only needed for the serial and RS-232 VISCA backends")
        print("         install with:  pip install 'camerafollow[serial]'")

    print("\ncapture devices")
    try:
        from .sources import list_devices

        found = list_devices()
        if found:
            _line(_OK, "found", ", ".join(f"index {i}" for i in found))
        else:
            _line(_WARN, "found", "none -- fine if you use a file or a stream URL")
    except Exception as exc:
        _line(_BAD, "probe", str(exc))
        problems += 1

    print()
    if problems:
        print(f"{problems} problem(s) need attention before this will run.")
        return 1
    print("ready. try:  camerafollow run --source 0 --serve")
    return 0


def _line(status: str, label: str, detail: str) -> None:
    print(f"  [{status}] {label:<14} {detail}")


def _report_import(module: str, label: str, *, required: bool = False) -> int:
    try:
        mod = __import__(module)
    except ImportError:
        _line(_BAD if required else _WARN, label, "not installed")
        return 1
    _line(_OK, label, getattr(mod, "__version__", "installed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
