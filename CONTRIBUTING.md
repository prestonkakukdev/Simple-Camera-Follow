# Contributing

Thanks for looking. This project only becomes good with hardware and rooms the
maintainers don't have, so contributions of *evidence* are as valuable as
contributions of code.

## Setup

```bash
git clone https://github.com/prestonkakukdev/camerafollow && cd camerafollow
pip install -e '.[all,dev]'
pytest
camerafollow doctor
```

## The most useful things you can contribute

**1. A field report.** Which camera, which room, which config, what looked
wrong. Attach 30 seconds of footage if you can. Every framing default in this
repo is a guess that wants replacing with evidence.

**2. Support for hardware we can't buy.** A new PTZ protocol, a new capture
path, a new accelerator. See below — each is one class.

**3. Appearance-based re-identification.** Currently the biggest functional
gap: two people crossing with one fully hidden for over a second can transfer
the lock. `Track.appearance` is reserved for a colour histogram or embedding and
is unused. This is a well-scoped, high-impact piece of work.

## Adding things

Every extension point is a single class with a small interface.

**A camera head** — subclass `BaseBackend` in `camerafollow/backends/`,
implement `move(pan, tilt, zoom, dt)` with velocities normalised to ±1, and
register it in `backends/__init__.py`. Override `view_rect()` only if your
backend changes which part of the sensor is shown (as the digital-PTZ one does).

**A detector** — any object with `detect(frame) -> list[Detection]`. Add it to
`detectors.py` and to `build_detector`.

**An output** — subclass `_ThreadedSink` in `outputs.py` and implement
`_consume(frame)`. Pick your queue depth deliberately: shallow for live
destinations (show the present), deep for recordings (lose nothing).

**Not a GUI.** `server.py` is a JSON API for headless control and live tuning,
off by default, and it should stay that way. This is a component you wire into
a rig, not an application. Diagnostics belong in the director overlay, which
costs nothing when nobody is looking at it.

**A video source** — `sources.py` is one OpenCV-backed class covering devices,
files, and URLs. Genuinely different transports (NDI, GigE Vision) belong in a
sibling class exposing the same `read() -> (frame, timestamp)` contract.

## What we care about in review

**Never block the control loop.** Inference, encoding, network I/O, and disk all
run on their own threads. A frame delayed is a motor command delayed, and that
shows up on screen. If you add work to the loop, measure it.

**Motion quality is a feature, not a detail.** Changes to `control.py`,
`kalman.py`, or `framing.py` need to keep `tests/test_closed_loop.py` passing —
it simulates a subject walking a stage and asserts the camera holds the shot,
doesn't oscillate, doesn't jerk between frames, and stays still when the subject
does. Those assertions have caught four real bugs so far. If you change the
motion deliberately, change the assertions deliberately too, and say why.

**Fail loudly on config, gently on hardware.** An unknown config key is an
error with a list of valid keys — silently ignoring a typo costs someone an
afternoon. A camera that disconnects mid-session must never take the process
down.

**Units in names.** `max_age` is seconds, not frames. We had that bug: a hold
window configured as 2.5 s silently became 1.55 s because two layers counted in
different units. Frame counts mean different things on different machines.

## Testing

```bash
pytest                     # everything, ~15s
pytest tests/test_closed_loop.py -v    # the motion-quality suite
```

Tests must not need a camera, a motor, or a network. Use generated video files,
synthetic detections, and the stand-in binaries the existing tests use as
patterns — `tests/test_outputs.py` fakes ffmpeg with a shell script,
`tests/test_server.py` runs a real socket on port 0.

## Style

Ruff with the config in `pyproject.toml`; `ruff check` and `ruff format` before
you push. Comments should explain *why*, especially where a value was chosen by
feel — those are the lines that stop the next person "simplifying" a dead zone
out of existence.

## Reporting a bug

Include the output of `camerafollow doctor`, your config, and what the director
window showed. A 10-second screen recording of it is worth more than any
description — nearly every motion problem is legible there.
