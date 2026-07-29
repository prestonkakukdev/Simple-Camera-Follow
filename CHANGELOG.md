# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [semantic versioning](https://semver.org/).

## [0.1.0] — unreleased

First public release. Alpha: the control loop is tested in simulation and the
plumbing runs end to end, but it has not yet been validated across long
real-world sessions. Field reports welcome.

### Added

- **Tracking pipeline** — YOLO or HOG person detection, IoU + Kalman multi-object
  tracking with stable IDs, sticky subject selection, and cinematographic
  framing (headroom, lead room, shot size).
- **Latency compensation** — the framing stage aims `lead_time` seconds ahead
  using the Kalman velocity estimate, cancelling detector and actuator delay.
- **Motion controller** — dead zone with hysteresis, smoothstep soft entry,
  filtered derivative, and slew limiting, so movement eases in and out and never
  fidgets.
- **Asynchronous detection** — inference runs on its own thread; the control
  loop runs at full capture rate and never blocks on it.
- **Camera backends** — digital PTZ (crop window, no motors), VISCA over IP and
  serial, and a DIY serial head with matching Arduino firmware.
- **Output sinks** — virtual camera (Zoom/OBS/Meet/Teams), ffmpeg streaming
  (RTMP/SRT/RTSP/HLS/UDP), and file recording.
- **Tracking zones** — normalised include/exclude polygons with a configurable
  test anchor, gating eligibility without disturbing track identity.
- **Control API** — optional, off-by-default JSON HTTP API for headless rigs:
  status, subject lock, home, JPEG/MJPEG diagnostic frames, and live config
  overrides that apply on the next frame without a restart. No web interface;
  `GET /` lists the endpoints.
- **Source resilience** — automatic reconnection with backoff plus a watchdog
  that catches streams which freeze without erroring.
- **`camerafollow doctor`** — environment self-check for dependencies, compute
  device, ffmpeg, and capture devices.
- **Closed-loop test suite** — simulates a subject walking a stage and asserts
  the camera holds the shot, doesn't oscillate, doesn't jerk between frames,
  stays still when the subject does, coasts through occlusions, and measurably
  benefits from latency compensation.

### Notes for anyone reading the history

Four real bugs were caught by the closed-loop tests during development and are
worth knowing about, because they are easy to reintroduce:

- A raw derivative term produced visible chatter on every direction change;
  each new detection steps the Kalman state, and dividing that step by `dt`
  spikes the derivative. Fixed with a filtered derivative (`kd_tau`).
- `coast_to_stop` decayed exponentially and never reached zero, leaving the head
  creeping too slowly to see but fast enough to keep motors energised.
- Track lifetime was counted in detection cycles while the subject hold window
  was in seconds, so a configured 2.5 s hold silently became 1.55 s. Track
  lifetime is now `max_age`, in seconds.
- The subject selector initialised its "last seen" timestamp to `0.0` rather
  than the current time, so a subject occluded immediately after acquisition
  lost its lock instantly.
