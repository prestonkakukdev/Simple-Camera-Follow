"""The control loop.

    capture -> (async) detect -> track -> select subject -> predict ahead
            -> frame -> control -> head, and -> output sinks

Three properties are what make this look professional rather than robotic, and
all three are easy to lose in a refactor:

1. The loop runs at capture rate and **never blocks** -- not on inference, not
   on encoding, not on the network. Detection and every output sink run on their
   own threads.
2. Between detections the subject position comes from the Kalman filter, and the
   framing stage aims ``lead_time`` seconds into the *future* to cancel
   end-to-end latency.
3. Losing the subject, or the whole camera, degrades gently: the head coasts to
   a stop and holds, then parks. It never snaps, and it never dies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from .backends import build_backend
from .config import AppConfig
from .control import AxisController
from .detect_worker import DetectorWorker
from .detectors import build_detector
from .framing import Composer
from .outputs import build_sinks
from .overlay import draw_director
from .server import ControlServer, SharedState
from .sources import VideoSource
from .subject import SubjectSelector
from .tracker import MultiTracker
from .types import FramingError, Rect
from .zones import Zones

_PROGRAM_WINDOW = "camerafollow :: program"
_DIRECTOR_WINDOW = "camerafollow :: director"


@dataclass
class Stats:
    control_fps: float = 0.0
    detect_fps: float = 0.0
    frames: int = 0
    detections: int = 0


class FollowPipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.stats = Stats()

        self.source = VideoSource(cfg.source)
        self.detector = build_detector(cfg.detector)
        self.worker = DetectorWorker(self.detector, max_rate=cfg.runtime.detect_rate)
        self.tracker = MultiTracker(
            iou_threshold=cfg.tracker.iou_threshold,
            max_age=cfg.tracker.max_age,
            min_hits=cfg.tracker.min_hits,
        )
        self.zones = Zones(cfg.zone)
        self.selector = SubjectSelector(cfg.subject, self.zones)
        self.composer = Composer(cfg.framing)
        self.pan = AxisController(cfg.pan)
        self.tilt = AxisController(cfg.tilt)
        self.zoom = AxisController(cfg.zoom)
        self.backend = build_backend(cfg.backend.kind, cfg.backend.options)

        cfg.output.fps = cfg.output.fps or cfg.source.fps or 30
        self.sinks = build_sinks(cfg.output)

        self.state = SharedState()
        self.server = (
            ControlServer(cfg.server, self.state, lambda: self.cfg) if cfg.server.enabled else None
        )

        self._show_overlay = cfg.runtime.show_overlay
        self._last_subject_time = time.perf_counter()
        self._last_publish = 0.0
        self._homed = False
        self._parked = False
        self._running = False
        self._last_view = Rect(0, 0, 1, 1)

    # -- lifecycle ----------------------------------------------------------
    def run(self) -> None:
        self.source.open()
        self.worker.start()
        if self.server is not None:
            self.server.start()
            print(f"control API: {self.server.url}")
            if self.cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
                print("  ! bound to a non-local address; this API can move your camera")
        if self.cfg.runtime.show_preview:
            print("keys:  q quit   l lock nearest   u auto   h home   o overlay")
        self._running = True
        try:
            self._loop()
        finally:
            self.close()

    def close(self) -> None:
        self._running = False
        try:
            self.backend.stop()
        except Exception:
            pass
        if self.server is not None:
            self.server.stop()
        self.worker.stop()
        self.source.close()
        self.backend.close()
        for sink in self.sinks:
            sink.close()
        if self.cfg.runtime.show_preview:
            cv2.destroyAllWindows()

    # -- main loop ----------------------------------------------------------
    def _loop(self) -> None:
        rt = self.cfg.runtime
        min_period = 1.0 / rt.control_rate if rt.control_rate else 0.0
        last_tick = time.perf_counter()
        last_submitted = 0.0
        fps_window_start = last_tick
        fps_frames = 0
        disconnected_since: float | None = None

        while self._running and self.source.running:
            self.worker.raise_if_failed()
            self._drain_commands()

            frame, frame_time = self.source.read()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            dt = min(0.25, max(1e-4, now - last_tick))
            last_tick = now

            live = self.source.connected
            if live:
                disconnected_since = None
                self._parked = False
            elif disconnected_since is None:
                disconnected_since = now

            # A stale frame must never drive a motor. Hold position, park after
            # a grace period, and keep serving the UI so an operator can see why.
            if not live:
                self._handle_disconnected(now, disconnected_since, dt)
                self._publish(frame, self.backend.render(frame), None, None, None, None, live)
                self._sleep_remainder(min_period, now)
                continue

            if frame_time > last_submitted:
                self.worker.submit(frame, frame_time)
                last_submitted = frame_time

            # 1. advance every track to *now*
            self.tracker.predict(dt)

            # 2. fold in a detection if one arrived since the last tick
            result = self.worker.poll()
            if result is not None:
                staleness = max(0.0, now - result.frame_time)
                self.tracker.update(result.detections, staleness=staleness)
                self.stats.detections += len(result.detections)

            # 3. what part of the sensor are we outputting?
            view = self.backend.view_rect(frame.shape)
            self._last_view = view

            # 4. who are we following?
            subject = self.selector.select(self.tracker, view, frame.shape)

            error: FramingError | None = None
            predicted = None
            target_point = None

            if subject is not None:
                self._last_subject_time = now
                self._homed = False
                # 5. aim where they *will* be, not where they were
                predicted = subject.box_ahead(rt.lead_time)
                error = self.composer.compute(predicted, view, subject.velocity, dt)
                target_point = self._target_point(view)

                pan = self.pan.step(error.x, dt)
                tilt = self.tilt.step(-error.y, dt)  # +tilt raises the camera
                zoom = self.zoom.step(error.zoom, dt) if self.cfg.framing.zoom_enabled else 0.0
            else:
                pan = self.pan.coast_to_stop(dt)
                tilt = self.tilt.coast_to_stop(dt)
                zoom = self.zoom.coast_to_stop(dt)
                self.composer.reset()
                self._maybe_return_home(now, pan, tilt)

            # 6. drive the head
            self.backend.move(pan, tilt, zoom, dt)

            # 7. deliver the picture
            program = self.backend.render(frame)
            for sink in self.sinks:
                sink.write(program)

            fps_frames += 1
            if now - fps_window_start >= 0.5:
                self.stats.control_fps = fps_frames / (now - fps_window_start)
                self.stats.detect_fps = self.worker.fps
                fps_window_start = now
                fps_frames = 0
            self.stats.frames += 1

            director = self._publish(
                frame, program, subject, view, error, predicted, live, target_point
            )
            if rt.show_preview and not self._present(program, director):
                break

            self._sleep_remainder(min_period, now)

    def _sleep_remainder(self, min_period: float, started: float) -> None:
        if not min_period:
            return
        slack = min_period - (time.perf_counter() - started)
        if slack > 0:
            time.sleep(slack)

    # -- degraded operation -------------------------------------------------
    def _handle_disconnected(self, now: float, since: float | None, dt: float) -> None:
        pan = self.pan.coast_to_stop(dt)
        tilt = self.tilt.coast_to_stop(dt)
        zoom = self.zoom.coast_to_stop(dt)
        self.backend.move(pan, tilt, zoom, dt)
        if self._parked or since is None:
            return
        if (now - since) < self.cfg.runtime.park_after_disconnect:
            return
        self.backend.stop()
        self._parked = True

    # -- helpers ------------------------------------------------------------
    def _target_point(self, view: Rect) -> tuple[float, float]:
        cfg = self.cfg.framing
        return (view.x + cfg.target_x * view.w, view.y + cfg.target_y * view.h)

    def _maybe_return_home(self, now: float, pan: float, tilt: float) -> None:
        after = self.cfg.runtime.return_home_after
        if after is None or self._homed:
            return
        if (now - self._last_subject_time) < after:
            return
        if abs(pan) > 1e-3 or abs(tilt) > 1e-3:
            return  # let the axes finish coasting before recentring
        home = getattr(self.backend, "home", None)
        if callable(home):
            home()
        self._homed = True

    def _status(self, subject, live: bool) -> dict:
        return {
            "control_fps": round(self.stats.control_fps, 1),
            "detect_fps": round(self.stats.detect_fps, 1),
            "detect_ms": round(self.worker.latency * 1000, 1),
            "subject": f"#{subject.id}" if subject else None,
            "people": sum(1 for t in self.tracker.tracks if t.confirmed),
            "pan": round(self.pan.velocity, 3),
            "tilt": round(self.tilt.velocity, 3),
            "zoom": round(self.zoom.velocity, 3),
            "locked": self.selector.manual,
            "backend": self.backend.name,
            "connected": live,
            "reconnects": self.source.reconnects,
            "frames": self.stats.frames,
            "sinks": [
                {"name": s.name, "dropped": s.dropped, "failed": s.failed} for s in self.sinks
            ],
        }

    def _needs_director(self) -> bool:
        if self.cfg.runtime.show_preview and self._show_overlay:
            return True
        return self.server is not None and self.state.director_wanted

    def _publish(self, frame, program, subject, view, error, predicted, live, target_point=None):
        """Build the director view (only when someone is looking) and share it."""
        director = None
        if self._needs_director():
            director = draw_director(
                frame,
                tracks=self.tracker.tracks,
                subject=subject,
                view=view if view is not None else self._last_view,
                error=error,
                predicted=predicted,
                deadzone=(self.cfg.pan.deadzone, self.cfg.tilt.deadzone),
                stats=self._status(subject, live),
                target_point=target_point,
                zones=self.zones,
                key_hints=self.cfg.runtime.show_preview,
            )

        if self.server is not None:
            cfg = self.cfg.server
            now = time.perf_counter()
            due = (now - self._last_publish) >= 1.0 / max(1.0, cfg.stream_fps)
            if due:
                self._last_publish = now
                self.state.publish(
                    director=director,
                    program=program,
                    status=self._status(subject, live),
                    quality=cfg.stream_quality,
                    width=cfg.stream_width,
                )
            else:
                self.state.publish(status=self._status(subject, live))
        return director

    def _drain_commands(self) -> None:
        """Apply UI actions on the control thread, where they are safe."""
        if self.server is None:
            return
        for name, payload in self.state.take_commands():
            if name == "lock":
                track_id = payload.get("id")
                if track_id is None:
                    self._lock_nearest()
                else:
                    self.selector.lock(int(track_id))
            elif name == "unlock":
                self.selector.unlock()
            elif name == "home":
                home = getattr(self.backend, "home", None)
                if callable(home):
                    home()

    def _lock_nearest(self) -> None:
        """Lock whoever is most central -- the usual 'that one' gesture."""
        view = self._last_view
        best, best_d = None, float("inf")
        for t in self.tracker.tracks:
            if not t.confirmed:
                continue
            d = abs(t.box.cx - view.cx) + abs(t.box.cy - view.cy)
            if d < best_d:
                best_d, best = d, t
        if best is not None:
            self.selector.lock(best.id)

    # -- local preview ------------------------------------------------------
    def _present(self, program, director) -> bool:
        cv2.imshow(_PROGRAM_WINDOW, program)
        if self._show_overlay and director is not None:
            cv2.imshow(_DIRECTOR_WINDOW, director)
        return self._handle_keys()

    def _handle_keys(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("l"):
            self._lock_nearest()
        elif key == ord("u"):
            self.selector.unlock()
        elif key == ord("o"):
            self._show_overlay = not self._show_overlay
            if not self._show_overlay:
                cv2.destroyWindow(_DIRECTOR_WINDOW)
        elif key == ord("h"):
            home = getattr(self.backend, "home", None)
            if callable(home):
                home()
        return True
