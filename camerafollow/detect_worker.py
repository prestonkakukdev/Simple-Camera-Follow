"""Runs the detector off the control loop.

This is the architectural decision that makes the motion smooth, and it is worth
stating plainly: **the control loop must never wait for inference.**

If you detect inline, your loop runs at the detector's rate. At 15 fps that is
67 ms of quantisation in the motor command, and the result visibly steps. Here
the detector runs in its own thread at whatever rate it can manage, while the
control loop runs at full capture rate and rides the Kalman prediction between
detections. A 15 fps detector then drives a 60 Hz control loop, and it looks
like a fluid head rather than a stepper motor.

Each result carries the capture time of the frame it saw, so the tracker knows
exactly how stale the measurement is and can weight it accordingly.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .detectors import Detector
from .types import DetectionResult


class DetectorWorker:
    def __init__(self, detector: Detector, *, max_rate: float | None = None) -> None:
        self.detector = detector
        self.min_interval = 1.0 / max_rate if max_rate else 0.0
        self._in_frame: np.ndarray | None = None
        self._in_time = 0.0
        self._in_lock = threading.Lock()
        self._out: DetectionResult | None = None
        self._out_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        #: Detections actually delivered per second, including any rate cap.
        #: Reporting raw inference speed here would flatter the system and hide
        #: the number that matters: how often the tracker gets corrected.
        self.fps = 0.0
        #: Seconds spent inside the model on the last frame.
        self.latency = 0.0

    def start(self) -> DetectorWorker:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def submit(self, frame: np.ndarray, frame_time: float) -> None:
        """Offer the newest frame. Older un-consumed frames are dropped."""
        with self._in_lock:
            self._in_frame = frame
            self._in_time = frame_time
        self._wake.set()

    def poll(self) -> DetectionResult | None:
        """Take the newest result, or ``None`` if nothing new since last call."""
        with self._out_lock:
            out, self._out = self._out, None
        return out

    def _run(self) -> None:
        smoothed = 0.0
        last_done = 0.0
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.1):
                continue
            self._wake.clear()

            with self._in_lock:
                frame = self._in_frame
                frame_time = self._in_time
                self._in_frame = None
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                dets = self.detector.detect(frame)
            except BaseException as exc:  # surface it on the main thread
                self._error = exc
                self._stop.set()
                return
            t1 = time.perf_counter()

            with self._out_lock:
                self._out = DetectionResult(
                    detections=tuple(dets), frame_time=frame_time, latency=t1 - t0
                )

            self.latency = t1 - t0
            if last_done:
                interval = t1 - last_done
                smoothed = interval if smoothed == 0.0 else smoothed * 0.9 + interval * 0.1
                self.fps = 1.0 / smoothed if smoothed > 0 else 0.0
            last_done = t1

            elapsed = t1 - t0

            if self.min_interval:
                slack = self.min_interval - elapsed
                if slack > 0:
                    time.sleep(slack)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
