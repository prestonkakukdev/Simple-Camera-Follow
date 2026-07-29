"""Person detectors.

Any object with a ``detect(frame) -> list[Detection]`` method plugs in here, so
swapping in a different model (RT-DETR, a Hailo/Coral/RKNN-accelerated graph, a
face detector for tight shots) is a single class.

Two are shipped:

* ``YoloDetector`` -- ultralytics YOLO. What you should actually run. ``yolo11n``
  is ~5 ms on a modest GPU, ~25 ms on Apple Silicon CPU, ~15 ms on a Jetson Orin
  Nano. Nano is plenty; a person occupying a third of frame is an easy target.
* ``HogDetector`` -- OpenCV's built-in HOG people detector. No model download,
  no torch, no GPU. Slow and twitchy, but it proves out an entire rig before you
  install anything, and it is a genuine fallback on very weak hardware.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .types import Box, Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> Sequence[Detection]: ...


@dataclass
class DetectorConfig:
    #: "yolo" or "hog".
    kind: str = "yolo"
    model: str = "yolo11n.pt"
    #: "cpu", "cuda", "cuda:0", "mps" (Apple Silicon). None => auto.
    device: str | None = None
    confidence: float = 0.35
    #: Long edge the frame is resized to before inference. Lower == faster.
    imgsz: int = 640
    #: Run inference at half precision where supported.
    half: bool = False


def build_detector(cfg: DetectorConfig) -> Detector:
    kind = cfg.kind.lower()
    if kind == "yolo":
        return YoloDetector(cfg)
    if kind == "hog":
        return HogDetector(cfg)
    raise ValueError(f"unknown detector kind {cfg.kind!r} (expected 'yolo' or 'hog')")


class YoloDetector:
    #: COCO class 0 is "person".
    PERSON_CLASS = 0

    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "the yolo detector needs ultralytics:  pip install 'camerafollow[yolo]'\n"
                "or run with  detector.kind: hog  for a zero-dependency fallback."
            ) from exc

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.device = cfg.device or _auto_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            classes=[self.PERSON_CLASS],
            conf=self.cfg.confidence,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.cfg.half,
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, conf, strict=True):
            out.append(Detection(Box(float(x1), float(y1), float(x2), float(y2)), float(score)))
        return out


class HogDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._width = 640

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = self._width / float(w)
        small = cv2.resize(frame, (self._width, int(h * scale)))
        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        out: list[Detection] = []
        for (x, y, rw, rh), score in zip(rects, weights, strict=True):
            s = float(score)
            if s < self.cfg.confidence:
                continue
            inv = 1.0 / scale
            out.append(
                Detection(
                    Box(x * inv, y * inv, (x + rw) * inv, (y + rh) * inv),
                    min(1.0, s / 2.0),
                )
            )
        return out


def _auto_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
