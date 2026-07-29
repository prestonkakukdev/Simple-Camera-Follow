"""Constant-velocity Kalman filter over a bounding box.

State is ``[cx, cy, w, h, vx, vy]``. Width/height are modelled as random-walk
(no velocity) because subject size changes slowly and estimating its rate just
adds noise. Position gets a velocity term -- that velocity is what lets us
extrapolate ahead of the detector and hide end-to-end latency, which is the
single biggest contributor to "it feels like it knows where I'm going".
"""

from __future__ import annotations

import numpy as np

from .types import Box

_DIM = 6
_MEAS = 4


class BoxKalman:
    def __init__(
        self,
        box: Box,
        *,
        process_pos: float = 12.0,
        process_vel: float = 90.0,
        process_size: float = 8.0,
        meas_pos: float = 6.0,
        meas_size: float = 12.0,
    ) -> None:
        cx, cy, w, h = box.to_cxcywh()
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 100.0, 100.0, 400.0, 400.0])

        self._q = np.array(
            [process_pos, process_pos, process_size, process_size, process_vel, process_vel],
            dtype=np.float64,
        )
        self.R = np.diag([meas_pos**2, meas_pos**2, meas_size**2, meas_size**2])

        self.H = np.zeros((_MEAS, _DIM))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

    # -- prediction ---------------------------------------------------------
    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(_DIM)
        F[0, 4] = dt
        F[1, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # Noise grows with dt so long gaps between detections widen the
        # covariance and the filter re-locks quickly when one finally arrives.
        return np.diag((self._q * dt) ** 2)

    def predict(self, dt: float) -> None:
        dt = max(1e-4, dt)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.x[2] = max(2.0, self.x[2])
        self.x[3] = max(2.0, self.x[3])

    # -- correction ---------------------------------------------------------
    def update(self, box: Box, *, staleness: float = 0.0, trust: float = 1.0) -> None:
        """Fold in a measurement.

        ``staleness`` is how old the detection is, in seconds. An async
        detector hands us a box that describes where the subject *was*, so we
        inflate R rather than pretending it is current -- the alternative is a
        filter that lags and then snaps.

        ``trust`` in ``(0, 1]`` scales confidence; a low-scoring detection
        nudges the state instead of yanking it.
        """
        z = np.array(box.to_cxcywh(), dtype=np.float64)
        speed = float(np.hypot(self.x[4], self.x[5]))
        inflate = (1.0 + speed * staleness / 8.0) ** 2 / max(0.05, trust)
        R = self.R * inflate

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # -- accessors ----------------------------------------------------------
    @property
    def box(self) -> Box:
        return Box.from_cxcywh(*self.x[:4])

    @property
    def velocity(self) -> tuple[float, float]:
        """Subject velocity in pixels/second."""
        return float(self.x[4]), float(self.x[5])

    def box_ahead(self, lead: float) -> Box:
        """Where the subject will be ``lead`` seconds from now.

        Feeding this to the framing stage instead of the current box is what
        cancels detector + motor latency.
        """
        cx = self.x[0] + self.x[4] * lead
        cy = self.x[1] + self.x[5] * lead
        return Box.from_cxcywh(cx, cy, self.x[2], self.x[3])
