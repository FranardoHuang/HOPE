"""Stage 1 - Ball state estimation.

Fits a 2nd-order polynomial to the most recent N position samples and
differentiates analytically to obtain a smoothed position and velocity.
The buffer is cleared on each detected table bounce so the polynomial
never fits across the velocity discontinuity.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 3.
"""

from typing import List, Optional, Tuple

import numpy as np

from .constants import PlannerConfig


class BallStateEstimator:
    """Estimate ball position and velocity from a position stream.

    Maintains a sliding window of recent position measurements and performs
    a least-squares polynomial fit to extract smoothed position and velocity.

    Bounce detection uses a three-sample pattern (descend -> contact -> rise)
    to identify the actual table impact event and clear the buffer.
    """

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.t_buffer: List[float] = []
        self.p_buffer: List[np.ndarray] = []

        # Bounce detection: three-sample z-height ring buffer.
        # Initialized to None to suppress false triggers before
        # enough measurements are collected.
        self._z_hist: List[Optional[float]] = [None, None, None]
        self._bounce_detected: bool = False

    def reset(self) -> None:
        """Clear the estimation buffer (call on bounce detection)."""
        self.t_buffer.clear()
        self.p_buffer.clear()

    def push(self, t: float, p: np.ndarray) -> None:
        """Add a new position measurement.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic, e.g. from ROS clock).
        p : np.ndarray, shape (3,)
            Ball position [x, y, z] in the HOPE canonical frame.
        """
        # Update z history ring buffer
        self._z_hist[0] = self._z_hist[1]
        self._z_hist[1] = self._z_hist[2]
        self._z_hist[2] = p[2]

        # Bounce detection: three-sample pattern.
        # z_prev_prev was above table, z_prev dipped to contact,
        # z_curr is rising again -> actual bounce event.
        self._bounce_detected = False
        z_pp, z_p, z_c = self._z_hist
        tol = self.config.bounce_z_tol
        if z_pp is not None and z_p is not None and z_c is not None:
            if z_pp > tol and z_p <= tol and z_c > tol:
                self._bounce_detected = True
                self.reset()

        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        if len(self.t_buffer) > self.config.fit_window:
            self.t_buffer.pop(0)
            self.p_buffer.pop(0)

    @property
    def bounce_detected(self) -> bool:
        """True if the most recent push() detected a table bounce."""
        return self._bounce_detected

    @property
    def ready(self) -> bool:
        """True if enough samples exist for a reliable fit."""
        return len(self.t_buffer) >= 6

    def estimate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute smoothed ball position and velocity at the latest timestamp.

        Returns
        -------
        p_est : np.ndarray, shape (3,)
            Smoothed position estimate [x, y, z] in HOPE frame.
        v_est : np.ndarray, shape (3,)
            Velocity estimate [vx, vy, vz] in HOPE frame (m/s).
        t_est : float
            Timestamp of the estimate (latest sample time).
        """
        if not self.ready:
            raise RuntimeError(f"Need >= 6 samples, have {len(self.t_buffer)}")

        t_arr = np.array(self.t_buffer)
        p_arr = np.array(self.p_buffer)

        # Normalize time to improve numerical conditioning
        t_ref = t_arr[-1]
        t_norm = t_arr - t_ref

        p_est = np.zeros(3)
        v_est = np.zeros(3)

        for axis in range(3):
            # np.polyfit returns [a2, a1, a0] (descending order)
            coeffs = np.polyfit(t_norm, p_arr[:, axis], deg=self.config.poly_order)
            p_est[axis] = coeffs[-1]   # a0 at t_norm = 0
            v_est[axis] = coeffs[-2]   # a1 at t_norm = 0

        return p_est, v_est, t_ref
