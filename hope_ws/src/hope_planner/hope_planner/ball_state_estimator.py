"""Stage 1 - Ball state estimation.

Fits a 2nd-order polynomial to the most recent N position samples and
differentiates analytically to obtain a smoothed position and velocity.
The buffer is cleared on each detected table bounce or racket-impact-sized
velocity jump so the polynomial never fits across a contact discontinuity.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 3.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .constants import PlannerConfig


class BallStateEstimator:
    """Estimate ball position and velocity from a position stream.

    Maintains a sliding window of recent position measurements and performs
    a least-squares polynomial fit to extract smoothed position and velocity.

    Bounce detection uses a three-sample pattern (descend -> contact -> rise)
    to identify a table impact. A second, geometry-independent detector compares
    short secant velocities on either side of each candidate breakpoint; this
    catches opponent/racket hits away from the table.
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
        self._discontinuity_detected: bool = False
        self._last_velocity_jump_mps: float = float("nan")

    def reset(self) -> None:
        """Clear the polynomial-fit buffer after a contact or stream reset."""
        self.t_buffer.clear()
        self.p_buffer.clear()
        self._discontinuity_detected = False

    def _velocity_discontinuity_tail(self) -> Optional[Tuple[List[float], List[np.ndarray], float]]:
        """Return the post-jump tail when a contact-sized velocity jump is present.

        For a candidate breakpoint, compare the displacement secant over ``k``
        intervals before it with the secant over ``k`` intervals after it.  With
        the default k=3 at 300 Hz these are two 10 ms averages.  Ballistic gravity
        and venue drag change velocity by only a few tenths of a metre per second
        over the combined 20 ms span; a racket impact changes it by several m/s.
        The default 3 m/s threshold is therefore physical rather than a noise-fit
        magic number.  Detection is delayed only k frames, and the returned tail
        lets the caller discard every pre-impact sample while retaining the
        already-observed post-impact samples.
        """
        k = int(getattr(self.config, "discontinuity_window", 3))
        threshold = float(getattr(self.config, "discontinuity_velocity_jump_mps", 3.0))
        max_spread = float(getattr(self.config, "discontinuity_segment_spread_mps", 2.5))
        if (
            k < 1
            or not np.isfinite(threshold)
            or threshold <= 0.0
            or not np.isfinite(max_spread)
            or max_spread <= 0.0
        ):
            return None
        n_required = 2 * k + 1
        if len(self.t_buffer) < n_required:
            return None

        ts = self.t_buffer[-n_required:]
        ps = self.p_buffer[-n_required:]
        dt_before = float(ts[k] - ts[0])
        dt_after = float(ts[-1] - ts[k])
        if dt_before <= 0.0 or dt_after <= 0.0:
            return None

        v_before = (ps[k] - ps[0]) / dt_before
        v_after = (ps[-1] - ps[k]) / dt_after
        jump = float(np.linalg.norm(v_after - v_before))
        if not np.isfinite(jump) or jump < threshold:
            return None

        # Do not trigger on a candidate whose own before/after segment still
        # contains the impact.  Per-interval velocities must agree with that
        # segment's secant within the venue-noise allowance; otherwise wait for
        # the centered breakpoint to acquire k genuinely post-impact frames.
        before_dt = np.diff(np.asarray(ts[: k + 1], dtype=float))
        after_dt = np.diff(np.asarray(ts[k:], dtype=float))
        if np.any(before_dt <= 0.0) or np.any(after_dt <= 0.0):
            return None
        before_step_v = np.diff(np.asarray(ps[: k + 1]), axis=0) / before_dt[:, None]
        after_step_v = np.diff(np.asarray(ps[k:]), axis=0) / after_dt[:, None]
        before_spread = float(np.max(np.linalg.norm(before_step_v - v_before, axis=1)))
        after_spread = float(np.max(np.linalg.norm(after_step_v - v_after, axis=1)))
        if before_spread > max_spread or after_spread > max_spread:
            return None
        return list(ts[k:]), [p.copy() for p in ps[k:]], jump

    def push(self, t: float, p: np.ndarray) -> None:
        """Add a new position measurement.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic, e.g. from ROS clock).
        p : np.ndarray, shape (3,)
            Ball position [x, y, z] in the HOPE canonical frame.
        """
        self._discontinuity_detected = False

        # Update z history ring buffer
        self._z_hist[0] = self._z_hist[1]
        self._z_hist[1] = self._z_hist[2]
        self._z_hist[2] = p[2]

        # Bounce detection: three-sample pattern, two geometries (paper §IV-A: clear the buffer
        # on bounce so the polyfit never spans the velocity discontinuity).
        #   (a) LEGACY point-ball dip: z_prev at/below bounce_z_tol between two above-tol samples
        #       (sim harness feeds ideal point-ball z that reaches ~0 at contact).
        #   (b) CENTER geometry (real mocap, audit 2026-07-07): the rigid-body CENTER never goes
        #       below the ball RADIUS (0.02 m >> bounce_z_tol), so detect a LOCAL Z-MINIMUM below
        #       bounce_center_z_max — descending then rising with the dip in the contact band.
        self._bounce_detected = False
        z_pp, z_p, z_c = self._z_hist
        tol = self.config.bounce_z_tol
        center_max = getattr(self.config, "bounce_center_z_max", 0.05)
        if z_pp is not None and z_p is not None and z_c is not None:
            legacy_dip = z_pp > tol and z_p <= tol and z_c > tol
            center_min = z_p <= center_max and z_pp > z_p and z_c > z_p
            if legacy_dip or center_min:
                self._bounce_detected = True
                self.reset()

        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        # A racket/opponent hit need not occur near the table, so the local-z
        # bounce detector above cannot see it.  Test each centered breakpoint
        # once k post-event samples exist, then keep only the breakpoint and
        # post-event tail.  This prevents the first actionable plan after a hit
        # from fitting one polynomial through both incoming and outgoing arcs.
        discontinuity = self._velocity_discontinuity_tail()
        if discontinuity is not None:
            tail_t, tail_p, jump = discontinuity
            self.reset()
            self.t_buffer.extend(tail_t)
            self.p_buffer.extend(tail_p)
            self._discontinuity_detected = True
            self._last_velocity_jump_mps = jump

        if len(self.t_buffer) > self.config.fit_window:
            self.t_buffer.pop(0)
            self.p_buffer.pop(0)

    @property
    def bounce_detected(self) -> bool:
        """True if the most recent push() detected a table bounce."""
        return self._bounce_detected

    @property
    def discontinuity_detected(self) -> bool:
        """True if the most recent push cleared a non-bounce velocity jump."""
        return self._discontinuity_detected

    @property
    def last_velocity_jump_mps(self) -> float:
        """Magnitude of the last detected before/after secant jump (m/s)."""
        return self._last_velocity_jump_mps

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
