"""Stage 2 - Ball trajectory prediction.

Forward-integrate the ball trajectory with explicit Euler at 1 kHz using a
hybrid flight (quadratic drag + gravity) / bounce (diagonal restitution)
model, and return the predicted ball state at the virtual hitting plane.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 4.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .constants import BallPhysics, PlannerConfig, TableParams
from .fastmath import cross3


@dataclass
class StrikeTarget:
    """Output of Stage 2: predicted ball state at the hitting plane."""

    p_ball: np.ndarray        # predicted ball position at strike [x, y, z]
    v_ball: np.ndarray        # predicted ball velocity at strike [vx, vy, vz]
    t_strike: float           # absolute time of strike
    num_bounces: int          # number of table bounces before strike
    valid: bool               # True if a valid crossing was found


class BallTrajectoryPredictor:
    """Forward-integrate ball trajectory and find the hitting-plane crossing.

    Uses explicit Euler integration with the hybrid flight/bounce model
    from HITTER Section III-B.
    """

    def __init__(self, physics: BallPhysics, config: PlannerConfig, table: TableParams):
        self.physics = physics
        self.config = config
        self.table = table

    def _is_on_table(self, p: np.ndarray) -> bool:
        """Check if the ball could contact the table surface.

        Bounds are expanded by ball radius to handle edge contacts.
        """
        r = self.physics.radius
        y_hi = self.table.y_max
        return (
            -r <= p[0] <= self.table.length + r
            and y_hi - self.table.width - r <= p[1] <= y_hi + r
        )

    def _flight_acceleration(self, v: np.ndarray, omega: Optional[np.ndarray] = None) -> np.ndarray:
        """Flight acceleration: a = -k|v|v + g [+ k_m (omega x v)].

        omega is the ball spin (rad/s); when omega is None or zero the Magnus
        term contributes exactly 0.0 so legacy (spin-blind) behavior is
        bit-identical. k_m is the venue Magnus coefficient
        (configs/ball_physics_venue.yaml flight.k_m).

        The Magnus cross product is the hand-written fastmath.cross3 —
        BIT-IDENTICAL to np.cross on float64 (asserted by
        test_strike_spec_fast.py over random states incl. zeros/subnormals)
        but ~10x cheaper per call; np.cross dispatch was ~60 % of the step
        cost in this 1 kHz loop (benchmark --profile, 2026-07-06).
        """
        speed = np.linalg.norm(v)
        a = -self.physics.k * speed * v + self.physics.g
        if omega is not None:
            a = a + self.config.k_m * cross3(omega, v)
        return a

    def _apply_bounce(self, v: np.ndarray) -> np.ndarray:
        """Apply table bounce restitution: v+ = diag(C_h, C_h, -C_v) @ v-"""
        C = np.diag([self.physics.C_h, self.physics.C_h, -self.physics.C_v])
        return C @ v

    def _apply_bounce_nakashima(
        self, v: np.ndarray, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spin-coupled table bounce (Ace/Nakashima impulse model).

        Contact-point tangential velocity (contact at -r*ez from center):
            v_T = [vx - r*wy, vy + r*wx, 0]
        A friction impulse -alpha*m*v_T acts at the contact point, with
        alpha = mu (1 + e_n) |vz| / |v_T| while sliding persists, capped at
        the ROLLING value alpha = 2/5 (hollow-sphere inertia I = (2/3) m r^2)
        at which v_T+ = (1 - 5/2 alpha) v_T reaches exactly zero. The spin
        rows follow from the impulse torque delta_w = -(3/2r) ez x delta_v:
            w+_x = wx - (3 alpha / 2r) v_Ty
            w+_y = wy + (3 alpha / 2r) v_Tx
            w+_z = wz
        e_n is the venue-fit table restitution (BallPhysics.C_v = contact.
        table.e_eff); mu is an UNFITTED Ace prior (constants.mu_table) — the
        venue tangential refit was degenerate (F5 in the fit report).
        """
        r = self.physics.radius
        e_n = self.physics.C_v
        mu = self.config.mu_table

        v_T = np.array([v[0] - r * w[1], v[1] + r * w[0], 0.0])
        v_T_norm = np.linalg.norm(v_T)

        if v_T_norm < 1e-9:
            alpha = 0.0
        else:
            alpha_slide = mu * (1.0 + e_n) * abs(v[2]) / v_T_norm
            nu_s = 1.0 - 2.5 * alpha_slide
            alpha = alpha_slide if nu_s > 0.0 else 0.4  # rolling cap = 2/5

        v_out = np.array([
            v[0] - alpha * v_T[0],
            v[1] - alpha * v_T[1],
            -e_n * v[2],
        ])
        gain = 3.0 * alpha / (2.0 * r)
        w_out = np.array([
            w[0] - gain * v_T[1],
            w[1] + gain * v_T[0],
            w[2],
        ])
        return v_out, w_out

    def _rk4_flight_step(
        self, p: np.ndarray, v: np.ndarray, omega: np.ndarray, h: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One RK4 step of the SAME flight ODE (_flight_acceleration).

        Used only by the adaptive (dt_integrate_coarse > 0) cruise phase; the
        legacy fixed-dt path never calls this. Same tableau as the training
        side virtual_ball.rk4_step so the coarse cruise stays a high-order
        approximation of the identical physics, not a second model.
        """
        a1 = self._flight_acceleration(v, omega)
        v2 = v + 0.5 * h * a1
        a2 = self._flight_acceleration(v2, omega)
        v3 = v + 0.5 * h * a2
        a3 = self._flight_acceleration(v3, omega)
        v4 = v + h * a3
        a4 = self._flight_acceleration(v4, omega)
        p_new = p + (h / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v_new = v + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        return p_new, v_new

    def _adaptive_dts(self) -> Tuple[float, float, int]:
        """(dt_fine, dt_coarse, n_sub) for the adaptive path (flag > fine)."""
        dt_f = self.config.dt_integrate
        dt_c = float(getattr(self.config, "dt_integrate_coarse", 0.0))
        n_sub = max(1, int(round(dt_c / dt_f)))
        return dt_f, dt_c, n_sub

    def integrate_to_table_plane(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        omega0: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """First DOWNWARD table-plane (z = 0) crossing of a post-strike flight.

        Shares _flight_acceleration (drag + gravity + Magnus) and the Euler
        scheme with predict() so the strike-spec planner's landing forward
        model is the same physics as the incoming-ball prediction — no second
        integrator to keep in sync. Spin is constant in flight (venue
        measurement: spin decay consistent with zero, fit report §11.2).

        Returns (landing_xy, flight_time_s) or None if the ball never crosses
        z = 0 from above within config.max_predict_time.

        With config.dt_integrate_coarse > dt_integrate the cruise phase uses
        RK4 steps of the coarse size and only intervals that can contain the
        landing are replayed with the legacy fine Euler scheme (远粗近细);
        flag off (0.0, the default) keeps this method byte-identical.
        """
        if getattr(self.config, "dt_integrate_coarse", 0.0) > self.config.dt_integrate:
            return self._integrate_to_table_plane_adaptive(p0, v0, omega0)
        dt = self.config.dt_integrate
        max_steps = int(self.config.max_predict_time / dt)
        p = np.asarray(p0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()
        omega = np.zeros(3) if omega0 is None else np.asarray(omega0, dtype=float)

        t = 0.0
        for _ in range(max_steps):
            a = self._flight_acceleration(v, omega)
            p_new = p + v * dt + 0.5 * a * dt ** 2
            v_new = v + a * dt
            t += dt
            if p[2] > 0.0 and p_new[2] <= 0.0:
                dz = p[2] - p_new[2]
                frac = p[2] / dz if dz > 1e-12 else 0.5
                frac = float(np.clip(frac, 0.0, 1.0))
                p_land = p + frac * (p_new - p)
                return p_land[:2].copy(), (t - dt) + frac * dt
            p, v = p_new, v_new
        return None

    def _integrate_to_table_plane_adaptive(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        omega0: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Adaptive-dt twin of integrate_to_table_plane (远粗近细).

        RK4 cruise at dt_integrate_coarse; a coarse interval whose endpoints
        straddle z = 0 (or whose descent could reach it: z + min(vz, 0)*dt_c
        <= 0) is replayed from its start with the LEGACY Euler fine steps, so
        the returned crossing has the legacy interpolation resolution. If the
        fine replay does not actually cross, integration continues from the
        fine end state (never skips the event).
        """
        dt_f, dt_c, n_sub = self._adaptive_dts()
        t_end = self.config.max_predict_time
        p = np.asarray(p0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()
        omega = np.zeros(3) if omega0 is None else np.asarray(omega0, dtype=float)

        t = 0.0
        while t < t_end - 1e-12:
            p_new, v_new = self._rk4_flight_step(p, v, omega, dt_c)
            can_reach = p[2] + min(v[2], v_new[2], 0.0) * dt_c <= 0.0
            if p[2] > 0.0 and (p_new[2] <= 0.0 or can_reach):
                for i in range(n_sub):
                    a = self._flight_acceleration(v, omega)
                    p_n = p + v * dt_f + 0.5 * a * dt_f ** 2
                    v_n = v + a * dt_f
                    if p[2] > 0.0 and p_n[2] <= 0.0:
                        dz = p[2] - p_n[2]
                        frac = p[2] / dz if dz > 1e-12 else 0.5
                        frac = float(np.clip(frac, 0.0, 1.0))
                        p_land = p + frac * (p_n - p)
                        return p_land[:2].copy(), t + i * dt_f + frac * dt_f
                    p, v = p_n, v_n
                t += dt_c
                continue
            p, v = p_new, v_new
            t += dt_c
        return None

    def _step_with_events(
        self,
        p: np.ndarray,
        v: np.ndarray,
        omega: np.ndarray,
        t: float,
        bounces: int,
        dt: float,
        use_nakashima: bool,
    ):
        """One LEGACY predict() Euler step (bounce + hit-plane logic), reusable.

        Exact transcription of the predict() loop body at step size ``dt``.
        Returns (strike_or_none, p, v, omega, t, bounces); a non-None first
        element is the finished StrikeTarget (hit-plane crossing found).
        Only the adaptive path calls this — predict() keeps its own inline
        loop so the flag-off path stays byte-identical.
        """
        x_hit = self.config.x_hit
        p_prev_x = p[0]

        a = self._flight_acceleration(v, omega)
        v_new = v + a * dt
        p_new = p + v * dt + 0.5 * a * dt ** 2
        t += dt
        bounce_this_step = False
        p_bounce = p
        v_post = v
        remaining_dt = dt

        if p_new[2] < 0.0 and v_new[2] < 0.0:
            if self._is_on_table(p_new):
                dz = p[2] - p_new[2]
                frac = p[2] / dz if dz > 1e-9 else 0.5
                frac = np.clip(frac, 0.0, 1.0)

                p_bounce = p + frac * (p_new - p)
                p_bounce[2] = 0.0
                v_at_bounce = v + a * (frac * dt)

                if use_nakashima:
                    v_post, omega = self._apply_bounce_nakashima(v_at_bounce, omega)
                else:
                    v_post = self._apply_bounce(v_at_bounce)

                remaining_dt = (1.0 - frac) * dt
                a_post = self._flight_acceleration(v_post, omega)
                p_new = p_bounce + v_post * remaining_dt + 0.5 * a_post * remaining_dt ** 2
                v_new = v_post + a_post * remaining_dt
                bounces += 1
                bounce_this_step = True
            else:
                p_new[2] = max(p_new[2], 0.0)

        if p_prev_x > x_hit and p_new[0] <= x_hit and v_new[0] < 0:
            if bounce_this_step:
                dx_arc = p_bounce[0] - p_new[0]
                if abs(dx_arc) > 1e-9:
                    frac_cross = (p_bounce[0] - x_hit) / dx_arc
                else:
                    frac_cross = 0.5
                frac_cross = np.clip(frac_cross, 0.0, 1.0)
                p_cross = p_bounce + frac_cross * (p_new - p_bounce)
                v_cross = v_post + frac_cross * (v_new - v_post)
                t_cross = (t - remaining_dt) + frac_cross * remaining_dt
            else:
                dx_step = p[0] - p_new[0]
                if abs(dx_step) > 1e-9:
                    frac_cross = (p[0] - x_hit) / dx_step
                else:
                    frac_cross = 0.5
                frac_cross = np.clip(frac_cross, 0.0, 1.0)
                p_cross = p + frac_cross * (p_new - p)
                v_cross = v + frac_cross * (v_new - v)
                t_cross = t - dt + frac_cross * dt

            p_cross[0] = x_hit
            dead_ball = p_cross[2] < 0.05 and v_cross[2] < 0.0
            strike = StrikeTarget(
                p_ball=p_cross, v_ball=v_cross,
                t_strike=t_cross, num_bounces=bounces, valid=not dead_ball,
            )
            return strike, p_new, v_new, omega, t, bounces

        return None, p_new, v_new, omega, t, bounces

    def _predict_adaptive(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        t0: float,
        omega0: Optional[np.ndarray] = None,
    ) -> StrikeTarget:
        """Adaptive-dt twin of predict() (远粗近细).

        RK4 cruise at dt_integrate_coarse; a coarse interval that could
        contain an event — table reach (z + min(vz,0)*dt_c <= 0 or endpoint
        below the plane) or a hit-plane straddle — is replayed from its start
        with n_sub LEGACY fine Euler steps carrying the full bounce +
        crossing logic (_step_with_events), so events keep dt_integrate
        resolution. Fine replays that end without the event hand their end
        state back to the cruise.
        """
        dt_f, dt_c, n_sub = self._adaptive_dts()
        x_hit = self.config.x_hit
        horizon = self.config.max_predict_time

        p = p0.astype(float).copy()
        v = v0.astype(float).copy()
        omega = np.zeros(3) if omega0 is None else np.asarray(omega0, dtype=float).copy()
        use_nakashima = self.config.bounce_model == "nakashima"
        t = t0
        bounces = 0

        elapsed = 0.0
        while elapsed < horizon - 1e-12:
            p_new, v_new = self._rk4_flight_step(p, v, omega, dt_c)
            near_table = (
                p_new[2] < 0.0
                or min(p[2], p_new[2]) + min(v[2], v_new[2], 0.0) * dt_c <= 0.0
            )
            straddles_hit = p[0] > x_hit >= p_new[0]
            if near_table or straddles_hit:
                for _ in range(n_sub):
                    strike, p, v, omega, t, bounces = self._step_with_events(
                        p, v, omega, t, bounces, dt_f, use_nakashima
                    )
                    if strike is not None:
                        return strike
                elapsed += dt_c
                continue
            p, v = p_new, v_new
            t += dt_c
            elapsed += dt_c

        return StrikeTarget(
            p_ball=p, v_ball=v, t_strike=t,
            num_bounces=bounces, valid=False,
        )

    def predict(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        t0: float,
        omega0: Optional[np.ndarray] = None,
    ) -> StrikeTarget:
        """Forward-integrate and find the hitting-plane crossing.

        Parameters
        ----------
        p0 : Current ball position in HOPE frame.
        v0 : Current ball velocity in HOPE frame.
        t0 : Current timestamp (s).
        omega0 : Optional ball spin (rad/s) for the Magnus term and (with
            config.bounce_model == "nakashima") the spin-coupled bounce.
            Default None -> zero spin -> identical to the legacy prediction.

        Returns
        -------
        StrikeTarget with predicted ball state at the virtual hitting plane.

        With config.dt_integrate_coarse > dt_integrate the adaptive twin
        (_predict_adaptive) runs instead; flag off (0.0, the default) keeps
        this method byte-identical.
        """
        if getattr(self.config, "dt_integrate_coarse", 0.0) > self.config.dt_integrate:
            return self._predict_adaptive(p0, v0, t0, omega0)
        dt = self.config.dt_integrate
        max_steps = int(self.config.max_predict_time / dt)
        x_hit = self.config.x_hit

        p = p0.copy()
        v = v0.copy()
        # Spin is carried as a constant during flight (no spin-decay model)
        # and updated at bounces when the nakashima map is active.
        omega = np.zeros(3) if omega0 is None else np.asarray(omega0, dtype=float).copy()
        use_nakashima = self.config.bounce_model == "nakashima"
        t = t0
        bounces = 0

        # Track the most recent bounce state so a hit-plane crossing that
        # happens in the same step as a bounce interpolates on a continuous arc.
        p_bounce = p.copy()
        v_post = v.copy()
        remaining_dt = dt

        for _step in range(max_steps):
            p_prev_x = p[0]

            # --- Euler integration step ---
            a = self._flight_acceleration(v, omega)
            v_new = v + a * dt
            p_new = p + v * dt + 0.5 * a * dt ** 2
            t += dt
            bounce_this_step = False

            # --- Bounce detection ---
            if p_new[2] < 0.0 and v_new[2] < 0.0:
                if self._is_on_table(p_new):
                    # Sub-step interpolation to find exact bounce time
                    dz = p[2] - p_new[2]
                    frac = p[2] / dz if dz > 1e-9 else 0.5
                    frac = np.clip(frac, 0.0, 1.0)

                    p_bounce = p + frac * (p_new - p)
                    p_bounce[2] = 0.0
                    v_at_bounce = v + a * (frac * dt)

                    if use_nakashima:
                        v_post, omega = self._apply_bounce_nakashima(v_at_bounce, omega)
                    else:
                        v_post = self._apply_bounce(v_at_bounce)

                    # Continue from bounce with second-order correction
                    remaining_dt = (1.0 - frac) * dt
                    a_post = self._flight_acceleration(v_post, omega)
                    p_new = p_bounce + v_post * remaining_dt + 0.5 * a_post * remaining_dt ** 2
                    v_new = v_post + a_post * remaining_dt
                    bounces += 1
                    bounce_this_step = True
                else:
                    p_new[2] = max(p_new[2], 0.0)

            # --- Hitting plane crossing detection ---
            if p_prev_x > x_hit and p_new[0] <= x_hit and v_new[0] < 0:
                if bounce_this_step:
                    # Use post-bounce arc for interpolation
                    dx_arc = p_bounce[0] - p_new[0]
                    if abs(dx_arc) > 1e-9:
                        frac_cross = (p_bounce[0] - x_hit) / dx_arc
                    else:
                        frac_cross = 0.5
                    frac_cross = np.clip(frac_cross, 0.0, 1.0)
                    p_cross = p_bounce + frac_cross * (p_new - p_bounce)
                    v_cross = v_post + frac_cross * (v_new - v_post)
                    t_cross = (t - remaining_dt) + frac_cross * remaining_dt
                else:
                    dx_step = p[0] - p_new[0]
                    if abs(dx_step) > 1e-9:
                        frac_cross = (p[0] - x_hit) / dx_step
                    else:
                        frac_cross = 0.5
                    frac_cross = np.clip(frac_cross, 0.0, 1.0)
                    p_cross = p + frac_cross * (p_new - p)
                    v_cross = v + frac_cross * (v_new - v)
                    t_cross = t - dt + frac_cross * dt

                p_cross[0] = x_hit

                # DEAD-BALL GUARD (2026-07-05): a crossing at table-skim height with the
                # ball still falling means the prediction never modeled a bounce (off-table
                # per _is_on_table, z clamped to 0 while vz kept integrating) — publishing
                # it as valid=True produced garbage strike plans (z=0.00, vz=-7 m/s) that
                # the runner's gate had to reject every time. A dead ball is NOT a plan.
                dead_ball = p_cross[2] < 0.05 and v_cross[2] < 0.0
                return StrikeTarget(
                    p_ball=p_cross, v_ball=v_cross,
                    t_strike=t_cross, num_bounces=bounces, valid=not dead_ball,
                )

            p = p_new
            v = v_new

        return StrikeTarget(
            p_ball=p, v_ball=v, t_strike=t,
            num_bounces=bounces, valid=False,
        )
