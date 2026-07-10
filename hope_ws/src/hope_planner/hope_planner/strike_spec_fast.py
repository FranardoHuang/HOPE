"""Fast strike-spec solver: numpy-batched LM probes + adaptive integration.

PRODUCTION home of the benchmark winner "ss_fastnp_a20_warm"
(benchmarks/benchmark_planner_latency.py, N=200 venue scenarios, Mac CPU):

    StrikeSpecPlanner.solve (baseline)   med 451 ms / p90 788 ms @ 18.9 mm
    FastStrikeSpecPlanner.solve_fast     med  15 ms / p90  42 ms @ 18.6 mm
    (dt_integrate_coarse=0.02 + warm start + iter budget 6)

Recipe — each piece is individually ablated in the benchmark curves:
  * numpy-BATCHED Jacobian probes: per LM iteration the 5 probe rollouts are
    integrated as ONE (5, 3) batch (batch_integrate_to_table_plane below),
    amortizing the python-per-step overhead across probes. Contact stays
    per-row — it is ~1000x cheaper than the flight.
  * adaptive 远粗近细 integration via the EXISTING PlannerConfig.
    dt_integrate_coarse flag (RK4 cruise at the coarse step, legacy fine
    Euler replay only inside intervals that can contain the landing).
    0.02 s is the productionized setting; 0.0 (the default) = legacy 1 kHz.
  * warm start: pass q0 = the previous tick's solution. The tilt origin
    (mirror-law seed normal) is recomputed from the CURRENT inputs, so warm
    starts stay valid across small per-tick changes of ball state/target.
  * iteration budget: max_iter=6 suffices from a warm start (median 1-2).
  * sensitivities OFF the hot path: solve_fast never rolls them out;
    solve_fast_spec(with_sensitivities=True) brings them back on demand.

NO physics is duplicated: the class subclasses StrikeSpecPlanner and reuses
its contact model (ball_contact.predict_paddle_contact), geometry helpers,
LM schedule/tolerances/residual and spec assembly; the batched integrator
below is the row-parallel twin of BallTrajectoryPredictor.
integrate_to_table_plane with the same per-row arithmetic (cross products
via fastmath.cross3/cross_rows — bit-identical to np.cross, see fastmath).

Flag story (team rule: un-ablated changes merge flag-off): nothing in this
module runs unless a caller instantiates FastStrikeSpecPlanner; node.py
gates it behind use_fast_strike_spec (default False) and the scalar
StrikeSpecPlanner path stays byte-identical.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .ball_contact import predict_paddle_contact
from .constants import BallPhysics, PlannerConfig
from .fastmath import cross3, cross_rows
from .strike_spec_planner import StrikeSpec, StrikeSpecPlanner


def batch_integrate_to_table_plane(
    p0: np.ndarray,
    Vp: np.ndarray,
    Wp: np.ndarray,
    physics: BallPhysics,
    config: PlannerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched twin of integrate_to_table_plane: M rows stepped together.

    Fixed-dt mode uses the scalar Euler formula (same per-row arithmetic);
    with config.dt_integrate_coarse > dt the cruise is batched RK4 and rows
    whose interval could contain the physical ball-center contact crossing
    (z = ball radius) are replayed scalar-fine.
    Returns (land_xy (M,2) with NaN for no-crossing, t_land (M,)).
    """
    dt = config.dt_integrate
    dt_c = float(getattr(config, "dt_integrate_coarse", 0.0))
    k, g, k_m = physics.k, physics.g, config.k_m
    M = Vp.shape[0]
    P = np.tile(np.asarray(p0, float), (M, 1))
    V = Vp.astype(float).copy()
    W = Wp.astype(float)
    land = np.full((M, 2), np.nan)
    t_land = np.full(M, np.nan)
    active = np.ones(M, bool)
    contact_z = float(physics.radius)

    def accel(Vx):
        sp = np.linalg.norm(Vx, axis=1, keepdims=True)
        return -k * sp * Vx + g + k_m * cross_rows(W, Vx)

    if dt_c > dt:
        n_sub = max(1, int(round(dt_c / dt)))
        t = 0.0
        while t < config.max_predict_time - 1e-12 and active.any():
            A1 = accel(V)
            V2 = V + 0.5 * dt_c * A1
            A2 = accel(V2)
            V3 = V + 0.5 * dt_c * A2
            A3 = accel(V3)
            V4 = V + dt_c * A3
            A4 = accel(V4)
            Pn = P + (dt_c / 6.0) * (V + 2.0 * V2 + 2.0 * V3 + V4)
            Vn = V + (dt_c / 6.0) * (A1 + 2.0 * A2 + 2.0 * A3 + A4)
            reach = (
                P[:, 2] + np.minimum(np.minimum(V[:, 2], Vn[:, 2]), 0.0) * dt_c
                <= contact_z
            )
            flagged = active & (P[:, 2] > contact_z) & ((Pn[:, 2] <= contact_z) | reach)
            for i in np.where(flagged)[0]:
                p, v, w = P[i].copy(), V[i].copy(), W[i]
                for si in range(n_sub):
                    a = -k * np.linalg.norm(v) * v + g + k_m * cross3(w, v)
                    pn = p + v * dt + 0.5 * a * dt * dt
                    vn = v + a * dt
                    if p[2] > contact_z and pn[2] <= contact_z:
                        dz = p[2] - pn[2]
                        frac = (p[2] - contact_z) / dz if dz > 1e-12 else 0.5
                        frac = float(np.clip(frac, 0.0, 1.0))
                        land[i] = (p + frac * (pn - p))[:2]
                        t_land[i] = t + si * dt + frac * dt
                        active[i] = False
                        break
                    p, v = pn, vn
                else:
                    Pn[i], Vn[i] = p, v
            P, V = Pn, Vn
            t += dt_c
        return land, t_land

    max_steps = int(config.max_predict_time / dt)
    t = 0.0
    for _ in range(max_steps):
        if not active.any():
            break
        A = accel(V)
        Pn = P + V * dt + 0.5 * A * dt * dt
        Vn = V + A * dt
        t += dt
        cross = active & (P[:, 2] > contact_z) & (Pn[:, 2] <= contact_z)
        if cross.any():
            dz = P[cross, 2] - Pn[cross, 2]
            frac = np.where(
                dz > 1e-12,
                (P[cross, 2] - contact_z) / np.maximum(dz, 1e-12),
                0.5,
            ).clip(0.0, 1.0)
            land[cross] = P[cross, :2] + frac[:, None] * (Pn[cross, :2] - P[cross, :2])
            t_land[cross] = (t - dt) + frac * dt
            active[cross] = False
        P, V = Pn, Vn
    return land, t_land


class FastStrikeSpecPlanner(StrikeSpecPlanner):
    """StrikeSpecPlanner with numpy-batched Jacobian probes (see module doc).

    solve() / solve_fixed_normal() are inherited untouched; the fast path is
    the ADDITIVE solve_fast (raw dict, the hot loop) and solve_fast_spec
    (full StrikeSpec incl. net-clearance annotation, for node diagnostics).
    """

    #: productionized defaults measured in the benchmark ("ss_fastnp_a20_warm")
    PROD_DT_COARSE = 0.02
    PROD_MAX_ITER = 6

    def _forward_batch(self, Q, phi0, theta0, p_strike, v_ball, omega_ball):
        Mq = Q.shape[0]
        Ns = np.zeros((Mq, 3))
        Vr = np.zeros((Mq, 3))
        Vp = np.zeros((Mq, 3))
        Wp = np.zeros((Mq, 3))
        for i in range(Mq):
            n = self._normal_from_tilt(phi0, theta0, Q[i, 0], Q[i, 1])
            b1, b2 = self._face_basis(n)
            v_r = Q[i, 2] * n + Q[i, 3] * b1 + Q[i, 4] * b2
            v_plus, w_plus = predict_paddle_contact(
                v_ball, v_r, n, omega_ball, self.physics, self.config)
            Ns[i], Vr[i], Vp[i], Wp[i] = n, v_r, v_plus, w_plus
        land, t_land = batch_integrate_to_table_plane(
            p_strike, Vp, Wp, self.physics, self.config)
        return land, t_land, Ns, Vr, Vp, Wp

    def solve_fast(
        self,
        p_ball: np.ndarray,
        v_ball: np.ndarray,
        omega_ball: Optional[np.ndarray],
        landing_target_xy: np.ndarray,
        racket_speed_budget: float,
        max_iter: Optional[int] = None,
        tol_m: Optional[float] = None,
        q0: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Fast LM solve; same schedule/tolerances/residual as solve().

        Returns None on failure (like solve()), else a dict with the command
        (n, v_r), the outcome (landing_xy, t_land, v_plus, omega_plus),
        solver state for warm starting the next tick (q) and diagnostics
        (resid_m, iterations, phi0, theta0). No sensitivities here — that is
        the point; use solve_fast_spec(with_sensitivities=True) on demand.
        """
        max_iter = self.MAX_ITER if max_iter is None else int(max_iter)
        tol = self.TOL_M if tol_m is None else float(tol_m)
        p_ball = np.asarray(p_ball, float)
        v_ball = np.asarray(v_ball, float)
        omega_ball = np.zeros(3) if omega_ball is None else np.asarray(omega_ball, float)
        target_xy = np.asarray(landing_target_xy, float)[:2]

        phi0, theta0, v_n0 = self._initial_guess(p_ball, v_ball, target_xy)
        q = np.array([0.0, 0.0, v_n0, 0.0, 0.0]) if q0 is None else np.asarray(q0, float).copy()

        def fwd_one(qq):
            land, tl, Ns, Vr, Vp, Wp = self._forward_batch(
                qq[None], phi0, theta0, p_ball, v_ball, omega_ball)
            if np.isnan(land[0]).any():
                return None
            return dict(landing_xy=land[0], t_land=tl[0], n=Ns[0], v_r=Vr[0],
                        v_plus=Vp[0], omega_plus=Wp[0])

        fwd = fwd_one(q)
        if fwd is None:
            return None
        r = self._residual(fwd, q, target_xy)
        cost = float(r @ r)
        h = np.array([0.2, 0.2, 0.02, 0.02, 0.02])
        lam = 1e-3
        iterations = 0

        for _ in range(max_iter):
            iterations += 1
            if np.linalg.norm(fwd["landing_xy"] - target_xy) < tol:
                break
            Qp = np.repeat(q[None], 5, axis=0) + np.diag(h)   # 5 probes, ONE batch
            land_p, _, _, _, _, _ = self._forward_batch(
                Qp, phi0, theta0, p_ball, v_ball, omega_ball)
            if np.isnan(land_p).any():
                lam *= 10.0
                if lam > 1e6:
                    return None
                continue
            J = np.zeros((r.size, 5))
            for j in range(5):
                fj = dict(landing_xy=land_p[j])
                J[:, j] = (self._residual(fj, Qp[j], target_xy) - r) / h[j]

            JtJ = J.T @ J
            g = J.T @ r
            damp = np.diag(np.diag(JtJ)) + 1e-9 * np.eye(5)
            accepted = False
            for _try in range(6):
                try:
                    dq = np.linalg.solve(JtJ + lam * damp, -g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                q_new = q + dq
                fwd_new = fwd_one(q_new)
                if fwd_new is not None:
                    r_new = self._residual(fwd_new, q_new, target_xy)
                    cost_new = float(r_new @ r_new)
                    if cost_new < cost:
                        q, fwd, r, cost = q_new, fwd_new, r_new, cost_new
                        lam = max(lam * 0.3, 1e-8)
                        accepted = True
                        break
                lam *= 10.0
            if not accepted and lam > 1e6:
                return None

        resid = float(np.linalg.norm(fwd["landing_xy"] - target_xy))
        if resid >= tol:
            return None
        if float(np.linalg.norm(fwd["v_r"])) > racket_speed_budget + 1e-9:
            return None
        return dict(n=fwd["n"], v_r=fwd["v_r"], landing_xy=fwd["landing_xy"],
                    t_land=float(fwd["t_land"]), v_plus=fwd["v_plus"],
                    omega_plus=fwd["omega_plus"], resid_m=resid,
                    iterations=iterations, q=q.copy(),
                    phi0=phi0, theta0=theta0)

    def solve_fast_spec(
        self,
        p_ball: np.ndarray,
        v_ball: np.ndarray,
        omega_ball: Optional[np.ndarray],
        landing_target_xy: np.ndarray,
        racket_speed_budget: float,
        max_iter: Optional[int] = None,
        tol_m: Optional[float] = None,
        q0: Optional[np.ndarray] = None,
        with_sensitivities: bool = False,
    ) -> Optional[StrikeSpec]:
        """solve_fast + full StrikeSpec assembly (the node diagnostics path).

        Same fields as solve()'s StrikeSpec, incl. the post-solve
        net-clearance annotation (compute_net_clearance runs on the SAME
        predictor, so with dt_integrate_coarse set it uses the adaptive
        path). Sensitivities default OFF (NaN) — with_sensitivities=True
        adds the 8 central-difference rollouts back, on demand.

        Warm start q0 for the next tick is recoverable from the returned
        spec: [tilt_pitch_deg, tilt_yaw_deg, v_n_signed, *v_t_vec].
        """
        p_ball = np.asarray(p_ball, float)
        v_ball = np.asarray(v_ball, float)
        omega = np.zeros(3) if omega_ball is None else np.asarray(omega_ball, float)
        out = self.solve_fast(p_ball, v_ball, omega, landing_target_xy,
                              racket_speed_budget, max_iter=max_iter,
                              tol_m=tol_m, q0=q0)
        if out is None:
            return None
        b1, b2 = self._face_basis(out["n"])
        fwd = {
            "n": out["n"], "b1": b1, "b2": b2, "v_r": out["v_r"],
            "v_plus": out["v_plus"], "omega_plus": out["omega_plus"],
            "landing_xy": out["landing_xy"], "t_land": out["t_land"],
        }
        return self._build_spec(
            out["q"], fwd, out["phi0"], out["theta0"], p_ball, v_ball, omega,
            out["resid_m"], out["iterations"], with_sensitivities=with_sensitivities,
        )
