"""Batched torch StrikeSpec inversion — training-inline "solved question" generator.

Given a batch of incoming balls (at-strike position/velocity/spin, env-local frame) and landing
targets, solve for the racket state each ball DEMANDS: face normal + contact-point velocity.
This is the torch port of hope_planner.strike_spec_planner.StrikeSpecPlanner.solve() (numpy LM,
the deploy/eval oracle) built on the SAME training-side physics (virtual_ball.predict_paddle_contact
+ coarse_landing — venue fit, so training targets and training scoring share one model).

Decision record (franco 2026-07-05): torch batched solver over an offline question bank — the
training loop solves inline at resample time (cost: a few dozen batched rollouts, kernel-launch
bound), and deploy can later share the same solver. The numpy planner stays the slow reference
oracle; scripts/test_strike_spec_torch.py cross-checks the two.

Frames: everything env-local (z above the FLOOR, table surface at ``surface_z``, net plane at
``net_x``) — the same convention as coarse_landing/_vb_evaluate. The numpy planner works with
z relative to the TABLE; the test handles the offset.

Solver: damped Gauss-Newton over q = (theta, phi, v_n, v_t1, v_t2) — absolute spherical face
angles + velocity in the face basis — seeded by the mirror law + e(u_n) restitution fixed point
(same construction as the numpy planner). Per-problem Levenberg damping with batched 5x5 solves;
problems that fail to reach ``tol_m`` / exceed the speed budget / never land return ok=False
(callers resample those balls, same semantics as VenueBallSampler).
"""
from __future__ import annotations

import torch

from .virtual_ball import VirtualBallParams, coarse_landing, predict_paddle_contact

_EPS = 1e-9


def _face_from_angles(theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Unit normal from spherical (theta=elevation, phi=azimuth) + face basis (b1, b2).

    b1 = z_hat x n (horizontal in-face), b2 = n x b1 (face-up). Near-vertical faces fall back
    to the x-axis cross (never a real return; keeps the math total, matching the numpy planner).
    """
    ct = torch.cos(theta)
    n = torch.stack([ct * torch.cos(phi), ct * torch.sin(phi), torch.sin(theta)], dim=-1)
    z_hat = torch.zeros_like(n)
    z_hat[..., 2] = 1.0
    b1 = torch.cross(z_hat, n, dim=-1)
    b1n = torch.linalg.norm(b1, dim=-1, keepdim=True)
    x_hat = torch.zeros_like(n)
    x_hat[..., 0] = 1.0
    b1_alt = torch.cross(x_hat, n, dim=-1)
    b1 = torch.where(b1n > 1e-6, b1, b1_alt)
    b1 = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + _EPS)
    b2 = torch.cross(n, b1, dim=-1)
    return n, b1, b2


def _forward_landing(
    q: torch.Tensor,
    p_strike: torch.Tensor,
    v_ball: torch.Tensor,
    w_ball: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    h: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """q (N,5) -> (landing_xy (N,2), valid (N,), v_r (N,3), n (N,3))."""
    n, b1, b2 = _face_from_angles(q[:, 0], q[:, 1])
    v_r = q[:, 2:3] * n + q[:, 3:4] * b1 + q[:, 4:5] * b2
    v_plus, w_plus = predict_paddle_contact(v_ball, v_r, n, w_ball, prm)
    land = coarse_landing(p_strike, v_plus, w_plus, prm, surface_z=surface_z, net_x=net_x,
                          h=h, n_steps=n_steps)
    return land["land_xy"], land["land_valid"], v_r, n


def _seed(
    p_strike: torch.Tensor,
    v_ball: torch.Tensor,
    target_xy: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    delta_t_flight: float,
) -> torch.Tensor:
    """Mirror-law + e(u_n) fixed-point seed, batched (numpy planner _initial_guess port)."""
    T = delta_t_flight
    p_land = torch.cat([target_xy, torch.full_like(target_xy[:, :1], surface_z)], dim=-1)
    g_vec = torch.zeros_like(p_strike)
    g_vec[:, 2] = -prm.g
    v_out = (p_land - p_strike) / T - 0.5 * g_vec * T

    delta_v = v_out - v_ball
    n0 = delta_v / (torch.linalg.norm(delta_v, dim=-1, keepdim=True) + _EPS)
    flip = n0[:, 0:1] < 0.0
    n0 = torch.where(flip, -n0, n0)          # opponent-facing (+x) convention

    v_o_n = torch.sum(v_out * n0, dim=-1)
    v_i_n = torch.sum(v_ball * n0, dim=-1)
    e = torch.full_like(v_o_n, 0.5)
    v_n0 = (v_o_n + e * v_i_n) / (1.0 + e)
    for _ in range(3):                        # e(u_n) fixed point, converges in 2-3 iters
        u_n = torch.abs(v_i_n - v_n0)
        e = (prm.paddle_e_g1 * torch.exp(prm.paddle_e_g2 * u_n)).clamp(0.05, 0.95)
        v_n0 = (v_o_n + e * v_i_n) / (1.0 + e)

    theta0 = torch.asin(n0[:, 2].clamp(-1.0, 1.0))
    phi0 = torch.atan2(n0[:, 1], n0[:, 0])
    zeros = torch.zeros_like(v_n0)
    return torch.stack([theta0, phi0, v_n0, zeros, zeros], dim=-1)


@torch.no_grad()
def solve_strike_specs(
    p_strike: torch.Tensor,
    v_ball: torch.Tensor,
    w_ball: torch.Tensor,
    target_xy: torch.Tensor,
    prm: VirtualBallParams,
    surface_z: float,
    net_x: float,
    ref_normal: torch.Tensor | None = None,
    speed_budget: float = 10.0,
    n_iters: int = 12,
    tol_m: float = 0.02,
    w_speed: float = 0.03,
    delta_t_flight: float = 0.45,
    h: float = 0.01,
    n_steps: int = 100,
) -> dict[str, torch.Tensor]:
    """Solve racket (normal, velocity) demands for a batch of balls. All args env-local.

    Returns dict: ``v_r`` (N,3), ``n`` (N,3, sign-matched to ``ref_normal`` when given),
    ``landing_xy`` (N,2), ``resid_m`` (N,), ``ok`` (N,) — ok=False rows carry the best attempt
    but failed tolerance/budget/landing; callers should resample those balls.
    """
    N = p_strike.shape[0]
    dev, dt = p_strike.device, p_strike.dtype
    fwd = lambda qq: _forward_landing(qq, p_strike, v_ball, w_ball, prm, surface_z, net_x, h, n_steps)  # noqa: E731

    q = _seed(p_strike, v_ball, target_xy, prm, surface_z, delta_t_flight)
    land_xy, valid, v_r, n = fwd(q)

    def residual(land_xy_, q_):
        return torch.cat([land_xy_ - target_xy, w_speed * q_[:, 2:5]], dim=-1)   # (N,5)

    BIG = torch.tensor(1e6, device=dev, dtype=dt)
    r = residual(land_xy, q)
    cost = torch.where(valid, torch.sum(r * r, dim=-1), BIG)

    # Jacobian steps sized to each variable's natural scale (rad / m/s) — numpy planner parity
    # (0.2 deg = 3.5e-3 rad; 0.02 m/s).
    hstep = torch.tensor([3.5e-3, 3.5e-3, 0.02, 0.02, 0.02], device=dev, dtype=dt)
    lam = torch.full((N,), 1e-3, device=dev, dtype=dt)

    for _ in range(n_iters):
        # forward-difference Jacobian: 5 batched rollouts
        J = torch.zeros(N, 5, 5, device=dev, dtype=dt)
        for j in range(5):
            qj = q.clone()
            qj[:, j] += hstep[j]
            lx_j, valid_j, _, _ = fwd(qj)
            rj = residual(lx_j, qj)
            col = (rj - r) / hstep[j]
            # a probe that leaves the landing manifold gives garbage — zero its column so the
            # step ignores that direction for this problem (mirrors numpy lam*10 retreat)
            col = torch.where(valid_j.unsqueeze(-1), col, torch.zeros_like(col))
            J[:, :, j] = col

        JtJ = J.transpose(1, 2) @ J                                   # (N,5,5)
        g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)         # (N,5)
        damp = torch.diag_embed(torch.diagonal(JtJ, dim1=1, dim2=2)) + 1e-9 * torch.eye(5, device=dev, dtype=dt)
        accepted_any = torch.zeros(N, dtype=torch.bool, device=dev)
        for _try in range(4):
            A = JtJ + lam.view(N, 1, 1) * damp
            try:
                dq = torch.linalg.solve(A, -g.unsqueeze(-1)).squeeze(-1)
            except Exception:
                dq = torch.linalg.lstsq(A, -g.unsqueeze(-1)).solution.squeeze(-1)
            q_new = q + torch.where(accepted_any.unsqueeze(-1), torch.zeros_like(dq), dq)
            lx_new, valid_new, _, _ = fwd(q_new)
            r_new = residual(lx_new, q_new)
            cost_new = torch.where(valid_new, torch.sum(r_new * r_new, dim=-1), BIG)
            better = (cost_new < cost) & (~accepted_any)
            q = torch.where(better.unsqueeze(-1), q_new, q)
            r = torch.where(better.unsqueeze(-1), r_new, r)
            cost = torch.where(better, cost_new, cost)
            lam = torch.where(better, (lam * 0.3).clamp(min=1e-8), lam)
            lam = torch.where(~better & ~accepted_any, lam * 10.0, lam)
            accepted_any = accepted_any | better
            if bool(accepted_any.all()):
                break

    land_xy, valid, v_r, n = fwd(q)
    resid = torch.linalg.norm(land_xy - target_xy, dim=-1)
    speed = torch.linalg.norm(v_r, dim=-1)
    ok = valid & (resid < tol_m) & (speed <= speed_budget + 1e-9)

    if ref_normal is not None:
        flip = torch.sum(n * ref_normal, dim=-1, keepdim=True) < 0.0
        n = torch.where(flip, -n, n)         # ±n span the same face; match the clip-face sign

    return {"v_r": v_r, "n": n, "landing_xy": land_xy, "resid_m": resid, "ok": ok}
