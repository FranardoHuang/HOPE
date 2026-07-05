"""Vectorized torch port of the spin-equation contact model.

This is a 1:1 port of ``Record/analysis/contact_model/spin_equation.py`` (the fitted, validated
impulse model) to batched torch tensors, so it can run inside the Isaac Lab env over ``(N, 3)`` ball
states on GPU. The numpy original remains the regression oracle (see tests).

Equation (SI units), per contact:

    n        = normalize(n), oriented so the incoming relative velocity approaches the face
    u        = v_minus + omega_minus x (-R n) - v_r
    u_n      = u . n                                  (signed normal component)
    u_t      = u - u_n n                              (tangential vector)
    s(theta) = clip((a_t + b_t cos(theta)) |u_t|, 0, mu_safety (1 + e_eff) |u_n|)
    dv_t     = -s unit(u_t)
    dv_n     = -(1 + e_eff) u_n n
    dw       = -(1 / (c R)) (n x dv_t)
    v_plus   = v_minus + dv_n + dv_t
    w_plus   = omega_minus + dw

Both ``v_r`` (contact-point velocity) and ``n`` (face normal) are inputs, so the same function serves
both the static table bounce (``v_r = 0``, ``n = +Z``) and the moving paddle hit.

Venue F4: when ``params.use_e_exp`` is set (the paddle section of ``ball_physics_venue.yaml``),
``e_eff`` above is replaced per contact by ``e(u_n) = e_exp_g1 * exp(e_exp_g2 * |u_n|)`` clamped to
[0.05, 0.95] — same form and clamp as the tracking-task ``virtual_ball.predict_paddle_contact``.
"""

from __future__ import annotations

import torch

from .params import ContactParams

_EPS = 1e-12


def _normalize(v: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


def orient_normal(n: torch.Tensor, v_minus: torch.Tensor, v_r: torch.Tensor) -> torch.Tensor:
    """Orient ``n`` (per env) so the incoming relative velocity approaches the face.

    Mirrors ``spin_equation.orient_normal``: flip ``n`` where ``(v_minus - v_r) . n > 0``.
    """
    n = _normalize(n)
    approaching = torch.sum((v_minus - v_r) * n, dim=-1, keepdim=True) > 0.0
    return torch.where(approaching, -n, n)


def predict_contact(
    v_minus: torch.Tensor,
    v_r: torch.Tensor,
    n: torch.Tensor,
    omega_minus: torch.Tensor,
    params: ContactParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outgoing ``(v_plus, omega_plus)`` from the fitted spin equation. All inputs ``(N, 3)``."""
    R = params.ball_radius
    c = params.inertia_coeff

    n = orient_normal(n, v_minus, v_r)
    r = -R * n

    u = v_minus + torch.cross(omega_minus, r, dim=-1) - v_r
    u_n_signed = torch.sum(u * n, dim=-1, keepdim=True)        # (N,1)
    u_t_vec = u - u_n_signed * n                               # (N,3)
    u_t_mag = torch.linalg.norm(u_t_vec, dim=-1, keepdim=True)  # (N,1)
    u_n_abs = torch.abs(u_n_signed)                            # (N,1)

    # F4 (venue fit): velocity-dependent normal restitution for the paddle,
    # e(u_n) = g1 * exp(g2 * |u_n|), clamped so far-out-of-envelope contacts stay physical.
    if params.use_e_exp:
        e_eff: torch.Tensor | float = (
            params.e_exp_g1 * torch.exp(params.e_exp_g2 * u_n_abs)
        ).clamp(0.05, 0.95)
    else:
        e_eff = params.e_eff

    cos_theta = u_n_abs / (torch.hypot(u_t_mag, u_n_signed) + _EPS)
    raw = (params.a_t + params.b_t * cos_theta) * u_t_mag
    cap = params.mu_safety * (1.0 + e_eff) * u_n_abs
    s = torch.clamp(raw, min=0.0).minimum(cap)                 # clip(raw, 0, cap)

    # dv_t = -s * unit(u_t); guard the zero-tangential case.
    safe_dir = u_t_vec / (u_t_mag + _EPS)
    delta_v_t = torch.where(u_t_mag > _EPS, -s * safe_dir, torch.zeros_like(u_t_vec))

    delta_v_n = -(1.0 + e_eff) * u_n_signed * n
    delta_omega = -(1.0 / (c * R)) * torch.cross(n, delta_v_t, dim=-1)

    v_plus = v_minus + delta_v_n + delta_v_t
    omega_plus = omega_minus + delta_omega
    return v_plus, omega_plus
