"""Outgoing-shot landing predictor (vectorized torch).

Given the ball state ``(p, v, omega)`` at the instant the racket hits it, forward-integrate the SAME
flight model (gravity + quadratic drag + Magnus) used everywhere else and return the FIRST descending
crossing of the table plane — i.e. where the shot lands. This is the flight half of the Record
``simulator.simulate`` first-bounce; the contact that produced ``(v, omega)`` has already happened, so
the predictor just flies the post-hit state.

Geometry is passed in (frame-dependent), so this stays frame-agnostic:
* ``contact_z``  — ball-CENTRE height at table contact (= table-surface z + ball radius)
* ``table_x``/``table_y`` — court rectangle (min, max) in the same frame
* ``net_x``      — net plane along x; the opponent half is ``x > net_x``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .flight import rk4_step
from .params import FlightParams


@dataclass
class LandingResult:
    """Per-env landing prediction (all tensors shape ``(N,)`` except ``xy`` ``(N, 2)``)."""

    xy: torch.Tensor          # predicted landing point (x, y)
    valid: torch.Tensor       # bool: a descending table-plane crossing was found within the horizon
    in_bounds: torch.Tensor   # bool: crossing falls inside the table rectangle
    on_opponent: torch.Tensor # bool: in_bounds AND on the opponent half (x > net_x)
    t_flight: torch.Tensor    # time from the hit to the landing (s)
    net_z: torch.Tensor       # ball-CENTRE height (HOPE z) where the shot first crosses the net plane
    net_valid: torch.Tensor   # bool: a forward (toward-opponent) crossing of x = net_x was found


def _z_after(p, v, omega, h, flight) -> torch.Tensor:
    return rk4_step(p, v, omega, h, flight)[0][:, 2]


def predict_landing(
    p: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    flight: FlightParams,
    *,
    contact_z: float,
    table_x: tuple[float, float],
    table_y: tuple[float, float],
    net_x: float,
    max_time: float = 2.0,
    dt: float = 1.0e-3,
    bisect_iters: int = 24,
) -> LandingResult:
    """Forward-integrate to the first descending table-plane crossing. Inputs ``(N, 3)``."""
    n = p.shape[0]
    device, dtype = p.device, p.dtype

    p = p.clone()
    v = v.clone()
    landed = torch.zeros(n, dtype=torch.bool, device=device)
    land_p = p.clone()
    t_flight = torch.zeros(n, dtype=dtype, device=device)
    # Net-plane crossing (height at x = net_x). Tracked independently of landing: a forward shot crosses
    # the net (in x) while still airborne, well before its descending table-plane crossing (in z).
    net_z = torch.zeros(n, dtype=dtype, device=device)
    net_found = torch.zeros(n, dtype=torch.bool, device=device)

    t = 0.0
    n_steps = max(1, int(round(max_time / dt)))
    for _ in range(n_steps):
        p_next, v_next = rk4_step(p, v, omega, dt, flight)

        # Forward crossing of the net plane within this step (toward the opponent, +x), for still-flying,
        # not-yet-found envs. Bisect the sub-step fraction so that x(f) == net_x.
        net_cross = (~net_found) & (~landed) & (p[:, 0] <= net_x) & (p_next[:, 0] > net_x)
        if torch.any(net_cross):
            lo = torch.zeros(n, dtype=dtype, device=device)
            hi = torch.full((n,), dt, dtype=dtype, device=device)
            for _b in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                x_mid = rk4_step(p, v, omega, mid.unsqueeze(-1), flight)[0][:, 0]
                before = x_mid < net_x
                lo = torch.where(before, mid, lo)
                hi = torch.where(before, hi, mid)
            fn = (0.5 * (lo + hi)).unsqueeze(-1)
            p_net = rk4_step(p, v, omega, fn, flight)[0]
            net_z = torch.where(net_cross, p_net[:, 2], net_z)
            net_found = net_found | net_cross

        # Descending crossing of the contact plane within this step, for not-yet-landed envs.
        crossing = (~landed) & (p[:, 2] > contact_z) & (p_next[:, 2] <= contact_z) & (v[:, 2] < 0.0)

        if torch.any(crossing):
            # Bisection on the sub-step fraction f in [0, dt] so that z(f) == contact_z (descending).
            lo = torch.zeros(n, dtype=dtype, device=device)
            hi = torch.full((n,), dt, dtype=dtype, device=device)
            for _b in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                z_mid = _z_after(p, v, omega, mid.unsqueeze(-1), flight)
                above = z_mid > contact_z
                lo = torch.where(above, mid, lo)
                hi = torch.where(above, hi, mid)
            f = (0.5 * (lo + hi)).unsqueeze(-1)
            p_cross = rk4_step(p, v, omega, f, flight)[0]

            land_p = torch.where(crossing.unsqueeze(-1), p_cross, land_p)
            t_flight = torch.where(crossing, t + f.squeeze(-1), t_flight)
            landed = landed | crossing

        if torch.all(landed):
            break

        # Advance only the still-flying envs (landed envs are frozen at their crossing).
        keep = (~landed).unsqueeze(-1)
        p = torch.where(keep, p_next, p)
        v = torch.where(keep, v_next, v)
        t += dt

    x, y = land_p[:, 0], land_p[:, 1]
    in_bounds = landed & (x >= table_x[0]) & (x <= table_x[1]) & (y >= table_y[0]) & (y <= table_y[1])
    on_opponent = in_bounds & (x > net_x)
    return LandingResult(
        xy=land_p[:, :2].clone(),
        valid=landed,
        in_bounds=in_bounds,
        on_opponent=on_opponent,
        t_flight=t_flight,
        net_z=net_z,
        net_valid=net_found,
    )
