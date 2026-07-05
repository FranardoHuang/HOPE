"""Vectorized torch flight model: gravity + quadratic drag + Magnus.

Port of ``Record/analysis/flight_model/simulator.py`` (``flight_accel`` / ``rk4``) to batched torch
tensors. ``omega`` is the PHYSICAL spin and is treated as constant during flight (no aero spin decay),
matching the reference. Used by the landing predictor and the unit tests; the per-substep aero force in
the live env is applied separately by :mod:`..ball` (PhysX integrates it).
"""

from __future__ import annotations

import torch

from .params import FlightParams


def gravity_vec(g: float, like: torch.Tensor) -> torch.Tensor:
    """``(0, 0, -g)`` broadcast to the shape/device/dtype of ``like`` (N, 3)."""
    out = torch.zeros_like(like)
    out[..., 2] = -g
    return out


def flight_accel(v: torch.Tensor, omega: torch.Tensor, flight: FlightParams) -> torch.Tensor:
    """a = g - k_d |v| v + k_m (omega x v). Inputs ``(N, 3)``."""
    speed = torch.linalg.norm(v, dim=-1, keepdim=True)
    a = gravity_vec(flight.g, v) - flight.k_d * speed * v
    if flight.k_m != 0.0:
        a = a + flight.k_m * torch.cross(omega, v, dim=-1)
    return a


def rk4_step(
    p: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    h: float,
    flight: FlightParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One RK4 flight step of duration ``h`` (omega constant). Returns ``(p_new, v_new)``.

    Matches ``simulator.rk4``: position is advanced with the RK4-weighted velocity samples.
    """
    a1 = flight_accel(v, omega, flight)
    a2 = flight_accel(v + 0.5 * h * a1, omega, flight)
    a3 = flight_accel(v + 0.5 * h * a2, omega, flight)
    a4 = flight_accel(v + h * a3, omega, flight)
    v_new = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    p_new = p + (h / 6.0) * (v + 2 * (v + 0.5 * h * a1) + 2 * (v + 0.5 * h * a2) + (v + h * a3))
    return p_new, v_new
