"""Load the experimentally-fitted ball-physics constants from the shared YAML.

``configs/ball_physics_venue.yaml`` (repo root; 2026-07-03 Avatar Pro venue fit on the 3.4 g
coated MATCH ball) is the **single source of truth** shared by the Isaac training env, the MuJoCo
C++ sim, the tracking-task virtual ball, and the hope_planner constant mirror. This module parses
it into small frozen dataclasses. It is pure Python (only ``PyYAML``) so it can be imported
without Isaac Lab.

Resolution order for the YAML path:
  1. ``$HOPE_BALL_PHYSICS_YAML`` if set,
  2. otherwise walk up from this file until ``configs/ball_physics_venue.yaml`` is found (the repo root).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class BallParams:
    """Physical properties of the ball (SI)."""

    mass: float            # kg
    radius: float          # m
    inertia_coeff: float   # c in I = c * m * R^2 (hollow sphere -> 2/3)


@dataclass(frozen=True)
class FlightParams:
    """Aerodynamic flight law: a = g - k_d |v| v + k_m_eff (omega x v).

    ``k_m_eff = k_m`` by default (the fitted constant-k_m law, oracle parity). With
    ``use_saturating_magnus=True`` the venue-yaml F2 saturating lift law is applied as
    ``k_m_eff(SR) = k_m / (1 + SR / magnus_sr_sat)`` with spin ratio ``SR = R |omega| / |v|`` —
    algebraically the yaml's ``C_L(SR) = cl_slope*SR / (1 + cl_slope*SR/cl_max)`` expressed
    relative to its own low-SR linearization (``magnus_sr_sat = cl_max / cl_slope``), so it is
    identical to constant k_m at low SR and only bends where the fit has no coverage. Behavior
    switch stays OFF by default (team convention: flag-gated, default off); flip it per-run with
    ``dataclasses.replace(cfg.flight, use_saturating_magnus=True)``.
    """

    k_d: float             # 1/m  (quadratic-drag accel coefficient; NOT s/m)
    k_m: float             # Magnus coefficient bridging PHYSICAL spin -> force
    g: float               # m/s^2 gravity magnitude (acts along -Z)
    rk4_h: float           # s reference RK4 step
    ball_radius: float = 0.020        # m (for SR); folded in from the ball section
    magnus_sr_sat: float = 0.0        # cl_max/cl_slope from the yaml; 0 -> saturation data absent
    use_saturating_magnus: bool = False  # behavior switch (NOT read from yaml)


@dataclass(frozen=True)
class ContactParams:
    """Spin-equation impulse parameters for one contact type (table or paddle).

    ``ball_radius`` (R) and ``inertia_coeff`` (c) are folded in from the ball section so a single
    ``ContactParams`` fully parameterizes :func:`..physics.spin_contact.predict_contact`.

    F4 (venue fit): the paddle's normal restitution is velocity-dependent. When ``use_e_exp`` is
    set, ``predict_contact`` replaces the constant ``e_eff`` with
    ``e(u_n) = e_exp_g1 * exp(e_exp_g2 * |u_n|)`` (clamped to [0.05, 0.95]); ``e_eff`` remains as
    the documented constant fallback. The table stays constant-e (venue F3: flat in v_n).
    """

    e_eff: float           # normal restitution (constant, or fallback when use_e_exp)
    a_t: float             # tangential gain (constant term)
    b_t: float             # tangential gain (cos(theta) term)
    mu_safety: float       # Coulomb cap multiplier
    ball_radius: float     # m
    inertia_coeff: float   # c
    e_exp_g1: float = 0.0  # e(u_n) = g1 * exp(g2 * |u_n|)  (venue F4, paddle only)
    e_exp_g2: float = 0.0
    use_e_exp: bool = False


@dataclass(frozen=True)
class BallPhysicsConfig:
    """Everything the flight + contact + landing models need."""

    ball: BallParams
    air_rho: float
    flight: FlightParams
    table: ContactParams
    paddle: ContactParams
    source_path: str


def default_yaml_path() -> str:
    """Resolve the canonical ``ball_physics_venue.yaml`` path (env override, then upward search)."""
    env = os.environ.get("HOPE_BALL_PHYSICS_YAML")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "configs", "ball_physics_venue.yaml")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "ball_physics_venue.yaml not found; set $HOPE_BALL_PHYSICS_YAML or place it at "
        "<repo>/configs/ball_physics_venue.yaml"
    )


def load_ball_physics(path: str | None = None) -> BallPhysicsConfig:
    """Parse the shared YAML into a :class:`BallPhysicsConfig`."""
    path = path or default_yaml_path()
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    ball = BallParams(
        mass=float(raw["ball"]["mass"]),
        radius=float(raw["ball"]["radius"]),
        inertia_coeff=float(raw["ball"]["inertia_coeff"]),
    )
    fl = raw["flight"]
    sr_sat = 0.0
    if "cl_slope" in fl and "cl_max" in fl and float(fl["cl_slope"]) > 0.0:
        sr_sat = float(fl["cl_max"]) / float(fl["cl_slope"])
    flight = FlightParams(
        k_d=float(fl["k_d"]),
        k_m=float(fl["k_m"]),
        g=float(fl["g"]),
        rk4_h=float(fl["rk4_h"]),
        ball_radius=ball.radius,
        magnus_sr_sat=sr_sat,
    )

    def _contact(section: dict) -> ContactParams:
        has_e_exp = "e_exp_g1" in section and "e_exp_g2" in section
        return ContactParams(
            e_eff=float(section["e_eff"]),
            a_t=float(section["a_t"]),
            b_t=float(section["b_t"]),
            mu_safety=float(section["mu_safety"]),
            ball_radius=ball.radius,
            inertia_coeff=ball.inertia_coeff,
            e_exp_g1=float(section["e_exp_g1"]) if has_e_exp else 0.0,
            e_exp_g2=float(section["e_exp_g2"]) if has_e_exp else 0.0,
            use_e_exp=has_e_exp,
        )

    return BallPhysicsConfig(
        ball=ball,
        air_rho=float(raw["air"]["rho"]),
        flight=flight,
        table=_contact(raw["contact"]["table"]),
        paddle=_contact(raw["contact"]["paddle"]),
        source_path=os.path.abspath(path),
    )


@functools.lru_cache(maxsize=4)
def get_ball_physics(path: str | None = None) -> BallPhysicsConfig:
    """Cached :func:`load_ball_physics` (the YAML never changes within a run)."""
    return load_ball_physics(path)
