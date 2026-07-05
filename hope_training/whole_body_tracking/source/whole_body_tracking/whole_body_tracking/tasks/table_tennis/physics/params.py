"""Load the experimentally-fitted ball-physics constants from the shared YAML.

``configs/ball_physics.yaml`` (repo root) is the **single source of truth** shared by the Isaac
training env, the MuJoCo C++ sim, and the offline Record reference. This module parses it into small
frozen dataclasses. It is pure Python (only ``PyYAML``) so it can be imported without Isaac Lab.

Resolution order for the YAML path:
  1. ``$HOPE_BALL_PHYSICS_YAML`` if set,
  2. otherwise walk up from this file until ``configs/ball_physics.yaml`` is found (the repo root).
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
    """Aerodynamic flight law: a = g - k_d |v| v + k_m (omega x v)."""

    k_d: float             # 1/m  (quadratic-drag accel coefficient; NOT s/m)
    k_m: float             # Magnus coefficient bridging PHYSICAL spin -> force
    g: float               # m/s^2 gravity magnitude (acts along -Z)
    rk4_h: float           # s reference RK4 step


@dataclass(frozen=True)
class ContactParams:
    """Spin-equation impulse parameters for one contact type (table or paddle).

    ``ball_radius`` (R) and ``inertia_coeff`` (c) are folded in from the ball section so a single
    ``ContactParams`` fully parameterizes :func:`..physics.spin_contact.predict_contact`.
    """

    e_eff: float           # normal restitution
    a_t: float             # tangential gain (constant term)
    b_t: float             # tangential gain (cos(theta) term)
    mu_safety: float       # Coulomb cap multiplier
    ball_radius: float     # m
    inertia_coeff: float   # c


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
    """Resolve the canonical ``ball_physics.yaml`` path (env override, then upward search)."""
    env = os.environ.get("HOPE_BALL_PHYSICS_YAML")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "configs", "ball_physics.yaml")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "ball_physics.yaml not found; set $HOPE_BALL_PHYSICS_YAML or place it at <repo>/configs/ball_physics.yaml"
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
    flight = FlightParams(
        k_d=float(raw["flight"]["k_d"]),
        k_m=float(raw["flight"]["k_m"]),
        g=float(raw["flight"]["g"]),
        rk4_h=float(raw["flight"]["rk4_h"]),
    )

    def _contact(section: dict) -> ContactParams:
        return ContactParams(
            e_eff=float(section["e_eff"]),
            a_t=float(section["a_t"]),
            b_t=float(section["b_t"]),
            mu_safety=float(section["mu_safety"]),
            ball_radius=ball.radius,
            inertia_coeff=ball.inertia_coeff,
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
