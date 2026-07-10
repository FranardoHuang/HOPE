"""Simulator-free virtual hit/return scorer shared by offline evaluation.

This module is the NumPy specification of the Isaac training metric implemented in
``tracking/mdp/virtual_ball.py`` and consumed by ``hope_commands._vb_evaluate``.  It deliberately
does not import Isaac, MuJoCo, or ``hope_planner``.  The scoring contract is:

* contact: strict ``pos_err < capture_radius`` and strict oriented approach-speed threshold;
* flight: the venue drag + Magnus ODE, fixed-step RK4, constant spin;
* landing: first ball-CENTRE crossing of ``table_surface_z + ball_radius``;
* net: first +X net-plane crossing before a previously detected landing, with clearance measured
  at ``table_surface_z + net_height + ball_radius``;
* legal return: contacted, net clear, and the first landing lies on the opponent half in bounds.

The fixed 10 ms / 100-step rollout is intentional: it is the fast scorer used by the Isaac
training metric.  Do not replace it with the planner's legacy 1 ms Euler / bare-z=0 landing
routine; that is a different forward model and shifts representative landings by centimetres.
Training rewards are not imported or changed here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


_EPS = 1.0e-12


@dataclass(frozen=True)
class VirtualBallParams:
    """Venue-fitted flight and paddle-contact constants."""

    k_d: float
    k_m: float
    g: float
    ball_radius: float
    inertia_coeff: float
    paddle_a_t: float
    paddle_b_t: float
    paddle_mu: float
    paddle_e_g1: float
    paddle_e_g2: float
    source_path: str


@dataclass(frozen=True)
class VirtualReturnSpec:
    """Geometry and threshold contract for one scorer instance."""

    table_surface_z: float
    net_x: float
    far_x: float
    half_width: float
    net_height: float
    capture_radius: float = 0.095
    min_approach_speed: float = 0.3
    rollout_h: float = 0.01
    rollout_steps: int = 100


@dataclass(frozen=True)
class FlightEvents:
    """First landing/net events from the fixed-length RK4 rollout."""

    landing_valid: bool
    landing_xy: np.ndarray
    flight_time: float
    net_crossed: bool
    net_z: float


@dataclass(frozen=True)
class VirtualReturnOutcome:
    """Complete score plus outgoing state, useful for parity tests and diagnostics."""

    contacted: bool
    approach_speed: float
    outgoing_vel: np.ndarray
    outgoing_spin: np.ndarray
    landing_valid: bool
    landing_xy: np.ndarray
    flight_time: float
    on_opponent: bool
    net_crossed: bool
    net_z: float
    net_clear: bool
    landed_ok: bool
    land_err: float


def default_venue_yaml_path() -> str:
    """Resolve the repository's venue-fit YAML without importing another package."""

    env = os.environ.get("HOPE_BALL_PHYSICS_YAML")
    if env:
        return env
    directory = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(directory, "configs", "ball_physics_venue.yaml")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    raise FileNotFoundError(
        "ball_physics_venue.yaml not found; set $HOPE_BALL_PHYSICS_YAML or place it at "
        "<repo>/configs/ball_physics_venue.yaml"
    )


def load_venue_params(path: str | None = None) -> VirtualBallParams:
    """Load the same venue YAML fields as the Isaac Torch scorer."""

    import yaml  # optional for callers that pass already-loaded/mirrored constants

    source = os.path.abspath(path or default_venue_yaml_path())
    with open(source, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    paddle = raw["contact"]["paddle"]
    return VirtualBallParams(
        k_d=float(raw["flight"]["k_d"]),
        k_m=float(raw["flight"]["k_m"]),
        g=float(raw["flight"]["g"]),
        ball_radius=float(raw["ball"]["radius"]),
        inertia_coeff=float(raw["ball"]["inertia_coeff"]),
        paddle_a_t=float(paddle["a_t"]),
        paddle_b_t=float(paddle["b_t"]),
        paddle_mu=float(paddle["mu_safety"]),
        paddle_e_g1=float(paddle["e_exp_g1"]),
        paddle_e_g2=float(paddle["e_exp_g2"]),
        source_path=source,
    )


def _vec3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite, got {array}")
    return array


def orient_normal(normal, incoming_vel, racket_vel) -> np.ndarray:
    """Orient a face normal exactly like ``virtual_ball.orient_normal``."""

    n = _vec3(normal, "normal")
    v_in = _vec3(incoming_vel, "incoming_vel")
    v_r = _vec3(racket_vel, "racket_vel")
    n = n / (float(np.linalg.norm(n)) + _EPS)
    if float(np.dot(v_in - v_r, n)) > 0.0:
        n = -n
    return n


def predict_paddle_contact(
    incoming_vel,
    racket_vel,
    normal,
    incoming_spin,
    params: VirtualBallParams,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy port of the Isaac scorer's fitted paddle impulse law."""

    v_in = _vec3(incoming_vel, "incoming_vel")
    v_r = _vec3(racket_vel, "racket_vel")
    omega = _vec3(incoming_spin, "incoming_spin")
    n = orient_normal(normal, v_in, v_r)

    radius = params.ball_radius
    contact_offset = -radius * n
    relative = v_in + np.cross(omega, contact_offset) - v_r
    normal_signed = float(np.dot(relative, n))
    tangent = relative - normal_signed * n
    tangent_mag = float(np.linalg.norm(tangent))
    normal_abs = abs(normal_signed)

    restitution = float(np.clip(
        params.paddle_e_g1 * np.exp(params.paddle_e_g2 * normal_abs), 0.05, 0.95
    ))
    cos_theta = normal_abs / (float(np.hypot(tangent_mag, normal_signed)) + _EPS)
    raw_tangent = (params.paddle_a_t + params.paddle_b_t * cos_theta) * tangent_mag
    friction_cap = params.paddle_mu * (1.0 + restitution) * normal_abs
    impulse_tangent = min(max(raw_tangent, 0.0), friction_cap)

    if tangent_mag > _EPS:
        delta_v_tangent = -impulse_tangent * tangent / (tangent_mag + _EPS)
    else:
        delta_v_tangent = np.zeros(3, dtype=np.float64)
    delta_v_normal = -(1.0 + restitution) * normal_signed * n
    delta_spin = -(1.0 / (params.inertia_coeff * radius)) * np.cross(n, delta_v_tangent)
    return v_in + delta_v_normal + delta_v_tangent, omega + delta_spin


def flight_accel(velocity, spin, params: VirtualBallParams) -> np.ndarray:
    """Venue flight ODE: gravity + quadratic drag + Magnus acceleration."""

    v = _vec3(velocity, "velocity")
    omega = _vec3(spin, "spin")
    accel = -params.k_d * float(np.linalg.norm(v)) * v + params.k_m * np.cross(omega, v)
    accel[2] -= params.g
    return accel


def rk4_step(position, velocity, spin, h: float, params: VirtualBallParams) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 flight step with the same tableau as the Isaac Torch scorer."""

    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    omega = _vec3(spin, "spin")
    a1 = flight_accel(v, omega, params)
    v2 = v + 0.5 * h * a1
    a2 = flight_accel(v2, omega, params)
    v3 = v + 0.5 * h * a2
    a3 = flight_accel(v3, omega, params)
    v4 = v + h * a3
    a4 = flight_accel(v4, omega, params)
    p_new = p + (h / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4)
    v_new = v + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return p_new, v_new


def coarse_flight_events(
    position,
    velocity,
    spin,
    params: VirtualBallParams,
    *,
    contact_plane_z: float,
    net_x: float,
    h: float = 0.01,
    n_steps: int = 100,
) -> FlightEvents:
    """First table/net events under the Isaac metric's fixed-length RK4 rollout."""

    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    p = _vec3(position, "position").copy()
    v = _vec3(velocity, "velocity").copy()
    omega = _vec3(spin, "spin")

    landed = False
    net_crossed = False
    landing_xy = np.full(2, np.nan, dtype=np.float64)
    flight_time = float("nan")
    net_z = float("nan")
    for step in range(int(n_steps)):
        p_new, v_new = rk4_step(p, v, omega, float(h), params)

        # Keep the event ordering/masks of virtual_ball.coarse_landing: the net check reads the
        # landing latch from the previous step, then the landing check updates it.
        net_event = (
            (not net_crossed)
            and (not landed)
            and (p[0] < net_x)
            and (p_new[0] >= net_x)
        )
        if net_event:
            dx = float(p_new[0] - p[0])
            fraction = float(np.clip((net_x - p[0]) / max(dx, _EPS), 0.0, 1.0))
            net_z = float(p[2] + (p_new[2] - p[2]) * fraction)
            net_crossed = True

        landing_event = (not landed) and (p[2] > contact_plane_z) and (p_new[2] <= contact_plane_z)
        if landing_event:
            dz = float(p[2] - p_new[2])
            fraction = float(np.clip((p[2] - contact_plane_z) / max(dz, _EPS), 0.0, 1.0))
            landing_xy = p[:2] + (p_new[:2] - p[:2]) * fraction
            flight_time = (step + fraction) * float(h)
            landed = True

        p, v = p_new, v_new

    return FlightEvents(
        landing_valid=landed,
        landing_xy=landing_xy,
        flight_time=flight_time,
        net_crossed=net_crossed,
        net_z=net_z,
    )


class VirtualReturnScorer:
    """Score achieved racket state using the Isaac metric contract, without a simulator."""

    def __init__(self, params: VirtualBallParams, spec: VirtualReturnSpec):
        self.params = params
        self.spec = spec
        if spec.net_x >= spec.far_x:
            raise ValueError(f"net_x must be before far_x, got {spec.net_x} >= {spec.far_x}")
        if spec.half_width <= 0.0:
            raise ValueError(f"half_width must be positive, got {spec.half_width}")

    @property
    def contact_plane_z(self) -> float:
        return self.spec.table_surface_z + self.params.ball_radius

    @property
    def net_clear_center_z(self) -> float:
        return self.spec.table_surface_z + self.spec.net_height + self.params.ball_radius

    def flight_events(self, position, velocity, spin) -> FlightEvents:
        return coarse_flight_events(
            position,
            velocity,
            spin,
            self.params,
            contact_plane_z=self.contact_plane_z,
            net_x=self.spec.net_x,
            h=self.spec.rollout_h,
            n_steps=self.spec.rollout_steps,
        )

    def score(
        self,
        *,
        ball_vel,
        ball_spin,
        racket_pos,
        racket_vel,
        racket_normal,
        pos_err: float,
        intended_landing_xy=None,
    ) -> VirtualReturnOutcome:
        """Score one exact-strike attempt; all vectors are in the same world/env-local frame."""

        v_in = _vec3(ball_vel, "ball_vel")
        v_r = _vec3(racket_vel, "racket_vel")
        omega_in = _vec3(ball_spin, "ball_spin")
        normal = _vec3(racket_normal, "racket_normal")
        racket_position = _vec3(racket_pos, "racket_pos")
        if not np.isfinite(float(pos_err)):
            raise ValueError(f"pos_err must be finite, got {pos_err}")

        oriented = orient_normal(normal, v_in, v_r)
        approach = float(np.dot(v_r, oriented))
        contacted = (
            float(pos_err) < self.spec.capture_radius
            and approach > self.spec.min_approach_speed
        )
        v_plus, omega_plus = predict_paddle_contact(v_in, v_r, normal, omega_in, self.params)
        events = self.flight_events(racket_position, v_plus, omega_plus)

        on_opponent = bool(
            events.landing_valid
            and events.landing_xy[0] > self.spec.net_x
            and events.landing_xy[0] <= self.spec.far_x
            and abs(float(events.landing_xy[1])) <= self.spec.half_width
        )
        net_clear = bool(events.net_crossed and events.net_z > self.net_clear_center_z)
        landed_ok = bool(contacted and net_clear and on_opponent)
        land_err = float("nan")
        if intended_landing_xy is not None and events.landing_valid:
            intended = np.asarray(intended_landing_xy, dtype=np.float64)
            if intended.shape != (2,) or not np.isfinite(intended).all():
                raise ValueError(f"intended_landing_xy must be finite shape (2,), got {intended}")
            land_err = float(np.linalg.norm(events.landing_xy - intended))

        return VirtualReturnOutcome(
            contacted=contacted,
            approach_speed=approach,
            outgoing_vel=v_plus,
            outgoing_spin=omega_plus,
            landing_valid=events.landing_valid,
            landing_xy=events.landing_xy,
            flight_time=events.flight_time,
            on_opponent=on_opponent,
            net_crossed=events.net_crossed,
            net_z=events.net_z,
            net_clear=net_clear,
            landed_ok=landed_ok,
            land_err=land_err,
        )
