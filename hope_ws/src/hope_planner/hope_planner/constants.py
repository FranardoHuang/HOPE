"""Physical constants and tuning parameters for the HOPE planner.

Values follow the HOPE canonical world frame (origin at P1 near-side left
corner, X toward P2, Y left, Z up) and ITTF regulation table dimensions.
See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 2.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TableParams:
    """ITTF regulation table dimensions in the HOPE canonical frame."""

    length: float = 2.74          # m, along X
    width: float = 1.525          # m, along -Y
    height: float = 0.76          # m, table surface above floor
    net_x: float = 1.37           # m, net position along X
    net_height: float = 0.1525    # m, net height above table surface
    net_overhang: float = 0.15    # m, net extends past each table edge in Y


@dataclass
class BallPhysics:
    """Aerodynamic and restitution parameters.

    Defaults are the 2026-07-03 venue fit on the MATCH ball (retro-reflective
    coated, 3.4 g) — single source of truth: configs/ball_physics_venue.yaml
    + docs/ball_physics_fit_report.md. calibrate_ball_physics can refit from
    recorded trajectories; its estimator is cruder than the yaml pipeline, so
    prefer the yaml values unless the venue changes.
    """

    k: float = 0.1261            # 1/m — QUADRATIC drag accel coefficient (a = -k|v|v).
                                 # Venue fit (C_d ~ 0.57 coated ball). The old default 0.5
                                 # (mislabeled "s/m") over-dragged 4x.
    C_h: float = 0.64            # horizontal (tangential) bounce retention. No-spin equivalent
                                 # of the grip tangential block: v_t+ = (1 - a_t) v_t with
                                 # a_t = 0.369 (101-bounce M-matrix 0.641). NOTE this diagonal
                                 # model cannot represent spin<->velocity coupling at the bounce;
                                 # incoming topspin makes the real outgoing v_t larger.
    C_v: float = 0.9215          # vertical restitution e_n, venue table fit (v_n 1.0-4.5 m/s,
                                 # forensics-corrected; old table read 0.908).
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    radius: float = 0.02         # ball radius, 40 mm diameter
    mass: float = 0.0034         # 3.4 g — the coated MATCH ball (clean ITTF ball is 2.7 g)


@dataclass
class PlannerConfig:
    """Planner tuning parameters."""

    # State estimation
    poly_order: int = 2           # polynomial fit order
    fit_window: int = 31          # number of position samples for velocity fit
                                  # (~103 ms at 300 Hz; venue noise-floor MC recommends >=100 ms —
                                  # the rig noise is ~1.9 mm white + AR(1) rho~0.94 colored)
    mocap_hz: float = 300.0       # ChingMu/VRPN venue rig streams 300 Hz (was 360 = old OptiTrack)

    # Trajectory prediction
    dt_integrate: float = 0.001   # integration time step (s)
    max_predict_time: float = 2.0  # max forward prediction horizon (s)
    bounce_z_tol: float = 0.005   # z threshold for bounce detection (m)

    # Racket planning
    x_hit: float = 0.0            # virtual hitting plane X coordinate (m)
    target_land: np.ndarray = field(
        default_factory=lambda: np.array([2.055, -0.7625, 0.0])
    )                             # center of opponent's half
    delta_t_flight: float = 0.5   # desired post-strike flight time (s)
    C_r: float = 0.654            # ball-racket normal restitution — FIRST real racket fit
                                  # (150 strikes, venue 2026-07-03). Used as the constant
                                  # fallback / fixed-point seed; the planner prefers the
                                  # velocity-dependent form below (F4: e falls with |u_n|).
    e_exp_g1: float = 0.759       # e(u_n) = g1 * exp(g2 * |u_n|), valid u_n 1.4-7.2 m/s
    e_exp_g2: float = -0.0441     # (u_n = normal approach speed in the racket frame)
    racket_radius: float = 0.075  # 7.5 cm paddle radius
