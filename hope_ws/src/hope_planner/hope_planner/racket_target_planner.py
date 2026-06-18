"""Stage 3 - Racket target planning.

Given the predicted ball state at the hitting plane (Stage 2), compute the
desired racket velocity and face orientation to return the ball to the
opponent's half center, with a net-clearance check and flight-time fallback.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 5.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .ball_trajectory_predictor import StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams


@dataclass
class RacketCommand:
    """Output of Stage 3: desired racket state at strike time.

    This is the planner's output to the whole-body controller. The racket's
    actual pose is inferred by the humanoid via forward kinematics - it is
    never measured by the motion capture system.
    """

    p_intercept: np.ndarray   # desired racket center position at interception
    v_racket: np.ndarray      # desired racket velocity vector [vx, vy, vz]
    n_racket: np.ndarray      # desired racket face normal (unit vector)
    t_strike: float           # predicted time of strike
    v_ball_outgoing: np.ndarray  # expected outgoing ball velocity
    target_land: np.ndarray   # intended landing point
    clears_net: bool          # True if return trajectory clears the net
    bypasses_net_posts: bool  # True if ball passes outside net Y extent
    valid: bool               # True if all computations succeeded
    num_bounces: int = 0      # bounces predicted before the strike (from Stage 2)


class RacketTargetPlanner:
    """Compute desired racket velocity and orientation for a valid return.

    Implements HITTER Section III-C: racket-ball interaction model.
    """

    def __init__(self, physics: BallPhysics, config: PlannerConfig, table: TableParams):
        self.physics = physics
        self.config = config
        self.table = table

    def _compute_outgoing_velocity(
        self, p_strike: np.ndarray, p_land: np.ndarray, delta_t: float,
    ) -> np.ndarray:
        """v_o = (p_land - p_strike) / dt + 0.5 g dt"""
        return (p_land - p_strike) / delta_t + 0.5 * self.physics.g * delta_t

    def _compute_racket_velocity(
        self, v_incoming: np.ndarray, v_outgoing: np.ndarray, C_r: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute desired racket velocity and face normal from impact model."""
        delta_v = v_outgoing - v_incoming
        delta_v_norm = np.linalg.norm(delta_v)

        if delta_v_norm < 1e-6:
            n = -v_incoming / max(np.linalg.norm(v_incoming), 1e-6)
            return np.zeros(3), n

        u_hat = delta_v / delta_v_norm
        v_o_n = np.dot(v_outgoing, u_hat)
        v_i_n = np.dot(v_incoming, u_hat)
        v_r_n = (v_o_n + C_r * v_i_n) / (1.0 + C_r)

        return v_r_n * u_hat, u_hat

    def _check_net_clearance(
        self, p_strike: np.ndarray, v_outgoing: np.ndarray, margin: float = 0.03,
    ) -> Tuple[bool, bool]:
        """Check height clearance and Y-axis net extent."""
        x_net = self.table.net_x
        z_net = self.table.net_height

        dx = x_net - p_strike[0]
        if v_outgoing[0] <= 0:
            return False, False

        t_net = dx / v_outgoing[0]
        if t_net < 0:
            return False, False

        z_at_net = p_strike[2] + v_outgoing[2] * t_net + 0.5 * self.physics.g[2] * t_net ** 2
        y_at_net = p_strike[1] + v_outgoing[1] * t_net

        y_net_min = -self.table.width - self.table.net_overhang
        y_net_max = self.table.net_overhang

        bypasses_posts = (y_at_net < y_net_min) or (y_at_net > y_net_max)
        if bypasses_posts:
            return False, True

        return z_at_net > (z_net + margin), False

    def plan(self, strike: StrikeTarget) -> RacketCommand:
        """Compute racket target state for a valid return."""
        if not strike.valid:
            return RacketCommand(
                p_intercept=strike.p_ball, v_racket=np.zeros(3),
                n_racket=np.array([1.0, 0.0, 0.0]), t_strike=strike.t_strike,
                v_ball_outgoing=np.zeros(3), target_land=self.config.target_land.copy(),
                clears_net=False, bypasses_net_posts=False, valid=False,
                num_bounces=strike.num_bounces,
            )

        p_strike = strike.p_ball
        v_incoming = strike.v_ball
        p_land = self.config.target_land.copy()

        v_outgoing = self._compute_outgoing_velocity(
            p_strike, p_land, self.config.delta_t_flight
        )
        v_racket, n_racket = self._compute_racket_velocity(
            v_incoming, v_outgoing, self.config.C_r
        )
        clears, bypasses = self._check_net_clearance(p_strike, v_outgoing)

        # Auto-adjust flight time if net clearance fails
        if not clears:
            for dt_adj in [0.4, 0.6, 0.35, 0.7, 0.3]:
                v_out_adj = self._compute_outgoing_velocity(p_strike, p_land, dt_adj)
                clears_adj, bypasses_adj = self._check_net_clearance(p_strike, v_out_adj)
                if clears_adj:
                    v_outgoing = v_out_adj
                    v_racket, n_racket = self._compute_racket_velocity(
                        v_incoming, v_outgoing, self.config.C_r
                    )
                    clears, bypasses = True, bypasses_adj
                    break

        return RacketCommand(
            p_intercept=p_strike, v_racket=v_racket, n_racket=n_racket,
            t_strike=strike.t_strike, v_ball_outgoing=v_outgoing,
            target_land=p_land, clears_net=clears,
            bypasses_net_posts=bypasses, valid=True,
            num_bounces=strike.num_bounces,
        )
