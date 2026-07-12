"""Top-level HOPE planner pipeline combining Stages 1-3.

Call .update() with each ball position at the motion capture frame rate;
read .racket_command for the latest desired racket state.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 6.
"""

from typing import Optional

import numpy as np

from .ball_state_estimator import BallStateEstimator
from .ball_trajectory_predictor import BallTrajectoryPredictor, StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams
from .racket_target_planner import RacketCommand, RacketTargetPlanner


class HOPEPlanner:
    """Top-level planner combining Stages 1-3."""

    def __init__(
        self,
        physics: Optional[BallPhysics] = None,
        config: Optional[PlannerConfig] = None,
        table: Optional[TableParams] = None,
    ):
        self.physics = physics or BallPhysics()
        self.config = config or PlannerConfig()
        self.table = table or TableParams()

        self.estimator = BallStateEstimator(self.config)
        self.predictor = BallTrajectoryPredictor(self.physics, self.config, self.table)
        self.target_planner = RacketTargetPlanner(self.physics, self.config, self.table)

        self._latest_command: Optional[RacketCommand] = None
        self._latest_strike: Optional[StrikeTarget] = None
        self._latest_t: Optional[float] = None

    def update(self, t: float, p_ball: np.ndarray) -> Optional[RacketCommand]:
        """Process a new ball position measurement.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic).
        p_ball : np.ndarray, shape (3,)
            Ball position [x, y, z] in HOPE canonical frame.

        Returns
        -------
        RacketCommand or None
        """
        self.estimator.push(t, p_ball)

        if not self.estimator.ready:
            self._latest_command = None
            return None

        p_est, v_est, t_est = self.estimator.estimate()
        self._latest_t = t_est

        # Only predict if the ball is moving toward P1 (v_x < 0).
        if v_est[0] >= 0:
            self._latest_command = None
            self._latest_strike = None
            return None

        strike = self.predictor.predict(p_est, v_est, t_est)
        self._latest_strike = strike

        command = self.target_planner.plan(strike)
        self._latest_command = command
        return command

    def push_measurement(self, t: float, p_ball: np.ndarray) -> None:
        """Ingest one measurement without solving or mutating cached output.

        The ROS node uses this on cadence-rejected high-rate samples.  The
        estimator keeps the full mocap window, while ``racket_command``,
        ``strike_target`` and the command publication age remain unchanged.
        """

        self.estimator.push(t, p_ball)

    def replan_latest(self) -> Optional[RacketCommand]:
        """Re-run Stage 3 after an optional per-side aim change.

        Stage 2 already established the current ball intercept and strike
        timestamp, so this does not touch estimator state or the strike clock.
        """

        if self._latest_strike is None:
            return self._latest_command
        self._latest_command = self.target_planner.plan(self._latest_strike)
        return self._latest_command

    @property
    def racket_command(self) -> Optional[RacketCommand]:
        return self._latest_command

    @property
    def strike_target(self) -> Optional[StrikeTarget]:
        """Latest Stage-2 prediction (ball state at the hitting plane).

        Exposed for the strike-spec diagnostics path (node.py), which needs
        the predicted ball state AT the strike, not just the racket command.
        """
        return self._latest_strike

    @property
    def time_to_strike(self) -> Optional[float]:
        """Seconds remaining until the predicted strike (positive, decreasing).

        NOTE: this returns time *remaining* (t_strike - latest sample time),
        not the absolute strike time. The reference doc skeleton returned the
        absolute time; that is corrected here so the value is positive and
        decreases as the ball approaches, per the verification gate.
        """
        if self._latest_command is None or not self._latest_command.valid:
            return None
        if self._latest_strike is None or self._latest_t is None:
            return None
        return self._latest_strike.t_strike - self._latest_t
