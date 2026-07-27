"""Top-level HOPE planner pipeline combining Stages 1-3.

Call .update() with each ball position at the motion capture frame rate;
read .racket_command for the latest desired racket state.

See HOPE_7DOF_Racket_Model_based_Planner_Reference_Setup.md, Section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .ball_state_estimator import BallStateEstimator
from .ball_trajectory_predictor import BallTrajectoryPredictor, StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams
from .racket_target_planner import RacketCommand, RacketTargetPlanner


@dataclass(frozen=True)
class PlannerEstimateSnapshot:
    """Immutable estimator window and config used by one asynchronous solve."""

    t_buffer: tuple[float, ...]
    p_buffer: np.ndarray
    config: PlannerConfig


@dataclass(frozen=True)
class PlannerPipelineResult:
    """Stage-2/3 result that is inert until the ROS executor accepts it."""

    command: Optional[RacketCommand]
    strike: Optional[StrikeTarget]
    t_est: float
    config: PlannerConfig


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
        snapshot = self.ingest_measurement(t, p_ball)
        if snapshot is None:
            self._latest_command = None
            return None
        result = self.solve_snapshot(snapshot)
        self.accept_result(result)
        return result.command

    def ingest_measurement(
        self, t: float, p_ball: np.ndarray
    ) -> Optional[PlannerEstimateSnapshot]:
        """Ingest one sample and copy the latest Stage-1 estimate cheaply.

        The returned object owns every numpy buffer that the asynchronous
        Stage-2/3 worker reads.  Slow prediction therefore never holds a lock
        around the live estimator and cannot block the next 300 Hz ingest.
        ``None`` means the fit window has not become ready yet.
        """

        self.estimator.push(t, p_ball)
        if not self.estimator.ready:
            return None
        config = replace(self.config, target_land=self.config.target_land.copy())
        return PlannerEstimateSnapshot(
            t_buffer=tuple(float(v) for v in self.estimator.t_buffer),
            p_buffer=np.asarray(self.estimator.p_buffer, dtype=float).copy(),
            config=config,
        )

    def solve_snapshot(self, snapshot: PlannerEstimateSnapshot) -> PlannerPipelineResult:
        """Run Stage 2/3 from an immutable estimate without touching live caches."""

        if not isinstance(snapshot, PlannerEstimateSnapshot):
            raise TypeError("solve_snapshot requires a PlannerEstimateSnapshot")
        config = replace(snapshot.config, target_land=snapshot.config.target_land.copy())
        estimator = BallStateEstimator(config)
        estimator.t_buffer = list(snapshot.t_buffer)
        estimator.p_buffer = [row.copy() for row in snapshot.p_buffer]
        p_est, v_est, t_est = estimator.estimate()
        if v_est[0] >= 0:
            return PlannerPipelineResult(
                command=None, strike=None, t_est=float(t_est), config=config
            )
        predictor = BallTrajectoryPredictor(self.physics, config, self.table)
        target_planner = RacketTargetPlanner(self.physics, config, self.table)
        strike = predictor.predict(p_est, v_est, t_est)
        command = target_planner.plan(strike)
        return PlannerPipelineResult(
            command=command,
            strike=strike,
            t_est=float(t_est),
            config=config,
        )

    def replan_result(
        self,
        result: PlannerPipelineResult,
        *,
        target_land_y: float | None = None,
        delta_t_flight: float | None = None,
    ) -> PlannerPipelineResult:
        """Pure Stage-3 replan for a side-specific aim selected by the worker."""

        if result.strike is None:
            return result
        config = replace(result.config, target_land=result.config.target_land.copy())
        if target_land_y is not None:
            config.target_land[1] = float(target_land_y)
        if delta_t_flight is not None:
            config.delta_t_flight = float(delta_t_flight)
        command = RacketTargetPlanner(self.physics, config, self.table).plan(result.strike)
        return PlannerPipelineResult(
            command=command,
            strike=result.strike,
            t_est=result.t_est,
            config=config,
        )

    def accept_result(self, result: PlannerPipelineResult) -> None:
        """Install a freshness-checked result on the executor thread."""

        if not isinstance(result, PlannerPipelineResult):
            raise TypeError("accept_result requires a PlannerPipelineResult")
        self._latest_command = result.command
        self._latest_strike = result.strike
        self._latest_t = result.t_est
        # Side-specific aim remains visible to the strike-spec diagnostics, but
        # an old asynchronous result must never roll back the live hit plane.
        self.config.target_land = result.config.target_land.copy()
        self.config.delta_t_flight = float(result.config.delta_t_flight)

    def push_measurement(self, t: float, p_ball: np.ndarray) -> None:
        """Ingest one measurement without solving or mutating cached output.

        The ROS node uses this on cadence-rejected high-rate samples.  The
        estimator keeps the full mocap window, while ``racket_command``,
        ``strike_target`` and the command publication age remain unchanged.
        """

        self.ingest_measurement(t, p_ball)

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
