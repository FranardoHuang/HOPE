"""Stage 2 unit tests (HOPE_7DOF...Planner_Reference_Setup.md Section 4 gates)."""

import numpy as np

from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams


def _predictor():
    return BallTrajectoryPredictor(BallPhysics(), PlannerConfig(), TableParams())


def test_incoming_trajectory_crosses_hit_plane():
    pred = _predictor()
    # Ball above the table, moving toward P1 (-x), high enough to avoid a bounce.
    p0 = np.array([0.5, -0.7625, 0.5])
    v0 = np.array([-4.0, 0.0, 2.0])
    strike = pred.predict(p0, v0, 0.0)
    assert strike.valid
    assert abs(strike.p_ball[0] - 0.0) < 1e-6   # x_hit default = 0.0
    assert strike.num_bounces == 0
    assert 0.05 < strike.t_strike < 0.4


def test_ball_moving_away_produces_no_valid_command():
    pred = _predictor()
    p0 = np.array([0.5, -0.7625, 0.5])
    v0 = np.array([3.0, 0.0, 1.0])  # moving toward P2 (+x): never crosses x_hit
    strike = pred.predict(p0, v0, 0.0)
    assert not strike.valid


def test_table_bounce_reverses_z_velocity():
    pred = _predictor()
    v_minus = np.array([2.0, 0.0, -3.0])
    v_plus = pred._apply_bounce(v_minus)
    assert v_plus[2] > 0.0                       # downward -> upward
    assert np.isclose(v_plus[2], BallPhysics().C_v * 3.0)
    assert np.isclose(v_plus[0], BallPhysics().C_h * 2.0)


def test_bounce_then_cross_hit_plane_path():
    pred = _predictor()
    p0 = np.array([0.2, -0.7625, 0.03])
    v0 = np.array([-1.5, 0.0, -2.0])
    strike = pred.predict(p0, v0, 0.0)
    assert strike.valid
    assert strike.num_bounces == 1
    assert abs(strike.p_ball[0]) < 1e-6
    assert strike.p_ball[2] > 0.0


def test_bounce_outside_table_bounds_not_valid():
    pred = _predictor()
    on_table = np.array([1.0, -0.7625, -0.01])   # within bounds (expanded by radius)
    off_table = np.array([3.0, -0.7625, -0.01])  # x beyond table length + radius
    assert pred._is_on_table(on_table)
    assert not pred._is_on_table(off_table)
