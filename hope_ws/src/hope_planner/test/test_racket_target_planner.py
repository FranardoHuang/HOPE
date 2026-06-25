"""Stage 3 unit tests (HOPE_7DOF...Planner_Reference_Setup.md Section 5 gates)."""

import numpy as np

from hope_planner.ball_trajectory_predictor import StrikeTarget
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.racket_target_planner import RacketTargetPlanner


def _planner():
    return RacketTargetPlanner(BallPhysics(), PlannerConfig(), TableParams())


def _incoming_strike():
    return StrikeTarget(
        p_ball=np.array([0.0, -0.7625, 0.3]),
        v_ball=np.array([-3.0, 0.0, -0.5]),
        t_strike=0.4, num_bounces=1, valid=True,
    )


def test_normal_incoming_ball_produces_valid_command():
    cmd = _planner().plan(_incoming_strike())
    assert cmd.valid
    assert cmd.num_bounces == 1


def test_normal_vector_is_unit_length():
    cmd = _planner().plan(_incoming_strike())
    assert np.isclose(np.linalg.norm(cmd.n_racket), 1.0, atol=1e-9)


def test_racket_normal_faces_opponent_side():
    cmd = _planner().plan(_incoming_strike())
    assert cmd.n_racket[0] > 0.0


def test_degenerate_delta_v_fallback_normal_faces_opponent_side():
    pl = _planner()
    _v_racket, n = pl._compute_racket_velocity(
        np.array([3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]), BallPhysics().C_h
    )
    assert n[0] > 0.0


def test_sideways_delta_v_still_gets_forward_normal_component():
    pl = _planner()
    _v_racket, n = pl._compute_racket_velocity(
        np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), BallPhysics().C_h
    )
    assert n[0] > 0.0


def test_outgoing_velocity_lands_under_drag_model():
    pl = _planner()
    strike = _incoming_strike()
    v_out = pl._compute_outgoing_velocity(
        strike.p_ball, pl.config.target_land, pl.config.delta_t_flight
    )
    p_end, _ = pl._integrate_flight(strike.p_ball, v_out, pl.config.delta_t_flight)
    assert np.allclose(p_end, pl.config.target_land, atol=2e-3)


def test_net_clearance_reported_correctly():
    pl = _planner()
    p_strike = np.array([0.0, -0.7625, 0.3])
    clears_hi, bypass_hi = pl._check_net_clearance(p_strike, np.array([5.0, 0.0, 2.0]))
    clears_lo, _ = pl._check_net_clearance(p_strike, np.array([5.0, 0.0, -2.0]))
    assert clears_hi and not bypass_hi
    assert not clears_lo


def test_invalid_strike_produces_valid_false():
    invalid = StrikeTarget(
        p_ball=np.array([0.0, -0.7625, 0.3]),
        v_ball=np.array([1.0, 0.0, 0.0]),
        t_strike=0.0, num_bounces=0, valid=False,
    )
    cmd = _planner().plan(invalid)
    assert not cmd.valid
