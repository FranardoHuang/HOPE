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


def test_normal_vector_faces_opponent_side():
    cmd = _planner().plan(_incoming_strike())
    assert np.dot(cmd.n_racket, np.array([1.0, 0.0, 0.0])) > 0.0


def test_degenerate_sideways_and_reversed_normals_face_opponent():
    pl = _planner()
    cases = [
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),   # delta_v ~= 0
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # pure sideways delta_v
        (np.array([3.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),  # raw delta_v points -x
    ]
    for v_in, v_out in cases:
        _, n = pl._compute_racket_velocity(v_in, v_out, pl.config.C_r)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-9)
        assert n[0] > 0.0


def test_outgoing_velocity_with_drag_lands_near_target():
    pl = _planner()
    p_strike = np.array([0.0, -0.7625, 0.3])
    # Positions are ball-center positions: physical table contact is z=radius.
    p_land = np.array([2.055, -0.7625, pl.physics.radius])
    dt = 0.55
    v_out = pl._compute_outgoing_velocity(p_strike, p_land, dt)
    p_end, _ = pl._integrate_free_flight(p_strike, v_out, dt)
    assert np.allclose(p_end, p_land, atol=5e-4)


def test_net_clearance_reported_correctly():
    pl = _planner()
    p_strike = np.array([0.0, -0.7625, 0.3])
    clears_hi, bypass_hi = pl._check_net_clearance(p_strike, np.array([5.0, 0.0, 2.0]))
    clears_lo, _ = pl._check_net_clearance(p_strike, np.array([5.0, 0.0, -2.0]))
    assert clears_hi and not bypass_hi
    assert not clears_lo


def test_net_post_band_tracks_shifted_table_y_max():
    """The centered sim table must not inherit the arena-only [-width, 0] band."""
    table = TableParams(y_max=0.80)
    pl = RacketTargetPlanner(BallPhysics(k=0.0), PlannerConfig(), table)
    p_strike = np.array([0.0, 0.50, 0.40])

    # y=0.50 is inside the shifted net span [y_max-width-overhang,
    # y_max+overhang], but outside the old hardcoded max y=overhang.
    clears, bypasses = pl._check_net_clearance(
        p_strike, np.array([4.0, 0.0, 2.0])
    )
    assert clears and not bypasses

    # A trajectory beyond the shifted table's +Y post is still identified.
    p_outside = np.array([0.0, table.y_max + table.net_overhang + 0.05, 0.40])
    clears, bypasses = pl._check_net_clearance(
        p_outside, np.array([4.0, 0.0, 2.0])
    )
    assert not clears and bypasses


def test_plan_uses_ball_center_contact_height():
    """The legacy command path must aim the center at z=R, not through z=0."""
    pl = _planner()
    cmd = pl.plan(_incoming_strike())
    # plan may select a fallback flight time to clear the net; infer that time
    # by checking the advertised target against the candidate set.
    candidates = [pl.config.delta_t_flight, 0.4, 0.6, 0.35, 0.7, 0.3]
    errors = []
    for duration in candidates:
        end, _ = pl._integrate_free_flight(
            _incoming_strike().p_ball, cmd.v_ball_outgoing, duration
        )
        errors.append(np.linalg.norm(end - np.array([
            cmd.target_land[0], cmd.target_land[1], pl.physics.radius,
        ])))
    assert min(errors) < 5e-4


def test_invalid_strike_produces_valid_false():
    invalid = StrikeTarget(
        p_ball=np.array([0.0, -0.7625, 0.3]),
        v_ball=np.array([1.0, 0.0, 0.0]),
        t_strike=0.0, num_bounces=0, valid=False,
    )
    cmd = _planner().plan(invalid)
    assert not cmd.valid


def test_velocity_dependent_restitution_self_consistent():
    """The fixed-point e(u_n) solve must satisfy the restitution identity
    v_o_n - v_r_n = -e(u_n) * (v_i_n - v_r_n) with e = g1*exp(g2*|u_n|)."""
    pl = _planner()
    v_in = np.array([-6.0, 0.3, -1.0])
    v_out = np.array([4.0, -0.2, 2.5])
    v_r, n = pl._compute_racket_velocity(v_in, v_out, pl.config.C_r)
    v_r_n = float(np.dot(v_r, n))
    v_i_n = float(np.dot(v_in, n))
    v_o_n = float(np.dot(v_out, n))
    u_n = abs(v_i_n - v_r_n)
    e = pl.config.e_exp_g1 * np.exp(pl.config.e_exp_g2 * u_n)
    assert abs((v_o_n - v_r_n) + e * (v_i_n - v_r_n)) < 1e-3


def test_velocity_dependent_restitution_needs_faster_racket():
    """F4 physics: a faster contact has LOWER e, so the planner must command
    more racket speed than the constant-C_r model would."""
    pl = _planner()
    v_in = np.array([-7.0, 0.0, -1.0])
    v_out = np.array([5.0, 0.0, 2.0])
    v_r, n = pl._compute_racket_velocity(v_in, v_out, pl.config.C_r)
    # constant-e reference
    v_o_n = float(np.dot(v_out, n))
    v_i_n = float(np.dot(v_in, n))
    v_r_n_const = (v_o_n + pl.config.C_r * v_i_n) / (1.0 + pl.config.C_r)
    assert float(np.dot(v_r, n)) > v_r_n_const
