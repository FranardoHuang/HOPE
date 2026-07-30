"""Host-only contract tests for the N=1 counter-rally objective."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking"
    / "whole_body_tracking/tasks/tracking/mdp/counter_rally.py"
)
SPEC = importlib.util.spec_from_file_location("_counter_rally_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CR
SPEC.loader.exec_module(CR)


def _profile():
    return CR.CounterRallyObjectiveProfile()


def _task(*, yaw=0.0, incoming=(-1.0, 0.0), landing_x=2.5, speed=3.0):
    return CR.derive_counter_rally_task(
        base_goal_env_xy_m=(0.55, 0.10),
        base_yaw_env_rad=yaw,
        contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
        incoming_direction_b_yaw=incoming,
        incoming_ball_speed_at_contact_mps=speed,
        landing_depth_env_x_m=landing_x,
        profile=_profile(),
    )


def test_profile_is_strict_hashed_and_disables_only_dependent_landing_y_arms():
    profile = _profile()
    assert CR.CounterRallyObjectiveProfile.from_mapping(
        profile.to_mapping()
    ) == profile
    assert len(profile.sha256) == 64
    assert profile.inactive_curriculum_arms == (
        "landing_aim_y_lower",
        "landing_aim_y_upper",
    )
    bad = dict(profile.to_mapping())
    bad["unreviewed"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        CR.CounterRallyObjectiveProfile.from_mapping(bad)
    with pytest.raises(ValueError, match="sum to one"):
        replace(profile, reward_speed_fraction=0.24)


def test_reverse_ray_is_base_relative_and_landing_y_is_derived_not_sampled():
    task = _task()
    assert task.contact_env_m == pytest.approx((0.80, 0.0, 1.0))
    assert task.return_direction_b_yaw_xy == pytest.approx((1.0, 0.0))
    assert task.return_direction_env_xy == pytest.approx((1.0, 0.0))
    assert task.landing_aim_env_xy_m == pytest.approx((2.5, 0.0))
    assert task.target_baseline_speed_mps == pytest.approx(3.0)

    angled = _task(incoming=(-1.0, -0.1), landing_x=2.0)
    expected_y = angled.contact_env_m[1] + (
        (2.0 - angled.contact_env_m[0])
        * angled.return_direction_env_xy[1]
        / angled.return_direction_env_xy[0]
    )
    assert angled.landing_aim_env_xy_m == pytest.approx((2.0, expected_y))


def test_base_yaw_rotates_reverse_direction_but_all_coordinates_remain_env_local():
    task = _task(yaw=math.radians(10.0), landing_x=2.0)
    assert task.return_direction_env_xy == pytest.approx(
        (math.cos(math.radians(10.0)), math.sin(math.radians(10.0)))
    )
    # There is deliberately no global env origin input.  Translation into a
    # simulator's global/world frame is a later, explicit operation.
    assert task.contact_env_m[0] < _profile().opponent_baseline_x_env_m


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"yaw": math.pi}, "reverse_ray_not_opponent_bound"),
        ({"landing_x": 0.51}, "landing_depth_outside_table"),
        ({"landing_x": 1.20}, "landing_depth_not_opponent_half"),
        ({"speed": 0.99}, "incoming_speed_outside_venue_support"),
        ({"speed": 7.01}, "incoming_speed_outside_venue_support"),
        (
            {"incoming": (-0.86, -0.51), "landing_x": 2.5},
            "reverse_ray_misses_table",
        ),
    ),
)
def test_invalid_questions_fail_closed_with_named_reasons(kwargs, reason):
    with pytest.raises(CR.CounterRallyRejected) as caught:
        _task(**kwargs)
    assert caught.value.reason == reason


def test_opponent_half_boundary_is_strict_but_has_no_hidden_gap():
    profile = _profile()
    net_x = profile.table_near_x_env_m + 0.5 * profile.table_length_m
    with pytest.raises(CR.CounterRallyRejected) as caught:
        _task(landing_x=net_x)
    assert caught.value.reason == "landing_depth_not_opponent_half"
    accepted = _task(landing_x=math.nextafter(net_x, math.inf))
    assert accepted.landing_aim_env_xy_m[0] > net_x


def test_venue_yaml_loads_the_reviewed_fitted_ball_and_table_parameters():
    physics = CR.VenueBallPhysics.from_venue_yaml(
        ROOT / "configs/ball_physics_venue.yaml"
    )
    assert physics.ball_mass_kg == pytest.approx(0.0034)
    assert physics.ball_radius_m == pytest.approx(0.020)
    assert physics.drag_k_d_per_m == pytest.approx(0.1261)
    assert physics.magnus_k_m == pytest.approx(0.00444)
    assert physics.table_e_eff == pytest.approx(0.9215)
    assert physics.table_a_t == pytest.approx(0.369)
    assert physics.table_mu_safety == pytest.approx(2.0)
    assert len(physics.sha256) == 64


def test_fitted_table_impulse_reverses_normal_and_couples_slip_to_spin():
    physics = CR.VenueBallPhysics()
    velocity, spin = CR.fitted_table_impulse(
        velocity_before_mps=(4.0, 1.0, -2.0),
        spin_before_radps=(0.0, 0.0, 0.0),
        physics=physics,
    )
    assert velocity[2] == pytest.approx(2.0 * physics.table_e_eff)
    assert 0.0 < velocity[0] < 4.0
    assert 0.0 < velocity[1] < 1.0
    assert spin[0] < 0.0
    assert spin[1] > 0.0
    with pytest.raises(CR.CounterRallyRejected) as caught:
        CR.fitted_table_impulse(
            velocity_before_mps=(4.0, 0.0, 0.1),
            spin_before_radps=(0.0, 0.0, 0.0),
            physics=physics,
        )
    assert caught.value.reason == "table_contact_not_descending"


def _rollout(dt_s):
    return CR.rollout_counter_rally_eager(
        position_after_paddle_env_m=(0.8, 0.0, 1.0),
        velocity_after_paddle_mps=(5.0, 0.0, 2.0),
        spin_after_paddle_radps=(0.0, 0.0, 0.0),
        profile=_profile(),
        physics=CR.VenueBallPhysics(),
        dt_s=dt_s,
    )


def _task_for_landing(landing_x, *, speed=3.0):
    return CR.derive_counter_rally_task(
        base_goal_env_xy_m=(0.55, 0.10),
        base_yaw_env_rad=0.0,
        contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
        incoming_direction_b_yaw=(-1.0, 0.0),
        incoming_ball_speed_at_contact_mps=speed,
        landing_depth_env_x_m=landing_x,
        profile=_profile(),
    )


def test_rollout_proves_net_first_landing_one_fitted_bounce_and_far_baseline():
    outcome = _rollout(0.001)
    assert outcome.net_crossed and outcome.net_clear
    assert outcome.first_landing_valid
    assert outcome.first_landing_env_xy_m == pytest.approx(
        (2.9187215081, 0.0), abs=2.0e-6
    )
    assert outcome.table_bounce_count == 1
    assert outcome.opponent_baseline_crossed
    assert outcome.baseline_velocity_mps is not None
    assert outcome.baseline_velocity_mps[0] > 0.0
    assert outcome.rejection_reason is None


def test_half_millisecond_and_one_millisecond_rollouts_agree():
    coarse = _rollout(0.001)
    fine = _rollout(0.0005)
    assert coarse.first_landing_env_xy_m == pytest.approx(
        fine.first_landing_env_xy_m, abs=2.0e-5
    )
    assert coarse.baseline_velocity_mps == pytest.approx(
        fine.baseline_velocity_mps, abs=2.0e-4
    )
    assert coarse.baseline_time_s == pytest.approx(
        fine.baseline_time_s, abs=1.0e-3
    )


def test_eager_and_batched_reference_are_identical_for_same_state():
    eager = _rollout(0.001)
    batch = CR.rollout_counter_rally_batch(
        position_after_paddle_env_m=((0.8, 0.0, 1.0), (0.8, 0.0, 1.0)),
        velocity_after_paddle_mps=((5.0, 0.0, 2.0), (5.0, 0.0, 2.0)),
        spin_after_paddle_radps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        profile=_profile(),
        physics=CR.VenueBallPhysics(),
        dt_s=0.001,
    )
    assert batch == (eager, eager)


def test_assessment_and_raw_reward_decomposition_are_bounded_and_monotone():
    outcome = _rollout(0.001)
    assert outcome.first_landing_env_xy_m is not None
    assert outcome.baseline_velocity_mps is not None
    measured_speed = math.sqrt(
        sum(component * component for component in outcome.baseline_velocity_mps)
    )
    task = CR.derive_counter_rally_task(
        base_goal_env_xy_m=(0.55, 0.10),
        base_yaw_env_rad=0.0,
        contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
        incoming_direction_b_yaw=(-1.0, 0.0),
        incoming_ball_speed_at_contact_mps=measured_speed,
        landing_depth_env_x_m=outcome.first_landing_env_xy_m[0],
        profile=_profile(),
    )
    assessment = CR.assess_counter_rally_outcome(
        task=task, outcome=outcome, profile=_profile()
    )
    assert assessment.accepted, assessment.reasons
    raw = CR.counter_rally_reward_raw(
        task=task, outcome=outcome, profile=_profile()
    )
    assert set(raw) == {"legal", "landing", "reverse", "speed", "total"}
    assert raw == pytest.approx(
        {"legal": 1.0, "landing": 1.0, "reverse": 1.0, "speed": 1.0, "total": 1.0}
    )

    wrong = replace(
        outcome,
        first_landing_env_xy_m=(
            outcome.first_landing_env_xy_m[0],
            outcome.first_landing_env_xy_m[1] + 0.10,
        ),
        baseline_velocity_mps=(
            outcome.baseline_velocity_mps[0] * 0.5,
            outcome.baseline_velocity_mps[1],
            outcome.baseline_velocity_mps[2] * 0.5,
        ),
    )
    wrong_raw = CR.counter_rally_reward_raw(
        task=task, outcome=wrong, profile=_profile()
    )
    assert 0.0 <= wrong_raw["total"] < raw["total"] <= 1.0
    wrong_assessment = CR.assess_counter_rally_outcome(
        task=task, outcome=wrong, profile=_profile()
    )
    assert not wrong_assessment.accepted
    assert {"landing_aim_miss", "baseline_speed_miss"} <= set(
        wrong_assessment.reasons
    )


@pytest.mark.parametrize(
    ("net_crossed", "net_clear", "expected_reason"),
    (
        (False, True, "net_not_crossed"),
        (True, False, "net_not_clear"),
    ),
)
def test_assessment_does_not_emit_reverse_diagnostics_for_illegal_net_paths(
    net_crossed,
    net_clear,
    expected_reason,
):
    outcome = replace(
        _rollout(0.001),
        net_crossed=net_crossed,
        net_clear=net_clear,
    )
    assert outcome.first_landing_env_xy_m is not None
    task = _task_for_landing(outcome.first_landing_env_xy_m[0])
    assessment = CR.assess_counter_rally_outcome(
        task=task,
        outcome=outcome,
        profile=_profile(),
    )
    assert not assessment.accepted
    assert expected_reason in assessment.reasons
    assert assessment.reverse_direction_error_deg is None
    assert assessment.baseline_direction_error_deg is None
    assert assessment.baseline_speed_mps is None
    assert assessment.baseline_speed_error_mps is None


def test_objective_profile_sha_mismatch_is_identity_error_not_difficulty():
    outcome = _rollout(0.001)
    assert outcome.first_landing_env_xy_m is not None
    task = _task_for_landing(outcome.first_landing_env_xy_m[0])
    changed_profile = replace(_profile(), landing_tolerance_m=0.04)
    assert task.objective_profile_sha256 != changed_profile.sha256
    with pytest.raises(
        CR.CounterRallyIdentityError,
        match="objective_profile_sha256_mismatch",
    ):
        CR.counter_rally_reward_raw(
            task=task,
            outcome=outcome,
            profile=changed_profile,
        )
    with pytest.raises(
        CR.CounterRallyIdentityError,
        match="objective_profile_sha256_mismatch",
    ):
        CR.assess_counter_rally_outcome(
            task=task,
            outcome=outcome,
            profile=changed_profile,
        )


def test_fixed_solver_precheck_preserves_action_and_owns_only_named_rejection():
    profile = _profile()
    accepted = CR.precheck_counter_rally_fixed_solver_proposal(
        frozen_action_uid=17,
        solver_action_uid=17,
        expected_objective_profile_sha256=profile.sha256,
        base_goal_env_xy_m=(0.55, 0.10),
        base_yaw_env_rad=0.0,
        contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
        incoming_direction_b_yaw=(-1.0, 0.0),
        incoming_ball_speed_at_contact_mps=3.0,
        landing_depth_env_x_m=2.5,
        profile=profile,
    )
    assert accepted.proposal_count == 1
    assert accepted.eligible_for_solver
    assert accepted.task is not None
    assert accepted.rejection_reason is None
    with pytest.raises(ValueError, match="no solver-admission ledger delta"):
        accepted.rejected_ledger_counts

    rejected = CR.precheck_counter_rally_fixed_solver_proposal(
        frozen_action_uid=17,
        solver_action_uid=17,
        expected_objective_profile_sha256=profile.sha256,
        base_goal_env_xy_m=(0.55, 0.10),
        base_yaw_env_rad=0.0,
        contact_offset_b_yaw_m=(0.25, -0.10, 1.0),
        incoming_direction_b_yaw=(-0.86, -0.51),
        incoming_ball_speed_at_contact_mps=3.0,
        landing_depth_env_x_m=2.5,
        profile=profile,
    )
    assert rejected.proposal_count == 1
    assert not rejected.eligible_for_solver
    assert rejected.task is None
    assert rejected.rejection_reason == "reverse_ray_misses_table"
    assert rejected.rejected_ledger_counts == (1, 0)


def test_ordered_solver_rejection_schema_and_all_arms_are_p1_a0():
    expected = (
        "reverse_ray_not_opponent_bound",
        "landing_depth_outside_table",
        "landing_depth_not_opponent_half",
        "landing_behind_contact",
        "reverse_ray_misses_table",
        "incoming_speed_outside_venue_support",
        "target_speed_outside_venue_support",
    )
    assert CR.COUNTER_RALLY_SOLVER_REJECTION_REASON_SCHEMA == expected
    cases = (
        ({"base_yaw_env_rad": math.pi}, expected[0]),
        ({"landing_depth_env_x_m": 0.51}, expected[1]),
        ({"landing_depth_env_x_m": 1.20}, expected[2]),
        (
            {
                "base_goal_env_xy_m": (3.0, 0.10),
                "landing_depth_env_x_m": 2.5,
            },
            expected[3],
        ),
        (
            {
                "incoming_direction_b_yaw": (-0.86, -0.51),
            },
            expected[4],
        ),
        ({"incoming_ball_speed_at_contact_mps": 0.99}, expected[5]),
        (
            {
                "profile": replace(
                    _profile(),
                    target_baseline_speed_ratio=2.0,
                ),
                "incoming_ball_speed_at_contact_mps": 4.0,
            },
            expected[6],
        ),
    )
    for overrides, reason in cases:
        profile = overrides.get("profile", _profile())
        kwargs = {
            "frozen_action_uid": 17,
            "solver_action_uid": 17,
            "expected_objective_profile_sha256": profile.sha256,
            "base_goal_env_xy_m": (0.55, 0.10),
            "base_yaw_env_rad": 0.0,
            "contact_offset_b_yaw_m": (0.25, -0.10, 1.0),
            "incoming_direction_b_yaw": (-1.0, 0.0),
            "incoming_ball_speed_at_contact_mps": 3.0,
            "landing_depth_env_x_m": 2.5,
            "profile": profile,
        }
        kwargs.update(overrides)
        result = CR.precheck_counter_rally_fixed_solver_proposal(
            **kwargs
        )
        assert result.rejection_reason == reason
        assert result.rejected_ledger_counts == (1, 0)


@pytest.mark.parametrize(
    "overrides",
    (
        {"solver_action_uid": 18},
        {"expected_objective_profile_sha256": "0" * 64},
        {"base_goal_env_xy_m": (0.55,)},
        {"incoming_direction_b_yaw": (0.0, 0.0)},
    ),
)
def test_fixed_solver_precheck_identity_or_malformed_drift_hard_stops(
    overrides,
):
    profile = _profile()
    kwargs = {
        "frozen_action_uid": 17,
        "solver_action_uid": 17,
        "expected_objective_profile_sha256": profile.sha256,
        "base_goal_env_xy_m": (0.55, 0.10),
        "base_yaw_env_rad": 0.0,
        "contact_offset_b_yaw_m": (0.25, -0.10, 1.0),
        "incoming_direction_b_yaw": (-1.0, 0.0),
        "incoming_ball_speed_at_contact_mps": 3.0,
        "landing_depth_env_x_m": 2.5,
        "profile": profile,
    }
    kwargs.update(overrides)
    with pytest.raises(CR.CounterRallyIdentityError):
        CR.precheck_counter_rally_fixed_solver_proposal(**kwargs)


def test_real_own_half_path_has_no_task_quality_diagnostics():
    profile = _profile()
    outcome = CR.rollout_counter_rally_eager(
        position_after_paddle_env_m=(0.8, 0.0, 0.85),
        velocity_after_paddle_mps=(4.0, 0.0, 0.0),
        spin_after_paddle_radps=(0.0, 0.0, 0.0),
        profile=profile,
        physics=CR.VenueBallPhysics(),
        dt_s=0.001,
    )
    assessment = CR.assess_counter_rally_outcome(
        task=_task_for_landing(2.5),
        outcome=outcome,
        profile=profile,
    )
    assert not assessment.accepted
    assert "first_landing_not_opponent_half" in assessment.reasons
    assert "landing_aim_miss" not in assessment.reasons
    assert assessment.landing_error_m is None
    assert assessment.reverse_direction_error_deg is None
    assert assessment.baseline_direction_error_deg is None
    assert assessment.baseline_speed_mps is None
    assert assessment.baseline_speed_error_mps is None


@pytest.mark.parametrize(
    (
        "position",
        "velocity",
        "max_time_s",
        "expected_reason",
        "expected_stage_reward",
    ),
    (
        (
            (0.8, 0.0, 0.95),
            (7.0, 0.0, 0.0),
            2.0,
            "net_not_clear",
            0.0,
        ),
        (
            (0.8, 0.0, 0.85),
            (4.0, 0.0, 0.0),
            2.0,
            "first_landing_own_half",
            0.0,
        ),
        (
            (0.8, 0.0, 1.0),
            (5.0, 3.0, 2.0),
            2.0,
            "first_landing_outside_table",
            0.0,
        ),
        (
            (0.8, 0.0, 1.0),
            (2.5, 0.0, 3.0),
            2.0,
            "second_table_bounce_before_baseline",
            0.65,
        ),
        (
            (0.8, 0.0, 1.0),
            (4.0, 0.0, 2.0),
            0.8,
            "rollout_horizon_exceeded",
            0.65,
        ),
    ),
)
def test_invalid_paths_have_one_primary_reason_and_cannot_harvest_shaping(
    position,
    velocity,
    max_time_s,
    expected_reason,
    expected_stage_reward,
):
    outcome = CR.rollout_counter_rally_eager(
        position_after_paddle_env_m=position,
        velocity_after_paddle_mps=velocity,
        spin_after_paddle_radps=(0.0, 0.0, 0.0),
        profile=_profile(),
        physics=CR.VenueBallPhysics(),
        dt_s=0.001,
        max_time_s=max_time_s,
    )
    assert outcome.rejection_reason == expected_reason
    net_x = (
        _profile().table_near_x_env_m
        + 0.5 * _profile().table_length_m
    )
    landing_x = (
        outcome.first_landing_env_xy_m[0]
        if outcome.first_landing_env_xy_m is not None
        and net_x < outcome.first_landing_env_xy_m[0]
        <= _profile().opponent_baseline_x_env_m
        else 2.5
    )
    raw = CR.counter_rally_reward_raw(
        task=_task_for_landing(landing_x),
        outcome=outcome,
        profile=_profile(),
    )
    assert raw["total"] == pytest.approx(expected_stage_reward, abs=1.0e-10)
    if expected_stage_reward == 0.0:
        assert raw == {
            "legal": 0.0,
            "landing": 0.0,
            "reverse": 0.0,
            "speed": 0.0,
            "total": 0.0,
        }
    else:
        assert raw["legal"] == 1.0
        assert raw["landing"] == pytest.approx(1.0)
        assert raw["reverse"] == 0.0
        assert raw["speed"] == 0.0
