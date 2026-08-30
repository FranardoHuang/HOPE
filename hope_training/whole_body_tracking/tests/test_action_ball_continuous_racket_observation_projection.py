"""R08 leaf tests for the retained Racket installed-task projection.

These tests exercise one owner-private publication registry.  Device-R05's
independent current-task/shot observation ABI is not frozen, so the production
binder remains HOLD and none of these fixtures closes the production graph.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

import test_action_ball_continuous_racket_selected_reset as reset_test


HC = reset_test.HC


def _device_r05_harness(rows: int = 2):
    device_test = reset_test.device_r05_test
    return device_test._harness(rows, device="cpu")


def _racket(rows: int = 2):
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = rows
    racket.device = "cpu"
    racket._env = SimpleNamespace(common_step_counter=7)
    racket._action_ball_continuous_racket_mutation_version = 3
    racket._action_ball_full_mdp_device_r05_owner = None
    racket._action_ball_continuous_racket_observation_d05_validator = None
    racket._action_ball_continuous_racket_observation_publication_sequence = 0
    racket._action_ball_continuous_racket_observation_current_token = None
    racket._action_ball_continuous_racket_observation_records = {}

    racket._action_ball_reset_generation = torch.arange(
        1, rows + 1, dtype=torch.int64
    )
    racket._action_ball_swing_generation = torch.arange(
        3, rows + 3, dtype=torch.int64
    )
    racket._action_ball_continuous_racket_observation_scheduled_ordinal = (
        torch.arange(rows, dtype=torch.int64)
    )
    racket._action_ball_action_uid = torch.arange(
        101, rows + 101, dtype=torch.int64
    )
    racket._action_ball_action_slot = torch.arange(rows, dtype=torch.int64)
    racket._action_ball_continuous_racket_observation_task_identity = (
        torch.arange(1, rows * 32 + 1, dtype=torch.int64)
        .remainder(255)
        .to(torch.uint8)
        .reshape(rows, 32)
    )
    racket._action_ball_task_valid = torch.ones(rows, dtype=torch.bool)

    base = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    racket.racket_target_pos_w = base + 0.1
    racket.racket_target_vel_w = base + 0.2
    racket.target_normal_cmd = torch.zeros((rows, 3), dtype=torch.float32)
    racket.target_normal_cmd[:, 1] = 1.0
    racket.base_target_pos_w = base[:, :2] + 0.3
    racket._action_ball_ball_contact_target_w = base + 0.4
    racket._action_ball_face_center_velocity_target_w = base + 0.5
    racket._action_ball_racket_command_quat_w = torch.zeros(
        (rows, 4), dtype=torch.float32
    )
    racket._action_ball_racket_command_quat_w[:, 0] = 1.0
    racket.vb_vel_in_w = base + 0.6
    racket.vb_spin_in_w = base + 0.7
    racket._vb_target_xy_per_env = base[:, :2] + 0.8
    racket.time_to_strike = torch.arange(
        1, rows + 1, dtype=torch.float32
    ) * 0.1
    return racket


def test_empty_owner_token_private_registry_and_clone_only_one_shot_view():
    racket = _racket()
    with pytest.raises(TypeError, match="owner-issued"):
        HC.ActionBallContinuousRacketObservationToken()
    with pytest.raises(TypeError, match="cannot be serialized"):
        token_type = HC.ActionBallContinuousRacketObservationToken
        object.__new__(token_type).__reduce__()

    token = racket._publish_action_ball_continuous_racket_observation_for_test()
    assert HC.ActionBallContinuousRacketObservationToken.__slots__ == ()
    assert tuple(racket._action_ball_continuous_racket_observation_records) == (
        token,
    )
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(token)

    view = (
        racket.require_owned_action_ball_continuous_racket_observation_projection(
            token
        )
    )
    assert type(view) is HC.ActionBallContinuousRacketObservationView
    assert view.racket_owner is racket
    assert torch.equal(view.action_uid, racket._action_ball_action_uid)
    assert torch.equal(
        view.task_identity,
        racket._action_ball_continuous_racket_observation_task_identity,
    )
    view.action_uid.zero_()
    view.desired_position_w.add_(1000)
    assert torch.count_nonzero(racket._action_ball_action_uid) == 2
    assert torch.max(racket.racket_target_pos_w) < 1000

    with pytest.raises(RuntimeError, match="replayed, or stale"):
        racket.require_owned_action_ball_continuous_racket_observation(token)
    with pytest.raises(RuntimeError, match="forged, replayed, or stale"):
        racket.require_owned_action_ball_continuous_racket_observation(
            object()
        )


def test_stale_retained_task_identity_or_numeric_mutation_blocks_view():
    racket = _racket()
    token = racket._publish_action_ball_continuous_racket_observation_for_test()
    racket._action_ball_action_uid[0] += 1
    with pytest.raises(RuntimeError, match="stale current task or numerics"):
        racket.require_owned_action_ball_continuous_racket_observation(token)

    racket = _racket()
    token = racket._publish_action_ball_continuous_racket_observation_for_test()
    racket.racket_target_vel_w[1, 2] += 1.0
    with pytest.raises(RuntimeError, match="stale current task or numerics"):
        racket.require_owned_action_ball_continuous_racket_observation(token)

    racket = _racket()
    token = racket._publish_action_ball_continuous_racket_observation_for_test()
    racket._env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="stale current task or numerics"):
        racket.require_owned_action_ball_continuous_racket_observation(token)


def test_nonfinite_or_incomplete_installed_task_blocks_before_token_mint():
    racket = _racket()
    racket.vb_spin_in_w[0, 1] = float("nan")
    with pytest.raises(RuntimeError, match="numerics are nonfinite"):
        racket._publish_action_ball_continuous_racket_observation_for_test()
    assert racket._action_ball_continuous_racket_observation_records == {}
    assert racket._action_ball_continuous_racket_observation_publication_sequence == 0

    racket = _racket()
    racket._action_ball_continuous_racket_observation_task_identity[1].zero_()
    with pytest.raises(RuntimeError, match="identity is incomplete"):
        racket._publish_action_ball_continuous_racket_observation_for_test()
    assert racket._action_ball_continuous_racket_observation_records == {}

    # Construction tombstones are legitimate for rows that are not installed.
    racket._action_ball_task_valid[1] = False
    token = racket._publish_action_ball_continuous_racket_observation_for_test()
    view = racket.require_owned_action_ball_continuous_racket_observation(token)
    assert not bool(view.task_valid[1])


def test_production_projection_and_binder_hold_for_unfrozen_device_r05_abi():
    racket = _racket()
    with pytest.raises(
        HC.ActionBallContinuousRacketObservationHold,
        match="Device-R05 current identity ABI",
    ):
        racket.action_ball_continuous_racket_observation_projection()

    harness = _device_r05_harness(2)
    production = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    production.num_envs = 2
    production.device = "cpu"
    production._action_ball_enabled = False
    production._action_ball_full_mdp_enabled = True
    production._action_ball_continuous_fresh_racket_lane_bound = False
    with pytest.raises(
        HC.ActionBallContinuousRacketObservationHold,
        match="current-observation projection/validator ABI",
    ):
        production.bind_action_ball_full_mdp_racket_staging(harness.owner)
    assert harness.owner._genesis_child_projections == {}
    assert not hasattr(production, "_action_ball_reset_generation")
