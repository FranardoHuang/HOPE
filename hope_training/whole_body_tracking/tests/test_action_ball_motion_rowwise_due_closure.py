"""Focused Motion-owned row-wise due/closure projection tests.

These tests cover one mechanics-only Motion slice.  They do not install an
ActionEpoch consumer, replace the old D05 callpoint, or authorize training.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "source" / "whole_body_tracking"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

import action_ball_motion_cadence_device as cadence  # noqa: E402
import test_action_ball_continuous_motion_bridge as bridge  # noqa: E402
import test_action_ball_continuous_motion_selected_reset as reset_test  # noqa: E402
import test_action_ball_motion_genesis_cadence_activation as genesis  # noqa: E402


C = bridge.C


def _fresh_motion(device: torch.device):
    command, cadence_owner, device_owner, epoch_owner = (
        genesis._fresh_command_and_owners(device)
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    for common_step in range(3):
        command._env.common_step_counter = common_step
        command._advance_action_ball_continuous_motion_cadence()
    return command, cadence_owner


def _seed_current_task_rows(command) -> None:
    command._action_ball_continuous_motion_active.fill_(True)
    command._action_ball_continuous_current_policy_opportunity.fill_(True)
    command._action_ball_continuous_canonical_task_valid.fill_(True)
    command._action_ball_continuous_canonical_task_identity.copy_(
        torch.tensor([101, 102], dtype=torch.int64, device=command.device)
    )
    command._action_ball_continuous_canonical_cadence_identity.copy_(
        torch.tensor([201, 202], dtype=torch.int64, device=command.device)
    )
    command._action_ball_continuous_canonical_action_uid.copy_(
        torch.as_tensor(
            command._action_ball_action_uids,
            dtype=torch.int64,
            device=command.device,
        )[command.clip_id]
    )
    command._action_ball_task_timing_active.fill_(True)
    command._action_ball_pre_swing_wait_s.fill_(100.0)
    command._action_ball_scaled_t_cycle_s.fill_(1.0)
    command._action_ball_teacher_rate.fill_(1.0)
    command._action_ball_continuous_canonical_phase.fill_(
        C.ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE
    )


def test_motion_lifecycle_keeps_playback_publication_but_not_legacy_close() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            C.MotionCommand._advance_action_ball_continuous_canonical_lifecycle,
            C.MotionCommand.action_epoch_playback_transition_mask,
        )
    )
    assert "publish_motion_playback_started" in source
    assert "publish_motion_closed_unplayed" not in source
    assert "MOTION_CLOSED_UNPLAYED" not in source


@pytest.mark.parametrize("device", genesis._DEVICES)
def test_post_d05_publication_reseals_exact_current_row_projection(
    device: torch.device,
) -> None:
    command, owner = _fresh_motion(device)
    before = owner.project_current_action_epoch_rows()
    command._invalidate_action_ball_continuous_observation_publication()
    command._action_ball_continuous_current_projection = None

    command.publish_action_ball_full_mdp_post_d05_observation()

    after = owner.project_current_action_epoch_rows()
    assert after.common_step == before.common_step
    assert torch.equal(after.reveal_due, before.reveal_due)
    assert torch.equal(after.closed_mask, before.closed_mask)
    assert torch.equal(after.close_reason, before.close_reason)
    token = command.action_ball_continuous_motion_observation_projection()
    observation = (
        command.require_owned_action_ball_continuous_motion_observation(token)
    )
    assert observation.common_step == before.common_step


@pytest.mark.parametrize("device", genesis._DEVICES)
def test_rowwise_unplayed_close_keeps_inactive_peer_and_clears_next_tick(
    device: torch.device,
) -> None:
    command, owner = _fresh_motion(device)
    _seed_current_task_rows(command)
    command._action_ball_continuous_canonical_task_close_tick.copy_(
        torch.tensor([6, 10], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_sequence_active[1] = False
    command._action_ball_continuous_motion_active[1] = False
    command._action_ball_task_timing_active[1] = False
    peer_identity = (
        command._action_ball_continuous_canonical_task_identity[1].clone()
    )

    for common_step in range(3, 7):
        command._env.common_step_counter = common_step
        command._advance_action_ball_continuous_motion_cadence()

    projected = owner.project_current_action_epoch_rows()
    assert torch.equal(
        projected.closed_mask,
        torch.tensor([True, False], dtype=torch.bool, device=device),
    )
    assert torch.equal(
        projected.close_reason,
        torch.tensor(
            [
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_UNPLAYED,
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE,
            ],
            dtype=torch.int64,
            device=device,
        ),
    )
    assert command._action_ball_continuous_canonical_task_identity[0] == -1
    assert torch.equal(
        command._action_ball_continuous_canonical_task_identity[1],
        peer_identity,
    )

    command._env.common_step_counter = 7
    command._advance_action_ball_continuous_motion_cadence()
    next_projection = owner.project_current_action_epoch_rows()
    assert not torch.any(next_projection.closed_mask)
    assert torch.all(
        next_projection.close_reason.eq(
            C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE
        )
    )


@pytest.mark.parametrize("device", genesis._DEVICES)
def test_rowwise_played_suffix_close_is_not_broadcast_to_peer(
    device: torch.device,
) -> None:
    command, owner = _fresh_motion(device)
    _seed_current_task_rows(command)
    command._action_ball_continuous_canonical_playback_started.fill_(True)
    command._action_ball_continuous_canonical_task_close_tick.fill_(100)
    command._action_ball_pre_swing_wait_s.zero_()
    command._action_ball_task_age_s.copy_(
        torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
    )
    peer_identity = (
        command._action_ball_continuous_canonical_task_identity[1].clone()
    )

    command._env.common_step_counter = 3
    command._advance_action_ball_continuous_motion_cadence()
    projected = owner.project_current_action_epoch_rows()

    assert torch.equal(
        projected.closed_mask,
        torch.tensor([True, False], dtype=torch.bool, device=device),
    )
    assert torch.equal(
        projected.close_reason,
        torch.tensor(
            [
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_PLAYED_SUFFIX,
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE,
            ],
            dtype=torch.int64,
            device=device,
        ),
    )
    assert command._action_ball_continuous_canonical_task_identity[0] == -1
    assert torch.equal(
        command._action_ball_continuous_canonical_task_identity[1],
        peer_identity,
    )


def test_selected_reset_clears_only_selected_close_edge() -> None:
    command, _owner, r05_owner = reset_test._command()
    reset_test._seed_nontrivial_live_state(command)
    command._action_ball_continuous_closed_mask.copy_(
        torch.tensor([True, True, True], dtype=torch.bool)
    )
    command._action_ball_continuous_close_reason.copy_(
        torch.tensor(
            [
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_PLAYED_SUFFIX,
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_UNPLAYED,
                C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_PLAYED_SUFFIX,
            ],
            dtype=torch.int64,
        )
    )
    before_mask = command._action_ball_continuous_closed_mask.clone()
    before_reason = command._action_ball_continuous_close_reason.clone()
    reset_test._refresh_selection_generations(command, r05_owner)

    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    command.commit_prevalidated_selected_reset(armed)

    assert not command._action_ball_continuous_closed_mask[1]
    assert command._action_ball_continuous_close_reason[1] == (
        C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE
    )
    assert torch.equal(
        command._action_ball_continuous_closed_mask[[0, 2]],
        before_mask[[0, 2]],
    )
    assert torch.equal(
        command._action_ball_continuous_close_reason[[0, 2]],
        before_reason[[0, 2]],
    )


def _call_names(function) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def test_close_and_eager_seal_order_and_new_projection_path_have_no_hot_d2h(
) -> None:
    advance_source = textwrap.dedent(
        inspect.getsource(
            C.MotionCommand._advance_action_ball_continuous_motion_cadence
        )
    )
    advance_tree = ast.parse(advance_source)
    ordered_calls = sorted(
        (
            node.lineno,
            node.func.attr,
        )
        for node in ast.walk(advance_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    call_line = {name: line for line, name in ordered_calls}
    assert call_line["_write_action_ball_continuous_close_edge"] < call_line[
        "_advance_action_ball_continuous_canonical_lifecycle"
    ]
    assert call_line["_publish_action_ball_continuous_observation"] < call_line[
        "_seal_action_ball_continuous_current_projection"
    ]

    # Inspect the complete new eager-seal/current-projection call graph one
    # function at a time.  The old cadence advance reaches pre-existing async
    # assertions elsewhere and is intentionally not represented as clean.
    inspected = (
        C.MotionCommand._seal_action_ball_continuous_current_projection,
        C.MotionCommand._clone_action_ball_continuous_projection,
        C.MotionCommand._write_action_ball_continuous_close_edge,
        C.MotionCommand.action_ball_continuous_current_projection,
        C.MotionCommand._require_action_ball_continuous_current_publication,
        C.MotionCommand._require_action_ball_continuous_projection_current,
        C.MotionCommand._require_action_ball_continuous_motion_leaf_idle,
        C.MotionCommand._action_ball_continuous_motion_leaf_is_active,
        C.MotionCommand._action_ball_continuous_motion_selected_reset_is_active,
        C.MotionCommand._require_action_ball_continuous_parent_authorities,
        cadence.ActionBallMotionCadenceAuthority.project_current_action_epoch_rows,
    )
    banned = {
        "item",
        "cpu",
        "numpy",
        "tolist",
        "equal",
        "_assert_async",
    }
    for function in inspected:
        calls = _call_names(function)
        assert banned.isdisjoint(calls), function.__qualname__
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bool"
            for node in ast.walk(tree)
        ), function.__qualname__
