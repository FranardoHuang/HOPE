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
import types

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


def _install_frozen_task_frame_latch(command) -> tuple[torch.Tensor, torch.Tensor]:
    """Install the constructor-owned D05 task-frame latch for this harness.

    The focused Motion harness bypasses ``MotionCommand.__init__`` and the
    FullMDP construction path.  Use a fixed, explicit identity/zero frame:
    this keeps the fixture independent of the D05 writer that the tests
    exercise, while matching the production latch ABI.
    """

    yaw_wxyz = torch.zeros(
        command.num_envs,
        4,
        dtype=command.motion.body_quat_w.dtype,
        device=command.device,
    )
    yaw_wxyz[:, 0] = 1.0
    translation_w = torch.zeros(
        command.num_envs,
        3,
        dtype=command.motion.body_pos_w.dtype,
        device=command.device,
    )
    command._action_ball_full_mdp_task_yaw_wxyz = yaw_wxyz
    command._action_ball_full_mdp_task_translation_w = translation_w
    return yaw_wxyz, translation_w


def _refresh_revealed_reference(command, reveal: torch.Tensor) -> None:
    """Call the fresh-lane API with the fixture-owned frozen task frame."""

    command.refresh_action_ball_revealed_body_reference(
        reveal,
        task_yaw_wxyz=command._action_ball_full_mdp_task_yaw_wxyz,
        task_translation_w=command._action_ball_full_mdp_task_translation_w,
    )


def _fresh_motion(device: torch.device):
    command, cadence_owner, device_owner, epoch_owner = (
        genesis._fresh_command_and_owners(device)
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    _install_frozen_task_frame_latch(command)
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


def _bound_fresh_motion():
    command, _cadence_owner, device_owner, epoch_owner = (
        genesis._fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    _install_frozen_task_frame_latch(command)
    return command, epoch_owner


def _motion_row_snapshot(command, row: int) -> dict[str, torch.Tensor]:
    return {
        field: getattr(command, attr)[row].detach().clone()
        for field, attr, _nonnegative in (
            C._ACTION_BALL_CONTINUOUS_MOTION_CHECKPOINT_TENSORS
        )
        if field
        not in {
            "frozen_root_pos_w",
            "frozen_root_quat_wxyz",
            "frozen_root_valid",
        }
    }


def _assert_motion_row_unchanged(command, row: int, before) -> None:
    after = _motion_row_snapshot(command, row)
    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name]), name


def _advance_once(command) -> None:
    command._env.common_step_counter = 0
    command._advance_action_ball_continuous_motion_cadence()


def test_named_cadence_overdue_fault_freezes_only_bad_row() -> None:
    command, epoch_owner = _bound_fresh_motion()
    command._action_ball_continuous_episode_step[0] = (
        command._action_ball_continuous_next_reveal_step[0]
    )
    bad_before = _motion_row_snapshot(command, 0)
    peer_step_before = command._action_ball_continuous_episode_step[1].clone()

    _advance_once(command)

    assert epoch_owner._undrained_row_fault_bits.tolist() == [
        genesis.E.ROW_FAULT_MOTION_CADENCE_OVERDUE,
        0,
    ]
    _assert_motion_row_unchanged(command, 0, bad_before)
    assert command._action_ball_continuous_episode_step[1] == (
        peer_step_before + 1
    )


def test_cadence_overdue_is_not_compounded_with_task_timing_fault() -> None:
    command, epoch_owner = _bound_fresh_motion()
    command._action_ball_continuous_episode_step[0] = (
        command._action_ball_continuous_next_reveal_step[0]
    )
    command._action_ball_continuous_motion_active[0] = True
    command._action_ball_task_timing_active[0] = False
    bad_before = _motion_row_snapshot(command, 0)
    peer_step_before = command._action_ball_continuous_episode_step[1].clone()

    _advance_once(command)

    bits = epoch_owner._undrained_row_fault_bits
    assert bits.tolist() == [
        genesis.E.ROW_FAULT_MOTION_CADENCE_OVERDUE,
        0,
    ]
    assert not torch.any(
        bits.bitwise_and(genesis.E.ROW_FAULT_MOTION_TASK_TIMING_CONTRACT)
    )
    _assert_motion_row_unchanged(command, 0, bad_before)
    assert command._action_ball_continuous_episode_step[1] == (
        peer_step_before + 1
    )


def test_named_task_timing_fault_freezes_age_and_cadence_before_write() -> None:
    command, epoch_owner = _bound_fresh_motion()
    command._action_ball_continuous_motion_active[0] = True
    command._action_ball_task_timing_active[0] = False
    command._action_ball_task_age_s[0] = 7.0
    bad_before = _motion_row_snapshot(command, 0)
    peer_step_before = command._action_ball_continuous_episode_step[1].clone()

    _advance_once(command)

    assert epoch_owner._undrained_row_fault_bits.tolist() == [
        genesis.E.ROW_FAULT_MOTION_TASK_TIMING_CONTRACT,
        0,
    ]
    _assert_motion_row_unchanged(command, 0, bad_before)
    assert command._action_ball_continuous_episode_step[1] == (
        peer_step_before + 1
    )


def _configure_reveal_cache(command) -> None:
    command.action_ball_diagnostic_split_ready_teacher = True
    command.canonical_ready_mode = False
    command.clip_id.copy_(torch.tensor([0, 1], dtype=torch.int64))
    frame_zero = command.motion.seg_start[command.clip_id]
    command.time_steps.copy_(frame_zero)
    command._action_ball_safe_ready_reference_pending = torch.tensor(
        [True, False], dtype=torch.bool
    )
    command._action_ball_public_task_valid = torch.tensor(
        [True, True], dtype=torch.bool
    )
    command.motion.body_pos_w = torch.zeros_like(command.motion.body_pos_w)
    command.motion.body_pos_w[frame_zero[0], 0] = torch.tensor([1.0, 2.0, 3.0])
    command.motion.body_pos_w[frame_zero[1], 0] = torch.tensor([4.0, 5.0, 6.0])
    command.motion.body_quat_w = torch.zeros_like(command.motion.body_quat_w)
    command.motion.body_quat_w[:, 0, 0] = 1.0
    command.motion_anchor_body_index = 0
    command.robot_anchor_body_index = 0
    command.body_indexes = [0]
    command.robot.data.body_pos_w = torch.zeros((2, 1, 3))
    command.robot.data.body_quat_w = torch.zeros((2, 1, 4))
    command.robot.data.body_quat_w[:, 0, 0] = 1.0
    command.body_pos_relative_w = torch.full((2, 1, 3), -9.0)
    command.body_quat_relative_w = torch.full((2, 1, 4), -9.0)


def _latch_out_of_range_reveal_fault(command, epoch_owner) -> torch.Tensor:
    _configure_reveal_cache(command)
    command._action_ball_safe_ready_reference_pending.zero_()
    command._action_ball_safe_ready_pending_count = 0
    command._action_ball_continuous_policy_opportunities_created.fill_(1)
    command.time_steps[0] = int(command.motion.time_step_total) + 17
    command.time_steps_f[0] = command.time_steps[0].float()
    _refresh_revealed_reference(
        command, torch.tensor([True, False], dtype=torch.bool)
    )
    assert epoch_owner._undrained_row_fault_bits.tolist() == [
        genesis.E.ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
        0,
    ]
    return command.motion.seg_start[command.clip_id]


def test_named_reveal_reference_fault_masks_cache_and_keeps_peer() -> None:
    command, epoch_owner = _bound_fresh_motion()
    _configure_reveal_cache(command)
    bad_pos_before = command.body_pos_relative_w[0].clone()
    bad_quat_before = command.body_quat_relative_w[0].clone()
    peer_pos_before = command.body_pos_relative_w[1].clone()

    _refresh_revealed_reference(
        command, torch.tensor([True, True], dtype=torch.bool)
    )

    assert epoch_owner._undrained_row_fault_bits.tolist() == [
        genesis.E.ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
        0,
    ]
    assert torch.equal(command.body_pos_relative_w[0], bad_pos_before)
    assert torch.equal(command.body_quat_relative_w[0], bad_quat_before)
    assert not torch.equal(command.body_pos_relative_w[1], peer_pos_before)


def test_fresh_reveal_fault_without_epoch_owner_fails_closed_before_cache_write(
) -> None:
    command, _cadence_owner, device_owner, _epoch_owner = (
        genesis._fresh_command_and_owners(torch.device("cpu"))
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    _install_frozen_task_frame_latch(command)
    _configure_reveal_cache(command)
    before_pos = command.body_pos_relative_w.clone()
    before_quat = command.body_quat_relative_w.clone()

    with pytest.raises(
        RuntimeError,
        match="fresh Motion row fault requires its exact ActionEpoch owner",
    ):
        _refresh_revealed_reference(
            command, torch.tensor([True, False], dtype=torch.bool)
        )

    assert torch.equal(command.body_pos_relative_w, before_pos)
    assert torch.equal(command.body_quat_relative_w, before_quat)


def test_out_of_range_fault_quarantines_every_motion_reference_getter() -> None:
    command, epoch_owner = _bound_fresh_motion()
    frame_zero = _latch_out_of_range_reveal_fault(command, epoch_owner)
    bridge._add_velocity_reference_tensors(command)
    command.robot.data.default_joint_pos = torch.zeros_like(
        command.motion.joint_pos[: command.num_envs]
    )

    safe_steps = command._action_ball_full_mdp_safe_pose_reference_steps()
    assert safe_steps.tolist() == [frame_zero[0].item(), frame_zero[1].item()]
    references = {
        name: getattr(command, name)
        for name in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "anchor_pos_w",
            "anchor_quat_w",
            "anchor_lin_vel_w",
            "anchor_ang_vel_w",
        )
    }
    for name, value in references.items():
        assert value.shape[0] == command.num_envs, name
        assert torch.all(torch.isfinite(value)), name
    assert torch.equal(
        references["joint_pos"][0], command.motion.joint_pos[frame_zero[0]]
    )
    assert torch.equal(
        references["body_pos_w"][0, 0],
        command.motion.body_pos_w[frame_zero[0], 0]
        + command._env.scene.env_origins[0],
    )


def test_out_of_range_fault_survives_real_update_tail_without_cache_write(
) -> None:
    command, epoch_owner = _bound_fresh_motion()
    _latch_out_of_range_reveal_fault(command, epoch_owner)
    bad_pos_before = command.body_pos_relative_w[0].clone()
    bad_quat_before = command.body_quat_relative_w[0].clone()
    peer_pos_before = command.body_pos_relative_w[1].clone()
    peer_quat_before = command.body_quat_relative_w[1].clone()
    command._stagger_ep_pending = False
    command._event_scheduler = None
    command._multiseg = True
    command._advance_action_ball_continuous_motion_cadence = types.MethodType(
        lambda self: (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        ),
        command,
    )

    C.MotionCommand._update_command(command)

    assert epoch_owner._undrained_row_fault_bits.tolist() == [
        genesis.E.ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
        0,
    ]
    assert torch.equal(command.body_pos_relative_w[0], bad_pos_before)
    assert torch.equal(command.body_quat_relative_w[0], bad_quat_before)
    assert not torch.equal(command.body_pos_relative_w[1], peer_pos_before)
    assert not torch.equal(command.body_quat_relative_w[1], peer_quat_before)


def test_task_valid_false_alone_is_not_a_reveal_reference_fault() -> None:
    command, epoch_owner = _bound_fresh_motion()
    _configure_reveal_cache(command)
    command._action_ball_safe_ready_reference_pending.zero_()
    command._action_ball_public_task_valid[0] = False
    before = command.body_pos_relative_w.clone()

    _refresh_revealed_reference(
        command, torch.tensor([True, False], dtype=torch.bool)
    )

    assert epoch_owner._undrained_row_fault_bits.tolist() == [0, 0]
    assert not torch.equal(command.body_pos_relative_w[0], before[0])
    assert torch.equal(command.body_pos_relative_w[1], before[1])


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
    command._action_ball_continuous_canonical_task_valid[1] = False
    peer_identity = (
        command._action_ball_continuous_canonical_task_identity[1].clone()
    )

    for common_step in range(3, 7):
        command._env.common_step_counter = common_step
        command._advance_action_ball_continuous_motion_cadence()

    projected = owner.project_current_action_epoch_rows()
    assert command.action_ball_current_task_receipt_active.tolist() == [
        False,
        False,
    ]
    assert command.action_ball_task_timing_active.tolist() == [False, False]
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

    assert command.action_ball_current_task_receipt_active.tolist() == [False, True]
    assert command.action_ball_task_timing_active.tolist() == [False, True]
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
    command._action_ball_continuous_canonical_task_valid.fill_(True)
    command._action_ball_task_timing_active.fill_(True)
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
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    )

    assert not command._action_ball_continuous_closed_mask[1]
    assert command._action_ball_continuous_close_reason[1] == (
        C.ACTION_BALL_CONTINUOUS_MOTION_CLOSE_NONE
    )
    assert command.action_ball_current_task_receipt_active.tolist() == [
        True,
        False,
        True,
    ]
    assert command.action_ball_task_timing_active.tolist() == [True, False, True]
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

    # Inspect the complete cadence/eager-seal/current-projection call graph one
    # function at a time.  Fresh cadence faults must use the named packed Epoch
    # path; an anonymous CUDA assertion is not a write authorization boundary.
    inspected = (
        C.MotionCommand._advance_action_ball_continuous_motion_cadence,
        C.MotionCommand._latch_action_ball_full_mdp_motion_epoch_row_fault,
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
