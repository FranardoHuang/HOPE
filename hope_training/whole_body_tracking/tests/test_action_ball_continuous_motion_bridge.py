"""Fresh ActionBall Motion-side cadence/recovery bridge regression.

These tests exercise the real repository ``MotionCommand`` through the pinned
Pod1 IsaacLab test loader.  Motion owns only its task timing/playback leaf; it
never installs a Racket target or physical ball.  The current row-wise D05
writer is covered by ``test_action_ball_motion_rowwise_accept_writer.py``;
this file retains the independent scheduler and sealed-projection invariants.

Run only on Pod1 with CUDA hidden.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
from pathlib import Path
import sys
import types

import pytest
import torch


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import test_action_ball_motion_batch_handoff as batch_handoff
import test_action_ball_motion_birth as motion_birth
C = motion_birth.C
CONTINUOUS_CONTRACT_AUTHORITY_SHA256 = "7" * 64
RECOVERY_CONTRACT_AUTHORITY_SHA256 = "8" * 64


def _profile():
    payload = {
        "schema_version": 1,
        "kind": "whole_body_tracking.action_ball_continuous_motion_projection_v1",
        "clock_kind": "episode_tick_v1",
        "continuous_contract_authority_sha256": (
            CONTINUOUS_CONTRACT_AUTHORITY_SHA256
        ),
        "recovery_contract_authority_sha256": (
            RECOVERY_CONTRACT_AUTHORITY_SHA256
        ),
        "ready_reference_kind": (
            "completed_action_frame0_zero_velocity_v1"
        ),
    }
    return {
        **payload,
        "canonical_sha256": hashlib.sha256(
            C._canonical_json_bytes(payload)
        ).hexdigest(),
    }


def _schedule_projection(
    *, first_reveal_step=2, cadence_steps=5, deadline_offset_steps=2
):
    return {
        "frozen_at_step": 0,
        "sequence_origin_step": 0,
        "first_reveal_step": first_reveal_step,
        "cadence_steps": cadence_steps,
        "deadline_offset_steps": deadline_offset_steps,
    }


def _reseal_profile(value):
    payload = {
        name: field
        for name, field in value.items()
        if name != "canonical_sha256"
    }
    return {
        **payload,
        "canonical_sha256": hashlib.sha256(
            C._canonical_json_bytes(payload)
        ).hexdigest(),
    }


def _configure_unbound_command(*, num_envs=1, profile=None):
    (
        command,
        broker,
        authority,
        env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        elapsed_s,
    ) = batch_handoff._prepare_batch(
        num_envs=num_envs, swing_generation=0
    )
    command.resolve_action_ball_task_timing_now(env_ids)
    # Real ManagerBasedRLEnv always owns this integer manager-tick clock.
    # The lightweight Motion harness must expose it too, even in tests that
    # advance the episode cadence directly rather than through the helper.
    command._env.common_step_counter = 0
    command._continuous_test_context = types.SimpleNamespace(
        broker=broker,
        authority=authority,
        env_ids=env_ids,
        host_identity_rows=host_identity_rows,
        receipts=receipts,
        task_refs=task_refs,
        elapsed_s=elapsed_s,
        runtime=command._action_ball_runtime_module_bound,
    )
    command.canonical_ready_mode = True
    command.action_ball_single_stroke_timeout_enabled = False
    command.cfg.action_ball_continuous_motion_cadence = (
        _profile() if profile is None else profile
    )
    command.cfg.clip_switch_prob = 0.0
    command.cfg.stagger_initial_clock = False
    command.cfg.rsi_skip_settle_frames = 0
    command._event_timing_mode = C.EVENT_TIMING_MODE_DISABLED
    command._event_scheduler = None
    command.retiming_active = False
    command.planner_revision_enabled = False
    command.just_resampled = torch.zeros(num_envs, dtype=torch.bool)
    command._configure_action_ball_continuous_motion_cadence()
    return command, env_ids


def _task_refs(command):
    return command._continuous_test_context.task_refs


def _install_successor_task(command, *, env_id=0):
    """Publish and resolve the next exact task under the same true-reset birth."""

    context = command._continuous_test_context
    reset_generation = int(
        command._action_ball_reset_generation[env_id].item()
    )
    birth = context.broker._consumed_receipts[
        (env_id, reset_generation)
    ]
    swing_generation = int(
        command._action_ball_swing_generation[env_id].item()
    )
    receipt = motion_birth._task_receipt(
        context.runtime,
        birth,
        swing_generation=swing_generation,
    )
    context.authority.install((receipt,))
    env_ids = torch.tensor([env_id], dtype=torch.long)
    command._begin_action_ball_task_pending(
        env_ids,
        elapsed_s=float(command._env.step_dt),
    )
    command.resolve_action_ball_task_timing_now(env_ids)
    return receipt.task_ref()


def _parent_binding_kwargs(schedule):
    return {
        "continuous_contract_authority_sha256": (
            CONTINUOUS_CONTRACT_AUTHORITY_SHA256
        ),
        "recovery_contract_authority_sha256": (
            RECOVERY_CONTRACT_AUTHORITY_SHA256
        ),
        **schedule,
    }


def _continuous_command(*, num_envs=1, **schedule_kwargs):
    command, env_ids = _configure_unbound_command(num_envs=num_envs)
    schedule = _schedule_projection(**schedule_kwargs)
    command.bind_action_ball_continuous_parent_authorities(
        **_parent_binding_kwargs(schedule)
    )
    command._reset_action_ball_continuous_motion_cadence(env_ids)
    return command, env_ids


def _add_velocity_reference_tensors(command):
    frame_count = command.motion.joint_pos.shape[0]
    joint_count = command.motion.joint_pos.shape[1]
    body_count = command.motion.body_pos_w.shape[1]
    command.motion.joint_vel = torch.ones(frame_count, joint_count)
    command.motion.body_lin_vel_w = torch.ones(
        frame_count, body_count, 3
    )
    command.motion.body_ang_vel_w = torch.full(
        (frame_count, body_count, 3), 2.0
    )
    command.motion_anchor_body_index = 0


def _advance_at_common_step(command, common_step):
    command._env.common_step_counter = common_step
    return command._advance_action_ball_continuous_motion_cadence()


def _motion_transaction_snapshot(command):
    tensor_names = (
        "_action_ball_continuous_episode_step",
        "_action_ball_continuous_scheduled_ordinal",
        "_action_ball_continuous_current_reveal_step",
        "_action_ball_continuous_current_deadline_step",
        "_action_ball_continuous_next_reveal_step",
        "_action_ball_continuous_reveal_due",
        "_action_ball_continuous_task_commit_pending",
        "_action_ball_continuous_task_commit_missed",
        "_action_ball_continuous_task_committed",
        "_action_ball_continuous_motion_active",
        "_action_ball_continuous_suffix_complete",
        "_action_ball_continuous_ready_reference_active",
        "_action_ball_continuous_ready_at_reveal",
        "_action_ball_continuous_motion_release_pending",
        "_action_ball_continuous_motion_release_missed",
        "_action_ball_task_pending_elapsed_s",
        "_action_ball_task_age_s",
        "_action_ball_time_to_contact_s",
        "_action_ball_teacher_rate",
        "_action_ball_scaled_t_hit_s",
        "_action_ball_scaled_t_cycle_s",
        "_action_ball_pre_swing_wait_s",
        "_action_ball_task_timing_active",
    )
    return {
        "tensors": {
            name: getattr(command, name).clone() for name in tensor_names
        },
        "active_refs": tuple(command._action_ball_active_task_refs),
        "committed_refs": tuple(
            command._action_ball_continuous_committed_task_refs
        ),
        "prepared": command._action_ball_continuous_prepared_task_commit,
        "prepared_receipts": (
            command._action_ball_continuous_prepared_task_commit_receipts
        ),
        "next_serial": (
            command._action_ball_continuous_next_commit_token_serial
        ),
    }


def _assert_motion_transaction_snapshot(command, expected):
    actual = _motion_transaction_snapshot(command)
    assert actual.keys() == expected.keys()
    for name, value in expected["tensors"].items():
        assert torch.equal(actual["tensors"][name], value), name
    for name in (
        "active_refs",
        "committed_refs",
        "next_serial",
    ):
        assert actual[name] == expected[name]
    assert actual["prepared"] is expected["prepared"]
    assert actual["prepared_receipts"] is expected["prepared_receipts"]


def test_profile_is_sealed_and_default_is_literal_legacy_path():
    assert C.MotionCommandCfg.action_ball_continuous_motion_cadence is None
    value = _profile()
    assert C._parse_action_ball_continuous_motion_profile(value) == value

    mutated = deepcopy(value)
    mutated["recovery_contract_authority_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="canonical SHA-256 differs"):
        C._parse_action_ball_continuous_motion_profile(mutated)

    command, _env_ids = _configure_unbound_command()
    with pytest.raises(ValueError, match="smaller than cadence"):
        command.bind_action_ball_continuous_parent_authorities(
            **_parent_binding_kwargs(
                _schedule_projection(
                    cadence_steps=2,
                    deadline_offset_steps=2,
                )
            )
        )


def test_profile_cannot_reseal_or_swap_away_from_external_c01_c02_authority():
    original = _profile()

    changed_timing = deepcopy(original)
    changed_timing["first_reveal_step"] = 3
    changed_timing = _reseal_profile(changed_timing)
    with pytest.raises(ValueError, match="keys differ"):
        _configure_unbound_command(profile=changed_timing)

    command, env_ids = _configure_unbound_command(profile=original)
    with pytest.raises(RuntimeError, match="requires external C01/C02"):
        command._reset_action_ball_continuous_motion_cadence(env_ids)

    changed_authority = deepcopy(original)
    changed_authority["recovery_contract_authority_sha256"] = "9" * 64
    changed_authority = _reseal_profile(changed_authority)
    command, _env_ids = _configure_unbound_command(profile=changed_authority)
    with pytest.raises(RuntimeError, match="differs from external C01/C02"):
        command.bind_action_ball_continuous_parent_authorities(
            **_parent_binding_kwargs(_schedule_projection())
        )

    command, _env_ids = _continuous_command()
    with pytest.raises(TypeError):
        command._action_ball_continuous_motion_profile[
            "recovery_contract_authority_sha256"
        ] = "9" * 64
    with pytest.raises(TypeError):
        command._action_ball_continuous_schedule_projection[
            "cadence_steps"
        ] = 6

    drifted = _parent_binding_kwargs(_schedule_projection())
    drifted["continuous_contract_authority_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="differs from external C01/C02"):
        command.bind_action_ball_continuous_parent_authorities(**drifted)

    drifted = _parent_binding_kwargs(
        _schedule_projection(first_reveal_step=3)
    )
    with pytest.raises(RuntimeError, match="may not drift"):
        command.bind_action_ball_continuous_parent_authorities(**drifted)


def test_fresh_profile_rejects_single_shot_wrap_and_second_clock():
    command, *_ = batch_handoff._prepare_batch(
        num_envs=1,
        swing_generation=0,
    )
    command.canonical_ready_mode = True
    command.action_ball_single_stroke_timeout_enabled = True
    command.cfg.action_ball_continuous_motion_cadence = _profile()
    command.cfg.clip_switch_prob = 0.0
    command.cfg.stagger_initial_clock = False
    command.cfg.rsi_skip_settle_frames = 0
    command.cfg.wrap_teleport = True
    command._event_timing_mode = C.EVENT_TIMING_MODE_POST_STRIKE_T1
    command.retiming_active = False
    command.planner_revision_enabled = False

    with pytest.raises(ValueError, match="conflicts") as error:
        command._configure_action_ball_continuous_motion_cadence()
    text = str(error.value)
    assert "single_stroke_timeout" in text
    assert "wrap_teleport" in text
    assert "event_timing_mode" in text


def test_current_projection_is_order_locked_current_only_and_non_aliasing():
    command, _env_ids = _continuous_command(
        num_envs=2,
        first_reveal_step=1,
    )
    command._env.common_step_counter = 0
    with pytest.raises(RuntimeError, match="stale|swapped"):
        command.action_ball_continuous_current_projection()

    _advance_at_common_step(command, 0)
    projection = command.action_ball_continuous_current_projection()
    expected_fields = {
        "common_step",
        "episode_tick",
        "reveal_due",
        "deadline_due",
        "scheduled_ordinal",
        "reveal_tick",
        "deadline_tick",
        "next_reveal_tick",
        "ready_at_reveal",
        "motion_active",
        "ready_reference_active",
        "suffix_complete",
        "closed_mask",
        "close_reason",
        "reset_generation",
        "swing_generation",
    }
    assert {field.name for field in fields(projection)} == expected_fields
    assert all(
        forbidden not in field_name
        for field_name in expected_fields
        for forbidden in ("future", "target", "ball", "task")
    )
    live = {
        "episode_tick": command._action_ball_continuous_episode_step,
        "reveal_due": command._action_ball_continuous_reveal_due,
        "deadline_due": command._action_ball_continuous_deadline_due,
        "scheduled_ordinal": (
            command._action_ball_continuous_scheduled_ordinal
        ),
        "reveal_tick": (
            command._action_ball_continuous_current_reveal_step
        ),
        "deadline_tick": (
            command._action_ball_continuous_current_deadline_step
        ),
        "next_reveal_tick": (
            command._action_ball_continuous_next_reveal_step
        ),
        "ready_at_reveal": (
            command._action_ball_continuous_ready_at_reveal
        ),
        "motion_active": command._action_ball_continuous_motion_active,
        "ready_reference_active": (
            command._action_ball_continuous_ready_reference_active
        ),
        "suffix_complete": (
            command._action_ball_continuous_suffix_complete
        ),
        "closed_mask": command._action_ball_continuous_closed_mask,
        "close_reason": command._action_ball_continuous_close_reason,
        "reset_generation": command._action_ball_reset_generation,
        "swing_generation": command._action_ball_swing_generation,
    }
    for name, owner_tensor in live.items():
        projected = getattr(projection, name)
        assert torch.equal(projected, owner_tensor), name
        assert projected.data_ptr() != owner_tensor.data_ptr(), name

    owner_before = command._action_ball_continuous_episode_step.clone()
    projection.episode_tick.add_(100)
    assert torch.equal(
        command._action_ball_continuous_episode_step,
        owner_before,
    )
    with pytest.raises(FrozenInstanceError):
        projection.common_step = 9
    repeated = command.action_ball_continuous_current_projection()
    assert repeated.episode_tick.tolist() != projection.episode_tick.tolist()
    assert repeated.episode_tick.data_ptr() != projection.episode_tick.data_ptr()

    # A later manager tick before Motion executes cannot reuse the snapshot.
    command._env.common_step_counter = 1
    with pytest.raises(RuntimeError, match="stale|swapped"):
        command.action_ball_continuous_current_projection()


def test_two_phase_commit_prevalidates_full_batch_without_live_racket_read():
    command, _env_ids = _continuous_command(
        num_envs=2,
        first_reveal_step=1,
    )
    command.bind_action_ball_continuous_ready_authority(
        torch.tensor([False, False])
    )
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    assert command.action_ball_continuous_reveal_due.tolist() == [True, True]

    context = command._continuous_test_context
    task_refs = context.task_refs
    receipts = context.receipts
    state_before = _motion_transaction_snapshot(command)
    with pytest.raises(RuntimeError, match="complete.*ordered"):
        command.commit_action_ball_continuous_task(
            [0], [0], task_refs[:1]
        )
    _assert_motion_transaction_snapshot(command, state_before)

    with pytest.raises(RuntimeError, match="complete.*ordered"):
        command.prepare_action_ball_continuous_task_commit(
            [0], [0], task_refs[:1], receipts[:1]
        )
    _assert_motion_transaction_snapshot(command, state_before)

    with pytest.raises(RuntimeError, match="scheduled env/ordinal"):
        command.prepare_action_ball_continuous_task_commit(
            [0, 1],
            [0, 0],
            tuple(reversed(task_refs)),
            tuple(reversed(receipts)),
        )
    _assert_motion_transaction_snapshot(command, state_before)

    invalid_second_receipt = deepcopy(receipts[1])
    object.__setattr__(
        invalid_second_receipt,
        "sample_sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="changed the requested immutable ref"):
        command.prepare_action_ball_continuous_task_commit(
            [0, 1],
            [0, 0],
            task_refs,
            (receipts[0], invalid_second_receipt),
        )
    # Row 0 validated before row 1 failed; neither row was published.
    _assert_motion_transaction_snapshot(command, state_before)

    with pytest.raises(RuntimeError, match="complete.*ordered"):
        command.prepare_action_ball_continuous_task_commit(
            [1, 0],
            [0, 0],
            tuple(reversed(task_refs)),
            tuple(reversed(receipts)),
        )
    _assert_motion_transaction_snapshot(command, state_before)

    authority_calls_before = (
        context.authority.ref_calls,
        context.authority.resolve_calls,
    )
    token = command.prepare_action_ball_continuous_task_commit(
        [0, 1], [0, 0], task_refs, receipts
    )
    assert (
        context.authority.ref_calls,
        context.authority.resolve_calls,
    ) == authority_calls_before
    assert command._action_ball_continuous_task_commit_pending.tolist() == [
        True,
        True,
    ]
    assert command._action_ball_continuous_task_committed.tolist() == [
        False,
        False,
    ]
    with pytest.raises(FrozenInstanceError):
        token.common_step = 2

    forged = replace(token, serial=token.serial + 1)
    with pytest.raises(RuntimeError, match="forged|stale|consumed"):
        command.commit_prepared_action_ball_continuous_task(forged)
    assert command._action_ball_continuous_task_commit_pending.tolist() == [
        True,
        True,
    ]

    command.commit_prepared_action_ball_continuous_task(token)
    assert (
        context.authority.ref_calls,
        context.authority.resolve_calls,
    ) == authority_calls_before
    assert command._action_ball_continuous_task_commit_pending.tolist() == [
        False,
        False,
    ]
    assert command._action_ball_continuous_task_committed.tolist() == [
        True,
        True,
    ]
    assert command._action_ball_task_timing_active.tolist() == [True, True]
    assert command._action_ball_active_task_refs == list(task_refs)
    # Not-ready controls playback only; it never cancels the new task.
    assert command._action_ball_continuous_motion_active.tolist() == [
        False,
        False,
    ]
    assert command._action_ball_continuous_ready_reference_active.tolist() == [
        True,
        True,
    ]
    with pytest.raises(RuntimeError, match="stale|consumed"):
        command.commit_prepared_action_ball_continuous_task(token)


def test_prepared_token_stales_on_next_tick_without_partial_publication():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    context = command._continuous_test_context
    token = command.prepare_action_ball_continuous_task_commit(
        [0], [0], context.task_refs, context.receipts
    )
    next_reveal = command._action_ball_continuous_next_reveal_step.clone()

    _advance_at_common_step(command, 2)
    with pytest.raises(RuntimeError, match="stale|consumed"):
        command.commit_prepared_action_ball_continuous_task(token)
    assert command._action_ball_continuous_task_committed.tolist() == [False]
    assert command._action_ball_continuous_task_commit_missed.tolist() == [True]
    assert torch.equal(
        command._action_ball_continuous_next_reveal_step,
        next_reveal,
    )


@pytest.mark.parametrize(
    "drift",
    ("clip_id", "ready_at_reveal", "motion_release_pending"),
)
def test_prepared_token_rejects_owner_tensor_drift_before_any_commit(drift):
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    context = command._continuous_test_context
    token = command.prepare_action_ball_continuous_task_commit(
        [0], [0], context.task_refs, context.receipts
    )

    if drift == "clip_id":
        command.clip_id.add_(1)
    elif drift == "ready_at_reveal":
        command._action_ball_continuous_ready_at_reveal.logical_not_()
    else:
        command._action_ball_continuous_motion_release_pending.logical_not_()
    with pytest.raises(RuntimeError, match="owner state drifted"):
        command.commit_prepared_action_ball_continuous_task(token)
    assert command._action_ball_continuous_task_commit_pending.tolist() == [True]
    assert command._action_ball_continuous_task_committed.tolist() == [False]


def test_ready_suffix_failure_is_rejected_during_prepare_without_writes():
    command, _env_ids = _continuous_command(
        first_reveal_step=1,
        cadence_steps=80,
        deadline_offset_steps=2,
    )
    command.bind_action_ball_continuous_ready_authority(torch.tensor([True]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    context = command._continuous_test_context
    reveal_receipt = command.action_ball_continuous_current_projection()
    state_before = _motion_transaction_snapshot(command)

    with pytest.raises(RuntimeError, match="cannot complete before"):
        command.prepare_action_ball_continuous_task_commit(
            [0], [0], context.task_refs, context.receipts
        )
    _assert_motion_transaction_snapshot(command, state_before)
    command.acknowledge_action_ball_continuous_infrastructure_invalid(
        [0], [0], reveal_receipt
    )
    assert command._action_ball_continuous_task_commit_missed.tolist() == [True]


def test_release_rejects_swapped_manager_order_without_writes():
    command, _env_ids = _continuous_command(
        first_reveal_step=1,
        cadence_steps=81,
        deadline_offset_steps=2,
    )
    command.bind_action_ball_continuous_ready_authority(torch.tensor([True]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    context = command._continuous_test_context
    token = command.prepare_action_ball_continuous_task_commit(
        [0], [0], context.task_refs, context.receipts
    )
    command.commit_prepared_action_ball_continuous_task(token)
    state_before = _motion_transaction_snapshot(command)

    command._env.common_step_counter = 2
    with pytest.raises(RuntimeError, match="stale|swapped"):
        command.release_action_ball_continuous_motion_playback([0], [0])
    _assert_motion_transaction_snapshot(command, state_before)


def test_legacy_commit_rejects_swapped_manager_order_without_writes():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    state_before = _motion_transaction_snapshot(command)

    command._env.common_step_counter = 2
    with pytest.raises(RuntimeError, match="stale|swapped"):
        command.commit_action_ball_continuous_task(
            [0], [0], _task_refs(command)
        )
    _assert_motion_transaction_snapshot(command, state_before)


def test_legacy_ack_commits_mixed_batch_and_releases_only_ready_rows():
    command, _env_ids = _continuous_command(
        num_envs=2,
        first_reveal_step=1,
        cadence_steps=81,
        deadline_offset_steps=2,
    )
    command.bind_action_ball_continuous_ready_authority(
        torch.tensor([True, False])
    )
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    context = command._continuous_test_context

    command.acknowledge_action_ball_continuous_motion_task(
        [0, 1], [0, 0], context.task_refs
    )

    assert command._action_ball_continuous_task_committed.tolist() == [
        True,
        True,
    ]
    assert command._action_ball_continuous_motion_active.tolist() == [
        True,
        False,
    ]
    assert command._action_ball_continuous_motion_release_pending.tolist() == [
        False,
        False,
    ]
    assert command._action_ball_continuous_ready_reference_active.tolist() == [
        False,
        True,
    ]


def test_explicit_infrastructure_invalid_ack_consumes_token_not_cadence():
    command, _env_ids = _continuous_command(
        first_reveal_step=1,
        cadence_steps=5,
        deadline_offset_steps=2,
    )
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    context = command._continuous_test_context
    token = command.prepare_action_ball_continuous_task_commit(
        [0], [0], context.task_refs, context.receipts
    )
    next_reveal = command._action_ball_continuous_next_reveal_step.clone()
    root_before = command.robot.data.root_state_w.clone()
    joint_before = command.robot.data.joint_pos.clone()

    command.acknowledge_action_ball_continuous_infrastructure_invalid(
        [0], [0], token
    )
    assert command._action_ball_continuous_task_commit_pending.tolist() == [
        False
    ]
    assert command._action_ball_continuous_task_commit_missed.tolist() == [True]
    assert command._action_ball_continuous_motion_release_pending.tolist() == [
        False
    ]
    infrastructure_invalid_code = C.ACTION_BALL_CONTINUOUS_MOTION_PHASES.index(
        "infrastructure_invalid"
    )
    assert command.action_ball_continuous_motion_phase.tolist() == [
        infrastructure_invalid_code
    ]
    with pytest.raises(RuntimeError, match="stale|consumed"):
        command.commit_prepared_action_ball_continuous_task(token)
    assert torch.equal(
        command._action_ball_continuous_next_reveal_step,
        next_reveal,
    )
    assert torch.equal(command.robot.data.root_state_w, root_before)
    assert torch.equal(command.robot.data.joint_pos, joint_before)

    reveal_steps = []
    deadline_steps = []
    for step in range(2, 7):
        _advance_at_common_step(command, step)
        if command.action_ball_continuous_reveal_due.item():
            reveal_steps.append(step)
        if command.action_ball_continuous_deadline_due.item():
            deadline_steps.append(step)
    assert deadline_steps == [3]
    assert reveal_steps == [6]
    assert command._action_ball_continuous_scheduled_ordinal.tolist() == [1]
    assert command._action_ball_swing_generation.tolist() == [1]
    assert command._action_ball_continuous_next_reveal_step.tolist() == [11]


@pytest.mark.parametrize("receipt_kind", ("projection", "token"))
def test_infrastructure_ack_rejects_receipt_from_prior_true_reset(
    receipt_kind,
):
    command, env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    projection = command.action_ball_continuous_current_projection()
    if receipt_kind == "token":
        context = command._continuous_test_context
        old_receipt = command.prepare_action_ball_continuous_task_commit(
            [0], [0], context.task_refs, context.receipts
        )
    else:
        old_receipt = projection

    # Model only the identity changes the true-reset owner performs before
    # Motion resets its episode-local cadence.  The new episode deliberately
    # reuses scheduled ordinal Q0.
    command._action_ball_reset_generation.add_(1)
    command._action_ball_swing_generation.zero_()
    command._reset_action_ball_continuous_motion_cadence(env_ids)
    _advance_at_common_step(command, 2)
    _advance_at_common_step(command, 3)
    assert command._action_ball_continuous_scheduled_ordinal.tolist() == [0]
    state_before = _motion_transaction_snapshot(command)

    with pytest.raises(RuntimeError, match="forged|stale|consumed"):
        command.acknowledge_action_ball_continuous_infrastructure_invalid(
            [0], [0], old_receipt
        )
    _assert_motion_transaction_snapshot(command, state_before)


def test_projection_ack_rejects_swapped_manager_order_without_writes():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))
    _advance_at_common_step(command, 0)
    _advance_at_common_step(command, 1)
    projection = command.action_ball_continuous_current_projection()
    state_before = _motion_transaction_snapshot(command)

    # The manager tick advanced, but Motion has not yet published for it.
    command._env.common_step_counter = 2
    with pytest.raises(RuntimeError, match="stale"):
        command.acknowledge_action_ball_continuous_infrastructure_invalid(
            [0], [0], projection
        )
    _assert_motion_transaction_snapshot(command, state_before)


def test_absolute_cadence_advances_after_miss_and_not_ready_reveal():
    command, _env_ids = _continuous_command()
    ready = torch.tensor([True])
    command.bind_action_ball_continuous_ready_authority(ready)

    reveal_steps = []
    deadline_steps = []
    commit_missed_steps = []
    infrastructure_invalid_steps = []
    infrastructure_invalid_code = C.ACTION_BALL_CONTINUOUS_MOTION_PHASES.index(
        "infrastructure_invalid"
    )
    for expected_step in range(10):
        command._advance_action_ball_continuous_motion_cadence()
        assert command._action_ball_continuous_episode_step.item() == expected_step
        if command.action_ball_continuous_reveal_due.item():
            reveal_steps.append(expected_step)
        if command.action_ball_continuous_deadline_due.item():
            deadline_steps.append(expected_step)
        if command._action_ball_continuous_task_commit_missed.item():
            commit_missed_steps.append(expected_step)
        if (
            command.action_ball_continuous_motion_phase.item()
            == infrastructure_invalid_code
        ):
            infrastructure_invalid_steps.append(expected_step)
        # Q0 is deliberately not acknowledged.  That is a failed/missing
        # opportunity, not permission to slide Q1's frozen reveal.
        if expected_step == 4:
            ready.fill_(False)

    assert reveal_steps == [2, 7]
    assert deadline_steps == [4, 9]
    assert commit_missed_steps == [3, 4, 5, 6, 8, 9]
    assert infrastructure_invalid_steps == [3, 4, 8, 9]
    assert command._action_ball_continuous_opportunities_consumed.tolist() == [2]
    assert command._action_ball_continuous_scheduled_ordinal.tolist() == [1]
    assert command._action_ball_swing_generation.tolist() == [1]
    assert command._action_ball_continuous_next_reveal_step.tolist() == [12]
    assert command._action_ball_continuous_last_closed_ordinal.tolist() == [1]
    assert command._action_ball_continuous_ready_at_reveal.tolist() == [False]
    assert command._action_ball_continuous_recovery_unavailable.tolist() == [False]
    assert command._action_ball_continuous_task_commit_missed.tolist() == [True]


def test_not_ready_is_sampled_only_at_scheduled_reveal_and_never_retroactive():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    ready = torch.tensor([False])
    command.bind_action_ball_continuous_ready_authority(ready)

    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # Q0 reveal, step 1
    assert command.action_ball_continuous_reveal_due.tolist() == [True]
    assert command.action_ball_continuous_recovery_unavailable.tolist() == [True]
    assert command._action_ball_continuous_ready_at_reveal.tolist() == [False]
    unavailable_code = C.ACTION_BALL_CONTINUOUS_MOTION_PHASES.index(
        "recovery_unavailable"
    )
    assert command.action_ball_continuous_motion_phase.tolist() == [
        unavailable_code
    ]
    command.commit_action_ball_continuous_task(
        [0], [0], _task_refs(command)
    )
    assert command._action_ball_continuous_task_committed.tolist() == [True]
    assert command._action_ball_continuous_ready_reference_active.tolist() == [True]
    assert command._action_ball_continuous_motion_active.tolist() == [False]
    with pytest.raises(RuntimeError, match="not a ready committed reveal"):
        command.release_action_ball_continuous_motion_playback([0], [0])

    ready.fill_(True)
    command._advance_action_ball_continuous_motion_cadence()
    assert command._action_ball_continuous_ready_at_reveal.tolist() == [False]
    with pytest.raises(RuntimeError, match="not a ready committed reveal"):
        command.release_action_ball_continuous_motion_playback([0], [0])


def test_not_ready_commits_each_new_task_but_never_releases_playback():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))

    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # Q0 reveal
    q0_ref = _task_refs(command)[0]
    command.commit_action_ball_continuous_task([0], [0], (q0_ref,))
    assert command._action_ball_active_task_refs == [q0_ref]
    assert command._action_ball_task_timing_active.tolist() == [True]
    assert command._action_ball_continuous_motion_active.tolist() == [False]

    command._advance_action_ball_continuous_motion_cadence()  # step 2
    command._advance_action_ball_continuous_motion_cadence()  # Q0 deadline
    assert command._action_ball_continuous_opportunities_consumed.tolist() == [1]
    assert command._action_ball_task_timing_active.tolist() == [False]
    command._advance_action_ball_continuous_motion_cadence()  # step 4
    command._advance_action_ball_continuous_motion_cadence()  # step 5
    command._advance_action_ball_continuous_motion_cadence()  # Q1 reveal

    q1_ref = _install_successor_task(command)
    assert q1_ref != q0_ref
    assert q1_ref.swing_generation == 1
    command.commit_action_ball_continuous_task([0], [1], (q1_ref,))
    assert command._action_ball_active_task_refs == [q1_ref]
    assert command._action_ball_task_timing_active.tolist() == [True]
    assert command._action_ball_continuous_task_commit_missed.tolist() == [False]
    assert command._action_ball_continuous_motion_active.tolist() == [False]
    assert command._action_ball_continuous_ready_reference_active.tolist() == [True]

    command._advance_action_ball_continuous_motion_cadence()  # step 7
    command._advance_action_ball_continuous_motion_cadence()  # Q1 deadline
    assert command._action_ball_continuous_opportunities_consumed.tolist() == [2]
    assert command._action_ball_continuous_last_closed_ordinal.tolist() == [1]
    assert command._action_ball_task_timing_active.tolist() == [False]


def test_successor_reveal_rejects_reused_task_identity_and_timing():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([False]))

    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # Q0 reveal
    q0_ref = _task_refs(command)[0]
    command.commit_action_ball_continuous_task([0], [0], (q0_ref,))
    for _ in range(5):
        command._advance_action_ball_continuous_motion_cadence()
    assert command.action_ball_continuous_reveal_due.tolist() == [True]
    assert command._action_ball_continuous_scheduled_ordinal.tolist() == [1]
    assert command._action_ball_active_task_refs == [q0_ref]
    assert command._action_ball_task_timing_active.tolist() == [False]

    with pytest.raises(RuntimeError, match="differs from scheduled env/ordinal"):
        command.commit_action_ball_continuous_task([0], [1], (q0_ref,))
    assert command._action_ball_continuous_task_commit_pending.tolist() == [True]
    assert command._action_ball_continuous_task_committed.tolist() == [False]
    command._advance_action_ball_continuous_motion_cadence()
    assert command._action_ball_continuous_task_commit_missed.tolist() == [True]
    infrastructure_invalid_code = C.ACTION_BALL_CONTINUOUS_MOTION_PHASES.index(
        "infrastructure_invalid"
    )
    assert command.action_ball_continuous_motion_phase.tolist() == [
        infrastructure_invalid_code
    ]


def test_ready_commit_without_playback_release_is_infrastructure_invalid():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    command.bind_action_ball_continuous_ready_authority(torch.tensor([True]))

    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # Q0 reveal
    command.commit_action_ball_continuous_task(
        [0], [0], _task_refs(command)
    )
    assert command._action_ball_continuous_task_commit_pending.tolist() == [False]
    assert command._action_ball_continuous_motion_release_pending.tolist() == [True]
    command._advance_action_ball_continuous_motion_cadence()

    assert command._action_ball_continuous_task_commit_missed.tolist() == [False]
    assert command._action_ball_continuous_motion_release_missed.tolist() == [True]
    infrastructure_invalid_code = C.ACTION_BALL_CONTINUOUS_MOTION_PHASES.index(
        "infrastructure_invalid"
    )
    assert command.action_ball_continuous_motion_phase.tolist() == [
        infrastructure_invalid_code
    ]


def test_release_rejects_suffix_that_would_finish_on_next_reveal_tick():
    command, _env_ids = _continuous_command(
        first_reveal_step=1,
        cadence_steps=80,
        deadline_offset_steps=2,
    )
    command.bind_action_ball_continuous_ready_authority(torch.tensor([True]))
    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # Q0 reveal
    command.commit_action_ball_continuous_task(
        [0], [0], _task_refs(command)
    )

    gap_steps = (
        command._action_ball_continuous_next_reveal_step.item()
        - command._action_ball_continuous_episode_step.item()
    )
    task_age_s = command._action_ball_task_age_s.item()
    cycle_total_s = (
        command._action_ball_pre_swing_wait_s.item()
        + command._action_ball_scaled_t_cycle_s.item()
    )
    step_dt = float(command._env.step_dt)
    assert task_age_s + (gap_steps - 2) * step_dt < cycle_total_s
    assert task_age_s + (gap_steps - 1) * step_dt == pytest.approx(
        cycle_total_s
    )
    with pytest.raises(RuntimeError, match="cannot complete before"):
        command.release_action_ball_continuous_motion_playback([0], [0])
    assert command._action_ball_continuous_task_committed.tolist() == [True]
    assert command._action_ball_continuous_motion_active.tolist() == [False]

    for _ in range(gap_steps):
        command._advance_action_ball_continuous_motion_cadence()
    assert command.action_ball_continuous_reveal_due.tolist() == [True]
    assert command._action_ball_continuous_scheduled_ordinal.tolist() == [1]
    q1_ref = _install_successor_task(command)
    command.commit_action_ball_continuous_task([0], [1], (q1_ref,))
    assert command._action_ball_continuous_task_committed.tolist() == [True]
    assert command._action_ball_active_task_refs == [q1_ref]


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("active_ref", "not owned by the committed task ref"),
        ("live_ref", "differs from live task authority"),
        ("teacher_rate", "stale teacher_rate"),
    ),
)
def test_release_revalidates_committed_ref_authority_and_timing(drift, message):
    command, _env_ids = _continuous_command(
        first_reveal_step=1,
        cadence_steps=81,
        deadline_offset_steps=2,
    )
    command.bind_action_ball_continuous_ready_authority(torch.tensor([True]))
    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # Q0 reveal
    command.commit_action_ball_continuous_task(
        [0], [0], _task_refs(command)
    )

    if drift == "active_ref":
        command._action_ball_active_task_refs[0] = None
    elif drift == "live_ref":
        context = command._continuous_test_context
        birth = context.broker._consumed_receipts[(0, 1)]
        context.authority.install(
            (
                motion_birth._task_receipt(
                    context.runtime,
                    birth,
                    swing_generation=1,
                ),
            )
        )
    else:
        command._action_ball_teacher_rate.add_(0.125)

    with pytest.raises(RuntimeError, match=message):
        command.release_action_ball_continuous_motion_playback([0], [0])
    assert command._action_ball_continuous_motion_active.tolist() == [False]
    assert command._action_ball_continuous_motion_release_pending.tolist() == [True]


def test_cadence_has_no_four_shot_or_ready_driven_auto_stop():
    command, _env_ids = _continuous_command(first_reveal_step=1)
    reveals = []
    deadlines = []
    # No ready authority is bound: every row is unavailable, but the Motion
    # clock is an unbounded cadence until the real sequence reset/horizon owner
    # acts.  The four-shot pure tape is an acceptance minimum, not a stop rule.
    for expected_step in range(29):
        command._advance_action_ball_continuous_motion_cadence()
        if command.action_ball_continuous_reveal_due.item():
            reveals.append(expected_step)
        if command.action_ball_continuous_deadline_due.item():
            deadlines.append(expected_step)

    assert reveals == [1, 6, 11, 16, 21, 26]
    assert deadlines == [3, 8, 13, 18, 23, 28]
    assert command._action_ball_continuous_scheduled_ordinal.tolist() == [5]
    assert command._action_ball_continuous_opportunities_consumed.tolist() == [6]
    assert command._action_ball_swing_generation.tolist() == [5]
    assert command._action_ball_continuous_sequence_active.tolist() == [True]


def test_complete_suffix_enters_frame0_zero_velocity_reference_without_plant_write():
    command, _env_ids = _continuous_command(
        first_reveal_step=1,
        cadence_steps=81,
        deadline_offset_steps=2,
    )
    _add_velocity_reference_tensors(command)
    # Exercise the measured split-ready branch: before Q0 it may use the
    # physical birth tuple, but post-suffix recovery must use completed-action
    # frame 0 even while the future task remains hidden.
    command.action_ball_diagnostic_split_ready_teacher = True
    command._action_ball_public_task_valid = torch.tensor([False])
    command._action_ball_dynamic_ready_physical_joint_pos_rad = torch.full(
        (command.motion.num_segments, command.motion.joint_pos.shape[1]),
        99.0,
    )
    command._action_ball_safe_ready_body_pos_w = torch.full_like(
        command.motion.body_pos_w[:1], 99.0
    )
    command._action_ball_safe_ready_body_quat_w = torch.zeros_like(
        command.motion.body_quat_w[:1]
    )
    command._action_ball_safe_ready_body_quat_w[..., 0] = 1.0
    ready = torch.tensor([True])
    command.bind_action_ball_continuous_ready_authority(ready)

    def unexpected_pending_scan(self):
        raise AssertionError("continuous active ticks must not scan pending ids")

    command._resolve_pending_action_ball_tasks = types.MethodType(
        unexpected_pending_scan,
        command,
    )

    root_before = command.robot.data.root_state_w.clone()
    joint_before = command.robot.data.joint_pos.clone()
    command._advance_action_ball_continuous_motion_cadence()  # step 0
    command._advance_action_ball_continuous_motion_cadence()  # reveal 1
    command.acknowledge_action_ball_continuous_motion_task(
        [0], [0], _task_refs(command)
    )
    assert command._action_ball_continuous_motion_active.tolist() == [True]
    assert command._action_ball_continuous_task_committed.tolist() == [True]
    assert command._action_ball_continuous_motion_release_pending.tolist() == [False]
    assert not bool(command.just_resampled.any())

    frame0 = int(command.motion.seg_start[0].item())
    _held, suffix_due = command._advance_action_ball_continuous_motion_cadence()
    assert suffix_due.tolist() == [False]
    assert command._action_ball_task_age_s.tolist() == pytest.approx([0.02])
    assert command._action_ball_task_timing_active.tolist() == [True]

    playback_advanced = False
    for _ in range(150):
        _held, suffix_due = command._advance_action_ball_continuous_motion_cadence()
        if command.time_steps.item() > frame0:
            playback_advanced = True
            assert command.speed_scale.item() > 0.0
            assert command.hold_counter.item() == 0
        if suffix_due.item():
            break
    else:
        pytest.fail("real task suffix did not complete within its bounded cycle")

    assert playback_advanced
    assert suffix_due.tolist() == [True]
    assert (
        command._action_ball_continuous_episode_step.item()
        < command._action_ball_continuous_next_reveal_step.item()
    )
    assert command._action_ball_continuous_suffix_complete.tolist() == [True]
    assert command._action_ball_continuous_motion_active.tolist() == [False]
    assert command._action_ball_continuous_ready_reference_active.tolist() == [True]
    assert command.time_steps.tolist() == [command.motion.seg_start[0].item()]
    assert command.speed_scale.tolist() == [0.0]
    assert command.in_hold.tolist() == [True]
    assert torch.equal(
        command.joint_pos,
        command.motion.joint_pos[command.motion.seg_start[:1]],
    )
    assert not bool((command.joint_pos == 99.0).any())
    assert torch.count_nonzero(command.joint_vel).item() == 0
    assert torch.count_nonzero(command.body_lin_vel_w).item() == 0
    assert torch.count_nonzero(command.body_ang_vel_w).item() == 0
    assert torch.count_nonzero(command.anchor_lin_vel_w).item() == 0
    assert torch.count_nonzero(command.anchor_ang_vel_w).item() == 0
    assert torch.equal(command.robot.data.root_state_w, root_before)
    assert torch.equal(command.robot.data.joint_pos, joint_before)
    assert not bool(command.just_resampled.any())
    assert not bool(command.action_ball_single_stroke_complete.any())
    cadence_source = inspect.getsource(
        C.MotionCommand._advance_action_ball_continuous_motion_cadence
    )
    assert "resolve_pending=False" in cadence_source


def test_live_update_routes_continuous_suffix_boundary_to_empty_wrap_batch():
    command, _env_ids = _continuous_command(num_envs=4)
    command._stagger_ep_pending = False
    command._multiseg = True
    due = torch.tensor([False, True, False, True])
    command._advance_action_ball_continuous_motion_cadence = types.MethodType(
        lambda self: (torch.zeros(4, dtype=torch.bool), due),
        command,
    )
    selected = {}

    class _SelectionCaptured(Exception):
        pass

    def capture_resample(self, env_ids):
        selected["env_ids"] = env_ids.clone()
        raise _SelectionCaptured

    command._resample_command = types.MethodType(capture_resample, command)
    with pytest.raises(_SelectionCaptured):
        command._update_command()

    assert selected["env_ids"].numel() == 0
    assert not bool(command._action_ball_single_stroke_complete.any())


def test_live_update_fails_closed_before_birth_authority_is_bound():
    command, _env_ids = _continuous_command()
    command._stagger_ep_pending = False
    command._action_ball_birth_broker = None

    with pytest.raises(RuntimeError, match="requires the action-ball birth"):
        command._update_command()


def test_bridge_has_no_reset_teleport_history_or_target_ball_writer():
    ready_source = inspect.getsource(
        C.MotionCommand._hold_action_ball_continuous_ready_reference
    )
    cadence_source = inspect.getsource(
        C.MotionCommand._advance_action_ball_continuous_motion_cadence
    )
    commit_source = inspect.getsource(
        C.MotionCommand.commit_action_ball_continuous_task
    )
    release_source = inspect.getsource(
        C.MotionCommand.release_action_ball_continuous_motion_playback
    )
    validation_source = inspect.getsource(
        C.MotionCommand._validate_action_ball_continuous_task_timing_binding
    )
    acknowledgement_source = inspect.getsource(
        C.MotionCommand.acknowledge_action_ball_continuous_motion_task
    )
    projection_source = inspect.getsource(
        C.MotionCommand.action_ball_continuous_current_projection
    )
    projection_validation_source = inspect.getsource(
        C.MotionCommand._require_action_ball_continuous_projection_current
    )
    prepare_source = inspect.getsource(
        C.MotionCommand.prepare_action_ball_continuous_task_commit
    )
    prepared_commit_source = inspect.getsource(
        C.MotionCommand.commit_prepared_action_ball_continuous_task
    )
    token_validation_source = inspect.getsource(
        C.MotionCommand._validate_action_ball_continuous_task_commit_token
    )
    infrastructure_ack_source = inspect.getsource(
        C.MotionCommand
        .acknowledge_action_ball_continuous_infrastructure_invalid
    )
    suffix_window_source = inspect.getsource(
        C.MotionCommand
        ._validate_action_ball_continuous_full_suffix_window
    )
    full_reveal_source = inspect.getsource(
        C.MotionCommand._action_ball_continuous_require_full_reveal_batch
    )
    full_release_source = inspect.getsource(
        C.MotionCommand._action_ball_continuous_require_full_release_batch
    )
    publication_guard_source = inspect.getsource(
        C.MotionCommand._require_action_ball_continuous_current_publication
    )
    combined = (
        ready_source
        + cadence_source
        + commit_source
        + release_source
        + validation_source
        + acknowledgement_source
        + projection_source
        + projection_validation_source
        + prepare_source
        + prepared_commit_source
        + token_validation_source
        + infrastructure_ack_source
        + suffix_window_source
        + full_reveal_source
        + full_release_source
        + publication_guard_source
    )
    for forbidden in (
        "write_root_state_to_sim",
        "write_joint_state_to_sim",
        "_reset_actor_target_state",
        "_resample_command",
        "reset_idx",
        "episode_length_buf",
        "terminated",
        "truncated",
        "target_pos",
        "ball_pos",
    ):
        assert forbidden not in combined

    # Preflight accepts staged immutable receipt objects and cannot require a
    # live Racket task to have been published already.
    assert "_action_ball_task_ref_for_env" not in prepare_source
    assert "_action_ball_task_receipt_resolver" not in prepare_source
    assert "_action_ball_continuous_require_full_reveal_batch" in commit_source
    assert "_action_ball_continuous_require_full_release_batch" in release_source
    assert "_require_action_ball_continuous_current_publication" in commit_source
    assert "_require_action_ball_continuous_current_publication" in release_source

    update_source = inspect.getsource(C.MotionCommand._update_command)
    wrap_owner = update_source.index(
        "# Receipt timing is the sole ActionBall wrap owner."
    )
    continuous_start = update_source.index(
        "if self.action_ball_continuous_motion_enabled:", wrap_owner
    )
    continuous_branch = update_source[
        continuous_start : update_source.index("elif (", continuous_start)
    ]
    assert "torch.empty(" in continuous_branch
    assert "self._resample_command(" not in continuous_branch
