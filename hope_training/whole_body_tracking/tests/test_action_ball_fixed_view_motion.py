"""Focused CPU-Torch tests for Motion's immutable-N1 fixed timing view.

The production seam remains diagnostic-only.  These tests exercise Motion's
local bind/install/checkpoint contract with the real dependency-light
ActionBall broker and the synthetic Racket owner from
``test_action_ball_motion_birth``; they do not authorize whole-command
diagnostic resume or replace an Isaac reset/auto-reset check.

Run in the repository's CPU-Torch test environment::

    python -m pytest -q \
      hope_training/whole_body_tracking/tests/test_action_ball_fixed_view_motion.py
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import types

import pytest
import torch

import test_action_ball_motion_batch_handoff as motion_batch
import test_action_ball_motion_birth as motion_birth


_FIXED_VIEW_IDENTITY = motion_birth._digest("motion-fixed-view-n1")
_TIMING_ROW = (
    0.80,  # time_to_contact_s
    motion_birth._REFERENCE_T_HIT_S,
    motion_birth._REFERENCE_T_CYCLE_S,
    3.0,  # reference_racket_site_speed_mps
    3.0,  # required_racket_site_speed_mps
    0.8,  # teacher_rate_min
    1.2,  # teacher_rate_max
    1.0,  # teacher_rate
    motion_birth._REFERENCE_T_HIT_S,  # scaled_t_hit_s
    motion_birth._REFERENCE_T_CYCLE_S,  # scaled_t_cycle_s
    0.80 - motion_birth._REFERENCE_T_HIT_S,  # pre_swing_wait_s
    0.05,  # reaction_margin_s
    3.0,
    0.0,
    0.0,  # racket_site_velocity_w_mps
)
_COMPACT_TIMING_ROW = (
    _TIMING_ROW[0],
    _TIMING_ROW[7],
    _TIMING_ROW[8],
    _TIMING_ROW[9],
    _TIMING_ROW[10],
)


def _n1_diagnostic_motion(num_envs: int):
    """Narrow the existing real-broker Motion harness to exact diagnostic N1."""

    command, runtime, _old_broker, _old_provider, _old_domain = (
        motion_birth._motion_harness(
            num_envs,
            diagnostic_fast_path=True,
        )
    )
    bindings = motion_birth._bindings(runtime, count=1)
    broker = runtime.ActionBirthBroker(
        bindings,
        motion_birth._pins(runtime),
        "no_move",
        diagnostic_unauthorized=True,
    )
    domain = motion_birth._DomainAuthority(runtime, bindings, "no_move")
    provider = motion_birth._BirthProvider(runtime)
    broker.bind_domain_claim_authority(domain)
    broker.bind_provider(provider)

    old_motion = command.motion
    command.motion = types.SimpleNamespace(
        num_segments=1,
        seg_start=old_motion.seg_start[:1].clone(),
        seg_len=old_motion.seg_len[:1].clone(),
        time_step_total=motion_birth._SEGMENT_FRAMES,
        body_pos_w=old_motion.body_pos_w[
            : motion_birth._SEGMENT_FRAMES
        ].clone(),
        body_quat_w=old_motion.body_quat_w[
            : motion_birth._SEGMENT_FRAMES
        ].clone(),
        joint_pos=old_motion.joint_pos[
            : motion_birth._SEGMENT_FRAMES
        ].clone(),
    )
    command.clip_id.zero_()
    command._multiseg = False
    clip_order = (bindings[0].motion_path,)
    command._balanced_clip_sampler = motion_birth._counted_balanced_sampler(
        1, clip_order
    )
    command._motion_files = clip_order
    command._motion_file_sha256 = (bindings[0].motion_sha256,)
    command._action_ball_birth_broker = broker
    command._action_ball_action_uids = broker.ordered_action_uids
    command._action_ball_motion_sha256 = (bindings[0].motion_sha256,)
    command._action_ball_segment_lengths = (
        motion_birth._SEGMENT_FRAMES,
    )
    command._action_ball_ready_root_z = (
        command._action_ball_ready_root_z[0],
    )
    command._action_ball_ready_root_quat = (
        command._action_ball_ready_root_quat[0],
    )

    # ``_motion_harness`` bypasses MotionCommand.__init__; spell out the new
    # production lifecycle fields locally rather than editing the shared helper.
    command.action_ball_diagnostic_split_ready_teacher = False
    command._action_ball_fixed_view_identity_sha256 = None
    command._action_ball_fixed_view_timing_row = None
    command._action_ball_fixed_view_timing_row_device = None
    command._action_ball_fixed_view_broker_state_accessor = None
    command._action_ball_diagnostic_pending_row_count = 0
    command._action_ball_single_stroke_complete = torch.zeros(
        num_envs, dtype=torch.bool
    )
    command._action_ball_public_task_valid = None
    command._action_ball_safe_ready_reference_pending = torch.zeros(
        num_envs, dtype=torch.bool
    )
    command._action_ball_safe_ready_pending_count = 0

    task_authority = motion_birth._bind_task_authority(
        command, runtime, broker
    )
    return command, runtime, broker, provider, domain, task_authority


def _bind_fixed_view(command):
    command.bind_action_ball_fixed_view_timing(
        fixed_view_identity_sha256=_FIXED_VIEW_IDENTITY,
        timing_row=_TIMING_ROW,
        broker_exact_state=(
            command._action_ball_task_ref_for_env.__self__
            .action_ball_fixed_view_broker_exact_state
        ),
    )


def _stage_fixed_true_reset(command, runtime, broker, env_ids):
    previous_swing = command._action_ball_swing_generation.clone()
    active_before = command._action_ball_task_timing_active.clone()
    transaction, _rollback = motion_birth._reserve_write_commit(
        command, env_ids
    )
    # Fixed view removes per-swing task receipts, not physical episode births.
    consumed = motion_birth._consume_committed(
        runtime, broker, transaction["receipts"]
    )
    task_authority = command._action_ball_task_ref_for_env.__self__
    task_authority.record_consumed_births(consumed)
    command._begin_action_ball_task_pending(env_ids, elapsed_s=0.0)
    rows = tuple(
        (
            env_id,
            0,
            command._action_ball_action_uids[0],
            int(command._action_ball_reset_generation[env_id].item()),
            int(command._action_ball_swing_generation[env_id].item()),
            int(previous_swing[env_id].item()),
            bool(active_before[env_id].item()),
        )
        for env_id in range(command.num_envs)
    )
    return transaction, rows


def _install_staged_fixed_view(command, env_ids, rows):
    command.install_action_ball_fixed_view_timing_now(
        env_ids=env_ids,
        host_identity_rows=rows,
        fixed_view_identity_sha256=_FIXED_VIEW_IDENTITY,
    )


def _install_fixed_true_reset(command, runtime, broker, env_ids):
    transaction, rows = _stage_fixed_true_reset(
        command, runtime, broker, env_ids
    )
    _install_staged_fixed_view(command, env_ids, rows)
    return transaction, rows


def test_fixed_view_bind_is_diagnostic_n1_one_shot_and_caches_compact_row():
    command, _runtime, _broker, _provider, _domain, _authority = (
        _n1_diagnostic_motion(3)
    )

    _bind_fixed_view(command)

    assert command.action_ball_fixed_view_enabled is True
    assert command._action_ball_fixed_view_identity_sha256 == (
        _FIXED_VIEW_IDENTITY
    )
    assert command._action_ball_fixed_view_timing_row == _COMPACT_TIMING_ROW
    assert tuple(command._action_ball_fixed_view_timing_row_device.shape) == (
        1,
        5,
    )
    assert command._action_ball_fixed_view_timing_row_device.dtype == (
        torch.float64
    )
    torch.testing.assert_close(
        command._action_ball_fixed_view_timing_row_device[0],
        torch.tensor(_COMPACT_TIMING_ROW, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="bound exactly once"):
        _bind_fixed_view(command)


def test_fixed_view_install_activates_exact_timing_without_task_resolution():
    (
        command,
        runtime,
        broker,
        _provider,
        _domain,
        task_authority,
    ) = _n1_diagnostic_motion(3)
    _bind_fixed_view(command)
    env_ids = torch.arange(3, dtype=torch.long)
    _transaction, rows = _stage_fixed_true_reset(
        command, runtime, broker, env_ids
    )
    assert command._action_ball_diagnostic_pending_row_count == 3
    calls_before = (
        task_authority.ref_calls,
        task_authority.resolve_calls,
    )

    _install_staged_fixed_view(command, env_ids, rows)

    expected = torch.tensor(_COMPACT_TIMING_ROW, dtype=torch.float64)
    assert command._action_ball_diagnostic_pending_row_count == 0
    assert command._action_ball_active_task_refs == [None, None, None]
    assert torch.equal(
        command._action_ball_task_pending_elapsed_s,
        torch.zeros(3, dtype=torch.float64),
    )
    assert torch.equal(
        command._action_ball_task_age_s,
        torch.zeros(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        command._action_ball_time_to_contact_s,
        expected[0].expand(3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        command._action_ball_teacher_rate,
        expected[1].expand(3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        command._action_ball_scaled_t_hit_s,
        expected[2].expand(3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        command._action_ball_scaled_t_cycle_s,
        expected[3].expand(3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        command._action_ball_pre_swing_wait_s,
        expected[4].expand(3),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        command._action_ball_task_timing_active,
        torch.ones(3, dtype=torch.bool),
    )
    assert (
        task_authority.ref_calls,
        task_authority.resolve_calls,
    ) == calls_before


def test_fixed_view_second_true_reset_reopens_the_same_timing_row():
    command, runtime, broker, provider, domain, _task_authority = (
        _n1_diagnostic_motion(2)
    )
    _bind_fixed_view(command)
    env_ids = torch.arange(2, dtype=torch.long)
    first, _rows = _install_fixed_true_reset(
        command, runtime, broker, env_ids
    )
    first_timing = motion_batch._snapshot_motion_timing(command)
    provider_calls = provider.issue_invocations
    domain_calls = domain.claim_invocations

    second, _rows = _install_fixed_true_reset(
        command, runtime, broker, env_ids
    )

    assert torch.equal(
        command._action_ball_reset_generation,
        torch.full((2,), 2, dtype=torch.long),
    )
    assert torch.equal(
        command._action_ball_swing_generation,
        torch.zeros(2, dtype=torch.long),
    )
    assert tuple(first["receipt_sha256"]) != tuple(second["receipt_sha256"])
    assert provider.issue_invocations == provider_calls + 2
    assert domain.claim_invocations == domain_calls + 2
    assert command._action_ball_diagnostic_pending_row_count == 0
    assert command._action_ball_active_task_refs == [None, None]
    motion_batch._assert_motion_timing_equal(
        motion_batch._snapshot_motion_timing(command),
        first_timing,
    )


def test_fixed_view_second_generation_checkpoint_uses_full_birth_history():
    command, runtime, broker, provider, domain, task_authority = (
        _n1_diagnostic_motion(2)
    )
    _bind_fixed_view(command)
    env_ids = torch.arange(2, dtype=torch.long)
    _install_fixed_true_reset(command, runtime, broker, env_ids)
    _install_fixed_true_reset(command, runtime, broker, env_ids)
    producer_calls_before = (
        provider.issue_invocations,
        domain.claim_invocations,
    )

    broker_state = (
        task_authority.action_ball_fixed_view_broker_exact_state()
    )
    saved = command.exact_resume_state_dict()

    assert [
        (row["env_id"], row["reset_generation"])
        for row in broker_state["consumed_receipts"]
    ] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert saved["schema_version"] == 5
    assert (
        provider.issue_invocations,
        domain.claim_invocations,
    ) == producer_calls_before


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    (
        ("identity", RuntimeError, "outside its immutable N1"),
        ("action_uid", ValueError, "identity row 0 is invalid"),
        ("reset_generation", RuntimeError, None),
        ("swing_generation", RuntimeError, None),
        ("clip_id", RuntimeError, None),
    ),
)
def test_fixed_view_install_rejects_identity_drift_before_timing_write(
    mutation,
    error_type,
    message,
):
    command, runtime, broker, _provider, _domain, _authority = (
        _n1_diagnostic_motion(1)
    )
    _bind_fixed_view(command)
    env_ids = torch.tensor([0], dtype=torch.long)
    _transaction, rows = _stage_fixed_true_reset(
        command, runtime, broker, env_ids
    )
    before = motion_batch._snapshot_motion_timing(command)
    identity = _FIXED_VIEW_IDENTITY
    row = list(rows[0])
    if mutation == "identity":
        identity = motion_birth._digest("wrong-fixed-view")
    elif mutation == "action_uid":
        row[2] += 1
    elif mutation == "reset_generation":
        row[3] += 1
    elif mutation == "swing_generation":
        row[4] = 1
        row[5] = 0
    else:
        command.clip_id[0] = 1

    context = pytest.raises(error_type, match=message)
    with context:
        command.install_action_ball_fixed_view_timing_now(
            env_ids=env_ids,
            host_identity_rows=(tuple(row),),
            fixed_view_identity_sha256=identity,
        )

    motion_batch._assert_motion_timing_equal(
        motion_batch._snapshot_motion_timing(command),
        before,
    )
    assert command._action_ball_diagnostic_pending_row_count == 1


def test_fixed_view_install_has_no_device_to_host_or_legacy_task_resolution():
    source = inspect.getsource(
        motion_birth.C.MotionCommand.install_action_ball_fixed_view_timing_now
    )

    assert ".cpu(" not in source
    assert ".tolist(" not in source
    assert ".item(" not in source
    assert "_action_ball_task_ref_for_env" not in source
    assert "_action_ball_task_receipt_resolver" not in source
    assert source.count("torch.tensor(") == 1
    assert ".expand(" in source


def test_fixed_view_exact_checkpoint_rejects_incomplete_reset_handoff():
    command, runtime, broker, _provider, _domain, task_authority = (
        _n1_diagnostic_motion(2)
    )
    _bind_fixed_view(command)
    env_ids = torch.arange(2, dtype=torch.long)
    _transaction, _rows = _stage_fixed_true_reset(
        command, runtime, broker, env_ids
    )
    digest_calls_before = task_authority.digest_calls

    with pytest.raises(RuntimeError, match="incomplete.*timing handoff"):
        command.exact_resume_state_dict()

    assert command._action_ball_diagnostic_pending_row_count == 2
    assert not bool(command._action_ball_task_timing_active.any())
    assert task_authority.digest_calls == digest_calls_before


@pytest.mark.parametrize("mutation", ("fixed_identity", "task_ref"))
def test_fixed_view_exact_load_rejects_identity_or_task_ref_atomically(
    mutation,
):
    command, runtime, broker, _provider, _domain, _task_authority = (
        _n1_diagnostic_motion(1)
    )
    _bind_fixed_view(command)
    env_ids = torch.tensor([0], dtype=torch.long)
    _install_fixed_true_reset(command, runtime, broker, env_ids)
    saved = command.exact_resume_state_dict()
    tampered = deepcopy(saved)
    if mutation == "fixed_identity":
        tampered["action_ball_birth"]["fixed_view_identity_sha256"] = (
            motion_birth._digest("wrong-resume-fixed-view")
        )
        match = "immutable identity differs"
    else:
        tampered["action_ball_birth"]["active_task_refs"][0] = {
            "forbidden": "legacy-task-ref"
        }
        match = "must not contain task refs"

    with pytest.raises(ValueError, match=match):
        command.load_exact_resume_state_dict(tampered, strict=True)

    motion_birth._assert_nested_equal(command.exact_resume_state_dict(), saved)


def test_fixed_view_motion_exact_state_load_finalize_roundtrip_is_data_only():
    (
        command,
        runtime,
        broker,
        provider,
        domain,
        task_authority,
    ) = _n1_diagnostic_motion(4)
    _bind_fixed_view(command)
    env_ids = torch.arange(4, dtype=torch.long)
    _install_fixed_true_reset(command, runtime, broker, env_ids)
    saved_shared_racket = deepcopy(task_authority.state_dict())
    saved = command.exact_resume_state_dict()

    assert saved["schema_version"] == 5
    assert (
        saved["identity"]["action_ball"]["fixed_view_identity_sha256"]
        == _FIXED_VIEW_IDENTITY
    )
    assert (
        saved["action_ball_birth"]["fixed_view_identity_sha256"]
        == _FIXED_VIEW_IDENTITY
    )
    assert saved["action_ball_birth"]["active_task_refs"] == [
        None,
        None,
        None,
        None,
    ]

    # Move every Motion-local and synthetic shared owner away from the save.
    command._balanced_clip_sampler.sample(7)
    _install_fixed_true_reset(command, runtime, broker, env_ids)
    command.bin_failed_count[:] = 11.0
    command._current_bin_failed[:] = 13.0
    task_authority.load_state_dict(saved_shared_racket)

    broker_load_calls = 0
    original_broker_load = broker.load_state_dict

    def counted_broker_load(state):
        nonlocal broker_load_calls
        broker_load_calls += 1
        return original_broker_load(state)

    broker.load_state_dict = counted_broker_load
    provider_issue_before = provider.issue_invocations
    domain_claim_before = domain.claim_invocations
    sampler_calls_before = command._balanced_clip_sampler.sample_invocations
    root_writes_before = command.robot.root_write_calls
    joint_writes_before = command.robot.joint_write_calls
    ref_calls_before = task_authority.ref_calls
    resolve_calls_before = task_authority.resolve_calls
    digest_calls_before = task_authority.digest_calls

    command.load_exact_resume_state_dict(saved, strict=True)

    assert broker_load_calls == 0
    assert provider.issue_invocations == provider_issue_before
    assert domain.claim_invocations == domain_claim_before
    assert command._balanced_clip_sampler.sample_invocations == (
        sampler_calls_before
    )
    assert command.robot.root_write_calls == root_writes_before
    assert command.robot.joint_write_calls == joint_writes_before
    assert task_authority.ref_calls == ref_calls_before
    assert task_authority.resolve_calls == resolve_calls_before
    assert task_authority.digest_calls == digest_calls_before
    assert command._action_ball_active_task_refs == [None] * 4
    assert torch.equal(command.clip_id, torch.zeros(4, dtype=torch.long))
    for name in motion_batch._TIMING_BUFFER_NAMES:
        value = getattr(command, name)
        assert not bool(value.any()), name
    assert command._action_ball_expected_shared_racket_state_sha256 == (
        saved["action_ball_birth"]["shared_racket_state_sha256"]
    )

    # Finalization detects a shared-owner mismatch, then accepts the exact
    # Racket-first restore without sampling or simulator writes.
    task_authority._nonce += 1
    with pytest.raises(RuntimeError, match="live Racket state differs"):
        command.finalize_action_ball_exact_resume()
    task_authority.load_state_dict(saved_shared_racket)
    command.finalize_action_ball_exact_resume()

    assert provider.issue_invocations == provider_issue_before
    assert domain.claim_invocations == domain_claim_before
    assert command._balanced_clip_sampler.sample_invocations == (
        sampler_calls_before
    )
    assert command.robot.root_write_calls == root_writes_before
    assert command.robot.joint_write_calls == joint_writes_before
    assert task_authority.ref_calls == ref_calls_before
    assert task_authority.resolve_calls == resolve_calls_before
    motion_birth._assert_nested_equal(command.exact_resume_state_dict(), saved)
