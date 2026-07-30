"""Focused guards for the diagnostic Motion timing batch handoff.

Run in the repository's Torch test environment:

    python -m pytest -q \
      hope_training/whole_body_tracking/tests/test_action_ball_motion_batch_handoff.py
"""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

import test_action_ball_motion_birth as motion_birth
import test_reward_flags_mdp as loaded_mdp


_TIMING_BUFFER_NAMES = (
    "_action_ball_task_pending_elapsed_s",
    "_action_ball_task_age_s",
    "_action_ball_time_to_contact_s",
    "_action_ball_teacher_rate",
    "_action_ball_scaled_t_hit_s",
    "_action_ball_scaled_t_cycle_s",
    "_action_ball_pre_swing_wait_s",
    "_action_ball_task_timing_active",
)


class _DiagnosticBrokerView:
    """Expose the already-bound synthetic broker under diagnostic branding."""

    diagnostic_fast_path = True

    def __init__(self, broker):
        self._broker = broker

    def __getattr__(self, name):
        return getattr(self._broker, name)


def _snapshot_motion_timing(command):
    snapshot = {
        name: getattr(command, name).clone()
        for name in _TIMING_BUFFER_NAMES
    }
    snapshot["active_task_refs"] = tuple(
        command._action_ball_active_task_refs
    )
    return snapshot


def _assert_motion_timing_equal(actual, expected):
    assert set(actual) == set(expected)
    for name in _TIMING_BUFFER_NAMES:
        assert torch.equal(actual[name], expected[name]), name
    assert actual["active_task_refs"] == expected["active_task_refs"]


def _prepare_batch(*, num_envs: int, swing_generation: int):
    command, runtime, broker, _provider, _domain = (
        motion_birth._motion_harness(num_envs)
    )
    authority = motion_birth._bind_task_authority(
        command, runtime, broker
    )
    env_ids = torch.arange(num_envs, dtype=torch.long)
    transaction, _rollback = motion_birth._reserve_write_commit(
        command, env_ids
    )
    command._action_ball_swing_generation[env_ids] = swing_generation
    elapsed_s = (
        0.0
        if swing_generation == 0
        else float(command._env.step_dt)
    )
    receipts = motion_birth._install_current_tasks(
        command,
        runtime,
        broker,
        authority,
        transaction["receipts"],
        elapsed_s=elapsed_s,
    )
    task_refs = tuple(receipt.task_ref() for receipt in receipts)
    host_identity_rows = tuple(
        (
            env_id,
            action_slot,
            command._action_ball_action_uids[action_slot],
            int(command._action_ball_reset_generation[env_id]),
            swing_generation,
            max(swing_generation - 1, 0),
            False,
        )
        for env_id, action_slot in enumerate(command.clip_id.tolist())
    )
    command._action_ball_segment_lengths = tuple(
        int(value) for value in command.motion.seg_len.tolist()
    )
    return (
        command,
        broker,
        authority,
        env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        elapsed_s,
    )


@pytest.mark.parametrize(
    ("swing_generation", "expected_age_s"),
    (
        (0, 0.0),
        (1, motion_birth._POLICY_DT_S),
    ),
)
def test_diagnostic_batch_matches_existing_timing_resolver(
    swing_generation,
    expected_age_s,
):
    (
        command,
        broker,
        authority,
        env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        elapsed_s,
    ) = _prepare_batch(
        num_envs=3,
        swing_generation=swing_generation,
    )

    command.resolve_action_ball_task_timing_now(env_ids)
    expected = _snapshot_motion_timing(command)
    assert torch.equal(
        command._action_ball_task_age_s[env_ids],
        torch.full(
            (len(env_ids),),
            expected_age_s,
            dtype=torch.float64,
        ),
    )

    command._begin_action_ball_task_pending(
        env_ids, elapsed_s=elapsed_s
    )
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    authority_calls_before = (
        authority.ref_calls,
        authority.resolve_calls,
    )
    command.install_action_ball_task_timing_diagnostic_many(
        host_identity_rows=host_identity_rows,
        receipts=receipts,
        task_refs=task_refs,
    )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        expected,
    )
    assert (
        authority.ref_calls,
        authority.resolve_calls,
    ) == authority_calls_before


def test_forged_second_identity_row_cannot_partially_write_motion():
    (
        command,
        broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        _elapsed_s,
    ) = _prepare_batch(num_envs=3, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)

    for index, name in enumerate(_TIMING_BUFFER_NAMES[:-1], start=1):
        getattr(command, name).fill_(float(index))
    command._action_ball_task_timing_active[:] = False
    command._action_ball_active_task_refs[:] = [
        f"sentinel-{env_id}" for env_id in range(command.num_envs)
    ]
    before = _snapshot_motion_timing(command)

    forged_rows = list(host_identity_rows)
    forged = list(forged_rows[1])
    forged[2] += 1
    forged_rows[1] = tuple(forged)
    with pytest.raises(
        ValueError,
        match="action UID/slot binding changed",
    ):
        command.install_action_ball_task_timing_diagnostic_many(
            host_identity_rows=tuple(forged_rows),
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_forged_second_timing_field_cannot_partially_write_motion():
    (
        command,
        broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        _elapsed_s,
    ) = _prepare_batch(num_envs=3, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)

    for index, name in enumerate(_TIMING_BUFFER_NAMES[:-1], start=1):
        getattr(command, name).fill_(float(index))
    command._action_ball_task_timing_active[:] = False
    command._action_ball_active_task_refs[:] = [
        f"sentinel-{env_id}" for env_id in range(command.num_envs)
    ]
    before = _snapshot_motion_timing(command)

    object.__setattr__(
        receipts[1],
        "pre_swing_wait_s",
        receipts[1].pre_swing_wait_s + 0.125,
    )
    with pytest.raises(
        ValueError,
        match="pre_swing_wait_s",
    ):
        command.install_action_ball_task_timing_diagnostic_many(
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_forged_valid_task_digest_cannot_partially_write_motion():
    (
        command,
        broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        _elapsed_s,
    ) = _prepare_batch(num_envs=3, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    before = _snapshot_motion_timing(command)

    forged_refs = list(task_refs)
    replacement_digest = "f" * 64
    if replacement_digest == forged_refs[1].task_sha256:
        replacement_digest = "e" * 64
    forged_refs[1] = replace(
        forged_refs[1],
        task_sha256=replacement_digest,
    )
    with pytest.raises(
        ValueError,
        match="task ref changed receipt identity",
    ):
        command.install_action_ball_task_timing_diagnostic_many(
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=tuple(forged_refs),
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_direct_batch_handoff_is_rejected_for_formal_broker():
    (
        command,
        _broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        _elapsed_s,
    ) = _prepare_batch(num_envs=1, swing_generation=0)
    before = _snapshot_motion_timing(command)

    with pytest.raises(RuntimeError, match="diagnostic-only"):
        command.install_action_ball_task_timing_diagnostic_many(
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_diagnostic_batch_handoff_has_no_device_to_host_readback():
    install_source = inspect.getsource(
        motion_birth.C.MotionCommand
        .install_action_ball_task_timing_diagnostic_many
    )
    validator_source = inspect.getsource(
        motion_birth.C.MotionCommand
        ._validate_action_ball_task_ref_and_receipt_host
    )
    combined = install_source + validator_source

    assert ".item(" not in combined
    assert ".cpu(" not in combined
    assert "_action_ball_task_ref_for_env" not in combined
    assert "_action_ball_task_receipt_resolver" not in combined


def test_racket_routes_only_diagnostic_rows_to_direct_batch_handoff():
    sample_source = inspect.getsource(
        loaded_mdp.hope_commands_mod.RacketTargetCommand
        ._sample_targets_action_ball
    )
    handoff_start = sample_source.index("# Motion resets before Racket.")
    handoff = sample_source[handoff_start:]
    commit = sample_source.index("self._action_ball_commit_install(")
    install = sample_source.index(
        "motion.install_action_ball_task_timing_diagnostic_many("
    )
    branch = handoff.index(
        "if self._action_ball_diagnostic_unauthorized:"
    )
    direct = handoff.index(
        "motion.install_action_ball_task_timing_diagnostic_many("
    )
    fallback = handoff.index(
        "motion.resolve_action_ball_task_timing_now(ids)"
    )
    latch = handoff.index(
        "self._action_ball_reference_term_center_latch[ids]"
    )

    assert commit < install
    assert branch < direct < fallback < latch
    diagnostic_seam = sample_source[commit:fallback]
    assert "try:" not in diagnostic_seam
    assert "except" not in diagnostic_seam
    assert "poisons the whole diagnostic run" in diagnostic_seam
