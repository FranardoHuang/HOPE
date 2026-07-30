"""Focused guards for the diagnostic Motion timing batch handoff.

Run in the repository's Torch test environment:

    python -m pytest -q \
      hope_training/whole_body_tracking/tests/test_action_ball_motion_batch_handoff.py
"""

from __future__ import annotations

from dataclasses import replace
import inspect
import types

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


def _resolve_diagnostic_batch(
    command,
    *,
    host_identity_rows,
    receipts,
    task_refs,
):
    command.resolve_action_ball_task_timing_now(
        diagnostic_host_identity_rows=host_identity_rows,
        diagnostic_receipts=receipts,
        diagnostic_task_refs=task_refs,
    )


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
            swing_generation - 1,
            False,
        )
        for env_id, action_slot in enumerate(command.clip_id.tolist())
    )
    command._action_ball_segment_lengths = tuple(
        int(value) for value in command.motion.seg_len.tolist()
    )
    command._action_ball_diagnostic_pending_row_count = 0
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

    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    command._begin_action_ball_task_pending(
        env_ids, elapsed_s=elapsed_s
    )
    assert command._action_ball_diagnostic_pending_row_count == len(
        env_ids
    )
    authority_calls_before = (
        authority.ref_calls,
        authority.resolve_calls,
    )
    _resolve_diagnostic_batch(
        command,
        host_identity_rows=host_identity_rows,
        receipts=receipts,
        task_refs=task_refs,
    )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        expected,
    )
    assert command._action_ball_diagnostic_pending_row_count == 0
    with pytest.raises(RuntimeError, match="no pending selected batch"):
        _resolve_diagnostic_batch(
            command,
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


@pytest.mark.parametrize("swing_generation", (0, 1))
def test_diagnostic_prevalidated_validator_matches_full_validator(
    swing_generation,
):
    (
        command,
        broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        _elapsed_s,
    ) = _prepare_batch(
        num_envs=3,
        swing_generation=swing_generation,
    )
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)

    for identity, receipt, task_ref in zip(
        host_identity_rows,
        receipts,
        task_refs,
    ):
        (
            env_id,
            action_slot,
            _action_uid,
            reset_generation,
            row_swing_generation,
            _previous_swing_generation,
            _active_before_install,
        ) = identity
        pending_elapsed_s = (
            0.0
            if row_swing_generation == 0
            else float(command._env.step_dt)
        )
        kwargs = {
            "env_id": env_id,
            "reset_generation": reset_generation,
            "swing_generation": row_swing_generation,
            "action_slot": action_slot,
            "segment_length": command._action_ball_segment_lengths[
                action_slot
            ],
            "pending_elapsed_s": pending_elapsed_s,
        }
        full = (
            command._validate_action_ball_task_ref_and_receipt_host(
                task_ref,
                receipt,
                **kwargs,
            )
        )
        lean = (
            command
            ._validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host(
                task_ref,
                receipt,
                **kwargs,
            )
        )
        assert lean == full


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
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=tuple(forged_rows),
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_bad_segment_cannot_partially_write_motion():
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

    segment_lengths = list(command._action_ball_segment_lengths)
    last_action_slot = host_identity_rows[-1][1]
    segment_lengths[last_action_slot] += 1
    command._action_ball_segment_lengths = tuple(segment_lengths)
    with pytest.raises(
        ValueError,
        match="reference_t_cycle_s vs admitted motion",
    ):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_forged_last_timing_field_cannot_partially_write_motion():
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
        receipts[-1],
        "pre_swing_wait_s",
        receipts[-1].pre_swing_wait_s + 0.125,
    )
    with pytest.raises(
        ValueError,
        match="pre_swing_wait_s",
    ):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_coordinated_last_timing_forgery_cannot_partially_write_motion():
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

    forged = receipts[-1]
    forged_rate = 0.9
    forged_required_speed = (
        forged.reference_racket_site_speed_mps * forged_rate
    )
    forged_scaled_hit = forged.reference_t_hit_s / forged_rate
    object.__setattr__(
        forged,
        "required_racket_site_speed_mps",
        forged_required_speed,
    )
    object.__setattr__(forged, "teacher_rate", forged_rate)
    object.__setattr__(
        forged,
        "scaled_t_hit_s",
        forged_scaled_hit,
    )
    object.__setattr__(
        forged,
        "scaled_t_cycle_s",
        forged.reference_t_cycle_s / forged_rate,
    )
    object.__setattr__(
        forged,
        "pre_swing_wait_s",
        forged.time_to_contact_s - forged_scaled_hit,
    )
    with pytest.raises(
        ValueError,
        match="required racket-site speed",
    ):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_last_row_bad_identity_leaves_zero_partial_writes():
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
    last = list(forged_rows[-1])
    last[2] += 1
    forged_rows[-1] = tuple(last)
    with pytest.raises(
        ValueError,
        match="action UID/slot binding changed",
    ):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=tuple(forged_rows),
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_runtime_horizon_tamper_cannot_partially_write_motion():
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

    command._env.max_episode_length = 1
    with pytest.raises(
        ValueError,
        match="cycle plus close tick exceeds runtime episode horizon",
    ):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_pending_elapsed_tamper_is_rejected_by_prevalidated_validator():
    (
        command,
        broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        _elapsed_s,
    ) = _prepare_batch(num_envs=1, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    identity = host_identity_rows[0]
    receipt = receipts[0]

    with pytest.raises(
        RuntimeError,
        match="arrived after its certified ready-wait ended",
    ):
        (
            command
            ._validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host(
                task_refs[0],
                receipt,
                env_id=identity[0],
                reset_generation=identity[3],
                swing_generation=identity[4],
                action_slot=identity[1],
                segment_length=command._action_ball_segment_lengths[
                    identity[1]
                ],
                pending_elapsed_s=receipt.pre_swing_wait_s + 0.001,
            )
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
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=tuple(forged_refs),
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_diagnostic_handoff_rejects_an_incomplete_selected_batch():
    (
        command,
        broker,
        _authority,
        env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        elapsed_s,
    ) = _prepare_batch(num_envs=5, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    command._begin_action_ball_task_pending(
        env_ids, elapsed_s=elapsed_s
    )
    before = _snapshot_motion_timing(command)

    with pytest.raises(
        RuntimeError,
        match="selected row count does not match the pending row count",
    ):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows[:-1],
            receipts=receipts[:-1],
            task_refs=task_refs[:-1],
        )

    assert command._action_ball_diagnostic_pending_row_count == 5
    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_diagnostic_handoff_rejects_same_size_active_row_substitution():
    (
        command,
        broker,
        _authority,
        _env_ids,
        host_identity_rows,
        receipts,
        task_refs,
        elapsed_s,
    ) = _prepare_batch(num_envs=3, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    pending_ids = torch.tensor([0, 1], dtype=torch.long)
    command._begin_action_ball_task_pending(
        pending_ids, elapsed_s=elapsed_s
    )

    # The device assertion is deliberately asynchronous on CUDA.  A forged
    # same-size substitution therefore poisons the process, but its partially
    # queued diagnostic state is not a rollback surface and must not be
    # inspected or retried after the exception.
    with pytest.raises(RuntimeError):
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=(
                host_identity_rows[0],
                host_identity_rows[2],
            ),
            receipts=(receipts[0], receipts[2]),
            task_refs=(task_refs[0], task_refs[2]),
        )


def test_diagnostic_pending_row_count_round_trips_reset_rollback():
    (
        command,
        broker,
        _authority,
        env_ids,
        _host_identity_rows,
        _receipts,
        _task_refs,
        elapsed_s,
    ) = _prepare_batch(num_envs=5, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    snapshot = command._action_ball_reset_motion_snapshot(env_ids)
    command._begin_action_ball_task_pending(
        env_ids, elapsed_s=elapsed_s
    )
    assert command._action_ball_diagnostic_pending_row_count == 5

    command._restore_action_ball_reset_motion_snapshot(
        env_ids, snapshot
    )

    assert command._action_ball_diagnostic_pending_row_count == 0


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
        _resolve_diagnostic_batch(
            command,
            host_identity_rows=host_identity_rows,
            receipts=receipts,
            task_refs=task_refs,
        )

    _assert_motion_timing_equal(
        _snapshot_motion_timing(command),
        before,
    )


def test_diagnostic_batch_handoff_has_no_device_to_host_readback():
    selected_source = inspect.getsource(
        motion_birth.C.MotionCommand
        ._resolve_action_ball_task_timing_diagnostic_selected
    )
    validator_source = inspect.getsource(
        motion_birth.C.MotionCommand
        ._validate_action_ball_task_ref_and_receipt_diagnostic_prevalidated_host
    )
    combined = selected_source + validator_source

    assert ".item(" not in combined
    assert ".cpu(" not in combined
    assert "_action_ball_task_ref_for_env" not in combined
    assert "_action_ball_task_receipt_resolver" not in combined
    assert selected_source.count("torch.tensor(") == 1
    assert "torch.as_tensor(" not in selected_source
    assert (
        "_validate_action_ball_task_ref_and_receipt_host("
        not in selected_source
    )


def test_diagnostic_pending_begin_defers_host_rows_to_racket_handoff():
    begin_source = inspect.getsource(
        motion_birth.C.MotionCommand
        ._begin_action_ball_task_pending
    )
    diagnostic_guard = begin_source.index(
        "if not diagnostic_fast_path:"
    )
    formal_readback = begin_source.index(".detach().cpu().tolist()")
    counter = begin_source.index(
        "self._action_ball_diagnostic_pending_row_count += int("
    )

    assert diagnostic_guard < formal_readback < counter


def test_diagnostic_normal_step_skips_global_pending_scan(monkeypatch):
    (
        command,
        broker,
        _authority,
        _env_ids,
        _host_identity_rows,
        _receipts,
        _task_refs,
        _elapsed_s,
    ) = _prepare_batch(num_envs=1, swing_generation=0)
    command._action_ball_birth_broker = _DiagnosticBrokerView(broker)
    command._action_ball_diagnostic_pending_row_count = 0

    def forbidden_where(*_args, **_kwargs):
        raise AssertionError("normal diagnostic step scanned all environments")

    monkeypatch.setattr(torch, "where", forbidden_where)
    command._resolve_pending_action_ball_tasks()

    advance_source = inspect.getsource(
        motion_birth.C.MotionCommand._advance_action_ball_task_timing
    )
    diagnostic_check = advance_source.index(
        "if self._action_ball_birth_broker.diagnostic_fast_path:"
    )
    formal_reduction = advance_source.index("elif bool(")
    assert diagnostic_check < formal_reduction
    assert ".any()" not in advance_source[
        diagnostic_check:formal_reduction
    ]


def test_action_ball_update_selects_cycle_due_rows_without_event_sync():
    command, _broker, _authority, _env_ids, *_rest = _prepare_batch(
        num_envs=4,
        swing_generation=0,
    )
    command._stagger_ep_pending = False
    command._event_scheduler = None
    command._multiseg = True
    due = torch.tensor([False, True, False, True])
    command._advance_action_ball_task_timing = types.MethodType(
        lambda self: (torch.zeros(4, dtype=torch.bool), due),
        command,
    )
    selected = {}

    class _SelectionCaptured(Exception):
        pass

    def capture_resample(self, env_ids):
        selected["env_ids"] = env_ids.clone()
        raise _SelectionCaptured

    command._resample_command = types.MethodType(
        capture_resample,
        command,
    )
    with pytest.raises(_SelectionCaptured):
        command._update_command()

    assert torch.equal(
        selected["env_ids"],
        torch.tensor([1, 3], dtype=torch.long),
    )


def test_action_ball_update_fast_branch_excludes_event_reductions():
    update_source = inspect.getsource(
        motion_birth.C.MotionCommand._update_command
    )
    guard = update_source.index(
        "if action_ball_active and self._event_scheduler is not None:"
    )
    fast_start = update_source.index(
        "# Receipt timing is the sole ActionBall wrap owner."
    )
    fast_end = update_source.index(
        "elif self._multiseg:",
        fast_start,
    )
    fast_branch = update_source[fast_start:fast_end]

    assert guard < fast_start
    assert "torch.where(action_ball_cycle_due)" in fast_branch
    assert "clamp =" not in fast_branch
    assert ".any()" not in fast_branch
    assert "event_owned" not in fast_branch


def test_racket_routes_only_diagnostic_rows_to_selected_batch_resolver():
    sample_source = inspect.getsource(
        loaded_mdp.hope_commands_mod.RacketTargetCommand
        ._sample_targets_action_ball
    )
    handoff_start = sample_source.index("# Motion resets before Racket.")
    handoff = sample_source[handoff_start:]
    commit = sample_source.index("self._action_ball_commit_install(")
    install = sample_source.index(
        "diagnostic_host_identity_rows=host_identity_rows"
    )
    branch = handoff.index(
        "if self._action_ball_diagnostic_unauthorized:"
    )
    direct = handoff.index(
        "diagnostic_host_identity_rows=host_identity_rows"
    )
    fallback = handoff.index(
        "motion.resolve_action_ball_task_timing_now(ids)"
    )
    latch = handoff.index(
        "self._action_ball_reference_term_center_latch[ids]"
    )

    assert commit < install
    assert branch < direct < fallback < latch
    diagnostic_seam = sample_source[
        commit : handoff_start + fallback
    ]
    assert "try:" not in diagnostic_seam
    assert "except" not in diagnostic_seam
    assert "poisons the whole run" in diagnostic_seam
