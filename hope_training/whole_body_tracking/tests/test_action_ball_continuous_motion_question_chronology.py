"""Focused Motion-owned exact question chronology and production-HOLD tests."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "source" / "whole_body_tracking"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

import action_ball_physical_question_device as physical  # noqa: E402
import test_action_ball_continuous_motion_bridge as bridge  # noqa: E402


C = bridge.C
_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    _DEVICES.append(torch.device("cuda", torch.cuda.current_device()))


def _seed_complete_canonical_prepare(command) -> None:
    device = torch.device(command.device)
    command._action_ball_continuous_motion_device_r05_owner = object()
    command._action_ball_continuous_sequence_active.fill_(True)
    command._action_ball_continuous_canonical_phase.fill_(
        C.ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE
    )
    command._action_ball_continuous_canonical_phase_start_tick.copy_(
        command._action_ball_continuous_episode_step
    )
    command._action_ball_continuous_canonical_task_identity.copy_(
        torch.tensor([101, 102], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_cadence_identity.copy_(
        torch.tensor([201, 202], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_action_uid.copy_(
        torch.as_tensor(command._action_ball_action_uids, device=device)[
            command.clip_id
        ]
    )
    command._action_ball_continuous_canonical_shot_index.copy_(
        torch.tensor([301, 302], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_outcome_identity.copy_(
        torch.tensor([351, 352], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_task_receipt_sha256.fill_(0x11)
    command._action_ball_continuous_canonical_cadence_receipt_sha256.fill_(0x22)
    command._action_ball_continuous_canonical_candidate_identity.copy_(
        torch.tensor([401, 402], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_contact_tick.copy_(
        torch.tensor([50, 60], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_launch_tick.copy_(
        torch.tensor([45, 55], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_canonical_chosen_horizon_tick.fill_(5)
    command._action_ball_continuous_canonical_task_valid.fill_(True)
    command._action_ball_continuous_canonical_playback_started.fill_(False)
    command._action_ball_task_timing_active.fill_(True)
    command._action_ball_task_pending_elapsed_s.fill_(0.1)
    command._action_ball_task_age_s.fill_(0.1)
    command._action_ball_time_to_contact_s.fill_(0.9)
    command._action_ball_teacher_rate.fill_(1.0)
    command._action_ball_scaled_t_hit_s.fill_(0.4)
    command._action_ball_scaled_t_cycle_s.fill_(0.6)
    command._action_ball_pre_swing_wait_s.fill_(0.05)
    command._action_ball_continuous_current_deadline_step.copy_(
        torch.tensor([30, 40], dtype=torch.int64, device=device)
    )
    command._action_ball_continuous_current_reveal_step.copy_(
        command._action_ball_continuous_episode_step
    )
    command._action_ball_continuous_next_reveal_step.copy_(
        command._action_ball_continuous_current_deadline_step + 5
    )
    command._action_ball_continuous_scheduled_ordinal.fill_(0)
    command._action_ball_continuous_last_closed_ordinal.fill_(-1)
    command._action_ball_continuous_opportunities_consumed.zero_()


def _physical_batch(device: torch.device) -> physical.PhysicalQuestionCandidateBatch:
    return physical.PhysicalQuestionCandidateBatch(
        candidate_identity=torch.tensor(
            [[101, 102], [201, 202]], dtype=torch.int64, device=device
        ),
        contact_position_env_m=torch.tensor(
            [
                [[-0.45, 0.10, 1.05], [-0.42, -0.12, 0.95]],
                [[-0.48, 0.08, 1.03], [-0.40, -0.10, 0.98]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        incoming_linear_velocity_world_mps=torch.tensor(
            [
                [[-3.2, 0.25, -0.5], [-3.0, -0.2, 0.2]],
                [[-3.1, 0.20, -0.4], [-2.9, -0.1, 0.1]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        incoming_angular_velocity_world_radps=torch.tensor(
            [
                [[20.0, -10.0, 15.0], [30.0, 5.0, -20.0]],
                [[18.0, -8.0, 12.0], [25.0, 4.0, -15.0]],
            ],
            dtype=torch.float32,
            device=device,
        ),
    )


def _physical_owner() -> physical.PhysicalQuestionNumericCore:
    return physical.make_test_physical_question_numeric_core(
        params=physical.PhysicalQuestionFlightParams(
            k_d=0.08, k_m=0.001, g=9.81, ball_radius_m=0.02
        ),
        config=physical.PhysicalQuestionNumericConfig(
            motion_tick_s=0.02,
            integration_substeps_per_motion_tick=1,
            max_final_segment_motion_ticks=5,
            table_surface_z_m=0.76,
        ),
    )


def _motion(device: torch.device = torch.device("cpu")) -> tuple[object, tuple]:
    command, _ = bridge._configure_unbound_command(num_envs=2)
    if device.type == "cuda":
        for name, value in tuple(vars(command).items()):
            if torch.is_tensor(value):
                setattr(command, name, value.to(device))
        for name, value in tuple(vars(command.motion).items()):
            if torch.is_tensor(value):
                setattr(command.motion, name, value.to(device))
        command._env.scene.env_origins = (
            command._env.scene.env_origins.to(device)
        )
        command.device = str(device)
    command._action_ball_continuous_episode_step.copy_(
        torch.tensor([10, 20], dtype=torch.int64, device=device)
    )
    return command, command._continuous_test_context.receipts


def _issue(command, receipts, *, candidate_delta: int = 0):
    owner = _physical_owner()
    batch = _physical_batch(torch.device(command.device))
    horizon_receipt = owner.issue_horizon_for_test(batch)
    candidate = batch.candidate_identity.clone()
    if candidate_delta:
        candidate[0, 0] += candidate_delta
    receipt = command.issue_action_ball_motion_question_chronology(
        selected_env_index=torch.tensor(
            [0, 1], dtype=torch.int64, device=command.device
        ),
        candidate_identity=candidate,
        runtime_task_receipts=receipts,
        physical_horizon_owner=owner,
        physical_horizon_receipt=horizon_receipt,
    )
    return owner, horizon_receipt, receipt


def test_n2_contact_tick_is_task_fact_not_deadline_and_uses_max_horizon() -> None:
    command, receipts = _motion()
    owner, horizon_receipt, receipt = _issue(command, receipts)
    view = command.require_owned_action_ball_motion_question_chronology(receipt)
    horizon = owner.project_horizon_for_test(horizon_receipt)

    expected_contact = torch.tensor(
        [
            10 + round(receipts[0].time_to_contact_s / command._env.step_dt),
            20 + round(receipts[1].time_to_contact_s / command._env.step_dt),
        ],
        dtype=torch.int64,
    )
    expected_ticks = torch.minimum(
        horizon.max_feasible_motion_ticks,
        expected_contact.unsqueeze(1),
    )
    assert torch.equal(view.contact_tick, expected_contact)
    assert torch.equal(view.launch_tick, expected_contact.unsqueeze(1) - expected_ticks)
    assert torch.equal(view.earliest_launch_tick, view.launch_tick)
    assert torch.equal(view.chosen_horizon_s, expected_ticks.float() * 0.02)
    assert tuple(view.task_receipt_sha256.shape) == (2, 32)

    # Cadence close is a different fact; a one-tick mutation cannot move contact.
    command._action_ball_continuous_current_deadline_step.add_(1)
    _, _, second = _issue(command, receipts)
    second_view = command.require_owned_action_ball_motion_question_chronology(second)
    assert torch.equal(second_view.contact_tick, expected_contact)


def test_one_tick_current_contact_clock_mutation_moves_only_its_chronology() -> None:
    command, receipts = _motion()
    _, _, base_receipt = _issue(command, receipts)
    command._action_ball_continuous_episode_step[0] += 1
    _, _, shifted_receipt = _issue(command, receipts)
    base = command.require_owned_action_ball_motion_question_chronology(base_receipt)
    changed = command.require_owned_action_ball_motion_question_chronology(
        shifted_receipt
    )
    assert torch.equal(changed.contact_tick - base.contact_tick, torch.tensor([1, 0]))
    assert torch.equal(
        changed.task_receipt_sha256[0], base.task_receipt_sha256[0]
    )


def test_foreign_physical_receipt_and_candidate_binding_fail_closed() -> None:
    command, receipts = _motion()
    left = _physical_owner()
    right = _physical_owner()
    batch = _physical_batch(torch.device(command.device))
    foreign = left.issue_horizon_for_test(batch)
    with pytest.raises(physical.PhysicalQuestionConflictError, match="foreign"):
        command.issue_action_ball_motion_question_chronology(
            selected_env_index=torch.tensor([0, 1], dtype=torch.int64),
            candidate_identity=batch.candidate_identity,
            runtime_task_receipts=receipts,
            physical_horizon_owner=right,
            physical_horizon_receipt=foreign,
        )

    _, _, receipt = _issue(command, receipts, candidate_delta=1)
    view = command.require_owned_action_ball_motion_question_chronology(receipt)
    assert torch.all(
        torch.bitwise_and(
            view.producer_fault[0],
            C._ACTION_BALL_MOTION_QUESTION_FAULT_UNATTRIBUTABLE,
        ).ne(0)
    )


def test_hot_stage_is_real_device_ingress_but_explicit_negative_hold() -> None:
    source = inspect.getsource(
        C.MotionCommand.stage_action_ball_continuous_motion_device_r05_reveal
    )
    assert "require_owned_prepared_reveal_for_child" in source
    assert 'owner_kind="motion"' in source
    assert "ActionBallMotionQuestionProductionHold" in source
    assert "stage_action_ball_continuous_motion_reveal" not in source
    assert "portable" not in source.lower()
    assert not hasattr(
        C.MotionCommand,
        "_install_action_ball_continuous_canonical_prepare_from_device_r05",
    )


def test_chronology_hot_methods_have_no_host_tensor_observation() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            C.MotionCommand.issue_action_ball_motion_question_chronology,
            C.MotionCommand.require_owned_action_ball_motion_question_chronology,
            C.MotionCommand.stage_action_ball_continuous_motion_device_r05_reveal,
        )
    )
    for forbidden in (".item(", ".cpu(", ".tolist(", ".numpy(", "torch._assert_async"):
        assert forbidden not in source


def test_canonical_lifecycle_uses_teacher_deadline_suffix_and_r07_ready() -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    start = command.motion.seg_start[command.clip_id]
    command.time_steps.copy_(start + 1)
    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=torch.ones(2, dtype=torch.bool),
        suffix_due=torch.zeros(2, dtype=torch.bool),
        closed_without_playback=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(
        command._action_ball_continuous_canonical_phase,
        torch.full(
            (2,), C.ACTION_BALL_CONTINUOUS_CANONICAL_SWING, dtype=torch.int64
        ),
    )
    assert torch.all(command._action_ball_continuous_canonical_playback_started)

    command._action_ball_continuous_episode_step.copy_(
        command._action_ball_continuous_current_deadline_step
    )
    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=torch.ones(2, dtype=torch.bool),
        suffix_due=torch.zeros(2, dtype=torch.bool),
        closed_without_playback=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.all(
        command._action_ball_continuous_canonical_phase.eq(
            C.ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH
        )
    )

    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=torch.ones(2, dtype=torch.bool),
        suffix_due=torch.ones(2, dtype=torch.bool),
        closed_without_playback=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.all(
        command._action_ball_continuous_canonical_phase.eq(
            C.ACTION_BALL_CONTINUOUS_CANONICAL_RECOVER_HIDDEN
        )
    )
    assert not torch.any(command._action_ball_continuous_canonical_task_valid)
    assert not torch.any(
        command._action_ball_continuous_canonical_playback_started
    )
    assert torch.all(command._action_ball_continuous_canonical_action_uid.eq(-1))

    class _Ready:
        def __init__(self, owner):
            self.owner = owner

        def validate(self, projection, *, owner_kind):
            assert projection is self
            return type(
                "ReadyView",
                (),
                {
                    "ready_projection": self,
                    "owner_kind": owner_kind,
                    "ready": torch.ones(2, dtype=torch.bool),
                    "control_tick": self.owner._action_ball_continuous_episode_step.clone(),
                },
            )()

    ready = _Ready(command)
    command._action_ball_continuous_r07_ready_projection = ready
    command._action_ball_continuous_r07_ready_validator = ready.validate
    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=torch.zeros(2, dtype=torch.bool),
        suffix_due=torch.zeros(2, dtype=torch.bool),
        closed_without_playback=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.all(
        command._action_ball_continuous_canonical_phase.eq(
            C.ACTION_BALL_CONTINUOUS_CANONICAL_READY_HOLD
        )
    )


def test_zero_or_ordinary_miss_does_not_invent_a_canonical_phase() -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    before = command._action_ball_continuous_canonical_phase.clone()
    command.time_steps.copy_(command.motion.seg_start[command.clip_id])
    command._action_ball_task_age_s.zero_()
    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=torch.zeros(2, dtype=torch.bool),
        suffix_due=torch.zeros(2, dtype=torch.bool),
        closed_without_playback=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(command._action_ball_continuous_canonical_phase, before)


def test_teacher_first_leaves_frame_zero_at_deadline_enters_follow_directly() -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    command.time_steps.copy_(command.motion.seg_start[command.clip_id] + 1)
    command._action_ball_continuous_episode_step.copy_(
        command._action_ball_continuous_current_deadline_step
    )
    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=torch.ones(2, dtype=torch.bool),
        suffix_due=torch.zeros(2, dtype=torch.bool),
        closed_without_playback=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.all(
        command._action_ball_continuous_canonical_phase.eq(
            C.ACTION_BALL_CONTINUOUS_CANONICAL_FOLLOW_THROUGH
        )
    )


def test_r07_validation_fault_leaves_canonical_owner_bytes_unchanged() -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    command.time_steps.copy_(command.motion.seg_start[command.clip_id] + 1)

    def _fault(_projection, *, owner_kind):
        assert owner_kind == "motion"
        raise RuntimeError("injected R07 fault")

    command._action_ball_continuous_r07_ready_projection = object()
    command._action_ball_continuous_r07_ready_validator = _fault
    tracked = (
        "_action_ball_continuous_canonical_phase",
        "_action_ball_continuous_canonical_phase_start_tick",
        "_action_ball_continuous_canonical_task_valid",
        "_action_ball_continuous_canonical_playback_started",
        "_action_ball_continuous_canonical_task_identity",
        "_action_ball_continuous_canonical_cadence_identity",
        "_action_ball_continuous_canonical_action_uid",
        "_action_ball_continuous_canonical_shot_index",
        "_action_ball_continuous_canonical_outcome_identity",
        "_action_ball_continuous_canonical_task_receipt_sha256",
        "_action_ball_continuous_canonical_cadence_receipt_sha256",
        "_action_ball_continuous_canonical_candidate_identity",
        "_action_ball_continuous_canonical_contact_tick",
        "_action_ball_continuous_canonical_launch_tick",
        "_action_ball_continuous_canonical_chosen_horizon_tick",
    )
    before = {name: getattr(command, name).clone() for name in tracked}
    with pytest.raises(RuntimeError, match="R07 readiness projection became stale"):
        command._advance_action_ball_continuous_canonical_lifecycle(
            motion_active_before=torch.ones(2, dtype=torch.bool),
            suffix_due=torch.zeros(2, dtype=torch.bool),
            closed_without_playback=torch.zeros(2, dtype=torch.bool),
        )
    for name, expected in before.items():
        assert torch.equal(getattr(command, name), expected), name


@pytest.mark.parametrize("device", _DEVICES)
def test_observation_is_pre_published_opaque_and_exact_narrow_snapshot(
    device: torch.device,
) -> None:
    command, _receipts = _motion(device)
    with pytest.raises(RuntimeError, match="not published|stale"):
        command.action_ball_continuous_motion_observation_projection()
    _seed_complete_canonical_prepare(command)
    command._action_ball_continuous_published_common_step = 0
    command._publish_action_ball_continuous_observation()
    token = command.action_ball_continuous_motion_observation_projection()
    view = command.require_owned_action_ball_continuous_motion_observation(token)

    record = command._action_ball_continuous_observation_record
    isolated = ("task_identity", "time_to_contact_remaining_s")
    expected = {name: getattr(record, name).clone() for name in isolated}
    for name in isolated:
        getattr(view, name).zero_()
    second = command.require_owned_action_ball_continuous_motion_observation(token)
    assert second is not view
    for name in isolated:
        assert torch.equal(getattr(record, name), expected[name])
        assert torch.equal(getattr(second, name), expected[name])
    assert tuple(type(view).__dataclass_fields__) == (
        "motion_owner",
        "publication_identity",
        "common_step",
        "control_tick",
        "phase",
        "reset_generation",
        "swing_generation",
        "action_uid",
        "task_identity",
        "task_valid",
        "time_to_contact_remaining_s",
        "time_to_teacher_start_remaining_s",
        "time_to_next_reveal_s",
    )
    for name in tuple(type(view).__dataclass_fields__)[3:]:
        value = getattr(view, name)
        dtype = (
            torch.bool
            if name == "task_valid"
            else torch.float64 if name.endswith("_s") else torch.int64
        )
        assert type(value) is torch.Tensor
        assert tuple(value.shape) == (2,)
        assert value.dtype == dtype
        assert value.device == device
    assert view.motion_owner is command
    assert view.publication_identity is not None
    assert view.common_step == 0


def test_observation_republication_rejects_old_token_and_retains_old_snapshot(
) -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    command._action_ball_continuous_published_common_step = 0
    command._publish_action_ball_continuous_observation()
    old_token = command.action_ball_continuous_motion_observation_projection()
    old_view = command.require_owned_action_ball_continuous_motion_observation(
        old_token
    )
    old_task_identity = old_view.task_identity.clone()
    old_time_to_contact = old_view.time_to_contact_remaining_s.clone()

    command._action_ball_continuous_canonical_task_identity.add_(1000)
    command._action_ball_task_age_s.add_(0.25)
    command._publish_action_ball_continuous_observation()
    new_token = command.action_ball_continuous_motion_observation_projection()
    new_view = command.require_owned_action_ball_continuous_motion_observation(
        new_token
    )

    with pytest.raises(RuntimeError, match="forged or stale"):
        command.require_owned_action_ball_continuous_motion_observation(old_token)
    assert new_view is not old_view
    assert new_view.publication_identity is not old_view.publication_identity
    assert torch.equal(old_view.task_identity, old_task_identity)
    assert torch.equal(
        old_view.time_to_contact_remaining_s, old_time_to_contact
    )
    assert torch.equal(new_view.task_identity, old_task_identity + 1000)
    assert torch.equal(
        new_view.time_to_contact_remaining_s,
        old_time_to_contact - 0.25,
    )


@pytest.mark.parametrize(
    "field",
    [
        "_action_ball_continuous_canonical_task_receipt_sha256",
        "_action_ball_continuous_canonical_cadence_receipt_sha256",
        "_action_ball_continuous_canonical_action_uid",
        "_action_ball_continuous_canonical_shot_index",
        "_action_ball_continuous_canonical_outcome_identity",
        "_action_ball_continuous_canonical_candidate_identity",
        "_action_ball_continuous_canonical_chosen_horizon_tick",
        "_action_ball_task_timing_active",
    ],
)
def test_active_canonical_checkpoint_fails_closed_when_incomplete(field) -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    command._require_action_ball_continuous_canonical_checkpoint_complete()
    value = getattr(command, field)
    if value.dtype == torch.bool:
        value[0] = False
    elif value.ndim == 2:
        value[0].zero_()
    else:
        value[0] = -1
    with pytest.raises(RuntimeError, match="lacks exact mid-task"):
        command._require_action_ball_continuous_canonical_checkpoint_complete()


def test_portable_and_legacy_paths_cannot_publish_canonical_observation() -> None:
    command, _receipts = _motion()
    command._action_ball_continuous_published_common_step = 0
    command._publish_action_ball_continuous_observation()
    with pytest.raises(RuntimeError, match="not published"):
        command.action_ball_continuous_motion_observation_projection()
    source = inspect.getsource(
        C.MotionCommand._advance_action_ball_continuous_canonical_lifecycle
    )
    assert "_action_ball_continuous_motion_device_r05_owner is None" in source


def test_active_midtask_checkpoint_payload_keeps_complete_identity_and_timing() -> None:
    command, _receipts = _motion()
    _seed_complete_canonical_prepare(command)
    command._action_ball_continuous_motion_device_mutation_version = torch.zeros(
        1, dtype=torch.int64
    )
    command._action_ball_continuous_motion_mutation_version = 0
    command._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version = 0
    command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack = False
    command._action_ball_continuous_motion_fault_count_device = torch.zeros(
        1, dtype=torch.int64
    )
    command._action_ball_continuous_motion_terminal_resolution_total_device = torch.zeros(
        1, dtype=torch.int64
    )
    command._action_ball_continuous_motion_terminal_resolution_total = 0
    command._action_ball_continuous_motion_selected_reset_next_serial = 0
    payload = command._action_ball_continuous_motion_checkpoint_payload()
    tensors = payload["tensors"]
    for name in (
        "canonical_phase",
        "canonical_phase_start_tick",
        "task_identity",
        "cadence_identity",
        "action_uid",
        "task_receipt_sha256",
        "cadence_receipt_sha256",
        "candidate_identity",
        "contact_tick",
        "launch_tick",
        "chosen_horizon_tick",
        "timing_active",
        "pending_elapsed_s",
        "task_age_s",
        "time_to_contact_s",
        "teacher_rate",
        "scaled_t_hit_s",
        "scaled_t_cycle_s",
        "pre_swing_wait_s",
        "action_slot",
        "teacher_time_step",
        "teacher_time_step_f",
        "teacher_speed_scale",
    ):
        assert name in tensors
    prepared = command._prepare_action_ball_continuous_motion_checkpoint(payload)
    assert torch.equal(prepared["tensors"]["action_uid"], tensors["action_uid"])
    assert payload["observation_common_step"] is None
    retained_publication = {**payload, "observation_common_step": 0}
    with pytest.raises(ValueError, match="metadata is invalid"):
        command._prepare_action_ball_continuous_motion_checkpoint(
            retained_publication
        )
    impossible = {**payload, "tensors": dict(payload["tensors"])}
    impossible["tensors"]["sequence_active"] = torch.zeros_like(
        payload["tensors"]["sequence_active"]
    )
    impossible_canonical = {
        name: value
        for name, value in impossible.items()
        if name != "canonical_sha256"
    }
    impossible["canonical_sha256"] = hashlib.sha256(
        C._canonical_json_bytes(
            {
                **impossible_canonical,
                "device_owner_mutation_version": impossible_canonical[
                    "device_owner_mutation_version"
                ].tolist(),
                "terminal_resolution_total_device": impossible_canonical[
                    "terminal_resolution_total_device"
                ].tolist(),
                "fault_count_device": impossible_canonical[
                    "fault_count_device"
                ].tolist(),
                "tensors": {
                    name: value.tolist()
                    for name, value in impossible_canonical["tensors"].items()
                },
            }
        )
    ).hexdigest()
    with pytest.raises(ValueError, match="lifecycle invariants"):
        command._prepare_action_ball_continuous_motion_checkpoint(impossible)
    damaged = {**payload, "tensors": dict(payload["tensors"])}
    damaged["tensors"]["teacher_rate"] = payload["tensors"]["teacher_rate"].clone()
    damaged["tensors"]["teacher_rate"][0] = 0.0
    with pytest.raises(ValueError, match="lifecycle invariants"):
        command._prepare_action_ball_continuous_motion_checkpoint(damaged)
