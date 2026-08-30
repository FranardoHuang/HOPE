"""Focused Motion tests for D05's token-only full-N ACCEPT writer."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (SOURCE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import action_ball_continuous_runtime_transaction_device as D05  # noqa: E402
import test_action_ball_continuous_motion_bridge as bridge  # noqa: E402


C = bridge.C
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda", torch.cuda.current_device()))


WRITTEN_TENSORS = (
    "_action_ball_full_mdp_task_yaw_wxyz",
    "_action_ball_full_mdp_task_translation_w",
    "_action_ball_task_pending_elapsed_s",
    "_action_ball_task_age_s",
    "_action_ball_time_to_contact_s",
    "_action_ball_teacher_rate",
    "_action_ball_scaled_t_hit_s",
    "_action_ball_scaled_t_cycle_s",
    "_action_ball_pre_swing_wait_s",
    "_action_ball_task_timing_active",
    "_action_ball_continuous_task_commit_pending",
    "_action_ball_continuous_task_commit_missed",
    "_action_ball_continuous_task_committed",
    "_action_ball_continuous_motion_reset_pending",
    "_action_ball_continuous_motion_release_pending",
    "_action_ball_continuous_motion_release_missed",
    "_action_ball_continuous_motion_active",
    "_action_ball_continuous_suffix_complete",
    "_action_ball_continuous_ready_reference_active",
    "_action_ball_continuous_phase",
    "_action_ball_continuous_current_policy_opportunity",
    "_action_ball_continuous_policy_opportunities_created",
    "_action_ball_continuous_canonical_phase",
    "_action_ball_continuous_canonical_phase_start_tick",
    "_action_ball_continuous_canonical_task_identity",
    "_action_ball_continuous_canonical_cadence_identity",
    "_action_ball_continuous_canonical_action_uid",
    "_action_ball_continuous_canonical_shot_index",
    "_action_ball_continuous_canonical_outcome_identity",
    "_action_ball_continuous_canonical_candidate_identity",
    "_action_ball_continuous_canonical_contact_tick",
    "_action_ball_continuous_canonical_launch_tick",
    "_action_ball_continuous_canonical_chosen_horizon_tick",
    "_action_ball_continuous_canonical_task_close_tick",
    "_action_ball_continuous_canonical_task_valid",
    "_action_ball_continuous_canonical_playback_started",
)


def _to_device(command, device: torch.device) -> None:
    if device.type == "cpu":
        return
    for name, value in tuple(vars(command).items()):
        if torch.is_tensor(value):
            setattr(command, name, value.to(device))
    for name, value in tuple(vars(command.motion).items()):
        if torch.is_tensor(value):
            setattr(command.motion, name, value.to(device))
    command._env.scene.env_origins = command._env.scene.env_origins.to(device)
    command.device = str(device)


def _install_frozen_task_frame_sources(command, n: int, device: torch.device) -> None:
    dtype = torch.float32
    command._action_ball_full_mdp_source_strike_root_xy = torch.tensor(
        [[0.25, -0.15]], dtype=dtype, device=device
    )
    command._action_ball_full_mdp_source_strike_yaw_wxyz = torch.tensor(
        [[math.cos(0.15), 0.0, 0.0, math.sin(0.15)]],
        dtype=dtype,
        device=device,
    )
    command._action_ball_full_mdp_frozen_root_quat_wxyz = torch.zeros(
        n, 4, dtype=dtype, device=device
    )
    command._action_ball_full_mdp_frozen_root_quat_wxyz[:, 0] = math.cos(0.35)
    command._action_ball_full_mdp_frozen_root_quat_wxyz[:, 3] = math.sin(0.35)
    command._action_ball_full_mdp_task_yaw_wxyz = torch.zeros(
        n, 4, dtype=dtype, device=device
    )
    command._action_ball_full_mdp_task_yaw_wxyz[:, 0] = 1.0
    command._action_ball_full_mdp_task_translation_w = torch.zeros(
        n, 3, dtype=dtype, device=device
    )


def _masked_i64(
    values: torch.Tensor, accepted: torch.Tensor
) -> torch.Tensor:
    return torch.where(accepted, values, torch.full_like(values, -1)).contiguous()


def _candidate(
    n: int,
    device: torch.device,
    *,
    candidate_valid: torch.Tensor,
    mutated_key_field: str | None = None,
) -> object:
    epoch_mod = D05._require_action_epoch_module()
    shape = (n, 1)
    rows = torch.arange(n, dtype=torch.int64, device=device).reshape(shape)
    valid = candidate_valid.reshape(shape)
    key_values = {
        "reset_generation": torch.zeros(shape, dtype=torch.int64, device=device),
        "ball_generation": rows + 11,
        "action_uid": rows + 101,
        "action_slot": torch.zeros(shape, dtype=torch.int64, device=device),
        "shot_index": rows + 201,
        "task_identity": rows + 301,
        "outcome_identity": rows + 401,
        "ball_identity": rows + 501,
    }
    key_values = {
        name: _masked_i64(value, valid) for name, value in key_values.items()
    }
    if mutated_key_field is not None:
        key_values[mutated_key_field] = key_values[
            mutated_key_field
        ].clone()
        key_values[mutated_key_field][0, 0] = -1
    key = epoch_mod.ActionEpochShotKey(**key_values)
    task = torch.zeros(
        (n, 1, epoch_mod.TASK_F32_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    timing = torch.stack(
        (
            rows[:, 0].to(torch.float32) + 0.8,
            rows[:, 0].to(torch.float32) + 1.0,
            rows[:, 0].to(torch.float32) + 0.4,
            rows[:, 0].to(torch.float32) + 0.7,
            rows[:, 0].to(torch.float32) + 0.1,
        ),
        dim=1,
    )
    task[:, 0, : epoch_mod.MOTION_TASK_F32_WIDTH] = timing
    task[:, 0, epoch_mod.MOTION_TASK_F32_WIDTH + 19] = rows[:, 0] + 1.25
    task[:, 0, epoch_mod.MOTION_TASK_F32_WIDTH + 20] = rows[:, 0] - 0.35
    task = torch.where(valid.unsqueeze(2), task, torch.zeros_like(task)).contiguous()
    identity = epoch_mod.EpochIdentityPayload(
        shot_key=key,
        scheduled_ordinal=_masked_i64(rows + 601, valid),
        target_generation=_masked_i64(rows + 701, valid),
        selected_cell=_masked_i64(torch.zeros_like(rows), valid),
        candidate_identity=_masked_i64(rows + 801, valid),
    )
    clocks = epoch_mod.EpochClockPayload(
        reveal_tick=_masked_i64(rows + 10, valid),
        contact_tick=_masked_i64(rows + 30, valid),
        launch_tick=_masked_i64(rows + 20, valid),
        deadline_tick=_masked_i64(rows + 40, valid),
        next_reveal_tick=_masked_i64(rows + 50, valid),
    )
    zeros = torch.zeros(shape, dtype=torch.int64, device=device)
    return epoch_mod.ActionEpochD05CandidateProjection(
        identity=identity,
        clocks=clocks,
        task=epoch_mod.EpochTaskPayload(
            task_f32=task, task_valid=valid.contiguous()
        ),
        rng_counter=torch.where(valid, rows + 901, zeros).contiguous(),
        construction_admissible=valid.contiguous(),
        playback_admissible=valid.contiguous(),
        owner_fault_bits=torch.zeros(
            (n, 1, epoch_mod.OWNER_COUNT), dtype=torch.int64, device=device
        ),
    )


def _install_exact_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n: int,
    device: torch.device,
    accept_mask: torch.Tensor,
    candidate_valid: torch.Tensor | None = None,
    playback_admissible: torch.Tensor | None = None,
    mutated_key_field: str | None = None,
):
    epoch_mod = D05._require_action_epoch_module()
    command, _ = bridge._configure_unbound_command(num_envs=n)
    _to_device(command, device)
    _install_frozen_task_frame_sources(command, n, device)
    epoch = epoch_mod.ActionEpochOwner(num_envs=n, device=device)
    token = object.__new__(D05.DeviceR05RowTransaction)
    owner = object.__new__(D05.DeviceR05Owner)
    if candidate_valid is None:
        candidate_valid = accept_mask.clone()
    candidate = _candidate(
        n,
        device,
        candidate_valid=candidate_valid.to(device=device, dtype=torch.bool),
        mutated_key_field=mutated_key_field,
    )
    if playback_admissible is not None:
        candidate = replace(
            candidate,
            playback_admissible=playback_admissible.to(
                device=device, dtype=torch.bool
            ).reshape(n, 1).contiguous(),
        )
    accepted = accept_mask.to(device=device, dtype=torch.bool).contiguous()
    record = types.SimpleNamespace(
        token=token,
        candidate=candidate,
        accept_mask=accepted,
        accepted_consumers=set(),
    )
    owner._num_envs = n
    owner._device = device
    command._action_ball_continuous_motion_device_r05_owner = owner
    command._action_ball_full_mdp_motion_epoch_owner = epoch
    if command._action_ball_continuous_motion_device_mutation_version is None:
        command._action_ball_continuous_motion_device_mutation_version = torch.zeros(
            1, dtype=torch.int64, device=device
        )

    active_calls: list[str] = []

    def require_active(
        self, actual: object, *, owner_kind: str
    ) -> object:
        assert self is epoch and actual is record.token and owner_kind == "motion"
        active_calls.append(owner_kind)
        return epoch_mod.ActionEpochD05AcceptedRows(
            accept_mask=record.accept_mask.reshape(n, 1).clone(),
            publication_ordinal=torch.full(
                (n, 1), 77, dtype=torch.int64, device=device
            ),
        )

    monkeypatch.setattr(
        epoch_mod.ActionEpochOwner,
        "require_active_d05_accepted_rows",
        require_active,
    )

    def require_accepted(
        self, actual: object, *, owner_kind: str
    ) -> D05.DeviceR05AcceptedRowsView:
        if (
            self is not owner
            or actual is not record.token
            or owner_kind != "motion"
            or owner_kind in record.accepted_consumers
        ):
            raise D05.DeviceR05ConflictError(
                "accepted row view is stale or foreign"
            )
        accepted_rows = epoch.require_active_d05_accepted_rows(
            actual, owner_kind="motion"
        )
        record.accepted_consumers.add(owner_kind)
        current_candidate = record.candidate
        mask = accepted_rows.accept_mask

        def masked_i64(value: torch.Tensor) -> torch.Tensor:
            return torch.where(
                mask, value, torch.full_like(value, -1)
            ).contiguous()

        key = epoch_mod.ActionEpochShotKey(
            **{
                field.name: masked_i64(
                    getattr(current_candidate.identity.shot_key, field.name)
                )
                for field in fields(epoch_mod.ActionEpochShotKey)
            }
        )
        identity = epoch_mod.EpochIdentityPayload(
            shot_key=key,
            **{
                field.name: masked_i64(
                    getattr(current_candidate.identity, field.name)
                )
                for field in fields(epoch_mod.EpochIdentityPayload)
                if field.name != "shot_key"
            },
        )
        clocks = epoch_mod.EpochClockPayload(
            **{
                field.name: masked_i64(
                    getattr(current_candidate.clocks, field.name)
                )
                for field in fields(epoch_mod.EpochClockPayload)
            }
        )
        task_mask = mask.unsqueeze(2)
        task = epoch_mod.EpochTaskPayload(
            task_f32=torch.where(
                task_mask,
                current_candidate.task.task_f32,
                torch.zeros_like(current_candidate.task.task_f32),
            ).contiguous(),
            task_valid=(current_candidate.task.task_valid & mask).contiguous(),
        )
        return D05.DeviceR05AcceptedRowsView(
            transaction=record.token,
            publication_ordinal=masked_i64(
                accepted_rows.publication_ordinal
            ),
            target_xy_m=torch.zeros(
                (n, 1, 2), dtype=torch.float32, device=device
            ),
            identity=identity,
            clocks=clocks,
            task=task,
            rng_counter=torch.where(
                mask,
                current_candidate.rng_counter,
                torch.zeros_like(current_candidate.rng_counter),
            ).contiguous(),
            playback_admissible=(
                current_candidate.playback_admissible & mask
            ).contiguous(),
        )

    monkeypatch.setattr(
        D05.DeviceR05Owner,
        "require_owned_action_epoch_accepted",
        require_accepted,
    )
    return command, owner, epoch, token, record, active_calls


class _RealD05AcceptedPeer:
    """Consume one real D05 view without adding another business writer."""

    def __init__(self, owner: D05.DeviceR05Owner, owner_kind: str) -> None:
        self.owner = owner
        self.owner_kind = owner_kind
        self.view = None

    def commit(self, token: object) -> None:
        self.view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind=self.owner_kind
        )


def _install_real_d05_record(
    *,
    device: torch.device,
    invalid_key_field: str | None = None,
    task_valid: bool = True,
    construction_admissible: bool = True,
    corrupt_accept_mask: bool = True,
):
    """Arm real D05/Epoch internals, then corrupt only D05's private mask."""

    epoch_mod = D05._require_action_epoch_module()
    n = 2
    command, _ = bridge._configure_unbound_command(num_envs=n)
    _to_device(command, device)
    _install_frozen_task_frame_sources(command, n, device)
    epoch_owner = epoch_mod.ActionEpochOwner(num_envs=n, device=device)
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(n, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(n, dtype=torch.int64, device=device),
    )

    candidate = _candidate(
        n,
        device,
        candidate_valid=torch.ones(n, dtype=torch.bool, device=device),
        mutated_key_field=invalid_key_field,
    )
    assert C._ACTION_BALL_ROW_IDENTITY is epoch_mod.row_identity
    command._action_ball_reset_generation.copy_(
        candidate.identity.shot_key.reset_generation[:, 0]
    )
    command._action_ball_swing_generation.copy_(
        candidate.identity.shot_key.ball_generation[:, 0]
    )
    command.clip_id.copy_(candidate.identity.shot_key.action_slot[:, 0])
    candidate.task.task_valid[0, 0] = task_valid
    candidate.construction_admissible[0, 0] = construction_admissible

    due_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    admitted = torch.tensor(
        [construction_admissible, False], dtype=torch.bool, device=device
    )
    prepared = types.SimpleNamespace(
        owner_fault_free=torch.ones(n, dtype=torch.bool, device=device),
        admissible=admitted,
        projection=types.SimpleNamespace(
            ready_at_reveal=torch.ones(n, dtype=torch.bool, device=device)
        ),
        selected_target_xy_m=torch.arange(
            n * 2, dtype=torch.float32, device=device
        ).reshape(n, 2),
    )
    rows = epoch_mod.ActionEpochDueRows(
        common_step=1,
        due_mask=due_mask,
        construct_mask=due_mask.clone(),
    )
    token = object.__new__(D05.DeviceR05RowTransaction)
    owner = object.__new__(D05.DeviceR05Owner)
    owner._num_envs = n
    owner._device = device
    owner._action_epoch_candidate = lambda _prepared: candidate
    record = owner._build_row_transaction(
        token,
        rows,
        prepared,
        types.SimpleNamespace(prepared=prepared),
    )
    if corrupt_accept_mask:
        assert not bool(record.accept_mask[0])
        record.accept_mask[0] = True
    else:
        assert bool(record.accept_mask[0])
    record.stage = "settling"
    owner._row_transaction_records = {token: record}
    owner._active_row_transaction = token
    owner._diagnostic_epoch_owner = epoch_owner

    command._action_ball_continuous_motion_device_r05_owner = owner
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    if command._action_ball_continuous_motion_device_mutation_version is None:
        command._action_ball_continuous_motion_device_mutation_version = torch.zeros(
            1, dtype=torch.int64, device=device
        )

    racket_peer = _RealD05AcceptedPeer(owner, "racket")
    physical_peer = _RealD05AcceptedPeer(owner, "physical_ball")
    epoch_owner._active_d05 = epoch_mod._ActiveD05Transaction(
        rows=rows,
        publication_ordinal=17,
        base_version=epoch_owner.current().version,
    )
    epoch_owner._d05_owner = owner
    epoch_owner._d05_candidate_projector = (
        owner.require_owned_action_epoch_candidate
    )
    epoch_owner._d05_accept_writers = (
        command.commit_action_ball_full_mdp_motion_epoch_rows,
        racket_peer.commit,
        physical_peer.commit,
    )
    return (
        command,
        owner,
        epoch_owner,
        token,
        record,
        racket_peer,
        physical_peer,
    )


def _arm_next_real_d05_record(
    command,
    owner: D05.DeviceR05Owner,
    epoch_owner,
    candidate: object,
):
    """Arm a later exact row transaction on the same owner and slot 0."""

    epoch_mod = D05._require_action_epoch_module()
    n = command.num_envs
    device = torch.device(command.device)
    due_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    prepared = types.SimpleNamespace(
        owner_fault_free=torch.ones(n, dtype=torch.bool, device=device),
        admissible=torch.tensor([True, False], dtype=torch.bool, device=device),
        projection=types.SimpleNamespace(
            ready_at_reveal=torch.ones(n, dtype=torch.bool, device=device)
        ),
        selected_target_xy_m=torch.arange(
            n * 2, dtype=torch.float32, device=device
        ).reshape(n, 2),
    )
    rows = epoch_mod.ActionEpochDueRows(2, due_mask, due_mask.clone())
    token = object.__new__(D05.DeviceR05RowTransaction)
    owner._action_epoch_candidate = lambda _prepared: candidate
    record = owner._build_row_transaction(
        token,
        rows,
        prepared,
        types.SimpleNamespace(prepared=prepared),
    )
    assert record.accept_mask.tolist() == [True, False]
    record.stage = "settling"
    owner._row_transaction_records[token] = record
    owner._active_row_transaction = token
    command._action_ball_reset_generation.copy_(
        candidate.identity.shot_key.reset_generation[:, 0]
    )
    command._action_ball_swing_generation.copy_(
        candidate.identity.shot_key.ball_generation[:, 0]
    )
    command.clip_id.copy_(candidate.identity.shot_key.action_slot[:, 0])
    racket_peer = _RealD05AcceptedPeer(owner, "racket")
    physical_peer = _RealD05AcceptedPeer(owner, "physical_ball")
    epoch_owner._active_d05 = epoch_mod._ActiveD05Transaction(
        rows=rows,
        publication_ordinal=18,
        base_version=epoch_owner.current().version,
    )
    epoch_owner._d05_candidate_projector = (
        owner.require_owned_action_epoch_candidate
    )
    epoch_owner._d05_accept_writers = (
        command.commit_action_ball_full_mdp_motion_epoch_rows,
        racket_peer.commit,
        physical_peer.commit,
    )
    return token, record


def _bytes(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8).clone()


def _snapshot(command, *, row: int | None = None) -> dict[str, torch.Tensor]:
    result = {}
    for name in WRITTEN_TENSORS:
        value = getattr(command, name)
        result[name] = _bytes(value if row is None else value[row])
    return result


def _seed_peer_payloads(command, row: int) -> None:
    f32_payloads = torch.tensor(
        [2143294004, -2147483648],
        dtype=torch.int32,
        device=command.device,
    ).view(torch.float32)
    f64_payloads = torch.tensor(
        [9221120237041095220, -9223372036854775808],
        dtype=torch.int64,
        device=command.device,
    ).view(torch.float64)
    float_names = [
        name
        for name in WRITTEN_TENSORS
        if getattr(command, name).is_floating_point()
    ]
    for ordinal, name in enumerate(float_names):
        destination = getattr(command, name)
        payloads = f32_payloads if destination.dtype is torch.float32 else f64_payloads
        destination[row] = payloads[ordinal % 2]
    command._action_ball_continuous_canonical_task_identity[row] = 987654321
    command._action_ball_continuous_canonical_cadence_identity[row] = 123456789


def _make_teacher_start_reachable(command) -> None:
    command._action_ball_continuous_canonical_task_valid.fill_(True)
    command._action_ball_task_timing_active.fill_(True)
    command._action_ball_continuous_motion_active.fill_(True)
    command._action_ball_continuous_canonical_playback_started.zero_()
    command._action_ball_continuous_canonical_phase.fill_(
        C.ACTION_BALL_CONTINUOUS_CANONICAL_PREPARE_VISIBLE
    )
    command._action_ball_pre_swing_wait_s.zero_()
    command._action_ball_task_age_s.fill_(1.0)
    command.time_steps.copy_(command.motion.seg_start[command.clip_id] + 1)


def _advance_canonical_lifecycle_once(command) -> None:
    empty = torch.zeros(
        command.num_envs, dtype=torch.bool, device=torch.device(command.device)
    )
    command._advance_action_ball_continuous_canonical_lifecycle(
        motion_active_before=command._action_ball_continuous_motion_active.clone(),
        suffix_due=empty,
        closed_without_playback=empty.clone(),
    )


def _install_frame0_reference_contract(
    monkeypatch: pytest.MonkeyPatch,
    command,
) -> None:
    upper = (
        "torso_Link",
        "left_shoulder_roll_Link",
        "left_elbow_Link",
        "left_wrist_yaw_Link",
        "right_shoulder_roll_Link",
        "right_elbow_Link",
        "right_wrist_yaw_Link",
    )
    command.robot.body_names = ["root_Link", *upper]
    command.cfg.body_names = list(command.robot.body_names)
    command.motion.body_pos_w = command.motion.body_pos_w.repeat(1, 8, 1)
    command.motion.body_quat_w = command.motion.body_quat_w.repeat(1, 8, 1)
    uid = command._action_ball_action_uids[0]
    command.clip_id.zero_()
    command._action_ball_continuous_schedule_projection = types.MappingProxyType(
        {"upcoming_action_slot": 0, "upcoming_action_uid": uid}
    )
    robots = types.ModuleType("whole_body_tracking.robots")
    agibot_a3 = types.ModuleType("whole_body_tracking.robots.agibot_a3")
    agibot_a3.A3_UPPER_TRACKED = list(upper)
    robots.agibot_a3 = agibot_a3
    monkeypatch.setitem(sys.modules, "whole_body_tracking.robots", robots)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.robots.agibot_a3", agibot_a3
    )
    monkeypatch.setattr(
        sys.modules["whole_body_tracking"], "robots", robots, raising=False
    )


def _assert_reference_key_row(
    reference,
    expected,
    *,
    row: int,
) -> None:
    for field in fields(C._ACTION_BALL_ROW_IDENTITY.ActionEpochShotKey):
        assert torch.equal(
            getattr(reference.shot_key, field.name)[row],
            getattr(expected, field.name)[row, 0],
        ), field.name


@pytest.mark.parametrize("device", DEVICES)
def test_r07_bootstrap_reference_has_neutral_full_key_without_scalar_epoch(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> None:
    epoch_mod = D05._require_action_epoch_module()
    command, _ = bridge._configure_unbound_command(num_envs=2)
    _to_device(command, device)
    epoch_owner = epoch_mod.ActionEpochOwner(num_envs=2, device=device)
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(2, dtype=torch.int64, device=device),
    )
    command._action_ball_full_mdp_motion_epoch_owner = epoch_owner
    _install_frame0_reference_contract(monkeypatch, command)

    reference = command.project_action_ball_full_mdp_recovery_ready_reference()

    assert not hasattr(reference, "epoch")
    assert reference.epoch_version == epoch_owner.current().version
    assert reference.reference_kind.tolist() == [1, 1]
    assert reference.reference_action_slot.tolist() == [0, 0]
    assert reference.reference_action_uid.tolist() == [
        command._action_ball_action_uids[0],
        command._action_ball_action_uids[0],
    ]
    assert reference.validity.tolist() == [True, True]
    for field in fields(C._ACTION_BALL_ROW_IDENTITY.ActionEpochShotKey):
        value = getattr(reference.shot_key, field.name)
        assert value.dtype is torch.int64
        assert tuple(value.shape) == (2,)
        assert value.is_contiguous()
        assert value.tolist() == [-1, -1]

    reference.shot_key.shot_index.fill_(999)
    replay = command.project_action_ball_full_mdp_recovery_ready_reference()
    assert replay.shot_key.shot_index.tolist() == [-1, -1]


@pytest.mark.parametrize("device", DEVICES)
def test_r07_completed_reference_tracks_same_slot_uid_next_shot_full_key(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> None:
    (
        command,
        owner,
        epoch_owner,
        first_token,
        first_record,
        _racket_peer,
        _physical_peer,
    ) = _install_real_d05_record(
        device=device,
        corrupt_accept_mask=False,
    )
    first_record.candidate.identity.shot_key.action_uid[0, 0] = (
        command._action_ball_action_uids[0]
    )
    _install_frame0_reference_contract(monkeypatch, command)
    epoch_owner.settle_d05_transaction(first_token)
    first_public = epoch_owner.current().identity.shot_key.clone()
    first = command.project_action_ball_full_mdp_recovery_ready_reference()
    assert first.reference_kind.tolist() == [2, 1]
    assert first.validity.tolist() == [True, True]
    _assert_reference_key_row(first, first_public, row=0)
    # A completed R07 row consumes the exact same accepted non-identity frame
    # for root and every tracked body; bootstrap row 1 remains source-frame.
    angle = torch.tensor(torch.pi / 2, device=device)
    command._action_ball_full_mdp_task_yaw_wxyz[0] = torch.tensor(
        [torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)], device=device
    )
    command._action_ball_full_mdp_task_translation_w[0] = torch.tensor(
        [0.3, -0.2, 0.0], device=device
    )
    transformed = command.project_action_ball_full_mdp_recovery_ready_reference()
    q = command._action_ball_full_mdp_task_yaw_wxyz[0:1]
    t = command._action_ball_full_mdp_task_translation_w[0]
    origin = command._env.scene.env_origins[0]
    start = command.motion.seg_start[0]
    raw_root = command.motion.body_pos_w[start, 0]
    raw_root_q = command.motion.body_quat_w[start, 0:1]
    upper = tuple(sys.modules["whole_body_tracking.robots.agibot_a3"].A3_UPPER_TRACKED)
    slots = [tuple(command.cfg.body_names).index(name) for name in upper]
    raw_body = command.motion.body_pos_w[start, slots]
    raw_body_q = command.motion.body_quat_w[start, slots]
    assert torch.allclose(
        transformed.root_position_m[0],
        C.quat_apply(q, raw_root[None])[0] + t + origin,
    )
    assert torch.allclose(
        transformed.root_orientation_wxyz[0],
        C.quat_mul(q, raw_root_q)[0],
    )
    body_q = q[:, None, :].expand_as(first.body_orientation_wxyz[0:1])
    assert torch.allclose(
        transformed.body_position_m[0],
        C.quat_apply(body_q, raw_body[None])[0] + t + origin,
    )
    assert torch.allclose(
        transformed.body_orientation_wxyz[0],
        C.quat_mul(body_q, raw_body_q[None])[0],
    )
    for field in fields(C._ACTION_BALL_ROW_IDENTITY.ActionEpochShotKey):
        assert getattr(first.shot_key, field.name)[1] == -1

    second = _candidate(
        2,
        device,
        candidate_valid=torch.ones(2, dtype=torch.bool, device=device),
    )
    second.identity.shot_key.action_uid[:, 0] = (
        command._action_ball_action_uids[0]
    )
    for name in (
        "ball_generation",
        "shot_index",
        "task_identity",
        "outcome_identity",
        "ball_identity",
    ):
        getattr(second.identity.shot_key, name).add_(1000)
    second.identity.candidate_identity.add_(1000)
    second_token, _second_record = _arm_next_real_d05_record(
        command, owner, epoch_owner, second
    )
    epoch_owner.settle_d05_transaction(second_token)
    second_public = epoch_owner.current().identity.shot_key.clone()

    next_reference = command.project_action_ball_full_mdp_recovery_ready_reference()

    assert epoch_owner.current().epoch == -1
    assert next_reference.reference_kind.tolist() == [2, 1]
    assert torch.equal(
        next_reference.reference_action_slot,
        first.reference_action_slot,
    )
    assert torch.equal(
        next_reference.reference_action_uid,
        first.reference_action_uid,
    )
    _assert_reference_key_row(next_reference, second_public, row=0)
    for name in (
        "ball_generation",
        "shot_index",
        "task_identity",
        "outcome_identity",
        "ball_identity",
    ):
        assert getattr(next_reference.shot_key, name)[0] != getattr(
            first.shot_key, name
        )[0]
    for field in fields(C._ACTION_BALL_ROW_IDENTITY.ActionEpochShotKey):
        assert getattr(next_reference.shot_key, field.name)[1] == -1


@pytest.mark.parametrize("device", DEVICES)
def test_partial_accept_writes_only_exact_row_and_preserves_peer_bytes(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> None:
    command, _owner, _epoch, token, record, active_calls = _install_exact_sources(
        monkeypatch,
        n=2,
        device=device,
        accept_mask=torch.tensor([True, False]),
    )
    _seed_peer_payloads(command, 1)
    peer_before = _snapshot(command, row=1)
    host_version = command._action_ball_continuous_motion_mutation_version
    device_version = _bytes(
        command._action_ball_continuous_motion_device_mutation_version
    )
    checkpoint_debt = (
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack
    )

    command.commit_action_ball_full_mdp_motion_epoch_rows(token)

    assert active_calls == ["motion"]
    assert record.accepted_consumers == {"motion"}
    assert torch.equal(
        command._action_ball_time_to_contact_s[0],
        torch.tensor(0.8, dtype=torch.float32, device=device),
    )
    assert command._action_ball_continuous_canonical_task_identity[0] == 301
    assert command._action_ball_continuous_canonical_cadence_identity[0] == 501
    assert command._action_ball_continuous_canonical_chosen_horizon_tick[0] == 10
    task_yaw = command._action_ball_full_mdp_task_yaw_wxyz[0]
    source_yaw = command._action_ball_full_mdp_source_strike_yaw_wxyz[0]
    frozen_yaw = bridge.C.yaw_quat(
        command._action_ball_full_mdp_frozen_root_quat_wxyz[0:1]
    )[0]
    source_yaw_inverse = source_yaw.clone()
    source_yaw_inverse[1:].neg_()
    expected_task_yaw = bridge.C.quat_mul(
        frozen_yaw[None, :], source_yaw_inverse[None, :]
    )[0]
    # The non-commutative order is fixed: installed physical yaw times the
    # inverse source-strike yaw.  A reversed multiply or identity fallback is
    # therefore observable even when both source values are valid quaternions.
    torch.testing.assert_close(
        task_yaw,
        expected_task_yaw,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        task_yaw,
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
    )
    source_root = torch.tensor([[0.25, -0.15, 0.0]], device=device)
    mapped_source_root = bridge.C.quat_apply(task_yaw[None, :], source_root)[0]
    expected_translation = torch.tensor([1.25, -0.35, 0.0], device=device)
    expected_translation[:2] -= mapped_source_root[:2]
    torch.testing.assert_close(
        command._action_ball_full_mdp_task_translation_w[0],
        expected_translation,
        rtol=0.0,
        atol=1.0e-7,
    )
    peer_after = _snapshot(command, row=1)
    assert peer_before.keys() == peer_after.keys()
    assert all(torch.equal(peer_before[name], peer_after[name]) for name in peer_before)
    assert command._action_ball_continuous_motion_mutation_version == host_version
    assert torch.equal(
        _bytes(command._action_ball_continuous_motion_device_mutation_version),
        device_version,
    )
    assert (
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack
        is checkpoint_debt
    )


@pytest.mark.parametrize("device", DEVICES)
def test_not_ready_accept_installs_task_without_starting_teacher_or_launch(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> None:
    command, _owner, _epoch, token, _record, _calls = _install_exact_sources(
        monkeypatch,
        n=2,
        device=device,
        accept_mask=torch.tensor([True, False]),
        playback_admissible=torch.tensor([False, False]),
    )

    command.commit_action_ball_full_mdp_motion_epoch_rows(token)

    assert command._action_ball_continuous_task_committed.tolist() == [True, False]
    assert command._action_ball_continuous_canonical_task_valid.tolist() == [True, False]
    assert command._action_ball_continuous_current_policy_opportunity.tolist() == [True, False]
    assert command._action_ball_continuous_motion_active.tolist() == [False, False]
    assert command._action_ball_continuous_ready_reference_active.tolist() == [True, False]
    assert command._action_ball_continuous_canonical_playback_started.tolist() == [False, False]
    assert command._action_ball_continuous_phase[0].item() == (
        C._ACTION_BALL_CONTINUOUS_MOTION_PHASE_CODE["recovery_unavailable"]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_all_zero_n64_is_exact_byte_noop_but_consumes_callback(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> None:
    command, _owner, _epoch, token, record, active_calls = _install_exact_sources(
        monkeypatch,
        n=64,
        device=device,
        accept_mask=torch.zeros(64, dtype=torch.bool),
    )
    _seed_peer_payloads(command, 17)
    before = _snapshot(command)
    host_accounting = (
        command._action_ball_continuous_motion_mutation_version,
        _bytes(command._action_ball_continuous_motion_device_mutation_version),
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack,
        command._action_ball_continuous_motion_global_drain_sequence,
        command._action_ball_continuous_motion_global_drain_last_update,
        command._action_ball_continuous_motion_global_drain_last_completed_steps,
        command._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version,
    )

    command.commit_action_ball_full_mdp_motion_epoch_rows(token)

    after = _snapshot(command)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert active_calls == ["motion"]
    assert record.accepted_consumers == {"motion"}
    assert command._action_ball_continuous_motion_mutation_version == host_accounting[0]
    assert torch.equal(
        _bytes(command._action_ball_continuous_motion_device_mutation_version),
        host_accounting[1],
    )
    assert (
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack
        is host_accounting[2]
    )
    assert (
        command._action_ball_continuous_motion_global_drain_sequence,
        command._action_ball_continuous_motion_global_drain_last_update,
        command._action_ball_continuous_motion_global_drain_last_completed_steps,
        command._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version,
    ) == host_accounting[3:]


@pytest.mark.parametrize("n", (1, 2, 64))
def test_n1_n2_n64_use_same_typed_token_only_surface(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    command, _owner, _epoch, token, record, _calls = _install_exact_sources(
        monkeypatch,
        n=n,
        device=torch.device("cpu"),
        accept_mask=torch.ones(n, dtype=torch.bool),
    )
    command.commit_action_ball_full_mdp_motion_epoch_rows(token)
    assert type(token) is D05.DeviceR05RowTransaction
    assert record.accepted_consumers == {"motion"}
    assert tuple(command._action_ball_continuous_canonical_action_uid.shape) == (n,)


def test_same_slot0_action_uid_accepts_next_shot_with_different_full_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    command, _owner, _epoch, first_token, record, _calls = (
        _install_exact_sources(
            monkeypatch,
            n=1,
            device=device,
            accept_mask=torch.ones(1, dtype=torch.bool),
        )
    )
    command.commit_action_ball_full_mdp_motion_epoch_rows(first_token)
    assert command._action_ball_continuous_canonical_task_valid.tolist() == [True]
    assert command.action_ball_task_timing_active.tolist() == [True]
    first_uid = command._action_ball_continuous_canonical_action_uid.clone()
    first_task = command._action_ball_continuous_canonical_task_identity.clone()
    first_ball = command._action_ball_continuous_canonical_cadence_identity.clone()
    first_candidate = (
        command._action_ball_continuous_canonical_candidate_identity.clone()
    )

    # Natural close owns the prior receipt and timing teardown.  The next D05
    # must install both facts together for the replacement task.
    command._action_ball_continuous_canonical_task_valid.zero_()
    command._action_ball_task_timing_active.zero_()
    assert command._action_ball_continuous_canonical_task_valid.tolist() == [False]
    assert command.action_ball_task_timing_active.tolist() == [False]

    second = _candidate(
        1,
        device,
        candidate_valid=torch.ones(1, dtype=torch.bool),
    )
    for field_name in (
        "shot_index",
        "task_identity",
        "outcome_identity",
        "ball_identity",
    ):
        getattr(second.identity.shot_key, field_name).add_(1000)
    second.identity.candidate_identity.add_(1000)
    second_token = object.__new__(D05.DeviceR05RowTransaction)
    record.token = second_token
    record.candidate = second
    record.accepted_consumers.clear()

    command.commit_action_ball_full_mdp_motion_epoch_rows(second_token)

    assert command._action_ball_continuous_canonical_task_valid.tolist() == [True]
    assert command.action_ball_task_timing_active.tolist() == [True]
    assert command.clip_id.tolist() == [0]
    assert torch.equal(
        command._action_ball_continuous_canonical_action_uid, first_uid
    )
    assert torch.equal(
        command._action_ball_continuous_canonical_task_identity,
        first_task + 1000,
    )
    assert torch.equal(
        command._action_ball_continuous_canonical_cadence_identity,
        first_ball + 1000,
    )
    assert torch.equal(
        command._action_ball_continuous_canonical_candidate_identity,
        first_candidate + 1000,
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "reset_generation",
        "ball_generation",
        "action_uid",
        "action_slot",
        "shot_index",
        "task_identity",
        "outcome_identity",
        "ball_identity",
    ),
)
def test_every_invalid_full_key_field_is_neutralized_before_motion_write(
    monkeypatch: pytest.MonkeyPatch, field_name: str
) -> None:
    command, owner, _epoch, token, _record, _calls = _install_exact_sources(
        monkeypatch,
        n=2,
        device=torch.device("cpu"),
        accept_mask=torch.tensor([False, True]),
        candidate_valid=torch.ones(2, dtype=torch.bool),
        mutated_key_field=field_name,
    )
    _seed_peer_payloads(command, 0)
    peer_before = _snapshot(command, row=0)
    view = owner.require_owned_action_epoch_accepted(token, owner_kind="motion")
    for field in fields(type(view.identity.shot_key)):
        assert getattr(view.identity.shot_key, field.name)[0, 0] == -1
    assert not view.task.task_valid[0, 0]
    assert torch.equal(
        view.task.task_f32[0], torch.zeros_like(view.task.task_f32[0])
    )
    peer_after = _snapshot(command, row=0)
    assert peer_before.keys() == peer_after.keys()
    assert all(
        torch.equal(peer_before[name], peer_after[name]) for name in peer_before
    )


@pytest.mark.parametrize("device", DEVICES)
def test_real_d05_corrupt_accept_cannot_authorize_invalid_key_row(
    device: torch.device,
) -> None:
    (
        command,
        _owner,
        epoch_owner,
        token,
        record,
        racket_peer,
        physical_peer,
    ) = _install_real_d05_record(
        device=device,
        invalid_key_field="ball_identity",
    )
    _seed_peer_payloads(command, 0)
    before = _snapshot(command)
    accounting_before = (
        command._action_ball_continuous_motion_mutation_version,
        _bytes(command._action_ball_continuous_motion_device_mutation_version),
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack,
        command._action_ball_continuous_motion_global_drain_sequence,
        command._action_ball_continuous_motion_global_drain_last_update,
        command._action_ball_continuous_motion_global_drain_last_completed_steps,
        command._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version,
    )

    epoch_owner.settle_d05_transaction(token)
    settled = epoch_owner.current()

    after = _snapshot(command)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert record.accept_mask.tolist() == [True, False]
    assert record.accepted_consumers == {"motion", "racket", "physical_ball"}
    assert racket_peer.view is not None and physical_peer.view is not None
    assert not command._action_ball_continuous_motion_poisoned
    assert not epoch_owner.poisoned
    assert (
        command._action_ball_continuous_motion_mutation_version,
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack,
        command._action_ball_continuous_motion_global_drain_sequence,
        command._action_ball_continuous_motion_global_drain_last_update,
        command._action_ball_continuous_motion_global_drain_last_completed_steps,
        command._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version,
    ) == (
        accounting_before[0],
        accounting_before[2],
        accounting_before[3],
        accounting_before[4],
        accounting_before[5],
        accounting_before[6],
    )
    assert torch.equal(
        _bytes(command._action_ball_continuous_motion_device_mutation_version),
        accounting_before[1],
    )

    log = epoch_owner._publication.pending_log
    settled_entry = next(
        entry for entry in log if entry.transition == "D05_SETTLED"
    )
    decision = settled_entry.delta.values[
        settled_entry.delta.names.index("decision")
    ]
    assert decision[:, 0].tolist() == [
        D05._require_action_epoch_module().D05_DECISION_CENSOR,
        D05._require_action_epoch_module().D05_DECISION_NONE,
    ]
    committed = next(
        entry
        for entry in log
        if entry.transition == "WRITES_COMMITTED:motion"
    )
    event_mask = committed.delta.values[
        committed.delta.names.index("event_mask")
    ]
    assert not bool(event_mask.any())
    assert not bool(
        settled.writes_committed[
            :, :, D05._require_action_epoch_module().OWNER_ORDER.index("motion")
        ].any()
    )
    assert settled.phase[:, 0].tolist() == [
        D05._require_action_epoch_module().PHASE_IDLE,
        D05._require_action_epoch_module().PHASE_IDLE,
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_real_d05_corrupt_accept_task_invalid_row_is_motion_byte_noop(
    device: torch.device,
) -> None:
    (
        command,
        _owner,
        epoch_owner,
        token,
        record,
        _racket_peer,
        _physical_peer,
    ) = _install_real_d05_record(
        device=device,
        task_valid=False,
        construction_admissible=False,
    )
    _seed_peer_payloads(command, 0)
    before = _snapshot(command)
    device_version = _bytes(
        command._action_ball_continuous_motion_device_mutation_version
    )
    host_version = command._action_ball_continuous_motion_mutation_version

    epoch_owner.settle_d05_transaction(token)
    settled = epoch_owner.current()

    after = _snapshot(command)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert record.accept_mask.tolist() == [True, False]
    assert command._action_ball_continuous_motion_mutation_version == host_version
    assert torch.equal(
        _bytes(command._action_ball_continuous_motion_device_mutation_version),
        device_version,
    )
    settled_entry = next(
        entry
        for entry in epoch_owner._publication.pending_log
        if entry.transition == "D05_SETTLED"
    )
    decision = settled_entry.delta.values[
        settled_entry.delta.names.index("decision")
    ]
    assert decision[:, 0].tolist() == [
        D05._require_action_epoch_module().D05_DECISION_REJECT,
        D05._require_action_epoch_module().D05_DECISION_NONE,
    ]
    assert not bool(settled.task.task_valid.any())


@pytest.mark.parametrize("device", DEVICES)
def test_real_motion_callpoint_publishes_active_full_key_while_epoch_stays_minus_one(
    device: torch.device,
) -> None:
    (
        command,
        _owner,
        epoch_owner,
        token,
        _record,
        _racket_peer,
        _physical_peer,
    ) = _install_real_d05_record(
        device=device,
        corrupt_accept_mask=False,
    )
    epoch_owner.settle_d05_transaction(token)
    assert epoch_owner.current().epoch == -1
    _make_teacher_start_reachable(command)

    _advance_canonical_lifecycle_once(command)

    current = epoch_owner.current()
    assert current.motion_playback_started[:, 0].tolist() == [True, False]
    assert command._action_ball_continuous_canonical_playback_started.tolist() == [
        True,
        False,
    ]
    entry = epoch_owner._publication.pending_log[-1]
    assert entry.transition == D05._require_action_epoch_module().MOTION_PLAYBACK_STARTED
    event_mask = entry.delta.values[entry.delta.names.index("event_mask")]
    assert event_mask[:, 0].tolist() == [True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_real_motion_callpoint_genesis_without_active_key_is_empty_not_authority(
    device: torch.device,
) -> None:
    epoch_mod = D05._require_action_epoch_module()
    command, _ = bridge._configure_unbound_command(num_envs=2)
    _to_device(command, device)
    command._action_ball_continuous_motion_device_r05_owner = object.__new__(
        D05.DeviceR05Owner
    )
    epoch_owner = epoch_mod.ActionEpochOwner(num_envs=2, device=device)
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(2, dtype=torch.int64, device=device),
    )
    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
    _make_teacher_start_reachable(command)

    _advance_canonical_lifecycle_once(command)

    current = epoch_owner.current()
    assert current.epoch == -1
    assert not bool(current.motion_playback_started.any())
    assert not bool(
        command._action_ball_continuous_canonical_playback_started.any()
    )
    entry = epoch_owner._publication.pending_log[-1]
    event_mask = entry.delta.values[entry.delta.names.index("event_mask")]
    assert not bool(event_mask.any())
    assert not epoch_owner.poisoned


@pytest.mark.parametrize("device", DEVICES)
def test_same_slot_uid_next_shot_rejects_stale_motion_key_then_publishes_exact_key(
    device: torch.device,
) -> None:
    epoch_mod = D05._require_action_epoch_module()
    (
        command,
        owner,
        epoch_owner,
        first_token,
        _record,
        _racket_peer,
        _physical_peer,
    ) = _install_real_d05_record(
        device=device,
        corrupt_accept_mask=False,
    )
    epoch_owner.settle_d05_transaction(first_token)
    first_uid = command._action_ball_continuous_canonical_action_uid.clone()
    first_slot = command.clip_id.clone()
    first_shot = command._action_ball_continuous_canonical_shot_index.clone()

    second = _candidate(
        2,
        device,
        candidate_valid=torch.ones(2, dtype=torch.bool, device=device),
    )
    for name in (
        "ball_generation",
        "shot_index",
        "task_identity",
        "outcome_identity",
        "ball_identity",
    ):
        getattr(second.identity.shot_key, name).add_(1000)
    second.identity.candidate_identity.add_(1000)
    peer_before_second = _snapshot(command, row=1)
    second_token, _second_record = _arm_next_real_d05_record(
        command, owner, epoch_owner, second
    )
    epoch_owner.settle_d05_transaction(second_token)
    second_shot = second.identity.shot_key.shot_index[:, 0]
    assert torch.equal(
        command._action_ball_continuous_canonical_action_uid, first_uid
    )
    assert torch.equal(command.clip_id, first_slot)
    assert command._action_ball_continuous_canonical_shot_index[0] == second_shot[0]
    second_retained_shot = (
        command._action_ball_continuous_canonical_shot_index.clone()
    )
    peer_after_second = _snapshot(command, row=1)
    assert all(
        torch.equal(peer_before_second[name], peer_after_second[name])
        for name in peer_before_second
    )

    # Simulate stale Motion state A after Epoch has already installed shot B.
    command._action_ball_continuous_canonical_shot_index.copy_(first_shot)
    _make_teacher_start_reachable(command)
    _advance_canonical_lifecycle_once(command)
    assert not bool(epoch_owner.current().motion_playback_started.any())
    assert not bool(
        command._action_ball_continuous_canonical_playback_started.any()
    )
    stale_entry = epoch_owner._publication.pending_log[-1]
    stale_mask = stale_entry.delta.values[
        stale_entry.delta.names.index("event_mask")
    ]
    assert not bool(stale_mask.any())
    assert not epoch_owner.poisoned

    # Restoring B's exact full key makes the same owner-derived edge reachable.
    command._action_ball_continuous_canonical_shot_index.copy_(
        second_retained_shot
    )
    _advance_canonical_lifecycle_once(command)
    assert epoch_owner.current().motion_playback_started[:, 0].tolist() == [
        True,
        False,
    ]
    exact_entry = epoch_owner._publication.pending_log[-1]
    exact_mask = exact_entry.delta.values[
        exact_entry.delta.names.index("event_mask")
    ]
    exact_shot = exact_entry.delta.values[
        exact_entry.delta.names.index("shot_key.shot_index")
    ]
    assert exact_mask[:, 0].tolist() == [True, False]
    assert exact_shot[0, 0] == second_shot[0]
    assert exact_shot[1, 0] == -1


def test_playback_path_cold_pins_the_epoch_row_identity_module() -> None:
    epoch_mod = D05._require_action_epoch_module()
    assert C._ACTION_BALL_ROW_IDENTITY is epoch_mod.row_identity
    source = inspect.getsource(C.MotionCommand.action_epoch_playback_transition_mask)
    assert "import action_ball_full_mdp_row_identity" not in source
    assert "torch.arange" not in source
    assert "torch.clamp" not in source
    assert "projection.phase," in source
    assert "projection.selected_mask," in source
    assert "slots.eq(0)" in source
    assert "epoch_owner.current(" not in source
    assert ".version" not in source
    assert "action_epoch_shot_key_valid(public_key)" in source
    assert "action_epoch_shot_key_valid(retained_key)" in source
    assert "action_epoch_shot_key_equal(public_key, retained_key)" in source
    lifecycle = inspect.getsource(
        C.MotionCommand._advance_action_ball_continuous_canonical_lifecycle
    )
    assert "current().epoch" not in lifecycle


def test_motion_epoch_cold_bind_uses_current_sole_mask_factory_abi() -> None:
    epoch_mod = D05._require_action_epoch_module()
    command, _ = bridge._configure_unbound_command(num_envs=2)
    command._action_ball_continuous_motion_device_r05_owner = object.__new__(
        D05.DeviceR05Owner
    )
    epoch_owner = epoch_mod.ActionEpochOwner(
        num_envs=2, device=torch.device("cpu")
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )

    command.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)

    assert command._action_ball_full_mdp_motion_epoch_owner is epoch_owner
    assert epoch_owner._motion_playback.__self__ is command
    binder = inspect.getsource(
        C.MotionCommand.bind_action_ball_full_mdp_motion_epoch_owner
    )
    factory = inspect.getsource(
        D05.construct_action_ball_full_mdp_device_r05
    )
    assert "require_active_d05_accepted_rows" in binder
    assert "bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)" in factory
    assert "num_envs = epoch_owner.num_envs" in factory
    assert "num_envs=2" not in factory
    assert "require_active_d05_" + "writer" not in binder
    assert "require_active_d05_" + "publication_ordinal" not in binder


def test_r07_reference_surface_has_no_scalar_epoch_or_hot_host_verdict() -> None:
    assert tuple(
        field.name
        for field in fields(C.ActionBallFullMdpCompletedActionFrame0Reference)
    ) == (
        "motion_owner",
        "epoch_owner",
        "epoch_version",
        "cadence_tick",
        "shot_key",
        "reference_kind",
        "reference_action_slot",
        "reference_action_uid",
        "root_position_m",
        "root_orientation_wxyz",
        "joint_position_rad",
        "body_position_m",
        "body_orientation_wxyz",
        "station_anchor_xy_m",
        "validity",
        "producer_fault_bits",
    )
    source = inspect.getsource(
        C.MotionCommand.project_action_ball_full_mdp_recovery_ready_reference
    )
    for forbidden in (
        "record.epoch",
        "PHASE_REJECTED",
        "PHASE_DEFERRED",
        ".item(",
        ".cpu(",
        ".numpy(",
        ".tolist(",
        "bool(",
        "torch._assert_async",
        "torch.arange",
        "nonzero",
        "masked_select",
    ):
        assert forbidden not in source
    assert "recovery_reference_lifecycle_masks()" in source
    assert "require_action_epoch_shot_key" in source
    assert "empty_action_epoch_shot_key" in source


def test_writer_replay_and_foreign_token_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, _owner, _epoch, token, _record, _calls = _install_exact_sources(
        monkeypatch,
        n=2,
        device=torch.device("cpu"),
        accept_mask=torch.tensor([True, False]),
    )
    command.commit_action_ball_full_mdp_motion_epoch_rows(token)
    with pytest.raises(D05.DeviceR05ConflictError, match="stale|foreign"):
        command.commit_action_ball_full_mdp_motion_epoch_rows(token)
    assert command._action_ball_continuous_motion_poisoned

    fresh, _owner, _epoch, _token, _record, _calls = _install_exact_sources(
        monkeypatch,
        n=2,
        device=torch.device("cpu"),
        accept_mask=torch.tensor([True, False]),
    )
    foreign = object.__new__(D05.DeviceR05RowTransaction)
    with pytest.raises(D05.DeviceR05ConflictError, match="stale|foreign"):
        fresh.commit_action_ball_full_mdp_motion_epoch_rows(foreign)
    assert fresh._action_ball_continuous_motion_poisoned


def test_writer_and_post_publication_have_no_compact_or_host_verdict_path() -> None:
    parameters = tuple(
        inspect.signature(
            C.MotionCommand.commit_action_ball_full_mdp_motion_epoch_rows
        ).parameters
    )
    assert parameters == ("self", "token")
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            C.MotionCommand.commit_action_ball_full_mdp_motion_epoch_rows,
            C.MotionCommand._commit_action_ball_full_mdp_motion_epoch_rows_impl,
            C.MotionCommand._action_ball_full_mdp_motion_exact_tensor,
            C.MotionCommand.publish_action_ball_full_mdp_post_d05_observation,
            C.MotionCommand._publish_action_ball_full_mdp_post_d05_observation_impl,
        )
    )
    for forbidden in (
        ".item(",
        ".cpu(",
        ".numpy(",
        ".tolist(",
        "bool(",
        "nonzero",
        "masked_select",
        "index_copy",
        "torch.arange",
        "torch._assert_async",
        "epoch_owner.current(",
    ):
        assert forbidden not in source
    assert "torch.where" in source
    assert "require_owned_action_epoch_accepted" in source
    assert "_seal_action_ball_continuous_current_projection" in source
    assert "require_owned_prepared_reveal_for_child" not in source
