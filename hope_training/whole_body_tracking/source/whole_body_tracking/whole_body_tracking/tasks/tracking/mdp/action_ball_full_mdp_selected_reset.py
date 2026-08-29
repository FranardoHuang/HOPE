"""Typed, publication-free selected-reset transaction for ActionEpoch.
It imports no owner and cannot publish, log, perform D2H, or call a leaf."""
from __future__ import annotations
from dataclasses import dataclass
import importlib
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Protocol

import torch
class SelectedResetProtocolError(RuntimeError):
    """The selected-reset lease, boundary, or tensor contract differs."""
class ActionEpochPreparedSelectedReset:
    """Opaque epoch-issued lease for one packed-preflighted selected reset."""

    __slots__ = ()
    def __new__(cls):
        raise TypeError("selected-reset leases are minted only by ActionEpoch")


class SelectedResetRecordView(Protocol):
    """Structural read-only view required by the pure after-image builders."""
    epoch: int
    version: int
    phase: torch.Tensor
    identity: object
    clocks: object
    task: object
    rng_counter: torch.Tensor
    current_task_slot: torch.Tensor
    publication_ordinal: torch.Tensor
    owner_fault_bits: torch.Tensor
    writes_started: torch.Tensor
    writes_committed: torch.Tensor
    physical_launch_requested: torch.Tensor
    launch_succeeded: torch.Tensor
    late_launch: torch.Tensor
    outcome_code: torch.Tensor
    reward_cycle_open: torch.Tensor
    reward_due: torch.Tensor
    reward_paid: torch.Tensor
    fact_valid_bits: torch.Tensor
    fact_source_step: torch.Tensor
    fact_f32: torch.Tensor
    reset_selected_mask: torch.Tensor
    motion_playback_started: torch.Tensor
    motion_close_reason: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor
    poison_reason: torch.Tensor


@dataclass(frozen=True)
class SelectedResetPendingPlan:
    changes: Mapping[str, object]
    changed_fields: tuple[str, ...]
    generation_after: torch.Tensor
    overflow: torch.Tensor
@dataclass(frozen=True)
class SelectedResetCommitPlan:
    lease: ActionEpochPreparedSelectedReset
    changes: Mapping[str, object]
    changed_fields: tuple[str, ...]
    generation_after: torch.Tensor
    selected_mask: torch.Tensor
    overflow: torch.Tensor
    terminal_reset_facts_i64: torch.Tensor
@dataclass(frozen=True)
class _SelectedResetOwnerBinding:
    owner: object
    coordinator_method: Callable[..., object]
    preflight_validator: Callable[..., object]
    commit_validator: Callable[..., object]


@dataclass(frozen=True)
class _FrozenSelectedReset:
    lease: ActionEpochPreparedSelectedReset
    top_preflight: object
    version: int
    selected_mask: torch.Tensor
    generation_after: torch.Tensor
    overflow: torch.Tensor
    terminal_reset_facts_i64: torch.Tensor


def _require_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
    ):
        raise SelectedResetProtocolError(
            f"{label} must be {shape} {dtype} on {device}"
        )
    return value.detach().clone().contiguous()


def build_pending_selected_reset_plan(
    *,
    record: SelectedResetRecordView,
    selected_mask: torch.Tensor,
    generation: torch.Tensor,
    num_envs: int,
    device: torch.device,
    i64_max: int,
    generation_overflow_fault_bit: int,
    poisoned_phase: int,
) -> SelectedResetPendingPlan:
    """Build the legacy selected-reset device-fact publication without writes."""

    selected = _require_tensor(
        selected_mask,
        label="selected_mask",
        shape=(num_envs,),
        dtype=torch.bool,
        device=device,
    )
    overflow = selected & generation.eq(i64_max)
    safe = selected & ~overflow
    safe_base = torch.where(safe, generation, torch.zeros_like(generation))
    after_generation = torch.where(
        safe, safe_base + safe.to(torch.int64), generation
    )
    phase = torch.where(
        overflow[:, None],
        torch.full_like(record.phase, poisoned_phase),
        record.phase,
    )
    faults = record.owner_fault_bits.clone()
    faults[:, :, 0] = torch.where(
        overflow[:, None],
        torch.bitwise_or(
            faults[:, :, 0],
            torch.full_like(faults[:, :, 0], generation_overflow_fault_bit),
        ),
        faults[:, :, 0],
    )
    reasons = torch.where(
        overflow[:, None],
        torch.full_like(record.poison_reason, generation_overflow_fault_bit),
        record.poison_reason,
    )
    changes = MappingProxyType({
        "phase": phase,
        "owner_fault_bits": faults,
        "reset_generation": after_generation,
        "reset_selected_mask": selected,
        "poison_reason": reasons,
    })
    return SelectedResetPendingPlan(
        changes=changes,
        changed_fields=(
            "phase",
            "owner_fault_bits",
            "reset_generation",
            "reset_selected_mask",
            "poison_reason",
        ),
        generation_after=after_generation,
        overflow=overflow,
    )


def _masked_after_image(
    record: SelectedResetRecordView,
    frozen: _FrozenSelectedReset,
    *,
    num_envs: int,
    idle_phase: int,
) -> tuple[Mapping[str, object], tuple[str, ...]]:
    selected = frozen.selected_mask
    shot = selected[:, None]
    owner_rows = shot[:, :, None]

    def masked(value: torch.Tensor, fill: int | float | bool) -> torch.Tensor:
        mask = selected.reshape((num_envs,) + (1,) * (value.ndim - 1))
        return torch.where(mask, torch.full_like(value, fill), value)

    identity_type = type(record.identity)
    clock_type = type(record.clocks)
    task_type = type(record.task)
    identity_fields = tuple(record.identity.__dataclass_fields__)
    clock_fields = tuple(record.clocks.__dataclass_fields__)
    shot_key_type = type(record.identity.shot_key)
    shot_key = shot_key_type(**{
        name: masked(getattr(record.identity.shot_key, name), -1)
        for name in record.identity.shot_key.__dataclass_fields__
    })
    identity = identity_type(
        shot_key=shot_key,
        **{
            name: masked(getattr(record.identity, name), -1)
            for name in identity_fields
            if name != "shot_key"
        },
    )
    clocks = clock_type(**{
        name: masked(getattr(record.clocks, name), -1) for name in clock_fields
    })
    task = task_type(
        task_f32=masked(record.task.task_f32, 0.0),
        task_valid=masked(record.task.task_valid, False),
    )
    changes = MappingProxyType(
        {
            "phase": torch.where(
                shot, torch.full_like(record.phase, idle_phase), record.phase),
            "identity": identity,
            "clocks": clocks,
            "task": task,
            "rng_counter": masked(record.rng_counter, -1),
            "current_task_slot": torch.where(
                selected,
                torch.zeros_like(record.current_task_slot),
                record.current_task_slot,
            ),
            "publication_ordinal": masked(record.publication_ordinal, -1),
            "owner_fault_bits": torch.where(owner_rows,
                torch.zeros_like(record.owner_fault_bits), record.owner_fault_bits),
            "writes_started": torch.where(owner_rows,
                torch.zeros_like(record.writes_started), record.writes_started),
            "writes_committed": torch.where(owner_rows,
                torch.zeros_like(record.writes_committed), record.writes_committed),
            "physical_launch_requested": masked(
                record.physical_launch_requested, False),
            "launch_succeeded": masked(record.launch_succeeded, False),
            "late_launch": masked(record.late_launch, False),
            "outcome_code": masked(record.outcome_code, -1),
            "reward_cycle_open": torch.where(selected,
                torch.zeros_like(record.reward_cycle_open), record.reward_cycle_open),
            "reward_due": masked(record.reward_due, False),
            "reward_paid": masked(record.reward_paid, False),
            "fact_valid_bits": torch.where(owner_rows,
                torch.zeros_like(record.fact_valid_bits), record.fact_valid_bits),
            "fact_source_step": torch.where(owner_rows,
                torch.full_like(record.fact_source_step, -1), record.fact_source_step),
            "fact_f32": masked(record.fact_f32, 0.0),
            "reset_generation": frozen.generation_after,
            "reset_selected_mask": torch.where(
                selected, torch.ones_like(selected), record.reset_selected_mask),
            "motion_playback_started": masked(record.motion_playback_started, False),
            "motion_close_reason": masked(record.motion_close_reason, 0),
            "settlement_step": masked(record.settlement_step, -1),
            "payment_step": masked(record.payment_step, -1),
            "poison_reason": masked(record.poison_reason, 0),
        }
    )
    changed = (
        "phase",
        *("identity." + name for name in identity_fields),
        *("clocks." + name for name in clock_fields),
        "task.task_f32",
        "task.task_valid",
        "rng_counter",
        "current_task_slot",
        "publication_ordinal",
        "owner_fault_bits",
        "writes_started",
        "writes_committed",
        "physical_launch_requested",
        "launch_succeeded",
        "late_launch",
        "outcome_code",
        "reward_cycle_open",
        "reward_due",
        "reward_paid",
        "fact_valid_bits",
        "fact_source_step",
        "fact_f32",
        "reset_generation",
        "reset_selected_mask",
        "motion_playback_started",
        "motion_close_reason",
        "settlement_step",
        "payment_step",
        "poison_reason",
    )
    return changes, changed

class SelectedResetTransaction:
    """One bound, single-flight selected-reset lease and pure planner."""
    def __init__(self, *, num_envs: int, device: torch.device) -> None:
        self._num_envs = num_envs
        self._device = device
        self._binding: Optional[_SelectedResetOwnerBinding] = None
        self._active: Optional[_FrozenSelectedReset] = None

    @property
    def active(self) -> bool:
        return self._active is not None

    def bind(self, owner: object, *, canonical_genesis_idle: bool) -> None:
        if owner is None or self._binding is not None:
            raise SelectedResetProtocolError(
                "selected-reset owner is absent or already bound"
            )
        if not canonical_genesis_idle:
            raise SelectedResetProtocolError(
                "selected-reset owner requires canonical genesis IDLE"
            )
        module_name = ((__package__ + ".") if __package__ else "") + (
            "action_ball_full_mdp_lean_runtime"
        )
        owner_type = getattr(
            importlib.import_module(module_name),
            "ActionBallFullMdpLeanRuntimeOwner", None,
        )
        if type(owner_type) is not type or type(owner) is not owner_type:
            raise SelectedResetProtocolError(
                "selected-reset owner must be the canonical Lean runtime"
            )
        names = (
            "selected_true_reset",
            "require_owned_epoch_selected_reset_preflight",
            "require_owned_epoch_selected_reset_commit",
        )
        pairs = tuple(
            (getattr(owner, name, None), getattr(owner_type, name, None))
            for name in names
        )
        if any(
            not callable(bound)
            or not callable(direct)
            or getattr(bound, "__self__", None) is not owner
            or getattr(bound, "__func__", None) is not direct
            for bound, direct in pairs
        ):
            raise SelectedResetProtocolError(
                "selected-reset owner must expose exact bound methods"
            )
        self._binding = _SelectedResetOwnerBinding(
            owner=owner,
            coordinator_method=pairs[0][0],
            preflight_validator=pairs[1][0],
            commit_validator=pairs[2][0],
        )

    def prepare(
        self,
        *,
        owner: object,
        top_preflight: object,
        record: Optional[SelectedResetRecordView],
        boundary_is_open: bool,
        selected_env_index: torch.Tensor,
        selected_mask: torch.Tensor,
        generation_before: torch.Tensor,
        generation_after: torch.Tensor,
        generation_overflow_fault: torch.Tensor,
        terminal_reset_facts_i64: torch.Tensor,
    ) -> ActionEpochPreparedSelectedReset:
        binding = self._binding
        if (
            binding is None
            or owner is not binding.owner
            or record is None
            or self._active is not None
            or not boundary_is_open
        ):
            raise SelectedResetProtocolError(
                "selected true reset prepare owner or epoch boundary differs"
            )
        owned = binding.preflight_validator(
            top_preflight,
            selected_env_index=selected_env_index,
            selected_mask=selected_mask,
            generation_before=generation_before,
            generation_after=generation_after,
            generation_overflow_fault=generation_overflow_fault,
            terminal_reset_facts_i64=terminal_reset_facts_i64,
        )
        if owned is not top_preflight:
            raise SelectedResetProtocolError(
                "selected true reset preflight identity differs"
            )
        if (
            type(selected_env_index) is not torch.Tensor
            or selected_env_index.ndim != 1
            or selected_env_index.shape[0] < 1
        ):
            raise SelectedResetProtocolError(
                "selected_env_index must be nonempty rank-1"
            )
        index = _require_tensor(
            selected_env_index,
            label="selected_env_index",
            shape=(selected_env_index.shape[0],),
            dtype=torch.int64,
            device=self._device,
        )
        tensors = (
            ("selected_mask", selected_mask, torch.bool),
            ("generation_before", generation_before, torch.int64),
            ("generation_after", generation_after, torch.int64),
            ("generation_overflow_fault", generation_overflow_fault, torch.bool),
        )
        selected, _before, after, overflow = (
            _require_tensor(
                value,
                label=label,
                shape=(self._num_envs,),
                dtype=dtype,
                device=self._device,
            )
            for label, value, dtype in tensors
        )
        terminal_facts = _require_tensor(
            terminal_reset_facts_i64,
            label="terminal_reset_facts_i64",
            shape=(self._num_envs, 3),
            dtype=torch.int64,
            device=self._device,
        )
        lease = object.__new__(ActionEpochPreparedSelectedReset)
        self._active = _FrozenSelectedReset(
            lease=lease,
            top_preflight=top_preflight,
            version=record.version,
            selected_mask=selected,
            generation_after=after,
            overflow=overflow,
            terminal_reset_facts_i64=terminal_facts,
        )
        return lease

    def abort(
        self, *, owner: object, lease: ActionEpochPreparedSelectedReset
    ) -> None:
        active = self._require_active(owner=owner, lease=lease, label="abort")
        if active.lease is not lease:
            raise SelectedResetProtocolError("selected true reset abort lease differs")
        self._active = None

    def plan_commit(
        self,
        *,
        owner: object,
        lease: ActionEpochPreparedSelectedReset,
        record: Optional[SelectedResetRecordView],
        boundary_is_quiescent: bool,
        idle_phase: int,
    ) -> SelectedResetCommitPlan:
        active = self._require_active(owner=owner, lease=lease, label="commit")
        binding = self._binding
        if (
            binding is None
            or record is None
            or record.version != active.version
            or not boundary_is_quiescent
        ):
            raise SelectedResetProtocolError(
                "selected true reset owner or quiescent epoch boundary differs"
            )
        if binding.commit_validator(active.top_preflight, lease) is not lease:
            raise SelectedResetProtocolError(
                "selected true reset lacks the exact completed top transaction"
            )
        changes, changed = _masked_after_image(
            record, active, num_envs=self._num_envs, idle_phase=idle_phase
        )
        return SelectedResetCommitPlan(
            lease=lease,
            changes=changes,
            changed_fields=changed,
            generation_after=active.generation_after,
            selected_mask=active.selected_mask,
            overflow=active.overflow,
            terminal_reset_facts_i64=(
                active.terminal_reset_facts_i64.clone()
            ),
        )

    def complete_commit(self, lease: ActionEpochPreparedSelectedReset) -> None:
        active = self._active
        if active is None or active.lease is not lease:
            raise SelectedResetProtocolError(
                "selected true reset completion lease differs"
            )
        self._active = None

    def _require_active(
        self,
        *,
        owner: object,
        lease: ActionEpochPreparedSelectedReset,
        label: str,
    ) -> _FrozenSelectedReset:
        binding = self._binding
        active = self._active
        if (
            binding is None
            or owner is not binding.owner
            or active is None
            or type(lease) is not ActionEpochPreparedSelectedReset
            or active.lease is not lease
        ):
            raise SelectedResetProtocolError(
                f"selected true reset {label} lease differs"
            )
        return active


__all__ = [
    "ActionEpochPreparedSelectedReset", "SelectedResetCommitPlan",
    "SelectedResetPendingPlan", "SelectedResetProtocolError",
    "SelectedResetTransaction", "build_pending_selected_reset_plan",
]
