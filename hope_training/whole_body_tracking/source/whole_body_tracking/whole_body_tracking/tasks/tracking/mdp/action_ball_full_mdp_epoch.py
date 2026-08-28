"""Row-wise ActionBall carry state and the sole packed business journal.

The public record contains only the current or just-completed shot in every
environment row.  A D05 opportunity is a private, single-flight transaction:
REJECT/DEFER/CENSOR are journal events and never replace the public shot;
ACCEPT alone performs one masked replacement after the old row is closed.

All row-validity decisions remain device resident.  Motion and R06 are
cold-bound real producers and are called without a caller mask, index, key, or
verdict.  The after-command boundary performs one exact all-idle reduction so
the common K=0 case never enters the dense D05 numeric/writer chain.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
import importlib
import threading
from typing import Iterator, Optional, Union

import torch

try:
    import action_ball_full_mdp_row_identity as row_identity
except ImportError:  # pragma: no cover - package-style test import
    from whole_body_tracking import action_ball_full_mdp_row_identity as row_identity

try:
    from whole_body_tracking import action_ball_continuous_recovery_device as r07_device
except ImportError:  # pragma: no cover - direct-source focused tests
    import action_ball_continuous_recovery_device as r07_device

try:
    from . import action_ball_full_mdp_selected_reset as selected_reset
except ImportError:
    import action_ball_full_mdp_selected_reset as selected_reset

try:
    from . import action_ball_full_mdp_milestone_tensors as milestone_tensors
except ImportError:
    import action_ball_full_mdp_milestone_tensors as milestone_tensors

try:
    from . import action_ball_full_mdp_lean_checkpoint_txn as carry_txn
except ImportError:
    import action_ball_full_mdp_lean_checkpoint_txn as carry_txn

try:
    from . import action_ball_full_mdp_action_strata as action_strata
except ImportError:
    import action_ball_full_mdp_action_strata as action_strata

ActionEpochPreparedSelectedReset = selected_reset.ActionEpochPreparedSelectedReset
ActionEpochShotKey = row_identity.ActionEpochShotKey

DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

OWNER_ORDER = (
    "r05_runtime",
    "motion",
    "racket",
    "physical_ball",
    "r06_landing_outcome",
    "r03_strike_fact",
    "r07_recovery",
)
OWNER_COUNT = len(OWNER_ORDER)
_LEAN_REWARD_PHYSICAL_SLOT = OWNER_ORDER.index("physical_ball")
_LEAN_REWARD_R06_SLOT = OWNER_ORDER.index("r06_landing_outcome")
_LEAN_REWARD_R03_SLOT = OWNER_ORDER.index("r03_strike_fact")
_LEAN_REWARD_R07_SLOT = OWNER_ORDER.index("r07_recovery")
REVEAL_WRITE_OWNER_ORDER = ("motion", "racket", "r05_runtime")
LAUNCH_WRITE_OWNER_ORDER = ("physical_ball",)

REWARD_CONSUMER_ORDER = (
    "r03:racket_position",
    "r03:racket_velocity",
    "r03:racket_normal",
    "r03:racket_position_coarse",
    "r03:racket_velocity_coarse",
    "r03:racket_normal_coarse",
    "r03:racket_position_precision",
    "r03:racket_velocity_precision",
    "r03:racket_normal_precision",
    "r03:paddle_center_proximity",
    "physical:physical_selected_contact",
    "r06:common_on_table_outcome",
    "r06:post_contact_placement_guidance",
    "r07:common_recovery_reward_v1",
)
LIFECYCLE_PAYMENT_COUNT = 14
REWARD_CONSUMER_COUNT = LIFECYCLE_PAYMENT_COUNT

MOTION_TASK_F32_WIDTH = 5
RACKET_TASK_F32_WIDTH = 27
PHYSICAL_TASK_F32_WIDTH = 13
TASK_F32_WIDTH = MOTION_TASK_F32_WIDTH + RACKET_TASK_F32_WIDTH + PHYSICAL_TASK_F32_WIDTH
OWNER_FACT_F32_WIDTH = 32

# Public carry-state phases.  Candidate classifications deliberately do not
# share this enum.
PHASE_IDLE = 0
PHASE_REVEAL_COMMITTED = 2
PHASE_LAUNCH_SETTLED = 5
PHASE_OUTCOME_SETTLED = 6
PHASE_RETIRED = 8  # closed but retained until the next ACCEPT
PHASE_POISONED = 9

# Legacy exports are kept as values only; the row-wise owner never writes them
# into the public record.
PHASE_PREPARED = 1
PHASE_CENSORED = 3
PHASE_REJECTED = 4
PHASE_REWARD_SETTLED = 7
PHASE_DEFERRED = 10


def action_epoch_open_shot_phase_mask(phase: torch.Tensor) -> torch.Tensor:
    """Rows whose published shot may still receive Motion playback/close edges."""

    return (
        phase.eq(PHASE_REVEAL_COMMITTED)
        | phase.eq(PHASE_LAUNCH_SETTLED)
        | phase.eq(PHASE_OUTCOME_SETTLED)
    )

D05_DECISION_NONE = 0
D05_DECISION_ACCEPT = 1
D05_DECISION_CENSOR = 2
D05_DECISION_REJECT = 3
D05_DECISION_DEFER = 4

MOTION_CLOSE_NONE = 0
MOTION_CLOSE_PLAYED_SUFFIX = 1
MOTION_CLOSE_UNPLAYED = 2
MOTION_PLAYBACK_STARTED = "MOTION_PLAYBACK_STARTED"
MOTION_CLOSED = "MOTION_CLOSED"

_I64_MAX = 2**63 - 1
_FAULT_ILLEGAL_REPLAY = 1 << 60
_FAULT_ABORT_AFTER_PUBLICATION = 1 << 59

# Device-resident row faults share the existing single packed drain transfer.
# Keep these causes orthogonal: a fatal run must say which causal join failed,
# rather than collapsing unrelated lifecycle contracts into one boolean.
ROW_FAULT_RESET_GENESIS_CONTRACT = 1 << 0
ROW_FAULT_MOTION_CLOSE_CONTRACT = 1 << 1
ROW_FAULT_R06_PREVIOUS_PAID_CONTRACT = 1 << 2
ROW_FAULT_D05_RESET_GENERATION_JOIN = 1 << 3
ROW_FAULT_PHYSICAL_POSTPHYSICS_JOIN = 1 << 4
ROW_FAULT_R06_OUTCOME_JOIN = 1 << 5
ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY = 1 << 6
ROW_FAULT_OWNER_FACT_ACTIVE_JOIN = 1 << 7
ROW_FAULT_R07_FIRST_READY_JOIN = 1 << 8
ROW_FAULT_PHYSICAL_LAUNCH_JOIN = 1 << 9
ROW_FAULT_SELECTED_RESET_GENERATION_OVERFLOW = 1 << 10
ROW_FAULT_R06_LAUNCH_SELECTION_CONTRACT = 1 << 11
ROW_FAULT_R06_LAUNCH_IDENTITY_CONTRACT = 1 << 12
ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE = 1 << 13
ROW_FAULT_R06_PAYMENT_PROJECTION_CONTRACT = 1 << 14
ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE = 1 << 15
ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED = 1 << 16
ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT = 1 << 17
ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION = 1 << 18
ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE = 1 << 19
ROW_FAULT_R06_CLOSED_PROJECTION_CONTRACT = 1 << 20
ROW_FAULT_R06_CLOSED_DEBT_MISMATCH = 1 << 21
ROW_FAULT_R06_CURRENT_FLIGHT_DUPLICATE = 1 << 22
ROW_FAULT_MOTION_CADENCE_OVERDUE = 1 << 23
ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW = 1 << 24
ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT = 1 << 25
ROW_FAULT_MOTION_TASK_TIMING_CONTRACT = 1 << 26
# R03 keeps its compact source-local word in ``owner_fault_bits`` for exact
# audit, but these disjoint ActionEpoch bits are the optimizer-facing causes.
# A rejected source row therefore cannot turn into a neutral Reward and still
# enter PPO merely because the Reward view correctly ignores owner faults.
ROW_FAULT_R03_EPOCH_IDENTITY = 1 << 27
ROW_FAULT_R03_STALE_SOURCE_STEP = 1 << 28
ROW_FAULT_R03_NONFINITE_FACT = 1 << 29
# The terminal R07 cell is a distinct cross-clock chronology fact.  It is not
# the active-row publication join (bit 7) or first-ready identity join (bit 8).
ROW_FAULT_R07_TERMINAL_FACT_CONTRACT = 1 << 30
# Physical already publishes these two exact source bits.  Keep the values
# identical at the optimizer boundary so the causal producer/nonfinite split is
# not lost while moving from the owner journal into the one packed row word.
ROW_FAULT_PHYSICAL_POSTPHYSICS_PRODUCER = 1 << 41
ROW_FAULT_PHYSICAL_POSTPHYSICS_NONFINITE = 1 << 42
# R06's private fault namespace overlaps earlier ActionEpoch bits, so map its
# retained source word into a disjoint public row-fault range.  The raw R06 word
# remains in the journal for exact sub-cause audit.
ROW_FAULT_R06_OWNER_PRODUCER_CONTRACT = 1 << 43
ROW_FAULT_R06_OWNER_ENGINE_OVERFLOW = 1 << 44
ROW_FAULT_R06_OWNER_NONFINITE = 1 << 45
ROW_FAULT_R06_OWNER_OTHER = 1 << 46

ACTION_EPOCH_ROW_FAULT_NAMES = (
    (ROW_FAULT_RESET_GENESIS_CONTRACT, "reset_genesis_contract"),
    (ROW_FAULT_MOTION_CLOSE_CONTRACT, "motion_close_contract"),
    (ROW_FAULT_R06_PREVIOUS_PAID_CONTRACT, "r06_previous_paid_contract"),
    (ROW_FAULT_D05_RESET_GENERATION_JOIN, "d05_reset_generation_join"),
    (ROW_FAULT_PHYSICAL_POSTPHYSICS_JOIN, "physical_postphysics_join"),
    (ROW_FAULT_R06_OUTCOME_JOIN, "r06_outcome_join"),
    (ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY, "reward_payment_chronology"),
    (ROW_FAULT_OWNER_FACT_ACTIVE_JOIN, "owner_fact_active_join"),
    (ROW_FAULT_R07_FIRST_READY_JOIN, "r07_first_ready_join"),
    (ROW_FAULT_PHYSICAL_LAUNCH_JOIN, "physical_launch_join"),
    (
        ROW_FAULT_SELECTED_RESET_GENERATION_OVERFLOW,
        "selected_reset_generation_overflow",
    ),
    (
        ROW_FAULT_R06_LAUNCH_SELECTION_CONTRACT,
        "r06_launch_selection_contract",
    ),
    (
        ROW_FAULT_R06_LAUNCH_IDENTITY_CONTRACT,
        "r06_launch_identity_contract",
    ),
    (
        ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE,
        "r06_outcome_projection_duplicate",
    ),
    (
        ROW_FAULT_R06_PAYMENT_PROJECTION_CONTRACT,
        "r06_payment_projection_contract",
    ),
    (
        ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE,
        "r06_payment_mailbox_duplicate",
    ),
    (
        ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED,
        "r06_payment_missing_or_mismatched",
    ),
    (
        ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT,
        "r06_payment_before_settlement",
    ),
    (
        ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION,
        "r06_payment_highwater_regression",
    ),
    (
        ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
        "r06_payment_unconsumed_debt_overwrite",
    ),
    (
        ROW_FAULT_R06_CLOSED_PROJECTION_CONTRACT,
        "r06_closed_projection_contract",
    ),
    (
        ROW_FAULT_R06_CLOSED_DEBT_MISMATCH,
        "r06_closed_debt_mismatch",
    ),
    (
        ROW_FAULT_R06_CURRENT_FLIGHT_DUPLICATE,
        "r06_current_flight_duplicate",
    ),
    (ROW_FAULT_MOTION_CADENCE_OVERDUE, "motion_cadence_overdue"),
    (
        ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW,
        "motion_swing_generation_overflow",
    ),
    (
        ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
        "motion_reveal_reference_contract",
    ),
    (
        ROW_FAULT_MOTION_TASK_TIMING_CONTRACT,
        "motion_task_timing_contract",
    ),
    (ROW_FAULT_R03_EPOCH_IDENTITY, "r03_epoch_identity"),
    (ROW_FAULT_R03_STALE_SOURCE_STEP, "r03_stale_source_step"),
    (ROW_FAULT_R03_NONFINITE_FACT, "r03_nonfinite_fact"),
    (
        ROW_FAULT_R07_TERMINAL_FACT_CONTRACT,
        "r07_terminal_fact_contract",
    ),
    (
        ROW_FAULT_PHYSICAL_POSTPHYSICS_PRODUCER,
        "physical_postphysics_producer",
    ),
    (
        ROW_FAULT_PHYSICAL_POSTPHYSICS_NONFINITE,
        "physical_postphysics_nonfinite",
    ),
    (
        ROW_FAULT_R06_OWNER_PRODUCER_CONTRACT,
        "r06_owner_producer_contract",
    ),
    (
        ROW_FAULT_R06_OWNER_ENGINE_OVERFLOW,
        "r06_owner_engine_overflow",
    ),
    (ROW_FAULT_R06_OWNER_NONFINITE, "r06_owner_nonfinite"),
    (ROW_FAULT_R06_OWNER_OTHER, "r06_owner_other"),
)
_KNOWN_ROW_FAULT_MASK = sum(bit for bit, _name in ACTION_EPOCH_ROW_FAULT_NAMES)
_R06_RUNTIME_ROW_FAULT_BITS = frozenset(
    (
        ROW_FAULT_R06_LAUNCH_SELECTION_CONTRACT,
        ROW_FAULT_R06_LAUNCH_IDENTITY_CONTRACT,
        ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE,
        ROW_FAULT_R06_PAYMENT_PROJECTION_CONTRACT,
        ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE,
        ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED,
        ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT,
        ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION,
        ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE,
        ROW_FAULT_R06_CLOSED_PROJECTION_CONTRACT,
        ROW_FAULT_R06_CLOSED_DEBT_MISMATCH,
        ROW_FAULT_R06_CURRENT_FLIGHT_DUPLICATE,
    )
)
_RUNTIME_ROW_FAULT_BITS_BY_OWNER = {
    "r03_strike_fact": frozenset(
        (
            ROW_FAULT_R03_EPOCH_IDENTITY,
            ROW_FAULT_R03_STALE_SOURCE_STEP,
            ROW_FAULT_R03_NONFINITE_FACT,
        )
    ),
    "r06_landing_outcome": _R06_RUNTIME_ROW_FAULT_BITS,
    "motion": frozenset(
        (
            ROW_FAULT_MOTION_CADENCE_OVERDUE,
            ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW,
            ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
            ROW_FAULT_MOTION_TASK_TIMING_CONTRACT,
        )
    ),
}
_FAULT_GENERATION_OVERFLOW = 1 << 58
_FAULT_REWARD_CYCLE_OVERFLOW = 1 << 57
_FAULT_ASYNC_REPLAY = 1 << 52
_FAULT_MOTION_PLAYBACK = 1 << 51
_FAULT_D05_OWNER = 1 << 50
_FAULT_R06_OUTCOME = 1 << 49
_FAULT_D05_RESET_GENERATION = 1 << 48
_FAULT_MILESTONE_TELEMETRY = 1 << 47


class ActionEpochError(RuntimeError):
    """The row-wise epoch contract or one of its real producers differs."""


@dataclass(frozen=True)
class EpochIdentityPayload:
    """The sole typed shot key plus non-key D05 construction chronology."""

    shot_key: ActionEpochShotKey
    scheduled_ordinal: torch.Tensor
    target_generation: torch.Tensor
    selected_cell: torch.Tensor
    candidate_identity: torch.Tensor

    def clone(self) -> "EpochIdentityPayload":
        return EpochIdentityPayload(
            shot_key=self.shot_key.clone(),
            scheduled_ordinal=self.scheduled_ordinal.clone(),
            target_generation=self.target_generation.clone(),
            selected_cell=self.selected_cell.clone(),
            candidate_identity=self.candidate_identity.clone(),
        )

    # Read-only migration views.  New joins must consume ``shot_key``.
    @property
    def action_uid(self) -> torch.Tensor:
        return self.shot_key.action_uid

    @property
    def action_slot(self) -> torch.Tensor:
        return self.shot_key.action_slot

    @property
    def task_identity(self) -> torch.Tensor:
        return self.shot_key.task_identity

    @property
    def outcome_identity(self) -> torch.Tensor:
        return self.shot_key.outcome_identity

    @property
    def ball_identity(self) -> torch.Tensor:
        return self.shot_key.ball_identity


@dataclass(frozen=True)
class EpochClockPayload:
    reveal_tick: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    deadline_tick: torch.Tensor
    next_reveal_tick: torch.Tensor

    def clone(self) -> "EpochClockPayload":
        return EpochClockPayload(
            **{field.name: getattr(self, field.name).clone() for field in fields(self)}
        )


@dataclass(frozen=True)
class EpochTaskPayload:
    task_f32: torch.Tensor
    task_valid: torch.Tensor

    def clone(self) -> "EpochTaskPayload":
        return EpochTaskPayload(self.task_f32.clone(), self.task_valid.clone())


@dataclass(frozen=True)
class RecoveryReferenceLifecycleMasks:
    upcoming: torch.Tensor
    completed: torch.Tensor


@dataclass(frozen=True)
class ActionEpochIdleObservationChronology:
    """Clone-only reset chronology for an unkeyed neutral observation."""

    epoch_version: int
    reset_generation: torch.Tensor


@dataclass(frozen=True)
class ActionEpochCurrentShotProjection:
    """Transient clone-only value for the current shot in every row.

    The owner gathers one internally selected slot and returns no authority,
    token, or caller-selected mask.  Consumers must use the value immediately
    on the current device stream and must not retain it across an Epoch write.
    """

    slot_valid: torch.Tensor
    phase: torch.Tensor
    shot_key: ActionEpochShotKey
    publication_ordinal: torch.Tensor


@dataclass(frozen=True)
class ActionEpochMotionPlaybackProjection:
    """Narrow public Epoch facts needed by Motion's full-key playback join.

    The projection is consumed synchronously while Epoch owns the publication
    operation.  It carries no publication version or mutable owner authority:
    freshness comes from Epoch constructing it from the locked current record.
    """

    current_task_slot: torch.Tensor
    phase: torch.Tensor
    selected_mask: torch.Tensor
    shot_key: ActionEpochShotKey


@dataclass(frozen=True)
class ActionEpochRecord:
    """One immutable public carry-state version.

    ``epoch`` is a temporary scalar compatibility spelling for publication
    order.  It is not part of shot identity.  ``publication_ordinal`` is the
    per-cell publication that Physical/R06 retain alongside ``shot_key``.
    """

    epoch: int
    version: int
    phase: torch.Tensor
    identity: EpochIdentityPayload
    clocks: EpochClockPayload
    task: EpochTaskPayload
    rng_counter: torch.Tensor
    current_task_slot: torch.Tensor
    publication_ordinal: torch.Tensor
    owner_fault_bits: torch.Tensor
    writes_started: torch.Tensor
    writes_committed: torch.Tensor
    launch_succeeded: torch.Tensor
    late_launch: torch.Tensor
    outcome_code: torch.Tensor
    reward_cycle_age: torch.Tensor
    reward_cycle_fault: torch.Tensor
    reward_cycle_open: torch.Tensor
    reward_due: torch.Tensor
    reward_paid: torch.Tensor
    fact_valid_bits: torch.Tensor
    fact_source_step: torch.Tensor
    fact_f32: torch.Tensor
    reset_generation: torch.Tensor
    reset_selected_mask: torch.Tensor
    motion_playback_started: torch.Tensor
    motion_close_reason: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor
    poison_reason: torch.Tensor
    diagnostic_unauthorized: bool = True

    def clone(self) -> "ActionEpochRecord":
        changes: dict[str, object] = {
            "identity": self.identity.clone(),
            "clocks": self.clocks.clone(),
            "task": self.task.clone(),
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is torch.Tensor:
                changes[field.name] = value.clone()
        return replace(self, **changes)

    @property
    def action_uid(self) -> torch.Tensor:
        return self.identity.shot_key.action_uid

    @property
    def selected_mask(self) -> torch.Tensor:
        """Compatibility view derived from public phase, never a due mask."""

        return ~self.phase.eq(PHASE_IDLE)

    @property
    def motion_closed_unplayed(self) -> torch.Tensor:
        return self.motion_close_reason.eq(MOTION_CLOSE_UNPLAYED)

    def recovery_reference_lifecycle_masks(self) -> RecoveryReferenceLifecycleMasks:
        """Project row-wise frame-0 authority without consulting scalar epoch.

        An IDLE cell remains the upcoming scheduled-action reference even after
        REJECT/DEFER/CENSOR journal events because those events never publish a
        replacement shot.  A published shot is the completed reference from
        REVEAL through RETIRED only after Motion's ACCEPT writer both started
        and committed.  The two masks are full ``[N, S]`` and disjoint.
        """

        motion_slot = OWNER_ORDER.index("motion")
        started = self.writes_started[:, :, motion_slot]
        committed = self.writes_committed[:, :, motion_slot]
        upcoming = self.phase.eq(PHASE_IDLE)
        completed = (
            self.phase.eq(PHASE_REVEAL_COMMITTED)
            | self.phase.eq(PHASE_LAUNCH_SETTLED)
            | self.phase.eq(PHASE_OUTCOME_SETTLED)
            | self.phase.eq(PHASE_RETIRED)
        ) & started & committed
        return RecoveryReferenceLifecycleMasks(upcoming.clone(), completed.clone())


@dataclass(frozen=True)
class _LeanRewardCycleSnapshot:
    """Selected, copied before-image consumed by lifecycle Reward rows only."""

    reward_cycle_age: torch.Tensor
    reward_cycle_fault: torch.Tensor
    phase: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor
    r03: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    physical: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    r06: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    r07: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ActionEpochDueRows:
    common_step: int
    due_mask: torch.Tensor
    construct_mask: torch.Tensor

    def clone(self) -> "ActionEpochDueRows":
        return ActionEpochDueRows(
            self.common_step, self.due_mask.clone(), self.construct_mask.clone()
        )


@dataclass(frozen=True)
class ActionEpochD05CandidateProjection:
    """Full-N D05-private candidate facts; no selection authority is present."""

    identity: EpochIdentityPayload
    clocks: EpochClockPayload
    task: EpochTaskPayload
    rng_counter: torch.Tensor
    construction_admissible: torch.Tensor
    playback_admissible: torch.Tensor
    owner_fault_bits: torch.Tensor


@dataclass(frozen=True)
class ActionEpochD05AcceptedRows:
    """Epoch-owned authorization for the one currently active D05 writer."""

    accept_mask: torch.Tensor
    publication_ordinal: torch.Tensor

    def clone(self) -> "ActionEpochD05AcceptedRows":
        return ActionEpochD05AcceptedRows(
            accept_mask=self.accept_mask.clone(),
            publication_ordinal=self.publication_ordinal.clone(),
        )


@dataclass(frozen=True)
class ActionEpochRewardPaymentRows:
    """Current/previous real Reward payment facts aligned as one row per env."""

    valid: torch.Tensor
    shot_key: ActionEpochShotKey
    payment_step: torch.Tensor

    def clone(self) -> "ActionEpochRewardPaymentRows":
        return ActionEpochRewardPaymentRows(
            valid=self.valid.clone(),
            shot_key=self.shot_key.clone(),
            payment_step=self.payment_step.clone(),
        )


@dataclass(frozen=True)
class _RequiredPreviousPaidRows:
    """Validated R06 mailbox after-image retained for the local full join."""

    valid: torch.Tensor
    shot_key: ActionEpochShotKey
    publication_ordinal: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor


@dataclass(frozen=True)
class ActionEpochClosedRows:
    """Transient full-key close acknowledgement exposed only to bound R06."""

    valid: torch.Tensor
    shot_key: ActionEpochShotKey

    def clone(self) -> "ActionEpochClosedRows":
        return ActionEpochClosedRows(
            self.valid.clone(),
            self.shot_key.clone(),
        )


@dataclass(frozen=True)
class PackedDelta:
    names: tuple[str, ...]
    values: tuple[torch.Tensor, ...]

    def clone(self) -> "PackedDelta":
        return PackedDelta(self.names, tuple(value.clone() for value in self.values))


@dataclass(frozen=True)
class CommitEntry:
    sequence: int
    epoch: int  # publication order only
    transition: str
    before_version: int
    after_version: int
    delta: PackedDelta

    def clone(self) -> "CommitEntry":
        return replace(self, delta=self.delta.clone())


@dataclass(frozen=True)
class ActionEpochMaterializedDrain:
    """The sole host image minted by one frozen drain materialization."""

    entries: tuple[CommitEntry, ...]
    row_fault_bits: torch.Tensor
    milestone_i64: torch.Tensor
    milestone_f64: torch.Tensor


_DRAIN_PACK_VERSION = 5
_DRAIN_DTYPE_NBYTES = {
    torch.bool: 1,
    torch.int64: 8,
    torch.float32: 4,
    torch.float64: 8,
}


@dataclass(frozen=True)
class _DrainTensorLayout:
    """Static-version byte offset for one device tensor in a frozen drain."""

    version: int
    entry_index: int
    value_index: int
    name: str
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


def _single_d2h_packed_bytes(device_bytes: torch.Tensor) -> torch.Tensor:
    """Perform the drain's one physical device-to-host transfer."""

    return device_bytes.to(device="cpu", copy=True).contiguous()


_RECORD_DIRECT_TENSOR_NAMES = (
    "phase",
    "rng_counter",
    "current_task_slot",
    "publication_ordinal",
    "owner_fault_bits",
    "writes_started",
    "writes_committed",
    "launch_succeeded",
    "late_launch",
    "outcome_code",
    "reward_cycle_age",
    "reward_cycle_fault",
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


def _record_tensor_items(
    record: ActionEpochRecord,
) -> tuple[tuple[str, torch.Tensor], ...]:
    items: list[tuple[str, torch.Tensor]] = []
    for field in fields(ActionEpochShotKey):
        items.append((
            "identity.shot_key." + field.name,
            getattr(record.identity.shot_key, field.name),
        ))
    for name in (
        "scheduled_ordinal", "target_generation", "selected_cell",
        "candidate_identity",
    ):
        items.append(("identity." + name, getattr(record.identity, name)))
    for field in fields(EpochClockPayload):
        items.append(("clocks." + field.name, getattr(record.clocks, field.name)))
    items.extend((
        ("task.task_f32", record.task.task_f32),
        ("task.task_valid", record.task.task_valid),
    ))
    items.extend(
        (name, getattr(record, name)) for name in _RECORD_DIRECT_TENSOR_NAMES
    )
    return tuple(items)


def _record_from_tensor_values(
    template: ActionEpochRecord,
    values: tuple[torch.Tensor, ...],
) -> ActionEpochRecord:
    iterator = iter(values)

    def take() -> torch.Tensor:
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover - internal schema bug
            raise ActionEpochError("checkpoint record tensor image is truncated") from exc

    shot_key = ActionEpochShotKey(
        **{field.name: take() for field in fields(ActionEpochShotKey)}
    )
    identity = EpochIdentityPayload(
        shot_key=shot_key,
        scheduled_ordinal=take(),
        target_generation=take(),
        selected_cell=take(),
        candidate_identity=take(),
    )
    clocks = EpochClockPayload(
        **{field.name: take() for field in fields(EpochClockPayload)}
    )
    task = EpochTaskPayload(task_f32=take(), task_valid=take())
    direct = {name: take() for name in _RECORD_DIRECT_TENSOR_NAMES}
    try:
        next(iterator)
    except StopIteration:
        pass
    else:  # pragma: no cover - internal schema bug
        raise ActionEpochError("checkpoint record tensor image is extended")
    return ActionEpochRecord(
        epoch=template.epoch,
        version=template.version,
        identity=identity,
        clocks=clocks,
        task=task,
        diagnostic_unauthorized=True,
        **direct,
    )


def _require_exact_tensor_items(
    items: tuple[tuple[str, torch.Tensor], ...],
    specs: tuple[tuple[str, tuple[int, ...], torch.dtype], ...],
    *,
    device: torch.device,
    label: str,
    require_disjoint: bool = True,
) -> None:
    if tuple(name for name, _ in items) != tuple(name for name, _, _ in specs):
        raise ActionEpochError(label + " tensor names differ")
    occupied: list[tuple[int, int, int, str]] = []
    for (name, value), (_, shape, dtype) in zip(items, specs):
        if (
            type(value) is not torch.Tensor
            or value.device != device
            or value.dtype is not dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ActionEpochError(label + "." + name + " tensor ABI differs")
        pointer = value.untyped_storage().data_ptr()
        start = value.storage_offset() * value.element_size()
        end = start + value.numel() * value.element_size()
        if require_disjoint:
            for prior_pointer, prior_start, prior_end, prior_name in occupied:
                if (
                    pointer == prior_pointer
                    and start < prior_end
                    and prior_start < end
                ):
                    raise ActionEpochError(
                        label + "." + name + " aliases "
                        + label + "." + prior_name
                    )
        occupied.append((pointer, start, end, name))


@dataclass(frozen=True)
class _Publication:
    current: Optional[ActionEpochRecord]
    pending_log: tuple[CommitEntry, ...]


@dataclass
class _ActiveD05Transaction:
    rows: ActionEpochDueRows
    publication_ordinal: int
    base_version: int
    token: Optional[object] = None
    accept_mask: Optional[torch.Tensor] = None
    active_writer_kind: Optional[str] = None
    next_writer_ordinal: int = 0
    publication_started: bool = False


@dataclass(frozen=True)
class ActionEpochCheckpoint:
    commit_head: int
    drain_frontier: int
    next_epoch: int
    reset_generation: torch.Tensor
    reward_cycle_age: torch.Tensor
    current: ActionEpochRecord
    diagnostic_unauthorized: bool = True

    def clone(self) -> "ActionEpochCheckpoint":
        return replace(
            self,
            reset_generation=self.reset_generation.clone(),
            reward_cycle_age=self.reward_cycle_age.clone(),
            current=self.current.clone(),
        )


@dataclass(frozen=True)
class _ActionEpochCarryState:
    """Root-private, journal-free ActionEpoch state at one ACK boundary."""

    num_envs: int
    shot_slot_capacity: int
    commit_head: int
    drain_frontier: int
    next_epoch: int
    reward_ordinal: int
    last_motion_common_step: int
    current: ActionEpochRecord
    undrained_row_fault_bits: torch.Tensor
    action_uids_by_slot: torch.Tensor
    family_codes_by_slot: torch.Tensor
    last_r06_paid_payment_step: torch.Tensor
    current_payment_rows: ActionEpochRewardPaymentRows
    diagnostic_unauthorized: bool = True


_SHOT_EVENT_NAMES = (
    "event_mask",
    "shot_key.reset_generation",
    "shot_key.ball_generation",
    "shot_key.action_uid",
    "shot_key.action_slot",
    "shot_key.shot_index",
    "shot_key.task_identity",
    "shot_key.outcome_identity",
    "shot_key.ball_identity",
)


class ActionEpochOwner:
    """One row-wise public record, one private D05 transaction, one journal."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        shot_slot_capacity: int = 1,
        initial_reset_generation: Union[int, torch.Tensor] = 0,
        initial_reward_cycle_age: Union[int, torch.Tensor] = 0,
    ) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ActionEpochError("num_envs must be a positive exact int")
        if type(shot_slot_capacity) is not int or shot_slot_capacity < 1:
            raise ActionEpochError("shot_slot_capacity must be a positive exact int")
        self.num_envs = num_envs
        self.shot_slot_capacity = shot_slot_capacity
        self.device = torch.device(device)
        self._shot_shape = (num_envs, shot_slot_capacity)
        self._owner_shape = (*self._shot_shape, OWNER_COUNT)
        self._reset_generation = self._initial_env_row(
            initial_reset_generation, label="initial_reset_generation"
        )
        self._reward_cycle_age = self._initial_env_row(
            initial_reward_cycle_age, label="initial_reward_cycle_age"
        )
        self._reward_cycle_fault = torch.zeros(
            (num_envs,), dtype=torch.int64, device=self.device
        )
        self._next_epoch = 0
        self._publication = _Publication(None, ())
        self._commit_head = 0
        self._drain_frontier = 0
        self._pending_drain: Optional[tuple[int, int]] = None
        self._pending_drain_materialized = False
        self._drain_decoded_row_fault = False
        self._undrained_row_fault_bits = torch.zeros(
            (num_envs,), dtype=torch.int64, device=self.device
        )
        self.milestone = milestone_tensors.MilestoneTensorAccumulator(
            num_envs, self.device, shot_slot_capacity
        )
        self._poisoned = False
        self._operation_active = False
        self._genesis_activated = False
        self._reward_open = False
        self._reward_ordinal = 0
        self._active_d05: Optional[_ActiveD05Transaction] = None
        self._d05_owner: Optional[object] = None
        self._d05_candidate_projector: Optional[object] = None
        self._d05_accept_writers: Optional[tuple[object, object, object]] = None
        self._motion_owner: Optional[object] = None
        self._motion_projection: Optional[object] = None
        self._action_uids_by_slot = torch.empty(
            0, dtype=torch.int64, device=self.device
        )
        self._family_codes_by_slot = self._action_uids_by_slot.clone()
        self._motion_playback_owner: Optional[object] = None
        self._motion_playback: Optional[object] = None
        self._physical_launch_projection: Optional[object] = None
        self._physical_projection: Optional[object] = None
        self._r06_owner: Optional[object] = None
        self._r06_paid_projection: Optional[object] = None
        self._r06_outcome_projection: Optional[object] = None
        self._r06_consume_closed: Optional[object] = None
        self._fact_owner_identities: dict[str, object] = {}
        self._async_owner_identities: dict[str, object] = {}
        self._last_motion_common_step = -1
        self._last_r06_paid_payment_step = torch.full(
            (num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._current_payment_rows = self._empty_payment_rows()
        self._current_closed_rows: Optional[ActionEpochClosedRows] = None
        self._selected_reset = selected_reset.SelectedResetTransaction(
            num_envs=num_envs, device=self.device
        )
        self._lean_carry_coordinator = None
        self._lock = threading.RLock()

    def _initial_env_row(
        self, value: Union[int, torch.Tensor], *, label: str
    ) -> torch.Tensor:
        if type(value) is int:
            if value < 0 or value > _I64_MAX:
                raise ActionEpochError(label + " must be a non-negative int64")
            return torch.full(
                (self.num_envs,), value, dtype=torch.int64, device=self.device
            )
        return self._tensor(
            value, label=label, shape=(self.num_envs,), dtype=torch.int64
        )

    def _empty_payment_rows(self) -> ActionEpochRewardPaymentRows:
        return ActionEpochRewardPaymentRows(
            valid=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            shot_key=row_identity.empty_action_epoch_shot_key(
                (self.num_envs,), device=self.device
            ),
            payment_step=torch.full(
                (self.num_envs,), -1, dtype=torch.int64, device=self.device
            ),
        )

    @property
    def diagnostic_unauthorized(self) -> bool:
        return True

    @property
    def commit_head(self) -> int:
        return self._commit_head

    @property
    def drain_frontier(self) -> int:
        return self._drain_frontier

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def current(self) -> ActionEpochRecord:
        with self._lock:
            if self._publication.current is None:
                raise ActionEpochError("no ActionEpoch record exists")
            return self._publication.current.clone()

    def project_current_shot(self) -> ActionEpochCurrentShotProjection:
        """Gather the current device row without cloning the full record.

        This is a same-stream read projection, not an admission verdict.  An
        invalid internal slot is represented by ``slot_valid=False`` and
        neutral ``-1`` values so each consumer can route only that row to its
        existing named fault without a host synchronization.
        """

        with self._lock:
            record = self._publication.current
            if record is None:
                raise ActionEpochError("no ActionEpoch record exists")
            slot = record.current_task_slot
            slot_valid = slot.ge(0) & slot.lt(self.shot_slot_capacity)
            safe_slot = slot.clamp(0, self.shot_slot_capacity - 1)
            index = safe_slot[:, None]

            def selected(value: torch.Tensor) -> torch.Tensor:
                gathered = torch.gather(value, 1, index).squeeze(1)
                return torch.where(
                    slot_valid,
                    gathered,
                    torch.full_like(gathered, -1),
                ).detach()

            return ActionEpochCurrentShotProjection(
                slot_valid=slot_valid.detach(),
                phase=selected(record.phase),
                shot_key=ActionEpochShotKey(
                    **{
                        field.name: selected(
                            getattr(record.identity.shot_key, field.name)
                        )
                        for field in fields(ActionEpochShotKey)
                    }
                ),
                publication_ordinal=selected(record.publication_ordinal),
            )

    def project_recovery_postphysics_activity_mask(
        self,
        *,
        owner: object,
        motion_cadence_tick: torch.Tensor,
    ) -> torch.Tensor:
        """Project keyed rows due for one fixed-window R07 producer cell.

        Recovery age is computed in Motion's per-environment cadence domain;
        selected reset can rewind those rows without rewinding the global
        counter.  Physical independently owns and validates the global cached
        host-verdict chronology, so duplicating that scalar here would add no
        row fact or safety invariant.
        Outcome settlement is a Reward-consumer condition and deliberately
        does not gate production.
        """

        with self._lock:
            self._healthy()
            record = self._publication.current
            if (
                record is None
                or self._async_owner_identities.get("physical_ball") is not owner
            ):
                raise ActionEpochError(
                    "recovery postphysics activity owner identity differs"
                )
            if (
                type(motion_cadence_tick) is not torch.Tensor
                or motion_cadence_tick.dtype != torch.int64
                or motion_cadence_tick.device != self.device
                or tuple(motion_cadence_tick.shape) != (self.num_envs,)
                or not motion_cadence_tick.is_contiguous()
            ):
                raise ActionEpochError(
                    "recovery postphysics Motion cadence tick ABI differs"
                )
            motion_slot = self._owner_slot("motion")
            phase = record.phase
            deadline = record.clocks.deadline_tick
            recovery_age = motion_cadence_tick[:, None] - deadline
            return (
                row_identity.action_epoch_shot_key_valid(
                    record.identity.shot_key
                )
                & record.writes_started[:, :, motion_slot]
                & record.writes_committed[:, :, motion_slot]
                & deadline.ge(0)
                & motion_cadence_tick[:, None].ge(deadline)
                & recovery_age.ge(r07_device.RECOVERY_START_AGE_TICK)
                & recovery_age.le(r07_device.RECOVERY_END_AGE_TICK)
                & (
                    phase.eq(PHASE_REVEAL_COMMITTED)
                    | phase.eq(PHASE_LAUNCH_SETTLED)
                    | phase.eq(PHASE_OUTCOME_SETTLED)
                )
            ).detach()

    def snapshot_idle_observation_chronology(
        self, *, owner: object
    ) -> ActionEpochIdleObservationChronology:
        """Clone only reset chronology for the unkeyed observation."""

        with self._lock:
            self._healthy()
            record = self._publication.current
            if (
                record is None
                or self._fact_owner_identities.get("r07_recovery") is not owner
            ):
                raise ActionEpochError(
                    "idle observation owner identity differs"
                )
            return ActionEpochIdleObservationChronology(
                epoch_version=record.version,
                reset_generation=record.reset_generation.detach().clone(),
            )

    @contextmanager
    def _operation(
        self, label: str, *, allow_pending_drain: bool = False
    ) -> Iterator[None]:
        del label
        with self._lock:
            carry_txn._require_leaf_mutable(self)
            if self._pending_drain is not None and not allow_pending_drain:
                raise ActionEpochError("mutation cannot overlap a frozen drain")
            if self._operation_active:
                self._poison_locked(
                    reason=_FAULT_ILLEGAL_REPLAY,
                    owner_kind="r05_runtime",
                    transition="REENTRANT_POISON",
                )
                raise ActionEpochError("reentrant ActionEpoch operation poisoned")
            self._operation_active = True
            try:
                yield
            finally:
                self._operation_active = False

    def _healthy(self) -> None:
        if self._poisoned:
            raise ActionEpochError("ActionEpoch owner is poisoned")

    def _latch_device_row_fault(
        self, fault: torch.Tensor, *, reason_bit: int
    ) -> torch.Tensor:
        """Latch one named device-only cause and return rows safe to mutate."""

        if (
            type(reason_bit) is not int
            or reason_bit <= 0
            or reason_bit & (reason_bit - 1)
            or reason_bit & ~_KNOWN_ROW_FAULT_MASK
        ):
            raise ActionEpochError("device row fault reason bit differs")
        self._undrained_row_fault_bits = torch.bitwise_or(
            self._undrained_row_fault_bits,
            torch.where(fault, reason_bit, 0),
        )
        return self._undrained_row_fault_bits.eq(0)

    def latch_runtime_row_fault(
        self,
        owner_kind: str,
        reason_bit: int,
        rows: torch.Tensor,
        *,
        owner: object,
    ) -> torch.Tensor:
        """Latch one pre-registered runtime-owner fault at the packed boundary.

        This narrow ingress deliberately remains callable from an exact owner
        projection that Epoch is currently pulling, so it cannot use the
        ordinary non-reentrant mutation guard.  The owner identity, owner/bit
        registry, tensor ABI, carry lease, and frozen-drain boundary are still
        exact; callers cannot select another owner's or a compound reason.
        """

        with self._lock:
            carry_txn._require_leaf_mutable(self)
            self._healthy()
            if self._pending_drain is not None:
                raise ActionEpochError("mutation cannot overlap a frozen drain")
            allowed_bits = (
                _RUNTIME_ROW_FAULT_BITS_BY_OWNER.get(owner_kind)
                if type(owner_kind) is str
                else None
            )
            expected_owner = None
            if type(owner_kind) is str:
                if owner_kind == "motion":
                    expected_owner = self._motion_playback_owner
                elif owner_kind == "r03_strike_fact":
                    expected_owner = self._fact_owner_identities.get(owner_kind)
                else:
                    expected_owner = self._async_owner_identities.get(owner_kind)
            exact_owner_binding = expected_owner is owner
            if owner_kind == "r06_landing_outcome":
                exact_owner_binding &= (
                    self._fact_owner_identities.get(owner_kind) is owner
                )
            if (
                allowed_bits is None
                or owner is None
                or not exact_owner_binding
            ):
                raise ActionEpochError("runtime row fault owner binding differs")
            if type(reason_bit) is not int or reason_bit not in allowed_bits:
                raise ActionEpochError("runtime row fault reason bit differs")
            exact_rows = self._borrow_tensor(
                rows,
                label="runtime row fault rows",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            return self._latch_device_row_fault(
                exact_rows, reason_bit=reason_bit
            )

    def _tensor(
        self,
        value: object,
        *,
        label: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            type(value) is not torch.Tensor
            or value.device != self.device
            or value.dtype is not dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ActionEpochError(
                f"{label} must be contiguous {dtype} on {self.device} with shape {shape}"
            )
        return value.detach().clone().contiguous()

    def _borrow_tensor(
        self,
        value: object,
        *,
        label: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Validate one synchronous owner projection without copying it.

        The caller must consume the view inside the current owner operation;
        it must never store the tensor in a record or journal entry.  This is
        the narrow seam used by Physical's one-shot postphysics packet.
        """

        if (
            type(value) is not torch.Tensor
            or value.device != self.device
            or value.dtype is not dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ActionEpochError(
                f"{label} must be contiguous {dtype} on {self.device} with shape {shape}"
            )
        return value.detach()

    @staticmethod
    def _owner_slot(owner_kind: str) -> int:
        if type(owner_kind) is not str or owner_kind not in OWNER_ORDER:
            raise ActionEpochError("owner_kind is not one of the fixed seven slots")
        return OWNER_ORDER.index(owner_kind)

    def _append(
        self,
        record: ActionEpochRecord,
        *,
        transition: str,
        names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        epoch: Optional[int] = None,
        journal_epoch: Optional[int] = None,
        changes: Optional[dict[str, object]] = None,
    ) -> ActionEpochRecord:
        if self._pending_drain is not None:
            raise ActionEpochError("mutation cannot overlap a frozen drain")
        if not names or len(names) != len(values) or len(set(names)) != len(names):
            raise ActionEpochError("packed delta names/values differ")
        if any(type(value) is not torch.Tensor for value in values):
            raise ActionEpochError("packed delta values must be tensors")
        before = record.version
        after = replace(
            record,
            epoch=record.epoch if epoch is None else epoch,
            version=before + 1,
            **({} if changes is None else changes),
        )
        entry = CommitEntry(
            sequence=self._commit_head,
            epoch=after.epoch if journal_epoch is None else journal_epoch,
            transition=transition,
            before_version=before,
            after_version=after.version,
            delta=PackedDelta(names, values),
        )
        self._publication = _Publication(
            after, (*self._publication.pending_log, entry)
        )
        self._commit_head += 1
        return after

    @staticmethod
    def _event(
        mask: torch.Tensor, key: ActionEpochShotKey
    ) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
        return _SHOT_EVENT_NAMES, (
            mask,
            *(getattr(key, field.name) for field in fields(ActionEpochShotKey)),
        )

    def _poison_locked(
        self, *, reason: int, owner_kind: str, transition: str
    ) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        record = self._publication.current
        if record is None:
            return
        slot = self._owner_slot(owner_kind)
        faults = record.owner_fault_bits.clone()
        faults[:, :, slot] = torch.bitwise_or(
            faults[:, :, slot], torch.full_like(faults[:, :, slot], reason)
        )
        phase = torch.where(
            ~record.phase.eq(PHASE_IDLE),
            torch.full_like(record.phase, PHASE_POISONED),
            record.phase,
        )
        poison = torch.where(
            ~record.phase.eq(PHASE_IDLE),
            torch.full_like(record.poison_reason, reason),
            record.poison_reason,
        )
        self._append(
            record,
            transition=transition,
            names=("phase", "owner_fault_bits", "poison_reason"),
            values=(phase, faults, poison),
            changes={
                "phase": phase,
                "owner_fault_bits": faults,
                "poison_reason": poison,
            },
        )

    def _milestone_after_business_write(
        self, owner_kind: str, method_name: str, *args: object
    ) -> None:
        """Write telemetry only after its business write; any failure is terminal."""

        try:
            getattr(self.milestone, method_name)(*args)
        except BaseException:
            self._poison_locked(
                reason=_FAULT_MILESTONE_TELEMETRY,
                owner_kind=owner_kind,
                transition="OWNER_WRITE_POISON:" + owner_kind,
            )
            raise

    def poison_owner_write(
        self, owner_kind: str, reason_code: int, *, owner: object
    ) -> ActionEpochRecord:
        if type(reason_code) is not int or not (0 < reason_code <= _I64_MAX):
            raise ActionEpochError("reason_code must be a positive exact int64")
        with self._operation("poison owner write"):
            self._owner_slot(owner_kind)
            expected_owner = (
                self._d05_owner
                if owner_kind == "r05_runtime"
                else self._fact_owner_identities.get(owner_kind)
            )
            if expected_owner is None or owner is not expected_owner:
                raise ActionEpochError("owner-write poison attribution differs")
            self._poison_locked(
                reason=reason_code,
                owner_kind=owner_kind,
                transition="OWNER_WRITE_POISON:" + owner_kind,
            )
            assert self._publication.current is not None
            return self._publication.current.clone()

    def _canonical_genesis(self) -> bool:
        current = self._publication.current
        return (
            self._genesis_activated
            and current is not None
            and current.epoch == -1
            and current.version == 0
            and len(self._publication.pending_log) == 1
            and self._publication.pending_log[0].transition == "RESET_GENESIS_IDLE"
            and self._active_d05 is None
            and not self._reward_open
            and not self._selected_reset.active
        )

    @staticmethod
    def _exact_bound(owner: object, name: str) -> object:
        method = getattr(owner, name, None)
        direct = getattr(type(owner), name, None)
        if (
            not callable(method)
            or not callable(direct)
            or getattr(method, "__self__", None) is not owner
            or getattr(method, "__func__", None) is not direct
        ):
            raise ActionEpochError(name + " must be one exact bound method")
        return method

    @staticmethod
    def _exact_d05_owner_type() -> type:
        # The runtime factory intentionally constructs DeviceR05Owner from
        # this one flat module name.  Do not fall back to a package alias:
        # importing both would create two distinct class identities.
        d05 = importlib.import_module(
            "action_ball_continuous_runtime_transaction_device"
        )
        owner_type = getattr(d05, "DeviceR05Owner", None)
        if type(owner_type) is not type:
            raise ActionEpochError("canonical DeviceR05Owner type is unavailable")
        return owner_type

    def bind_d05_accept_writers(
        self, *, motion_write: object, racket_write: object, r05_write: object
    ) -> None:
        with self._operation("bind D05"):
            if not self._canonical_genesis() or self._d05_accept_writers is not None:
                raise ActionEpochError("D05 writers require canonical genesis IDLE")
            writers = (motion_write, racket_write, r05_write)
            writer_names = (
                "_commit_action_epoch_motion_write",
                "_commit_action_epoch_racket_write",
                "_commit_action_epoch_r05_write",
            )
            d05_owner = getattr(r05_write, "__self__", None)
            d05_owner_type = self._exact_d05_owner_type()
            if any(
                not callable(writer)
                or type(d05_owner) is not d05_owner_type
                or getattr(writer, "__self__", None) is not d05_owner
                or getattr(writer, "__func__", None)
                is not getattr(d05_owner_type, writer_name, None)
                for writer, writer_name in zip(writers, writer_names)
            ):
                raise ActionEpochError("D05 writers must be exact bound callables")
            projector = self._exact_bound(
                d05_owner, "require_owned_action_epoch_candidate"
            )
            self._d05_owner = d05_owner
            self._d05_candidate_projector = projector
            self._d05_accept_writers = writers

    def require_active_d05_accepted_rows(
        self, token: object, *, owner_kind: str
    ) -> ActionEpochD05AcceptedRows:
        """Return the sole full-N authorization during one exact writer window."""

        with self._lock:
            tx = self._active_d05
            if (
                owner_kind not in REVEAL_WRITE_OWNER_ORDER
                or tx is None
                or tx.token is not token
                or tx.active_writer_kind != owner_kind
                or tx.accept_mask is None
            ):
                raise ActionEpochError("D05 accepted rows token/writer lifetime differs")
            return ActionEpochD05AcceptedRows(
                accept_mask=tx.accept_mask.clone(),
                publication_ordinal=torch.full(
                    self._shot_shape,
                    tx.publication_ordinal,
                    dtype=torch.int64,
                    device=self.device,
                ),
            )

    def bind_fact_owner(self, owner_kind: str, owner: object) -> None:
        with self._operation("bind fact owner"):
            self._owner_slot(owner_kind)
            if (
                owner is None
                or owner_kind in self._fact_owner_identities
                or not self._canonical_genesis()
            ):
                raise ActionEpochError("fact owner binding differs")
            self._fact_owner_identities[owner_kind] = owner

    def bind_async_owner(self, owner_kind: str, owner: object) -> None:
        with self._operation("bind async owner"):
            if (
                owner_kind not in ("physical_ball", "r06_landing_outcome")
                or owner is None
                or owner_kind in self._async_owner_identities
                or not self._canonical_genesis()
                or self._fact_owner_identities.get(owner_kind) is not owner
            ):
                raise ActionEpochError("async owner binding differs")
            self._async_owner_identities[owner_kind] = owner
            if owner_kind == "physical_ball":
                self._physical_launch_projection = self._exact_bound(
                    owner, "action_epoch_r06_launch_projection"
                )
                projection = getattr(
                    owner,
                    "require_owned_action_epoch_r06_postphysics_projection",
                    None,
                )
                direct = getattr(
                    type(owner),
                    "require_owned_action_epoch_r06_postphysics_projection",
                    None,
                )
                if (
                    callable(projection)
                    and callable(direct)
                    and getattr(projection, "__self__", None) is owner
                    and getattr(projection, "__func__", None) is direct
                ):
                    self._physical_projection = projection
            if owner_kind == "r06_landing_outcome":
                self._r06_owner = owner
                self._r06_paid_projection = self._exact_bound(
                    owner, "project_previous_paid_action_epoch_rows"
                )
                self._r06_consume_closed = self._exact_bound(
                    owner, "consume_closed_action_epoch_rows"
                )
                outcome = getattr(owner, "project_current_action_epoch_outcome_rows", None)
                direct = getattr(type(owner), "project_current_action_epoch_outcome_rows", None)
                if (
                    callable(outcome)
                    and callable(direct)
                    and getattr(outcome, "__self__", None) is owner
                    and getattr(outcome, "__func__", None) is direct
                ):
                    self._r06_outcome_projection = outcome

    def bind_motion_cadence_owner(self, owner: object) -> None:
        """Cold-bind the sole full-N Motion due/close producer."""

        with self._operation("bind Motion cadence"):
            if owner is None or self._motion_owner is not None or not self._canonical_genesis():
                raise ActionEpochError("Motion cadence owner binding differs")
            self._motion_projection = self._exact_bound(
                owner, "project_current_action_epoch_rows"
            )
            project_catalog = getattr(
                owner, "project_action_stroke_family_catalog", None
            )
            if project_catalog is not None:
                direct = getattr(
                    type(owner), "project_action_stroke_family_catalog", None
                )
                if (
                    not callable(project_catalog)
                    or getattr(project_catalog, "__self__", None) is not owner
                    or getattr(project_catalog, "__func__", None) is not direct
                ):
                    raise ActionEpochError("Motion action catalog method differs")
                catalog = project_catalog()
                if type(catalog) is not action_strata.ActionStrokeFamilyCatalog:
                    raise ActionEpochError("Motion action catalog projection differs")
                self._action_uids_by_slot = torch.tensor(
                    catalog.action_uids, dtype=torch.int64, device=self.device
                )
                self._family_codes_by_slot = torch.tensor(
                    catalog.family_codes, dtype=torch.int64, device=self.device
                )
            self._motion_owner = owner

    def bind_motion_playback_owner(self, owner: object) -> None:
        """Bind Motion's leaf playback transition; it is not cadence authority."""

        with self._operation("bind Motion playback"):
            if self._motion_playback is not None or not self._canonical_genesis():
                raise ActionEpochError("Motion playback owner binding differs")
            self._motion_playback = self._exact_bound(
                owner, "action_epoch_playback_transition_mask"
            )
            self._motion_playback_owner = owner

    def bind_selected_reset_owner(self, owner: object) -> None:
        with self._operation("bind selected reset"):
            try:
                self._selected_reset.bind(
                    owner, canonical_genesis_idle=self._canonical_genesis()
                )
            except selected_reset.SelectedResetProtocolError as exc:
                raise ActionEpochError(str(exc)) from exc

    def activate_reset_genesis(
        self, *, selected_mask: torch.Tensor, reset_generation: torch.Tensor
    ) -> None:
        with self._operation("activate reset genesis"):
            if self._genesis_activated or self._publication.current is not None:
                raise ActionEpochError("reset genesis is stale or duplicated")
            selected = self._tensor(
                selected_mask,
                label="selected_mask",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            generation = self._tensor(
                reset_generation,
                label="reset_generation",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            genesis_fault = ~selected | generation.ne(self._reset_generation)
            self._undrained_row_fault_bits = torch.bitwise_or(
                self._undrained_row_fault_bits,
                torch.where(
                    genesis_fault, ROW_FAULT_RESET_GENESIS_CONTRACT, 0
                ),
            )
            self._genesis_activated = True
            empty_i64 = torch.full(
                self._shot_shape, -1, dtype=torch.int64, device=self.device
            )
            empty_bool = torch.zeros(
                self._shot_shape, dtype=torch.bool, device=self.device
            )
            identity = EpochIdentityPayload(
                shot_key=row_identity.empty_action_epoch_shot_key(
                    self._shot_shape, device=self.device
                ),
                scheduled_ordinal=empty_i64.clone(),
                target_generation=empty_i64.clone(),
                selected_cell=empty_i64.clone(),
                candidate_identity=empty_i64.clone(),
            )
            clocks = EpochClockPayload(
                **{field.name: empty_i64.clone() for field in fields(EpochClockPayload)}
            )
            record = ActionEpochRecord(
                epoch=-1,
                version=0,
                phase=torch.full_like(empty_i64, PHASE_IDLE),
                identity=identity,
                clocks=clocks,
                task=EpochTaskPayload(
                    task_f32=torch.zeros(
                        (*self._shot_shape, TASK_F32_WIDTH),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    task_valid=empty_bool.clone(),
                ),
                rng_counter=empty_i64.clone(),
                current_task_slot=torch.zeros(
                    self.num_envs, dtype=torch.int64, device=self.device
                ),
                publication_ordinal=empty_i64.clone(),
                owner_fault_bits=torch.zeros(
                    self._owner_shape, dtype=torch.int64, device=self.device
                ),
                writes_started=torch.zeros(
                    self._owner_shape, dtype=torch.bool, device=self.device
                ),
                writes_committed=torch.zeros(
                    self._owner_shape, dtype=torch.bool, device=self.device
                ),
                launch_succeeded=empty_bool.clone(),
                late_launch=empty_bool.clone(),
                outcome_code=empty_i64.clone(),
                reward_cycle_age=self._reward_cycle_age.clone(),
                reward_cycle_fault=self._reward_cycle_fault.clone(),
                reward_cycle_open=torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                ),
                reward_due=torch.zeros(
                    (self.num_envs, REWARD_CONSUMER_COUNT),
                    dtype=torch.bool,
                    device=self.device,
                ),
                reward_paid=torch.zeros(
                    (self.num_envs, REWARD_CONSUMER_COUNT),
                    dtype=torch.bool,
                    device=self.device,
                ),
                fact_valid_bits=torch.zeros(
                    self._owner_shape, dtype=torch.int64, device=self.device
                ),
                fact_source_step=torch.full(
                    self._owner_shape, -1, dtype=torch.int64, device=self.device
                ),
                fact_f32=torch.zeros(
                    (*self._owner_shape, OWNER_FACT_F32_WIDTH),
                    dtype=torch.float32,
                    device=self.device,
                ),
                reset_generation=generation,
                reset_selected_mask=selected,
                motion_playback_started=empty_bool.clone(),
                motion_close_reason=empty_i64.clone(),
                settlement_step=empty_i64.clone(),
                payment_step=empty_i64.clone(),
                poison_reason=torch.zeros_like(empty_i64),
            )
            self._publication = _Publication(
                record,
                (
                    CommitEntry(
                        sequence=0,
                        epoch=-1,
                        transition="RESET_GENESIS_IDLE",
                        before_version=-1,
                        after_version=0,
                        delta=PackedDelta(
                            ("reset_generation", "reset_selected_mask"),
                            (generation, selected),
                        ),
                    ),
                ),
            )
            self._reset_generation = record.reset_generation
            self._reward_cycle_age = record.reward_cycle_age
            self._reward_cycle_fault = record.reward_cycle_fault
            self._commit_head = 1

    # ------------------------------------------------------------------
    # Row/key validation and masked immutable after-images

    def _require_identity(self, value: object) -> EpochIdentityPayload:
        if type(value) is not EpochIdentityPayload:
            raise ActionEpochError("candidate identity must be exact EpochIdentityPayload")
        try:
            key = row_identity.require_action_epoch_shot_key(
                value.shot_key,
                shape=self._shot_shape,
                device=self.device,
                label="candidate.identity.shot_key",
            )
        except row_identity.ActionEpochShotKeyError as exc:
            raise ActionEpochError(str(exc)) from exc
        tensors = {
            field.name: self._tensor(
                getattr(value, field.name),
                label="candidate.identity." + field.name,
                shape=self._shot_shape,
                dtype=torch.int64,
            )
            for field in fields(EpochIdentityPayload)
            if field.name != "shot_key"
        }
        return EpochIdentityPayload(shot_key=key.clone(), **tensors)

    def _require_clocks(self, value: object) -> EpochClockPayload:
        if type(value) is not EpochClockPayload:
            raise ActionEpochError("candidate clocks must be exact EpochClockPayload")
        return EpochClockPayload(
            **{
                field.name: self._tensor(
                    getattr(value, field.name),
                    label="candidate.clocks." + field.name,
                    shape=self._shot_shape,
                    dtype=torch.int64,
                )
                for field in fields(EpochClockPayload)
            }
        )

    def _require_task(self, value: object) -> EpochTaskPayload:
        if type(value) is not EpochTaskPayload:
            raise ActionEpochError("candidate task must be exact EpochTaskPayload")
        return EpochTaskPayload(
            task_f32=self._tensor(
                value.task_f32,
                label="candidate.task.task_f32",
                shape=(*self._shot_shape, TASK_F32_WIDTH),
                dtype=torch.float32,
            ),
            task_valid=self._tensor(
                value.task_valid,
                label="candidate.task.task_valid",
                shape=self._shot_shape,
                dtype=torch.bool,
            ),
        )

    def _require_candidate(
        self, value: object
    ) -> ActionEpochD05CandidateProjection:
        if type(value) is not ActionEpochD05CandidateProjection:
            raise ActionEpochError(
                "D05 projector must return exact ActionEpochD05CandidateProjection"
            )
        return ActionEpochD05CandidateProjection(
            identity=self._require_identity(value.identity),
            clocks=self._require_clocks(value.clocks),
            task=self._require_task(value.task),
            rng_counter=self._tensor(
                value.rng_counter,
                label="candidate.rng_counter",
                shape=self._shot_shape,
                dtype=torch.int64,
            ),
            construction_admissible=self._tensor(
                value.construction_admissible,
                label="candidate.construction_admissible",
                shape=self._shot_shape,
                dtype=torch.bool,
            ),
            playback_admissible=self._tensor(
                value.playback_admissible,
                label="candidate.playback_admissible",
                shape=self._shot_shape,
                dtype=torch.bool,
            ),
            owner_fault_bits=self._tensor(
                value.owner_fault_bits,
                label="candidate.owner_fault_bits",
                shape=(*self._shot_shape, OWNER_COUNT),
                dtype=torch.int64,
            ),
        )

    @staticmethod
    def _where_key(
        mask: torch.Tensor, yes: ActionEpochShotKey, no: ActionEpochShotKey
    ) -> ActionEpochShotKey:
        return ActionEpochShotKey(
            **{
                field.name: torch.where(
                    mask, getattr(yes, field.name), getattr(no, field.name)
                )
                for field in fields(ActionEpochShotKey)
            }
        )

    @staticmethod
    def _where_identity(
        mask: torch.Tensor, yes: EpochIdentityPayload, no: EpochIdentityPayload
    ) -> EpochIdentityPayload:
        return EpochIdentityPayload(
            shot_key=ActionEpochOwner._where_key(mask, yes.shot_key, no.shot_key),
            **{
                field.name: torch.where(
                    mask, getattr(yes, field.name), getattr(no, field.name)
                )
                for field in fields(EpochIdentityPayload)
                if field.name != "shot_key"
            },
        )

    @staticmethod
    def _where_clocks(
        mask: torch.Tensor, yes: EpochClockPayload, no: EpochClockPayload
    ) -> EpochClockPayload:
        return EpochClockPayload(
            **{
                field.name: torch.where(
                    mask, getattr(yes, field.name), getattr(no, field.name)
                )
                for field in fields(EpochClockPayload)
            }
        )

    @staticmethod
    def _where_task(
        mask: torch.Tensor, yes: EpochTaskPayload, no: EpochTaskPayload
    ) -> EpochTaskPayload:
        return EpochTaskPayload(
            task_f32=torch.where(mask[..., None], yes.task_f32, no.task_f32),
            task_valid=torch.where(mask, yes.task_valid, no.task_valid),
        )

    def _current_slot_mask(self, row_mask: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(
            self.shot_slot_capacity, dtype=torch.int64, device=self.device
        )[None, :]
        assert self._publication.current is not None
        return row_mask[:, None] & slots.eq(
            self._publication.current.current_task_slot[:, None]
        )

    def _gather_current(self, value: torch.Tensor) -> torch.Tensor:
        assert self._publication.current is not None
        slot = self._publication.current.current_task_slot[:, None]
        tail = (1,) * (value.ndim - 2)
        index = slot.reshape((self.num_envs, 1) + tail).expand(
            (self.num_envs, 1) + tuple(value.shape[2:])
        )
        return value.gather(1, index).squeeze(1)

    def _gather_current_key(self, key: ActionEpochShotKey) -> ActionEpochShotKey:
        return ActionEpochShotKey(
            **{
                field.name: self._gather_current(getattr(key, field.name))
                for field in fields(ActionEpochShotKey)
            }
        )

    def _require_previous_paid_rows(
        self, value: object, *, label: str
    ) -> _RequiredPreviousPaidRows:
        try:
            from .action_ball_landing_outcome_device import (
                PreviousPaidActionEpochRows,
            )
        except ImportError:  # pragma: no cover - focused standalone import
            from action_ball_landing_outcome_device import PreviousPaidActionEpochRows
        if type(value) is not PreviousPaidActionEpochRows:
            raise ActionEpochError(label + " must be exact PreviousPaidActionEpochRows")
        try:
            key = row_identity.require_action_epoch_shot_key(
                value.shot_key,
                shape=(self.num_envs,),
                device=self.device,
                label=label + ".shot_key",
            )
        except row_identity.ActionEpochShotKeyError as exc:
            raise ActionEpochError(str(exc)) from exc
        return _RequiredPreviousPaidRows(
            valid=self._tensor(
                value.valid,
                label=label + ".valid",
                shape=(self.num_envs,),
                dtype=torch.bool,
            ),
            shot_key=key.clone(),
            publication_ordinal=self._tensor(
                value.publication_ordinal,
                label=label + ".publication_ordinal",
                shape=(self.num_envs,),
                dtype=torch.int64,
            ),
            settlement_step=self._tensor(
                value.settlement_step,
                label=label + ".settlement_step",
                shape=(self.num_envs,),
                dtype=torch.int64,
            ),
            payment_step=self._tensor(
                value.payment_step,
                label=label + ".payment_step",
                shape=(self.num_envs,),
                dtype=torch.int64,
            ),
        )

    # ------------------------------------------------------------------
    # Motion after-command close/due and D05-private opportunity transaction

    def prepare_after_command_rows(self) -> Optional[ActionEpochDueRows]:
        """Freeze a real D05 opportunity, or advance an exactly idle tick."""

        with self._operation("prepare after command rows"):
            self._healthy()
            record = self._publication.current
            if (
                record is None
                or not self._genesis_activated
                or self._active_d05 is not None
                or self._reward_open
                or self._selected_reset.active
                or self._pending_drain is not None
                or self._motion_projection is None
                or self._r06_paid_projection is None
                or self._r06_consume_closed is None
            ):
                raise ActionEpochError("after-command row boundary is not quiescent/bound")
            projection = self._motion_projection()
            common_step = getattr(projection, "common_step", None)
            if type(common_step) is not int or common_step <= self._last_motion_common_step:
                raise ActionEpochError("Motion current projection chronology differs")
            episode_tick = self._tensor(
                getattr(projection, "episode_tick", None),
                label="Motion.episode_tick",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            due = self._tensor(
                getattr(projection, "reveal_due", None),
                label="Motion.reveal_due",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            closed = self._tensor(
                getattr(projection, "closed_mask", None),
                label="Motion.closed_mask",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            close_reason_rows = self._tensor(
                getattr(projection, "close_reason", None),
                label="Motion.close_reason",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            paid = self._require_previous_paid_rows(
                self._r06_paid_projection(), label="R06.previous_paid_rows"
            )
            current_key = self._gather_current_key(record.identity.shot_key)
            current_key_valid = row_identity.action_epoch_shot_key_valid(
                current_key
            )
            current_phase = self._gather_current(record.phase)
            motion_slot = self._owner_slot("motion")
            current_motion_committed = self._gather_current(
                record.writes_started[:, :, motion_slot]
                & record.writes_committed[:, :, motion_slot]
            )
            current_deadline = self._gather_current(record.clocks.deadline_tick)
            r07_slot = self._owner_slot("r07_recovery")
            r07_valid = self._gather_current(
                record.fact_valid_bits[:, :, r07_slot]
            )
            r07_fault = self._gather_current(
                record.owner_fault_bits[:, :, r07_slot]
            )
            r07_source = self._gather_current(
                record.fact_source_step[:, :, r07_slot]
            )
            r07_fact = self._gather_current(
                record.fact_f32[:, :, r07_slot, :]
            )
            r07_terminal_due = (
                current_key_valid
                & current_motion_committed
                & action_epoch_open_shot_phase_mask(current_phase)
                & current_deadline.ge(0)
                # The prior post-physics cell at cadence age 77 is visible at
                # the next Motion boundary, local episode tick D+78.
                & episode_tick.eq(
                    current_deadline
                    + r07_device.RECOVERY_END_AGE_TICK
                    + 1
                )
            )
            required_r07_bits = (
                r07_device.R07_EPOCH_FACT_PRESENT
                | r07_device.R07_EPOCH_FACT_NUMERICALLY_VALID
            )
            # Motion cadence and the environment source step have a constant
            # per-row offset for one reset generation.  Recover the only global
            # source that can name cadence age 77 instead of accepting any old
            # retained producer cell.  At the exact D+78 join this simplifies
            # to ``common_step - 1``; a later R06 payment retains the same cell
            # while both clocks advance together.
            expected_r07_terminal_source = (
                common_step
                - episode_tick
                + current_deadline
                + r07_device.RECOVERY_END_AGE_TICK
            )
            r07_terminal_fact_clean = (
                r07_valid.eq(required_r07_bits)
                & r07_fault.eq(0)
                & r07_source.eq(expected_r07_terminal_source)
                & r07_fact[:, 3].eq(1.0)
                & r07_fact[:, 4].eq(0.0)
                & r07_fact[:, 6].eq(
                    float(r07_device.RECOVERY_END_AGE_TICK)
                )
                & torch.isfinite(r07_fact).all(dim=1)
            )
            r07_terminal_fault = r07_terminal_due & ~r07_terminal_fact_clean
            r07_window_complete = (
                current_key_valid
                & current_motion_committed
                & action_epoch_open_shot_phase_mask(current_phase)
                & current_deadline.ge(0)
                & episode_tick.ge(
                    current_deadline
                    + r07_device.RECOVERY_END_AGE_TICK
                    + 1
                )
                & r07_terminal_fact_clean
            )
            # This exact D+78 check is independent of fact publication and
            # first-ready joins.  It never inspects Reward eligibility, ready,
            # score, or component thresholds.
            r07_terminal_safe = self._latch_device_row_fault(
                r07_terminal_fault,
                reason_bit=ROW_FAULT_R07_TERMINAL_FACT_CONTRACT,
            )
            external_business_rows = (
                due
                | closed
                | close_reason_rows.ne(MOTION_CLOSE_NONE)
            )
            business_rows = (
                external_business_rows | paid.valid | r07_terminal_fault
            )
            if torch.equal(business_rows, torch.zeros_like(business_rows)):
                # Preserve scalar chronology without manufacturing a private
                # transaction, empty journal rows, or neutral writer calls.
                self._next_epoch += 1
                self._last_motion_common_step = common_step
                return None
            valid_reason = (
                close_reason_rows.eq(MOTION_CLOSE_NONE)
                | close_reason_rows.eq(MOTION_CLOSE_PLAYED_SUFFIX)
                | close_reason_rows.eq(MOTION_CLOSE_UNPLAYED)
            )
            current_playback = self._gather_current(
                record.motion_playback_started
            )
            current_phase_for_close = self._gather_current(record.phase)
            close_has_active_row = (
                current_key_valid
                & action_epoch_open_shot_phase_mask(current_phase_for_close)
            )
            causal = (
                (~close_reason_rows.eq(MOTION_CLOSE_PLAYED_SUFFIX) | current_playback)
                & (~close_reason_rows.eq(MOTION_CLOSE_UNPLAYED) | ~current_playback)
            )
            close_valid = (
                valid_reason
                & closed.eq(close_reason_rows.ne(MOTION_CLOSE_NONE))
                & causal
                & (~closed | close_has_active_row)
            )
            safe_rows = self._latch_device_row_fault(
                ~close_valid, reason_bit=ROW_FAULT_MOTION_CLOSE_CONTRACT
            )
            safe_rows &= r07_terminal_safe
            due = due & safe_rows
            closed = closed & safe_rows
            close_reason_rows = torch.where(
                safe_rows,
                close_reason_rows,
                torch.zeros_like(close_reason_rows),
            )
            close_slots = self._current_slot_mask(closed)
            close_reason = torch.where(
                close_slots,
                close_reason_rows[:, None],
                record.motion_close_reason,
            )
            current_publication = self._gather_current(
                record.publication_ordinal
            )
            current_settlement = self._gather_current(record.settlement_step)
            current_payment = self._gather_current(record.payment_step)
            debt = self._gather_current(close_reason).ne(MOTION_CLOSE_NONE)
            paid_chronology = (
                paid.publication_ordinal.ge(0)
                & paid.settlement_step.ge(0)
                & paid.payment_step.ge(paid.settlement_step)
                & paid.payment_step.le(common_step)
                & paid.payment_step.ge(self._last_r06_paid_payment_step)
            )
            same_paid_key = (
                row_identity.action_epoch_shot_key_valid(current_key)
                & row_identity.action_epoch_shot_key_equal(
                    current_key, paid.shot_key
                )
            )
            current_payment_chronology = (
                current_settlement.ge(0)
                & current_payment.ge(current_settlement)
                & current_payment.le(common_step)
            )
            # Payment, Motion close, and the fixed R07 producer suffix are
            # independent debts.  A valid R06 mailbox may arrive before either
            # other edge; retain it until all three exact facts join.
            paid_assertion_contract = (
                paid_chronology
                & same_paid_key
                & paid.publication_ordinal.eq(current_publication)
                & paid.settlement_step.eq(current_settlement)
                & paid.payment_step.eq(current_payment)
                & current_phase.eq(PHASE_OUTCOME_SETTLED)
                & current_payment_chronology
            )
            invalid_paid_assertion = paid.valid & ~paid_assertion_contract
            paid_safe = self._latch_device_row_fault(
                invalid_paid_assertion,
                reason_bit=ROW_FAULT_R06_PREVIOUS_PAID_CONTRACT,
            )
            due = due & paid_safe
            paid_close_matches = (
                paid.valid
                & paid_assertion_contract
                & debt
                & paid_safe
            )
            # R06 retains one paid mailbox until Motion close and the terminal
            # R07 producer cell both join.  Re-reading that same valid assertion
            # is chronology validation, not a new lifecycle event.  Keep scalar
            # Motion time monotonic, but do not manufacture two empty epoch
            # events plus a zero-mask D05/writer transaction on every pending
            # tick.  A real due/close, malformed assertion, exact terminal
            # fault, or three-debt retirement still takes the journal path.
            actionable_rows = (
                external_business_rows
                | invalid_paid_assertion
                | (paid_close_matches & r07_window_complete)
                | r07_terminal_fault
            )
            if torch.equal(
                actionable_rows, torch.zeros_like(actionable_rows)
            ):
                self._next_epoch += 1
                self._last_motion_common_step = common_step
                return None

            key = record.identity.shot_key
            names, values = self._event(close_slots, key)
            record = self._append(
                record,
                transition=MOTION_CLOSED,
                names=(*names, "motion_close_reason"),
                values=(*values, close_reason),
                changes={"motion_close_reason": close_reason},
            )
            retire_rows = (
                paid_close_matches
                & r07_window_complete
                & r07_terminal_safe
            )
            self._last_r06_paid_payment_step = torch.where(
                retire_rows,
                torch.maximum(
                    self._last_r06_paid_payment_step, paid.payment_step
                ),
                self._last_r06_paid_payment_step,
            )
            retire_slots = self._current_slot_mask(retire_rows)
            phase = torch.where(
                retire_slots,
                torch.full_like(record.phase, PHASE_RETIRED),
                record.phase,
            )
            names, values = self._event(retire_slots, record.identity.shot_key)
            retirement_step = torch.where(
                retire_slots,
                torch.full_like(record.phase, common_step),
                torch.full_like(record.phase, -1),
            )
            record = self._append(
                record,
                transition="RETIRED",
                names=(
                    *names,
                    "motion_close_reason",
                    "payment_step",
                    "retirement_step",
                ),
                values=(
                    *values,
                    record.motion_close_reason,
                    record.payment_step,
                    retirement_step,
                ),
                changes={"phase": phase},
            )
            self._current_closed_rows = ActionEpochClosedRows(
                valid=retire_rows.clone(),
                shot_key=current_key.clone(),
            )
            try:
                self._r06_consume_closed()
                empty_payment = self._empty_payment_rows()
                self._current_payment_rows = ActionEpochRewardPaymentRows(
                    valid=torch.where(
                        retire_rows,
                        empty_payment.valid,
                        self._current_payment_rows.valid,
                    ),
                    shot_key=self._where_key(
                        retire_rows,
                        empty_payment.shot_key,
                        self._current_payment_rows.shot_key,
                    ),
                    payment_step=torch.where(
                        retire_rows,
                        empty_payment.payment_step,
                        self._current_payment_rows.payment_step,
                    ),
                )
            finally:
                self._current_closed_rows = None
            available = self._gather_current(phase).eq(PHASE_IDLE) | self._gather_current(
                phase
            ).eq(PHASE_RETIRED)
            rows = ActionEpochDueRows(
                common_step=common_step,
                due_mask=due,
                construct_mask=due & available,
            )
            self._active_d05 = _ActiveD05Transaction(
                rows=rows.clone(),
                publication_ordinal=self._next_epoch,
                base_version=record.version,
            )
            self._next_epoch += 1
            self._last_motion_common_step = common_step
            return rows.clone()

    def project_current_closed_action_epoch_rows(
        self, *, owner: object
    ) -> ActionEpochClosedRows:
        """Return the transient close image only to the construction-bound R06."""

        with self._lock:
            if owner is not self._r06_owner or self._current_closed_rows is None:
                raise ActionEpochError("R06 close projection owner/lifetime differs")
            return self._current_closed_rows.clone()

    def active_after_command_rows(self) -> ActionEpochDueRows:
        """D05-only full-N due view; it carries no candidate or public key."""

        with self._lock:
            tx = self._active_d05
            if tx is None:
                raise ActionEpochError("no active after-command D05 transaction")
            return tx.rows.clone()

    def abort_d05_transaction(self, *, owner: object) -> None:
        """Abort only the still-private, zero-writer D05 opportunity."""

        with self._operation("abort D05 transaction"):
            self._healthy()
            tx = self._active_d05
            if (
                owner is not self._d05_owner
                or tx is None
                or tx.token is not None
                or tx.publication_started
                or tx.active_writer_kind is not None
            ):
                raise ActionEpochError("D05 abort owner or zero-write boundary differs")
            self._active_d05 = None

    def settle_d05_transaction(self, token: object) -> None:
        """Classify the D05-private projection; ACCEPT alone replaces public rows."""

        with self._operation("settle D05 transaction"):
            self._healthy()
            tx = self._active_d05
            record = self._publication.current
            if (
                tx is None
                or record is None
                or tx.token is not None
                or self._d05_candidate_projector is None
                or self._d05_accept_writers is None
            ):
                raise ActionEpochError("D05 transaction is absent, replayed, or unbound")
            tx.token = token
            candidate = self._require_candidate(self._d05_candidate_projector(token))
            due_slots = self._current_slot_mask(tx.rows.due_mask)
            construct_slots = self._current_slot_mask(tx.rows.construct_mask)
            key_valid = row_identity.action_epoch_shot_key_valid(
                candidate.identity.shot_key
            )
            reset_generation_mismatch = (
                construct_slots
                & candidate.construction_admissible
                & key_valid
                & candidate.identity.shot_key.reset_generation.ne(
                    self._reset_generation[:, None]
                )
            )
            device_safe = self._latch_device_row_fault(
                reset_generation_mismatch.any(dim=1),
                reason_bit=ROW_FAULT_D05_RESET_GENERATION_JOIN,
            )
            candidate_faults = candidate.owner_fault_bits.clone()
            r05_slot = self._owner_slot("r05_runtime")
            candidate_faults[:, :, r05_slot] = torch.bitwise_or(
                candidate_faults[:, :, r05_slot],
                torch.where(
                    reset_generation_mismatch,
                    torch.full_like(
                        candidate_faults[:, :, r05_slot],
                        _FAULT_D05_RESET_GENERATION,
                    ),
                    torch.zeros_like(candidate_faults[:, :, r05_slot]),
                ),
            )
            any_fault = (
                candidate_faults.ne(0).any(dim=-1) | ~device_safe[:, None]
            )
            censor = due_slots & (
                ~construct_slots
                | any_fault
                | (
                    construct_slots
                    & candidate.construction_admissible
                    & ~key_valid
                )
            )
            reject = (
                construct_slots
                & ~any_fault
                & ~candidate.construction_admissible
            )
            defer = (
                construct_slots
                & key_valid
                & ~any_fault
                & candidate.construction_admissible
                & ~candidate.playback_admissible
            )
            accept = (
                construct_slots
                & key_valid
                & ~any_fault
                & candidate.construction_admissible
                & candidate.playback_admissible
            )
            decision = torch.full(
                self._shot_shape,
                D05_DECISION_NONE,
                dtype=torch.int64,
                device=self.device,
            )
            decision = torch.where(
                censor, torch.full_like(decision, D05_DECISION_CENSOR), decision
            )
            decision = torch.where(
                reject, torch.full_like(decision, D05_DECISION_REJECT), decision
            )
            decision = torch.where(
                defer, torch.full_like(decision, D05_DECISION_DEFER), decision
            )
            decision = torch.where(
                accept, torch.full_like(decision, D05_DECISION_ACCEPT), decision
            )
            family = torch.zeros_like(decision)
            attribution = torch.zeros_like(due_slots)
            catalog_size = self._action_uids_by_slot.numel()
            if catalog_size:
                slot = candidate.identity.shot_key.action_slot
                in_range = due_slots & slot.ge(0) & slot.lt(catalog_size)
                safe_slot = slot.clamp(0, catalog_size - 1)
                attribution = in_range & candidate.identity.shot_key.action_uid.eq(
                    self._action_uids_by_slot[safe_slot]
                )
                family = torch.where(
                    attribution, self._family_codes_by_slot[safe_slot], family
                )
            names, values = self._event(due_slots, candidate.identity.shot_key)
            record = self._append(
                record,
                transition="D05_SETTLED",
                names=(
                    *names,
                    "due_mask",
                    "selected_mask",
                    "decision",
                    "construction_admissible",
                    "playback_admissible",
                    "owner_fault_bits",
                    "stroke_family_code",
                    "action_attribution_valid",
                ),
                values=(
                    *values,
                    due_slots,
                    construct_slots,
                    decision,
                    candidate.construction_admissible,
                    candidate.playback_admissible,
                    candidate_faults,
                    family,
                    attribution,
                ),
                journal_epoch=tx.publication_ordinal,
            )
            construction_admitted = (
                construct_slots
                & candidate.construction_admissible
                & ~any_fault
            )
            self._milestone_after_business_write(
                "r05_runtime",
                "add_d05_events",
                due_slots,
                construct_slots,
                construction_admitted,
                construction_admitted & key_valid,
            )
            tx.accept_mask = accept
            for ordinal, (owner_kind, writer) in enumerate(
                zip(REVEAL_WRITE_OWNER_ORDER, self._d05_accept_writers)
            ):
                tx.active_writer_kind = owner_kind
                tx.next_writer_ordinal = ordinal
                names, values = self._event(accept, candidate.identity.shot_key)
                record = self._append(
                    record,
                    transition="WRITES_STARTED:" + owner_kind,
                    names=names,
                    values=values,
                    journal_epoch=tx.publication_ordinal,
                )
                try:
                    writer(token)
                except Exception:
                    self._poison_locked(
                        reason=_FAULT_D05_OWNER,
                        owner_kind=owner_kind,
                        transition="D05_WRITER_POISON:" + owner_kind,
                    )
                    raise
                record = self._publication.current
                assert record is not None
                record = self._append(
                    record,
                    transition="WRITES_COMMITTED:" + owner_kind,
                    names=names,
                    values=values,
                    journal_epoch=tx.publication_ordinal,
                )
                tx.active_writer_kind = None

            tx.publication_started = True
            mask3 = accept[..., None]
            writes_started = torch.where(
                mask3, torch.zeros_like(record.writes_started), record.writes_started
            )
            writes_committed = torch.where(
                mask3,
                torch.zeros_like(record.writes_committed),
                record.writes_committed,
            )
            for owner_kind in REVEAL_WRITE_OWNER_ORDER:
                slot = self._owner_slot(owner_kind)
                writes_started[:, :, slot] |= accept
                writes_committed[:, :, slot] |= accept
            phase = torch.where(
                accept,
                torch.full_like(record.phase, PHASE_REVEAL_COMMITTED),
                record.phase,
            )
            publication = torch.where(
                accept,
                torch.full_like(record.publication_ordinal, tx.publication_ordinal),
                record.publication_ordinal,
            )
            changes = {
                "phase": phase,
                "identity": self._where_identity(accept, candidate.identity, record.identity),
                "clocks": self._where_clocks(accept, candidate.clocks, record.clocks),
                "task": self._where_task(accept, candidate.task, record.task),
                "rng_counter": torch.where(accept, candidate.rng_counter, record.rng_counter),
                "publication_ordinal": publication,
                "owner_fault_bits": torch.where(
                    mask3, candidate.owner_fault_bits, record.owner_fault_bits
                ),
                "writes_started": writes_started,
                "writes_committed": writes_committed,
                "launch_succeeded": torch.where(
                    accept, torch.zeros_like(record.launch_succeeded), record.launch_succeeded
                ),
                "late_launch": torch.where(
                    accept, torch.zeros_like(record.late_launch), record.late_launch
                ),
                "outcome_code": torch.where(
                    accept, torch.full_like(record.outcome_code, -1), record.outcome_code
                ),
                "fact_valid_bits": torch.where(
                    mask3, torch.zeros_like(record.fact_valid_bits), record.fact_valid_bits
                ),
                "fact_source_step": torch.where(
                    mask3,
                    torch.full_like(record.fact_source_step, -1),
                    record.fact_source_step,
                ),
                "fact_f32": torch.where(
                    mask3[..., None], torch.zeros_like(record.fact_f32), record.fact_f32
                ),
                "motion_playback_started": torch.where(
                    accept,
                    torch.zeros_like(record.motion_playback_started),
                    record.motion_playback_started,
                ),
                "motion_close_reason": torch.where(
                    accept,
                    torch.zeros_like(record.motion_close_reason),
                    record.motion_close_reason,
                ),
                "settlement_step": torch.where(
                    accept,
                    torch.full_like(record.settlement_step, -1),
                    record.settlement_step,
                ),
                "payment_step": torch.where(
                    accept,
                    torch.full_like(record.payment_step, -1),
                    record.payment_step,
                ),
                "poison_reason": torch.where(
                    accept, torch.zeros_like(record.poison_reason), record.poison_reason
                ),
            }
            names, values = self._event(accept, candidate.identity.shot_key)
            record = self._append(
                record,
                transition="D05_ACCEPT_PUBLISHED",
                names=(*names, "publication_ordinal"),
                values=(*values, publication),
                journal_epoch=tx.publication_ordinal,
                changes=changes,
            )
            self._milestone_after_business_write(
                "r05_runtime", "reset_event_rows", accept
            )
            self._active_d05 = None
            return None

    # ------------------------------------------------------------------
    # Bound Motion/Physical/R06 publications

    def publish_motion_playback_started(self, *, owner: object) -> torch.Tensor:
        """Publish Motion's edge and return only the cumulative started mask."""

        with self._operation("Motion playback started"):
            self._healthy()
            record = self._publication.current
            if record is None or self._motion_playback is None:
                raise ActionEpochError("Motion playback owner is not bound")
            if getattr(self._motion_playback, "__self__", None) is not owner:
                raise ActionEpochError("Motion playback caller differs")
            try:
                projection = ActionEpochMotionPlaybackProjection(
                    current_task_slot=record.current_task_slot.detach().clone(),
                    phase=record.phase.detach().clone(),
                    # ``selected_mask`` is derived from ``phase`` and already
                    # owns fresh storage; detach without a redundant copy.
                    selected_mask=record.selected_mask.detach(),
                    shot_key=record.identity.shot_key.clone(),
                )
                mask = self._tensor(
                    self._motion_playback(MOTION_PLAYBACK_STARTED, projection),
                    label="Motion playback mask",
                    shape=self._shot_shape,
                    dtype=torch.bool,
                )
            except Exception:
                self._poison_locked(
                    reason=_FAULT_MOTION_PLAYBACK,
                    owner_kind="motion",
                    transition="MOTION_PLAYBACK_POISON",
                )
                raise
            eligible = (
                mask
                & action_epoch_open_shot_phase_mask(record.phase)
                & ~record.motion_playback_started
            )
            started = record.motion_playback_started | eligible
            names, values = self._event(eligible, record.identity.shot_key)
            record = self._append(
                record,
                transition=MOTION_PLAYBACK_STARTED,
                names=names,
                values=values,
                changes={"motion_playback_started": started},
            )
            # Do not expose the record-owned tensor: callers receive exactly
            # the typed fact they consume, without a full-record round trip.
            return record.motion_playback_started.detach().clone().contiguous()

    def refresh_physical_postphysics_rows(self) -> None:
        """Pull the sole active Physical packet and full-key join its fact planes."""

        with self._operation("refresh Physical postphysics rows"):
            self._healthy()
            record = self._publication.current
            if record is None or self._physical_projection is None:
                raise ActionEpochError("Physical postphysics projection is not bound")
            projection = self._physical_projection()
            try:
                from .action_ball_physical_flight_device import (
                    ActionEpochR06PostPhysicsProjection,
                )
            except ImportError:  # pragma: no cover - focused standalone import
                from action_ball_physical_flight_device import (
                    ActionEpochR06PostPhysicsProjection,
                )
            if type(projection) is not ActionEpochR06PostPhysicsProjection:
                raise ActionEpochError(
                    "Physical must return exact ActionEpochR06PostPhysicsProjection"
                )
            flight_slot_value = getattr(projection, "flight_slot", None)
            if (
                type(flight_slot_value) is not torch.Tensor
                or flight_slot_value.ndim != 2
                or flight_slot_value.shape[0] != self.num_envs
                or flight_slot_value.shape[1] < 1
            ):
                raise ActionEpochError("Physical.flight_slot must establish [N,K]")
            physical_shape = (self.num_envs, flight_slot_value.shape[1])
            self._borrow_tensor(
                flight_slot_value,
                label="Physical.flight_slot",
                shape=physical_shape,
                dtype=torch.int64,
            )
            observe = self._borrow_tensor(
                projection.observe_mask,
                label="Physical.observe_mask",
                shape=physical_shape,
                dtype=torch.bool,
            )
            try:
                key = row_identity.require_action_epoch_shot_key(
                    projection.shot_key,
                    shape=physical_shape,
                    device=self.device,
                    label="Physical.shot_key",
                )
            except row_identity.ActionEpochShotKeyError as exc:
                raise ActionEpochError(str(exc)) from exc
            publication = self._borrow_tensor(
                projection.publication_ordinal,
                label="Physical.publication_ordinal",
                shape=physical_shape,
                dtype=torch.int64,
            )
            faults = self._borrow_tensor(
                projection.owner_fault_bits,
                label="Physical.owner_fault_bits",
                shape=physical_shape,
                dtype=torch.int64,
            )
            valid_bits = self._borrow_tensor(
                projection.fact_valid_bits,
                label="Physical.fact_valid_bits",
                shape=physical_shape,
                dtype=torch.int64,
            )
            source_step = self._borrow_tensor(
                projection.fact_source_step,
                label="Physical.fact_source_step",
                shape=physical_shape,
                dtype=torch.int64,
            )
            fact_f32 = self._borrow_tensor(
                projection.fact_f32,
                label="Physical.fact_f32",
                shape=(*physical_shape, OWNER_FACT_F32_WIDTH),
                dtype=torch.float32,
            )
            join = (
                observe[:, :, None]
                & row_identity.action_epoch_shot_key_valid(key)[:, :, None]
                & publication[:, :, None].eq(
                    record.publication_ordinal[:, None, :]
                )
                & (
                    record.phase.eq(PHASE_LAUNCH_SETTLED)
                    | record.phase.eq(PHASE_OUTCOME_SETTLED)
                )[:, None, :]
            )
            for field in fields(ActionEpochShotKey):
                join &= getattr(key, field.name)[:, :, None].eq(
                    getattr(record.identity.shot_key, field.name)[:, None, :]
                )
            destination_count = join.sum(dim=1)
            source_count = join.sum(dim=2)
            invalid_rows = (
                (observe & source_count.ne(1)).any(dim=1)
                | destination_count.gt(1).any(dim=1)
            )
            safe_rows = self._latch_device_row_fault(
                invalid_rows, reason_bit=ROW_FAULT_PHYSICAL_POSTPHYSICS_JOIN
            )
            join &= safe_rows[:, None, None]
            joined = join.any(dim=1)
            source_index = join.to(torch.int64).argmax(dim=1)

            def gather_physical(value: torch.Tensor) -> torch.Tensor:
                tail = (1,) * (value.ndim - 2)
                index = source_index.reshape(
                    (*self._shot_shape, *tail)
                ).expand((*self._shot_shape, *value.shape[2:]))
                return value.gather(1, index)

            joined_faults = gather_physical(faults)
            joined_valid = gather_physical(valid_bits)
            joined_source = gather_physical(source_step)
            joined_facts = gather_physical(fact_f32)
            physical_fault = joined & joined_faults.ne(0)
            physical_nonfinite = physical_fault & torch.bitwise_and(
                joined_faults,
                ROW_FAULT_PHYSICAL_POSTPHYSICS_NONFINITE,
            ).ne(0)
            recognized_physical_fault_mask = (
                ROW_FAULT_PHYSICAL_POSTPHYSICS_PRODUCER
                | ROW_FAULT_PHYSICAL_POSTPHYSICS_NONFINITE
            )
            # An unregistered/foreign Physical source bit is itself a producer
            # contract fault.  It must stop the optimizer rather than becoming
            # an unscored miss merely because this consumer cannot name it.
            physical_producer = physical_fault & (
                torch.bitwise_and(
                    joined_faults,
                    ROW_FAULT_PHYSICAL_POSTPHYSICS_PRODUCER,
                ).ne(0)
                | torch.bitwise_and(
                    joined_faults, ~recognized_physical_fault_mask
                ).ne(0)
            )
            self._latch_device_row_fault(
                physical_producer.any(dim=1),
                reason_bit=ROW_FAULT_PHYSICAL_POSTPHYSICS_PRODUCER,
            )
            owner_safe_rows = self._latch_device_row_fault(
                physical_nonfinite.any(dim=1),
                reason_bit=ROW_FAULT_PHYSICAL_POSTPHYSICS_NONFINITE,
            )
            business_joined = joined & owner_safe_rows[:, None]
            journal_faults = torch.where(
                joined, joined_faults, torch.zeros_like(joined_faults)
            )
            owner_slot = self._owner_slot("physical_ball")
            owner_faults = record.owner_fault_bits.clone()
            owner_faults[:, :, owner_slot] = torch.where(
                joined,
                torch.bitwise_or(
                    owner_faults[:, :, owner_slot], journal_faults
                ),
                owner_faults[:, :, owner_slot],
            )
            record_valid = record.fact_valid_bits.clone()
            record_valid[:, :, owner_slot] = torch.where(
                business_joined,
                torch.bitwise_or(
                    record_valid[:, :, owner_slot], joined_valid
                ),
                record_valid[:, :, owner_slot],
            )
            previous_valid = record.fact_valid_bits[:, :, owner_slot]
            present = business_joined & torch.bitwise_and(joined_valid, 1).ne(0)
            first_observed = (
                present & ~torch.bitwise_and(previous_valid, 1).ne(0)
            )
            first_contact = (
                business_joined
                & torch.bitwise_and(joined_valid, 2).ne(0)
                & ~torch.bitwise_and(previous_valid, 2).ne(0)
            )
            record_source = record.fact_source_step.clone()
            record_source[:, :, owner_slot] = torch.where(
                first_contact, joined_source, record_source[:, :, owner_slot]
            )
            record_facts = record.fact_f32.clone()
            physical_facts = record_facts[:, :, owner_slot, :].clone()
            physical_facts[..., 0:3] = torch.where(
                present[..., None], joined_facts[..., 0:3], physical_facts[..., 0:3]
            )
            physical_facts[..., 9] = torch.where(
                present, joined_facts[..., 9], physical_facts[..., 9]
            )
            physical_facts[..., 3:9] = torch.where(
                first_contact[..., None],
                joined_facts[..., 3:9],
                physical_facts[..., 3:9],
            )
            record_facts[:, :, owner_slot, :] = physical_facts
            names, values = self._event(joined, record.identity.shot_key)
            record = self._append(
                record,
                transition="PHYSICAL_POSTPHYSICS_ROWS",
                names=(
                    *names,
                    "publication_ordinal",
                    "owner_fault_bits",
                    "fact_valid_bits",
                    "fact_source_step",
                ),
                values=(
                    *values,
                    record.publication_ordinal,
                    journal_faults,
                    record_valid[:, :, owner_slot],
                    record_source[:, :, owner_slot],
                ),
                changes={
                    "owner_fault_bits": owner_faults,
                    "fact_valid_bits": record_valid,
                    "fact_source_step": record_source,
                    "fact_f32": record_facts,
                },
            )
            self._milestone_after_business_write(
                "physical_ball",
                "add_physical_events",
                first_observed,
                first_contact,
            )

    def refresh_r06_outcome_rows(self) -> None:
        """Pull and full-key join R06's current settled outcome rows."""

        with self._operation("refresh R06 outcome rows"):
            self._healthy()
            record = self._publication.current
            if record is None or self._r06_outcome_projection is None:
                raise ActionEpochError("R06 outcome row projection is not bound")
            projection = self._r06_outcome_projection()
            try:
                from . import action_ball_landing_outcome_device as r06_device
            except ImportError:  # pragma: no cover - focused standalone import
                import action_ball_landing_outcome_device as r06_device
            if type(projection) is not r06_device.ActionEpochR06OutcomeRows:
                raise ActionEpochError("R06 must return exact ActionEpochR06OutcomeRows")
            valid = self._tensor(
                projection.valid,
                label="R06.outcome.valid",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            try:
                key = row_identity.require_action_epoch_shot_key(
                    projection.shot_key,
                    shape=(self.num_envs,),
                    device=self.device,
                    label="R06.outcome.shot_key",
                ).clone()
            except row_identity.ActionEpochShotKeyError as exc:
                raise ActionEpochError(str(exc)) from exc
            publication = self._tensor(
                projection.publication_ordinal,
                label="R06.outcome.publication_ordinal",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            settlement = self._tensor(
                projection.settlement_step,
                label="R06.outcome.settlement_step",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            valid_bits = self._tensor(
                projection.valid_bits,
                label="R06.outcome.valid_bits",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            facts = self._tensor(
                projection.fact_values,
                label="R06.outcome.fact_values",
                shape=(self.num_envs, OWNER_FACT_F32_WIDTH),
                dtype=torch.float32,
            )
            outcome = self._tensor(
                projection.outcome_code,
                label="R06.outcome.outcome_code",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            faults = self._tensor(
                projection.owner_fault_bits,
                label="R06.outcome.owner_fault_bits",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            owner_slot = self._owner_slot("r06_landing_outcome")
            current_key = self._gather_current_key(record.identity.shot_key)
            key_matches = (
                row_identity.action_epoch_shot_key_valid(key)
                & row_identity.action_epoch_shot_key_equal(key, current_key)
            )
            current_phase = self._gather_current(record.phase)
            current_publication = self._gather_current(record.publication_ordinal)
            first_join = (
                key_matches
                & current_phase.eq(PHASE_LAUNCH_SETTLED)
                & publication.ge(0)
                & publication.eq(current_publication)
                & settlement.ge(0)
            )
            current_owner_faults = self._gather_current(
                record.owner_fault_bits[:, :, owner_slot]
            )
            current_facts = self._gather_current(
                record.fact_f32[:, :, owner_slot, :]
            )
            same_after_image = (
                publication.eq(current_publication)
                & settlement.eq(self._gather_current(record.settlement_step))
                & valid_bits.eq(
                    self._gather_current(record.fact_valid_bits[:, :, owner_slot])
                )
                & facts.view(torch.int32)
                .eq(current_facts.view(torch.int32))
                .all(dim=1)
                & outcome.eq(self._gather_current(record.outcome_code))
                & torch.bitwise_or(current_owner_faults, faults).eq(
                    current_owner_faults
                )
            )
            already_joined = (
                key_matches
                & current_phase.eq(PHASE_OUTCOME_SETTLED)
                & same_after_image
            )
            safe_rows = self._latch_device_row_fault(
                valid & ~(first_join | already_joined),
                reason_bit=ROW_FAULT_R06_OUTCOME_JOIN,
            )
            observed_rows = valid & first_join & safe_rows
            r06_producer_rows = observed_rows & torch.bitwise_and(
                faults, r06_device.FAULT_PRODUCER_CONTRACT
            ).ne(0)
            r06_overflow_rows = observed_rows & torch.bitwise_and(
                faults, r06_device.FAULT_ENGINE_OVERFLOW
            ).ne(0)
            r06_nonfinite_rows = observed_rows & torch.bitwise_and(
                faults, r06_device.FAULT_NONFINITE
            ).ne(0)
            recognized_r06_fault_mask = (
                r06_device.FAULT_PRODUCER_CONTRACT
                | r06_device.FAULT_ENGINE_OVERFLOW
                | r06_device.FAULT_NONFINITE
            )
            r06_other_rows = (
                observed_rows
                & faults.ne(0)
                & torch.bitwise_and(faults, ~recognized_r06_fault_mask).ne(0)
            )
            self._latch_device_row_fault(
                r06_producer_rows,
                reason_bit=ROW_FAULT_R06_OWNER_PRODUCER_CONTRACT,
            )
            self._latch_device_row_fault(
                r06_overflow_rows,
                reason_bit=ROW_FAULT_R06_OWNER_ENGINE_OVERFLOW,
            )
            self._latch_device_row_fault(
                r06_nonfinite_rows,
                reason_bit=ROW_FAULT_R06_OWNER_NONFINITE,
            )
            owner_safe_rows = self._latch_device_row_fault(
                r06_other_rows,
                reason_bit=ROW_FAULT_R06_OWNER_OTHER,
            )
            join_rows = observed_rows & owner_safe_rows
            audit_rows = observed_rows
            join = self._current_slot_mask(join_rows)
            audit_join = self._current_slot_mask(audit_rows)
            owner_faults = record.owner_fault_bits.clone()
            owner_faults[:, :, owner_slot] = torch.where(
                audit_join,
                torch.bitwise_or(owner_faults[:, :, owner_slot], faults[:, None]),
                owner_faults[:, :, owner_slot],
            )
            record_valid = record.fact_valid_bits.clone()
            record_valid[:, :, owner_slot] = torch.where(
                join, valid_bits[:, None], record_valid[:, :, owner_slot]
            )
            record_source = record.fact_source_step.clone()
            record_source[:, :, owner_slot] = torch.where(
                join, settlement[:, None], record_source[:, :, owner_slot]
            )
            record_facts = record.fact_f32.clone()
            record_facts[:, :, owner_slot, :] = torch.where(
                join[..., None], facts[:, None, :], record_facts[:, :, owner_slot, :]
            )
            outcome_code = torch.where(join, outcome[:, None], record.outcome_code)
            settlement_step = torch.where(
                join, settlement[:, None], record.settlement_step
            )
            phase = torch.where(
                join,
                torch.full_like(record.phase, PHASE_OUTCOME_SETTLED),
                record.phase,
            )
            names, values = self._event(audit_join, record.identity.shot_key)
            predicate_bits = (
                facts[:, r06_device.R06_ACTION_EPOCH_CONTACT_VALID_F32]
                .eq(1.0).to(torch.int64)
                | facts[:, r06_device.R06_ACTION_EPOCH_NET_CROSSED_F32]
                .eq(1.0).to(torch.int64).mul(2)
                | facts[:, r06_device.R06_ACTION_EPOCH_NET_CLEAR_F32]
                .eq(1.0).to(torch.int64).mul(4)
                | facts[:, r06_device.R06_ACTION_EPOCH_CROSSING_VALID_F32]
                .eq(1.0).to(torch.int64).mul(8)
                | facts[:, r06_device.R06_ACTION_EPOCH_ON_TABLE_F32]
                .eq(1.0).to(torch.int64).mul(16)
                | facts[:, r06_device.R06_ACTION_EPOCH_COMMON_ON_TABLE_F32]
                .eq(1.0).to(torch.int64).mul(32)
            )
            predicate_bits = torch.where(
                join, predicate_bits[:, None], torch.zeros_like(join, dtype=torch.int64)
            )
            record = self._append(
                record,
                transition="R06_OUTCOME_ROWS",
                names=(
                    *names,
                    "publication_ordinal",
                    "settlement_step",
                    "valid_bits",
                    "outcome_code",
                    "owner_fault_bits",
                    "predicate_bits",
                ),
                values=(
                    *values,
                    record.publication_ordinal,
                    settlement_step,
                    record_valid[:, :, owner_slot],
                    outcome_code,
                    torch.where(
                        audit_join,
                        faults[:, None],
                        torch.zeros_like(owner_faults[:, :, owner_slot]),
                    ),
                    predicate_bits,
                ),
                changes={
                    "phase": phase,
                    "settlement_step": settlement_step,
                    "outcome_code": outcome_code,
                    "owner_fault_bits": owner_faults,
                    "fact_valid_bits": record_valid,
                    "fact_source_step": record_source,
                    "fact_f32": record_facts,
                },
            )
            event_fact = lambda offset: (  # noqa: E731 - compact typed projection
                join & facts[:, None, offset].eq(1.0)
            )
            net_crossed = event_fact(r06_device.R06_ACTION_EPOCH_NET_CROSSED_F32)
            net_clear = (
                net_crossed
                & event_fact(r06_device.R06_ACTION_EPOCH_NET_CLEAR_F32)
            )
            self._milestone_after_business_write(
                "r06_landing_outcome",
                "add_r06_events",
                join,
                event_fact(r06_device.R06_ACTION_EPOCH_CONTACT_VALID_F32),
                net_crossed,
                net_clear,
                event_fact(r06_device.R06_ACTION_EPOCH_CROSSING_VALID_F32),
                event_fact(r06_device.R06_ACTION_EPOCH_ON_TABLE_F32),
                join
                & facts[:, None, r06_device.R06_ACTION_EPOCH_COMMON_ON_TABLE_F32].eq(1.0),
            )

    # ------------------------------------------------------------------
    # Reward-cycle completion and actual control-step payment

    def _open_reward_cycle_record_locked(self) -> ActionEpochRecord:
        self._healthy()
        record = self._publication.current
        if (
            record is None
            or self._reward_open
            or self._active_d05 is not None
            or self._selected_reset.active
        ):
            raise ActionEpochError("reward cycle cannot open at this boundary")
        overflow = self._reward_cycle_age.eq(_I64_MAX)
        safe_age = torch.where(
            overflow, self._reward_cycle_age, self._reward_cycle_age + 1
        )
        fault = torch.where(
            overflow,
            torch.bitwise_or(
                self._reward_cycle_fault,
                torch.full_like(
                    self._reward_cycle_fault, _FAULT_REWARD_CYCLE_OVERFLOW
                ),
            ),
            self._reward_cycle_fault,
        )
        due = torch.ones(
            (self.num_envs, REWARD_CONSUMER_COUNT),
            dtype=torch.bool,
            device=self.device,
        )
        paid = torch.zeros_like(due)
        opened = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        record = self._append(
            record,
            transition="REWARD_CYCLE_OPEN",
            names=("reward_cycle_age", "reward_cycle_fault"),
            values=(safe_age, fault),
            changes={
                "reward_cycle_age": safe_age,
                "reward_cycle_fault": fault,
                "reward_cycle_open": opened,
                "reward_due": due,
                "reward_paid": paid,
            },
        )
        self._reward_cycle_age = safe_age
        self._reward_cycle_fault = fault
        self._reward_open = True
        self._reward_ordinal = 0
        return record

    def open_reward_cycle(self) -> ActionEpochRecord:
        with self._operation("open reward cycle"):
            return self._open_reward_cycle_record_locked().clone()

    def _open_lean_reward_cycle_snapshot(self) -> _LeanRewardCycleSnapshot:
        """Open once and copy only selected fields used by sealed Lean Reward."""

        with self._operation("open lean reward cycle"):
            record = self._open_reward_cycle_record_locked()
            index = record.current_task_slot

            def selected(value: torch.Tensor) -> torch.Tensor:
                suffix = (1,) * (value.ndim - 2)
                gather_index = index.reshape(
                    self.num_envs, 1, *suffix
                ).expand(self.num_envs, 1, *value.shape[2:])
                return torch.gather(value, 1, gather_index).squeeze(1)

            def owner_field(value: torch.Tensor, owner_slot: int) -> torch.Tensor:
                return selected(value[:, :, owner_slot])

            return _LeanRewardCycleSnapshot(
                reward_cycle_age=record.reward_cycle_age.clone(),
                reward_cycle_fault=record.reward_cycle_fault.clone(),
                phase=selected(record.phase),
                settlement_step=selected(record.settlement_step),
                payment_step=selected(record.payment_step),
                r03=(
                    owner_field(record.fact_valid_bits, _LEAN_REWARD_R03_SLOT),
                    owner_field(record.fact_source_step, _LEAN_REWARD_R03_SLOT),
                    owner_field(record.fact_f32, _LEAN_REWARD_R03_SLOT),
                    owner_field(record.owner_fault_bits, _LEAN_REWARD_R03_SLOT),
                ),
                physical=(
                    owner_field(record.fact_valid_bits, _LEAN_REWARD_PHYSICAL_SLOT),
                    owner_field(record.fact_source_step, _LEAN_REWARD_PHYSICAL_SLOT),
                    owner_field(record.owner_fault_bits, _LEAN_REWARD_PHYSICAL_SLOT),
                ),
                r06=(
                    owner_field(record.fact_valid_bits, _LEAN_REWARD_R06_SLOT),
                    owner_field(record.fact_f32, _LEAN_REWARD_R06_SLOT),
                    owner_field(record.owner_fault_bits, _LEAN_REWARD_R06_SLOT),
                ),
                r07=(
                    owner_field(record.fact_valid_bits, _LEAN_REWARD_R07_SLOT),
                    owner_field(record.fact_source_step, _LEAN_REWARD_R07_SLOT),
                    owner_field(record.fact_f32, _LEAN_REWARD_R07_SLOT),
                    owner_field(record.owner_fault_bits, _LEAN_REWARD_R07_SLOT),
                ),
            )

    def pay_reward(self, ordinal: int) -> None:
        if type(ordinal) is not int:
            raise ActionEpochError("reward ordinal must be exact int")
        with self._operation("pay reward"):
            self._healthy()
            record = self._publication.current
            if (
                record is None
                or not self._reward_open
                or ordinal != self._reward_ordinal
                or not (0 <= ordinal < REWARD_CONSUMER_COUNT)
            ):
                raise ActionEpochError("reward payment chronology differs")
            paid = record.reward_paid.clone()
            paid[:, ordinal] = record.reward_due[:, ordinal]
            last = ordinal + 1 == REWARD_CONSUMER_COUNT
            opened = (
                torch.zeros_like(record.reward_cycle_open)
                if last
                else record.reward_cycle_open
            )
            self._append(
                record,
                transition="REWARD_CONSUMER_PAID",
                names=("reward_consumer_ordinal", "reward_paid"),
                values=(
                    torch.full(
                        (self.num_envs,),
                        ordinal,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    paid,
                ),
                changes={"reward_paid": paid, "reward_cycle_open": opened},
            )
            self._reward_ordinal += 1
            if last:
                self._reward_open = False

    def publish_reward_payment(self, control_step: int) -> None:
        """Publish the actual completed Reward edge as a command."""

        if type(control_step) is not int or control_step < 0:
            raise ActionEpochError("control_step must be a non-negative exact int")
        with self._operation("publish reward payment"):
            self._healthy()
            record = self._publication.current
            if (
                record is None
                or self._reward_open
                or self._active_d05 is not None
                or self._selected_reset.active
            ):
                raise ActionEpochError("reward payment boundary is not closed")
            owner_slot = self._owner_slot("r06_landing_outcome")
            current_key = self._gather_current_key(record.identity.shot_key)
            complete = record.reward_due.all(dim=1) & record.reward_due.eq(
                record.reward_paid
            ).all(dim=1)
            settlement_step = self._gather_current(record.settlement_step)
            prior_payment_step = self._gather_current(record.payment_step)
            candidate = (
                complete
                & self._gather_current(record.phase).eq(PHASE_OUTCOME_SETTLED)
                & row_identity.action_epoch_shot_key_valid(current_key)
                & self._gather_current(
                    record.fact_valid_bits[:, :, owner_slot]
                ).ne(0)
            )
            pending = candidate & prior_payment_step.lt(0)
            payment_contract = (
                settlement_step.ge(0)
                & prior_payment_step.eq(-1)
                & settlement_step.le(control_step)
            )
            safe_rows = self._latch_device_row_fault(
                pending & ~payment_contract,
                reason_bit=ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY,
            )
            valid = pending & payment_contract & safe_rows
            mask = self._current_slot_mask(valid)
            payment_step = torch.where(
                mask,
                torch.full_like(record.payment_step, control_step),
                record.payment_step,
            )
            record = self._append(
                record,
                transition="PAYMENT_RECORDED",
                names=(
                    *self._event(mask, record.identity.shot_key)[0],
                    "payment_step",
                ),
                values=(
                    *self._event(mask, record.identity.shot_key)[1],
                    payment_step,
                ),
                changes={"payment_step": payment_step},
            )
            self._current_payment_rows = ActionEpochRewardPaymentRows(
                valid=valid.clone(),
                shot_key=current_key.clone(),
                payment_step=torch.where(
                    valid,
                    self._gather_current(record.payment_step),
                    torch.full(
                        (self.num_envs,),
                        -1,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                ),
            )

    def project_current_reward_payment_rows(self) -> ActionEpochRewardPaymentRows:
        with self._lock:
            self._healthy()
            return self._current_payment_rows.clone()

    # ------------------------------------------------------------------
    # Existing bound owner facts and Physical launch publication

    def merge_runtime_owner_fault(
        self, owner_kind: str, fault_bits: torch.Tensor, *, owner: object
    ) -> None:
        with self._operation("merge runtime owner fault"):
            self._healthy()
            record = self._publication.current
            owner_slot = self._owner_slot(owner_kind)
            if (
                record is None
                or self._fact_owner_identities.get(owner_kind) is not owner
            ):
                raise ActionEpochError("runtime fault owner identity differs")
            faults = self._tensor(
                fault_bits,
                label=owner_kind + " fault_bits",
                shape=self._shot_shape,
                dtype=torch.int64,
            )
            merged = record.owner_fault_bits.clone()
            merged[:, :, owner_slot] = torch.bitwise_or(
                merged[:, :, owner_slot], faults
            )
            event_mask = faults.ne(0)
            names, values = self._event(event_mask, record.identity.shot_key)
            record = self._append(
                record,
                transition="OWNER_FAULT:" + owner_kind,
                names=(*names, "owner_fault_bits"),
                values=(*values, merged[:, :, owner_slot]),
                changes={"owner_fault_bits": merged},
            )

    def publish_owner_facts(
        self,
        owner_kind: str,
        *,
        owner: object,
        valid_bits: torch.Tensor,
        source_step: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        with self._operation("publish owner facts"):
            self._healthy()
            record = self._publication.current
            owner_slot = self._owner_slot(owner_kind)
            if (
                record is None
                or self._fact_owner_identities.get(owner_kind) is not owner
            ):
                raise ActionEpochError("fact owner identity differs")
            valid = self._tensor(
                valid_bits,
                label=owner_kind + ".valid_bits",
                shape=self._shot_shape,
                dtype=torch.int64,
            )
            step = self._tensor(
                source_step,
                label=owner_kind + ".source_step",
                shape=self._shot_shape,
                dtype=torch.int64,
            )
            fact_values = self._tensor(
                values,
                label=owner_kind + ".values",
                shape=(*self._shot_shape, OWNER_FACT_F32_WIDTH),
                dtype=torch.float32,
            )
            requested = valid.ne(0)
            active = (
                row_identity.action_epoch_shot_key_valid(
                    record.identity.shot_key
                )
                & action_epoch_open_shot_phase_mask(record.phase)
            )
            invalid_rows = (requested & ~active).any(dim=1)
            safe_rows = self._latch_device_row_fault(
                invalid_rows, reason_bit=ROW_FAULT_OWNER_FACT_ACTIVE_JOIN
            )
            selected = requested & active & safe_rows[:, None]
            owner_fault_free = record.owner_fault_bits[:, :, owner_slot].eq(0)
            fully_valid = None
            if owner_kind == "r03_strike_fact":
                fully_valid = (
                    selected & valid.bitwise_and(3).eq(3) & owner_fault_free
                )
            elif owner_kind == "r07_recovery":
                fully_valid = (
                    selected
                    & valid.bitwise_and(3).eq(3)
                    & fact_values[..., 2].eq(1.0)
                    & fact_values[..., 3].eq(1.0)
                    & fact_values[..., 4].eq(0.0)
                    & owner_fault_free
                )
            all_valid = record.fact_valid_bits.clone()
            all_steps = record.fact_source_step.clone()
            all_values = record.fact_f32.clone()
            all_valid[:, :, owner_slot] = torch.where(
                selected, valid, all_valid[:, :, owner_slot]
            )
            all_steps[:, :, owner_slot] = torch.where(
                selected, step, all_steps[:, :, owner_slot]
            )
            all_values[:, :, owner_slot, :] = torch.where(
                selected[..., None],
                fact_values,
                all_values[:, :, owner_slot, :],
            )
            names, event_values = self._event(selected, record.identity.shot_key)
            qualified = (
                fully_valid
                if fully_valid is not None
                else torch.zeros_like(selected)
            )
            record = self._append(
                record,
                transition="OWNER_FACTS:" + owner_kind,
                names=(*names, "valid_bits", "source_step", "qualified"),
                values=(*event_values, valid, step, qualified),
                changes={
                    "fact_valid_bits": all_valid,
                    "fact_source_step": all_steps,
                    "fact_f32": all_values,
                },
            )
            if fully_valid is not None:
                self._milestone_after_business_write(
                    owner_kind,
                    "add_first_fact_event",
                    owner_kind,
                    fully_valid,
                )

    def publish_r07_first_ready(
        self,
        *,
        owner: object,
        first_ready: torch.Tensor,
        shot_key: object,
        source_step: torch.Tensor,
    ) -> None:
        """Join R07's monotonic first-ready write to the current full key."""

        with self._operation("publish R07 first ready"):
            self._healthy()
            record = self._publication.current
            if (
                record is None
                or self._fact_owner_identities.get("r07_recovery") is not owner
            ):
                raise ActionEpochError("R07 first-ready owner identity differs")
            rows = self._tensor(
                first_ready,
                label="R07.first_ready",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            step = self._tensor(
                source_step,
                label="R07.first_ready.source_step",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            try:
                key = row_identity.require_action_epoch_shot_key(
                    shot_key,
                    shape=(self.num_envs,),
                    device=self.device,
                    label="R07.first_ready.shot_key",
                )
            except row_identity.ActionEpochShotKeyError as exc:
                raise ActionEpochError(str(exc)) from exc
            joined = rows[:, None] & row_identity.action_epoch_shot_key_valid(
                record.identity.shot_key
            )
            for field in fields(ActionEpochShotKey):
                joined &= getattr(record.identity.shot_key, field.name).eq(
                    getattr(key, field.name)[:, None]
                )
            safe = self._latch_device_row_fault(
                rows & (joined.sum(dim=1).ne(1) | step.lt(0)),
                reason_bit=ROW_FAULT_R07_FIRST_READY_JOIN,
            )
            joined &= safe[:, None]
            names, values = self._event(joined, record.identity.shot_key)
            self._append(
                record,
                transition="R07_FIRST_READY",
                names=(*names, "source_step"),
                values=(
                    *values,
                    torch.where(
                        joined,
                        step[:, None],
                        torch.full(self._shot_shape, -1, dtype=torch.int64, device=self.device),
                    ),
                ),
                changes={},
            )
            self._milestone_after_business_write(
                "r07_recovery", "add_r07_first_ready", joined
            )

    def refresh_physical_launch_rows(self) -> None:
        """Pull Physical's active launch after-image and join its full row key."""

        with self._operation("refresh Physical launch rows"):
            self._healthy()
            record = self._publication.current
            if record is None or self._physical_launch_projection is None:
                raise ActionEpochError("Physical launch projection is not bound")
            try:
                from .action_ball_physical_flight_device import (
                    ActionEpochR06LaunchProjection,
                )
            except ImportError:  # pragma: no cover - focused standalone import
                from action_ball_physical_flight_device import (
                    ActionEpochR06LaunchProjection,
                )
            projection = self._physical_launch_projection()
            if type(projection) is not ActionEpochR06LaunchProjection:
                raise ActionEpochError(
                    "Physical launch projection must be exact ActionEpochR06LaunchProjection"
                )
            due = self._tensor(
                projection.due,
                label="Physical.launch.due",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            try:
                shot_key = row_identity.require_action_epoch_shot_key(
                    projection.shot_key,
                    shape=(self.num_envs,),
                    device=self.device,
                    label="Physical.launch.shot_key",
                )
            except row_identity.ActionEpochShotKeyError as exc:
                raise ActionEpochError(str(exc)) from exc
            publication_ordinal = self._tensor(
                projection.publication_ordinal,
                label="Physical.launch.publication_ordinal",
                shape=(self.num_envs,),
                dtype=torch.int64,
            )
            late = self._tensor(
                projection.late_launch,
                label="Physical.launch.late_launch",
                shape=(self.num_envs,),
                dtype=torch.bool,
            )
            target_xy_m = self._tensor(
                projection.target_xy_m,
                label="Physical.launch.target_xy_m",
                shape=(self.num_envs, 2),
                dtype=torch.float32,
            )
            current_key = self._gather_current_key(record.identity.shot_key)
            current_phase = self._gather_current(record.phase)
            current_publication = self._gather_current(
                record.publication_ordinal
            )
            joined = (
                row_identity.action_epoch_shot_key_valid(shot_key)
                & row_identity.action_epoch_shot_key_equal(shot_key, current_key)
                & current_phase.eq(PHASE_REVEAL_COMMITTED)
            )
            chronology = publication_ordinal.eq(current_publication)
            target_finite = torch.isfinite(target_xy_m).all(dim=1)
            invalid_rows = (
                (due & (~joined | ~chronology))
                | (late & ~due)
                | ~target_finite
            )
            safe_rows = self._latch_device_row_fault(
                invalid_rows, reason_bit=ROW_FAULT_PHYSICAL_LAUNCH_JOIN
            )
            safe_due = due & joined & chronology & safe_rows
            safe_late = late & safe_due
            event = self._current_slot_mask(safe_due)
            phase = torch.where(
                event,
                torch.full_like(record.phase, PHASE_LAUNCH_SETTLED),
                record.phase,
            )
            launch = torch.where(event, event, record.launch_succeeded)
            late_rows = torch.where(
                event, self._current_slot_mask(safe_late), record.late_launch
            )
            owner_slot = self._owner_slot("physical_ball")
            started = record.writes_started.clone()
            committed = record.writes_committed.clone()
            started[:, :, owner_slot] |= event
            committed[:, :, owner_slot] |= event
            event_target_xy_m = torch.where(
                event[..., None],
                target_xy_m[:, None, :],
                torch.zeros(
                    (self.num_envs, self.shot_slot_capacity, 2),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ).contiguous()
            names, values = self._event(event, record.identity.shot_key)
            record = self._append(
                record,
                transition="PHYSICAL_LAUNCH_ROWS",
                names=(
                    *names,
                    "publication_ordinal",
                    "launch_succeeded",
                    "late_launch",
                    "owner_fault_bits",
                    "target_xy_m",
                ),
                values=(
                    *values,
                    record.publication_ordinal,
                    launch,
                    late_rows,
                    record.owner_fault_bits[:, :, owner_slot],
                    event_target_xy_m,
                ),
                changes={
                    "phase": phase,
                    "launch_succeeded": launch,
                    "late_launch": late_rows,
                    "writes_started": started,
                    "writes_committed": committed,
                },
            )

    # ------------------------------------------------------------------
    # Publication-free selected reset lease followed by one masked commit

    def prepare_selected_true_reset(
        self,
        *,
        owner: object,
        top_preflight: object,
        selected_env_index: torch.Tensor,
        selected_mask: torch.Tensor,
        generation_before: torch.Tensor,
        generation_after: torch.Tensor,
        generation_overflow_fault: torch.Tensor,
        terminal_reset_facts_i64: torch.Tensor,
    ) -> ActionEpochPreparedSelectedReset:
        with self._operation("prepare selected true reset"):
            self._healthy()
            try:
                return self._selected_reset.prepare(
                    owner=owner,
                    top_preflight=top_preflight,
                    record=self._publication.current,
                    boundary_is_open=(
                        self._active_d05 is None
                        and not self._reward_open
                        and self._pending_drain is None
                    ),
                    selected_env_index=selected_env_index,
                    selected_mask=selected_mask,
                    generation_before=generation_before,
                    generation_after=generation_after,
                    generation_overflow_fault=generation_overflow_fault,
                    terminal_reset_facts_i64=terminal_reset_facts_i64,
                )
            except selected_reset.SelectedResetProtocolError as exc:
                raise ActionEpochError(str(exc)) from exc

    def abort_selected_true_reset(
        self, *, owner: object, prepared_reset: ActionEpochPreparedSelectedReset
    ) -> None:
        with self._operation("abort selected true reset"):
            try:
                self._selected_reset.abort(owner=owner, lease=prepared_reset)
            except selected_reset.SelectedResetProtocolError as exc:
                raise ActionEpochError(str(exc)) from exc

    def commit_selected_true_reset(
        self, *, owner: object, prepared_reset: ActionEpochPreparedSelectedReset
    ) -> ActionEpochRecord:
        with self._operation("commit selected true reset"):
            self._healthy()
            record = self._publication.current
            try:
                plan = self._selected_reset.plan_commit(
                    owner=owner,
                    lease=prepared_reset,
                    record=record,
                    boundary_is_quiescent=(
                        self._active_d05 is None
                        and not self._reward_open
                        and self._pending_drain is None
                    ),
                    idle_phase=PHASE_IDLE,
                )
            except selected_reset.SelectedResetProtocolError as exc:
                raise ActionEpochError(str(exc)) from exc
            if record is None:
                raise ActionEpochError("selected reset has no public record")
            record = self._append(
                record,
                transition="RESET_SELECTED",
                names=(
                    "selected_mask",
                    "reset_generation",
                    "terminal_reset_facts_i64",
                ),
                values=(
                    plan.selected_mask,
                    plan.generation_after,
                    plan.terminal_reset_facts_i64,
                ),
                changes=dict(plan.changes),
            )
            self._milestone_after_business_write(
                "r05_runtime", "reset_event_envs", plan.selected_mask
            )
            self._reset_generation = record.reset_generation
            self._undrained_row_fault_bits = torch.bitwise_or(
                self._undrained_row_fault_bits,
                torch.where(
                    plan.overflow,
                    ROW_FAULT_SELECTED_RESET_GENERATION_OVERFLOW,
                    0,
                ),
            )
            selected = plan.selected_mask
            self._current_payment_rows = ActionEpochRewardPaymentRows(
                valid=torch.where(
                    selected,
                    torch.zeros_like(self._current_payment_rows.valid),
                    self._current_payment_rows.valid,
                ),
                shot_key=self._where_key(
                    selected,
                    row_identity.empty_action_epoch_shot_key(
                        (self.num_envs,), device=self.device
                    ),
                    self._current_payment_rows.shot_key,
                ),
                payment_step=torch.where(
                    selected,
                    torch.full_like(self._current_payment_rows.payment_step, -1),
                    self._current_payment_rows.payment_step,
                ),
            )
            self._selected_reset.complete_commit(prepared_reset)
            return record.clone()

    # ------------------------------------------------------------------
    # Frozen journal transfer; only materialization may be followed by D2H

    def _drain_expected_names(self, transition: str) -> tuple[str, ...]:
        event = _SHOT_EVENT_NAMES
        if transition == "RESET_GENESIS_IDLE":
            return ("reset_generation", "reset_selected_mask")
        if transition == "RESET_SELECTED":
            return (
                "selected_mask",
                "reset_generation",
                "terminal_reset_facts_i64",
            )
        if transition == MOTION_CLOSED:
            return (*event, "motion_close_reason")
        if transition == "RETIRED":
            return (*event, "motion_close_reason", "payment_step", "retirement_step")
        if transition == "D05_SETTLED":
            return (
                *event,
                "due_mask",
                "selected_mask",
                "decision",
                "construction_admissible",
                "playback_admissible",
                "owner_fault_bits",
                "stroke_family_code",
                "action_attribution_valid",
            )
        if transition.startswith("WRITES_STARTED:") or transition.startswith(
            "WRITES_COMMITTED:"
        ):
            return event
        if transition == "D05_ACCEPT_PUBLISHED":
            return (*event, "publication_ordinal")
        if transition == MOTION_PLAYBACK_STARTED:
            return event
        if transition == "PHYSICAL_POSTPHYSICS_ROWS":
            return (
                *event,
                "publication_ordinal",
                "owner_fault_bits",
                "fact_valid_bits",
                "fact_source_step",
            )
        if transition == "R06_OUTCOME_ROWS":
            return (
                *event,
                "publication_ordinal",
                "settlement_step",
                "valid_bits",
                "outcome_code",
                "owner_fault_bits",
                "predicate_bits",
            )
        if transition == "R07_FIRST_READY":
            return (*event, "source_step")
        if transition == "REWARD_CYCLE_OPEN":
            return ("reward_cycle_age", "reward_cycle_fault")
        if transition == "REWARD_CONSUMER_PAID":
            return ("reward_consumer_ordinal", "reward_paid")
        if transition == "PAYMENT_RECORDED":
            return (*event, "payment_step")
        if transition.startswith("OWNER_FAULT:"):
            return (*event, "owner_fault_bits")
        if transition.startswith("OWNER_FACTS:"):
            return (*event, "valid_bits", "source_step", "qualified")
        if transition == "PHYSICAL_LAUNCH_ROWS":
            return (
                *event,
                "publication_ordinal",
                "launch_succeeded",
                "late_launch",
                "owner_fault_bits",
                "target_xy_m",
            )
        if (
            transition == "REENTRANT_POISON"
            or transition == "MOTION_PLAYBACK_POISON"
            or transition.startswith("OWNER_WRITE_POISON:")
            or transition.startswith("D05_WRITER_POISON:")
        ):
            return ("phase", "owner_fault_bits", "poison_reason")
        raise ActionEpochError("drain transition has no version-1 byte schema")

    def _drain_expected_value_spec(
        self, *, transition: str, name: str
    ) -> tuple[tuple[int, ...], torch.dtype]:
        if transition == "RESET_GENESIS_IDLE":
            return (
                ((self.num_envs,), torch.int64)
                if name == "reset_generation"
                else ((self.num_envs,), torch.bool)
            )
        if transition == "RESET_SELECTED":
            if name == "selected_mask":
                return (self.num_envs,), torch.bool
            if name == "reset_generation":
                return (self.num_envs,), torch.int64
            return (self.num_envs, 3), torch.int64
        if name in _SHOT_EVENT_NAMES:
            return self._shot_shape, (
                torch.bool if name == "event_mask" else torch.int64
            )
        if transition == "D05_SETTLED" and name == "owner_fault_bits":
            return self._owner_shape, torch.int64
        if name == "owner_fault_bits" and (
            transition == "REENTRANT_POISON"
            or transition == "MOTION_PLAYBACK_POISON"
            or transition.startswith("OWNER_WRITE_POISON:")
            or transition.startswith("D05_WRITER_POISON:")
        ):
            return self._owner_shape, torch.int64
        if transition == "REWARD_CYCLE_OPEN":
            return (self.num_envs,), torch.int64
        if transition == "REWARD_CONSUMER_PAID":
            if name == "reward_consumer_ordinal":
                return (self.num_envs,), torch.int64
            return (self.num_envs, REWARD_CONSUMER_COUNT), torch.bool
        if transition == "PHYSICAL_LAUNCH_ROWS" and name == "target_xy_m":
            return (*self._shot_shape, 2), torch.float32
        bool_names = {
            "due_mask",
            "selected_mask",
            "construction_admissible",
            "playback_admissible",
            "launch_succeeded",
            "late_launch",
            "action_attribution_valid",
            "qualified",
        }
        dtype = torch.bool if name in bool_names else torch.int64
        return self._shot_shape, dtype

    def _drain_layout(
        self, entries: tuple[CommitEntry, ...]
    ) -> tuple[_DrainTensorLayout, ...]:
        layout: list[_DrainTensorLayout] = []
        offset = 0
        for entry_index, entry in enumerate(entries):
            expected_names = self._drain_expected_names(entry.transition)
            if entry.delta.names != expected_names or len(entry.delta.values) != len(
                expected_names
            ):
                raise ActionEpochError("drain delta differs from version-1 schema")
            for value_index, (name, value) in enumerate(
                zip(expected_names, entry.delta.values)
            ):
                shape, dtype = self._drain_expected_value_spec(
                    transition=entry.transition, name=name
                )
                if (
                    type(value) is not torch.Tensor
                    or value.device != self.device
                    or value.dtype is not dtype
                    or tuple(value.shape) != shape
                ):
                    raise ActionEpochError(
                        "drain tensor differs from version-1 dtype/shape/device schema"
                    )
                numel = 1
                for extent in shape:
                    numel *= extent
                nbytes = numel * _DRAIN_DTYPE_NBYTES[dtype]
                layout.append(
                    _DrainTensorLayout(
                        version=_DRAIN_PACK_VERSION,
                        entry_index=entry_index,
                        value_index=value_index,
                        name=name,
                        offset=offset,
                        nbytes=nbytes,
                        dtype=dtype,
                        shape=shape,
                    )
                )
                offset += nbytes
        layout.append(
            _DrainTensorLayout(
                _DRAIN_PACK_VERSION, -2, -1, "milestone_i64", offset,
                milestone_tensors.I64_NUMEL * 8, torch.int64,
                (milestone_tensors.I64_NUMEL,),
            )
        )
        offset += milestone_tensors.I64_NUMEL * 8
        layout.append(
            _DrainTensorLayout(
                _DRAIN_PACK_VERSION, -3, -1, "milestone_f64", offset,
                milestone_tensors.F64_NUMEL * 8, torch.float64,
                (milestone_tensors.F64_NUMEL,),
            )
        )
        offset += milestone_tensors.F64_NUMEL * 8
        layout.append(
            _DrainTensorLayout(
                version=_DRAIN_PACK_VERSION,
                entry_index=-1,
                value_index=-1,
                name="row_fault_bits",
                offset=offset,
                nbytes=self.num_envs * 8,
                dtype=torch.int64,
                shape=(self.num_envs,),
            )
        )
        return tuple(layout)

    def _decode_drain_bytes(
        self,
        *,
        host_bytes: torch.Tensor,
        entries: tuple[CommitEntry, ...],
        layout: tuple[_DrainTensorLayout, ...],
    ) -> ActionEpochMaterializedDrain:
        expected_layout = self._drain_layout(entries)
        if layout != expected_layout:
            raise ActionEpochError("drain byte offsets or dtype/shape schema differ")
        expected_nbytes = layout[-1].offset + layout[-1].nbytes
        if (
            type(host_bytes) is not torch.Tensor
            or host_bytes.device.type != "cpu"
            or host_bytes.dtype is not torch.uint8
            or tuple(host_bytes.shape) != (expected_nbytes,)
            or not host_bytes.is_contiguous()
        ):
            raise ActionEpochError("drain byte payload is truncated, extended, or malformed")
        decoded: list[list[torch.Tensor]] = [list() for _ in entries]
        row_fault_bits: Optional[torch.Tensor] = None
        milestone_i64: Optional[torch.Tensor] = None
        milestone_f64: Optional[torch.Tensor] = None
        for segment in layout:
            raw = host_bytes.narrow(0, segment.offset, segment.nbytes).clone()
            value = raw.view(segment.dtype).reshape(segment.shape).contiguous()
            if segment.name == "milestone_i64":
                milestone_i64 = value
            elif segment.name == "milestone_f64":
                milestone_f64 = value
            elif segment.entry_index < 0:
                row_fault_bits = value
            else:
                decoded[segment.entry_index].append(value)
        if row_fault_bits is None or milestone_i64 is None or milestone_f64 is None:
            raise ActionEpochError("drain fixed tensor bytes are absent")
        decoded_entries = tuple(
            replace(
                entry,
                delta=PackedDelta(entry.delta.names, tuple(values)),
            )
            for entry, values in zip(entries, decoded)
        )
        return ActionEpochMaterializedDrain(
            entries=decoded_entries,
            row_fault_bits=row_fault_bits,
            milestone_i64=milestone_i64, milestone_f64=milestone_f64,
        )

    def prepare_drain(self) -> tuple[int, int]:
        with self._operation("prepare drain"):
            self._healthy()
            if (
                self._pending_drain is not None
                or self._active_d05 is not None
                or self._reward_open
                or self._current_closed_rows is not None
                or self._selected_reset.active
            ):
                raise ActionEpochError("drain overlaps a transaction")
            start, end = self._drain_frontier, self._commit_head
            count = sum(
                start <= entry.sequence < end
                for entry in self._publication.pending_log
            )
            if count != end - start:
                raise ActionEpochError("journal frontier is not contiguous")
            self.milestone.freeze_window_()
            self._pending_drain = (start, end)
            self._pending_drain_materialized = False
            return start, end

    def materialize_drain(
        self, *, start: int, end: int
    ) -> ActionEpochMaterializedDrain:
        with self._operation("materialize drain", allow_pending_drain=True):
            if self._pending_drain != (start, end) or self._pending_drain_materialized:
                raise ActionEpochError("drain materialization lease differs")
            entries = tuple(
                entry
                for entry in self._publication.pending_log
                if start <= entry.sequence < end
            )
            layout = self._drain_layout(entries)
            device_segments = tuple(
                value.detach().contiguous().view(torch.uint8).reshape(-1)
                for entry in entries
                for value in entry.delta.values
            ) + tuple(
                value.detach().contiguous().view(torch.uint8).reshape(-1)
                for value in self.milestone.pack_views()
            ) + (
                self._undrained_row_fault_bits.detach()
                .contiguous()
                .view(torch.uint8)
                .reshape(-1),
            )
            device_bytes = torch.cat(device_segments, dim=0).contiguous()
            host_bytes = _single_d2h_packed_bytes(device_bytes)
            materialized = self._decode_drain_bytes(
                host_bytes=host_bytes, entries=entries, layout=layout
            )
            decoded_row_fault = bool(
                materialized.row_fault_bits.ne(0).any()
            )
            self._pending_drain_materialized = True
            self._drain_decoded_row_fault = decoded_row_fault
            if decoded_row_fault:
                self._poisoned = True
            return materialized

    def abort_drain(self, *, start: int, end: int) -> None:
        with self._operation("abort drain", allow_pending_drain=True):
            if self._pending_drain != (start, end) or self._pending_drain_materialized:
                raise ActionEpochError("only an exact pre-materialization drain can abort")
            self.milestone.abort_window_()
            self._pending_drain = None

    def acknowledge_drain(self, *, start: int, end: int) -> None:
        with self._operation("acknowledge drain", allow_pending_drain=True):
            if (
                self._pending_drain != (start, end)
                or not self._pending_drain_materialized
                or self._drain_decoded_row_fault
            ):
                raise ActionEpochError("drain ACK lease differs")
            with torch.inference_mode(False):
                replacement_fault_bits = torch.zeros(
                    (self.num_envs,), dtype=torch.int64, device=self.device
                )
            self._publication = _Publication(
                self._publication.current,
                tuple(
                    entry
                    for entry in self._publication.pending_log
                    if entry.sequence >= end
                ),
            )
            self._drain_frontier = end
            self._pending_drain = None
            self._pending_drain_materialized = False
            self._drain_decoded_row_fault = False
            self._undrained_row_fault_bits = replacement_fault_bits
            self.milestone.clear_window_()

    def _checkpoint_record_specs(
        self,
    ) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        shot = self._shot_shape
        owner = self._owner_shape
        specs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        specs.extend(
            ("identity.shot_key." + field.name, shot, torch.int64)
            for field in fields(ActionEpochShotKey)
        )
        specs.extend(
            ("identity." + name, shot, torch.int64)
            for name in (
                "scheduled_ordinal", "target_generation", "selected_cell",
                "candidate_identity",
            )
        )
        specs.extend(
            ("clocks." + field.name, shot, torch.int64)
            for field in fields(EpochClockPayload)
        )
        specs.extend((
            ("task.task_f32", (*shot, TASK_F32_WIDTH), torch.float32),
            ("task.task_valid", shot, torch.bool),
            ("phase", shot, torch.int64),
            ("rng_counter", shot, torch.int64),
            ("current_task_slot", (self.num_envs,), torch.int64),
            ("publication_ordinal", shot, torch.int64),
            ("owner_fault_bits", owner, torch.int64),
            ("writes_started", owner, torch.bool),
            ("writes_committed", owner, torch.bool),
            ("launch_succeeded", shot, torch.bool),
            ("late_launch", shot, torch.bool),
            ("outcome_code", shot, torch.int64),
            ("reward_cycle_age", (self.num_envs,), torch.int64),
            ("reward_cycle_fault", (self.num_envs,), torch.int64),
            ("reward_cycle_open", (self.num_envs,), torch.bool),
            (
                "reward_due", (self.num_envs, REWARD_CONSUMER_COUNT),
                torch.bool,
            ),
            (
                "reward_paid", (self.num_envs, REWARD_CONSUMER_COUNT),
                torch.bool,
            ),
            ("fact_valid_bits", owner, torch.int64),
            ("fact_source_step", owner, torch.int64),
            ("fact_f32", (*owner, OWNER_FACT_F32_WIDTH), torch.float32),
            ("reset_generation", (self.num_envs,), torch.int64),
            ("reset_selected_mask", (self.num_envs,), torch.bool),
            ("motion_playback_started", shot, torch.bool),
            ("motion_close_reason", shot, torch.int64),
            ("settlement_step", shot, torch.int64),
            ("payment_step", shot, torch.int64),
            ("poison_reason", shot, torch.int64),
        ))
        return tuple(specs)

    @staticmethod
    def _payment_tensor_items(
        rows: ActionEpochRewardPaymentRows,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        items: list[tuple[str, torch.Tensor]] = [("payment.valid", rows.valid)]
        items.extend(
            ("payment.shot_key." + field.name, getattr(rows.shot_key, field.name))
            for field in fields(ActionEpochShotKey)
        )
        items.append(("payment.payment_step", rows.payment_step))
        return tuple(items)

    def _checkpoint_extra_items(
        self,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("owner.undrained_row_fault_bits", self._undrained_row_fault_bits),
            ("owner.action_uids_by_slot", self._action_uids_by_slot),
            ("owner.family_codes_by_slot", self._family_codes_by_slot),
            (
                "owner.last_r06_paid_payment_step",
                self._last_r06_paid_payment_step,
            ),
            *self._payment_tensor_items(self._current_payment_rows),
        )

    def _checkpoint_extra_specs(
        self, action_count: int,
    ) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        if type(action_count) is not int or action_count < 1:
            raise ActionEpochError("checkpoint action catalog must be nonempty")
        env = (self.num_envs,)
        specs: list[tuple[str, tuple[int, ...], torch.dtype]] = [
            ("owner.undrained_row_fault_bits", env, torch.int64),
            ("owner.action_uids_by_slot", (action_count,), torch.int64),
            ("owner.family_codes_by_slot", (action_count,), torch.int64),
            ("owner.last_r06_paid_payment_step", env, torch.int64),
            ("payment.valid", env, torch.bool),
        ]
        specs.extend(
            ("payment.shot_key." + field.name, env, torch.int64)
            for field in fields(ActionEpochShotKey)
        )
        specs.append(("payment.payment_step", env, torch.int64))
        return tuple(specs)

    def _require_restore_boundary(
        self, *, dormant: bool, allow_prepared: bool = False,
    ) -> ActionEpochRecord:
        del allow_prepared
        self._healthy()
        current = self._publication.current
        common_bad = (
            current is None
            or self._operation_active
            or self._active_d05 is not None
            or self._reward_open
            or self._current_closed_rows is not None
            or self._selected_reset.active
            or self._pending_drain is not None
            or self._pending_drain_materialized
            or self._drain_decoded_row_fault
            or bool(self._undrained_row_fault_bits.ne(0).any())
        )
        if common_bad:
            raise ActionEpochError("restore boundary is not quiescent")
        if dormant:
            if not self._canonical_genesis():
                raise ActionEpochError("restore target is not dormant genesis")
            _require_exact_tensor_items(
                _record_tensor_items(current),
                self._checkpoint_record_specs(),
                device=self.device,
                label="restore target current",
                require_disjoint=False,
            )
        elif (
            not self._genesis_activated
            or self._commit_head != self._drain_frontier
            or self._publication.pending_log
        ):
            raise ActionEpochError("checkpoint source is not an ACK boundary")
        assert current is not None
        return current

    def _state_extra_items(
        self, state: _ActionEpochCarryState,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            (
                "owner.undrained_row_fault_bits",
                state.undrained_row_fault_bits,
            ),
            ("owner.action_uids_by_slot", state.action_uids_by_slot),
            ("owner.family_codes_by_slot", state.family_codes_by_slot),
            (
                "owner.last_r06_paid_payment_step",
                state.last_r06_paid_payment_step,
            ),
            *self._payment_tensor_items(state.current_payment_rows),
        )

    def _require_carry_state(
        self, value: object,
    ) -> _ActionEpochCarryState:
        if type(value) is not _ActionEpochCarryState:
            raise ActionEpochError("restore state must be exact ActionEpoch carry")
        state = value
        if (
            type(state.num_envs) is not int
            or state.num_envs != self.num_envs
            or type(state.shot_slot_capacity) is not int
            or state.shot_slot_capacity != self.shot_slot_capacity
            or type(state.commit_head) is not int
            or type(state.drain_frontier) is not int
            or state.commit_head < 1
            or state.commit_head != state.drain_frontier
            or type(state.next_epoch) is not int
            or state.next_epoch < 0
            or type(state.reward_ordinal) is not int
            or state.reward_ordinal not in (0, REWARD_CONSUMER_COUNT)
            or type(state.last_motion_common_step) is not int
            or state.last_motion_common_step < -1
            or ((state.next_epoch == 0) != (state.last_motion_common_step == -1))
            or state.diagnostic_unauthorized is not True
            or type(state.current) is not ActionEpochRecord
            or type(state.current.epoch) is not int
            or state.current.epoch != -1
            or type(state.current.version) is not int
            or state.next_epoch > state.current.version
            or state.current.version != state.commit_head - 1
            or state.current.diagnostic_unauthorized is not True
            or type(state.current.identity) is not EpochIdentityPayload
            or type(state.current.identity.shot_key) is not ActionEpochShotKey
            or type(state.current.clocks) is not EpochClockPayload
            or type(state.current.task) is not EpochTaskPayload
            or type(state.current_payment_rows) is not ActionEpochRewardPaymentRows
            or type(state.action_uids_by_slot) is not torch.Tensor
            or type(state.family_codes_by_slot) is not torch.Tensor
        ):
            raise ActionEpochError("ActionEpoch carry scalar/type ABI differs")
        action_count = state.action_uids_by_slot.numel()
        epoch_items = (
            *_record_tensor_items(state.current),
            *self._state_extra_items(state),
        )
        epoch_specs = (
            *self._checkpoint_record_specs(),
            *self._checkpoint_extra_specs(action_count),
        )
        _require_exact_tensor_items(
            epoch_items,
            epoch_specs,
            device=torch.device("cpu"),
            label="checkpoint",
        )
        try:
            action_strata.ActionStrokeFamilyCatalog(
                tuple(int(value) for value in state.action_uids_by_slot.tolist()),
                tuple(int(value) for value in state.family_codes_by_slot.tolist()),
            )
        except (TypeError, ValueError) as exc:
            raise ActionEpochError("checkpoint action catalog differs") from exc
        if bool(state.undrained_row_fault_bits.ne(0).any()):
            raise ActionEpochError("checkpoint undrained row fault differs")
        phase = state.current.phase
        allowed_phase = torch.zeros_like(phase, dtype=torch.bool)
        for exact_phase in (
            PHASE_IDLE, PHASE_REVEAL_COMMITTED, PHASE_LAUNCH_SETTLED,
            PHASE_OUTCOME_SETTLED, PHASE_RETIRED,
        ):
            allowed_phase |= phase.eq(exact_phase)
        occupied = row_identity.action_epoch_shot_key_valid(
            state.current.identity.shot_key
        )
        current_slots = state.current.current_task_slot
        publication = state.current.publication_ordinal
        key_reset = state.current.identity.shot_key.reset_generation
        if bool((~allowed_phase).any()) or not torch.equal(
            occupied, ~phase.eq(PHASE_IDLE)
        ) or bool((current_slots < 0).any()) or bool(
            (current_slots >= self.shot_slot_capacity).any()
        ) or bool(current_slots.ne(0).any()) or bool(
            (occupied & (publication < 0)).any()
        ) or bool(
            (occupied & (publication >= state.next_epoch)).any()
        ) or bool((~occupied & publication.ne(-1)).any()) or bool(
            (
                occupied
                & key_reset.ne(state.current.reset_generation[:, None])
            ).any()
        ):
            raise ActionEpochError("checkpoint current phase/key occupancy differs")
        if bool(occupied.any()):
            slots = state.current.identity.shot_key.action_slot[occupied]
            if bool((slots < 0).any()) or bool((slots >= action_count).any()):
                raise ActionEpochError("checkpoint current action slot is foreign")
            expected_uid = state.action_uids_by_slot[slots]
            if not torch.equal(
                expected_uid, state.current.identity.shot_key.action_uid[occupied]
            ):
                raise ActionEpochError("checkpoint current action UID differs")
        payment_valid = state.current_payment_rows.valid
        payment_key_valid = row_identity.action_epoch_shot_key_valid(
            state.current_payment_rows.shot_key
        )
        payment_step = state.current_payment_rows.payment_step
        slot_index = current_slots[:, None]
        current_key = ActionEpochShotKey(**{
            field.name: getattr(
                state.current.identity.shot_key, field.name
            ).gather(1, slot_index).squeeze(1)
            for field in fields(ActionEpochShotKey)
        })
        current_phase = phase.gather(1, slot_index).squeeze(1)
        current_settlement = state.current.settlement_step.gather(
            1, slot_index
        ).squeeze(1)
        current_payment = state.current.payment_step.gather(
            1, slot_index
        ).squeeze(1)
        reward_due = state.current.reward_due.all(dim=1)
        payment_key_equal = row_identity.action_epoch_shot_key_equal(
            state.current_payment_rows.shot_key, current_key
        )
        paid_current = current_phase.eq(PHASE_OUTCOME_SETTLED) & current_payment.ge(0)
        if (
            bool((payment_valid & ~payment_key_valid).any())
            or bool((payment_valid & ~payment_key_equal).any())
            or bool(
                (payment_valid & current_phase.ne(PHASE_OUTCOME_SETTLED)).any()
            )
            or bool((payment_valid & ~reward_due).any())
            or bool((payment_valid & payment_step.lt(0)).any())
            or bool((payment_valid & payment_step.ne(current_payment)).any())
            or bool((payment_valid & current_settlement.lt(0)).any())
            or bool(
                (payment_valid & current_settlement.gt(payment_step)).any()
            )
            or bool((~payment_valid & payment_step.ne(-1)).any())
            or bool((paid_current & ~payment_valid).any())
            or bool((state.current.reset_generation < 0).any())
            or bool((state.current.reward_cycle_age < 0).any())
            or bool((state.last_r06_paid_payment_step < -1).any())
            or bool(
                (state.last_r06_paid_payment_step > state.last_motion_common_step).any()
            )
            or bool(state.current.reward_cycle_open.any())
            or not torch.equal(
                state.current.reward_due, state.current.reward_paid
            )
        ):
            raise ActionEpochError("checkpoint payment/reward chronology differs")
        return state

    def _lean_carry_schema(self) -> carry_txn._LeanCarrySchema:
        action_count = self._action_uids_by_slot.numel()
        specs = (*self._checkpoint_record_specs(), *self._checkpoint_extra_specs(action_count))
        return carry_txn._LeanCarrySchema(
            "epoch",
            (
                ("num_envs", int), ("shot_slot_capacity", int),
                ("commit_head", int), ("drain_frontier", int),
                ("next_epoch", int), ("reward_ordinal", int),
                ("last_motion_common_step", int), ("current_epoch", int),
                ("current_version", int),
            ),
            tuple(
                carry_txn._LeanCarryTensorSpec(
                    name, shape, dtype,
                    "attest" if name in (
                        "owner.action_uids_by_slot", "owner.family_codes_by_slot"
                    ) else "copy",
                )
                for name, shape, dtype in specs
            ),
        )

    def _lean_carry_construction_views(self):
        current = self._publication.current
        if current is None:
            raise ActionEpochError("ActionEpoch construction current is absent")
        return tuple(value for _name, value in (
            *_record_tensor_items(current), *self._checkpoint_extra_items()
        ))

    def _lean_carry_capture(self, lease: object) -> carry_txn._LeanCarryCapture:
        with self._lock:
            if (
                getattr(lease, "coordinator", None) is not self._lean_carry_coordinator
                or getattr(lease, "kind", None) != "capture"
            ):
                raise ActionEpochError("ActionEpoch carry lease differs")
            current = self._require_restore_boundary(dormant=False)
            if (
                self._reset_generation is not current.reset_generation
                or self._reward_cycle_age is not current.reward_cycle_age
                or self._reward_cycle_fault is not current.reward_cycle_fault
            ):
                raise ActionEpochError("live carry owner mirror differs")
            items = (*_record_tensor_items(current), *self._checkpoint_extra_items())
            return carry_txn._LeanCarryCapture((
                self.num_envs, self.shot_slot_capacity, self._commit_head,
                self._drain_frontier, self._next_epoch, self._reward_ordinal,
                self._last_motion_common_step, current.epoch, current.version,
            ), tuple(value for _name, value in items))

    def _lean_carry_state_from_host(self, scalars, values) -> _ActionEpochCarryState:
        record_count = len(_record_tensor_items(self._publication.current))
        template = replace(
            self._publication.current, epoch=scalars[7], version=scalars[8]
        )
        record = _record_from_tensor_values(template, values[:record_count])
        names = tuple(name for name, _value in self._checkpoint_extra_items())
        by_name = dict(zip(names, values[record_count:]))
        payment_key = ActionEpochShotKey(**{
            field.name: by_name["payment.shot_key." + field.name]
            for field in fields(ActionEpochShotKey)
        })
        return self._require_carry_state(_ActionEpochCarryState(
            num_envs=scalars[0], shot_slot_capacity=scalars[1],
            commit_head=scalars[2], drain_frontier=scalars[3],
            next_epoch=scalars[4], reward_ordinal=scalars[5],
            last_motion_common_step=scalars[6], current=record,
            undrained_row_fault_bits=by_name[
                "owner.undrained_row_fault_bits"
            ],
            action_uids_by_slot=by_name["owner.action_uids_by_slot"],
            family_codes_by_slot=by_name["owner.family_codes_by_slot"],
            last_r06_paid_payment_step=by_name["owner.last_r06_paid_payment_step"],
            current_payment_rows=ActionEpochRewardPaymentRows(
                valid=by_name["payment.valid"], shot_key=payment_key,
                payment_step=by_name["payment.payment_step"],
            ),
        ))

    def _lean_carry_stage(self, lease, scalars, host_tensors):
        with self._lock:
            if getattr(lease, "coordinator", None) is not self._lean_carry_coordinator:
                raise ActionEpochError("ActionEpoch target lease differs")
            self._require_restore_boundary(dormant=True)
            self._lean_carry_state_from_host(scalars, host_tensors)
            current = self._publication.current
            targets = tuple(value for _name, value in (
                *_record_tensor_items(current), *self._checkpoint_extra_items()
            ))
            staging = tuple(
                value.to(device=self.device, copy=True).contiguous()
                for value in host_tensors
            )
            return carry_txn._LeanCarryStage(scalars, staging, targets)

    def _lean_carry_target_views(self, lease, stage):
        current = self._publication.current
        if lease is not self._lean_carry_coordinator._active_lease or current is None:
            raise ActionEpochError("ActionEpoch commit target lease differs")
        return tuple(value for _name, value in (
            *_record_tensor_items(current), *self._checkpoint_extra_items()
        ))

    def _lean_carry_apply_scalars(self, lease, stage) -> None:
        with self._lock:
            if not stage.commit_started or lease is not self._lean_carry_coordinator._active_lease:
                raise ActionEpochError("ActionEpoch carry commit was not armed")
            scalars = stage.scalars
            record_count = len(_record_tensor_items(self._publication.current))
            template = replace(
                self._publication.current, epoch=scalars[7], version=scalars[8]
            )
            current = _record_from_tensor_values(template, stage.targets[:record_count])
            extra_names = tuple(name for name, _value in self._checkpoint_extra_items())
            by_name = dict(zip(extra_names, stage.targets[record_count:]))
            payment_key = ActionEpochShotKey(**{
                field.name: by_name["payment.shot_key." + field.name]
                for field in fields(ActionEpochShotKey)
            })
            self._publication = _Publication(current, ())
            self._commit_head, self._drain_frontier = scalars[2], scalars[3]
            self._next_epoch, self._reward_ordinal = scalars[4], scalars[5]
            self._last_motion_common_step = scalars[6]
            self._reset_generation = current.reset_generation
            self._reward_cycle_age = current.reward_cycle_age
            self._reward_cycle_fault = current.reward_cycle_fault
            self._undrained_row_fault_bits = by_name[
                "owner.undrained_row_fault_bits"
            ]
            self._last_r06_paid_payment_step = by_name[
                "owner.last_r06_paid_payment_step"
            ]
            self._current_payment_rows = ActionEpochRewardPaymentRows(
                valid=by_name["payment.valid"], shot_key=payment_key,
                payment_step=by_name["payment.payment_step"],
            )
            self._pending_drain = None
            self._pending_drain_materialized = False
            self._drain_decoded_row_fault = False
            self._active_d05 = None
            self._reward_open = False
            self._current_closed_rows = None
            self._genesis_activated = True

    def checkpoint(self) -> ActionEpochCheckpoint:
        """Return bounded carry state/frontiers; journal history is excluded."""

        with self._lock:
            self._healthy()
            current = self._publication.current
            if (
                current is None
                or self._active_d05 is not None
                or self._reward_open
                or self._pending_drain is not None
                or self._current_closed_rows is not None
                or self._selected_reset.active
                or self._commit_head != self._drain_frontier
            ):
                raise ActionEpochError("checkpoint boundary is not quiescent")
            return ActionEpochCheckpoint(
                commit_head=self._commit_head,
                drain_frontier=self._drain_frontier,
                next_epoch=self._next_epoch,
                reset_generation=self._reset_generation.clone(),
                reward_cycle_age=self._reward_cycle_age.clone(),
                current=current.clone(),
            )


__all__ = [
    "ACTION_EPOCH_ROW_FAULT_NAMES",
    "ActionEpochIdleObservationChronology",
    "ActionEpochCheckpoint",
    "ActionEpochClosedRows",
    "ActionEpochCurrentShotProjection",
    "ActionEpochD05AcceptedRows",
    "ActionEpochD05CandidateProjection",
    "ActionEpochDueRows",
    "ActionEpochError",
    "ActionEpochMaterializedDrain",
    "ActionEpochMotionPlaybackProjection",
    "ActionEpochOwner",
    "ActionEpochPreparedSelectedReset",
    "ActionEpochRecord",
    "ActionEpochRewardPaymentRows",
    "ActionEpochShotKey",
    "CommitEntry",
    "EpochClockPayload",
    "EpochIdentityPayload",
    "EpochTaskPayload",
    "PackedDelta",
    "ROW_FAULT_D05_RESET_GENERATION_JOIN",
    "ROW_FAULT_MOTION_CADENCE_OVERDUE",
    "ROW_FAULT_MOTION_CLOSE_CONTRACT",
    "ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT",
    "ROW_FAULT_MOTION_SWING_GENERATION_OVERFLOW",
    "ROW_FAULT_MOTION_TASK_TIMING_CONTRACT",
    "ROW_FAULT_OWNER_FACT_ACTIVE_JOIN",
    "ROW_FAULT_PHYSICAL_LAUNCH_JOIN",
    "ROW_FAULT_PHYSICAL_POSTPHYSICS_JOIN",
    "ROW_FAULT_PHYSICAL_POSTPHYSICS_NONFINITE",
    "ROW_FAULT_PHYSICAL_POSTPHYSICS_PRODUCER",
    "ROW_FAULT_R03_EPOCH_IDENTITY",
    "ROW_FAULT_R03_NONFINITE_FACT",
    "ROW_FAULT_R03_STALE_SOURCE_STEP",
    "ROW_FAULT_R06_CLOSED_DEBT_MISMATCH",
    "ROW_FAULT_R06_CLOSED_PROJECTION_CONTRACT",
    "ROW_FAULT_R06_CURRENT_FLIGHT_DUPLICATE",
    "ROW_FAULT_R06_LAUNCH_IDENTITY_CONTRACT",
    "ROW_FAULT_R06_LAUNCH_SELECTION_CONTRACT",
    "ROW_FAULT_R06_OUTCOME_JOIN",
    "ROW_FAULT_R06_OWNER_ENGINE_OVERFLOW",
    "ROW_FAULT_R06_OWNER_NONFINITE",
    "ROW_FAULT_R06_OWNER_OTHER",
    "ROW_FAULT_R06_OWNER_PRODUCER_CONTRACT",
    "ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE",
    "ROW_FAULT_R06_PAYMENT_BEFORE_SETTLEMENT",
    "ROW_FAULT_R06_PAYMENT_HIGHWATER_REGRESSION",
    "ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE",
    "ROW_FAULT_R06_PAYMENT_MISSING_OR_MISMATCHED",
    "ROW_FAULT_R06_PAYMENT_PROJECTION_CONTRACT",
    "ROW_FAULT_R06_PAYMENT_UNCONSUMED_DEBT_OVERWRITE",
    "ROW_FAULT_R06_PREVIOUS_PAID_CONTRACT",
    "ROW_FAULT_R07_FIRST_READY_JOIN",
    "ROW_FAULT_R07_TERMINAL_FACT_CONTRACT",
    "ROW_FAULT_RESET_GENESIS_CONTRACT",
    "ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY",
    "ROW_FAULT_SELECTED_RESET_GENERATION_OVERFLOW",
    "action_epoch_open_shot_phase_mask",
]
