#!/usr/bin/env python3
"""Device-resident D05 owner for fixed full-N ActionBall transactions.

ActionEpoch owns settlement chronology; D05 owns candidate construction,
private sampler state, opaque row capabilities, and ACCEPT-only publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import struct
from threading import RLock
from typing import Optional, Protocol, Tuple

import torch


def _require_action_epoch_module() -> object:
    """Import the one code-owned lean epoch implementation."""

    try:
        return importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
        )
    except ModuleNotFoundError:
        return importlib.import_module("action_ball_full_mdp_epoch")


def _require_lean_carry_module() -> object:
    """Import the root-private carry ABI without exporting leaf authority."""

    try:
        return importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_checkpoint_txn"
        )
    except ModuleNotFoundError:
        return importlib.import_module("action_ball_full_mdp_lean_checkpoint_txn")


def _require_canonical_action_epoch_idle(
    owner: object,
    *,
    epoch_module: object,
    device: torch.device,
    num_envs: int,
) -> None:
    """Require the one explicit genesis IDLE publication before D05 binds."""

    if (
        type(owner) is not epoch_module.ActionEpochOwner
        or owner.num_envs != num_envs
        or owner.shot_slot_capacity != 1
        or owner.device != device
        or owner.poisoned
        or owner.commit_head != 1
        or owner.drain_frontier != 0
    ):
        raise DeviceR05Error(
            "diagnostic ActionEpoch is not the canonical genesis IDLE owner"
        )
    try:
        current = owner.current()
    except Exception as exc:
        raise DeviceR05Error(
            "diagnostic ActionEpoch genesis IDLE publication is absent"
        ) from exc
    if (
        type(current) is not epoch_module.ActionEpochRecord
        or current.epoch != -1
        or current.version != 0
    ):
        raise DeviceR05Error(
            "diagnostic ActionEpoch genesis IDLE chronology differs"
        )
    # Device values are not decoded here.  The exact Epoch owner validates its
    # own canonical genesis atomically when the three D05 writers bind; a
    # second same-writer projection would add no independent authority.


INTEGRATION_STATUS = "device_r05_hot_owner_integration_hold"
GLOBAL_DRAIN_OWNER_KIND = "r05_runtime"

DECISION_ACCEPT = 1
DECISION_CENSOR = 2
DECISION_CONSTRUCTION_REJECT = 3
DECISION_DEFER = 4

JOURNAL_ABORT = 1
JOURNAL_CONSTRUCTION_REJECT = 2
JOURNAL_ACCEPT = 3
JOURNAL_CENSOR = 4
JOURNAL_TRUE_RESET = 5
JOURNAL_MIXED_EPOCH = 6

REASON_ADMISSIBLE = 0
REASON_NO_FEASIBLE_TARGET = 1
REASON_ONLY_PREVIOUS_TARGET_FEASIBLE = 2
REASON_BATCH_PEER_INFEASIBLE = 3
REASON_ABORTED_BEFORE_TRANSFER = 4
REASON_TRUE_RESET = 5
FAULT_RESET_GENERATION_OVERFLOW = 12
PRODUCER_FAULT_QUESTION_CHRONOLOGY = 1 << 50

SEQUENCE_EMPTY = 0
SEQUENCE_COMMITTED = 1
SEQUENCE_INFRA_CENSORED = 2

CHILD_OWNER_ORDER = (
    "motion",
    "racket",
    "physical_ball",
    "r06_flight",
)
GENESIS_CONSUMER_ORDER = (*CHILD_OWNER_ORDER, "full_mdp_env",)

MOTION_TASK_F32_FIELDS = (
    "time_to_contact_s",
    "teacher_rate",
    "scaled_t_hit_s",
    "scaled_t_cycle_s",
    "pre_swing_wait_s",
)
QUESTION_CONSTRUCTION_REASON_ADMITTED = -1
QUESTION_CONSTRUCTION_REASON_MIN_REJECT = 0
QUESTION_CONSTRUCTION_REASON_MAX_REJECT = 13
QUESTION_CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL = 13
RACKET_F32_FIELDS = (
    "racket_site_target_env_x",
    "racket_site_target_env_y",
    "racket_site_target_env_z",
    "racket_site_velocity_w_x",
    "racket_site_velocity_w_y",
    "racket_site_velocity_w_z",
    "racket_normal_w_x",
    "racket_normal_w_y",
    "racket_normal_w_z",
    "ball_contact_target_env_x",
    "ball_contact_target_env_y",
    "ball_contact_target_env_z",
    "racket_face_center_velocity_w_x",
    "racket_face_center_velocity_w_y",
    "racket_face_center_velocity_w_z",
    "command_quaternion_w",
    "command_quaternion_x",
    "command_quaternion_y",
    "command_quaternion_z",
    "base_goal_env_x",
    "base_goal_env_y",
    "incoming_velocity_w_x",
    "incoming_velocity_w_y",
    "incoming_velocity_w_z",
    "incoming_spin_w_x",
    "incoming_spin_w_y",
    "incoming_spin_w_z",
)
PHYSICAL_STATE_F32_FIELDS = (
    "position_env_m_x",
    "position_env_m_y",
    "position_env_m_z",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "linear_velocity_world_mps_x",
    "linear_velocity_world_mps_y",
    "linear_velocity_world_mps_z",
    "angular_velocity_world_radps_x",
    "angular_velocity_world_radps_y",
    "angular_velocity_world_radps_z",
)

_U16_MASK = 0xFFFF
_U32_MASK = 0xFFFFFFFF
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_MAX_SUPPORT = (1 << 31) - 1
_I64_MAX = (1 << 63) - 1
INTERNAL_QUESTION_DRAW_WIDTH = 6
INTERNAL_QUESTION_REDRAW_ROUNDS = 3
INTERNAL_QUESTION_TOTAL_DRAW_WIDTH = (
    INTERNAL_QUESTION_DRAW_WIDTH * INTERNAL_QUESTION_REDRAW_ROUNDS + 1
)

_LIVE_PUBLICATION_TENSOR_NAMES = frozenset(
    {
        "_rng_lo",
        "_rng_hi",
        "_draw_count",
        "_target_generation",
        "_previous_cell_index",
        "_reset_generation",
        "_scheduled_ordinal",
        "_outcome_shot_index",
        "_sequence_kind",
        "_task_identity",
        "_outcome_identity",
        "_ball_identity",
        "_next_outcome_identity",
        "_next_ball_identity",
        "_policy_opportunity",
        "_mutation_version",
        "_poisoned",
        "_poison_reason",
    }
)
_PUBLICATION_REGISTRY_NAMES = frozenset(
    {
        "_prepared_records",
        "_preview_records",
        "_pretransfer_records",
        "_claim_records",
        "_armed_records",
        "_terminal_receipts",
        "_prepared_true_resets",
        "_true_reset_receipts",
    }
)
_PUBLICATION_SCALAR_NAMES = frozenset(
    {
        "_active",
        "_journal_head",
        "_mutation_version_host",
        "_last_candidate_bank_sequence",
        "_next_candidate_identity",
    }
)
_LEAN_CARRY_COPY_NAMES = (
    "next_outcome_identity", "next_ball_identity", "rng_lo", "rng_hi",
    "draw_count", "target_generation", "previous_cell_index",
    "reset_generation", "scheduled_ordinal", "outcome_shot_index",
    "sequence_kind", "task_identity", "outcome_identity", "ball_identity",
    "policy_opportunity", "mutation_version",
)
_LEAN_CARRY_ATTEST_NAMES = (
    "profile_targets_xy_m", "poisoned", "poison_reason",
    "genesis_reset_generations", "row_axis",
)
_JOURNAL_FIELD_NAMES = (
    "meta",
    "selected",
    "feasible",
    "construction_reason",
    "candidate_identity",
    "round_construction_reason",
    "round_candidate_identity",
    "round_producer_fault",
    "chosen_round",
    "rounds_attempted",
    "reason",
    "rng_before_lo",
    "rng_before_hi",
    "rng_after_lo",
    "rng_after_hi",
    "draw_before",
    "draw_after",
    "sampler_generation_before",
    "sampler_generation_after",
    "previous_before",
    "previous_after",
    "reset_before",
    "reset_after",
    "ordinal_before",
    "ordinal_after",
    "outcome_before",
    "outcome_after",
    "selected_cell",
    "candidate_bank_sequence",
    "selected_candidate_identity",
    "task_identity",
    "outcome_identity",
    "ball_identity",
    "cadence_identity",
    "primary_fault",
    "question_producer_fault",
    "settlement",
    "child_completion",
)
_ROUND_INTERNAL_JOURNAL_FIELD_NAMES = frozenset(
    {
        "round_construction_reason",
        "round_candidate_identity",
        "round_producer_fault",
        "chosen_round",
        "rounds_attempted",
    }
)
_GLOBAL_JOURNAL_FIELD_NAMES = tuple(
    name
    for name in _JOURNAL_FIELD_NAMES
    if name not in _ROUND_INTERNAL_JOURNAL_FIELD_NAMES
)

GLOBAL_DRAIN_SCALAR_FIELD_NAMES = (
    "mutation_version",
    "fault_count",
    "invariant_count",
    "terminal_resolution_total",
    "policy_opportunity_total",
    "journal_count",
    "journal_start_sequence",
    "journal_end_sequence",
)
GLOBAL_DRAIN_JOURNAL_FIELD_NAMES = tuple(
    f"journal_{name}" for name in _GLOBAL_JOURNAL_FIELD_NAMES
)
GLOBAL_DRAIN_FIELD_NAMES = (
    *GLOBAL_DRAIN_SCALAR_FIELD_NAMES,
    *GLOBAL_DRAIN_JOURNAL_FIELD_NAMES,
)


class DeviceR05Error(RuntimeError):
    """Base contract error."""


class DeviceR05ConflictError(DeviceR05Error):
    """A stale, foreign, replayed, or out-of-phase capability was used."""


class DeviceR05PoisonedError(DeviceR05Error):
    """The owner failed closed after an irreversible boundary or overflow."""


class DeviceR05FullMdpIdentityProductionHold(DeviceR05Error):
    """Upstream authorities did not issue the complete hot-leaf identity."""


class DeviceCadenceAuthority(Protocol):
    """Construction-bound no-argument full-N Motion projection."""

    def project_current_action_epoch_rows(self) -> object: ...


class DeviceRevealBoundaryAuthority(Protocol):
    """Owner of the one packed reveal transfer and its terminal decision.

    This authority receives the exact owner-retained preview, validates its
    feasibility/admissibility and primary-fault lanes during that transfer,
    and later validates four real child arm/commit identities in fixed order.
    Merely returning the child names or four freshly allocated objects does
    not satisfy this protocol.
    """

    def require_owned_r05_reveal_boundary(
        self,
        preview: "DeviceR05PreviewToken",
        receipt: object,
        *,
        owner_view: "DeviceR05RevealBoundaryInput",
    ) -> "DeviceRevealBoundaryProjection": ...

    def require_owned_r05_terminal_arm(
        self, claim: "DeviceR05TerminalClaim", receipt: object
    ) -> "DeviceTerminalArmProjection": ...

    def require_owned_r05_terminal_commit(
        self, armed: "DeviceR05ArmedTerminal", receipt: object
    ) -> "DeviceTerminalCommitProjection": ...


class DeviceChildCompletionAuthority(Protocol):
    """Exact validator for one construction-fixed child owner."""

    def require_owned_r05_child_completion(
        self, receipt: object
    ) -> "DeviceChildCompletionProjection": ...


class DeviceDrainAuthority(Protocol):
    """Legacy constructor placeholder for the retired local drain path.

    Device-R05 now participates directly as the ``r05_runtime`` leaf of the
    seven-owner global PPO drain.  These methods are deliberately never called
    by production code; the constructor argument remains temporarily so old
    factories fail by explicit tombstone rather than silently selecting a
    second materialization/D2H path.
    """

    def materialize_r05_device_drain(
        self, batch: "DeviceR05DrainView"
    ) -> object: ...

    def require_owned_r05_drain_ack(
        self, receipt: object
    ) -> "DeviceDrainAckProjection": ...


class DeviceCheckpointAuthority(Protocol):
    """External authority retaining an exact cold checkpoint.

    The checkpoint must be retained after the global stream sync and attest
    deterministic C03 invariants in addition to its exact bytes.
    """

    def require_owned_r05_device_checkpoint(
        self, receipt: object
    ) -> "DeviceR05Checkpoint": ...


class DeviceGenesisAuthority(Protocol):
    """Construction authority for the exact live reset-generation genesis.

    It is issued by independent world-reset chronology, not reconstructed from
    a caller tuple or from a device-to-host observation.
    """

    def require_owned_r05_genesis(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
    ) -> "DeviceGenesisProjection": ...


class DeviceTrueResetAuthority(Protocol):
    """Top reset-event and four-child commit coordinator.

    ``project_r05_true_reset`` must bind the event selection to the supplied
    live generation.  ``require_owned_r05_true_reset_commit`` must validate
    four real owner-issued child commit identities in ``CHILD_OWNER_ORDER``;
    names or caller booleans are not sufficient evidence.  The event authority
    owns selection range/order/uniqueness, mask/index equality, and the live
    generation join.  Abort authority is issued only while no child commit has
    begun.
    """

    def project_r05_true_reset(
        self,
        receipt: object,
        *,
        device: torch.device,
        num_envs: int,
        live_reset_ledger_identity: object,
        live_reset_generation: torch.Tensor,
    ) -> "DeviceTrueResetEventProjection": ...

    def require_owned_r05_true_reset_commit(
        self,
        prepared: "DeviceR05PreparedTrueReset",
        *,
        owner_view: "DeviceR05TrueResetCommitInput",
    ) -> "DeviceTrueResetCommitProjection": ...

    def require_owned_r05_true_reset_preflight(
        self,
        prepared: "DeviceR05PreparedTrueReset",
        *,
        preflight_capability: object,
    ) -> "DeviceTrueResetPreflightProjection": ...

    def require_owned_r05_true_reset_abort(
        self, prepared: "DeviceR05PreparedTrueReset"
    ) -> "DeviceTrueResetAbortProjection": ...

    def require_owned_r05_true_reset_child_completion(
        self,
        receipt: "DeviceR05TrueResetReceipt",
        *,
        child_kind: str,
        child_receipt: object,
    ) -> "DeviceTrueResetChildCompletionProjection": ...


@dataclass(frozen=True)
class DeviceCadenceProjection:
    selected_count: int
    selected_env_index: torch.Tensor
    episode_tick: torch.Tensor
    reveal_tick: torch.Tensor
    deadline_tick: torch.Tensor
    next_reveal_tick: torch.Tensor
    swing_generation: torch.Tensor
    ready_at_reveal: torch.Tensor
    action_slot: torch.Tensor
    pending_elapsed_s: torch.Tensor
    reset_generation: torch.Tensor
    scheduled_ordinal: torch.Tensor
    outcome_shot_index: torch.Tensor
    sampler_generation: torch.Tensor
    task_identity: torch.Tensor
    cadence_identity: torch.Tensor
    cadence_producer_fault: Optional[torch.Tensor] = None
    action_uid: Optional[torch.Tensor] = None
    sequence_kind: Optional[torch.Tensor] = None
    task_sha256: Optional[torch.Tensor] = None
    cadence_sha256: Optional[torch.Tensor] = None
    task_receipt_sha256: Optional[torch.Tensor] = None
    cadence_receipt_sha256: Optional[torch.Tensor] = None
    contact_tick: Optional[torch.Tensor] = None
    launch_tick: Optional[torch.Tensor] = None
    chosen_horizon_ticks: Optional[torch.Tensor] = None
    task_close_tick: Optional[torch.Tensor] = None
    cadence_owner_receipt_identity: Optional[object] = None


@dataclass(frozen=True)
class DeviceR05CandidateBank:
    """Opaque-producer numeric bank indexed by ``[selected, cell]``.

    Every tensor is device-resident.  The construction-bound question/solver
    authority owns the exact field semantics and cross-owner joins.  R05 only
    selects a cell and slices these rows; it never manufactures candidate
    numerics or treats its own copies as independent facts.
    """

    candidate_identity: torch.Tensor
    construction_reason: torch.Tensor
    motion_task_f32: torch.Tensor
    racket_task_f32: torch.Tensor
    physical_state_f32: torch.Tensor


@dataclass(frozen=True)
class DeviceR05CandidateRoundBank:
    """Internal fixed-redraw bank indexed by ``[selected, round, cell]``.

    The recurring diagnostic question owner computes all three rounds in one
    device call.  D05 remains the sole selector: it applies previous-cell
    exclusion, producer-fault precedence, and the one final selection draw.
    The private journal retains the complete raw tape plus the chosen round
    and number of rounds attempted.
    """

    candidate_identity: torch.Tensor
    construction_reason: torch.Tensor
    producer_fault: torch.Tensor
    motion_task_f32: torch.Tensor
    racket_task_f32: torch.Tensor
    physical_state_f32: torch.Tensor


@dataclass(frozen=True)
class DeviceProfileProjection:
    profile_sha256: str
    profile_binding_sha256: str
    cell_ids: Tuple[str, ...]
    semantic_sha256s: Tuple[str, ...]
    targets_xy_m: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.targets_xy_m.device

    @property
    def support_size(self) -> int:
        return len(self.cell_ids)


class DeviceProfileAuthority(Protocol):
    """Independent cold authority for exact profile bytes and identity."""

    def require_owned_r05_profile(
        self, receipt: object
    ) -> DeviceProfileProjection: ...


@dataclass(frozen=True)
class DeviceQuestionChronology:
    """Question-owned per-candidate action and absolute tick chronology."""

    action_uid: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_ticks: torch.Tensor
    task_close_tick: torch.Tensor


@dataclass(frozen=True)
class DeviceQuestionRoundChronology:
    """Question-owned chronology indexed by ``[selected, round, cell]``."""

    action_uid: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_ticks: torch.Tensor
    task_close_tick: torch.Tensor


@dataclass(frozen=True)
class DeviceQuestionProjection:
    """Question/solver-owned candidate bank bound to one cadence selection."""

    cadence_receipt_identity: object
    bank_identity: object
    bank_sequence: int
    bank: Optional[DeviceR05CandidateBank]
    producer_fault: Optional[torch.Tensor]
    selected_count: int
    support_size: int
    chronology: Optional[DeviceQuestionChronology] = None
    round_bank: Optional[DeviceR05CandidateRoundBank] = None
    round_chronology: Optional[DeviceQuestionRoundChronology] = None
    full_key_sha256: Optional[torch.Tensor] = None
    task_sha256: Optional[torch.Tensor] = None
    physical_question_receipt_identity: Optional[object] = None


class DeviceQuestionAuthority(Protocol):
    """Independent producer/validator for the complete per-cell numeric bank.

    The authority owns monotonic ``bank_sequence`` and binds each positive,
    unique candidate identity to the frozen profile/cell order and complete
    numeric rows.  R05 deliberately does not re-label its own lane checks as
    independent solver evidence.
    """

    def project_r05_candidate_bank(
        self,
        receipt: object,
        *,
        cadence_receipt: object,
        cadence_projection: DeviceCadenceProjection,
        device: torch.device,
        support_size: int,
    ) -> DeviceQuestionProjection: ...


class DeviceInternalQuestionCompositionAuthority(Protocol):
    """Construction-bound question seam for D05-private cadence facts.

    This is deliberately not another cadence receipt.  D05 invokes the
    retained composer itself, immediately after its private cadence
    projection.  Source completeness is a construction-time responsibility;
    once composition starts, failure is an irreversible partial-write
    boundary.
    """

    def compose_r05_candidate_bank_inside_prepare(
        self,
        internal_context: object,
    ) -> DeviceQuestionProjection: ...


class _DeviceR05InternalQuestionContext:
    """Ephemeral whole-call capability; not a cadence or tensor receipt."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("D05 internal question contexts are owner-issued")


_ACTIVE_INTERNAL_QUESTION_CONTEXTS: dict[object, tuple[object, ...]] = {}
_ACTIVE_INTERNAL_QUESTION_CONTEXTS_LOCK = RLock()


def _consume_internal_question_context(
    internal_context: object,
    authority: object,
) -> tuple[
    object,
    DeviceCadenceProjection,
    DeviceProfileProjection,
    torch.device,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Consume one D05 whole-call context before any composer mutation."""

    with _ACTIVE_INTERNAL_QUESTION_CONTEXTS_LOCK:
        retained = _ACTIVE_INTERNAL_QUESTION_CONTEXTS.pop(
            internal_context, None
        )
        if type(internal_context) is not _DeviceR05InternalQuestionContext:
            retained = None
        if retained is None or retained[0] is not authority:
            raise DeviceR05ConflictError(
                "internal question composition context is foreign or inactive"
            )
        return retained[1:]  # type: ignore[return-value]


@dataclass(frozen=True)
class DeviceGenesisProjection:
    world_reset_identity: object
    reset_generations: torch.Tensor


class _OpaqueDeviceR05Capability:
    """Empty owner-issued identity; ordinary construction/copy is forbidden."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("Device-R05 capabilities are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Device-R05 capabilities are immutable")

    def __copy__(self):
        raise TypeError("Device-R05 capabilities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Device-R05 capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("Device-R05 capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Device-R05 capabilities cannot be serialized")


class DeviceR05GenesisProjection(_OpaqueDeviceR05Capability):
    """Opaque construction capability; fields are returned only by validator."""

    __slots__ = ()


@dataclass(frozen=True)
class DeviceR05GenesisView:
    """Clone-only data returned after exact projection registry validation."""

    device_r05_owner: "DeviceR05Owner"
    owner_kind: str
    world_reset_identity: object
    reset_generation: torch.Tensor


class DeviceR05EnvResetBinding(_OpaqueDeviceR05Capability):
    """Opaque construction capability for the top/env reset coordinator."""

    __slots__ = ()


class DeviceR05LiveResetLedger(_OpaqueDeviceR05Capability):
    """Opaque identity for the sole Device-R05 reset-generation writer."""

    __slots__ = ()


@dataclass(frozen=True)
class DeviceR05EnvResetBindingView:
    """Opaque ledger identity plus clone-only construction snapshot."""

    device_r05_owner: "DeviceR05Owner"
    world_reset_identity: object
    live_reset_ledger_identity: DeviceR05LiveResetLedger
    reset_generation_snapshot: torch.Tensor


@dataclass(frozen=True)
class _GenesisProjectionRecord:
    projection: DeviceR05GenesisProjection
    owner_kind: str
    view: DeviceR05GenesisView


@dataclass(frozen=True)
class DeviceRevealBoundaryProjection:
    preview_identity: object
    construction_admissible: bool
    owner_fault_present: bool
    decision: int
    primary_fault: torch.Tensor
    transfer_sequence: int


@dataclass(frozen=True)
class DeviceTerminalArmProjection:
    claim_identity: object
    decision: int
    child_kinds: Tuple[str, ...]
    child_arm_identities: Tuple[object, ...]


@dataclass(frozen=True)
class DeviceTerminalCommitProjection:
    armed_identity: object
    claim_identity: object
    decision: int
    child_kinds: Tuple[str, ...]
    child_commit_identities: Tuple[object, ...]


@dataclass(frozen=True)
class DeviceChildCompletionProjection:
    terminal_identity: object
    child_kind: str
    decision: int


@dataclass(frozen=True)
class DeviceDrainAckProjection:
    drain_identity: object
    start_sequence: int
    end_sequence: int
    global_drain_sequence: int
    materialized: bool
    continuation_allowed: bool


@dataclass(frozen=True)
class DeviceTrueResetEventProjection:
    reset_event_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor


@dataclass(frozen=True)
class DeviceR05PreparedTrueResetProjection:
    prepared_true_reset: "DeviceR05PreparedTrueReset"
    owner_kind: str
    prepared_identity: object
    reset_event_identity: object
    selected_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    generation_overflow_fault: torch.Tensor
    writer_fault: torch.Tensor


@dataclass(frozen=True)
class DeviceR05TerminalClaimProjection:
    """Exact boundary decision joined to one retained prepared reveal."""

    terminal_claim: "DeviceR05TerminalClaim"
    owner_kind: str
    claim_identity: object
    preview_identity: object
    decision: int
    primary_fault: torch.Tensor


@dataclass(frozen=True)
class DeviceR05TerminalReceiptProjection:
    """Read-only R05-last publication fact for one fixed child owner."""

    terminal_receipt: "DeviceR05TerminalReceipt"
    owner_kind: str
    terminal_identity: object
    decision: int
    journal_sequence: int


@dataclass(frozen=True)
class DeviceTrueResetCommitProjection:
    prepared_true_reset: "DeviceR05PreparedTrueReset"
    reset_event_identity: object
    child_kinds: Tuple[str, ...]
    child_commit_identities: Tuple[object, ...]
    preflight_capability: object = None


@dataclass(frozen=True)
class DeviceTrueResetPreflightProjection:
    prepared_true_reset: "DeviceR05PreparedTrueReset"
    reset_event_identity: object
    preflight_capability: object


@dataclass(frozen=True)
class DeviceR05TrueResetCommitInput:
    """Clone-only writer facts settled by the global reset authority."""

    prepared_true_reset: "DeviceR05PreparedTrueReset"
    reset_event_identity: object
    selected_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    generation_overflow_fault: torch.Tensor


@dataclass(frozen=True)
class DeviceTrueResetAbortProjection:
    prepared_true_reset: "DeviceR05PreparedTrueReset"
    reset_event_identity: object
    child_commits_started: bool


@dataclass(frozen=True)
class DeviceTrueResetChildCompletionProjection:
    true_reset_receipt: "DeviceR05TrueResetReceipt"
    child_kind: str
    child_receipt: object


def _snapshot_device_profile(
    profile_authority: DeviceProfileAuthority,
    profile_receipt: object,
) -> DeviceProfileProjection:
    """Validate and privately snapshot a cold profile before any callback."""

    validator = getattr(profile_authority, "require_owned_r05_profile", None)
    if not callable(validator):
        raise DeviceR05Error("profile authority is not callable")
    profile = validator(profile_receipt)
    if type(profile) is not DeviceProfileProjection:
        raise DeviceR05Error("profile authority projection type differs")
    profile_sha256 = profile.profile_sha256
    cell_ids = profile.cell_ids
    semantic_sha256s = profile.semantic_sha256s
    targets_source = profile.targets_xy_m
    profile_binding_sha256 = profile.profile_binding_sha256
    if (
        type(profile_sha256) is not str
        or len(profile_sha256) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha256)
        or type(profile_binding_sha256) is not str
        or len(profile_binding_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in profile_binding_sha256
        )
        or type(cell_ids) is not tuple
        or type(semantic_sha256s) is not tuple
        or type(targets_source) is not torch.Tensor
        or not (2 <= len(cell_ids) <= _MAX_SUPPORT)
        or len(cell_ids) != len(semantic_sha256s)
        or tuple(targets_source.shape) != (len(cell_ids), 2)
        or targets_source.dtype != torch.float32
        or not targets_source.is_contiguous()
        or any(type(cell_id) is not str or not cell_id for cell_id in cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
        or any(
            type(semantic) is not str
            or len(semantic) != 64
            or any(character not in "0123456789abcdef" for character in semantic)
            for semantic in semantic_sha256s
        )
        or len(set(semantic_sha256s)) != len(semantic_sha256s)
    ):
        raise DeviceR05Error("device profile structural binding differs")
    # This is a cold construction/restore check, never a reveal-path transfer.
    # Recompute the binding from the copied float32 bytes instead of trusting a
    # digest supplied by the same authority.  ``untyped_storage`` avoids
    # scalar observation APIs; the only device transfer here is the explicit
    # cold copy made before any runtime owner exists.
    targets_owned = targets_source.detach().to(
        device=torch.device("cpu"), dtype=torch.float32
    ).contiguous().clone()
    if not bool(torch.all(torch.isfinite(targets_owned))):
        raise DeviceR05Error("device profile structural binding differs")
    raw_native = bytes(targets_owned.view(torch.uint8).untyped_storage())[
        : targets_owned.numel() * 4
    ]
    native_one = struct.pack("=f", 1.0)
    if native_one == struct.pack(">f", 1.0):
        target_bytes = raw_native
    else:
        target_bytes = b"".join(
            raw_native[offset : offset + 4][::-1]
            for offset in range(0, len(raw_native), 4)
        )
    binding = hashlib.sha256()
    binding.update(profile_sha256.encode("ascii"))
    for cell_id, semantic in zip(cell_ids, semantic_sha256s):
        cell_id_bytes = cell_id.encode("utf-8")
        binding.update(len(cell_id_bytes).to_bytes(8, "big"))
        binding.update(cell_id_bytes)
        binding.update(bytes.fromhex(semantic))
    binding.update(target_bytes)
    if binding.hexdigest() != profile_binding_sha256:
        raise DeviceR05Error("device profile content binding differs")
    return DeviceProfileProjection(
        profile_sha256=profile_sha256,
        profile_binding_sha256=profile_binding_sha256,
        cell_ids=tuple(cell_ids),
        semantic_sha256s=tuple(semantic_sha256s),
        targets_xy_m=targets_owned.to(device=targets_source.device).clone(),
    )


class DeviceR05DrainBatch(_OpaqueDeviceR05Capability):
    """Opaque drain capability; packed bytes stay owner-private."""

    __slots__ = ()


@dataclass(frozen=True)
class DeviceR05DrainView:
    """Clone-only view for tests/diagnostics; never accepted for ACK."""

    drain_identity: object
    schema_version: int
    num_envs: int
    support_size: int
    row_count: int
    start_sequence: int
    end_sequence: int
    packed: torch.Tensor
    packed_schema: Tuple[Tuple[str, int, int, Tuple[int, ...]], ...]


@dataclass(frozen=True)
class DeviceR05Checkpoint:
    """Fully-drained, non-poisoned device checkpoint."""

    profile_sha256: str
    profile_binding_sha256: str
    seed: int
    num_envs: int
    journal_capacity: int
    max_reveal_epochs_per_drain: int
    internal_question_redraw_rounds: int
    epoch: int
    journal_sequence: int
    last_transfer_sequence: int
    last_candidate_bank_sequence: int
    next_candidate_identity: int
    last_global_drain_sequence: int
    last_global_update_index: int
    last_global_completed_environment_steps: int
    last_global_ack_mutation_version: int
    last_global_terminal_resolution_total: int
    last_global_policy_opportunity_total: int
    mutation_version_host: int
    next_outcome_identity: torch.Tensor
    next_ball_identity: torch.Tensor
    rng_lo: torch.Tensor
    rng_hi: torch.Tensor
    draw_count: torch.Tensor
    target_generation: torch.Tensor
    previous_cell_index: torch.Tensor
    reset_generation: torch.Tensor
    scheduled_ordinal: torch.Tensor
    outcome_shot_index: torch.Tensor
    sequence_kind: torch.Tensor
    task_identity: torch.Tensor
    outcome_identity: torch.Tensor
    ball_identity: torch.Tensor
    policy_opportunity: torch.Tensor
    mutation_version: torch.Tensor


class DeviceR05PreparedToken(_OpaqueDeviceR05Capability):
    """Opaque owner-issued staged C03 after-image capability."""

    __slots__ = ()


class DeviceR05RowTransaction(_OpaqueDeviceR05Capability):
    """Opaque full-N opportunity transaction consumed only by ActionEpoch."""

    __slots__ = ()


@dataclass(frozen=True)
class DeviceR05AcceptedRowsView:
    """Full-N candidate after-image visible only inside an ACCEPT writer."""

    transaction: DeviceR05RowTransaction
    publication_ordinal: torch.Tensor
    target_xy_m: torch.Tensor
    identity: object
    clocks: object
    task: object
    rng_counter: torch.Tensor
    playback_admissible: torch.Tensor


class DeviceR05PreviewToken(_OpaqueDeviceR05Capability):
    """Opaque exclusive reveal preview lease."""

    __slots__ = ()


class DeviceR05PreTransferBoundaryToken(_OpaqueDeviceR05Capability):
    """Single-use view capability consumed inside the sole boundary packet."""

    __slots__ = ()


class DeviceR05TerminalClaim(_OpaqueDeviceR05Capability):
    """Terminal decision issued by the global reveal authority."""

    __slots__ = ()


class DeviceR05ArmedTerminal(_OpaqueDeviceR05Capability):
    """Last R05 pre-publication gate."""

    __slots__ = ()


class DeviceR05TerminalReceipt(_OpaqueDeviceR05Capability):
    """Opaque R05-last capability; all authority facts stay owner-private."""

    __slots__ = ()


class DeviceR05ConstructionRejection(_OpaqueDeviceR05Capability):
    """Zero-opportunity, non-advancing rejection from the reveal boundary."""

    __slots__ = ()


class DeviceR05PreparedTrueReset(_OpaqueDeviceR05Capability):
    """Opaque owner-retained selected true-reset after-image."""

    __slots__ = ()


class DeviceR05TrueResetReceipt(_OpaqueDeviceR05Capability):
    """Exact top-last Device-R05 reset receipt; portable audit stays in drain."""

    __slots__ = ()


@dataclass
class _PreparedRecord:
    capability: DeviceR05PreparedToken
    epoch: int
    journal_slot: int
    projection: DeviceCadenceProjection
    question_projection: DeviceQuestionProjection
    question_producer_fault: torch.Tensor
    selected_candidate_identity: torch.Tensor
    selected_construction_reason: torch.Tensor
    bank_candidate_identity: torch.Tensor
    bank_construction_reason: torch.Tensor
    round_candidate_identity: torch.Tensor
    round_construction_reason: torch.Tensor
    round_producer_fault: torch.Tensor
    chosen_round: torch.Tensor
    rounds_attempted: torch.Tensor
    selected_motion_task_f32: torch.Tensor
    selected_racket_task_f32: torch.Tensor
    selected_physical_state_f32: torch.Tensor
    selected_full_key_sha256: Optional[torch.Tensor]
    selected_task_sha256: Optional[torch.Tensor]
    reserved_outcome_identity: torch.Tensor
    reserved_ball_identity: torch.Tensor
    outcome_identity_highwater_before: torch.Tensor
    ball_identity_highwater_before: torch.Tensor
    identity_advance_count: torch.Tensor
    identity_counter_room: torch.Tensor
    internal_question: bool
    candidate_identity_highwater_before: int
    candidate_identity_highwater_after: int
    selected_index: torch.Tensor
    selected_mask: torch.Tensor
    feasible: torch.Tensor
    eligible: torch.Tensor
    admissible: torch.Tensor
    owner_fault_free: torch.Tensor
    counter_overflow_fault: torch.Tensor
    reason: torch.Tensor
    rng_before_lo: torch.Tensor
    rng_before_hi: torch.Tensor
    rng_after_lo: torch.Tensor
    rng_after_hi: torch.Tensor
    draw_before: torch.Tensor
    draw_after: torch.Tensor
    rng_advance_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    previous_before: torch.Tensor
    reset_before: torch.Tensor
    ordinal_before: torch.Tensor
    outcome_before: torch.Tensor
    selected_cell: torch.Tensor
    selected_target_xy_m: torch.Tensor
    stage: str


@dataclass
class _RowTransactionRecord:
    capability: DeviceR05RowTransaction
    candidate: object
    prepared: _PreparedRecord
    preview: "_PreviewRecord"
    due_mask: torch.Tensor
    construct_mask: torch.Tensor
    accept_mask: torch.Tensor
    reject_mask: torch.Tensor
    defer_mask: torch.Tensor
    censor_mask: torch.Tensor
    candidate_consumed: bool
    accepted_consumers: set[str]
    stage: str


@dataclass
class _PreviewRecord:
    capability: DeviceR05PreviewToken
    prepared: _PreparedRecord
    preview_identity: object
    stage: str
    pretransfer_token: Optional[DeviceR05PreTransferBoundaryToken] = None
    pretransfer_consumed: bool = False


@dataclass
class _ClaimRecord:
    capability: DeviceR05TerminalClaim
    preview: _PreviewRecord
    decision: int
    owner_fault_present: bool
    primary_fault: torch.Tensor
    transfer_sequence: int
    claim_identity: object
    stage: str


@dataclass
class _ArmedRecord:
    capability: DeviceR05ArmedTerminal
    claim: _ClaimRecord
    armed_identity: object
    stage: str


@dataclass
class _PreparedTrueResetRecord:
    capability: DeviceR05PreparedTrueReset
    prepared_identity: object
    reset_event_receipt: object
    reset_event_identity: object
    selected_index: torch.Tensor
    selected_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    generation_overflow_fault: torch.Tensor
    mutation_before: torch.Tensor
    mutation_after: torch.Tensor
    scheduled_ordinal_before: torch.Tensor
    outcome_shot_index_before: torch.Tensor
    sequence_kind_before: torch.Tensor
    task_identity_before: torch.Tensor
    outcome_identity_before: torch.Tensor
    ball_identity_before: torch.Tensor
    policy_opportunity_before: torch.Tensor
    device_fault: torch.Tensor
    journal_slot: int
    epoch: int
    stage: str
    preflight_capability: object = None


@dataclass
class _TerminalRecord:
    receipt: DeviceR05TerminalReceipt
    terminal_identity: object
    epoch: int
    journal_slot: int
    decision: int
    selected_count: int
    journal_sequence: int
    prepared_reveal: DeviceR05PreviewToken
    completed_children: set[str]


@dataclass(frozen=True)
class _TrueResetRecord:
    receipt: DeviceR05TrueResetReceipt
    prepared: DeviceR05PreparedTrueReset
    journal_sequence: int
    completed_children: set[str]


@dataclass
class _DrainRecord:
    capability: object
    authority: object
    update_index: int
    completed_environment_steps: int
    mutation_version_host: int
    journal_head: int
    journal_tail: int
    row_count: int
    stage: str = "prepared"


@dataclass
class _PublicationState:
    """One swappable owner after-image.

    Builders use copy-on-write: live tensors are cloned, retained receipt
    registries and immutable journal rows are shallow-copied.  The owner makes
    a terminal/reset result visible with one Python pointer replacement only.
    """

    live: dict[str, torch.Tensor]
    registries: dict[str, dict[object, object]]
    counters: dict[str, object]
    journal_rows: dict[int, dict[str, torch.Tensor]]

    def fork(self) -> "_PublicationState":
        return _PublicationState(
            live={name: value.clone() for name, value in self.live.items()},
            registries={
                name: dict(value) for name, value in self.registries.items()
            },
            counters=dict(self.counters),
            journal_rows=dict(self.journal_rows),
        )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _initial_stream_state(
    *, profile_sha256: str, seed: int, env_id: int
) -> int:
    digest = _canonical_sha256(
        {
            "schema_version": 2,
            "kind": "action_ball_continuous_target_sampler_stream_v2",
            "profile_sha256": profile_sha256,
            "seed": seed,
            "env_id": env_id,
        }
    )
    return int(digest[:16], 16)


def _require_tensor(
    value: object,
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise DeviceR05Error(f"{label} must be an exact Tensor")
    if value.device != device or value.dtype != dtype or tuple(value.shape) != shape:
        raise DeviceR05Error(f"{label} device/dtype/shape differs")
    return value


def materialize_pre_optimizer_ppo_boundary_leaf_schema(
    *,
    leaf_schema_type: type,
    field_spec_type: type,
    journal_capacity: int,
    num_envs: int,
    support_size: int,
) -> object:
    """Build Device-R05's real fixed-width global-drain schema.

    The bounded journal is transferred as exact full-capacity fields.  The
    scalar ``journal_count`` and monotonic start/end sequences distinguish an
    empty logical batch from the zero sentinel in unused capacity.  No
    portable R05 object or local drain receipt is created here.
    """

    if (
        type(journal_capacity) is not int
        or journal_capacity < 1
        or type(num_envs) is not int
        or num_envs < 1
        or type(support_size) is not int
        or support_size < 2
    ):
        raise DeviceR05Error("global drain journal dimensions differ")
    scalar_specs = tuple(
        field_spec_type(name=name)
        for name in GLOBAL_DRAIN_SCALAR_FIELD_NAMES
    )
    fixed_specs = tuple(
        field_spec_type(
            name=f"journal_{name}",
            cardinality="fixed",
            fixed_width=2
            * journal_capacity
            * _journal_field_width_per_row(
                name,
                num_envs=num_envs,
                support_size=support_size,
            ),
        )
        for name in _GLOBAL_JOURNAL_FIELD_NAMES
    )
    return leaf_schema_type(
        owner_kind=GLOBAL_DRAIN_OWNER_KIND,
        fields=scalar_specs + fixed_specs,
    )


def _journal_field_width_per_row(name: str, *, num_envs: Optional[int] = None,
                                 support_size: Optional[int] = None) -> int:
    if name == "meta":
        return DeviceR05Owner._META_WIDTH
    if name in ("feasible", "construction_reason", "candidate_identity"):
        if num_envs is None or support_size is None:
            raise DeviceR05Error(
                "num_envs/support_size are required for candidate journal width"
            )
        return num_envs * support_size
    if name == "child_completion":
        return len(CHILD_OWNER_ORDER)
    if num_envs is None:
        raise DeviceR05Error("num_envs is required for per-environment journal width")
    return num_envs


def _u64_add_const(
    lo: torch.Tensor, hi: torch.Tensor, constant: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lo_sum = lo + (constant & _U32_MASK)
    carry = torch.bitwise_right_shift(lo_sum, 32)
    result_lo = torch.bitwise_and(lo_sum, _U32_MASK)
    result_hi = torch.bitwise_and(
        hi + ((constant >> 32) & _U32_MASK) + carry, _U32_MASK
    )
    return result_lo, result_hi


def _u64_shr(
    lo: torch.Tensor, hi: torch.Tensor, amount: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0 < amount < 32):
        raise DeviceR05Error("internal u64 shift must be in (0, 32)")
    low_hi_bits = torch.bitwise_and(hi, (1 << amount) - 1)
    result_lo = torch.bitwise_or(
        torch.bitwise_right_shift(lo, amount),
        torch.bitwise_left_shift(low_hi_bits, 32 - amount),
    )
    result_hi = torch.bitwise_right_shift(hi, amount)
    return (
        torch.bitwise_and(result_lo, _U32_MASK),
        torch.bitwise_and(result_hi, _U32_MASK),
    )


def _u64_mul_const_low(
    lo: torch.Tensor, hi: torch.Tensor, constant: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact low-u64 product using nonnegative 16-bit limbs."""

    a = (
        torch.bitwise_and(lo, _U16_MASK),
        torch.bitwise_right_shift(lo, 16),
        torch.bitwise_and(hi, _U16_MASK),
        torch.bitwise_right_shift(hi, 16),
    )
    b = tuple((constant >> (16 * index)) & _U16_MASK for index in range(4))
    carry = torch.zeros_like(lo)
    out = []
    for k in range(4):
        total = carry
        for i in range(k + 1):
            total = total + a[i] * b[k - i]
        out.append(torch.bitwise_and(total, _U16_MASK))
        carry = torch.bitwise_right_shift(total, 16)
    result_lo = torch.bitwise_or(out[0], torch.bitwise_left_shift(out[1], 16))
    result_hi = torch.bitwise_or(out[2], torch.bitwise_left_shift(out[3], 16))
    return result_lo, result_hi


def _splitmix64_lanes(
    state_lo: torch.Tensor, state_hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_lo, next_hi = _u64_add_const(state_lo, state_hi, _SPLITMIX_GAMMA)
    shr_lo, shr_hi = _u64_shr(next_lo, next_hi, 30)
    value_lo = torch.bitwise_xor(next_lo, shr_lo)
    value_hi = torch.bitwise_xor(next_hi, shr_hi)
    value_lo, value_hi = _u64_mul_const_low(value_lo, value_hi, _SPLITMIX_M1)
    shr_lo, shr_hi = _u64_shr(value_lo, value_hi, 27)
    value_lo = torch.bitwise_xor(value_lo, shr_lo)
    value_hi = torch.bitwise_xor(value_hi, shr_hi)
    value_lo, value_hi = _u64_mul_const_low(value_lo, value_hi, _SPLITMIX_M2)
    shr_lo, shr_hi = _u64_shr(value_lo, value_hi, 31)
    draw_lo = torch.bitwise_and(torch.bitwise_xor(value_lo, shr_lo), _U32_MASK)
    draw_hi = torch.bitwise_and(torch.bitwise_xor(value_hi, shr_hi), _U32_MASK)
    return next_lo, next_hi, draw_lo, draw_hi


def _draw_internal_question_uniform01(
    state_lo: torch.Tensor,
    state_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reserve three fixed rounds of replayable incoming-ball draws.

    Each round maps six values to three linear-velocity and three spin
    coordinates.  All rounds are generated branchlessly before construction;
    D05 then reserves and burns one final-selection lane after these eighteen
    question draws.  Every typed ACCEPT, REJECT, DEFER, or CENSOR settlement
    therefore consumes exactly nineteen lanes, independent of whether a clean
    eligible round exists.  This keeps replay chronology independent of solver
    admission and avoids a second RNG owner.
    """

    current_lo = state_lo
    current_hi = state_hi
    values = []
    for _ in range(
        INTERNAL_QUESTION_REDRAW_ROUNDS * INTERNAL_QUESTION_DRAW_WIDTH
    ):
        current_lo, current_hi, _draw_lo, draw_hi = _splitmix64_lanes(
            current_lo, current_hi
        )
        mantissa = torch.bitwise_right_shift(draw_hi, 8).to(torch.float32)
        values.append((mantissa + 0.5) * (1.0 / float(1 << 24)))
    return (
        current_lo,
        current_hi,
        torch.stack(values, dim=1)
        .reshape(
            state_lo.shape[0],
            INTERNAL_QUESTION_REDRAW_ROUNDS,
            INTERNAL_QUESTION_DRAW_WIDTH,
        )
        .contiguous(),
    )


def _bitwise_or_rounds(value: torch.Tensor) -> torch.Tensor:
    """Fixed-width device reduction without host observation."""

    result = torch.zeros(
        value.shape[0], dtype=torch.int64, device=value.device
    )
    for round_index in range(value.shape[1]):
        result = torch.bitwise_or(result, value[:, round_index])
    return result.contiguous()


def _u64_mul_u31_high(
    lo: torch.Tensor, hi: torch.Tensor, multiplier: torch.Tensor
) -> torch.Tensor:
    """Exact unsigned ``(u64 * u31) >> 64`` using 16-bit limbs."""

    a = (
        torch.bitwise_and(lo, _U16_MASK),
        torch.bitwise_right_shift(lo, 16),
        torch.bitwise_and(hi, _U16_MASK),
        torch.bitwise_right_shift(hi, 16),
    )
    b = (
        torch.bitwise_and(multiplier, _U16_MASK),
        torch.bitwise_right_shift(multiplier, 16),
    )
    carry = torch.zeros_like(lo)
    out = []
    for k in range(6):
        total = carry
        for i in range(4):
            j = k - i
            if 0 <= j < 2:
                total = total + a[i] * b[j]
        out.append(torch.bitwise_and(total, _U16_MASK))
        carry = torch.bitwise_right_shift(total, 16)
    return torch.bitwise_or(out[4], torch.bitwise_left_shift(out[5], 16))


class DeviceR05Owner:
    """N-vector R05 hot owner with exact C03 sampling and bounded evidence."""

    _META_WIDTH = 9
    _META_SEQUENCE = 0
    _META_EPOCH = 1
    _META_OPERATION = 2
    _META_MUTATION_BEFORE = 3
    _META_MUTATION_AFTER = 4
    _META_SELECTED_COUNT = 5
    _META_ADMISSIBLE_COUNT = 6
    _META_TRANSFER_SEQUENCE = 7
    _META_DECISION = 8
    _GLOBAL_DRAIN_SENTINEL = 0
    _STATE_VIEW_NAMES = frozenset(
        {
            "rng_lo",
            "rng_hi",
            "draw_count",
            "target_generation",
            "previous_cell_index",
            "reset_generation",
            "scheduled_ordinal",
            "outcome_shot_index",
            "sequence_kind",
            "task_identity",
            "outcome_identity",
            "ball_identity",
            "policy_opportunity",
            "mutation_version",
            "poisoned",
            "poison_reason",
        }
    )

    @property
    def profile(self) -> DeviceProfileProjection:
        self._enter_public_operation()
        self._close_construction_window()
        return DeviceProfileProjection(
            profile_sha256=self._profile.profile_sha256,
            profile_binding_sha256=self._profile.profile_binding_sha256,
            cell_ids=self._profile.cell_ids,
            semantic_sha256s=self._profile.semantic_sha256s,
            targets_xy_m=self._profile.targets_xy_m.clone(),
        )

    @property
    def seed(self) -> int:
        self._enter_public_operation()
        self._close_construction_window()
        return self._seed

    @property
    def num_envs(self) -> int:
        self._enter_public_operation()
        self._close_construction_window()
        return self._num_envs

    @property
    def journal_capacity(self) -> int:
        self._enter_public_operation()
        self._close_construction_window()
        return self._journal_capacity

    @property
    def max_reveal_epochs_per_drain(self) -> int:
        self._enter_public_operation()
        self._close_construction_window()
        return self._max_reveal_epochs_per_drain

    @property
    def device(self) -> torch.device:
        self._enter_public_operation()
        return self._device

    def __getattribute__(self, name: str) -> object:
        if name in object.__getattribute__(self, "_STATE_VIEW_NAMES"):
            state = object.__getattribute__(self, "__dict__")
            if state.get("_authority_callback_active", False):
                state["_authority_reentry_detected"] = True
                raise DeviceR05ConflictError(
                    "authority callback re-entered a Device-R05 state view"
                )
            if state.get("_construction_window_open", False):
                state["_construction_window_open"] = False
                if (
                    state.get("_true_reset_authority") is None
                    and state.get("_diagnostic_epoch_owner") is None
                ):
                    raise DeviceR05ConflictError(
                        "business state read before true-reset authority bind"
                    )
            private_name = f"_{name}"
            publication = state.get("_publication")
            if publication is not None:
                return publication.live[private_name].clone()
            return object.__getattribute__(self, private_name).clone()
        if name in _LIVE_PUBLICATION_TENSOR_NAMES:
            publication = object.__getattribute__(self, "__dict__").get(
                "_publication"
            )
            if publication is not None:
                return publication.live[name]
        if name in _PUBLICATION_REGISTRY_NAMES:
            publication = object.__getattribute__(self, "__dict__").get(
                "_publication"
            )
            if publication is not None:
                return publication.registries[name]
        if name in _PUBLICATION_SCALAR_NAMES:
            publication = object.__getattribute__(self, "__dict__").get(
                "_publication"
            )
            if publication is not None:
                return publication.counters[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        publication = self.__dict__.get("_publication")
        if publication is not None:
            if name in _LIVE_PUBLICATION_TENSOR_NAMES:
                publication.live[name] = value  # type: ignore[assignment]
                return
            if name in _PUBLICATION_REGISTRY_NAMES:
                publication.registries[name] = value  # type: ignore[assignment]
                return
            if name in _PUBLICATION_SCALAR_NAMES:
                publication.counters[name] = value
                return
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise TypeError("Device-R05 owners cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("Device-R05 owners cannot be copied")

    def __reduce__(self):
        raise TypeError("Device-R05 owners cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("Device-R05 owners cannot be serialized")

    def __init__(
        self,
        profile_authority: DeviceProfileAuthority,
        profile_receipt: object,
        *,
        seed: int,
        num_envs: int,
        journal_capacity: int,
        max_reveal_epochs_per_drain: int,
        genesis_authority: DeviceGenesisAuthority,
        genesis_receipt: object,
        cadence_authority: DeviceCadenceAuthority,
        question_authority: DeviceQuestionAuthority,
        reveal_boundary_authority: Optional[DeviceRevealBoundaryAuthority] = None,
        child_completion_authorities: Tuple[
            DeviceChildCompletionAuthority, ...
        ] = (),
        drain_authority: Optional[DeviceDrainAuthority] = None,
        true_reset_authority: Optional[DeviceTrueResetAuthority] = None,
        diagnostic_epoch_owner: Optional[object] = None,
    ) -> None:
        # Snapshot caller-owned bytes before invoking any construction authority.
        # Frozen dataclasses do not make their Tensor fields immutable, and an
        # authority callback must never be able to rewrite C03 profile truth
        # between validation and stream initialization.
        owned_profile = _snapshot_device_profile(profile_authority, profile_receipt)
        if type(seed) is not int or not (0 <= seed < (1 << 64)):
            raise DeviceR05Error("seed must be an exact uint64")
        if type(num_envs) is not int or num_envs < 1:
            raise DeviceR05Error("num_envs must be a positive exact int")
        if (
            type(journal_capacity) is not int
            or type(max_reveal_epochs_per_drain) is not int
            or max_reveal_epochs_per_drain < 1
            or journal_capacity < max_reveal_epochs_per_drain
        ):
            raise DeviceR05Error(
                "journal capacity must cover the PPO reveal horizon"
            )
        if not callable(
            getattr(genesis_authority, "require_owned_r05_genesis", None)
        ):
            raise DeviceR05Error("reset-generation genesis authority is not callable")
        genesis = genesis_authority.require_owned_r05_genesis(
            genesis_receipt,
            device=owned_profile.device,
            num_envs=num_envs,
        )
        if type(genesis) is not DeviceGenesisProjection:
            raise DeviceR05Error("reset-generation genesis projection type differs")
        genesis_world_reset_identity = genesis.world_reset_identity
        genesis_reset_generations = genesis.reset_generations
        if genesis_world_reset_identity is None:
            raise DeviceR05Error("genesis world-reset identity is absent")
        initial_reset_generations = _require_tensor(
            genesis_reset_generations,
            label="genesis.reset_generations",
            device=owned_profile.device,
            dtype=torch.int64,
            shape=(num_envs,),
        ).clone()
        # Construction is a cold boundary.  Reject an exhausted or nonpositive
        # world ledger before any child can retain an incompatible genesis.
        if not bool(
            torch.all(
                torch.logical_and(
                    initial_reset_generations >= 1,
                    initial_reset_generations < _I64_MAX,
                )
            )
        ):
            raise DeviceR05Error(
                "reset-generation genesis has no positive int64 continuation"
            )
        if not callable(
            getattr(cadence_authority, "project_current_action_epoch_rows", None)
        ):
            raise DeviceR05Error("Motion cadence authority is not callable")
        internal_question_compose = getattr(
            question_authority,
            "compose_r05_candidate_bank_inside_prepare",
            None,
        )
        if not callable(internal_question_compose) and not callable(
            getattr(question_authority, "project_r05_candidate_bank", None)
        ):
            raise DeviceR05Error("question/solver authority is not callable")
        children = tuple(child_completion_authorities)
        if diagnostic_epoch_owner is None:
            if not callable(
                getattr(
                    reveal_boundary_authority,
                    "project_owned_r05_reveal_boundary",
                    None,
                )
            ) or not callable(
                getattr(
                    reveal_boundary_authority,
                    "require_owned_r05_terminal_arm",
                    None,
                )
            ) or not callable(
                getattr(
                    reveal_boundary_authority,
                    "require_owned_r05_terminal_commit",
                    None,
                )
            ):
                raise DeviceR05Error("reveal boundary authority is not callable")
            if len(children) != len(CHILD_OWNER_ORDER) or any(
                not callable(
                    getattr(authority, "require_owned_r05_child_completion", None)
                )
                for authority in children
            ):
                raise DeviceR05Error("child completion authorities differ")
        else:
            epoch = _require_action_epoch_module()
            if (
                reveal_boundary_authority is not None
                or children
                or true_reset_authority is not None
            ):
                raise DeviceR05Error(
                    "diagnostic epoch mode excludes legacy reveal/reset authorities"
                )
            _require_canonical_action_epoch_idle(
                diagnostic_epoch_owner,
                epoch_module=epoch,
                device=owned_profile.device,
                num_envs=num_envs,
            )
        if true_reset_authority is not None:
            self._validate_true_reset_authority(true_reset_authority)

        self._profile = owned_profile
        self._profile_binding_sha256 = owned_profile.profile_binding_sha256
        self._profile_authority = profile_authority
        self._profile_receipt = profile_receipt
        self._seed = seed
        self._num_envs = num_envs
        self._journal_capacity = journal_capacity
        self._max_reveal_epochs_per_drain = max_reveal_epochs_per_drain
        self._device = owned_profile.device
        self._row_axis = torch.arange(num_envs, dtype=torch.int64, device=self._device)
        self._genesis_authority = genesis_authority
        self._genesis_world_reset_identity = genesis_world_reset_identity
        self._genesis_reset_generations = initial_reset_generations.clone()
        self._genesis_child_projections: dict[
            str, _GenesisProjectionRecord
        ] = {}
        self._env_reset_binding: Optional[DeviceR05EnvResetBinding] = None
        self._cadence_authority = cadence_authority
        self._question_authority = question_authority
        self._internal_question_compose = (
            internal_question_compose
            if callable(internal_question_compose)
            else None
        )
        self._question_prepare_lock = RLock()
        self._question_composition_in_progress = False
        self._reveal_boundary_authority = reveal_boundary_authority
        self._child_completion_authorities = children
        self._diagnostic_epoch_owner = diagnostic_epoch_owner
        self._action_epoch_is_single_business_log = (
            diagnostic_epoch_owner is not None
        )
        self._diagnostic_motion_owner: Optional[object] = None
        self._diagnostic_racket_owner: Optional[object] = None
        self._diagnostic_physical_owner: Optional[object] = None
        self._active_diagnostic_epoch_leaf_writer: Optional[str] = None
        self._active_row_transaction: Optional[DeviceR05RowTransaction] = None
        self._row_transaction_records: dict[
            DeviceR05RowTransaction, _RowTransactionRecord
        ] = {}
        del drain_authority
        self._true_reset_authority = true_reset_authority
        self._construction_window_open = True
        self._identity = object()
        self._live_reset_ledger_identity = object.__new__(DeviceR05LiveResetLedger)
        self._epoch = 0
        self._active: Optional[object] = None
        self._prepared_records: dict[
            DeviceR05PreparedToken, _PreparedRecord
        ] = {}
        self._preview_records: dict[
            DeviceR05PreviewToken, _PreviewRecord
        ] = {}
        self._pretransfer_records: dict[
            DeviceR05PreTransferBoundaryToken, _PreviewRecord
        ] = {}
        self._claim_records: dict[
            DeviceR05TerminalClaim, _ClaimRecord
        ] = {}
        self._armed_records: dict[
            DeviceR05ArmedTerminal, _ArmedRecord
        ] = {}
        self._active_drain: Optional[_DrainRecord] = None
        self._terminal_receipts: dict[
            DeviceR05TerminalReceipt, _TerminalRecord
        ] = {}
        self._prepared_true_resets: dict[
            DeviceR05PreparedTrueReset, _PreparedTrueResetRecord
        ] = {}
        self._true_reset_receipts: dict[
            DeviceR05TrueResetReceipt, _TrueResetRecord
        ] = {}
        self._poisoned_python = False
        self._authority_callback_active = False
        self._authority_reentry_detected = False
        self._journal_head = 0
        self._journal_tail = 0
        self._last_transfer_sequence = 0
        self._last_candidate_bank_sequence = 0
        self._next_candidate_identity = 1
        self._last_global_drain_sequence = 0
        self._last_global_update_index = -1
        self._last_global_completed_environment_steps = -1
        self._last_global_ack_mutation_version = 0
        self._last_global_terminal_resolution_total = 0
        self._last_global_policy_opportunity_total = 0
        self._checkpoint_requires_global_drain_ack = (
            not self._action_epoch_is_single_business_log
        )
        self._global_drain_poison_reason: Optional[str] = None

        initial = tuple(
            _initial_stream_state(
                profile_sha256=owned_profile.profile_sha256,
                seed=seed,
                env_id=env_id,
            )
            for env_id in range(num_envs)
        )
        self._rng_lo = torch.tensor(
            tuple(value & _U32_MASK for value in initial),
            dtype=torch.int64,
            device=self._device,
        )
        self._rng_hi = torch.tensor(
            tuple((value >> 32) & _U32_MASK for value in initial),
            dtype=torch.int64,
            device=self._device,
        )
        zeros = torch.zeros(num_envs, dtype=torch.int64, device=self._device)
        self._draw_count = zeros.clone()
        self._target_generation = zeros.clone()
        self._previous_cell_index = torch.full_like(zeros, -1)
        self._reset_generation = initial_reset_generations
        self._scheduled_ordinal = torch.full_like(zeros, -1)
        self._outcome_shot_index = zeros.clone()
        self._sequence_kind = zeros.clone()
        self._task_identity = torch.full_like(zeros, -1)
        self._outcome_identity = torch.full_like(zeros, -1)
        self._ball_identity = torch.full_like(zeros, -1)
        self._next_outcome_identity = torch.ones(
            (), dtype=torch.int64, device=self._device
        )
        self._next_ball_identity = torch.ones(
            (), dtype=torch.int64, device=self._device
        )
        self._policy_opportunity = torch.zeros(
            num_envs, dtype=torch.bool, device=self._device
        )
        self._mutation_version = torch.zeros(
            (), dtype=torch.int64, device=self._device
        )
        self._mutation_version_host = 0
        self._poisoned = torch.zeros((), dtype=torch.bool, device=self._device)
        self._poison_reason = torch.zeros(
            (), dtype=torch.int64, device=self._device
        )
        self._allocate_journal()
        # Terminal and true-reset commits never mutate these objects in place.
        # They fork this complete state, build live+journal+receipt after-images
        # off-owner, then publish them with one pointer replacement.
        self._publication = _PublicationState(
            live={
                name: object.__getattribute__(self, name)
                for name in _LIVE_PUBLICATION_TENSOR_NAMES
            },
            registries={
                name: object.__getattribute__(self, name)
                for name in _PUBLICATION_REGISTRY_NAMES
            },
            counters={
                name: object.__getattribute__(self, name)
                for name in _PUBLICATION_SCALAR_NAMES
            },
            journal_rows={},
        )

    @property
    def integration_status(self) -> str:
        self._enter_public_operation()
        return INTEGRATION_STATUS

    @property
    def runtime_wiring_connected(self) -> bool:
        self._enter_public_operation()
        self._close_construction_window()
        return self._diagnostic_epoch_owner is not None

    @staticmethod
    def _validate_true_reset_authority(
        authority: DeviceTrueResetAuthority,
    ) -> None:
        if not all(
            callable(getattr(authority, name, None))
            for name in (
                "project_r05_true_reset",
                "require_owned_r05_true_reset_preflight",
                "require_owned_r05_true_reset_commit",
                "require_owned_r05_true_reset_abort",
                "require_owned_r05_true_reset_child_completion",
            )
        ):
            raise DeviceR05Error("true-reset authority is not callable")

    def bind_true_reset_authority(
        self, authority: DeviceTrueResetAuthority
    ) -> None:
        """Close the sole construction-cycle seam exactly once.

        No business operation, checkpoint, drain, or public state access may
        precede this bind.  The method exists only to break the top-owner/R05
        construction cycle and is not a runtime late-bind capability.
        """

        self._enter_public_operation()
        if (
            not self._construction_window_open
            or self._true_reset_authority is not None
            or self._active is not None
            or self._active_drain is not None
            or self._journal_head != 0
            or self._journal_tail != 0
            or self._epoch != 0
        ):
            raise DeviceR05ConflictError(
                "true-reset authority construction window is closed"
            )
        coordinator = getattr(self, "_lean_carry_coordinator", None)
        if coordinator is not None:
            frozen, current = self._lean_carry_frozen_bindings, self._lean_carry_bindings()
            if len(frozen) != 15 or any(current[index] is not frozen[index] for index in (*range(13), 14)): self._construction_window_open = False; raise DeviceR05ConflictError("construction binding changed before final reset bind")
        self._validate_true_reset_authority(authority)
        self._true_reset_authority = authority
        self._construction_window_open = False
        if coordinator is not None: self._lean_carry_frozen_bindings = (*frozen[:13], authority, *frozen[14:])

    def project_owned_genesis_for_child(
        self, *, owner_kind: str
    ) -> DeviceR05GenesisProjection:
        """Mint/return one opaque exact genesis capability per child."""

        self._enter_public_operation()
        if not self._construction_window_open:
            raise DeviceR05ConflictError("genesis construction window is closed")
        if owner_kind not in GENESIS_CONSUMER_ORDER:
            raise DeviceR05Error("genesis child kind differs")
        record = self._genesis_child_projections.get(owner_kind)
        if record is not None:
            return record.projection
        projection = object.__new__(DeviceR05GenesisProjection)
        view = DeviceR05GenesisView(
            device_r05_owner=self,
            owner_kind=owner_kind,
            world_reset_identity=self._genesis_world_reset_identity,
            reset_generation=self._genesis_reset_generations.clone(),
        )
        self._genesis_child_projections[owner_kind] = _GenesisProjectionRecord(
            projection=projection,
            owner_kind=owner_kind,
            view=view,
        )
        return projection

    def require_owned_genesis_projection(
        self,
        projection: DeviceR05GenesisProjection,
        *,
        owner_kind: str,
    ) -> DeviceR05GenesisView:
        """Validate exact opaque identity and return fresh clone-only data."""

        self._enter_public_operation()
        if not self._construction_window_open:
            raise DeviceR05ConflictError("genesis construction window is closed")
        record = self._genesis_child_projections.get(owner_kind)
        if (
            owner_kind not in GENESIS_CONSUMER_ORDER
            or record is None
            or record.projection is not projection
        ):
            raise DeviceR05ConflictError("genesis projection is stale or foreign")
        view = record.view
        return DeviceR05GenesisView(
            device_r05_owner=self,
            owner_kind=owner_kind,
            world_reset_identity=view.world_reset_identity,
            reset_generation=view.reset_generation.clone(),
        )

    def project_full_mdp_env_reset_binding(self) -> DeviceR05EnvResetBinding:
        """Mint the sole top/env capability during the construction window."""

        self._enter_public_operation()
        if not self._construction_window_open:
            raise DeviceR05ConflictError("env reset binding window is closed")
        if self._env_reset_binding is None:
            capability = object.__new__(DeviceR05EnvResetBinding)
            self._env_reset_binding = capability
        return self._env_reset_binding

    def require_owned_full_mdp_env_reset_binding(
        self, capability: DeviceR05EnvResetBinding
    ) -> DeviceR05EnvResetBindingView:
        """Validate exact identity and expose only an opaque ledger + clone."""

        self._enter_public_operation()
        if (
            not self._construction_window_open
            or type(capability) is not DeviceR05EnvResetBinding
            or capability is not self._env_reset_binding
        ):
            raise DeviceR05ConflictError("env reset binding is stale or foreign")
        return DeviceR05EnvResetBindingView(
            device_r05_owner=self,
            world_reset_identity=self._genesis_world_reset_identity,
            live_reset_ledger_identity=self._live_reset_ledger_identity,
            reset_generation_snapshot=self._reset_generation.clone(),
        )

    def _close_construction_window(self) -> None:
        if (
            self._true_reset_authority is None
            and self._diagnostic_epoch_owner is None
        ):
            self._construction_window_open = False
            raise DeviceR05ConflictError(
                "true-reset authority is not construction-bound"
            )
        self._construction_window_open = False

    def _enter_public_operation(self) -> None:
        """Reject every owner call made recursively by an authority callback."""

        if getattr(self, "_lean_carry_coordinator", None) is not None:
            _require_lean_carry_module()._require_leaf_mutable(self)
        if self._authority_callback_active:
            self._authority_reentry_detected = True
            raise DeviceR05ConflictError(
                "authority callback re-entered the Device-R05 owner"
            )
        if self._question_composition_in_progress:
            raise DeviceR05ConflictError(
                "D05-internal question composition is in progress"
            )

    def _call_authority(self, callback: object, /, *args: object, **kwargs: object) -> object:
        """Run one external callback with a sticky nested-call tripwire."""

        if self._authority_callback_active:
            self._authority_reentry_detected = True
            raise DeviceR05ConflictError("nested authority callback is forbidden")
        self._authority_callback_active = True
        self._authority_reentry_detected = False
        try:
            result = callback(*args, **kwargs)  # type: ignore[operator]
        except Exception:
            reentered = self._authority_reentry_detected
            self._authority_reentry_detected = False
            if reentered:
                self._poison(11)
            raise
        finally:
            self._authority_callback_active = False
        if self._authority_reentry_detected:
            self._authority_reentry_detected = False
            self._poison(11)
            raise DeviceR05ConflictError(
                "authority callback attempted owner re-entry"
            )
        return result

    def _allocate_journal(self) -> None:
        cap = self._journal_capacity
        n = self._num_envs
        c = self._profile.support_size
        options = {"dtype": torch.int64, "device": self._device}
        self._journal_meta = torch.zeros((cap, self._META_WIDTH), **options)
        self._journal_selected = torch.zeros(
            (cap, n), dtype=torch.bool, device=self._device
        )
        self._journal_feasible = torch.zeros(
            (cap, n, c), dtype=torch.bool, device=self._device
        )
        self._journal_construction_reason = torch.full(
            (cap, n, c), -1, **options
        )
        self._journal_candidate_identity = torch.full(
            (cap, n, c), -1, **options
        )
        self._journal_round_construction_reason = torch.full(
            (cap, n, INTERNAL_QUESTION_REDRAW_ROUNDS, c), -1, **options
        )
        self._journal_round_candidate_identity = torch.full(
            (cap, n, INTERNAL_QUESTION_REDRAW_ROUNDS, c), -1, **options
        )
        self._journal_round_producer_fault = torch.zeros(
            (cap, n, INTERNAL_QUESTION_REDRAW_ROUNDS), **options
        )
        self._journal_chosen_round = torch.full((cap, n), -1, **options)
        self._journal_rounds_attempted = torch.zeros((cap, n), **options)
        self._journal_reason = torch.full((cap, n), -1, **options)
        for name in (
            "rng_before_lo",
            "rng_before_hi",
            "rng_after_lo",
            "rng_after_hi",
            "draw_before",
            "draw_after",
            "sampler_generation_before",
            "sampler_generation_after",
            "previous_before",
            "previous_after",
            "reset_before",
            "reset_after",
            "ordinal_before",
            "ordinal_after",
            "outcome_before",
            "outcome_after",
        ):
            setattr(self, f"_journal_{name}", torch.zeros((cap, n), **options))
        self._journal_selected_cell = torch.full((cap, n), -1, **options)
        self._journal_candidate_bank_sequence = torch.full(
            (cap, n), -1, **options
        )
        self._journal_selected_candidate_identity = torch.full(
            (cap, n), -1, **options
        )
        self._journal_task_identity = torch.full((cap, n), -1, **options)
        self._journal_outcome_identity = torch.full((cap, n), -1, **options)
        self._journal_ball_identity = torch.full((cap, n), -1, **options)
        self._journal_cadence_identity = torch.full((cap, n), -1, **options)
        self._journal_primary_fault = torch.zeros((cap, n), **options)
        self._journal_question_producer_fault = torch.zeros((cap, n), **options)
        self._journal_settlement = torch.zeros((cap, n), **options)
        self._journal_child_completion = torch.zeros(
            (cap, len(CHILD_OWNER_ORDER)),
            dtype=torch.bool,
            device=self._device,
        )

    def _require_operable(self) -> None:
        self._enter_public_operation()
        self._close_construction_window()
        if self._poisoned_python:
            raise DeviceR05PoisonedError("device R05 owner is sticky-poisoned")

    def _require_idle(self) -> None:
        self._require_operable()
        if self._active is not None:
            raise DeviceR05ConflictError("another R05 transaction is active")
        if self._active_drain is not None:
            raise DeviceR05ConflictError("a PPO-boundary drain is unacknowledged")
        if any(
            len(record.completed_children) != len(CHILD_OWNER_ORDER)
            for record in self._terminal_receipts.values()
        ):
            raise DeviceR05ConflictError("terminal child completions are incomplete")
        if any(
            len(record.completed_children) != len(CHILD_OWNER_ORDER)
            for record in self._true_reset_receipts.values()
        ):
            raise DeviceR05ConflictError(
                "true-reset child completions are incomplete"
            )

    def _poison(self, reason: int) -> None:
        if self._poisoned_python:
            return
        self._poisoned_python = True
        self._poisoned.fill_(True)
        self._poison_reason.fill_(reason)

    def _publish(self, after: _PublicationState) -> None:
        """Make one already-complete after-image visible; this must not throw."""

        object.__setattr__(self, "_publication", after)

    def _publication_afterimage(self) -> _PublicationState:
        """Clone all mutable publication state before any business write."""

        return self._publication.fork()

    def _journal_row_afterimage(self, slot: int) -> dict[str, torch.Tensor]:
        """Clone one reserved journal slot without mutating live evidence."""

        return {
            name: getattr(self, f"_journal_{name}")[slot].clone()
            for name in _JOURNAL_FIELD_NAMES
        }

    @staticmethod
    def _set_afterimage(after: _PublicationState, name: str, value: object) -> None:
        if name in _LIVE_PUBLICATION_TENSOR_NAMES:
            after.live[name] = value  # type: ignore[assignment]
            return
        if name in _PUBLICATION_REGISTRY_NAMES:
            after.registries[name] = value  # type: ignore[assignment]
            return
        if name in _PUBLICATION_SCALAR_NAMES:
            after.counters[name] = value
            return
        raise DeviceR05Error(f"unknown publication field {name}")

    def _install_journal_row_afterimage(
        self,
        after: _PublicationState,
        *,
        slot: int,
        row: dict[str, torch.Tensor],
    ) -> None:
        if set(row) != set(_JOURNAL_FIELD_NAMES):
            raise DeviceR05Error("journal after-image fields differ")
        after.journal_rows[slot] = row

    def _apply_journal_row(self, slot: int, row: dict[str, torch.Tensor]) -> None:
        """Cold compatibility mirror; the immutable row remains drain truth."""

        for name in _JOURNAL_FIELD_NAMES:
            getattr(self, f"_journal_{name}")[slot].copy_(row[name])

    def _retire_action_epoch_shadow_journal(self) -> None:
        """Make the lean-only debug row nonblocking immediately.

        ActionEpoch already owns the causal packed delta and its optimizer ACK.
        Advancing this private tail does not claim a second acknowledgement;
        it only prevents the redundant fixed-capacity mirror from becoming a
        runtime liveness gate.  The latest physical slot stays available for
        cold debugging and is never consulted as business authority.
        """

        if not self._action_epoch_is_single_business_log:
            return
        self._journal_tail = self._journal_head
        self._checkpoint_requires_global_drain_ack = False

    def poison_from_external_failure(self, reason_code: int) -> None:
        """Sticky top-owner failure ingress; reason code is diagnostic only."""

        self._enter_public_operation()
        self._close_construction_window()
        if type(reason_code) is not int or reason_code < 1:
            raise DeviceR05Error("external poison reason must be a positive int")
        self._poison(reason_code)

    def require_healthy(self) -> None:
        """Fail closed without inspecting device poison lanes on the host."""

        self._require_operable()

    def _reserve_journal_slot(self) -> int:
        outstanding = self._journal_head - self._journal_tail
        if (
            self._journal_head >= _I64_MAX
            or (
                not self._action_epoch_is_single_business_log
                and (
                    outstanding >= self._journal_capacity
                    or outstanding >= self._max_reveal_epochs_per_drain
                )
            )
        ):
            self._poison(1)
            raise DeviceR05PoisonedError(
                "device R05 journal exceeded the PPO-drain bound"
            )
        slot = self._journal_head % self._journal_capacity
        self._journal_meta[slot].zero_()
        self._journal_selected[slot].zero_()
        self._journal_feasible[slot].zero_()
        self._journal_construction_reason[slot].fill_(-1)
        self._journal_candidate_identity[slot].fill_(-1)
        self._journal_round_construction_reason[slot].fill_(-1)
        self._journal_round_candidate_identity[slot].fill_(-1)
        self._journal_round_producer_fault[slot].zero_()
        self._journal_chosen_round[slot].fill_(-1)
        self._journal_rounds_attempted[slot].zero_()
        self._journal_reason[slot].fill_(-1)
        for name in (
            "rng_before_lo",
            "rng_before_hi",
            "rng_after_lo",
            "rng_after_hi",
            "draw_before",
            "draw_after",
            "sampler_generation_before",
            "sampler_generation_after",
            "previous_before",
            "previous_after",
            "reset_before",
            "reset_after",
            "ordinal_before",
            "ordinal_after",
            "outcome_before",
            "outcome_after",
            "primary_fault",
            "question_producer_fault",
            "settlement",
        ):
            getattr(self, f"_journal_{name}")[slot].zero_()
        for name in (
            "selected_cell",
            "candidate_bank_sequence",
            "selected_candidate_identity",
            "task_identity",
            "outcome_identity",
            "ball_identity",
            "cadence_identity",
        ):
            getattr(self, f"_journal_{name}")[slot].fill_(-1)
        self._journal_child_completion[slot].zero_()
        return slot

    def _settle_failed_reveal(
        self,
        prepared: _PreparedRecord,
        *,
        poison_reason: int,
        transfer_sequence: int = 0,
        decision: int = 0,
        primary_fault: Optional[torch.Tensor] = None,
    ) -> None:
        """Make an irreversible reveal failure visible to the sole drain.

        The prepared row already owns one reserved journal slot.  Post-boundary
        failures must consume that slot exactly once; otherwise a first-ever
        failure would leave ``head == tail`` and the sticky poison could never
        be materialized.  The row is settlement evidence only and never claims
        that any business after-image committed.
        """

        if self._journal_head != prepared.epoch - 1:
            self._poison(poison_reason)
            raise DeviceR05PoisonedError(
                "reveal failure row was already settled"
            )
        if primary_fault is None:
            fault = torch.zeros(
                prepared.projection.selected_count,
                dtype=torch.int64,
                device=self._device,
            )
        else:
            fault = _require_tensor(
                primary_fault,
                label="failed_reveal.primary_fault",
                device=self._device,
                dtype=torch.int64,
                shape=(prepared.projection.selected_count,),
            ).clone()
        owner_fault = torch.bitwise_or(
            prepared.question_producer_fault,
            prepared.counter_overflow_fault.to(torch.int64) * (1 << 30),
        )
        fault = torch.bitwise_or(fault, owner_fault)
        fault = torch.where(
            fault.ne(0), fault, torch.full_like(fault, poison_reason)
        )
        row = self._build_reveal_journal_afterimage(
            prepared,
            operation=JOURNAL_ABORT,
            decision=decision,
            transfer_sequence=transfer_sequence,
            primary_fault=fault,
            state_committed=False,
            reason=torch.where(
                prepared.admissible,
                torch.full_like(
                    prepared.reason, REASON_ABORTED_BEFORE_TRANSFER
                ),
                prepared.reason,
            ),
        )
        after = self._publication_afterimage()
        after.live["_poisoned"].fill_(True)
        after.live["_poison_reason"].fill_(poison_reason)
        after.counters["_active"] = None
        after.counters["_journal_head"] = self._journal_head + 1
        self._install_journal_row_afterimage(
            after, slot=prepared.journal_slot, row=row
        )
        self._poisoned_python = True
        self._publish(after)

    def _settle_failed_true_reset(
        self,
        prepared: _PreparedTrueResetRecord,
        *,
        poison_reason: int,
    ) -> None:
        """Retain one drainable settlement row for a failed selected reset."""

        if self._journal_head != prepared.epoch - 1:
            self._poison(poison_reason)
            raise DeviceR05PoisonedError(
                "true-reset failure row was already settled"
            )
        row = self._build_true_reset_journal_afterimage(
            prepared, committed=False
        )
        after = self._publication_afterimage()
        after.live["_poisoned"].fill_(True)
        after.live["_poison_reason"].fill_(poison_reason)
        after.counters["_active"] = None
        after.counters["_journal_head"] = self._journal_head + 1
        self._install_journal_row_afterimage(
            after, slot=prepared.journal_slot, row=row
        )
        self._poisoned_python = True
        self._publish(after)

    def _current_row_cadence(
        self, due: object, motion: object
    ) -> DeviceCadenceProjection:
        """Build the private fixed full-N composer input."""

        epoch = _require_action_epoch_module()
        if type(due) is not epoch.ActionEpochDueRows:
            raise DeviceR05ConflictError("ActionEpoch due-row projection type differs")
        if type(getattr(due, "common_step", None)) is not int or type(
            getattr(motion, "common_step", None)
        ) is not int:
            raise DeviceR05ConflictError("full-N row common-step type differs")
        if due.common_step != motion.common_step:
            raise DeviceR05ConflictError("Epoch and Motion row steps differ")
        n = self._num_envs
        for name in ("due_mask", "construct_mask"):
            _require_tensor(
                getattr(due, name, None),
                label=f"action_epoch.{name}",
                device=self._device,
                dtype=torch.bool,
                shape=(n,),
            )
        motion_i64 = {}
        for name in (
            "episode_tick",
            "scheduled_ordinal",
            "reveal_tick",
            "deadline_tick",
            "next_reveal_tick",
            "reset_generation",
            "swing_generation",
        ):
            motion_i64[name] = _require_tensor(
                getattr(motion, name, None),
                label=f"motion_rows.{name}",
                device=self._device,
                dtype=torch.int64,
                shape=(n,),
            )
        motion_bool = {}
        for name in ("reveal_due", "ready_at_reveal"):
            motion_bool[name] = _require_tensor(
                getattr(motion, name, None),
                label=f"motion_rows.{name}",
                device=self._device,
                dtype=torch.bool,
                shape=(n,),
            )
        k = n

        def pick(value: torch.Tensor) -> torch.Tensor:
            return value.clone().contiguous()

        live_reset = pick(self._reset_generation)
        live_ordinal = pick(self._scheduled_ordinal)
        live_outcome = pick(self._outcome_shot_index)
        live_sampler = pick(self._target_generation)
        room = (
            live_ordinal.lt(_I64_MAX)
            & live_outcome.lt(_I64_MAX)
            & live_sampler.lt(_I64_MAX)
        )
        next_ordinal = torch.where(room, live_ordinal + 1, live_ordinal)
        next_outcome = torch.where(room, live_outcome + 1, live_outcome)
        next_sampler = torch.where(room, live_sampler + 1, live_sampler)
        producer_fault = torch.zeros(k, dtype=torch.int64, device=self._device)
        for fault in (
            pick(motion_i64["reset_generation"]).ne(live_reset),
            pick(motion_i64["scheduled_ordinal"]).ne(next_ordinal),
            pick(motion_bool["reveal_due"]).ne(pick(due.due_mask)),
            ~room,
        ):
            producer_fault = torch.bitwise_or(
                producer_fault,
                fault.to(torch.int64) * PRODUCER_FAULT_QUESTION_CHRONOLOGY,
            )
        return DeviceCadenceProjection(
            selected_count=k,
            selected_env_index=pick(self._row_axis),
            episode_tick=pick(motion_i64["episode_tick"]),
            reveal_tick=pick(motion_i64["reveal_tick"]),
            deadline_tick=pick(motion_i64["deadline_tick"]),
            next_reveal_tick=pick(motion_i64["next_reveal_tick"]),
            swing_generation=pick(motion_i64["swing_generation"]),
            ready_at_reveal=pick(motion_bool["ready_at_reveal"]),
            action_slot=torch.zeros(k, dtype=torch.int64, device=self._device),
            pending_elapsed_s=torch.zeros(
                k, dtype=torch.float32, device=self._device
            ),
            reset_generation=live_reset,
            scheduled_ordinal=next_ordinal,
            outcome_shot_index=next_outcome,
            sampler_generation=next_sampler,
            task_identity=torch.full_like(live_reset, -1),
            cadence_identity=torch.full_like(live_reset, -1),
            cadence_producer_fault=producer_fault.contiguous(),
            cadence_owner_receipt_identity=None,
        )

    @torch.no_grad()
    def advance_action_ball_full_mdp_rows(self) -> None:
        """Settle one real after-command opportunity, or return exactly idle."""

        with self._question_prepare_lock:
            self._require_idle()
            if self._internal_question_compose is None:
                raise DeviceR05ConflictError(
                    "internal question composer is not construction-bound"
                )
            epoch = _require_action_epoch_module()
            epoch_owner = self._diagnostic_epoch_owner
            if type(epoch_owner) is not epoch.ActionEpochOwner:
                raise DeviceR05ConflictError("diagnostic ActionEpoch is not bound")
            due = epoch_owner.prepare_after_command_rows()
            if due is None:
                return None
            started = [False]
            self._question_composition_in_progress = True
            try:
                motion = self._call_authority(
                    self._cadence_authority.project_current_action_epoch_rows
                )
                construct = _require_tensor(
                    getattr(due, "construct_mask", None),
                    label="action_epoch.construct_mask",
                    device=self._device,
                    dtype=torch.bool,
                    shape=(self._num_envs,),
                )
                cadence_receipt = object()
                cadence = self._current_row_cadence(due, motion)
                cadence = DeviceCadenceProjection(
                    **{
                        **cadence.__dict__,
                        "cadence_owner_receipt_identity": cadence_receipt,
                    }
                )
                prepared_token = self._prepare_many_impl(
                    cadence_receipt,
                    question_receipt=None,
                    internal_question_compose=self._internal_question_compose,
                    internal_callback_started=started,
                    owned_projection=cadence,
                    construction_mask=construct,
                    transaction_ordinal=due.common_step,
                )
                preview_token = self._preview_impl(prepared_token)
                preview = self._require_preview(preview_token)
                prepared = preview.prepared
                token = object.__new__(DeviceR05RowTransaction)
                record = self._build_row_transaction(
                    token, due, prepared, preview
                )
            except BaseException as exc:
                self._poison(22)
                epoch_owner.poison_owner_write("r05_runtime", 22, owner=self)
                raise DeviceR05PoisonedError(
                    "full-N D05 composition failed after due freeze"
                ) from exc
            finally:
                self._question_composition_in_progress = False
            self._row_transaction_records[token] = record
            self._active_row_transaction = token
            record.stage = "settling"
            try:
                epoch_owner.settle_d05_transaction(token)
                if record.stage != "settled":
                    raise DeviceR05ConflictError(
                        "ActionEpoch did not complete all three D05 writers"
                    )
                publisher = getattr(
                    self._diagnostic_motion_owner,
                    "publish_action_ball_full_mdp_post_d05_observation",
                    None,
                )
                if (
                    not callable(publisher)
                    or getattr(publisher, "__self__", None)
                    is not self._diagnostic_motion_owner
                    or getattr(publisher, "__func__", None)
                    is not getattr(
                        type(self._diagnostic_motion_owner),
                        "publish_action_ball_full_mdp_post_d05_observation",
                        None,
                    )
                ):
                    raise DeviceR05ConflictError(
                        "Motion post-D05 observation publisher differs"
                    )
                publisher()
            except BaseException as exc:
                record.stage = "failed"
                self._active_row_transaction = None
                if not self._poisoned_python:
                    self._poison(24)
                epoch_owner.poison_owner_write("r05_runtime", 24, owner=self)
                raise DeviceR05PoisonedError(
                    "full-N D05 settlement failed"
                ) from exc
            finally:
                self._row_transaction_records.pop(token, None)

    def _prepare_many_impl(
        self,
        cadence_receipt: object,
        *,
        question_receipt: object,
        internal_question_compose: object,
        internal_callback_started: Optional[list[bool]],
        owned_projection: DeviceCadenceProjection,
        construction_mask: torch.Tensor,
        transaction_ordinal: int,
    ) -> DeviceR05PreparedToken:
        """Locked common prepare path; internal cadence never leaves D05 first."""

        projection = owned_projection
        k = self._num_envs
        index = self._row_axis
        if projection.selected_count != k:
            raise DeviceR05ConflictError("D05 composer input is not full-N")
        construction_mask = _require_tensor(
            construction_mask,
            label="D05 construction_mask",
            device=self._device,
            dtype=torch.bool,
            shape=(k,),
        ).clone()
        question_rng_before_lo = self._rng_lo[index].clone()
        question_rng_before_hi = self._rng_hi[index].clone()
        question_rng_after_lo = question_rng_before_lo
        question_rng_after_hi = question_rng_before_hi
        question_draw_width = 0
        question_draw_u01 = torch.empty(
            (k, 0), dtype=torch.float32, device=self._device
        )
        if type(transaction_ordinal) is not int or transaction_ordinal < 0:
            raise DeviceR05ConflictError("D05 transaction ordinal differs")
        candidate_count = (
            k * INTERNAL_QUESTION_REDRAW_ROUNDS * self._profile.support_size
        )
        candidate_identity_highwater_before = (
            transaction_ordinal * candidate_count + 1
        )
        candidate_identity_highwater_after = (
            candidate_identity_highwater_before
        )
        cadence_for_question = DeviceCadenceProjection(
            **{
                name: value.clone() if type(value) is torch.Tensor else value
                for name, value in projection.__dict__.items()
            }
        )
        if internal_question_compose is None:
            question = self._call_authority(
                self._question_authority.project_r05_candidate_bank,
                question_receipt,
                cadence_receipt=cadence_receipt,
                cadence_projection=cadence_for_question,
                device=self._device,
                support_size=self._profile.support_size,
            )
        else:
            if internal_callback_started is None:
                raise DeviceR05Error("internal callback marker is absent")
            internal_callback_started[0] = True
            internal_context = object.__new__(
                _DeviceR05InternalQuestionContext
            )
            (
                question_rng_after_lo,
                question_rng_after_hi,
                question_draw_u01,
            ) = _draw_internal_question_uniform01(
                question_rng_before_lo,
                question_rng_before_hi,
            )
            question_draw_width = (
                INTERNAL_QUESTION_REDRAW_ROUNDS
                * INTERNAL_QUESTION_DRAW_WIDTH
            )
            profile_for_question = DeviceProfileProjection(
                profile_sha256=self._profile.profile_sha256,
                profile_binding_sha256=self._profile.profile_binding_sha256,
                cell_ids=self._profile.cell_ids,
                semantic_sha256s=self._profile.semantic_sha256s,
                targets_xy_m=self._profile.targets_xy_m.clone(),
            )
            if (
                transaction_ordinal >= _I64_MAX
                or candidate_identity_highwater_before
                > (_I64_MAX - candidate_count)
            ):
                self._poison(13)
                raise DeviceR05PoisonedError(
                    "device R05 candidate identity chronology exhausted int64"
                )
            candidate_identity = torch.arange(
                candidate_count,
                dtype=torch.int64,
                device=self._device,
            ).reshape(
                k,
                INTERNAL_QUESTION_REDRAW_ROUNDS,
                self._profile.support_size,
            )
            candidate_identity = (
                candidate_identity
                + candidate_identity_highwater_before
            ).contiguous()
            candidate_identity_highwater_after = (
                candidate_identity_highwater_before + candidate_count
            )
            bank_sequence = transaction_ordinal + 1
            with _ACTIVE_INTERNAL_QUESTION_CONTEXTS_LOCK:
                _ACTIVE_INTERNAL_QUESTION_CONTEXTS[internal_context] = (
                    self._question_authority,
                    cadence_receipt,
                    cadence_for_question,
                    profile_for_question,
                    self._device,
                    self._profile.support_size,
                    question_draw_u01,
                    candidate_identity,
                    construction_mask.clone(),
                    bank_sequence,
                )
            callback_failed = False
            try:
                question = self._call_authority(
                    internal_question_compose,
                    internal_context,
                )
            except BaseException:
                callback_failed = True
                raise
            finally:
                with _ACTIVE_INTERNAL_QUESTION_CONTEXTS_LOCK:
                    unconsumed = _ACTIVE_INTERNAL_QUESTION_CONTEXTS.pop(
                        internal_context, None
                    )
                if unconsumed is not None and not callback_failed:
                    raise DeviceR05ConflictError(
                        "internal question composer did not consume its context"
                    )
        if type(question) is DeviceQuestionProjection:
            cadence_receipt_identity = question.cadence_receipt_identity
            bank_identity = question.bank_identity
            bank_sequence = question.bank_sequence
            bank_source = question.bank
            round_bank_source = question.round_bank
            producer_fault_source = question.producer_fault
            question_selected_count = question.selected_count
            question_support_size = question.support_size
            chronology_source = question.chronology
            round_chronology_source = question.round_chronology
        else:
            cadence_receipt_identity = None
            bank_identity = None
            bank_sequence = None
            bank_source = None
            round_bank_source = None
            producer_fault_source = None
            question_selected_count = None
            question_support_size = None
            chronology_source = None
            round_chronology_source = None
        internal_round_bank = internal_question_compose is not None
        if (
            type(question) is not DeviceQuestionProjection
            or cadence_receipt_identity is not cadence_receipt
            or bank_identity is None
            or type(bank_sequence) is not int
            or bank_sequence > _I64_MAX
            or bank_sequence != transaction_ordinal + 1
            or question_selected_count != k
            or question_support_size != self._profile.support_size
            or (
                internal_round_bank
                and (
                    bank_source is not None
                    or chronology_source is not None
                    or type(round_bank_source)
                    is not DeviceR05CandidateRoundBank
                    or type(round_chronology_source)
                    is not DeviceQuestionRoundChronology
                )
            )
            or (
                not internal_round_bank
                and (
                    type(bank_source) is not DeviceR05CandidateBank
                    or round_bank_source is not None
                    or round_chronology_source is not None
                )
            )
        ):
            raise DeviceR05ConflictError("candidate bank authority differs")
        support = self._profile.support_size
        rounds = INTERNAL_QUESTION_REDRAW_ROUNDS if internal_round_bank else 1
        prefix_shape = (k, rounds, support)
        bank_fields = (
            ("candidate_identity", torch.int64, prefix_shape),
            ("construction_reason", torch.int64, prefix_shape),
            (
                "motion_task_f32",
                torch.float32,
                (*prefix_shape, len(MOTION_TASK_F32_FIELDS)),
            ),
            (
                "racket_task_f32",
                torch.float32,
                (*prefix_shape, len(RACKET_F32_FIELDS)),
            ),
            (
                "physical_state_f32",
                torch.float32,
                (*prefix_shape, len(PHYSICAL_STATE_F32_FIELDS)),
            ),
        )
        source_bank = round_bank_source if internal_round_bank else bank_source
        bank_sources = {
            name: (
                getattr(source_bank, name)
                if internal_round_bank
                else getattr(source_bank, name).unsqueeze(1)
            )
            for name, _, _ in bank_fields
        }
        copied_bank = {}
        for name, dtype, shape in bank_fields:
            value = _require_tensor(
                bank_sources[name],
                label=f"candidate_bank.{name}",
                device=self._device,
                dtype=dtype,
                shape=shape,
            )
            if not value.is_contiguous():
                raise DeviceR05Error(
                    f"candidate_bank.{name} must be contiguous"
                )
            copied_bank[name] = value.clone()
        round_fault_source = (
            _require_tensor(
                round_bank_source.producer_fault,
                label="candidate_round_bank.producer_fault",
                device=self._device,
                dtype=torch.int64,
                shape=(k, rounds),
            )
            if internal_round_bank
            else _require_tensor(
                producer_fault_source,
                label="question.producer_fault",
                device=self._device,
                dtype=torch.int64,
                shape=(k,),
            ).unsqueeze(1)
        )
        if not round_fault_source.is_contiguous():
            raise DeviceR05Error(
                "candidate_round_bank.producer_fault must be contiguous"
            )
        round_bank = DeviceR05CandidateRoundBank(
            **copied_bank,
            producer_fault=round_fault_source.clone(),
        )
        structural_fault = torch.bitwise_or(
            projection.cadence_producer_fault,
            (
                _require_tensor(
                    producer_fault_source,
                    label="question.structural_producer_fault",
                    device=self._device,
                    dtype=torch.int64,
                    shape=(k,),
                ).clone()
                if internal_round_bank
                else torch.zeros(k, dtype=torch.int64, device=self._device)
            ),
        ).contiguous()
        structural_fault = torch.where(
            construction_mask,
            structural_fault,
            torch.zeros_like(structural_fault),
        ).contiguous()
        # Structural faults are row-owned; a malformed peer must not censor
        # any other environment in the full-N payload.
        source_structural_fault_bits = structural_fault.clone()
        structural_round_fault = torch.zeros(
            (k, rounds), dtype=torch.int64, device=self._device
        )
        structural_round_fault = torch.bitwise_or(
            structural_round_fault,
            round_bank.producer_fault.lt(0).to(torch.int64)
            * PRODUCER_FAULT_QUESTION_CHRONOLOGY,
        ).contiguous()
        chronology = None
        if internal_round_bank:
            chronology = DeviceQuestionRoundChronology(
                **{
                    name: _require_tensor(
                        getattr(round_chronology_source, name),
                        label=f"question.round_chronology.{name}",
                        device=self._device,
                        dtype=torch.int64,
                        shape=prefix_shape,
                    ).clone()
                    for name in DeviceQuestionRoundChronology.__dataclass_fields__
                }
            )
        elif chronology_source is not None:
            if type(chronology_source) is not DeviceQuestionChronology:
                raise DeviceR05ConflictError("question chronology type differs")
            chronology = DeviceQuestionRoundChronology(
                **{
                    name: _require_tensor(
                        getattr(chronology_source, name),
                        label=f"question.chronology.{name}",
                        device=self._device,
                        dtype=torch.int64,
                        shape=(k, support),
                    ).unsqueeze(1).clone()
                    for name in DeviceQuestionChronology.__dataclass_fields__
                }
            )
        if internal_round_bank:
            expected_candidate_identity = candidate_identity
            candidate_identity_fault = round_bank.candidate_identity.ne(
                expected_candidate_identity
            ).any(dim=2)
            structural_round_fault = torch.bitwise_or(
                structural_round_fault,
                candidate_identity_fault.to(torch.int64)
                * PRODUCER_FAULT_QUESTION_CHRONOLOGY,
            ).contiguous()
        if chronology is not None:
            chronology_fault = (
                chronology.action_uid.le(0)
                | chronology.contact_tick.lt(0)
                | chronology.launch_tick.lt(0)
                | chronology.contact_tick.le(chronology.launch_tick)
                | chronology.chosen_horizon_ticks.le(0)
                | chronology.chosen_horizon_ticks.ne(
                    chronology.contact_tick - chronology.launch_tick
                )
            ).any(dim=2)
            structural_round_fault = torch.bitwise_or(
                structural_round_fault,
                chronology_fault.to(torch.int64)
                * PRODUCER_FAULT_QUESTION_CHRONOLOGY,
            ).contiguous()
            admitted = round_bank.construction_reason.eq(
                QUESTION_CONSTRUCTION_REASON_ADMITTED
            )
            suffix_crosses = chronology.task_close_tick.ge(
                projection.next_reveal_tick.reshape(k, 1, 1)
            )
            suffix_reason = round_bank.construction_reason.eq(
                QUESTION_CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL
            )
            active_task_chronology = admitted | suffix_reason
            task_close_invalid = active_task_chronology & chronology.task_close_tick.lt(
                projection.reveal_tick.reshape(k, 1, 1)
            )
            chronology_reason_fault = (
                task_close_invalid
                | (suffix_reason & ~suffix_crosses)
                | (admitted & suffix_crosses)
            ).any(dim=2)
            structural_round_fault = torch.bitwise_or(
                structural_round_fault,
                chronology_reason_fault.to(torch.int64)
                * PRODUCER_FAULT_QUESTION_CHRONOLOGY,
            ).contiguous()
        reason_domain_fault = (
            round_bank.construction_reason.lt(
                QUESTION_CONSTRUCTION_REASON_ADMITTED
            )
            | round_bank.construction_reason.gt(
                QUESTION_CONSTRUCTION_REASON_MAX_REJECT
            )
        ).any(dim=2)
        structural_round_fault = torch.bitwise_or(
            structural_round_fault,
            reason_domain_fault.to(torch.int64)
            * PRODUCER_FAULT_QUESTION_CHRONOLOGY,
        ).contiguous()
        if internal_round_bank:
            structural_round_fault = torch.where(
                construction_mask[:, None],
                structural_round_fault,
                torch.zeros_like(structural_round_fault),
            ).contiguous()
        # Reduce only the fixed redraw rounds, independently for every row.
        structural_fault = torch.bitwise_or(
            structural_fault,
            _bitwise_or_rounds(structural_round_fault),
        ).contiguous()
        question = DeviceQuestionProjection(
            cadence_receipt_identity=cadence_receipt_identity,
            bank_identity=bank_identity,
            bank_sequence=bank_sequence,
            bank=(
                None
                if internal_round_bank
                else DeviceR05CandidateBank(
                    **{
                        name: getattr(round_bank, name).squeeze(1)
                        for name in DeviceR05CandidateBank.__dataclass_fields__
                    }
                )
            ),
            producer_fault=(
                structural_fault.clone()
                if internal_round_bank
                else round_bank.producer_fault.squeeze(1).clone()
            ),
            selected_count=question_selected_count,
            support_size=question_support_size,
            chronology=(
                None
                if internal_round_bank or chronology is None
                else DeviceQuestionChronology(
                    **{
                        name: getattr(chronology, name).squeeze(1)
                        for name in DeviceQuestionChronology.__dataclass_fields__
                    }
                )
            ),
            round_bank=(round_bank if internal_round_bank else None),
            round_chronology=(chronology if internal_round_bank else None),
            full_key_sha256=question.full_key_sha256,
            task_sha256=question.task_sha256,
            physical_question_receipt_identity=(
                question.physical_question_receipt_identity
            ),
        )
        if question.full_key_sha256 is not None:
            question = DeviceQuestionProjection(
                **{
                    **question.__dict__,
                    "full_key_sha256": _require_tensor(
                        question.full_key_sha256,
                        label="question.full_key_sha256",
                        device=self._device,
                        dtype=torch.uint8,
                        shape=(k, support, 32),
                    ).clone(),
                    "task_sha256": _require_tensor(
                        question.task_sha256,
                        label="question.task_sha256",
                        device=self._device,
                        dtype=torch.uint8,
                        shape=(k, support, 32),
                    ).clone(),
                }
            )
        elif question.task_sha256 is not None:
            raise DeviceR05ConflictError(
                "question full-key/task identity pair differs"
            )
        if internal_round_bank and (
            question.full_key_sha256 is not None
            or question.task_sha256 is not None
        ):
            raise DeviceR05ConflictError(
                "internal round bank cannot carry legacy digest identity"
            )
        round_feasible = round_bank.construction_reason.eq(
            QUESTION_CONSTRUCTION_REASON_ADMITTED
        ) & construction_mask.reshape(k, 1, 1)
        # ActionEpoch owns the only row-wise chronology and settlement log.
        slot = -1

        before_lo = question_rng_before_lo
        before_hi = question_rng_before_hi
        draw_before = self._draw_count[index].clone()
        generation_before = self._target_generation[index].clone()
        previous = self._previous_cell_index[index].clone()
        cells = torch.arange(
            self._profile.support_size,
            dtype=torch.int64,
            device=self._device,
        )
        round_eligible = torch.logical_and(
            round_feasible,
            cells.reshape(1, 1, support)
            != previous.reshape(k, 1, 1),
        )
        round_eligible_count = torch.sum(
            round_eligible, dim=2, dtype=torch.int64
        )
        round_success = round_eligible_count.gt(0)
        numeric_round_fault = round_bank.producer_fault.ne(0)
        event = round_success | numeric_round_fault
        has_event = event.any(dim=1)
        first_event_index = torch.argmax(event.to(torch.int64), dim=1)
        row_index = torch.arange(k, dtype=torch.int64, device=self._device)
        first_event_fault = numeric_round_fault[
            row_index, first_event_index
        ]
        first_event_success = round_success[
            row_index, first_event_index
        ] & ~first_event_fault

        structural_round_event = structural_round_fault.ne(0)
        has_structural_round_fault = structural_round_event.any(dim=1)
        first_structural_round_index = torch.argmax(
            structural_round_event.to(torch.int64), dim=1
        )
        has_source_structural_fault = source_structural_fault_bits.ne(0)
        has_structural_fault = structural_fault.ne(0)
        structural_terminal_index = torch.where(
            has_source_structural_fault,
            torch.zeros_like(first_structural_round_index),
            first_structural_round_index,
        )
        chosen_round_index = torch.where(
            has_event & first_event_success & ~has_structural_fault,
            first_event_index,
            torch.full_like(first_event_index, -1),
        )
        rounds_attempted = torch.where(
            has_structural_fault,
            structural_terminal_index + 1,
            torch.where(
                has_event,
                first_event_index + 1,
                torch.full_like(first_event_index, rounds),
            ),
        ).contiguous()
        admissible = chosen_round_index.ge(0)
        terminal_round_index = torch.where(
            admissible,
            chosen_round_index,
            rounds_attempted - 1,
        ).contiguous()
        attempted_round = (
            torch.arange(rounds, dtype=torch.int64, device=self._device)
            .reshape(1, rounds)
            .lt(rounds_attempted.reshape(k, 1))
        )
        prefix_numeric_fault = _bitwise_or_rounds(
            torch.where(
                attempted_round,
                round_bank.producer_fault,
                torch.zeros_like(round_bank.producer_fault),
            )
        )
        producer_fault = torch.bitwise_or(
            structural_fault, prefix_numeric_fault
        ).contiguous()
        draw_counter_overflow = (
            draw_before > (_I64_MAX - (question_draw_width + 1))
            if internal_round_bank
            else (
                admissible
                & (draw_before > (_I64_MAX - (question_draw_width + 1)))
            )
        )
        generation_counter_overflow = admissible & generation_before.eq(
            _I64_MAX
        )
        counter_overflow_fault = (
            draw_counter_overflow | generation_counter_overflow
        ).contiguous()
        outcome_highwater = self._next_outcome_identity.clone()
        ball_highwater = self._next_ball_identity.clone()
        # ``next`` is an exclusive positive high-water.  Reserve at most k
        # identities while leaving the post-commit high-water representable.
        # The unsafe branch uses base 1, so no device expression wraps even
        # when a restored/fault-injected counter is at int64 MAX.
        identity_counter_room = torch.logical_and(
            outcome_highwater <= (_I64_MAX - k),
            ball_highwater <= (_I64_MAX - k),
        )
        counter_overflow_fault = torch.logical_or(
            counter_overflow_fault,
            torch.logical_and(admissible, ~identity_counter_room),
        )
        owner_fault_free = torch.logical_and(
            torch.logical_and(producer_fault.eq(0), ~counter_overflow_fault),
            ~self._poisoned,
        )
        # Candidate construction is independent of the plant's playback
        # readiness.  A clean not-ready row retains the composed candidate and
        # is still published as a task; readiness only controls whether Motion
        # playback and Physical launch may start for that task.
        install_admissible = torch.logical_and(admissible, owner_fault_free)
        attempted_cells = attempted_round.unsqueeze(2)
        any_feasible = (round_feasible & attempted_cells).reshape(k, -1).any(
            dim=1
        )
        reason = torch.where(
            admissible,
            torch.zeros_like(rounds_attempted),
            torch.where(
                any_feasible,
                torch.full_like(
                    rounds_attempted, REASON_ONLY_PREVIOUS_TARGET_FEASIBLE
                ),
                torch.full_like(
                    rounds_attempted, REASON_NO_FEASIBLE_TARGET
                ),
            ),
        )
        next_lo, next_hi, draw_lo, draw_hi = _splitmix64_lanes(
            question_rng_after_lo, question_rng_after_hi
        )
        terminal_eligible = round_eligible[
            row_index, terminal_round_index
        ]
        terminal_eligible_count = round_eligible_count[
            row_index, terminal_round_index
        ]
        rank = _u64_mul_u31_high(
            draw_lo, draw_hi, terminal_eligible_count
        )
        compact_rank = torch.cumsum(
            terminal_eligible.to(torch.int64), dim=1
        ) - 1
        match = torch.logical_and(
            terminal_eligible, compact_rank == rank.unsqueeze(1)
        )
        selected = torch.argmax(match.to(torch.int64), dim=1)
        selected = torch.where(
            install_admissible, selected, torch.full_like(selected, -1)
        )
        safe_selected = torch.clamp(selected, min=0)
        selected_target = self._profile.targets_xy_m[safe_selected]
        selected_target = torch.where(
            install_admissible.unsqueeze(1),
            selected_target,
            torch.zeros_like(selected_target),
        )
        terminal_bank = {
            name: getattr(round_bank, name)[row_index, terminal_round_index]
            for name in DeviceR05CandidateBank.__dataclass_fields__
        }
        feasible = terminal_bank["construction_reason"].eq(
            QUESTION_CONSTRUCTION_REASON_ADMITTED
        )
        eligible = terminal_eligible
        chosen_round = torch.where(
            admissible,
            chosen_round_index + 1,
            torch.full_like(chosen_round_index, -1),
        ).contiguous()
        def selected_bank(name: str) -> torch.Tensor:
            candidate = terminal_bank[name]
            selected_value = candidate[row_index, safe_selected]
            expand = (k,) + (1,) * (selected_value.ndim - 1)
            return torch.where(
                install_admissible.reshape(expand),
                selected_value,
                torch.zeros_like(selected_value),
            )

        def selected_question_identity(
            source: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if source is None:
                return None
            selected_value = source[row_index, safe_selected]
            return torch.where(
                install_admissible.reshape(k, 1),
                selected_value,
                torch.zeros_like(selected_value),
            )

        identity_advance_count = torch.sum(
            install_admissible.to(torch.int64)
        ).reshape(())
        identity_offset = (
            torch.cumsum(install_admissible.to(torch.int64), dim=0) - 1
        )
        safe_outcome_base = torch.where(
            identity_counter_room,
            outcome_highwater,
            torch.ones_like(outcome_highwater),
        )
        safe_ball_base = torch.where(
            identity_counter_room,
            ball_highwater,
            torch.ones_like(ball_highwater),
        )
        reserved_outcome_identity = torch.where(
            install_admissible,
            safe_outcome_base + identity_offset,
            torch.full_like(identity_offset, -1),
        ).contiguous()
        reserved_ball_identity = torch.where(
            install_admissible,
            safe_ball_base + identity_offset,
            torch.full_like(identity_offset, -1),
        ).contiguous()
        if chronology is not None:
            selected_chronology = {
                name: getattr(chronology, name)[
                    row_index, terminal_round_index, safe_selected
                ]
                for name in DeviceQuestionRoundChronology.__dataclass_fields__
            }
            selected_chronology = {
                name: torch.where(
                    install_admissible,
                    value,
                    torch.full_like(value, -1),
                ).contiguous()
                for name, value in selected_chronology.items()
            }
            projection = DeviceCadenceProjection(
                **{
                    **projection.__dict__,
                    "task_identity": reserved_outcome_identity.clone(),
                    "cadence_identity": reserved_ball_identity.clone(),
                    **selected_chronology,
                }
            )
        rng_advance_mask = (
            construction_mask & ~draw_counter_overflow
            if internal_round_bank
            else construction_mask & install_admissible
        )
        safe_draw_before = torch.where(
            draw_counter_overflow,
            torch.zeros_like(draw_before),
            draw_before,
        )

        token = object.__new__(DeviceR05PreparedToken)
        selected_mask = torch.ones(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        record = _PreparedRecord(
            capability=token,
            epoch=transaction_ordinal + 1,
            journal_slot=slot,
            projection=projection,
            question_projection=question,
            question_producer_fault=producer_fault,
            selected_candidate_identity=selected_bank("candidate_identity"),
            selected_construction_reason=selected_bank("construction_reason"),
            bank_candidate_identity=terminal_bank[
                "candidate_identity"
            ].clone(),
            bank_construction_reason=terminal_bank[
                "construction_reason"
            ].clone(),
            round_candidate_identity=round_bank.candidate_identity.clone(),
            round_construction_reason=(
                round_bank.construction_reason.clone()
            ),
            round_producer_fault=round_bank.producer_fault.clone(),
            chosen_round=chosen_round,
            rounds_attempted=rounds_attempted,
            selected_motion_task_f32=selected_bank("motion_task_f32"),
            selected_racket_task_f32=selected_bank("racket_task_f32"),
            selected_physical_state_f32=selected_bank("physical_state_f32"),
            selected_full_key_sha256=selected_question_identity(
                question.full_key_sha256
            ),
            selected_task_sha256=selected_question_identity(
                question.task_sha256
            ),
            reserved_outcome_identity=reserved_outcome_identity,
            reserved_ball_identity=reserved_ball_identity,
            outcome_identity_highwater_before=outcome_highwater,
            ball_identity_highwater_before=ball_highwater,
            identity_advance_count=identity_advance_count,
            identity_counter_room=identity_counter_room,
            internal_question=internal_round_bank,
            candidate_identity_highwater_before=(
                candidate_identity_highwater_before
            ),
            candidate_identity_highwater_after=(
                candidate_identity_highwater_after
            ),
            selected_index=index,
            selected_mask=selected_mask,
            feasible=feasible,
            eligible=eligible,
            admissible=admissible,
            owner_fault_free=owner_fault_free,
            counter_overflow_fault=counter_overflow_fault,
            reason=reason,
            rng_before_lo=before_lo,
            rng_before_hi=before_hi,
            rng_after_lo=torch.where(rng_advance_mask, next_lo, before_lo),
            rng_after_hi=torch.where(rng_advance_mask, next_hi, before_hi),
            draw_before=draw_before,
            draw_after=torch.where(
                rng_advance_mask,
                safe_draw_before + (question_draw_width + 1),
                draw_before,
            ),
            rng_advance_mask=rng_advance_mask.contiguous(),
            generation_before=generation_before,
            generation_after=(
                generation_before + install_admissible.to(torch.int64)
            ),
            previous_before=previous,
            reset_before=self._reset_generation[index].clone(),
            ordinal_before=self._scheduled_ordinal[index].clone(),
            outcome_before=self._outcome_shot_index[index].clone(),
            selected_cell=selected,
            selected_target_xy_m=selected_target,
            stage="prepared",
        )
        self._prepared_records[token] = record
        self._active = token
        return token

    def _require_prepared(
        self, token: DeviceR05PreparedToken
    ) -> _PreparedRecord:
        record = self._prepared_records.get(token)
        if (
            type(token) is not DeviceR05PreparedToken
            or token is not self._active
            or record is None
            or record.capability is not token
            or record.stage != "prepared"
        ):
            raise DeviceR05ConflictError("prepared token is not active")
        return record

    def _preview_impl(
        self, token: DeviceR05PreparedToken
    ) -> DeviceR05PreviewToken:
        """Advance the active prepared capability without public-entry checks."""

        prepared = self._require_prepared(token)
        preview = object.__new__(DeviceR05PreviewToken)
        prepared.stage = "previewed"
        self._preview_records[preview] = _PreviewRecord(
            capability=preview,
            prepared=prepared,
            preview_identity=preview,
            stage="previewed",
        )
        self._active = preview
        return preview

    def preview(self, token: DeviceR05PreparedToken) -> DeviceR05PreviewToken:
        self._enter_public_operation()
        return self._preview_impl(token)

    def _require_preview(
        self, token: DeviceR05PreviewToken
    ) -> _PreviewRecord:
        record = self._preview_records.get(token)
        if (
            type(token) is not DeviceR05PreviewToken
            or token is not self._active
            or record is None
            or record.capability is not token
            or record.stage != "previewed"
        ):
            raise DeviceR05ConflictError("preview token is not active")
        return record

    def _action_epoch_candidate(self, prepared: _PreparedRecord) -> object:
        """Pack owner-private construction facts into the frozen full-N ABI."""

        epoch = _require_action_epoch_module()
        n = self._num_envs
        projection = prepared.projection
        if projection.selected_count != n:
            raise DeviceR05ConflictError("D05 candidate plane is not full-N")

        def required_i64(name: str) -> torch.Tensor:
            value = getattr(projection, name)
            if value is None:
                raise DeviceR05ConflictError(
                    f"diagnostic D05 cadence lacks exact {name}"
                )
            return _require_tensor(
                value,
                label=f"diagnostic_epoch.{name}",
                device=self._device,
                dtype=torch.int64,
                shape=(n,),
            )

        def plane_i64(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(n, 1).clone().contiguous()

        shot_key = epoch.ActionEpochShotKey(
            reset_generation=plane_i64(projection.reset_generation),
            ball_generation=plane_i64(projection.swing_generation),
            action_uid=plane_i64(required_i64("action_uid")),
            action_slot=plane_i64(projection.action_slot),
            shot_index=plane_i64(projection.outcome_shot_index),
            task_identity=plane_i64(projection.task_identity),
            outcome_identity=plane_i64(prepared.reserved_outcome_identity),
            ball_identity=plane_i64(prepared.reserved_ball_identity),
        )
        identity = epoch.EpochIdentityPayload(
            shot_key=shot_key,
            scheduled_ordinal=plane_i64(projection.scheduled_ordinal),
            target_generation=plane_i64(prepared.generation_after),
            selected_cell=plane_i64(prepared.selected_cell),
            candidate_identity=plane_i64(
                prepared.selected_candidate_identity
            ),
        )
        clocks = epoch.EpochClockPayload(
            reveal_tick=plane_i64(projection.reveal_tick),
            contact_tick=plane_i64(required_i64("contact_tick")),
            launch_tick=plane_i64(required_i64("launch_tick")),
            # Epoch deadline is the question-owned complete Motion task close.
            # C01's policy-opportunity deadline remains a separate cadence
            # field in the child projection and never closes the action.
            deadline_tick=plane_i64(required_i64("task_close_tick")),
            next_reveal_tick=plane_i64(projection.next_reveal_tick),
        )
        selected_task = torch.cat(
            (
                prepared.selected_motion_task_f32,
                prepared.selected_racket_task_f32,
                prepared.selected_physical_state_f32,
            ),
            dim=1,
        ).contiguous()
        if tuple(selected_task.shape) != (
            n,
            epoch.TASK_F32_WIDTH,
        ):
            raise DeviceR05ConflictError("diagnostic epoch task width differs")
        task = epoch.EpochTaskPayload(
            task_f32=selected_task.reshape(n, 1, epoch.TASK_F32_WIDTH),
            task_valid=(
                prepared.admissible & prepared.owner_fault_free
            ).reshape(n, 1).contiguous(),
        )
        owner_fault_bits = torch.zeros(
            (n, 1, len(epoch.OWNER_ORDER)),
            dtype=torch.int64,
            device=self._device,
        )
        owner_fault_bits[:, 0, epoch.OWNER_ORDER.index("r05_runtime")] = (
            torch.bitwise_or(
                prepared.question_producer_fault,
                prepared.counter_overflow_fault.to(torch.int64) * (1 << 30),
            )
        )
        return epoch.ActionEpochD05CandidateProjection(
            identity=identity,
            clocks=clocks,
            task=task,
            rng_counter=prepared.draw_after.reshape(n, 1).clone().contiguous(),
            construction_admissible=(
                prepared.admissible.reshape(n, 1).clone().contiguous()
            ),
            playback_admissible=(
                projection.ready_at_reveal.reshape(n, 1).clone().contiguous()
            ),
            owner_fault_bits=owner_fault_bits.contiguous(),
        )

    def _build_row_transaction(
        self,
        token: DeviceR05RowTransaction,
        due: object,
        prepared: _PreparedRecord,
        preview: _PreviewRecord,
    ) -> _RowTransactionRecord:
        n = self._num_envs
        due_mask = due.due_mask.clone()
        construct_mask = due.construct_mask.clone()
        candidate = self._action_epoch_candidate(prepared)
        epoch = _require_action_epoch_module()
        construct_slots = construct_mask[:, None]

        def private_i64(value: torch.Tensor) -> torch.Tensor:
            return torch.where(
                construct_slots, value, torch.full_like(value, -1)
            ).contiguous()

        candidate = epoch.ActionEpochD05CandidateProjection(
            identity=epoch.EpochIdentityPayload(
                shot_key=epoch.ActionEpochShotKey(
                    **{
                        name: private_i64(
                            getattr(candidate.identity.shot_key, name)
                        )
                        for name in epoch.ActionEpochShotKey.__dataclass_fields__
                    }
                ),
                **{
                    name: private_i64(getattr(candidate.identity, name))
                    for name in (
                        "scheduled_ordinal",
                        "target_generation",
                        "selected_cell",
                        "candidate_identity",
                    )
                },
            ),
            clocks=epoch.EpochClockPayload(
                **{
                    name: private_i64(getattr(candidate.clocks, name))
                    for name in candidate.clocks.__dataclass_fields__
                }
            ),
            task=epoch.EpochTaskPayload(
                task_f32=torch.where(
                    construct_slots.unsqueeze(2),
                    candidate.task.task_f32,
                    torch.zeros_like(candidate.task.task_f32),
                ).contiguous(),
                task_valid=(
                    candidate.task.task_valid & construct_slots
                ).contiguous(),
            ),
            rng_counter=torch.where(
                construct_slots,
                candidate.rng_counter,
                torch.zeros_like(candidate.rng_counter),
            ).contiguous(),
            construction_admissible=(
                candidate.construction_admissible & construct_slots
            ).contiguous(),
            playback_admissible=(
                candidate.playback_admissible & construct_slots
            ).contiguous(),
            owner_fault_bits=torch.where(
                construct_slots.unsqueeze(2),
                candidate.owner_fault_bits,
                torch.zeros_like(candidate.owner_fault_bits),
            ).contiguous(),
        )
        key_valid = epoch.row_identity.action_epoch_shot_key_valid(
            candidate.identity.shot_key
        )[:, 0]
        active = construct_mask
        clean = prepared.owner_fault_free
        valid_candidate = key_valid
        admitted = prepared.admissible
        accept = active & clean & admitted & valid_candidate
        reject = active & clean & ~admitted
        # A due row whose prior shot still owns the single lifecycle slot is a
        # real resource deferral.  Readiness is not: swallowing the task here
        # would turn balance -> mimic into an implicit serial stage.
        defer = due_mask & ~construct_mask
        censor = due_mask & active & (
            ~clean | (admitted & ~valid_candidate)
        )
        return _RowTransactionRecord(
            capability=token,
            candidate=candidate,
            prepared=prepared,
            preview=preview,
            due_mask=due_mask,
            construct_mask=construct_mask,
            accept_mask=accept.contiguous(),
            reject_mask=reject.contiguous(),
            defer_mask=defer.contiguous(),
            censor_mask=censor.contiguous(),
            candidate_consumed=False,
            accepted_consumers=set(),
            stage="prepared",
        )

    def require_owned_action_epoch_candidate(self, token: object) -> object:
        """Consume the one D05-private candidate projection for ActionEpoch."""

        record = self._row_transaction_records.get(token)
        if (
            type(token) is not DeviceR05RowTransaction
            or token is not self._active_row_transaction
            or record is None
            or record.capability is not token
            or record.stage != "settling"
            or record.candidate_consumed
        ):
            raise DeviceR05ConflictError("row transaction is stale or foreign")
        record.candidate_consumed = True
        candidate = record.candidate
        epoch = _require_action_epoch_module()
        return epoch.ActionEpochD05CandidateProjection(
            identity=candidate.identity.clone(),
            clocks=candidate.clocks.clone(),
            task=candidate.task.clone(),
            rng_counter=candidate.rng_counter.clone(),
            construction_admissible=candidate.construction_admissible.clone(),
            playback_admissible=candidate.playback_admissible.clone(),
            owner_fault_bits=candidate.owner_fault_bits.clone(),
        )

    def require_owned_action_epoch_accepted(
        self, token: object, *, owner_kind: str
    ) -> DeviceR05AcceptedRowsView:
        """Return a non-aliased full-N view only during one ACCEPT writer."""

        record = self._row_transaction_records.get(token)
        epoch_owner = self._diagnostic_epoch_owner
        if (
            type(token) is not DeviceR05RowTransaction
            or token is not self._active_row_transaction
            or record is None
            or record.stage != "settling"
            or owner_kind in record.accepted_consumers
            or epoch_owner is None
        ):
            raise DeviceR05ConflictError("accepted row view is stale or foreign")
        epoch_owner_kind = (
            "r05_runtime" if owner_kind == "physical_ball" else owner_kind
        )
        if owner_kind not in ("motion", "racket", "physical_ball"):
            raise DeviceR05ConflictError("accepted row consumer kind differs")
        accepted_rows = epoch_owner.require_active_d05_accepted_rows(
            token, owner_kind=epoch_owner_kind
        )
        epoch = _require_action_epoch_module()
        if type(accepted_rows) is not epoch.ActionEpochD05AcceptedRows:
            raise DeviceR05ConflictError("Epoch accepted row type differs")
        mask = _require_tensor(
            accepted_rows.accept_mask,
            label="action_epoch.accept_mask",
            device=self._device,
            dtype=torch.bool,
            shape=(self._num_envs, 1),
        ).clone()
        publication_ordinal = _require_tensor(
            accepted_rows.publication_ordinal,
            label="action_epoch.publication_ordinal",
            device=self._device,
            dtype=torch.int64,
            shape=(self._num_envs, 1),
        )
        record.accepted_consumers.add(owner_kind)
        candidate = record.candidate

        def masked_i64(value: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, value, torch.full_like(value, -1)).contiguous()

        key = epoch.ActionEpochShotKey(
            **{
                name: masked_i64(getattr(candidate.identity.shot_key, name))
                for name in epoch.ActionEpochShotKey.__dataclass_fields__
            }
        )
        identity = epoch.EpochIdentityPayload(
            shot_key=key,
            **{
                name: masked_i64(getattr(candidate.identity, name))
                for name in (
                    "scheduled_ordinal",
                    "target_generation",
                    "selected_cell",
                    "candidate_identity",
                )
            },
        )
        clocks = epoch.EpochClockPayload(
            **{
                name: masked_i64(getattr(candidate.clocks, name))
                for name in candidate.clocks.__dataclass_fields__
            }
        )
        task_mask = mask.unsqueeze(2)
        task = epoch.EpochTaskPayload(
            task_f32=torch.where(
                task_mask, candidate.task.task_f32, torch.zeros_like(candidate.task.task_f32)
            ).contiguous(),
            task_valid=(candidate.task.task_valid & mask).contiguous(),
        )
        target_xy_m = torch.where(
            mask.unsqueeze(2),
            record.prepared.selected_target_xy_m.reshape(
                self._num_envs, 1, 2
            ),
            torch.zeros(
                (self._num_envs, 1, 2),
                dtype=torch.float32,
                device=self._device,
            ),
        ).contiguous()
        return DeviceR05AcceptedRowsView(
            transaction=token,
            publication_ordinal=masked_i64(publication_ordinal),
            target_xy_m=target_xy_m,
            identity=identity,
            clocks=clocks,
            task=task,
            rng_counter=torch.where(
                mask, candidate.rng_counter, torch.zeros_like(candidate.rng_counter)
            ).contiguous(),
            playback_admissible=(
                candidate.playback_admissible & mask
            ).contiguous(),
        )

    def _finish_row_transaction(
        self,
        record: _RowTransactionRecord,
        *,
        accepted: torch.Tensor,
        retain_physical: bool = False,
    ) -> None:
        self._publish_action_epoch_afterimage(
            record.preview,
            accepted=accepted,
            settled_due=record.due_mask,
        )
        if retain_physical:
            method = getattr(
                self._diagnostic_physical_owner, "retain_action_epoch_launch", None
            )
            direct = getattr(
                type(self._diagnostic_physical_owner),
                "retain_action_epoch_launch",
                None,
            )
            if (
                not callable(method)
                or not callable(direct)
                or getattr(method, "__self__", None)
                is not self._diagnostic_physical_owner
                or getattr(method, "__func__", None) is not direct
            ):
                raise DeviceR05ConflictError("Physical row consumer is not bound")
            method(record.capability)
            if "physical_ball" not in record.accepted_consumers:
                raise DeviceR05ConflictError(
                    "Physical did not consume accepted rows"
                )
        record.stage = "settled"
        self._active_row_transaction = None

    def _commit_action_epoch_leaf_write(
        self, token: object, *, owner: object, owner_kind: str, method_name: str
    ) -> None:
        row = self._row_transaction_records.get(token)
        if self._diagnostic_epoch_owner is None or row is None:
            raise DeviceR05ConflictError("leaf callback is outside D05 settle")
        if self._active_diagnostic_epoch_leaf_writer is not None:
            raise DeviceR05ConflictError(
                "diagnostic epoch leaf callback is reordered or duplicated"
            )
        self._active_diagnostic_epoch_leaf_writer = owner_kind
        try:
            method = getattr(owner, method_name, None)
            direct = getattr(type(owner), method_name, None)
            if (
                not callable(method)
                or not callable(direct)
                or getattr(method, "__self__", None) is not owner
                or getattr(method, "__func__", None) is not direct
            ):
                raise DeviceR05ConflictError(owner_kind + " row writer is not bound")
            method(token)
            if owner_kind not in row.accepted_consumers:
                raise DeviceR05ConflictError(
                    owner_kind + " did not consume accepted rows"
                )
        finally:
            self._active_diagnostic_epoch_leaf_writer = None

    def _commit_action_epoch_motion_write(self, token: object) -> None:
        self._commit_action_epoch_leaf_write(
            token,
            owner=self._diagnostic_motion_owner,
            owner_kind="motion",
            method_name="commit_action_ball_full_mdp_motion_epoch_rows",
        )

    def _commit_action_epoch_racket_write(self, token: object) -> None:
        self._commit_action_epoch_leaf_write(
            token,
            owner=self._diagnostic_racket_owner,
            owner_kind="racket",
            method_name="commit_action_ball_full_mdp_racket_epoch_rows",
        )

    def _commit_action_epoch_r05_write(self, token: object) -> None:
        """Publish D05 last using only its active opaque row transaction."""

        epoch_owner = self._diagnostic_epoch_owner
        row = self._row_transaction_records.get(token)
        if epoch_owner is None or row is None:
            raise DeviceR05ConflictError("D05 epoch callback is outside settlement")
        if self._active_diagnostic_epoch_leaf_writer is not None:
            raise DeviceR05ConflictError(
                "diagnostic epoch leaf callback is reordered or duplicated"
            )
        self._active_diagnostic_epoch_leaf_writer = "r05_runtime"
        try:
            accepted_rows = epoch_owner.require_active_d05_accepted_rows(
                token, owner_kind="r05_runtime"
            )
            epoch = _require_action_epoch_module()
            if type(accepted_rows) is not epoch.ActionEpochD05AcceptedRows:
                raise DeviceR05ConflictError("Epoch accepted row type differs")
            _require_tensor(
                accepted_rows.publication_ordinal,
                label="action_epoch.publication_ordinal",
                device=self._device,
                dtype=torch.int64,
                shape=(self._num_envs, 1),
            )
            accepted = _require_tensor(
                accepted_rows.accept_mask,
                label="action_epoch.accept_mask",
                device=self._device,
                dtype=torch.bool,
                shape=(self._num_envs, 1),
            )[:, 0].clone()
            self._finish_row_transaction(
                row, accepted=accepted, retain_physical=True
            )
        finally:
            self._active_diagnostic_epoch_leaf_writer = None

    def stage_terminal(
        self,
        token: DeviceR05PreviewToken,
        boundary_receipt: object,
    ) -> DeviceR05TerminalClaim | DeviceR05ConstructionRejection:
        """Consume the sole global boundary fact and stage its exact outcome."""

        self._enter_public_operation()
        preview = self._require_preview(token)
        prepared = preview.prepared
        try:
            projection = self._call_authority(
                self._reveal_boundary_authority.project_owned_r05_reveal_boundary,
                token,
                boundary_receipt,
            )
            if type(projection) is DeviceRevealBoundaryProjection:
                boundary_preview_identity = projection.preview_identity
                construction_admissible = projection.construction_admissible
                owner_fault_present = projection.owner_fault_present
                transfer_sequence = projection.transfer_sequence
                decision = projection.decision
                primary_fault_source = projection.primary_fault
            else:
                boundary_preview_identity = None
                construction_admissible = None
                owner_fault_present = None
                transfer_sequence = None
                decision = None
                primary_fault_source = None
            if (
                type(projection) is not DeviceRevealBoundaryProjection
                or boundary_preview_identity is not preview.preview_identity
                or type(construction_admissible) is not bool
                or type(owner_fault_present) is not bool
                or type(transfer_sequence) is not int
                or transfer_sequence > _I64_MAX
                or transfer_sequence <= self._last_transfer_sequence
                or decision
                not in (
                    DECISION_ACCEPT,
                    DECISION_CENSOR,
                    DECISION_CONSTRUCTION_REJECT,
                )
            ):
                raise DeviceR05ConflictError("reveal boundary projection differs")
            fault = _require_tensor(
                primary_fault_source,
                label="primary_fault",
                device=self._device,
                dtype=torch.int64,
                shape=(prepared.projection.selected_count,),
            ).clone()
            if construction_admissible != (
                decision != DECISION_CONSTRUCTION_REJECT
            ):
                raise DeviceR05ConflictError(
                    "construction decision and admissibility differ"
                )
            if owner_fault_present and decision != DECISION_CENSOR:
                raise DeviceR05ConflictError(
                    "owner fault and typed CENSOR decision differ"
                )
        except Exception as exc:
            preview.stage = "failed_after_transfer"
            prepared.stage = "failed_after_transfer"
            self._active = None
            decoded_fault = locals().get("fault")
            self._settle_failed_reveal(
                prepared,
                poison_reason=2,
                transfer_sequence=(
                    transfer_sequence
                    if type(decoded_fault) is torch.Tensor
                    else 0
                ),
                decision=(
                    decision if type(decoded_fault) is torch.Tensor else 0
                ),
                primary_fault=(
                    decoded_fault if type(decoded_fault) is torch.Tensor else None
                ),
            )
            raise DeviceR05PoisonedError(
                "packed reveal boundary failed and cannot retry"
            ) from exc

        self._last_transfer_sequence = transfer_sequence
        if decision == DECISION_CONSTRUCTION_REJECT:
            # A clean recurring question draw is an attempted sample even when
            # every candidate is ordinarily infeasible.  Publish only D05's
            # reserved RNG after-image here; target generation and previous
            # cell belong to an installed target and therefore do not move.
            # Producer/counter faults retained ``rng_after == rng_before`` at
            # prepare time, so the same code is a byte-preserving no-op for a
            # censored source.
            rejection_reason = torch.where(
                prepared.admissible,
                torch.full_like(
                    prepared.reason, REASON_BATCH_PEER_INFEASIBLE
                ),
                prepared.reason,
            )
            row = self._build_reveal_journal_afterimage(
                prepared,
                operation=JOURNAL_CONSTRUCTION_REJECT,
                decision=DECISION_CONSTRUCTION_REJECT,
                transfer_sequence=transfer_sequence,
                primary_fault=fault,
                state_committed=False,
                reason=rejection_reason,
            )
            row["rng_after_lo"][prepared.selected_index] = torch.where(
                prepared.rng_advance_mask,
                prepared.rng_after_lo,
                prepared.rng_before_lo,
            )
            row["rng_after_hi"][prepared.selected_index] = torch.where(
                prepared.rng_advance_mask,
                prepared.rng_after_hi,
                prepared.rng_before_hi,
            )
            row["draw_after"][prepared.selected_index] = torch.where(
                prepared.rng_advance_mask,
                prepared.draw_after,
                prepared.draw_before,
            )
            after = self._publication_afterimage()
            after.live["_rng_lo"].index_copy_(
                0, prepared.selected_index, prepared.rng_after_lo
            )
            after.live["_rng_hi"].index_copy_(
                0, prepared.selected_index, prepared.rng_after_hi
            )
            after.live["_draw_count"].index_copy_(
                0, prepared.selected_index, prepared.draw_after
            )
            self._install_journal_row_afterimage(
                after, slot=prepared.journal_slot, row=row
            )
            after.counters["_journal_head"] = self._journal_head + 1
            after.counters["_active"] = None
            if prepared.internal_question:
                after.counters["_last_candidate_bank_sequence"] = (
                    prepared.question_projection.bank_sequence
                )
                after.counters["_next_candidate_identity"] = (
                    prepared.candidate_identity_highwater_after
                )
            rejection = object.__new__(DeviceR05ConstructionRejection)
            preview.stage = "construction_rejected"
            prepared.stage = "construction_rejected"
            self._active = None
            if preview.pretransfer_token is not None:
                after.registries["_pretransfer_records"].pop(
                    preview.pretransfer_token, None
                )
            self._publish(after)
            self._preview_records.pop(token, None)
            self._prepared_records.pop(prepared.capability, None)
            return rejection

        claim = object.__new__(DeviceR05TerminalClaim)
        self._claim_records[claim] = _ClaimRecord(
            capability=claim,
            preview=preview,
            decision=decision,
            owner_fault_present=owner_fault_present,
            primary_fault=fault,
            transfer_sequence=transfer_sequence,
            claim_identity=claim,
            stage="staged",
        )
        preview.stage = "staged"
        self._active = claim
        return claim

    def require_owned_terminal_claim_for_child(
        self,
        claim: DeviceR05TerminalClaim,
        *,
        owner_kind: str,
        expected_prepared_reveal: DeviceR05PreviewToken,
    ) -> DeviceR05TerminalClaimProjection:
        """Return the boundary-authenticated decision for one fixed leaf."""

        self._enter_public_operation()
        record = self._claim_records.get(claim)
        if (
            type(claim) is not DeviceR05TerminalClaim
            or claim is not self._active
            or record is None
            or record.capability is not claim
            or record.stage != "staged"
            or owner_kind not in CHILD_OWNER_ORDER
            or record.preview.capability is not expected_prepared_reveal
        ):
            raise DeviceR05ConflictError("terminal claim is stale or foreign")
        return DeviceR05TerminalClaimProjection(
            terminal_claim=claim,
            owner_kind=owner_kind,
            claim_identity=record.claim_identity,
            preview_identity=record.preview.preview_identity,
            decision=record.decision,
            primary_fault=record.primary_fault.clone(),
        )

    def arm_terminal(
        self, claim: DeviceR05TerminalClaim, arm_receipt: object
    ) -> DeviceR05ArmedTerminal:
        self._enter_public_operation()
        claim_record = self._claim_records.get(claim)
        if (
            type(claim) is not DeviceR05TerminalClaim
            or claim is not self._active
            or claim_record is None
            or claim_record.capability is not claim
            or claim_record.stage != "staged"
        ):
            raise DeviceR05ConflictError("terminal claim is not active")
        try:
            projection = self._call_authority(
                self._reveal_boundary_authority.require_owned_r05_terminal_arm,
                claim,
                arm_receipt,
            )
            if type(projection) is DeviceTerminalArmProjection:
                arm_claim_identity = projection.claim_identity
                arm_decision = projection.decision
                arm_child_kinds = projection.child_kinds
                child_arm_identities = projection.child_arm_identities
            else:
                arm_claim_identity = None
                arm_decision = None
                arm_child_kinds = None
                child_arm_identities = None
            if (
                type(projection) is not DeviceTerminalArmProjection
                or arm_claim_identity is not claim_record.claim_identity
                or arm_decision != claim_record.decision
                or arm_child_kinds != CHILD_OWNER_ORDER
                or type(child_arm_identities) is not tuple
                or len(child_arm_identities) != len(CHILD_OWNER_ORDER)
                or len({id(value) for value in child_arm_identities})
                != len(CHILD_OWNER_ORDER)
            ):
                raise DeviceR05ConflictError("terminal child arm proof differs")
        except Exception as exc:
            claim_record.stage = "arm_failed"
            claim_record.preview.stage = "arm_failed"
            claim_record.preview.prepared.stage = "arm_failed"
            self._active = None
            self._settle_failed_reveal(
                claim_record.preview.prepared,
                poison_reason=8,
                transfer_sequence=claim_record.transfer_sequence,
                decision=claim_record.decision,
                primary_fault=claim_record.primary_fault,
            )
            raise DeviceR05PoisonedError(
                "terminal child arm proof failed after transfer"
            ) from exc
        armed = object.__new__(DeviceR05ArmedTerminal)
        self._armed_records[armed] = _ArmedRecord(
            capability=armed,
            claim=claim_record,
            armed_identity=armed,
            stage="armed",
        )
        claim_record.stage = "armed"
        self._active = armed
        return armed

    def commit_terminal(
        self,
        armed: DeviceR05ArmedTerminal,
        commit_receipt: object,
    ) -> DeviceR05TerminalReceipt:
        """Publish a prevalidated device after-image; do no encoding or D2H."""

        self._enter_public_operation()
        armed_record = self._armed_records.get(armed)
        if (
            type(armed) is not DeviceR05ArmedTerminal
            or armed is not self._active
            or armed_record is None
            or armed_record.capability is not armed
            or armed_record.stage != "armed"
        ):
            raise DeviceR05ConflictError("armed terminal is not active")
        try:
            commit_projection = self._call_authority(
                self._reveal_boundary_authority.require_owned_r05_terminal_commit,
                armed,
                commit_receipt,
            )
            if type(commit_projection) is DeviceTerminalCommitProjection:
                commit_armed_identity = commit_projection.armed_identity
                commit_claim_identity = commit_projection.claim_identity
                commit_decision = commit_projection.decision
                commit_child_kinds = commit_projection.child_kinds
                child_commit_identities = (
                    commit_projection.child_commit_identities
                )
            else:
                commit_armed_identity = None
                commit_claim_identity = None
                commit_decision = None
                commit_child_kinds = None
                child_commit_identities = None
            if (
                type(commit_projection) is not DeviceTerminalCommitProjection
                or commit_armed_identity is not armed_record.armed_identity
                or commit_claim_identity is not armed_record.claim.claim_identity
                or commit_decision != armed_record.claim.decision
                or commit_child_kinds != CHILD_OWNER_ORDER
                or type(child_commit_identities) is not tuple
                or len(child_commit_identities)
                != len(CHILD_OWNER_ORDER)
                or len(
                    {id(value) for value in child_commit_identities}
                )
                != len(CHILD_OWNER_ORDER)
            ):
                raise DeviceR05ConflictError("terminal child commit proof differs")
        except Exception as exc:
            armed_record.stage = "commit_proof_failed"
            armed_record.claim.stage = "commit_proof_failed"
            self._active = None
            self._settle_failed_reveal(
                armed_record.claim.preview.prepared,
                poison_reason=10,
                transfer_sequence=armed_record.claim.transfer_sequence,
                decision=armed_record.claim.decision,
                primary_fault=armed_record.claim.primary_fault,
            )
            raise DeviceR05PoisonedError(
                "terminal child commit proof failed after child commits"
            ) from exc
        claim_record = armed_record.claim
        prepared = claim_record.preview.prepared
        projection = prepared.projection
        index = prepared.selected_index
        mutation_before = self._mutation_version.clone()
        if self._mutation_version_host >= _I64_MAX:
            armed_record.stage = "mutation_exhausted"
            claim_record.stage = "mutation_exhausted"
            prepared.stage = "mutation_exhausted"
            self._active = None
            self._settle_failed_reveal(
                prepared,
                poison_reason=16,
                transfer_sequence=claim_record.transfer_sequence,
                decision=claim_record.decision,
                primary_fault=claim_record.primary_fault,
            )
            raise DeviceR05PoisonedError(
                "terminal publication exhausted mutation chronology"
            )
        owner_was_clean = ~self._poisoned.clone()
        mutation_room = mutation_before != _I64_MAX
        boundary_fault_free = (
            torch.zeros_like(prepared.owner_fault_free)
            if claim_record.owner_fault_present
            else claim_record.primary_fault.eq(0)
        )
        row_fault_free = torch.logical_and(
            prepared.owner_fault_free, boundary_fault_free
        )
        settlement_mask = torch.logical_and(
            owner_was_clean.expand_as(row_fault_free),
            mutation_room.expand_as(row_fault_free),
        )
        sampler_advance_mask = torch.logical_and(
            settlement_mask,
            row_fault_free,
        )
        rng_advance_mask = torch.logical_and(
            settlement_mask,
            prepared.rng_after_lo.ne(prepared.rng_before_lo)
            | prepared.rng_after_hi.ne(prepared.rng_before_hi),
        )
        policy_accept_mask = torch.logical_and(
            sampler_advance_mask,
            (
                torch.ones_like(row_fault_free)
                if claim_record.decision == DECISION_ACCEPT
                else torch.zeros_like(row_fault_free)
            ),
        )

        try:
            after = self._publication_afterimage()
            after.live["_rng_lo"].index_copy_(
                0,
                index,
                torch.where(
                    rng_advance_mask,
                    prepared.rng_after_lo,
                    prepared.rng_before_lo,
                ),
            )
            after.live["_rng_hi"].index_copy_(
                0,
                index,
                torch.where(
                    rng_advance_mask,
                    prepared.rng_after_hi,
                    prepared.rng_before_hi,
                ),
            )
            after.live["_draw_count"].index_copy_(
                0,
                index,
                torch.where(
                    rng_advance_mask,
                    prepared.draw_after,
                    prepared.draw_before,
                ),
            )
            after.live["_target_generation"].index_copy_(
                0,
                index,
                torch.where(
                    sampler_advance_mask,
                    prepared.generation_after,
                    prepared.generation_before,
                ),
            )
            after.live["_previous_cell_index"].index_copy_(
                0,
                index,
                torch.where(
                    sampler_advance_mask,
                    prepared.selected_cell,
                    prepared.previous_before,
                ),
            )
            after.live["_scheduled_ordinal"].index_copy_(
                0,
                index,
                torch.where(
                    settlement_mask,
                    projection.scheduled_ordinal,
                    prepared.ordinal_before,
                ),
            )
            after.live["_outcome_shot_index"].index_copy_(
                0,
                index,
                torch.where(
                    settlement_mask,
                    projection.outcome_shot_index,
                    prepared.outcome_before,
                ),
            )
            current_sequence = self._sequence_kind[index]
            after.live["_sequence_kind"].index_copy_(
                0,
                index,
                torch.where(
                    settlement_mask,
                    torch.where(
                        policy_accept_mask,
                        torch.full_like(current_sequence, SEQUENCE_COMMITTED),
                        torch.full_like(
                            current_sequence, SEQUENCE_INFRA_CENSORED
                        ),
                    ),
                    current_sequence,
                ),
            )
            for destination, source in (
                (after.live["_task_identity"], projection.task_identity),
                (
                    after.live["_outcome_identity"],
                    prepared.reserved_outcome_identity,
                ),
                (
                    after.live["_ball_identity"],
                    prepared.reserved_ball_identity,
                ),
            ):
                current = destination[index]
                destination.index_copy_(
                    0,
                    index,
                    torch.where(
                        settlement_mask,
                        torch.where(
                            policy_accept_mask,
                            source,
                            torch.full_like(source, -1),
                        ),
                        current,
                    ),
                )
            if claim_record.decision == DECISION_ACCEPT:
                after.live["_next_outcome_identity"] = (
                    prepared.outcome_identity_highwater_before
                    + prepared.identity_advance_count
                )
                after.live["_next_ball_identity"] = (
                    prepared.ball_identity_highwater_before
                    + prepared.identity_advance_count
                )
            current_opportunity = self._policy_opportunity[index]
            after.live["_policy_opportunity"].index_copy_(
                0,
                index,
                torch.where(
                    settlement_mask,
                    policy_accept_mask,
                    current_opportunity,
                ),
            )
            if claim_record.decision == DECISION_ACCEPT:
                operation = JOURNAL_ACCEPT
            else:
                operation = JOURNAL_CENSOR
            mutation_after = torch.where(
                mutation_room,
                mutation_before + mutation_room.to(torch.int64),
                mutation_before,
            )
            after.live["_mutation_version"] = mutation_after
            after.counters["_mutation_version_host"] = (
                self._mutation_version_host + 1
            )
            device_fault = torch.logical_or(
                torch.logical_and(
                    torch.tensor(
                        claim_record.decision == DECISION_ACCEPT,
                        dtype=torch.bool,
                        device=self._device,
                    ),
                    ~torch.all(row_fault_free),
                ),
                ~mutation_room,
            )
            after.live["_poisoned"].logical_or_(device_fault)
            after.live["_poison_reason"].copy_(
                torch.where(
                    torch.logical_and(
                        device_fault,
                        after.live["_poison_reason"].eq(0),
                    ),
                    torch.full_like(self._poison_reason, 16),
                    after.live["_poison_reason"],
                )
            )
            row = self._build_reveal_journal_afterimage(
                prepared,
                operation=operation,
                decision=claim_record.decision,
                transfer_sequence=claim_record.transfer_sequence,
                primary_fault=claim_record.primary_fault,
                state_committed=True,
                reason=prepared.reason,
                mutation_before=mutation_before,
                mutation_after=mutation_after,
                committed_mask=sampler_advance_mask,
            )
            receipt = object.__new__(DeviceR05TerminalReceipt)
            after.registries["_terminal_receipts"][receipt] = _TerminalRecord(
                receipt=receipt,
                terminal_identity=receipt,
                epoch=prepared.epoch,
                journal_slot=prepared.journal_slot,
                decision=claim_record.decision,
                selected_count=prepared.projection.selected_count,
                journal_sequence=self._journal_head,
                prepared_reveal=claim_record.preview.capability,
                completed_children=set(),
            )
            after.registries["_armed_records"].pop(armed, None)
            after.registries["_claim_records"].pop(
                claim_record.capability, None
            )
            after.registries["_preview_records"].pop(
                claim_record.preview.capability, None
            )
            if claim_record.preview.pretransfer_token is not None:
                after.registries["_pretransfer_records"].pop(
                    claim_record.preview.pretransfer_token, None
                )
            after.registries["_prepared_records"].pop(
                prepared.capability, None
            )
            after.counters["_active"] = None
            after.counters["_journal_head"] = self._journal_head + 1
            self._install_journal_row_afterimage(
                after, slot=prepared.journal_slot, row=row
            )
        except Exception as exc:
            armed_record.stage = "commit_write_failed"
            claim_record.stage = "commit_write_failed"
            prepared.stage = "commit_write_failed"
            self._active = None
            self._settle_failed_reveal(
                prepared,
                poison_reason=15,
                transfer_sequence=claim_record.transfer_sequence,
                decision=claim_record.decision,
                primary_fault=claim_record.primary_fault,
            )
            raise DeviceR05PoisonedError(
                "terminal publication failed after child commits"
            ) from exc
        self._publish(after)
        if not self._action_epoch_is_single_business_log:
            self._checkpoint_requires_global_drain_ack = True
        armed_record.stage = "committed"
        claim_record.stage = "committed"
        claim_record.preview.stage = "committed"
        prepared.stage = "committed"
        return receipt

    def _publish_action_epoch_afterimage(
        self,
        preview: _PreviewRecord,
        *,
        accepted: torch.Tensor,
        settled_due: torch.Tensor,
    ) -> None:
        """Publish cadence settlement separately from ACCEPT task state."""

        prepared = preview.prepared
        n = self._num_envs
        accept = _require_tensor(
            accepted,
            label="diagnostic_epoch.accepted",
            device=self._device,
            dtype=torch.bool,
            shape=(n,),
        )
        due = _require_tensor(
            settled_due,
            label="diagnostic_epoch.settled_due",
            device=self._device,
            dtype=torch.bool,
            shape=(n,),
        )
        # ``accept`` and ``due`` have the same exact ActionEpoch writer:
        # ActionEpoch constructs candidates only from ``due & available`` and
        # its ACCEPT predicate begins with those constructed rows.  Rechecking
        # that theorem here was an anonymous CUDA-poisoning writer echo, not an
        # independent trust boundary.  The writer-side rowwise regression is
        # the executable contract; this consumer only applies its afterimage.
        rng_advance = prepared.rng_advance_mask
        try:
            after = self._publication_afterimage()
            for name, committed, before in (
                ("_rng_lo", prepared.rng_after_lo, prepared.rng_before_lo),
                ("_rng_hi", prepared.rng_after_hi, prepared.rng_before_hi),
                ("_draw_count", prepared.draw_after, prepared.draw_before),
            ):
                after.live[name] = torch.where(
                    rng_advance, committed, before
                ).contiguous()
            for name, committed, before, mask in (
                (
                    "_target_generation",
                    prepared.generation_after,
                    prepared.generation_before,
                    accept,
                ),
                (
                    "_previous_cell_index",
                    prepared.selected_cell,
                    prepared.previous_before,
                    accept,
                ),
                (
                    "_scheduled_ordinal",
                    prepared.projection.scheduled_ordinal,
                    prepared.ordinal_before,
                    due,
                ),
                (
                    "_outcome_shot_index",
                    prepared.projection.outcome_shot_index,
                    prepared.outcome_before,
                    due,
                ),
                (
                    "_task_identity",
                    prepared.projection.task_identity,
                    after.live["_task_identity"],
                    accept,
                ),
                (
                    "_outcome_identity",
                    prepared.reserved_outcome_identity,
                    after.live["_outcome_identity"],
                    accept,
                ),
                (
                    "_ball_identity",
                    prepared.reserved_ball_identity,
                    after.live["_ball_identity"],
                    accept,
                ),
            ):
                after.live[name] = torch.where(
                    mask, committed, before
                ).contiguous()
            after.live["_sequence_kind"] = torch.where(
                accept,
                torch.full_like(after.live["_sequence_kind"], SEQUENCE_COMMITTED),
                after.live["_sequence_kind"],
            ).contiguous()
            after.live["_policy_opportunity"] = torch.where(
                accept,
                torch.ones_like(after.live["_policy_opportunity"]),
                after.live["_policy_opportunity"],
            ).contiguous()
            after.live["_next_outcome_identity"] = (
                prepared.outcome_identity_highwater_before
                + prepared.identity_advance_count
            )
            after.live["_next_ball_identity"] = (
                prepared.ball_identity_highwater_before
                + prepared.identity_advance_count
            )
            after.registries["_preview_records"].pop(preview.capability, None)
            after.registries["_prepared_records"].pop(
                prepared.capability, None
            )
            after.counters["_active"] = None
        except BaseException as exc:
            preview.stage = "d05_epoch_write_failed"
            prepared.stage = "d05_epoch_write_failed"
            self._active = None
            self._poison(25)
            raise DeviceR05PoisonedError(
                "D05 publication failed after child commits"
            ) from exc
        self._publish(after)
        preview.stage = "committed"
        prepared.stage = "committed"

    def _finish_reveal_journal(
        self,
        prepared: _PreparedRecord,
        *,
        operation: int,
        decision: int,
        transfer_sequence: int,
        primary_fault: torch.Tensor,
        state_committed: bool,
        reason: torch.Tensor,
        mutation_before: Optional[torch.Tensor] = None,
        mutation_after: Optional[torch.Tensor] = None,
        committed_mask: Optional[torch.Tensor] = None,
    ) -> None:
        row = self._build_reveal_journal_afterimage(
            prepared,
            operation=operation,
            decision=decision,
            transfer_sequence=transfer_sequence,
            primary_fault=primary_fault,
            state_committed=state_committed,
            reason=reason,
            mutation_before=mutation_before,
            mutation_after=mutation_after,
            committed_mask=committed_mask,
        )
        after = self._publication_afterimage()
        if prepared.internal_question:
            index = prepared.selected_index
            for name, committed, before in (
                ("_rng_lo", prepared.rng_after_lo, prepared.rng_before_lo),
                ("_rng_hi", prepared.rng_after_hi, prepared.rng_before_hi),
                ("_draw_count", prepared.draw_after, prepared.draw_before),
            ):
                after.live[name].index_copy_(
                    0,
                    index,
                    torch.where(
                        prepared.rng_advance_mask,
                        committed,
                        before,
                    ),
                )
            row["rng_after_lo"][index] = torch.where(
                prepared.rng_advance_mask,
                prepared.rng_after_lo,
                prepared.rng_before_lo,
            )
            row["rng_after_hi"][index] = torch.where(
                prepared.rng_advance_mask,
                prepared.rng_after_hi,
                prepared.rng_before_hi,
            )
            row["draw_after"][index] = torch.where(
                prepared.rng_advance_mask,
                prepared.draw_after,
                prepared.draw_before,
            )
        self._install_journal_row_afterimage(
            after, slot=prepared.journal_slot, row=row
        )
        after.counters["_journal_head"] = self._journal_head + 1
        self._publish(after)
        if self._action_epoch_is_single_business_log:
            self._retire_action_epoch_shadow_journal()
        else:
            self._checkpoint_requires_global_drain_ack = True

    def _build_reveal_journal_afterimage(
        self,
        prepared: _PreparedRecord,
        *,
        operation: int,
        decision: int,
        transfer_sequence: int,
        primary_fault: torch.Tensor,
        state_committed: bool,
        reason: torch.Tensor,
        mutation_before: Optional[torch.Tensor] = None,
        mutation_after: Optional[torch.Tensor] = None,
        committed_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Build a complete immutable row without touching published state."""

        slot = prepared.journal_slot
        index = prepared.selected_index
        row = self._journal_row_afterimage(slot)
        before_mutation = (
            self._mutation_version if mutation_before is None else mutation_before
        )
        after_mutation = (
            self._mutation_version if mutation_after is None else mutation_after
        )
        row_committed = (
            torch.ones_like(prepared.admissible)
            if state_committed and committed_mask is None
            else (
                _require_tensor(
                    committed_mask,
                    label="reveal_journal.committed_mask",
                    device=self._device,
                    dtype=torch.bool,
                    shape=(prepared.projection.selected_count,),
                )
                if state_committed
                else torch.zeros_like(prepared.admissible)
            )
        )
        meta = row["meta"]
        meta[self._META_SEQUENCE] = self._journal_head
        meta[self._META_EPOCH] = prepared.epoch
        meta[self._META_OPERATION] = operation
        meta[self._META_MUTATION_BEFORE] = before_mutation
        meta[self._META_MUTATION_AFTER] = after_mutation
        meta[self._META_SELECTED_COUNT] = prepared.projection.selected_count
        meta[self._META_ADMISSIBLE_COUNT] = torch.sum(
            row_committed, dtype=torch.int64
        )
        meta[self._META_TRANSFER_SEQUENCE] = transfer_sequence
        meta[self._META_DECISION] = decision
        row["selected"][index] = True
        row["feasible"][index] = prepared.feasible
        row["construction_reason"][index] = prepared.bank_construction_reason
        row["candidate_identity"][index] = prepared.bank_candidate_identity
        retained_rounds = prepared.round_construction_reason.shape[1]
        row["round_construction_reason"][
            index, :retained_rounds
        ] = prepared.round_construction_reason
        row["round_candidate_identity"][
            index, :retained_rounds
        ] = prepared.round_candidate_identity
        row["round_producer_fault"][
            index, :retained_rounds
        ] = prepared.round_producer_fault
        row["chosen_round"][index] = prepared.chosen_round
        row["rounds_attempted"][index] = prepared.rounds_attempted
        row["reason"][index] = reason
        row["rng_before_lo"][index] = prepared.rng_before_lo
        row["rng_before_hi"][index] = prepared.rng_before_hi
        row["rng_after_lo"][index] = (
            torch.where(row_committed, prepared.rng_after_lo, prepared.rng_before_lo)
        )
        row["rng_after_hi"][index] = (
            torch.where(row_committed, prepared.rng_after_hi, prepared.rng_before_hi)
        )
        row["draw_before"][index] = prepared.draw_before
        row["draw_after"][index] = (
            torch.where(row_committed, prepared.draw_after, prepared.draw_before)
        )
        row["sampler_generation_before"][index] = prepared.generation_before
        row["sampler_generation_after"][index] = (
            torch.where(
                row_committed,
                prepared.generation_after,
                prepared.generation_before,
            )
        )
        row["previous_before"][index] = prepared.previous_before
        row["previous_after"][index] = (
            torch.where(
                row_committed, prepared.selected_cell, prepared.previous_before
            )
        )
        row["reset_before"][index] = prepared.reset_before
        row["reset_after"][index] = torch.where(
            row_committed,
            prepared.projection.reset_generation,
            prepared.reset_before,
        )
        row["ordinal_before"][index] = prepared.ordinal_before
        row["ordinal_after"][index] = (
            torch.where(
                row_committed,
                prepared.projection.scheduled_ordinal,
                prepared.ordinal_before,
            )
        )
        row["outcome_before"][index] = prepared.outcome_before
        row["outcome_after"][index] = (
            torch.where(
                row_committed,
                prepared.projection.outcome_shot_index,
                prepared.outcome_before,
            )
        )
        row["candidate_bank_sequence"][index] = (
            prepared.question_projection.bank_sequence
        )
        if state_committed:
            row["selected_cell"][index] = torch.where(
                row_committed,
                prepared.selected_cell,
                torch.full_like(prepared.selected_cell, -1),
            )
            row["selected_candidate_identity"][index] = torch.where(
                row_committed,
                prepared.selected_candidate_identity,
                torch.full_like(prepared.selected_candidate_identity, -1),
            )
        row["task_identity"][index] = prepared.projection.task_identity
        identity_committed = torch.logical_and(
            row_committed,
            torch.full_like(row_committed, decision == DECISION_ACCEPT),
        )
        row["outcome_identity"][index] = torch.where(
            identity_committed,
            prepared.reserved_outcome_identity,
            torch.full_like(prepared.reserved_outcome_identity, -1),
        )
        row["ball_identity"][index] = torch.where(
            identity_committed,
            prepared.reserved_ball_identity,
            torch.full_like(prepared.reserved_ball_identity, -1),
        )
        row["cadence_identity"][index] = prepared.projection.cadence_identity
        row["primary_fault"][index] = primary_fault
        row["question_producer_fault"][index] = (
            prepared.question_producer_fault
        )
        row["settlement"].zero_()
        return row

    def record_child_completion(
        self,
        terminal: DeviceR05TerminalReceipt,
        *,
        child_kind: str,
        child_receipt: object,
    ) -> None:
        """Record one independently issued child completion identity."""

        self._enter_public_operation()
        self._close_construction_window()
        if self._active is not None or self._active_drain is not None:
            raise DeviceR05ConflictError(
                "child completion cannot overlap another R05 operation"
            )
        record = self._terminal_receipts.get(terminal)
        if (
            type(terminal) is not DeviceR05TerminalReceipt
            or record is None
            or record.receipt is not terminal
            or child_kind not in CHILD_OWNER_ORDER
            or child_kind in record.completed_children
        ):
            raise DeviceR05ConflictError("terminal child completion is stale")
        child_index = CHILD_OWNER_ORDER.index(child_kind)
        authority = self._child_completion_authorities[child_index]
        try:
            projection = self._call_authority(
                authority.require_owned_r05_child_completion,
                child_receipt,
            )
            if (
                type(projection) is not DeviceChildCompletionProjection
                or projection.terminal_identity is not record.terminal_identity
                or projection.child_kind != child_kind
                or projection.decision != record.decision
            ):
                raise DeviceR05ConflictError("child completion projection differs")
        except Exception as exc:
            self._poison(3)
            raise DeviceR05PoisonedError(
                "child completion failed after R05 publication"
            ) from exc
        after = self._publication_afterimage()
        after.registries["_terminal_receipts"][terminal] = _TerminalRecord(
            receipt=record.receipt,
            terminal_identity=record.terminal_identity,
            epoch=record.epoch,
            journal_slot=record.journal_slot,
            decision=record.decision,
            selected_count=record.selected_count,
            journal_sequence=record.journal_sequence,
            prepared_reveal=record.prepared_reveal,
            completed_children={*record.completed_children, child_kind},
        )
        row = dict(after.journal_rows[record.journal_slot])
        row["child_completion"] = row["child_completion"].clone()
        row["child_completion"][child_index] = True
        after.journal_rows[record.journal_slot] = row
        self._publish(after)

    def require_owned_terminal_receipt_for_child(
        self,
        terminal: DeviceR05TerminalReceipt,
        *,
        owner_kind: str,
        expected_prepared_reveal: DeviceR05PreviewToken,
    ) -> DeviceR05TerminalReceiptProjection:
        """Return the exact R05-last identity; it is not a child-complete fact."""

        self._enter_public_operation()
        self._close_construction_window()
        record = self._terminal_receipts.get(terminal)
        if (
            type(terminal) is not DeviceR05TerminalReceipt
            or record is None
            or record.receipt is not terminal
            or owner_kind not in CHILD_OWNER_ORDER
            or record.prepared_reveal is not expected_prepared_reveal
        ):
            raise DeviceR05ConflictError("terminal receipt is stale or foreign")
        return DeviceR05TerminalReceiptProjection(
            terminal_receipt=terminal,
            owner_kind=owner_kind,
            terminal_identity=record.terminal_identity,
            decision=record.decision,
            journal_sequence=record.journal_sequence,
        )

    def prepare_true_reset_many(
        self, reset_event_receipt: object
    ) -> DeviceR05PreparedTrueReset:
        """Prepare one owner-authoritative selected reset with no live write."""

        self._require_idle()
        # This host mirror is owned by Device-R05, so exhaustion can be
        # rejected synchronously before the authority callback, journal
        # reservation, capability mint, or any child method.  It must never be
        # deferred to the post-child commit path.
        if self._mutation_version_host >= _I64_MAX:
            raise DeviceR05PoisonedError(
                "true-reset publication exhausted mutation chronology"
            )
        if self._epoch >= _I64_MAX or self._journal_head >= _I64_MAX:
            self._poison(13)
            raise DeviceR05PoisonedError("device R05 chronology exhausted int64")
        projection = self._call_authority(
            self._true_reset_authority.project_r05_true_reset,
            reset_event_receipt,
            device=self._device,
            num_envs=self._num_envs,
            live_reset_ledger_identity=self._live_reset_ledger_identity,
            live_reset_generation=self._reset_generation.clone(),
        )
        if type(projection) is not DeviceTrueResetEventProjection:
            raise DeviceR05Error("true-reset event projection type differs")
        reset_event_identity = projection.reset_event_identity
        raw_index = projection.selected_env_index
        raw_mask = projection.selected_mask
        if (
            type(raw_index) is not torch.Tensor
            or raw_index.ndim != 1
            or raw_index.shape[0] < 1
        ):
            raise DeviceR05Error("true-reset selection is empty or not rank-1")
        k = raw_index.shape[0]
        index = _require_tensor(
            raw_index,
            label="true_reset.selected_env_index",
            device=self._device,
            dtype=torch.int64,
            shape=(k,),
        ).clone()
        mask = _require_tensor(
            raw_mask,
            label="true_reset.selected_mask",
            device=self._device,
            dtype=torch.bool,
            shape=(self._num_envs,),
        ).clone()
        # The authority projection is not trusted yet.  Never feed raw indices
        # to a CUDA indexing kernel before the top owner has consumed its sole
        # packed host preflight: an out-of-range value would otherwise poison
        # the CUDA context instead of producing a clean fail-closed verdict.
        index_in_domain = torch.logical_and(index >= 0, index < self._num_envs)
        safe_index = torch.clamp(index, min=0, max=self._num_envs - 1)
        reconstructed_mask = torch.zeros_like(mask)
        reconstructed_mask.index_fill_(0, safe_index, True)
        selection_fault = torch.logical_or(
            torch.any(~index_in_domain),
            torch.logical_or(
                torch.any(index[1:] <= index[:-1]),
                torch.any(reconstructed_mask.logical_xor(mask)),
            ),
        )
        generation_before = self._reset_generation.clone()
        generation_overflow_fault = torch.logical_and(
            mask, generation_before == torch.iinfo(torch.int64).max
        )
        safe_increment = torch.logical_and(
            mask, ~generation_overflow_fault
        ).to(torch.int64)
        generation_after = generation_before + safe_increment
        slot = self._reserve_journal_slot()
        self._epoch += 1

        prepared = object.__new__(DeviceR05PreparedTrueReset)
        mutation_before = self._mutation_version.clone()
        record = _PreparedTrueResetRecord(
            capability=prepared,
            prepared_identity=prepared,
            reset_event_receipt=reset_event_receipt,
            reset_event_identity=reset_event_identity,
            # All later private/journal indexing uses only the sanitized
            # after-image.  The raw env index remains retained by the top/env
            # active record and is joined in the packed preflight.
            selected_index=safe_index,
            selected_mask=mask,
            generation_before=generation_before,
            generation_after=generation_after,
            generation_overflow_fault=generation_overflow_fault,
            mutation_before=mutation_before,
            mutation_after=torch.where(
                mutation_before == _I64_MAX,
                mutation_before,
                mutation_before + 1,
            ),
            scheduled_ordinal_before=self._scheduled_ordinal.clone(),
            outcome_shot_index_before=self._outcome_shot_index.clone(),
            sequence_kind_before=self._sequence_kind.clone(),
            task_identity_before=self._task_identity.clone(),
            outcome_identity_before=self._outcome_identity.clone(),
            ball_identity_before=self._ball_identity.clone(),
            policy_opportunity_before=self._policy_opportunity.clone(),
            device_fault=torch.logical_or(
                selection_fault,
                torch.logical_or(
                    torch.any(generation_overflow_fault),
                    mutation_before == _I64_MAX,
                ),
            ),
            journal_slot=slot,
            epoch=self._epoch,
            stage="prepared",
        )
        # Overflow remains device-resident and is resolved by the independent
        # reset authority at commit.  Until then the after-image never wraps.
        record.generation_after = torch.where(
            generation_overflow_fault,
            generation_before,
            generation_after,
        )
        self._prepared_true_resets[prepared] = record
        self._active = prepared
        return prepared

    def _require_prepared_true_reset(
        self, prepared: DeviceR05PreparedTrueReset
    ) -> _PreparedTrueResetRecord:
        record = self._prepared_true_resets.get(prepared)
        if (
            type(prepared) is not DeviceR05PreparedTrueReset
            or record is None
            or record.capability is not prepared
            or prepared is not self._active
            or record.stage != "prepared"
        ):
            raise DeviceR05ConflictError("prepared true reset is stale or foreign")
        return record

    def require_owned_prepared_true_reset(
        self,
        prepared: DeviceR05PreparedTrueReset,
        *,
        owner_kind: str,
    ) -> DeviceR05PreparedTrueResetProjection:
        """Return a clone-only leaf projection for one fixed child kind.

        ``owner_kind`` routes to a construction-fixed leaf lane; it is not
        treated as proof that the leaf committed.  That proof is required from
        ``DeviceTrueResetAuthority`` at the top-last commit.
        """

        self._enter_public_operation()
        owned = self._require_prepared_true_reset(prepared)
        if owner_kind not in CHILD_OWNER_ORDER:
            raise DeviceR05Error("true-reset leaf kind differs")
        return DeviceR05PreparedTrueResetProjection(
            prepared_true_reset=prepared,
            owner_kind=owner_kind,
            prepared_identity=owned.prepared_identity,
            reset_event_identity=owned.reset_event_identity,
            selected_mask=owned.selected_mask.clone(),
            generation_before=owned.generation_before.clone(),
            generation_after=owned.generation_after.clone(),
            generation_overflow_fault=owned.generation_overflow_fault.clone(),
            writer_fault=owned.device_fault.clone(),
        )

    def register_true_reset_preflight(
        self,
        prepared: DeviceR05PreparedTrueReset,
        preflight_capability: object,
    ) -> None:
        """Bind the top's sole packed-D2H edge before any child method."""

        self._enter_public_operation()
        owned = self._require_prepared_true_reset(prepared)
        if preflight_capability is None or owned.preflight_capability is not None:
            raise DeviceR05ConflictError(
                "true-reset packed preflight is absent or already registered"
            )
        try:
            callback = getattr(
                self._true_reset_authority,
                "require_owned_r05_true_reset_preflight",
                None,
            )
            if not callable(callback):
                raise DeviceR05ConflictError(
                    "true-reset authority lacks packed-preflight validation"
                )
            projection = self._call_authority(
                callback,
                prepared,
                preflight_capability=preflight_capability,
            )
            if (
                type(projection) is not DeviceTrueResetPreflightProjection
                or projection.prepared_true_reset is not prepared
                or projection.reset_event_identity is not owned.reset_event_identity
                or projection.preflight_capability is not preflight_capability
            ):
                raise DeviceR05ConflictError(
                    "true-reset packed preflight authority differs"
                )
        except Exception as exc:
            raise DeviceR05ConflictError(
                "true-reset packed preflight registration failed"
            ) from exc
        owned.preflight_capability = preflight_capability

    def abort_true_reset_many(
        self, prepared: DeviceR05PreparedTrueReset
    ) -> None:
        self._enter_public_operation()
        owned = self._require_prepared_true_reset(prepared)
        try:
            projection = self._call_authority(
                self._true_reset_authority.require_owned_r05_true_reset_abort,
                prepared,
            )
            if (
                type(projection) is not DeviceTrueResetAbortProjection
                or projection.prepared_true_reset is not prepared
                or projection.reset_event_identity is not owned.reset_event_identity
                or projection.child_commits_started is not False
            ):
                raise DeviceR05ConflictError("true-reset abort authority differs")
        except Exception as exc:
            owned.stage = "abort_failed"
            self._active = None
            self._settle_failed_true_reset(owned, poison_reason=4)
            raise DeviceR05PoisonedError(
                "true-reset abort could not prove zero child commits"
            ) from exc
        self._finish_true_reset_journal(owned, committed=False)
        owned.stage = "aborted"
        self._active = None
        self._prepared_true_resets.pop(prepared, None)

    def commit_true_reset_many(
        self, prepared: DeviceR05PreparedTrueReset
    ) -> DeviceR05TrueResetReceipt:
        """Commit Device-R05 last after four exact child commits."""

        self._enter_public_operation()
        owned = self._require_prepared_true_reset(prepared)
        # The sole packed D2H was consumed and registered before any child
        # method.  Never perform a second post-child host verdict here: the
        # exact active capability is rejoined through the construction-bound
        # top callback below.
        if owned.preflight_capability is None:
            owned.stage = "missing_preflight_after_children"
            self._active = None
            self._settle_failed_true_reset(owned, poison_reason=17)
            raise DeviceR05PoisonedError(
                "true-reset packed preflight was not registered before children"
            )
        try:
            owner_view = DeviceR05TrueResetCommitInput(
                prepared_true_reset=prepared,
                reset_event_identity=owned.reset_event_identity,
                selected_mask=owned.selected_mask.clone(),
                generation_before=owned.generation_before.clone(),
                generation_after=owned.generation_after.clone(),
                generation_overflow_fault=(
                    owned.generation_overflow_fault.clone()
                ),
            )
            projection = self._call_authority(
                self._true_reset_authority.require_owned_r05_true_reset_commit,
                prepared,
                owner_view=owner_view,
            )
            if type(projection) is DeviceTrueResetCommitProjection:
                reset_prepared = projection.prepared_true_reset
                reset_event_identity = projection.reset_event_identity
                reset_child_kinds = projection.child_kinds
                reset_child_commit_identities = (
                    projection.child_commit_identities
                )
            else:
                reset_prepared = None
                reset_event_identity = None
                reset_child_kinds = None
                reset_child_commit_identities = None
            if (
                type(projection) is not DeviceTrueResetCommitProjection
                or reset_prepared is not prepared
                or reset_event_identity is not owned.reset_event_identity
                or reset_child_kinds != CHILD_OWNER_ORDER
                or type(reset_child_commit_identities) is not tuple
                or len(reset_child_commit_identities)
                != len(CHILD_OWNER_ORDER)
                or len(
                    {id(identity) for identity in reset_child_commit_identities}
                )
                != len(CHILD_OWNER_ORDER)
                or projection.preflight_capability
                is not owned.preflight_capability
            ):
                raise DeviceR05ConflictError("true-reset commit authority differs")
        except Exception as exc:
            owned.stage = "commit_failed_after_children"
            self._active = None
            self._settle_failed_true_reset(owned, poison_reason=5)
            raise DeviceR05PoisonedError(
                "true-reset child proof failed after child commits"
            ) from exc

        index = owned.selected_index
        # C03 state is deliberately untouched by true reset.  Host mutation
        # exhaustion was synchronously rejected by prepare, before the top
        # could call any child method.
        try:
            after = self._publication_afterimage()
            mutation_room = owned.mutation_before != _I64_MAX
            commit_mask = torch.logical_and(
                owned.selected_mask, ~owned.generation_overflow_fault
            )
            after.live["_reset_generation"].copy_(
                torch.where(
                    commit_mask,
                    owned.generation_after,
                    owned.generation_before,
                )
            )
            for name, reset_value, before in (
                (
                    "_scheduled_ordinal",
                    torch.full_like(self._scheduled_ordinal, -1),
                    owned.scheduled_ordinal_before,
                ),
                (
                    "_outcome_shot_index",
                    torch.zeros_like(self._outcome_shot_index),
                    owned.outcome_shot_index_before,
                ),
                (
                    "_sequence_kind",
                    torch.full_like(self._sequence_kind, SEQUENCE_EMPTY),
                    owned.sequence_kind_before,
                ),
                (
                    "_task_identity",
                    torch.full_like(self._task_identity, -1),
                    owned.task_identity_before,
                ),
                (
                    "_outcome_identity",
                    torch.full_like(self._outcome_identity, -1),
                    owned.outcome_identity_before,
                ),
                (
                    "_ball_identity",
                    torch.full_like(self._ball_identity, -1),
                    owned.ball_identity_before,
                ),
            ):
                after.live[name].copy_(
                    torch.where(commit_mask, reset_value, before)
                )
            after.live["_policy_opportunity"].copy_(
                torch.where(
                    commit_mask,
                    torch.zeros_like(self._policy_opportunity),
                    owned.policy_opportunity_before,
                )
            )
            after.live["_mutation_version"].copy_(
                torch.where(
                    mutation_room,
                    owned.mutation_before + mutation_room.to(torch.int64),
                    owned.mutation_before,
                )
            )
            after.counters["_mutation_version_host"] = (
                self._mutation_version_host + 1
            )
            device_fault = owned.device_fault
            after.live["_poisoned"].logical_or_(device_fault)
            after.live["_poison_reason"].copy_(
                torch.where(
                    torch.logical_and(
                        device_fault,
                        after.live["_poison_reason"].eq(0),
                    ),
                    torch.full_like(self._poison_reason, 17),
                    after.live["_poison_reason"],
                )
            )
            row = self._build_true_reset_journal_afterimage(
                owned, committed=True
            )
            receipt = object.__new__(DeviceR05TrueResetReceipt)
            after.registries["_true_reset_receipts"][receipt] = (
                _TrueResetRecord(
                    receipt=receipt,
                    prepared=prepared,
                    journal_sequence=self._journal_head,
                    completed_children=set(),
                )
            )
            after.registries["_prepared_true_resets"].pop(prepared, None)
            after.counters["_active"] = None
            after.counters["_journal_head"] = self._journal_head + 1
            self._install_journal_row_afterimage(
                after, slot=owned.journal_slot, row=row
            )
        except Exception as exc:
            owned.stage = "commit_write_failed"
            self._active = None
            self._settle_failed_true_reset(owned, poison_reason=9)
            raise DeviceR05PoisonedError(
                "true-reset publication failed after child commits"
            ) from exc
        self._publish(after)
        if not self._action_epoch_is_single_business_log:
            self._checkpoint_requires_global_drain_ack = True
        owned.stage = "committed"
        return receipt

    def record_true_reset_child_completion(
        self,
        receipt: DeviceR05TrueResetReceipt,
        *,
        child_kind: str,
        child_receipt: object,
    ) -> None:
        """Record one exact child completion after the R05-last reset write.

        The same construction-bound child authorities validate reveal and
        selected-reset completions.  A caller-supplied boolean or a journal
        self-write is never accepted as evidence that another owner settled.
        """

        self._enter_public_operation()
        self._close_construction_window()
        if self._active is not None or self._active_drain is not None:
            raise DeviceR05ConflictError(
                "true-reset completion cannot overlap another R05 operation"
            )
        record = self._true_reset_receipts.get(receipt)
        if (
            type(receipt) is not DeviceR05TrueResetReceipt
            or record is None
            or record.receipt is not receipt
            or child_kind not in CHILD_OWNER_ORDER
            or child_kind in record.completed_children
        ):
            raise DeviceR05ConflictError(
                "true-reset child completion is stale"
            )
        try:
            projection = self._call_authority(
                self._true_reset_authority.require_owned_r05_true_reset_child_completion,
                receipt,
                child_kind=child_kind,
                child_receipt=child_receipt,
            )
            if (
                type(projection)
                is not DeviceTrueResetChildCompletionProjection
                or projection.true_reset_receipt is not receipt
                or projection.child_kind != child_kind
                or projection.child_receipt is not child_receipt
            ):
                raise DeviceR05ConflictError(
                    "true-reset child completion projection differs"
                )
        except Exception as exc:
            self._poison(18)
            raise DeviceR05PoisonedError(
                "true-reset child completion failed after publication"
            ) from exc
        child_index = CHILD_OWNER_ORDER.index(child_kind)
        slot = record.journal_sequence % self._journal_capacity
        after = self._publication_afterimage()
        completed_children = {*record.completed_children, child_kind}
        after.registries["_true_reset_receipts"][receipt] = _TrueResetRecord(
            receipt=record.receipt,
            prepared=record.prepared,
            journal_sequence=record.journal_sequence,
            completed_children=completed_children,
        )
        row = dict(after.journal_rows[slot])
        row["child_completion"] = row["child_completion"].clone()
        row["child_completion"][child_index] = True
        after.journal_rows[slot] = row
        if (
            self._action_epoch_is_single_business_log
            and len(completed_children) == len(CHILD_OWNER_ORDER)
        ):
            after.registries["_true_reset_receipts"].pop(receipt, None)
        self._publish(after)
        if (
            self._action_epoch_is_single_business_log
            and len(completed_children) == len(CHILD_OWNER_ORDER)
        ):
            self._retire_action_epoch_shadow_journal()

    def require_owned_true_reset_receipt(
        self,
        receipt: DeviceR05TrueResetReceipt,
        *,
        expected_prepared_true_reset: DeviceR05PreparedTrueReset,
    ) -> DeviceR05TrueResetReceipt:
        """Repeatably validate exact receipt and exact retained prepare token."""

        self._enter_public_operation()
        if type(receipt) is not DeviceR05TrueResetReceipt:
            raise DeviceR05ConflictError("true-reset receipt type differs")
        record = self._true_reset_receipts.get(receipt)
        if (
            record is None
            or record.receipt is not receipt
            or record.prepared is not expected_prepared_true_reset
        ):
            raise DeviceR05ConflictError("true-reset receipt is stale or foreign")
        return receipt

    def _finish_true_reset_journal(
        self,
        prepared: _PreparedTrueResetRecord,
        *,
        committed: bool,
    ) -> None:
        row = self._build_true_reset_journal_afterimage(
            prepared, committed=committed
        )
        self._apply_journal_row(prepared.journal_slot, row)
        self._publication.journal_rows[prepared.journal_slot] = row
        self._journal_head += 1

    def _build_true_reset_journal_afterimage(
        self,
        prepared: _PreparedTrueResetRecord,
        *,
        committed: bool,
    ) -> dict[str, torch.Tensor]:
        """Build a reset row off-owner for the terminal pointer swap."""

        slot = prepared.journal_slot
        index = prepared.selected_index
        row = self._journal_row_afterimage(slot)
        meta = row["meta"]
        meta[self._META_SEQUENCE] = self._journal_head
        meta[self._META_EPOCH] = prepared.epoch
        meta[self._META_OPERATION] = (
            JOURNAL_TRUE_RESET if committed else JOURNAL_ABORT
        )
        meta[self._META_MUTATION_BEFORE] = prepared.mutation_before
        meta[self._META_MUTATION_AFTER] = (
            torch.where(
                prepared.device_fault,
                prepared.mutation_before,
                prepared.mutation_after,
            )
            if committed
            else prepared.mutation_before
        )
        meta[self._META_SELECTED_COUNT] = prepared.selected_index.shape[0]
        meta[self._META_ADMISSIBLE_COUNT] = 0
        meta[self._META_TRANSFER_SEQUENCE] = 0
        meta[self._META_DECISION] = 0
        row["selected"][index] = True
        row["reason"][index] = (
            REASON_TRUE_RESET if committed else REASON_ABORTED_BEFORE_TRANSFER
        )
        row["rng_before_lo"][index] = self._rng_lo[index]
        row["rng_before_hi"][index] = self._rng_hi[index]
        row["rng_after_lo"][index] = self._rng_lo[index]
        row["rng_after_hi"][index] = self._rng_hi[index]
        row["draw_before"][index] = self._draw_count[index]
        row["draw_after"][index] = self._draw_count[index]
        row["sampler_generation_before"][index] = self._target_generation[index]
        row["sampler_generation_after"][index] = self._target_generation[index]
        row["previous_before"][index] = self._previous_cell_index[index]
        row["previous_after"][index] = self._previous_cell_index[index]
        row["reset_before"][index] = prepared.generation_before[index]
        row["reset_after"][index] = (
            torch.where(
                prepared.generation_overflow_fault[index],
                prepared.generation_before[index],
                prepared.generation_after[index],
            )
            if committed
            else prepared.generation_before[index]
        )
        row["ordinal_before"][index] = prepared.scheduled_ordinal_before[index]
        row["ordinal_after"][index] = (
            torch.full_like(index, -1)
            if committed
            else prepared.scheduled_ordinal_before[index]
        )
        row["outcome_before"][index] = prepared.outcome_shot_index_before[index]
        row["outcome_after"][index] = (
            torch.zeros_like(index)
            if committed
            else prepared.outcome_shot_index_before[index]
        )
        row["primary_fault"][index] = torch.where(
            prepared.generation_overflow_fault[index],
            torch.full_like(index, 17),
            torch.where(
                prepared.device_fault.expand_as(index),
                torch.full_like(index, 16),
                torch.zeros_like(index),
            ),
        )
        return row

    def _drainable(self) -> None:
        self._close_construction_window()
        if self._active is not None:
            raise DeviceR05ConflictError("active transaction cannot drain")
        if self._active_drain is not None:
            raise DeviceR05ConflictError("a PPO-boundary drain is unacknowledged")
        if not self._poisoned_python and any(
            len(record.completed_children) != len(CHILD_OWNER_ORDER)
            for record in self._terminal_receipts.values()
        ):
            raise DeviceR05ConflictError("terminal child completions are incomplete")
        if not self._poisoned_python and any(
            len(record.completed_children) != len(CHILD_OWNER_ORDER)
            for record in self._true_reset_receipts.values()
        ):
            raise DeviceR05ConflictError(
                "true-reset child completions are incomplete"
            )

    def _journal_source(self, name: str) -> torch.Tensor:
        """Return the complete bounded journal in physical-slot order.

        ACK compaction clears retired slots.  The full fixed-capacity tensor is
        therefore a canonical device row for the global drain; logical
        membership remains ``[journal_start_sequence, journal_end_sequence)``.
        """

        legacy = getattr(self, f"_journal_{name}")
        rows = []
        for slot in range(self._journal_capacity):
            published = self._publication.journal_rows.get(slot)
            rows.append(legacy[slot] if published is None else published[name])
        return torch.stack(rows, dim=0)

    def _global_journal_flat(self, name: str) -> torch.Tensor:
        raw = self._journal_source(name).to(torch.int64).reshape(-1)
        # Global schemas intentionally reject negative host integers.  Encode
        # every exact signed int64 as (nonnegative_bit, payload) without
        # arithmetic overflow: bitwise-not maps every negative value into
        # [0, INT64_MAX] and is exactly reversible from the sign bit.
        nonnegative = raw.ge(0).to(torch.int64)
        payload = torch.where(raw.ge(0), raw, torch.bitwise_not(raw))
        return torch.cat((nonnegative, payload), dim=0)

    @staticmethod
    def _require_global_drain_module() -> object:
        return importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        )

    def _require_global_drain_authority(self, authority: object) -> object:
        drain = self._require_global_drain_module()
        mint = getattr(authority, "mint_device_pack", None)
        ack = getattr(authority, "require_owned_ack", None)
        expected_width = len(GLOBAL_DRAIN_SCALAR_FIELD_NAMES) + sum(
            2 * self._journal_capacity
            * _journal_field_width_per_row(
                name,
                num_envs=self._num_envs,
                support_size=self._profile.support_size,
            )
            for name in _GLOBAL_JOURNAL_FIELD_NAMES
        )
        if (
            type(authority) is not drain.LeafDevicePackAuthority
            or getattr(authority, "owner_kind", None) != GLOBAL_DRAIN_OWNER_KIND
            or tuple(getattr(authority, "field_names", ()))
            != GLOBAL_DRAIN_FIELD_NAMES
            or getattr(authority, "expected_width", None) != expected_width
            or not callable(mint)
            or getattr(mint, "__self__", None) is not authority
            or getattr(mint, "__func__", None)
            is not drain.LeafDevicePackAuthority.mint_device_pack
            or not callable(ack)
            or getattr(ack, "__self__", None) is not authority
            or getattr(ack, "__func__", None)
            is not drain.LeafDevicePackAuthority.require_owned_ack
        ):
            raise DeviceR05ConflictError(
                "Device-R05 global drain exact authority/schema differs"
            )
        return drain

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Mint Device-R05's exact fixed-capacity journal leaf row."""

        self._enter_public_operation()
        self._close_construction_window()
        if self._action_epoch_is_single_business_log:
            raise DeviceR05ConflictError(
                "ActionEpoch-owned Device-R05 rejects a second global drain"
            )
        self._drainable()
        if (
            type(update_index) is not int
            or update_index <= self._last_global_update_index
            or type(completed_environment_steps) is not int
            or completed_environment_steps
            <= self._last_global_completed_environment_steps
        ):
            raise DeviceR05ConflictError(
                "Device-R05 global drain frontier did not strictly advance"
            )
        drain = self._require_global_drain_authority(authority)
        del drain
        row_count = self._journal_head - self._journal_tail
        meta = self._journal_source("meta")
        active_slots = torch.zeros(
            self._journal_capacity, dtype=torch.bool, device=self._device
        )
        for sequence in range(self._journal_tail, self._journal_head):
            active_slots[sequence % self._journal_capacity] = True
        operation = meta[:, self._META_OPERATION]
        selected_count = meta[:, self._META_SELECTED_COUNT]
        settlement = self._journal_source("settlement")
        mixed_terminal = (
            settlement.eq(DECISION_ACCEPT) | settlement.eq(DECISION_CENSOR)
        ).sum(dim=1, dtype=torch.int64)
        mixed_policy = settlement.eq(DECISION_ACCEPT).sum(
            dim=1, dtype=torch.int64
        )
        terminal_delta = torch.where(
            active_slots,
            torch.where(
                operation.eq(JOURNAL_MIXED_EPOCH),
                mixed_terminal,
                torch.where(
                    (operation == JOURNAL_ACCEPT)
                    | (operation == JOURNAL_CENSOR),
                    selected_count,
                    torch.zeros_like(selected_count),
                ),
            ),
            torch.zeros_like(selected_count),
        ).sum().reshape(1)
        policy_delta = torch.where(
            active_slots,
            torch.where(
                operation.eq(JOURNAL_MIXED_EPOCH),
                mixed_policy,
                torch.where(
                    operation.eq(JOURNAL_ACCEPT),
                    selected_count,
                    torch.zeros_like(selected_count),
                ),
            ),
            torch.zeros_like(selected_count),
        ).sum().reshape(1)
        terminal_total = terminal_delta + self._last_global_terminal_resolution_total
        policy_total = policy_delta + self._last_global_policy_opportunity_total
        invariant_count = (
            (self._mutation_version < 0).to(torch.int64).reshape(1)
            + (terminal_delta < policy_delta).to(torch.int64)
        )
        scalar_values = torch.cat(
            (
                self._mutation_version.reshape(1),
                self._poisoned.to(torch.int64).reshape(1),
                invariant_count,
                terminal_total,
                policy_total,
                torch.tensor(
                    (row_count, self._journal_tail, self._journal_head),
                    dtype=torch.int64,
                    device=self._device,
                ),
            )
        )
        journal_values = tuple(
            self._global_journal_flat(name)
            for name in _GLOBAL_JOURNAL_FIELD_NAMES
        )
        values = torch.cat((scalar_values, *journal_values), dim=0).contiguous()
        if values.shape != (getattr(authority, "expected_width", -1),):
            raise DeviceR05ConflictError("Device-R05 global drain width differs")
        pack = authority.mint_device_pack(leaf=self, values=values)
        self._active_drain = _DrainRecord(
            capability=pack,
            authority=authority,
            update_index=update_index,
            completed_environment_steps=completed_environment_steps,
            mutation_version_host=self._mutation_version_host,
            journal_head=self._journal_head,
            journal_tail=self._journal_tail,
            row_count=row_count,
        )
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(
        self, *, pack: object
    ) -> None:
        """Release a pre-transfer global lease without compacting evidence."""

        if self._action_epoch_is_single_business_log:
            raise DeviceR05ConflictError(
                "ActionEpoch-owned Device-R05 has no second drain to abort"
            )

        active = self._active_drain
        if (
            active is None
            or pack is not active.capability
            or active.stage != "prepared"
        ):
            raise DeviceR05ConflictError(
                "Device-R05 global drain abort pack is stale or foreign"
            )
        if (
            self._mutation_version_host != active.mutation_version_host
            or self._journal_head != active.journal_head
            or self._journal_tail != active.journal_tail
        ):
            self._poison(20)
            active.stage = "poisoned"
            raise DeviceR05PoisonedError(
                "Device-R05 mutated while global drain lease was active"
            )
        active.stage = "aborted"
        self._active_drain = None

    @staticmethod
    def _global_owner_row_mapping(owner_row: object) -> dict[str, object]:
        if getattr(owner_row, "owner_kind", None) != GLOBAL_DRAIN_OWNER_KIND:
            raise DeviceR05ConflictError("global receipt owner row kind differs")
        values = getattr(owner_row, "values", None)
        if type(values) is not tuple:
            raise DeviceR05ConflictError("global receipt owner row values differ")
        names = tuple(
            item[0]
            for item in values
            if type(item) is tuple and len(item) == 2 and type(item[0]) is str
        )
        if names != GLOBAL_DRAIN_FIELD_NAMES or len(names) != len(values):
            raise DeviceR05ConflictError("global receipt owner row schema differs")
        return dict(values)

    def _clear_acknowledged_journal(self, end_sequence: int) -> None:
        self._journal_tail = end_sequence
        self._terminal_receipts = {
            receipt: record
            for receipt, record in self._terminal_receipts.items()
            if record.journal_sequence >= self._journal_tail
        }
        self._true_reset_receipts = {
            receipt: record
            for receipt, record in self._true_reset_receipts.items()
            if record.journal_sequence >= self._journal_tail
        }
        self._prepared_true_resets = {
            prepared: record
            for prepared, record in self._prepared_true_resets.items()
            if record.stage == "prepared"
        }
        retained_slots = {
            sequence % self._journal_capacity
            for sequence in range(self._journal_tail, self._journal_head)
        }
        retired_slots = set(range(self._journal_capacity)) - retained_slots
        for slot in retired_slots:
            for name in _JOURNAL_FIELD_NAMES:
                tensor = getattr(self, f"_journal_{name}")[slot]
                if name in (
                    "construction_reason",
                    "candidate_identity",
                    "round_construction_reason",
                    "round_candidate_identity",
                    "chosen_round",
                    "reason",
                    "selected_cell",
                    "candidate_bank_sequence",
                    "selected_candidate_identity",
                    "task_identity",
                    "outcome_identity",
                    "ball_identity",
                    "cadence_identity",
                ):
                    tensor.fill_(-1)
                else:
                    tensor.zero_()
        self._publication.journal_rows = {
            slot: row
            for slot, row in self._publication.journal_rows.items()
            if slot in retained_slots
        }

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Consume the exact post-optimizer global ACK and compact journal."""

        if self._action_epoch_is_single_business_log:
            raise DeviceR05ConflictError(
                "ActionEpoch-owned Device-R05 rejects a second global ACK"
            )

        active = self._active_drain
        try:
            if active is None:
                raise DeviceR05ConflictError(
                    "Device-R05 global acknowledgement has no active pack"
                )
            # Exact authority must be the first receipt/row consumer.  Equal
            # values from a second real coordinator are not authority.
            active.authority.require_owned_ack(
                leaf=self,
                pack=pack,
                receipt=receipt,
                owner_row=owner_row,
            )
            drain = self._require_global_drain_module()
            if (
                active.stage != "prepared"
                or pack is not active.capability
                or type(receipt) is not drain.PreOptimizerPpoBoundaryReceipt
                or type(owner_row) is not drain.OwnerDrainRow
                or getattr(receipt, "update_index", None) != active.update_index
                or getattr(receipt, "completed_environment_steps", None)
                != active.completed_environment_steps
                or getattr(receipt, "device_to_host_transfers", None) != 1
                or type(getattr(receipt, "drain_sequence", None)) is not int
                or receipt.drain_sequence != self._last_global_drain_sequence + 1
                or not any(row is owner_row for row in receipt.owner_rows)
                or receipt.acknowledged
                or self._mutation_version_host != active.mutation_version_host
            ):
                raise DeviceR05ConflictError(
                    "Device-R05 global acknowledgement differs"
                )
            row = self._global_owner_row_mapping(owner_row)
            if (
                row["mutation_version"] != active.mutation_version_host
                or row["journal_count"] != active.row_count
                or row["journal_start_sequence"]
                != active.journal_tail
                or row["journal_end_sequence"] != active.journal_head
            ):
                raise DeviceR05ConflictError(
                    "Device-R05 global acknowledgement frontier differs"
                )
        except BaseException as exc:
            self.poison_pre_optimizer_ppo_boundary(
                reason=(
                    "Device-R05 global acknowledgement failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            raise
        self._clear_acknowledged_journal(active.journal_head)
        self._last_global_drain_sequence = receipt.drain_sequence
        self._last_global_update_index = active.update_index
        self._last_global_completed_environment_steps = (
            active.completed_environment_steps
        )
        self._last_global_ack_mutation_version = active.mutation_version_host
        self._last_global_terminal_resolution_total = int(
            row["terminal_resolution_total"]
        )
        self._last_global_policy_opportunity_total = int(
            row["policy_opportunity_total"]
        )
        self._checkpoint_requires_global_drain_ack = False
        active.stage = "acknowledged"
        self._active_drain = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason: str) -> None:
        """Idempotently fail-stop this owner after global boundary failure."""
        self._enter_public_operation()
        if self._global_drain_poison_reason is None:
            self._global_drain_poison_reason = (
                reason
                if type(reason) is str and bool(reason.strip())
                else "unspecified Device-R05 global drain failure"
            )
        self._poison(21)

    def pack_drain(self) -> DeviceR05DrainBatch:
        """Tombstone the retired local drain capability."""

        raise DeviceR05ConflictError(
            "local Device-R05 drain is tombstoned; use the seven-leaf global drain"
        )

    def materialize_portable(self, batch: DeviceR05DrainBatch) -> object:
        """Tombstone the retired local portable-materialization path."""

        del batch
        raise DeviceR05ConflictError(
            "local Device-R05 materialization is tombstoned; "
            "the seven-leaf global drain owns the sole D2H"
        )

    def require_owned_drain_view(
        self, batch: DeviceR05DrainBatch
    ) -> DeviceR05DrainView:
        """Tombstone the retired local drain-view path."""

        del batch
        raise DeviceR05ConflictError(
            "local Device-R05 drain view is tombstoned; use global owner rows"
        )

    def ack_drain(
        self, batch: DeviceR05DrainBatch, drain_ack_receipt: object
    ) -> None:
        """Tombstone the retired local ACK path."""

        del batch, drain_ack_receipt
        raise DeviceR05ConflictError(
            "local Device-R05 ACK is tombstoned; use the exact global leaf ACK"
        )

    def _lean_carry_schema(self):
        """Describe only D05 business continuation and construction attestations."""

        carry = _require_lean_carry_module()
        n, support = self._num_envs, self._profile.support_size
        i64, boolean = torch.int64, torch.bool
        shapes = (
            (), (), (n,), (n,), (n,), (n,), (n,), (n,), (n,), (n,),
            (n,), (n,), (n,), (n,), (n,), (), (support, 2), (), (), (n,), (n,),
        )
        dtypes = (
            i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
            i64, i64, i64, i64, boolean, i64, torch.float32, boolean,
            i64, i64, i64,
        )
        fields = tuple(
            carry._LeanCarryTensorSpec(
                name, shape, dtype,
                "copy" if name in _LEAN_CARRY_COPY_NAMES else "attest",
            )
            for name, shape, dtype in zip(
                _LEAN_CARRY_COPY_NAMES + _LEAN_CARRY_ATTEST_NAMES, shapes, dtypes
            )
        )
        return carry._LeanCarrySchema(
            role="d05",
            scalar_fields=(
                ("profile_sha256", str), ("profile_binding_sha256", str),
                ("cell_ids", tuple), ("semantic_sha256s", tuple),
                ("seed", int), ("num_envs", int), ("journal_capacity", int),
                ("max_reveal_epochs_per_drain", int),
                ("internal_question_redraw_rounds", int),
                ("action_epoch_single_business_log", bool),
                ("epoch", int), ("last_transfer_sequence", int),
                ("mutation_version_host", int),
            ),
            tensor_fields=fields,
        )
    def _lean_carry_scalars(self) -> tuple[object, ...]:
        return (
            self._profile.profile_sha256, self._profile_binding_sha256,
            self._profile.cell_ids, self._profile.semantic_sha256s,
            self._seed, self._num_envs, self._journal_capacity,
            self._max_reveal_epochs_per_drain,
            INTERNAL_QUESTION_REDRAW_ROUNDS,
            self._action_epoch_is_single_business_log,
            self._epoch, self._last_transfer_sequence,
            self._mutation_version_host,
        )
    def _lean_carry_views(self) -> tuple[torch.Tensor, ...]:
        values = tuple(getattr(self, "_" + name) for name in _LEAN_CARRY_COPY_NAMES)
        values += (
            self._profile.targets_xy_m, self._poisoned, self._poison_reason,
            self._genesis_reset_generations, self._row_axis,
        )
        schema = self._lean_carry_schema()
        for value, field in zip(values, schema.tensor_fields):
            _require_tensor(
                value, label="lean_carry." + field.name, device=self._device,
                dtype=field.dtype, shape=field.shape,
            )
            if not value.is_contiguous():
                raise DeviceR05ConflictError(
                    "lean carry live tensor is not contiguous: " + field.name
                )
        return values
    def _require_lean_carry_lease(self, lease: object, *, kind: str) -> object:
        carry = _require_lean_carry_module()
        coordinator = getattr(self, "_lean_carry_coordinator", None)
        if (
            type(lease) is not carry._LeanCarryLease
            or lease.coordinator is not coordinator
            or getattr(coordinator, "_active_lease", None) is not lease
            or lease.kind != kind
        ):
            raise DeviceR05ConflictError("D05 Lean carry lease differs")
        return carry
    def _lean_carry_bindings(self) -> tuple[object, ...]:
        return (self._profile, self._profile_authority, self._profile_receipt, self._genesis_authority, self._genesis_world_reset_identity, self._live_reset_ledger_identity, self._cadence_authority, self._question_authority, self._internal_question_compose, self._diagnostic_epoch_owner, self._diagnostic_motion_owner, self._diagnostic_racket_owner, self._diagnostic_physical_owner, self._true_reset_authority, self._env_reset_binding)
    def _require_lean_carry_quiescent(self, *, fresh: bool) -> None:
        if (
            not self._action_epoch_is_single_business_log
            or self._construction_window_open
            or self._poisoned_python
            or self._authority_callback_active
            or self._authority_reentry_detected
            or self._question_composition_in_progress
            or self._active_diagnostic_epoch_leaf_writer is not None
            or self._active_row_transaction is not None
            or self._row_transaction_records
            or self._active is not None
            or self._active_drain is not None
            or any(getattr(self, name) for name in _PUBLICATION_REGISTRY_NAMES)
            or tuple(map(id, self._lean_carry_bindings())) != tuple(map(id, getattr(self, "_lean_carry_frozen_bindings", ())))
        ):
            raise DeviceR05ConflictError("D05 Lean carry call point is not quiescent")
        if fresh and (
            self._epoch != 0
            or self._last_transfer_sequence != 0
            or self._mutation_version_host != 0
            or self._journal_head != 0
            or self._journal_tail != 0
            or self._last_candidate_bank_sequence != 0
            or self._next_candidate_identity != 1
        ):
            raise DeviceR05ConflictError("D05 Lean carry target is not fresh")
    @staticmethod
    def _lean_profile_binding(
        profile_sha256: str, cell_ids: tuple, semantics: tuple,
        targets: torch.Tensor,
    ) -> str:
        digest = hashlib.sha256(profile_sha256.encode("ascii"))
        for cell_id, semantic in zip(cell_ids, semantics):
            encoded = cell_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(bytes.fromhex(semantic))
        for row in targets.tolist():
            for value in row:
                digest.update(struct.pack(">f", value))
        return digest.hexdigest()
    def _validate_lean_carry_host(
        self, scalars: object, tensors: object,
    ) -> tuple[torch.Tensor, ...]:
        carry, schema = _require_lean_carry_module(), self._lean_carry_schema()
        values = carry._require_scalars(schema, scalars, label="d05.host")
        named = carry._require_tensors(schema, tensors, label="d05.host")
        host = tuple(value for _name, value in named)
        if named and named[0][1].device.type != "cpu":
            raise DeviceR05ConflictError("D05 Lean carry host image is not CPU")
        if values[:10] != self._lean_carry_scalars()[:10]:
            raise DeviceR05ConflictError("D05 Lean carry construction differs")
        epoch, transfer, mutation_host = values[10:]
        if any(type(value) is not int or not (0 <= value < _I64_MAX)
               for value in (epoch, transfer, mutation_host)):
            raise DeviceR05ConflictError("D05 Lean carry scalar chronology differs")
        data = dict(zip(_LEAN_CARRY_COPY_NAMES + _LEAN_CARRY_ATTEST_NAMES, host))
        profile = data["profile_targets_xy_m"]
        if (
            not bool(torch.all(torch.isfinite(profile)))
            or self._lean_profile_binding(values[0], values[2], values[3], profile)
            != values[1]
            or bool(data["poisoned"])
            or int(data["poison_reason"]) != 0
            or not torch.equal(data["row_axis"], torch.arange(self._num_envs))
            or not bool(torch.all((data["genesis_reset_generations"] >= 1)
                                  & (data["genesis_reset_generations"] < _I64_MAX)))
        ):
            raise DeviceR05ConflictError("D05 Lean carry attestation differs")
        draw, generation = data["draw_count"], data["target_generation"]
        reset, genesis = data["reset_generation"], data["genesis_reset_generations"]
        previous, sequence = data["previous_cell_index"], data["sequence_kind"]
        scheduled, outcome = data["scheduled_ordinal"], data["outcome_shot_index"]
        task, outcome_id, ball = (
            data["task_identity"], data["outcome_identity"], data["ball_identity"]
        )
        next_outcome, next_ball = (
            data["next_outcome_identity"], data["next_ball_identity"]
        )
        valid = (
            torch.all((draw >= 0) & (draw < _I64_MAX))
            & torch.all(torch.remainder(draw, INTERNAL_QUESTION_TOTAL_DRAW_WIDTH).eq(0))
            & torch.all((generation >= 0) & (generation < _I64_MAX) & (generation <= draw // INTERNAL_QUESTION_TOTAL_DRAW_WIDTH) & generation.eq(0).eq(previous.eq(-1)))
            & torch.all((previous >= -1) & (previous < self._profile.support_size))
            & torch.all((reset >= genesis) & (reset < _I64_MAX))
            & torch.all((scheduled >= -1) & (scheduled < _I64_MAX))
            & torch.all((outcome >= 0) & (outcome < _I64_MAX))
            & torch.all((sequence >= SEQUENCE_EMPTY)
                        & (sequence <= SEQUENCE_COMMITTED))
            & torch.all(data["policy_opportunity"].eq(sequence.eq(SEQUENCE_COMMITTED)))
            & torch.all(task.eq(outcome_id) & task.eq(ball) & torch.where(sequence.eq(SEQUENCE_COMMITTED), (task >= 1) & (generation >= 1) & (previous >= 0), task == -1))
            & next_outcome.eq(next_ball)
            & (next_outcome >= 1) & (next_outcome < _I64_MAX)
            & torch.all(task.lt(next_outcome))
            & data["mutation_version"].eq(mutation_host)
        )
        empty = sequence.eq(SEQUENCE_EMPTY)
        # Cadence settlement and task acceptance are deliberately separate.
        # A due row can be rejected/deferred/censored before any task has ever
        # been accepted, so SEQUENCE_EMPTY does not imply an untouched cadence
        # cursor.  The one invariant shared by genesis and consumed due rows is
        # that the next outcome index is exactly one past the last scheduled
        # ordinal (-1/0 at genesis, N/N+1 afterwards).
        valid &= torch.all(outcome.eq(scheduled + 1))
        valid &= torch.all(~empty | task.eq(-1))
        valid &= torch.all(empty | ((scheduled >= 0) & (outcome >= 0)))
        if not bool(valid) or len({value for value in task.tolist() if value >= 1}) != int(torch.sum(task >= 1)):
            raise DeviceR05ConflictError("D05 Lean carry business state differs")
        initial = tuple(
            _initial_stream_state(profile_sha256=values[0], seed=values[4], env_id=i)
            for i in range(self._num_envs)
        )
        draws = tuple(int(value) for value in draw.tolist())
        expected = tuple(
            (state + count * _SPLITMIX_GAMMA) & ((1 << 64) - 1)
            for state, count in zip(initial, draws)
        )
        expected_lo = torch.tensor(tuple(value & _U32_MASK for value in expected))
        expected_hi = torch.tensor(tuple(value >> 32 for value in expected))
        if not torch.equal(data["rng_lo"], expected_lo) or not torch.equal(
            data["rng_hi"], expected_hi
        ):
            raise DeviceR05ConflictError("D05 Lean carry RNG chronology differs")
        return host
    def _lean_carry_capture(self, lease: object):
        carry = self._require_lean_carry_lease(lease, kind="capture")
        self._require_lean_carry_quiescent(fresh=False)
        return carry._LeanCarryCapture(self._lean_carry_scalars(), self._lean_carry_views())
    def _lean_carry_construction_views(self):
        current, frozen = self._lean_carry_bindings(), getattr(self, "_lean_carry_frozen_bindings", None)
        if frozen is not None and any(now is not before for now, before in zip(current, frozen)): raise DeviceR05ConflictError("D05 Lean carry construction binding drifted")
        if frozen is None: self._lean_carry_frozen_bindings = current
        return self._lean_carry_views()
    def _lean_carry_stage(self, lease: object, scalars: object, host_tensors: object):
        carry = self._require_lean_carry_lease(lease, kind="prepare")
        self._require_lean_carry_quiescent(fresh=True)
        host = self._validate_lean_carry_host(scalars, host_tensors)
        targets = self._lean_carry_views()
        initial = tuple(
            _initial_stream_state(
                profile_sha256=self._profile.profile_sha256,
                seed=self._seed, env_id=index,
            )
            for index in range(self._num_envs)
        )
        expected = (
            torch.ones((), dtype=torch.int64, device=self._device),
            torch.ones((), dtype=torch.int64, device=self._device),
            torch.tensor(tuple(value & _U32_MASK for value in initial), device=self._device),
            torch.tensor(tuple(value >> 32 for value in initial), device=self._device),
        )
        fresh = (
            torch.equal(targets[0], expected[0]) and torch.equal(targets[1], expected[1])
            and torch.equal(targets[2], expected[2]) and torch.equal(targets[3], expected[3])
            and all(not bool(torch.any(value)) for value in targets[4:6])
            and bool(torch.all(targets[6].eq(-1)))
            and torch.equal(targets[7], targets[19])
            and bool(torch.all(targets[8].eq(-1)))
            and all(not bool(torch.any(value)) for value in targets[9:11])
            and all(bool(torch.all(value.eq(-1))) for value in targets[11:14])
            and not bool(torch.any(targets[14]))
            and not bool(targets[15]) and not bool(targets[17])
            and int(targets[18]) == 0
        )
        if not fresh:
            raise DeviceR05ConflictError("D05 Lean carry target is not genesis-fresh")
        staging = tuple(
            value.to(device=self._device, copy=True).contiguous() for value in host
        )
        return carry._LeanCarryStage(
            scalars=scalars, staging=staging, targets=targets
        )
    def _lean_carry_target_views(self, lease: object, stage: object):
        carry = self._require_lean_carry_lease(lease, kind="prepare")
        self._require_lean_carry_quiescent(fresh=True)
        if type(stage) is not carry._LeanCarryStage or stage.commit_started or stage.scalars[:10] != self._lean_carry_scalars()[:10]:
            raise DeviceR05ConflictError("D05 Lean carry stage differs")
        current = self._lean_carry_views()
        for index in range(len(_LEAN_CARRY_COPY_NAMES), len(current)):
            if not torch.equal(current[index], stage.staging[index]):
                raise DeviceR05ConflictError("D05 Lean carry attestation drifted")
        return current
    def _lean_carry_apply_scalars(self, lease: object, stage: object) -> None:
        carry = self._require_lean_carry_lease(lease, kind="prepare")
        if type(stage) is not carry._LeanCarryStage or not stage.commit_started:
            raise DeviceR05ConflictError("D05 Lean carry commit stage differs")
        carry._require_scalars(
            self._lean_carry_schema(), stage.scalars, label="d05.commit"
        )
        self._epoch, self._last_transfer_sequence, self._mutation_version_host = (
            stage.scalars[10:]
        )

    def checkpoint_device(self) -> DeviceR05Checkpoint:
        """Create a resumable checkpoint at the canonical business frontier.

        Legacy owners still require their exact global journal ACK.  In the
        lean ActionEpoch-owned mode the private R05 row is only a nonblocking
        debug snapshot, so ActionEpoch is the sole checkpoint/drain ledger.
        """

        self._enter_public_operation()
        self._require_idle()
        if self._action_epoch_is_single_business_log:
            raise DeviceR05ConflictError(
                "Lean mode legacy Device-R05 checkpoint is tombstoned"
            )
        if (
            not self._action_epoch_is_single_business_log
            and self._journal_head != self._journal_tail
        ):
            raise DeviceR05ConflictError(
                "device checkpoint requires a fully acknowledged drain"
            )
        if (
            not self._action_epoch_is_single_business_log
            and (
                self._checkpoint_requires_global_drain_ack
                or self._last_global_ack_mutation_version
                != self._mutation_version_host
            )
        ):
            raise DeviceR05ConflictError(
                "device checkpoint lacks the exact global ACK mutation frontier"
            )
        if (
            self._epoch >= _I64_MAX
            or self._journal_head >= _I64_MAX
            or self._last_transfer_sequence >= _I64_MAX
            or self._last_candidate_bank_sequence >= _I64_MAX
            or self._next_candidate_identity >= _I64_MAX
            or self._last_global_drain_sequence >= _I64_MAX
            or self._mutation_version_host >= _I64_MAX
            or bool(self._next_outcome_identity >= _I64_MAX)
            or bool(self._next_ball_identity >= _I64_MAX)
        ):
            raise DeviceR05ConflictError(
                "device checkpoint has no healthy int64 continuation frontier"
            )
        return DeviceR05Checkpoint(
            profile_sha256=self._profile.profile_sha256,
            profile_binding_sha256=self._profile_binding_sha256,
            seed=self._seed,
            num_envs=self._num_envs,
            journal_capacity=self._journal_capacity,
            max_reveal_epochs_per_drain=self._max_reveal_epochs_per_drain,
            internal_question_redraw_rounds=(
                INTERNAL_QUESTION_REDRAW_ROUNDS
            ),
            epoch=self._epoch,
            journal_sequence=self._journal_head,
            last_transfer_sequence=self._last_transfer_sequence,
            last_candidate_bank_sequence=self._last_candidate_bank_sequence,
            next_candidate_identity=self._next_candidate_identity,
            last_global_drain_sequence=self._last_global_drain_sequence,
            last_global_update_index=self._last_global_update_index,
            last_global_completed_environment_steps=(
                self._last_global_completed_environment_steps
            ),
            last_global_ack_mutation_version=(
                self._last_global_ack_mutation_version
            ),
            last_global_terminal_resolution_total=(
                self._last_global_terminal_resolution_total
            ),
            last_global_policy_opportunity_total=(
                self._last_global_policy_opportunity_total
            ),
            mutation_version_host=self._mutation_version_host,
            next_outcome_identity=self._next_outcome_identity.clone(),
            next_ball_identity=self._next_ball_identity.clone(),
            rng_lo=self._rng_lo.clone(),
            rng_hi=self._rng_hi.clone(),
            draw_count=self._draw_count.clone(),
            target_generation=self._target_generation.clone(),
            previous_cell_index=self._previous_cell_index.clone(),
            reset_generation=self._reset_generation.clone(),
            scheduled_ordinal=self._scheduled_ordinal.clone(),
            outcome_shot_index=self._outcome_shot_index.clone(),
            sequence_kind=self._sequence_kind.clone(),
            task_identity=self._task_identity.clone(),
            outcome_identity=self._outcome_identity.clone(),
            ball_identity=self._ball_identity.clone(),
            policy_opportunity=self._policy_opportunity.clone(),
            mutation_version=self._mutation_version.clone(),
        )

    @classmethod
    def from_device_checkpoint(
        cls,
        *,
        profile_authority: DeviceProfileAuthority,
        profile_receipt: object,
        checkpoint_authority: DeviceCheckpointAuthority,
        authority_receipt: object,
        genesis_authority: DeviceGenesisAuthority,
        genesis_receipt: object,
        cadence_authority: DeviceCadenceAuthority,
        question_authority: DeviceQuestionAuthority,
        reveal_boundary_authority: DeviceRevealBoundaryAuthority,
        child_completion_authorities: Tuple[
            DeviceChildCompletionAuthority, ...
        ],
        drain_authority: Optional[DeviceDrainAuthority] = None,
        true_reset_authority: Optional[DeviceTrueResetAuthority] = None,
    ) -> "DeviceR05Owner":
        """Restore only the exact checkpoint retained by external authority."""

        owned_profile = _snapshot_device_profile(
            profile_authority, profile_receipt
        )
        validator = getattr(
            checkpoint_authority, "require_owned_r05_device_checkpoint", None
        )
        if not callable(validator):
            raise DeviceR05Error("checkpoint authority is not callable")
        checkpoint = validator(authority_receipt)
        if type(checkpoint) is not DeviceR05Checkpoint:
            raise DeviceR05ConflictError("checkpoint authority rejected receipt")
        profile_sha256 = checkpoint.profile_sha256
        profile_binding_sha256 = checkpoint.profile_binding_sha256
        seed = checkpoint.seed
        num_envs = checkpoint.num_envs
        journal_capacity = checkpoint.journal_capacity
        max_reveal_epochs_per_drain = checkpoint.max_reveal_epochs_per_drain
        internal_question_redraw_rounds = (
            checkpoint.internal_question_redraw_rounds
        )
        epoch = checkpoint.epoch
        journal_sequence = checkpoint.journal_sequence
        last_transfer_sequence = checkpoint.last_transfer_sequence
        last_candidate_bank_sequence = checkpoint.last_candidate_bank_sequence
        next_candidate_identity = checkpoint.next_candidate_identity
        last_global_drain_sequence = checkpoint.last_global_drain_sequence
        last_global_update_index = checkpoint.last_global_update_index
        last_global_completed_environment_steps = (
            checkpoint.last_global_completed_environment_steps
        )
        last_global_ack_mutation_version = (
            checkpoint.last_global_ack_mutation_version
        )
        last_global_terminal_resolution_total = (
            checkpoint.last_global_terminal_resolution_total
        )
        last_global_policy_opportunity_total = (
            checkpoint.last_global_policy_opportunity_total
        )
        mutation_version_host = checkpoint.mutation_version_host
        next_outcome_identity_source = checkpoint.next_outcome_identity
        next_ball_identity_source = checkpoint.next_ball_identity
        if (
            profile_sha256 != owned_profile.profile_sha256
            or profile_binding_sha256 != owned_profile.profile_binding_sha256
            or type(seed) is not int
            or type(num_envs) is not int
            or num_envs < 1
            or type(journal_capacity) is not int
            or type(max_reveal_epochs_per_drain) is not int
            or journal_capacity > _I64_MAX
            or max_reveal_epochs_per_drain > _I64_MAX
            or max_reveal_epochs_per_drain < 1
            or journal_capacity < max_reveal_epochs_per_drain
            or type(internal_question_redraw_rounds) is not int
            or internal_question_redraw_rounds
            != INTERNAL_QUESTION_REDRAW_ROUNDS
            or type(epoch) is not int
            or epoch < 0
            or epoch >= _I64_MAX
            or type(journal_sequence) is not int
            or journal_sequence < 0
            or journal_sequence >= _I64_MAX
            or type(last_transfer_sequence) is not int
            or last_transfer_sequence < 0
            or last_transfer_sequence >= _I64_MAX
            or type(last_candidate_bank_sequence) is not int
            or last_candidate_bank_sequence < 0
            or last_candidate_bank_sequence >= _I64_MAX
            or type(next_candidate_identity) is not int
            or next_candidate_identity < 1
            or next_candidate_identity >= _I64_MAX
            or type(last_global_drain_sequence) is not int
            or last_global_drain_sequence < 0
            or last_global_drain_sequence >= _I64_MAX
            or type(last_global_update_index) is not int
            or last_global_update_index < 0
            or last_global_update_index >= _I64_MAX
            or type(last_global_completed_environment_steps) is not int
            or last_global_completed_environment_steps < 0
            or last_global_completed_environment_steps >= _I64_MAX
            or type(last_global_ack_mutation_version) is not int
            or last_global_ack_mutation_version < 0
            or last_global_ack_mutation_version >= _I64_MAX
            or type(last_global_terminal_resolution_total) is not int
            or last_global_terminal_resolution_total < 0
            or last_global_terminal_resolution_total >= _I64_MAX
            or type(last_global_policy_opportunity_total) is not int
            or last_global_policy_opportunity_total < 0
            or last_global_policy_opportunity_total
            > last_global_terminal_resolution_total
            or type(mutation_version_host) is not int
            or mutation_version_host < 0
            or mutation_version_host >= _I64_MAX
        ):
            raise DeviceR05ConflictError("checkpoint scalar binding differs")

        # The authority may retain the dataclass it returned.  Snapshot every
        # byte before the owner constructor invokes genesis or other external
        # callbacks, so those callbacks cannot rewrite the restore after-image.
        tensor_names = (
            "next_outcome_identity",
            "next_ball_identity",
            "rng_lo",
            "rng_hi",
            "draw_count",
            "target_generation",
            "previous_cell_index",
            "reset_generation",
            "scheduled_ordinal",
            "outcome_shot_index",
            "sequence_kind",
            "task_identity",
            "outcome_identity",
            "ball_identity",
            "policy_opportunity",
            "mutation_version",
        )
        tensor_sources = {
            name: getattr(checkpoint, name) for name in tensor_names
        }
        tensor_snapshot: dict[str, torch.Tensor] = {}
        for name, source in (
            ("next_outcome_identity", next_outcome_identity_source),
            ("next_ball_identity", next_ball_identity_source),
        ):
            tensor_snapshot[name] = _require_tensor(
                source,
                label=f"checkpoint.{name}",
                device=owned_profile.device,
                dtype=torch.int64,
                shape=(),
            ).clone()
        for name in (
            "rng_lo",
            "rng_hi",
            "draw_count",
            "target_generation",
            "previous_cell_index",
            "reset_generation",
            "scheduled_ordinal",
            "outcome_shot_index",
            "sequence_kind",
            "task_identity",
            "outcome_identity",
            "ball_identity",
        ):
            tensor_snapshot[name] = _require_tensor(
                tensor_sources[name],
                label=f"checkpoint.{name}",
                device=owned_profile.device,
                dtype=torch.int64,
                shape=(num_envs,),
            ).clone()
        tensor_snapshot["policy_opportunity"] = _require_tensor(
            tensor_sources["policy_opportunity"],
            label="checkpoint.policy_opportunity",
            device=owned_profile.device,
            dtype=torch.bool,
            shape=(num_envs,),
        ).clone()
        tensor_snapshot["mutation_version"] = _require_tensor(
            tensor_sources["mutation_version"],
            label="checkpoint.mutation_version",
            device=owned_profile.device,
            dtype=torch.int64,
            shape=(),
        ).clone()
        if (
            not torch.equal(
                tensor_snapshot["mutation_version"],
                torch.clamp(
                    tensor_snapshot["mutation_version"],
                    min=0,
                    max=_I64_MAX - 1,
                ),
            )
            or not torch.equal(
                tensor_snapshot["draw_count"],
                torch.clamp(
                    tensor_snapshot["draw_count"], min=0, max=_I64_MAX - 1
                ),
            )
            or not torch.equal(
                tensor_snapshot["target_generation"],
                torch.clamp(
                    tensor_snapshot["target_generation"],
                    min=0,
                    max=_I64_MAX - 1,
                ),
            )
            or not torch.equal(
                tensor_snapshot["reset_generation"],
                torch.clamp(
                    tensor_snapshot["reset_generation"],
                    min=0,
                    max=_I64_MAX - 1,
                ),
            )
            or not bool(
                (tensor_snapshot["next_outcome_identity"] >= 1)
                & (tensor_snapshot["next_outcome_identity"] < _I64_MAX)
                & (tensor_snapshot["next_ball_identity"] >= 1)
                & (tensor_snapshot["next_ball_identity"] < _I64_MAX)
            )
        ):
            raise DeviceR05ConflictError(
                "checkpoint device chronology has no int64 continuation"
            )
        outcome_identity = tensor_snapshot["outcome_identity"]
        ball_identity = tensor_snapshot["ball_identity"]
        next_outcome_identity = tensor_snapshot["next_outcome_identity"]
        next_ball_identity = tensor_snapshot["next_ball_identity"]
        if (
            not torch.equal(outcome_identity, ball_identity)
            or not torch.equal(next_outcome_identity, next_ball_identity)
            or not bool(
                torch.all(
                    outcome_identity.eq(-1)
                    | (
                        outcome_identity.gt(0)
                        & outcome_identity.lt(next_outcome_identity)
                    )
                )
            )
            or not bool(
                torch.all(
                    ball_identity.eq(-1)
                    | (
                        ball_identity.gt(0)
                        & ball_identity.lt(next_ball_identity)
                    )
                )
            )
        ):
            raise DeviceR05ConflictError(
                "checkpoint live identity differs from its owner high-water"
            )
        for identity in (outcome_identity, ball_identity):
            positive = identity[identity.gt(0)]
            if positive.numel() != torch.unique(positive).numel():
                raise DeviceR05ConflictError(
                    "checkpoint reuses one live owner identity"
                )
        checkpoint = DeviceR05Checkpoint(
            profile_sha256=profile_sha256,
            profile_binding_sha256=profile_binding_sha256,
            seed=seed,
            num_envs=num_envs,
            journal_capacity=journal_capacity,
            max_reveal_epochs_per_drain=max_reveal_epochs_per_drain,
            internal_question_redraw_rounds=(
                internal_question_redraw_rounds
            ),
            epoch=epoch,
            journal_sequence=journal_sequence,
            last_transfer_sequence=last_transfer_sequence,
            last_candidate_bank_sequence=last_candidate_bank_sequence,
            next_candidate_identity=next_candidate_identity,
            last_global_drain_sequence=last_global_drain_sequence,
            last_global_update_index=last_global_update_index,
            last_global_completed_environment_steps=(
                last_global_completed_environment_steps
            ),
            last_global_ack_mutation_version=last_global_ack_mutation_version,
            last_global_terminal_resolution_total=(
                last_global_terminal_resolution_total
            ),
            last_global_policy_opportunity_total=(
                last_global_policy_opportunity_total
            ),
            mutation_version_host=mutation_version_host,
            **tensor_snapshot,
        )
        if not torch.equal(
            checkpoint.mutation_version,
            torch.full_like(
                checkpoint.mutation_version, checkpoint.mutation_version_host
            ),
        ) or checkpoint.last_global_ack_mutation_version != checkpoint.mutation_version_host:
            raise DeviceR05ConflictError(
                "checkpoint host/device mutation chronology differs"
            )
        owner = cls(
            profile_authority,
            profile_receipt,
            seed=checkpoint.seed,
            num_envs=checkpoint.num_envs,
            journal_capacity=checkpoint.journal_capacity,
            max_reveal_epochs_per_drain=(
                checkpoint.max_reveal_epochs_per_drain
            ),
            genesis_authority=genesis_authority,
            genesis_receipt=genesis_receipt,
            cadence_authority=cadence_authority,
            question_authority=question_authority,
            reveal_boundary_authority=reveal_boundary_authority,
            child_completion_authorities=child_completion_authorities,
            drain_authority=drain_authority,
            true_reset_authority=true_reset_authority,
        )
        # The public constructor deliberately performs the only supported
        # profile installation path.  Restore snapshots the authority once
        # before the other construction callbacks and the constructor reads it
        # once more; require exact content equality so a mutable authority
        # cannot win a callback-order TOCTOU.  There is no caller-visible
        # ``_owned_profile`` bypass.
        if (
            owner._profile.profile_sha256 != owned_profile.profile_sha256
            or owner._profile.profile_binding_sha256
            != owned_profile.profile_binding_sha256
            or owner._profile.cell_ids != owned_profile.cell_ids
            or owner._profile.semantic_sha256s != owned_profile.semantic_sha256s
            or not torch.equal(
                owner._profile.targets_xy_m, owned_profile.targets_xy_m
            )
        ):
            owner._poison(14)
            raise DeviceR05ConflictError(
                "profile authority changed during checkpoint restore construction"
            )
        # A cold process necessarily has new opaque Python identities.  The
        # checkpoint therefore carries no raw world-reset object.  Bind the
        # new process to the durable numeric ledger after-image here; a future
        # portable world-root authority remains an explicit integration HOLD.
        if not torch.equal(
            owner._genesis_reset_generations,
            checkpoint.reset_generation,
        ):
            owner._poison(19)
            raise DeviceR05ConflictError(
                "checkpoint reset generation differs from fresh genesis authority"
            )
        for name in (
            "next_outcome_identity",
            "next_ball_identity",
            "rng_lo",
            "rng_hi",
            "draw_count",
            "target_generation",
            "previous_cell_index",
            "reset_generation",
            "scheduled_ordinal",
            "outcome_shot_index",
            "sequence_kind",
            "task_identity",
            "outcome_identity",
            "ball_identity",
            "policy_opportunity",
            "mutation_version",
        ):
            target = getattr(owner, f"_{name}")
            source = _require_tensor(
                getattr(checkpoint, name),
                label=f"checkpoint.{name}",
                device=owner.device,
                dtype=target.dtype,
                shape=tuple(target.shape),
            )
            target.copy_(source)
        owner._epoch = checkpoint.epoch
        owner._journal_head = checkpoint.journal_sequence
        owner._journal_tail = checkpoint.journal_sequence
        owner._last_transfer_sequence = checkpoint.last_transfer_sequence
        owner._last_candidate_bank_sequence = (
            checkpoint.last_candidate_bank_sequence
        )
        owner._next_candidate_identity = checkpoint.next_candidate_identity
        owner._last_global_drain_sequence = checkpoint.last_global_drain_sequence
        owner._last_global_update_index = checkpoint.last_global_update_index
        owner._last_global_completed_environment_steps = (
            checkpoint.last_global_completed_environment_steps
        )
        owner._last_global_ack_mutation_version = (
            checkpoint.last_global_ack_mutation_version
        )
        owner._last_global_terminal_resolution_total = (
            checkpoint.last_global_terminal_resolution_total
        )
        owner._last_global_policy_opportunity_total = (
            checkpoint.last_global_policy_opportunity_total
        )
        owner._checkpoint_requires_global_drain_ack = False
        owner._mutation_version_host = checkpoint.mutation_version_host
        owner._construction_window_open = False
        return owner


def construct_action_ball_full_mdp_device_r05(
    profile_authority: object,
    profile_receipt: object,
    *,
    seed: int,
    genesis_authority: object,
    genesis_receipt: object,
    cadence_authority: object,
    question_authority: object,
    epoch_owner: object,
    motion_owner: object,
    racket_owner: object,
    physical_owner: object,
    journal_capacity: int = 64,
    max_reveal_epochs_per_drain: int = 64,
) -> DeviceR05Owner:
    """Cold-construct the exact generic-N lean D05/ActionEpoch writer chain.

    This diagnostic constructor is intentionally separate from the legacy
    receipt graph.  Every retained source is an exact code-owned class and the
    epoch binds the only Motion -> Racket -> D05 callback order once.
    """

    epoch = _require_action_epoch_module()
    profile = importlib.import_module("action_ball_device_profile_authority")
    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    cadence = importlib.import_module("action_ball_motion_cadence_device")
    question = importlib.import_module(
        "action_ball_full_mdp_canary_question_owner"
    )
    commands = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.commands"
    )
    hope_commands = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.hope_commands"
    )
    physical = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_physical_flight_device"
    )
    post_d05_publisher = getattr(
        motion_owner,
        "publish_action_ball_full_mdp_post_d05_observation",
        None,
    )
    if (
        type(profile_authority) is not profile.DeviceProfileAuthorityOwner
        or type(profile_receipt) is not profile.DeviceProfileReceipt
        or type(genesis_authority)
        is not genesis.ActionBallFullMdpResetGenesisAuthority
        or type(genesis_receipt)
        is not genesis.ActionBallFullMdpResetGenesisReceipt
        or type(cadence_authority) is not cadence.ActionBallMotionCadenceAuthority
        or type(question_authority)
        is not question.RecurringD05InternalQuestionBundle
        or type(epoch_owner) is not epoch.ActionEpochOwner
        or type(motion_owner) is not commands.MotionCommand
        or type(racket_owner) is not hope_commands.RacketTargetCommand
        or type(physical_owner) is not physical.ActionBallPhysicalFlightDeviceOwner
        or not callable(post_d05_publisher)
        or getattr(post_d05_publisher, "__self__", None) is not motion_owner
        or getattr(post_d05_publisher, "__func__", None)
        is not commands.MotionCommand.publish_action_ball_full_mdp_post_d05_observation
        or type(seed) is not int
        or seed != 20260804
        or type(journal_capacity) is not int
        or type(max_reveal_epochs_per_drain) is not int
        or journal_capacity != 64
        or max_reveal_epochs_per_drain != 64
    ):
        raise DeviceR05Error("diagnostic lean D05 constructor inputs differ")
    num_envs = epoch_owner.num_envs
    if type(num_envs) is not int or num_envs < 1:
        raise DeviceR05Error("ActionEpoch num_envs is not a positive exact int")
    owner = DeviceR05Owner(
        profile_authority,
        profile_receipt,
        seed=seed,
        num_envs=num_envs,
        journal_capacity=journal_capacity,
        max_reveal_epochs_per_drain=max_reveal_epochs_per_drain,
        genesis_authority=genesis_authority,
        genesis_receipt=genesis_receipt,
        cadence_authority=cadence_authority,
        question_authority=question_authority,
        reveal_boundary_authority=None,
        child_completion_authorities=(),
        diagnostic_epoch_owner=epoch_owner,
    )
    try:
        epoch_owner.bind_motion_cadence_owner(cadence_authority)
        motion_owner.bind_action_ball_continuous_motion_device_r05_reveal(owner)
        motion_owner.bind_action_ball_full_mdp_motion_epoch_owner(epoch_owner)
        racket_owner.bind_action_ball_full_mdp_racket_epoch_sources(
            owner, epoch_owner
        )
        physical_owner.bind_action_epoch_owner(
            epoch_owner,
            device_r05_owner=owner,
            motion_owner=motion_owner,
            racket_owner=racket_owner,
        )
        owner._diagnostic_motion_owner = motion_owner
        owner._diagnostic_racket_owner = racket_owner
        owner._diagnostic_physical_owner = physical_owner
        epoch_owner.bind_d05_accept_writers(
            motion_write=owner._commit_action_epoch_motion_write,
            racket_write=owner._commit_action_epoch_racket_write,
            r05_write=owner._commit_action_epoch_r05_write,
        )
    except BaseException as exc:
        owner._poison(26)
        poison_physical = getattr(
            physical_owner, "poison_global_reveal_epoch", None
        )
        if callable(poison_physical):
            try:
                poison_physical(
                    "diagnostic lean D05 construction failed after Physical bind began"
                )
            except BaseException:
                pass
        raise DeviceR05PoisonedError(
            "diagnostic lean D05 construction failed after writer binding began"
        ) from exc
    return owner


__all__ = [
    "CHILD_OWNER_ORDER",
    "GENESIS_CONSUMER_ORDER",
    "GLOBAL_DRAIN_FIELD_NAMES",
    "GLOBAL_DRAIN_OWNER_KIND",
    "DECISION_ACCEPT",
    "DECISION_CENSOR",
    "DECISION_CONSTRUCTION_REJECT",
    "DECISION_DEFER",
    "DeviceInternalQuestionCompositionAuthority",
    "DeviceQuestionAuthority",
    "DeviceQuestionChronology",
    "DeviceQuestionProjection",
    "DeviceCheckpointAuthority",
    "DeviceChildCompletionAuthority",
    "DeviceChildCompletionProjection",
    "DeviceDrainAckProjection",
    "DeviceDrainAuthority",
    "DeviceGenesisAuthority",
    "DeviceGenesisProjection",
    "DeviceR05GenesisProjection",
    "DeviceR05GenesisView",
    "DeviceR05ArmedTerminal",
    "DeviceR05AcceptedRowsView",
    "DeviceR05CandidateBank",
    "DeviceR05Checkpoint",
    "DeviceR05ConflictError",
    "DeviceR05ConstructionRejection",
    "DeviceR05DrainBatch",
    "DeviceR05DrainView",
    "DeviceR05EnvResetBinding",
    "DeviceR05EnvResetBindingView",
    "DeviceR05Error",
    "DeviceR05Owner",
    "DeviceR05PoisonedError",
    "DeviceR05FullMdpIdentityProductionHold",
    "DeviceR05PreparedToken",
    "DeviceR05RowTransaction",
    "DeviceR05PreparedTrueReset",
    "DeviceR05PreparedTrueResetProjection",
    "DeviceR05PreviewToken",
    "DeviceR05PreTransferBoundaryToken",
    "DeviceR05TerminalClaim",
    "DeviceR05TerminalClaimProjection",
    "DeviceR05TerminalReceipt",
    "DeviceR05TerminalReceiptProjection",
    "DeviceR05TrueResetReceipt",
    "materialize_pre_optimizer_ppo_boundary_leaf_schema",
    "DeviceRevealBoundaryAuthority",
    "DeviceRevealBoundaryProjection",
    "DeviceTerminalArmProjection",
    "DeviceTerminalCommitProjection",
    "DeviceTrueResetAbortProjection",
    "DeviceTrueResetAuthority",
    "DeviceTrueResetChildCompletionProjection",
    "DeviceTrueResetCommitProjection",
    "DeviceTrueResetPreflightProjection",
    "construct_action_ball_full_mdp_device_r05",
    "DeviceTrueResetEventProjection",
    "INTEGRATION_STATUS",
    "JOURNAL_ABORT",
    "JOURNAL_ACCEPT",
    "JOURNAL_CENSOR",
    "JOURNAL_CONSTRUCTION_REJECT",
    "JOURNAL_TRUE_RESET",
    "MOTION_TASK_F32_FIELDS",
    "PHYSICAL_STATE_F32_FIELDS",
    "RACKET_F32_FIELDS",
    "QUESTION_CONSTRUCTION_REASON_ADMITTED",
    "QUESTION_CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL",
    "QUESTION_CONSTRUCTION_REASON_MIN_REJECT",
    "QUESTION_CONSTRUCTION_REASON_MAX_REJECT",
    "INTERNAL_QUESTION_REDRAW_ROUNDS",
    "INTERNAL_QUESTION_TOTAL_DRAW_WIDTH",
    "PRODUCER_FAULT_QUESTION_CHRONOLOGY",
    "REASON_ADMISSIBLE",
    "REASON_ABORTED_BEFORE_TRANSFER",
    "REASON_BATCH_PEER_INFEASIBLE",
    "REASON_NO_FEASIBLE_TARGET",
    "REASON_ONLY_PREVIOUS_TARGET_FEASIBLE",
    "REASON_TRUE_RESET",
]
