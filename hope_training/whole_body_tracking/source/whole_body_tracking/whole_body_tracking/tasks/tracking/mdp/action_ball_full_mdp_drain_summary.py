"""One-pass, CPU-only decoder for one frozen ActionEpoch PPO suffix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    import action_ball_full_mdp_row_identity as row_identity
except ImportError:
    from whole_body_tracking import action_ball_full_mdp_row_identity as row_identity

try:
    from . import action_ball_full_mdp_milestone_tensors as milestone_tensors
except ImportError:
    import action_ball_full_mdp_milestone_tensors as milestone_tensors

try:
    from . import action_ball_full_mdp_action_strata as action_strata
except ImportError:
    import action_ball_full_mdp_action_strata as action_strata

try:
    from . import action_ball_full_mdp_portable_catalog as portable_catalog
except ImportError:
    import action_ball_full_mdp_portable_catalog as portable_catalog

ActionEpochShotKey = row_identity.ActionEpochShotKey


DRAIN_SCHEMA_VERSION = 11
DRAIN_SUMMARY_KIND = "action_ball_epoch_ppo_boundary_summary_v11"
LIFECYCLE_PAYMENT_COUNT = 14
SHOT_KEY_FIELDS = (
    "reset_generation",
    "ball_generation",
    "action_uid",
    "action_slot",
    "shot_index",
    "task_identity",
    "outcome_identity",
    "ball_identity",
)
SHOT_EVENT_PREFIX = (
    "event_mask",
    *("shot_key." + field for field in SHOT_KEY_FIELDS),
)

D05_DECISION_NONE = 0
D05_DECISION_ACCEPT = 1
D05_DECISION_CENSOR = 2
D05_DECISION_REJECT = 3
D05_DECISION_DEFER = 4

MOTION_CLOSE_PLAYED_SUFFIX = 1
MOTION_CLOSE_UNPLAYED = 2

_B, _I = torch.bool, torch.int64
_SHOT_PREFIX_SPEC = (("event_mask", _B, "shot"),) + tuple(
    ("shot_key." + field, _I, "shot") for field in SHOT_KEY_FIELDS
)
_SHOT_REMAINDERS = {
    "D05_SETTLED": (
        ("due_mask", _B, "shot"),
        ("selected_mask", _B, "shot"),
        ("decision", _I, "shot"),
        ("construction_admissible", _B, "shot"),
        ("playback_admissible", _B, "shot"),
        ("owner_fault_bits", _I, "fault7"),
        ("stroke_family_code", _I, "shot"),
        ("action_attribution_valid", _B, "shot"),
    ),
    "D05_ACCEPT_PUBLISHED": (("publication_ordinal", _I, "shot"),),
    "MOTION_PLAYBACK_STARTED": (),
    "MOTION_CLOSED": (("motion_close_reason", _I, "shot"),),
    "PHYSICAL_LAUNCH_ROWS": (
        ("publication_ordinal", _I, "shot"),
        ("launch_succeeded", _B, "shot"),
        ("late_launch", _B, "shot"),
        ("owner_fault_bits", _I, "shot"),
        ("target_xy_m", torch.float32, "target_xy"),
    ),
    "PHYSICAL_POSTPHYSICS_ROWS": (
        ("publication_ordinal", _I, "shot"),
        ("owner_fault_bits", _I, "shot"),
        ("fact_valid_bits", _I, "shot"),
        ("fact_source_step", _I, "shot"),
    ),
    "R06_OUTCOME_ROWS": (
        ("publication_ordinal", _I, "shot"),
        ("settlement_step", _I, "shot"),
        ("valid_bits", _I, "shot"),
        ("outcome_code", _I, "shot"),
        ("owner_fault_bits", _I, "shot"),
        ("predicate_bits", _I, "shot"),
    ),
    "R07_FIRST_READY": (("source_step", _I, "shot"),),
    "PAYMENT_RECORDED": (("payment_step", _I, "shot"),),
    "RETIRED": (
        ("motion_close_reason", _I, "shot"),
        ("payment_step", _I, "shot"),
        ("retirement_step", _I, "shot"),
    ),
}
_NONSHOT = {
    "RESET_GENESIS_IDLE": (
        ("reset_generation", _I, "env"),
        ("reset_selected_mask", _B, "env"),
    ),
    "RESET_SELECTED": (
        ("selected_mask", _B, "env"),
        ("reset_generation", _I, "env"),
        ("terminal_reset_facts_i64", _I, "reset_fact"),
    ),
    "REWARD_CYCLE_OPEN": (
        ("reward_cycle_age", _I, "env"),
        ("reward_cycle_fault", _I, "env"),
    ),
    "REWARD_CONSUMER_PAID": (
        ("reward_consumer_ordinal", _I, "env"),
        ("reward_paid", _B, "reward"),
    ),
}
_WRITERS = (
    "WRITES_STARTED:motion",
    "WRITES_COMMITTED:motion",
    "WRITES_STARTED:racket",
    "WRITES_COMMITTED:racket",
    "WRITES_STARTED:r05_runtime",
    "WRITES_COMMITTED:r05_runtime",
    "D05_ACCEPT_PUBLISHED",
)
_OWNERS = frozenset(
    (
        "r05_runtime",
        "motion",
        "racket",
        "physical_ball",
        "r06_landing_outcome",
        "r03_strike_fact",
        "r07_recovery",
    )
)

SHOT_LIFECYCLE_FLAGS = (
    "reveal_committed",
    "playback_started",
    "motion_closed",
    "physical_launched",
    "outcome_settled",
    "payment_recorded",
)
SHOT_LIFECYCLE_BITS = {
    name: 1 << ordinal for ordinal, name in enumerate(SHOT_LIFECYCLE_FLAGS)
}
SHOT_LIFECYCLE_ALLOWED_MASK = (1 << len(SHOT_LIFECYCLE_FLAGS)) - 1

_R06_CONTACT_VALID = 1
_R06_NET_CROSSED = 2
_R06_NET_CLEAR = 4
_R06_CROSSING_VALID = 8
_R06_ON_TABLE = 16
_R06_COMMON_LEGAL = 32


class ActionEpochDrainDecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class D05SettlementCounts:
    transactions: int
    due_rows: int
    selected_rows: int
    accepted: int
    censored: int
    rejected: int
    deferred: int
    not_ready: int


@dataclass(frozen=True)
class RevealCommitCounts:
    motion_committed_rows: int
    racket_committed_rows: int
    r05_committed_rows: int


@dataclass(frozen=True)
class LifecycleEdgeCounts:
    playback_started_rows: int
    closed_unplayed_rows: int
    physical_launch_rows: int
    outcome_settled_rows: int
    payment_recorded_rows: int
    retired_rows: int
    terminal_shot_rows: int = 0


@dataclass(frozen=True)
class OwnerFaultCounts:
    attributed_fault_rows: int


@dataclass(frozen=True)
class ContinuationCounts:
    active_before: int
    active_after: int
    awaiting_playback_after: int
    awaiting_outcome_after: int
    awaiting_payment_after: int


@dataclass(frozen=True)
class EpochDrainFrontier:
    schema_version: int
    kind: str
    num_envs: int
    shot_slot_capacity: int
    device_type: str
    device_index: Optional[int]
    update_index: int
    next_update_index: int
    completed_environment_steps: int
    operation_sequence: int
    drain_sequence: int
    start_commit: int
    end_commit: int
    due_terminal_overlap_rows: int
    diagnostic_unauthorized: bool = True


@dataclass(frozen=True)
class ActionEpochShotEvidence:
    lifecycle_bits: int
    r03_valid_bits: int
    r03_source_step: int
    physical_valid_bits: int
    physical_actor_pair_contact_source_step: int
    r06_valid_bits: int
    r06_outcome_code: int
    r06_predicate_bits: int
    r07_valid_bits: int
    r07_qualified_source_step: int
    r07_first_ready_source_step: int


@dataclass(frozen=True)
class _ActionEpochShotSnapshot:
    env_row: int
    slot_index: int
    reset_generation: int
    ball_generation: int
    action_uid: int
    action_slot: int
    shot_index: int
    task_identity: int
    outcome_identity: int
    ball_identity: int
    target_x_m: float
    target_y_m: float
    motion_close_reason: int
    settlement_step: int
    payment_step: int
    stroke_family: str = "unknown"
    action_attribution_valid: bool = False
    evidence: ActionEpochShotEvidence | None = None


@dataclass(frozen=True)
class CompletedActionEpochShot(_ActionEpochShotSnapshot):
    retirement_step: int = -1


@dataclass(frozen=True)
class TerminalActionEpochShot(_ActionEpochShotSnapshot):
    reset_generation_after: int = -1
    reset_common_step: int = -1
    reset_episode_tick: int = -1
    reset_reason_bits: int = 0


@dataclass(frozen=True)
class D05ActionOpportunity:
    env_row: int
    slot_index: int
    action_uid: int
    action_slot: int
    stroke_family: str
    attribution_valid: bool
    selected: bool
    accepted: bool
    censored: bool
    rejected: bool
    deferred: bool


@dataclass(frozen=True)
class ResetTelemetry:
    env_row: int
    reset_generation: int
    common_step: int
    episode_tick: int
    reason_bits: int


RESET_TELEMETRY_EXAMPLE_LIMIT = 8


@dataclass(frozen=True)
class ResetTelemetryAggregate:
    row_count: int
    episode_length_sum: int
    reason_bit_counts: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ActionEpochPpoBoundarySummary:
    frontier: EpochDrainFrontier
    settlement: D05SettlementCounts
    reveal_commit: RevealCommitCounts
    lifecycle: LifecycleEdgeCounts
    owner_faults: OwnerFaultCounts
    continuation: ContinuationCounts
    action_opportunities: tuple[D05ActionOpportunity, ...]
    completed_shots: tuple[CompletedActionEpochShot, ...]
    terminal_shots: tuple[TerminalActionEpochShot, ...]
    terminal_resets: tuple[ResetTelemetry, ...]
    terminal_reset_aggregate: ResetTelemetryAggregate
    milestone: milestone_tensors.MilestoneWindowTelemetry


@dataclass(frozen=True)
class ActionEpochDrainContinuation:
    occupied: torch.Tensor
    key: ActionEpochShotKey
    reveal_committed: torch.Tensor
    playback_started: torch.Tensor
    motion_closed: torch.Tensor
    motion_close_reason: torch.Tensor
    physical_launch_requested: torch.Tensor
    physical_launched: torch.Tensor
    physical_target_xy_m: torch.Tensor
    outcome_settled: torch.Tensor
    payment_recorded: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor
    stroke_family_code: torch.Tensor
    action_attribution_valid: torch.Tensor
    r03_valid_bits: torch.Tensor
    r03_source_step: torch.Tensor
    physical_valid_bits: torch.Tensor
    physical_actor_pair_contact_source_step: torch.Tensor
    r06_valid_bits: torch.Tensor
    r06_outcome_code: torch.Tensor
    r06_predicate_bits: torch.Tensor
    r07_valid_bits: torch.Tensor
    r07_qualified_source_step: torch.Tensor
    r07_first_ready_source_step: torch.Tensor

    @classmethod
    def empty(cls, *, num_envs: int, shot_slot_capacity: int):
        if (
            type(num_envs) is not int
            or num_envs < 1
            or type(shot_slot_capacity) is not int
            or shot_slot_capacity < 1
        ):
            raise ActionEpochDrainDecodeError("continuation shape differs")
        shape = (num_envs, shot_slot_capacity)
        boolean = lambda: torch.zeros(shape, dtype=_B)
        integer = lambda: torch.full(shape, -1, dtype=_I)
        return cls(
            occupied=boolean(),
            key=ActionEpochShotKey(
                **{field: integer() for field in SHOT_KEY_FIELDS}
            ),
            reveal_committed=boolean(),
            playback_started=boolean(),
            motion_closed=boolean(),
            motion_close_reason=integer(),
            physical_launch_requested=boolean(),
            physical_launched=boolean(),
            physical_target_xy_m=torch.zeros((*shape, 2), dtype=torch.float32),
            outcome_settled=boolean(),
            payment_recorded=boolean(),
            settlement_step=integer(),
            payment_step=integer(),
            stroke_family_code=torch.zeros(shape, dtype=_I),
            action_attribution_valid=boolean(),
            r03_valid_bits=torch.zeros(shape, dtype=_I),
            r03_source_step=integer(),
            physical_valid_bits=torch.zeros(shape, dtype=_I),
            physical_actor_pair_contact_source_step=integer(),
            r06_valid_bits=integer(),
            r06_outcome_code=integer(),
            r06_predicate_bits=torch.zeros(shape, dtype=_I),
            r07_valid_bits=torch.zeros(shape, dtype=_I),
            r07_qualified_source_step=integer(),
            r07_first_ready_source_step=integer(),
        )

    def clone(self):
        return ActionEpochDrainContinuation(
            **{
                field: getattr(self, field).clone()
                for field in (
                    "occupied",
                    "reveal_committed",
                    "playback_started",
                    "motion_closed",
                    "motion_close_reason",
                    "physical_launch_requested",
                    "physical_launched",
                    "physical_target_xy_m",
                    "outcome_settled",
                    "payment_recorded",
                    "settlement_step",
                    "payment_step",
                    "stroke_family_code",
                    "action_attribution_valid",
                    "r03_valid_bits",
                    "r03_source_step",
                    "physical_valid_bits",
                    "physical_actor_pair_contact_source_step",
                    "r06_valid_bits",
                    "r06_outcome_code",
                    "r06_predicate_bits",
                    "r07_valid_bits",
                    "r07_qualified_source_step",
                    "r07_first_ready_source_step",
                )
            },
            key=ActionEpochShotKey(
                **{
                    field: getattr(self.key, field).clone()
                    for field in SHOT_KEY_FIELDS
                }
            ),
        )


@dataclass(frozen=True)
class DecodedEpochDrain:
    settlement: D05SettlementCounts
    reveal_commit: RevealCommitCounts
    lifecycle: LifecycleEdgeCounts
    owner_faults: OwnerFaultCounts
    continuation: ContinuationCounts
    action_opportunities: tuple[D05ActionOpportunity, ...]
    completed_shots: tuple[CompletedActionEpochShot, ...]
    terminal_shots: tuple[TerminalActionEpochShot, ...]
    terminal_resets: tuple[ResetTelemetry, ...]
    terminal_reset_aggregate: ResetTelemetryAggregate
    due_terminal_overlap_rows: int
    milestone: milestone_tensors.MilestoneWindowTelemetry
    next_continuation: ActionEpochDrainContinuation

    def with_frontier(self, frontier: EpochDrainFrontier):
        if type(frontier) is not EpochDrainFrontier:
            raise ActionEpochDrainDecodeError("frontier type differs")
        return ActionEpochPpoBoundarySummary(
            frontier,
            self.settlement,
            self.reveal_commit,
            self.lifecycle,
            self.owner_faults,
            self.continuation,
            self.action_opportunities,
            self.completed_shots,
            self.terminal_shots,
            self.terminal_resets,
            self.terminal_reset_aggregate,
            self.milestone,
        )


@dataclass
class _PendingD05:
    mask: torch.Tensor
    key: ActionEpochShotKey
    publication: int
    next_writer: int = 0


def _shape(kind: str, shot: tuple[int, int]):
    return {
        "shot": shot,
        "fault7": (*shot, 7),
        "target_xy": (*shot, 2),
        "crossing_xy": (*shot, 2),
        "env": (shot[0],),
        "reset_fact": (shot[0], 3),
        "reward": (shot[0], 14),
    }[kind]


def _unpack(entry: object, specs: tuple, shot_shape: tuple[int, int]):
    delta = getattr(entry, "delta", None)
    names = getattr(delta, "names", None)
    values = getattr(delta, "values", None)
    expected = tuple(name for name, _dtype, _kind in specs)
    if type(names) is not tuple or names != expected or type(values) is not tuple:
        raise ActionEpochDrainDecodeError(
            str(getattr(entry, "transition", "<missing>")) + " schema differs"
        )
    result = {}
    for (name, dtype, kind), value in zip(specs, values):
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != _shape(kind, shot_shape)
            or not value.is_contiguous()
        ):
            raise ActionEpochDrainDecodeError(name + " tensor ABI differs")
        result[name] = value
    return result


def _shot(entry: object, remainder: tuple, shape: tuple[int, int]):
    data = _unpack(entry, _SHOT_PREFIX_SPEC + remainder, shape)
    key = ActionEpochShotKey(
        **{field: data["shot_key." + field] for field in SHOT_KEY_FIELDS}
    )
    _require_key(key, shape, "journal.shot_key")
    return data, data["event_mask"], key


def _require_key(key, shape, label):
    try:
        return row_identity.require_action_epoch_shot_key(
            key, shape=shape, device="cpu", label=label
        )
    except row_identity.ActionEpochShotKeyError as exc:
        raise ActionEpochDrainDecodeError(str(exc)) from exc


def _valid(key: ActionEpochShotKey):
    return row_identity.action_epoch_shot_key_valid(key)


def _keys_equal(left: ActionEpochShotKey, right: ActionEpochShotKey, mask):
    return bool(
        torch.all(
            ~mask | row_identity.action_epoch_shot_key_equal(left, right)
        ).item()
    )


def _current(state: ActionEpochDrainContinuation, mask, key, transition):
    if bool(torch.any(mask & ~state.occupied).item()) or not _keys_equal(
        state.key, key, mask
    ):
        raise ActionEpochDrainDecodeError(
            transition + " does not match the active full shot key"
        )


def _clear(state: ActionEpochDrainContinuation, mask):
    state.occupied[mask] = False
    for field in SHOT_KEY_FIELDS:
        getattr(state.key, field)[mask] = -1
    for field in (
        "reveal_committed",
        "playback_started",
        "motion_closed",
        "physical_launch_requested",
        "physical_launched",
        "outcome_settled",
        "payment_recorded",
        "action_attribution_valid",
    ):
        getattr(state, field)[mask] = False
    for field in ("motion_close_reason", "settlement_step", "payment_step"):
        getattr(state, field)[mask] = -1
    state.physical_target_xy_m[mask] = 0.0
    state.stroke_family_code[mask] = action_strata.STROKE_FAMILY_UNKNOWN
    for field in (
        "r03_source_step",
        "physical_actor_pair_contact_source_step",
        "r06_valid_bits",
        "r06_outcome_code",
        "r07_qualified_source_step",
        "r07_first_ready_source_step",
    ):
        getattr(state, field)[mask] = -1
    for field in (
        "r03_valid_bits",
        "physical_valid_bits",
        "r06_predicate_bits",
        "r07_valid_bits",
    ):
        getattr(state, field)[mask] = 0


def _activate(
    state: ActionEpochDrainContinuation,
    mask,
    key,
    family,
    attributed,
    launch_requested,
):
    _clear(state, mask)
    for field in SHOT_KEY_FIELDS:
        getattr(state.key, field)[mask] = getattr(key, field)[mask]
    state.stroke_family_code[mask] = family[mask]
    state.action_attribution_valid[mask] = attributed[mask]
    state.physical_launch_requested[mask] = launch_requested[mask]
    state.occupied[mask] = True


def _shot_evidence(
    state: ActionEpochDrainContinuation, env_row: int, slot_index: int
) -> ActionEpochShotEvidence:
    lifecycle_bits = 0
    for field in (
        "reveal_committed",
        "playback_started",
        "motion_closed",
        "physical_launched",
        "outcome_settled",
        "payment_recorded",
    ):
        if bool(getattr(state, field)[env_row, slot_index].item()):
            lifecycle_bits |= SHOT_LIFECYCLE_BITS[field]
    return ActionEpochShotEvidence(
        lifecycle_bits=lifecycle_bits,
        r03_valid_bits=int(state.r03_valid_bits[env_row, slot_index].item()),
        r03_source_step=int(state.r03_source_step[env_row, slot_index].item()),
        physical_valid_bits=int(
            state.physical_valid_bits[env_row, slot_index].item()
        ),
        physical_actor_pair_contact_source_step=int(
            state.physical_actor_pair_contact_source_step[
                env_row, slot_index
            ].item()
        ),
        r06_valid_bits=int(state.r06_valid_bits[env_row, slot_index].item()),
        r06_outcome_code=int(state.r06_outcome_code[env_row, slot_index].item()),
        r06_predicate_bits=int(
            state.r06_predicate_bits[env_row, slot_index].item()
        ),
        r07_valid_bits=int(state.r07_valid_bits[env_row, slot_index].item()),
        r07_qualified_source_step=int(
            state.r07_qualified_source_step[env_row, slot_index].item()
        ),
        r07_first_ready_source_step=int(
            state.r07_first_ready_source_step[env_row, slot_index].item()
        ),
    )


def _shot_snapshot(
    state: ActionEpochDrainContinuation, env_row: int, slot_index: int
) -> dict[str, object]:
    return {
        "env_row": int(env_row),
        "slot_index": int(slot_index),
        **{
            field: int(getattr(state.key, field)[env_row, slot_index].item())
            for field in SHOT_KEY_FIELDS
        },
        "target_x_m": float(
            state.physical_target_xy_m[env_row, slot_index, 0].item()
        ),
        "target_y_m": float(
            state.physical_target_xy_m[env_row, slot_index, 1].item()
        ),
        "motion_close_reason": int(
            state.motion_close_reason[env_row, slot_index].item()
        ),
        "settlement_step": int(state.settlement_step[env_row, slot_index].item()),
        "payment_step": int(state.payment_step[env_row, slot_index].item()),
        "stroke_family": action_strata.STROKE_FAMILY_NAMES[
            int(state.stroke_family_code[env_row, slot_index].item())
        ],
        "action_attribution_valid": bool(
            state.action_attribution_valid[env_row, slot_index].item()
        ),
        "evidence": _shot_evidence(state, env_row, slot_index),
    }


def _event(entry, transition, shape):
    if transition in _SHOT_REMAINDERS:
        return _shot(entry, _SHOT_REMAINDERS[transition], shape)
    for prefix, remainder in (
        ("OWNER_FAULT:", (("owner_fault_bits", _I, "shot"),)),
        (
            "OWNER_FACTS:",
            (
                ("valid_bits", _I, "shot"),
                ("source_step", _I, "shot"),
                ("qualified", _B, "shot"),
            ),
        ),
    ):
        if transition.startswith(prefix) and transition[len(prefix) :] in _OWNERS:
            return _shot(entry, remainder, shape)
    raise ActionEpochDrainDecodeError("epoch drain transition is foreign")


def decode_epoch_drain_suffix(
    entries: tuple[object, ...],
    *,
    start_commit: int,
    end_commit: int,
    previous: ActionEpochDrainContinuation,
    milestone_i64: torch.Tensor,
    milestone_f64: torch.Tensor,
) -> DecodedEpochDrain:
    """Reduce exactly one materialized suffix into counts and carry state."""

    if (
        type(entries) is not tuple
        or type(start_commit) is not int
        or start_commit < 0
        or type(end_commit) is not int
        or end_commit - start_commit != len(entries)
        or type(previous) is not ActionEpochDrainContinuation
        or previous.occupied.ndim != 2
    ):
        raise ActionEpochDrainDecodeError("drain frontier differs")
    try:
        milestone = milestone_tensors.decode_host_window(
            milestone_i64, milestone_f64
        )
    except ValueError as exc:
        raise ActionEpochDrainDecodeError(str(exc)) from exc
    shape = tuple(previous.occupied.shape)
    if shape[0] < 1 or shape[1] < 1:
        raise ActionEpochDrainDecodeError("continuation shape differs")
    # Validate the caller-held continuation with the same CPU tensor ABI.
    probe = ActionEpochDrainContinuation.empty(
        num_envs=shape[0], shot_slot_capacity=shape[1]
    )
    _require_key(previous.key, shape, "continuation.shot_key")
    for field in probe.__dataclass_fields__:
        if field == "key":
            for key_field in SHOT_KEY_FIELDS:
                value = getattr(previous.key, key_field)
                expected = getattr(probe.key, key_field)
                if (
                    type(value) is not torch.Tensor
                    or value.device.type != "cpu"
                    or value.dtype != expected.dtype
                    or tuple(value.shape) != shape
                    or not value.is_contiguous()
                ):
                    raise ActionEpochDrainDecodeError("continuation key ABI differs")
        else:
            value, expected = getattr(previous, field), getattr(probe, field)
            if (
                type(value) is not torch.Tensor
                or value.device.type != "cpu"
                or value.dtype != expected.dtype
                or tuple(value.shape) != tuple(expected.shape)
                or not value.is_contiguous()
            ):
                raise ActionEpochDrainDecodeError("continuation tensor ABI differs")

    state = previous.clone()
    r03_present = state.r03_valid_bits.bitwise_and(1).ne(0)
    physical_present = state.physical_valid_bits.bitwise_and(1).ne(0)
    physical_contact = state.physical_valid_bits.bitwise_and(2).ne(0)
    r06_present = state.r06_valid_bits.ge(0) & state.r06_valid_bits.bitwise_and(1).ne(0)
    r06_eligible = state.r06_valid_bits.ge(0) & state.r06_valid_bits.bitwise_and(2).ne(0)
    r06_source_valid = state.r06_valid_bits.ge(0) & state.r06_valid_bits.bitwise_and(4).ne(0)
    r07_present = state.r07_valid_bits.bitwise_and(1).ne(0)
    r07_qualified = state.r07_qualified_source_step.ge(0)
    r07_ready = state.r07_first_ready_source_step.ge(0)
    invalid_carry = (
        state.r03_valid_bits.lt(0)
        | state.r03_valid_bits.bitwise_and(~3).ne(0)
        | (r03_present ^ state.r03_source_step.ge(0))
        | (state.r03_valid_bits.bitwise_and(2).ne(0) & ~r03_present)
        | state.physical_valid_bits.lt(0)
        | state.physical_valid_bits.bitwise_and(~3).ne(0)
        | (physical_contact ^ state.physical_actor_pair_contact_source_step.ge(0))
        | (physical_contact & ~physical_present)
        | (state.r06_valid_bits.lt(-1) | state.r06_valid_bits.gt(7))
        | (state.outcome_settled ^ state.r06_valid_bits.ge(0))
        | (state.outcome_settled & ~r06_present)
        | (r06_eligible & ~r06_source_valid)
        | (r06_source_valid & ~r06_present)
        | (state.r06_predicate_bits.lt(0) | state.r06_predicate_bits.gt(63))
        | (state.r06_predicate_bits.ne(0) & ~r06_source_valid)
        | (state.r06_predicate_bits.bitwise_and(_R06_NET_CLEAR).ne(0)
           & state.r06_predicate_bits.bitwise_and(_R06_NET_CROSSED).eq(0))
        | (r06_present & (state.r06_outcome_code.lt(1) | state.r06_outcome_code.gt(7)))
        | (~r06_present & state.r06_outcome_code.ne(-1))
        | state.r07_valid_bits.lt(0)
        | state.r07_valid_bits.bitwise_and(~3).ne(0)
        | (state.r07_valid_bits.bitwise_and(2).ne(0) & ~r07_present)
        | (r07_qualified & ~r07_present)
        | (r07_ready & ~r07_qualified)
    )
    if bool(torch.any(invalid_carry).item()):
        raise ActionEpochDrainDecodeError("continuation evidence differs")
    active_before = int(previous.occupied.sum().item())
    count = {
        name: 0
        for name in (
            "transactions due selected accepted censored rejected deferred not_ready "
            "construction motion racket r05 playback unplayed launch outcome payment retired terminal faults"
        ).split()
    }
    pending: Optional[_PendingD05] = None
    reward_ordinal: Optional[int] = None
    opportunities: list[D05ActionOpportunity] = []
    completed: list[CompletedActionEpochShot] = []
    terminal_resets: list[ResetTelemetry] = []
    terminal_reset_row_count = 0
    terminal_reset_episode_length_sum = 0
    terminal_reset_reason_counts = [0, 0, 0, 0, 0]
    due_terminal_overlap_rows = 0
    terminal_shots: list[TerminalActionEpochShot] = []

    for offset, entry in enumerate(entries):
        sequence = getattr(entry, "sequence", None)
        publication = getattr(entry, "epoch", None)
        transition = getattr(entry, "transition", None)
        before = getattr(entry, "before_version", None)
        after = getattr(entry, "after_version", None)
        genesis_transition = transition == "RESET_GENESIS_IDLE"
        if (
            type(sequence) is not int
            or sequence != start_commit + offset
            or type(publication) is not int
            or (genesis_transition and sequence != 0)
            or type(transition) is not str
            or type(before) is not int
            or type(after) is not int
            or after <= before
        ):
            raise ActionEpochDrainDecodeError("journal entry chronology differs")

        if pending is not None:
            expected = _WRITERS[pending.next_writer]
            if transition != expected:
                raise ActionEpochDrainDecodeError("D05 writer order differs")
            remainder = _SHOT_REMAINDERS.get(transition, ())
            data, mask, key = _shot(entry, remainder, shape)
            if (
                publication != pending.publication
                or not torch.equal(mask, pending.mask)
                or not _keys_equal(key, pending.key, mask)
            ):
                raise ActionEpochDrainDecodeError("D05 writer identity differs")
            if transition.startswith("WRITES_COMMITTED:"):
                count[transition.split(":", 1)[1].replace("_runtime", "")] += int(
                    mask.sum().item()
                )
            if transition == "D05_ACCEPT_PUBLISHED":
                ordinal = data["publication_ordinal"]
                if bool(torch.any(mask & ordinal.ne(publication)).item()):
                    raise ActionEpochDrainDecodeError("D05 publication differs")
                state.reveal_committed[mask] = True
            pending.next_writer += 1
            if pending.next_writer == len(_WRITERS):
                pending = None
            continue

        if reward_ordinal is not None:
            if transition != "REWARD_CONSUMER_PAID":
                raise ActionEpochDrainDecodeError("Reward payment order differs")
            data = _unpack(entry, _NONSHOT[transition], shape)
            if (
                not bool(
                    torch.all(
                        data["reward_consumer_ordinal"].eq(reward_ordinal)
                    ).item()
                )
                or not bool(torch.all(data["reward_paid"][:, reward_ordinal]).item())
            ):
                raise ActionEpochDrainDecodeError("Reward payment differs")
            reward_ordinal += 1
            if reward_ordinal == LIFECYCLE_PAYMENT_COUNT:
                reward_ordinal = None
            continue

        if transition in _NONSHOT:
            data = _unpack(entry, _NONSHOT[transition], shape)
            if transition == "REWARD_CONSUMER_PAID":
                raise ActionEpochDrainDecodeError("Reward payment lacks open cycle")
            if transition == "REWARD_CYCLE_OPEN":
                reward_ordinal = 0
            elif transition.startswith("RESET_"):
                selected_rows = data[
                    "selected_mask"
                    if transition == "RESET_SELECTED"
                    else "reset_selected_mask"
                ]
                if transition == "RESET_SELECTED":
                    generations = data["reset_generation"]
                    facts = data["terminal_reset_facts_i64"]
                    common_step = facts[:, 0]
                    episode_tick = facts[:, 1]
                    reason_bits = facts[:, 2]
                    unselected = ~selected_rows
                    invalid = (
                        generations.lt(0)
                        | (
                            unselected
                            & (
                                common_step.ne(-1)
                                | episode_tick.ne(-1)
                                | reason_bits.ne(0)
                            )
                        )
                        | (
                            selected_rows
                            & (
                                generations.lt(1)
                                | common_step.lt(1)
                                | episode_tick.lt(1)
                                | reason_bits.eq(0)
                                | torch.bitwise_and(
                                    reason_bits,
                                    torch.full_like(reason_bits, ~31),
                                ).ne(0)
                            )
                        )
                    )
                    if bool(torch.any(invalid).item()):
                        raise ActionEpochDrainDecodeError(
                            "selected-reset telemetry differs"
                        )
                    selected_env_rows = torch.nonzero(
                        selected_rows, as_tuple=False
                    ).flatten()
                    reset_fact_rows = torch.stack(
                        (
                            selected_env_rows,
                            generations[selected_env_rows],
                            common_step[selected_env_rows],
                            episode_tick[selected_env_rows],
                            reason_bits[selected_env_rows],
                        ),
                        dim=1,
                    ).tolist()
                    for (
                        env_row,
                        generation_after,
                        reset_common_step,
                        reset_episode_tick,
                        reset_reason_bits,
                    ) in reset_fact_rows:
                        terminal_reset_row_count += 1
                        terminal_reset_episode_length_sum += reset_episode_tick
                        for ordinal, bit in enumerate((1, 2, 4, 8, 16)):
                            terminal_reset_reason_counts[ordinal] += int(
                                bool(reset_reason_bits & bit)
                            )
                        due_terminal_overlap_rows += int(
                            reset_episode_tick
                            in portable_catalog.FRESH_REFERENCE_DUE_TICKS
                        )
                        if (
                            len(terminal_resets)
                            < RESET_TELEMETRY_EXAMPLE_LIMIT
                        ):
                            terminal_resets.append(
                                ResetTelemetry(
                                    env_row=env_row,
                                    reset_generation=generation_after,
                                    common_step=reset_common_step,
                                    episode_tick=reset_episode_tick,
                                    reason_bits=reset_reason_bits,
                                )
                            )
                        for slot_index in torch.nonzero(
                            state.occupied[env_row], as_tuple=False
                        ).flatten().tolist():
                            prior_generation = int(
                                state.key.reset_generation[
                                    env_row, slot_index
                                ].item()
                            )
                            if generation_after <= prior_generation:
                                raise ActionEpochDrainDecodeError(
                                    "selected reset did not advance the active shot generation"
                                )
                            terminal_shots.append(
                                TerminalActionEpochShot(
                                    **_shot_snapshot(state, env_row, slot_index),
                                    reset_generation_after=generation_after,
                                    reset_common_step=reset_common_step,
                                    reset_episode_tick=reset_episode_tick,
                                    reset_reason_bits=reset_reason_bits,
                                )
                            )
                            count["terminal"] += 1
                _clear(state, selected_rows[:, None].expand(shape))
            continue

        if transition == "D05_SETTLED":
            # Only D05 consumes ``CommitEntry.epoch`` as a publication
            # ordinal.  Other transitions deliberately leave this scalar
            # compatibility spelling unused (and the producer writes -1).
            if publication < 0:
                raise ActionEpochDrainDecodeError(
                    "D05 publication chronology differs"
                )
            data, mask, key = _event(entry, transition, shape)
            due, selected = data["due_mask"], data["selected_mask"]
            decision = data["decision"]
            construction, playback = (
                data["construction_admissible"],
                data["playback_admissible"],
            )
            family = data["stroke_family_code"]
            attributed = data["action_attribution_valid"]
            faulted = data["owner_fault_bits"].ne(0).any(dim=2)
            valid = _valid(key)
            if bool(torch.any(mask ^ due).item()) or bool(
                torch.any(selected & ~due).item()
            ):
                raise ActionEpochDrainDecodeError("D05 denominators differ")
            censor = due & selected & (
                faulted | (construction & ~valid)
            )
            reject = selected & ~faulted & ~construction
            defer = due & ~selected
            accept = selected & valid & ~faulted & construction
            expected = torch.full_like(decision, D05_DECISION_NONE)
            for rows, value in (
                (censor, D05_DECISION_CENSOR),
                (reject, D05_DECISION_REJECT),
                (defer, D05_DECISION_DEFER),
                (accept, D05_DECISION_ACCEPT),
            ):
                expected[rows] = value
            if bool(torch.any(decision.ne(expected)).item()) or bool(
                torch.any(accept & state.occupied).item()
            ):
                raise ActionEpochDrainDecodeError("D05 decision/state differs")
            known_family = family.eq(1) | family.eq(2)
            if bool(
                torch.any(
                    family.lt(0)
                    | family.gt(2)
                    | attributed.ne(known_family)
                    | (~due & family.ne(action_strata.STROKE_FAMILY_UNKNOWN))
                ).item()
            ):
                raise ActionEpochDrainDecodeError("D05 action attribution differs")
            for env_row, slot_index in torch.nonzero(due, as_tuple=False).tolist():
                value = int(decision[env_row, slot_index].item())
                opportunities.append(D05ActionOpportunity(
                    int(env_row), int(slot_index),
                    int(key.action_uid[env_row, slot_index].item()),
                    int(key.action_slot[env_row, slot_index].item()),
                    action_strata.STROKE_FAMILY_NAMES[
                        int(family[env_row, slot_index].item())
                    ],
                    bool(attributed[env_row, slot_index].item()),
                    bool(selected[env_row, slot_index].item()),
                    value == D05_DECISION_ACCEPT,
                    value == D05_DECISION_CENSOR,
                    value == D05_DECISION_REJECT,
                    value == D05_DECISION_DEFER,
                ))
            _activate(
                state,
                accept,
                key,
                family,
                attributed,
                playback,
            )
            count["transactions"] += 1
            count["due"] += int(due.sum().item())
            count["selected"] += int(selected.sum().item())
            count["construction"] += int(
                (selected & construction & ~faulted).sum().item()
            )
            for name, rows in (
                ("accepted", accept),
                ("censored", censor),
                ("rejected", reject),
                ("deferred", defer),
            ):
                count[name] += int(rows.sum().item())
            count["not_ready"] += int((accept & ~playback).sum().item())
            count["faults"] += int((due & faulted).sum().item())
            pending = _PendingD05(accept.clone(), key, publication)
            continue

        data, mask, key = _event(entry, transition, shape)
        _current(state, mask, key, transition)
        if bool(torch.any(mask & ~_valid(key)).item()):
            raise ActionEpochDrainDecodeError("shot event full key is invalid")

        if transition == "MOTION_PLAYBACK_STARTED":
            if bool(
                torch.any(
                    mask
                    & (~state.reveal_committed | state.playback_started | state.motion_closed)
                ).item()
            ):
                raise ActionEpochDrainDecodeError("Motion playback order differs")
            state.playback_started[mask] = True
            count["playback"] += int(mask.sum().item())
        elif transition == "MOTION_CLOSED":
            reason = data["motion_close_reason"]
            invalid = (
                state.motion_closed
                | (
                    reason.eq(MOTION_CLOSE_PLAYED_SUFFIX)
                    & ~state.playback_started
                )
                | (reason.eq(MOTION_CLOSE_UNPLAYED) & state.playback_started)
                | (
                    ~reason.eq(MOTION_CLOSE_PLAYED_SUFFIX)
                    & ~reason.eq(MOTION_CLOSE_UNPLAYED)
                )
            )
            if bool(torch.any(mask & invalid).item()):
                raise ActionEpochDrainDecodeError("Motion close differs")
            state.motion_closed[mask] = True
            state.motion_close_reason[mask] = reason[mask]
            count["unplayed"] += int(
                (mask & reason.eq(MOTION_CLOSE_UNPLAYED)).sum().item()
            )
        elif transition == "PHYSICAL_LAUNCH_ROWS":
            target = data["target_xy_m"]
            if bool(
                torch.any(
                    mask
                    & (
                        ~state.reveal_committed
                        | ~state.physical_launch_requested
                        | state.physical_launched
                        | data["publication_ordinal"].lt(0)
                        | ~torch.isfinite(target).all(dim=2)
                    )
                ).item()
            ) or bool(torch.any(target[~mask].ne(0.0)).item()):
                raise ActionEpochDrainDecodeError("Physical launch differs")
            state.physical_launched[mask] = True
            state.physical_target_xy_m[mask] = target[mask]
            count["launch"] += int(mask.sum().item())
        elif transition == "PHYSICAL_POSTPHYSICS_ROWS":
            valid_bits = data["fact_valid_bits"]
            source_step = data["fact_source_step"]
            present = mask & valid_bits.bitwise_and(1).ne(0)
            contact = mask & valid_bits.bitwise_and(2).ne(0)
            if bool(
                torch.any(
                    mask
                    & (
                        ~state.reveal_committed
                        | data["publication_ordinal"].lt(0)
                        | valid_bits.lt(0)
                        | valid_bits.bitwise_and(~3).ne(0)
                        | (contact & source_step.lt(0))
                    )
                ).item()
            ):
                raise ActionEpochDrainDecodeError("Physical facts differ")
            new_contact = contact & state.physical_valid_bits.bitwise_and(2).eq(0)
            state.physical_valid_bits[mask] |= valid_bits[mask]
            state.physical_actor_pair_contact_source_step[new_contact] = (
                source_step[new_contact]
            )
        elif transition == "R06_OUTCOME_ROWS":
            settlement = data["settlement_step"]
            valid_bits = data["valid_bits"]
            predicates = data["predicate_bits"]
            present = valid_bits.bitwise_and(1).ne(0)
            eligible = valid_bits.bitwise_and(2).ne(0)
            source_valid = valid_bits.bitwise_and(4).ne(0)
            outcome_code = data["outcome_code"]
            if bool(
                torch.any(
                    mask
                    & (
                        ~state.physical_launched
                        | state.outcome_settled
                        | settlement.lt(0)
                        | data["publication_ordinal"].lt(0)
                        | valid_bits.lt(0)
                        | valid_bits.bitwise_and(~7).ne(0)
                        | predicates.lt(0)
                        | predicates.bitwise_and(~63).ne(0)
                        | (present & (outcome_code.lt(1) | outcome_code.gt(7)))
                        | (~present & outcome_code.ne(-1))
                        | (eligible & ~source_valid)
                        | (source_valid & ~present)
                        | (predicates.ne(0) & ~source_valid)
                        | (
                            predicates.bitwise_and(_R06_NET_CLEAR).ne(0)
                            & predicates.bitwise_and(_R06_NET_CROSSED).eq(0)
                        )
                    )
                ).item()
            ):
                raise ActionEpochDrainDecodeError("R06 outcome differs")
            state.outcome_settled[mask] = True
            state.settlement_step[mask] = settlement[mask]
            state.r06_valid_bits[mask] = valid_bits[mask]
            state.r06_outcome_code[mask] = outcome_code[mask]
            state.r06_predicate_bits[mask] = predicates[mask]
            count["outcome"] += int(mask.sum().item())
        elif transition == "PAYMENT_RECORDED":
            payment = data["payment_step"]
            if bool(
                torch.any(
                    mask
                    & (
                        ~state.outcome_settled
                        | state.payment_recorded
                        | payment.lt(state.settlement_step)
                    )
                ).item()
            ):
                raise ActionEpochDrainDecodeError("Reward payment differs")
            state.payment_recorded[mask] = True
            state.payment_step[mask] = payment[mask]
            count["payment"] += int(mask.sum().item())
        elif transition == "RETIRED":
            retirement = data["retirement_step"]
            ordinary_paid = (
                state.motion_closed
                & state.outcome_settled
                & state.payment_recorded
                & retirement.ge(state.payment_step)
            )
            explicit_no_launch = (
                state.motion_closed
                & state.motion_close_reason.eq(MOTION_CLOSE_UNPLAYED)
                & ~state.physical_launch_requested
                & ~state.playback_started
                & ~state.physical_launched
                & ~state.outcome_settled
                & ~state.payment_recorded
                & state.payment_step.eq(-1)
                & retirement.ge(0)
            )
            invalid = (
                ~(ordinary_paid | explicit_no_launch)
                | ~torch.isfinite(state.physical_target_xy_m).all(dim=2)
                | data["motion_close_reason"].ne(state.motion_close_reason)
                | data["payment_step"].ne(state.payment_step)
            )
            if bool(torch.any(mask & invalid).item()) or bool(
                torch.any(~mask & retirement.ne(-1)).item()
            ):
                raise ActionEpochDrainDecodeError("retirement differs")
            for env_row, slot_index in torch.nonzero(mask, as_tuple=False).tolist():
                completed.append(
                    CompletedActionEpochShot(
                        **_shot_snapshot(state, env_row, slot_index),
                        retirement_step=int(
                            retirement[env_row, slot_index].item()
                        ),
                    )
                )
            count["retired"] += int(mask.sum().item())
            _clear(state, mask)
        elif transition.startswith("OWNER_FAULT:"):
            if bool(torch.any(mask & data["owner_fault_bits"].eq(0)).item()):
                raise ActionEpochDrainDecodeError("owner fault attribution differs")
            count["faults"] += int(mask.sum().item())
        elif transition.startswith("OWNER_FACTS:"):
            owner_kind = transition.split(":", 1)[1]
            valid_bits = data["valid_bits"]
            source_step = data["source_step"]
            qualified = data["qualified"]
            present = mask & valid_bits.bitwise_and(1).ne(0)
            if bool(
                torch.any(
                    mask
                    & (
                        valid_bits.lt(0)
                        | valid_bits.bitwise_and(~3).ne(0)
                        | (valid_bits.bitwise_and(2).ne(0) & ~present)
                        | (qualified & ~present)
                        | (qualified & valid_bits.bitwise_and(2).eq(0))
                        | (qualified & source_step.lt(0))
                    )
                ).item()
            ):
                raise ActionEpochDrainDecodeError("owner facts differ")
            if owner_kind == "r03_strike_fact":
                if bool(torch.any(present & source_step.lt(0)).item()):
                    raise ActionEpochDrainDecodeError("R03 fact source differs")
                new = present & state.r03_valid_bits.bitwise_and(1).eq(0)
                state.r03_valid_bits[mask] |= valid_bits[mask]
                state.r03_source_step[new] = source_step[new]
            elif owner_kind == "r07_recovery":
                new_qualified = qualified & state.r07_qualified_source_step.lt(0)
                state.r07_valid_bits[mask] |= valid_bits[mask]
                state.r07_qualified_source_step[new_qualified] = source_step[
                    new_qualified
                ]
            elif bool(torch.any(qualified).item()):
                raise ActionEpochDrainDecodeError("foreign owner qualification differs")
        elif transition == "R07_FIRST_READY":
            source_step = data["source_step"]
            replay = mask & state.r07_first_ready_source_step.ge(0)
            if bool(
                torch.any(
                    mask
                    & (source_step.lt(0) | state.r07_qualified_source_step.lt(0))
                ).item()
            ) or bool(
                torch.any(
                    replay & source_step.ne(state.r07_first_ready_source_step)
                ).item()
            ):
                raise ActionEpochDrainDecodeError("R07 first-ready source differs")
            new = mask & state.r07_first_ready_source_step.lt(0)
            state.r07_first_ready_source_step[new] = source_step[new]

    if pending is not None or reward_ordinal is not None:
        raise ActionEpochDrainDecodeError("suffix ends inside a transaction")
    if count["due"] != sum(
        count[name] for name in ("accepted", "censored", "rejected", "deferred")
    ):
        raise ActionEpochDrainDecodeError("D05 conservation differs")
    active_after = int(state.occupied.sum().item())
    episode = dict(zip(
        milestone_tensors.EPISODE_I64_NAMES, milestone.episode_i64_counts
    ))
    expected_episode = {
        "completed": terminal_reset_row_count,
        "length_sum": terminal_reset_episode_length_sum,
        **{
            name: terminal_reset_reason_counts[ordinal]
            for ordinal, name in enumerate(
                milestone_tensors.EPISODE_I64_NAMES[2:]
            )
        },
    }
    if episode != expected_episode:
        raise ActionEpochDrainDecodeError(
            "milestone episode/reset suffix relationship differs"
        )
    return DecodedEpochDrain(
        settlement=D05SettlementCounts(
            count["transactions"],
            count["due"],
            count["selected"],
            count["accepted"],
            count["censored"],
            count["rejected"],
            count["deferred"],
            count["not_ready"],
        ),
        reveal_commit=RevealCommitCounts(
            count["motion"], count["racket"], count["r05"]
        ),
        lifecycle=LifecycleEdgeCounts(
            count["playback"],
            count["unplayed"],
            count["launch"],
            count["outcome"],
            count["payment"],
            count["retired"],
            count["terminal"],
        ),
        owner_faults=OwnerFaultCounts(count["faults"]),
        continuation=ContinuationCounts(
            active_before,
            active_after,
            int(
                (
                    state.occupied
                    & ~state.playback_started
                    & ~state.motion_closed
                ).sum().item()
            ),
            int((state.occupied & ~state.outcome_settled).sum().item()),
            int(
                (
                    state.occupied
                    & state.outcome_settled
                    & ~state.payment_recorded
                ).sum().item()
            ),
        ),
        action_opportunities=tuple(opportunities),
        completed_shots=tuple(completed),
        terminal_shots=tuple(terminal_shots),
        terminal_resets=tuple(terminal_resets),
        terminal_reset_aggregate=ResetTelemetryAggregate(
            row_count=terminal_reset_row_count,
            episode_length_sum=terminal_reset_episode_length_sum,
            reason_bit_counts=tuple(terminal_reset_reason_counts),
        ),
        due_terminal_overlap_rows=due_terminal_overlap_rows,
        milestone=milestone,
        next_continuation=state,
    )
