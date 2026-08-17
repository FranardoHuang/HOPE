"""Pure transaction owner for atomic continuous ActionBall reveals.

This module is deliberately marked ``PRE_INTEGRATION_HOLD``.  It owns no
Isaac, MuJoCo, Command, Reward, or physical-ball state and therefore cannot
authorize a launch.  Its narrow job is to make the future question a private,
checkpointable prepared row and to publish that row atomically in its own
state only after the frozen reveal facts agree.

The four counters are intentionally distinct and machine checked::

    scheduled_ordinal          = 0, 1, 2, ...
    runtime_swing_generation   = scheduled_ordinal
    sampler_generation         = per-env absolute target/RNG high-water + 1
    outcome_shot_index         = scheduled_ordinal + 1

Every carry chain starts its runtime swing generation at zero.  Target sampler
generation is intentionally different: it is the absolute per-environment
RNG high-water and therefore continues across a typed true reset.  Retired
chains remain content-addressed checkpoint evidence; reset never deletes or
renumbers committed rows.

Preparation reconstructs one private full-owner clone of
``ContinuousTargetSampler`` and stages an ordered, unique set of environment
rows.  The live sampler checkpoint is unchanged until commit.  Commit installs
all selected rows with one owner-state swap while byte-preserving every
unselected row, so independently prepared environments cannot erase one
another's RNG progress.  The single-environment API is the K=1 form of the
same authority and cannot split a multi-environment batch.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields
from functools import cached_property
import hashlib
import json
import math
from numbers import Real
import struct
import threading
from typing import Callable, ClassVar, Mapping, Optional, Sequence, Tuple, Union
import weakref

import action_ball_continuous_successor as _successor
import action_ball_continuous_target_sampler as _target_sampler
import action_ball_landing_outcome_mailbox as _mailbox


SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 5
INTEGRATION_STATUS = "PRE_INTEGRATION_HOLD"
RUNTIME_WIRING_CONNECTED = False
CHECKPOINT_KIND = "action_ball_continuous_runtime_transaction_checkpoint_v5"
LEGACY_CHECKPOINT_KIND = (
    "action_ball_continuous_runtime_transaction_checkpoint_v1"
)
LEGACY_V2_CHECKPOINT_KIND = (
    "action_ball_continuous_runtime_transaction_checkpoint_v2"
)
LEGACY_V3_CHECKPOINT_KIND = (
    "action_ball_continuous_runtime_transaction_checkpoint_v3"
)
LEGACY_V4_CHECKPOINT_KIND = (
    "action_ball_continuous_runtime_transaction_checkpoint_v4"
)

EMPTY = "EMPTY"
PREPARED = "PREPARED"
REVEAL_FINAL_PREVIEWED = "REVEAL_FINAL_PREVIEWED"
INFRA_CENSORED = "INFRA_CENSORED"
COMMITTED = "COMMITTED"

BALL_EMPTY = "empty"
BALL_INBOUND = "inbound"
BALL_OPEN = "open"
BALL_SETTLED_UNPAID = "settled_unpaid"
BALL_PAID = "paid"
BALL_CLOSED = "closed"
BALL_STATES = frozenset(
    (
        BALL_EMPTY,
        BALL_INBOUND,
        BALL_OPEN,
        BALL_SETTLED_UNPAID,
        BALL_PAID,
        BALL_CLOSED,
    )
)
_BALL_LIFECYCLE_RANK = {
    BALL_INBOUND: 0,
    BALL_OPEN: 1,
    BALL_SETTLED_UNPAID: 2,
    BALL_PAID: 3,
    BALL_CLOSED: 4,
}

_CONSTRUCTION_REASONS = frozenset(_successor.CONSTRUCTION_REASONS)
_NO_REJECTION = "none"
_INFRA_CENSOR_REASONS = frozenset(
    (
        "physical_ball_preflight_rejected",
        "runtime_task_receipt_preflight_rejected",
        "owner_crossbind_preflight_rejected",
        "owner_preterminal_receipt_censored",
        "all_owner_batch_censored",
        "reveal_install_reservation_lost",
    )
)
_PREARM_CHILD_OWNER_KINDS = (
    "motion",
    "racket",
    "physical_ball",
    "r06_flight",
)
TERMINAL_DECISION_ACCEPT = "ACCEPT"
TERMINAL_DECISION_CENSOR = "CENSOR"
TERMINAL_DECISIONS = (
    TERMINAL_DECISION_ACCEPT,
    TERMINAL_DECISION_CENSOR,
)
PREPARED_REVEAL_TERMINAL_CLAIM_KIND = (
    "action_ball_continuous_prepared_reveal_terminal_claim_v1"
)
TERMINAL_BOUNDARY_AUTHORITY_KIND = (
    "action_ball_continuous_terminal_boundary_authority_v1"
)
TERMINAL_BOUNDARY_DECISION_MAPPING_SCHEMA_VERSION = 1
TERMINAL_BOUNDARY_SOURCE_DECISION_PASS = "PASS"
_TERMINAL_BOUNDARY_DECISION_MAP_V1 = {
    TERMINAL_BOUNDARY_SOURCE_DECISION_PASS: TERMINAL_DECISION_ACCEPT,
    TERMINAL_DECISION_CENSOR: TERMINAL_DECISION_CENSOR,
}
_HEX = frozenset("0123456789abcdef")
_FaultInjector = Optional[Callable[[str], None]]


class ContinuousRuntimeTransactionError(RuntimeError):
    """The proposed transaction does not satisfy the pre-integration contract."""


class Float32TargetAliasError(ContinuousRuntimeTransactionError):
    """Two target cells become the same numerical target at runtime precision."""


class BallSlotCapacityError(ContinuousRuntimeTransactionError):
    """No physical slot can accept the next ball without overwriting an owner."""


class TransactionConflictError(ContinuousRuntimeTransactionError):
    """A stale, duplicate, or cross-environment transaction was requested."""


def canonical_sha256(value: object) -> str:
    encoded = _canonical_json_bytes(value)
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_sha256(payload)
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ContinuousRuntimeTransactionError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise ContinuousRuntimeTransactionError(f"{label} must be an exact int")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContinuousRuntimeTransactionError(f"{label} is outside its range")
    return value


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ContinuousRuntimeTransactionError(f"{label} must be an exact bool")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ContinuousRuntimeTransactionError(
            f"{label} must be a non-empty string"
        )
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContinuousRuntimeTransactionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContinuousRuntimeTransactionError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _xy(value: object, *, label: str) -> Tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContinuousRuntimeTransactionError(f"{label} must be an XY sequence")
    rows = tuple(value)
    if len(rows) != 2:
        raise ContinuousRuntimeTransactionError(f"{label} must contain two values")
    return (
        _finite(rows[0], label=f"{label}[0]"),
        _finite(rows[1], label=f"{label}[1]"),
    )


def _float32(value: object, *, label: str) -> float:
    number = _finite(value, label=label)
    try:
        result = struct.unpack("!f", struct.pack("!f", number))[0]
    except (OverflowError, struct.error) as exc:
        raise ContinuousRuntimeTransactionError(
            f"{label} is not finite float32"
        ) from exc
    if not math.isfinite(result):
        raise ContinuousRuntimeTransactionError(f"{label} is not finite float32")
    return 0.0 if result == 0.0 else float(result)


def _runtime_xy(value: object, *, label: str) -> Tuple[float, float]:
    x, y = _xy(value, label=label)
    return (
        _float32(x, label=f"{label}[0]"),
        _float32(y, label=f"{label}[1]"),
    )


def _encode(value: object) -> object:
    if isinstance(value, _SealedRecord):
        return value.to_mapping()
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return value.to_mapping()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _canonical_clone(value: object) -> object:
    """Detach one canonical-JSON value from caller-owned nested containers."""

    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ContinuousRuntimeTransactionError(
            "value is not finite canonical JSON"
        ) from exc


def _verified_values(
    value: object,
    *,
    cls: type,
    kind: str,
    schema_version: int,
) -> dict[str, object]:
    label = cls.__name__
    if not isinstance(value, Mapping):
        raise ContinuousRuntimeTransactionError(f"{label} must be a mapping")
    names = tuple(field.name for field in fields(cls))
    expected_payload = frozenset(("schema_version", "kind", *names))
    expected = expected_payload | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise ContinuousRuntimeTransactionError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    payload = {key: value[key] for key in expected_payload}
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != schema_version
    ):
        raise ContinuousRuntimeTransactionError(f"{label} schema_version differs")
    if type(payload["kind"]) is not str or payload["kind"] != kind:
        raise ContinuousRuntimeTransactionError(f"{label} kind differs")
    declared = _sha256(value["canonical_sha256"], label=f"{label}.canonical_sha256")
    if canonical_sha256(payload) != declared:
        raise ContinuousRuntimeTransactionError(f"{label} canonical SHA differs")
    return {name: payload[name] for name in names}


class _SealedRecord:
    KIND: ClassVar[str]
    RECORD_SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.RECORD_SCHEMA_VERSION,
            "kind": self.KIND,
            **{
                field.name: _encode(getattr(self, field.name))
                for field in fields(self)
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def _mapping_values(cls, value: object) -> dict[str, object]:
        return _verified_values(
            value,
            cls=cls,
            kind=cls.KIND,
            schema_version=cls.RECORD_SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class BallSlotSnapshot(_SealedRecord):
    """One physical slot plus its independent delayed-outcome owner state."""

    KIND: ClassVar[str] = "action_ball_continuous_ball_slot_snapshot_v1"

    slot_index: int
    lifecycle_state: str
    physical_retired: bool
    owner_key_sha256: Optional[str]
    ball_generation: Optional[int]
    inbound_ball_sha256: Optional[str]
    dynamic_state_sha256: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "slot_index",
            _plain_int(self.slot_index, label="slot_index"),
        )
        state = _text(self.lifecycle_state, label="lifecycle_state")
        if state not in BALL_STATES:
            raise ContinuousRuntimeTransactionError("ball lifecycle_state is unknown")
        object.__setattr__(self, "lifecycle_state", state)
        object.__setattr__(
            self,
            "physical_retired",
            _exact_bool(self.physical_retired, label="physical_retired"),
        )
        if state == BALL_EMPTY:
            if any(
                value is not None
                for value in (
                    self.owner_key_sha256,
                    self.ball_generation,
                    self.inbound_ball_sha256,
                    self.dynamic_state_sha256,
                )
            ):
                raise ContinuousRuntimeTransactionError(
                    "empty ball slot carries owner or ball state"
                )
            if not self.physical_retired:
                raise ContinuousRuntimeTransactionError(
                    "empty ball slot must be physically retired"
                )
            return
        for name in (
            "owner_key_sha256",
            "inbound_ball_sha256",
            "dynamic_state_sha256",
        ):
            value = getattr(self, name)
            if value is None:
                raise ContinuousRuntimeTransactionError(
                    f"non-empty ball slot lacks {name}"
                )
            object.__setattr__(self, name, _sha256(value, label=name))
        if self.ball_generation is None:
            raise ContinuousRuntimeTransactionError(
                "non-empty ball slot lacks ball_generation"
            )
        object.__setattr__(
            self,
            "ball_generation",
            _plain_int(self.ball_generation, label="ball_generation"),
        )
        if state in (BALL_INBOUND, BALL_OPEN) and self.physical_retired:
            raise ContinuousRuntimeTransactionError(
                "live inbound/open ball cannot be physically retired"
            )
        if state == BALL_CLOSED and not self.physical_retired:
            raise ContinuousRuntimeTransactionError(
                "closed ball slot must be physically retired"
            )

    @property
    def reusable(self) -> bool:
        return self.lifecycle_state == BALL_EMPTY or (
            self.lifecycle_state in (
                BALL_SETTLED_UNPAID,
                BALL_PAID,
                BALL_CLOSED,
            )
            and self.physical_retired
        )

    @classmethod
    def from_mapping(cls, value: object) -> "BallSlotSnapshot":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class CandidateTaskMaterialization(_SealedRecord):
    """Construction result for exactly one profile cell.

    Feasible rows carry an exact eight-field task ref and new inbound ball.
    Rejected rows carry no task/ball identity and therefore can never become a
    policy opportunity.
    """

    KIND: ClassVar[str] = "action_ball_continuous_candidate_materialization_v2"

    cell_id: str
    target_xy_m: Tuple[float, float]
    target_semantic_sha256: str
    evaluated_step: int
    construction_feasible: bool
    rejection_reason: str
    feasibility_authority_sha256: str
    solver_receipt_sha256: str
    support_receipt_sha256: str
    task_ref: Optional[_successor.ContinuousActionTaskReceiptRef]
    receipt_content_sha256: Optional[str]
    inbound_ball_generation: Optional[int]
    inbound_ball_sha256: Optional[str]
    installed_ball_dynamic_state_sha256: Optional[str]
    physical_ball_install_payload_sha256: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _text(self.cell_id, label="cell_id"))
        object.__setattr__(
            self, "target_xy_m", _xy(self.target_xy_m, label="target_xy_m")
        )
        for name in (
            "target_semantic_sha256",
            "feasibility_authority_sha256",
            "solver_receipt_sha256",
            "support_receipt_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "evaluated_step",
            _plain_int(self.evaluated_step, label="evaluated_step"),
        )
        feasible = _exact_bool(
            self.construction_feasible, label="construction_feasible"
        )
        object.__setattr__(self, "construction_feasible", feasible)
        reason = _text(self.rejection_reason, label="rejection_reason")
        if feasible:
            if reason != _NO_REJECTION:
                raise ContinuousRuntimeTransactionError(
                    "feasible candidate carries a rejection reason"
                )
            if not isinstance(
                self.task_ref, _successor.ContinuousActionTaskReceiptRef
            ):
                raise ContinuousRuntimeTransactionError(
                    "feasible candidate lacks an exact eight-field task ref"
                )
            for name in (
                "receipt_content_sha256",
                "inbound_ball_sha256",
                "installed_ball_dynamic_state_sha256",
                "physical_ball_install_payload_sha256",
            ):
                value = getattr(self, name)
                if value is None:
                    raise ContinuousRuntimeTransactionError(
                        f"feasible candidate lacks {name}"
                    )
                object.__setattr__(self, name, _sha256(value, label=name))
            if self.inbound_ball_generation is None:
                raise ContinuousRuntimeTransactionError(
                    "feasible candidate lacks inbound_ball_generation"
                )
            object.__setattr__(
                self,
                "inbound_ball_generation",
                _plain_int(
                    self.inbound_ball_generation,
                    label="inbound_ball_generation",
                ),
            )
        else:
            if reason not in _CONSTRUCTION_REASONS:
                raise ContinuousRuntimeTransactionError(
                    "infeasible candidate rejection reason is unknown"
                )
            if any(
                value is not None
                for value in (
                    self.task_ref,
                    self.receipt_content_sha256,
                    self.inbound_ball_generation,
                    self.inbound_ball_sha256,
                    self.installed_ball_dynamic_state_sha256,
                    self.physical_ball_install_payload_sha256,
                )
            ):
                raise ContinuousRuntimeTransactionError(
                    "infeasible candidate acquired task or ball identity"
                )
        object.__setattr__(self, "rejection_reason", reason)

    @property
    def policy_opportunity_created(self) -> bool:
        return False

    @classmethod
    def from_mapping(cls, value: object) -> "CandidateTaskMaterialization":
        values = cls._mapping_values(value)
        raw_ref = values["task_ref"]
        values["task_ref"] = (
            None
            if raw_ref is None
            else _successor.ContinuousActionTaskReceiptRef.from_mapping(raw_ref)
        )
        return cls(**values)


@dataclass(frozen=True)
class ContinuousPrepareRequest(_SealedRecord):
    """Immutable expected identity for one future scheduled reveal."""

    KIND: ClassVar[str] = "action_ball_continuous_prepare_request_v1"

    env_id: int
    reset_generation: int
    scheduled_ordinal: int
    runtime_swing_generation: int
    sampler_generation: int
    outcome_shot_index: int
    action_uid: int
    action_slot: int
    birth_sha256: str
    run_id: str
    carry_chain_id: str
    schedule_sha256: str
    scheduled_reveal_step: int
    scheduled_deadline_step: int
    admission_evaluated_step: int
    selection_authority_sha256: str
    source_sha256: str
    config_sha256: str
    task_birth_snapshot_id: str
    ball_slots: Tuple[BallSlotSnapshot, ...]
    previous_ball_slot_index: Optional[int]

    def __post_init__(self) -> None:
        for name, minimum in (
            ("env_id", 0),
            ("reset_generation", 1),
            ("scheduled_ordinal", 0),
            ("runtime_swing_generation", 0),
            ("sampler_generation", 1),
            ("outcome_shot_index", 1),
            ("action_uid", 1),
            ("action_slot", 0),
            ("scheduled_reveal_step", 0),
            ("scheduled_deadline_step", 1),
            ("admission_evaluated_step", 0),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name, minimum=minimum),
            )
        if self.action_uid > (1 << 53) - 1:
            raise ContinuousRuntimeTransactionError("action_uid exceeds exact range")
        ordinal = self.scheduled_ordinal
        if self.runtime_swing_generation != ordinal:
            raise ContinuousRuntimeTransactionError(
                "runtime_swing_generation must equal zero-based scheduled_ordinal"
            )
        if self.sampler_generation < ordinal + 1:
            raise ContinuousRuntimeTransactionError(
                "sampler_generation cannot trail scheduled_ordinal + 1"
            )
        if self.outcome_shot_index != ordinal + 1:
            raise ContinuousRuntimeTransactionError(
                "outcome_shot_index must equal scheduled_ordinal + 1"
            )
        if not (
            self.admission_evaluated_step < self.scheduled_reveal_step
            < self.scheduled_deadline_step
        ):
            raise ContinuousRuntimeTransactionError(
                "admission/reveal/deadline order differs"
            )
        for name in (
            "birth_sha256",
            "schedule_sha256",
            "selection_authority_sha256",
            "source_sha256",
            "config_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in ("run_id", "carry_chain_id", "task_birth_snapshot_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), label=name)
            )
        slots = tuple(self.ball_slots)
        if any(not isinstance(slot, BallSlotSnapshot) for slot in slots):
            raise ContinuousRuntimeTransactionError(
                "ball_slots must contain BallSlotSnapshot rows"
            )
        object.__setattr__(self, "ball_slots", slots)
        if self.previous_ball_slot_index is not None:
            object.__setattr__(
                self,
                "previous_ball_slot_index",
                _plain_int(
                    self.previous_ball_slot_index,
                    label="previous_ball_slot_index",
                ),
            )

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousPrepareRequest":
        values = cls._mapping_values(value)
        raw_slots = values["ball_slots"]
        if not isinstance(raw_slots, (tuple, list)):
            raise ContinuousRuntimeTransactionError("ball_slots must be a sequence")
        values["ball_slots"] = tuple(
            BallSlotSnapshot.from_mapping(slot) for slot in raw_slots
        )
        return cls(**values)


@dataclass(frozen=True)
class PreparedBallSlotReservation(_SealedRecord):
    """Preparation-time capacity observation, never an install decision."""

    KIND: ClassVar[str] = "action_ball_continuous_ball_slot_reservation_v2"

    capacity: int
    snapshot_sha256: str
    previous_slot_index: Optional[int]
    reusable_slot_indices: Tuple[int, ...]
    capacity_available_at_prepare: bool
    observed_prior_owner_key_sha256: Tuple[str, ...]
    new_ball_generation: int
    new_inbound_ball_sha256: str
    new_ball_dynamic_state_sha256: str
    physical_ball_install_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capacity",
            _plain_int(self.capacity, label="capacity", minimum=1),
        )
        if self.previous_slot_index is not None:
            previous = _plain_int(
                self.previous_slot_index,
                label="previous_slot_index",
            )
            if previous >= self.capacity:
                raise ContinuousRuntimeTransactionError(
                    "previous ball slot is out of range"
                )
            object.__setattr__(self, "previous_slot_index", previous)
        reusable = tuple(
            _plain_int(value, label="reusable_slot_indices")
            for value in self.reusable_slot_indices
        )
        if (
            reusable != tuple(sorted(set(reusable)))
            or any(value >= self.capacity for value in reusable)
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared reusable ball slots are not unique ordered indices"
            )
        object.__setattr__(self, "reusable_slot_indices", reusable)
        available = _exact_bool(
            self.capacity_available_at_prepare,
            label="capacity_available_at_prepare",
        )
        if available != bool(reusable):
            raise ContinuousRuntimeTransactionError(
                "prepared capacity fact differs from reusable slots"
            )
        object.__setattr__(self, "capacity_available_at_prepare", available)
        observed = tuple(
            _sha256(value, label="observed_prior_owner_key_sha256")
            for value in self.observed_prior_owner_key_sha256
        )
        if len(set(observed)) != len(observed):
            raise ContinuousRuntimeTransactionError(
                "prepared prior owner identities contain duplicates"
            )
        object.__setattr__(self, "observed_prior_owner_key_sha256", observed)
        object.__setattr__(
            self,
            "new_ball_generation",
            _plain_int(self.new_ball_generation, label="new_ball_generation"),
        )
        for name in (
            "snapshot_sha256",
            "new_inbound_ball_sha256",
            "new_ball_dynamic_state_sha256",
            "physical_ball_install_payload_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), label=name),
            )

    @classmethod
    def from_mapping(cls, value: object) -> "PreparedBallSlotReservation":
        values = cls._mapping_values(value)
        values["reusable_slot_indices"] = tuple(values["reusable_slot_indices"])
        values["observed_prior_owner_key_sha256"] = tuple(
            values["observed_prior_owner_key_sha256"]
        )
        return cls(**values)


@dataclass(frozen=True)
class BallSlotPlan(_SealedRecord):
    """Prepared reservation or reveal-time final plan over one slot snapshot.

    ``PreparedReveal`` carries a reservation evaluated on the preparation
    snapshot.  ``CommittedReveal`` carries a newly derived final plan evaluated
    on the reveal snapshot after monotone prior-ball evolution.  Consumers
    must never treat the prepared reservation as the installed slot fact.
    """

    KIND: ClassVar[str] = "action_ball_continuous_ball_slot_plan_v2"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 2

    capacity: int
    snapshot_sha256: str
    selected_slot_index: int
    previous_slot_index: Optional[int]
    reused_previous_slot: bool
    preserved_live_owner_key_sha256: Tuple[str, ...]
    new_ball_generation: int
    new_inbound_ball_sha256: str
    new_ball_dynamic_state_sha256: str
    physical_ball_install_payload_sha256: str
    reused_retired_owner_key_sha256: Optional[str]

    def __post_init__(self) -> None:
        for name, minimum in (
            ("capacity", 1),
            ("selected_slot_index", 0),
            ("new_ball_generation", 0),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name, minimum=minimum),
            )
        if self.selected_slot_index >= self.capacity:
            raise ContinuousRuntimeTransactionError("selected ball slot is out of range")
        if self.previous_slot_index is not None:
            previous = _plain_int(
                self.previous_slot_index, label="previous_slot_index"
            )
            if previous >= self.capacity:
                raise ContinuousRuntimeTransactionError(
                    "previous ball slot is out of range"
                )
            object.__setattr__(self, "previous_slot_index", previous)
        object.__setattr__(
            self,
            "reused_previous_slot",
            _exact_bool(self.reused_previous_slot, label="reused_previous_slot"),
        )
        expected_reuse = (
            self.previous_slot_index is not None
            and self.selected_slot_index == self.previous_slot_index
        )
        if self.reused_previous_slot != expected_reuse:
            raise ContinuousRuntimeTransactionError("ball slot reuse fact differs")
        for name in (
            "snapshot_sha256",
            "new_inbound_ball_sha256",
            "new_ball_dynamic_state_sha256",
            "physical_ball_install_payload_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if self.reused_retired_owner_key_sha256 is not None:
            object.__setattr__(
                self,
                "reused_retired_owner_key_sha256",
                _sha256(
                    self.reused_retired_owner_key_sha256,
                    label="reused_retired_owner_key_sha256",
                ),
            )
        preserved = tuple(
            _sha256(value, label="preserved_live_owner_key_sha256")
            for value in self.preserved_live_owner_key_sha256
        )
        if len(set(preserved)) != len(preserved):
            raise ContinuousRuntimeTransactionError(
                "preserved live ball owner list contains duplicates"
            )
        object.__setattr__(self, "preserved_live_owner_key_sha256", preserved)

    @classmethod
    def from_mapping(cls, value: object) -> "BallSlotPlan":
        values = cls._mapping_values(value)
        values["preserved_live_owner_key_sha256"] = tuple(
            values["preserved_live_owner_key_sha256"]
        )
        return cls(**values)


def _target_selection_from_mapping(value: object) -> _target_sampler.TargetSelection:
    if not isinstance(value, Mapping):
        raise ContinuousRuntimeTransactionError("target selection must be a mapping")
    if (
        value.get("schema_version") == 1
        or value.get("kind")
        == "action_ball_continuous_target_selection_v1"
    ):
        raise ContinuousRuntimeTransactionError(
            "legacy v1 target selection is tombstoned"
        )
    keys = (
        "schema_version",
        "kind",
        "profile_sha256",
        "env_id",
        "target_generation",
        "cell_id",
        "frame_id",
        "frame_binding_sha256",
        "runtime_dtype",
        "quantization_contract",
        "components",
        "target",
        "semantic_sha256",
        "canonical_sha256",
    )
    if frozenset(value) != frozenset(keys):
        raise ContinuousRuntimeTransactionError("target selection keys differ")
    payload = {key: value[key] for key in keys if key != "canonical_sha256"}
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != _target_sampler.SCHEMA_VERSION
        or type(payload["kind"]) is not str
        or payload["kind"] != _target_sampler.SELECTION_KIND
    ):
        raise ContinuousRuntimeTransactionError("target selection schema differs")
    declared = _sha256(
        value["canonical_sha256"], label="target selection canonical_sha256"
    )
    if _target_sampler.canonical_sha256(payload) != declared:
        raise ContinuousRuntimeTransactionError("target selection canonical SHA differs")
    return _target_sampler.TargetSelection(
        profile_sha256=payload["profile_sha256"],
        env_id=payload["env_id"],
        target_generation=payload["target_generation"],
        cell_id=payload["cell_id"],
        frame_id=payload["frame_id"],
        frame_binding_sha256=payload["frame_binding_sha256"],
        runtime_dtype=payload["runtime_dtype"],
        quantization_contract=payload["quantization_contract"],
        components=tuple(payload["components"]),
        target=tuple(payload["target"]),
        semantic_sha256=payload["semantic_sha256"],
    )


def _ordered_unique_env_ids(
    values: Sequence[object], *, label: str
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContinuousRuntimeTransactionError(
            f"{label} must be a finite sequence"
        )
    result = tuple(
        _plain_int(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ContinuousRuntimeTransactionError(f"{label} cannot be empty")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ContinuousRuntimeTransactionError(
            f"{label} must be strictly ordered and unique"
        )
    return result


def _sampler_checkpoint_sha256(
    checkpoint: Mapping[str, object], *, label: str
) -> str:
    if not isinstance(checkpoint, Mapping):
        raise ContinuousRuntimeTransactionError(f"{label} must be a mapping")
    return _sha256(
        checkpoint.get("checkpoint_sha256"),
        label=f"{label}.checkpoint_sha256",
    )


def _sampler_rows_excluding(
    checkpoint: Mapping[str, object], selected_env_ids: Sequence[int]
) -> Tuple[dict[str, object], ...]:
    selected = frozenset(selected_env_ids)
    rows = checkpoint.get("environments")
    if not isinstance(rows, list):
        raise ContinuousRuntimeTransactionError(
            "sampler checkpoint environments are malformed"
        )
    result = []
    prior = -1
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContinuousRuntimeTransactionError(
                "sampler checkpoint environment row is malformed"
            )
        env_id = _plain_int(row.get("env_id"), label="sampler row env_id")
        if env_id <= prior:
            raise ContinuousRuntimeTransactionError(
                "sampler checkpoint environments are not strictly sorted"
            )
        prior = env_id
        if env_id not in selected:
            result.append(dict(row))
    return tuple(result)


def _sampler_untouched_rows_root(
    checkpoint: Mapping[str, object], selected_env_ids: Sequence[int]
) -> str:
    selected = _ordered_unique_env_ids(
        tuple(selected_env_ids), label="selected_env_ids"
    )
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_sampler_untouched_rows_root_v1",
            "profile_sha256": checkpoint.get("profile_sha256"),
            "runtime_dtype": checkpoint.get("runtime_dtype"),
            "quantization_contract": checkpoint.get("quantization_contract"),
            "seed": checkpoint.get("seed"),
            "selected_env_ids": list(selected),
            "untouched_environments": list(
                _sampler_rows_excluding(checkpoint, selected)
            ),
        }
    )


@dataclass(frozen=True)
class PreparedReveal(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_prepared_reveal_v1"

    integration_status: str
    phase: str
    public_visible: bool
    policy_opportunity_created: bool
    request: ContinuousPrepareRequest
    candidates: Tuple[CandidateTaskMaterialization, ...]
    selection: _target_sampler.TargetSelection
    runtime_target_receipt: _successor.TargetSelectionReceipt
    selected_task_ref: _successor.ContinuousActionTaskReceiptRef
    outcome_key: _mailbox.LandingOutcomeShotKey
    prepared_ball_slot_reservation: PreparedBallSlotReservation
    sampler_before_env_state_sha256: str
    sampler_after_env_state_sha256: str
    sampler_before_env_state: Optional[Mapping[str, object]]
    sampler_after_env_state: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError(
                "prepared reveal lost PRE_INTEGRATION_HOLD status"
            )
        if self.phase != PREPARED:
            raise ContinuousRuntimeTransactionError("prepared reveal phase differs")
        if self.public_visible or self.policy_opportunity_created:
            raise ContinuousRuntimeTransactionError(
                "prepared future question became public or entered denominator"
            )
        if not isinstance(self.request, ContinuousPrepareRequest):
            raise ContinuousRuntimeTransactionError("prepared request type differs")
        candidates = tuple(self.candidates)
        if any(not isinstance(row, CandidateTaskMaterialization) for row in candidates):
            raise ContinuousRuntimeTransactionError("prepared candidates type differs")
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.selection, _target_sampler.TargetSelection):
            raise ContinuousRuntimeTransactionError("prepared selection type differs")
        if not isinstance(
            self.runtime_target_receipt, _successor.TargetSelectionReceipt
        ):
            raise ContinuousRuntimeTransactionError(
                "runtime target receipt type differs"
            )
        if not isinstance(
            self.selected_task_ref, _successor.ContinuousActionTaskReceiptRef
        ):
            raise ContinuousRuntimeTransactionError("selected task ref type differs")
        if not isinstance(self.outcome_key, _mailbox.LandingOutcomeShotKey):
            raise ContinuousRuntimeTransactionError("outcome key type differs")
        if not isinstance(
            self.prepared_ball_slot_reservation,
            PreparedBallSlotReservation,
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared ball slot reservation type differs"
            )
        for name in (
            "sampler_before_env_state_sha256",
            "sampler_after_env_state_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if self.sampler_before_env_state is not None and not isinstance(
            self.sampler_before_env_state, Mapping
        ):
            raise ContinuousRuntimeTransactionError(
                "sampler_before_env_state must be a mapping or None"
            )
        if not isinstance(self.sampler_after_env_state, Mapping):
            raise ContinuousRuntimeTransactionError(
                "sampler_after_env_state must be a mapping"
            )
        object.__setattr__(
            self,
            "sampler_before_env_state",
            None
            if self.sampler_before_env_state is None
            else dict(self.sampler_before_env_state),
        )
        object.__setattr__(
            self, "sampler_after_env_state", dict(self.sampler_after_env_state)
        )

    @classmethod
    def from_mapping(cls, value: object) -> "PreparedReveal":
        values = cls._mapping_values(value)
        values["request"] = ContinuousPrepareRequest.from_mapping(values["request"])
        values["candidates"] = tuple(
            CandidateTaskMaterialization.from_mapping(row)
            for row in values["candidates"]
        )
        values["selection"] = _target_selection_from_mapping(values["selection"])
        values["runtime_target_receipt"] = (
            _successor.TargetSelectionReceipt.from_mapping(
                values["runtime_target_receipt"]
            )
        )
        values["selected_task_ref"] = (
            _successor.ContinuousActionTaskReceiptRef.from_mapping(
                values["selected_task_ref"]
            )
        )
        values["outcome_key"] = _mailbox.LandingOutcomeShotKey.from_mapping(
            values["outcome_key"]
        )
        values["prepared_ball_slot_reservation"] = (
            PreparedBallSlotReservation.from_mapping(
                values["prepared_ball_slot_reservation"]
            )
        )
        return cls(**values)


@dataclass(frozen=True)
class ContinuousRevealFacts(_SealedRecord):
    """Frozen Motion facts observed on the exact reveal tick."""

    KIND: ClassVar[str] = "action_ball_continuous_reveal_facts_v1"

    env_id: int
    reset_generation: int
    scheduled_ordinal: int
    runtime_swing_generation: int
    sampler_generation: int
    outcome_shot_index: int
    schedule_sha256: str
    reveal_step: int
    deadline_step: int
    ready_at_reveal: bool
    boundary_reset: bool
    boundary_teleported: bool
    pre_reveal_hidden: _successor.PreRevealHiddenWitness
    carry_before_reveal: _successor.CarryContinuityWitness
    carry_after_reveal: _successor.CarryContinuityWitness
    ball_slots: Tuple[BallSlotSnapshot, ...]

    def __post_init__(self) -> None:
        for name, minimum in (
            ("env_id", 0),
            ("reset_generation", 1),
            ("scheduled_ordinal", 0),
            ("runtime_swing_generation", 0),
            ("sampler_generation", 1),
            ("outcome_shot_index", 1),
            ("reveal_step", 0),
            ("deadline_step", 1),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name, minimum=minimum),
            )
        if self.runtime_swing_generation != self.scheduled_ordinal:
            raise ContinuousRuntimeTransactionError(
                "reveal runtime_swing_generation mapping differs"
            )
        if self.sampler_generation < self.scheduled_ordinal + 1:
            raise ContinuousRuntimeTransactionError(
                "reveal sampler_generation trails scheduled ordinal"
            )
        if self.outcome_shot_index != self.scheduled_ordinal + 1:
            raise ContinuousRuntimeTransactionError(
                "reveal outcome_shot_index mapping differs"
            )
        if self.deadline_step <= self.reveal_step:
            raise ContinuousRuntimeTransactionError("reveal deadline is not later")
        object.__setattr__(
            self,
            "schedule_sha256",
            _sha256(self.schedule_sha256, label="schedule_sha256"),
        )
        object.__setattr__(
            self,
            "ready_at_reveal",
            _exact_bool(self.ready_at_reveal, label="ready_at_reveal"),
        )
        for name in ("boundary_reset", "boundary_teleported"):
            value = _exact_bool(getattr(self, name), label=name)
            if value:
                raise ContinuousRuntimeTransactionError(
                    f"reveal {name} cannot be true"
                )
            object.__setattr__(self, name, value)
        if not isinstance(
            self.pre_reveal_hidden, _successor.PreRevealHiddenWitness
        ):
            raise ContinuousRuntimeTransactionError(
                "pre_reveal_hidden type differs"
            )
        if not self.pre_reveal_hidden.all_future_facts_hidden:
            raise ContinuousRuntimeTransactionError(
                "future question leaked before scheduled reveal"
            )
        for name in ("carry_before_reveal", "carry_after_reveal"):
            if not isinstance(
                getattr(self, name), _successor.CarryContinuityWitness
            ):
                raise ContinuousRuntimeTransactionError(
                    f"{name} type differs"
                )
        before = self.carry_before_reveal
        after = self.carry_after_reveal
        if (
            before.episode_step != self.reveal_step
            or after.episode_step != self.reveal_step
            or before.reset_generation != self.reset_generation
            or after.reset_generation != self.reset_generation
        ):
            raise ContinuousRuntimeTransactionError(
                "carry witness step/reset binding differs"
            )
        if before != after:
            raise ContinuousRuntimeTransactionError(
                "question install teleported state or cleared carried history"
            )
        slots = tuple(self.ball_slots)
        if any(not isinstance(slot, BallSlotSnapshot) for slot in slots):
            raise ContinuousRuntimeTransactionError(
                "reveal ball_slots must contain BallSlotSnapshot rows"
            )
        object.__setattr__(self, "ball_slots", slots)

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRevealFacts":
        values = cls._mapping_values(value)
        values["pre_reveal_hidden"] = _successor.PreRevealHiddenWitness.from_mapping(
            values["pre_reveal_hidden"]
        )
        values["carry_before_reveal"] = (
            _successor.CarryContinuityWitness.from_mapping(
                values["carry_before_reveal"]
            )
        )
        values["carry_after_reveal"] = (
            _successor.CarryContinuityWitness.from_mapping(
                values["carry_after_reveal"]
            )
        )
        values["ball_slots"] = tuple(
            BallSlotSnapshot.from_mapping(row) for row in values["ball_slots"]
        )
        return cls(**values)


@dataclass(frozen=True)
class RevealFinalInstallRow(_SealedRecord):
    """Reveal-time exact physical install preview; no owner state is published."""

    KIND: ClassVar[str] = "action_ball_continuous_reveal_final_install_row_v2"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 2

    integration_status: str
    phase: str
    public_visible: bool
    policy_opportunity_created: bool
    prepared_reveal: PreparedReveal
    reveal_facts: ContinuousRevealFacts
    ball_slot_plan: BallSlotPlan
    selected_task_ref_sha256: str
    outcome_key_sha256: str
    physical_ball_install_payload_sha256: str
    pre_install_ball_slots: Tuple[BallSlotSnapshot, ...]
    post_install_ball_slots: Tuple[BallSlotSnapshot, ...]

    def __post_init__(self) -> None:
        if (
            self.integration_status != INTEGRATION_STATUS
            or self.phase != REVEAL_FINAL_PREVIEWED
            or self.public_visible
            or self.policy_opportunity_created
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final row phase/privacy differs"
            )
        if not isinstance(self.prepared_reveal, PreparedReveal):
            raise ContinuousRuntimeTransactionError(
                "reveal-final prepared row type differs"
            )
        if not isinstance(self.reveal_facts, ContinuousRevealFacts):
            raise ContinuousRuntimeTransactionError(
                "reveal-final facts type differs"
            )
        if not isinstance(self.ball_slot_plan, BallSlotPlan):
            raise ContinuousRuntimeTransactionError(
                "reveal-final ball plan type differs"
            )
        task_ref_sha = _sha256(
            self.selected_task_ref_sha256,
            label="selected_task_ref_sha256",
        )
        outcome_sha = _sha256(
            self.outcome_key_sha256,
            label="outcome_key_sha256",
        )
        physical_payload_sha = _sha256(
            self.physical_ball_install_payload_sha256,
            label="physical_ball_install_payload_sha256",
        )
        if (
            task_ref_sha
            != self.prepared_reveal.selected_task_ref.canonical_sha256
            or outcome_sha != self.prepared_reveal.outcome_key.canonical_sha256
            or physical_payload_sha
            != self.ball_slot_plan.physical_ball_install_payload_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final task/outcome identity differs"
            )
        before = tuple(self.pre_install_ball_slots)
        after = tuple(self.post_install_ball_slots)
        plan = self.ball_slot_plan
        facts = self.reveal_facts
        if (
            before != facts.ball_slots
            or len(before) != plan.capacity
            or len(after) != plan.capacity
            or tuple(row.slot_index for row in before)
            != tuple(range(plan.capacity))
            or tuple(row.slot_index for row in after)
            != tuple(range(plan.capacity))
            or plan.snapshot_sha256 != _ball_snapshot_sha256(before)
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final slot snapshot binding differs"
            )
        selected = plan.selected_slot_index
        for index, (prior, installed) in enumerate(zip(before, after)):
            if index != selected and installed != prior:
                raise ContinuousRuntimeTransactionError(
                    "reveal-final changed an unselected physical owner"
                )
        installed = after[selected]
        if (
            installed.lifecycle_state != BALL_INBOUND
            or installed.physical_retired
            or installed.owner_key_sha256 != outcome_sha
            or installed.ball_generation != plan.new_ball_generation
            or installed.inbound_ball_sha256 != plan.new_inbound_ball_sha256
            or installed.dynamic_state_sha256
            != plan.new_ball_dynamic_state_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final selected install fact differs"
            )
        object.__setattr__(self, "selected_task_ref_sha256", task_ref_sha)
        object.__setattr__(self, "outcome_key_sha256", outcome_sha)
        object.__setattr__(
            self,
            "physical_ball_install_payload_sha256",
            physical_payload_sha,
        )
        object.__setattr__(self, "pre_install_ball_slots", before)
        object.__setattr__(self, "post_install_ball_slots", after)

    @property
    def env_id(self) -> int:
        return self.prepared_reveal.request.env_id

    @classmethod
    def from_mapping(cls, value: object) -> "RevealFinalInstallRow":
        values = cls._mapping_values(value)
        values["prepared_reveal"] = PreparedReveal.from_mapping(
            values["prepared_reveal"]
        )
        values["reveal_facts"] = ContinuousRevealFacts.from_mapping(
            values["reveal_facts"]
        )
        values["ball_slot_plan"] = BallSlotPlan.from_mapping(
            values["ball_slot_plan"]
        )
        values["pre_install_ball_slots"] = tuple(
            BallSlotSnapshot.from_mapping(row)
            for row in values["pre_install_ball_slots"]
        )
        values["post_install_ball_slots"] = tuple(
            BallSlotSnapshot.from_mapping(row)
            for row in values["post_install_ball_slots"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CommittedReveal(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_committed_reveal_v2"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 2

    integration_status: str
    phase: str
    runtime_wiring_connected: bool
    identity_committed: bool
    policy_opportunity_created: bool
    prepared_reveal: PreparedReveal
    reveal_facts: ContinuousRevealFacts
    ball_slot_plan: BallSlotPlan
    playback_release_requested: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError(
                "committed reveal lost PRE_INTEGRATION_HOLD status"
            )
        if self.phase != COMMITTED:
            raise ContinuousRuntimeTransactionError("committed reveal phase differs")
        if self.runtime_wiring_connected:
            raise ContinuousRuntimeTransactionError(
                "pure transaction cannot claim runtime wiring"
            )
        if not self.identity_committed or not self.policy_opportunity_created:
            raise ContinuousRuntimeTransactionError(
                "committed admitted row did not enter its opportunity lane"
            )
        if not isinstance(self.prepared_reveal, PreparedReveal):
            raise ContinuousRuntimeTransactionError("committed prepared row type differs")
        if not isinstance(self.reveal_facts, ContinuousRevealFacts):
            raise ContinuousRuntimeTransactionError("committed reveal facts type differs")
        if not isinstance(self.ball_slot_plan, BallSlotPlan):
            raise ContinuousRuntimeTransactionError(
                "committed ball slot plan type differs"
            )
        requested = _exact_bool(
            self.playback_release_requested,
            label="playback_release_requested",
        )
        if requested != self.reveal_facts.ready_at_reveal:
            raise ContinuousRuntimeTransactionError(
                "playback request differs from ready-at-reveal fact"
            )
        object.__setattr__(self, "playback_release_requested", requested)

    @classmethod
    def from_mapping(cls, value: object) -> "CommittedReveal":
        values = cls._mapping_values(value)
        values["prepared_reveal"] = PreparedReveal.from_mapping(
            values["prepared_reveal"]
        )
        values["reveal_facts"] = ContinuousRevealFacts.from_mapping(
            values["reveal_facts"]
        )
        values["ball_slot_plan"] = BallSlotPlan.from_mapping(
            values["ball_slot_plan"]
        )
        return cls(**values)


@dataclass(frozen=True)
class AbortReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_abort_receipt_v1"

    integration_status: str
    phase_before: str
    phase_after: str
    env_id: int
    prepared_reveal_sha256: str
    sampler_checkpoint_sha256: str
    policy_opportunity_created: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError("abort status differs")
        if self.phase_before != PREPARED or self.phase_after != EMPTY:
            raise ContinuousRuntimeTransactionError("abort phase transition differs")
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, label="env_id")
        )
        for name in (
            "prepared_reveal_sha256",
            "sampler_checkpoint_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if self.policy_opportunity_created:
            raise ContinuousRuntimeTransactionError(
                "abort created a policy opportunity"
            )

    @classmethod
    def from_mapping(cls, value: object) -> "AbortReceipt":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class PreparedRevealBatch(_SealedRecord):
    """One K-environment private prepare performed on one full-owner clone."""

    KIND: ClassVar[str] = "action_ball_continuous_prepared_reveal_batch_v1"

    integration_status: str
    phase: str
    public_visible: bool
    policy_opportunity_created: bool
    selected_env_ids: Tuple[int, ...]
    task_birth_snapshot_ids: Tuple[str, ...]
    sampler_checkpoint_before_sha256: str
    sampler_checkpoint_after_private_sha256: str
    untouched_rows_before_sha256: str
    untouched_rows_after_sha256: str
    sampler_checkpoint_before: Mapping[str, object]
    sampler_checkpoint_after_private: Mapping[str, object]
    prepared_reveals: Tuple[PreparedReveal, ...]

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError(
                "prepared batch lost PRE_INTEGRATION_HOLD status"
            )
        if self.phase != PREPARED:
            raise ContinuousRuntimeTransactionError(
                "prepared batch phase differs"
            )
        if self.public_visible or self.policy_opportunity_created:
            raise ContinuousRuntimeTransactionError(
                "prepared batch became public or entered the denominator"
            )
        env_ids = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        births = tuple(
            _text(value, label=f"task_birth_snapshot_ids[{index}]")
            for index, value in enumerate(self.task_birth_snapshot_ids)
        )
        rows = tuple(self.prepared_reveals)
        if len(births) != len(env_ids) or len(rows) != len(env_ids):
            raise ContinuousRuntimeTransactionError(
                "prepared batch widths differ"
            )
        if any(not isinstance(row, PreparedReveal) for row in rows):
            raise ContinuousRuntimeTransactionError(
                "prepared batch rows must be PreparedReveal"
            )
        if tuple(row.request.env_id for row in rows) != env_ids:
            raise ContinuousRuntimeTransactionError(
                "prepared batch row order differs from selected envs"
            )
        if tuple(row.request.task_birth_snapshot_id for row in rows) != births:
            raise ContinuousRuntimeTransactionError(
                "prepared batch task-birth snapshot binding differs"
            )
        before = _canonical_clone(self.sampler_checkpoint_before)
        after = _canonical_clone(self.sampler_checkpoint_after_private)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ContinuousRuntimeTransactionError(
                "prepared batch sampler checkpoints must be mappings"
            )
        before_sha = _sha256(
            self.sampler_checkpoint_before_sha256,
            label="sampler_checkpoint_before_sha256",
        )
        after_sha = _sha256(
            self.sampler_checkpoint_after_private_sha256,
            label="sampler_checkpoint_after_private_sha256",
        )
        if before_sha != _sampler_checkpoint_sha256(
            before, label="sampler_checkpoint_before"
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared batch before checkpoint SHA differs"
            )
        if after_sha != _sampler_checkpoint_sha256(
            after, label="sampler_checkpoint_after_private"
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared batch private checkpoint SHA differs"
            )
        untouched_before = _sha256(
            self.untouched_rows_before_sha256,
            label="untouched_rows_before_sha256",
        )
        untouched_after = _sha256(
            self.untouched_rows_after_sha256,
            label="untouched_rows_after_sha256",
        )
        if untouched_before != _sampler_untouched_rows_root(before, env_ids):
            raise ContinuousRuntimeTransactionError(
                "prepared batch untouched-before root differs"
            )
        if untouched_after != _sampler_untouched_rows_root(after, env_ids):
            raise ContinuousRuntimeTransactionError(
                "prepared batch untouched-after root differs"
            )
        if (
            untouched_before != untouched_after
            or _sampler_rows_excluding(before, env_ids)
            != _sampler_rows_excluding(after, env_ids)
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared batch changed an unselected sampler row"
            )
        before_rows = _sampler_env_rows_by_id(before)
        after_rows = _sampler_env_rows_by_id(after)
        for row in rows:
            env_id = row.request.env_id
            if (
                before_rows.get(env_id) != row.sampler_before_env_state
                or after_rows.get(env_id) != row.sampler_after_env_state
            ):
                raise ContinuousRuntimeTransactionError(
                    "prepared batch selected sampler row binding differs"
                )
        if _merge_sampler_env_rows(before, rows) != after:
            raise ContinuousRuntimeTransactionError(
                "prepared batch full-owner transition differs"
            )
        object.__setattr__(self, "selected_env_ids", env_ids)
        object.__setattr__(self, "task_birth_snapshot_ids", births)
        object.__setattr__(self, "prepared_reveals", rows)
        object.__setattr__(self, "sampler_checkpoint_before", before)
        object.__setattr__(self, "sampler_checkpoint_after_private", after)
        object.__setattr__(self, "sampler_checkpoint_before_sha256", before_sha)
        object.__setattr__(
            self, "sampler_checkpoint_after_private_sha256", after_sha
        )
        object.__setattr__(self, "untouched_rows_before_sha256", untouched_before)
        object.__setattr__(self, "untouched_rows_after_sha256", untouched_after)

    @classmethod
    def from_mapping(cls, value: object) -> "PreparedRevealBatch":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["task_birth_snapshot_ids"] = tuple(
            values["task_birth_snapshot_ids"]
        )
        values["prepared_reveals"] = tuple(
            PreparedReveal.from_mapping(row) for row in values["prepared_reveals"]
        )
        return cls(**values)

    @property
    def batch_sha256(self) -> str:
        return self.canonical_sha256


def _all_owner_install_root(rows: Sequence[RevealFinalInstallRow]) -> str:
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_all_owner_install_root_v1",
            "rows": [
                {
                    "env_id": row.env_id,
                    "reveal_final_install_row_sha256": row.canonical_sha256,
                    "pre_install_ball_slots_sha256": _ball_snapshot_sha256(
                        row.pre_install_ball_slots
                    ),
                    "post_install_ball_slots_sha256": _ball_snapshot_sha256(
                        row.post_install_ball_slots
                    ),
                }
                for row in rows
            ],
        }
    )


@dataclass(frozen=True)
class RevealFinalPreviewBatch(_SealedRecord):
    """Sealed K-env reveal-final token; construction alone changes no live state."""

    KIND: ClassVar[str] = "action_ball_continuous_reveal_final_preview_batch_v2"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 2

    integration_status: str
    phase: str
    public_visible: bool
    policy_opportunity_created: bool
    owner_checkpoint_before_sha256: str
    prepared_batch: PreparedRevealBatch
    sampler_checkpoint_before_commit_sha256: str
    sampler_checkpoint_after_commit_sha256: str
    untouched_rows_before_sha256: str
    untouched_rows_after_sha256: str
    sampler_checkpoint_before_commit: Mapping[str, object]
    sampler_checkpoint_after_commit: Mapping[str, object]
    reveal_final_rows: Tuple[RevealFinalInstallRow, ...]
    all_owner_install_root_sha256: str

    def __post_init__(self) -> None:
        if (
            self.integration_status != INTEGRATION_STATUS
            or self.phase != REVEAL_FINAL_PREVIEWED
            or self.public_visible
            or self.policy_opportunity_created
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final batch phase/privacy differs"
            )
        if not isinstance(self.prepared_batch, PreparedRevealBatch):
            raise ContinuousRuntimeTransactionError(
                "reveal-final prepared batch type differs"
            )
        rows = tuple(self.reveal_final_rows)
        if (
            len(rows) != len(self.prepared_batch.prepared_reveals)
            or any(not isinstance(row, RevealFinalInstallRow) for row in rows)
            or tuple(row.prepared_reveal for row in rows)
            != self.prepared_batch.prepared_reveals
            or tuple(row.env_id for row in rows)
            != self.prepared_batch.selected_env_ids
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final batch row identity/order differs"
            )
        owner_sha = _sha256(
            self.owner_checkpoint_before_sha256,
            label="owner_checkpoint_before_sha256",
        )
        before = _canonical_clone(self.sampler_checkpoint_before_commit)
        after = _canonical_clone(self.sampler_checkpoint_after_commit)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ContinuousRuntimeTransactionError(
                "reveal-final sampler checkpoints must be mappings"
            )
        before_sha = _sha256(
            self.sampler_checkpoint_before_commit_sha256,
            label="sampler_checkpoint_before_commit_sha256",
        )
        after_sha = _sha256(
            self.sampler_checkpoint_after_commit_sha256,
            label="sampler_checkpoint_after_commit_sha256",
        )
        selected = self.prepared_batch.selected_env_ids
        untouched_before = _sha256(
            self.untouched_rows_before_sha256,
            label="untouched_rows_before_sha256",
        )
        untouched_after = _sha256(
            self.untouched_rows_after_sha256,
            label="untouched_rows_after_sha256",
        )
        if (
            before_sha
            != _sampler_checkpoint_sha256(
                before, label="sampler_checkpoint_before_commit"
            )
            or after_sha
            != _sampler_checkpoint_sha256(
                after, label="sampler_checkpoint_after_commit"
            )
            or untouched_before
            != _sampler_untouched_rows_root(before, selected)
            or untouched_after
            != _sampler_untouched_rows_root(after, selected)
            or untouched_before != untouched_after
            or _sampler_rows_excluding(before, selected)
            != _sampler_rows_excluding(after, selected)
            or _merge_sampler_env_rows(
                before, self.prepared_batch.prepared_reveals
            )
            != after
            or any(
                _sampler_env_row(before, row.request.env_id)
                != row.sampler_before_env_state
                for row in self.prepared_batch.prepared_reveals
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final sampler transition differs"
            )
        install_root = _sha256(
            self.all_owner_install_root_sha256,
            label="all_owner_install_root_sha256",
        )
        if install_root != _all_owner_install_root(rows):
            raise ContinuousRuntimeTransactionError(
                "reveal-final all-owner install root differs"
            )
        object.__setattr__(self, "owner_checkpoint_before_sha256", owner_sha)
        object.__setattr__(self, "reveal_final_rows", rows)
        object.__setattr__(
            self, "sampler_checkpoint_before_commit", before
        )
        object.__setattr__(
            self, "sampler_checkpoint_after_commit", after
        )
        object.__setattr__(
            self, "sampler_checkpoint_before_commit_sha256", before_sha
        )
        object.__setattr__(
            self, "sampler_checkpoint_after_commit_sha256", after_sha
        )
        object.__setattr__(self, "untouched_rows_before_sha256", untouched_before)
        object.__setattr__(self, "untouched_rows_after_sha256", untouched_after)
        object.__setattr__(self, "all_owner_install_root_sha256", install_root)

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self.prepared_batch.selected_env_ids

    @property
    def prepared_batch_sha256(self) -> str:
        return self.prepared_batch.canonical_sha256

    @classmethod
    def from_mapping(cls, value: object) -> "RevealFinalPreviewBatch":
        values = cls._mapping_values(value)
        values["prepared_batch"] = PreparedRevealBatch.from_mapping(
            values["prepared_batch"]
        )
        values["reveal_final_rows"] = tuple(
            RevealFinalInstallRow.from_mapping(row)
            for row in values["reveal_final_rows"]
        )
        return cls(**values)


@dataclass(frozen=True, eq=False)
class RevealFinalPreviewAbortReceipt(_SealedRecord):
    """Owner-issued proof that an unstaged preview lease was discarded."""

    KIND: ClassVar[str] = (
        "action_ball_continuous_reveal_final_preview_abort_receipt_v1"
    )

    integration_status: str
    phase_before: str
    phase_after: str
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    prepared_batch_sha256: str
    selected_env_ids: Tuple[int, ...]
    sampler_checkpoint_sha256: str
    owner_state_unchanged: bool
    policy_opportunity_created: bool
    terminal_claim_created: bool

    def __post_init__(self) -> None:
        if (
            self.integration_status != INTEGRATION_STATUS
            or self.phase_before != REVEAL_FINAL_PREVIEWED
            or self.phase_after != PREPARED
            or _exact_bool(
                self.owner_state_unchanged,
                label="owner_state_unchanged",
            )
            is not True
            or _exact_bool(
                self.policy_opportunity_created,
                label="policy_opportunity_created",
            )
            is not False
            or _exact_bool(
                self.terminal_claim_created,
                label="terminal_claim_created",
            )
            is not False
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-final preview abort phase/effects differ"
            )
        object.__setattr__(
            self,
            "reveal_final_preview_schema_version",
            _plain_int(
                self.reveal_final_preview_schema_version,
                label="reveal_final_preview_schema_version",
                minimum=1,
            ),
        )
        for name in (
            "reveal_final_preview_sha256",
            "prepared_batch_sha256",
            "sampler_checkpoint_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "selected_env_ids",
            _ordered_unique_env_ids(
                self.selected_env_ids, label="selected_env_ids"
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "RevealFinalPreviewAbortReceipt":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        return cls(**values)


@dataclass(frozen=True)
class InfrastructureCensorFact(_SealedRecord):
    """Externally pinned proof that an external reveal preflight failed."""

    KIND: ClassVar[str] = "action_ball_continuous_infrastructure_censor_fact_v1"

    env_id: int
    reveal_final_install_row_sha256: str
    observed_at_step: int
    reason: str
    failed_owner_kind: str
    failure_receipt_sha256: str
    producer_schema_sha256: str
    producer_source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_id", _plain_int(self.env_id, label="env_id"))
        object.__setattr__(
            self,
            "observed_at_step",
            _plain_int(self.observed_at_step, label="observed_at_step"),
        )
        reason = _text(self.reason, label="reason")
        if reason not in _INFRA_CENSOR_REASONS:
            raise ContinuousRuntimeTransactionError(
                "infrastructure censor reason differs"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "failed_owner_kind",
            _text(self.failed_owner_kind, label="failed_owner_kind"),
        )
        for name in (
            "reveal_final_install_row_sha256",
            "failure_receipt_sha256",
            "producer_schema_sha256",
            "producer_source_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )

    @classmethod
    def from_mapping(cls, value: object) -> "InfrastructureCensorFact":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class CensoredReveal(_SealedRecord):
    """One consumed scheduled identity with zero task, ball, or denominator install."""

    KIND: ClassVar[str] = "action_ball_continuous_censored_reveal_v1"

    integration_status: str
    phase: str
    sequence_advanced: bool
    sampler_consumed: bool
    task_installed: bool
    ball_installed: bool
    policy_opportunity_created: bool
    reveal_final_install_row: RevealFinalInstallRow
    censor_fact: InfrastructureCensorFact

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS or self.phase != INFRA_CENSORED:
            raise ContinuousRuntimeTransactionError(
                "censored reveal phase/status differs"
            )
        if not _exact_bool(self.sequence_advanced, label="sequence_advanced"):
            raise ContinuousRuntimeTransactionError(
                "censored reveal did not advance sequence"
            )
        if not _exact_bool(self.sampler_consumed, label="sampler_consumed"):
            raise ContinuousRuntimeTransactionError(
                "censored reveal did not consume sampler progress"
            )
        for name in (
            "task_installed",
            "ball_installed",
            "policy_opportunity_created",
        ):
            if _exact_bool(getattr(self, name), label=name):
                raise ContinuousRuntimeTransactionError(
                    "infrastructure censor installed task/ball/opportunity"
                )
        if not isinstance(self.reveal_final_install_row, RevealFinalInstallRow):
            raise ContinuousRuntimeTransactionError(
                "censored reveal final row type differs"
            )
        if not isinstance(self.censor_fact, InfrastructureCensorFact):
            raise ContinuousRuntimeTransactionError(
                "censored reveal fact type differs"
            )
        preview = self.reveal_final_install_row
        if (
            self.censor_fact.env_id != preview.env_id
            or self.censor_fact.reveal_final_install_row_sha256
            != preview.canonical_sha256
            or self.censor_fact.observed_at_step
            != preview.reveal_facts.reveal_step
        ):
            raise ContinuousRuntimeTransactionError(
                "censored reveal fact/final-row binding differs"
            )

    @property
    def prepared_reveal(self) -> PreparedReveal:
        return self.reveal_final_install_row.prepared_reveal

    @classmethod
    def from_mapping(cls, value: object) -> "CensoredReveal":
        values = cls._mapping_values(value)
        values["reveal_final_install_row"] = RevealFinalInstallRow.from_mapping(
            values["reveal_final_install_row"]
        )
        values["censor_fact"] = InfrastructureCensorFact.from_mapping(
            values["censor_fact"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CensoredRevealBatch(_SealedRecord):
    """One all-or-none K-env sampler advance with zero runtime install.

    Production terminal publication always carries
    ``terminal_boundary_marker``.  ``None`` is reserved solely for the
    private compatibility-pure censor helper and is never owner-registered as
    a production terminal receipt.
    """

    KIND: ClassVar[str] = "action_ball_continuous_censored_reveal_batch_v2"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 2

    integration_status: str
    phase: str
    reveal_final_preview: RevealFinalPreviewBatch
    censored_reveals: Tuple[CensoredReveal, ...]
    sampler_checkpoint_before_sha256: str
    sampler_checkpoint_after_sha256: str
    untouched_rows_before_sha256: str
    untouched_rows_after_sha256: str
    sampler_checkpoint_before: Mapping[str, object]
    sampler_checkpoint_after: Mapping[str, object]
    task_install_count: int
    ball_install_count: int
    policy_opportunity_count: int
    terminal_boundary_marker: Optional["RevealTerminalBoundaryMarker"] = None

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS or self.phase != INFRA_CENSORED:
            raise ContinuousRuntimeTransactionError(
                "censored batch phase/status differs"
            )
        if not isinstance(self.reveal_final_preview, RevealFinalPreviewBatch):
            raise ContinuousRuntimeTransactionError(
                "censored batch lacks reveal-final authority"
            )
        marker = self.terminal_boundary_marker
        if marker is not None and (
            type(marker) is not RevealTerminalBoundaryMarker
            or marker.selected_env_ids
            != self.reveal_final_preview.selected_env_ids
            or marker.reveal_final_preview_sha256
            != self.reveal_final_preview.canonical_sha256
            or marker.terminal_boundary_projection.decision
            != TERMINAL_DECISION_CENSOR
        ):
            raise ContinuousRuntimeTransactionError(
                "censored batch terminal boundary authority differs"
            )
        rows = tuple(self.censored_reveals)
        preview_rows = self.reveal_final_preview.reveal_final_rows
        if (
            len(rows) != len(preview_rows)
            or any(not isinstance(row, CensoredReveal) for row in rows)
            or tuple(row.reveal_final_install_row for row in rows)
            != preview_rows
        ):
            raise ContinuousRuntimeTransactionError(
                "censored batch row authority differs"
            )
        for name in (
            "task_install_count",
            "ball_install_count",
            "policy_opportunity_count",
        ):
            if _plain_int(getattr(self, name), label=name) != 0:
                raise ContinuousRuntimeTransactionError(
                    "censored batch created install/opportunity count"
                )
        before = _canonical_clone(self.sampler_checkpoint_before)
        after = _canonical_clone(self.sampler_checkpoint_after)
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or before != self.reveal_final_preview.sampler_checkpoint_before_commit
            or after != self.reveal_final_preview.sampler_checkpoint_after_commit
        ):
            raise ContinuousRuntimeTransactionError(
                "censored batch sampler authority differs"
            )
        before_sha = _sha256(
            self.sampler_checkpoint_before_sha256,
            label="sampler_checkpoint_before_sha256",
        )
        after_sha = _sha256(
            self.sampler_checkpoint_after_sha256,
            label="sampler_checkpoint_after_sha256",
        )
        untouched_before = _sha256(
            self.untouched_rows_before_sha256,
            label="untouched_rows_before_sha256",
        )
        untouched_after = _sha256(
            self.untouched_rows_after_sha256,
            label="untouched_rows_after_sha256",
        )
        if (
            before_sha != before["checkpoint_sha256"]
            or after_sha != after["checkpoint_sha256"]
            or untouched_before
            != self.reveal_final_preview.untouched_rows_before_sha256
            or untouched_after
            != self.reveal_final_preview.untouched_rows_after_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "censored batch sampler roots differ"
            )
        object.__setattr__(self, "censored_reveals", rows)
        object.__setattr__(self, "sampler_checkpoint_before", before)
        object.__setattr__(self, "sampler_checkpoint_after", after)
        object.__setattr__(self, "sampler_checkpoint_before_sha256", before_sha)
        object.__setattr__(self, "sampler_checkpoint_after_sha256", after_sha)
        object.__setattr__(self, "untouched_rows_before_sha256", untouched_before)
        object.__setattr__(self, "untouched_rows_after_sha256", untouched_after)

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self.reveal_final_preview.selected_env_ids

    @property
    def prepared_batch_sha256(self) -> str:
        return self.reveal_final_preview.prepared_batch_sha256

    @classmethod
    def from_mapping(cls, value: object) -> "CensoredRevealBatch":
        values = cls._mapping_values(value)
        values["reveal_final_preview"] = RevealFinalPreviewBatch.from_mapping(
            values["reveal_final_preview"]
        )
        values["censored_reveals"] = tuple(
            CensoredReveal.from_mapping(row)
            for row in values["censored_reveals"]
        )
        raw_marker = values["terminal_boundary_marker"]
        if raw_marker is not None:
            values["terminal_boundary_marker"] = (
                RevealTerminalBoundaryMarker.from_mapping(raw_marker)
            )
        return cls(**values)


@dataclass(frozen=True)
class TerminalBoundaryParticipantRoot(_SealedRecord):
    """One ordered participant root in a dependency-neutral terminal packet."""

    KIND: ClassVar[str] = (
        "action_ball_continuous_terminal_boundary_participant_root_v1"
    )

    participant_domain: str
    participant_kind: str
    participant_root_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_domain",
            _text(self.participant_domain, label="participant_domain"),
        )
        object.__setattr__(
            self,
            "participant_kind",
            _text(self.participant_kind, label="participant_kind"),
        )
        object.__setattr__(
            self,
            "participant_root_sha256",
            _sha256(
                self.participant_root_sha256,
                label="participant_root_sha256",
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "TerminalBoundaryParticipantRoot":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class TerminalBoundaryCensorEvidence(_SealedRecord):
    """Owner-issued batch-primary CENSOR cause applied to one selected env.

    ``primary_failure_env_id`` identifies the selected environment that
    triggered the normalized batch cause.  Every selected environment gets
    one ordered row, including healthy rows censored only because publication
    is all-or-none; all rows therefore repeat the same primary cause identity.
    """

    KIND: ClassVar[str] = (
        "action_ball_continuous_terminal_boundary_censor_evidence_v1"
    )

    env_id: int
    primary_failure_env_id: int
    participant_domain: str
    participant_kind: str
    participant_root_sha256: str
    failure_receipt_sha256: str
    reason: str
    censor_fact_sha256: str
    producer_schema_sha256: str
    producer_source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_id", _plain_int(self.env_id, label="env_id"))
        object.__setattr__(
            self,
            "primary_failure_env_id",
            _plain_int(
                self.primary_failure_env_id,
                label="primary_failure_env_id",
            ),
        )
        object.__setattr__(
            self,
            "participant_domain",
            _text(self.participant_domain, label="participant_domain"),
        )
        object.__setattr__(
            self,
            "participant_kind",
            _text(self.participant_kind, label="participant_kind"),
        )
        reason = _text(self.reason, label="reason")
        if reason not in _INFRA_CENSOR_REASONS:
            raise ContinuousRuntimeTransactionError(
                "terminal boundary CENSOR reason differs"
            )
        object.__setattr__(self, "reason", reason)
        for name in (
            "participant_root_sha256",
            "failure_receipt_sha256",
            "censor_fact_sha256",
            "producer_schema_sha256",
            "producer_source_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )

    @classmethod
    def from_mapping(cls, value: object) -> "TerminalBoundaryCensorEvidence":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class TerminalBoundaryProjection(_SealedRecord):
    """Normalized projection returned by one prebound boundary validator.

    The projection is the only downstream-neutral information R05 consumes.
    The retained validator closure, rather than any caller-supplied owner,
    establishes the opaque receipt's concrete owner identity.
    """

    KIND: ClassVar[str] = (
        "action_ball_continuous_terminal_boundary_projection_v1"
    )

    authority_domain: str
    authority_schema_sha256: str
    authority_source_sha256: str
    decision_mapping_schema_version: int
    source_decision: str
    decision: str
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    selected_env_ids: Tuple[int, ...]
    boundary_receipt_kind: str
    boundary_receipt_sha256: str
    boundary_packet_schema_version: int
    boundary_packet_sha256: str
    ordered_participant_roots: Tuple[TerminalBoundaryParticipantRoot, ...]
    ordered_censor_evidence: Tuple[TerminalBoundaryCensorEvidence, ...]

    def __post_init__(self) -> None:
        domain = _text(self.authority_domain, label="authority_domain")
        mapping_version = _plain_int(
            self.decision_mapping_schema_version,
            label="decision_mapping_schema_version",
            minimum=1,
        )
        source_decision = _text(
            self.source_decision, label="source_decision"
        )
        decision = _text(self.decision, label="decision")
        if (
            mapping_version
            != TERMINAL_BOUNDARY_DECISION_MAPPING_SCHEMA_VERSION
            or source_decision not in _TERMINAL_BOUNDARY_DECISION_MAP_V1
            or _TERMINAL_BOUNDARY_DECISION_MAP_V1[source_decision]
            != decision
        ):
            raise ContinuousRuntimeTransactionError(
                "terminal boundary source/terminal decision mapping differs"
            )
        preview_schema = _plain_int(
            self.reveal_final_preview_schema_version,
            label="reveal_final_preview_schema_version",
            minimum=1,
        )
        packet_schema = _plain_int(
            self.boundary_packet_schema_version,
            label="boundary_packet_schema_version",
            minimum=1,
        )
        selected = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        rows = tuple(self.ordered_participant_roots)
        if not rows or any(
            type(row) is not TerminalBoundaryParticipantRoot for row in rows
        ):
            raise ContinuousRuntimeTransactionError(
                "terminal boundary participant roots are empty or inexact"
            )
        identities = tuple(
            (row.participant_domain, row.participant_kind) for row in rows
        )
        if len(set(identities)) != len(identities):
            raise ContinuousRuntimeTransactionError(
                "terminal boundary participant roots are duplicated"
            )
        censor_evidence = tuple(self.ordered_censor_evidence)
        if any(
            type(row) is not TerminalBoundaryCensorEvidence
            for row in censor_evidence
        ):
            raise ContinuousRuntimeTransactionError(
                "terminal boundary CENSOR evidence type differs"
            )
        if decision == TERMINAL_DECISION_ACCEPT and censor_evidence:
            raise ContinuousRuntimeTransactionError(
                "ACCEPT terminal boundary carries CENSOR evidence"
            )
        censor_causes = {
            (
                row.primary_failure_env_id,
                row.participant_domain,
                row.participant_kind,
                row.participant_root_sha256,
                row.failure_receipt_sha256,
                row.reason,
                row.producer_schema_sha256,
                row.producer_source_sha256,
            )
            for row in censor_evidence
        }
        if decision == TERMINAL_DECISION_CENSOR and (
            tuple(row.env_id for row in censor_evidence) != selected
            or len(censor_causes) != 1
            or censor_evidence[0].primary_failure_env_id not in selected
            or any(
                sum(
                    participant.participant_domain
                    == evidence.participant_domain
                    and participant.participant_kind
                    == evidence.participant_kind
                    and participant.participant_root_sha256
                    == evidence.participant_root_sha256
                    for participant in rows
                )
                != 1
                for evidence in censor_evidence
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "CENSOR evidence selection/participant root differs"
            )
        object.__setattr__(self, "authority_domain", domain)
        object.__setattr__(
            self,
            "authority_schema_sha256",
            _sha256(
                self.authority_schema_sha256,
                label="authority_schema_sha256",
            ),
        )
        object.__setattr__(
            self,
            "authority_source_sha256",
            _sha256(
                self.authority_source_sha256,
                label="authority_source_sha256",
            ),
        )
        object.__setattr__(
            self, "decision_mapping_schema_version", mapping_version
        )
        object.__setattr__(self, "source_decision", source_decision)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(
            self, "reveal_final_preview_schema_version", preview_schema
        )
        object.__setattr__(
            self,
            "reveal_final_preview_sha256",
            _sha256(
                self.reveal_final_preview_sha256,
                label="reveal_final_preview_sha256",
            ),
        )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(
            self,
            "boundary_receipt_kind",
            _text(self.boundary_receipt_kind, label="boundary_receipt_kind"),
        )
        object.__setattr__(
            self,
            "boundary_receipt_sha256",
            _sha256(
                self.boundary_receipt_sha256,
                label="boundary_receipt_sha256",
            ),
        )
        object.__setattr__(
            self, "boundary_packet_schema_version", packet_schema
        )
        object.__setattr__(
            self,
            "boundary_packet_sha256",
            _sha256(
                self.boundary_packet_sha256,
                label="boundary_packet_sha256",
            ),
        )
        object.__setattr__(self, "ordered_participant_roots", rows)
        object.__setattr__(
            self, "ordered_censor_evidence", censor_evidence
        )

    @classmethod
    def from_mapping(cls, value: object) -> "TerminalBoundaryProjection":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["ordered_participant_roots"] = tuple(
            TerminalBoundaryParticipantRoot.from_mapping(row)
            for row in values["ordered_participant_roots"]
        )
        values["ordered_censor_evidence"] = tuple(
            TerminalBoundaryCensorEvidence.from_mapping(row)
            for row in values["ordered_censor_evidence"]
        )
        return cls(**values)


@dataclass(frozen=True)
class PreparedTerminalContentPin(_SealedRecord):
    """Prebuilt canonical bytes for the exact future terminal receipt."""

    KIND: ClassVar[str] = (
        "action_ball_continuous_prepared_terminal_content_pin_v1"
    )

    terminal_schema_version: int
    terminal_kind: str
    terminal_canonical_sha256: str
    content_bytes_base64: str
    content_byte_length: int
    content_bytes_sha256: str

    @cached_property
    def canonical_sha256(self) -> str:
        """Cache the root because every field is a frozen scalar value."""

        return canonical_sha256(self.payload())

    def __post_init__(self) -> None:
        schema_version = _plain_int(
            self.terminal_schema_version,
            label="terminal_schema_version",
            minimum=1,
        )
        terminal_kind = _text(self.terminal_kind, label="terminal_kind")
        terminal_sha = _sha256(
            self.terminal_canonical_sha256,
            label="terminal_canonical_sha256",
        )
        byte_length = _plain_int(
            self.content_byte_length,
            label="content_byte_length",
            minimum=1,
        )
        content_sha = _sha256(
            self.content_bytes_sha256, label="content_bytes_sha256"
        )
        if type(self.content_bytes_base64) is not str:
            raise ContinuousRuntimeTransactionError(
                "content_bytes_base64 must be an exact string"
            )
        try:
            raw = base64.b64decode(
                self.content_bytes_base64.encode("ascii"), validate=True
            )
            decoded = json.loads(raw.decode("ascii"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ContinuousRuntimeTransactionError(
                "prepared terminal content is not canonical JSON ASCII"
            ) from exc
        if (
            len(raw) != byte_length
            or hashlib.sha256(raw).hexdigest() != content_sha
            or not isinstance(decoded, Mapping)
            or _canonical_json_bytes(decoded) != raw
            or decoded.get("schema_version") != schema_version
            or decoded.get("kind") != terminal_kind
            or decoded.get("canonical_sha256") != terminal_sha
            or canonical_sha256(
                {
                    key: decoded[key]
                    for key in decoded
                    if key != "canonical_sha256"
                }
            )
            != terminal_sha
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared terminal content pin differs from its sealed mapping"
            )
        object.__setattr__(self, "terminal_schema_version", schema_version)
        object.__setattr__(self, "terminal_kind", terminal_kind)
        object.__setattr__(
            self, "terminal_canonical_sha256", terminal_sha
        )
        object.__setattr__(self, "content_byte_length", byte_length)
        object.__setattr__(self, "content_bytes_sha256", content_sha)

    @classmethod
    def from_mapping(cls, value: object) -> "PreparedTerminalContentPin":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class RevealTerminalBoundaryMarker(_SealedRecord):
    """R05-normalized terminal boundary retained for either decision."""

    KIND: ClassVar[str] = (
        "action_ball_continuous_reveal_terminal_boundary_marker_v1"
    )

    terminal_boundary_authority_sha256: str
    terminal_boundary_projection: TerminalBoundaryProjection

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_boundary_authority_sha256",
            _sha256(
                self.terminal_boundary_authority_sha256,
                label="terminal_boundary_authority_sha256",
            ),
        )
        if type(self.terminal_boundary_projection) is not TerminalBoundaryProjection:
            raise ContinuousRuntimeTransactionError(
                "terminal boundary marker projection type differs"
            )

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self.terminal_boundary_projection.selected_env_ids

    @property
    def reveal_final_preview_sha256(self) -> str:
        return self.terminal_boundary_projection.reveal_final_preview_sha256

    @classmethod
    def from_mapping(cls, value: object) -> "RevealTerminalBoundaryMarker":
        values = cls._mapping_values(value)
        values["terminal_boundary_projection"] = (
            TerminalBoundaryProjection.from_mapping(
                values["terminal_boundary_projection"]
            )
        )
        return cls(**values)


@dataclass(frozen=True)
class RevealPrepareBoundaryMarker(_SealedRecord):
    """One packed-D2H, all-owner prearm decision for a reveal batch."""

    KIND: ClassVar[str] = (
        "action_ball_continuous_reveal_prepare_boundary_marker_v1"
    )

    selected_env_ids: Tuple[int, ...]
    reveal_final_preview_sha256: str
    boundary_packet_version: int
    boundary_packet_root_sha256: str
    boundary_transfer_count: int
    selected_pass_count: int
    selected_fault_count: int
    ordered_child_token_roots: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        selected = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        preview_sha = _sha256(
            self.reveal_final_preview_sha256,
            label="reveal_final_preview_sha256",
        )
        version = _plain_int(
            self.boundary_packet_version,
            label="boundary_packet_version",
            minimum=1,
        )
        transfer_count = _plain_int(
            self.boundary_transfer_count,
            label="boundary_transfer_count",
            minimum=0,
        )
        pass_count = _plain_int(
            self.selected_pass_count,
            label="selected_pass_count",
            minimum=0,
        )
        fault_count = _plain_int(
            self.selected_fault_count,
            label="selected_fault_count",
            minimum=0,
        )
        if (
            transfer_count != 1
            or pass_count != len(selected)
            or fault_count != 0
        ):
            raise ContinuousRuntimeTransactionError(
                "reveal-prepare boundary is not one all-selected pass packet"
            )
        raw_roots = tuple(self.ordered_child_token_roots)
        if len(raw_roots) != len(_PREARM_CHILD_OWNER_KINDS):
            raise ContinuousRuntimeTransactionError(
                "prearm marker child-token root width differs"
            )
        roots = []
        for index, raw in enumerate(raw_roots):
            if (
                not isinstance(raw, Sequence)
                or isinstance(raw, (str, bytes))
                or len(raw) != 2
                or type(raw[0]) is not str
                or raw[0] != _PREARM_CHILD_OWNER_KINDS[index]
            ):
                raise ContinuousRuntimeTransactionError(
                    "prearm marker child-token root order/kind differs"
                )
            roots.append(
                (
                    raw[0],
                    _sha256(
                        raw[1],
                        label=(
                            "ordered_child_token_roots"
                            f"[{index}][1]"
                        ),
                    ),
                )
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "reveal_final_preview_sha256", preview_sha)
        object.__setattr__(self, "boundary_packet_version", version)
        object.__setattr__(
            self,
            "boundary_packet_root_sha256",
            _sha256(
                self.boundary_packet_root_sha256,
                label="boundary_packet_root_sha256",
            ),
        )
        object.__setattr__(self, "boundary_transfer_count", transfer_count)
        object.__setattr__(self, "selected_pass_count", pass_count)
        object.__setattr__(self, "selected_fault_count", fault_count)
        object.__setattr__(self, "ordered_child_token_roots", tuple(roots))

    @classmethod
    def from_mapping(cls, value: object) -> "RevealPrepareBoundaryMarker":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["ordered_child_token_roots"] = tuple(
            tuple(row) for row in values["ordered_child_token_roots"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CommittedRevealBatch(_SealedRecord):
    """One all-or-none K-environment commit and its sampler state swap."""

    KIND: ClassVar[str] = "action_ball_continuous_committed_reveal_batch_v2"
    RECORD_SCHEMA_VERSION: ClassVar[int] = 2

    integration_status: str
    phase: str
    runtime_wiring_connected: bool
    identity_committed: bool
    policy_opportunity_created: bool
    reveal_final_preview: RevealFinalPreviewBatch
    global_prearm_marker: Union[
        RevealTerminalBoundaryMarker, RevealPrepareBoundaryMarker
    ]
    prepared_batch: PreparedRevealBatch
    sampler_checkpoint_before_commit_sha256: str
    sampler_checkpoint_after_commit_sha256: str
    untouched_rows_before_sha256: str
    untouched_rows_after_sha256: str
    sampler_checkpoint_before_commit: Mapping[str, object]
    sampler_checkpoint_after_commit: Mapping[str, object]
    committed_reveals: Tuple[CommittedReveal, ...]

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError(
                "committed batch lost PRE_INTEGRATION_HOLD status"
            )
        if self.phase != COMMITTED:
            raise ContinuousRuntimeTransactionError(
                "committed batch phase differs"
            )
        if self.runtime_wiring_connected:
            raise ContinuousRuntimeTransactionError(
                "pure committed batch cannot claim runtime wiring"
            )
        if not self.identity_committed or not self.policy_opportunity_created:
            raise ContinuousRuntimeTransactionError(
                "committed batch did not enter its opportunity lanes"
            )
        if not isinstance(self.reveal_final_preview, RevealFinalPreviewBatch):
            raise ContinuousRuntimeTransactionError(
                "committed batch lacks exact reveal-final authority"
            )
        marker = self.global_prearm_marker
        if (
            type(marker)
            not in (RevealTerminalBoundaryMarker, RevealPrepareBoundaryMarker)
            or marker.selected_env_ids
            != self.reveal_final_preview.selected_env_ids
            or marker.reveal_final_preview_sha256
            != self.reveal_final_preview.canonical_sha256
            or (
                type(marker) is RevealTerminalBoundaryMarker
                and marker.terminal_boundary_projection.decision
                != TERMINAL_DECISION_ACCEPT
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch global prearm authority differs"
            )
        if not isinstance(self.prepared_batch, PreparedRevealBatch):
            raise ContinuousRuntimeTransactionError(
                "committed batch prepared authority type differs"
            )
        if self.reveal_final_preview.prepared_batch != self.prepared_batch:
            raise ContinuousRuntimeTransactionError(
                "committed batch reveal-final/prepared authority differs"
            )
        rows = tuple(self.committed_reveals)
        if len(rows) != len(self.prepared_batch.prepared_reveals) or any(
            not isinstance(row, CommittedReveal) for row in rows
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch row width/type differs"
            )
        if tuple(row.prepared_reveal for row in rows) != (
            self.prepared_batch.prepared_reveals
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch rows differ from prepared authority"
            )
        if any(
            row.prepared_reveal != preview_row.prepared_reveal
            or row.reveal_facts != preview_row.reveal_facts
            or row.ball_slot_plan != preview_row.ball_slot_plan
            for row, preview_row in zip(
                rows, self.reveal_final_preview.reveal_final_rows
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch differs from reveal-final install facts"
            )
        env_ids = self.prepared_batch.selected_env_ids
        before = _canonical_clone(self.sampler_checkpoint_before_commit)
        after = _canonical_clone(self.sampler_checkpoint_after_commit)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ContinuousRuntimeTransactionError(
                "committed batch sampler checkpoints must be mappings"
            )
        before_sha = _sha256(
            self.sampler_checkpoint_before_commit_sha256,
            label="sampler_checkpoint_before_commit_sha256",
        )
        after_sha = _sha256(
            self.sampler_checkpoint_after_commit_sha256,
            label="sampler_checkpoint_after_commit_sha256",
        )
        if before_sha != _sampler_checkpoint_sha256(
            before, label="sampler_checkpoint_before_commit"
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch before checkpoint SHA differs"
            )
        if after_sha != _sampler_checkpoint_sha256(
            after, label="sampler_checkpoint_after_commit"
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch after checkpoint SHA differs"
            )
        untouched_before = _sha256(
            self.untouched_rows_before_sha256,
            label="untouched_rows_before_sha256",
        )
        untouched_after = _sha256(
            self.untouched_rows_after_sha256,
            label="untouched_rows_after_sha256",
        )
        if untouched_before != _sampler_untouched_rows_root(before, env_ids):
            raise ContinuousRuntimeTransactionError(
                "committed batch untouched-before root differs"
            )
        if untouched_after != _sampler_untouched_rows_root(after, env_ids):
            raise ContinuousRuntimeTransactionError(
                "committed batch untouched-after root differs"
            )
        if (
            untouched_before != untouched_after
            or _sampler_rows_excluding(before, env_ids)
            != _sampler_rows_excluding(after, env_ids)
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch changed an unselected sampler row"
            )
        for row in self.prepared_batch.prepared_reveals:
            env_id = row.request.env_id
            if (
                _sampler_env_row(before, env_id)
                != row.sampler_before_env_state
                or _sampler_env_row(after, env_id)
                != row.sampler_after_env_state
            ):
                raise ContinuousRuntimeTransactionError(
                    "committed batch selected sampler row binding differs"
                )
        if (
            _merge_sampler_env_rows(
                before, self.prepared_batch.prepared_reveals
            )
            != after
        ):
            raise ContinuousRuntimeTransactionError(
                "committed batch full-owner transition differs"
            )
        object.__setattr__(self, "committed_reveals", rows)
        object.__setattr__(self, "sampler_checkpoint_before_commit", before)
        object.__setattr__(self, "sampler_checkpoint_after_commit", after)
        object.__setattr__(
            self, "sampler_checkpoint_before_commit_sha256", before_sha
        )
        object.__setattr__(
            self, "sampler_checkpoint_after_commit_sha256", after_sha
        )
        object.__setattr__(self, "untouched_rows_before_sha256", untouched_before)
        object.__setattr__(self, "untouched_rows_after_sha256", untouched_after)

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self.prepared_batch.selected_env_ids

    @property
    def batch_sha256(self) -> str:
        return self.canonical_sha256

    @classmethod
    def from_mapping(cls, value: object) -> "CommittedRevealBatch":
        values = cls._mapping_values(value)
        values["reveal_final_preview"] = RevealFinalPreviewBatch.from_mapping(
            values["reveal_final_preview"]
        )
        raw_marker = values["global_prearm_marker"]
        if not isinstance(raw_marker, Mapping):
            raise ContinuousRuntimeTransactionError(
                "committed batch global prearm marker must be a mapping"
            )
        marker_kind = raw_marker.get("kind")
        if marker_kind == RevealTerminalBoundaryMarker.KIND:
            values["global_prearm_marker"] = (
                RevealTerminalBoundaryMarker.from_mapping(raw_marker)
            )
        elif marker_kind == RevealPrepareBoundaryMarker.KIND:
            values["global_prearm_marker"] = (
                RevealPrepareBoundaryMarker.from_mapping(raw_marker)
            )
        else:
            raise ContinuousRuntimeTransactionError(
                "committed batch global prearm marker kind differs"
            )
        values["prepared_batch"] = PreparedRevealBatch.from_mapping(
            values["prepared_batch"]
        )
        values["committed_reveals"] = tuple(
            CommittedReveal.from_mapping(row) for row in values["committed_reveals"]
        )
        return cls(**values)


@dataclass(frozen=True)
class AbortBatchReceipt(_SealedRecord):
    """Proof that one entire private batch was discarded without sampler change."""

    KIND: ClassVar[str] = "action_ball_continuous_abort_batch_receipt_v1"

    integration_status: str
    phase_before: str
    phase_after: str
    prepared_batch_sha256: str
    selected_env_ids: Tuple[int, ...]
    task_birth_snapshot_ids: Tuple[str, ...]
    prepared_reveal_sha256s: Tuple[str, ...]
    prepared_batch: PreparedRevealBatch
    sampler_checkpoint_sha256: str
    untouched_rows_sha256: str
    sampler_checkpoint: Mapping[str, object]
    policy_opportunity_created: bool
    abort_receipts: Tuple[AbortReceipt, ...]

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError("batch abort status differs")
        if self.phase_before != PREPARED or self.phase_after != EMPTY:
            raise ContinuousRuntimeTransactionError(
                "batch abort phase transition differs"
            )
        if self.policy_opportunity_created:
            raise ContinuousRuntimeTransactionError(
                "batch abort created a policy opportunity"
            )
        if not isinstance(self.prepared_batch, PreparedRevealBatch):
            raise ContinuousRuntimeTransactionError(
                "batch abort prepared authority type differs"
            )
        env_ids = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        births = tuple(
            _text(value, label=f"task_birth_snapshot_ids[{index}]")
            for index, value in enumerate(self.task_birth_snapshot_ids)
        )
        rows = tuple(self.abort_receipts)
        prepared_shas = tuple(
            _sha256(value, label=f"prepared_reveal_sha256s[{index}]")
            for index, value in enumerate(self.prepared_reveal_sha256s)
        )
        if (
            len(births) != len(env_ids)
            or len(prepared_shas) != len(env_ids)
            or len(rows) != len(env_ids)
            or any(not isinstance(row, AbortReceipt) for row in rows)
            or tuple(row.env_id for row in rows) != env_ids
            or tuple(row.prepared_reveal_sha256 for row in rows)
            != prepared_shas
            or self.prepared_batch.selected_env_ids != env_ids
            or self.prepared_batch.task_birth_snapshot_ids != births
            or tuple(
                row.canonical_sha256
                for row in self.prepared_batch.prepared_reveals
            )
            != prepared_shas
        ):
            raise ContinuousRuntimeTransactionError(
                "batch abort row width/order differs"
            )
        object.__setattr__(
            self,
            "prepared_batch_sha256",
            _sha256(
                self.prepared_batch_sha256, label="prepared_batch_sha256"
            ),
        )
        if self.prepared_batch_sha256 != self.prepared_batch.canonical_sha256:
            raise ContinuousRuntimeTransactionError(
                "batch abort prepared authority SHA differs"
            )
        object.__setattr__(
            self,
            "sampler_checkpoint_sha256",
            _sha256(
                self.sampler_checkpoint_sha256,
                label="sampler_checkpoint_sha256",
            ),
        )
        object.__setattr__(
            self,
            "untouched_rows_sha256",
            _sha256(self.untouched_rows_sha256, label="untouched_rows_sha256"),
        )
        object.__setattr__(self, "selected_env_ids", env_ids)
        object.__setattr__(self, "task_birth_snapshot_ids", births)
        object.__setattr__(self, "prepared_reveal_sha256s", prepared_shas)
        object.__setattr__(self, "abort_receipts", rows)
        if any(
            row.sampler_checkpoint_sha256 != self.sampler_checkpoint_sha256
            for row in rows
        ):
            raise ContinuousRuntimeTransactionError(
                "batch abort sampler checkpoint binding differs"
            )
        sampler_checkpoint = _canonical_clone(self.sampler_checkpoint)
        if not isinstance(sampler_checkpoint, dict):
            raise ContinuousRuntimeTransactionError(
                "batch abort sampler checkpoint must be a mapping"
            )
        if (
            _sampler_checkpoint_sha256(
                sampler_checkpoint, label="sampler_checkpoint"
            )
            != self.sampler_checkpoint_sha256
            or _sampler_untouched_rows_root(sampler_checkpoint, env_ids)
            != self.untouched_rows_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "batch abort untouched sampler binding differs"
            )
        object.__setattr__(self, "sampler_checkpoint", sampler_checkpoint)

    @classmethod
    def from_mapping(cls, value: object) -> "AbortBatchReceipt":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["task_birth_snapshot_ids"] = tuple(
            values["task_birth_snapshot_ids"]
        )
        values["prepared_reveal_sha256s"] = tuple(
            values["prepared_reveal_sha256s"]
        )
        values["prepared_batch"] = PreparedRevealBatch.from_mapping(
            values["prepared_batch"]
        )
        values["abort_receipts"] = tuple(
            AbortReceipt.from_mapping(row) for row in values["abort_receipts"]
        )
        return cls(**values)


def _chain_transcript_root(receipt_shas: Sequence[str]) -> str:
    root = canonical_sha256(
        {
            "kind": "action_ball_continuous_retired_chain_transcript_root_v1",
            "receipts": [],
        }
    )
    for receipt_sha in receipt_shas:
        root = canonical_sha256(
            {
                "kind": "action_ball_continuous_retired_chain_transcript_step_v1",
                "parent_sha256": root,
                "committed_reveal_sha256": receipt_sha,
            }
        )
    return root


def _sequence_chain_transcript_root(
    event_kinds: Sequence[str], event_shas: Sequence[str]
) -> str:
    root = canonical_sha256(
        {
            "kind": "action_ball_continuous_sequence_chain_root_v1",
            "events": [],
        }
    )
    for kind, event_sha in zip(event_kinds, event_shas):
        root = canonical_sha256(
            {
                "kind": "action_ball_continuous_sequence_chain_step_v1",
                "parent_sha256": root,
                "event_kind": kind,
                "event_sha256": event_sha,
            }
        )
    return root


@dataclass(frozen=True)
class TrueResetClosureReceipt(_SealedRecord):
    """External zero-open proof required before retiring one active chain."""

    KIND: ClassVar[str] = "action_ball_continuous_true_reset_closure_v3"

    env_id: int
    prior_reset_generation: int
    latest_sequence_event_kind: str
    latest_sequence_event_sha256: str
    latest_committed_reveal_sha256: Optional[str]
    latest_outcome_key_sha256: Optional[str]
    closed_at_step: int
    closure_disposition: str
    open_flight_count: int
    open_mailbox_count: int
    settled_unpaid_mailbox_count: int
    open_recovery_count: int
    open_payment_epoch_count: int
    hard_terminal_pending_outcome_count: int
    flight_closure_authority_sha256: str
    mailbox_closure_authority_sha256: str
    recovery_closure_authority_sha256: str
    payment_closure_authority_sha256: str
    hard_terminal_censor_authority_sha256: str
    closure_snapshot_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, label="env_id")
        )
        object.__setattr__(
            self,
            "prior_reset_generation",
            _plain_int(
                self.prior_reset_generation,
                label="prior_reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "closed_at_step",
            _plain_int(self.closed_at_step, label="closed_at_step"),
        )
        disposition = _text(
            self.closure_disposition, label="closure_disposition"
        )
        if disposition not in (
            "CLOSED_AFTER_DEADLINE",
            "CENSORED_TRUE_RESET",
        ):
            raise ContinuousRuntimeTransactionError(
                "true reset closure disposition differs"
            )
        object.__setattr__(self, "closure_disposition", disposition)
        sequence_kind = _text(
            self.latest_sequence_event_kind,
            label="latest_sequence_event_kind",
        )
        if sequence_kind not in ("COMMITTED", "INFRA_CENSORED"):
            raise ContinuousRuntimeTransactionError(
                "true reset latest sequence event kind differs"
            )
        object.__setattr__(
            self, "latest_sequence_event_kind", sequence_kind
        )
        object.__setattr__(
            self,
            "latest_sequence_event_sha256",
            _sha256(
                self.latest_sequence_event_sha256,
                label="latest_sequence_event_sha256",
            ),
        )
        for name in (
            "open_flight_count",
            "open_mailbox_count",
            "settled_unpaid_mailbox_count",
            "open_recovery_count",
            "open_payment_epoch_count",
            "hard_terminal_pending_outcome_count",
        ):
            value = _plain_int(getattr(self, name), label=name)
            if value != 0:
                raise ContinuousRuntimeTransactionError(
                    f"true reset requires {name} == 0"
                )
            object.__setattr__(self, name, value)
        for name in (
            "flight_closure_authority_sha256",
            "mailbox_closure_authority_sha256",
            "recovery_closure_authority_sha256",
            "payment_closure_authority_sha256",
            "hard_terminal_censor_authority_sha256",
            "closure_snapshot_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if (self.latest_committed_reveal_sha256 is None) != (
            self.latest_outcome_key_sha256 is None
        ):
            raise ContinuousRuntimeTransactionError(
                "true reset latest committed/outcome optionality differs"
            )
        for name in (
            "latest_committed_reveal_sha256",
            "latest_outcome_key_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(value, label=name))

    @classmethod
    def from_mapping(cls, value: object) -> "TrueResetClosureReceipt":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class TrueResetClosureBatch(_SealedRecord):
    """Externally pinned all-selected closure barrier; R05 is not its producer."""

    KIND: ClassVar[str] = "action_ball_continuous_true_reset_closure_batch_v1"

    integration_status: str
    selected_env_ids: Tuple[int, ...]
    owner_checkpoint_before_sha256: str
    closure_barrier_id: str
    closure_barrier_sha256: str
    producer_schema_sha256: str
    producer_source_sha256: str
    receipts: Tuple[TrueResetClosureReceipt, ...]

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError(
                "closure batch integration status differs"
            )
        selected = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        rows = tuple(self.receipts)
        if (
            len(rows) != len(selected)
            or any(not isinstance(row, TrueResetClosureReceipt) for row in rows)
            or tuple(row.env_id for row in rows) != selected
        ):
            raise ContinuousRuntimeTransactionError(
                "closure batch row identity/order differs"
            )
        object.__setattr__(
            self,
            "closure_barrier_id",
            _text(self.closure_barrier_id, label="closure_barrier_id"),
        )
        for name in (
            "owner_checkpoint_before_sha256",
            "closure_barrier_sha256",
            "producer_schema_sha256",
            "producer_source_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "receipts", rows)

    @classmethod
    def from_mapping(cls, value: object) -> "TrueResetClosureBatch":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["receipts"] = tuple(
            TrueResetClosureReceipt.from_mapping(row)
            for row in values["receipts"]
        )
        return cls(**values)


@dataclass(frozen=True)
class RetiredContinuousChain(_SealedRecord):
    """Durable historical evidence for one closed reset generation."""

    KIND: ClassVar[str] = "action_ball_continuous_retired_chain_v2"

    env_id: int
    reset_generation: int
    birth_sha256: str
    run_id: str
    carry_chain_id: str
    first_scheduled_ordinal: int
    last_scheduled_ordinal: int
    first_sampler_generation: int
    last_sampler_generation: int
    sequence_event_kinds: Tuple[str, ...]
    sequence_event_sha256s: Tuple[str, ...]
    sequence_chain_transcript_sha256: str
    committed_reveal_sha256s: Tuple[str, ...]
    committed_chain_transcript_sha256: str
    sampler_high_water_env_state_sha256: str
    latest_outcome_key_sha256: Optional[str]
    closure_receipt: TrueResetClosureReceipt
    next_q0_request: ContinuousPrepareRequest

    def __post_init__(self) -> None:
        for name, minimum in (
            ("env_id", 0),
            ("reset_generation", 1),
            ("first_scheduled_ordinal", 0),
            ("last_scheduled_ordinal", 0),
            ("first_sampler_generation", 1),
            ("last_sampler_generation", 1),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name, minimum=minimum),
            )
        if (
            self.first_scheduled_ordinal != 0
            or self.last_scheduled_ordinal < self.first_scheduled_ordinal
            or self.last_sampler_generation < self.first_sampler_generation
            or self.last_sampler_generation - self.first_sampler_generation
            != self.last_scheduled_ordinal
        ):
            raise ContinuousRuntimeTransactionError(
                "retired chain ordinal/generation range differs"
            )
        for name in ("run_id", "carry_chain_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "birth_sha256",
            _sha256(self.birth_sha256, label="birth_sha256"),
        )
        sequence_kinds = tuple(
            _text(value, label="sequence_event_kinds")
            for value in self.sequence_event_kinds
        )
        sequence_shas = tuple(
            _sha256(value, label="sequence_event_sha256s")
            for value in self.sequence_event_sha256s
        )
        if (
            len(sequence_kinds) != self.last_scheduled_ordinal + 1
            or len(sequence_shas) != len(sequence_kinds)
            or len(set(sequence_shas)) != len(sequence_shas)
            or any(
                kind not in ("COMMITTED", "INFRA_CENSORED")
                for kind in sequence_kinds
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "retired chain sequence event range differs"
            )
        sequence_transcript = _sha256(
            self.sequence_chain_transcript_sha256,
            label="sequence_chain_transcript_sha256",
        )
        if sequence_transcript != _sequence_chain_transcript_root(
            sequence_kinds, sequence_shas
        ):
            raise ContinuousRuntimeTransactionError(
                "retired chain sequence transcript differs"
            )
        receipts = tuple(
            _sha256(value, label="committed_reveal_sha256s")
            for value in self.committed_reveal_sha256s
        )
        if (
            len(set(receipts)) != len(receipts)
            or receipts
            != tuple(
                event_sha
                for kind, event_sha in zip(sequence_kinds, sequence_shas)
                if kind == "COMMITTED"
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "retired chain committed receipt range differs"
            )
        object.__setattr__(self, "committed_reveal_sha256s", receipts)
        object.__setattr__(self, "sequence_event_kinds", sequence_kinds)
        object.__setattr__(self, "sequence_event_sha256s", sequence_shas)
        object.__setattr__(
            self, "sequence_chain_transcript_sha256", sequence_transcript
        )
        transcript = _sha256(
            self.committed_chain_transcript_sha256,
            label="committed_chain_transcript_sha256",
        )
        if transcript != _chain_transcript_root(receipts):
            raise ContinuousRuntimeTransactionError(
                "retired chain transcript differs"
            )
        object.__setattr__(
            self, "committed_chain_transcript_sha256", transcript
        )
        for name in ("sampler_high_water_env_state_sha256",):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if self.latest_outcome_key_sha256 is not None:
            object.__setattr__(
                self,
                "latest_outcome_key_sha256",
                _sha256(
                    self.latest_outcome_key_sha256,
                    label="latest_outcome_key_sha256",
                ),
            )
        if not isinstance(self.closure_receipt, TrueResetClosureReceipt):
            raise ContinuousRuntimeTransactionError(
                "retired chain closure receipt type differs"
            )
        if not isinstance(self.next_q0_request, ContinuousPrepareRequest):
            raise ContinuousRuntimeTransactionError(
                "retired chain next Q0 request type differs"
            )
        next_request = self.next_q0_request
        if (
            self.closure_receipt.env_id != self.env_id
            or self.closure_receipt.prior_reset_generation
            != self.reset_generation
            or self.closure_receipt.latest_outcome_key_sha256
            != self.latest_outcome_key_sha256
            or self.closure_receipt.latest_sequence_event_kind
            != sequence_kinds[-1]
            or self.closure_receipt.latest_sequence_event_sha256
            != sequence_shas[-1]
            or self.closure_receipt.latest_committed_reveal_sha256
            != (None if not receipts else receipts[-1])
            or next_request.env_id != self.env_id
            or next_request.reset_generation != self.reset_generation + 1
            or next_request.scheduled_ordinal != 0
            or next_request.runtime_swing_generation != 0
            or next_request.outcome_shot_index != 1
            or next_request.previous_ball_slot_index is not None
            or next_request.sampler_generation != self.last_sampler_generation + 1
            or any(
                slot.lifecycle_state != BALL_EMPTY
                for slot in next_request.ball_slots
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "retired chain closure/next-Q0 binding differs"
            )

    @classmethod
    def from_mapping(cls, value: object) -> "RetiredContinuousChain":
        values = cls._mapping_values(value)
        values["sequence_event_kinds"] = tuple(values["sequence_event_kinds"])
        values["sequence_event_sha256s"] = tuple(
            values["sequence_event_sha256s"]
        )
        values["committed_reveal_sha256s"] = tuple(
            values["committed_reveal_sha256s"]
        )
        values["closure_receipt"] = TrueResetClosureReceipt.from_mapping(
            values["closure_receipt"]
        )
        values["next_q0_request"] = ContinuousPrepareRequest.from_mapping(
            values["next_q0_request"]
        )
        return cls(**values)


@dataclass(frozen=True)
class PendingTrueResetQ0(_SealedRecord):
    """The only Q0 identity admitted after one selected true reset."""

    KIND: ClassVar[str] = "action_ball_continuous_pending_true_reset_q0_v1"

    env_id: int
    retired_chain_sha256: str
    sampler_before_env_state_sha256: str
    next_q0_request: ContinuousPrepareRequest

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, label="env_id")
        )
        for name in (
            "retired_chain_sha256",
            "sampler_before_env_state_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if (
            not isinstance(self.next_q0_request, ContinuousPrepareRequest)
            or self.next_q0_request.env_id != self.env_id
            or self.next_q0_request.scheduled_ordinal != 0
        ):
            raise ContinuousRuntimeTransactionError(
                "pending true-reset Q0 identity differs"
            )

    @classmethod
    def from_mapping(cls, value: object) -> "PendingTrueResetQ0":
        values = cls._mapping_values(value)
        values["next_q0_request"] = ContinuousPrepareRequest.from_mapping(
            values["next_q0_request"]
        )
        return cls(**values)


@dataclass(frozen=True)
class TrueResetBatchReceipt(_SealedRecord):
    """One selected K-env active-chain retirement and new-Q0 registration."""

    KIND: ClassVar[str] = "action_ball_continuous_true_reset_batch_v2"

    integration_status: str
    selected_env_ids: Tuple[int, ...]
    retired_chains: Tuple[RetiredContinuousChain, ...]
    pending_q0: Tuple[PendingTrueResetQ0, ...]
    sampler_checkpoint_sha256: str
    committed_transcript_sha256: str
    active_owner_root_before_sha256: str
    active_owner_root_after_sha256: str
    unselected_active_owner_root_before_sha256: str
    unselected_active_owner_root_after_sha256: str
    unselected_prepared_owner_root_sha256: str
    parent_true_reset_transcript_sha256: str
    external_closure_batch: TrueResetClosureBatch
    external_closure_batch_sha256: str
    policy_opportunity_created: bool

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRuntimeTransactionError(
                "true reset batch integration status differs"
            )
        selected = _ordered_unique_env_ids(
            self.selected_env_ids, label="selected_env_ids"
        )
        retired = tuple(self.retired_chains)
        pending = tuple(self.pending_q0)
        if (
            len(retired) != len(selected)
            or len(pending) != len(selected)
            or any(not isinstance(row, RetiredContinuousChain) for row in retired)
            or any(not isinstance(row, PendingTrueResetQ0) for row in pending)
            or tuple(row.env_id for row in retired) != selected
            or tuple(row.env_id for row in pending) != selected
            or any(
                registration.retired_chain_sha256 != chain.canonical_sha256
                or registration.next_q0_request != chain.next_q0_request
                for chain, registration in zip(retired, pending)
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "true reset batch row identity/order differs"
            )
        for name in (
            "sampler_checkpoint_sha256",
            "committed_transcript_sha256",
            "active_owner_root_before_sha256",
            "active_owner_root_after_sha256",
            "unselected_active_owner_root_before_sha256",
            "unselected_active_owner_root_after_sha256",
            "unselected_prepared_owner_root_sha256",
            "parent_true_reset_transcript_sha256",
            "external_closure_batch_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        if (
            not isinstance(self.external_closure_batch, TrueResetClosureBatch)
            or self.external_closure_batch.selected_env_ids != selected
            or self.external_closure_batch.canonical_sha256
            != self.external_closure_batch_sha256
            or tuple(
                row.closure_receipt for row in retired
            )
            != self.external_closure_batch.receipts
        ):
            raise ContinuousRuntimeTransactionError(
                "true reset external closure batch binding differs"
            )
        if (
            self.unselected_active_owner_root_before_sha256
            != self.unselected_active_owner_root_after_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "true reset changed an unselected active owner"
            )
        if _exact_bool(
            self.policy_opportunity_created,
            label="policy_opportunity_created",
        ):
            raise ContinuousRuntimeTransactionError(
                "true reset created a policy opportunity"
            )
        object.__setattr__(self, "selected_env_ids", selected)
        object.__setattr__(self, "retired_chains", retired)
        object.__setattr__(self, "pending_q0", pending)

    @classmethod
    def from_mapping(cls, value: object) -> "TrueResetBatchReceipt":
        values = cls._mapping_values(value)
        values["selected_env_ids"] = tuple(values["selected_env_ids"])
        values["retired_chains"] = tuple(
            RetiredContinuousChain.from_mapping(row)
            for row in values["retired_chains"]
        )
        values["pending_q0"] = tuple(
            PendingTrueResetQ0.from_mapping(row)
            for row in values["pending_q0"]
        )
        values["external_closure_batch"] = TrueResetClosureBatch.from_mapping(
            values["external_closure_batch"]
        )
        return cls(**values)


@dataclass(frozen=True)
class ResetHighWater(_SealedRecord):
    """Per-environment active/archive and absolute sampler progress owner."""

    KIND: ClassVar[str] = "action_ball_continuous_reset_high_water_v1"

    env_id: int
    latest_reset_generation: int
    sampler_draw_high_water: int
    latest_retired_chain_sha256: Optional[str]
    active_sequence_event_kind: Optional[str]
    active_sequence_event_sha256: Optional[str]
    active_committed_reveal_sha256: Optional[str]
    pending_true_reset_q0_sha256: Optional[str]

    def __post_init__(self) -> None:
        for name, minimum in (
            ("env_id", 0),
            ("latest_reset_generation", 1),
            ("sampler_draw_high_water", 1),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name, minimum=minimum),
            )
        for name in (
            "latest_retired_chain_sha256",
            "active_sequence_event_sha256",
            "active_committed_reveal_sha256",
            "pending_true_reset_q0_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(value, label=name))
        sequence_kind = self.active_sequence_event_kind
        if sequence_kind is not None:
            sequence_kind = _text(
                sequence_kind, label="active_sequence_event_kind"
            )
            if sequence_kind not in ("COMMITTED", "INFRA_CENSORED"):
                raise ContinuousRuntimeTransactionError(
                    "reset high-water sequence kind differs"
                )
            object.__setattr__(
                self, "active_sequence_event_kind", sequence_kind
            )
        if (
            (self.active_sequence_event_sha256 is None)
            == (self.pending_true_reset_q0_sha256 is None)
            or (self.active_sequence_event_kind is None)
            != (self.active_sequence_event_sha256 is None)
            or (
                self.pending_true_reset_q0_sha256 is not None
                and self.active_committed_reveal_sha256 is not None
            )
            or (
                self.active_sequence_event_kind == "COMMITTED"
                and self.active_committed_reveal_sha256
                != self.active_sequence_event_sha256
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "reset high-water active sequence/pending binding differs"
            )

    @classmethod
    def from_mapping(cls, value: object) -> "ResetHighWater":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class OwnerTransitionRef(_SealedRecord):
    """Unified chronology across commit and selected true-reset transitions."""

    KIND: ClassVar[str] = "action_ball_continuous_owner_transition_ref_v1"

    transition_kind: str
    transition_sha256: str

    def __post_init__(self) -> None:
        kind = _text(self.transition_kind, label="transition_kind")
        if kind not in (
            "PREPARE_BATCH",
            "ABORT_BATCH",
            "COMMIT_BATCH",
            "CENSOR_BATCH",
            "TRUE_RESET_BATCH",
        ):
            raise ContinuousRuntimeTransactionError(
                "owner transition kind differs"
            )
        object.__setattr__(self, "transition_kind", kind)
        object.__setattr__(
            self,
            "transition_sha256",
            _sha256(self.transition_sha256, label="transition_sha256"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "OwnerTransitionRef":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class _OwnerState:
    sampler_checkpoint_json: str
    prepared: Tuple[PreparedReveal, ...]
    prepared_batches: Tuple[PreparedRevealBatch, ...]
    prepared_event_batches: Tuple[PreparedRevealBatch, ...]
    abort_event_receipts: Tuple[AbortBatchReceipt, ...]
    committed: Tuple[CommittedReveal, ...]
    committed_batches: Tuple[CommittedRevealBatch, ...]
    committed_receipt_shas: Tuple[str, ...]
    committed_batch_shas: Tuple[str, ...]
    censored: Tuple[CensoredReveal, ...]
    censored_batches: Tuple[CensoredRevealBatch, ...]
    retired_chains: Tuple[RetiredContinuousChain, ...]
    pending_true_reset_q0: Tuple[PendingTrueResetQ0, ...]
    true_reset_batches: Tuple[TrueResetBatchReceipt, ...]
    reset_high_waters: Tuple[ResetHighWater, ...]
    owner_transitions: Tuple[OwnerTransitionRef, ...]


@dataclass
class _RevealFinalPreviewAbortPayload:
    owner_id: int
    receipt_sha256: str
    public_preview: RevealFinalPreviewBatch
    owner_state: _OwnerState


@dataclass
class _RevealFinalLease:
    private_token: RevealFinalPreviewBatch
    public_token: RevealFinalPreviewBatch
    private_token_json: str
    preview_root_sha256: str
    committed_result: Optional[CommittedRevealBatch] = None
    committed_state: Optional[_OwnerState] = None
    global_prearm_marker_json: Optional[str] = None
    global_prearm_marker_sha256: Optional[str] = None
    armed_handle: Optional["ArmedRevealFinalHandle"] = None
    armed_payload: Optional["_ArmedPreviewPayload"] = None
    terminal_claim: Optional["PreparedRevealTerminalClaim"] = None
    terminal_claim_payload: Optional[
        "_PreparedRevealTerminalClaimPayload"
    ] = None


@dataclass
class _ArmedPreviewPayload:
    owner_id: int
    preview_root_sha256: str
    global_prearm_marker_sha256: str
    status: str = "armed"


class ArmedRevealFinalHandle:
    """Opaque single-use authority returned only after global token cross-check."""

    __slots__ = ("__weakref__",)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("armed reveal-final handles are immutable")

    @property
    def status(self) -> str:
        payload = _lookup_armed_preview(self)
        return "invalid" if payload is None else payload.status


@dataclass
class _ArmedRevealTerminalPayload:
    owner_id: int
    claim_sha256: str
    status: str = "prepared"


class ArmedRevealTerminalHandle:
    """Preallocated, opaque, single-use terminal publication authority."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        raise TypeError(
            "armed reveal terminal handles are issued only by their owner"
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("armed reveal terminal handles are immutable")

    def __copy__(self):
        raise TypeError("armed reveal terminal handles cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("armed reveal terminal handles cannot be copied")

    def __reduce__(self):
        raise TypeError("armed reveal terminal handles cannot be serialized")

    @property
    def status(self) -> str:
        payload = _lookup_armed_terminal(self)
        return "invalid" if payload is None else payload.status


@dataclass
class _TerminalBoundaryAuthorityPayload:
    owner_id: int
    authority_domain: str
    authority_schema_sha256: str
    authority_source_sha256: str
    authority_sha256: str
    validator: Callable[[object], TerminalBoundaryProjection]
    status: str = "bound"


class TerminalBoundaryAuthority:
    """Opaque identity for the one validator prebound to an R05 owner."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        raise TypeError(
            "terminal boundary authorities are issued only by their owner"
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("terminal boundary authorities are immutable")

    def __copy__(self):
        raise TypeError("terminal boundary authorities cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("terminal boundary authorities cannot be copied")

    def __reduce__(self):
        raise TypeError("terminal boundary authorities cannot be serialized")

    @staticmethod
    def _payload_for(
        authority: "TerminalBoundaryAuthority",
    ) -> _TerminalBoundaryAuthorityPayload:
        payload = _lookup_terminal_boundary_authority(authority)
        if payload is None:
            raise TransactionConflictError(
                "terminal boundary authority is not owner-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def kind(self) -> str:
        return TERMINAL_BOUNDARY_AUTHORITY_KIND

    @property
    def canonical_sha256(self) -> str:
        return self._payload_for(self).authority_sha256

    @property
    def status(self) -> str:
        payload = _lookup_terminal_boundary_authority(self)
        return "invalid" if payload is None else payload.status

    @property
    def authority_domain(self) -> str:
        return self._payload_for(self).authority_domain

    @property
    def authority_schema_sha256(self) -> str:
        return self._payload_for(self).authority_schema_sha256

    @property
    def authority_source_sha256(self) -> str:
        return self._payload_for(self).authority_source_sha256


def _terminal_boundary_authority_sha256(
    *,
    authority_domain: str,
    authority_schema_sha256: str,
    authority_source_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": TERMINAL_BOUNDARY_AUTHORITY_KIND,
            "authority_domain": authority_domain,
            "authority_schema_sha256": authority_schema_sha256,
            "authority_source_sha256": authority_source_sha256,
        }
    )


@dataclass
class _PreparedRevealTerminalClaimPayload:
    owner_id: int
    decision: str
    selected_env_ids: Tuple[int, ...]
    reveal_final_preview_schema_version: int
    reveal_final_preview_sha256: str
    global_boundary_receipt_kind: str
    global_boundary_receipt_sha256: str
    global_boundary_packet_schema_version: int
    global_boundary_packet_sha256: str
    terminal_kind: str
    terminal_sha256: str
    terminal_boundary_authority: TerminalBoundaryAuthority
    terminal_boundary_authority_sha256: str
    terminal_boundary_projection: TerminalBoundaryProjection
    terminal_boundary_projection_sha256: str
    terminal_boundary_marker: RevealTerminalBoundaryMarker
    terminal_content_pin: PreparedTerminalContentPin
    terminal_content_pin_sha256: str
    claim_sha256: str
    terminal_result: Union[CommittedRevealBatch, CensoredRevealBatch]
    terminal_state: _OwnerState
    public_preview: RevealFinalPreviewBatch
    private_preview: RevealFinalPreviewBatch
    boundary_receipt: object
    armed_handle: ArmedRevealTerminalHandle
    armed_payload: _ArmedRevealTerminalPayload
    status: str = "prepared"


class PreparedRevealTerminalClaim:
    """Opaque owner-issued identity for one exact future ACCEPT/CENSOR root.

    The public scalar properties let child owners bind the same terminal
    future without exposing the retained result or next owner state.  There is
    intentionally no constructor, mapping form, copy protocol, or decoder.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls):
        raise TypeError(
            "prepared reveal terminal claims are issued only by their owner"
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("prepared reveal terminal claims are immutable")

    def __copy__(self):
        raise TypeError("prepared reveal terminal claims cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("prepared reveal terminal claims cannot be copied")

    def __reduce__(self):
        raise TypeError("prepared reveal terminal claims cannot be serialized")

    @staticmethod
    def _payload_for(
        claim: "PreparedRevealTerminalClaim",
    ) -> "_PreparedRevealTerminalClaimPayload":
        payload = _lookup_terminal_claim(claim)
        if payload is None:
            raise TransactionConflictError(
                "prepared reveal terminal claim is not owner-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def kind(self) -> str:
        return PREPARED_REVEAL_TERMINAL_CLAIM_KIND

    @property
    def canonical_sha256(self) -> str:
        return self._payload_for(self).claim_sha256

    @property
    def status(self) -> str:
        payload = _lookup_terminal_claim(self)
        return "invalid" if payload is None else payload.status

    @property
    def decision(self) -> str:
        return self._payload_for(self).decision

    @property
    def selected_env_ids(self) -> Tuple[int, ...]:
        return self._payload_for(self).selected_env_ids

    @property
    def reveal_final_preview_schema_version(self) -> int:
        return self._payload_for(self).reveal_final_preview_schema_version

    @property
    def reveal_final_preview_sha256(self) -> str:
        return self._payload_for(self).reveal_final_preview_sha256

    @property
    def global_boundary_receipt_kind(self) -> str:
        return self._payload_for(self).global_boundary_receipt_kind

    @property
    def global_boundary_receipt_sha256(self) -> str:
        return self._payload_for(self).global_boundary_receipt_sha256

    @property
    def global_boundary_packet_schema_version(self) -> int:
        return self._payload_for(self).global_boundary_packet_schema_version

    @property
    def global_boundary_packet_sha256(self) -> str:
        return self._payload_for(self).global_boundary_packet_sha256

    @property
    def terminal_boundary_authority_sha256(self) -> str:
        return self._payload_for(self).terminal_boundary_authority_sha256

    @property
    def terminal_boundary_projection(self) -> TerminalBoundaryProjection:
        return self._payload_for(self).terminal_boundary_projection

    @property
    def terminal_boundary_projection_sha256(self) -> str:
        return self._payload_for(self).terminal_boundary_projection_sha256

    @property
    def terminal_content_pin(self) -> PreparedTerminalContentPin:
        return self._payload_for(self).terminal_content_pin

    @property
    def terminal_content_pin_sha256(self) -> str:
        return self._payload_for(self).terminal_content_pin_sha256

    @property
    def terminal_kind(self) -> str:
        return self._payload_for(self).terminal_kind

    @property
    def terminal_sha256(self) -> str:
        return self._payload_for(self).terminal_sha256


def _terminal_claim_sha256(
    *,
    decision: str,
    selected_env_ids: Tuple[int, ...],
    reveal_final_preview_schema_version: int,
    reveal_final_preview_sha256: str,
    global_boundary_receipt_kind: str,
    global_boundary_receipt_sha256: str,
    global_boundary_packet_schema_version: int,
    global_boundary_packet_sha256: str,
    terminal_boundary_authority_sha256: str,
    terminal_boundary_projection_sha256: str,
    terminal_content_pin_sha256: str,
    terminal_kind: str,
    terminal_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": PREPARED_REVEAL_TERMINAL_CLAIM_KIND,
            "decision": decision,
            "selected_env_ids": list(selected_env_ids),
            "reveal_final_preview_schema_version": (
                reveal_final_preview_schema_version
            ),
            "reveal_final_preview_sha256": reveal_final_preview_sha256,
            "global_boundary_receipt_kind": global_boundary_receipt_kind,
            "global_boundary_receipt_sha256": (
                global_boundary_receipt_sha256
            ),
            "global_boundary_packet_schema_version": (
                global_boundary_packet_schema_version
            ),
            "global_boundary_packet_sha256": global_boundary_packet_sha256,
            "terminal_boundary_authority_sha256": (
                terminal_boundary_authority_sha256
            ),
            "terminal_boundary_projection_sha256": (
                terminal_boundary_projection_sha256
            ),
            "terminal_content_pin_sha256": terminal_content_pin_sha256,
            "terminal_kind": terminal_kind,
            "terminal_sha256": terminal_sha256,
        }
    )

def _make_armed_preview_registry():
    payloads = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def attach(
        handle: ArmedRevealFinalHandle, payload: _ArmedPreviewPayload
    ) -> None:
        with lock:
            if handle in payloads:
                raise ContinuousRuntimeTransactionError(
                    "armed reveal-final handle is already attached"
                )
            payloads[handle] = payload

    def lookup(
        handle: ArmedRevealFinalHandle,
    ) -> Optional[_ArmedPreviewPayload]:
        with lock:
            return payloads.get(handle)

    return attach, lookup


_attach_armed_preview, _lookup_armed_preview = _make_armed_preview_registry()
del _make_armed_preview_registry


def _make_reveal_final_preview_abort_registry():
    receipts = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def attach(
        receipt: RevealFinalPreviewAbortReceipt,
        payload: _RevealFinalPreviewAbortPayload,
    ) -> None:
        with lock:
            if receipt in receipts:
                raise ContinuousRuntimeTransactionError(
                    "reveal-final preview abort receipt is already attached"
                )
            receipts[receipt] = payload

    def lookup(
        receipt: RevealFinalPreviewAbortReceipt,
    ) -> Optional[_RevealFinalPreviewAbortPayload]:
        with lock:
            return receipts.get(receipt)

    return attach, lookup


(
    _attach_reveal_final_preview_abort,
    _lookup_reveal_final_preview_abort,
) = _make_reveal_final_preview_abort_registry()
del _make_reveal_final_preview_abort_registry


def _make_terminal_boundary_authority_registry():
    authorities = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _TerminalBoundaryAuthorityPayload,
    ) -> TerminalBoundaryAuthority:
        authority = object.__new__(TerminalBoundaryAuthority)
        with lock:
            authorities[authority] = payload
        return authority

    def lookup(
        authority: TerminalBoundaryAuthority,
    ) -> Optional[_TerminalBoundaryAuthorityPayload]:
        with lock:
            return authorities.get(authority)

    return mint, lookup


(
    _mint_terminal_boundary_authority,
    _lookup_terminal_boundary_authority,
) = _make_terminal_boundary_authority_registry()
del _make_terminal_boundary_authority_registry


def _make_terminal_claim_registries():
    claims = weakref.WeakKeyDictionary()
    handles = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint_handle(
        payload: _ArmedRevealTerminalPayload,
    ) -> ArmedRevealTerminalHandle:
        handle = object.__new__(ArmedRevealTerminalHandle)
        with lock:
            handles[handle] = payload
        return handle

    def lookup_handle(
        handle: ArmedRevealTerminalHandle,
    ) -> Optional[_ArmedRevealTerminalPayload]:
        with lock:
            return handles.get(handle)

    def mint_claim(
        payload: _PreparedRevealTerminalClaimPayload,
    ) -> PreparedRevealTerminalClaim:
        claim = object.__new__(PreparedRevealTerminalClaim)
        with lock:
            claims[claim] = payload
        return claim

    def lookup_claim(
        claim: PreparedRevealTerminalClaim,
    ) -> Optional[_PreparedRevealTerminalClaimPayload]:
        with lock:
            return claims.get(claim)

    return mint_handle, lookup_handle, mint_claim, lookup_claim


(
    _mint_armed_terminal,
    _lookup_armed_terminal,
    _mint_terminal_claim,
    _lookup_terminal_claim,
) = _make_terminal_claim_registries()
del _make_terminal_claim_registries


def _owner_transition_root(rows: Sequence[OwnerTransitionRef]) -> str:
    root = canonical_sha256(
        {
            "kind": "action_ball_continuous_owner_transition_root_v1",
            "transitions": [],
        }
    )
    for row in rows:
        root = canonical_sha256(
            {
                "kind": "action_ball_continuous_owner_transition_step_v1",
                "parent_sha256": root,
                "transition_kind": row.transition_kind,
                "transition_sha256": row.transition_sha256,
            }
        )
    return root


def _retired_row_shas(state: _OwnerState) -> frozenset[str]:
    return frozenset(
        receipt_sha
        for chain in state.retired_chains
        for receipt_sha in chain.committed_reveal_sha256s
    )


def _active_committed_rows(state: _OwnerState, env_id: int) -> Tuple[CommittedReveal, ...]:
    retired = _retired_row_shas(state)
    return tuple(
        row
        for row in state.committed
        if row.prepared_reveal.request.env_id == env_id
        and row.canonical_sha256 not in retired
    )


def _active_censored_rows(state: _OwnerState, env_id: int) -> Tuple[CensoredReveal, ...]:
    retired = frozenset(
        event_sha
        for chain in state.retired_chains
        for event_sha in getattr(chain, "sequence_event_sha256s", ())
    )
    return tuple(
        row
        for row in state.censored
        if row.prepared_reveal.request.env_id == env_id
        and row.canonical_sha256 not in retired
    )


def _active_sequence_rows(state: _OwnerState, env_id: int) -> Tuple[object, ...]:
    return tuple(
        sorted(
            (*_active_committed_rows(state, env_id), *_active_censored_rows(state, env_id)),
            key=lambda row: row.prepared_reveal.request.scheduled_ordinal,
        )
    )


def _active_owner_root(
    state: _OwnerState,
    *,
    excluded_env_ids: Sequence[int] = (),
) -> str:
    """Bind every per-environment owner subtree, excluding global chronology.

    The name is retained for the sealed receipt ABI, but this is deliberately
    wider than the current active head: sampler progress, every committed row
    and batch membership, retired archives, pending Q0, and reset high-water
    are all included.  Private prepared rows have their own opaque root because
    prepare/abort are intentionally absent from the public owner chronology.
    The unified owner-event transcript is global and is bound separately.
    """

    excluded = frozenset(excluded_env_ids)
    sampler_checkpoint = _checkpoint_from_json(state.sampler_checkpoint_json)
    env_ids = sorted(
        {
            row.prepared_reveal.request.env_id for row in state.committed
        }
        | {row.prepared_reveal.request.env_id for row in state.censored}
        | {row.env_id for row in state.retired_chains}
        | {row.env_id for row in state.pending_true_reset_q0}
        | {row.env_id for row in state.reset_high_waters}
        | {row["env_id"] for row in sampler_checkpoint["environments"]}
    )
    payload_rows = []
    for env_id in env_ids:
        if env_id in excluded:
            continue
        committed = tuple(
            row
            for row in state.committed
            if row.prepared_reveal.request.env_id == env_id
        )
        censored = tuple(
            row
            for row in state.censored
            if row.prepared_reveal.request.env_id == env_id
        )
        active = _active_committed_rows(state, env_id)
        retired = tuple(
            row for row in state.retired_chains if row.env_id == env_id
        )
        pending = next(
            (row for row in state.pending_true_reset_q0 if row.env_id == env_id),
            None,
        )
        high_water = next(
            (row for row in state.reset_high_waters if row.env_id == env_id),
            None,
        )
        payload_rows.append(
            {
                "env_id": env_id,
                "sampler_env_state": _sampler_env_row(
                    sampler_checkpoint, env_id
                ),
                "committed_reveals": [row.to_mapping() for row in committed],
                "committed_batch_sha256s": [
                    batch.canonical_sha256
                    for batch in state.committed_batches
                    if any(
                        row.prepared_reveal.request.env_id == env_id
                        for row in batch.committed_reveals
                    )
                ],
                "censored_reveals": [row.to_mapping() for row in censored],
                "censored_batch_sha256s": [
                    batch.canonical_sha256
                    for batch in state.censored_batches
                    if any(
                        row.prepared_reveal.request.env_id == env_id
                        for row in batch.censored_reveals
                    )
                ],
                "active_committed_reveal_sha256s": [
                    row.canonical_sha256 for row in active
                ],
                "retired_chains": [row.to_mapping() for row in retired],
                "pending_true_reset_q0": (
                    None if pending is None else pending.to_mapping()
                ),
                "reset_high_water": (
                    None if high_water is None else high_water.to_mapping()
                ),
            }
        )
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_per_env_owner_root_v2",
            "rows": payload_rows,
        }
    )


def _prepared_owner_root(
    state: _OwnerState,
    *,
    excluded_env_ids: Sequence[int] = (),
) -> str:
    """Opaque byte root for private prepared rows and batch membership."""

    excluded = frozenset(excluded_env_ids)
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_prepared_owner_root_v1",
            "rows": [
                {
                    "env_id": env_id,
                    "prepared_reveals": [
                        row.to_mapping()
                        for row in state.prepared
                        if row.request.env_id == env_id
                    ],
                    "prepared_batch_sha256s": [
                        batch.canonical_sha256
                        for batch in state.prepared_batches
                        if any(
                            row.request.env_id == env_id
                            for row in batch.prepared_reveals
                        )
                    ],
                }
                for env_id in sorted(
                    {row.request.env_id for row in state.prepared}
                )
                if env_id not in excluded
            ],
        }
    )


def _checkpoint_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _checkpoint_from_json(value: str) -> dict[str, object]:
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ContinuousRuntimeTransactionError("sampler checkpoint is not a mapping")
    return result


def _sampler_env_row(
    checkpoint: Mapping[str, object], env_id: int
) -> Optional[dict[str, object]]:
    rows = checkpoint.get("environments")
    if not isinstance(rows, list):
        raise ContinuousRuntimeTransactionError(
            "sampler checkpoint environments are malformed"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise ContinuousRuntimeTransactionError(
                "sampler checkpoint environment row is malformed"
            )
        if row.get("env_id") == env_id:
            return dict(row)
    return None


def _sampler_env_rows_by_id(
    checkpoint: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    """Index a validated checkpoint once for batched owner operations."""

    rows = checkpoint.get("environments")
    if not isinstance(rows, list):
        raise ContinuousRuntimeTransactionError(
            "sampler checkpoint environments are malformed"
        )
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ContinuousRuntimeTransactionError(
                "sampler checkpoint environment row is malformed"
            )
        env_id = row.get("env_id")
        if type(env_id) is not int or env_id < 0:
            raise ContinuousRuntimeTransactionError(
                "sampler checkpoint environment id is malformed"
            )
        if env_id in result:
            raise ContinuousRuntimeTransactionError(
                "sampler checkpoint environment id is duplicated"
            )
        result[env_id] = dict(row)
    return result


def _sampler_env_state_sha256(
    env_id: int,
    row: Optional[Mapping[str, object]],
    *,
    phase: str,
) -> str:
    if phase not in ("before", "after"):
        raise ContinuousRuntimeTransactionError(
            "sampler env-state digest phase differs"
        )
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_sampler_env_state_v1",
            "env_id": _plain_int(env_id, label="env_id"),
            "phase": phase,
            "state": None if row is None else dict(row),
        }
    )


def _single_env_sampler_checkpoint(
    profile: _target_sampler.ContinuousTargetProfile,
    *,
    seed: int,
    before: Optional[Mapping[str, object]],
) -> dict[str, object]:
    payload = {
        "schema_version": _target_sampler.SCHEMA_VERSION,
        "kind": _target_sampler.CHECKPOINT_KIND,
        "profile_sha256": profile.profile_sha256,
        "runtime_dtype": profile.runtime_dtype,
        "quantization_contract": profile.quantization_contract,
        "seed": seed,
        "environments": [] if before is None else [dict(before)],
    }
    return {
        **payload,
        "checkpoint_sha256": _target_sampler.canonical_sha256(payload),
    }


def _merge_sampler_env_row(
    checkpoint: Mapping[str, object],
    *,
    env_id: int,
    expected_before: Optional[Mapping[str, object]],
    staged_after: Mapping[str, object],
) -> dict[str, object]:
    current = _sampler_env_row(checkpoint, env_id)
    expected = None if expected_before is None else dict(expected_before)
    if current != expected:
        raise TransactionConflictError(
            "live sampler env row differs from prepared before-state"
        )
    rows = [
        dict(row)
        for row in checkpoint["environments"]
        if row["env_id"] != env_id
    ]
    after = dict(staged_after)
    if after.get("env_id") != env_id:
        raise ContinuousRuntimeTransactionError(
            "staged sampler row belongs to another environment"
        )
    rows.append(after)
    rows.sort(key=lambda row: row["env_id"])
    payload = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_sha256"
    }
    payload["environments"] = rows
    result = dict(payload)
    result["checkpoint_sha256"] = _target_sampler.canonical_sha256(payload)
    return result


def _merge_sampler_env_rows(
    checkpoint: Mapping[str, object],
    prepared_rows: Sequence[PreparedReveal],
) -> dict[str, object]:
    """Stage every selected row in one checkpoint payload and seal once."""

    rows = tuple(prepared_rows)
    env_ids = _ordered_unique_env_ids(
        tuple(row.request.env_id for row in rows), label="selected_env_ids"
    )
    selected = frozenset(env_ids)
    current_by_env = {
        row["env_id"]: dict(row) for row in checkpoint["environments"]
    }
    for prepared in rows:
        env_id = prepared.request.env_id
        if current_by_env.get(env_id) != prepared.sampler_before_env_state:
            raise TransactionConflictError(
                "live sampler env row differs from prepared before-state"
            )
    merged_rows = [
        dict(row)
        for row in checkpoint["environments"]
        if row["env_id"] not in selected
    ]
    for prepared in rows:
        after = dict(prepared.sampler_after_env_state)
        if after.get("env_id") != prepared.request.env_id:
            raise ContinuousRuntimeTransactionError(
                "staged sampler row belongs to another environment"
            )
        merged_rows.append(after)
    merged_rows.sort(key=lambda row: row["env_id"])
    payload = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_sha256"
    }
    payload["environments"] = merged_rows
    result = dict(payload)
    result["checkpoint_sha256"] = _target_sampler.canonical_sha256(payload)
    return result


def _ball_snapshot_sha256(slots: Sequence[BallSlotSnapshot]) -> str:
    return canonical_sha256([slot.to_mapping() for slot in slots])


def _transcript_root(receipt_shas: Sequence[str]) -> str:
    root = canonical_sha256(
        {
            "kind": "action_ball_continuous_commit_transcript_root_v1",
            "receipts": [],
        }
    )
    for receipt_sha in receipt_shas:
        root = canonical_sha256(
            {
                "kind": "action_ball_continuous_commit_transcript_step_v1",
                "parent_sha256": root,
                "committed_reveal_sha256": receipt_sha,
            }
        )
    return root


def _batch_transcript_root(batch_shas: Sequence[str]) -> str:
    root = canonical_sha256(
        {
            "kind": "action_ball_continuous_commit_batch_transcript_root_v1",
            "batches": [],
        }
    )
    for batch_sha in batch_shas:
        root = canonical_sha256(
            {
                "kind": "action_ball_continuous_commit_batch_transcript_step_v1",
                "parent_sha256": root,
                "committed_reveal_batch_sha256": _sha256(
                    batch_sha, label="committed_reveal_batch_sha256"
                ),
            }
        )
    return root


def _clone_prepared(value: PreparedReveal) -> PreparedReveal:
    """Return a sealed deep copy so mutable checkpoint rows never escape."""

    return PreparedReveal.from_mapping(value.to_mapping())


def _clone_committed(value: CommittedReveal) -> CommittedReveal:
    """Return a sealed deep copy so nested prepared rows never escape."""

    return CommittedReveal.from_mapping(value.to_mapping())


def _clone_prepared_batch(value: PreparedRevealBatch) -> PreparedRevealBatch:
    return PreparedRevealBatch.from_mapping(value.to_mapping())


def _clone_committed_batch(value: CommittedRevealBatch) -> CommittedRevealBatch:
    return CommittedRevealBatch.from_mapping(value.to_mapping())


class ContinuousRuntimeTransactionOwner:
    """Per-process pure owner for private prepare and atomic logical commit."""

    def __init__(
        self,
        profile: _target_sampler.ContinuousTargetProfile,
        *,
        seed: int,
        ball_slot_capacity: int,
        sampler_checkpoint: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not isinstance(profile, _target_sampler.ContinuousTargetProfile):
            raise ContinuousRuntimeTransactionError(
                "profile must be ContinuousTargetProfile"
            )
        if len(profile.components) != 2:
            raise ContinuousRuntimeTransactionError(
                "runtime transaction requires an XY target profile"
            )
        if (
            type(profile.runtime_dtype) is not str
            or profile.runtime_dtype != _target_sampler.RUNTIME_DTYPE
            or type(profile.quantization_contract) is not str
            or profile.quantization_contract
            != _target_sampler.QUANTIZATION_CONTRACT
        ):
            raise ContinuousRuntimeTransactionError(
                "target profile runtime dtype/quantization contract differs"
            )
        self.profile = profile
        self.seed = _plain_int(seed, label="seed", maximum=(1 << 64) - 1)
        self.ball_slot_capacity = _plain_int(
            ball_slot_capacity, label="ball_slot_capacity", minimum=1
        )
        runtime_targets: dict[Tuple[float, float], str] = {}
        for cell in profile.cells:
            runtime_target = _runtime_xy(
                cell.target, label=f"profile.cells[{cell.cell_id}].target"
            )
            previous = runtime_targets.get(runtime_target)
            if previous is not None:
                raise Float32TargetAliasError(
                    "different target cells alias after float32 conversion: "
                    f"{previous} and {cell.cell_id}"
                )
            runtime_targets[runtime_target] = cell.cell_id

        if sampler_checkpoint is None:
            target_sampler = _target_sampler.ContinuousTargetSampler(
                profile, seed=self.seed
            )
            checkpoint = target_sampler.checkpoint()
        else:
            checkpoint = dict(sampler_checkpoint)
            target_sampler = _target_sampler.ContinuousTargetSampler.from_checkpoint(
                profile, checkpoint
            )
            if target_sampler.seed != self.seed:
                raise ContinuousRuntimeTransactionError(
                    "sampler checkpoint seed differs from owner seed"
                )
            checkpoint = target_sampler.checkpoint()
        self._state = _OwnerState(
            sampler_checkpoint_json=_checkpoint_json(checkpoint),
            prepared=(),
            prepared_batches=(),
            prepared_event_batches=(),
            abort_event_receipts=(),
            committed=(),
            committed_batches=(),
            committed_receipt_shas=(),
            committed_batch_shas=(),
            censored=(),
            censored_batches=(),
            retired_chains=(),
            pending_true_reset_q0=(),
            true_reset_batches=(),
            reset_high_waters=(),
            owner_transitions=(),
        )
        self._operation_active = False
        self._active_preview: Optional[_RevealFinalLease] = None
        self._terminal_boundary_bind_open = True
        self._terminal_boundary_authority: Optional[
            TerminalBoundaryAuthority
        ] = None
        self._terminal_boundary_authority_payload: Optional[
            _TerminalBoundaryAuthorityPayload
        ] = None
        self._poisoned = False
        self._poison_reason: Optional[str] = None

    @property
    def integration_status(self) -> str:
        return INTEGRATION_STATUS

    @property
    def runtime_wiring_connected(self) -> bool:
        return RUNTIME_WIRING_CONNECTED

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def poison_reason(self) -> Optional[str]:
        return self._poison_reason

    def poison_global_reveal_epoch(self, reason: str) -> None:
        """Sticky, idempotent, non-raising stop used by broadcast failure paths."""

        if self._poisoned:
            return
        clean_reason = (
            reason
            if type(reason) is str and bool(reason)
            else "invalid_or_empty_global_reveal_poison_reason"
        )
        self._poisoned = True
        self._poison_reason = clean_reason
        lease = self._active_preview
        if lease is not None and lease.terminal_claim_payload is not None:
            lease.terminal_claim_payload.status = "poisoned"
            lease.terminal_claim_payload.armed_payload.status = "poisoned"
        if self._terminal_boundary_authority_payload is not None:
            self._terminal_boundary_authority_payload.status = "poisoned"

    def bind_terminal_boundary_authority(
        self,
        validator: Callable[[object], TerminalBoundaryProjection],
        *,
        authority_domain: str,
        authority_schema_sha256: str,
        authority_source_sha256: str,
    ) -> TerminalBoundaryAuthority:
        """One-time construction bind for an exact downstream receipt owner.

        The retained callable closes over the concrete downstream owner and
        must perform its exact receipt-registry check.  Later staging receives
        only an opaque receipt, so a caller cannot pair an arbitrary owner with
        an unrelated receipt.
        """

        self._begin_operation(allow_terminal_boundary_bind=True)
        try:
            if (
                not self._terminal_boundary_bind_open
                or self._terminal_boundary_authority is not None
                or self._terminal_boundary_authority_payload is not None
            ):
                raise TransactionConflictError(
                    "terminal boundary authority is already bound"
                )
            if not callable(validator):
                raise ContinuousRuntimeTransactionError(
                    "terminal boundary authority validator must be callable"
                )
            domain = _text(authority_domain, label="authority_domain")
            schema_sha = _sha256(
                authority_schema_sha256,
                label="authority_schema_sha256",
            )
            source_sha = _sha256(
                authority_source_sha256,
                label="authority_source_sha256",
            )
            authority_sha = _terminal_boundary_authority_sha256(
                authority_domain=domain,
                authority_schema_sha256=schema_sha,
                authority_source_sha256=source_sha,
            )
            payload = _TerminalBoundaryAuthorityPayload(
                owner_id=id(self),
                authority_domain=domain,
                authority_schema_sha256=schema_sha,
                authority_source_sha256=source_sha,
                authority_sha256=authority_sha,
                validator=validator,
            )
            authority = _mint_terminal_boundary_authority(payload)
            self._terminal_boundary_authority = authority
            self._terminal_boundary_authority_payload = payload
            self._terminal_boundary_bind_open = False
            return authority
        finally:
            self._finish_operation()

    @property
    def terminal_boundary_authority(
        self,
    ) -> Optional[TerminalBoundaryAuthority]:
        """Return the stable exact authority identity, if construction bound."""

        return self._terminal_boundary_authority

    def require_owned_terminal_boundary_authority(
        self,
        authority: TerminalBoundaryAuthority,
        *,
        expected_authority_sha256: str,
        expected_authority_domain: str,
        expected_authority_schema_sha256: str,
        expected_authority_source_sha256: str,
    ) -> TerminalBoundaryAuthority:
        """Read-only exact-identity validator for construction wiring."""

        self._begin_operation(allow_active_preview=True)
        try:
            payload = (
                None
                if type(authority) is not TerminalBoundaryAuthority
                else _lookup_terminal_boundary_authority(authority)
            )
            if (
                authority is not self._terminal_boundary_authority
                or payload is None
                or payload is not self._terminal_boundary_authority_payload
                or payload.owner_id != id(self)
                or payload.status != "bound"
                or payload.authority_sha256
                != _sha256(
                    expected_authority_sha256,
                    label="expected_authority_sha256",
                )
                or payload.authority_domain
                != _text(
                    expected_authority_domain,
                    label="expected_authority_domain",
                )
                or payload.authority_schema_sha256
                != _sha256(
                    expected_authority_schema_sha256,
                    label="expected_authority_schema_sha256",
                )
                or payload.authority_source_sha256
                != _sha256(
                    expected_authority_source_sha256,
                    label="expected_authority_source_sha256",
                )
            ):
                raise TransactionConflictError(
                    "terminal boundary authority identity/domain/pins differ"
                )
            return authority
        finally:
            self._finish_operation()

    def _require_no_active_preview(self) -> None:
        if self._poisoned:
            raise TransactionConflictError(
                "transaction owner is poisoned by a global reveal failure"
            )
        if self._active_preview is not None:
            raise TransactionConflictError(
                "owner read forbidden while reveal-final preview lease is active"
            )
        self._terminal_boundary_bind_open = False

    def sampler_checkpoint(self) -> dict[str, object]:
        self._require_no_active_preview()
        return _checkpoint_from_json(self._state.sampler_checkpoint_json)

    def prepared_for_env(self, env_id: int) -> Optional[PreparedReveal]:
        self._require_no_active_preview()
        clean = _plain_int(env_id, label="env_id")
        row = next(
            (row for row in self._state.prepared if row.request.env_id == clean),
            None,
        )
        return None if row is None else _clone_prepared(row)

    def prepared_batch(self, batch_sha256: str) -> Optional[PreparedRevealBatch]:
        self._require_no_active_preview()
        digest = _sha256(batch_sha256, label="prepared_batch_sha256")
        row = next(
            (
                row
                for row in self._state.prepared_batches
                if row.canonical_sha256 == digest
            ),
            None,
        )
        return None if row is None else _clone_prepared_batch(row)

    def committed_for_env(self, env_id: int) -> Optional[CommittedReveal]:
        self._require_no_active_preview()
        clean = _plain_int(env_id, label="env_id")
        sequence = _active_sequence_rows(self._state, clean)
        row = sequence[-1] if sequence and isinstance(
            sequence[-1], CommittedReveal
        ) else None
        return None if row is None else _clone_committed(row)

    def reset_high_water_for_env(self, env_id: int) -> Optional[ResetHighWater]:
        self._require_no_active_preview()
        clean = _plain_int(env_id, label="env_id")
        row = next(
            (row for row in self._state.reset_high_waters if row.env_id == clean),
            None,
        )
        return (
            None
            if row is None
            else ResetHighWater.from_mapping(row.to_mapping())
        )

    def true_reset_many(
        self,
        next_q0_requests: Sequence[ContinuousPrepareRequest],
        closure_batch: TrueResetClosureBatch,
        *,
        expected_owner_checkpoint_sha256: object,
        expected_closure_batch_sha256: object,
        fault_injector: _FaultInjector = None,
    ) -> TrueResetBatchReceipt:
        """Retire selected active chains and register their only admissible Q0.

        Flight, mailbox, recovery, payment, and hard-terminal closure are
        external authorities.  This method accepts them only as one exact
        content-addressed batch whose digest and pre-owner checkpoint root are
        independently supplied by the caller.
        """

        self._begin_operation()
        try:
            requests = tuple(next_q0_requests)
            if not requests or any(
                type(row) is not ContinuousPrepareRequest for row in requests
            ):
                raise ContinuousRuntimeTransactionError(
                    "true_reset_many requests must be non-empty exact Q0 requests"
                )
            selected = _ordered_unique_env_ids(
                tuple(row.env_id for row in requests),
                label="selected_env_ids",
            )
            if type(closure_batch) is not TrueResetClosureBatch:
                raise ContinuousRuntimeTransactionError(
                    "true reset requires an exact external closure batch"
                )
            expected_owner = _sha256(
                expected_owner_checkpoint_sha256,
                label="expected_owner_checkpoint_sha256",
            )
            expected_closure = _sha256(
                expected_closure_batch_sha256,
                label="expected_closure_batch_sha256",
            )
            before_checkpoint = self.checkpoint()
            if (
                before_checkpoint["canonical_sha256"] != expected_owner
                or closure_batch.owner_checkpoint_before_sha256 != expected_owner
            ):
                raise TransactionConflictError(
                    "true reset owner checkpoint external root differs"
                )
            if closure_batch.canonical_sha256 != expected_closure:
                raise TransactionConflictError(
                    "true reset closure batch external root differs"
                )
            if closure_batch.selected_env_ids != selected:
                raise TransactionConflictError(
                    "true reset closure selected worlds differ"
                )
            selected_set = frozenset(selected)
            if any(
                row.request.env_id in selected_set for row in self._state.prepared
            ):
                raise TransactionConflictError(
                    "selected true reset world has a prepared reveal"
                )
            if any(
                row.env_id in selected_set
                for row in self._state.pending_true_reset_q0
            ):
                raise TransactionConflictError(
                    "selected world already has a pending true-reset Q0"
                )
            live_sampler = self.sampler_checkpoint()
            active_before = _active_owner_root(self._state)
            unselected_before = _active_owner_root(
                self._state, excluded_env_ids=selected
            )
            closures = {
                row.env_id: row for row in closure_batch.receipts
            }
            high_waters = {
                row.env_id: row for row in self._state.reset_high_waters
            }
            new_chains = []
            new_pending = []
            for request in requests:
                env_id = request.env_id
                active_rows = _active_sequence_rows(self._state, env_id)
                if not active_rows:
                    raise TransactionConflictError(
                        "true reset selected world lacks an active sequence chain"
                    )
                latest = active_rows[-1]
                first_request = active_rows[0].prepared_reveal.request
                latest_request = latest.prepared_reveal.request
                ordinals = tuple(
                    row.prepared_reveal.request.scheduled_ordinal
                    for row in active_rows
                )
                if ordinals != tuple(range(len(active_rows))):
                    raise ContinuousRuntimeTransactionError(
                        "active chain ordinals are not contiguous"
                    )
                for row in active_rows:
                    row_request = row.prepared_reveal.request
                    if (
                        row_request.reset_generation
                        != first_request.reset_generation
                        or row_request.birth_sha256 != first_request.birth_sha256
                        or row_request.run_id != first_request.run_id
                        or row_request.carry_chain_id
                        != first_request.carry_chain_id
                    ):
                        raise ContinuousRuntimeTransactionError(
                            "active chain identity is not stable"
                        )
                closure = closures[env_id]
                active_committed_rows = tuple(
                    row for row in active_rows if isinstance(row, CommittedReveal)
                )
                latest_committed = (
                    None if not active_committed_rows else active_committed_rows[-1]
                )
                if (
                    closure.prior_reset_generation
                    != latest_request.reset_generation
                    or closure.latest_sequence_event_kind
                    != (
                        "COMMITTED"
                        if isinstance(latest, CommittedReveal)
                        else "INFRA_CENSORED"
                    )
                    or closure.latest_sequence_event_sha256
                    != latest.canonical_sha256
                    or closure.latest_committed_reveal_sha256
                    != (
                        None
                        if latest_committed is None
                        else latest_committed.canonical_sha256
                    )
                    or closure.latest_outcome_key_sha256
                    != (
                        None
                        if latest_committed is None
                        else latest_committed.prepared_reveal.outcome_key.canonical_sha256
                    )
                    or (
                        closure.closure_disposition == "CLOSED_AFTER_DEADLINE"
                        and closure.closed_at_step
                        < latest_request.scheduled_deadline_step
                    )
                    or (
                        closure.closure_disposition == "CENSORED_TRUE_RESET"
                        and closure.closed_at_step
                        < latest_request.scheduled_reveal_step
                    )
                ):
                    raise TransactionConflictError(
                        "external closure differs from the active chain head"
                    )
                live_row = _sampler_env_row(live_sampler, env_id)
                if (
                    live_row is None
                    or live_row != latest.prepared_reveal.sampler_after_env_state
                ):
                    raise TransactionConflictError(
                        "true reset sampler high-water differs from active head"
                    )
                prior_high_water = high_waters.get(env_id)
                if (
                    prior_high_water is None
                    or prior_high_water.active_sequence_event_sha256
                    != latest.canonical_sha256
                    or prior_high_water.active_sequence_event_kind
                    != closure.latest_sequence_event_kind
                    or prior_high_water.latest_reset_generation
                    != latest_request.reset_generation
                    or prior_high_water.sampler_draw_high_water
                    != live_row["draw_count"]
                ):
                    raise TransactionConflictError(
                        "true reset active reset high-water differs"
                    )
                if (
                    request.scheduled_ordinal != 0
                    or request.runtime_swing_generation != 0
                    or request.outcome_shot_index != 1
                    or request.reset_generation
                    != latest_request.reset_generation + 1
                    or request.sampler_generation != live_row["draw_count"] + 1
                    or request.previous_ball_slot_index is not None
                    or any(slot.lifecycle_state != BALL_EMPTY for slot in request.ball_slots)
                    or request.birth_sha256 == latest_request.birth_sha256
                    or request.carry_chain_id == latest_request.carry_chain_id
                    or request.task_birth_snapshot_id
                    == latest_request.task_birth_snapshot_id
                ):
                    raise TransactionConflictError(
                        "true reset next Q0 identity/generation differs"
                    )
                sequence_kinds = tuple(
                    "COMMITTED"
                    if isinstance(row, CommittedReveal)
                    else "INFRA_CENSORED"
                    for row in active_rows
                )
                sequence_shas = tuple(
                    row.canonical_sha256 for row in active_rows
                )
                committed_shas = tuple(
                    row.canonical_sha256 for row in active_committed_rows
                )
                chain = RetiredContinuousChain(
                    env_id=env_id,
                    reset_generation=latest_request.reset_generation,
                    birth_sha256=latest_request.birth_sha256,
                    run_id=latest_request.run_id,
                    carry_chain_id=latest_request.carry_chain_id,
                    first_scheduled_ordinal=0,
                    last_scheduled_ordinal=latest_request.scheduled_ordinal,
                    first_sampler_generation=(
                        active_rows[0].prepared_reveal.request.sampler_generation
                    ),
                    last_sampler_generation=latest_request.sampler_generation,
                    sequence_event_kinds=sequence_kinds,
                    sequence_event_sha256s=sequence_shas,
                    sequence_chain_transcript_sha256=(
                        _sequence_chain_transcript_root(
                            sequence_kinds, sequence_shas
                        )
                    ),
                    committed_reveal_sha256s=committed_shas,
                    committed_chain_transcript_sha256=(
                        _chain_transcript_root(committed_shas)
                    ),
                    sampler_high_water_env_state_sha256=(
                        latest.prepared_reveal.sampler_after_env_state_sha256
                    ),
                    latest_outcome_key_sha256=(
                        None
                        if latest_committed is None
                        else latest_committed.prepared_reveal.outcome_key.canonical_sha256
                    ),
                    closure_receipt=closure,
                    next_q0_request=request,
                )
                pending = PendingTrueResetQ0(
                    env_id=env_id,
                    retired_chain_sha256=chain.canonical_sha256,
                    sampler_before_env_state_sha256=_sampler_env_state_sha256(
                        env_id, live_row, phase="before"
                    ),
                    next_q0_request=request,
                )
                new_chains.append(chain)
                new_pending.append(pending)
                high_waters[env_id] = ResetHighWater(
                    env_id=env_id,
                    latest_reset_generation=request.reset_generation,
                    sampler_draw_high_water=live_row["draw_count"],
                    latest_retired_chain_sha256=chain.canonical_sha256,
                    active_sequence_event_kind=None,
                    active_sequence_event_sha256=None,
                    active_committed_reveal_sha256=None,
                    pending_true_reset_q0_sha256=pending.canonical_sha256,
                )
                self._inject(fault_injector, "true_reset_many_after_row")
            provisional = _OwnerState(
                sampler_checkpoint_json=self._state.sampler_checkpoint_json,
                prepared=self._state.prepared,
                prepared_batches=self._state.prepared_batches,
                prepared_event_batches=self._state.prepared_event_batches,
                abort_event_receipts=self._state.abort_event_receipts,
                committed=self._state.committed,
                committed_batches=self._state.committed_batches,
                committed_receipt_shas=self._state.committed_receipt_shas,
                committed_batch_shas=self._state.committed_batch_shas,
                censored=self._state.censored,
                censored_batches=self._state.censored_batches,
                retired_chains=tuple(
                    sorted(
                        (*self._state.retired_chains, *new_chains),
                        key=lambda row: (row.env_id, row.reset_generation),
                    )
                ),
                pending_true_reset_q0=tuple(
                    sorted(
                        (*self._state.pending_true_reset_q0, *new_pending),
                        key=lambda row: row.env_id,
                    )
                ),
                true_reset_batches=self._state.true_reset_batches,
                reset_high_waters=tuple(
                    high_waters[env_id] for env_id in sorted(high_waters)
                ),
                owner_transitions=self._state.owner_transitions,
            )
            active_after = _active_owner_root(provisional)
            unselected_after = _active_owner_root(
                provisional, excluded_env_ids=selected
            )
            receipt = TrueResetBatchReceipt(
                integration_status=INTEGRATION_STATUS,
                selected_env_ids=selected,
                retired_chains=tuple(new_chains),
                pending_q0=tuple(new_pending),
                sampler_checkpoint_sha256=live_sampler["checkpoint_sha256"],
                committed_transcript_sha256=_transcript_root(
                    self._state.committed_receipt_shas
                ),
                active_owner_root_before_sha256=active_before,
                active_owner_root_after_sha256=active_after,
                unselected_active_owner_root_before_sha256=unselected_before,
                unselected_active_owner_root_after_sha256=unselected_after,
                unselected_prepared_owner_root_sha256=_prepared_owner_root(
                    self._state, excluded_env_ids=selected
                ),
                parent_true_reset_transcript_sha256=_owner_transition_root(
                    self._state.owner_transitions
                ),
                external_closure_batch=closure_batch,
                external_closure_batch_sha256=closure_batch.canonical_sha256,
                policy_opportunity_created=False,
            )
            result = TrueResetBatchReceipt.from_mapping(receipt.to_mapping())
            transition = OwnerTransitionRef(
                transition_kind="TRUE_RESET_BATCH",
                transition_sha256=receipt.canonical_sha256,
            )
            final_state = _OwnerState(
                sampler_checkpoint_json=provisional.sampler_checkpoint_json,
                prepared=provisional.prepared,
                prepared_batches=provisional.prepared_batches,
                prepared_event_batches=provisional.prepared_event_batches,
                abort_event_receipts=provisional.abort_event_receipts,
                committed=provisional.committed,
                committed_batches=provisional.committed_batches,
                committed_receipt_shas=provisional.committed_receipt_shas,
                committed_batch_shas=provisional.committed_batch_shas,
                censored=provisional.censored,
                censored_batches=provisional.censored_batches,
                retired_chains=provisional.retired_chains,
                pending_true_reset_q0=provisional.pending_true_reset_q0,
                true_reset_batches=(*self._state.true_reset_batches, receipt),
                reset_high_waters=provisional.reset_high_waters,
                owner_transitions=(*self._state.owner_transitions, transition),
            )
            self._inject(fault_injector, "true_reset_many_before_publish")
            self._state = final_state
            return result
        finally:
            self._finish_operation()

    def _begin_operation(
        self,
        *,
        allow_active_preview: bool = False,
        allow_terminal_boundary_bind: bool = False,
    ) -> None:
        if self._poisoned:
            raise TransactionConflictError(
                "transaction owner is poisoned by a global reveal failure"
            )
        if self._operation_active:
            raise TransactionConflictError("transaction owner is not reentrant")
        if self._active_preview is not None and not allow_active_preview:
            raise TransactionConflictError(
                "reveal-final preview lease is active"
            )
        if (
            allow_terminal_boundary_bind
            and not self._terminal_boundary_bind_open
        ):
            if (
                self._terminal_boundary_authority is not None
                or self._terminal_boundary_authority_payload is not None
            ):
                raise TransactionConflictError(
                    "terminal boundary authority is already bound"
                )
            raise TransactionConflictError(
                "terminal boundary authority bind window is closed"
            )
        if not allow_terminal_boundary_bind:
            self._terminal_boundary_bind_open = False
        self._operation_active = True

    def _finish_operation(self) -> None:
        self._operation_active = False

    @staticmethod
    def _inject(fault_injector: _FaultInjector, point: str) -> None:
        if fault_injector is not None:
            if not callable(fault_injector):
                raise ContinuousRuntimeTransactionError(
                    "fault_injector must be callable"
                )
            fault_injector(point)

    def _validate_slots(
        self, slots: Sequence[BallSlotSnapshot]
    ) -> Tuple[BallSlotSnapshot, ...]:
        rows = tuple(slots)
        if len(rows) != self.ball_slot_capacity:
            raise BallSlotCapacityError(
                "ball slot snapshot width differs from configured capacity"
            )
        if tuple(slot.slot_index for slot in rows) != tuple(range(len(rows))):
            raise ContinuousRuntimeTransactionError(
                "ball slots must be complete and index ordered"
            )
        for name in ("owner_key_sha256", "inbound_ball_sha256"):
            identities = [
                getattr(slot, name)
                for slot in rows
                if getattr(slot, name) is not None
            ]
            if len(set(identities)) != len(identities):
                raise ContinuousRuntimeTransactionError(
                    f"ball slots contain duplicate {name} identities"
                )
        return rows

    @staticmethod
    def _validate_slot_evolution(
        prepared_slots: Sequence[BallSlotSnapshot],
        reveal_slots: Sequence[BallSlotSnapshot],
    ) -> None:
        """Allow physical progress while preserving every prepared owner.

        Empty slots are logical reservations and therefore cannot acquire an
        unrelated owner between preparation and reveal.  A non-empty prior
        ball may progress monotonically through its mailbox lifecycle and its
        dynamic-state digest may change, but its owner/generation/inbound-ball
        identity cannot change or disappear before the atomic new-ball install.
        """

        if len(prepared_slots) != len(reveal_slots):
            raise BallSlotCapacityError(
                "ball slot capacity changed after preparation"
            )
        for before, after in zip(prepared_slots, reveal_slots):
            if before.slot_index != after.slot_index:
                raise TransactionConflictError(
                    "ball slot ordering changed after preparation"
                )
            if before.lifecycle_state == BALL_EMPTY:
                if after.lifecycle_state != BALL_EMPTY:
                    raise TransactionConflictError(
                        "prepared empty ball-slot reservation acquired an owner"
                    )
                continue
            if after.lifecycle_state == BALL_EMPTY:
                raise TransactionConflictError(
                    "prepared prior-ball owner disappeared before reveal"
                )
            for name in (
                "owner_key_sha256",
                "ball_generation",
                "inbound_ball_sha256",
            ):
                if getattr(after, name) != getattr(before, name):
                    raise TransactionConflictError(
                        f"prior-ball {name} changed after preparation"
                    )
            if (
                _BALL_LIFECYCLE_RANK[after.lifecycle_state]
                < _BALL_LIFECYCLE_RANK[before.lifecycle_state]
            ):
                raise TransactionConflictError(
                    "prior-ball lifecycle moved backwards before reveal"
                )
            if before.physical_retired and not after.physical_retired:
                raise TransactionConflictError(
                    "prior ball became physically live after retirement"
                )

    def _current_committed(self, env_id: int) -> Optional[CommittedReveal]:
        rows = _active_committed_rows(self._state, env_id)
        return (
            None
            if not rows
            else max(
                rows,
                key=lambda item: item.prepared_reveal.request.scheduled_ordinal,
            )
        )

    def _current_sequence_advance(self, env_id: int) -> Optional[object]:
        rows = _active_sequence_rows(self._state, env_id)
        return None if not rows else rows[-1]

    def _pending_q0(self, env_id: int) -> Optional[PendingTrueResetQ0]:
        return next(
            (
                row
                for row in self._state.pending_true_reset_q0
                if row.env_id == env_id
            ),
            None,
        )

    def _validate_sequence_parent(
        self,
        request: ContinuousPrepareRequest,
        slots: Tuple[BallSlotSnapshot, ...],
    ) -> Optional[CommittedReveal]:
        previous = self._current_sequence_advance(request.env_id)
        return self._validate_sequence_parent_against(
            previous,
            request=request,
            slots=slots,
            pending_q0=self._pending_q0(request.env_id),
            physical_previous=self._current_committed(request.env_id),
        )

    def _validate_sequence_parent_against(
        self,
        previous: Optional[object],
        *,
        request: ContinuousPrepareRequest,
        slots: Tuple[BallSlotSnapshot, ...],
        pending_q0: Optional[PendingTrueResetQ0] = None,
        physical_previous: Optional[CommittedReveal] = None,
    ) -> Optional[object]:
        """Validate one successor against the latest committed/censored advance."""

        if request.scheduled_ordinal == 0:
            if previous is not None or request.previous_ball_slot_index is not None:
                raise TransactionConflictError(
                    "Q0 unexpectedly names a previous committed ball"
                )
            if pending_q0 is None:
                if request.sampler_generation != 1:
                    raise TransactionConflictError(
                        "initial Q0 sampler generation must equal one"
                    )
            elif request != pending_q0.next_q0_request:
                raise TransactionConflictError(
                    "Q0 differs from the registered selected true reset"
                )
            return None
        if previous is None or not isinstance(
            previous, (CommittedReveal, CensoredReveal)
        ):
            raise TransactionConflictError(
                "successor prepare lacks previous sequence advance"
            )
        parent = previous.prepared_reveal
        parent_request = parent.request
        parent_ref = parent.selected_task_ref
        for name in (
            "env_id",
            "reset_generation",
            "action_uid",
            "action_slot",
            "birth_sha256",
            "run_id",
            "carry_chain_id",
            "schedule_sha256",
            "selection_authority_sha256",
            "source_sha256",
            "config_sha256",
        ):
            if getattr(request, name) != getattr(parent_request, name):
                raise TransactionConflictError(f"successor {name} differs")
        if request.scheduled_ordinal != parent_request.scheduled_ordinal + 1:
            raise TransactionConflictError("scheduled ordinal did not advance by one")
        if request.runtime_swing_generation != parent_ref.swing_generation + 1:
            raise TransactionConflictError(
                "runtime swing generation did not advance by one"
            )
        if (
            request.sampler_generation
            != parent.selection.target_generation + 1
        ):
            raise TransactionConflictError(
                "absolute sampler generation did not advance by one"
            )
        if (
            request.admission_evaluated_step
            <= parent_request.scheduled_deadline_step
            or request.scheduled_reveal_step
            <= parent_request.scheduled_deadline_step
        ):
            raise TransactionConflictError(
                "successor admission/reveal did not follow the prior deadline"
            )
        if physical_previous is None and isinstance(previous, CommittedReveal):
            physical_previous = previous
        previous_slot = request.previous_ball_slot_index
        if physical_previous is None:
            if previous_slot is not None or any(
                slot.lifecycle_state != BALL_EMPTY for slot in slots
            ):
                raise TransactionConflictError(
                    "censored-only chain unexpectedly names a physical ball"
                )
        else:
            physical_parent = physical_previous.prepared_reveal
            if previous_slot != physical_previous.ball_slot_plan.selected_slot_index:
                raise TransactionConflictError(
                    "previous ball slot differs from physical parent"
                )
            assert previous_slot is not None
            slot = slots[previous_slot]
            if (
                slot.owner_key_sha256
                != physical_parent.outcome_key.canonical_sha256
                or slot.ball_generation
                != physical_previous.ball_slot_plan.new_ball_generation
                or slot.inbound_ball_sha256
                != physical_previous.ball_slot_plan.new_inbound_ball_sha256
            ):
                raise TransactionConflictError(
                    "previous physical ball owner differs from committed parent"
                )
        return previous

    def _validate_prepared_integrity(self, prepared: PreparedReveal) -> None:
        """Re-bind every nested identity carried by one private prepared row."""

        if not isinstance(prepared, PreparedReveal):
            raise ContinuousRuntimeTransactionError("prepared row type differs")
        request = prepared.request
        slots = self._validate_slots(request.ball_slots)
        if len(prepared.candidates) != len(self.profile.cells):
            raise ContinuousRuntimeTransactionError(
                "prepared candidate width differs from target profile"
            )
        by_cell: dict[str, CandidateTaskMaterialization] = {}
        for cell, row in zip(self.profile.cells, prepared.candidates):
            if (
                row.cell_id != cell.cell_id
                or row.target_xy_m != cell.target
                or row.target_semantic_sha256
                != self.profile.semantic_sha256(cell)
                or row.evaluated_step != request.admission_evaluated_step
            ):
                raise ContinuousRuntimeTransactionError(
                    "prepared candidate/profile binding differs"
                )
            by_cell[row.cell_id] = row

        selection = prepared.selection
        if (
            selection.profile_sha256 != self.profile.profile_sha256
            or selection.env_id != request.env_id
            or selection.target_generation != request.sampler_generation
            or selection.frame_id != self.profile.frame_id
            or selection.frame_binding_sha256
            != self.profile.frame_binding_sha256
            or selection.runtime_dtype != self.profile.runtime_dtype
            or selection.quantization_contract
            != self.profile.quantization_contract
            or selection.components != self.profile.components
            or selection.cell_id not in by_cell
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared sampler selection binding differs"
            )
        selected = by_cell[selection.cell_id]
        if (
            not selected.construction_feasible
            or selected.task_ref is None
            or tuple(selection.target) != selected.target_xy_m
            or selection.semantic_sha256 != selected.target_semantic_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared selected candidate binding differs"
            )
        ref = prepared.selected_task_ref
        if ref != selected.task_ref:
            raise ContinuousRuntimeTransactionError(
                "prepared selected task ref differs from candidate"
            )
        expected_ref_values = {
            "env_id": request.env_id,
            "reset_generation": request.reset_generation,
            "swing_generation": request.runtime_swing_generation,
            "action_uid": request.action_uid,
            "action_slot": request.action_slot,
            "birth_sha256": request.birth_sha256,
        }
        if any(
            getattr(ref, name) != expected
            for name, expected in expected_ref_values.items()
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared task ref/request binding differs"
            )

        target = prepared.runtime_target_receipt
        runtime_target = _runtime_xy(selection.target, label="prepared target")
        if (
            target.profile_sha256 != self.profile.profile_sha256
            or target.selection_authority_sha256
            != request.selection_authority_sha256
            or target.runtime_dtype != self.profile.runtime_dtype
            or target.target_generation != request.runtime_swing_generation
            or target.task_ref_sha256 != ref.canonical_sha256
            or (
                target.requested_target_x_m,
                target.requested_target_y_m,
            )
            != tuple(selection.target)
            or target.runtime_target_xy_m != runtime_target
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared runtime target receipt binding differs"
            )

        assert selected.receipt_content_sha256 is not None
        expected_outcome = {
            **ref.runtime_dict(),
            "run_id": request.run_id,
            "carry_chain_id": request.carry_chain_id,
            "shot_index": request.outcome_shot_index,
            "source_sha256": request.source_sha256,
            "config_sha256": request.config_sha256,
            "receipt_content_sha256": selected.receipt_content_sha256,
        }
        if prepared.outcome_key.full_key_dict() != expected_outcome:
            raise ContinuousRuntimeTransactionError(
                "prepared full landing-outcome key binding differs"
            )

        if selected.inbound_ball_generation != request.runtime_swing_generation:
            raise ContinuousRuntimeTransactionError(
                "prepared inbound ball generation differs"
            )
        expected_reservation = self._ball_reservation(
            request=request,
            slots=slots,
            candidate=selected,
        )
        if prepared.prepared_ball_slot_reservation != expected_reservation:
            raise ContinuousRuntimeTransactionError(
                "prepared ball-slot reservation binding differs"
            )

        before = prepared.sampler_before_env_state
        after = dict(prepared.sampler_after_env_state)
        expected_row_keys = frozenset(
            (
                "env_id",
                "rng_state",
                "draw_count",
                "target_generation",
                "previous_cell_id",
                "previous_semantic_sha256",
            )
        )
        if frozenset(after) != expected_row_keys:
            raise ContinuousRuntimeTransactionError(
                "prepared sampler after-row keys differ"
            )
        if (
            after["env_id"] != request.env_id
            or after["draw_count"] != request.sampler_generation
            or after["target_generation"] != request.sampler_generation
            or after["previous_cell_id"] != selection.cell_id
            or after["previous_semantic_sha256"] != selection.semantic_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared sampler after-row binding differs"
            )
        if before is None:
            if request.sampler_generation != 1:
                raise ContinuousRuntimeTransactionError(
                    "non-initial prepared row lacks sampler before-state"
                )
        else:
            prior = dict(before)
            if frozenset(prior) != expected_row_keys:
                raise ContinuousRuntimeTransactionError(
                    "prepared sampler before-row keys differ"
                )
            if (
                prior["env_id"] != request.env_id
                or prior["draw_count"] != request.sampler_generation - 1
                or prior["target_generation"] != request.sampler_generation - 1
            ):
                raise ContinuousRuntimeTransactionError(
                    "prepared sampler before-row binding differs"
                )
        if prepared.sampler_before_env_state_sha256 != _sampler_env_state_sha256(
            request.env_id,
            before,
            phase="before",
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared sampler before-row SHA differs"
            )
        if prepared.sampler_after_env_state_sha256 != _sampler_env_state_sha256(
            request.env_id,
            after,
            phase="after",
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared sampler after-row SHA differs"
            )

        replay = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile,
            _single_env_sampler_checkpoint(
                self.profile,
                seed=self.seed,
                before=before,
            ),
        )
        replayed_selection = replay.sample_next(
            request.env_id,
            feasible_mask=tuple(
                row.construction_feasible for row in prepared.candidates
            ),
        )
        replayed_after = _sampler_env_row(replay.checkpoint(), request.env_id)
        if (
            replayed_selection.to_mapping() != selection.to_mapping()
            or replayed_after != after
        ):
            raise ContinuousRuntimeTransactionError(
                "prepared sampler transition is not one exact clone draw"
            )

    def _validate_committed_integrity(self, committed: CommittedReveal) -> None:
        """Re-bind a published identity row to the frozen reveal facts."""

        if not isinstance(committed, CommittedReveal):
            raise ContinuousRuntimeTransactionError("committed row type differs")
        self._validate_prepared_integrity(committed.prepared_reveal)
        request = committed.prepared_reveal.request
        facts = committed.reveal_facts
        expected = {
            "env_id": request.env_id,
            "reset_generation": request.reset_generation,
            "scheduled_ordinal": request.scheduled_ordinal,
            "runtime_swing_generation": request.runtime_swing_generation,
            "sampler_generation": request.sampler_generation,
            "outcome_shot_index": request.outcome_shot_index,
            "schedule_sha256": request.schedule_sha256,
            "reveal_step": request.scheduled_reveal_step,
            "deadline_step": request.scheduled_deadline_step,
        }
        if any(getattr(facts, name) != value for name, value in expected.items()):
            raise ContinuousRuntimeTransactionError(
                "committed reveal facts/prepared binding differs"
            )
        reveal_slots = self._validate_slots(facts.ball_slots)
        self._validate_slot_evolution(request.ball_slots, reveal_slots)
        selected = next(
            row
            for row in committed.prepared_reveal.candidates
            if row.cell_id == committed.prepared_reveal.selection.cell_id
        )
        expected_plan = self._ball_plan(
            request=request,
            slots=reveal_slots,
            candidate=selected,
        )
        if committed.ball_slot_plan != expected_plan:
            raise ContinuousRuntimeTransactionError(
                "committed reveal-time ball-slot plan binding differs"
            )
        hidden = facts.pre_reveal_hidden
        if (
            hidden.hidden_from_step != request.admission_evaluated_step
            or hidden.hidden_through_step != request.scheduled_reveal_step - 1
            or hidden.observed_tick_count
            != request.scheduled_reveal_step
            - request.admission_evaluated_step
            or hidden.first_visible_step != request.scheduled_reveal_step
        ):
            raise ContinuousRuntimeTransactionError(
                "pre-reveal hidden witness interval differs from preparation"
            )

    @staticmethod
    def _validate_successor_content_against(
        previous: object,
        prepared: PreparedReveal,
    ) -> None:
        """Validate content freshness that is not carried by request counters."""

        if not isinstance(previous, (CommittedReveal, CensoredReveal)):
            raise ContinuousRuntimeTransactionError(
                "successor content parent type differs"
            )
        parent = previous.prepared_reveal
        if (
            prepared.runtime_target_receipt.runtime_target_xy_m
            == parent.runtime_target_receipt.runtime_target_xy_m
            or prepared.selection.semantic_sha256
            == parent.selection.semantic_sha256
        ):
            raise Float32TargetAliasError(
                "adjacent target identity is reused"
            )
        if (
            prepared.selected_task_ref.sample_sha256
            == parent.selected_task_ref.sample_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "adjacent sample identity is reused"
            )
        if (
            prepared.selected_task_ref.task_sha256
            == parent.selected_task_ref.task_sha256
        ):
            raise ContinuousRuntimeTransactionError(
                "adjacent task identity is reused"
            )

    def _ball_reservation(
        self,
        *,
        request: ContinuousPrepareRequest,
        slots: Tuple[BallSlotSnapshot, ...],
        candidate: CandidateTaskMaterialization,
    ) -> PreparedBallSlotReservation:
        if candidate.inbound_ball_sha256 in {
            slot.inbound_ball_sha256
            for slot in slots
            if slot.inbound_ball_sha256 is not None
        }:
            raise ContinuousRuntimeTransactionError(
                "new inbound ball identity reuses an occupied slot identity"
            )
        reusable = tuple(slot.slot_index for slot in slots if slot.reusable)
        observed_owners = tuple(
            slot.owner_key_sha256
            for slot in slots
            if slot.owner_key_sha256 is not None
        )
        assert candidate.inbound_ball_generation is not None
        assert candidate.inbound_ball_sha256 is not None
        assert candidate.installed_ball_dynamic_state_sha256 is not None
        assert candidate.physical_ball_install_payload_sha256 is not None
        return PreparedBallSlotReservation(
            capacity=self.ball_slot_capacity,
            snapshot_sha256=_ball_snapshot_sha256(slots),
            previous_slot_index=request.previous_ball_slot_index,
            reusable_slot_indices=reusable,
            capacity_available_at_prepare=bool(reusable),
            observed_prior_owner_key_sha256=observed_owners,
            new_ball_generation=candidate.inbound_ball_generation,
            new_inbound_ball_sha256=candidate.inbound_ball_sha256,
            new_ball_dynamic_state_sha256=(
                candidate.installed_ball_dynamic_state_sha256
            ),
            physical_ball_install_payload_sha256=(
                candidate.physical_ball_install_payload_sha256
            ),
        )

    def _ball_plan(
        self,
        *,
        request: ContinuousPrepareRequest,
        slots: Tuple[BallSlotSnapshot, ...],
        candidate: CandidateTaskMaterialization,
    ) -> BallSlotPlan:
        previous = request.previous_ball_slot_index
        if previous is not None and slots[previous].reusable:
            selected = previous
        else:
            reusable = [slot.slot_index for slot in slots if slot.reusable]
            if not reusable:
                raise BallSlotCapacityError(
                    "no reusable ball slot; OPEN/inbound owner would be overwritten"
                )
            selected = min(reusable)
        if candidate.inbound_ball_sha256 in {
            slot.inbound_ball_sha256
            for slot in slots
            if slot.inbound_ball_sha256 is not None
        }:
            raise ContinuousRuntimeTransactionError(
                "new inbound ball identity reuses an occupied slot identity"
            )
        preserved = tuple(
            slot.owner_key_sha256
            for slot in slots
            if slot.slot_index != selected
            and slot.lifecycle_state != BALL_EMPTY
            and slot.owner_key_sha256 is not None
        )
        assert candidate.inbound_ball_generation is not None
        assert candidate.inbound_ball_sha256 is not None
        assert candidate.installed_ball_dynamic_state_sha256 is not None
        assert candidate.physical_ball_install_payload_sha256 is not None
        reused_owner = (
            None
            if slots[selected].lifecycle_state == BALL_EMPTY
            else slots[selected].owner_key_sha256
        )
        return BallSlotPlan(
            capacity=self.ball_slot_capacity,
            snapshot_sha256=_ball_snapshot_sha256(slots),
            selected_slot_index=selected,
            previous_slot_index=previous,
            reused_previous_slot=(previous is not None and selected == previous),
            preserved_live_owner_key_sha256=preserved,
            new_ball_generation=candidate.inbound_ball_generation,
            new_inbound_ball_sha256=candidate.inbound_ball_sha256,
            new_ball_dynamic_state_sha256=(
                candidate.installed_ball_dynamic_state_sha256
            ),
            physical_ball_install_payload_sha256=(
                candidate.physical_ball_install_payload_sha256
            ),
            reused_retired_owner_key_sha256=reused_owner,
        )

    def _prepare_inputs(
        self,
        request: ContinuousPrepareRequest,
        candidates: Sequence[CandidateTaskMaterialization],
    ) -> tuple[
        Tuple[CandidateTaskMaterialization, ...],
        Tuple[BallSlotSnapshot, ...],
        Optional[object],
        Tuple[bool, ...],
        dict[str, CandidateTaskMaterialization],
    ]:
        if not isinstance(request, ContinuousPrepareRequest):
            raise ContinuousRuntimeTransactionError(
                "prepare request must be ContinuousPrepareRequest"
            )
        if any(
            row.request.env_id == request.env_id for row in self._state.prepared
        ):
            raise TransactionConflictError(
                "environment already has a prepared reveal"
            )
        rows = tuple(candidates)
        slots = self._validate_slots(request.ball_slots)
        previous = self._validate_sequence_parent(request, slots)
        if len(rows) != len(self.profile.cells):
            raise ContinuousRuntimeTransactionError(
                "candidate materialization width differs from target profile"
            )
        feasible_mask = []
        by_cell: dict[str, CandidateTaskMaterialization] = {}
        for cell, row in zip(self.profile.cells, rows):
            if not isinstance(row, CandidateTaskMaterialization):
                raise ContinuousRuntimeTransactionError("candidate row type differs")
            expected_semantic = self.profile.semantic_sha256(cell)
            if (
                row.cell_id != cell.cell_id
                or row.target_xy_m != cell.target
                or row.target_semantic_sha256 != expected_semantic
            ):
                raise ContinuousRuntimeTransactionError(
                    "candidate target identity differs from profile cell"
                )
            if row.evaluated_step != request.admission_evaluated_step:
                raise ContinuousRuntimeTransactionError(
                    "candidate evaluation step differs from admission"
                )
            feasible_mask.append(row.construction_feasible)
            by_cell[row.cell_id] = row
        return rows, slots, previous, tuple(feasible_mask), by_cell

    def _prepare_one_on_clone(
        self,
        *,
        request: ContinuousPrepareRequest,
        rows: Tuple[CandidateTaskMaterialization, ...],
        slots: Tuple[BallSlotSnapshot, ...],
        previous: Optional[object],
        feasible_mask: Tuple[bool, ...],
        by_cell: Mapping[str, CandidateTaskMaterialization],
        sampler_before_env_state: Optional[Mapping[str, object]],
        clone: _target_sampler.ContinuousTargetSampler,
    ) -> PreparedReveal:
        selection = clone.sample_next(request.env_id, feasible_mask=feasible_mask)
        if selection.target_generation != request.sampler_generation:
            raise TransactionConflictError(
                "sampler generation differs from scheduled mapping"
            )
        selected = by_cell[selection.cell_id]
        if not selected.construction_feasible or selected.task_ref is None:
            raise ContinuousRuntimeTransactionError(
                "sampler selected an infeasible candidate"
            )
        ref = selected.task_ref
        for name in (
            "env_id",
            "reset_generation",
            "runtime_swing_generation",
            "action_uid",
            "action_slot",
            "birth_sha256",
        ):
            ref_name = (
                "swing_generation" if name == "runtime_swing_generation" else name
            )
            if getattr(ref, ref_name) != getattr(request, name):
                raise ContinuousRuntimeTransactionError(
                    f"selected task ref {ref_name} differs"
                )
        if selected.inbound_ball_generation != request.runtime_swing_generation:
            raise ContinuousRuntimeTransactionError(
                "inbound ball generation differs from runtime swing"
            )
        assert selected.inbound_ball_sha256 is not None
        prior_inbound_ball_shas = {
            row.ball_slot_plan.new_inbound_ball_sha256
            for row in _active_committed_rows(self._state, request.env_id)
        }
        if selected.inbound_ball_sha256 in prior_inbound_ball_shas:
            raise ContinuousRuntimeTransactionError(
                "inbound ball identity is reused within the carry chain"
            )
        runtime_target = _runtime_xy(selection.target, label="selected target")
        if previous is not None:
            previous_prepared = previous.prepared_reveal
            if (
                runtime_target
                == previous_prepared.runtime_target_receipt.runtime_target_xy_m
            ):
                raise Float32TargetAliasError(
                    "adjacent runtime float32 targets are identical"
                )
            previous_ref = previous_prepared.selected_task_ref
            if ref.sample_sha256 == previous_ref.sample_sha256:
                raise ContinuousRuntimeTransactionError(
                    "adjacent sample identity is reused"
                )
            if ref.task_sha256 == previous_ref.task_sha256:
                raise ContinuousRuntimeTransactionError(
                    "adjacent task identity is reused"
                )
        if (
            _runtime_xy(selected.target_xy_m, label="materialized target")
            != runtime_target
        ):
            raise ContinuousRuntimeTransactionError(
                "materialized task target differs from selected runtime target"
            )
        runtime_target_receipt = _successor.TargetSelectionReceipt(
            profile_sha256=self.profile.profile_sha256,
            selection_authority_sha256=request.selection_authority_sha256,
            runtime_dtype=_successor.RUNTIME_TARGET_DTYPE,
            target_generation=request.runtime_swing_generation,
            task_ref_sha256=ref.canonical_sha256,
            requested_target_x_m=selection.target[0],
            requested_target_y_m=selection.target[1],
            runtime_target_x_m=runtime_target[0],
            runtime_target_y_m=runtime_target[1],
            semantic_sha256=_successor.target_semantic_sha256(
                self.profile.profile_sha256,
                runtime_target[0],
                runtime_target[1],
            ),
        )
        assert selected.receipt_content_sha256 is not None
        outcome_key = _mailbox.LandingOutcomeShotKey(
            **ref.runtime_dict(),
            run_id=request.run_id,
            carry_chain_id=request.carry_chain_id,
            shot_index=request.outcome_shot_index,
            source_sha256=request.source_sha256,
            config_sha256=request.config_sha256,
            receipt_content_sha256=selected.receipt_content_sha256,
        )
        ball_reservation = self._ball_reservation(
            request=request, slots=slots, candidate=selected
        )
        before_row = (
            None
            if sampler_before_env_state is None
            else dict(sampler_before_env_state)
        )
        after_row = clone.checkpoint_env_state(request.env_id)
        pending_q0 = self._pending_q0(request.env_id)
        if (
            request.scheduled_ordinal == 0
            and pending_q0 is not None
            and _sampler_env_state_sha256(
                request.env_id, before_row, phase="before"
            )
            != pending_q0.sampler_before_env_state_sha256
        ):
            raise TransactionConflictError(
                "true-reset Q0 sampler high-water differs"
            )
        if after_row is None:
            raise ContinuousRuntimeTransactionError(
                "staged sampler checkpoint lacks selected environment"
            )
        prepared = PreparedReveal(
            integration_status=INTEGRATION_STATUS,
            phase=PREPARED,
            public_visible=False,
            policy_opportunity_created=False,
            request=request,
            candidates=rows,
            selection=selection,
            runtime_target_receipt=runtime_target_receipt,
            selected_task_ref=ref,
            outcome_key=outcome_key,
            prepared_ball_slot_reservation=ball_reservation,
            sampler_before_env_state_sha256=_sampler_env_state_sha256(
                request.env_id, before_row, phase="before"
            ),
            sampler_after_env_state_sha256=_sampler_env_state_sha256(
                request.env_id, after_row, phase="after"
            ),
            sampler_before_env_state=before_row,
            sampler_after_env_state=after_row,
        )
        self._validate_prepared_integrity(prepared)
        if previous is not None:
            self._validate_successor_content_against(previous, prepared)
        return prepared

    def _validate_prepared_batch_integrity(
        self, batch: PreparedRevealBatch
    ) -> None:
        if not isinstance(batch, PreparedRevealBatch):
            raise ContinuousRuntimeTransactionError(
                "prepared batch type differs"
            )
        for row in batch.prepared_reveals:
            self._validate_prepared_integrity(row)
        before_sampler = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, batch.sampler_checkpoint_before
        )
        after_sampler = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, batch.sampler_checkpoint_after_private
        )
        if before_sampler.seed != self.seed or after_sampler.seed != self.seed:
            raise ContinuousRuntimeTransactionError(
                "prepared batch sampler seed differs"
            )
        before_rows = _sampler_env_rows_by_id(
            batch.sampler_checkpoint_before
        )
        after_rows = _sampler_env_rows_by_id(
            batch.sampler_checkpoint_after_private
        )
        for row in batch.prepared_reveals:
            env_id = row.request.env_id
            if (
                before_rows.get(env_id) != row.sampler_before_env_state
                or after_rows.get(env_id) != row.sampler_after_env_state
            ):
                raise ContinuousRuntimeTransactionError(
                    "prepared batch selected sampler row binding differs"
                )
        expected_after = _merge_sampler_env_rows(
            batch.sampler_checkpoint_before, batch.prepared_reveals
        )
        if expected_after != batch.sampler_checkpoint_after_private:
            raise ContinuousRuntimeTransactionError(
                "prepared batch is not one full-owner clone transition"
            )

    def _prepare_many_impl(
        self,
        requests: Tuple[ContinuousPrepareRequest, ...],
        candidate_rows: Tuple[Tuple[CandidateTaskMaterialization, ...], ...],
        *,
        fault_injector: _FaultInjector,
        legacy_fault_points: bool,
    ) -> PreparedRevealBatch:
        if len(requests) != len(candidate_rows) or not requests:
            raise ContinuousRuntimeTransactionError(
                "prepare_many request/candidate widths differ or are empty"
            )
        if any(not isinstance(request, ContinuousPrepareRequest) for request in requests):
            raise ContinuousRuntimeTransactionError(
                "prepare_many requests must be ContinuousPrepareRequest"
            )
        _ordered_unique_env_ids(
            tuple(request.env_id for request in requests),
            label="selected_env_ids",
        )
        validated = tuple(
            self._prepare_inputs(request, rows)
            for request, rows in zip(requests, candidate_rows)
        )
        self._inject(fault_injector, "prepare_many_after_validation")
        if legacy_fault_points:
            self._inject(fault_injector, "prepare_after_validation")
        live_checkpoint = self.sampler_checkpoint()
        live_rows_by_env = _sampler_env_rows_by_id(live_checkpoint)
        clone = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, live_checkpoint
        )
        prepared_rows = []
        for request, values in zip(requests, validated):
            rows, slots, previous, feasible_mask, by_cell = values
            prepared_rows.append(
                self._prepare_one_on_clone(
                    request=request,
                    rows=rows,
                    slots=slots,
                    previous=previous,
                    feasible_mask=feasible_mask,
                    by_cell=by_cell,
                    sampler_before_env_state=live_rows_by_env.get(
                        request.env_id
                    ),
                    clone=clone,
                )
            )
            self._inject(fault_injector, "prepare_many_after_row")
        private_checkpoint = clone.checkpoint()
        self._inject(fault_injector, "prepare_many_after_sampler_clone")
        if legacy_fault_points:
            self._inject(fault_injector, "prepare_after_sampler_clone")
        env_ids = tuple(request.env_id for request in requests)
        batch = PreparedRevealBatch(
            integration_status=INTEGRATION_STATUS,
            phase=PREPARED,
            public_visible=False,
            policy_opportunity_created=False,
            selected_env_ids=env_ids,
            task_birth_snapshot_ids=tuple(
                request.task_birth_snapshot_id for request in requests
            ),
            sampler_checkpoint_before_sha256=live_checkpoint[
                "checkpoint_sha256"
            ],
            sampler_checkpoint_after_private_sha256=private_checkpoint[
                "checkpoint_sha256"
            ],
            untouched_rows_before_sha256=_sampler_untouched_rows_root(
                live_checkpoint, env_ids
            ),
            untouched_rows_after_sha256=_sampler_untouched_rows_root(
                private_checkpoint, env_ids
            ),
            sampler_checkpoint_before=live_checkpoint,
            sampler_checkpoint_after_private=private_checkpoint,
            prepared_reveals=tuple(prepared_rows),
        )
        self._validate_prepared_batch_integrity(batch)
        self._inject(fault_injector, "prepare_many_before_publish")
        if legacy_fault_points:
            self._inject(fault_injector, "prepare_before_publish")
        # ``batch`` was constructed entirely from owner-cloned inputs and its
        # post-init canonicalizes every nested mapping.  Retain it privately
        # and build exactly one isolated public image.  The former
        # private->private->public double round-trip serialized the full K-row
        # graph twice without adding another authority boundary.
        stored_batch = batch
        result_batch = _clone_prepared_batch(stored_batch)
        state = self._state
        transition = OwnerTransitionRef(
            transition_kind="PREPARE_BATCH",
            transition_sha256=stored_batch.canonical_sha256,
        )
        self._state = _OwnerState(
            sampler_checkpoint_json=state.sampler_checkpoint_json,
            prepared=tuple(
                sorted(
                    (*state.prepared, *stored_batch.prepared_reveals),
                    key=lambda row: row.request.env_id,
                )
            ),
            prepared_batches=tuple(
                sorted(
                    (*state.prepared_batches, stored_batch),
                    key=lambda row: row.selected_env_ids,
                )
            ),
            prepared_event_batches=(
                *state.prepared_event_batches,
                stored_batch,
            ),
            abort_event_receipts=state.abort_event_receipts,
            committed=state.committed,
            committed_batches=state.committed_batches,
            committed_receipt_shas=state.committed_receipt_shas,
            committed_batch_shas=state.committed_batch_shas,
            censored=state.censored,
            censored_batches=state.censored_batches,
            retired_chains=state.retired_chains,
            pending_true_reset_q0=state.pending_true_reset_q0,
            true_reset_batches=state.true_reset_batches,
            reset_high_waters=state.reset_high_waters,
            owner_transitions=(*state.owner_transitions, transition),
        )
        return result_batch

    def prepare_many(
        self,
        requests: Sequence[ContinuousPrepareRequest],
        candidate_rows: Sequence[Sequence[CandidateTaskMaterialization]],
        *,
        fault_injector: _FaultInjector = None,
    ) -> PreparedRevealBatch:
        """Privately prepare ordered unique environments on one sampler clone."""

        self._begin_operation()
        try:
            request_tuple = tuple(requests)
            candidate_tuple = tuple(tuple(rows) for rows in candidate_rows)
            return self._prepare_many_impl(
                request_tuple,
                candidate_tuple,
                fault_injector=fault_injector,
                legacy_fault_points=False,
            )
        finally:
            self._finish_operation()

    def prepare(
        self,
        request: ContinuousPrepareRequest,
        candidates: Sequence[CandidateTaskMaterialization],
        *,
        fault_injector: _FaultInjector = None,
    ) -> PreparedReveal:
        """K=1 compatibility API backed by the batch authority."""

        self._begin_operation()
        try:
            batch = self._prepare_many_impl(
                (request,),
                (tuple(candidates),),
                fault_injector=fault_injector,
                legacy_fault_points=True,
            )
            return batch.prepared_reveals[0]
        finally:
            self._finish_operation()

    def _prepared_batch_by_sha(self, digest: str) -> PreparedRevealBatch:
        batch = next(
            (
                row
                for row in self._state.prepared_batches
                if row.canonical_sha256 == digest
            ),
            None,
        )
        if batch is None:
            raise TransactionConflictError("unknown prepared batch")
        return batch

    def _batch_for_prepared_reveal(self, digest: str) -> PreparedRevealBatch:
        matches = tuple(
            batch
            for batch in self._state.prepared_batches
            if any(
                row.canonical_sha256 == digest
                for row in batch.prepared_reveals
            )
        )
        if len(matches) != 1:
            raise TransactionConflictError("unknown prepared reveal")
        if len(matches[0].prepared_reveals) != 1:
            raise TransactionConflictError(
                "cannot partially operate on a multi-environment batch"
            )
        return matches[0]

    def _abort_many_impl(
        self,
        batch: PreparedRevealBatch,
        *,
        fault_injector: _FaultInjector,
        legacy_fault_points: bool,
    ) -> AbortBatchReceipt:
        self._validate_prepared_batch_integrity(batch)
        live_checkpoint = self.sampler_checkpoint()
        for prepared in batch.prepared_reveals:
            if (
                _sampler_env_row(live_checkpoint, prepared.request.env_id)
                != prepared.sampler_before_env_state
            ):
                raise TransactionConflictError(
                    "live sampler changed before prepared batch abort"
                )
        abort_rows = tuple(
            AbortReceipt(
                integration_status=INTEGRATION_STATUS,
                phase_before=PREPARED,
                phase_after=EMPTY,
                env_id=prepared.request.env_id,
                prepared_reveal_sha256=prepared.canonical_sha256,
                sampler_checkpoint_sha256=live_checkpoint[
                    "checkpoint_sha256"
                ],
                policy_opportunity_created=False,
            )
            for prepared in batch.prepared_reveals
        )
        receipt = AbortBatchReceipt(
            integration_status=INTEGRATION_STATUS,
            phase_before=PREPARED,
            phase_after=EMPTY,
            prepared_batch_sha256=batch.canonical_sha256,
            selected_env_ids=batch.selected_env_ids,
            task_birth_snapshot_ids=batch.task_birth_snapshot_ids,
            prepared_reveal_sha256s=tuple(
                row.canonical_sha256 for row in batch.prepared_reveals
            ),
            prepared_batch=batch,
            sampler_checkpoint_sha256=live_checkpoint["checkpoint_sha256"],
            untouched_rows_sha256=_sampler_untouched_rows_root(
                live_checkpoint, batch.selected_env_ids
            ),
            sampler_checkpoint=live_checkpoint,
            policy_opportunity_created=False,
            abort_receipts=abort_rows,
        )
        self._inject(fault_injector, "abort_many_before_publish")
        if legacy_fault_points:
            self._inject(fault_injector, "abort_before_publish")
        removed = {
            row.canonical_sha256 for row in batch.prepared_reveals
        }
        result_receipt = AbortBatchReceipt.from_mapping(receipt.to_mapping())
        state = self._state
        transition = OwnerTransitionRef(
            transition_kind="ABORT_BATCH",
            transition_sha256=receipt.canonical_sha256,
        )
        self._state = _OwnerState(
            sampler_checkpoint_json=state.sampler_checkpoint_json,
            prepared=tuple(
                row
                for row in state.prepared
                if row.canonical_sha256 not in removed
            ),
            prepared_batches=tuple(
                row
                for row in state.prepared_batches
                if row.canonical_sha256 != batch.canonical_sha256
            ),
            prepared_event_batches=state.prepared_event_batches,
            abort_event_receipts=(*state.abort_event_receipts, receipt),
            committed=state.committed,
            committed_batches=state.committed_batches,
            committed_receipt_shas=state.committed_receipt_shas,
            committed_batch_shas=state.committed_batch_shas,
            censored=state.censored,
            censored_batches=state.censored_batches,
            retired_chains=state.retired_chains,
            pending_true_reset_q0=state.pending_true_reset_q0,
            true_reset_batches=state.true_reset_batches,
            reset_high_waters=state.reset_high_waters,
            owner_transitions=(*state.owner_transitions, transition),
        )
        return result_receipt

    def abort_many(
        self,
        prepared_batch_sha256: str,
        *,
        fault_injector: _FaultInjector = None,
    ) -> AbortBatchReceipt:
        digest = _sha256(
            prepared_batch_sha256, label="prepared_batch_sha256"
        )
        self._begin_operation()
        try:
            return self._abort_many_impl(
                self._prepared_batch_by_sha(digest),
                fault_injector=fault_injector,
                legacy_fault_points=False,
            )
        finally:
            self._finish_operation()

    def abort(
        self,
        prepared_reveal_sha256: str,
        *,
        fault_injector: _FaultInjector = None,
    ) -> AbortReceipt:
        """K=1 compatibility API; multi-env batches cannot be split."""

        digest = _sha256(
            prepared_reveal_sha256, label="prepared_reveal_sha256"
        )
        self._begin_operation()
        try:
            receipt = self._abort_many_impl(
                self._batch_for_prepared_reveal(digest),
                fault_injector=fault_injector,
                legacy_fault_points=True,
            )
            return receipt.abort_receipts[0]
        finally:
            self._finish_operation()

    def _preview_reveal_final_row(
        self,
        prepared: PreparedReveal,
        facts: ContinuousRevealFacts,
    ) -> RevealFinalInstallRow:
        if not isinstance(facts, ContinuousRevealFacts):
            raise ContinuousRuntimeTransactionError(
                "reveal_facts must be ContinuousRevealFacts"
            )
        request = prepared.request
        equality = (
            ("env_id", facts.env_id, request.env_id),
            ("reset_generation", facts.reset_generation, request.reset_generation),
            ("scheduled_ordinal", facts.scheduled_ordinal, request.scheduled_ordinal),
            (
                "runtime_swing_generation",
                facts.runtime_swing_generation,
                request.runtime_swing_generation,
            ),
            ("sampler_generation", facts.sampler_generation, request.sampler_generation),
            (
                "outcome_shot_index",
                facts.outcome_shot_index,
                request.outcome_shot_index,
            ),
            ("schedule_sha256", facts.schedule_sha256, request.schedule_sha256),
            ("reveal_step", facts.reveal_step, request.scheduled_reveal_step),
            ("deadline_step", facts.deadline_step, request.scheduled_deadline_step),
        )
        for name, actual, expected in equality:
            if actual != expected:
                raise TransactionConflictError(
                    f"reveal fact {name} differs from prepared row"
                )
        slots = self._validate_slots(facts.ball_slots)
        self._validate_slot_evolution(request.ball_slots, slots)
        selected = next(
            row
            for row in prepared.candidates
            if row.cell_id == prepared.selection.cell_id
        )
        plan = self._ball_plan(
            request=request, slots=slots, candidate=selected
        )
        post_install = list(slots)
        post_install[plan.selected_slot_index] = BallSlotSnapshot(
            slot_index=plan.selected_slot_index,
            lifecycle_state=BALL_INBOUND,
            physical_retired=False,
            owner_key_sha256=prepared.outcome_key.canonical_sha256,
            ball_generation=plan.new_ball_generation,
            inbound_ball_sha256=plan.new_inbound_ball_sha256,
            dynamic_state_sha256=plan.new_ball_dynamic_state_sha256,
        )
        return RevealFinalInstallRow(
            integration_status=INTEGRATION_STATUS,
            phase=REVEAL_FINAL_PREVIEWED,
            public_visible=False,
            policy_opportunity_created=False,
            prepared_reveal=prepared,
            reveal_facts=facts,
            ball_slot_plan=plan,
            selected_task_ref_sha256=(
                prepared.selected_task_ref.canonical_sha256
            ),
            outcome_key_sha256=prepared.outcome_key.canonical_sha256,
            physical_ball_install_payload_sha256=(
                plan.physical_ball_install_payload_sha256
            ),
            pre_install_ball_slots=slots,
            post_install_ball_slots=tuple(post_install),
        )

    def _committed_row_from_preview(
        self, preview: RevealFinalInstallRow
    ) -> CommittedReveal:
        committed = CommittedReveal(
            integration_status=INTEGRATION_STATUS,
            phase=COMMITTED,
            runtime_wiring_connected=False,
            identity_committed=True,
            policy_opportunity_created=True,
            prepared_reveal=preview.prepared_reveal,
            reveal_facts=preview.reveal_facts,
            ball_slot_plan=preview.ball_slot_plan,
            playback_release_requested=preview.reveal_facts.ready_at_reveal,
        )
        self._validate_committed_integrity(committed)
        return committed

    def _preview_many_impl(
        self,
        batch: PreparedRevealBatch,
        facts: Tuple[ContinuousRevealFacts, ...],
        *,
        fault_injector: _FaultInjector,
    ) -> RevealFinalPreviewBatch:
        if len(facts) != len(batch.prepared_reveals):
            raise ContinuousRuntimeTransactionError(
                "preview_many reveal-fact width differs"
            )
        if tuple(getattr(row, "env_id", None) for row in facts) != (
            batch.selected_env_ids
        ):
            raise ContinuousRuntimeTransactionError(
                "preview_many reveal facts must follow selected env order"
            )
        live_owner = self._checkpoint_for_state(self._state)
        live_sampler = _checkpoint_from_json(
            self._state.sampler_checkpoint_json
        )
        live_rows = _sampler_env_rows_by_id(live_sampler)
        for prepared in batch.prepared_reveals:
            if (
                live_rows.get(prepared.request.env_id)
                != prepared.sampler_before_env_state
            ):
                raise TransactionConflictError(
                    "selected sampler row changed before reveal-final preview"
                )
        rows = []
        for prepared, reveal in zip(batch.prepared_reveals, facts):
            rows.append(self._preview_reveal_final_row(prepared, reveal))
            self._inject(fault_injector, "preview_many_after_row_validation")
        merged_sampler = _merge_sampler_env_rows(
            live_sampler, batch.prepared_reveals
        )
        restored = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, merged_sampler
        )
        merged_sampler = restored.checkpoint()
        env_ids = batch.selected_env_ids
        preview = RevealFinalPreviewBatch(
            integration_status=INTEGRATION_STATUS,
            phase=REVEAL_FINAL_PREVIEWED,
            public_visible=False,
            policy_opportunity_created=False,
            owner_checkpoint_before_sha256=live_owner["canonical_sha256"],
            prepared_batch=batch,
            sampler_checkpoint_before_commit_sha256=live_sampler[
                "checkpoint_sha256"
            ],
            sampler_checkpoint_after_commit_sha256=merged_sampler[
                "checkpoint_sha256"
            ],
            untouched_rows_before_sha256=_sampler_untouched_rows_root(
                live_sampler, env_ids
            ),
            untouched_rows_after_sha256=_sampler_untouched_rows_root(
                merged_sampler, env_ids
            ),
            sampler_checkpoint_before_commit=live_sampler,
            sampler_checkpoint_after_commit=merged_sampler,
            reveal_final_rows=tuple(rows),
            all_owner_install_root_sha256=_all_owner_install_root(rows),
        )
        self._inject(fault_injector, "preview_many_after_sampler_stage")
        self._inject(fault_injector, "preview_many_before_return")
        return preview

    def _validate_reveal_final_preview_integrity(
        self,
        preview: RevealFinalPreviewBatch,
        live_batch: PreparedRevealBatch,
    ) -> None:
        if type(preview) is not RevealFinalPreviewBatch:
            raise ContinuousRuntimeTransactionError(
                "commit_many requires an exact reveal-final preview token"
            )
        if preview.prepared_batch != live_batch:
            raise TransactionConflictError(
                "reveal-final token differs from the live prepared batch"
            )
        live_owner = self._checkpoint_for_state(self._state)
        live_sampler = _checkpoint_from_json(
            self._state.sampler_checkpoint_json
        )
        if (
            preview.owner_checkpoint_before_sha256
            != live_owner["canonical_sha256"]
            or preview.sampler_checkpoint_before_commit != live_sampler
        ):
            raise TransactionConflictError(
                "reveal-final token owner before-state is stale"
            )
        expected_rows = tuple(
            self._preview_reveal_final_row(
                prepared, preview_row.reveal_facts
            )
            for prepared, preview_row in zip(
                live_batch.prepared_reveals,
                preview.reveal_final_rows,
            )
        )
        if expected_rows != preview.reveal_final_rows:
            raise ContinuousRuntimeTransactionError(
                "reveal-final token final install facts differ"
            )
        expected_after = _merge_sampler_env_rows(
            live_sampler, live_batch.prepared_reveals
        )
        expected_after = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, expected_after
        ).checkpoint()
        if preview.sampler_checkpoint_after_commit != expected_after:
            raise ContinuousRuntimeTransactionError(
                "reveal-final token sampler after-state differs"
            )

    def _validate_committed_batch_integrity(
        self, batch: CommittedRevealBatch
    ) -> None:
        if not isinstance(batch, CommittedRevealBatch):
            raise ContinuousRuntimeTransactionError(
                "committed batch type differs"
            )
        self._validate_prepared_batch_integrity(batch.prepared_batch)
        for row in batch.committed_reveals:
            self._validate_committed_integrity(row)
        before_sampler = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, batch.sampler_checkpoint_before_commit
        )
        after_sampler = _target_sampler.ContinuousTargetSampler.from_checkpoint(
            self.profile, batch.sampler_checkpoint_after_commit
        )
        if before_sampler.seed != self.seed or after_sampler.seed != self.seed:
            raise ContinuousRuntimeTransactionError(
                "committed batch sampler seed differs"
            )
        before_rows = _sampler_env_rows_by_id(
            batch.sampler_checkpoint_before_commit
        )
        after_rows = _sampler_env_rows_by_id(
            batch.sampler_checkpoint_after_commit
        )
        for row in batch.prepared_batch.prepared_reveals:
            env_id = row.request.env_id
            if (
                before_rows.get(env_id) != row.sampler_before_env_state
                or after_rows.get(env_id) != row.sampler_after_env_state
            ):
                raise ContinuousRuntimeTransactionError(
                    "committed batch selected sampler row binding differs"
                )
        expected_after = _merge_sampler_env_rows(
            batch.sampler_checkpoint_before_commit,
            batch.prepared_batch.prepared_reveals,
        )
        if expected_after != batch.sampler_checkpoint_after_commit:
            raise ContinuousRuntimeTransactionError(
                "committed batch is not one full-owner state transition"
            )

    def _stage_commit_many_impl(
        self,
        preview: RevealFinalPreviewBatch,
        global_prearm_marker: Union[
            RevealTerminalBoundaryMarker, RevealPrepareBoundaryMarker
        ],
        *,
        fault_injector: _FaultInjector,
        legacy_fault_points: bool,
    ) -> tuple[CommittedRevealBatch, _OwnerState, dict[str, object]]:
        if (
            type(global_prearm_marker)
            not in (RevealTerminalBoundaryMarker, RevealPrepareBoundaryMarker)
            or global_prearm_marker.selected_env_ids
            != preview.selected_env_ids
            or global_prearm_marker.reveal_final_preview_sha256
            != preview.canonical_sha256
            or (
                type(global_prearm_marker)
                is RevealTerminalBoundaryMarker
                and global_prearm_marker.terminal_boundary_projection.decision
                != TERMINAL_DECISION_ACCEPT
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "commit staging global prearm marker differs"
            )
        batch = self._prepared_batch_by_sha(preview.prepared_batch_sha256)
        self._validate_reveal_final_preview_integrity(preview, batch)
        committed_rows = []
        for row in preview.reveal_final_rows:
            committed_rows.append(self._committed_row_from_preview(row))
            self._inject(fault_injector, "commit_many_after_row_validation")
        self._inject(fault_injector, "commit_many_after_validation")
        if legacy_fault_points:
            self._inject(fault_injector, "commit_after_validation")
        live_checkpoint = preview.sampler_checkpoint_before_commit
        merged_checkpoint = preview.sampler_checkpoint_after_commit
        env_ids = batch.selected_env_ids
        committed_batch = CommittedRevealBatch(
            integration_status=INTEGRATION_STATUS,
            phase=COMMITTED,
            runtime_wiring_connected=False,
            identity_committed=True,
            policy_opportunity_created=True,
            reveal_final_preview=preview,
            global_prearm_marker=global_prearm_marker,
            prepared_batch=batch,
            sampler_checkpoint_before_commit_sha256=live_checkpoint[
                "checkpoint_sha256"
            ],
            sampler_checkpoint_after_commit_sha256=merged_checkpoint[
                "checkpoint_sha256"
            ],
            untouched_rows_before_sha256=_sampler_untouched_rows_root(
                live_checkpoint, env_ids
            ),
            untouched_rows_after_sha256=_sampler_untouched_rows_root(
                merged_checkpoint, env_ids
            ),
            sampler_checkpoint_before_commit=live_checkpoint,
            sampler_checkpoint_after_commit=merged_checkpoint,
            committed_reveals=tuple(committed_rows),
        )
        self._validate_committed_batch_integrity(committed_batch)
        self._inject(fault_injector, "commit_many_after_sampler_stage")
        if legacy_fault_points:
            self._inject(fault_injector, "commit_after_sampler_stage")
        self._inject(fault_injector, "commit_many_before_publish")
        if legacy_fault_points:
            self._inject(fault_injector, "commit_before_publish")
        stored_batch = committed_batch
        stored_mapping = stored_batch.to_mapping()
        stored_batch_sha = _sha256(
            stored_mapping["canonical_sha256"],
            label="committed_reveal_batch_sha256",
        )
        result_batch = CommittedRevealBatch.from_mapping(stored_mapping)
        state = self._state
        prepared_mapping = stored_mapping["prepared_batch"]
        removed = {
            _sha256(
                row["canonical_sha256"],
                label="prepared_reveal_sha256",
            )
            for row in prepared_mapping["prepared_reveals"]
        }
        new_committed = tuple(
            sorted(
                (*state.committed, *stored_batch.committed_reveals),
                key=lambda row: (
                    row.prepared_reveal.request.env_id,
                    row.prepared_reveal.request.reset_generation,
                    row.prepared_reveal.request.scheduled_ordinal,
                ),
            )
        )
        row_shas = tuple(
            _sha256(
                row["canonical_sha256"],
                label="committed_reveal_sha256",
            )
            for row in stored_mapping["committed_reveals"]
        )
        high_waters = {
            row.env_id: row for row in state.reset_high_waters
        }
        consumed_pending_envs = set()
        for row, row_sha in zip(
            stored_batch.committed_reveals,
            row_shas,
        ):
            request = row.prepared_reveal.request
            prior = high_waters.get(request.env_id)
            pending = next(
                (
                    item
                    for item in state.pending_true_reset_q0
                    if item.env_id == request.env_id
                ),
                None,
            )
            if pending is not None:
                if (
                    request.scheduled_ordinal != 0
                    or request != pending.next_q0_request
                    or prior is None
                    or prior.pending_true_reset_q0_sha256
                    != pending.canonical_sha256
                    or prior.latest_reset_generation
                    != request.reset_generation
                ):
                    raise ContinuousRuntimeTransactionError(
                        "committed Q0 differs from reset high-water"
                    )
                consumed_pending_envs.add(request.env_id)
            elif prior is not None and (
                prior.active_committed_reveal_sha256 is None
                or prior.latest_reset_generation != request.reset_generation
            ):
                raise ContinuousRuntimeTransactionError(
                    "committed successor differs from active reset high-water"
                )
            high_waters[request.env_id] = ResetHighWater(
                env_id=request.env_id,
                latest_reset_generation=request.reset_generation,
                sampler_draw_high_water=request.sampler_generation,
                latest_retired_chain_sha256=(
                    None
                    if prior is None
                    else prior.latest_retired_chain_sha256
                ),
                active_sequence_event_kind="COMMITTED",
                active_sequence_event_sha256=row_sha,
                active_committed_reveal_sha256=row_sha,
                pending_true_reset_q0_sha256=None,
            )
        transition = OwnerTransitionRef(
            transition_kind="COMMIT_BATCH",
            transition_sha256=stored_batch_sha,
        )
        final_state = _OwnerState(
            sampler_checkpoint_json=_checkpoint_json(merged_checkpoint),
            prepared=tuple(
                row
                for row in state.prepared
                if row.canonical_sha256 not in removed
            ),
            prepared_batches=tuple(
                row
                for row in state.prepared_batches
                if row.canonical_sha256 != preview.prepared_batch_sha256
            ),
            prepared_event_batches=state.prepared_event_batches,
            abort_event_receipts=state.abort_event_receipts,
            committed=new_committed,
            committed_batches=(*state.committed_batches, stored_batch),
            committed_receipt_shas=(*state.committed_receipt_shas, *row_shas),
            committed_batch_shas=(
                *state.committed_batch_shas,
                stored_batch_sha,
            ),
            censored=state.censored,
            censored_batches=state.censored_batches,
            retired_chains=state.retired_chains,
            pending_true_reset_q0=tuple(
                row
                for row in state.pending_true_reset_q0
                if row.env_id not in consumed_pending_envs
            ),
            true_reset_batches=state.true_reset_batches,
            reset_high_waters=tuple(
                high_waters[env_id] for env_id in sorted(high_waters)
            ),
            owner_transitions=(*state.owner_transitions, transition),
        )
        return result_batch, final_state, stored_mapping

    def _stage_censor_many_impl(
        self,
        preview: RevealFinalPreviewBatch,
        censor_facts: Sequence[InfrastructureCensorFact],
        *,
        terminal_boundary_marker: RevealTerminalBoundaryMarker,
        fault_injector: _FaultInjector,
    ) -> tuple[CensoredRevealBatch, _OwnerState, dict[str, object]]:
        """Prebuild one exact CENSOR receipt/state without publishing either."""

        live_batch = self._prepared_batch_by_sha(
            preview.prepared_batch_sha256
        )
        self._validate_reveal_final_preview_integrity(preview, live_batch)
        facts = tuple(censor_facts)
        if (
            len(facts) != len(preview.reveal_final_rows)
            or any(type(row) is not InfrastructureCensorFact for row in facts)
            or tuple(row.env_id for row in facts)
            != preview.selected_env_ids
            or type(terminal_boundary_marker)
            is not RevealTerminalBoundaryMarker
            or terminal_boundary_marker.selected_env_ids
            != preview.selected_env_ids
            or terminal_boundary_marker.reveal_final_preview_sha256
            != preview.canonical_sha256
            or terminal_boundary_marker.terminal_boundary_projection.decision
            != TERMINAL_DECISION_CENSOR
        ):
            raise ContinuousRuntimeTransactionError(
                "terminal CENSOR facts/order/boundary authority differ"
            )
        censored_rows = []
        for preview_row, fact in zip(preview.reveal_final_rows, facts):
            censored_rows.append(
                CensoredReveal(
                    integration_status=INTEGRATION_STATUS,
                    phase=INFRA_CENSORED,
                    sequence_advanced=True,
                    sampler_consumed=True,
                    task_installed=False,
                    ball_installed=False,
                    policy_opportunity_created=False,
                    reveal_final_install_row=preview_row,
                    censor_fact=fact,
                )
            )
            self._inject(
                fault_injector, "terminal_censor_after_row_validation"
            )
        staged = CensoredRevealBatch(
            integration_status=INTEGRATION_STATUS,
            phase=INFRA_CENSORED,
            reveal_final_preview=preview,
            censored_reveals=tuple(censored_rows),
            sampler_checkpoint_before_sha256=(
                preview.sampler_checkpoint_before_commit_sha256
            ),
            sampler_checkpoint_after_sha256=(
                preview.sampler_checkpoint_after_commit_sha256
            ),
            untouched_rows_before_sha256=(
                preview.untouched_rows_before_sha256
            ),
            untouched_rows_after_sha256=(
                preview.untouched_rows_after_sha256
            ),
            sampler_checkpoint_before=(
                preview.sampler_checkpoint_before_commit
            ),
            sampler_checkpoint_after=(
                preview.sampler_checkpoint_after_commit
            ),
            task_install_count=0,
            ball_install_count=0,
            policy_opportunity_count=0,
            terminal_boundary_marker=terminal_boundary_marker,
        )
        self._inject(fault_injector, "terminal_censor_after_sampler_stage")
        stored = staged
        stored_mapping = stored.to_mapping()
        stored_sha = _sha256(
            stored_mapping["canonical_sha256"],
            label="censored_reveal_batch_sha256",
        )
        result = CensoredRevealBatch.from_mapping(stored_mapping)
        state = self._state
        preview_mapping = stored_mapping["reveal_final_preview"]
        prepared_mapping = preview_mapping["prepared_batch"]
        removed = {
            _sha256(
                row["canonical_sha256"],
                label="prepared_reveal_sha256",
            )
            for row in prepared_mapping["prepared_reveals"]
        }
        censored_row_shas = tuple(
            _sha256(
                row["canonical_sha256"],
                label="censored_reveal_sha256",
            )
            for row in stored_mapping["censored_reveals"]
        )
        high_waters = {row.env_id: row for row in state.reset_high_waters}
        consumed_pending_envs = set()
        for row, row_sha in zip(
            stored.censored_reveals,
            censored_row_shas,
        ):
            request = row.prepared_reveal.request
            prior = high_waters.get(request.env_id)
            pending = next(
                (
                    item
                    for item in state.pending_true_reset_q0
                    if item.env_id == request.env_id
                ),
                None,
            )
            if pending is not None:
                if (
                    request.scheduled_ordinal != 0
                    or request != pending.next_q0_request
                    or prior is None
                    or prior.pending_true_reset_q0_sha256
                    != pending.canonical_sha256
                ):
                    raise ContinuousRuntimeTransactionError(
                        "censored Q0 differs from reset high-water"
                    )
                consumed_pending_envs.add(request.env_id)
            elif prior is not None and (
                prior.active_sequence_event_sha256 is None
                or prior.latest_reset_generation != request.reset_generation
            ):
                raise ContinuousRuntimeTransactionError(
                    "censored successor differs from active reset high-water"
                )
            high_waters[request.env_id] = ResetHighWater(
                env_id=request.env_id,
                latest_reset_generation=request.reset_generation,
                sampler_draw_high_water=request.sampler_generation,
                latest_retired_chain_sha256=(
                    None
                    if prior is None
                    else prior.latest_retired_chain_sha256
                ),
                active_sequence_event_kind="INFRA_CENSORED",
                active_sequence_event_sha256=row_sha,
                active_committed_reveal_sha256=(
                    None
                    if prior is None
                    else prior.active_committed_reveal_sha256
                ),
                pending_true_reset_q0_sha256=None,
            )
        transition = OwnerTransitionRef(
            transition_kind="CENSOR_BATCH",
            transition_sha256=stored_sha,
        )
        final_state = _OwnerState(
            sampler_checkpoint_json=_checkpoint_json(
                stored.sampler_checkpoint_after
            ),
            prepared=tuple(
                row
                for row in state.prepared
                if row.canonical_sha256 not in removed
            ),
            prepared_batches=tuple(
                row
                for row in state.prepared_batches
                if row.canonical_sha256 != preview.prepared_batch_sha256
            ),
            prepared_event_batches=state.prepared_event_batches,
            abort_event_receipts=state.abort_event_receipts,
            committed=state.committed,
            committed_batches=state.committed_batches,
            committed_receipt_shas=state.committed_receipt_shas,
            committed_batch_shas=state.committed_batch_shas,
            censored=tuple(
                sorted(
                    (*state.censored, *stored.censored_reveals),
                    key=lambda row: (
                        row.prepared_reveal.request.env_id,
                        row.prepared_reveal.request.reset_generation,
                        row.prepared_reveal.request.scheduled_ordinal,
                    ),
                )
            ),
            censored_batches=(*state.censored_batches, stored),
            retired_chains=state.retired_chains,
            pending_true_reset_q0=tuple(
                row
                for row in state.pending_true_reset_q0
                if row.env_id not in consumed_pending_envs
            ),
            true_reset_batches=state.true_reset_batches,
            reset_high_waters=tuple(
                high_waters[env_id] for env_id in sorted(high_waters)
            ),
            owner_transitions=(*state.owner_transitions, transition),
        )
        self._inject(fault_injector, "terminal_censor_before_retain")
        return result, final_state, stored_mapping

    def preview_many(
        self,
        prepared_batch_sha256: str,
        reveal_facts: Sequence[ContinuousRevealFacts],
        *,
        fault_injector: _FaultInjector = None,
    ) -> RevealFinalPreviewBatch:
        digest = _sha256(
            prepared_batch_sha256, label="prepared_batch_sha256"
        )
        self._begin_operation()
        try:
            facts = tuple(reveal_facts)
            private_preview = self._preview_many_impl(
                self._prepared_batch_by_sha(digest),
                facts,
                fault_injector=fault_injector,
            )
            private_mapping = private_preview.to_mapping()
            private_token_json = _checkpoint_json(private_mapping)
            public_preview = RevealFinalPreviewBatch.from_mapping(
                _canonical_clone(private_mapping)
            )
            self._active_preview = _RevealFinalLease(
                private_token=private_preview,
                public_token=public_preview,
                private_token_json=private_token_json,
                preview_root_sha256=private_preview.canonical_sha256,
            )
            return public_preview
        finally:
            self._finish_operation()

    def abort_reveal_final_preview(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
    ) -> RevealFinalPreviewAbortReceipt:
        """Clear one exact unstaged preview before any boundary transfer.

        This is the pre-transfer retry path.  It is legal only while the exact
        public preview is still unstaged and unarmed; once a terminal claim
        exists the only legal failure response is global poison.  The external
        coordinator must likewise stop calling this method once its concrete
        boundary transfer begins.
        """

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            if (
                lease is None
                or type(reveal_final_token) is not RevealFinalPreviewBatch
                or reveal_final_token is not lease.public_token
                or lease.terminal_claim is not None
                or lease.terminal_claim_payload is not None
                or lease.armed_handle is not None
                or lease.armed_payload is not None
                or lease.committed_result is not None
                or lease.committed_state is not None
            ):
                raise TransactionConflictError(
                    "preview abort token is not the exact unstaged lease"
                )
            state_before = self._state
            private_preview = lease.private_token
            receipt = RevealFinalPreviewAbortReceipt(
                integration_status=INTEGRATION_STATUS,
                phase_before=REVEAL_FINAL_PREVIEWED,
                phase_after=PREPARED,
                reveal_final_preview_schema_version=(
                    RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
                ),
                reveal_final_preview_sha256=lease.preview_root_sha256,
                prepared_batch_sha256=private_preview.prepared_batch_sha256,
                selected_env_ids=private_preview.selected_env_ids,
                sampler_checkpoint_sha256=(
                    private_preview.sampler_checkpoint_before_commit_sha256
                ),
                owner_state_unchanged=True,
                policy_opportunity_created=False,
                terminal_claim_created=False,
            )
            receipt_sha = receipt.canonical_sha256
            _attach_reveal_final_preview_abort(
                receipt,
                _RevealFinalPreviewAbortPayload(
                    owner_id=id(self),
                    receipt_sha256=receipt_sha,
                    public_preview=reveal_final_token,
                    owner_state=state_before,
                ),
            )
            self._active_preview = None
            return receipt
        finally:
            self._finish_operation()

    def require_owned_active_reveal_final_preview(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
        *,
        expected_reveal_final_preview_sha256: str,
    ) -> RevealFinalPreviewBatch:
        """Return the retained private image for one exact active public token.

        The owner already retained an isolated private image and the canonical
        root when it minted this exact public object.  Child owners need the
        typed rows, not a second portable decoder.  The final boundary
        projection still has to bind the retained root before terminal stage,
        so this removes only redundant same-process serialization.
        """

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            expected = _sha256(
                expected_reveal_final_preview_sha256,
                label="expected_reveal_final_preview_sha256",
            )
            if lease is not None and expected != lease.preview_root_sha256:
                raise TransactionConflictError(
                    "reveal-final preview external SHA pin differs"
                )
            if (
                lease is None
                or type(reveal_final_token) is not RevealFinalPreviewBatch
                or reveal_final_token is not lease.public_token
                or lease.terminal_claim is not None
                or lease.terminal_claim_payload is not None
                or lease.armed_handle is not None
                or lease.armed_payload is not None
                or lease.committed_result is not None
                or lease.committed_state is not None
            ):
                raise TransactionConflictError(
                    "reveal-final preview is not the exact active unstaged lease"
                )
            return lease.private_token
        finally:
            self._finish_operation()

    def require_owned_reveal_final_preview_abort_receipt(
        self,
        receipt: RevealFinalPreviewAbortReceipt,
        *,
        expected_receipt_sha256: str,
        expected_reveal_final_preview_sha256: str,
        expected_prepared_batch_sha256: str,
        expected_selected_env_ids: Tuple[int, ...],
    ) -> RevealFinalPreviewAbortReceipt:
        """Repeatably validate one exact owner-issued pre-transfer abort."""

        self._begin_operation()
        try:
            payload = (
                None
                if type(receipt) is not RevealFinalPreviewAbortReceipt
                else _lookup_reveal_final_preview_abort(receipt)
            )
            if (
                payload is None
                or payload.owner_id != id(self)
                or type(expected_receipt_sha256) is not str
                or payload.receipt_sha256 != expected_receipt_sha256
                or type(expected_reveal_final_preview_sha256) is not str
                or receipt.reveal_final_preview_sha256
                != expected_reveal_final_preview_sha256
                or type(expected_prepared_batch_sha256) is not str
                or receipt.prepared_batch_sha256
                != expected_prepared_batch_sha256
                or type(expected_selected_env_ids) is not tuple
                or receipt.selected_env_ids != expected_selected_env_ids
            ):
                raise TransactionConflictError(
                    "preview abort receipt identity/root/selection differs"
                )
            return receipt
        finally:
            self._finish_operation()

    def stage_terminal_claim(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
        terminal_boundary_receipt: object,
        *,
        decision: str,
        fault_injector: _FaultInjector = None,
    ) -> PreparedRevealTerminalClaim:
        """Prebuild the exact future terminal receipt/state before child arm.

        The construction-bound validator closure owns the concrete downstream
        owner and performs its receipt-registry check.  This method therefore
        accepts no caller-selected owner, marker, projection, or digest as
        authority.  ``self._state`` remains byte-for-byte unchanged until the
        final terminal commit.
        """

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            if (
                lease is None
                or type(reveal_final_token) is not RevealFinalPreviewBatch
                or reveal_final_token is not lease.public_token
                or lease.armed_handle is not None
                or lease.terminal_claim is not None
            ):
                raise TransactionConflictError(
                    "terminal claim token is not the active unstaged preview lease"
                )
            clean_decision = _text(decision, label="decision")
            if clean_decision not in TERMINAL_DECISIONS:
                raise ContinuousRuntimeTransactionError(
                    "terminal claim decision differs"
                )
            authority = self._terminal_boundary_authority
            authority_payload = self._terminal_boundary_authority_payload
            if (
                authority is None
                or authority_payload is None
                or _lookup_terminal_boundary_authority(authority)
                is not authority_payload
                or authority_payload.owner_id != id(self)
                or authority_payload.status != "bound"
            ):
                raise TransactionConflictError(
                    "terminal boundary authority is not prebound"
                )
            try:
                projected = authority_payload.validator(
                    terminal_boundary_receipt
                )
            except Exception as exc:
                raise ContinuousRuntimeTransactionError(
                    "terminal boundary receipt is not exact prebound authority"
                ) from exc
            if (
                type(projected) is not TerminalBoundaryProjection
            ):
                raise ContinuousRuntimeTransactionError(
                    "terminal boundary validator returned an inexact projection"
                )
            projection = TerminalBoundaryProjection.from_mapping(
                projected.to_mapping()
            )
            if (
                projection.authority_domain
                != authority_payload.authority_domain
                or projection.authority_schema_sha256
                != authority_payload.authority_schema_sha256
                or projection.authority_source_sha256
                != authority_payload.authority_source_sha256
                or projection.reveal_final_preview_schema_version
                != RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
                or projection.reveal_final_preview_sha256
                != lease.preview_root_sha256
                or projection.selected_env_ids
                != lease.private_token.selected_env_ids
                or projection.decision != clean_decision
            ):
                raise ContinuousRuntimeTransactionError(
                    "terminal boundary projection domain/preview/decision differs"
                )
            censor_evidence = projection.ordered_censor_evidence
            private_preview = lease.private_token
            marker = RevealTerminalBoundaryMarker(
                terminal_boundary_authority_sha256=(
                    authority_payload.authority_sha256
                ),
                terminal_boundary_projection=projection,
            )
            fact_rows: Tuple[InfrastructureCensorFact, ...] = ()
            if clean_decision == TERMINAL_DECISION_CENSOR:
                derived_facts = []
                for preview_row, evidence in zip(
                    private_preview.reveal_final_rows,
                    censor_evidence,
                ):
                    fact = InfrastructureCensorFact(
                        env_id=evidence.env_id,
                        reveal_final_install_row_sha256=(
                            preview_row.canonical_sha256
                        ),
                        observed_at_step=(
                            preview_row.reveal_facts.reveal_step
                        ),
                        reason=evidence.reason,
                        failed_owner_kind=evidence.participant_kind,
                        failure_receipt_sha256=(
                            evidence.failure_receipt_sha256
                        ),
                        producer_schema_sha256=(
                            evidence.producer_schema_sha256
                        ),
                        producer_source_sha256=(
                            evidence.producer_source_sha256
                        ),
                    )
                    if fact.canonical_sha256 != evidence.censor_fact_sha256:
                        raise ContinuousRuntimeTransactionError(
                            "terminal CENSOR fact root differs from "
                            "owner-issued evidence"
                        )
                    derived_facts.append(fact)
                fact_rows = tuple(derived_facts)
            if clean_decision == TERMINAL_DECISION_ACCEPT:
                terminal_result, terminal_state, terminal_mapping = (
                    self._stage_commit_many_impl(
                        private_preview,
                        marker,
                        fault_injector=fault_injector,
                        legacy_fault_points=False,
                    )
                )
            else:
                terminal_result, terminal_state, terminal_mapping = (
                    self._stage_censor_many_impl(
                        private_preview,
                        fact_rows,
                        terminal_boundary_marker=marker,
                        fault_injector=fault_injector,
                    )
                )
            self._inject(fault_injector, "terminal_claim_after_terminal_stage")

            boundary_receipt_sha = _sha256(
                projection.boundary_receipt_sha256,
                label="global_boundary_receipt_sha256",
            )
            packet_sha = _sha256(
                projection.boundary_packet_sha256,
                label="global_boundary_packet_sha256",
            )
            terminal_kind = _text(
                terminal_result.KIND, label="terminal_kind"
            )
            terminal_sha = _sha256(
                terminal_mapping["canonical_sha256"],
                label="terminal_sha256",
            )
            terminal_bytes = _canonical_json_bytes(terminal_mapping)
            terminal_content_pin = PreparedTerminalContentPin(
                terminal_schema_version=terminal_mapping["schema_version"],
                terminal_kind=terminal_kind,
                terminal_canonical_sha256=terminal_sha,
                content_bytes_base64=base64.b64encode(
                    terminal_bytes
                ).decode("ascii"),
                content_byte_length=len(terminal_bytes),
                content_bytes_sha256=hashlib.sha256(
                    terminal_bytes
                ).hexdigest(),
            )
            projection_sha = projection.canonical_sha256
            content_pin_sha = terminal_content_pin.canonical_sha256
            selected = private_preview.selected_env_ids
            claim_sha = _terminal_claim_sha256(
                decision=clean_decision,
                selected_env_ids=selected,
                reveal_final_preview_schema_version=(
                    RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
                ),
                reveal_final_preview_sha256=lease.preview_root_sha256,
                global_boundary_receipt_kind=(
                    projection.boundary_receipt_kind
                ),
                global_boundary_receipt_sha256=boundary_receipt_sha,
                global_boundary_packet_schema_version=(
                    projection.boundary_packet_schema_version
                ),
                global_boundary_packet_sha256=packet_sha,
                terminal_boundary_authority_sha256=(
                    authority_payload.authority_sha256
                ),
                terminal_boundary_projection_sha256=(
                    projection_sha
                ),
                terminal_content_pin_sha256=(
                    content_pin_sha
                ),
                terminal_kind=terminal_kind,
                terminal_sha256=terminal_sha,
            )
            armed_payload = _ArmedRevealTerminalPayload(
                owner_id=id(self),
                claim_sha256=claim_sha,
            )
            armed_handle = _mint_armed_terminal(armed_payload)
            claim_payload = _PreparedRevealTerminalClaimPayload(
                owner_id=id(self),
                decision=clean_decision,
                selected_env_ids=selected,
                reveal_final_preview_schema_version=(
                    RevealFinalPreviewBatch.RECORD_SCHEMA_VERSION
                ),
                reveal_final_preview_sha256=lease.preview_root_sha256,
                global_boundary_receipt_kind=(
                    projection.boundary_receipt_kind
                ),
                global_boundary_receipt_sha256=boundary_receipt_sha,
                global_boundary_packet_schema_version=(
                    projection.boundary_packet_schema_version
                ),
                global_boundary_packet_sha256=packet_sha,
                terminal_kind=terminal_kind,
                terminal_sha256=terminal_sha,
                terminal_boundary_authority=authority,
                terminal_boundary_authority_sha256=(
                    authority_payload.authority_sha256
                ),
                terminal_boundary_projection=projection,
                terminal_boundary_projection_sha256=projection_sha,
                terminal_boundary_marker=marker,
                terminal_content_pin=terminal_content_pin,
                terminal_content_pin_sha256=content_pin_sha,
                claim_sha256=claim_sha,
                terminal_result=terminal_result,
                terminal_state=terminal_state,
                public_preview=lease.public_token,
                private_preview=lease.private_token,
                boundary_receipt=terminal_boundary_receipt,
                armed_handle=armed_handle,
                armed_payload=armed_payload,
            )
            claim = _mint_terminal_claim(claim_payload)
            self._inject(fault_injector, "terminal_claim_before_retain")
            lease.terminal_claim = claim
            lease.terminal_claim_payload = claim_payload
            return claim
        finally:
            self._finish_operation()

    def _require_owned_terminal_claim_payload(
        self,
        claim: PreparedRevealTerminalClaim,
        *,
        expected_status: str,
        expected_claim_sha256: str,
        expected_decision: str,
        expected_reveal_final_preview_sha256: str,
        expected_global_boundary_receipt_sha256: str,
        expected_global_boundary_packet_sha256: str,
        expected_terminal_boundary_authority_sha256: str,
        expected_terminal_boundary_projection_sha256: str,
        expected_terminal_content_pin_sha256: str,
        expected_terminal_kind: str,
        expected_terminal_sha256: str,
        expected_selected_env_ids: Tuple[int, ...],
    ) -> _PreparedRevealTerminalClaimPayload:
        if type(claim) is not PreparedRevealTerminalClaim:
            raise TransactionConflictError(
                "terminal claim type/identity differs"
            )
        payload = _lookup_terminal_claim(claim)
        lease = self._active_preview
        if (
            payload is None
            or payload.owner_id != id(self)
            or payload.status != expected_status
            or (
                expected_status in ("prepared", "armed")
                and (
                    lease is None
                    or lease.terminal_claim is not claim
                    or lease.terminal_claim_payload is not payload
                    or payload.public_preview is not lease.public_token
                    or payload.private_preview is not lease.private_token
                )
            )
            or (
                expected_status == "committed"
                and lease is not None
            )
            or payload.terminal_boundary_authority
            is not self._terminal_boundary_authority
            or _lookup_terminal_boundary_authority(
                payload.terminal_boundary_authority
            )
            is not self._terminal_boundary_authority_payload
            or payload.terminal_boundary_marker.terminal_boundary_projection
            is not payload.terminal_boundary_projection
            or payload.terminal_boundary_marker.terminal_boundary_authority_sha256
            != payload.terminal_boundary_authority_sha256
            or payload.terminal_boundary_projection.decision
            != payload.decision
            or payload.terminal_boundary_projection.selected_env_ids
            != payload.selected_env_ids
            or payload.terminal_boundary_projection.reveal_final_preview_sha256
            != payload.reveal_final_preview_sha256
            or payload.terminal_boundary_projection.boundary_receipt_sha256
            != payload.global_boundary_receipt_sha256
            or payload.terminal_boundary_projection.boundary_packet_sha256
            != payload.global_boundary_packet_sha256
            or payload.terminal_content_pin.terminal_kind
            != payload.terminal_kind
            or payload.terminal_content_pin.terminal_canonical_sha256
            != payload.terminal_sha256
        ):
            raise TransactionConflictError(
                "terminal claim is foreign, stale, copied, or in the wrong phase"
            )
        if (
            type(expected_selected_env_ids) is not tuple
            or type(expected_claim_sha256) is not str
            or payload.claim_sha256 != expected_claim_sha256
            or type(expected_decision) is not str
            or payload.decision != expected_decision
            or expected_decision not in TERMINAL_DECISIONS
            or type(expected_reveal_final_preview_sha256) is not str
            or payload.reveal_final_preview_sha256
            != expected_reveal_final_preview_sha256
            or type(expected_global_boundary_receipt_sha256) is not str
            or payload.global_boundary_receipt_sha256
            != expected_global_boundary_receipt_sha256
            or type(expected_global_boundary_packet_sha256) is not str
            or payload.global_boundary_packet_sha256
            != expected_global_boundary_packet_sha256
            or type(expected_terminal_boundary_authority_sha256) is not str
            or payload.terminal_boundary_authority_sha256
            != expected_terminal_boundary_authority_sha256
            or type(expected_terminal_boundary_projection_sha256) is not str
            or payload.terminal_boundary_projection_sha256
            != expected_terminal_boundary_projection_sha256
            or type(expected_terminal_content_pin_sha256) is not str
            or payload.terminal_content_pin_sha256
            != expected_terminal_content_pin_sha256
            or type(expected_terminal_kind) is not str
            or payload.terminal_kind
            != expected_terminal_kind
            or type(expected_terminal_sha256) is not str
            or payload.terminal_sha256
            != expected_terminal_sha256
            or payload.selected_env_ids != expected_selected_env_ids
        ):
            raise TransactionConflictError(
                "terminal claim decision/root/preview/boundary/selection differs"
            )
        return payload

    def require_owned_prepared_terminal_claim(
        self,
        claim: PreparedRevealTerminalClaim,
        *,
        expected_claim_sha256: str,
        expected_decision: str,
        expected_reveal_final_preview_sha256: str,
        expected_global_boundary_receipt_sha256: str,
        expected_global_boundary_packet_sha256: str,
        expected_terminal_boundary_authority_sha256: str,
        expected_terminal_boundary_projection_sha256: str,
        expected_terminal_content_pin_sha256: str,
        expected_terminal_kind: str,
        expected_terminal_sha256: str,
        expected_selected_env_ids: Tuple[int, ...],
    ) -> PreparedRevealTerminalClaim:
        """Return only this owner's exact still-prepared opaque claim."""

        self._begin_operation(allow_active_preview=True)
        try:
            self._require_owned_terminal_claim_payload(
                claim,
                expected_status="prepared",
                expected_claim_sha256=expected_claim_sha256,
                expected_decision=expected_decision,
                expected_reveal_final_preview_sha256=(
                    expected_reveal_final_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=(
                    expected_global_boundary_receipt_sha256
                ),
                expected_global_boundary_packet_sha256=(
                    expected_global_boundary_packet_sha256
                ),
                expected_terminal_boundary_authority_sha256=(
                    expected_terminal_boundary_authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    expected_terminal_boundary_projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    expected_terminal_content_pin_sha256
                ),
                expected_terminal_kind=expected_terminal_kind,
                expected_terminal_sha256=expected_terminal_sha256,
                expected_selected_env_ids=expected_selected_env_ids,
            )
            return claim
        finally:
            self._finish_operation()

    def require_owned_armed_terminal_claim(
        self,
        claim: PreparedRevealTerminalClaim,
        *,
        expected_claim_sha256: str,
        expected_decision: str,
        expected_reveal_final_preview_sha256: str,
        expected_global_boundary_receipt_sha256: str,
        expected_global_boundary_packet_sha256: str,
        expected_terminal_boundary_authority_sha256: str,
        expected_terminal_boundary_projection_sha256: str,
        expected_terminal_content_pin_sha256: str,
        expected_terminal_kind: str,
        expected_terminal_sha256: str,
        expected_selected_env_ids: Tuple[int, ...],
    ) -> PreparedRevealTerminalClaim:
        """Read-only coordinator audit after the last fallible R05 arm gate."""

        self._begin_operation(allow_active_preview=True)
        try:
            self._require_owned_terminal_claim_payload(
                claim,
                expected_status="armed",
                expected_claim_sha256=expected_claim_sha256,
                expected_decision=expected_decision,
                expected_reveal_final_preview_sha256=(
                    expected_reveal_final_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=(
                    expected_global_boundary_receipt_sha256
                ),
                expected_global_boundary_packet_sha256=(
                    expected_global_boundary_packet_sha256
                ),
                expected_terminal_boundary_authority_sha256=(
                    expected_terminal_boundary_authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    expected_terminal_boundary_projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    expected_terminal_content_pin_sha256
                ),
                expected_terminal_kind=expected_terminal_kind,
                expected_terminal_sha256=expected_terminal_sha256,
                expected_selected_env_ids=expected_selected_env_ids,
            )
            return claim
        finally:
            self._finish_operation()

    def arm_terminal_claim(
        self,
        claim: PreparedRevealTerminalClaim,
        *,
        expected_claim_sha256: str,
        expected_decision: str,
        expected_reveal_final_preview_sha256: str,
        expected_global_boundary_receipt_sha256: str,
        expected_global_boundary_packet_sha256: str,
        expected_terminal_boundary_authority_sha256: str,
        expected_terminal_boundary_projection_sha256: str,
        expected_terminal_content_pin_sha256: str,
        expected_terminal_kind: str,
        expected_terminal_sha256: str,
        expected_selected_env_ids: Tuple[int, ...],
    ) -> ArmedRevealTerminalHandle:
        """Last fallible gate; return the handle preallocated during staging."""

        self._begin_operation(allow_active_preview=True)
        try:
            payload = self._require_owned_terminal_claim_payload(
                claim,
                expected_status="prepared",
                expected_claim_sha256=expected_claim_sha256,
                expected_decision=expected_decision,
                expected_reveal_final_preview_sha256=(
                    expected_reveal_final_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=(
                    expected_global_boundary_receipt_sha256
                ),
                expected_global_boundary_packet_sha256=(
                    expected_global_boundary_packet_sha256
                ),
                expected_terminal_boundary_authority_sha256=(
                    expected_terminal_boundary_authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    expected_terminal_boundary_projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    expected_terminal_content_pin_sha256
                ),
                expected_terminal_kind=expected_terminal_kind,
                expected_terminal_sha256=expected_terminal_sha256,
                expected_selected_env_ids=expected_selected_env_ids,
            )
            armed_payload = _lookup_armed_terminal(payload.armed_handle)
            if (
                armed_payload is not payload.armed_payload
                or armed_payload.owner_id != id(self)
                or armed_payload.claim_sha256 != payload.claim_sha256
                or armed_payload.status != "prepared"
            ):
                raise TransactionConflictError(
                    "terminal armed handle registry differs"
                )
            payload.status = "armed"
            armed_payload.status = "armed"
            return payload.armed_handle
        finally:
            self._finish_operation()

    def commit_terminal_prevalidated(
        self,
        armed_handle: ArmedRevealTerminalHandle,
    ) -> Union[CommittedRevealBatch, CensoredRevealBatch]:
        """Publish the exact prebuilt terminal state with no decode/allocation."""

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            claim = None if lease is None else lease.terminal_claim
            payload = (
                None if lease is None else lease.terminal_claim_payload
            )
            armed_payload = (
                None
                if type(armed_handle) is not ArmedRevealTerminalHandle
                else _lookup_armed_terminal(armed_handle)
            )
            if (
                claim is None
                or payload is None
                or _lookup_terminal_claim(claim) is not payload
                or payload.owner_id != id(self)
                or payload.status != "armed"
                or armed_handle is not payload.armed_handle
                or armed_payload is not payload.armed_payload
                or armed_payload.status != "armed"
                or armed_payload.claim_sha256 != payload.claim_sha256
                or (
                    payload.decision == TERMINAL_DECISION_ACCEPT
                    and type(payload.terminal_result)
                    is not CommittedRevealBatch
                )
                or (
                    payload.decision == TERMINAL_DECISION_CENSOR
                    and type(payload.terminal_result)
                    is not CensoredRevealBatch
                )
            ):
                raise TransactionConflictError(
                    "terminal armed handle is not the active prevalidated lease"
                )
            result = payload.terminal_result
            next_state = payload.terminal_state
            payload.status = "committed"
            armed_payload.status = "committed"
            self._active_preview = None
            self._state = next_state
            return result
        finally:
            self._finish_operation()

    def require_owned_terminal_receipt(
        self,
        claim: PreparedRevealTerminalClaim,
        terminal_receipt: Union[CommittedRevealBatch, CensoredRevealBatch],
        *,
        expected_claim_sha256: str,
        expected_decision: str,
        expected_reveal_final_preview_sha256: str,
        expected_global_boundary_receipt_sha256: str,
        expected_global_boundary_packet_sha256: str,
        expected_terminal_boundary_authority_sha256: str,
        expected_terminal_boundary_projection_sha256: str,
        expected_terminal_content_pin_sha256: str,
        expected_terminal_kind: str,
        expected_terminal_sha256: str,
        expected_selected_env_ids: Tuple[int, ...],
    ) -> Union[CommittedRevealBatch, CensoredRevealBatch]:
        """Repeatable post-publication exact-identity validator for child acks."""

        self._begin_operation()
        try:
            payload = self._require_owned_terminal_claim_payload(
                claim,
                expected_status="committed",
                expected_claim_sha256=expected_claim_sha256,
                expected_decision=expected_decision,
                expected_reveal_final_preview_sha256=(
                    expected_reveal_final_preview_sha256
                ),
                expected_global_boundary_receipt_sha256=(
                    expected_global_boundary_receipt_sha256
                ),
                expected_global_boundary_packet_sha256=(
                    expected_global_boundary_packet_sha256
                ),
                expected_terminal_boundary_authority_sha256=(
                    expected_terminal_boundary_authority_sha256
                ),
                expected_terminal_boundary_projection_sha256=(
                    expected_terminal_boundary_projection_sha256
                ),
                expected_terminal_content_pin_sha256=(
                    expected_terminal_content_pin_sha256
                ),
                expected_terminal_kind=expected_terminal_kind,
                expected_terminal_sha256=expected_terminal_sha256,
                expected_selected_env_ids=expected_selected_env_ids,
            )
            expected_type = (
                CommittedRevealBatch
                if payload.decision == TERMINAL_DECISION_ACCEPT
                else CensoredRevealBatch
            )
            if (
                type(terminal_receipt) is not expected_type
                or terminal_receipt is not payload.terminal_result
            ):
                raise TransactionConflictError(
                    "terminal receipt identity/root/kind/preview differs from claim"
                )
            return terminal_receipt
        finally:
            self._finish_operation()

    def arm_preview_for_all_owner(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
        global_prearm_marker: RevealPrepareBoundaryMarker,
        *,
        expected_preview_root_sha256: str,
        expected_global_prearm_marker_sha256: str,
    ) -> ArmedRevealFinalHandle:
        """Tombstone the caller-marker production ACCEPT arm."""

        del (
            reveal_final_token,
            global_prearm_marker,
            expected_preview_root_sha256,
            expected_global_prearm_marker_sha256,
        )
        raise TransactionConflictError(
            "arm_preview_for_all_owner is a production tombstone; use "
            "stage_terminal_claim/arm_terminal_claim"
        )

    def _arm_preview_for_all_owner_compatibility_pure(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
        global_prearm_marker: RevealPrepareBoundaryMarker,
        *,
        expected_preview_root_sha256: str,
        expected_global_prearm_marker_sha256: str,
    ) -> ArmedRevealFinalHandle:
        """Cross-check the public token before any external owner may mutate."""

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            if (
                lease is None
                or type(reveal_final_token) is not RevealFinalPreviewBatch
                or reveal_final_token is not lease.public_token
                or lease.armed_handle is not None
                or lease.terminal_claim is not None
                or type(global_prearm_marker)
                is not RevealPrepareBoundaryMarker
            ):
                raise TransactionConflictError(
                    "arm token is not the unarmed reveal-final lease"
                )
            expected = _sha256(
                expected_preview_root_sha256,
                label="expected_preview_root_sha256",
            )
            expected_marker = _sha256(
                expected_global_prearm_marker_sha256,
                label="expected_global_prearm_marker_sha256",
            )
            try:
                public_mapping = reveal_final_token.to_mapping()
                public_json = _checkpoint_json(public_mapping)
            except (TypeError, ValueError) as exc:
                raise ContinuousRuntimeTransactionError(
                    "arm token is not finite canonical JSON"
                ) from exc
            if (
                expected != lease.preview_root_sha256
                or public_json != lease.private_token_json
                or public_mapping != lease.private_token.to_mapping()
                or reveal_final_token.canonical_sha256
                != lease.preview_root_sha256
            ):
                raise TransactionConflictError(
                    "arm token differs from the owner-retained private preview"
                )
            try:
                marker_mapping = global_prearm_marker.to_mapping()
                marker_json = _checkpoint_json(marker_mapping)
            except (TypeError, ValueError) as exc:
                raise ContinuousRuntimeTransactionError(
                    "global prearm marker is not finite canonical JSON"
                ) from exc
            if (
                global_prearm_marker.canonical_sha256 != expected_marker
                or global_prearm_marker.selected_env_ids
                != lease.private_token.selected_env_ids
                or global_prearm_marker.reveal_final_preview_sha256
                != lease.preview_root_sha256
            ):
                raise TransactionConflictError(
                    "global prearm marker differs from the active preview"
                )
            (
                committed_result,
                committed_state,
                _committed_mapping,
            ) = self._stage_commit_many_impl(
                lease.private_token,
                global_prearm_marker,
                fault_injector=None,
                legacy_fault_points=False,
            )
            handle = ArmedRevealFinalHandle()
            payload = _ArmedPreviewPayload(
                owner_id=id(self),
                preview_root_sha256=lease.preview_root_sha256,
                global_prearm_marker_sha256=expected_marker,
            )
            _attach_armed_preview(handle, payload)
            lease.committed_result = committed_result
            lease.committed_state = committed_state
            lease.global_prearm_marker_json = marker_json
            lease.global_prearm_marker_sha256 = expected_marker
            lease.armed_handle = handle
            lease.armed_payload = payload
            return handle
        finally:
            self._finish_operation()

    def commit_prevalidated(
        self,
        armed_handle: ArmedRevealFinalHandle,
    ) -> CommittedRevealBatch:
        """Tombstone the claim-free production ACCEPT publication."""

        del armed_handle
        raise TransactionConflictError(
            "commit_prevalidated is a production tombstone; use "
            "commit_terminal_prevalidated"
        )

    def _commit_prevalidated_compatibility_pure(
        self,
        armed_handle: ArmedRevealFinalHandle,
    ) -> CommittedRevealBatch:
        """Publish the prebuilt state behind one exact armed lease.

        The valid path intentionally performs no caller-token decoding, no
        receipt construction, no fault injection, and no new allocation.
        """

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            if (
                lease is None
                or type(armed_handle) is not ArmedRevealFinalHandle
                or armed_handle is not lease.armed_handle
                or lease.armed_payload is None
                or lease.armed_payload.status != "armed"
                or lease.committed_result is None
                or lease.committed_state is None
            ):
                raise TransactionConflictError(
                    "armed handle is not the active reveal-final lease"
                )
            result = lease.committed_result
            next_state = lease.committed_state
            lease.armed_payload.status = "committed"
            self._state = next_state
            self._active_preview = None
            return result
        finally:
            self._finish_operation()

    def commit_many(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
    ) -> CommittedRevealBatch:
        """Tombstone the pre-arm commit path."""

        del reveal_final_token
        raise TransactionConflictError(
            "commit_many is a production tombstone; use "
            "stage_terminal_claim/arm_terminal_claim/"
            "commit_terminal_prevalidated"
        )

    def censor_many(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
        censor_facts: Sequence[InfrastructureCensorFact],
        *,
        expected_censor_fact_sha256s: Sequence[str],
        expected_censor_producer_schema_sha256s: Sequence[str],
        expected_censor_producer_source_sha256s: Sequence[str],
        fault_injector: _FaultInjector = None,
    ) -> CensoredRevealBatch:
        """Tombstone the production-unsafe one-step CENSOR transition."""

        del (
            reveal_final_token,
            censor_facts,
            expected_censor_fact_sha256s,
            expected_censor_producer_schema_sha256s,
            expected_censor_producer_source_sha256s,
            fault_injector,
        )
        raise TransactionConflictError(
            "censor_many is a production tombstone; use "
            "stage_terminal_claim/arm_terminal_claim/"
            "commit_terminal_prevalidated"
        )

    def _censor_many_compatibility_pure(
        self,
        reveal_final_token: RevealFinalPreviewBatch,
        censor_facts: Sequence[InfrastructureCensorFact],
        *,
        expected_censor_fact_sha256s: Sequence[str],
        expected_censor_producer_schema_sha256s: Sequence[str],
        expected_censor_producer_source_sha256s: Sequence[str],
        fault_injector: _FaultInjector = None,
    ) -> CensoredRevealBatch:
        """Legacy pure-owner fixture path; fresh coordinators must not call it."""

        self._begin_operation(allow_active_preview=True)
        try:
            lease = self._active_preview
            if (
                lease is None
                or type(reveal_final_token) is not RevealFinalPreviewBatch
                or reveal_final_token is not lease.public_token
                or lease.armed_handle is not None
                or lease.terminal_claim is not None
            ):
                raise TransactionConflictError(
                    "compatibility CENSOR token is not the active unarmed preview lease"
                )
            try:
                public_mapping = reveal_final_token.to_mapping()
                public_json = _checkpoint_json(public_mapping)
            except (TypeError, ValueError) as exc:
                raise ContinuousRuntimeTransactionError(
                    "censor_many token is not finite canonical JSON"
                ) from exc
            if (
                public_json != lease.private_token_json
                or public_mapping != lease.private_token.to_mapping()
                or reveal_final_token.canonical_sha256
                != lease.preview_root_sha256
            ):
                raise TransactionConflictError(
                    "censor_many token differs from the owner-retained preview"
                )
            preview = lease.private_token
            live_batch = preview.prepared_batch
            facts = tuple(censor_facts)
            expected_fact_shas = tuple(
                _sha256(value, label="expected_censor_fact_sha256s")
                for value in expected_censor_fact_sha256s
            )
            expected_schema_shas = tuple(
                _sha256(
                    value,
                    label="expected_censor_producer_schema_sha256s",
                )
                for value in expected_censor_producer_schema_sha256s
            )
            expected_source_shas = tuple(
                _sha256(
                    value,
                    label="expected_censor_producer_source_sha256s",
                )
                for value in expected_censor_producer_source_sha256s
            )
            if (
                len(facts) != len(preview.reveal_final_rows)
                or any(type(row) is not InfrastructureCensorFact for row in facts)
                or any(
                    row.failed_owner_kind not in _PREARM_CHILD_OWNER_KINDS
                    for row in facts
                )
                or tuple(row.env_id for row in facts)
                != preview.selected_env_ids
                or len(expected_fact_shas) != len(facts)
                or len(expected_schema_shas) != len(facts)
                or len(expected_source_shas) != len(facts)
                or tuple(row.canonical_sha256 for row in facts)
                != expected_fact_shas
                or tuple(row.producer_schema_sha256 for row in facts)
                != expected_schema_shas
                or tuple(row.producer_source_sha256 for row in facts)
                != expected_source_shas
            ):
                raise ContinuousRuntimeTransactionError(
                    "censor_many facts/order/external pins differ"
                )
            censored_rows = []
            for preview_row, fact in zip(preview.reveal_final_rows, facts):
                censored_rows.append(
                    CensoredReveal(
                        integration_status=INTEGRATION_STATUS,
                        phase=INFRA_CENSORED,
                        sequence_advanced=True,
                        sampler_consumed=True,
                        task_installed=False,
                        ball_installed=False,
                        policy_opportunity_created=False,
                        reveal_final_install_row=preview_row,
                        censor_fact=fact,
                    )
                )
                self._inject(fault_injector, "censor_many_after_row_validation")
            staged = CensoredRevealBatch(
                integration_status=INTEGRATION_STATUS,
                phase=INFRA_CENSORED,
                reveal_final_preview=preview,
                censored_reveals=tuple(censored_rows),
                sampler_checkpoint_before_sha256=(
                    preview.sampler_checkpoint_before_commit_sha256
                ),
                sampler_checkpoint_after_sha256=(
                    preview.sampler_checkpoint_after_commit_sha256
                ),
                untouched_rows_before_sha256=(
                    preview.untouched_rows_before_sha256
                ),
                untouched_rows_after_sha256=(
                    preview.untouched_rows_after_sha256
                ),
                sampler_checkpoint_before=(
                    preview.sampler_checkpoint_before_commit
                ),
                sampler_checkpoint_after=(
                    preview.sampler_checkpoint_after_commit
                ),
                task_install_count=0,
                ball_install_count=0,
                policy_opportunity_count=0,
            )
            self._inject(fault_injector, "censor_many_after_sampler_stage")
            stored = CensoredRevealBatch.from_mapping(staged.to_mapping())
            result = CensoredRevealBatch.from_mapping(stored.to_mapping())
            state = self._state
            removed = {
                row.canonical_sha256 for row in live_batch.prepared_reveals
            }
            high_waters = {row.env_id: row for row in state.reset_high_waters}
            consumed_pending_envs = set()
            for row in stored.censored_reveals:
                request = row.prepared_reveal.request
                prior = high_waters.get(request.env_id)
                pending = next(
                    (
                        item
                        for item in state.pending_true_reset_q0
                        if item.env_id == request.env_id
                    ),
                    None,
                )
                if pending is not None:
                    if (
                        request.scheduled_ordinal != 0
                        or request != pending.next_q0_request
                        or prior is None
                        or prior.pending_true_reset_q0_sha256
                        != pending.canonical_sha256
                    ):
                        raise ContinuousRuntimeTransactionError(
                            "censored Q0 differs from reset high-water"
                        )
                    consumed_pending_envs.add(request.env_id)
                elif prior is not None and (
                    prior.active_sequence_event_sha256 is None
                    or prior.latest_reset_generation != request.reset_generation
                ):
                    raise ContinuousRuntimeTransactionError(
                        "censored successor differs from active reset high-water"
                    )
                high_waters[request.env_id] = ResetHighWater(
                    env_id=request.env_id,
                    latest_reset_generation=request.reset_generation,
                    sampler_draw_high_water=request.sampler_generation,
                    latest_retired_chain_sha256=(
                        None
                        if prior is None
                        else prior.latest_retired_chain_sha256
                    ),
                    active_sequence_event_kind="INFRA_CENSORED",
                    active_sequence_event_sha256=row.canonical_sha256,
                    active_committed_reveal_sha256=(
                        None
                        if prior is None
                        else prior.active_committed_reveal_sha256
                    ),
                    pending_true_reset_q0_sha256=None,
                )
            transition = OwnerTransitionRef(
                transition_kind="CENSOR_BATCH",
                transition_sha256=stored.canonical_sha256,
            )
            final_state = _OwnerState(
                sampler_checkpoint_json=_checkpoint_json(
                    stored.sampler_checkpoint_after
                ),
                prepared=tuple(
                    row
                    for row in state.prepared
                    if row.canonical_sha256 not in removed
                ),
                prepared_batches=tuple(
                    row
                    for row in state.prepared_batches
                    if row.canonical_sha256 != live_batch.canonical_sha256
                ),
                prepared_event_batches=state.prepared_event_batches,
                abort_event_receipts=state.abort_event_receipts,
                committed=state.committed,
                committed_batches=state.committed_batches,
                committed_receipt_shas=state.committed_receipt_shas,
                committed_batch_shas=state.committed_batch_shas,
                censored=tuple(
                    sorted(
                        (*state.censored, *stored.censored_reveals),
                        key=lambda row: (
                            row.prepared_reveal.request.env_id,
                            row.prepared_reveal.request.reset_generation,
                            row.prepared_reveal.request.scheduled_ordinal,
                        ),
                    )
                ),
                censored_batches=(*state.censored_batches, stored),
                retired_chains=state.retired_chains,
                pending_true_reset_q0=tuple(
                    row
                    for row in state.pending_true_reset_q0
                    if row.env_id not in consumed_pending_envs
                ),
                true_reset_batches=state.true_reset_batches,
                reset_high_waters=tuple(
                    high_waters[env_id] for env_id in sorted(high_waters)
                ),
                owner_transitions=(*state.owner_transitions, transition),
            )
            self._inject(fault_injector, "censor_many_before_publish")
            self._state = final_state
            self._active_preview = None
            return result
        finally:
            self._finish_operation()

    def commit(
        self,
        prepared_reveal_sha256: str,
        reveal_facts: ContinuousRevealFacts,
        *,
        fault_injector: _FaultInjector = None,
    ) -> CommittedReveal:
        """Tombstone the unsafe pre-arm K=1 commit path."""

        del prepared_reveal_sha256, reveal_facts, fault_injector
        raise TransactionConflictError(
            "commit requires the K-batch preview/arm/commit_prevalidated path"
        )

    def checkpoint(self) -> dict[str, object]:
        """Return exact pure-owner state; this is not a runtime R10 checkpoint."""

        self._require_no_active_preview()
        return self._checkpoint_for_state(self._state)

    def _checkpoint_for_state(
        self, state: _OwnerState
    ) -> dict[str, object]:
        sampler_checkpoint = _checkpoint_from_json(
            state.sampler_checkpoint_json
        )
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "integration_status": INTEGRATION_STATUS,
            "runtime_wiring_connected": False,
            "profile_sha256": self.profile.profile_sha256,
            "runtime_dtype": self.profile.runtime_dtype,
            "quantization_contract": self.profile.quantization_contract,
            "seed": self.seed,
            "ball_slot_capacity": self.ball_slot_capacity,
            "sampler_checkpoint": sampler_checkpoint,
            "prepared": [row.to_mapping() for row in state.prepared],
            "prepared_batches": [
                row.to_mapping() for row in state.prepared_batches
            ],
            "prepared_event_batches": [
                row.to_mapping() for row in state.prepared_event_batches
            ],
            "abort_event_receipts": [
                row.to_mapping() for row in state.abort_event_receipts
            ],
            "committed": [row.to_mapping() for row in state.committed],
            "committed_batches": [
                row.to_mapping() for row in state.committed_batches
            ],
            "committed_receipt_shas": list(state.committed_receipt_shas),
            "committed_transcript_sha256": _transcript_root(
                state.committed_receipt_shas
            ),
            "committed_batch_shas": list(state.committed_batch_shas),
            "committed_batch_transcript_sha256": _batch_transcript_root(
                state.committed_batch_shas
            ),
            "censored": [row.to_mapping() for row in state.censored],
            "censored_batches": [
                row.to_mapping() for row in state.censored_batches
            ],
            "retired_chains": [
                row.to_mapping() for row in state.retired_chains
            ],
            "pending_true_reset_q0": [
                row.to_mapping() for row in state.pending_true_reset_q0
            ],
            "true_reset_batches": [
                row.to_mapping() for row in state.true_reset_batches
            ],
            "reset_high_waters": [
                row.to_mapping() for row in state.reset_high_waters
            ],
            "owner_transitions": [
                row.to_mapping() for row in state.owner_transitions
            ],
            "owner_transition_root_sha256": _owner_transition_root(
                state.owner_transitions
            ),
        }
        return _sealed(payload)

    @classmethod
    def from_checkpoint(
        cls,
        profile: _target_sampler.ContinuousTargetProfile,
        value: object,
        *,
        expected_checkpoint_sha256: object,
    ) -> "ContinuousRuntimeTransactionOwner":
        """Restore only from an externally pinned checkpoint digest."""

        expected_sha = _sha256(
            expected_checkpoint_sha256,
            label="expected_checkpoint_sha256",
        )
        if not isinstance(value, Mapping):
            raise ContinuousRuntimeTransactionError("checkpoint must be a mapping")
        if (
            (
                type(value.get("schema_version")) is int
                and value.get("schema_version") in (1, 2, 3, 4)
            )
            or value.get("kind")
            in (
                LEGACY_CHECKPOINT_KIND,
                LEGACY_V2_CHECKPOINT_KIND,
                LEGACY_V3_CHECKPOINT_KIND,
                LEGACY_V4_CHECKPOINT_KIND,
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "legacy v1/v2/v3/v4 runtime transaction checkpoint is tombstoned"
            )
        keys = frozenset(
            (
                "schema_version",
                "kind",
                "integration_status",
                "runtime_wiring_connected",
                "profile_sha256",
                "runtime_dtype",
                "quantization_contract",
                "seed",
                "ball_slot_capacity",
                "sampler_checkpoint",
                "prepared",
                "prepared_batches",
                "prepared_event_batches",
                "abort_event_receipts",
                "committed",
                "committed_batches",
                "committed_receipt_shas",
                "committed_transcript_sha256",
                "committed_batch_shas",
                "committed_batch_transcript_sha256",
                "censored",
                "censored_batches",
                "retired_chains",
                "pending_true_reset_q0",
                "true_reset_batches",
                "reset_high_waters",
                "owner_transitions",
                "owner_transition_root_sha256",
                "canonical_sha256",
            )
        )
        if frozenset(value) != keys:
            raise ContinuousRuntimeTransactionError("checkpoint keys differ")
        declared = _sha256(value["canonical_sha256"], label="canonical_sha256")
        if declared != expected_sha:
            raise ContinuousRuntimeTransactionError(
                "checkpoint differs from externally pinned SHA"
            )
        payload = {key: value[key] for key in keys if key != "canonical_sha256"}
        if canonical_sha256(payload) != declared:
            raise ContinuousRuntimeTransactionError("checkpoint canonical SHA differs")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
            or type(payload["kind"]) is not str
            or payload["kind"] != CHECKPOINT_KIND
            or payload["integration_status"] != INTEGRATION_STATUS
            or payload["runtime_wiring_connected"] is not False
            or payload["profile_sha256"] != profile.profile_sha256
            or type(payload["runtime_dtype"]) is not str
            or payload["runtime_dtype"] != profile.runtime_dtype
            or type(payload["quantization_contract"]) is not str
            or payload["quantization_contract"]
            != profile.quantization_contract
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint immutable identity differs"
            )
        owner = cls(
            profile,
            seed=payload["seed"],
            ball_slot_capacity=payload["ball_slot_capacity"],
            sampler_checkpoint=payload["sampler_checkpoint"],
        )
        raw_prepared = payload["prepared"]
        raw_prepared_batches = payload["prepared_batches"]
        raw_prepared_event_batches = payload["prepared_event_batches"]
        raw_abort_event_receipts = payload["abort_event_receipts"]
        raw_committed = payload["committed"]
        raw_committed_batches = payload["committed_batches"]
        raw_history = payload["committed_receipt_shas"]
        raw_batch_history = payload["committed_batch_shas"]
        raw_censored = payload["censored"]
        raw_censored_batches = payload["censored_batches"]
        raw_retired_chains = payload["retired_chains"]
        raw_pending_q0 = payload["pending_true_reset_q0"]
        raw_true_reset_batches = payload["true_reset_batches"]
        raw_reset_high_waters = payload["reset_high_waters"]
        raw_owner_transitions = payload["owner_transitions"]
        if not all(
            isinstance(rows, list)
            for rows in (
                raw_prepared,
                raw_prepared_batches,
                raw_prepared_event_batches,
                raw_abort_event_receipts,
                raw_committed,
                raw_committed_batches,
                raw_censored,
                raw_censored_batches,
                raw_retired_chains,
                raw_pending_q0,
                raw_true_reset_batches,
                raw_reset_high_waters,
                raw_owner_transitions,
            )
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint transaction rows must be lists"
            )
        if not isinstance(raw_history, list) or not isinstance(
            raw_batch_history, list
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint committed histories must be lists"
            )
        prepared = tuple(PreparedReveal.from_mapping(row) for row in raw_prepared)
        prepared_batches = tuple(
            PreparedRevealBatch.from_mapping(row)
            for row in raw_prepared_batches
        )
        prepared_event_batches = tuple(
            PreparedRevealBatch.from_mapping(row)
            for row in raw_prepared_event_batches
        )
        abort_event_receipts = tuple(
            AbortBatchReceipt.from_mapping(row)
            for row in raw_abort_event_receipts
        )
        committed = tuple(CommittedReveal.from_mapping(row) for row in raw_committed)
        committed_batches = tuple(
            CommittedRevealBatch.from_mapping(row)
            for row in raw_committed_batches
        )
        censored = tuple(
            CensoredReveal.from_mapping(row) for row in raw_censored
        )
        censored_batches = tuple(
            CensoredRevealBatch.from_mapping(row)
            for row in raw_censored_batches
        )
        retired_chains = tuple(
            RetiredContinuousChain.from_mapping(row)
            for row in raw_retired_chains
        )
        pending_q0 = tuple(
            PendingTrueResetQ0.from_mapping(row) for row in raw_pending_q0
        )
        true_reset_batches = tuple(
            TrueResetBatchReceipt.from_mapping(row)
            for row in raw_true_reset_batches
        )
        reset_high_waters = tuple(
            ResetHighWater.from_mapping(row) for row in raw_reset_high_waters
        )
        owner_transitions = tuple(
            OwnerTransitionRef.from_mapping(row)
            for row in raw_owner_transitions
        )
        for row in prepared:
            owner._validate_prepared_integrity(row)
        for row in prepared_batches:
            owner._validate_prepared_batch_integrity(row)
        for row in prepared_event_batches:
            owner._validate_prepared_batch_integrity(row)
        for row in abort_event_receipts:
            owner._validate_prepared_batch_integrity(row.prepared_batch)
        for row in committed:
            owner._validate_committed_integrity(row)
        for row in committed_batches:
            owner._validate_committed_batch_integrity(row)
        for batch in censored_batches:
            if tuple(batch.censored_reveals) != tuple(
                row
                for row in censored
                if row in batch.censored_reveals
            ):
                raise ContinuousRuntimeTransactionError(
                    "checkpoint censored batch row authority differs"
                )
        prepared_envs = [row.request.env_id for row in prepared]
        committed_keys = [
            (
                row.prepared_reveal.request.env_id,
                row.prepared_reveal.request.reset_generation,
                row.prepared_reveal.request.scheduled_ordinal,
            )
            for row in committed
        ]
        censored_keys = [
            (
                row.prepared_reveal.request.env_id,
                row.prepared_reveal.request.reset_generation,
                row.prepared_reveal.request.scheduled_ordinal,
            )
            for row in censored
        ]
        prepared_batch_keys = [row.selected_env_ids for row in prepared_batches]
        if (
            prepared_envs != sorted(prepared_envs)
            or len(set(prepared_envs)) != len(prepared_envs)
            or prepared_batch_keys != sorted(prepared_batch_keys)
            or len(set(prepared_batch_keys)) != len(prepared_batch_keys)
            or committed_keys != sorted(committed_keys)
            or len(set(committed_keys)) != len(committed_keys)
            or censored_keys != sorted(censored_keys)
            or len(set(censored_keys)) != len(censored_keys)
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint transaction rows must be sorted and unique"
            )
        prepared_flat_shas = tuple(
            row.canonical_sha256 for row in prepared
        )
        prepared_batch_flat_shas = tuple(
            row.canonical_sha256
            for batch in prepared_batches
            for row in batch.prepared_reveals
        )
        if (
            len(set(prepared_batch_flat_shas))
            != len(prepared_batch_flat_shas)
            or set(prepared_batch_flat_shas) != set(prepared_flat_shas)
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint prepared rows/batches are not an exact bijection"
            )
        committed_flat_shas = tuple(
            row.canonical_sha256 for row in committed
        )
        committed_batch_flat_shas = tuple(
            row.canonical_sha256
            for batch in committed_batches
            for row in batch.committed_reveals
        )
        if (
            len(set(committed_batch_flat_shas))
            != len(committed_batch_flat_shas)
            or set(committed_batch_flat_shas) != set(committed_flat_shas)
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint committed rows/batches are not an exact bijection"
            )
        censored_flat_shas = tuple(row.canonical_sha256 for row in censored)
        censored_batch_flat_shas = tuple(
            row.canonical_sha256
            for batch in censored_batches
            for row in batch.censored_reveals
        )
        if (
            len(set(censored_batch_flat_shas))
            != len(censored_batch_flat_shas)
            or set(censored_batch_flat_shas) != set(censored_flat_shas)
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint censored rows/batches are not an exact bijection"
            )

        retired_keys = [
            (row.env_id, row.reset_generation) for row in retired_chains
        ]
        pending_envs = [row.env_id for row in pending_q0]
        high_water_envs = [row.env_id for row in reset_high_waters]
        if (
            retired_keys != sorted(retired_keys)
            or len(set(retired_keys)) != len(retired_keys)
            or pending_envs != sorted(pending_envs)
            or len(set(pending_envs)) != len(pending_envs)
            or high_water_envs != sorted(high_water_envs)
            or len(set(high_water_envs)) != len(high_water_envs)
            or len({row.canonical_sha256 for row in true_reset_batches})
            != len(true_reset_batches)
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint reset/archive rows must be sorted and unique"
            )
        sequence_rows = tuple(
            sorted(
                (*committed, *censored),
                key=lambda row: (
                    row.prepared_reveal.request.env_id,
                    row.prepared_reveal.request.reset_generation,
                    row.prepared_reveal.request.scheduled_ordinal,
                ),
            )
        )
        sequence_keys = tuple(
            (
                row.prepared_reveal.request.env_id,
                row.prepared_reveal.request.reset_generation,
                row.prepared_reveal.request.scheduled_ordinal,
            )
            for row in sequence_rows
        )
        sequence_by_sha = {
            row.canonical_sha256: row for row in sequence_rows
        }
        if (
            len(set(sequence_keys)) != len(sequence_keys)
            or len(sequence_by_sha) != len(sequence_rows)
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint committed/censored sequence events collide"
            )
        retired_event_shas: set[str] = set()
        retired_committed_shas: set[str] = set()
        retired_by_sha = {
            row.canonical_sha256: row for row in retired_chains
        }
        for chain in retired_chains:
            if retired_event_shas.intersection(
                chain.sequence_event_sha256s
            ):
                raise ContinuousRuntimeTransactionError(
                    "checkpoint sequence event is retired twice"
                )
            retired_event_shas.update(chain.sequence_event_sha256s)
            retired_committed_shas.update(chain.committed_reveal_sha256s)
            rows = tuple(
                sequence_by_sha.get(event_sha)
                for event_sha in chain.sequence_event_sha256s
            )
            if any(row is None for row in rows):
                raise ContinuousRuntimeTransactionError(
                    "retired chain references an unknown sequence event"
                )
            requests = tuple(row.prepared_reveal.request for row in rows)
            latest = rows[-1]
            event_kinds = tuple(
                "COMMITTED"
                if isinstance(row, CommittedReveal)
                else "INFRA_CENSORED"
                for row in rows
            )
            committed_rows = tuple(
                row for row in rows if isinstance(row, CommittedReveal)
            )
            committed_shas = tuple(
                row.canonical_sha256 for row in committed_rows
            )
            latest_committed = (
                None if not committed_rows else committed_rows[-1]
            )
            latest_request = latest.prepared_reveal.request
            if (
                tuple(request.scheduled_ordinal for request in requests)
                != tuple(range(len(rows)))
                or event_kinds != chain.sequence_event_kinds
                or tuple(row.canonical_sha256 for row in rows)
                != chain.sequence_event_sha256s
                or committed_shas != chain.committed_reveal_sha256s
                or any(
                    request.env_id != chain.env_id
                    or request.reset_generation != chain.reset_generation
                    or request.birth_sha256 != chain.birth_sha256
                    or request.run_id != chain.run_id
                    or request.carry_chain_id != chain.carry_chain_id
                    for request in requests
                )
                or requests[0].sampler_generation
                != chain.first_sampler_generation
                or requests[-1].sampler_generation
                != chain.last_sampler_generation
                or latest.prepared_reveal.sampler_after_env_state_sha256
                != chain.sampler_high_water_env_state_sha256
                or (
                    None
                    if latest_committed is None
                    else latest_committed.prepared_reveal.outcome_key.canonical_sha256
                )
                != chain.latest_outcome_key_sha256
                or chain.closure_receipt.latest_sequence_event_kind
                != event_kinds[-1]
                or chain.closure_receipt.latest_sequence_event_sha256
                != latest.canonical_sha256
                or chain.closure_receipt.latest_committed_reveal_sha256
                != (
                    None
                    if latest_committed is None
                    else latest_committed.canonical_sha256
                )
                or (
                    chain.closure_receipt.closure_disposition
                    == "CLOSED_AFTER_DEADLINE"
                    and chain.closure_receipt.closed_at_step
                    < latest_request.scheduled_deadline_step
                )
                or (
                    chain.closure_receipt.closure_disposition
                    == "CENSORED_TRUE_RESET"
                    and chain.closure_receipt.closed_at_step
                    < latest_request.scheduled_reveal_step
                )
            ):
                raise ContinuousRuntimeTransactionError(
                    "retired chain does not bind its sequence evidence"
                )
        for pending in pending_q0:
            chain = retired_by_sha.get(pending.retired_chain_sha256)
            if (
                chain is None
                or chain.next_q0_request != pending.next_q0_request
                or pending.env_id != chain.env_id
            ):
                raise ContinuousRuntimeTransactionError(
                    "pending Q0 does not bind a retired chain"
                )

        current_by_env: dict[int, object] = {}
        sampler_current_by_env: dict[int, object] = {}
        parent_by_chain: dict[tuple[int, int], object] = {}
        physical_by_chain: dict[tuple[int, int], CommittedReveal] = {}
        active_physical_by_env: dict[int, CommittedReveal] = {}
        inbound_ball_shas_by_chain: dict[tuple[int, int], set[str]] = {}
        for row in sequence_rows:
            request = row.prepared_reveal.request
            chain_key = (request.env_id, request.reset_generation)
            parent = parent_by_chain.get(chain_key)
            if parent is None:
                linked_archive = next(
                    (
                        chain
                        for chain in retired_chains
                        if chain.next_q0_request == request
                    ),
                    None,
                )
                if (
                    request.scheduled_ordinal != 0
                    or request.previous_ball_slot_index is not None
                    or (
                        request.sampler_generation != 1
                        and linked_archive is None
                    )
                ):
                    raise ContinuousRuntimeTransactionError(
                        "checkpoint Q0 lacks initial or retired-chain authority"
                    )
            else:
                owner._validate_sequence_parent_against(
                    parent,
                    request=request,
                    slots=owner._validate_slots(request.ball_slots),
                    physical_previous=physical_by_chain.get(chain_key),
                )
            if parent is not None:
                owner._validate_successor_content_against(
                    parent,
                    row.prepared_reveal,
                )
            if isinstance(row, CommittedReveal):
                inbound_sha = row.ball_slot_plan.new_inbound_ball_sha256
                seen_inbound = inbound_ball_shas_by_chain.setdefault(
                    chain_key,
                    set(),
                )
                if inbound_sha in seen_inbound:
                    raise ContinuousRuntimeTransactionError(
                        "checkpoint carry chain reuses inbound ball identity"
                    )
                seen_inbound.add(inbound_sha)
            sampler_parent = sampler_current_by_env.get(request.env_id)
            expected_before = (
                None
                if sampler_parent is None
                else sampler_parent.prepared_reveal.sampler_after_env_state
            )
            if row.prepared_reveal.sampler_before_env_state != expected_before:
                raise ContinuousRuntimeTransactionError(
                    "checkpoint sequence sampler chain differs"
                )
            parent_by_chain[chain_key] = row
            sampler_current_by_env[request.env_id] = row
            if isinstance(row, CommittedReveal):
                physical_by_chain[chain_key] = row
                if row.canonical_sha256 not in retired_committed_shas:
                    active_physical_by_env[request.env_id] = row
            if row.canonical_sha256 not in retired_event_shas:
                current_by_env[request.env_id] = row

        live_checkpoint = _checkpoint_from_json(
            owner._state.sampler_checkpoint_json
        )
        live_env_ids = [
            row["env_id"] for row in live_checkpoint["environments"]
        ]
        current_env_ids = sorted(sampler_current_by_env)
        if live_env_ids != current_env_ids:
            raise ContinuousRuntimeTransactionError(
                "checkpoint live sampler envs differ from committed envs"
            )
        for env_id, row in sampler_current_by_env.items():
            if (
                _sampler_env_row(live_checkpoint, env_id)
                != row.prepared_reveal.sampler_after_env_state
            ):
                raise ContinuousRuntimeTransactionError(
                    "checkpoint sequence sampler after-state differs"
                )
        for row in prepared:
            parent = current_by_env.get(row.request.env_id)
            pending = next(
                (
                    item
                    for item in pending_q0
                    if item.env_id == row.request.env_id
                ),
                None,
            )
            owner._validate_sequence_parent_against(
                parent,
                request=row.request,
                slots=owner._validate_slots(row.request.ball_slots),
                pending_q0=pending,
                physical_previous=active_physical_by_env.get(
                    row.request.env_id
                ),
            )
            if parent is not None:
                owner._validate_successor_content_against(parent, row)
            if (
                row.prepared_ball_slot_reservation.new_inbound_ball_sha256
                in inbound_ball_shas_by_chain.get(
                    (row.request.env_id, row.request.reset_generation), set()
                )
            ):
                raise ContinuousRuntimeTransactionError(
                    "checkpoint prepared row reuses inbound ball identity"
                )
            if (
                _sampler_env_row(live_checkpoint, row.request.env_id)
                != row.sampler_before_env_state
            ):
                raise ContinuousRuntimeTransactionError(
                    "checkpoint prepared sampler before-state differs"
                )
            staged = _merge_sampler_env_row(
                live_checkpoint,
                env_id=row.request.env_id,
                expected_before=row.sampler_before_env_state,
                staged_after=row.sampler_after_env_state,
            )
            _target_sampler.ContinuousTargetSampler.from_checkpoint(profile, staged)
        history = tuple(
            _sha256(item, label="committed_receipt_shas") for item in raw_history
        )
        batch_history = tuple(
            _sha256(item, label="committed_batch_shas")
            for item in raw_batch_history
        )
        transcript = _sha256(
            payload["committed_transcript_sha256"],
            label="committed_transcript_sha256",
        )
        if _transcript_root(history) != transcript:
            raise ContinuousRuntimeTransactionError(
                "checkpoint committed transcript differs"
            )
        committed_shas = {row.canonical_sha256 for row in committed}
        expected_history = tuple(
            row.canonical_sha256
            for batch in committed_batches
            for row in batch.committed_reveals
        )
        if (
            len(history) != len(committed)
            or len(set(history)) != len(history)
            or set(history) != committed_shas
            or history != expected_history
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint committed rows/transcript membership differs"
            )
        batch_transcript = _sha256(
            payload["committed_batch_transcript_sha256"],
            label="committed_batch_transcript_sha256",
        )
        expected_batch_history = tuple(
            row.canonical_sha256 for row in committed_batches
        )
        if (
            batch_history != expected_batch_history
            or len(set(batch_history)) != len(batch_history)
            or _batch_transcript_root(batch_history) != batch_transcript
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint committed batch transcript differs"
            )
        declared_owner_root = _sha256(
            payload["owner_transition_root_sha256"],
            label="owner_transition_root_sha256",
        )
        if declared_owner_root != _owner_transition_root(owner_transitions):
            raise ContinuousRuntimeTransactionError(
                "checkpoint unified owner transition root differs"
            )
        prepare_event_index = 0
        abort_event_index = 0
        commit_index = 0
        censor_index = 0
        reset_index = 0
        committed_prefix: list[CommittedReveal] = []
        committed_transcript_prefix: list[str] = []
        censored_prefix: list[CensoredReveal] = []
        retired_prefix: list[RetiredContinuousChain] = []
        pending_by_env: dict[int, PendingTrueResetQ0] = {}
        high_water_by_env: dict[int, ResetHighWater] = {}
        active_head_by_env: dict[int, object] = {}
        active_physical_by_env: dict[int, CommittedReveal] = {}
        replay_prepared_batches: list[PreparedRevealBatch] = []
        latest_retired_by_env: dict[int, str] = {}
        replayed_transitions: list[OwnerTransitionRef] = []
        replay_sampler_checkpoint = _target_sampler.ContinuousTargetSampler(
            profile, seed=owner.seed
        ).checkpoint()

        def replay_state() -> _OwnerState:
            return _OwnerState(
                sampler_checkpoint_json=_checkpoint_json(
                    replay_sampler_checkpoint
                ),
                prepared=tuple(
                    sorted(
                        (
                            row
                            for active_batch in replay_prepared_batches
                            for row in active_batch.prepared_reveals
                        ),
                        key=lambda row: row.request.env_id,
                    )
                ),
                prepared_batches=tuple(
                    sorted(
                        replay_prepared_batches,
                        key=lambda row: row.selected_env_ids,
                    )
                ),
                prepared_event_batches=tuple(
                    prepared_event_batches[:prepare_event_index]
                ),
                abort_event_receipts=tuple(
                    abort_event_receipts[:abort_event_index]
                ),
                committed=tuple(
                    sorted(
                        committed_prefix,
                        key=lambda row: (
                            row.prepared_reveal.request.env_id,
                            row.prepared_reveal.request.reset_generation,
                            row.prepared_reveal.request.scheduled_ordinal,
                        ),
                    )
                ),
                committed_batches=tuple(committed_batches[:commit_index]),
                committed_receipt_shas=tuple(committed_transcript_prefix),
                committed_batch_shas=tuple(
                    row.canonical_sha256
                    for row in committed_batches[:commit_index]
                ),
                censored=tuple(
                    sorted(
                        censored_prefix,
                        key=lambda row: (
                            row.prepared_reveal.request.env_id,
                            row.prepared_reveal.request.reset_generation,
                            row.prepared_reveal.request.scheduled_ordinal,
                        ),
                    )
                ),
                censored_batches=tuple(censored_batches[:censor_index]),
                retired_chains=tuple(
                    sorted(
                        retired_prefix,
                        key=lambda row: (row.env_id, row.reset_generation),
                    )
                ),
                pending_true_reset_q0=tuple(
                    pending_by_env[env_id] for env_id in sorted(pending_by_env)
                ),
                true_reset_batches=tuple(true_reset_batches[:reset_index]),
                reset_high_waters=tuple(
                    high_water_by_env[env_id]
                    for env_id in sorted(high_water_by_env)
                ),
                owner_transitions=tuple(replayed_transitions),
            )

        for transition in owner_transitions:
            if transition.transition_kind == "PREPARE_BATCH":
                if prepare_event_index >= len(prepared_event_batches):
                    raise ContinuousRuntimeTransactionError(
                        "owner transition names an extra prepare batch"
                    )
                batch = prepared_event_batches[prepare_event_index]
                prepare_event_index += 1
                if (
                    transition.transition_sha256 != batch.canonical_sha256
                    or batch.sampler_checkpoint_before
                    != replay_sampler_checkpoint
                ):
                    raise ContinuousRuntimeTransactionError(
                        "owner prepare transition lineage differs"
                    )
                active_prepared_envs = {
                    row.request.env_id
                    for active_batch in replay_prepared_batches
                    for row in active_batch.prepared_reveals
                }
                if active_prepared_envs.intersection(batch.selected_env_ids):
                    raise ContinuousRuntimeTransactionError(
                        "owner prepare transition overlaps a live private batch"
                    )
                for prepared_row in batch.prepared_reveals:
                    request = prepared_row.request
                    parent = active_head_by_env.get(request.env_id)
                    pending = pending_by_env.get(request.env_id)
                    owner._validate_sequence_parent_against(
                        parent,
                        request=request,
                        slots=owner._validate_slots(request.ball_slots),
                        pending_q0=pending,
                        physical_previous=active_physical_by_env.get(
                            request.env_id
                        ),
                    )
                    if parent is not None:
                        owner._validate_successor_content_against(
                            parent, prepared_row
                        )
                replay_prepared_batches.append(batch)
            elif transition.transition_kind == "ABORT_BATCH":
                if abort_event_index >= len(abort_event_receipts):
                    raise ContinuousRuntimeTransactionError(
                        "owner transition names an extra abort receipt"
                    )
                receipt = abort_event_receipts[abort_event_index]
                abort_event_index += 1
                matches = [
                    row
                    for row in replay_prepared_batches
                    if row.canonical_sha256 == receipt.prepared_batch_sha256
                ]
                if (
                    transition.transition_sha256 != receipt.canonical_sha256
                    or len(matches) != 1
                    or receipt.prepared_batch != matches[0]
                    or receipt.sampler_checkpoint != replay_sampler_checkpoint
                ):
                    raise ContinuousRuntimeTransactionError(
                        "owner abort transition lineage differs"
                    )
                replay_prepared_batches.remove(matches[0])
            elif transition.transition_kind == "COMMIT_BATCH":
                if commit_index >= len(committed_batches):
                    raise ContinuousRuntimeTransactionError(
                        "owner transition names an extra commit batch"
                )
                batch = committed_batches[commit_index]
                if transition.transition_sha256 != batch.canonical_sha256:
                    raise ContinuousRuntimeTransactionError(
                        "owner commit transition order differs"
                    )
                if (
                    batch.reveal_final_preview.owner_checkpoint_before_sha256
                    != owner._checkpoint_for_state(
                        replay_state()
                    )["canonical_sha256"]
                ):
                    raise ContinuousRuntimeTransactionError(
                        "owner commit reveal-final lease before-state differs"
                    )
                commit_index += 1
                prepared_matches = [
                    row
                    for row in replay_prepared_batches
                    if row.canonical_sha256
                    == batch.prepared_batch.canonical_sha256
                ]
                if len(prepared_matches) != 1:
                    raise ContinuousRuntimeTransactionError(
                        "owner commit transition lacks its live private batch"
                    )
                replay_prepared_batches.remove(prepared_matches[0])
                if batch.sampler_checkpoint_before_commit != replay_sampler_checkpoint:
                    raise ContinuousRuntimeTransactionError(
                        "owner event sampler before-state differs"
                    )
                replay_sampler_checkpoint = dict(
                    batch.sampler_checkpoint_after_commit
                )
                for committed_row in batch.committed_reveals:
                    request = committed_row.prepared_reveal.request
                    if request.scheduled_ordinal == 0:
                        pending = pending_by_env.pop(request.env_id, None)
                        if request.env_id in active_head_by_env:
                            raise ContinuousRuntimeTransactionError(
                                "owner event committed Q0 over an active chain"
                            )
                        if pending is not None and request != pending.next_q0_request:
                            raise ContinuousRuntimeTransactionError(
                                "owner event Q0 differs from pending reset"
                            )
                        if pending is None and any(
                            row.prepared_reveal.request.env_id == request.env_id
                            for row in committed_prefix
                        ):
                            raise ContinuousRuntimeTransactionError(
                                "owner event repeated Q0 without true reset"
                            )
                    else:
                        parent = active_head_by_env.get(request.env_id)
                        if parent is None:
                            raise ContinuousRuntimeTransactionError(
                                "owner event successor lacks active head"
                            )
                        owner._validate_sequence_parent_against(
                            parent,
                            request=request,
                            slots=owner._validate_slots(request.ball_slots),
                            physical_previous=active_physical_by_env.get(
                                request.env_id
                            ),
                        )
                    active_head_by_env[request.env_id] = committed_row
                    active_physical_by_env[request.env_id] = committed_row
                    committed_prefix.append(committed_row)
                    committed_transcript_prefix.append(
                        committed_row.canonical_sha256
                    )
                    prior_high_water = high_water_by_env.get(request.env_id)
                    high_water_by_env[request.env_id] = ResetHighWater(
                        env_id=request.env_id,
                        latest_reset_generation=request.reset_generation,
                        sampler_draw_high_water=request.sampler_generation,
                        latest_retired_chain_sha256=(
                            None
                            if prior_high_water is None
                            else prior_high_water.latest_retired_chain_sha256
                        ),
                        active_sequence_event_kind="COMMITTED",
                        active_sequence_event_sha256=(
                            committed_row.canonical_sha256
                        ),
                        active_committed_reveal_sha256=(
                            committed_row.canonical_sha256
                        ),
                        pending_true_reset_q0_sha256=None,
                    )
            elif transition.transition_kind == "CENSOR_BATCH":
                if censor_index >= len(censored_batches):
                    raise ContinuousRuntimeTransactionError(
                        "owner transition names an extra censor batch"
                    )
                batch = censored_batches[censor_index]
                if (
                    transition.transition_sha256 != batch.canonical_sha256
                    or batch.reveal_final_preview.owner_checkpoint_before_sha256
                    != owner._checkpoint_for_state(
                        replay_state()
                    )["canonical_sha256"]
                    or batch.sampler_checkpoint_before
                    != replay_sampler_checkpoint
                ):
                    raise ContinuousRuntimeTransactionError(
                        "owner censor transition lineage differs"
                    )
                censor_index += 1
                prepared_matches = [
                    row
                    for row in replay_prepared_batches
                    if row.canonical_sha256
                    == batch.reveal_final_preview.prepared_batch.canonical_sha256
                ]
                if len(prepared_matches) != 1:
                    raise ContinuousRuntimeTransactionError(
                        "owner censor transition lacks its live private batch"
                    )
                replay_prepared_batches.remove(prepared_matches[0])
                replay_sampler_checkpoint = dict(batch.sampler_checkpoint_after)
                for censored_row in batch.censored_reveals:
                    request = censored_row.prepared_reveal.request
                    if request.scheduled_ordinal == 0:
                        pending = pending_by_env.pop(request.env_id, None)
                        if request.env_id in active_head_by_env:
                            raise ContinuousRuntimeTransactionError(
                                "owner event censored Q0 over an active chain"
                            )
                        if pending is not None and request != pending.next_q0_request:
                            raise ContinuousRuntimeTransactionError(
                                "owner censored Q0 differs from pending reset"
                            )
                    else:
                        parent = active_head_by_env.get(request.env_id)
                        if parent is None:
                            raise ContinuousRuntimeTransactionError(
                                "owner censored successor lacks active head"
                            )
                        owner._validate_sequence_parent_against(
                            parent,
                            request=request,
                            slots=owner._validate_slots(request.ball_slots),
                            physical_previous=active_physical_by_env.get(
                                request.env_id
                            ),
                        )
                    active_head_by_env[request.env_id] = censored_row
                    censored_prefix.append(censored_row)
                    prior_high_water = high_water_by_env.get(request.env_id)
                    high_water_by_env[request.env_id] = ResetHighWater(
                        env_id=request.env_id,
                        latest_reset_generation=request.reset_generation,
                        sampler_draw_high_water=request.sampler_generation,
                        latest_retired_chain_sha256=(
                            None
                            if prior_high_water is None
                            else prior_high_water.latest_retired_chain_sha256
                        ),
                        active_sequence_event_kind="INFRA_CENSORED",
                        active_sequence_event_sha256=(
                            censored_row.canonical_sha256
                        ),
                        active_committed_reveal_sha256=(
                            None
                            if request.env_id not in active_physical_by_env
                            else active_physical_by_env[
                                request.env_id
                            ].canonical_sha256
                        ),
                        pending_true_reset_q0_sha256=None,
                    )
            else:
                if reset_index >= len(true_reset_batches):
                    raise ContinuousRuntimeTransactionError(
                        "owner transition names an extra true-reset batch"
                    )
                batch = true_reset_batches[reset_index]
                reset_index += 1
                if (
                    transition.transition_sha256 != batch.canonical_sha256
                    or batch.parent_true_reset_transcript_sha256
                    != _owner_transition_root(replayed_transitions)
                    or batch.sampler_checkpoint_sha256
                    != replay_sampler_checkpoint["checkpoint_sha256"]
                    or batch.committed_transcript_sha256
                    != _transcript_root(committed_transcript_prefix)
                ):
                    raise ContinuousRuntimeTransactionError(
                        "owner true-reset transition lineage differs"
                    )
                before_state = _OwnerState(
                    sampler_checkpoint_json=_checkpoint_json(
                        replay_sampler_checkpoint
                    ),
                    prepared=tuple(
                        sorted(
                            (
                                row
                                for active_batch in replay_prepared_batches
                                for row in active_batch.prepared_reveals
                            ),
                            key=lambda row: row.request.env_id,
                        )
                    ),
                    prepared_batches=tuple(
                        sorted(
                            replay_prepared_batches,
                            key=lambda row: row.selected_env_ids,
                        )
                    ),
                    prepared_event_batches=tuple(
                        prepared_event_batches[:prepare_event_index]
                    ),
                    abort_event_receipts=tuple(
                        abort_event_receipts[:abort_event_index]
                    ),
                    committed=tuple(
                        sorted(
                            committed_prefix,
                            key=lambda row: (
                                row.prepared_reveal.request.env_id,
                                row.prepared_reveal.request.reset_generation,
                                row.prepared_reveal.request.scheduled_ordinal,
                            ),
                        )
                    ),
                    committed_batches=tuple(committed_batches[:commit_index]),
                    committed_receipt_shas=tuple(committed_transcript_prefix),
                    committed_batch_shas=tuple(
                        row.canonical_sha256
                        for row in committed_batches[:commit_index]
                    ),
                    censored=tuple(
                        sorted(
                            censored_prefix,
                            key=lambda row: (
                                row.prepared_reveal.request.env_id,
                                row.prepared_reveal.request.reset_generation,
                                row.prepared_reveal.request.scheduled_ordinal,
                            ),
                        )
                    ),
                    censored_batches=tuple(censored_batches[:censor_index]),
                    retired_chains=tuple(retired_prefix),
                    pending_true_reset_q0=tuple(
                        pending_by_env[env_id] for env_id in sorted(pending_by_env)
                    ),
                    true_reset_batches=tuple(true_reset_batches[: reset_index - 1]),
                    reset_high_waters=tuple(
                        high_water_by_env[env_id]
                        for env_id in sorted(high_water_by_env)
                    ),
                    owner_transitions=tuple(replayed_transitions),
                )
                if (
                    any(
                        row.request.env_id in batch.selected_env_ids
                        for row in before_state.prepared
                    )
                    or batch.unselected_prepared_owner_root_sha256
                    != _prepared_owner_root(
                        before_state,
                        excluded_env_ids=batch.selected_env_ids,
                    )
                    or batch.active_owner_root_before_sha256
                    != _active_owner_root(before_state)
                    or batch.unselected_active_owner_root_before_sha256
                    != _active_owner_root(
                        before_state,
                        excluded_env_ids=batch.selected_env_ids,
                    )
                ):
                    raise ContinuousRuntimeTransactionError(
                        "true-reset before active-owner root differs"
                    )
                for chain, registration in zip(
                    batch.retired_chains, batch.pending_q0
                ):
                    active = active_head_by_env.get(chain.env_id)
                    if (
                        active is None
                        or active.canonical_sha256
                        != chain.sequence_event_sha256s[-1]
                        or chain.canonical_sha256 in latest_retired_by_env.values()
                        or chain not in retired_chains
                    ):
                        raise ContinuousRuntimeTransactionError(
                            "true-reset event archive differs from active head"
                        )
                    active_head_by_env.pop(chain.env_id)
                    active_physical_by_env.pop(chain.env_id, None)
                    pending_by_env[chain.env_id] = registration
                    retired_prefix.append(chain)
                    latest_retired_by_env[chain.env_id] = chain.canonical_sha256
                    prior_high_water = high_water_by_env.get(chain.env_id)
                    if prior_high_water is None:
                        raise ContinuousRuntimeTransactionError(
                            "true-reset event lacks prior high-water"
                        )
                    high_water_by_env[chain.env_id] = ResetHighWater(
                        env_id=chain.env_id,
                        latest_reset_generation=(
                            registration.next_q0_request.reset_generation
                        ),
                        sampler_draw_high_water=(
                            prior_high_water.sampler_draw_high_water
                        ),
                        latest_retired_chain_sha256=chain.canonical_sha256,
                        active_sequence_event_kind=None,
                        active_sequence_event_sha256=None,
                        active_committed_reveal_sha256=None,
                        pending_true_reset_q0_sha256=(
                            registration.canonical_sha256
                        ),
                    )
                after_state = _OwnerState(
                    sampler_checkpoint_json=before_state.sampler_checkpoint_json,
                    prepared=before_state.prepared,
                    prepared_batches=before_state.prepared_batches,
                    prepared_event_batches=before_state.prepared_event_batches,
                    abort_event_receipts=before_state.abort_event_receipts,
                    committed=before_state.committed,
                    committed_batches=before_state.committed_batches,
                    committed_receipt_shas=before_state.committed_receipt_shas,
                    committed_batch_shas=before_state.committed_batch_shas,
                    censored=before_state.censored,
                    censored_batches=before_state.censored_batches,
                    retired_chains=tuple(
                        sorted(
                            retired_prefix,
                            key=lambda row: (row.env_id, row.reset_generation),
                        )
                    ),
                    pending_true_reset_q0=tuple(
                        pending_by_env[env_id] for env_id in sorted(pending_by_env)
                    ),
                    true_reset_batches=tuple(true_reset_batches[:reset_index]),
                    reset_high_waters=tuple(
                        high_water_by_env[env_id]
                        for env_id in sorted(high_water_by_env)
                    ),
                    owner_transitions=tuple(replayed_transitions),
                )
                if (
                    batch.active_owner_root_after_sha256
                    != _active_owner_root(after_state)
                    or batch.unselected_active_owner_root_after_sha256
                    != _active_owner_root(
                        after_state,
                        excluded_env_ids=batch.selected_env_ids,
                    )
                ):
                    raise ContinuousRuntimeTransactionError(
                        "true-reset after active-owner root differs"
                    )
            replayed_transitions.append(transition)
        if (
            prepare_event_index != len(prepared_event_batches)
            or abort_event_index != len(abort_event_receipts)
            or commit_index != len(committed_batches)
            or censor_index != len(censored_batches)
            or reset_index != len(true_reset_batches)
            or tuple(
                sorted(
                    replay_prepared_batches,
                    key=lambda row: row.selected_env_ids,
                )
            )
            != prepared_batches
            or tuple(
                sorted(
                    (
                        row
                        for active_batch in replay_prepared_batches
                        for row in active_batch.prepared_reveals
                    ),
                    key=lambda row: row.request.env_id,
                )
            )
            != prepared
            or tuple(
                sorted(
                    censored_prefix,
                    key=lambda row: (
                        row.prepared_reveal.request.env_id,
                        row.prepared_reveal.request.reset_generation,
                        row.prepared_reveal.request.scheduled_ordinal,
                    ),
                )
            )
            != censored
            or tuple(
                sorted(
                    retired_prefix,
                    key=lambda row: (row.env_id, row.reset_generation),
                )
            )
            != retired_chains
            or tuple(
                pending_by_env[env_id] for env_id in sorted(pending_by_env)
            )
            != pending_q0
            or tuple(
                high_water_by_env[env_id]
                for env_id in sorted(high_water_by_env)
            )
            != reset_high_waters
        ):
            raise ContinuousRuntimeTransactionError(
                "checkpoint owner transition replay differs from final state"
            )
        expected_active_heads = {
            env_id: row.canonical_sha256
            for env_id, row in current_by_env.items()
        }
        if expected_active_heads != {
            env_id: row.canonical_sha256
            for env_id, row in active_head_by_env.items()
        }:
            raise ContinuousRuntimeTransactionError(
                "checkpoint active chain heads differ from event replay"
            )
        owner._state = _OwnerState(
            sampler_checkpoint_json=_checkpoint_json(live_checkpoint),
            prepared=prepared,
            prepared_batches=prepared_batches,
            prepared_event_batches=prepared_event_batches,
            abort_event_receipts=abort_event_receipts,
            committed=committed,
            committed_batches=committed_batches,
            committed_receipt_shas=history,
            committed_batch_shas=batch_history,
            censored=censored,
            censored_batches=censored_batches,
            retired_chains=retired_chains,
            pending_true_reset_q0=pending_q0,
            true_reset_batches=true_reset_batches,
            reset_high_waters=reset_high_waters,
            owner_transitions=owner_transitions,
        )
        if owner._checkpoint_for_state(owner._state) != dict(value):
            raise ContinuousRuntimeTransactionError(
                "checkpoint round-trip changed transaction state"
            )
        return owner


__all__ = [
    "AbortBatchReceipt",
    "AbortReceipt",
    "ArmedRevealTerminalHandle",
    "BALL_CLOSED",
    "BALL_EMPTY",
    "BALL_INBOUND",
    "BALL_OPEN",
    "BALL_PAID",
    "BALL_SETTLED_UNPAID",
    "BallSlotCapacityError",
    "BallSlotPlan",
    "BallSlotSnapshot",
    "CandidateTaskMaterialization",
    "COMMITTED",
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "CensoredReveal",
    "CensoredRevealBatch",
    "CommittedReveal",
    "CommittedRevealBatch",
    "ContinuousPrepareRequest",
    "ContinuousRevealFacts",
    "ContinuousRuntimeTransactionError",
    "ContinuousRuntimeTransactionOwner",
    "Float32TargetAliasError",
    "INTEGRATION_STATUS",
    "InfrastructureCensorFact",
    "LEGACY_CHECKPOINT_KIND",
    "LEGACY_V2_CHECKPOINT_KIND",
    "LEGACY_V3_CHECKPOINT_KIND",
    "OwnerTransitionRef",
    "PendingTrueResetQ0",
    "PREPARED",
    "PREPARED_REVEAL_TERMINAL_CLAIM_KIND",
    "PreparedBallSlotReservation",
    "PreparedReveal",
    "PreparedRevealBatch",
    "PreparedRevealTerminalClaim",
    "PreparedTerminalContentPin",
    "REVEAL_FINAL_PREVIEWED",
    "RevealFinalInstallRow",
    "RevealFinalPreviewAbortReceipt",
    "RevealFinalPreviewBatch",
    "RevealTerminalBoundaryMarker",
    "RUNTIME_WIRING_CONNECTED",
    "ResetHighWater",
    "RetiredContinuousChain",
    "TransactionConflictError",
    "TERMINAL_DECISION_ACCEPT",
    "TERMINAL_DECISION_CENSOR",
    "TERMINAL_DECISIONS",
    "TERMINAL_BOUNDARY_AUTHORITY_KIND",
    "TERMINAL_BOUNDARY_DECISION_MAPPING_SCHEMA_VERSION",
    "TERMINAL_BOUNDARY_SOURCE_DECISION_PASS",
    "TerminalBoundaryAuthority",
    "TerminalBoundaryCensorEvidence",
    "TerminalBoundaryParticipantRoot",
    "TerminalBoundaryProjection",
    "TrueResetBatchReceipt",
    "TrueResetClosureBatch",
    "TrueResetClosureReceipt",
    "canonical_sha256",
]
