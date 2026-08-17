"""Fail-closed placement success accounting for continuous ActionBall.

This module consumes two upstream contract objects rather than accepting an
unbound success label:

* a schedule/reveal ledger owns the committed shot denominator and carry-state
  ordinal chain; and
* a paid :mod:`action_ball_landing_outcome_mailbox` row owns contact, first
  crossing, target, frame, placement profile, score, source step, and payment
  identity.

The only success is ``opponent_on_table``.  ``opponent_off_table`` may carry
positive placement shaping but remains a failure.  Construction-infeasible
proposals are pre-reveal records and never enter the policy denominator.
Infrastructure-invalid committed rows use a closed censor receipt, remain in
committed conservation, and are excluded from policy statistics.

The current schedule ledger is still a pre-integration evidence carrier: no
live Isaac/MuJoCo runtime authority adapter is frozen.  Therefore every
non-empty aggregate is explicitly diagnostic and ``curriculum_signal`` always
fails closed.  Canonical records and arithmetic cannot promote self-reported
runtime evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, Union

from action_ball_landing_outcome_mailbox import (
    CLOSED,
    PAID,
    LandingOutcomePayment,
    LandingOutcomeShotKey,
    LandingOutcomeView,
)
from action_ball_landing_placement import (
    LandingPlacementFacts,
    LandingPlacementProfile,
    LandingPlacementScore,
    LandingPlacementTaskIdentity,
)


SCHEMA_VERSION = 2
CONTRACT_SCOPE = "dependency_light_runtime_preintegration_only_v2"
RUNTIME_AUTHORITY_FROZEN = False
LAUNCH_READY = False

SHOT_IDENTITY_KIND = "action_ball_placement_shot_identity_v2"
INFRA_CENSOR_KIND = "action_ball_placement_infra_censor_v2"
SCHEDULE_ENTRY_KIND = "action_ball_placement_schedule_entry_v2"
SCHEDULE_LEDGER_KIND = "action_ball_placement_schedule_ledger_v2"
CONSTRUCTION_REJECTION_KIND = (
    "action_ball_placement_construction_rejection_v2"
)
CONSUMER_RECEIPT_KIND = "action_ball_placement_consumer_receipt_v2"
MAILBOX_SETTLEMENT_KIND = "action_ball_placement_mailbox_settlement_v2"
INFRA_SETTLEMENT_KIND = "action_ball_placement_infra_settlement_v2"
AGGREGATE_KIND = "action_ball_placement_success_aggregate_v2"

SUCCESS_DEFINITION = "opponent_on_table_only"
ZERO_DENOMINATOR_SEMANTICS = "unmeasured"
CURRICULUM_METRIC_NAMES = (
    "on_table_rate",
    "raw_target_error_m",
    "placement_quality",
)
REWARD_CONSUMER_NAME = "landing_placement"
SCHEDULE_AUTHORITY_SCOPE = "preintegration_schedule_reveal_ledger_v2"

_MAX_ACTION_UID = (1 << 53) - 1
_RUNTIME_KEY_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
)
_SUCCESSOR_KEY_FIELDS = (
    "run_id",
    "carry_chain_id",
    "shot_index",
    "source_sha256",
    "config_sha256",
    "receipt_content_sha256",
)
_FULL_KEY_FIELDS = (*_RUNTIME_KEY_FIELDS, *_SUCCESSOR_KEY_FIELDS)
_ZERO_SHA256 = "0" * 64


class PlacementSuccessContractError(ValueError):
    """Placement authority, identity, partition, or arithmetic is invalid."""


class RuntimeAuthorityRequiredError(PlacementSuccessContractError):
    """A pre-integration aggregate was requested as curriculum evidence."""


class ConstructionRejectReason(str, Enum):
    SOLVER_INFEASIBLE = "solver_infeasible"
    KINEMATIC_INFEASIBLE = "kinematic_infeasible"
    TARGET_SUPPORT_EMPTY = "target_support_empty"


class InfraCensorReason(str, Enum):
    NONFINITE_STATE = "nonfinite_state"
    ENGINE_OVERFLOW = "engine_overflow"
    RECEIPT_FAULT = "receipt_fault"


class CommittedShotOutcome(str, Enum):
    NO_CONTACT = "no_contact"
    NONFINITE = "nonfinite"
    NO_CROSSING = "no_crossing"
    NET_FAIL = "net_fail"
    OWN_OR_BACK = "own_or_back"
    OPPONENT_OFF_TABLE = "opponent_off_table"
    OPPONENT_ON_TABLE = "opponent_on_table"
    INFRA_CENSOR = "infra_censor"


class MeasurementStatus(str, Enum):
    UNMEASURED = "UNMEASURED"
    DIAGNOSTIC_RUNTIME_AUTHORITY_REQUIRED = (
        "DIAGNOSTIC_RUNTIME_AUTHORITY_REQUIRED"
    )


_MAILBOX_REASON_TO_OUTCOME = {
    "no_contact": CommittedShotOutcome.NO_CONTACT,
    "nonfinite": CommittedShotOutcome.NONFINITE,
    "no_crossing": CommittedShotOutcome.NO_CROSSING,
    "net_not_crossed": CommittedShotOutcome.NET_FAIL,
    "net_not_clear": CommittedShotOutcome.NET_FAIL,
    "not_opponent_bound": CommittedShotOutcome.OWN_OR_BACK,
    "scored_off_table": CommittedShotOutcome.OPPONENT_OFF_TABLE,
    "scored_on_table": CommittedShotOutcome.OPPONENT_ON_TABLE,
}
_LANDING_OUTCOMES = frozenset(
    (
        CommittedShotOutcome.OPPONENT_OFF_TABLE,
        CommittedShotOutcome.OPPONENT_ON_TABLE,
    )
)
_ZERO_INCOME_OUTCOMES = frozenset(
    (
        CommittedShotOutcome.NO_CONTACT,
        CommittedShotOutcome.NONFINITE,
        CommittedShotOutcome.NO_CROSSING,
        CommittedShotOutcome.NET_FAIL,
        CommittedShotOutcome.OWN_OR_BACK,
    )
)


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PlacementSuccessContractError(
                "canonical JSON contains a non-finite float"
            )
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PlacementSuccessContractError(
                    "canonical JSON keys must be exact strings"
                )
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise PlacementSuccessContractError(
        "value is not dependency-light canonical JSON"
    )


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_sha256(payload)
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PlacementSuccessContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _optional_sha256(value: object, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _sha256(value, label=label)


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise PlacementSuccessContractError(f"{label} must be an exact int")
    if value < minimum:
        raise PlacementSuccessContractError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise PlacementSuccessContractError(f"{label} must be <= {maximum}")
    return value


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise PlacementSuccessContractError(f"{label} must be an exact bool")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PlacementSuccessContractError(
            f"{label} must be non-empty stripped text"
        )
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PlacementSuccessContractError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PlacementSuccessContractError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _finite_nonnegative(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0:
        raise PlacementSuccessContractError(f"{label} must be non-negative")
    return result


def _optional_finite_nonnegative(
    value: object, *, label: str
) -> Optional[float]:
    if value is None:
        return None
    return _finite_nonnegative(value, label=label)


def _enum(value: object, *, enum_cls: type[Enum], label: str):
    if isinstance(value, enum_cls):
        return value
    if type(value) is not str:
        raise PlacementSuccessContractError(f"{label} is not in the closed enum")
    try:
        return enum_cls(value)
    except ValueError as error:
        raise PlacementSuccessContractError(
            f"{label} is not in the closed enum"
        ) from error


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlacementSuccessContractError(f"{label} must be a sequence")
    return tuple(value)


def _verified_payload(
    value: object,
    *,
    expected_payload_keys: frozenset[str],
    kind: str,
    label: str,
) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping):
        raise PlacementSuccessContractError(f"{label} must be a mapping")
    expected = expected_payload_keys | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise PlacementSuccessContractError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    declared = _sha256(
        value["canonical_sha256"], label=f"{label}.canonical_sha256"
    )
    payload = {key: value[key] for key in expected_payload_keys}
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PlacementSuccessContractError(f"{label} schema_version differs")
    if payload["kind"] != kind:
        raise PlacementSuccessContractError(f"{label} kind differs")
    if canonical_sha256(payload) != declared:
        raise PlacementSuccessContractError(f"{label} canonical SHA differs")
    return payload, declared


def _record_root(kind: str, rows: Sequence[str]) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "ordering": "canonical_sha256_lexical",
            "record_sha256s": sorted(
                _sha256(item, label="record SHA") for item in rows
            ),
        }
    )


@dataclass(frozen=True)
class PlacementShotIdentity:
    """Exact 14-field equivalent of the mailbox full shot key."""

    KIND: ClassVar[str] = SHOT_IDENTITY_KIND

    env_id: int
    reset_generation: int
    swing_generation: int
    action_uid: int
    action_slot: int
    birth_sha256: str
    sample_sha256: str
    task_sha256: str
    run_id: str
    carry_chain_id: str
    shot_index: int
    source_sha256: str
    config_sha256: str
    receipt_content_sha256: str

    def __post_init__(self) -> None:
        minimums = {
            "env_id": 0,
            "reset_generation": 1,
            "swing_generation": 0,
            "action_uid": 1,
            "action_slot": 0,
            "shot_index": 1,
        }
        for name in (
            "env_id",
            "reset_generation",
            "swing_generation",
            "action_uid",
            "action_slot",
            "shot_index",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    label=name,
                    minimum=minimums[name],
                    maximum=_MAX_ACTION_UID if name == "action_uid" else None,
                ),
            )
        for name in ("run_id", "carry_chain_id"):
            object.__setattr__(
                self, name, _nonempty_text(getattr(self, name), label=name)
            )
        for name in (
            "birth_sha256",
            "sample_sha256",
            "task_sha256",
            "source_sha256",
            "config_sha256",
            "receipt_content_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )

    @classmethod
    def from_mailbox_key(
        cls, key: LandingOutcomeShotKey
    ) -> "PlacementShotIdentity":
        if not isinstance(key, LandingOutcomeShotKey):
            raise PlacementSuccessContractError(
                "shot identity requires a LandingOutcomeShotKey"
            )
        return cls(**{name: getattr(key, name) for name in _FULL_KEY_FIELDS})

    def to_mailbox_key(self) -> LandingOutcomeShotKey:
        return LandingOutcomeShotKey(
            **{name: getattr(self, name) for name in _FULL_KEY_FIELDS}
        )

    @property
    def task_key(self) -> tuple[object, ...]:
        return tuple(getattr(self, name) for name in _RUNTIME_KEY_FIELDS)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            **{name: getattr(self, name) for name in _FULL_KEY_FIELDS},
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "PlacementShotIdentity":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_SHOT_IDENTITY_KEYS,
            kind=cls.KIND,
            label="placement shot identity",
        )
        result = cls(**{name: payload[name] for name in _FULL_KEY_FIELDS})
        if result.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "shot identity normalization changed canonical SHA"
            )
        return result


_SHOT_IDENTITY_KEYS = frozenset(
    ("schema_version", "kind", *_FULL_KEY_FIELDS)
)


@dataclass(frozen=True)
class InfraCensorEvidence:
    """Closed evidence for a committed row excluded from policy statistics."""

    KIND: ClassVar[str] = INFRA_CENSOR_KIND

    shot_identity_sha256: str
    reason: InfraCensorReason
    detected_step: int
    producer_receipt_sha256: str
    detail_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "shot_identity_sha256",
            "producer_receipt_sha256",
            "detail_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "reason",
            _enum(
                self.reason,
                enum_cls=InfraCensorReason,
                label="infra censor reason",
            ),
        )
        object.__setattr__(
            self,
            "detected_step",
            _plain_int(self.detected_step, label="detected_step"),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "shot_identity_sha256": self.shot_identity_sha256,
            "reason": self.reason.value,
            "detected_step": self.detected_step,
            "producer_receipt_sha256": self.producer_receipt_sha256,
            "detail_sha256": self.detail_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "InfraCensorEvidence":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_INFRA_CENSOR_KEYS,
            kind=cls.KIND,
            label="infra censor evidence",
        )
        result = cls(
            shot_identity_sha256=payload["shot_identity_sha256"],
            reason=payload["reason"],
            detected_step=payload["detected_step"],
            producer_receipt_sha256=payload["producer_receipt_sha256"],
            detail_sha256=payload["detail_sha256"],
        )
        if result.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "infra censor normalization changed canonical SHA"
            )
        return result


_INFRA_CENSOR_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "shot_identity_sha256",
        "reason",
        "detected_step",
        "producer_receipt_sha256",
        "detail_sha256",
    )
)


@dataclass(frozen=True)
class ScheduleRevealEntry:
    """One committed absolute reveal/deadline row with carry witnesses."""

    KIND: ClassVar[str] = SCHEDULE_ENTRY_KIND

    shot: PlacementShotIdentity
    scheduled_ordinal: int
    schedule_authority_sha256: str
    schedule_trace_sha256: str
    scheduled_reveal_step: int
    actual_reveal_step: int
    scheduled_deadline_step: int
    close_step: int
    target_profile_sha256: str
    target_semantic_sha256: str
    target_x_m: float
    target_y_m: float
    episode_lineage_sha256: str
    carry_before_parent_sha256: str
    carry_before_sha256: str
    carry_after_close_parent_sha256: str
    carry_after_close_sha256: str
    clear_counter_signature_sha256: str
    boundary_terminated: bool
    boundary_truncated: bool
    boundary_reset: bool
    boundary_teleported: bool
    infrastructure_valid: bool
    infra_censor: Optional[InfraCensorEvidence]

    def __post_init__(self) -> None:
        if not isinstance(self.shot, PlacementShotIdentity):
            raise PlacementSuccessContractError(
                "schedule entry shot must be PlacementShotIdentity"
            )
        ordinal = _plain_int(self.scheduled_ordinal, label="scheduled_ordinal")
        object.__setattr__(self, "scheduled_ordinal", ordinal)
        if self.shot.shot_index != ordinal + 1:
            raise PlacementSuccessContractError(
                "shot_index must equal scheduled_ordinal + 1"
            )
        for name in (
            "schedule_authority_sha256",
            "schedule_trace_sha256",
            "target_profile_sha256",
            "target_semantic_sha256",
            "episode_lineage_sha256",
            "carry_before_parent_sha256",
            "carry_before_sha256",
            "carry_after_close_parent_sha256",
            "carry_after_close_sha256",
            "clear_counter_signature_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in (
            "scheduled_reveal_step",
            "actual_reveal_step",
            "scheduled_deadline_step",
            "close_step",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        if self.actual_reveal_step != self.scheduled_reveal_step:
            raise PlacementSuccessContractError(
                "actual reveal must equal the authoritative scheduled reveal"
            )
        if not self.scheduled_reveal_step < self.scheduled_deadline_step:
            raise PlacementSuccessContractError(
                "scheduled deadline must follow reveal"
            )
        if self.close_step != self.scheduled_deadline_step:
            raise PlacementSuccessContractError(
                "committed shot must close on scheduled deadline"
            )
        object.__setattr__(
            self, "target_x_m", _finite(self.target_x_m, label="target_x_m")
        )
        object.__setattr__(
            self, "target_y_m", _finite(self.target_y_m, label="target_y_m")
        )
        for name in (
            "boundary_terminated",
            "boundary_truncated",
            "boundary_reset",
            "boundary_teleported",
            "infrastructure_valid",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )
        if any(
            (
                self.boundary_terminated,
                self.boundary_truncated,
                self.boundary_reset,
                self.boundary_teleported,
            )
        ):
            raise PlacementSuccessContractError(
                "shot boundary cannot terminate, truncate, reset, or teleport"
            )
        if self.carry_after_close_parent_sha256 != self.carry_before_sha256:
            raise PlacementSuccessContractError(
                "within-shot carry parent chain differs"
            )
        if self.infrastructure_valid:
            if self.infra_censor is not None:
                raise PlacementSuccessContractError(
                    "infrastructure-valid entry cannot carry censor evidence"
                )
        else:
            if not isinstance(self.infra_censor, InfraCensorEvidence):
                raise PlacementSuccessContractError(
                    "infrastructure-invalid entry requires closed censor evidence"
                )
            if (
                self.infra_censor.shot_identity_sha256
                != self.shot.canonical_sha256
            ):
                raise PlacementSuccessContractError(
                    "infra censor full shot identity differs"
                )
            if not (
                self.scheduled_reveal_step
                <= self.infra_censor.detected_step
                <= self.close_step
            ):
                raise PlacementSuccessContractError(
                    "infra censor detection lies outside the committed row"
                )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "authority_scope": SCHEDULE_AUTHORITY_SCOPE,
            "runtime_authority_frozen": False,
            "shot": self.shot.to_mapping(),
            "scheduled_ordinal": self.scheduled_ordinal,
            "schedule_authority_sha256": self.schedule_authority_sha256,
            "schedule_trace_sha256": self.schedule_trace_sha256,
            "scheduled_reveal_step": self.scheduled_reveal_step,
            "actual_reveal_step": self.actual_reveal_step,
            "scheduled_deadline_step": self.scheduled_deadline_step,
            "close_step": self.close_step,
            "target_profile_sha256": self.target_profile_sha256,
            "target_semantic_sha256": self.target_semantic_sha256,
            "target_x_m": self.target_x_m,
            "target_y_m": self.target_y_m,
            "episode_lineage_sha256": self.episode_lineage_sha256,
            "carry_before_parent_sha256": self.carry_before_parent_sha256,
            "carry_before_sha256": self.carry_before_sha256,
            "carry_after_close_parent_sha256": (
                self.carry_after_close_parent_sha256
            ),
            "carry_after_close_sha256": self.carry_after_close_sha256,
            "clear_counter_signature_sha256": (
                self.clear_counter_signature_sha256
            ),
            "boundary_terminated": self.boundary_terminated,
            "boundary_truncated": self.boundary_truncated,
            "boundary_reset": self.boundary_reset,
            "boundary_teleported": self.boundary_teleported,
            "infrastructure_valid": self.infrastructure_valid,
            "infra_censor": (
                None
                if self.infra_censor is None
                else self.infra_censor.to_mapping()
            ),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "ScheduleRevealEntry":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_SCHEDULE_ENTRY_KEYS,
            kind=cls.KIND,
            label="schedule reveal entry",
        )
        if payload["authority_scope"] != SCHEDULE_AUTHORITY_SCOPE:
            raise PlacementSuccessContractError(
                "schedule entry authority scope differs"
            )
        if payload["runtime_authority_frozen"] is not False:
            raise PlacementSuccessContractError(
                "pre-integration entry cannot claim runtime authority"
            )
        result = cls(
            shot=PlacementShotIdentity.from_mapping(payload["shot"]),
            scheduled_ordinal=payload["scheduled_ordinal"],
            schedule_authority_sha256=payload["schedule_authority_sha256"],
            schedule_trace_sha256=payload["schedule_trace_sha256"],
            scheduled_reveal_step=payload["scheduled_reveal_step"],
            actual_reveal_step=payload["actual_reveal_step"],
            scheduled_deadline_step=payload["scheduled_deadline_step"],
            close_step=payload["close_step"],
            target_profile_sha256=payload["target_profile_sha256"],
            target_semantic_sha256=payload["target_semantic_sha256"],
            target_x_m=payload["target_x_m"],
            target_y_m=payload["target_y_m"],
            episode_lineage_sha256=payload["episode_lineage_sha256"],
            carry_before_parent_sha256=payload["carry_before_parent_sha256"],
            carry_before_sha256=payload["carry_before_sha256"],
            carry_after_close_parent_sha256=(
                payload["carry_after_close_parent_sha256"]
            ),
            carry_after_close_sha256=payload["carry_after_close_sha256"],
            clear_counter_signature_sha256=(
                payload["clear_counter_signature_sha256"]
            ),
            boundary_terminated=payload["boundary_terminated"],
            boundary_truncated=payload["boundary_truncated"],
            boundary_reset=payload["boundary_reset"],
            boundary_teleported=payload["boundary_teleported"],
            infrastructure_valid=payload["infrastructure_valid"],
            infra_censor=(
                None
                if payload["infra_censor"] is None
                else InfraCensorEvidence.from_mapping(payload["infra_censor"])
            ),
        )
        if result.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "schedule entry normalization changed canonical SHA"
            )
        return result


_SCHEDULE_ENTRY_FIELDS = (
    "authority_scope",
    "runtime_authority_frozen",
    "shot",
    "scheduled_ordinal",
    "schedule_authority_sha256",
    "schedule_trace_sha256",
    "scheduled_reveal_step",
    "actual_reveal_step",
    "scheduled_deadline_step",
    "close_step",
    "target_profile_sha256",
    "target_semantic_sha256",
    "target_x_m",
    "target_y_m",
    "episode_lineage_sha256",
    "carry_before_parent_sha256",
    "carry_before_sha256",
    "carry_after_close_parent_sha256",
    "carry_after_close_sha256",
    "clear_counter_signature_sha256",
    "boundary_terminated",
    "boundary_truncated",
    "boundary_reset",
    "boundary_teleported",
    "infrastructure_valid",
    "infra_censor",
)
_SCHEDULE_ENTRY_KEYS = frozenset(
    ("schema_version", "kind", *_SCHEDULE_ENTRY_FIELDS)
)


@dataclass(frozen=True)
class ScheduleRevealLedger:
    """Ordered committed-shot authority for one run; still pre-integration."""

    KIND: ClassVar[str] = SCHEDULE_LEDGER_KIND

    run_id: str
    schedule_authority_sha256: str
    entries: Tuple[ScheduleRevealEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_id", _nonempty_text(self.run_id, label="run_id")
        )
        object.__setattr__(
            self,
            "schedule_authority_sha256",
            _sha256(
                self.schedule_authority_sha256,
                label="schedule_authority_sha256",
            ),
        )
        raw_entries = _sequence(self.entries, label="schedule entries")
        if any(not isinstance(item, ScheduleRevealEntry) for item in raw_entries):
            raise PlacementSuccessContractError(
                "every schedule ledger row must be ScheduleRevealEntry"
            )
        entries = tuple(raw_entries)
        object.__setattr__(self, "entries", entries)
        expected_order = tuple(
            sorted(
                entries,
                key=lambda item: (
                    item.shot.env_id,
                    item.shot.carry_chain_id,
                    item.scheduled_ordinal,
                ),
            )
        )
        if entries != expected_order:
            raise PlacementSuccessContractError(
                "schedule ledger rows must use canonical env/chain/ordinal order"
            )
        if len({item.canonical_sha256 for item in entries}) != len(entries):
            raise PlacementSuccessContractError(
                "schedule ledger entry identity cannot be reused"
            )
        groups: dict[tuple[int, str], list[ScheduleRevealEntry]] = {}
        env_chain_identity: dict[int, tuple[int, str]] = {}
        for entry in entries:
            if entry.shot.run_id != self.run_id:
                raise PlacementSuccessContractError(
                    "schedule entry run_id differs from ledger"
                )
            if entry.schedule_authority_sha256 != self.schedule_authority_sha256:
                raise PlacementSuccessContractError(
                    "schedule entry authority differs from ledger"
                )
            chain_identity = (
                entry.shot.reset_generation,
                entry.shot.carry_chain_id,
            )
            previous_chain_identity = env_chain_identity.setdefault(
                entry.shot.env_id, chain_identity
            )
            if chain_identity != previous_chain_identity:
                raise PlacementSuccessContractError(
                    "an env schedule ledger cannot restart ordinal/reset or "
                    "switch carry chain"
                )
            groups.setdefault(
                (entry.shot.env_id, entry.shot.carry_chain_id), []
            ).append(entry)
        for chain_entries in groups.values():
            anchor = chain_entries[0]
            if anchor.scheduled_ordinal != 0:
                raise PlacementSuccessContractError(
                    "each carry chain must begin at scheduled ordinal zero"
                )
            for index, entry in enumerate(chain_entries):
                shot = entry.shot
                if entry.scheduled_ordinal != index:
                    raise PlacementSuccessContractError(
                        "scheduled ordinals must be contiguous"
                    )
                if shot.shot_index != index + 1:
                    raise PlacementSuccessContractError(
                        "shot indices must be contiguous"
                    )
                for name in (
                    "reset_generation",
                    "action_uid",
                    "action_slot",
                    "birth_sha256",
                    "source_sha256",
                    "config_sha256",
                ):
                    if getattr(shot, name) != getattr(anchor.shot, name):
                        raise PlacementSuccessContractError(
                            f"carry chain {name} changed"
                        )
                if shot.swing_generation != (
                    anchor.shot.swing_generation + index
                ):
                    raise PlacementSuccessContractError(
                        "swing generation must advance exactly once per shot"
                    )
                if entry.episode_lineage_sha256 != anchor.episode_lineage_sha256:
                    raise PlacementSuccessContractError(
                        "episode lineage changed within carry chain"
                    )
                if (
                    entry.clear_counter_signature_sha256
                    != anchor.clear_counter_signature_sha256
                ):
                    raise PlacementSuccessContractError(
                        "reset/clear counter signature changed within carry chain"
                    )
                if entry.target_profile_sha256 != anchor.target_profile_sha256:
                    raise PlacementSuccessContractError(
                        "target schedule profile changed within carry chain"
                    )
                if index > 0:
                    previous = chain_entries[index - 1]
                    if (
                        entry.carry_before_parent_sha256
                        != previous.carry_after_close_sha256
                    ):
                        raise PlacementSuccessContractError(
                            "cross-shot carry parent chain differs"
                        )
                    if not (
                        previous.scheduled_deadline_step
                        < entry.scheduled_reveal_step
                    ):
                        raise PlacementSuccessContractError(
                            "successor reveal did not advance after prior deadline"
                        )
                    if shot.sample_sha256 == previous.shot.sample_sha256:
                        raise PlacementSuccessContractError(
                            "adjacent sample receipt was reused"
                        )
                    if shot.task_sha256 == previous.shot.task_sha256:
                        raise PlacementSuccessContractError(
                            "adjacent task receipt was reused"
                        )
                    if (
                        entry.target_x_m,
                        entry.target_y_m,
                    ) == (
                        previous.target_x_m,
                        previous.target_y_m,
                    ):
                        raise PlacementSuccessContractError(
                            "adjacent numeric targets must differ"
                        )
                    if (
                        entry.target_semantic_sha256
                        == previous.target_semantic_sha256
                    ):
                        raise PlacementSuccessContractError(
                            "adjacent target semantic identity was reused"
                        )

    @property
    def runtime_authority_frozen(self) -> bool:
        return False

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "authority_scope": SCHEDULE_AUTHORITY_SCOPE,
            "runtime_authority_frozen": False,
            "run_id": self.run_id,
            "schedule_authority_sha256": self.schedule_authority_sha256,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "ScheduleRevealLedger":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_SCHEDULE_LEDGER_KEYS,
            kind=cls.KIND,
            label="schedule reveal ledger",
        )
        if payload["authority_scope"] != SCHEDULE_AUTHORITY_SCOPE:
            raise PlacementSuccessContractError(
                "schedule ledger authority scope differs"
            )
        if payload["runtime_authority_frozen"] is not False:
            raise PlacementSuccessContractError(
                "pre-integration ledger cannot claim runtime authority"
            )
        rows = payload["entries"]
        if not isinstance(rows, (list, tuple)):
            raise PlacementSuccessContractError(
                "schedule ledger entries must be a sequence"
            )
        result = cls(
            run_id=payload["run_id"],
            schedule_authority_sha256=payload["schedule_authority_sha256"],
            entries=tuple(ScheduleRevealEntry.from_mapping(row) for row in rows),
        )
        if result.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "schedule ledger normalization changed canonical SHA"
            )
        return result


_SCHEDULE_LEDGER_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "authority_scope",
        "runtime_authority_frozen",
        "run_id",
        "schedule_authority_sha256",
        "entries",
    )
)


@dataclass(frozen=True)
class ConstructionRejection:
    """Pre-reveal construction reject; never a committed policy row."""

    KIND: ClassVar[str] = CONSTRUCTION_REJECTION_KIND

    run_id: str
    carry_chain_id: str
    env_id: int
    before_scheduled_ordinal: int
    evaluated_step: int
    candidate_semantic_sha256: str
    schedule_authority_sha256: str
    reason: ConstructionRejectReason

    def __post_init__(self) -> None:
        for name in ("run_id", "carry_chain_id"):
            object.__setattr__(
                self, name, _nonempty_text(getattr(self, name), label=name)
            )
        for name in ("env_id", "before_scheduled_ordinal", "evaluated_step"):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        for name in (
            "candidate_semantic_sha256",
            "schedule_authority_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "reason",
            _enum(
                self.reason,
                enum_cls=ConstructionRejectReason,
                label="construction rejection reason",
            ),
        )

    @property
    def policy_denominator_increment(self) -> int:
        return 0

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "run_id": self.run_id,
            "carry_chain_id": self.carry_chain_id,
            "env_id": self.env_id,
            "before_scheduled_ordinal": self.before_scheduled_ordinal,
            "evaluated_step": self.evaluated_step,
            "candidate_semantic_sha256": self.candidate_semantic_sha256,
            "schedule_authority_sha256": self.schedule_authority_sha256,
            "reason": self.reason.value,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "ConstructionRejection":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_CONSTRUCTION_REJECTION_KEYS,
            kind=cls.KIND,
            label="construction rejection",
        )
        result = cls(
            run_id=payload["run_id"],
            carry_chain_id=payload["carry_chain_id"],
            env_id=payload["env_id"],
            before_scheduled_ordinal=payload["before_scheduled_ordinal"],
            evaluated_step=payload["evaluated_step"],
            candidate_semantic_sha256=payload["candidate_semantic_sha256"],
            schedule_authority_sha256=payload["schedule_authority_sha256"],
            reason=payload["reason"],
        )
        if result.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "construction rejection normalization changed canonical SHA"
            )
        return result


_CONSTRUCTION_REJECTION_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "run_id",
        "carry_chain_id",
        "env_id",
        "before_scheduled_ordinal",
        "evaluated_step",
        "candidate_semantic_sha256",
        "schedule_authority_sha256",
        "reason",
    )
)


@dataclass(frozen=True)
class PlacementRewardConsumerReceipt:
    """Exactly one eligible row and exactly one mailbox payment consumption."""

    KIND: ClassVar[str] = CONSUMER_RECEIPT_KIND

    consumer_name: str
    shot_identity_sha256: str
    mailbox_payment_idempotency_sha256: str
    source_step: int
    payment_step: int
    eligible_count: int
    payment_count: int
    raw_income: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "consumer_name",
            _nonempty_text(self.consumer_name, label="consumer_name"),
        )
        if self.consumer_name != REWARD_CONSUMER_NAME:
            raise PlacementSuccessContractError(
                "placement consumer name differs"
            )
        for name in (
            "shot_identity_sha256",
            "mailbox_payment_idempotency_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        source = _plain_int(self.source_step, label="source_step")
        payment = _plain_int(
            self.payment_step, label="payment_step", minimum=source
        )
        object.__setattr__(self, "source_step", source)
        object.__setattr__(self, "payment_step", payment)
        for name in ("eligible_count", "payment_count"):
            value = _plain_int(getattr(self, name), label=name)
            object.__setattr__(self, name, value)
            if value != 1:
                raise PlacementSuccessContractError(
                    "mailbox placement consumer requires count 1/1"
                )
        income = _finite_nonnegative(self.raw_income, label="raw_income")
        if income > 1.0:
            raise PlacementSuccessContractError("raw_income must be in [0,1]")
        object.__setattr__(self, "raw_income", income)

    @classmethod
    def from_mailbox_payment(
        cls,
        *,
        shot: PlacementShotIdentity,
        payment: LandingOutcomePayment,
    ) -> "PlacementRewardConsumerReceipt":
        if not isinstance(payment, LandingOutcomePayment):
            raise PlacementSuccessContractError(
                "consumer receipt requires LandingOutcomePayment"
            )
        if PlacementShotIdentity.from_mailbox_key(payment.task_key) != shot:
            raise PlacementSuccessContractError(
                "consumer receipt mailbox full key differs"
            )
        return cls(
            consumer_name=REWARD_CONSUMER_NAME,
            shot_identity_sha256=shot.canonical_sha256,
            mailbox_payment_idempotency_sha256=payment.idempotency_sha256,
            source_step=payment.source_step,
            payment_step=payment.payment_step,
            eligible_count=1,
            payment_count=1,
            raw_income=payment.score.total,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "consumer_name": self.consumer_name,
            "shot_identity_sha256": self.shot_identity_sha256,
            "mailbox_payment_idempotency_sha256": (
                self.mailbox_payment_idempotency_sha256
            ),
            "source_step": self.source_step,
            "payment_step": self.payment_step,
            "eligible_count": self.eligible_count,
            "payment_count": self.payment_count,
            "raw_income": self.raw_income,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "PlacementRewardConsumerReceipt":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_CONSUMER_RECEIPT_KEYS,
            kind=cls.KIND,
            label="placement consumer receipt",
        )
        result = cls(
            consumer_name=payload["consumer_name"],
            shot_identity_sha256=payload["shot_identity_sha256"],
            mailbox_payment_idempotency_sha256=(
                payload["mailbox_payment_idempotency_sha256"]
            ),
            source_step=payload["source_step"],
            payment_step=payload["payment_step"],
            eligible_count=payload["eligible_count"],
            payment_count=payload["payment_count"],
            raw_income=payload["raw_income"],
        )
        if result.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "consumer receipt normalization changed canonical SHA"
            )
        return result


_CONSUMER_RECEIPT_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "consumer_name",
        "shot_identity_sha256",
        "mailbox_payment_idempotency_sha256",
        "source_step",
        "payment_step",
        "eligible_count",
        "payment_count",
        "raw_income",
    )
)


_MAILBOX_FACTORY_TOKEN = object()
_INFRA_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class MailboxPlacementSettlement:
    """Settlement derivable only from a paid authoritative mailbox row."""

    KIND: ClassVar[str] = MAILBOX_SETTLEMENT_KIND

    schedule_entry_sha256: str
    shot: PlacementShotIdentity
    placement_profile: LandingPlacementProfile
    task_identity: LandingPlacementTaskIdentity
    facts: LandingPlacementFacts
    score: LandingPlacementScore
    source_step: int
    settlement_step: int
    payment_step: int
    mailbox_payment_idempotency_sha256: str
    consumer_receipt: PlacementRewardConsumerReceipt
    outcome: CommittedShotOutcome
    raw_target_error_m: Optional[float]
    placement_quality: Optional[float]
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _MAILBOX_FACTORY_TOKEN:
            raise PlacementSuccessContractError(
                "mailbox settlement must use from_mailbox_payment"
            )
        object.__setattr__(
            self,
            "schedule_entry_sha256",
            _sha256(self.schedule_entry_sha256, label="schedule_entry_sha256"),
        )
        if not isinstance(self.shot, PlacementShotIdentity):
            raise PlacementSuccessContractError("mailbox settlement shot differs")
        if not isinstance(self.placement_profile, LandingPlacementProfile):
            raise PlacementSuccessContractError(
                "placement profile must be canonical scorer profile"
            )
        if not isinstance(self.task_identity, LandingPlacementTaskIdentity):
            raise PlacementSuccessContractError(
                "task identity must be canonical placement identity"
            )
        if not isinstance(self.facts, LandingPlacementFacts):
            raise PlacementSuccessContractError(
                "facts must be canonical landing facts"
            )
        if not isinstance(self.score, LandingPlacementScore):
            raise PlacementSuccessContractError(
                "score must be canonical landing score"
            )
        if not isinstance(
            self.consumer_receipt, PlacementRewardConsumerReceipt
        ):
            raise PlacementSuccessContractError(
                "mailbox settlement requires consumer receipt"
            )
        source = _plain_int(self.source_step, label="source_step")
        settled = _plain_int(
            self.settlement_step, label="settlement_step", minimum=source
        )
        paid = _plain_int(
            self.payment_step, label="payment_step", minimum=settled
        )
        object.__setattr__(self, "source_step", source)
        object.__setattr__(self, "settlement_step", settled)
        object.__setattr__(self, "payment_step", paid)
        object.__setattr__(
            self,
            "mailbox_payment_idempotency_sha256",
            _sha256(
                self.mailbox_payment_idempotency_sha256,
                label="mailbox_payment_idempotency_sha256",
            ),
        )
        outcome = _enum(
            self.outcome,
            enum_cls=CommittedShotOutcome,
            label="committed shot outcome",
        )
        if outcome is CommittedShotOutcome.INFRA_CENSOR:
            raise PlacementSuccessContractError(
                "mailbox payment cannot classify an infra censor"
            )
        object.__setattr__(self, "outcome", outcome)
        error_m = _optional_finite_nonnegative(
            self.raw_target_error_m, label="raw_target_error_m"
        )
        quality = _optional_finite_nonnegative(
            self.placement_quality, label="placement_quality"
        )
        object.__setattr__(self, "raw_target_error_m", error_m)
        object.__setattr__(self, "placement_quality", quality)

        profile_sha = self.placement_profile.canonical_sha256
        identity_sha = self.task_identity.canonical_sha256
        facts_sha = self.facts.canonical_sha256
        if self.task_identity.profile_sha256 != profile_sha:
            raise PlacementSuccessContractError("task/profile binding differs")
        if self.task_identity.task_receipt_sha256 != self.shot.task_sha256:
            raise PlacementSuccessContractError("task receipt binding differs")
        if self.task_identity.frame_id != self.placement_profile.frame_id:
            raise PlacementSuccessContractError("task/profile frame differs")
        if (
            self.task_identity.frame_binding_sha256
            != self.placement_profile.frame_binding_sha256
        ):
            raise PlacementSuccessContractError("frame binding SHA differs")
        if self.facts.profile_sha256 != profile_sha:
            raise PlacementSuccessContractError("facts profile differs")
        if self.facts.task_identity_sha256 != identity_sha:
            raise PlacementSuccessContractError("facts task identity differs")
        if self.facts.frame_id != self.placement_profile.frame_id:
            raise PlacementSuccessContractError("facts frame differs")
        if self.score.profile_sha256 != profile_sha:
            raise PlacementSuccessContractError("score profile differs")
        if self.score.task_identity_sha256 != identity_sha:
            raise PlacementSuccessContractError("score task identity differs")
        if self.score.facts_sha256 != facts_sha:
            raise PlacementSuccessContractError("score facts identity differs")
        if self.score.task_receipt_sha256 != self.shot.task_sha256:
            raise PlacementSuccessContractError("score task receipt differs")
        if self.score.frame_id != self.placement_profile.frame_id:
            raise PlacementSuccessContractError("score frame differs")
        if (
            self.score.target_x_m,
            self.score.target_y_m,
        ) != (
            self.task_identity.target_x_m,
            self.task_identity.target_y_m,
        ):
            raise PlacementSuccessContractError("numeric target binding differs")
        expected = _MAILBOX_REASON_TO_OUTCOME.get(self.score.reason)
        if expected is None:
            raise RuntimeAuthorityRequiredError(
                "crossing contract fault must route to closed infra censor, "
                "not a paid policy outcome"
            )
        if outcome is not expected:
            raise PlacementSuccessContractError(
                "outcome differs from canonical mailbox score reason"
            )
        if outcome in _LANDING_OUTCOMES:
            if error_m != self.score.placement_error_m:
                raise PlacementSuccessContractError(
                    "raw target error differs from canonical score"
                )
            if quality != self.score.total:
                raise PlacementSuccessContractError(
                    "placement quality differs from canonical score"
                )
        elif error_m is not None or quality is not None:
            raise PlacementSuccessContractError(
                "non-landing policy outcome cannot carry placement metrics"
            )
        if outcome in _ZERO_INCOME_OUTCOMES and self.score.total != 0.0:
            raise PlacementSuccessContractError(
                "unscored policy failure must have zero raw income"
            )
        consumer = self.consumer_receipt
        if consumer.shot_identity_sha256 != self.shot.canonical_sha256:
            raise PlacementSuccessContractError("consumer shot identity differs")
        if (
            consumer.mailbox_payment_idempotency_sha256
            != self.mailbox_payment_idempotency_sha256
        ):
            raise PlacementSuccessContractError(
                "consumer mailbox payment identity differs"
            )
        if (
            consumer.source_step != source
            or consumer.payment_step != paid
            or consumer.raw_income != self.score.total
        ):
            raise PlacementSuccessContractError(
                "consumer source/payment/raw income differs"
            )

    @classmethod
    def from_mailbox_payment(
        cls,
        *,
        schedule_entry: ScheduleRevealEntry,
        payment: LandingOutcomePayment,
        paid_view: LandingOutcomeView,
        placement_profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        consumer_receipt: PlacementRewardConsumerReceipt,
    ) -> "MailboxPlacementSettlement":
        if not isinstance(schedule_entry, ScheduleRevealEntry):
            raise PlacementSuccessContractError(
                "mailbox settlement requires schedule entry"
            )
        if not schedule_entry.infrastructure_valid:
            raise PlacementSuccessContractError(
                "infra-invalid schedule row cannot consume mailbox payment"
            )
        if not isinstance(payment, LandingOutcomePayment):
            raise PlacementSuccessContractError(
                "settlement requires LandingOutcomePayment"
            )
        if not isinstance(paid_view, LandingOutcomeView):
            raise PlacementSuccessContractError(
                "settlement requires authoritative paid mailbox view"
            )
        if paid_view.state not in (PAID, CLOSED):
            raise PlacementSuccessContractError(
                "mailbox view must be PAID or CLOSED"
            )
        if paid_view.facts is None or paid_view.score is None:
            raise PlacementSuccessContractError(
                "paid mailbox view lacks canonical facts/score"
            )
        shot = PlacementShotIdentity.from_mailbox_key(payment.task_key)
        if shot != schedule_entry.shot:
            raise PlacementSuccessContractError(
                "mailbox payment full key differs from schedule owner"
            )
        if payment.task_key != paid_view.task_key:
            raise PlacementSuccessContractError(
                "mailbox payment/view full key differs"
            )
        if payment.profile_sha256 != paid_view.profile_sha256:
            raise PlacementSuccessContractError(
                "mailbox payment/view profile differs"
            )
        if payment.task_identity_sha256 != paid_view.task_identity_sha256:
            raise PlacementSuccessContractError(
                "mailbox payment/view task identity differs"
            )
        if (
            payment.target_x_m,
            payment.target_y_m,
        ) != (
            paid_view.target_x_m,
            paid_view.target_y_m,
        ):
            raise PlacementSuccessContractError(
                "mailbox payment/view numeric target differs"
            )
        if (
            schedule_entry.target_x_m,
            schedule_entry.target_y_m,
        ) != (
            task_identity.target_x_m,
            task_identity.target_y_m,
        ):
            raise PlacementSuccessContractError(
                "schedule numeric target differs from mailbox task"
            )
        if payment.source_step != paid_view.source_step:
            raise PlacementSuccessContractError(
                "mailbox payment/view source step differs"
            )
        if payment.settlement_step != paid_view.settlement_step:
            raise PlacementSuccessContractError(
                "mailbox payment/view settlement step differs"
            )
        if payment.payment_step != paid_view.payment_step:
            raise PlacementSuccessContractError(
                "mailbox payment/view payment step differs"
            )
        if not (
            schedule_entry.actual_reveal_step
            <= payment.source_step
            <= schedule_entry.close_step
        ):
            raise PlacementSuccessContractError(
                "mailbox source lies outside the authoritative "
                "reveal/close window"
            )
        if payment.idempotency_sha256 != paid_view.payment_idempotency_sha256:
            raise PlacementSuccessContractError(
                "mailbox payment/view idempotency differs"
            )
        if payment.score != paid_view.score:
            raise PlacementSuccessContractError(
                "mailbox payment/view canonical score differs"
            )
        if payment.profile_sha256 != placement_profile.canonical_sha256:
            raise PlacementSuccessContractError(
                "mailbox payment/scorer profile differs"
            )
        if payment.task_identity_sha256 != task_identity.canonical_sha256:
            raise PlacementSuccessContractError(
                "mailbox payment/task identity differs"
            )
        if not isinstance(consumer_receipt, PlacementRewardConsumerReceipt):
            raise PlacementSuccessContractError(
                "settlement requires placement consumer receipt"
            )
        expected_consumer = PlacementRewardConsumerReceipt.from_mailbox_payment(
            shot=shot,
            payment=payment,
        )
        if consumer_receipt != expected_consumer:
            raise PlacementSuccessContractError(
                "consumer receipt differs from mailbox payment"
            )
        outcome = _MAILBOX_REASON_TO_OUTCOME.get(payment.score.reason)
        if outcome is None:
            raise RuntimeAuthorityRequiredError(
                "mailbox contract fault requires infra censor routing"
            )
        error_m = (
            payment.score.placement_error_m
            if outcome in _LANDING_OUTCOMES
            else None
        )
        quality = payment.score.total if outcome in _LANDING_OUTCOMES else None
        return cls(
            schedule_entry_sha256=schedule_entry.canonical_sha256,
            shot=shot,
            placement_profile=placement_profile,
            task_identity=task_identity,
            facts=paid_view.facts,
            score=payment.score,
            source_step=payment.source_step,
            settlement_step=payment.settlement_step,
            payment_step=payment.payment_step,
            mailbox_payment_idempotency_sha256=payment.idempotency_sha256,
            consumer_receipt=consumer_receipt,
            outcome=outcome,
            raw_target_error_m=error_m,
            placement_quality=quality,
            _factory_token=_MAILBOX_FACTORY_TOKEN,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "schedule_entry_sha256": self.schedule_entry_sha256,
            "shot": self.shot.to_mapping(),
            "placement_profile": self.placement_profile.to_mapping(),
            "task_identity": self.task_identity.to_mapping(),
            "facts": self.facts.to_mapping(),
            "score": self.score.to_mapping(),
            "source_step": self.source_step,
            "settlement_step": self.settlement_step,
            "payment_step": self.payment_step,
            "mailbox_payment_idempotency_sha256": (
                self.mailbox_payment_idempotency_sha256
            ),
            "consumer_receipt": self.consumer_receipt.to_mapping(),
            "outcome": self.outcome.value,
            "raw_target_error_m": self.raw_target_error_m,
            "placement_quality": self.placement_quality,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())


@dataclass(frozen=True)
class InfraCensorSettlement:
    """Committed censor closure with no reward eligibility or payment."""

    KIND: ClassVar[str] = INFRA_SETTLEMENT_KIND

    schedule_entry_sha256: str
    shot: PlacementShotIdentity
    censor: InfraCensorEvidence
    eligible_count: int
    payment_count: int
    raw_income: float
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _INFRA_FACTORY_TOKEN:
            raise PlacementSuccessContractError(
                "infra settlement must use from_schedule_entry"
            )
        object.__setattr__(
            self,
            "schedule_entry_sha256",
            _sha256(self.schedule_entry_sha256, label="schedule_entry_sha256"),
        )
        if not isinstance(self.shot, PlacementShotIdentity):
            raise PlacementSuccessContractError("infra settlement shot differs")
        if not isinstance(self.censor, InfraCensorEvidence):
            raise PlacementSuccessContractError(
                "infra settlement requires closed censor evidence"
            )
        if self.censor.shot_identity_sha256 != self.shot.canonical_sha256:
            raise PlacementSuccessContractError(
                "infra settlement full shot key differs"
            )
        for name in ("eligible_count", "payment_count"):
            value = _plain_int(getattr(self, name), label=name)
            object.__setattr__(self, name, value)
            if value != 0:
                raise PlacementSuccessContractError(
                    "infra censor must have zero eligibility/payment"
                )
        income = _finite_nonnegative(self.raw_income, label="raw_income")
        if income != 0.0:
            raise PlacementSuccessContractError(
                "infra censor must have zero raw income"
            )
        object.__setattr__(self, "raw_income", income)

    @classmethod
    def from_schedule_entry(
        cls, entry: ScheduleRevealEntry
    ) -> "InfraCensorSettlement":
        if not isinstance(entry, ScheduleRevealEntry):
            raise PlacementSuccessContractError(
                "infra settlement requires schedule entry"
            )
        if entry.infrastructure_valid or entry.infra_censor is None:
            raise PlacementSuccessContractError(
                "schedule entry is not an infrastructure censor"
            )
        return cls(
            schedule_entry_sha256=entry.canonical_sha256,
            shot=entry.shot,
            censor=entry.infra_censor,
            eligible_count=0,
            payment_count=0,
            raw_income=0.0,
            _factory_token=_INFRA_FACTORY_TOKEN,
        )

    @property
    def outcome(self) -> CommittedShotOutcome:
        return CommittedShotOutcome.INFRA_CENSOR

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "schedule_entry_sha256": self.schedule_entry_sha256,
            "shot": self.shot.to_mapping(),
            "censor": self.censor.to_mapping(),
            "eligible_count": self.eligible_count,
            "payment_count": self.payment_count,
            "raw_income": self.raw_income,
            "outcome": self.outcome.value,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())


PlacementSettlement = Union[MailboxPlacementSettlement, InfraCensorSettlement]


@dataclass(frozen=True)
class PlacementSuccessAggregate:
    """Recomputed diagnostic aggregate; never current curriculum authority."""

    KIND: ClassVar[str] = AGGREGATE_KIND

    run_id: str
    schedule_ledger_sha256: str
    placement_profile_sha256: Optional[str]
    construction_infeasible_count: int
    committed_count: int
    no_contact_count: int
    nonfinite_count: int
    no_crossing_count: int
    net_fail_count: int
    own_or_back_count: int
    opponent_off_table_count: int
    opponent_on_table_count: int
    infra_censor_count: int
    policy_denominator: int
    success_count: int
    measurement_status: MeasurementStatus
    on_table_rate: Optional[float]
    placement_observation_count: int
    raw_target_error_sum_m: float
    raw_target_error_mean_m: Optional[float]
    placement_quality_sum: float
    placement_quality_mean: Optional[float]
    eligible_count: int
    payment_count: int
    raw_income_sum: float
    settlement_root_sha256: str
    construction_rejection_root_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_id", _nonempty_text(self.run_id, label="run_id")
        )
        object.__setattr__(
            self,
            "schedule_ledger_sha256",
            _sha256(self.schedule_ledger_sha256, label="schedule_ledger_sha256"),
        )
        object.__setattr__(
            self,
            "placement_profile_sha256",
            _optional_sha256(
                self.placement_profile_sha256,
                label="placement_profile_sha256",
            ),
        )
        count_fields = (
            "construction_infeasible_count",
            "committed_count",
            "no_contact_count",
            "nonfinite_count",
            "no_crossing_count",
            "net_fail_count",
            "own_or_back_count",
            "opponent_off_table_count",
            "opponent_on_table_count",
            "infra_censor_count",
            "policy_denominator",
            "success_count",
            "placement_observation_count",
            "eligible_count",
            "payment_count",
        )
        for name in count_fields:
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        status = _enum(
            self.measurement_status,
            enum_cls=MeasurementStatus,
            label="measurement_status",
        )
        object.__setattr__(self, "measurement_status", status)
        rate = _optional_finite_nonnegative(
            self.on_table_rate, label="on_table_rate"
        )
        if rate is not None and rate > 1.0:
            raise PlacementSuccessContractError("on_table_rate must be in [0,1]")
        object.__setattr__(self, "on_table_rate", rate)
        for name in (
            "raw_target_error_sum_m",
            "placement_quality_sum",
            "raw_income_sum",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), label=name),
            )
        for name in (
            "raw_target_error_mean_m",
            "placement_quality_mean",
        ):
            value = _optional_finite_nonnegative(getattr(self, name), label=name)
            if name == "placement_quality_mean" and value is not None and value > 1.0:
                raise PlacementSuccessContractError(
                    "placement_quality_mean must be in [0,1]"
                )
            object.__setattr__(self, name, value)
        for name in (
            "settlement_root_sha256",
            "construction_rejection_root_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )

        partition = (
            self.no_contact_count
            + self.nonfinite_count
            + self.no_crossing_count
            + self.net_fail_count
            + self.own_or_back_count
            + self.opponent_off_table_count
            + self.opponent_on_table_count
            + self.infra_censor_count
        )
        if partition != self.committed_count:
            raise PlacementSuccessContractError(
                "outcome partition does not conserve committed rows"
            )
        if self.policy_denominator != self.committed_count - self.infra_censor_count:
            raise PlacementSuccessContractError(
                "policy denominator must exclude only infra censors"
            )
        if self.success_count != self.opponent_on_table_count:
            raise PlacementSuccessContractError(
                "success count must equal opponent-on-table count only"
            )
        if self.success_count > self.policy_denominator:
            raise PlacementSuccessContractError(
                "success count exceeds policy denominator"
            )
        if self.eligible_count != self.policy_denominator:
            raise PlacementSuccessContractError(
                "eligible count must equal policy denominator"
            )
        if self.payment_count != self.eligible_count:
            raise PlacementSuccessContractError(
                "every eligible placement row must pay exactly once"
            )
        placement_count = (
            self.opponent_off_table_count + self.opponent_on_table_count
        )
        if self.placement_observation_count != placement_count:
            raise PlacementSuccessContractError(
                "placement count must equal on/off-table rows"
            )
        if self.policy_denominator == 0:
            if status is not MeasurementStatus.UNMEASURED or rate is not None:
                raise PlacementSuccessContractError(
                    "zero policy denominator must be UNMEASURED"
                )
        else:
            if status is not MeasurementStatus.DIAGNOSTIC_RUNTIME_AUTHORITY_REQUIRED:
                raise PlacementSuccessContractError(
                    "nonzero pre-integration window must remain diagnostic"
                )
            if rate != self.success_count / self.policy_denominator:
                raise PlacementSuccessContractError(
                    "on-table rate differs from numerator/denominator"
                )
        if placement_count == 0:
            if (
                self.raw_target_error_sum_m != 0.0
                or self.placement_quality_sum != 0.0
                or self.raw_target_error_mean_m is not None
                or self.placement_quality_mean is not None
            ):
                raise PlacementSuccessContractError(
                    "zero landing observations require zero sums/null means"
                )
        else:
            if (
                self.raw_target_error_mean_m
                != self.raw_target_error_sum_m / placement_count
            ):
                raise PlacementSuccessContractError(
                    "raw target error mean differs from sum/count"
                )
            if (
                self.placement_quality_mean
                != self.placement_quality_sum / placement_count
            ):
                raise PlacementSuccessContractError(
                    "placement quality mean differs from sum/count"
                )
            maximum_quality = (
                self.opponent_on_table_count
                + 0.5 * self.opponent_off_table_count
            )
            if self.placement_quality_sum > maximum_quality:
                raise PlacementSuccessContractError(
                    "placement quality exceeds table-gate maxima"
                )
        if self.raw_income_sum != self.placement_quality_sum:
            raise PlacementSuccessContractError(
                "raw income must equal canonical placement quality income"
            )

    @property
    def curriculum_authorized(self) -> bool:
        return False

    def curriculum_signal(self) -> None:
        raise RuntimeAuthorityRequiredError(
            "live schedule/reveal runtime authority is not frozen; "
            "diagnostic aggregate cannot drive curriculum"
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.KIND,
            "contract_scope": CONTRACT_SCOPE,
            "runtime_authority_frozen": False,
            "curriculum_authorized": False,
            "success_definition": SUCCESS_DEFINITION,
            "zero_denominator_semantics": ZERO_DENOMINATOR_SEMANTICS,
            "curriculum_metric_names": list(CURRICULUM_METRIC_NAMES),
            **{
                name: (
                    getattr(self, name).value
                    if name == "measurement_status"
                    else getattr(self, name)
                )
                for name in _AGGREGATE_FIELDS
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        schedule_ledger: ScheduleRevealLedger,
        settlements: Sequence[PlacementSettlement],
        construction_rejections: Sequence[ConstructionRejection] = (),
    ) -> "PlacementSuccessAggregate":
        payload, declared = _verified_payload(
            value,
            expected_payload_keys=_AGGREGATE_KEYS,
            kind=cls.KIND,
            label="placement success aggregate",
        )
        if payload["contract_scope"] != CONTRACT_SCOPE:
            raise PlacementSuccessContractError("aggregate scope differs")
        if payload["runtime_authority_frozen"] is not False:
            raise PlacementSuccessContractError(
                "aggregate cannot claim runtime authority"
            )
        if payload["curriculum_authorized"] is not False:
            raise PlacementSuccessContractError(
                "pre-integration aggregate cannot authorize curriculum"
            )
        if payload["success_definition"] != SUCCESS_DEFINITION:
            raise PlacementSuccessContractError(
                "aggregate success definition differs"
            )
        if (
            payload["zero_denominator_semantics"]
            != ZERO_DENOMINATOR_SEMANTICS
        ):
            raise PlacementSuccessContractError(
                "aggregate zero-denominator semantics differ"
            )
        if payload["curriculum_metric_names"] != list(CURRICULUM_METRIC_NAMES):
            raise PlacementSuccessContractError(
                "aggregate curriculum metric allow-list differs"
            )
        candidate = cls(**{name: payload[name] for name in _AGGREGATE_FIELDS})
        expected = validate_placement_success_aggregate(
            schedule_ledger=schedule_ledger,
            settlements=settlements,
            construction_rejections=construction_rejections,
        )
        if candidate != expected or candidate.canonical_sha256 != declared:
            raise PlacementSuccessContractError(
                "aggregate differs from recomputed authority rows"
            )
        return candidate


_AGGREGATE_FIELDS = (
    "run_id",
    "schedule_ledger_sha256",
    "placement_profile_sha256",
    "construction_infeasible_count",
    "committed_count",
    "no_contact_count",
    "nonfinite_count",
    "no_crossing_count",
    "net_fail_count",
    "own_or_back_count",
    "opponent_off_table_count",
    "opponent_on_table_count",
    "infra_censor_count",
    "policy_denominator",
    "success_count",
    "measurement_status",
    "on_table_rate",
    "placement_observation_count",
    "raw_target_error_sum_m",
    "raw_target_error_mean_m",
    "placement_quality_sum",
    "placement_quality_mean",
    "eligible_count",
    "payment_count",
    "raw_income_sum",
    "settlement_root_sha256",
    "construction_rejection_root_sha256",
)
_AGGREGATE_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "contract_scope",
        "runtime_authority_frozen",
        "curriculum_authorized",
        "success_definition",
        "zero_denominator_semantics",
        "curriculum_metric_names",
        *_AGGREGATE_FIELDS,
    )
)


_COUNT_FIELD_BY_OUTCOME = {
    CommittedShotOutcome.NO_CONTACT: "no_contact_count",
    CommittedShotOutcome.NONFINITE: "nonfinite_count",
    CommittedShotOutcome.NO_CROSSING: "no_crossing_count",
    CommittedShotOutcome.NET_FAIL: "net_fail_count",
    CommittedShotOutcome.OWN_OR_BACK: "own_or_back_count",
    CommittedShotOutcome.OPPONENT_OFF_TABLE: "opponent_off_table_count",
    CommittedShotOutcome.OPPONENT_ON_TABLE: "opponent_on_table_count",
    CommittedShotOutcome.INFRA_CENSOR: "infra_censor_count",
}


def validate_placement_success_aggregate(
    *,
    schedule_ledger: ScheduleRevealLedger,
    settlements: Sequence[PlacementSettlement],
    construction_rejections: Sequence[ConstructionRejection] = (),
) -> PlacementSuccessAggregate:
    """Recompute receipt roots, partition, payments, and diagnostic metrics."""

    if not isinstance(schedule_ledger, ScheduleRevealLedger):
        raise PlacementSuccessContractError(
            "aggregate requires ScheduleRevealLedger"
        )
    closed = _sequence(settlements, label="settlements")
    rejected = _sequence(
        construction_rejections, label="construction_rejections"
    )
    if any(
        not isinstance(item, (MailboxPlacementSettlement, InfraCensorSettlement))
        for item in closed
    ):
        raise PlacementSuccessContractError(
            "settlement rows must come from mailbox or infra factories"
        )
    if any(not isinstance(item, ConstructionRejection) for item in rejected):
        raise PlacementSuccessContractError(
            "construction rejection row type differs"
        )

    entries_by_sha = {
        entry.canonical_sha256: entry for entry in schedule_ledger.entries
    }
    if len(entries_by_sha) != len(schedule_ledger.entries):
        raise PlacementSuccessContractError("schedule entry identity reused")
    settlements_by_entry: dict[str, PlacementSettlement] = {}
    payment_ids: set[str] = set()
    consumer_ids: set[str] = set()
    fact_ids: set[str] = set()
    score_ids: set[str] = set()
    profiles: set[str] = set()
    for settlement in closed:
        entry = entries_by_sha.get(settlement.schedule_entry_sha256)
        if entry is None or settlement.shot != entry.shot:
            raise PlacementSuccessContractError(
                "settlement does not match authoritative schedule row"
            )
        if settlement.schedule_entry_sha256 in settlements_by_entry:
            raise PlacementSuccessContractError(
                "committed schedule row has multiple settlements"
            )
        if entry.infrastructure_valid:
            if not isinstance(settlement, MailboxPlacementSettlement):
                raise PlacementSuccessContractError(
                    "valid schedule row requires mailbox settlement"
                )
            if settlement.mailbox_payment_idempotency_sha256 in payment_ids:
                raise PlacementSuccessContractError(
                    "mailbox payment identity reused"
                )
            if settlement.consumer_receipt.canonical_sha256 in consumer_ids:
                raise PlacementSuccessContractError(
                    "reward consumer receipt reused"
                )
            if settlement.facts.canonical_sha256 in fact_ids:
                raise PlacementSuccessContractError(
                    "landing facts identity reused"
                )
            if settlement.score.canonical_sha256 in score_ids:
                raise PlacementSuccessContractError(
                    "landing score identity reused"
                )
            payment_ids.add(settlement.mailbox_payment_idempotency_sha256)
            consumer_ids.add(settlement.consumer_receipt.canonical_sha256)
            fact_ids.add(settlement.facts.canonical_sha256)
            score_ids.add(settlement.score.canonical_sha256)
            profiles.add(settlement.placement_profile.canonical_sha256)
        else:
            if not isinstance(settlement, InfraCensorSettlement):
                raise PlacementSuccessContractError(
                    "infra-invalid schedule row requires censor settlement"
                )
            if settlement.censor != entry.infra_censor:
                raise PlacementSuccessContractError(
                    "infra settlement evidence differs from schedule row"
                )
        settlements_by_entry[settlement.schedule_entry_sha256] = settlement
    missing = sorted(set(entries_by_sha) - set(settlements_by_entry))
    if missing:
        raise PlacementSuccessContractError(
            "every committed schedule row must close exactly once; missing="
            + repr(missing)
        )
    if len(profiles) > 1:
        raise PlacementSuccessContractError(
            "aggregate mailbox settlements must use one placement profile"
        )

    rejection_ids: set[str] = set()
    candidate_ids: set[str] = set()
    entry_lookup = {
        (
            entry.shot.env_id,
            entry.shot.carry_chain_id,
            entry.scheduled_ordinal,
        ): entry
        for entry in schedule_ledger.entries
    }
    for rejection in rejected:
        if rejection.run_id != schedule_ledger.run_id:
            raise PlacementSuccessContractError(
                "construction rejection run differs"
            )
        if (
            rejection.schedule_authority_sha256
            != schedule_ledger.schedule_authority_sha256
        ):
            raise PlacementSuccessContractError(
                "construction rejection schedule authority differs"
            )
        owner = entry_lookup.get(
            (
                rejection.env_id,
                rejection.carry_chain_id,
                rejection.before_scheduled_ordinal,
            )
        )
        if owner is None:
            raise PlacementSuccessContractError(
                "construction rejection has no scheduled ordinal owner"
            )
        if rejection.evaluated_step >= owner.scheduled_reveal_step:
            raise PlacementSuccessContractError(
                "construction infeasible record was not pre-reveal"
            )
        if rejection.candidate_semantic_sha256 in candidate_ids:
            raise PlacementSuccessContractError(
                "construction candidate identity reused"
            )
        if rejection.candidate_semantic_sha256 in {
            entry.target_semantic_sha256 for entry in schedule_ledger.entries
        }:
            raise PlacementSuccessContractError(
                "rejected construction candidate became committed target"
            )
        if rejection.canonical_sha256 in rejection_ids:
            raise PlacementSuccessContractError(
                "construction rejection identity reused"
            )
        candidate_ids.add(rejection.candidate_semantic_sha256)
        rejection_ids.add(rejection.canonical_sha256)

    counts = {name: 0 for name in _COUNT_FIELD_BY_OUTCOME.values()}
    errors: list[float] = []
    qualities: list[float] = []
    eligible_count = 0
    payment_count = 0
    raw_incomes: list[float] = []
    for settlement in settlements_by_entry.values():
        counts[_COUNT_FIELD_BY_OUTCOME[settlement.outcome]] += 1
        if isinstance(settlement, MailboxPlacementSettlement):
            eligible_count += settlement.consumer_receipt.eligible_count
            payment_count += settlement.consumer_receipt.payment_count
            raw_incomes.append(settlement.consumer_receipt.raw_income)
            if settlement.outcome in _LANDING_OUTCOMES:
                assert settlement.raw_target_error_m is not None
                assert settlement.placement_quality is not None
                errors.append(settlement.raw_target_error_m)
                qualities.append(settlement.placement_quality)

    committed_count = len(schedule_ledger.entries)
    policy_denominator = committed_count - counts["infra_censor_count"]
    success_count = counts["opponent_on_table_count"]
    if policy_denominator == 0:
        status = MeasurementStatus.UNMEASURED
        rate = None
    else:
        status = MeasurementStatus.DIAGNOSTIC_RUNTIME_AUTHORITY_REQUIRED
        rate = success_count / policy_denominator
    placement_count = len(errors)
    error_sum = math.fsum(errors)
    quality_sum = math.fsum(qualities)

    return PlacementSuccessAggregate(
        run_id=schedule_ledger.run_id,
        schedule_ledger_sha256=schedule_ledger.canonical_sha256,
        placement_profile_sha256=(next(iter(profiles)) if profiles else None),
        construction_infeasible_count=len(rejected),
        committed_count=committed_count,
        **counts,
        policy_denominator=policy_denominator,
        success_count=success_count,
        measurement_status=status,
        on_table_rate=rate,
        placement_observation_count=placement_count,
        raw_target_error_sum_m=error_sum,
        raw_target_error_mean_m=(
            None if placement_count == 0 else error_sum / placement_count
        ),
        placement_quality_sum=quality_sum,
        placement_quality_mean=(
            None if placement_count == 0 else quality_sum / placement_count
        ),
        eligible_count=eligible_count,
        payment_count=payment_count,
        raw_income_sum=math.fsum(raw_incomes),
        settlement_root_sha256=_record_root(
            "action_ball_placement_settlement_root_v2",
            tuple(
                settlement.canonical_sha256
                for settlement in settlements_by_entry.values()
            ),
        ),
        construction_rejection_root_sha256=_record_root(
            "action_ball_placement_construction_rejection_root_v2",
            tuple(rejection_ids),
        ),
    )
