"""Dependency-light v2 contract for continuous ActionBall successor shots.

This module describes a deterministic four-shot acceptance tape.  The
scheduler is the only timing authority: reveal and deadline steps are derived
from a schedule frozen before Q0.  Recovery readiness is an observed fact and
can never move a reveal or deadline.  A shot that is not ready at reveal is
still committed, closes as recovery_unavailable on its original deadline, and
the next scheduled shot still occurs.

The contract also separates construction-infeasible candidates from committed
opportunities and infrastructure-invalid censors; binds float32 target
semantics, ball lifecycle, recovery teacher/reference payments, rollout/GAE/
recurrent continuity, pre-reveal hiding, late outcome ownership, and a
mid-sequence exact-resume checkpoint.  It remains a pre-integration contract:
it does not mutate a simulator and cannot claim that a runtime is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from numbers import Real
import struct
from typing import ClassVar, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 2
MAX_ACTION_UID = (1 << 53) - 1
CONTRACT_SCOPE = "dependency_light_runtime_preintegration_only_v2"
CLOCK_KIND = "episode_tick_v1"
RUNTIME_TARGET_DTYPE = "float32"
RUNTIME_TASK_REF_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
)
CLOSE_REASONS = frozenset(
    ("hit", "policy_miss", "recovery_unavailable", "infra_invalid")
)
INFRA_CENSOR_REASONS = frozenset(
    ("none", "nonfinite_state", "engine_overflow", "receipt_fault")
)
CONSTRUCTION_REASONS = frozenset(
    ("solver_infeasible", "kinematic_infeasible", "target_support_empty")
)
OUTCOME_KINDS = frozenset(
    ("first_landing_placement", "outgoing_ball_state")
)
BALL_SETTLEMENT_REASONS = frozenset(
    ("owned_physical_outcome", "flight_horizon_zero")
)
CHECKPOINT_OUTCOME_STATES = frozenset(("pending", "settled_unpaid"))
SOURCE_EVENT_KINDS = frozenset(
    ("selected_rubber_contact", "first_landing_plane_crossing")
)
CHECKPOINT_PHASES = frozenset(
    (
        "active_opportunity",
        "recovery_hidden",
        "ready_hold",
        "recovery_unavailable",
        "infra_censored",
    )
)
ADMISSION_DECISIONS = frozenset(("admitted",))


class ContinuousSuccessorContractError(ValueError):
    """The supplied tape violates the v2 successor-shot contract."""


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _optional_int(
    value: object, *, label: str, minimum: int = 0
) -> Optional[int]:
    if value is None:
        return None
    return _plain_int(value, label=label, minimum=minimum)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an exact bool")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _closed_text(
    value: object, *, label: str, allowed: frozenset[str]
) -> str:
    result = _nonempty_text(value, label=label)
    if result not in allowed:
        raise ValueError(f"{label} is not in the closed enum")
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _optional_sha256(value: object, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _sha256(value, label=label)


def _float32(value: object, *, label: str) -> float:
    finite = _finite_number(value, label=label)
    try:
        result = struct.unpack("!f", struct.pack("!f", finite))[0]
    except OverflowError as error:
        raise ValueError(f"{label} is outside float32 range") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is outside finite float32 range")
    return 0.0 if result == 0.0 else float(result)


def target_semantic_sha256(
    profile_sha256: object, target_x_m: object, target_y_m: object
) -> str:
    profile = _sha256(profile_sha256, label="profile_sha256")
    x_value = _float32(target_x_m, label="target_x_m")
    y_value = _float32(target_y_m, label="target_y_m")
    return canonical_sha256(
        {
            "kind": "action_ball_runtime_target_semantics_v1",
            "profile_sha256": profile,
            "runtime_dtype": RUNTIME_TARGET_DTYPE,
            "target_x_m": x_value,
            "target_y_m": y_value,
        }
    )


def checkpoint_mailbox_sha256(entries: Sequence[object]) -> str:
    values = tuple(entries)
    for index, entry in enumerate(values):
        if (
            not isinstance(entry, _SealedRecord)
            or entry.KIND != "action_ball_checkpoint_outcome_mailbox_entry_v2"
        ):
            raise ValueError(
                f"mailbox entry {index} must be a typed checkpoint entry"
            )
    return canonical_sha256(
        {
            "kind": "action_ball_unpaid_outcome_mailbox_v2",
            "entries": [entry.to_mapping() for entry in values],
        }
    )


def _encode(value: object) -> object:
    if isinstance(value, _SealedRecord):
        return value.to_mapping()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_sha256(payload)
    return result


def _verified_values(
    value: object, *, cls: type, kind: str, label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    field_names = tuple(field.name for field in fields(cls))
    expected_payload = frozenset(("schema_version", "kind", *field_names))
    expected = expected_payload | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    declared = _sha256(value["canonical_sha256"], label="canonical_sha256")
    payload = {key: value[key] for key in expected_payload}
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version differs")
    if payload["kind"] != kind:
        raise ValueError(f"{label} kind differs")
    if canonical_sha256(payload) != declared:
        raise ValueError(f"{label} canonical SHA differs")
    return {name: payload[name] for name in field_names}


def _instance(value: object, cls: type, *, label: str):
    if not isinstance(value, cls):
        raise ValueError(f"{label} must be {cls.__name__}")
    return value


class _SealedRecord:
    KIND: ClassVar[str]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
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
            label=cls.__name__,
        )


@dataclass(frozen=True)
class ContinuousActionTaskReceiptRef(_SealedRecord):
    """Field-for-field equivalent of the runtime ActionTaskReceiptRef."""

    KIND: ClassVar[str] = "action_ball_continuous_task_receipt_ref_v2"

    env_id: int
    reset_generation: int
    swing_generation: int
    action_uid: int
    action_slot: int
    birth_sha256: str
    sample_sha256: str
    task_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, label="ref.env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                label="ref.reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "swing_generation",
            _plain_int(self.swing_generation, label="ref.swing_generation"),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                label="ref.action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, label="ref.action_slot"),
        )
        for name in ("birth_sha256", "sample_sha256", "task_sha256"):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), label=f"ref.{name}"),
            )

    def runtime_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in RUNTIME_TASK_REF_FIELDS}

    @classmethod
    def from_runtime_mapping(
        cls, value: object
    ) -> "ContinuousActionTaskReceiptRef":
        if not isinstance(value, Mapping):
            raise ValueError("runtime task receipt ref must be a mapping")
        expected = frozenset(RUNTIME_TASK_REF_FIELDS)
        actual = frozenset(value)
        if actual != expected:
            raise ValueError(
                "runtime task receipt ref keys differ: "
                f"missing={sorted(expected - actual)!r}, "
                f"unknown={sorted(actual - expected)!r}"
            )
        return cls(**{name: value[name] for name in RUNTIME_TASK_REF_FIELDS})

    @classmethod
    def from_runtime_ref(
        cls, value: object
    ) -> "ContinuousActionTaskReceiptRef":
        return cls(
            **{
                name: getattr(value, name)
                for name in RUNTIME_TASK_REF_FIELDS
            }
        )

    @classmethod
    def from_mapping(
        cls, value: object
    ) -> "ContinuousActionTaskReceiptRef":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class FrozenCadenceReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_frozen_cadence_receipt_v2"

    clock_kind: str
    clock_epoch_sha256: str
    schedule_authority_sha256: str
    frozen_at_step: int
    sequence_origin_step: int
    first_reveal_step: int
    cadence_steps: int
    deadline_offset_steps: int
    scheduled_shot_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clock_kind",
            _nonempty_text(self.clock_kind, label="clock_kind"),
        )
        for name in ("clock_epoch_sha256", "schedule_authority_sha256"):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in (
            "frozen_at_step",
            "sequence_origin_step",
            "first_reveal_step",
            "cadence_steps",
            "deadline_offset_steps",
            "scheduled_shot_count",
        ):
            minimum = 1 if name in (
                "cadence_steps",
                "deadline_offset_steps",
                "scheduled_shot_count",
            ) else 0
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name, minimum=minimum),
            )
        if self.clock_kind != CLOCK_KIND:
            raise ValueError("clock_kind differs")
        if not (
            self.sequence_origin_step
            <= self.frozen_at_step
            < self.first_reveal_step
        ):
            raise ValueError("schedule must freeze before first reveal")
        if self.deadline_offset_steps >= self.cadence_steps:
            raise ValueError("deadline offset must be smaller than cadence")

    def reveal_step(self, ordinal: int) -> int:
        index = _plain_int(ordinal, label="ordinal")
        return self.first_reveal_step + index * self.cadence_steps

    def deadline_step(self, ordinal: int) -> int:
        return self.reveal_step(ordinal) + self.deadline_offset_steps

    @classmethod
    def from_mapping(cls, value: object) -> "FrozenCadenceReceipt":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class TargetSelectionReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_target_selection_receipt_v2"

    profile_sha256: str
    selection_authority_sha256: str
    runtime_dtype: str
    target_generation: int
    task_ref_sha256: str
    requested_target_x_m: float
    requested_target_y_m: float
    runtime_target_x_m: float
    runtime_target_y_m: float
    semantic_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "profile_sha256",
            "selection_authority_sha256",
            "task_ref_sha256",
            "semantic_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "runtime_dtype",
            _nonempty_text(self.runtime_dtype, label="runtime_dtype"),
        )
        if self.runtime_dtype != RUNTIME_TARGET_DTYPE:
            raise ValueError("runtime target dtype differs")
        object.__setattr__(
            self,
            "target_generation",
            _plain_int(self.target_generation, label="target_generation"),
        )
        requested_x = _finite_number(
            self.requested_target_x_m, label="requested_target_x_m"
        )
        requested_y = _finite_number(
            self.requested_target_y_m, label="requested_target_y_m"
        )
        runtime_x = _float32(
            self.runtime_target_x_m, label="runtime_target_x_m"
        )
        runtime_y = _float32(
            self.runtime_target_y_m, label="runtime_target_y_m"
        )
        expected_x = _float32(requested_x, label="requested_target_x_m")
        expected_y = _float32(requested_y, label="requested_target_y_m")
        if (runtime_x, runtime_y) != (expected_x, expected_y):
            raise ValueError("runtime target differs from float32 request")
        object.__setattr__(self, "requested_target_x_m", requested_x)
        object.__setattr__(self, "requested_target_y_m", requested_y)
        object.__setattr__(self, "runtime_target_x_m", runtime_x)
        object.__setattr__(self, "runtime_target_y_m", runtime_y)
        expected_semantic = target_semantic_sha256(
            self.profile_sha256, runtime_x, runtime_y
        )
        if self.semantic_sha256 != expected_semantic:
            raise ValueError("target semantic SHA differs")

    @property
    def runtime_target_xy_m(self) -> Tuple[float, float]:
        return (self.runtime_target_x_m, self.runtime_target_y_m)

    @classmethod
    def from_mapping(cls, value: object) -> "TargetSelectionReceipt":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class PreRevealHiddenWitness(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_pre_reveal_hidden_witness_v2"

    hidden_from_step: int
    hidden_through_step: int
    observed_tick_count: int
    leak_tick_count: int
    first_visible_step: int
    future_task_ref_visible: bool
    future_target_visible: bool
    future_inbound_ball_visible: bool
    future_strike_deadline_visible: bool
    future_question_teacher_visible: bool
    future_action_visible: bool
    pre_reveal_actor_chain_sha256: str
    pre_reveal_critic_chain_sha256: str
    reveal_actor_observation_sha256: str
    reveal_critic_observation_sha256: str
    hidden_mask_authority_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hidden_from_step",
            _plain_int(self.hidden_from_step, label="hidden_from_step"),
        )
        for name in (
            "hidden_through_step",
            "observed_tick_count",
            "leak_tick_count",
            "first_visible_step",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        for name in (
            "future_task_ref_visible",
            "future_target_visible",
            "future_inbound_ball_visible",
            "future_strike_deadline_visible",
            "future_question_teacher_visible",
            "future_action_visible",
        ):
            object.__setattr__(
                self,
                name,
                _exact_bool(getattr(self, name), label=name),
            )
        for name in (
            "pre_reveal_actor_chain_sha256",
            "pre_reveal_critic_chain_sha256",
            "reveal_actor_observation_sha256",
            "reveal_critic_observation_sha256",
            "hidden_mask_authority_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )

    @property
    def all_future_facts_hidden(self) -> bool:
        return self.leak_tick_count == 0 and not any(
            (
                self.future_task_ref_visible,
                self.future_target_visible,
                self.future_inbound_ball_visible,
                self.future_strike_deadline_visible,
                self.future_question_teacher_visible,
                self.future_action_visible,
            )
        )

    @classmethod
    def from_mapping(cls, value: object) -> "PreRevealHiddenWitness":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class CarryContinuityWitness(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_carry_continuity_witness_v2"

    episode_step: int
    parent_witness_sha256: str
    episode_lineage_sha256: str
    reset_generation: int
    robot_state_sha256: str
    last_executed_action_sha256: str
    action_history_sha256: str
    observation_history_sha256: str
    target_delay_state_sha256: str
    noise_state_sha256: str
    recurrent_state_sha256: str
    gae_state_sha256: str
    return_bootstrap_sha256: str
    rollout_chain_sha256: str
    root_reset_count: int
    joint_reset_count: int
    velocity_reset_count: int
    action_history_clear_count: int
    observation_history_clear_count: int
    target_delay_clear_count: int
    noise_state_clear_count: int
    recurrent_state_clear_count: int
    gae_state_clear_count: int
    rollout_reset_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_step",
            _plain_int(self.episode_step, label="episode_step"),
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                label="reset_generation",
                minimum=1,
            ),
        )
        sha_names = (
            "parent_witness_sha256",
            "episode_lineage_sha256",
            "robot_state_sha256",
            "last_executed_action_sha256",
            "action_history_sha256",
            "observation_history_sha256",
            "target_delay_state_sha256",
            "noise_state_sha256",
            "recurrent_state_sha256",
            "gae_state_sha256",
            "return_bootstrap_sha256",
            "rollout_chain_sha256",
        )
        for name in sha_names:
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in self.clear_counter_names:
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )

    @property
    def clear_counter_names(self) -> Tuple[str, ...]:
        return (
            "root_reset_count",
            "joint_reset_count",
            "velocity_reset_count",
            "action_history_clear_count",
            "observation_history_clear_count",
            "target_delay_clear_count",
            "noise_state_clear_count",
            "recurrent_state_clear_count",
            "gae_state_clear_count",
            "rollout_reset_count",
        )

    @property
    def clear_counter_signature(self) -> Tuple[int, ...]:
        return tuple(getattr(self, name) for name in self.clear_counter_names)

    @classmethod
    def from_mapping(cls, value: object) -> "CarryContinuityWitness":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class BallSettlementReceipt(_SealedRecord):
    """Typed closure for the prior physical ball, including zero outcomes."""

    KIND: ClassVar[str] = "action_ball_ball_settlement_receipt_v2"

    owner_task_ref: ContinuousActionTaskReceiptRef
    ball_sha256: str
    settlement_reason: str
    source_step: int
    settlement_step: int
    settlement_state_sha256: str
    settlement_authority_sha256: str
    owned_outcome_event_sha256: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owner_task_ref",
            _instance(
                self.owner_task_ref,
                ContinuousActionTaskReceiptRef,
                label="owner_task_ref",
            ),
        )
        for name in (
            "ball_sha256",
            "settlement_state_sha256",
            "settlement_authority_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "owned_outcome_event_sha256",
            _optional_sha256(
                self.owned_outcome_event_sha256,
                label="owned_outcome_event_sha256",
            ),
        )
        object.__setattr__(
            self,
            "settlement_reason",
            _closed_text(
                self.settlement_reason,
                label="settlement_reason",
                allowed=BALL_SETTLEMENT_REASONS,
            ),
        )
        for name in ("source_step", "settlement_step"):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        if self.source_step > self.settlement_step:
            raise ValueError("ball settlement source follows settlement")
        if (
            self.settlement_reason == "owned_physical_outcome"
            and self.owned_outcome_event_sha256 is None
        ):
            raise ValueError("owned physical settlement lacks outcome event")
        if (
            self.settlement_reason == "flight_horizon_zero"
            and self.owned_outcome_event_sha256 is not None
        ):
            raise ValueError("zero closure unexpectedly names outcome event")

    @classmethod
    def from_mapping(cls, value: object) -> "BallSettlementReceipt":
        values = cls._mapping_values(value)
        values["owner_task_ref"] = ContinuousActionTaskReceiptRef.from_mapping(
            values["owner_task_ref"]
        )
        return cls(**values)


@dataclass(frozen=True)
class BallLifecycleReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_ball_lifecycle_receipt_v2"

    ball_generation: int
    inbound_ball_sha256: str
    installed_ball_state_sha256: str
    owner_task_ref_sha256: str
    installed_at_step: int
    prior_ball_sha256: Optional[str]
    prior_ball_retired: bool
    prior_ball_settlement: Optional[BallSettlementReceipt]
    prior_ball_retire_step: Optional[int]
    stale_contact_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ball_generation",
            _plain_int(self.ball_generation, label="ball_generation"),
        )
        for name in (
            "inbound_ball_sha256",
            "installed_ball_state_sha256",
            "owner_task_ref_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "installed_at_step",
            _plain_int(self.installed_at_step, label="installed_at_step"),
        )
        object.__setattr__(
            self,
            "prior_ball_sha256",
            _optional_sha256(self.prior_ball_sha256, label="prior_ball_sha256"),
        )
        object.__setattr__(
            self,
            "prior_ball_retired",
            _exact_bool(self.prior_ball_retired, label="prior_ball_retired"),
        )
        if self.prior_ball_settlement is not None:
            object.__setattr__(
                self,
                "prior_ball_settlement",
                _instance(
                    self.prior_ball_settlement,
                    BallSettlementReceipt,
                    label="prior_ball_settlement",
                ),
            )
        object.__setattr__(
            self,
            "prior_ball_retire_step",
            _optional_int(
                self.prior_ball_retire_step,
                label="prior_ball_retire_step",
            ),
        )
        object.__setattr__(
            self,
            "stale_contact_count",
            _plain_int(self.stale_contact_count, label="stale_contact_count"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "BallLifecycleReceipt":
        values = cls._mapping_values(value)
        raw = values["prior_ball_settlement"]
        values["prior_ball_settlement"] = (
            None if raw is None else BallSettlementReceipt.from_mapping(raw)
        )
        return cls(**values)


@dataclass(frozen=True)
class CommittedOpportunityReceipt(_SealedRecord):
    """Admission proof for the question that actually consumed a cadence slot."""

    KIND: ClassVar[str] = "action_ball_committed_opportunity_receipt_v2"

    task_ref_sha256: str
    target_semantic_sha256: str
    evaluated_step: int
    construction_feasible: bool
    admitted: bool
    admission_decision: str
    feasibility_authority_sha256: str
    solver_receipt_sha256: str
    support_receipt_sha256: str
    infrastructure_valid_at_admission: bool

    def __post_init__(self) -> None:
        for name in (
            "task_ref_sha256",
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
        for name in (
            "construction_feasible",
            "admitted",
            "infrastructure_valid_at_admission",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "admission_decision",
            _closed_text(
                self.admission_decision,
                label="admission_decision",
                allowed=ADMISSION_DECISIONS,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "CommittedOpportunityReceipt":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class ContinuousShotTrace(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_shot_trace_v2"

    shot_label: str
    scheduled_ordinal: int
    task_ref: ContinuousActionTaskReceiptRef
    target: TargetSelectionReceipt
    admission: CommittedOpportunityReceipt
    scheduled_reveal_step: int
    scheduled_deadline_step: int
    close_step: int
    committed: bool
    infrastructure_valid: bool
    infra_censor_reason: str
    ready_met_at_reveal: bool
    hit: bool
    close_reason: str
    closed: bool
    boundary_terminated: bool
    boundary_truncated: bool
    boundary_reset: bool
    boundary_teleported: bool
    pre_reveal_hidden: PreRevealHiddenWitness
    carry_before_reveal: CarryContinuityWitness
    carry_after_reveal: CarryContinuityWitness
    carry_after_close: CarryContinuityWitness
    ball: BallLifecycleReceipt

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shot_label",
            _nonempty_text(self.shot_label, label="shot_label"),
        )
        object.__setattr__(
            self,
            "scheduled_ordinal",
            _plain_int(self.scheduled_ordinal, label="scheduled_ordinal"),
        )
        for name, cls in (
            ("task_ref", ContinuousActionTaskReceiptRef),
            ("target", TargetSelectionReceipt),
            ("admission", CommittedOpportunityReceipt),
            ("pre_reveal_hidden", PreRevealHiddenWitness),
            ("carry_before_reveal", CarryContinuityWitness),
            ("carry_after_reveal", CarryContinuityWitness),
            ("carry_after_close", CarryContinuityWitness),
            ("ball", BallLifecycleReceipt),
        ):
            object.__setattr__(
                self,
                name,
                _instance(getattr(self, name), cls, label=name),
            )
        for name in (
            "scheduled_reveal_step",
            "scheduled_deadline_step",
            "close_step",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        for name in (
            "committed",
            "infrastructure_valid",
            "ready_met_at_reveal",
            "hit",
            "closed",
            "boundary_terminated",
            "boundary_truncated",
            "boundary_reset",
            "boundary_teleported",
        ):
            object.__setattr__(
                self,
                name,
                _exact_bool(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "infra_censor_reason",
            _closed_text(
                self.infra_censor_reason,
                label="infra_censor_reason",
                allowed=INFRA_CENSOR_REASONS,
            ),
        )
        object.__setattr__(
            self,
            "close_reason",
            _closed_text(
                self.close_reason,
                label="close_reason",
                allowed=CLOSE_REASONS,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousShotTrace":
        values = cls._mapping_values(value)
        values["task_ref"] = ContinuousActionTaskReceiptRef.from_mapping(
            values["task_ref"]
        )
        values["target"] = TargetSelectionReceipt.from_mapping(values["target"])
        values["admission"] = CommittedOpportunityReceipt.from_mapping(
            values["admission"]
        )
        values["pre_reveal_hidden"] = PreRevealHiddenWitness.from_mapping(
            values["pre_reveal_hidden"]
        )
        for name in (
            "carry_before_reveal",
            "carry_after_reveal",
            "carry_after_close",
        ):
            values[name] = CarryContinuityWitness.from_mapping(values[name])
        values["ball"] = BallLifecycleReceipt.from_mapping(values["ball"])
        return cls(**values)


@dataclass(frozen=True)
class ConstructionInfeasibleRecord(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_construction_infeasible_record_v2"

    before_scheduled_ordinal: int
    evaluated_step: int
    candidate_semantic_sha256: str
    rejection_reason: str
    committed: bool
    opportunity_created: bool
    infrastructure_valid: bool
    swing_generation_before: int
    swing_generation_after: int

    def __post_init__(self) -> None:
        for name in (
            "before_scheduled_ordinal",
            "evaluated_step",
            "swing_generation_before",
            "swing_generation_after",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "candidate_semantic_sha256",
            _sha256(
                self.candidate_semantic_sha256,
                label="candidate_semantic_sha256",
            ),
        )
        object.__setattr__(
            self,
            "rejection_reason",
            _closed_text(
                self.rejection_reason,
                label="rejection_reason",
                allowed=CONSTRUCTION_REASONS,
            ),
        )
        for name in ("committed", "opportunity_created", "infrastructure_valid"):
            object.__setattr__(
                self,
                name,
                _exact_bool(getattr(self, name), label=name),
            )

    @classmethod
    def from_mapping(cls, value: object) -> "ConstructionInfeasibleRecord":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class RecoveryWindowEvidence(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_recovery_window_evidence_v2"

    owner_task_ref: ContinuousActionTaskReceiptRef
    successor_task_ref: ContinuousActionTaskReceiptRef
    schedule_sha256: str
    window_start_step: int
    observation_end_step: int
    scheduled_successor_reveal_step: int
    teacher_suffix_sha256: str
    reference_sha256: str
    payment_ledger_sha256: str
    teacher_eligible_tick_count: int
    reference_eligible_tick_count: int
    recovery_eligible_tick_count: int
    conjunction_eligible_tick_count: int
    reward_payment_count: int
    reward_total: float
    ready_conjunction_tick_count: int
    first_ready_step: Optional[int]
    hold_ready_tick_count: int
    ready_met_at_scheduled_reveal: bool
    infrastructure_valid: bool

    def __post_init__(self) -> None:
        for name in ("owner_task_ref", "successor_task_ref"):
            object.__setattr__(
                self,
                name,
                _instance(
                    getattr(self, name),
                    ContinuousActionTaskReceiptRef,
                    label=name,
                ),
            )
        for name in (
            "schedule_sha256",
            "teacher_suffix_sha256",
            "reference_sha256",
            "payment_ledger_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in (
            "window_start_step",
            "observation_end_step",
            "scheduled_successor_reveal_step",
            "teacher_eligible_tick_count",
            "reference_eligible_tick_count",
            "recovery_eligible_tick_count",
            "conjunction_eligible_tick_count",
            "reward_payment_count",
            "ready_conjunction_tick_count",
            "hold_ready_tick_count",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "first_ready_step",
            _optional_int(self.first_ready_step, label="first_ready_step"),
        )
        object.__setattr__(
            self,
            "reward_total",
            _finite_number(self.reward_total, label="reward_total"),
        )
        for name in (
            "ready_met_at_scheduled_reveal",
            "infrastructure_valid",
        ):
            object.__setattr__(
                self,
                name,
                _exact_bool(getattr(self, name), label=name),
            )

    @classmethod
    def from_mapping(cls, value: object) -> "RecoveryWindowEvidence":
        values = cls._mapping_values(value)
        values["owner_task_ref"] = ContinuousActionTaskReceiptRef.from_mapping(
            values["owner_task_ref"]
        )
        values[
            "successor_task_ref"
        ] = ContinuousActionTaskReceiptRef.from_mapping(
            values["successor_task_ref"]
        )
        return cls(**values)


@dataclass(frozen=True)
class DelayedOutcomeTrace(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_delayed_outcome_trace_v2"

    outcome_kind: str
    source_event_kind: str
    source_event_sha256: str
    owner_task_ref: ContinuousActionTaskReceiptRef
    owner_target_semantic_sha256: str
    source_step: int
    settlement_step: Optional[int]
    settlement_payload_sha256: Optional[str]
    payment_step: Optional[int]
    payment_count: int
    active_task_ref_sha256_at_settlement: Optional[str]
    active_task_ref_sha256_at_payment: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_kind",
            _closed_text(
                self.outcome_kind,
                label="outcome_kind",
                allowed=OUTCOME_KINDS,
            ),
        )
        object.__setattr__(
            self,
            "source_event_kind",
            _closed_text(
                self.source_event_kind,
                label="source_event_kind",
                allowed=SOURCE_EVENT_KINDS,
            ),
        )
        expected_source_kind = {
            "first_landing_placement": "first_landing_plane_crossing",
            "outgoing_ball_state": "selected_rubber_contact",
        }[self.outcome_kind]
        if self.source_event_kind != expected_source_kind:
            raise ValueError("outcome/source event kind pairing differs")
        object.__setattr__(
            self,
            "source_event_sha256",
            _sha256(self.source_event_sha256, label="source_event_sha256"),
        )
        object.__setattr__(
            self,
            "owner_task_ref",
            _instance(
                self.owner_task_ref,
                ContinuousActionTaskReceiptRef,
                label="owner_task_ref",
            ),
        )
        object.__setattr__(
            self,
            "owner_target_semantic_sha256",
            _sha256(
                self.owner_target_semantic_sha256,
                label="owner_target_semantic_sha256",
            ),
        )
        object.__setattr__(
            self,
            "source_step",
            _plain_int(self.source_step, label="source_step"),
        )
        for name in ("settlement_step", "payment_step"):
            object.__setattr__(
                self,
                name,
                _optional_int(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "settlement_payload_sha256",
            _optional_sha256(
                self.settlement_payload_sha256,
                label="settlement_payload_sha256",
            ),
        )
        object.__setattr__(
            self,
            "payment_count",
            _plain_int(self.payment_count, label="payment_count"),
        )
        for name in (
            "active_task_ref_sha256_at_settlement",
            "active_task_ref_sha256_at_payment",
        ):
            object.__setattr__(
                self,
                name,
                _optional_sha256(getattr(self, name), label=name),
            )

    @property
    def event_identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "kind": "action_ball_delayed_outcome_event_identity_v2",
                "outcome_kind": self.outcome_kind,
                "source_event_kind": self.source_event_kind,
                "source_event_sha256": self.source_event_sha256,
                "owner_task_ref_sha256": self.owner_task_ref.canonical_sha256,
                "owner_target_semantic_sha256": (
                    self.owner_target_semantic_sha256
                ),
            }
        )

    @property
    def state(self) -> str:
        if self.payment_step is not None:
            return "paid"
        if self.settlement_step is not None:
            return "settled"
        return "pending"

    @classmethod
    def from_mapping(cls, value: object) -> "DelayedOutcomeTrace":
        values = cls._mapping_values(value)
        values["owner_task_ref"] = ContinuousActionTaskReceiptRef.from_mapping(
            values["owner_task_ref"]
        )
        return cls(**values)


@dataclass(frozen=True)
class CheckpointOutcomeMailboxEntry(_SealedRecord):
    """Exact unpaid-outcome state present at one checkpoint."""

    KIND: ClassVar[str] = "action_ball_checkpoint_outcome_mailbox_entry_v2"

    event_identity_sha256: str
    outcome_kind: str
    source_event_kind: str
    source_event_sha256: str
    owner_task_ref_sha256: str
    owner_target_semantic_sha256: str
    source_step: int
    checkpoint_state: str
    settlement_step_at_checkpoint: Optional[int]
    settlement_payload_sha256_at_checkpoint: Optional[str]
    payment_step_at_checkpoint: Optional[int]
    payment_count_at_checkpoint: int

    def __post_init__(self) -> None:
        for name in (
            "event_identity_sha256",
            "source_event_sha256",
            "owner_task_ref_sha256",
            "owner_target_semantic_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "outcome_kind",
            _closed_text(
                self.outcome_kind,
                label="outcome_kind",
                allowed=OUTCOME_KINDS,
            ),
        )
        object.__setattr__(
            self,
            "source_event_kind",
            _closed_text(
                self.source_event_kind,
                label="source_event_kind",
                allowed=SOURCE_EVENT_KINDS,
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_state",
            _closed_text(
                self.checkpoint_state,
                label="checkpoint_state",
                allowed=CHECKPOINT_OUTCOME_STATES,
            ),
        )
        object.__setattr__(
            self,
            "source_step",
            _plain_int(self.source_step, label="source_step"),
        )
        object.__setattr__(
            self,
            "settlement_step_at_checkpoint",
            _optional_int(
                self.settlement_step_at_checkpoint,
                label="settlement_step_at_checkpoint",
            ),
        )
        object.__setattr__(
            self,
            "settlement_payload_sha256_at_checkpoint",
            _optional_sha256(
                self.settlement_payload_sha256_at_checkpoint,
                label="settlement_payload_sha256_at_checkpoint",
            ),
        )
        object.__setattr__(
            self,
            "payment_step_at_checkpoint",
            _optional_int(
                self.payment_step_at_checkpoint,
                label="payment_step_at_checkpoint",
            ),
        )
        object.__setattr__(
            self,
            "payment_count_at_checkpoint",
            _plain_int(
                self.payment_count_at_checkpoint,
                label="payment_count_at_checkpoint",
            ),
        )
        if (
            self.payment_step_at_checkpoint is not None
            or self.payment_count_at_checkpoint != 0
        ):
            raise ValueError("unpaid checkpoint entry has payment")
        if self.checkpoint_state == "pending":
            if (
                self.settlement_step_at_checkpoint is not None
                or self.settlement_payload_sha256_at_checkpoint is not None
            ):
                raise ValueError("pending checkpoint entry carries settlement")
        elif (
            self.settlement_step_at_checkpoint is None
            or self.settlement_payload_sha256_at_checkpoint is None
        ):
            raise ValueError("settled-unpaid entry lacks settlement payload")

    @classmethod
    def from_mapping(cls, value: object) -> "CheckpointOutcomeMailboxEntry":
        return cls(**cls._mapping_values(value))


def checkpoint_mailbox_entries(
    outcomes: Sequence[DelayedOutcomeTrace], checkpoint_step: int
) -> Tuple[CheckpointOutcomeMailboxEntry, ...]:
    """Project final traces onto information actually present at checkpoint."""

    step = _plain_int(checkpoint_step, label="checkpoint_step")
    entries = []
    for index, outcome in enumerate(outcomes):
        _instance(
            outcome,
            DelayedOutcomeTrace,
            label=f"outcomes[{index}]",
        )
        if outcome.source_step > step:
            continue
        if outcome.payment_step is not None and outcome.payment_step <= step:
            continue
        settled = (
            outcome.settlement_step is not None
            and outcome.settlement_step <= step
        )
        entries.append(
            CheckpointOutcomeMailboxEntry(
                event_identity_sha256=outcome.event_identity_sha256,
                outcome_kind=outcome.outcome_kind,
                source_event_kind=outcome.source_event_kind,
                source_event_sha256=outcome.source_event_sha256,
                owner_task_ref_sha256=(
                    outcome.owner_task_ref.canonical_sha256
                ),
                owner_target_semantic_sha256=(
                    outcome.owner_target_semantic_sha256
                ),
                source_step=outcome.source_step,
                checkpoint_state=(
                    "settled_unpaid" if settled else "pending"
                ),
                settlement_step_at_checkpoint=(
                    outcome.settlement_step if settled else None
                ),
                settlement_payload_sha256_at_checkpoint=(
                    outcome.settlement_payload_sha256 if settled else None
                ),
                payment_step_at_checkpoint=None,
                payment_count_at_checkpoint=0,
            )
        )
    return tuple(sorted(entries, key=lambda item: item.event_identity_sha256))


@dataclass(frozen=True)
class RecoveryProgressReceipt(_SealedRecord):
    """Checkpoint prefix for a recovery window; counts cannot be replayed twice."""

    KIND: ClassVar[str] = "action_ball_recovery_progress_receipt_v2"

    owner_task_ref_sha256: str
    recovery_window_sha256: str
    observed_through_step: int
    observed_tick_count: int
    teacher_eligible_tick_count: int
    reference_eligible_tick_count: int
    recovery_eligible_tick_count: int
    conjunction_eligible_tick_count: int
    reward_payment_count: int
    reward_total: float
    ready_conjunction_tick_count: int
    hold_ready_tick_count: int
    ready_latched: bool
    first_ready_step: Optional[int]
    payment_ledger_prefix_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "owner_task_ref_sha256",
            "recovery_window_sha256",
            "payment_ledger_prefix_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in (
            "observed_through_step",
            "observed_tick_count",
            "teacher_eligible_tick_count",
            "reference_eligible_tick_count",
            "recovery_eligible_tick_count",
            "conjunction_eligible_tick_count",
            "reward_payment_count",
            "ready_conjunction_tick_count",
            "hold_ready_tick_count",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "reward_total",
            _finite_number(self.reward_total, label="reward_total"),
        )
        object.__setattr__(
            self,
            "ready_latched",
            _exact_bool(self.ready_latched, label="ready_latched"),
        )
        object.__setattr__(
            self,
            "first_ready_step",
            _optional_int(self.first_ready_step, label="first_ready_step"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "RecoveryProgressReceipt":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class ContinuousCheckpointReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_checkpoint_receipt_v2"

    contract_authority_sha256: str
    schedule_sha256: str
    checkpoint_step: int
    active_phase: str
    active_ordinal: Optional[int]
    active_task_ref: Optional[ContinuousActionTaskReceiptRef]
    last_closed_task_ref: ContinuousActionTaskReceiptRef
    next_scheduled_ordinal: int
    remaining_to_deadline_steps: int
    remaining_to_next_reveal_steps: int
    next_scheduled_reveal_step: int
    next_question_visible: bool
    sampler_rng_sha256: str
    current_carry: CarryContinuityWitness
    strike_latched: bool
    strike_latch_owner_task_ref_sha256: Optional[str]
    active_recovery_owner_task_ref_sha256: Optional[str]
    recovery_progress: RecoveryProgressReceipt
    current_ball_owner_task_ref_sha256: str
    current_ball_generation: int
    current_ball_state_sha256: str
    current_ball_retired: bool
    ball_contact_latched: bool
    ball_contact_latch_owner_task_ref_sha256: Optional[str]
    unpaid_outcome_entries: Tuple[CheckpointOutcomeMailboxEntry, ...]
    mailbox_sha256: str
    expected_replay_suffix_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "contract_authority_sha256",
            "schedule_sha256",
            "sampler_rng_sha256",
            "current_ball_owner_task_ref_sha256",
            "current_ball_state_sha256",
            "mailbox_sha256",
            "expected_replay_suffix_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "active_ordinal",
            _optional_int(self.active_ordinal, label="active_ordinal"),
        )
        for name in (
            "checkpoint_step",
            "next_scheduled_ordinal",
            "remaining_to_deadline_steps",
            "remaining_to_next_reveal_steps",
            "next_scheduled_reveal_step",
            "current_ball_generation",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "active_phase",
            _closed_text(
                self.active_phase,
                label="active_phase",
                allowed=CHECKPOINT_PHASES,
            ),
        )
        if self.active_task_ref is not None:
            object.__setattr__(
                self,
                "active_task_ref",
                _instance(
                    self.active_task_ref,
                    ContinuousActionTaskReceiptRef,
                    label="active_task_ref",
                ),
            )
        object.__setattr__(
            self,
            "last_closed_task_ref",
            _instance(
                self.last_closed_task_ref,
                ContinuousActionTaskReceiptRef,
                label="last_closed_task_ref",
            ),
        )
        for name, cls in (
            ("current_carry", CarryContinuityWitness),
            ("recovery_progress", RecoveryProgressReceipt),
        ):
            object.__setattr__(
                self,
                name,
                _instance(getattr(self, name), cls, label=name),
            )
        for name in (
            "next_question_visible",
            "strike_latched",
            "current_ball_retired",
            "ball_contact_latched",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )
        for name in (
            "strike_latch_owner_task_ref_sha256",
            "active_recovery_owner_task_ref_sha256",
            "ball_contact_latch_owner_task_ref_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _optional_sha256(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "unpaid_outcome_entries",
            tuple(
                _instance(
                    item,
                    CheckpointOutcomeMailboxEntry,
                    label="unpaid outcome entry",
                )
                for item in self.unpaid_outcome_entries
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        expected_contract_authority_sha256: str,
        expected_checkpoint_sha256: str,
    ) -> "ContinuousCheckpointReceipt":
        expected = _sha256(
            expected_contract_authority_sha256,
            label="expected_contract_authority_sha256",
        )
        values = cls._mapping_values(value)
        if values["contract_authority_sha256"] != expected:
            raise ValueError("checkpoint contract authority differs")
        active_raw = values["active_task_ref"]
        values["active_task_ref"] = (
            None
            if active_raw is None
            else ContinuousActionTaskReceiptRef.from_mapping(active_raw)
        )
        values["last_closed_task_ref"] = (
            ContinuousActionTaskReceiptRef.from_mapping(
                values["last_closed_task_ref"]
            )
        )
        values["current_carry"] = CarryContinuityWitness.from_mapping(
            values["current_carry"]
        )
        values["recovery_progress"] = RecoveryProgressReceipt.from_mapping(
            values["recovery_progress"]
        )
        unpaid = values["unpaid_outcome_entries"]
        if not isinstance(unpaid, Sequence) or isinstance(unpaid, (str, bytes)):
            raise ValueError("unpaid outcome entries must be a sequence")
        values["unpaid_outcome_entries"] = tuple(
            CheckpointOutcomeMailboxEntry.from_mapping(item)
            for item in unpaid
        )
        result = cls(**values)
        expected_checkpoint = _sha256(
            expected_checkpoint_sha256,
            label="expected_checkpoint_sha256",
        )
        if result.canonical_sha256 != expected_checkpoint:
            raise ValueError("checkpoint differs from externally pinned state")
        return result


@dataclass(frozen=True)
class CheckpointReplayEvidence(_SealedRecord):
    """Observed uninterrupted-vs-restored suffix equality, not a prediction."""

    KIND: ClassVar[str] = "action_ball_checkpoint_replay_evidence_v2"

    checkpoint_sha256: str
    save_count: int
    load_count: int
    replay_start_step: int
    replay_end_step: int
    saved_sampler_rng_sha256: str
    restored_sampler_rng_sha256: str
    restored_carry_sha256: str
    restored_ball_state_sha256: str
    restored_recovery_progress_sha256: str
    restored_mailbox_sha256: str
    uninterrupted_observer_authority_sha256: str
    restored_observer_authority_sha256: str
    uninterrupted_suffix_sha256: str
    resumed_suffix_sha256: str
    uninterrupted_rng_draw_chain_sha256: str
    resumed_rng_draw_chain_sha256: str
    first_post_restore_sample_sha256: str
    bit_exact: bool

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "saved_sampler_rng_sha256",
            "restored_sampler_rng_sha256",
            "restored_carry_sha256",
            "restored_ball_state_sha256",
            "restored_recovery_progress_sha256",
            "restored_mailbox_sha256",
            "uninterrupted_observer_authority_sha256",
            "restored_observer_authority_sha256",
            "uninterrupted_suffix_sha256",
            "resumed_suffix_sha256",
            "uninterrupted_rng_draw_chain_sha256",
            "resumed_rng_draw_chain_sha256",
            "first_post_restore_sample_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in (
            "save_count",
            "load_count",
            "replay_start_step",
            "replay_end_step",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "bit_exact",
            _exact_bool(self.bit_exact, label="bit_exact"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "CheckpointReplayEvidence":
        return cls(**cls._mapping_values(value))


def continuous_replay_suffix_sha256(
    shots: Sequence[ContinuousShotTrace],
    recoveries: Sequence[RecoveryWindowEvidence],
    outcomes: Sequence[DelayedOutcomeTrace],
    checkpoint_step: int,
) -> str:
    step = _plain_int(checkpoint_step, label="checkpoint_step")
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_replay_suffix_v2",
            "checkpoint_step": step,
            "shots": [
                shot.to_mapping()
                for shot in shots
                if shot.scheduled_deadline_step >= step
            ],
            "recoveries": [
                recovery.to_mapping()
                for recovery in recoveries
                if recovery.observation_end_step >= step
            ],
            "outcomes": [
                outcome.to_mapping()
                for outcome in outcomes
                if (
                    outcome.source_step >= step
                    or outcome.payment_step is None
                    or outcome.payment_step > step
                )
            ],
        }
    )


def continuous_contract_authority_sha256(
    schedule: FrozenCadenceReceipt,
    shots: Sequence[ContinuousShotTrace],
    recoveries: Sequence[RecoveryWindowEvidence],
    initial_ready_authority_sha256: str,
) -> str:
    """Seal immutable cadence/question/teacher facts for external pinning."""

    schedule_value = _instance(
        schedule, FrozenCadenceReceipt, label="schedule"
    )
    ready_authority = _sha256(
        initial_ready_authority_sha256,
        label="initial_ready_authority_sha256",
    )
    shot_values = tuple(shots)
    recovery_values = tuple(recoveries)
    for index, shot in enumerate(shot_values):
        _instance(shot, ContinuousShotTrace, label=f"shots[{index}]")
    for index, recovery in enumerate(recovery_values):
        _instance(
            recovery,
            RecoveryWindowEvidence,
            label=f"recoveries[{index}]",
        )
    return canonical_sha256(
        {
            "kind": "action_ball_continuous_contract_authority_v2",
            "schedule": schedule_value.to_mapping(),
            "initial_ready_authority_sha256": ready_authority,
            "questions": [
                {
                    "task_ref": shot.task_ref.to_mapping(),
                    "target": shot.target.to_mapping(),
                    "admission": shot.admission.to_mapping(),
                    "scheduled_ordinal": shot.scheduled_ordinal,
                    "scheduled_reveal_step": shot.scheduled_reveal_step,
                    "scheduled_deadline_step": shot.scheduled_deadline_step,
                    "ball_generation": shot.ball.ball_generation,
                    "inbound_ball_sha256": shot.ball.inbound_ball_sha256,
                    "installed_ball_state_sha256": (
                        shot.ball.installed_ball_state_sha256
                    ),
                    "ball_owner_task_ref_sha256": (
                        shot.ball.owner_task_ref_sha256
                    ),
                    "ball_installed_at_step": shot.ball.installed_at_step,
                    "hidden_mask_authority_sha256": (
                        shot.pre_reveal_hidden.hidden_mask_authority_sha256
                    ),
                }
                for shot in shot_values
            ],
            "recovery_authorities": [
                {
                    "owner_task_ref": recovery.owner_task_ref.to_mapping(),
                    "successor_task_ref": (
                        recovery.successor_task_ref.to_mapping()
                    ),
                    "schedule_sha256": recovery.schedule_sha256,
                    "scheduled_successor_reveal_step": (
                        recovery.scheduled_successor_reveal_step
                    ),
                    "window_start_step": recovery.window_start_step,
                    "observation_end_step": recovery.observation_end_step,
                    "teacher_suffix_sha256": recovery.teacher_suffix_sha256,
                    "reference_sha256": recovery.reference_sha256,
                    "payment_ledger_sha256": recovery.payment_ledger_sha256,
                }
                for recovery in recovery_values
            ],
        }
    )


@dataclass(frozen=True)
class ContinuousSuccessorTrace(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_successor_trace_v2"

    schedule: FrozenCadenceReceipt
    shots: Tuple[ContinuousShotTrace, ...]
    construction_rejections: Tuple[ConstructionInfeasibleRecord, ...]
    recoveries: Tuple[RecoveryWindowEvidence, ...]
    delayed_outcomes: Tuple[DelayedOutcomeTrace, ...]
    checkpoint: ContinuousCheckpointReceipt
    checkpoint_replay: CheckpointReplayEvidence
    contract_authority_sha256: str
    initial_ready_met: bool
    initial_ready_authority_sha256: str
    sequence_end_step: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule",
            _instance(
                self.schedule,
                FrozenCadenceReceipt,
                label="schedule",
            ),
        )
        tuple_specs = (
            ("shots", ContinuousShotTrace),
            ("construction_rejections", ConstructionInfeasibleRecord),
            ("recoveries", RecoveryWindowEvidence),
            ("delayed_outcomes", DelayedOutcomeTrace),
        )
        for name, cls in tuple_specs:
            values = tuple(getattr(self, name))
            for index, item in enumerate(values):
                _instance(item, cls, label=f"{name}[{index}]")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "checkpoint",
            _instance(
                self.checkpoint,
                ContinuousCheckpointReceipt,
                label="checkpoint",
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_replay",
            _instance(
                self.checkpoint_replay,
                CheckpointReplayEvidence,
                label="checkpoint_replay",
            ),
        )
        object.__setattr__(
            self,
            "contract_authority_sha256",
            _sha256(
                self.contract_authority_sha256,
                label="contract_authority_sha256",
            ),
        )
        object.__setattr__(
            self,
            "initial_ready_met",
            _exact_bool(self.initial_ready_met, label="initial_ready_met"),
        )
        object.__setattr__(
            self,
            "initial_ready_authority_sha256",
            _sha256(
                self.initial_ready_authority_sha256,
                label="initial_ready_authority_sha256",
            ),
        )
        object.__setattr__(
            self,
            "sequence_end_step",
            _plain_int(self.sequence_end_step, label="sequence_end_step"),
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        expected_contract_authority_sha256: str,
        expected_checkpoint_sha256: str,
        expected_replay_evidence_sha256: str,
        expected_trace_sha256: str,
    ) -> "ContinuousSuccessorTrace":
        expected = _sha256(
            expected_contract_authority_sha256,
            label="expected_contract_authority_sha256",
        )
        values = cls._mapping_values(value)
        if values["contract_authority_sha256"] != expected:
            raise ValueError("trace contract authority differs")
        values["schedule"] = FrozenCadenceReceipt.from_mapping(
            values["schedule"]
        )
        sequence_specs = (
            ("shots", ContinuousShotTrace),
            ("construction_rejections", ConstructionInfeasibleRecord),
            ("recoveries", RecoveryWindowEvidence),
            ("delayed_outcomes", DelayedOutcomeTrace),
        )
        for name, item_cls in sequence_specs:
            raw = values[name]
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"{name} must be a sequence")
            values[name] = tuple(item_cls.from_mapping(item) for item in raw)
        values["checkpoint"] = ContinuousCheckpointReceipt.from_mapping(
            values["checkpoint"],
            expected_contract_authority_sha256=expected,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        values["checkpoint_replay"] = CheckpointReplayEvidence.from_mapping(
            values["checkpoint_replay"]
        )
        result = cls(**values)
        validate_continuous_successor(
            result,
            expected_contract_authority_sha256=expected,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_replay_evidence_sha256=(
                expected_replay_evidence_sha256
            ),
            expected_trace_sha256=expected_trace_sha256,
        )
        return result


@dataclass(frozen=True)
class ContinuousSuccessorStateReceipt(_SealedRecord):
    KIND: ClassVar[str] = "action_ball_continuous_state_receipt_v2"

    trace_sha256: str
    contract_authority_sha256: str
    schedule_sha256: str
    birth_sha256: str
    env_id: int
    reset_generation: int
    action_uid: int
    action_slot: int
    first_swing_generation: int
    final_swing_generation: int
    sequence_end_step: int
    committed_shot_count: int
    successor_count: int
    construction_infeasible_count: int
    opportunity_count: int
    strike_eligible_count: int
    hit_count: int
    policy_miss_count: int
    recovery_unavailable_count: int
    infra_censored_count: int
    closed_count: int
    pre_reveal_hidden_count: int
    recovery_window_count: int
    recovery_payment_count: int
    ready_met_successor_count: int
    ball_retire_count: int
    stale_contact_count: int
    delayed_outcome_count: int
    paid_outcome_count: int
    late_outcome_count: int
    unpaid_at_checkpoint_count: int
    checkpoint_save_count: int
    checkpoint_load_count: int
    clear_counter_change_count: int
    contract_scope: str
    runtime_integrated: bool
    environment_fixed: bool
    launch_authorized: bool

    def __post_init__(self) -> None:
        for name in (
            "trace_sha256",
            "contract_authority_sha256",
            "schedule_sha256",
            "birth_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        minimum_one = (
            "reset_generation",
            "action_uid",
            "committed_shot_count",
        )
        int_names = (
            "env_id",
            "reset_generation",
            "action_uid",
            "action_slot",
            "first_swing_generation",
            "final_swing_generation",
            "sequence_end_step",
            "committed_shot_count",
            "successor_count",
            "construction_infeasible_count",
            "opportunity_count",
            "strike_eligible_count",
            "hit_count",
            "policy_miss_count",
            "recovery_unavailable_count",
            "infra_censored_count",
            "closed_count",
            "pre_reveal_hidden_count",
            "recovery_window_count",
            "recovery_payment_count",
            "ready_met_successor_count",
            "ball_retire_count",
            "stale_contact_count",
            "delayed_outcome_count",
            "paid_outcome_count",
            "late_outcome_count",
            "unpaid_at_checkpoint_count",
            "checkpoint_save_count",
            "checkpoint_load_count",
            "clear_counter_change_count",
        )
        for name in int_names:
            maximum = MAX_ACTION_UID if name == "action_uid" else None
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    label=name,
                    minimum=1 if name in minimum_one else 0,
                    maximum=maximum,
                ),
            )
        object.__setattr__(
            self,
            "contract_scope",
            _nonempty_text(self.contract_scope, label="contract_scope"),
        )
        for name in (
            "runtime_integrated",
            "environment_fixed",
            "launch_authorized",
        ):
            object.__setattr__(
                self,
                name,
                _exact_bool(getattr(self, name), label=name),
            )
        if self.contract_scope != CONTRACT_SCOPE:
            raise ValueError("contract_scope differs")
        if self.runtime_integrated or self.environment_fixed or self.launch_authorized:
            raise ValueError(
                "pre-integration receipt cannot claim runtime, environment, or launch"
            )
        if self.successor_count != self.committed_shot_count - 1:
            raise ValueError("successor_count identity differs")
        if self.closed_count != self.committed_shot_count:
            raise ValueError("closed_count identity differs")
        if (
            self.hit_count
            + self.policy_miss_count
            + self.recovery_unavailable_count
            + self.infra_censored_count
            != self.committed_shot_count
        ):
            raise ValueError("close reason counts do not partition committed shots")
        if (
            self.opportunity_count + self.infra_censored_count
            != self.committed_shot_count
        ):
            raise ValueError("opportunity/censor counts do not partition shots")
        if (
            self.strike_eligible_count
            != self.hit_count + self.policy_miss_count
        ):
            raise ValueError("strike eligible count identity differs")
        if self.pre_reveal_hidden_count != self.committed_shot_count:
            raise ValueError("pre-reveal hidden count identity differs")
        if self.ball_retire_count != self.successor_count:
            raise ValueError("ball retire count identity differs")
        if self.recovery_window_count != self.successor_count:
            raise ValueError("recovery window count identity differs")
        if self.final_swing_generation != (
            self.first_swing_generation + self.committed_shot_count - 1
        ):
            raise ValueError("swing generation count identity differs")
        if self.stale_contact_count != 0:
            raise ValueError("stale contact count must be zero")
        if self.clear_counter_change_count != 0:
            raise ValueError("clear counter change count must be zero")
        if self.checkpoint_save_count != 1 or self.checkpoint_load_count != 1:
            raise ValueError("checkpoint must be saved and loaded exactly once")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        validated_trace: ContinuousSuccessorTrace,
        expected_contract_authority_sha256: str,
        expected_checkpoint_sha256: str,
        expected_replay_evidence_sha256: str,
        expected_trace_sha256: str,
    ) -> "ContinuousSuccessorStateReceipt":
        candidate = cls(**cls._mapping_values(value))
        expected = validate_continuous_successor(
            validated_trace,
            expected_contract_authority_sha256=(
                expected_contract_authority_sha256
            ),
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_replay_evidence_sha256=(
                expected_replay_evidence_sha256
            ),
            expected_trace_sha256=expected_trace_sha256,
        )
        if candidate != expected:
            raise ValueError("state receipt differs from validated trace")
        return candidate


def _contract_error(message: str) -> ContinuousSuccessorContractError:
    return ContinuousSuccessorContractError(message)


def _active_shot_at_step(
    shots: Tuple[ContinuousShotTrace, ...], step: int
) -> Optional[ContinuousShotTrace]:
    for shot in shots:
        if shot.scheduled_reveal_step <= step <= shot.scheduled_deadline_step:
            return shot
    return None


def _expected_close_reason(shot: ContinuousShotTrace) -> str:
    if not shot.infrastructure_valid:
        return "infra_invalid"
    if not shot.ready_met_at_reveal:
        return "recovery_unavailable"
    return "hit" if shot.hit else "policy_miss"


def _validate_recovery(
    recovery: RecoveryWindowEvidence,
    *,
    owner: ContinuousShotTrace,
    successor: ContinuousShotTrace,
    schedule: FrozenCadenceReceipt,
) -> None:
    if recovery.owner_task_ref != owner.task_ref:
        raise _contract_error("recovery owner task ref differs")
    if recovery.successor_task_ref != successor.task_ref:
        raise _contract_error("recovery successor task ref differs")
    if recovery.schedule_sha256 != schedule.canonical_sha256:
        raise _contract_error("recovery schedule SHA differs")
    if recovery.window_start_step != owner.scheduled_deadline_step + 1:
        raise _contract_error("recovery window must start after owner deadline")
    if recovery.observation_end_step != successor.scheduled_reveal_step - 1:
        raise _contract_error(
            "recovery window must end immediately before successor reveal"
        )
    if (
        recovery.scheduled_successor_reveal_step
        != schedule.reveal_step(successor.scheduled_ordinal)
    ):
        raise _contract_error("recovery successor reveal is not frozen schedule")
    if not recovery.infrastructure_valid:
        raise _contract_error("recovery evidence is infrastructure invalid")
    window_size = (
        recovery.observation_end_step - recovery.window_start_step + 1
    )
    eligibility_counts = (
        recovery.teacher_eligible_tick_count,
        recovery.reference_eligible_tick_count,
        recovery.recovery_eligible_tick_count,
        recovery.conjunction_eligible_tick_count,
        recovery.reward_payment_count,
    )
    if any(count != window_size for count in eligibility_counts):
        raise _contract_error(
            "every recovery tick must satisfy teacher/reference/recovery "
            "eligibility and receive exactly one payment"
        )
    if recovery.reward_total <= 0.0:
        raise _contract_error("recovery reward total must be positive")
    if recovery.first_ready_step is None:
        if recovery.ready_conjunction_tick_count != 0:
            raise _contract_error("never-ready recovery has ready ticks")
        if recovery.hold_ready_tick_count != 0:
            raise _contract_error("never-ready recovery has hold-ready ticks")
        expected_ready = False
    else:
        if not (
            recovery.window_start_step
            <= recovery.first_ready_step
            <= recovery.observation_end_step
        ):
            raise _contract_error("first ready step is outside recovery window")
        if recovery.ready_conjunction_tick_count <= 0:
            raise _contract_error("ready fact lacks conjunction-eligible tick")
        expected_ready_ticks = (
            recovery.observation_end_step - recovery.first_ready_step + 1
        )
        if recovery.ready_conjunction_tick_count != expected_ready_ticks:
            raise _contract_error(
                "ready conjunction must cover every tick after first-ready"
            )
        expected_ready = True
        expected_hold = (
            recovery.scheduled_successor_reveal_step
            - recovery.first_ready_step
            if expected_ready
            else 0
        )
        if recovery.hold_ready_tick_count != expected_hold:
            raise _contract_error("hold-ready ticks do not reach frozen reveal")

    if recovery.ready_met_at_scheduled_reveal != expected_ready:
        raise _contract_error("ready-at-reveal fact differs from first ready step")
    if successor.ready_met_at_reveal != expected_ready:
        raise _contract_error("successor ready fact differs from recovery evidence")


def _validate_outcome(
    outcome: DelayedOutcomeTrace,
    *,
    shots: Tuple[ContinuousShotTrace, ...],
    shot_by_ref: Mapping[str, ContinuousShotTrace],
    sequence_end_step: int,
) -> None:
    owner_sha = outcome.owner_task_ref.canonical_sha256
    owner = shot_by_ref.get(owner_sha)
    if owner is None or outcome.owner_task_ref != owner.task_ref:
        raise _contract_error("delayed outcome owner full receipt differs")
    if not owner.hit or owner.close_reason != "hit":
        raise _contract_error("only a hit shot may own delayed outcome")
    if outcome.owner_target_semantic_sha256 != owner.target.semantic_sha256:
        raise _contract_error("delayed outcome target owner differs")
    if not (
        owner.scheduled_reveal_step
        <= outcome.source_step
        <= owner.scheduled_deadline_step
    ):
        raise _contract_error("delayed outcome source step is outside owner shot")

    if outcome.settlement_step is None:
        if outcome.settlement_payload_sha256 is not None:
            raise _contract_error("pending outcome has settlement payload")
        if outcome.payment_step is not None or outcome.payment_count != 0:
            raise _contract_error("pending outcome cannot be paid")
        if outcome.active_task_ref_sha256_at_settlement is not None:
            raise _contract_error("pending outcome names settlement active task")
        if outcome.active_task_ref_sha256_at_payment is not None:
            raise _contract_error("pending outcome names payment active task")
        return

    if outcome.settlement_payload_sha256 is None:
        raise _contract_error("settled outcome lacks settlement payload")

    if not outcome.source_step <= outcome.settlement_step <= sequence_end_step:
        raise _contract_error("outcome settlement step is outside sequence")
    settlement_active = _active_shot_at_step(shots, outcome.settlement_step)
    expected_settlement = (
        None
        if settlement_active is None
        else settlement_active.task_ref.canonical_sha256
    )
    if (
        outcome.active_task_ref_sha256_at_settlement
        != expected_settlement
    ):
        raise _contract_error("settlement active-task witness differs")

    if outcome.payment_step is None:
        if outcome.payment_count != 0:
            raise _contract_error("unpaid outcome payment_count must be zero")
        if outcome.active_task_ref_sha256_at_payment is not None:
            raise _contract_error("unpaid outcome names payment active task")
        return

    if not (
        outcome.settlement_step
        <= outcome.payment_step
        <= sequence_end_step
    ):
        raise _contract_error("outcome payment step is outside sequence")
    if outcome.payment_count != 1:
        raise _contract_error("paid outcome payment_count must equal one")
    payment_active = _active_shot_at_step(shots, outcome.payment_step)
    expected_payment = (
        None
        if payment_active is None
        else payment_active.task_ref.canonical_sha256
    )
    if outcome.active_task_ref_sha256_at_payment != expected_payment:
        raise _contract_error("payment active-task witness differs")


def validate_continuous_successor(
    trace: ContinuousSuccessorTrace,
    *,
    expected_contract_authority_sha256: str,
    expected_checkpoint_sha256: str,
    expected_replay_evidence_sha256: str,
    expected_trace_sha256: str,
) -> ContinuousSuccessorStateReceipt:
    """Validate the v2 Q0..Q3 successor tape and derive a sealed receipt."""

    if not isinstance(trace, ContinuousSuccessorTrace):
        raise _contract_error("trace must be ContinuousSuccessorTrace")
    expected_authority = _sha256(
        expected_contract_authority_sha256,
        label="expected_contract_authority_sha256",
    )
    expected_checkpoint = _sha256(
        expected_checkpoint_sha256,
        label="expected_checkpoint_sha256",
    )
    expected_replay_evidence = _sha256(
        expected_replay_evidence_sha256,
        label="expected_replay_evidence_sha256",
    )
    expected_trace = _sha256(
        expected_trace_sha256,
        label="expected_trace_sha256",
    )
    if trace.canonical_sha256 != expected_trace:
        raise _contract_error("trace artifact differs from external pin")
    schedule = trace.schedule
    shots = trace.shots
    if schedule.scheduled_shot_count != 4 or len(shots) != 4:
        raise _contract_error("acceptance tape must contain exactly Q0..Q3")
    if len(trace.recoveries) != 3:
        raise _contract_error("four-shot tape requires three recovery windows")
    if not trace.initial_ready_met:
        raise _contract_error("Q0 initial ready authority is not met")
    if not (
        shots[-1].scheduled_deadline_step
        < trace.sequence_end_step
        < schedule.reveal_step(4)
    ):
        raise _contract_error("sequence end must precede frozen Q4 reveal")

    computed_authority = continuous_contract_authority_sha256(
        schedule,
        shots,
        trace.recoveries,
        trace.initial_ready_authority_sha256,
    )
    if trace.contract_authority_sha256 != expected_authority:
        raise _contract_error("trace contract authority differs from external pin")
    if computed_authority != expected_authority:
        raise _contract_error("immutable question contract was resealed or drifted")

    anchor = shots[0].task_ref
    anchor_profile = shots[0].target.profile_sha256
    anchor_target_authority = shots[0].target.selection_authority_sha256
    expected_labels = ("Q0", "Q1", "Q2", "Q3")
    inbound_ball_shas = []
    close_counts = {reason: 0 for reason in CLOSE_REASONS}

    for index, shot in enumerate(shots):
        if shot.shot_label != expected_labels[index]:
            raise _contract_error(f"shot {index} label differs")
        if shot.scheduled_ordinal != index:
            raise _contract_error("scheduled ordinal differs")
        ref = shot.task_ref
        for name in (
            "env_id",
            "reset_generation",
            "action_uid",
            "action_slot",
            "birth_sha256",
        ):
            if getattr(ref, name) != getattr(anchor, name):
                raise _contract_error(f"successor {name} differs")
        if ref.swing_generation != anchor.swing_generation + index:
            raise _contract_error("swing_generation must advance by exactly one")
        if (
            shot.scheduled_reveal_step != schedule.reveal_step(index)
            or shot.scheduled_deadline_step != schedule.deadline_step(index)
        ):
            raise _contract_error("shot timing differs from frozen cadence")

        admission = shot.admission
        if (
            admission.task_ref_sha256 != ref.canonical_sha256
            or admission.target_semantic_sha256 != shot.target.semantic_sha256
        ):
            raise _contract_error("committed admission identity differs")
        if not (
            admission.construction_feasible
            and admission.admitted
            and admission.infrastructure_valid_at_admission
            and admission.admission_decision == "admitted"
        ):
            raise _contract_error("committed question lacks feasible admission")
        lower_bound = (
            schedule.frozen_at_step
            if index == 0
            else shots[index - 1].scheduled_deadline_step + 1
        )
        if not lower_bound <= admission.evaluated_step < shot.scheduled_reveal_step:
            raise _contract_error("admission was not decided before reveal")
        if shot.committed != admission.admitted:
            raise _contract_error("shot committed fact differs from admission")
        if not shot.closed or shot.close_step != shot.scheduled_deadline_step:
            raise _contract_error("committed shot must close on frozen deadline")
        for name in (
            "boundary_terminated",
            "boundary_truncated",
            "boundary_reset",
            "boundary_teleported",
        ):
            if getattr(shot, name):
                raise _contract_error(f"shot boundary {name} is true")

        expected_reason = _expected_close_reason(shot)
        if shot.close_reason != expected_reason:
            raise _contract_error("explicit close_reason differs from shot facts")
        if shot.hit and not shot.ready_met_at_reveal:
            raise _contract_error("not-ready committed shot cannot claim hit")
        if shot.infrastructure_valid:
            if shot.infra_censor_reason != "none":
                raise _contract_error("valid shot has infrastructure censor reason")
        else:
            if shot.infra_censor_reason == "none" or shot.hit:
                raise _contract_error("infra-invalid shot facts differ")
        close_counts[shot.close_reason] += 1

        target = shot.target
        if target.profile_sha256 != anchor_profile:
            raise _contract_error("target profile differs")
        if target.selection_authority_sha256 != anchor_target_authority:
            raise _contract_error("target selection authority differs")
        if target.target_generation != ref.swing_generation:
            raise _contract_error("target generation differs from swing")
        if target.task_ref_sha256 != ref.canonical_sha256:
            raise _contract_error("target task-ref binding differs")
        if index > 0:
            previous = shots[index - 1]
            if target.runtime_target_xy_m == previous.target.runtime_target_xy_m:
                raise _contract_error("adjacent float32 targets must differ")
            if target.semantic_sha256 == previous.target.semantic_sha256:
                raise _contract_error("adjacent target semantics must differ")
            if ref.sample_sha256 == previous.task_ref.sample_sha256:
                raise _contract_error("adjacent sample identity must differ")
            if ref.task_sha256 == previous.task_ref.task_sha256:
                raise _contract_error("adjacent task identity must differ")

        hidden = shot.pre_reveal_hidden
        if (
            hidden.hidden_from_step != schedule.frozen_at_step
            or hidden.hidden_through_step != shot.scheduled_reveal_step - 1
            or hidden.observed_tick_count
            != shot.scheduled_reveal_step - schedule.frozen_at_step
            or hidden.first_visible_step != shot.scheduled_reveal_step
        ):
            raise _contract_error("pre-reveal evidence does not span full interval")
        if not hidden.all_future_facts_hidden:
            raise _contract_error("future question fact leaked before reveal")

        if (
            shot.carry_before_reveal.episode_step != shot.scheduled_reveal_step
            or shot.carry_after_reveal.episode_step != shot.scheduled_reveal_step
            or shot.carry_after_close.episode_step != shot.scheduled_deadline_step
        ):
            raise _contract_error("carry witness step differs from schedule")
        if shot.carry_before_reveal != shot.carry_after_reveal:
            raise _contract_error("question install changed carried state")

        ball = shot.ball
        if ball.ball_generation != ref.swing_generation:
            raise _contract_error("ball generation differs from swing")
        if ball.owner_task_ref_sha256 != ref.canonical_sha256:
            raise _contract_error("ball owner task ref differs")
        if ball.installed_at_step != shot.scheduled_reveal_step:
            raise _contract_error("ball install step differs from reveal")
        if ball.stale_contact_count != 0:
            raise _contract_error("stale ball contact count is nonzero")
        if ball.inbound_ball_sha256 in inbound_ball_shas:
            raise _contract_error("inbound ball identity was reused")
        inbound_ball_shas.append(ball.inbound_ball_sha256)
        if index == 0:
            if (
                ball.prior_ball_sha256 is not None
                or ball.prior_ball_retired
                or ball.prior_ball_settlement is not None
                or ball.prior_ball_retire_step is not None
            ):
                raise _contract_error("Q0 unexpectedly names a prior ball")
        else:
            previous = shots[index - 1]
            if ball.prior_ball_sha256 != previous.ball.inbound_ball_sha256:
                raise _contract_error("prior ball identity differs")
            if not ball.prior_ball_retired:
                raise _contract_error("prior ball was not retired")
            settlement = ball.prior_ball_settlement
            if (
                settlement is None
                or ball.prior_ball_retire_step is None
                or not (
                    previous.scheduled_deadline_step
                    < settlement.settlement_step
                    < ball.prior_ball_retire_step
                    < shot.scheduled_reveal_step
                )
            ):
                raise _contract_error(
                    "prior ball must settle, then retire, before successor reveal"
                )
            if (
                settlement.owner_task_ref != previous.task_ref
                or settlement.ball_sha256 != previous.ball.inbound_ball_sha256
                or settlement.source_step
                != previous.scheduled_deadline_step + 1
            ):
                raise _contract_error("prior ball settlement identity differs")
            expected_settlement_reason = (
                "owned_physical_outcome"
                if previous.hit
                else "flight_horizon_zero"
            )
            if settlement.settlement_reason != expected_settlement_reason:
                raise _contract_error("prior ball settlement reason differs")

    if shots[0].ready_met_at_reveal != trace.initial_ready_met:
        raise _contract_error("Q0 ready fact differs from initial authority")
    if any(count != 1 for count in close_counts.values()):
        raise _contract_error("four-shot acceptance must exercise each close reason once")

    for index, recovery in enumerate(trace.recoveries):
        _validate_recovery(
            recovery,
            owner=shots[index],
            successor=shots[index + 1],
            schedule=schedule,
        )
        if index > 0 and (
            trace.recoveries[index - 1].observation_end_step
            >= recovery.window_start_step
        ):
            raise _contract_error("recovery windows overlap")

    if not trace.construction_rejections:
        raise _contract_error("tape must exercise construction infeasible rejection")
    seen_candidates = set()
    for rejection in trace.construction_rejections:
        ordinal = rejection.before_scheduled_ordinal
        if ordinal <= 0 or ordinal >= len(shots):
            raise _contract_error("construction rejection ordinal is invalid")
        if rejection.candidate_semantic_sha256 in seen_candidates:
            raise _contract_error("construction candidate identity was reused")
        seen_candidates.add(rejection.candidate_semantic_sha256)
        if (
            rejection.committed
            or rejection.opportunity_created
            or not rejection.infrastructure_valid
        ):
            raise _contract_error("infeasible candidate entered opportunity lane")
        expected_generation = shots[ordinal - 1].task_ref.swing_generation
        if (
            rejection.swing_generation_before != expected_generation
            or rejection.swing_generation_after != expected_generation
        ):
            raise _contract_error("rejection changed swing generation")
        if not (
            shots[ordinal - 1].scheduled_deadline_step
            < rejection.evaluated_step
            < shots[ordinal].scheduled_reveal_step
        ):
            raise _contract_error("infeasible evaluation was not pre-reveal")
        if rejection.candidate_semantic_sha256 == shots[ordinal].target.semantic_sha256:
            raise _contract_error("rejected candidate became committed target")

    carry_points = []
    for shot in shots:
        carry_points.extend((shot.carry_before_reveal, shot.carry_after_close))
    checkpoint = trace.checkpoint
    carry_points.append(checkpoint.current_carry)
    carry_points.sort(key=lambda item: item.episode_step)
    anchor_lineage = carry_points[0].episode_lineage_sha256
    anchor_counters = carry_points[0].clear_counter_signature
    for index, witness in enumerate(carry_points):
        if witness.episode_lineage_sha256 != anchor_lineage:
            raise _contract_error("carry episode lineage differs")
        if witness.reset_generation != anchor.reset_generation:
            raise _contract_error("carry reset generation differs")
        if witness.clear_counter_signature != anchor_counters:
            raise _contract_error("carried reset/history/noise/GAE counter changed")
        if index > 0 and witness.parent_witness_sha256 != carry_points[index - 1].canonical_sha256:
            raise _contract_error("carry witness parent chain differs")

    shot_by_ref = {shot.task_ref.canonical_sha256: shot for shot in shots}
    outcomes_by_owner = {key: [] for key in shot_by_ref}
    event_ids = set()
    for outcome in trace.delayed_outcomes:
        _validate_outcome(
            outcome,
            shots=shots,
            shot_by_ref=shot_by_ref,
            sequence_end_step=trace.sequence_end_step,
        )
        if outcome.event_identity_sha256 in event_ids:
            raise _contract_error("delayed outcome event identity is duplicated")
        event_ids.add(outcome.event_identity_sha256)
        outcomes_by_owner[outcome.owner_task_ref.canonical_sha256].append(outcome)
    for shot in shots:
        owned = outcomes_by_owner[shot.task_ref.canonical_sha256]
        if shot.hit:
            if (
                len(owned) != len(OUTCOME_KINDS)
                or {outcome.outcome_kind for outcome in owned} != OUTCOME_KINDS
            ):
                raise _contract_error(
                    "hit must own exactly one landing and one outgoing-ball outcome"
                )
        elif owned:
            raise _contract_error("non-hit shot owns delayed outcome")

    for index in range(1, len(shots)):
        previous = shots[index - 1]
        retire_step = shots[index].ball.prior_ball_retire_step
        settlement = shots[index].ball.prior_ball_settlement
        settlement_step = settlement.settlement_step
        owned = outcomes_by_owner[previous.task_ref.canonical_sha256]
        if any(outcome.settlement_step is None for outcome in owned):
            raise _contract_error("ball retired with unresolved physical outcome")
        if owned and max(outcome.settlement_step for outcome in owned) > settlement_step:
            raise _contract_error(
                "ball lifecycle settlement precedes owned physical outcome"
            )
        if settlement_step >= retire_step:
            raise _contract_error("ball retirement is not after settlement")
        if previous.hit:
            landing = next(
                outcome
                for outcome in owned
                if outcome.outcome_kind == "first_landing_placement"
            )
            if (
                settlement.owned_outcome_event_sha256
                != landing.event_identity_sha256
                or settlement.settlement_step != landing.settlement_step
                or settlement.settlement_state_sha256
                != landing.settlement_payload_sha256
            ):
                raise _contract_error(
                    "owned physical ball settlement differs from landing outcome"
                )

    if checkpoint.canonical_sha256 != expected_checkpoint:
        raise _contract_error("checkpoint state differs from external pin")
    if checkpoint.contract_authority_sha256 != expected_authority:
        raise _contract_error("checkpoint contract authority differs")
    if checkpoint.schedule_sha256 != schedule.canonical_sha256:
        raise _contract_error("checkpoint schedule SHA differs")
    cp_step = checkpoint.checkpoint_step
    if not schedule.first_reveal_step < cp_step < trace.sequence_end_step:
        raise _contract_error("checkpoint is outside sequence")
    if _active_shot_at_step(shots, cp_step) is not None:
        raise _contract_error("acceptance checkpoint must be pure recovery/hold")
    closed_before = [shot for shot in shots if shot.scheduled_deadline_step < cp_step]
    future = [shot for shot in shots if shot.scheduled_reveal_step > cp_step]
    if not closed_before or not future:
        raise _contract_error("checkpoint lacks closed owner or successor")
    last_closed = closed_before[-1]
    next_shot = future[0]
    active_recovery = next(
        (
            recovery
            for recovery in trace.recoveries
            if recovery.window_start_step <= cp_step <= recovery.observation_end_step
        ),
        None,
    )
    if active_recovery is None:
        raise _contract_error("checkpoint has no active recovery window")
    if checkpoint.active_ordinal is not None or checkpoint.active_task_ref is not None:
        raise _contract_error("pure recovery checkpoint names active opportunity")
    if checkpoint.last_closed_task_ref != last_closed.task_ref:
        raise _contract_error("checkpoint last closed task differs")
    if checkpoint.next_scheduled_ordinal != next_shot.scheduled_ordinal:
        raise _contract_error("checkpoint next ordinal differs")
    if checkpoint.remaining_to_deadline_steps != 0:
        raise _contract_error("recovery checkpoint has remaining deadline")
    if checkpoint.next_scheduled_reveal_step != next_shot.scheduled_reveal_step:
        raise _contract_error("checkpoint next reveal differs")
    if checkpoint.remaining_to_next_reveal_steps != next_shot.scheduled_reveal_step - cp_step:
        raise _contract_error("checkpoint remaining-to-reveal differs")
    if checkpoint.next_question_visible:
        raise _contract_error("next question visible during recovery gap")
    ready_at_cp = (
        active_recovery.first_ready_step is not None
        and active_recovery.first_ready_step <= cp_step
    )
    expected_phase = "ready_hold" if ready_at_cp else "recovery_hidden"
    if checkpoint.active_phase != expected_phase:
        raise _contract_error("checkpoint recovery phase differs")
    if checkpoint.active_recovery_owner_task_ref_sha256 != last_closed.task_ref.canonical_sha256:
        raise _contract_error("checkpoint recovery owner differs")
    if checkpoint.strike_latched or checkpoint.strike_latch_owner_task_ref_sha256 is not None:
        raise _contract_error("pure recovery checkpoint retains strike latch")

    if checkpoint.current_carry.episode_step != cp_step:
        raise _contract_error("checkpoint carry step differs")
    checkpoint_parent = max(
        (witness for witness in carry_points if witness.episode_step < cp_step),
        key=lambda item: item.episode_step,
    )
    if checkpoint.current_carry.parent_witness_sha256 != checkpoint_parent.canonical_sha256:
        raise _contract_error("checkpoint carry parent differs")
    if (
        checkpoint.current_carry.episode_lineage_sha256 != anchor_lineage
        or checkpoint.current_carry.reset_generation != anchor.reset_generation
        or checkpoint.current_carry.clear_counter_signature != anchor_counters
    ):
        raise _contract_error("checkpoint carried state continuity differs")

    retire_step = next_shot.ball.prior_ball_retire_step
    if (
        checkpoint.current_ball_owner_task_ref_sha256
        != last_closed.task_ref.canonical_sha256
        or checkpoint.current_ball_generation != last_closed.ball.ball_generation
        or checkpoint.current_ball_retired
        or retire_step is None
        or cp_step >= retire_step
    ):
        raise _contract_error("checkpoint live ball lifecycle differs")
    contact_before_cp = any(
        outcome.source_step <= cp_step
        for outcome in outcomes_by_owner[last_closed.task_ref.canonical_sha256]
    )
    if checkpoint.ball_contact_latched != contact_before_cp:
        raise _contract_error("checkpoint ball contact latch differs")
    expected_contact_owner = (
        last_closed.task_ref.canonical_sha256 if contact_before_cp else None
    )
    if checkpoint.ball_contact_latch_owner_task_ref_sha256 != expected_contact_owner:
        raise _contract_error("checkpoint ball contact owner differs")

    progress = checkpoint.recovery_progress
    prefix_ticks = cp_step - active_recovery.window_start_step + 1
    if (
        progress.owner_task_ref_sha256 != last_closed.task_ref.canonical_sha256
        or progress.recovery_window_sha256 != active_recovery.canonical_sha256
        or progress.observed_through_step != cp_step
        or progress.observed_tick_count != prefix_ticks
    ):
        raise _contract_error("checkpoint recovery prefix identity differs")
    prefix_counts = (
        progress.teacher_eligible_tick_count,
        progress.reference_eligible_tick_count,
        progress.recovery_eligible_tick_count,
        progress.conjunction_eligible_tick_count,
        progress.reward_payment_count,
    )
    if any(count != prefix_ticks for count in prefix_counts):
        raise _contract_error("checkpoint recovery prefix payment differs")
    if not 0.0 < progress.reward_total <= active_recovery.reward_total:
        raise _contract_error("checkpoint recovery reward prefix differs")
    expected_first_ready = active_recovery.first_ready_step if ready_at_cp else None
    expected_ready_ticks = cp_step - expected_first_ready + 1 if ready_at_cp else 0
    if (
        progress.ready_latched != ready_at_cp
        or progress.first_ready_step != expected_first_ready
        or progress.ready_conjunction_tick_count != expected_ready_ticks
        or progress.hold_ready_tick_count != expected_ready_ticks
    ):
        raise _contract_error("checkpoint ready/hold prefix differs")

    unpaid_at_checkpoint = checkpoint_mailbox_entries(
        trace.delayed_outcomes, cp_step
    )
    if checkpoint.unpaid_outcome_entries != unpaid_at_checkpoint:
        raise _contract_error("checkpoint unpaid mailbox differs")
    if checkpoint.mailbox_sha256 != checkpoint_mailbox_sha256(unpaid_at_checkpoint):
        raise _contract_error("checkpoint mailbox SHA differs")
    unpaid_event_ids = {
        entry.event_identity_sha256 for entry in unpaid_at_checkpoint
    }
    pending_unpaid = [
        outcome
        for outcome in trace.delayed_outcomes
        if outcome.event_identity_sha256 in unpaid_event_ids
        and (outcome.settlement_step is None or outcome.settlement_step > cp_step)
    ]
    settled_unpaid = [
        outcome
        for outcome in trace.delayed_outcomes
        if outcome.event_identity_sha256 in unpaid_event_ids
        and outcome.settlement_step is not None
        and outcome.settlement_step <= cp_step
        and (outcome.payment_step is None or outcome.payment_step > cp_step)
    ]
    if not pending_unpaid or not settled_unpaid:
        raise _contract_error("tape must exercise pending and settled-unpaid outcomes")

    expected_suffix = continuous_replay_suffix_sha256(
        shots, trace.recoveries, trace.delayed_outcomes, cp_step
    )
    if checkpoint.expected_replay_suffix_sha256 != expected_suffix:
        raise _contract_error("checkpoint replay suffix differs")
    replay = trace.checkpoint_replay
    if replay.canonical_sha256 != expected_replay_evidence:
        raise _contract_error("replay evidence differs from external pin")
    if replay.checkpoint_sha256 != checkpoint.canonical_sha256:
        raise _contract_error("restore evidence names another checkpoint")
    if replay.save_count != 1 or replay.load_count != 1:
        raise _contract_error("checkpoint was not saved and loaded exactly once")
    if replay.replay_start_step != cp_step or replay.replay_end_step != trace.sequence_end_step:
        raise _contract_error("restore replay interval differs")
    if (
        replay.saved_sampler_rng_sha256 != checkpoint.sampler_rng_sha256
        or replay.restored_sampler_rng_sha256 != checkpoint.sampler_rng_sha256
    ):
        raise _contract_error("sampler RNG was not restored exactly")
    if replay.restored_carry_sha256 != checkpoint.current_carry.canonical_sha256:
        raise _contract_error("carried state was not restored exactly")
    if replay.restored_ball_state_sha256 != checkpoint.current_ball_state_sha256:
        raise _contract_error("ball dynamic state was not restored exactly")
    if replay.restored_recovery_progress_sha256 != progress.canonical_sha256:
        raise _contract_error("recovery progress was not restored exactly")
    if replay.restored_mailbox_sha256 != checkpoint.mailbox_sha256:
        raise _contract_error("unpaid mailbox was not restored exactly")
    if replay.uninterrupted_observer_authority_sha256 == replay.restored_observer_authority_sha256:
        raise _contract_error("replay branches lack independent observers")
    if (
        replay.uninterrupted_suffix_sha256 != expected_suffix
        or replay.resumed_suffix_sha256 != expected_suffix
        or replay.uninterrupted_rng_draw_chain_sha256
        != replay.resumed_rng_draw_chain_sha256
        or replay.first_post_restore_sample_sha256 != next_shot.task_ref.sample_sha256
        or not replay.bit_exact
    ):
        raise _contract_error("uninterrupted/restored suffix is not bit exact")

    late_paid = [
        outcome
        for outcome in trace.delayed_outcomes
        if outcome.event_identity_sha256 in unpaid_event_ids
        and outcome.payment_step is not None
        and outcome.active_task_ref_sha256_at_payment is not None
        and outcome.active_task_ref_sha256_at_payment
        != outcome.owner_task_ref.canonical_sha256
    ]
    if not late_paid:
        raise _contract_error("tape lacks late payment under another active task")

    opportunity_count = (
        close_counts["hit"]
        + close_counts["policy_miss"]
        + close_counts["recovery_unavailable"]
    )
    strike_eligible_count = close_counts["hit"] + close_counts["policy_miss"]
    return ContinuousSuccessorStateReceipt(
        trace_sha256=trace.canonical_sha256,
        contract_authority_sha256=expected_authority,
        schedule_sha256=schedule.canonical_sha256,
        birth_sha256=anchor.birth_sha256,
        env_id=anchor.env_id,
        reset_generation=anchor.reset_generation,
        action_uid=anchor.action_uid,
        action_slot=anchor.action_slot,
        first_swing_generation=anchor.swing_generation,
        final_swing_generation=shots[-1].task_ref.swing_generation,
        sequence_end_step=trace.sequence_end_step,
        committed_shot_count=len(shots),
        successor_count=len(shots) - 1,
        construction_infeasible_count=len(trace.construction_rejections),
        opportunity_count=opportunity_count,
        strike_eligible_count=strike_eligible_count,
        hit_count=close_counts["hit"],
        policy_miss_count=close_counts["policy_miss"],
        recovery_unavailable_count=close_counts["recovery_unavailable"],
        infra_censored_count=close_counts["infra_invalid"],
        closed_count=len(shots),
        pre_reveal_hidden_count=len(shots),
        recovery_window_count=len(trace.recoveries),
        recovery_payment_count=sum(r.reward_payment_count for r in trace.recoveries),
        ready_met_successor_count=sum(int(r.ready_met_at_scheduled_reveal) for r in trace.recoveries),
        ball_retire_count=sum(int(shot.ball.prior_ball_retired) for shot in shots),
        stale_contact_count=0,
        delayed_outcome_count=len(trace.delayed_outcomes),
        paid_outcome_count=sum(int(outcome.state == "paid") for outcome in trace.delayed_outcomes),
        late_outcome_count=len(late_paid),
        unpaid_at_checkpoint_count=len(unpaid_at_checkpoint),
        checkpoint_save_count=replay.save_count,
        checkpoint_load_count=replay.load_count,
        clear_counter_change_count=0,
        contract_scope=CONTRACT_SCOPE,
        runtime_integrated=False,
        environment_fixed=False,
        launch_authorized=False,
    )


def validate_three_shot_successor(
    trace: ContinuousSuccessorTrace,
    *,
    expected_contract_authority_sha256: str,
    expected_checkpoint_sha256: str,
    expected_replay_evidence_sha256: str,
    expected_trace_sha256: str,
) -> ContinuousSuccessorStateReceipt:
    """Compatibility name; v2 now requires the four-shot acceptance tape."""

    return validate_continuous_successor(
        trace,
        expected_contract_authority_sha256=expected_contract_authority_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_replay_evidence_sha256=expected_replay_evidence_sha256,
        expected_trace_sha256=expected_trace_sha256,
    )


ContinuousThreeShotTrace = ContinuousSuccessorTrace


__all__ = (
    "ADMISSION_DECISIONS",
    "BALL_SETTLEMENT_REASONS",
    "CHECKPOINT_PHASES",
    "CHECKPOINT_OUTCOME_STATES",
    "CLOCK_KIND",
    "CLOSE_REASONS",
    "CONSTRUCTION_REASONS",
    "CONTRACT_SCOPE",
    "INFRA_CENSOR_REASONS",
    "OUTCOME_KINDS",
    "RUNTIME_TARGET_DTYPE",
    "RUNTIME_TASK_REF_FIELDS",
    "SOURCE_EVENT_KINDS",
    "BallLifecycleReceipt",
    "BallSettlementReceipt",
    "CarryContinuityWitness",
    "CheckpointReplayEvidence",
    "CheckpointOutcomeMailboxEntry",
    "CommittedOpportunityReceipt",
    "ConstructionInfeasibleRecord",
    "ContinuousActionTaskReceiptRef",
    "ContinuousCheckpointReceipt",
    "ContinuousShotTrace",
    "ContinuousSuccessorContractError",
    "ContinuousSuccessorStateReceipt",
    "ContinuousSuccessorTrace",
    "ContinuousThreeShotTrace",
    "DelayedOutcomeTrace",
    "FrozenCadenceReceipt",
    "PreRevealHiddenWitness",
    "RecoveryProgressReceipt",
    "RecoveryWindowEvidence",
    "TargetSelectionReceipt",
    "canonical_sha256",
    "checkpoint_mailbox_sha256",
    "checkpoint_mailbox_entries",
    "continuous_contract_authority_sha256",
    "continuous_replay_suffix_sha256",
    "target_semantic_sha256",
    "validate_continuous_successor",
    "validate_three_shot_successor",
)
