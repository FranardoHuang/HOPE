"""Pure continuous-recovery semantics for ActionBall (R07 first grid).

``PRE_INTEGRATION_HOLD`` is part of this module's contract, not a progress
label.  The module owns no Isaac, MuJoCo, environment configuration, command
manager, reward manager, or termination manager wiring.  It provides a
standard-library state owner which those adapters may later project into, but
it cannot authorize a training launch by itself.

The recovery clock is frozen at 50 Hz.  If ``d`` is the consumed shot
deadline tick, the reward-owner age is ``episode_tick - d`` and the complete
denominator is the inclusive interval ``10..77`` (68 samples).  A played row
must have completed its full motion suffix before age 10.  Missing that
boundary is a sticky infrastructure fault: the 68 expected denominator cells
remain visible and must still be paid as explicit zeroes.

Readiness and reward are deliberately different signals.  Readiness is a
hard conjunction followed by a consecutive dwell; one failed component
cannot be compensated by another.  Recovery reward is an additive, positive
Cauchy score and is *not* gated by readiness, so a finite but poor rollout-0
state still supplies broad learning signal.  A committed unplayed row owns
the same deadline-based recovery interval as a played row and needs no suffix
to become eligible.

Only the current committed complete C05 ``LandingOutcomeShotKey`` is accepted.
That is the 14-field owner spanning runtime receipt truth and successor
lineage.  Public Command and Reward projections contain no future task,
target, or deadline.  Every shot boundary projects literal false for done,
truncate, reset, and teleport.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Callable, ClassVar, Mapping, Optional, Sequence

try:  # Tests import the source root directly; installed packages use relative import.
    import action_ball_landing_outcome_mailbox as _mailbox
except ModuleNotFoundError:  # pragma: no cover - package installation path
    from . import action_ball_landing_outcome_mailbox as _mailbox


SCHEMA_VERSION = 1
INTEGRATION_STATUS = "PRE_INTEGRATION_HOLD"
RUNTIME_WIRING_CONNECTED = False

POLICY_RATE_HZ = 50
RECOVERY_START_AGE_TICK = 10
RECOVERY_END_AGE_TICK = 77
RECOVERY_SAMPLE_COUNT = 68

REFERENCE_KIND = "completed_action_frame0_zero_velocity_v1"
STATION_ANCHOR_KIND = "frozen_once_on_reference_entry_v1"
SUPPORT_SIGNAL_KIND = "normal_contact_signal_ge_threshold_v1"
REWARD_CONSUMER = "common_recovery_reward_v1"

SEQUENCE_BIRTH_OWNER = "sequence_birth"
COMMITTED_TASK_OWNER = "committed_task"
REFERENCE_OWNER_KINDS = frozenset((SEQUENCE_BIRTH_OWNER, COMMITTED_TASK_OWNER))

MOTION_PHASES = (
    "pre_reveal_hidden",
    "active_opportunity",
    "post_deadline_suffix",
    "recovery_hidden",
    "ready_hold",
    "recovery_unavailable",
    "infrastructure_invalid",
)

COMPONENT_NAMES = (
    "root_position_m",
    "root_orientation_rad",
    "root_linear_velocity_mps",
    "root_angular_velocity_radps",
    "joint_position_rad",
    "joint_velocity_radps",
    "body_position_m",
    "body_orientation_rad",
    "body_linear_velocity_mps",
    "body_angular_velocity_radps",
    "station_xy_m",
    "foot_slip_mps",
    "foot_support_deficit",
)
PLANT_ERROR_COMPONENT_NAMES = COMPONENT_NAMES[:-1]

# The caller must spell these out in every profile.  Keeping one exact value
# per component makes the numerical reduction a reviewable contract instead
# of an adapter-local choice.
REQUIRED_COMPONENT_REDUCTIONS = MappingProxyType(
    {
        "root_position_m": "l2_norm_v1",
        "root_orientation_rad": "quaternion_geodesic_rad_v1",
        "root_linear_velocity_mps": "l2_norm_v1",
        "root_angular_velocity_radps": "l2_norm_v1",
        "joint_position_rad": "root_mean_square_v1",
        "joint_velocity_radps": "root_mean_square_v1",
        "body_position_m": "root_mean_square_l2_v1",
        "body_orientation_rad": "root_mean_square_geodesic_rad_v1",
        "body_linear_velocity_mps": "root_mean_square_l2_v1",
        "body_angular_velocity_radps": "root_mean_square_l2_v1",
        "station_xy_m": "l2_norm_v1",
        "foot_slip_mps": "support_conditioned_max_l2_v1",
        "foot_support_deficit": "minimum_supported_feet_minus_count_clamped_v1",
    }
)

PROFILE_KIND = "action_ball_continuous_recovery_profile_v1"
REFERENCE_OWNER_KIND = "action_ball_continuous_recovery_reference_owner_v1"
MOTION_PROJECTION_KIND = "action_ball_continuous_recovery_motion_projection_v1"
COMMAND_PROJECTION_KIND = "action_ball_continuous_recovery_command_projection_v1"
DONE_TERM_PROJECTION_KIND = (
    "action_ball_continuous_recovery_done_term_projection_v1"
)
FACT_KIND = "action_ball_continuous_recovery_fact_v1"
REWARD_VIEW_KIND = "action_ball_continuous_recovery_reward_view_v1"
PAYMENT_IDEMPOTENCY_KIND = "action_ball_continuous_recovery_payment_v1"
LEDGER_KIND = "action_ball_continuous_recovery_ledger_v1"
CHECKPOINT_KIND = "action_ball_continuous_recovery_checkpoint_v1"

NO_INFRASTRUCTURE_FAULT = "none"
PLAYED_SUFFIX_MISSING_AT_REWARD_START = (
    "played_suffix_missing_at_recovery_age_10"
)

_FaultInjector = Optional[Callable[[str], None]]
_HEX = frozenset("0123456789abcdef")
_MIN_POSITIVE_FLOAT = math.ulp(0.0)


class ContinuousRecoveryError(RuntimeError):
    """The pure R07 recovery contract was violated."""


class ContinuousRecoveryConflictError(ContinuousRecoveryError):
    """A stale, duplicate, or cross-owner transition was requested."""


class ContinuousRecoveryInfrastructureError(ContinuousRecoveryError):
    """A sticky row-level infrastructure fault was observed."""


def canonical_sha256(value: object) -> str:
    """Return the canonical finite-JSON SHA-256 used by every sealed record."""

    encoded = json.dumps(
        value,
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


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise ContinuousRecoveryError(f"{label} must be an exact int")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContinuousRecoveryError(f"{label} is outside its allowed range")
    return value


def _optional_plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> Optional[int]:
    if value is None:
        return None
    return _plain_int(value, label=label, minimum=minimum)


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ContinuousRecoveryError(f"{label} must be an exact bool")
    return value


def _finite(
    value: object,
    *,
    label: str,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContinuousRecoveryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContinuousRecoveryError(f"{label} must be a finite number")
    if strictly_positive and result <= 0.0:
        raise ContinuousRecoveryError(f"{label} must be > 0")
    if minimum is not None and result < minimum:
        raise ContinuousRecoveryError(f"{label} must be >= {minimum}")
    return 0.0 if result == 0.0 else result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ContinuousRecoveryError(f"{label} must be one lowercase SHA-256")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContinuousRecoveryError(f"{label} must be a non-empty string")
    return value


def _optional_sha256(value: object, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _sha256(value, label=label)


def _xy(value: object, *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContinuousRecoveryError(f"{label} must be an XY sequence")
    rows = tuple(value)
    if len(rows) != 2:
        raise ContinuousRecoveryError(f"{label} must contain exactly two values")
    return (
        _finite(rows[0], label=f"{label}[0]"),
        _finite(rows[1], label=f"{label}[1]"),
    )


def _ordered_names(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContinuousRecoveryError(f"{label} must be an ordered sequence")
    result = tuple(_text(item, label=f"{label}[]") for item in value)
    if not result:
        raise ContinuousRecoveryError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ContinuousRecoveryError(f"{label} must contain unique names")
    return result


def _exact_number_mapping(
    value: object,
    *,
    label: str,
    strictly_positive: bool,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ContinuousRecoveryError(f"{label} must be a mapping")
    actual = frozenset(value)
    expected = frozenset(COMPONENT_NAMES)
    if actual != expected:
        raise ContinuousRecoveryError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return MappingProxyType(
        {
            name: _finite(
                value[name],
                label=f"{label}.{name}",
                minimum=0.0,
                strictly_positive=strictly_positive,
            )
            for name in COMPONENT_NAMES
        }
    )


def _component_errors(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ContinuousRecoveryError("component_errors must be a mapping")
    actual = frozenset(value)
    expected = frozenset(PLANT_ERROR_COMPONENT_NAMES)
    if actual != expected:
        raise ContinuousRecoveryError(
            "component_errors keys differ: "
            f"missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return MappingProxyType(
        {
            name: _finite(
                value[name], label=f"component_errors.{name}", minimum=0.0
            )
            for name in PLANT_ERROR_COMPONENT_NAMES
        }
    )


def _contact_signals(
    value: object,
    *,
    ordered_foot_names: tuple[str, ...],
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ContinuousRecoveryError("foot_contact_signals must be a mapping")
    actual = frozenset(value)
    expected = frozenset(ordered_foot_names)
    if actual != expected:
        raise ContinuousRecoveryError(
            "foot_contact_signals keys differ: "
            f"missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return MappingProxyType(
        {
            name: _finite(
                value[name], label=f"foot_contact_signals.{name}", minimum=0.0
            )
            for name in ordered_foot_names
        }
    )


def _string_mapping(value: object, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ContinuousRecoveryError(f"{label} must be a mapping")
    actual = frozenset(value)
    expected = frozenset(COMPONENT_NAMES)
    if actual != expected:
        raise ContinuousRecoveryError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    result = {
        name: _text(value[name], label=f"{label}.{name}")
        for name in COMPONENT_NAMES
    }
    for name, required in REQUIRED_COMPONENT_REDUCTIONS.items():
        if result[name] != required:
            raise ContinuousRecoveryError(
                f"{label}.{name} must equal {required!r}"
            )
    return MappingProxyType(result)


def _encode(value: object) -> object:
    if isinstance(value, _SealedRecord):
        return value.to_mapping()
    if isinstance(value, _mailbox.LandingOutcomeShotKey):
        return value.to_mapping()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _verified_values(
    value: object,
    *,
    cls: type,
    kind: str,
) -> dict[str, object]:
    label = cls.__name__
    if not isinstance(value, Mapping):
        raise ContinuousRecoveryError(f"{label} must be a mapping")
    names = tuple(field.name for field in fields(cls))
    expected_payload = frozenset(("schema_version", "kind", *names))
    expected = expected_payload | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise ContinuousRecoveryError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    payload = {key: value[key] for key in expected_payload}
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContinuousRecoveryError(f"{label} schema_version differs")
    if payload["kind"] != kind:
        raise ContinuousRecoveryError(f"{label} kind differs")
    declared = _sha256(value["canonical_sha256"], label=f"{label}.canonical_sha256")
    try:
        actual_sha = canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise ContinuousRecoveryError(
            f"{label} is not finite canonical JSON"
        ) from exc
    if actual_sha != declared:
        raise ContinuousRecoveryError(f"{label} canonical SHA differs")
    return {name: payload[name] for name in names}


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
        return _verified_values(value, cls=cls, kind=cls.KIND)


def coerce_landing_outcome_shot_key(
    value: object,
) -> _mailbox.LandingOutcomeShotKey:
    """Strictly copy the complete 14-field C05 owner key."""

    try:
        key = _mailbox.LandingOutcomeShotKey.coerce(value)
    except Exception as exc:
        raise ContinuousRecoveryError(
            "task key must be the exact complete C05 LandingOutcomeShotKey"
        ) from exc
    # A hostile lookalike must not be able to influence the returned object
    # after coercion.  Round-trip through the owner's own sealed format.
    try:
        return _mailbox.LandingOutcomeShotKey.from_mapping(key.to_mapping())
    except Exception as exc:  # pragma: no cover - protects dependency drift
        raise ContinuousRecoveryError("C05 shot-key round-trip failed") from exc


@dataclass(frozen=True)
class ContinuousRecoveryProfile(_SealedRecord):
    """All numerical and ordering choices required by the pure coordinator."""

    KIND: ClassVar[str] = PROFILE_KIND

    continuous_contract_authority_sha256: str
    recovery_contract_authority_sha256: str
    transaction_contract_authority_sha256: str
    source_sha256: str
    config_sha256: str
    plant_fact_schema_sha256: str
    ordered_joint_names: tuple[str, ...]
    ordered_body_names: tuple[str, ...]
    ordered_foot_names: tuple[str, ...]
    position_frame: str
    orientation_frame: str
    quaternion_order: str
    reference_semantics: str
    station_anchor_semantics: str
    support_signal_semantics: str
    policy_rate_hz: int
    recovery_start_age_tick: int
    recovery_end_age_tick: int
    component_weights: Mapping[str, float]
    component_scales: Mapping[str, float]
    component_reductions: Mapping[str, str]
    ready_tolerances: Mapping[str, float]
    support_contact_threshold: float
    minimum_supported_feet: int
    ready_dwell_ticks: int
    reward_weight: float

    def __post_init__(self) -> None:
        for name in (
            "continuous_contract_authority_sha256",
            "recovery_contract_authority_sha256",
            "transaction_contract_authority_sha256",
            "source_sha256",
            "config_sha256",
            "plant_fact_schema_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "ordered_joint_names",
            _ordered_names(self.ordered_joint_names, label="ordered_joint_names"),
        )
        object.__setattr__(
            self,
            "ordered_body_names",
            _ordered_names(self.ordered_body_names, label="ordered_body_names"),
        )
        object.__setattr__(
            self,
            "ordered_foot_names",
            _ordered_names(self.ordered_foot_names, label="ordered_foot_names"),
        )
        for name in ("position_frame", "orientation_frame"):
            object.__setattr__(self, name, _text(getattr(self, name), label=name))
        quaternion_order = _text(self.quaternion_order, label="quaternion_order")
        if quaternion_order not in ("wxyz", "xyzw"):
            raise ContinuousRecoveryError("quaternion_order must be wxyz or xyzw")
        object.__setattr__(self, "quaternion_order", quaternion_order)
        if self.reference_semantics != REFERENCE_KIND:
            raise ContinuousRecoveryError(
                f"reference_semantics must equal {REFERENCE_KIND!r}"
            )
        if self.station_anchor_semantics != STATION_ANCHOR_KIND:
            raise ContinuousRecoveryError(
                f"station_anchor_semantics must equal {STATION_ANCHOR_KIND!r}"
            )
        if self.support_signal_semantics != SUPPORT_SIGNAL_KIND:
            raise ContinuousRecoveryError(
                f"support_signal_semantics must equal {SUPPORT_SIGNAL_KIND!r}"
            )
        if _plain_int(self.policy_rate_hz, label="policy_rate_hz", minimum=1) != POLICY_RATE_HZ:
            raise ContinuousRecoveryError("policy_rate_hz must equal 50")
        if (
            _plain_int(
                self.recovery_start_age_tick,
                label="recovery_start_age_tick",
                minimum=0,
            )
            != RECOVERY_START_AGE_TICK
        ):
            raise ContinuousRecoveryError("recovery_start_age_tick must equal 10")
        if (
            _plain_int(
                self.recovery_end_age_tick,
                label="recovery_end_age_tick",
                minimum=0,
            )
            != RECOVERY_END_AGE_TICK
        ):
            raise ContinuousRecoveryError("recovery_end_age_tick must equal 77")
        if self.recovery_end_age_tick - self.recovery_start_age_tick + 1 != RECOVERY_SAMPLE_COUNT:
            raise ContinuousRecoveryError("recovery interval must contain 68 samples")
        object.__setattr__(
            self,
            "component_weights",
            _exact_number_mapping(
                self.component_weights,
                label="component_weights",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "component_scales",
            _exact_number_mapping(
                self.component_scales,
                label="component_scales",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "component_reductions",
            _string_mapping(self.component_reductions, label="component_reductions"),
        )
        object.__setattr__(
            self,
            "ready_tolerances",
            _exact_number_mapping(
                self.ready_tolerances,
                label="ready_tolerances",
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "support_contact_threshold",
            _finite(
                self.support_contact_threshold,
                label="support_contact_threshold",
                minimum=0.0,
            ),
        )
        minimum_supported_feet = _plain_int(
            self.minimum_supported_feet,
            label="minimum_supported_feet",
            minimum=1,
        )
        if minimum_supported_feet > len(self.ordered_foot_names):
            raise ContinuousRecoveryError(
                "minimum_supported_feet exceeds ordered_foot_names"
            )
        object.__setattr__(self, "minimum_supported_feet", minimum_supported_feet)
        object.__setattr__(
            self,
            "ready_dwell_ticks",
            _plain_int(self.ready_dwell_ticks, label="ready_dwell_ticks", minimum=1),
        )
        object.__setattr__(
            self,
            "reward_weight",
            _finite(
                self.reward_weight,
                label="reward_weight",
                strictly_positive=True,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRecoveryProfile":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class ContinuousRecoveryReferenceOwner(_SealedRecord):
    """One frozen frame-0 reference; it contains no future question fields."""

    KIND: ClassVar[str] = REFERENCE_OWNER_KIND

    owner_kind: str
    env_id: int
    reset_generation: int
    reference_generation: int
    task_key: Optional[_mailbox.LandingOutcomeShotKey]
    reference_snapshot_sha256: str
    frame0_sha256: str
    station_anchor_xy_m: tuple[float, float]
    reference_velocities_zero: bool

    def __post_init__(self) -> None:
        owner_kind = _text(self.owner_kind, label="owner_kind")
        if owner_kind not in REFERENCE_OWNER_KINDS:
            raise ContinuousRecoveryError("reference owner_kind is unknown")
        object.__setattr__(self, "owner_kind", owner_kind)
        env_id = _plain_int(self.env_id, label="env_id")
        reset_generation = _plain_int(
            self.reset_generation, label="reset_generation", minimum=1
        )
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "reset_generation", reset_generation)
        object.__setattr__(
            self,
            "reference_generation",
            _plain_int(
                self.reference_generation,
                label="reference_generation",
                minimum=0,
            ),
        )
        key = self.task_key
        if owner_kind == SEQUENCE_BIRTH_OWNER:
            if key is not None:
                raise ContinuousRecoveryError(
                    "sequence-birth reference must not carry a task key"
                )
        else:
            if key is None:
                raise ContinuousRecoveryError(
                    "committed-task reference requires the full C05 key"
                )
            key = coerce_landing_outcome_shot_key(key)
            if key.env_id != env_id or key.reset_generation != reset_generation:
                raise ContinuousRecoveryError(
                    "reference owner and task key environment differ"
                )
        object.__setattr__(self, "task_key", key)
        for name in ("reference_snapshot_sha256", "frame0_sha256"):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "station_anchor_xy_m",
            _xy(self.station_anchor_xy_m, label="station_anchor_xy_m"),
        )
        if not _exact_bool(
            self.reference_velocities_zero, label="reference_velocities_zero"
        ):
            raise ContinuousRecoveryError(
                "frame-0 recovery reference velocities must be literal zero"
            )

    @property
    def task_key_sha256(self) -> Optional[str]:
        return None if self.task_key is None else self.task_key.canonical_sha256

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRecoveryReferenceOwner":
        rows = cls._mapping_values(value)
        if rows["task_key"] is not None:
            rows["task_key"] = coerce_landing_outcome_shot_key(rows["task_key"])
        return cls(**rows)


@dataclass(frozen=True)
class ContinuousRecoveryMotionProjection(_SealedRecord):
    """Private Motion->common projection; deadline is present only once consumed."""

    KIND: ClassVar[str] = MOTION_PROJECTION_KIND

    source_step: int
    episode_tick: int
    phase: str
    reference_active: bool
    motion_active: bool
    deadline_consumed: bool
    consumed_task_key: Optional[_mailbox.LandingOutcomeShotKey]
    consumed_deadline_tick: Optional[int]
    played: Optional[bool]
    suffix_complete: Optional[bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_step", _plain_int(self.source_step, label="source_step")
        )
        object.__setattr__(
            self, "episode_tick", _plain_int(self.episode_tick, label="episode_tick")
        )
        phase = _text(self.phase, label="phase")
        if phase not in MOTION_PHASES:
            raise ContinuousRecoveryError("motion phase is unknown")
        object.__setattr__(self, "phase", phase)
        reference_active = _exact_bool(
            self.reference_active, label="reference_active"
        )
        motion_active = _exact_bool(self.motion_active, label="motion_active")
        if reference_active and motion_active:
            raise ContinuousRecoveryError(
                "reference_active and motion_active cannot both be true"
            )
        consumed = _exact_bool(self.deadline_consumed, label="deadline_consumed")
        if consumed:
            if self.consumed_task_key is None or self.consumed_deadline_tick is None:
                raise ContinuousRecoveryError(
                    "consumed deadline requires its complete task key and tick"
                )
            key = coerce_landing_outcome_shot_key(self.consumed_task_key)
            deadline = _plain_int(
                self.consumed_deadline_tick,
                label="consumed_deadline_tick",
            )
            if deadline > self.episode_tick:
                raise ContinuousRecoveryError("future deadline cannot be consumed")
            played = _exact_bool(self.played, label="played")
            suffix = _exact_bool(self.suffix_complete, label="suffix_complete")
            if not played and suffix:
                raise ContinuousRecoveryError(
                    "unplayed row cannot claim a motion suffix"
                )
        else:
            if any(
                value is not None
                for value in (
                    self.consumed_task_key,
                    self.consumed_deadline_tick,
                    self.played,
                    self.suffix_complete,
                )
            ):
                raise ContinuousRecoveryError(
                    "unconsumed projection must hide task, deadline, and suffix state"
                )
            key = None
            deadline = None
            played = None
            suffix = None
        object.__setattr__(self, "reference_active", reference_active)
        object.__setattr__(self, "motion_active", motion_active)
        object.__setattr__(self, "deadline_consumed", consumed)
        object.__setattr__(self, "consumed_task_key", key)
        object.__setattr__(self, "consumed_deadline_tick", deadline)
        object.__setattr__(self, "played", played)
        object.__setattr__(self, "suffix_complete", suffix)

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRecoveryMotionProjection":
        rows = cls._mapping_values(value)
        if rows["consumed_task_key"] is not None:
            rows["consumed_task_key"] = coerce_landing_outcome_shot_key(
                rows["consumed_task_key"]
            )
        return cls(**rows)


@dataclass(frozen=True)
class ContinuousRecoveryCommandProjection(_SealedRecord):
    """Public Command-side state with no future task/target/deadline."""

    KIND: ClassVar[str] = COMMAND_PROJECTION_KIND

    source_step: int
    episode_tick: int
    env_id: int
    reset_generation: int
    phase: str
    current_committed_task_key_sha256: Optional[str]
    reference_owner_sha256: str
    reference_active: bool
    motion_active: bool
    ready_authority: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_step", _plain_int(self.source_step, label="source_step")
        )
        object.__setattr__(
            self, "episode_tick", _plain_int(self.episode_tick, label="episode_tick")
        )
        object.__setattr__(self, "env_id", _plain_int(self.env_id, label="env_id"))
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(self.reset_generation, label="reset_generation", minimum=1),
        )
        if self.phase not in MOTION_PHASES:
            raise ContinuousRecoveryError("command phase is unknown")
        object.__setattr__(
            self,
            "current_committed_task_key_sha256",
            _optional_sha256(
                self.current_committed_task_key_sha256,
                label="current_committed_task_key_sha256",
            ),
        )
        object.__setattr__(
            self,
            "reference_owner_sha256",
            _sha256(self.reference_owner_sha256, label="reference_owner_sha256"),
        )
        for name in ("reference_active", "motion_active", "ready_authority"):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )


@dataclass(frozen=True)
class ContinuousRecoveryDoneTermProjection(_SealedRecord):
    """The shot boundary is never an RL or physical reset boundary."""

    KIND: ClassVar[str] = DONE_TERM_PROJECTION_KIND

    source_step: int
    episode_tick: int
    env_id: int
    reset_generation: int
    terminated: bool
    truncated: bool
    reset_requested: bool
    teleport_requested: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_step", _plain_int(self.source_step, label="source_step")
        )
        object.__setattr__(
            self, "episode_tick", _plain_int(self.episode_tick, label="episode_tick")
        )
        object.__setattr__(self, "env_id", _plain_int(self.env_id, label="env_id"))
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(self.reset_generation, label="reset_generation", minimum=1),
        )
        for name in (
            "terminated",
            "truncated",
            "reset_requested",
            "teleport_requested",
        ):
            if _exact_bool(getattr(self, name), label=name):
                raise ContinuousRecoveryError(
                    f"continuous shot boundary must keep {name}=False"
                )


def score_recovery_errors(
    profile: ContinuousRecoveryProfile,
    component_errors: Mapping[str, object],
) -> tuple[float, Mapping[str, float]]:
    """Return normalized additive Cauchy score and per-component kernels."""

    if not isinstance(profile, ContinuousRecoveryProfile):
        raise ContinuousRecoveryError("profile must be ContinuousRecoveryProfile")
    if not isinstance(component_errors, Mapping):
        raise ContinuousRecoveryError("component_errors must be a mapping")
    if frozenset(component_errors) != frozenset(COMPONENT_NAMES):
        raise ContinuousRecoveryError("score component-error keys differ")
    scores: dict[str, float] = {}
    numerator = 0.0
    denominator = 0.0
    for name in COMPONENT_NAMES:
        error = _finite(
            component_errors[name],
            label=f"component_errors.{name}",
            minimum=0.0,
        )
        scale = profile.component_scales[name]
        ratio = error / scale
        if ratio <= 1.0:
            kernel = 1.0 / (1.0 + ratio * ratio)
        else:
            inverse = scale / error
            kernel = (inverse * inverse) / (1.0 + inverse * inverse)
        # A finite error has a mathematically positive kernel.  Preserve that
        # contract even when the exact value is below binary64's range.
        kernel = max(kernel, _MIN_POSITIVE_FLOAT)
        scores[name] = kernel
        weight = profile.component_weights[name]
        numerator += weight * kernel
        denominator += weight
    raw_score = max(numerator / denominator, _MIN_POSITIVE_FLOAT)
    if not (0.0 < raw_score <= 1.0):  # positive for every finite error
        raise ContinuousRecoveryError("additive Cauchy score left (0, 1]")
    return raw_score, MappingProxyType(scores)


def recovery_payment_idempotency_sha256(
    *,
    profile_sha256: str,
    task_key_sha256: str,
    recovery_age_tick: int,
    source_step: int,
    consumer: str,
) -> str:
    """Seal the exact payment identity, independent of mutable row slots."""

    profile = _sha256(profile_sha256, label="profile_sha256")
    task_key = _sha256(task_key_sha256, label="task_key_sha256")
    age = _plain_int(recovery_age_tick, label="recovery_age_tick")
    source = _plain_int(source_step, label="source_step")
    if consumer != REWARD_CONSUMER:
        raise ContinuousRecoveryError("payment consumer differs")
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": PAYMENT_IDEMPOTENCY_KIND,
            "profile_sha256": profile,
            "task_key_sha256": task_key,
            "recovery_age_tick": age,
            "source_step": source,
            "consumer": consumer,
        }
    )


@dataclass(frozen=True)
class ContinuousRecoveryFact(_SealedRecord):
    """One post-physics, policy-rate recovery/readiness fact."""

    KIND: ClassVar[str] = FACT_KIND

    source_step: int
    episode_tick: int
    profile_sha256: str
    phase: str
    current_committed_task_key_sha256: Optional[str]
    recovery_owner_task_key_sha256: Optional[str]
    reference_owner_sha256: str
    recovery_age_tick: Optional[int]
    component_errors: Mapping[str, float]
    component_scores: Mapping[str, float]
    supported_foot_count: int
    facts_valid: bool
    hard_safety_ok: bool
    support_ok: bool
    reference_active: bool
    motion_active: bool
    ready_instant: bool
    ready_live: bool
    ready_streak: int
    played: Optional[bool]
    suffix_complete: Optional[bool]
    recovery_expected: bool
    reward_eligible: bool
    payment_idempotency_sha256: Optional[str]
    infrastructure_fault: str
    raw_score: float
    reward: float
    failed_ready_conjuncts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_step", _plain_int(self.source_step, label="source_step")
        )
        object.__setattr__(
            self, "episode_tick", _plain_int(self.episode_tick, label="episode_tick")
        )
        object.__setattr__(
            self,
            "profile_sha256",
            _sha256(self.profile_sha256, label="profile_sha256"),
        )
        if self.phase not in MOTION_PHASES:
            raise ContinuousRecoveryError("fact phase is unknown")
        for name in (
            "current_committed_task_key_sha256",
            "recovery_owner_task_key_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _optional_sha256(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "reference_owner_sha256",
            _sha256(self.reference_owner_sha256, label="reference_owner_sha256"),
        )
        if self.recovery_age_tick is not None:
            if type(self.recovery_age_tick) is not int:
                raise ContinuousRecoveryError("recovery_age_tick must be exact int")
        errors = _exact_number_mapping(
            self.component_errors,
            label="fact.component_errors",
            strictly_positive=False,
        )
        scores = _exact_number_mapping(
            self.component_scores,
            label="fact.component_scores",
            strictly_positive=True,
        )
        if any(score > 1.0 for score in scores.values()):
            raise ContinuousRecoveryError("component score exceeds one")
        object.__setattr__(self, "component_errors", errors)
        object.__setattr__(self, "component_scores", scores)
        object.__setattr__(
            self,
            "supported_foot_count",
            _plain_int(self.supported_foot_count, label="supported_foot_count"),
        )
        for name in (
            "facts_valid",
            "hard_safety_ok",
            "support_ok",
            "reference_active",
            "motion_active",
            "ready_instant",
            "ready_live",
            "recovery_expected",
            "reward_eligible",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )
        object.__setattr__(
            self,
            "ready_streak",
            _plain_int(self.ready_streak, label="ready_streak"),
        )
        if self.played is not None:
            object.__setattr__(
                self, "played", _exact_bool(self.played, label="played")
            )
        if self.suffix_complete is not None:
            object.__setattr__(
                self,
                "suffix_complete",
                _exact_bool(self.suffix_complete, label="suffix_complete"),
            )
        fault = _text(self.infrastructure_fault, label="infrastructure_fault")
        if fault not in (
            NO_INFRASTRUCTURE_FAULT,
            PLAYED_SUFFIX_MISSING_AT_REWARD_START,
        ):
            raise ContinuousRecoveryError("fact infrastructure fault is unknown")
        object.__setattr__(self, "infrastructure_fault", fault)
        raw = _finite(self.raw_score, label="raw_score", strictly_positive=True)
        if raw > 1.0:
            raise ContinuousRecoveryError("raw_score exceeds one")
        object.__setattr__(self, "raw_score", raw)
        object.__setattr__(
            self, "reward", _finite(self.reward, label="reward", minimum=0.0)
        )
        failed = tuple(
            _text(item, label="failed_ready_conjuncts[]")
            for item in self.failed_ready_conjuncts
        )
        if len(set(failed)) != len(failed):
            raise ContinuousRecoveryError("failed_ready_conjuncts contains duplicates")
        object.__setattr__(self, "failed_ready_conjuncts", failed)
        if self.ready_instant != (len(failed) == 0):
            raise ContinuousRecoveryError(
                "ready_instant disagrees with failed_ready_conjuncts"
            )
        if self.reward_eligible and not self.recovery_expected:
            raise ContinuousRecoveryError(
                "reward_eligible requires an expected denominator cell"
            )
        if not self.reward_eligible and self.reward != 0.0:
            raise ContinuousRecoveryError("ineligible fact must have zero reward")
        payment_id = _optional_sha256(
            self.payment_idempotency_sha256,
            label="payment_idempotency_sha256",
        )
        if self.recovery_expected != (payment_id is not None):
            raise ContinuousRecoveryError(
                "expected fact must carry exactly one payment idempotency key"
            )
        object.__setattr__(self, "payment_idempotency_sha256", payment_id)
        if self.recovery_expected:
            if (
                self.recovery_owner_task_key_sha256 is None
                or self.recovery_age_tick is None
            ):
                raise ContinuousRecoveryError(
                    "expected fact lacks reward owner or age"
                )
            expected_payment_id = recovery_payment_idempotency_sha256(
                profile_sha256=self.profile_sha256,
                task_key_sha256=self.recovery_owner_task_key_sha256,
                recovery_age_tick=self.recovery_age_tick,
                source_step=self.source_step,
                consumer=REWARD_CONSUMER,
            )
            if payment_id != expected_payment_id:
                raise ContinuousRecoveryError(
                    "payment idempotency key differs from full tuple"
                )

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRecoveryFact":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class ContinuousRecoveryRewardView(_SealedRecord):
    """Immutable one-consumer view; reading and payment are each exactly once."""

    KIND: ClassVar[str] = REWARD_VIEW_KIND

    consumer: str
    fact: ContinuousRecoveryFact
    payment_required: bool

    def __post_init__(self) -> None:
        if self.consumer != REWARD_CONSUMER:
            raise ContinuousRecoveryError("reward consumer differs")
        if not isinstance(self.fact, ContinuousRecoveryFact):
            raise ContinuousRecoveryError("reward view fact type differs")
        required = _exact_bool(self.payment_required, label="payment_required")
        if required != self.fact.recovery_expected:
            raise ContinuousRecoveryError(
                "payment_required must equal recovery_expected"
            )


@dataclass(frozen=True)
class ContinuousRecoveryLedger(_SealedRecord):
    """Per-sequence denominator and payment evidence."""

    KIND: ClassVar[str] = LEDGER_KIND

    expected_count: int
    eligible_count: int
    payment_count: int
    positive_payment_count: int
    suffix_fault_count: int
    raw_score_sum: float
    reward_sum: float
    first_expected_age_tick: Optional[int]
    last_expected_age_tick: Optional[int]

    def __post_init__(self) -> None:
        for name in (
            "expected_count",
            "eligible_count",
            "payment_count",
            "positive_payment_count",
            "suffix_fault_count",
        ):
            object.__setattr__(
                self, name, _plain_int(getattr(self, name), label=name)
            )
        for name in ("raw_score_sum", "reward_sum"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), label=name, minimum=0.0),
            )
        for name in ("first_expected_age_tick", "last_expected_age_tick"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise ContinuousRecoveryError(f"{name} must be exact int or None")
        if self.eligible_count > self.expected_count:
            raise ContinuousRecoveryError("eligible_count exceeds expected_count")
        if self.expected_count > RECOVERY_SAMPLE_COUNT:
            raise ContinuousRecoveryError("expected_count exceeds 68")
        if self.payment_count > self.expected_count:
            raise ContinuousRecoveryError("payment_count exceeds expected_count")
        if self.positive_payment_count > self.payment_count:
            raise ContinuousRecoveryError(
                "positive_payment_count exceeds payment_count"
            )
        if self.positive_payment_count > self.eligible_count:
            raise ContinuousRecoveryError(
                "positive_payment_count exceeds eligible_count"
            )
        if self.suffix_fault_count > self.expected_count:
            raise ContinuousRecoveryError(
                "suffix_fault_count exceeds expected_count"
            )
        if self.expected_count == 0:
            if (
                self.first_expected_age_tick is not None
                or self.last_expected_age_tick is not None
            ):
                raise ContinuousRecoveryError(
                    "empty ledger must not claim expected ages"
                )
        else:
            if self.first_expected_age_tick != RECOVERY_START_AGE_TICK:
                raise ContinuousRecoveryError(
                    "ledger first expected age must equal 10"
                )
            if self.last_expected_age_tick != (
                RECOVERY_START_AGE_TICK + self.expected_count - 1
            ):
                raise ContinuousRecoveryError(
                    "ledger expected ages must be consecutive"
                )

    @classmethod
    def empty(cls) -> "ContinuousRecoveryLedger":
        return cls(
            expected_count=0,
            eligible_count=0,
            payment_count=0,
            positive_payment_count=0,
            suffix_fault_count=0,
            raw_score_sum=0.0,
            reward_sum=0.0,
            first_expected_age_tick=None,
            last_expected_age_tick=None,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRecoveryLedger":
        return cls(**cls._mapping_values(value))


@dataclass(frozen=True)
class ContinuousRecoveryCheckpoint(_SealedRecord):
    """Canonical state carrying lineage authority, not its restore content pin."""

    KIND: ClassVar[str] = CHECKPOINT_KIND

    integration_status: str
    profile_sha256: str
    external_authority_sha256: str
    env_id: int
    reset_generation: int
    sequence_birth_owner: Optional[ContinuousRecoveryReferenceOwner]
    active_ready_reference_owner: Optional[ContinuousRecoveryReferenceOwner]
    pending_committed_reference_owner: Optional[ContinuousRecoveryReferenceOwner]
    current_committed_task_key: Optional[_mailbox.LandingOutcomeShotKey]
    recovery_owner_task_key: Optional[_mailbox.LandingOutcomeShotKey]
    recovery_deadline_tick: Optional[int]
    played: Optional[bool]
    suffix_complete: Optional[bool]
    infrastructure_fault: str
    phase: str
    reference_active: bool
    motion_active: bool
    last_source_step: int
    episode_tick: int
    last_publish_tick: Optional[int]
    ready_streak: int
    ready_live: bool
    first_ready_tick: Optional[int]
    latest_fact: Optional[ContinuousRecoveryFact]
    latest_viewed: bool
    latest_paid: bool
    latest_payment_step: Optional[int]
    paid_payment_idempotency_sha256s: tuple[str, ...]
    ledger: ContinuousRecoveryLedger

    def __post_init__(self) -> None:
        if self.integration_status != INTEGRATION_STATUS:
            raise ContinuousRecoveryError("checkpoint integration status differs")
        object.__setattr__(
            self,
            "profile_sha256",
            _sha256(self.profile_sha256, label="profile_sha256"),
        )
        object.__setattr__(
            self,
            "external_authority_sha256",
            _sha256(
                self.external_authority_sha256,
                label="external_authority_sha256",
            ),
        )
        object.__setattr__(self, "env_id", _plain_int(self.env_id, label="env_id"))
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(self.reset_generation, label="reset_generation", minimum=1),
        )
        for name in (
            "sequence_birth_owner",
            "active_ready_reference_owner",
            "pending_committed_reference_owner",
        ):
            owner = getattr(self, name)
            if owner is not None and not isinstance(
                owner, ContinuousRecoveryReferenceOwner
            ):
                raise ContinuousRecoveryError(f"{name} type differs")
        for name in (
            "current_committed_task_key",
            "recovery_owner_task_key",
        ):
            key = getattr(self, name)
            if key is not None:
                object.__setattr__(self, name, coerce_landing_outcome_shot_key(key))
        object.__setattr__(
            self,
            "recovery_deadline_tick",
            _optional_plain_int(
                self.recovery_deadline_tick, label="recovery_deadline_tick"
            ),
        )
        for name in ("played", "suffix_complete"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _exact_bool(value, label=name))
        if self.infrastructure_fault not in (
            NO_INFRASTRUCTURE_FAULT,
            PLAYED_SUFFIX_MISSING_AT_REWARD_START,
        ):
            raise ContinuousRecoveryError("checkpoint infrastructure fault differs")
        if self.phase not in MOTION_PHASES:
            raise ContinuousRecoveryError("checkpoint phase differs")
        for name in (
            "reference_active",
            "motion_active",
            "ready_live",
            "latest_viewed",
            "latest_paid",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )
        if self.reference_active and self.motion_active:
            raise ContinuousRecoveryError("checkpoint activates motion and reference")
        object.__setattr__(
            self,
            "last_source_step",
            _plain_int(self.last_source_step, label="last_source_step"),
        )
        object.__setattr__(
            self, "episode_tick", _plain_int(self.episode_tick, label="episode_tick")
        )
        object.__setattr__(
            self,
            "last_publish_tick",
            _optional_plain_int(self.last_publish_tick, label="last_publish_tick"),
        )
        object.__setattr__(
            self,
            "ready_streak",
            _plain_int(self.ready_streak, label="ready_streak"),
        )
        object.__setattr__(
            self,
            "first_ready_tick",
            _optional_plain_int(self.first_ready_tick, label="first_ready_tick"),
        )
        if self.latest_fact is not None and not isinstance(
            self.latest_fact, ContinuousRecoveryFact
        ):
            raise ContinuousRecoveryError("checkpoint latest_fact type differs")
        object.__setattr__(
            self,
            "latest_payment_step",
            _optional_plain_int(
                self.latest_payment_step, label="latest_payment_step"
            ),
        )
        paid = tuple(
            _sha256(value, label="paid_payment_idempotency_sha256s[]")
            for value in self.paid_payment_idempotency_sha256s
        )
        if len(set(paid)) != len(paid):
            raise ContinuousRecoveryError("paid fact replay guard has duplicates")
        object.__setattr__(self, "paid_payment_idempotency_sha256s", paid)
        if not isinstance(self.ledger, ContinuousRecoveryLedger):
            raise ContinuousRecoveryError("checkpoint ledger type differs")
        if self.ledger.payment_count != len(paid):
            raise ContinuousRecoveryError(
                "checkpoint payment ledger and idempotency guard differ"
            )
        if self.recovery_owner_task_key is None:
            if any(
                value is not None
                for value in (
                    self.recovery_deadline_tick,
                    self.played,
                    self.suffix_complete,
                )
            ):
                raise ContinuousRecoveryError(
                    "checkpoint has recovery facts without a reward owner"
                )
            if (
                self.infrastructure_fault != NO_INFRASTRUCTURE_FAULT
                or self.ledger.expected_count != 0
                or paid
            ):
                raise ContinuousRecoveryError(
                    "checkpoint has recovery evidence without a reward owner"
                )
        else:
            if (
                self.recovery_deadline_tick is None
                or self.played is None
                or self.suffix_complete is None
            ):
                raise ContinuousRecoveryError(
                    "checkpoint reward owner lacks deadline/play/suffix facts"
                )
            if not self.played and self.suffix_complete:
                raise ContinuousRecoveryError(
                    "checkpoint unplayed owner claims a completed suffix"
                )
        if self.latest_fact is None:
            if self.latest_viewed or self.latest_paid or self.latest_payment_step is not None:
                raise ContinuousRecoveryError("checkpoint has payment state without fact")
        else:
            if self.latest_paid and not self.latest_viewed:
                raise ContinuousRecoveryError("checkpoint paid fact was never viewed")
            in_guard = self.latest_fact.payment_idempotency_sha256 in set(paid)
            if in_guard != self.latest_paid:
                raise ContinuousRecoveryError("checkpoint payment replay guard differs")
            if self.latest_paid != (self.latest_payment_step is not None):
                raise ContinuousRecoveryError("checkpoint payment step differs")
            if self.latest_fact.recovery_expected:
                if self.recovery_owner_task_key is None:
                    raise ContinuousRecoveryError(
                        "checkpoint expected fact lacks a reward owner"
                    )
                if self.latest_fact.recovery_owner_task_key_sha256 != (
                    self.recovery_owner_task_key.canonical_sha256
                ):
                    raise ContinuousRecoveryError(
                        "checkpoint latest fact reward owner differs"
                    )
                if self.latest_fact.recovery_age_tick != (
                    self.ledger.last_expected_age_tick
                ):
                    raise ContinuousRecoveryError(
                        "checkpoint latest expected age and ledger differ"
                    )
                expected_payments = self.ledger.expected_count - int(
                    not self.latest_paid
                )
                if self.ledger.payment_count != expected_payments:
                    raise ContinuousRecoveryError(
                        "checkpoint latest payment state and ledger differ"
                    )

    @classmethod
    def from_mapping(cls, value: object) -> "ContinuousRecoveryCheckpoint":
        rows = cls._mapping_values(value)
        for name in (
            "sequence_birth_owner",
            "active_ready_reference_owner",
            "pending_committed_reference_owner",
        ):
            if rows[name] is not None:
                rows[name] = ContinuousRecoveryReferenceOwner.from_mapping(rows[name])
        for name in (
            "current_committed_task_key",
            "recovery_owner_task_key",
        ):
            if rows[name] is not None:
                rows[name] = coerce_landing_outcome_shot_key(rows[name])
        if rows["latest_fact"] is not None:
            rows["latest_fact"] = ContinuousRecoveryFact.from_mapping(
                rows["latest_fact"]
            )
        rows["ledger"] = ContinuousRecoveryLedger.from_mapping(rows["ledger"])
        return cls(**rows)


@dataclass(frozen=True)
class _RuntimeState:
    sequence_birth_owner: Optional[ContinuousRecoveryReferenceOwner]
    active_ready_reference_owner: Optional[ContinuousRecoveryReferenceOwner]
    pending_committed_reference_owner: Optional[ContinuousRecoveryReferenceOwner]
    current_committed_task_key: Optional[_mailbox.LandingOutcomeShotKey]
    recovery_owner_task_key: Optional[_mailbox.LandingOutcomeShotKey]
    recovery_deadline_tick: Optional[int]
    played: Optional[bool]
    suffix_complete: Optional[bool]
    infrastructure_fault: str
    phase: str
    reference_active: bool
    motion_active: bool
    last_source_step: int
    episode_tick: int
    last_publish_tick: Optional[int]
    ready_streak: int
    ready_live: bool
    first_ready_tick: Optional[int]
    latest_fact: Optional[ContinuousRecoveryFact]
    latest_viewed: bool
    latest_paid: bool
    latest_payment_step: Optional[int]
    paid_payment_idempotency_sha256s: tuple[str, ...]
    ledger: ContinuousRecoveryLedger


class ContinuousRecoveryRuntime:
    """Atomic one-environment owner for R07 recovery and readiness state."""

    integration_status = INTEGRATION_STATUS
    runtime_wiring_connected = RUNTIME_WIRING_CONNECTED

    def __init__(
        self,
        *,
        profile: ContinuousRecoveryProfile,
        env_id: int,
        reset_generation: int,
        fault_injector: _FaultInjector = None,
    ) -> None:
        if not isinstance(profile, ContinuousRecoveryProfile):
            raise ContinuousRecoveryError("profile must be ContinuousRecoveryProfile")
        self.profile = profile
        self.env_id = _plain_int(env_id, label="env_id")
        self.reset_generation = _plain_int(
            reset_generation, label="reset_generation", minimum=1
        )
        if fault_injector is not None and not callable(fault_injector):
            raise ContinuousRecoveryError("fault_injector must be callable or None")
        self._fault_injector = fault_injector
        self._state = _RuntimeState(
            sequence_birth_owner=None,
            active_ready_reference_owner=None,
            pending_committed_reference_owner=None,
            current_committed_task_key=None,
            recovery_owner_task_key=None,
            recovery_deadline_tick=None,
            played=None,
            suffix_complete=None,
            infrastructure_fault=NO_INFRASTRUCTURE_FAULT,
            phase="pre_reveal_hidden",
            reference_active=False,
            motion_active=False,
            last_source_step=0,
            episode_tick=0,
            last_publish_tick=None,
            ready_streak=0,
            ready_live=False,
            first_ready_tick=None,
            latest_fact=None,
            latest_viewed=False,
            latest_paid=False,
            latest_payment_step=None,
            paid_payment_idempotency_sha256s=(),
            ledger=ContinuousRecoveryLedger.empty(),
        )

    def _inject(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _require_owner_environment(
        self, owner: ContinuousRecoveryReferenceOwner
    ) -> None:
        if not isinstance(owner, ContinuousRecoveryReferenceOwner):
            raise ContinuousRecoveryError("reference owner type differs")
        if (
            owner.env_id != self.env_id
            or owner.reset_generation != self.reset_generation
        ):
            raise ContinuousRecoveryConflictError(
                "reference owner belongs to another environment generation"
            )

    def _require_key_environment(
        self, value: object
    ) -> _mailbox.LandingOutcomeShotKey:
        key = coerce_landing_outcome_shot_key(value)
        if key.env_id != self.env_id or key.reset_generation != self.reset_generation:
            raise ContinuousRecoveryConflictError(
                "task key belongs to another environment generation"
            )
        if (
            key.source_sha256 != self.profile.source_sha256
            or key.config_sha256 != self.profile.config_sha256
        ):
            raise ContinuousRecoveryConflictError(
                "task key source/config differs from the recovery profile"
            )
        return key

    def _replace_state(self, staged: _RuntimeState, *, stage: str) -> None:
        self._inject(stage)
        self._state = staged

    def bind_sequence_birth(
        self,
        *,
        source_step: int,
        episode_tick: int,
        reference_owner: ContinuousRecoveryReferenceOwner,
    ) -> None:
        """Bind the frame-0, zero-velocity reference before the first reveal."""

        source = _plain_int(source_step, label="source_step")
        tick = _plain_int(episode_tick, label="episode_tick")
        self._require_owner_environment(reference_owner)
        if reference_owner.owner_kind != SEQUENCE_BIRTH_OWNER:
            raise ContinuousRecoveryError("sequence birth requires birth reference")
        state = self._state
        if state.sequence_birth_owner is not None:
            raise ContinuousRecoveryConflictError("sequence birth already bound")
        if source < state.last_source_step or tick < state.episode_tick:
            raise ContinuousRecoveryConflictError("sequence birth clock regressed")
        staged = _RuntimeState(
            **{
                **state.__dict__,
                "sequence_birth_owner": reference_owner,
                "active_ready_reference_owner": reference_owner,
                "last_source_step": source,
                "episode_tick": tick,
                "ready_streak": 0,
                "ready_live": False,
                "first_ready_tick": None,
            }
        )
        self._replace_state(staged, stage="before_bind_sequence_birth_commit")

    def commit_reveal(
        self,
        *,
        source_step: int,
        episode_tick: int,
        task_key: object,
        reference_owner: ContinuousRecoveryReferenceOwner,
    ) -> None:
        """Publish the latest full key and keep its frame-0 owner pending.

        A commit is not evidence that a played motion suffix finished.  It
        therefore must not switch the active ready reference or clear an
        already earned dwell.  ``complete_suffix`` is the only promotion
        operation.
        """

        source = _plain_int(source_step, label="source_step")
        tick = _plain_int(episode_tick, label="episode_tick")
        key = self._require_key_environment(task_key)
        self._require_owner_environment(reference_owner)
        if reference_owner.owner_kind != COMMITTED_TASK_OWNER:
            raise ContinuousRecoveryError("committed reveal requires task reference")
        if reference_owner.task_key != key:
            raise ContinuousRecoveryConflictError(
                "reference frame-0 owner does not match committed full key"
            )
        state = self._state
        if state.sequence_birth_owner is None:
            raise ContinuousRecoveryConflictError("sequence birth is not bound")
        if source < state.last_source_step or tick < state.episode_tick:
            raise ContinuousRecoveryConflictError("commit clock regressed")
        if state.current_committed_task_key == key:
            raise ContinuousRecoveryConflictError("task key already committed")
        previous = state.current_committed_task_key
        if previous is not None:
            for name in (
                "env_id",
                "reset_generation",
                "action_uid",
                "action_slot",
                "birth_sha256",
                "run_id",
                "carry_chain_id",
                "source_sha256",
                "config_sha256",
            ):
                if getattr(key, name) != getattr(previous, name):
                    raise ContinuousRecoveryConflictError(
                        f"committed successor changed stable lineage field {name}"
                    )
            for name in (
                "sample_sha256",
                "task_sha256",
                "receipt_content_sha256",
            ):
                if getattr(key, name) == getattr(previous, name):
                    raise ContinuousRecoveryConflictError(
                        f"committed successor reused {name}"
                    )
            if (
                state.recovery_owner_task_key != previous
                or state.recovery_deadline_tick is None
            ):
                raise ContinuousRecoveryConflictError(
                    "next commit would overwrite a task before deadline close"
                )
            if tick - state.recovery_deadline_tick < RECOVERY_END_AGE_TICK + 1:
                raise ContinuousRecoveryConflictError(
                    "next reveal precedes deadline plus 78 ticks"
                )
            if (
                state.ledger.expected_count != RECOVERY_SAMPLE_COUNT
                or state.ledger.payment_count != RECOVERY_SAMPLE_COUNT
                or len(state.paid_payment_idempotency_sha256s)
                != RECOVERY_SAMPLE_COUNT
            ):
                raise ContinuousRecoveryInfrastructureError(
                    "next commit lacks the previous complete 68-cell ledger"
                )
            if state.played is True and not bool(state.suffix_complete):
                raise ContinuousRecoveryInfrastructureError(
                    "next commit would overwrite an incomplete played suffix"
                )
            if key.shot_index != previous.shot_index + 1:
                raise ContinuousRecoveryConflictError(
                    "committed shot_index is not the exact successor"
                )
            if key.swing_generation != previous.swing_generation + 1:
                raise ContinuousRecoveryConflictError(
                    "committed swing_generation is not the exact successor"
                )
        staged = _RuntimeState(
            **{
                **state.__dict__,
                "pending_committed_reference_owner": reference_owner,
                "current_committed_task_key": key,
                "last_source_step": source,
                "episode_tick": tick,
            }
        )
        self._replace_state(staged, stage="before_commit_reveal_commit")

    def reconcile_motion_projection(
        self,
        projection: ContinuousRecoveryMotionProjection,
    ) -> None:
        """Latch a consumed deadline without ever exposing a future deadline."""

        if not isinstance(projection, ContinuousRecoveryMotionProjection):
            raise ContinuousRecoveryError("projection type differs")
        state = self._state
        if (
            projection.source_step < state.last_source_step
            or projection.episode_tick < state.episode_tick
        ):
            raise ContinuousRecoveryConflictError("motion projection clock regressed")
        key = projection.consumed_task_key
        deadline = projection.consumed_deadline_tick
        played = projection.played
        suffix = projection.suffix_complete
        recovery_key = state.recovery_owner_task_key
        recovery_deadline = state.recovery_deadline_tick
        stored_played = state.played
        stored_suffix = state.suffix_complete
        fault = state.infrastructure_fault
        ledger = state.ledger
        paid_payment_ids = state.paid_payment_idempotency_sha256s
        latest_fact = state.latest_fact
        latest_viewed = state.latest_viewed
        latest_paid = state.latest_paid
        latest_payment_step = state.latest_payment_step
        if projection.deadline_consumed:
            assert key is not None and deadline is not None
            assert played is not None and suffix is not None
            key = self._require_key_environment(key)
            if state.current_committed_task_key is None:
                raise ContinuousRecoveryConflictError(
                    "deadline consumed before a task commit"
                )
            if key != state.current_committed_task_key:
                raise ContinuousRecoveryConflictError(
                    "deadline owner is not the current committed full key"
                )
            if recovery_key is None:
                if projection.episode_tick != deadline:
                    raise ContinuousRecoveryInfrastructureError(
                        "reward owner was not closed on its consumed deadline tick"
                    )
                if played and suffix:
                    raise ContinuousRecoveryConflictError(
                        "played suffix must be promoted by complete_suffix"
                    )
                recovery_key = key
                recovery_deadline = deadline
                stored_played = played
                stored_suffix = suffix
                fault = NO_INFRASTRUCTURE_FAULT
                ledger = ContinuousRecoveryLedger.empty()
                paid_payment_ids = ()
                latest_fact = None
                latest_viewed = False
                latest_paid = False
                latest_payment_step = None
            elif recovery_key == key:
                if recovery_deadline != deadline or stored_played != played:
                    raise ContinuousRecoveryConflictError(
                        "latched recovery owner facts changed"
                    )
                if stored_suffix and not suffix:
                    raise ContinuousRecoveryConflictError(
                        "completed suffix cannot become incomplete"
                    )
                if suffix and not stored_suffix:
                    raise ContinuousRecoveryConflictError(
                        "suffix promotion requires complete_suffix"
                    )
            else:
                assert recovery_deadline is not None
                if projection.episode_tick - recovery_deadline <= RECOVERY_END_AGE_TICK:
                    raise ContinuousRecoveryConflictError(
                        "new recovery owner would overwrite a live denominator"
                    )
                if (
                    ledger.expected_count != RECOVERY_SAMPLE_COUNT
                    or ledger.payment_count != RECOVERY_SAMPLE_COUNT
                    or len(paid_payment_ids) != RECOVERY_SAMPLE_COUNT
                ):
                    raise ContinuousRecoveryInfrastructureError(
                        "previous recovery owner lacks its complete 68-cell ledger"
                    )
                if played and suffix:
                    raise ContinuousRecoveryConflictError(
                        "played suffix must be promoted by complete_suffix"
                    )
                if projection.episode_tick != deadline:
                    raise ContinuousRecoveryInfrastructureError(
                        "next reward owner was not closed on its deadline tick"
                    )
                recovery_key = key
                recovery_deadline = deadline
                stored_played = played
                stored_suffix = suffix
                fault = NO_INFRASTRUCTURE_FAULT
                ledger = ContinuousRecoveryLedger.empty()
                paid_payment_ids = ()
                latest_fact = None
                latest_viewed = False
                latest_paid = False
                latest_payment_step = None
            age = projection.episode_tick - deadline
            if (
                played
                and age >= RECOVERY_START_AGE_TICK
                and not bool(stored_suffix)
            ):
                fault = PLAYED_SUFFIX_MISSING_AT_REWARD_START
        if recovery_deadline is not None:
            latched_age = projection.episode_tick - recovery_deadline
            if (
                stored_played is True
                and latched_age >= RECOVERY_START_AGE_TICK
                and not bool(stored_suffix)
            ):
                fault = PLAYED_SUFFIX_MISSING_AT_REWARD_START
        staged = _RuntimeState(
            **{
                **state.__dict__,
                "recovery_owner_task_key": recovery_key,
                "recovery_deadline_tick": recovery_deadline,
                "played": stored_played,
                "suffix_complete": stored_suffix,
                "infrastructure_fault": fault,
                "ledger": ledger,
                "paid_payment_idempotency_sha256s": paid_payment_ids,
                "latest_fact": latest_fact,
                "latest_viewed": latest_viewed,
                "latest_paid": latest_paid,
                "latest_payment_step": latest_payment_step,
                "phase": projection.phase,
                "reference_active": projection.reference_active,
                "motion_active": projection.motion_active,
                "last_source_step": projection.source_step,
                "episode_tick": projection.episode_tick,
            }
        )
        self._replace_state(staged, stage="before_reconcile_motion_commit")

    def close_deadline(
        self,
        projection: ContinuousRecoveryMotionProjection,
    ) -> None:
        """Create the next reward owner from one newly consumed deadline.

        The operation intentionally does not change the active ready reference
        and does not clear its dwell.  Later projections for the same owner
        use ``reconcile_motion_projection``; replaying this close is rejected.
        """

        if not isinstance(projection, ContinuousRecoveryMotionProjection):
            raise ContinuousRecoveryError("projection type differs")
        if not projection.deadline_consumed:
            raise ContinuousRecoveryError(
                "close_deadline requires a consumed deadline projection"
            )
        key = projection.consumed_task_key
        assert key is not None
        if self._state.recovery_owner_task_key == key:
            raise ContinuousRecoveryConflictError("deadline owner already closed")
        self.reconcile_motion_projection(projection)

    def complete_suffix(
        self,
        *,
        source_step: int,
        episode_tick: int,
        task_key: object,
    ) -> None:
        """Promote one played row's pending frame-0 reference exactly once.

        This is the only transition which changes the active ready reference
        after sequence birth.  Promotion is exact-full-key bound and clears
        the consecutive dwell because the reference itself changed.
        """

        source = _plain_int(source_step, label="source_step")
        tick = _plain_int(episode_tick, label="episode_tick")
        key = self._require_key_environment(task_key)
        state = self._state
        if source < state.last_source_step or tick < state.episode_tick:
            raise ContinuousRecoveryConflictError("suffix completion clock regressed")
        if state.recovery_owner_task_key != key:
            raise ContinuousRecoveryConflictError(
                "suffix completion does not match the reward owner full key"
            )
        if state.current_committed_task_key != key:
            raise ContinuousRecoveryConflictError(
                "suffix completion does not match the current committed full key"
            )
        if state.played is not True:
            raise ContinuousRecoveryConflictError(
                "only a played row can complete a motion suffix"
            )
        if state.suffix_complete:
            raise ContinuousRecoveryConflictError("motion suffix already completed")
        if state.recovery_deadline_tick is None:
            raise ContinuousRecoveryConflictError("suffix has no consumed deadline")
        if tick - state.recovery_deadline_tick > RECOVERY_START_AGE_TICK:
            raise ContinuousRecoveryInfrastructureError(
                "played suffix completed after recovery age 10"
            )
        if state.infrastructure_fault != NO_INFRASTRUCTURE_FAULT:
            raise ContinuousRecoveryInfrastructureError(
                "sticky suffix infrastructure fault cannot be repaired in place"
            )
        pending = state.pending_committed_reference_owner
        if pending is None or pending.task_key != key:
            raise ContinuousRecoveryConflictError(
                "suffix completion lacks its pending frame-0 reference"
            )
        staged = _RuntimeState(
            **{
                **state.__dict__,
                "active_ready_reference_owner": pending,
                "suffix_complete": True,
                "last_source_step": source,
                "episode_tick": tick,
                "ready_streak": 0,
                "ready_live": False,
                "first_ready_tick": None,
            }
        )
        self._replace_state(staged, stage="before_complete_suffix_commit")

    def publish_after_physics(
        self,
        *,
        source_step: int,
        episode_tick: int,
        component_errors: Mapping[str, object],
        foot_contact_signals: Mapping[str, object],
        facts_valid: bool,
        hard_safety_ok: bool,
    ) -> ContinuousRecoveryFact:
        """Publish one policy-rate fact after Motion projection and physics."""

        source = _plain_int(source_step, label="source_step")
        tick = _plain_int(episode_tick, label="episode_tick")
        state = self._state
        if source != state.last_source_step or tick != state.episode_tick:
            raise ContinuousRecoveryConflictError(
                "publish must match the reconciled Motion clock"
            )
        if state.active_ready_reference_owner is None:
            raise ContinuousRecoveryConflictError("no recovery reference is bound")
        if state.last_publish_tick is not None and tick <= state.last_publish_tick:
            raise ContinuousRecoveryConflictError("policy tick already published")
        if (
            state.latest_fact is not None
            and state.latest_fact.recovery_expected
            and not state.latest_paid
        ):
            raise ContinuousRecoveryConflictError(
                "expected recovery fact must be viewed and paid before overwrite"
            )
        errors = dict(_component_errors(component_errors))
        contacts = _contact_signals(
            foot_contact_signals,
            ordered_foot_names=self.profile.ordered_foot_names,
        )
        supported_count = sum(
            value >= self.profile.support_contact_threshold
            for value in contacts.values()
        )
        support_deficit = max(
            0, self.profile.minimum_supported_feet - supported_count
        )
        errors["foot_support_deficit"] = float(support_deficit)
        raw_score, scores = score_recovery_errors(self.profile, errors)
        facts_ok = _exact_bool(facts_valid, label="facts_valid")
        safety_ok = _exact_bool(hard_safety_ok, label="hard_safety_ok")
        support_ok = supported_count >= self.profile.minimum_supported_feet

        failed: list[str] = []
        if not state.reference_active:
            failed.append("reference_inactive")
        if state.motion_active:
            failed.append("motion_active")
        if not facts_ok:
            failed.append("facts_invalid")
        if not safety_ok:
            failed.append("hard_safety_failed")
        if not support_ok:
            failed.append("foot_support_failed")
        if state.played is True and not bool(state.suffix_complete):
            failed.append("played_suffix_incomplete")
        for name in COMPONENT_NAMES:
            if errors[name] > self.profile.ready_tolerances[name]:
                failed.append(f"{name}_outside_tolerance")
        ready_instant = not failed
        consecutive = (
            state.last_publish_tick is not None
            and tick == state.last_publish_tick + 1
        )
        if ready_instant:
            ready_streak = state.ready_streak + 1 if consecutive else 1
        else:
            ready_streak = 0
        ready_live = ready_streak >= self.profile.ready_dwell_ticks
        first_ready_tick = state.first_ready_tick
        if ready_live and first_ready_tick is None:
            first_ready_tick = tick

        recovery_age: Optional[int] = None
        expected = False
        full_key_match = False
        if (
            state.recovery_owner_task_key is not None
            and state.recovery_deadline_tick is not None
        ):
            recovery_age = tick - state.recovery_deadline_tick
            full_key_match = (
                state.current_committed_task_key == state.recovery_owner_task_key
            )
            expected = (
                RECOVERY_START_AGE_TICK
                <= recovery_age
                <= RECOVERY_END_AGE_TICK
                and full_key_match
            )
        if (
            expected
            and state.ledger.expected_count == 0
            and recovery_age != RECOVERY_START_AGE_TICK
        ):
            raise ContinuousRecoveryInfrastructureError(
                "first expected recovery denominator cell is not age 10"
            )
        if (
            expected
            and state.last_publish_tick is not None
            and state.latest_fact is not None
            and state.latest_fact.recovery_age_tick is not None
            and state.latest_fact.recovery_expected
            and state.latest_fact.recovery_owner_task_key_sha256
            == state.recovery_owner_task_key.canonical_sha256
            and recovery_age != state.latest_fact.recovery_age_tick + 1
        ):
            raise ContinuousRecoveryInfrastructureError(
                "expected recovery denominator is not consecutive"
            )
        suffix_allows = state.played is False or bool(state.suffix_complete)
        hard_context = (
            state.reference_active
            and not state.motion_active
            and facts_ok
            and full_key_match
        )
        eligible = (
            expected
            and hard_context
            and suffix_allows
            and state.infrastructure_fault == NO_INFRASTRUCTURE_FAULT
        )
        reward = self.profile.reward_weight * raw_score if eligible else 0.0
        payment_idempotency_sha256 = None
        if expected:
            assert state.recovery_owner_task_key is not None
            assert recovery_age is not None
            payment_idempotency_sha256 = recovery_payment_idempotency_sha256(
                profile_sha256=self.profile.canonical_sha256,
                task_key_sha256=(
                    state.recovery_owner_task_key.canonical_sha256
                ),
                recovery_age_tick=recovery_age,
                source_step=source,
                consumer=REWARD_CONSUMER,
            )

        fact = ContinuousRecoveryFact(
            source_step=source,
            episode_tick=tick,
            profile_sha256=self.profile.canonical_sha256,
            phase=state.phase,
            current_committed_task_key_sha256=(
                None
                if state.current_committed_task_key is None
                else state.current_committed_task_key.canonical_sha256
            ),
            recovery_owner_task_key_sha256=(
                None
                if state.recovery_owner_task_key is None
                else state.recovery_owner_task_key.canonical_sha256
            ),
            reference_owner_sha256=(
                state.active_ready_reference_owner.canonical_sha256
            ),
            recovery_age_tick=recovery_age,
            component_errors=errors,
            component_scores=scores,
            supported_foot_count=supported_count,
            facts_valid=facts_ok,
            hard_safety_ok=safety_ok,
            support_ok=support_ok,
            reference_active=state.reference_active,
            motion_active=state.motion_active,
            ready_instant=ready_instant,
            ready_live=ready_live,
            ready_streak=ready_streak,
            played=state.played,
            suffix_complete=state.suffix_complete,
            recovery_expected=expected,
            reward_eligible=eligible,
            payment_idempotency_sha256=payment_idempotency_sha256,
            infrastructure_fault=state.infrastructure_fault,
            raw_score=raw_score,
            reward=reward,
            failed_ready_conjuncts=tuple(failed),
        )
        ledger = state.ledger
        if expected:
            ledger = ContinuousRecoveryLedger(
                expected_count=ledger.expected_count + 1,
                eligible_count=ledger.eligible_count + int(eligible),
                payment_count=ledger.payment_count,
                positive_payment_count=ledger.positive_payment_count,
                suffix_fault_count=ledger.suffix_fault_count
                + int(
                    state.infrastructure_fault
                    == PLAYED_SUFFIX_MISSING_AT_REWARD_START
                ),
                raw_score_sum=ledger.raw_score_sum + raw_score,
                reward_sum=ledger.reward_sum,
                first_expected_age_tick=(
                    recovery_age
                    if ledger.first_expected_age_tick is None
                    else ledger.first_expected_age_tick
                ),
                last_expected_age_tick=recovery_age,
            )
        staged = _RuntimeState(
            **{
                **state.__dict__,
                "last_publish_tick": tick,
                "ready_streak": ready_streak,
                "ready_live": ready_live,
                "first_ready_tick": first_ready_tick,
                "latest_fact": fact,
                "latest_viewed": False,
                "latest_paid": False,
                "latest_payment_step": None,
                "ledger": ledger,
            }
        )
        self._replace_state(staged, stage="before_publish_after_physics_commit")
        return fact

    def reward_view(
        self,
        *,
        consumer: str,
        fact_sha256: str,
    ) -> ContinuousRecoveryRewardView:
        """Read the latest immutable fact exactly once for the common reward."""

        if consumer != REWARD_CONSUMER:
            raise ContinuousRecoveryError("unknown recovery reward consumer")
        fact_sha = _sha256(fact_sha256, label="fact_sha256")
        state = self._state
        if state.latest_fact is None:
            raise ContinuousRecoveryConflictError("no recovery fact is published")
        if state.latest_fact.canonical_sha256 != fact_sha:
            raise ContinuousRecoveryConflictError("reward view fact key differs")
        if state.latest_viewed:
            raise ContinuousRecoveryConflictError("reward fact already viewed")
        view = ContinuousRecoveryRewardView(
            consumer=consumer,
            fact=state.latest_fact,
            payment_required=state.latest_fact.recovery_expected,
        )
        staged = _RuntimeState(**{**state.__dict__, "latest_viewed": True})
        self._replace_state(staged, stage="before_reward_view_commit")
        return view

    def record_reward_payment(
        self,
        *,
        consumer: str,
        fact_sha256: str,
        payment_step: int,
        applied_reward: object,
    ) -> None:
        """Record exactly one numerical payment for an expected denominator cell."""

        if consumer != REWARD_CONSUMER:
            raise ContinuousRecoveryError("unknown recovery reward consumer")
        fact_sha = _sha256(fact_sha256, label="fact_sha256")
        payment = _finite(applied_reward, label="applied_reward", minimum=0.0)
        step = _plain_int(payment_step, label="payment_step")
        state = self._state
        fact = state.latest_fact
        if fact is None or fact.canonical_sha256 != fact_sha:
            raise ContinuousRecoveryConflictError("payment fact key differs")
        if not fact.recovery_expected:
            raise ContinuousRecoveryConflictError(
                "non-denominator fact must not create a payment row"
            )
        if not state.latest_viewed:
            raise ContinuousRecoveryConflictError("payment requires reward_view")
        payment_id = fact.payment_idempotency_sha256
        assert payment_id is not None
        if (
            state.latest_paid
            or payment_id in state.paid_payment_idempotency_sha256s
        ):
            raise ContinuousRecoveryConflictError("reward fact already paid")
        if step < fact.source_step:
            raise ContinuousRecoveryConflictError("payment precedes source fact")
        if payment != fact.reward:
            raise ContinuousRecoveryConflictError(
                "applied reward differs from immutable reward fact"
            )
        ledger = state.ledger
        ledger = ContinuousRecoveryLedger(
            expected_count=ledger.expected_count,
            eligible_count=ledger.eligible_count,
            payment_count=ledger.payment_count + 1,
            positive_payment_count=ledger.positive_payment_count + int(payment > 0.0),
            suffix_fault_count=ledger.suffix_fault_count,
            raw_score_sum=ledger.raw_score_sum,
            reward_sum=ledger.reward_sum + payment,
            first_expected_age_tick=ledger.first_expected_age_tick,
            last_expected_age_tick=ledger.last_expected_age_tick,
        )
        staged = _RuntimeState(
            **{
                **state.__dict__,
                "latest_paid": True,
                "latest_payment_step": step,
                "paid_payment_idempotency_sha256s": (
                    *state.paid_payment_idempotency_sha256s,
                    payment_id,
                ),
                "ledger": ledger,
            }
        )
        self._replace_state(staged, stage="before_record_reward_payment_commit")

    def command_projection(self) -> ContinuousRecoveryCommandProjection:
        state = self._state
        if state.active_ready_reference_owner is None:
            raise ContinuousRecoveryConflictError("no reference owner is bound")
        return ContinuousRecoveryCommandProjection(
            source_step=state.last_source_step,
            episode_tick=state.episode_tick,
            env_id=self.env_id,
            reset_generation=self.reset_generation,
            phase=state.phase,
            current_committed_task_key_sha256=(
                None
                if state.current_committed_task_key is None
                else state.current_committed_task_key.canonical_sha256
            ),
            reference_owner_sha256=(
                state.active_ready_reference_owner.canonical_sha256
            ),
            reference_active=state.reference_active,
            motion_active=state.motion_active,
            ready_authority=self._effective_ready_authority(state),
        )

    def done_term_projection(self) -> ContinuousRecoveryDoneTermProjection:
        state = self._state
        return ContinuousRecoveryDoneTermProjection(
            source_step=state.last_source_step,
            episode_tick=state.episode_tick,
            env_id=self.env_id,
            reset_generation=self.reset_generation,
            terminated=False,
            truncated=False,
            reset_requested=False,
            teleport_requested=False,
        )

    def ledger_view(self) -> ContinuousRecoveryLedger:
        return self._state.ledger

    @property
    def ready_authority(self) -> bool:
        return self._effective_ready_authority(self._state)

    @staticmethod
    def _effective_ready_authority(state: _RuntimeState) -> bool:
        """Project no stale true while Motion or an unfinished suffix is live."""

        return bool(
            state.ready_live
            and state.reference_active
            and not state.motion_active
            and not (state.played is True and not bool(state.suffix_complete))
        )

    @property
    def current_committed_task_key(
        self,
    ) -> Optional[_mailbox.LandingOutcomeShotKey]:
        return self._state.current_committed_task_key

    def checkpoint_state(
        self,
        *,
        external_authority_sha256: str,
    ) -> dict[str, object]:
        external = _sha256(
            external_authority_sha256, label="external_authority_sha256"
        )
        state = self._state
        checkpoint = ContinuousRecoveryCheckpoint(
            integration_status=INTEGRATION_STATUS,
            profile_sha256=self.profile.canonical_sha256,
            external_authority_sha256=external,
            env_id=self.env_id,
            reset_generation=self.reset_generation,
            sequence_birth_owner=state.sequence_birth_owner,
            active_ready_reference_owner=state.active_ready_reference_owner,
            pending_committed_reference_owner=(
                state.pending_committed_reference_owner
            ),
            current_committed_task_key=state.current_committed_task_key,
            recovery_owner_task_key=state.recovery_owner_task_key,
            recovery_deadline_tick=state.recovery_deadline_tick,
            played=state.played,
            suffix_complete=state.suffix_complete,
            infrastructure_fault=state.infrastructure_fault,
            phase=state.phase,
            reference_active=state.reference_active,
            motion_active=state.motion_active,
            last_source_step=state.last_source_step,
            episode_tick=state.episode_tick,
            last_publish_tick=state.last_publish_tick,
            ready_streak=state.ready_streak,
            ready_live=state.ready_live,
            first_ready_tick=state.first_ready_tick,
            latest_fact=state.latest_fact,
            latest_viewed=state.latest_viewed,
            latest_paid=state.latest_paid,
            latest_payment_step=state.latest_payment_step,
            paid_payment_idempotency_sha256s=(
                state.paid_payment_idempotency_sha256s
            ),
            ledger=state.ledger,
        )
        return checkpoint.to_mapping()

    @classmethod
    def restore_from_checkpoint(
        cls,
        *,
        profile: ContinuousRecoveryProfile,
        checkpoint: object,
        external_authority_sha256: str,
        expected_checkpoint_sha256: str,
        fault_injector: _FaultInjector = None,
    ) -> "ContinuousRecoveryRuntime":
        """Restore only the exact independently retained checkpoint content.

        ``expected_checkpoint_sha256`` must come from the trusted checkpoint
        registry/caller state, never by reading the incoming mapping.  The
        payload's ``external_authority_sha256`` binds lineage but is not a
        substitute for this content identity.
        """

        if not isinstance(profile, ContinuousRecoveryProfile):
            raise ContinuousRecoveryError("profile must be ContinuousRecoveryProfile")
        external = _sha256(
            external_authority_sha256, label="external_authority_sha256"
        )
        expected_checkpoint = _sha256(
            expected_checkpoint_sha256,
            label="expected_checkpoint_sha256",
        )
        # The checkpoint first proves that its mapping is internally sealed.
        # Only then may the caller's independently retained content identity
        # authorize that exact sealed mapping.  The authority stored *inside*
        # the payload is a separate lineage binding and cannot stand in for
        # this content pin.
        restored = ContinuousRecoveryCheckpoint.from_mapping(checkpoint)
        if restored.canonical_sha256 != expected_checkpoint:
            raise ContinuousRecoveryConflictError(
                "checkpoint content differs from independently retained SHA"
            )
        if restored.profile_sha256 != profile.canonical_sha256:
            raise ContinuousRecoveryConflictError("checkpoint profile differs")
        if (
            restored.latest_fact is not None
            and restored.latest_fact.profile_sha256 != restored.profile_sha256
        ):
            raise ContinuousRecoveryConflictError(
                "checkpoint latest fact profile differs"
            )
        if restored.external_authority_sha256 != external:
            raise ContinuousRecoveryConflictError(
                "checkpoint external authority differs from pinned restore"
            )
        if restored.sequence_birth_owner is None and any(
            value is not None
            for value in (
                restored.active_ready_reference_owner,
                restored.pending_committed_reference_owner,
                restored.current_committed_task_key,
                restored.recovery_owner_task_key,
            )
        ):
            raise ContinuousRecoveryConflictError(
                "checkpoint owns runtime state without sequence birth"
            )
        for owner_name in (
            "active_ready_reference_owner",
            "pending_committed_reference_owner",
        ):
            reference_owner = getattr(restored, owner_name)
            if reference_owner is not None and (
                reference_owner.env_id != restored.env_id
                or reference_owner.reset_generation != restored.reset_generation
            ):
                raise ContinuousRecoveryConflictError(
                    f"checkpoint {owner_name} environment differs"
                )
            if (
                reference_owner is not None
                and reference_owner.task_key is not None
                and (
                    reference_owner.task_key.source_sha256
                    != profile.source_sha256
                    or reference_owner.task_key.config_sha256
                    != profile.config_sha256
                )
            ):
                raise ContinuousRecoveryConflictError(
                    f"checkpoint {owner_name} source/config differs"
                )
        if (
            restored.sequence_birth_owner is not None
            and restored.sequence_birth_owner.owner_kind
            != SEQUENCE_BIRTH_OWNER
        ):
            raise ContinuousRecoveryConflictError(
                "checkpoint sequence-birth owner kind differs"
            )
        if (
            restored.pending_committed_reference_owner is not None
            and restored.pending_committed_reference_owner.owner_kind
            != COMMITTED_TASK_OWNER
        ):
            raise ContinuousRecoveryConflictError(
                "checkpoint pending reference owner kind differs"
            )
        for key_name in (
            "current_committed_task_key",
            "recovery_owner_task_key",
        ):
            task_key = getattr(restored, key_name)
            if task_key is not None and (
                task_key.source_sha256 != profile.source_sha256
                or task_key.config_sha256 != profile.config_sha256
            ):
                raise ContinuousRecoveryConflictError(
                    f"checkpoint {key_name} source/config differs"
                )
        if restored.current_committed_task_key is not None:
            if (
                restored.current_committed_task_key.env_id != restored.env_id
                or restored.current_committed_task_key.reset_generation
                != restored.reset_generation
            ):
                raise ContinuousRecoveryConflictError(
                    "checkpoint current key environment differs"
                )
            if (
                restored.pending_committed_reference_owner is None
                or restored.pending_committed_reference_owner.task_key
                != restored.current_committed_task_key
            ):
                raise ContinuousRecoveryConflictError(
                    "checkpoint latest committed key lost its pending frame-0 reference"
                )
        if (restored.recovery_owner_task_key is None) != (
            restored.recovery_deadline_tick is None
        ):
            raise ContinuousRecoveryConflictError(
                "checkpoint recovery owner and deadline differ"
            )
        owner = cls(
            profile=profile,
            env_id=restored.env_id,
            reset_generation=restored.reset_generation,
            fault_injector=fault_injector,
        )
        staged = _RuntimeState(
            sequence_birth_owner=restored.sequence_birth_owner,
            active_ready_reference_owner=restored.active_ready_reference_owner,
            pending_committed_reference_owner=(
                restored.pending_committed_reference_owner
            ),
            current_committed_task_key=restored.current_committed_task_key,
            recovery_owner_task_key=restored.recovery_owner_task_key,
            recovery_deadline_tick=restored.recovery_deadline_tick,
            played=restored.played,
            suffix_complete=restored.suffix_complete,
            infrastructure_fault=restored.infrastructure_fault,
            phase=restored.phase,
            reference_active=restored.reference_active,
            motion_active=restored.motion_active,
            last_source_step=restored.last_source_step,
            episode_tick=restored.episode_tick,
            last_publish_tick=restored.last_publish_tick,
            ready_streak=restored.ready_streak,
            ready_live=restored.ready_live,
            first_ready_tick=restored.first_ready_tick,
            latest_fact=restored.latest_fact,
            latest_viewed=restored.latest_viewed,
            latest_paid=restored.latest_paid,
            latest_payment_step=restored.latest_payment_step,
            paid_payment_idempotency_sha256s=(
                restored.paid_payment_idempotency_sha256s
            ),
            ledger=restored.ledger,
        )
        owner._replace_state(staged, stage="before_restore_checkpoint_commit")
        return owner


__all__ = [
    "CHECKPOINT_KIND",
    "COMMITTED_TASK_OWNER",
    "COMPONENT_NAMES",
    "ContinuousRecoveryCheckpoint",
    "ContinuousRecoveryCommandProjection",
    "ContinuousRecoveryConflictError",
    "ContinuousRecoveryDoneTermProjection",
    "ContinuousRecoveryError",
    "ContinuousRecoveryFact",
    "ContinuousRecoveryInfrastructureError",
    "ContinuousRecoveryLedger",
    "ContinuousRecoveryMotionProjection",
    "ContinuousRecoveryProfile",
    "ContinuousRecoveryReferenceOwner",
    "ContinuousRecoveryRewardView",
    "ContinuousRecoveryRuntime",
    "INTEGRATION_STATUS",
    "MOTION_PHASES",
    "NO_INFRASTRUCTURE_FAULT",
    "PAYMENT_IDEMPOTENCY_KIND",
    "PLANT_ERROR_COMPONENT_NAMES",
    "PLAYED_SUFFIX_MISSING_AT_REWARD_START",
    "POLICY_RATE_HZ",
    "RECOVERY_END_AGE_TICK",
    "RECOVERY_SAMPLE_COUNT",
    "RECOVERY_START_AGE_TICK",
    "REFERENCE_KIND",
    "REQUIRED_COMPONENT_REDUCTIONS",
    "REWARD_CONSUMER",
    "RUNTIME_WIRING_CONNECTED",
    "SEQUENCE_BIRTH_OWNER",
    "STATION_ANCHOR_KIND",
    "SUPPORT_SIGNAL_KIND",
    "canonical_sha256",
    "coerce_landing_outcome_shot_key",
    "recovery_payment_idempotency_sha256",
    "score_recovery_errors",
]
