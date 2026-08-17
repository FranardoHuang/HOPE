"""Receipt-keyed delayed landing outcomes for continuous ActionBall shots.

This module is a dependency-light, pre-integration mailbox.  It owns no
simulator state.  A runtime adapter opens one row from the complete shot
receipt at contact time, then submits finite ball-centre transitions until the
first descending crossing of the landing plane or the declared flight
horizon.  The row keeps its original task, target, profile, and source step
even after another question becomes active.

Endpoint interpolation alone cannot see a within-step
``descent -> table contact -> bounce``.  Therefore every observation must also
carry the authoritative post-physics crossing collector's supplemental report:
the hidden within-step crossing XY, or explicit ``None`` when no crossing was
hidden between the submitted endpoints.  The collector may also report a
visible endpoint crossing; in that case the two estimates must agree.  This
argument has no default, so a runtime adapter cannot accidentally omit the
bounce-sensitive evidence.  The simulator adapter remains responsible for
producing it at physics-step resolution.

The lifecycle is deliberately explicit::

    OPEN -> SETTLED_UNPAID -> PAID -> CLOSED

Both a crossing and a horizon miss settle exactly once.  A horizon miss is a
real zero-valued score which must still be paid exactly once, so misses cannot
silently disappear from denominators.  Missing, discontinuous, or non-finite
evidence is rejected as a crossing and settles into its own explicit zero
fault cell.  A bounded mailbox may reuse only an explicitly ``CLOSED`` slot;
it never evicts ``OPEN`` or unpaid evidence, and its per-environment monotone
replay guard survives slot reuse and checkpoints.

The produced facts and scores are the canonical objects from
``action_ball_landing_placement``.  This file does not wire either Isaac or
MuJoCo and passing its tests is not runtime evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Optional, Sequence

from action_ball_landing_placement import (
    LandingPlacementFacts,
    LandingPlacementProfile,
    LandingPlacementScore,
    LandingPlacementTaskIdentity,
    canonical_sha256,
    score_landing_placement,
)


SCHEMA_VERSION = 1
SHOT_KEY_KIND = "action_ball_landing_outcome_shot_key_v1"
ENTRY_KIND = "action_ball_landing_outcome_entry_v1"
CHECKPOINT_KIND = "action_ball_landing_outcome_mailbox_checkpoint_v1"

OPEN = "OPEN"
SETTLED_UNPAID = "SETTLED_UNPAID"
PAID = "PAID"
CLOSED = "CLOSED"
STATES = (OPEN, SETTLED_UNPAID, PAID, CLOSED)

FIRST_DESCENDING_CROSSING = "first_descending_crossing"
FLIGHT_HORIZON_NO_CROSSING = "flight_horizon_no_crossing"
NONFINITE_OBSERVATION = "nonfinite_observation"
OBSERVATION_CONTRACT_FAULT = "observation_contract_fault"
SETTLEMENT_CAUSES = (
    FIRST_DESCENDING_CROSSING,
    FLIGHT_HORIZON_NO_CROSSING,
    NONFINITE_OBSERVATION,
    OBSERVATION_CONTRACT_FAULT,
)
PAYMENT_IDEMPOTENCY_KIND = "action_ball_landing_outcome_payment_v1"
REPLAY_GUARD_KIND = "action_ball_landing_outcome_replay_guard_v1"
SEGMENT_CROSSING = "consecutive_segment"
POST_PHYSICS_EVENT_CROSSING = "post_physics_event"
CROSSING_OBSERVATION_SOURCES = (
    SEGMENT_CROSSING,
    POST_PHYSICS_EVENT_CROSSING,
)

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
_KEY_FIELDS = (*_RUNTIME_KEY_FIELDS, *_SUCCESSOR_KEY_FIELDS)
_INT_KEY_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "shot_index",
)
_DIGEST_KEY_FIELDS = (
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
    "source_sha256",
    "config_sha256",
    "receipt_content_sha256",
)
_TEXT_KEY_FIELDS = ("run_id", "carry_chain_id")
_MAX_ACTION_UID = (1 << 53) - 1


class LandingOutcomeMailboxError(RuntimeError):
    """A mailbox lifecycle, identity, geometry, or capacity contract failed."""


def _plain_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise LandingOutcomeMailboxError(f"{label} must be an exact int")
    if value < minimum:
        raise LandingOutcomeMailboxError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise LandingOutcomeMailboxError(f"{label} must be <= {maximum}")
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
        raise LandingOutcomeMailboxError(f"{label} must be an exact bool")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LandingOutcomeMailboxError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LandingOutcomeMailboxError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise LandingOutcomeMailboxError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise LandingOutcomeMailboxError(f"{label} must be a non-empty string")
    return value


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_sha256(payload)
    return result


def _verified_payload(
    value: object,
    *,
    expected_keys: frozenset[str],
    kind: str,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LandingOutcomeMailboxError(f"{label} must be a mapping")
    expected = expected_keys | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise LandingOutcomeMailboxError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    declared = _sha256(
        value["canonical_sha256"], label=f"{label}.canonical_sha256"
    )
    payload = {key: value[key] for key in expected_keys}
    if payload["schema_version"] != SCHEMA_VERSION:
        raise LandingOutcomeMailboxError(f"{label} schema_version differs")
    if payload["kind"] != kind:
        raise LandingOutcomeMailboxError(f"{label} kind differs")
    try:
        actual_sha = canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise LandingOutcomeMailboxError(
            f"{label} is not finite canonical JSON"
        ) from exc
    if actual_sha != declared:
        raise LandingOutcomeMailboxError(f"{label} canonical SHA differs")
    return payload


@dataclass(frozen=True)
class LandingOutcomeShotKey:
    """Runtime receipt truth plus continuous-successor lineage identity."""

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
        for name in _INT_KEY_FIELDS:
            maximum = _MAX_ACTION_UID if name == "action_uid" else None
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    label=f"shot_key.{name}",
                    minimum=minimums[name],
                    maximum=maximum,
                ),
            )
        for name in _DIGEST_KEY_FIELDS:
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), label=f"shot_key.{name}"),
            )
        for name in _TEXT_KEY_FIELDS:
            object.__setattr__(
                self,
                name,
                _nonempty_text(
                    getattr(self, name), label=f"shot_key.{name}"
                ),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": SHOT_KEY_KIND,
            **{name: getattr(self, name) for name in _KEY_FIELDS},
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    def runtime_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _RUNTIME_KEY_FIELDS}

    def full_key_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _KEY_FIELDS}

    @classmethod
    def from_mapping(cls, value: object) -> "LandingOutcomeShotKey":
        payload = _verified_payload(
            value,
            expected_keys=frozenset(("schema_version", "kind", *_KEY_FIELDS)),
            kind=SHOT_KEY_KIND,
            label="landing-outcome shot key",
        )
        return cls(**{name: payload[name] for name in _KEY_FIELDS})

    @classmethod
    def coerce(cls, value: object) -> "LandingOutcomeShotKey":
        """Copy a sealed key, full mapping, or full-key-like object."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if frozenset(value) == frozenset(_KEY_FIELDS):
                return cls(**{name: value[name] for name in _KEY_FIELDS})
            return cls.from_mapping(value)
        if not all(hasattr(value, name) for name in _KEY_FIELDS):
            raise LandingOutcomeMailboxError(
                "shot key must expose runtime truth and successor lineage fields"
            )
        return cls(**{name: getattr(value, name) for name in _KEY_FIELDS})


@dataclass(frozen=True)
class LandingOutcomeView:
    """Immutable inspection view; it cannot mutate mailbox-owned evidence."""

    task_key: LandingOutcomeShotKey
    profile_sha256: str
    task_identity_sha256: str
    target_x_m: float
    target_y_m: float
    source_step: int
    source_ball_center_xyz_m: tuple[float, float, float]
    flight_horizon_step: int
    last_observation_step: int
    last_ball_center_xyz_m: Optional[tuple[float, float, float]]
    state: str
    crossing_observation_source: Optional[str]
    settlement_previous_ball_center_xyz_m: Optional[
        tuple[float, float, float]
    ]
    settlement_cause: Optional[str]
    settlement_step: Optional[int]
    payment_step: Optional[int]
    close_step: Optional[int]
    payment_idempotency_sha256: Optional[str]
    facts: Optional[LandingPlacementFacts]
    score: Optional[LandingPlacementScore]

    @property
    def reward(self) -> Optional[float]:
        return None if self.score is None else self.score.total


@dataclass(frozen=True)
class LandingOutcomePayment:
    """Exactly-once reward returned by ``pay`` with source/payment attribution."""

    task_key: LandingOutcomeShotKey
    profile_sha256: str
    task_identity_sha256: str
    target_x_m: float
    target_y_m: float
    source_step: int
    settlement_step: int
    payment_step: int
    idempotency_sha256: str
    score: LandingPlacementScore

    @property
    def reward(self) -> float:
        return self.score.total


@dataclass
class _Entry:
    sequence: int
    task_key: LandingOutcomeShotKey
    profile: LandingPlacementProfile
    task_identity: LandingPlacementTaskIdentity
    source_step: int
    source_ball_center_xyz_m: tuple[float, float, float]
    flight_horizon_step: int
    contact_valid: bool
    state: str = OPEN
    last_observation_step: int = 0
    last_ball_center_xyz_m: Optional[tuple[float, float, float]] = None
    ball_center_net_crossed: bool = False
    ball_center_net_clear: bool = False
    crossing_observation_source: Optional[str] = None
    settlement_previous_ball_center_xyz_m: Optional[
        tuple[float, float, float]
    ] = None
    settlement_cause: Optional[str] = None
    settlement_step: Optional[int] = None
    payment_step: Optional[int] = None
    close_step: Optional[int] = None
    payment_idempotency_sha256: Optional[str] = None
    facts: Optional[LandingPlacementFacts] = None
    score: Optional[LandingPlacementScore] = None


@dataclass(frozen=True)
class _ReplayGuard:
    env_id: int
    reset_generation: int
    swing_generation: int
    task_key_sha256: str

    @property
    def generation(self) -> tuple[int, int]:
        return (self.reset_generation, self.swing_generation)

    def to_mapping(self) -> dict[str, object]:
        return _sealed(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": REPLAY_GUARD_KIND,
                "env_id": self.env_id,
                "reset_generation": self.reset_generation,
                "swing_generation": self.swing_generation,
                "task_key_sha256": self.task_key_sha256,
            }
        )


_REPLAY_GUARD_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "env_id",
        "reset_generation",
        "swing_generation",
        "task_key_sha256",
    )
)


_ENTRY_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "sequence",
        "task_key",
        "profile",
        "task_identity",
        "source_step",
        "source_ball_center_xyz_m",
        "flight_horizon_step",
        "contact_valid",
        "state",
        "last_observation_step",
        "last_ball_center_xyz_m",
        "ball_center_net_crossed",
        "ball_center_net_clear",
        "crossing_observation_source",
        "settlement_previous_ball_center_xyz_m",
        "settlement_cause",
        "settlement_step",
        "payment_step",
        "close_step",
        "payment_idempotency_sha256",
        "facts",
        "score",
    )
)
_CHECKPOINT_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "capacity",
        "next_sequence",
        "entries",
        "replay_guards",
    )
)


def _finite_xyz(value: object, *, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LandingOutcomeMailboxError(
            f"{label} must be a finite length-three sequence"
        )
    if len(value) != 3:
        raise LandingOutcomeMailboxError(
            f"{label} must be a finite length-three sequence"
        )
    return tuple(
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _finite_xy(value: object, *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LandingOutcomeMailboxError(
            f"{label} must be a finite length-two sequence"
        )
    if len(value) != 2:
        raise LandingOutcomeMailboxError(
            f"{label} must be a finite length-two sequence"
        )
    return (
        _finite_number(value[0], label=f"{label}[0]"),
        _finite_number(value[1], label=f"{label}[1]"),
    )


class LandingOutcomeMailbox:
    """Bounded receipt-keyed delayed outcome state machine."""

    def __init__(self, *, capacity: int) -> None:
        self.capacity = _plain_int(capacity, label="capacity", minimum=1)
        self._entries: dict[str, _Entry] = {}
        self._replay_guards: dict[int, _ReplayGuard] = {}
        self._next_sequence = 0

    @staticmethod
    def _validate_owner(
        task_key: LandingOutcomeShotKey,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
    ) -> None:
        if not isinstance(profile, LandingPlacementProfile):
            raise LandingOutcomeMailboxError(
                "profile must be a LandingPlacementProfile"
            )
        if not isinstance(task_identity, LandingPlacementTaskIdentity):
            raise LandingOutcomeMailboxError(
                "task_identity must be a LandingPlacementTaskIdentity"
            )
        if task_identity.profile_sha256 != profile.canonical_sha256:
            raise LandingOutcomeMailboxError(
                "task identity profile SHA differs"
            )
        if task_identity.frame_id != profile.frame_id:
            raise LandingOutcomeMailboxError("task identity frame_id differs")
        if task_identity.frame_binding_sha256 != profile.frame_binding_sha256:
            raise LandingOutcomeMailboxError(
                "task identity frame binding SHA differs"
            )
        if task_identity.task_receipt_sha256 != task_key.task_sha256:
            raise LandingOutcomeMailboxError(
                "task identity receipt SHA differs from full shot key"
            )
        if not (
            profile.opponent_table_x_min_m
            <= task_identity.target_x_m
            <= profile.opponent_table_x_max_m
            and profile.table_y_min_m
            <= task_identity.target_y_m
            <= profile.table_y_max_m
        ):
            raise LandingOutcomeMailboxError(
                "task target lies outside the opponent table"
            )

    def _entry_id(self, task_key: object) -> tuple[LandingOutcomeShotKey, str]:
        key = LandingOutcomeShotKey.coerce(task_key)
        return key, key.canonical_sha256

    def _require_binding(
        self,
        *,
        task_key: object,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        source_step: int,
    ) -> _Entry:
        key, key_id = self._entry_id(task_key)
        entry = self._entries.get(key_id)
        if entry is None or entry.task_key != key:
            raise LandingOutcomeMailboxError("unknown full shot key")
        source = _plain_int(source_step, label="source_step")
        if source != entry.source_step:
            raise LandingOutcomeMailboxError("source_step differs from owner")
        if not isinstance(profile, LandingPlacementProfile):
            raise LandingOutcomeMailboxError(
                "profile must be a LandingPlacementProfile"
            )
        if profile != entry.profile:
            raise LandingOutcomeMailboxError("profile differs from owner")
        if not isinstance(task_identity, LandingPlacementTaskIdentity):
            raise LandingOutcomeMailboxError(
                "task_identity must be a LandingPlacementTaskIdentity"
            )
        if task_identity != entry.task_identity:
            raise LandingOutcomeMailboxError(
                "target/task identity differs from owner"
            )
        return entry

    @staticmethod
    def _view(entry: _Entry) -> LandingOutcomeView:
        return LandingOutcomeView(
            task_key=entry.task_key,
            profile_sha256=entry.profile.canonical_sha256,
            task_identity_sha256=entry.task_identity.canonical_sha256,
            target_x_m=entry.task_identity.target_x_m,
            target_y_m=entry.task_identity.target_y_m,
            source_step=entry.source_step,
            source_ball_center_xyz_m=entry.source_ball_center_xyz_m,
            flight_horizon_step=entry.flight_horizon_step,
            last_observation_step=entry.last_observation_step,
            last_ball_center_xyz_m=entry.last_ball_center_xyz_m,
            state=entry.state,
            crossing_observation_source=entry.crossing_observation_source,
            settlement_previous_ball_center_xyz_m=(
                entry.settlement_previous_ball_center_xyz_m
            ),
            settlement_cause=entry.settlement_cause,
            settlement_step=entry.settlement_step,
            payment_step=entry.payment_step,
            close_step=entry.close_step,
            payment_idempotency_sha256=(
                entry.payment_idempotency_sha256
            ),
            facts=entry.facts,
            score=entry.score,
        )

    def _make_room(self) -> None:
        if len(self._entries) < self.capacity:
            return
        closed = [
            (entry.sequence, key_id)
            for key_id, entry in self._entries.items()
            if entry.state == CLOSED
        ]
        if not closed:
            raise LandingOutcomeMailboxError(
                "mailbox capacity exhausted; no CLOSED slot is reusable"
            )
        _, oldest_closed = min(closed)
        del self._entries[oldest_closed]

    def open(
        self,
        *,
        task_key: object,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        source_step: int,
        source_ball_center_xyz_m: object,
        flight_horizon_step: int,
        contact_valid: bool,
    ) -> LandingOutcomeView:
        """Open one immutable owner row; an existing key is never overwritten."""

        key, key_id = self._entry_id(task_key)
        if key_id in self._entries:
            raise LandingOutcomeMailboxError(
                "full shot key already exists; overwrite is forbidden"
            )
        self._validate_owner(key, profile, task_identity)
        source = _plain_int(source_step, label="source_step")
        source_ball = _finite_xyz(
            source_ball_center_xyz_m,
            label="source_ball_center_xyz_m",
        )
        horizon = _plain_int(
            flight_horizon_step,
            label="flight_horizon_step",
            minimum=source + 1,
        )
        contact = _exact_bool(contact_valid, label="contact_valid")
        generation = (key.reset_generation, key.swing_generation)
        guard = self._replay_guards.get(key.env_id)
        if guard is not None and generation <= guard.generation:
            raise LandingOutcomeMailboxError(
                "shot generation is not newer than the durable replay guard"
            )
        if guard is None and len(self._replay_guards) >= self.capacity:
            raise LandingOutcomeMailboxError(
                "mailbox capacity exhausted by distinct env replay guards"
            )
        self._make_room()
        entry = _Entry(
            sequence=self._next_sequence,
            task_key=key,
            profile=profile,
            task_identity=task_identity,
            source_step=source,
            source_ball_center_xyz_m=source_ball,
            flight_horizon_step=horizon,
            contact_valid=contact,
            last_observation_step=source,
            last_ball_center_xyz_m=source_ball,
        )
        self._entries[key_id] = entry
        self._replay_guards[key.env_id] = _ReplayGuard(
            env_id=key.env_id,
            reset_generation=key.reset_generation,
            swing_generation=key.swing_generation,
            task_key_sha256=key_id,
        )
        self._next_sequence += 1
        return self._view(entry)

    @staticmethod
    def _settle(
        entry: _Entry,
        *,
        step: int,
        cause: str,
        crossing_xy_m: Optional[tuple[float, float]],
        crossing_observation_source: Optional[str],
        settlement_previous_ball_center_xyz_m: Optional[
            tuple[float, float, float]
        ],
        ball_center_net_crossed: bool,
        ball_center_net_clear: bool,
        last_observation_step: int,
        last_ball_center_xyz_m: Optional[tuple[float, float, float]],
    ) -> None:
        if entry.state != OPEN:
            raise LandingOutcomeMailboxError(
                "landing outcome has already settled"
            )
        if cause not in SETTLEMENT_CAUSES:
            raise LandingOutcomeMailboxError("unknown settlement cause")
        crossing_valid = crossing_xy_m is not None
        if cause == FIRST_DESCENDING_CROSSING and not crossing_valid:
            raise LandingOutcomeMailboxError(
                "crossing settlement requires crossing coordinates"
            )
        if cause == FIRST_DESCENDING_CROSSING:
            if crossing_observation_source not in CROSSING_OBSERVATION_SOURCES:
                raise LandingOutcomeMailboxError(
                    "crossing settlement requires an observation source"
                )
        elif crossing_observation_source is not None:
            raise LandingOutcomeMailboxError(
                "non-crossing settlement cannot name a crossing source"
            )
        if cause in (
            FIRST_DESCENDING_CROSSING,
            FLIGHT_HORIZON_NO_CROSSING,
        ):
            if settlement_previous_ball_center_xyz_m is None:
                raise LandingOutcomeMailboxError(
                    "flight settlement requires its previous endpoint"
                )
        elif settlement_previous_ball_center_xyz_m is not None:
            raise LandingOutcomeMailboxError(
                "fault settlement cannot claim an accepted final transition"
            )
        if cause == FLIGHT_HORIZON_NO_CROSSING and crossing_valid:
            raise LandingOutcomeMailboxError(
                "horizon settlement cannot carry crossing coordinates"
            )
        nonfinite = cause == NONFINITE_OBSERVATION
        contract_fault = cause == OBSERVATION_CONTRACT_FAULT
        if crossing_valid and (nonfinite or contract_fault):
            raise LandingOutcomeMailboxError(
                "invalid observation cannot carry crossing coordinates"
            )
        try:
            facts = LandingPlacementFacts(
                frame_id=entry.profile.frame_id,
                profile_sha256=entry.profile.canonical_sha256,
                task_identity_sha256=entry.task_identity.canonical_sha256,
                contact_valid=entry.contact_valid,
                first_plane_crossing_valid=crossing_valid,
                first_plane_crossing_nonfinite=nonfinite,
                first_plane_crossing_contract_fault=contract_fault,
                first_plane_crossing_x_m=(
                    None if crossing_xy_m is None else crossing_xy_m[0]
                ),
                first_plane_crossing_y_m=(
                    None if crossing_xy_m is None else crossing_xy_m[1]
                ),
                ball_center_net_crossed=ball_center_net_crossed,
                ball_center_net_clear=ball_center_net_clear,
            )
            score = score_landing_placement(
                entry.profile, entry.task_identity, facts
            )
        except Exception as exc:
            raise LandingOutcomeMailboxError(
                "landing settlement is not finite and scoreable"
            ) from exc
        if cause != FIRST_DESCENDING_CROSSING and score.total != 0.0:
            raise LandingOutcomeMailboxError(
                "non-crossing/fault closure must have zero score"
            )
        payment_idempotency_sha256 = canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": PAYMENT_IDEMPOTENCY_KIND,
                "task_key_sha256": entry.task_key.canonical_sha256,
                "profile_sha256": entry.profile.canonical_sha256,
                "task_identity_sha256": entry.task_identity.canonical_sha256,
                "source_step": entry.source_step,
                "settlement_step": step,
                "crossing_observation_source": crossing_observation_source,
                "facts_sha256": facts.canonical_sha256,
                "score_sha256": score.canonical_sha256,
            }
        )
        entry.facts = facts
        entry.score = score
        entry.crossing_observation_source = crossing_observation_source
        entry.settlement_previous_ball_center_xyz_m = (
            settlement_previous_ball_center_xyz_m
        )
        entry.settlement_cause = cause
        entry.settlement_step = step
        entry.last_observation_step = last_observation_step
        entry.last_ball_center_xyz_m = last_ball_center_xyz_m
        entry.ball_center_net_crossed = ball_center_net_crossed
        entry.ball_center_net_clear = ball_center_net_clear
        entry.payment_idempotency_sha256 = payment_idempotency_sha256
        entry.state = SETTLED_UNPAID

    def _settle_observation_fault(
        self,
        entry: _Entry,
        *,
        detected_step: int,
        cause: str,
    ) -> LandingOutcomeView:
        """Consume an untrustworthy opportunity as an explicit zero row."""

        settlement_step = min(detected_step, entry.flight_horizon_step)
        if settlement_step <= entry.source_step:
            raise LandingOutcomeMailboxError(
                "observation fault does not identify a post-source step"
            )
        self._settle(
            entry,
            step=settlement_step,
            cause=cause,
            crossing_xy_m=None,
            crossing_observation_source=None,
            settlement_previous_ball_center_xyz_m=None,
            ball_center_net_crossed=entry.ball_center_net_crossed,
            ball_center_net_clear=entry.ball_center_net_clear,
            last_observation_step=entry.last_observation_step,
            last_ball_center_xyz_m=entry.last_ball_center_xyz_m,
        )
        return self._view(entry)

    def observe_flight(
        self,
        *,
        task_key: object,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        source_step: int,
        step: int,
        previous_ball_center_xyz_m: object,
        current_ball_center_xyz_m: object,
        ball_center_net_crossed: bool,
        ball_center_net_clear: bool,
        post_physics_descending_crossing_xy_m: object,
    ) -> LandingOutcomeView:
        """Record one consecutive transition and settle exactly once.

        Every step from ``source_step + 1`` through the horizon is required,
        and each transition's previous endpoint must exactly equal the prior
        transition's current endpoint.  This is what makes "first crossing"
        auditable rather than a claim over a caller-selected segment.  The net
        booleans are cumulative facts from one first-crossing event: they are
        monotone, clearance cannot exist without crossing, and the clearance
        value cannot change after crossing.  The required post-physics value
        is an authority report, not an endpoint-derived convenience: ``None``
        means that collector found no crossing hidden from the submitted
        endpoints during this transition.  Missing/discontinuous/non-finite
        evidence consumes the opportunity as an explicit fault zero.
        """

        entry = self._require_binding(
            task_key=task_key,
            profile=profile,
            task_identity=task_identity,
            source_step=source_step,
        )
        if entry.state != OPEN:
            raise LandingOutcomeMailboxError(
                "landing outcome has already settled"
            )
        try:
            observation_step = _plain_int(step, label="step")
        except LandingOutcomeMailboxError:
            return self._settle_observation_fault(
                entry,
                detected_step=entry.last_observation_step + 1,
                cause=OBSERVATION_CONTRACT_FAULT,
            )
        if observation_step <= entry.last_observation_step:
            return self._settle_observation_fault(
                entry,
                detected_step=entry.last_observation_step + 1,
                cause=OBSERVATION_CONTRACT_FAULT,
            )
        expected_step = entry.last_observation_step + 1
        if observation_step != expected_step:
            return self._settle_observation_fault(
                entry,
                detected_step=observation_step,
                cause=OBSERVATION_CONTRACT_FAULT,
            )
        if observation_step > entry.flight_horizon_step:
            return self._settle_observation_fault(
                entry,
                detected_step=observation_step,
                cause=OBSERVATION_CONTRACT_FAULT,
            )
        try:
            previous = _finite_xyz(
                previous_ball_center_xyz_m,
                label="previous_ball_center_xyz_m",
            )
            current = _finite_xyz(
                current_ball_center_xyz_m,
                label="current_ball_center_xyz_m",
            )
        except LandingOutcomeMailboxError:
            return self._settle_observation_fault(
                entry,
                detected_step=observation_step,
                cause=NONFINITE_OBSERVATION,
            )
        if (
            entry.last_ball_center_xyz_m is not None
            and previous != entry.last_ball_center_xyz_m
        ):
            return self._settle_observation_fault(
                entry,
                detected_step=observation_step,
                cause=OBSERVATION_CONTRACT_FAULT,
            )
        try:
            crossed_net = _exact_bool(
                ball_center_net_crossed, label="ball_center_net_crossed"
            )
            cleared_net = _exact_bool(
                ball_center_net_clear, label="ball_center_net_clear"
            )
        except LandingOutcomeMailboxError:
            return self._settle_observation_fault(
                entry,
                detected_step=observation_step,
                cause=OBSERVATION_CONTRACT_FAULT,
            )
        net_contract_fault = (
            (cleared_net and not crossed_net)
            or (entry.ball_center_net_crossed and not crossed_net)
            or (
                entry.ball_center_net_crossed
                and cleared_net != entry.ball_center_net_clear
            )
        )
        if net_contract_fault:
            return self._settle_observation_fault(
                entry,
                detected_step=observation_step,
                cause=OBSERVATION_CONTRACT_FAULT,
            )

        accumulated_net_crossed = crossed_net
        accumulated_net_clear = cleared_net

        if post_physics_descending_crossing_xy_m is None:
            event_crossing_xy = None
        else:
            try:
                event_crossing_xy = _finite_xy(
                    post_physics_descending_crossing_xy_m,
                    label="post_physics_descending_crossing_xy_m",
                )
            except LandingOutcomeMailboxError:
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=NONFINITE_OBSERVATION,
                )

        plane_z = entry.profile.ball_center_landing_plane_z_m
        descending_crossing = (
            current[2] < previous[2]
            and previous[2] >= plane_z
            and current[2] <= plane_z
        )
        segment_crossing_xy = None
        if descending_crossing:
            denominator = current[2] - previous[2]
            if not math.isfinite(denominator) or denominator == 0.0:
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=NONFINITE_OBSERVATION,
                )
            fraction = (plane_z - previous[2]) / denominator
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=OBSERVATION_CONTRACT_FAULT,
                )
            try:
                segment_crossing_xy = (
                    _finite_number(
                        previous[0] + fraction * (current[0] - previous[0]),
                        label="interpolated crossing x",
                    ),
                    _finite_number(
                        previous[1] + fraction * (current[1] - previous[1]),
                        label="interpolated crossing y",
                    ),
                )
            except LandingOutcomeMailboxError:
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=NONFINITE_OBSERVATION,
                )
        if event_crossing_xy is not None and segment_crossing_xy is not None:
            if not math.isclose(
                event_crossing_xy[0],
                segment_crossing_xy[0],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ) or not math.isclose(
                event_crossing_xy[1],
                segment_crossing_xy[1],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=OBSERVATION_CONTRACT_FAULT,
                )
        if event_crossing_xy is not None:
            crossing_xy = event_crossing_xy
            crossing_source = POST_PHYSICS_EVENT_CROSSING
        else:
            crossing_xy = segment_crossing_xy
            crossing_source = (
                None if crossing_xy is None else SEGMENT_CROSSING
            )

        if crossing_xy is not None:
            try:
                self._settle(
                    entry,
                    step=observation_step,
                    cause=FIRST_DESCENDING_CROSSING,
                    crossing_xy_m=crossing_xy,
                    crossing_observation_source=crossing_source,
                    settlement_previous_ball_center_xyz_m=previous,
                    ball_center_net_crossed=accumulated_net_crossed,
                    ball_center_net_clear=accumulated_net_clear,
                    last_observation_step=observation_step,
                    last_ball_center_xyz_m=current,
                )
            except LandingOutcomeMailboxError:
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=NONFINITE_OBSERVATION,
                )
        elif observation_step == entry.flight_horizon_step:
            try:
                self._settle(
                    entry,
                    step=observation_step,
                    cause=FLIGHT_HORIZON_NO_CROSSING,
                    crossing_xy_m=None,
                    crossing_observation_source=None,
                    settlement_previous_ball_center_xyz_m=previous,
                    ball_center_net_crossed=accumulated_net_crossed,
                    ball_center_net_clear=accumulated_net_clear,
                    last_observation_step=observation_step,
                    last_ball_center_xyz_m=current,
                )
            except LandingOutcomeMailboxError:
                return self._settle_observation_fault(
                    entry,
                    detected_step=observation_step,
                    cause=OBSERVATION_CONTRACT_FAULT,
                )
        else:
            entry.last_observation_step = observation_step
            entry.last_ball_center_xyz_m = current
            entry.ball_center_net_crossed = accumulated_net_crossed
            entry.ball_center_net_clear = accumulated_net_clear
        return self._view(entry)

    def pay(
        self,
        *,
        task_key: object,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        source_step: int,
        payment_step: int,
    ) -> LandingOutcomePayment:
        """Pay once and preserve source/payment attribution.

        ``idempotency_sha256`` is deterministic before and after checkpoint
        restore.  The downstream reward accumulator must use it as its
        idempotency key across the mailbox/checkpoint transaction boundary.
        """

        entry = self._require_binding(
            task_key=task_key,
            profile=profile,
            task_identity=task_identity,
            source_step=source_step,
        )
        if entry.state != SETTLED_UNPAID:
            raise LandingOutcomeMailboxError(
                "landing outcome is not SETTLED_UNPAID; duplicate payment forbidden"
            )
        assert entry.settlement_step is not None
        assert entry.score is not None
        assert entry.payment_idempotency_sha256 is not None
        paid_at = _plain_int(
            payment_step,
            label="payment_step",
            minimum=entry.settlement_step,
        )
        entry.payment_step = paid_at
        entry.state = PAID
        return LandingOutcomePayment(
            task_key=entry.task_key,
            profile_sha256=entry.profile.canonical_sha256,
            task_identity_sha256=entry.task_identity.canonical_sha256,
            target_x_m=entry.task_identity.target_x_m,
            target_y_m=entry.task_identity.target_y_m,
            source_step=entry.source_step,
            settlement_step=entry.settlement_step,
            payment_step=paid_at,
            idempotency_sha256=entry.payment_idempotency_sha256,
            score=entry.score,
        )

    def close(
        self,
        *,
        task_key: object,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        source_step: int,
        close_step: int,
    ) -> LandingOutcomeView:
        """Mark paid evidence reusable; OPEN or unpaid evidence cannot close."""

        entry = self._require_binding(
            task_key=task_key,
            profile=profile,
            task_identity=task_identity,
            source_step=source_step,
        )
        if entry.state != PAID:
            raise LandingOutcomeMailboxError(
                "only PAID landing outcomes can close"
            )
        assert entry.payment_step is not None
        closed_at = _plain_int(
            close_step,
            label="close_step",
            minimum=entry.payment_step,
        )
        entry.close_step = closed_at
        entry.state = CLOSED
        return self._view(entry)

    def inspect(
        self,
        *,
        task_key: object,
        profile: LandingPlacementProfile,
        task_identity: LandingPlacementTaskIdentity,
        source_step: int,
    ) -> LandingOutcomeView:
        """Inspect one row only after revalidating its complete owner binding."""

        return self._view(
            self._require_binding(
                task_key=task_key,
                profile=profile,
                task_identity=task_identity,
                source_step=source_step,
            )
        )

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def state_counts(self) -> dict[str, int]:
        return {
            state: sum(entry.state == state for entry in self._entries.values())
            for state in STATES
        }

    @staticmethod
    def _entry_mapping(entry: _Entry) -> dict[str, object]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": ENTRY_KIND,
            "sequence": entry.sequence,
            "task_key": entry.task_key.to_mapping(),
            "profile": entry.profile.to_mapping(),
            "task_identity": entry.task_identity.to_mapping(),
            "source_step": entry.source_step,
            "source_ball_center_xyz_m": list(
                entry.source_ball_center_xyz_m
            ),
            "flight_horizon_step": entry.flight_horizon_step,
            "contact_valid": entry.contact_valid,
            "state": entry.state,
            "last_observation_step": entry.last_observation_step,
            "last_ball_center_xyz_m": (
                None
                if entry.last_ball_center_xyz_m is None
                else list(entry.last_ball_center_xyz_m)
            ),
            "ball_center_net_crossed": entry.ball_center_net_crossed,
            "ball_center_net_clear": entry.ball_center_net_clear,
            "crossing_observation_source": (
                entry.crossing_observation_source
            ),
            "settlement_previous_ball_center_xyz_m": (
                None
                if entry.settlement_previous_ball_center_xyz_m is None
                else list(entry.settlement_previous_ball_center_xyz_m)
            ),
            "settlement_cause": entry.settlement_cause,
            "settlement_step": entry.settlement_step,
            "payment_step": entry.payment_step,
            "close_step": entry.close_step,
            "payment_idempotency_sha256": (
                entry.payment_idempotency_sha256
            ),
            "facts": None if entry.facts is None else entry.facts.to_mapping(),
            "score": None if entry.score is None else entry.score.to_mapping(),
        }
        return _sealed(payload)

    def to_checkpoint(self) -> dict[str, object]:
        """Return a canonical JSON checkpoint including OPEN/unpaid/paid rows."""

        entries = sorted(self._entries.values(), key=lambda item: item.sequence)
        return _sealed(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": CHECKPOINT_KIND,
                "capacity": self.capacity,
                "next_sequence": self._next_sequence,
                "entries": [self._entry_mapping(entry) for entry in entries],
                "replay_guards": [
                    self._replay_guards[env_id].to_mapping()
                    for env_id in sorted(self._replay_guards)
                ],
            }
        )

    @classmethod
    def _entry_from_mapping(cls, value: object) -> _Entry:
        payload = _verified_payload(
            value,
            expected_keys=_ENTRY_KEYS,
            kind=ENTRY_KIND,
            label="landing-outcome entry",
        )
        try:
            key = LandingOutcomeShotKey.from_mapping(payload["task_key"])
            profile = LandingPlacementProfile.from_mapping(payload["profile"])
            identity = LandingPlacementTaskIdentity.from_mapping(
                payload["task_identity"]
            )
        except (TypeError, ValueError, LandingOutcomeMailboxError) as exc:
            raise LandingOutcomeMailboxError(
                "landing-outcome owner checkpoint is invalid"
            ) from exc
        cls._validate_owner(key, profile, identity)
        sequence = _plain_int(payload["sequence"], label="entry.sequence")
        source = _plain_int(payload["source_step"], label="entry.source_step")
        source_ball = _finite_xyz(
            payload["source_ball_center_xyz_m"],
            label="entry.source_ball_center_xyz_m",
        )
        horizon = _plain_int(
            payload["flight_horizon_step"],
            label="entry.flight_horizon_step",
            minimum=source + 1,
        )
        last = _plain_int(
            payload["last_observation_step"],
            label="entry.last_observation_step",
            minimum=source,
        )
        if last > horizon:
            raise LandingOutcomeMailboxError(
                "entry last observation exceeds flight horizon"
            )
        raw_last_ball = payload["last_ball_center_xyz_m"]
        if raw_last_ball is None:
            last_ball = None
        else:
            last_ball = _finite_xyz(
                raw_last_ball, label="entry.last_ball_center_xyz_m"
            )
        contact = _exact_bool(
            payload["contact_valid"], label="entry.contact_valid"
        )
        net_crossed = _exact_bool(
            payload["ball_center_net_crossed"],
            label="entry.ball_center_net_crossed",
        )
        net_clear = _exact_bool(
            payload["ball_center_net_clear"],
            label="entry.ball_center_net_clear",
        )
        if net_clear and not net_crossed:
            raise LandingOutcomeMailboxError(
                "entry net clearance exists without a net crossing"
            )
        state = payload["state"]
        if state not in STATES:
            raise LandingOutcomeMailboxError("entry state is unknown")
        cause = payload["settlement_cause"]
        if cause is not None and cause not in SETTLEMENT_CAUSES:
            raise LandingOutcomeMailboxError("entry settlement cause is unknown")
        crossing_source = payload["crossing_observation_source"]
        if (
            crossing_source is not None
            and crossing_source not in CROSSING_OBSERVATION_SOURCES
        ):
            raise LandingOutcomeMailboxError(
                "entry crossing observation source is unknown"
            )
        raw_settlement_previous = payload[
            "settlement_previous_ball_center_xyz_m"
        ]
        if raw_settlement_previous is None:
            settlement_previous = None
        else:
            settlement_previous = _finite_xyz(
                raw_settlement_previous,
                label="entry.settlement_previous_ball_center_xyz_m",
            )
        settlement = _optional_plain_int(
            payload["settlement_step"], label="entry.settlement_step"
        )
        payment = _optional_plain_int(
            payload["payment_step"], label="entry.payment_step"
        )
        close = _optional_plain_int(
            payload["close_step"], label="entry.close_step"
        )
        raw_payment_idempotency = payload["payment_idempotency_sha256"]
        payment_idempotency = (
            None
            if raw_payment_idempotency is None
            else _sha256(
                raw_payment_idempotency,
                label="entry.payment_idempotency_sha256",
            )
        )

        facts = None
        score = None
        if payload["facts"] is not None:
            try:
                facts = LandingPlacementFacts.from_mapping(payload["facts"])
            except (OverflowError, TypeError, ValueError) as exc:
                raise LandingOutcomeMailboxError(
                    "entry landing facts checkpoint is invalid"
                ) from exc
        if payload["score"] is not None:
            try:
                score = LandingPlacementScore.from_mapping(payload["score"])
            except (OverflowError, TypeError, ValueError) as exc:
                raise LandingOutcomeMailboxError(
                    "entry landing score checkpoint is invalid"
                ) from exc

        if state == OPEN:
            if any(
                item is not None
                for item in (
                    cause,
                    crossing_source,
                    settlement_previous,
                    settlement,
                    payment,
                    close,
                    payment_idempotency,
                    facts,
                    score,
                )
            ):
                raise LandingOutcomeMailboxError(
                    "OPEN entry carries settled or paid state"
                )
            if last >= horizon:
                raise LandingOutcomeMailboxError(
                    "OPEN entry reached its flight horizon without settling"
                )
            if last_ball is None:
                raise LandingOutcomeMailboxError(
                    "OPEN entry is missing its last finite endpoint"
                )
            if last == source and last_ball != source_ball:
                raise LandingOutcomeMailboxError(
                    "OPEN entry source endpoint differs"
                )
        else:
            if (
                cause is None
                or settlement is None
                or payment_idempotency is None
                or facts is None
                or score is None
            ):
                raise LandingOutcomeMailboxError(
                    "settled entry is missing cause, step, idempotency, facts, or score"
                )
            if not source < settlement <= horizon or last > settlement:
                raise LandingOutcomeMailboxError(
                    "settlement step is outside owner flight interval"
                )
            if last_ball is None:
                raise LandingOutcomeMailboxError(
                    "settled entry is missing its last finite endpoint"
                )
            if last == source and last_ball != source_ball:
                raise LandingOutcomeMailboxError(
                    "settled entry source endpoint differs"
                )
            if facts.contact_valid != contact:
                raise LandingOutcomeMailboxError(
                    "settled facts contact differs from owner"
                )
            if facts.ball_center_net_crossed != net_crossed:
                raise LandingOutcomeMailboxError(
                    "settled facts net-crossed differs from owner"
                )
            if facts.ball_center_net_clear != net_clear:
                raise LandingOutcomeMailboxError(
                    "settled facts net-clear differs from owner"
                )
            try:
                recomputed = score_landing_placement(profile, identity, facts)
            except (OverflowError, TypeError, ValueError) as exc:
                raise LandingOutcomeMailboxError(
                    "settled facts fail landing scorer identity"
                ) from exc
            if recomputed != score:
                raise LandingOutcomeMailboxError(
                    "settled score differs from canonical scorer"
                )
            expected_payment_idempotency = canonical_sha256(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": PAYMENT_IDEMPOTENCY_KIND,
                    "task_key_sha256": key.canonical_sha256,
                    "profile_sha256": profile.canonical_sha256,
                    "task_identity_sha256": identity.canonical_sha256,
                    "source_step": source,
                    "settlement_step": settlement,
                    "crossing_observation_source": crossing_source,
                    "facts_sha256": facts.canonical_sha256,
                    "score_sha256": score.canonical_sha256,
                }
            )
            if payment_idempotency != expected_payment_idempotency:
                raise LandingOutcomeMailboxError(
                    "settled payment idempotency SHA differs"
                )
            if cause == FIRST_DESCENDING_CROSSING:
                if (
                    last != settlement
                    or last_ball is None
                    or settlement_previous is None
                    or not facts.first_plane_crossing_valid
                    or facts.first_plane_crossing_nonfinite
                    or facts.first_plane_crossing_contract_fault
                    or crossing_source not in CROSSING_OBSERVATION_SOURCES
                ):
                    raise LandingOutcomeMailboxError(
                        "crossing settlement lacks an exact valid transition"
                    )
                if (
                    settlement == source + 1
                    and settlement_previous != source_ball
                ):
                    raise LandingOutcomeMailboxError(
                        "first crossing transition is not source anchored"
                    )
                plane_z = profile.ball_center_landing_plane_z_m
                visible_segment_crossing = (
                    last_ball[2] < settlement_previous[2]
                    and settlement_previous[2] >= plane_z
                    and last_ball[2] <= plane_z
                )
                if (
                    crossing_source == SEGMENT_CROSSING
                    and not visible_segment_crossing
                ):
                    raise LandingOutcomeMailboxError(
                        "segment crossing endpoints do not cross the plane"
                    )
                if visible_segment_crossing:
                    denominator = last_ball[2] - settlement_previous[2]
                    if not math.isfinite(denominator) or denominator == 0.0:
                        raise LandingOutcomeMailboxError(
                            "segment crossing denominator is not finite"
                        )
                    fraction = (plane_z - settlement_previous[2]) / denominator
                    expected_x = settlement_previous[0] + fraction * (
                        last_ball[0] - settlement_previous[0]
                    )
                    expected_y = settlement_previous[1] + fraction * (
                        last_ball[1] - settlement_previous[1]
                    )
                    if not math.isfinite(expected_x) or not math.isfinite(
                        expected_y
                    ):
                        raise LandingOutcomeMailboxError(
                            "segment crossing interpolation is not finite"
                        )
                    tolerance = (
                        1.0e-12
                        if crossing_source == SEGMENT_CROSSING
                        else 1.0e-9
                    )
                    if not math.isclose(
                        facts.first_plane_crossing_x_m,
                        expected_x,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ) or not math.isclose(
                        facts.first_plane_crossing_y_m,
                        expected_y,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        label = (
                            "segment"
                            if crossing_source == SEGMENT_CROSSING
                            else "post-physics"
                        )
                        raise LandingOutcomeMailboxError(
                            f"{label} crossing XY differs from final transition"
                        )
                if (
                    crossing_source == SEGMENT_CROSSING
                    and last_ball[2]
                    > profile.ball_center_landing_plane_z_m
                ):
                    raise LandingOutcomeMailboxError(
                        "segment crossing ends above the landing plane"
                    )
            elif cause == FLIGHT_HORIZON_NO_CROSSING:
                if (
                    settlement != horizon
                    or last != settlement
                    or last_ball is None
                    or settlement_previous is None
                    or facts.first_plane_crossing_valid
                    or facts.first_plane_crossing_nonfinite
                    or facts.first_plane_crossing_contract_fault
                    or facts.first_plane_crossing_x_m is not None
                    or facts.first_plane_crossing_y_m is not None
                    or score.total != 0.0
                    or crossing_source is not None
                ):
                    raise LandingOutcomeMailboxError(
                        "horizon no-crossing facts are not exact"
                    )
                if (
                    settlement == source + 1
                    and settlement_previous != source_ball
                ):
                    raise LandingOutcomeMailboxError(
                        "horizon transition is not source anchored"
                    )
                if (
                    last_ball[2] < settlement_previous[2]
                    and settlement_previous[2]
                    >= profile.ball_center_landing_plane_z_m
                    and last_ball[2]
                    <= profile.ball_center_landing_plane_z_m
                ):
                    raise LandingOutcomeMailboxError(
                        "horizon no-crossing final transition crosses the plane"
                    )
            elif cause == NONFINITE_OBSERVATION:
                if (
                    last >= settlement
                    or facts.first_plane_crossing_valid
                    or not facts.first_plane_crossing_nonfinite
                    or facts.first_plane_crossing_contract_fault
                    or facts.first_plane_crossing_x_m is not None
                    or facts.first_plane_crossing_y_m is not None
                    or score.total != 0.0
                    or crossing_source is not None
                    or settlement_previous is not None
                ):
                    raise LandingOutcomeMailboxError(
                        "non-finite observation facts are not exact"
                    )
            elif cause == OBSERVATION_CONTRACT_FAULT:
                if (
                    last >= settlement
                    or facts.first_plane_crossing_valid
                    or facts.first_plane_crossing_nonfinite
                    or not facts.first_plane_crossing_contract_fault
                    or facts.first_plane_crossing_x_m is not None
                    or facts.first_plane_crossing_y_m is not None
                    or score.total != 0.0
                    or crossing_source is not None
                    or settlement_previous is not None
                ):
                    raise LandingOutcomeMailboxError(
                        "observation-contract-fault facts are not exact"
                    )

        if state == SETTLED_UNPAID:
            if payment is not None or close is not None:
                raise LandingOutcomeMailboxError(
                    "SETTLED_UNPAID entry carries payment or close step"
                )
        elif state == PAID:
            if payment is None or settlement is None or payment < settlement:
                raise LandingOutcomeMailboxError(
                    "PAID entry has invalid payment step"
                )
            if close is not None:
                raise LandingOutcomeMailboxError("PAID entry carries close step")
        elif state == CLOSED:
            if (
                payment is None
                or settlement is None
                or close is None
                or payment < settlement
                or close < payment
            ):
                raise LandingOutcomeMailboxError(
                    "CLOSED entry has invalid payment/close steps"
                )

        return _Entry(
            sequence=sequence,
            task_key=key,
            profile=profile,
            task_identity=identity,
            source_step=source,
            source_ball_center_xyz_m=source_ball,
            flight_horizon_step=horizon,
            contact_valid=contact,
            state=state,
            last_observation_step=last,
            last_ball_center_xyz_m=last_ball,
            ball_center_net_crossed=net_crossed,
            ball_center_net_clear=net_clear,
            crossing_observation_source=crossing_source,
            settlement_previous_ball_center_xyz_m=settlement_previous,
            settlement_cause=cause,
            settlement_step=settlement,
            payment_step=payment,
            close_step=close,
            payment_idempotency_sha256=payment_idempotency,
            facts=facts,
            score=score,
        )

    @classmethod
    def from_checkpoint(
        cls,
        value: object,
        *,
        expected_checkpoint_sha256: object,
    ) -> "LandingOutcomeMailbox":
        """Restore and revalidate canonical internal mailbox invariants.

        Canonical SHA seals detect accidental drift, but they are not a
        signature.  The required expected SHA must come from the runtime's
        externally pinned checkpoint authority; copying the digest out of an
        untrusted candidate defeats that boundary.
        """

        expected_checkpoint_sha = _sha256(
            expected_checkpoint_sha256,
            label="expected_checkpoint_sha256",
        )
        payload = _verified_payload(
            value,
            expected_keys=_CHECKPOINT_KEYS,
            kind=CHECKPOINT_KIND,
            label="landing-outcome mailbox checkpoint",
        )
        assert isinstance(value, Mapping)
        declared_checkpoint_sha = _sha256(
            value["canonical_sha256"],
            label="checkpoint.canonical_sha256",
        )
        if declared_checkpoint_sha != expected_checkpoint_sha:
            raise LandingOutcomeMailboxError(
                "checkpoint differs from externally pinned authority SHA"
            )
        capacity = _plain_int(payload["capacity"], label="capacity", minimum=1)
        next_sequence = _plain_int(
            payload["next_sequence"], label="next_sequence"
        )
        values = payload["entries"]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise LandingOutcomeMailboxError(
                "checkpoint entries must be a sequence"
            )
        if len(values) > capacity:
            raise LandingOutcomeMailboxError(
                "checkpoint entry count exceeds mailbox capacity"
            )
        entries = [cls._entry_from_mapping(item) for item in values]
        sequences = [entry.sequence for entry in entries]
        if len(set(sequences)) != len(sequences):
            raise LandingOutcomeMailboxError(
                "checkpoint entry sequences are not unique"
            )
        if sequences != sorted(sequences):
            raise LandingOutcomeMailboxError(
                "checkpoint entries are not sequence ordered"
            )
        if sequences and next_sequence <= max(sequences):
            raise LandingOutcomeMailboxError(
                "next_sequence does not follow restored entries"
            )
        last_generation_by_env: dict[int, tuple[int, int]] = {}
        for entry in entries:
            env_id = entry.task_key.env_id
            generation = (
                entry.task_key.reset_generation,
                entry.task_key.swing_generation,
            )
            previous_generation = last_generation_by_env.get(env_id)
            if (
                previous_generation is not None
                and generation <= previous_generation
            ):
                raise LandingOutcomeMailboxError(
                    "checkpoint entry generations do not increase by sequence"
                )
            last_generation_by_env[env_id] = generation
        guard_values = payload["replay_guards"]
        if isinstance(guard_values, (str, bytes)) or not isinstance(
            guard_values, Sequence
        ):
            raise LandingOutcomeMailboxError(
                "checkpoint replay_guards must be a sequence"
            )
        if len(guard_values) > capacity:
            raise LandingOutcomeMailboxError(
                "checkpoint replay guard count exceeds mailbox capacity"
            )
        guards: list[_ReplayGuard] = []
        for value_item in guard_values:
            guard_payload = _verified_payload(
                value_item,
                expected_keys=_REPLAY_GUARD_KEYS,
                kind=REPLAY_GUARD_KIND,
                label="landing-outcome replay guard",
            )
            guards.append(
                _ReplayGuard(
                    env_id=_plain_int(
                        guard_payload["env_id"], label="guard.env_id"
                    ),
                    reset_generation=_plain_int(
                        guard_payload["reset_generation"],
                        label="guard.reset_generation",
                        minimum=1,
                    ),
                    swing_generation=_plain_int(
                        guard_payload["swing_generation"],
                        label="guard.swing_generation",
                    ),
                    task_key_sha256=_sha256(
                        guard_payload["task_key_sha256"],
                        label="guard.task_key_sha256",
                    ),
                )
            )
        guard_env_ids = [guard.env_id for guard in guards]
        if guard_env_ids != sorted(guard_env_ids) or len(
            set(guard_env_ids)
        ) != len(guard_env_ids):
            raise LandingOutcomeMailboxError(
                "checkpoint replay guards are not uniquely env ordered"
            )
        guard_by_env = {guard.env_id: guard for guard in guards}
        for entry in entries:
            guard = guard_by_env.get(entry.task_key.env_id)
            if guard is None:
                raise LandingOutcomeMailboxError(
                    "checkpoint entry has no durable replay guard"
                )
            generation = (
                entry.task_key.reset_generation,
                entry.task_key.swing_generation,
            )
            if generation > guard.generation:
                raise LandingOutcomeMailboxError(
                    "checkpoint entry is newer than its replay guard"
                )
            if (
                generation == guard.generation
                and entry.task_key.canonical_sha256
                != guard.task_key_sha256
            ):
                raise LandingOutcomeMailboxError(
                    "checkpoint latest generation differs from replay guard key"
                )
        result = cls(capacity=capacity)
        for entry in entries:
            key_id = entry.task_key.canonical_sha256
            if key_id in result._entries:
                raise LandingOutcomeMailboxError(
                    "checkpoint contains duplicate full shot keys"
                )
            result._entries[key_id] = entry
        result._replay_guards = guard_by_env
        result._next_sequence = next_sequence
        return result


__all__ = (
    "CHECKPOINT_KIND",
    "CLOSED",
    "CROSSING_OBSERVATION_SOURCES",
    "FIRST_DESCENDING_CROSSING",
    "FLIGHT_HORIZON_NO_CROSSING",
    "LandingOutcomeMailbox",
    "LandingOutcomeMailboxError",
    "LandingOutcomePayment",
    "LandingOutcomeShotKey",
    "LandingOutcomeView",
    "NONFINITE_OBSERVATION",
    "OBSERVATION_CONTRACT_FAULT",
    "OPEN",
    "PAID",
    "PAYMENT_IDEMPOTENCY_KIND",
    "POST_PHYSICS_EVENT_CROSSING",
    "REPLAY_GUARD_KIND",
    "SCHEMA_VERSION",
    "SEGMENT_CROSSING",
    "SETTLED_UNPAID",
    "SETTLEMENT_CAUSES",
    "SHOT_KEY_KIND",
    "STATES",
)
