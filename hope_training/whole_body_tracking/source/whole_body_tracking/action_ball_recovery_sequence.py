"""Portable pre-integration contract for carry-state ActionBall recovery.

This module owns only an immutable schedule and a dependency-light state
ledger.  It does not import, mutate, or claim integration with Isaac, MuJoCo,
an RL environment, a robot, an action history, or a ball implementation.
Runtime adapters must later consume the emitted events without teleporting or
resetting physical/policy state, and must prove that wiring independently.

The contract deliberately separates three times that a one-shot environment
used to conflate:

* the scheduled strike deadline closes an attempt even when it misses;
* the admitted motion's cycle end closes the complete post-strike suffix;
* the next question reveals only on its pre-registered absolute reveal tick.

A question that is not ready on that exact reveal tick is never installed
late.  Its immutable deadline is still consumed and later rows continue on
their original absolute schedule.  Staged task keys and payload hashes remain
absent from :meth:`RecoverySequence.visible_state` until the reveal event.

A checkpoint self-hash detects accidental corruption but is not its own
authority.  Exact restore therefore requires an independently retained
``contract_authority_sha256`` covering the immutable schedule, references, and
clock configuration; changing a future question and re-sealing is rejected.

At 50 Hz, the frozen recovery window is age ticks 10 through 77 inclusive:
``0.20 <= age_s <= 1.55`` produces exactly 68 eligible samples.  Every admitted
cycle end must occur no later than age tick 10, and every successor reveal must
occur after age tick 77, so the window cannot be shortened by a wrap or reveal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping, Optional, Sequence


SCHEMA_VERSION = 1
STATE_KIND = "action_ball_recovery_sequence_state_v1"
CONTRACT_SCOPE = "dependency_light_runtime_preintegration_only"
POLICY_RATE_HZ = 50
RECOVERY_START_OFFSET_TICKS = 10
RECOVERY_END_OFFSET_TICKS = 77
RECOVERY_ELIGIBLE_TICK_COUNT = (
    RECOVERY_END_OFFSET_TICKS - RECOVERY_START_OFFSET_TICKS + 1
)
MAX_ACTION_UID = (1 << 53) - 1

TASK_KEY_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
)


class RecoverySequenceError(ValueError):
    """The portable schedule/state violates the recovery contract."""


class SequencePhase(str, Enum):
    """Public reference/lifecycle phase for one environment."""

    PREPARE_VISIBLE = "prepare_visible"
    SWING = "swing"
    FOLLOW_THROUGH = "follow_through"
    RECOVER_HIDDEN = "recover_hidden"
    READY_HOLD = "ready_hold"


class ReferenceMode(str, Enum):
    """Reference source selected by a backend adapter."""

    FRAME0_HOLD = "frame0_hold"
    MOTION_PLAYBACK = "motion_playback"
    MOTION_SUFFIX = "motion_suffix"
    RECOVERY_READY = "recovery_ready"


class RevealStatus(str, Enum):
    """One scheduled row's reveal disposition."""

    PENDING = "pending"
    REVEALED = "revealed"
    SKIPPED_NOT_READY = "skipped_not_ready"


def _plain_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise RecoverySequenceError(f"{name} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "unbounded" if maximum is None else str(maximum)
        raise RecoverySequenceError(
            f"{name} must be in [{minimum}, {upper}], got {value}"
        )
    return value


def _optional_int(value: object, *, name: str, minimum: int = 0) -> Optional[int]:
    if value is None:
        return None
    return _plain_int(value, name=name, minimum=minimum)


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise RecoverySequenceError(f"{name} must be an exact boolean")
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecoverySequenceError(f"{name} must be non-empty stripped text")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RecoverySequenceError(f"{name} must be a 64-character SHA-256")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise RecoverySequenceError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return repository-style canonical JSON and reject non-finite values."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RecoverySequenceError("state is not canonical-JSON encodable") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecoverySequenceError(f"checkpoint contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise RecoverySequenceError(
        f"checkpoint contains non-finite JSON constant {value!r}"
    )


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecoverySequenceError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise RecoverySequenceError(
            f"{name} keys differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


@dataclass(frozen=True)
class FullTaskKey:
    """Field-for-field portable owner key for one scheduled ActionBall task."""

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
            self, "env_id", _plain_int(self.env_id, name="task.env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="task.reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "swing_generation",
            _plain_int(
                self.swing_generation,
                name="task.swing_generation",
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="task.action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="task.action_slot"),
        )
        for field_name in ("birth_sha256", "sample_sha256", "task_sha256"):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), name=f"task.{field_name}"),
            )

    def to_mapping(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in TASK_KEY_FIELDS}

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: object) -> "FullTaskKey":
        mapping = _exact_keys(value, set(TASK_KEY_FIELDS), name="full task key")
        return cls(**{name: mapping[name] for name in TASK_KEY_FIELDS})


@dataclass(frozen=True)
class ScheduledShot:
    """One immutable absolute reveal/deadline/cycle-end row."""

    label: str
    task_key: FullTaskKey
    payload_sha256: str
    scheduled_reveal_tick: int
    scheduled_deadline_tick: int
    cycle_end_tick: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _nonempty_text(self.label, name="row.label"))
        if not isinstance(self.task_key, FullTaskKey):
            raise RecoverySequenceError("row.task_key must be FullTaskKey")
        object.__setattr__(
            self,
            "payload_sha256",
            _sha256(self.payload_sha256, name="row.payload_sha256"),
        )
        reveal = _plain_int(
            self.scheduled_reveal_tick,
            name="row.scheduled_reveal_tick",
        )
        deadline = _plain_int(
            self.scheduled_deadline_tick,
            name="row.scheduled_deadline_tick",
        )
        cycle_end = _plain_int(self.cycle_end_tick, name="row.cycle_end_tick")
        if deadline <= reveal:
            raise RecoverySequenceError("row deadline must be strictly after reveal")
        if cycle_end <= deadline:
            raise RecoverySequenceError("row cycle end must be strictly after deadline")
        if cycle_end > deadline + RECOVERY_START_OFFSET_TICKS:
            raise RecoverySequenceError(
                "row post-strike suffix overlaps the frozen recovery window"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "label": self.label,
            "task_key": self.task_key.to_mapping(),
            "payload_sha256": self.payload_sha256,
            "scheduled_reveal_tick": self.scheduled_reveal_tick,
            "scheduled_deadline_tick": self.scheduled_deadline_tick,
            "cycle_end_tick": self.cycle_end_tick,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ScheduledShot":
        keys = {
            "label",
            "task_key",
            "payload_sha256",
            "scheduled_reveal_tick",
            "scheduled_deadline_tick",
            "cycle_end_tick",
        }
        mapping = _exact_keys(value, keys, name="scheduled shot")
        return cls(
            label=mapping["label"],
            task_key=FullTaskKey.from_mapping(mapping["task_key"]),
            payload_sha256=mapping["payload_sha256"],
            scheduled_reveal_tick=mapping["scheduled_reveal_tick"],
            scheduled_deadline_tick=mapping["scheduled_deadline_tick"],
            cycle_end_tick=mapping["cycle_end_tick"],
        )


@dataclass(frozen=True)
class SequenceEvent:
    """Pure intent/evidence emitted by one exact policy tick."""

    tick: int
    phase_before: SequencePhase
    phase_after: SequencePhase
    reference_mode_before: ReferenceMode
    reference_mode_after: ReferenceMode
    playback_started: bool
    task_revealed: bool
    revealed_task_key: Optional[FullTaskKey]
    revealed_payload_sha256: Optional[str]
    actual_ready_tick: Optional[int]
    ready_at_scheduled_reveal: Optional[bool]
    reveal_skipped_not_ready: bool
    skipped_task_key: Optional[FullTaskKey]
    deadline_due: bool
    deadline_consumed: bool
    deadline_task_key: Optional[FullTaskKey]
    shot_closed: bool
    closed_task_key: Optional[FullTaskKey]
    cycle_end_reached: bool
    cycle_end_task_key: Optional[FullTaskKey]
    recovery_owner_task_key: Optional[FullTaskKey]
    recovery_age_ticks: Optional[int]
    recovery_eligible: bool
    terminal_requested: bool
    truncation_requested: bool
    physical_state_reset_requested: bool
    carry_state_reset_requested: bool
    pose_teleport_requested: bool

    def to_mapping(self) -> dict[str, object]:
        def task(value: Optional[FullTaskKey]):
            return None if value is None else value.to_mapping()

        return {
            "tick": self.tick,
            "phase_before": self.phase_before.value,
            "phase_after": self.phase_after.value,
            "reference_mode_before": self.reference_mode_before.value,
            "reference_mode_after": self.reference_mode_after.value,
            "playback_started": self.playback_started,
            "task_revealed": self.task_revealed,
            "revealed_task_key": task(self.revealed_task_key),
            "revealed_payload_sha256": self.revealed_payload_sha256,
            "actual_ready_tick": self.actual_ready_tick,
            "ready_at_scheduled_reveal": self.ready_at_scheduled_reveal,
            "reveal_skipped_not_ready": self.reveal_skipped_not_ready,
            "skipped_task_key": task(self.skipped_task_key),
            "deadline_due": self.deadline_due,
            "deadline_consumed": self.deadline_consumed,
            "deadline_task_key": task(self.deadline_task_key),
            "shot_closed": self.shot_closed,
            "closed_task_key": task(self.closed_task_key),
            "cycle_end_reached": self.cycle_end_reached,
            "cycle_end_task_key": task(self.cycle_end_task_key),
            "recovery_owner_task_key": task(self.recovery_owner_task_key),
            "recovery_age_ticks": self.recovery_age_ticks,
            "recovery_eligible": self.recovery_eligible,
            "terminal_requested": self.terminal_requested,
            "truncation_requested": self.truncation_requested,
            "physical_state_reset_requested": (
                self.physical_state_reset_requested
            ),
            "carry_state_reset_requested": self.carry_state_reset_requested,
            "pose_teleport_requested": self.pose_teleport_requested,
        }


class RecoverySequence:
    """One-environment portable recovery/schedule ledger.

    ``ready`` passed to :meth:`advance` is an adapter-computed, fail-closed
    conjunction.  This class intentionally knows no robot thresholds or
    simulator tensors.  A staged row is private coordinator state; only a
    successful exact-tick reveal places its key/hash in ``visible_state``.
    """

    def __init__(
        self,
        *,
        recovery_reference_sha256: str,
        frame0_reference_sha256: str,
        ready_hold_ticks: int,
        rows: Sequence[ScheduledShot] = (),
    ) -> None:
        self._recovery_reference_sha256 = _sha256(
            recovery_reference_sha256,
            name="recovery_reference_sha256",
        )
        self._frame0_reference_sha256 = _sha256(
            frame0_reference_sha256,
            name="frame0_reference_sha256",
        )
        self._ready_hold_ticks = _plain_int(
            ready_hold_ticks,
            name="ready_hold_ticks",
            minimum=1,
        )
        self._rows: list[ScheduledShot] = []
        self._reveal_statuses: list[RevealStatus] = []
        self._deadline_consumed: list[bool] = []

        self._current_tick = -1
        self._cursor = 0
        self._phase = SequencePhase.RECOVER_HIDDEN
        self._reference_mode = ReferenceMode.RECOVERY_READY
        self._public_row_index: Optional[int] = None
        self._followthrough_row_index: Optional[int] = None
        self._recovery_owner_row_index: Optional[int] = None
        self._recovery_owner_deadline_tick: Optional[int] = None
        self._ready_consecutive_ticks = 0
        self._ready_hold_met_tick: Optional[int] = None

        self._opportunities_consumed = 0
        self._tasks_revealed = 0
        self._not_ready_skips = 0
        self._shots_closed = 0
        self._cycles_completed = 0
        self._recovery_eligible_sample_count = 0

        for row in rows:
            self.stage_shot(row)
        self._assert_invariants()

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def phase(self) -> SequencePhase:
        return self._phase

    @property
    def reference_mode(self) -> ReferenceMode:
        return self._reference_mode

    @property
    def recovery_owner_task_key(self) -> Optional[FullTaskKey]:
        if self._recovery_owner_row_index is None:
            return None
        return self._rows[self._recovery_owner_row_index].task_key

    @property
    def recovery_eligible_sample_count(self) -> int:
        return self._recovery_eligible_sample_count

    @property
    def opportunities_consumed(self) -> int:
        return self._opportunities_consumed

    @property
    def rows(self) -> tuple[ScheduledShot, ...]:
        return tuple(self._rows)

    @property
    def reveal_statuses(self) -> tuple[RevealStatus, ...]:
        return tuple(self._reveal_statuses)

    @property
    def schedule_sha256(self) -> str:
        """Content identity of every immutable absolute schedule row."""

        return canonical_sha256([row.to_mapping() for row in self._rows])

    def _contract_authority_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": STATE_KIND,
            "contract_scope": CONTRACT_SCOPE,
            "policy_rate_hz": POLICY_RATE_HZ,
            "recovery_start_offset_ticks": RECOVERY_START_OFFSET_TICKS,
            "recovery_end_offset_ticks": RECOVERY_END_OFFSET_TICKS,
            "recovery_eligible_tick_count": RECOVERY_ELIGIBLE_TICK_COUNT,
            "ready_hold_ticks": self._ready_hold_ticks,
            "recovery_reference_sha256": self._recovery_reference_sha256,
            "frame0_reference_sha256": self._frame0_reference_sha256,
            "schedule": [row.to_mapping() for row in self._rows],
        }

    @property
    def contract_authority_sha256(self) -> str:
        """External authority required to restore a sealed checkpoint.

        The checkpoint's own canonical hash detects accidental mutation.  It
        is not an authority because an editor could change a future question
        and re-seal the whole checkpoint.  Restore therefore requires this
        independently retained contract identity.
        """

        return canonical_sha256(self._contract_authority_payload())

    def _validate_adjacent_rows(self, previous: ScheduledShot, row: ScheduledShot) -> None:
        previous_key = previous.task_key
        key = row.task_key
        same_lineage = (
            key.env_id == previous_key.env_id
            and key.reset_generation == previous_key.reset_generation
            and key.action_uid == previous_key.action_uid
            and key.action_slot == previous_key.action_slot
            and key.birth_sha256 == previous_key.birth_sha256
        )
        if not same_lineage:
            raise RecoverySequenceError("adjacent rows must keep one birth/action lineage")
        if key.swing_generation != previous_key.swing_generation + 1:
            raise RecoverySequenceError(
                "adjacent row swing_generation must advance by exactly one"
            )
        if key.sample_sha256 == previous_key.sample_sha256:
            raise RecoverySequenceError("adjacent rows must not reuse sample_sha256")
        if key.task_sha256 == previous_key.task_sha256:
            raise RecoverySequenceError("adjacent rows must not reuse task_sha256")
        if row.scheduled_deadline_tick <= previous.scheduled_deadline_tick:
            raise RecoverySequenceError("scheduled deadlines must strictly advance")
        if row.scheduled_reveal_tick <= previous.cycle_end_tick:
            raise RecoverySequenceError(
                "successor reveal must follow the complete prior motion suffix"
            )
        earliest_reveal = (
            previous.scheduled_deadline_tick + RECOVERY_END_OFFSET_TICKS + 1
        )
        if row.scheduled_reveal_tick < earliest_reveal:
            raise RecoverySequenceError(
                "successor reveal truncates the 68-tick recovery window"
            )

    def stage_shot(self, row: ScheduledShot) -> int:
        """Stage one off-live row without changing actor-visible state."""

        if not isinstance(row, ScheduledShot):
            raise RecoverySequenceError("staged row must be ScheduledShot")
        if row.scheduled_reveal_tick <= self._current_tick:
            raise RecoverySequenceError("cannot stage a row at or behind current tick")
        if self._rows:
            self._validate_adjacent_rows(self._rows[-1], row)
        self._rows.append(row)
        self._reveal_statuses.append(RevealStatus.PENDING)
        self._deadline_consumed.append(False)
        self._assert_invariants()
        return len(self._rows) - 1

    def _reference_sha256(self) -> str:
        if self._reference_mode is ReferenceMode.RECOVERY_READY:
            return self._recovery_reference_sha256
        return self._frame0_reference_sha256

    def _reference_velocity_semantics(self) -> str:
        if self._reference_mode in {
            ReferenceMode.RECOVERY_READY,
            ReferenceMode.FRAME0_HOLD,
        }:
            return "literal_zero"
        return "immutable_motion_payload"

    def visible_state(self) -> dict[str, object]:
        """Return the only task/reference view a policy adapter may expose.

        Future schedule rows, task keys, payload hashes, reveal ticks, and
        deadlines are deliberately absent.  The current closed task remains
        visible during its full motion suffix and disappears at cycle end.
        """

        row = (
            None
            if self._public_row_index is None
            else self._rows[self._public_row_index]
        )
        return {
            "contract_scope": CONTRACT_SCOPE,
            "current_tick": self._current_tick,
            "phase": self._phase.value,
            "reference_mode": self._reference_mode.value,
            "reference_sha256": self._reference_sha256(),
            "reference_velocity_semantics": self._reference_velocity_semantics(),
            "teacher_reference_valid": True,
            "task_visible": row is not None,
            "attempt_open": self._phase
            in {SequencePhase.PREPARE_VISIBLE, SequencePhase.SWING},
            "active_task_key": None if row is None else row.task_key.to_mapping(),
            "active_payload_sha256": None if row is None else row.payload_sha256,
        }

    def _ready_phase(self) -> bool:
        return self._phase in {
            SequencePhase.RECOVER_HIDDEN,
            SequencePhase.READY_HOLD,
        }

    def _update_ready(self, *, ready: bool, ready_facts_valid: bool) -> None:
        effective = ready and ready_facts_valid
        if not self._ready_phase():
            self._ready_consecutive_ticks = 0
            self._ready_hold_met_tick = None
            return
        if not effective:
            self._ready_consecutive_ticks = 0
            self._ready_hold_met_tick = None
            self._phase = SequencePhase.RECOVER_HIDDEN
            return
        self._ready_consecutive_ticks += 1
        if self._ready_consecutive_ticks >= self._ready_hold_ticks:
            if self._ready_hold_met_tick is None:
                self._ready_hold_met_tick = self._current_tick
            self._phase = SequencePhase.READY_HOLD
        else:
            self._phase = SequencePhase.RECOVER_HIDDEN

    def _preflight_advance(self, *, tick: int, playback_started: bool) -> None:
        """Reject invalid external transitions before mutating ledger state."""

        if playback_started and self._phase is not SequencePhase.PREPARE_VISIBLE:
            raise RecoverySequenceError(
                "playback may start only from prepare_visible"
            )
        if self._followthrough_row_index is not None:
            followthrough = self._rows[self._followthrough_row_index]
            if tick > followthrough.cycle_end_tick:
                raise RecoverySequenceError("cycle-end event was skipped")
        if self._cursor >= len(self._rows):
            return
        current_row = self._rows[self._cursor]
        status = self._reveal_statuses[self._cursor]
        if tick > current_row.scheduled_reveal_tick and status is RevealStatus.PENDING:
            raise RecoverySequenceError("scheduled reveal tick was skipped")
        if tick == current_row.scheduled_reveal_tick and status is not RevealStatus.PENDING:
            raise RecoverySequenceError("scheduled reveal fired more than once")
        if tick > current_row.scheduled_deadline_tick:
            raise RecoverySequenceError("scheduled deadline tick was skipped")
        if tick == current_row.scheduled_deadline_tick:
            if status is RevealStatus.PENDING:
                raise RecoverySequenceError(
                    "deadline reached before reveal disposition"
                )
            if status is RevealStatus.REVEALED:
                if self._public_row_index != self._cursor:
                    raise RecoverySequenceError(
                        "revealed deadline does not own public task"
                    )
                playback_will_start = (
                    playback_started
                    and self._phase is SequencePhase.PREPARE_VISIBLE
                )
                if (
                    self._phase is not SequencePhase.SWING
                    and not playback_will_start
                ):
                    raise RecoverySequenceError(
                        "revealed task reached deadline before playback"
                    )

    def advance(
        self,
        *,
        tick: int,
        ready: bool,
        ready_facts_valid: bool = True,
        playback_started: bool = False,
    ) -> SequenceEvent:
        """Advance exactly one policy tick and emit pure integration intents."""

        tick = _plain_int(tick, name="tick")
        ready = _exact_bool(ready, name="ready")
        ready_facts_valid = _exact_bool(
            ready_facts_valid, name="ready_facts_valid"
        )
        playback_started = _exact_bool(
            playback_started, name="playback_started"
        )
        if tick != self._current_tick + 1:
            raise RecoverySequenceError(
                f"ticks must advance exactly once: expected {self._current_tick + 1}, got {tick}"
            )
        self._preflight_advance(tick=tick, playback_started=playback_started)

        phase_before = self._phase
        reference_before = self._reference_mode
        self._current_tick = tick

        task_revealed = False
        revealed_task_key = None
        revealed_payload_sha256 = None
        actual_ready_tick = None
        ready_at_scheduled_reveal = None
        reveal_skipped_not_ready = False
        skipped_task_key = None
        deadline_due = False
        deadline_consumed = False
        deadline_task_key = None
        shot_closed = False
        closed_task_key = None
        cycle_end_reached = False
        cycle_end_task_key = None

        if playback_started:
            self._phase = SequencePhase.SWING
            self._reference_mode = ReferenceMode.MOTION_PLAYBACK

        if self._followthrough_row_index is not None:
            followthrough = self._rows[self._followthrough_row_index]
            if tick > followthrough.cycle_end_tick:
                raise RecoverySequenceError("cycle-end event was skipped")
            if tick == followthrough.cycle_end_tick:
                cycle_end_reached = True
                cycle_end_task_key = followthrough.task_key
                self._cycles_completed += 1
                self._public_row_index = None
                self._followthrough_row_index = None
                self._phase = SequencePhase.RECOVER_HIDDEN
                self._reference_mode = ReferenceMode.RECOVERY_READY

        self._update_ready(ready=ready, ready_facts_valid=ready_facts_valid)

        current_row = self._rows[self._cursor] if self._cursor < len(self._rows) else None
        if current_row is not None:
            status = self._reveal_statuses[self._cursor]
            if tick > current_row.scheduled_reveal_tick and status is RevealStatus.PENDING:
                raise RecoverySequenceError("scheduled reveal tick was skipped")
            if tick == current_row.scheduled_reveal_tick:
                if status is not RevealStatus.PENDING:
                    raise RecoverySequenceError("scheduled reveal fired more than once")
                ready_now = (
                    self._phase is SequencePhase.READY_HOLD
                    and self._ready_consecutive_ticks >= self._ready_hold_ticks
                )
                actual_ready_tick = self._ready_hold_met_tick
                ready_at_scheduled_reveal = ready_now
                if ready_now:
                    task_revealed = True
                    revealed_task_key = current_row.task_key
                    revealed_payload_sha256 = current_row.payload_sha256
                    self._reveal_statuses[self._cursor] = RevealStatus.REVEALED
                    self._tasks_revealed += 1
                    self._public_row_index = self._cursor
                    self._phase = SequencePhase.PREPARE_VISIBLE
                    self._reference_mode = ReferenceMode.FRAME0_HOLD
                    self._ready_consecutive_ticks = 0
                    self._ready_hold_met_tick = None
                    self._recovery_owner_row_index = None
                    self._recovery_owner_deadline_tick = None
                else:
                    reveal_skipped_not_ready = True
                    skipped_task_key = current_row.task_key
                    self._reveal_statuses[
                        self._cursor
                    ] = RevealStatus.SKIPPED_NOT_READY
                    self._not_ready_skips += 1

            if tick > current_row.scheduled_deadline_tick:
                raise RecoverySequenceError("scheduled deadline tick was skipped")
            if tick == current_row.scheduled_deadline_tick:
                deadline_due = True
                deadline_consumed = True
                deadline_task_key = current_row.task_key
                status = self._reveal_statuses[self._cursor]
                if status is RevealStatus.PENDING:
                    raise RecoverySequenceError(
                        "deadline reached before reveal disposition"
                    )
                if status is RevealStatus.REVEALED:
                    if self._public_row_index != self._cursor:
                        raise RecoverySequenceError(
                            "revealed deadline does not own public task"
                        )
                    if self._phase is not SequencePhase.SWING:
                        raise RecoverySequenceError(
                            "revealed task reached deadline before playback"
                        )
                    shot_closed = True
                    closed_task_key = current_row.task_key
                    self._shots_closed += 1
                    self._phase = SequencePhase.FOLLOW_THROUGH
                    self._reference_mode = ReferenceMode.MOTION_SUFFIX
                    self._followthrough_row_index = self._cursor
                    self._recovery_owner_row_index = self._cursor
                    self._recovery_owner_deadline_tick = (
                        current_row.scheduled_deadline_tick
                    )
                self._deadline_consumed[self._cursor] = True
                self._opportunities_consumed += 1
                self._cursor += 1

        recovery_age_ticks = None
        recovery_eligible = False
        if self._recovery_owner_row_index is not None:
            if self._recovery_owner_deadline_tick is None:
                raise RecoverySequenceError("recovery owner lacks deadline tick")
            recovery_age_ticks = tick - self._recovery_owner_deadline_tick
            recovery_eligible = (
                RECOVERY_START_OFFSET_TICKS
                <= recovery_age_ticks
                <= RECOVERY_END_OFFSET_TICKS
                and self._ready_phase()
            )
            if recovery_eligible:
                self._recovery_eligible_sample_count += 1

        self._assert_invariants()
        return SequenceEvent(
            tick=tick,
            phase_before=phase_before,
            phase_after=self._phase,
            reference_mode_before=reference_before,
            reference_mode_after=self._reference_mode,
            playback_started=playback_started,
            task_revealed=task_revealed,
            revealed_task_key=revealed_task_key,
            revealed_payload_sha256=revealed_payload_sha256,
            actual_ready_tick=actual_ready_tick,
            ready_at_scheduled_reveal=ready_at_scheduled_reveal,
            reveal_skipped_not_ready=reveal_skipped_not_ready,
            skipped_task_key=skipped_task_key,
            deadline_due=deadline_due,
            deadline_consumed=deadline_consumed,
            deadline_task_key=deadline_task_key,
            shot_closed=shot_closed,
            closed_task_key=closed_task_key,
            cycle_end_reached=cycle_end_reached,
            cycle_end_task_key=cycle_end_task_key,
            recovery_owner_task_key=self.recovery_owner_task_key,
            recovery_age_ticks=recovery_age_ticks,
            recovery_eligible=recovery_eligible,
            terminal_requested=False,
            truncation_requested=False,
            physical_state_reset_requested=False,
            carry_state_reset_requested=False,
            pose_teleport_requested=False,
        )

    def _assert_invariants(self) -> None:
        row_count = len(self._rows)
        if type(self._current_tick) is not int or self._current_tick < -1:
            raise RecoverySequenceError("current tick is invalid")
        if (
            len(self._reveal_statuses) != row_count
            or len(self._deadline_consumed) != row_count
        ):
            raise RecoverySequenceError("row/state vector lengths differ")
        if not 0 <= self._cursor <= row_count:
            raise RecoverySequenceError("schedule cursor is outside row range")
        if any(
            not isinstance(value, RevealStatus) for value in self._reveal_statuses
        ):
            raise RecoverySequenceError("reveal status vector contains invalid value")
        if any(type(value) is not bool for value in self._deadline_consumed):
            raise RecoverySequenceError("deadline-consumed vector contains invalid value")

        for name, index in (
            ("public", self._public_row_index),
            ("follow-through", self._followthrough_row_index),
            ("recovery owner", self._recovery_owner_row_index),
        ):
            if index is not None and (
                type(index) is not int or not 0 <= index < row_count
            ):
                raise RecoverySequenceError(f"{name} row index is invalid")

        for index, row in enumerate(self._rows):
            status = self._reveal_statuses[index]
            consumed = self._deadline_consumed[index]
            if index < self._cursor:
                if not consumed:
                    raise RecoverySequenceError(
                        "cursor passed an unconsumed deadline"
                    )
                if status is RevealStatus.PENDING:
                    raise RecoverySequenceError("cursor passed a pending reveal")
                if self._current_tick < row.scheduled_deadline_tick:
                    raise RecoverySequenceError(
                        "consumed row is ahead of current tick"
                    )
            elif index == self._cursor:
                if consumed:
                    raise RecoverySequenceError("current row consumed its deadline")
                if status is RevealStatus.PENDING:
                    if self._current_tick >= row.scheduled_reveal_tick:
                        raise RecoverySequenceError(
                            "pending row passed its scheduled reveal"
                        )
                elif not (
                    row.scheduled_reveal_tick
                    <= self._current_tick
                    < row.scheduled_deadline_tick
                ):
                    raise RecoverySequenceError(
                        "disposed current row lies outside reveal/deadline interval"
                    )
            else:
                if consumed:
                    raise RecoverySequenceError("future row has consumed deadline")
                if status is not RevealStatus.PENDING:
                    raise RecoverySequenceError("future row has reveal disposition")
                if self._current_tick >= row.scheduled_reveal_tick:
                    raise RecoverySequenceError(
                        "future row reveal is not in the future"
                    )

        active_phases = {
            SequencePhase.PREPARE_VISIBLE,
            SequencePhase.SWING,
        }
        ready_phases = {
            SequencePhase.RECOVER_HIDDEN,
            SequencePhase.READY_HOLD,
        }
        if self._phase in active_phases:
            if self._public_row_index != self._cursor:
                raise RecoverySequenceError("active phase public row/cursor differ")
            if self._public_row_index is None or self._public_row_index >= row_count:
                raise RecoverySequenceError("active phase lacks public row")
            if self._reveal_statuses[self._public_row_index] is not RevealStatus.REVEALED:
                raise RecoverySequenceError("active public row was not revealed")
            if self._followthrough_row_index is not None:
                raise RecoverySequenceError("active phase also owns follow-through")
            active_row = self._rows[self._public_row_index]
            if not (
                active_row.scheduled_reveal_tick
                <= self._current_tick
                < active_row.scheduled_deadline_tick
            ):
                raise RecoverySequenceError(
                    "active task lies outside reveal/deadline interval"
                )
        elif self._phase is SequencePhase.FOLLOW_THROUGH:
            if (
                self._public_row_index is None
                or self._public_row_index != self._followthrough_row_index
                or self._public_row_index != self._recovery_owner_row_index
            ):
                raise RecoverySequenceError("follow-through ownership differs")
            if not self._deadline_consumed[self._public_row_index]:
                raise RecoverySequenceError("follow-through deadline is unconsumed")
            suffix_row = self._rows[self._public_row_index]
            if not (
                suffix_row.scheduled_deadline_tick
                <= self._current_tick
                < suffix_row.cycle_end_tick
            ):
                raise RecoverySequenceError(
                    "follow-through lies outside deadline/cycle-end interval"
                )
        elif self._phase in ready_phases:
            if (
                self._public_row_index is not None
                or self._followthrough_row_index is not None
            ):
                raise RecoverySequenceError("hidden recovery exposes a public task")
        else:
            raise RecoverySequenceError("unknown sequence phase")

        if self._cursor < row_count:
            current_status = self._reveal_statuses[self._cursor]
            if (
                current_status is RevealStatus.REVEALED
                and self._phase not in active_phases
            ):
                raise RecoverySequenceError("revealed current row is not active")
            if (
                current_status is RevealStatus.SKIPPED_NOT_READY
                and self._phase not in ready_phases
            ):
                raise RecoverySequenceError("skipped current row is not hidden")

        expected_mode = {
            SequencePhase.PREPARE_VISIBLE: ReferenceMode.FRAME0_HOLD,
            SequencePhase.SWING: ReferenceMode.MOTION_PLAYBACK,
            SequencePhase.FOLLOW_THROUGH: ReferenceMode.MOTION_SUFFIX,
            SequencePhase.RECOVER_HIDDEN: ReferenceMode.RECOVERY_READY,
            SequencePhase.READY_HOLD: ReferenceMode.RECOVERY_READY,
        }[self._phase]
        if self._reference_mode is not expected_mode:
            raise RecoverySequenceError("phase/reference mode pair differs")

        if (
            type(self._ready_consecutive_ticks) is not int
            or self._ready_consecutive_ticks < 0
            or self._ready_consecutive_ticks > self._current_tick + 1
        ):
            raise RecoverySequenceError("ready streak is invalid")
        if self._phase is SequencePhase.READY_HOLD:
            if (
                self._ready_consecutive_ticks < self._ready_hold_ticks
                or self._ready_hold_met_tick is None
            ):
                raise RecoverySequenceError("ready_hold lacks a completed ready streak")
            expected_met_tick = (
                self._current_tick
                - self._ready_consecutive_ticks
                + self._ready_hold_ticks
            )
            if self._ready_hold_met_tick != expected_met_tick:
                raise RecoverySequenceError("ready_hold completion tick differs")
        elif self._phase is SequencePhase.RECOVER_HIDDEN:
            if self._ready_consecutive_ticks >= self._ready_hold_ticks:
                raise RecoverySequenceError(
                    "recover_hidden contains a completed ready streak"
                )
            if self._ready_hold_met_tick is not None:
                raise RecoverySequenceError("recover_hidden retains completion tick")
        elif self._ready_consecutive_ticks != 0 or self._ready_hold_met_tick is not None:
            raise RecoverySequenceError("non-recovery phase retains ready streak")

        if (self._recovery_owner_row_index is None) != (
            self._recovery_owner_deadline_tick is None
        ):
            raise RecoverySequenceError("recovery owner/deadline are half-initialized")
        expected_owner = None
        if self._phase not in active_phases:
            for index in range(self._cursor - 1, -1, -1):
                if self._reveal_statuses[index] is RevealStatus.REVEALED:
                    expected_owner = index
                    break
        if self._recovery_owner_row_index != expected_owner:
            raise RecoverySequenceError("recovery owner differs from latest closed shot")
        if self._recovery_owner_row_index is not None:
            owner = self._rows[self._recovery_owner_row_index]
            if owner.scheduled_deadline_tick != self._recovery_owner_deadline_tick:
                raise RecoverySequenceError("recovery owner deadline differs from schedule")
            if self._current_tick < owner.scheduled_deadline_tick:
                raise RecoverySequenceError("recovery owner starts before deadline")

        counters = (
            self._opportunities_consumed,
            self._tasks_revealed,
            self._not_ready_skips,
            self._shots_closed,
            self._cycles_completed,
            self._recovery_eligible_sample_count,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise RecoverySequenceError("sequence counters must be non-negative integers")
        if self._opportunities_consumed != self._cursor:
            raise RecoverySequenceError("opportunity counter differs from cursor")
        if self._tasks_revealed != sum(
            status is RevealStatus.REVEALED for status in self._reveal_statuses
        ):
            raise RecoverySequenceError("reveal counter differs from row statuses")
        if self._not_ready_skips != sum(
            status is RevealStatus.SKIPPED_NOT_READY
            for status in self._reveal_statuses
        ):
            raise RecoverySequenceError("not-ready counter differs from row statuses")
        closed_indices = [
            index
            for index in range(self._cursor)
            if self._reveal_statuses[index] is RevealStatus.REVEALED
        ]
        if self._shots_closed != len(closed_indices):
            raise RecoverySequenceError("closed-shot counter differs from schedule")
        expected_cycles_completed = sum(
            self._current_tick >= self._rows[index].cycle_end_tick
            for index in closed_indices
        )
        if self._cycles_completed != expected_cycles_completed:
            raise RecoverySequenceError("completed-cycle counter differs from schedule")
        expected_eligible_samples = 0
        for index in closed_indices:
            deadline = self._rows[index].scheduled_deadline_tick
            eligible_first = deadline + RECOVERY_START_OFFSET_TICKS
            eligible_last = min(
                self._current_tick,
                deadline + RECOVERY_END_OFFSET_TICKS,
            )
            if eligible_last >= eligible_first:
                expected_eligible_samples += eligible_last - eligible_first + 1
        if self._recovery_eligible_sample_count != expected_eligible_samples:
            raise RecoverySequenceError(
                "recovery eligible counter differs from absolute windows"
            )

    def _state_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": STATE_KIND,
            "contract_scope": CONTRACT_SCOPE,
            "config": {
                "policy_rate_hz": POLICY_RATE_HZ,
                "recovery_start_offset_ticks": RECOVERY_START_OFFSET_TICKS,
                "recovery_end_offset_ticks": RECOVERY_END_OFFSET_TICKS,
                "recovery_eligible_tick_count": RECOVERY_ELIGIBLE_TICK_COUNT,
                "ready_hold_ticks": self._ready_hold_ticks,
                "recovery_reference_sha256": self._recovery_reference_sha256,
                "frame0_reference_sha256": self._frame0_reference_sha256,
                "schedule_sha256": self.schedule_sha256,
                "contract_authority_sha256": self.contract_authority_sha256,
            },
            "schedule": [row.to_mapping() for row in self._rows],
            "state": {
                "current_tick": self._current_tick,
                "cursor": self._cursor,
                "phase": self._phase.value,
                "reference_mode": self._reference_mode.value,
                "reveal_statuses": [value.value for value in self._reveal_statuses],
                "deadline_consumed": list(self._deadline_consumed),
                "public_row_index": self._public_row_index,
                "followthrough_row_index": self._followthrough_row_index,
                "recovery_owner_row_index": self._recovery_owner_row_index,
                "recovery_owner_deadline_tick": self._recovery_owner_deadline_tick,
                "ready_consecutive_ticks": self._ready_consecutive_ticks,
                "ready_hold_met_tick": self._ready_hold_met_tick,
                "opportunities_consumed": self._opportunities_consumed,
                "tasks_revealed": self._tasks_revealed,
                "not_ready_skips": self._not_ready_skips,
                "shots_closed": self._shots_closed,
                "cycles_completed": self._cycles_completed,
                "recovery_eligible_sample_count": self._recovery_eligible_sample_count,
            },
        }

    def state_dict(self) -> dict[str, object]:
        """Return a sealed checkpoint containing pending and carry-state ledger."""

        self._assert_invariants()
        payload = self._state_payload()
        return {**payload, "canonical_sha256": canonical_sha256(payload)}

    def checkpoint_json(self) -> str:
        """Return canonical checkpoint JSON; this still makes no runtime claim."""

        return canonical_json_bytes(self.state_dict()).decode("ascii")

    @classmethod
    def from_state_dict(
        cls,
        value: object,
        *,
        expected_contract_authority_sha256: str,
    ) -> "RecoverySequence":
        expected_contract_authority_sha256 = _sha256(
            expected_contract_authority_sha256,
            name="expected_contract_authority_sha256",
        )
        top_keys = {
            "schema_version",
            "kind",
            "contract_scope",
            "config",
            "schedule",
            "state",
            "canonical_sha256",
        }
        mapping = _exact_keys(value, top_keys, name="recovery sequence checkpoint")
        declared = _sha256(
            mapping["canonical_sha256"], name="checkpoint.canonical_sha256"
        )
        payload = {key: mapping[key] for key in top_keys if key != "canonical_sha256"}
        if canonical_sha256(payload) != declared:
            raise RecoverySequenceError("checkpoint canonical SHA differs")
        if _plain_int(
            mapping["schema_version"],
            name="checkpoint.schema_version",
            minimum=1,
        ) != SCHEMA_VERSION:
            raise RecoverySequenceError("checkpoint schema_version differs")
        if mapping["kind"] != STATE_KIND:
            raise RecoverySequenceError("checkpoint kind differs")
        if mapping["contract_scope"] != CONTRACT_SCOPE:
            raise RecoverySequenceError("checkpoint contract_scope differs")

        config_keys = {
            "policy_rate_hz",
            "recovery_start_offset_ticks",
            "recovery_end_offset_ticks",
            "recovery_eligible_tick_count",
            "ready_hold_ticks",
            "recovery_reference_sha256",
            "frame0_reference_sha256",
            "schedule_sha256",
            "contract_authority_sha256",
        }
        config = _exact_keys(mapping["config"], config_keys, name="checkpoint config")
        frozen = {
            "policy_rate_hz": POLICY_RATE_HZ,
            "recovery_start_offset_ticks": RECOVERY_START_OFFSET_TICKS,
            "recovery_end_offset_ticks": RECOVERY_END_OFFSET_TICKS,
            "recovery_eligible_tick_count": RECOVERY_ELIGIBLE_TICK_COUNT,
        }
        for name, expected in frozen.items():
            if _plain_int(config[name], name=f"checkpoint.{name}") != expected:
                raise RecoverySequenceError(f"checkpoint frozen {name} differs")
        schedule = mapping["schedule"]
        if not isinstance(schedule, list):
            raise RecoverySequenceError("checkpoint schedule must be a list")
        machine = cls(
            recovery_reference_sha256=config["recovery_reference_sha256"],
            frame0_reference_sha256=config["frame0_reference_sha256"],
            ready_hold_ticks=config["ready_hold_ticks"],
            rows=tuple(ScheduledShot.from_mapping(row) for row in schedule),
        )
        declared_schedule_sha256 = _sha256(
            config["schedule_sha256"],
            name="checkpoint.schedule_sha256",
        )
        if machine.schedule_sha256 != declared_schedule_sha256:
            raise RecoverySequenceError("checkpoint schedule SHA differs")
        declared_contract_authority_sha256 = _sha256(
            config["contract_authority_sha256"],
            name="checkpoint.contract_authority_sha256",
        )
        if (
            machine.contract_authority_sha256
            != declared_contract_authority_sha256
        ):
            raise RecoverySequenceError(
                "checkpoint contract authority payload differs"
            )
        if (
            machine.contract_authority_sha256
            != expected_contract_authority_sha256
        ):
            raise RecoverySequenceError(
                "checkpoint differs from external contract authority"
            )

        state_keys = {
            "current_tick",
            "cursor",
            "phase",
            "reference_mode",
            "reveal_statuses",
            "deadline_consumed",
            "public_row_index",
            "followthrough_row_index",
            "recovery_owner_row_index",
            "recovery_owner_deadline_tick",
            "ready_consecutive_ticks",
            "ready_hold_met_tick",
            "opportunities_consumed",
            "tasks_revealed",
            "not_ready_skips",
            "shots_closed",
            "cycles_completed",
            "recovery_eligible_sample_count",
        }
        state = _exact_keys(mapping["state"], state_keys, name="checkpoint state")
        current_tick = state["current_tick"]
        if type(current_tick) is not int or current_tick < -1:
            raise RecoverySequenceError("checkpoint current_tick is invalid")
        machine._current_tick = current_tick
        machine._cursor = _plain_int(state["cursor"], name="state.cursor")
        try:
            machine._phase = SequencePhase(state["phase"])
            machine._reference_mode = ReferenceMode(state["reference_mode"])
        except (TypeError, ValueError) as exc:
            raise RecoverySequenceError("checkpoint phase/reference mode is invalid") from exc
        raw_statuses = state["reveal_statuses"]
        if not isinstance(raw_statuses, list):
            raise RecoverySequenceError("checkpoint reveal_statuses must be a list")
        try:
            machine._reveal_statuses = [RevealStatus(value) for value in raw_statuses]
        except (TypeError, ValueError) as exc:
            raise RecoverySequenceError("checkpoint reveal status is invalid") from exc
        raw_consumed = state["deadline_consumed"]
        if not isinstance(raw_consumed, list) or any(
            type(value) is not bool for value in raw_consumed
        ):
            raise RecoverySequenceError("checkpoint deadline_consumed is invalid")
        machine._deadline_consumed = list(raw_consumed)
        machine._public_row_index = _optional_int(
            state["public_row_index"], name="state.public_row_index"
        )
        machine._followthrough_row_index = _optional_int(
            state["followthrough_row_index"],
            name="state.followthrough_row_index",
        )
        machine._recovery_owner_row_index = _optional_int(
            state["recovery_owner_row_index"],
            name="state.recovery_owner_row_index",
        )
        machine._recovery_owner_deadline_tick = _optional_int(
            state["recovery_owner_deadline_tick"],
            name="state.recovery_owner_deadline_tick",
        )
        machine._ready_consecutive_ticks = _plain_int(
            state["ready_consecutive_ticks"],
            name="state.ready_consecutive_ticks",
        )
        machine._ready_hold_met_tick = _optional_int(
            state["ready_hold_met_tick"], name="state.ready_hold_met_tick"
        )
        for field_name in (
            "opportunities_consumed",
            "tasks_revealed",
            "not_ready_skips",
            "shots_closed",
            "cycles_completed",
            "recovery_eligible_sample_count",
        ):
            setattr(
                machine,
                f"_{field_name}",
                _plain_int(state[field_name], name=f"state.{field_name}"),
            )
        machine._assert_invariants()
        return machine

    @classmethod
    def from_checkpoint_json(
        cls,
        value: object,
        *,
        expected_contract_authority_sha256: str,
    ) -> "RecoverySequence":
        if not isinstance(value, str):
            raise RecoverySequenceError("checkpoint JSON must be text")
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=_reject_json_constant,
            )
        except RecoverySequenceError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RecoverySequenceError("checkpoint JSON is invalid") from exc
        return cls.from_state_dict(
            decoded,
            expected_contract_authority_sha256=(
                expected_contract_authority_sha256
            ),
        )


__all__ = [
    "CONTRACT_SCOPE",
    "FullTaskKey",
    "POLICY_RATE_HZ",
    "RECOVERY_ELIGIBLE_TICK_COUNT",
    "RECOVERY_END_OFFSET_TICKS",
    "RECOVERY_START_OFFSET_TICKS",
    "RecoverySequence",
    "RecoverySequenceError",
    "ReferenceMode",
    "RevealStatus",
    "SCHEMA_VERSION",
    "STATE_KIND",
    "ScheduledShot",
    "SequenceEvent",
    "SequencePhase",
    "TASK_KEY_FIELDS",
    "canonical_json_bytes",
    "canonical_sha256",
]
