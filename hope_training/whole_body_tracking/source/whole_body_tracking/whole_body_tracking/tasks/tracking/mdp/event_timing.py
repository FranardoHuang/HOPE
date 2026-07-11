"""Immutable post-strike event timing for continuous training (torch only).

This module deliberately has no Isaac imports.  It owns the timing ledger for the T1 training
treatment, while :class:`MotionCommand` owns reference playback and
:class:`RacketTargetCommand` owns the atomic question payload.  Keeping the ledger standalone
makes the safety-critical invariants dependency-light and deterministic:

* no row can reveal until an exact-strike origin is explicitly recorded;
* reveal and deadline ticks are absolute once the origin is accepted;
* a missed, unavailable, or kinematically infeasible row still consumes its immutable deadline;
* the next row is based on the previous *scheduled* deadline, never on a late install or hit;
* the exact JSON bytes, not a re-serialized approximation, are SHA-256 bound.

The scheduler never touches simulator state, actions, observation history, or noise state.  Its
only mutation is its own per-environment ledger.  The Isaac integration may use an emitted
``hold_steps`` value to align a native-speed clip strike to the immutable deadline, but may not
retime the clip or move the deadline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


EVENT_SCHEDULE_SCHEMA_VERSION = 1
EVENT_TIMING_MODE_DISABLED = "disabled"
EVENT_TIMING_MODE_POST_STRIKE_T1 = "post_strike_t1"
EVENT_TIMING_MODES = (
    EVENT_TIMING_MODE_DISABLED,
    EVENT_TIMING_MODE_POST_STRIKE_T1,
)


@dataclass(frozen=True)
class EventScheduleRow:
    """One immutable opportunity relative to the previous scheduled strike."""

    question_id: str
    clip_id: int
    bank_row: int
    reveal_ticks_after_prior_strike: int
    next_strike_ticks_after_prior_strike: int
    available: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class EventScheduleSequence:
    sequence_id: str
    rows: tuple[EventScheduleRow, ...]


@dataclass(frozen=True)
class EventSchedule:
    schema_version: int
    schedule_id: str
    policy_rate_hz: int
    sequences: tuple[EventScheduleSequence, ...]
    source_path: str
    source_sha256: str
    source_bytes: int

    @property
    def rows(self) -> tuple[EventScheduleRow, ...]:
        return tuple(row for sequence in self.sequences for row in sequence.rows)

    def hard_contract(self) -> dict[str, Any]:
        """Content-addressed facts safe to embed in a checkpoint hard contract."""

        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "policy_rate_hz": self.policy_rate_hz,
            "sha256": self.source_sha256,
            "bytes": self.source_bytes,
            "sequence_count": len(self.sequences),
            "sequence_lengths": [len(sequence.rows) for sequence in self.sequences],
        }


@dataclass(frozen=True)
class EventStep:
    """Transient events emitted atomically by one scheduler advance."""

    install_env_ids: torch.Tensor
    install_schedule_rows: torch.Tensor
    install_clip_ids: torch.Tensor
    install_bank_rows: torch.Tensor
    install_hold_steps: torch.Tensor
    unavailable_env_ids: torch.Tensor
    infeasible_env_ids: torch.Tensor
    deadline_env_ids: torch.Tensor


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"event schedule contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"event schedule contains non-finite JSON constant {value!r}")


def _object(value, *, name: str, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    return value


def _integer(value, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _nonempty_string(value, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    digest = value.strip()
    if (
        digest != value
        or len(digest) != 64
        or digest.lower() != digest
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 string")
    return digest


def load_event_schedule(path: str | Path, expected_sha256: str) -> EventSchedule:
    """Load and byte-bind a materialized T1 schedule, failing closed on ambiguity.

    The materialized schema is intentionally smaller than the preregistration/spec document.  It
    contains only executable immutable rows.  Unknown semantic fields and duplicate JSON keys are
    rejected so a producer cannot add a timing instruction that this consumer silently ignores.
    Optional ``metadata`` objects are provenance-only and never affect execution.
    """

    expected = _sha256(expected_sha256, name="event schedule expected_sha256")
    source = Path(path).expanduser().resolve()
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"event schedule is unreadable: {source}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(
            f"event schedule byte SHA mismatch: expected {expected}, got {actual} ({source})"
        )
    try:
        decoded = payload.decode("utf-8")
        root = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"event schedule is not strict UTF-8 JSON: {source}") from exc

    root = _object(
        root,
        name="event schedule",
        allowed={"schema_version", "schedule_id", "policy_rate_hz", "sequences", "metadata"},
        required={"schema_version", "schedule_id", "policy_rate_hz", "sequences"},
    )
    schema = _integer(root["schema_version"], name="schema_version", minimum=1)
    if schema != EVENT_SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported event schedule schema {schema}; expected {EVENT_SCHEDULE_SCHEMA_VERSION}"
        )
    schedule_id = _nonempty_string(root["schedule_id"], name="schedule_id")
    policy_rate = _integer(root["policy_rate_hz"], name="policy_rate_hz", minimum=1)
    raw_sequences = root["sequences"]
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError("event schedule sequences must be a non-empty array")

    sequences = []
    sequence_ids = set()
    for sequence_index, raw_sequence in enumerate(raw_sequences):
        sequence = _object(
            raw_sequence,
            name=f"sequences[{sequence_index}]",
            allowed={"sequence_id", "rows", "metadata"},
            required={"sequence_id", "rows"},
        )
        sequence_id = _nonempty_string(
            sequence["sequence_id"], name=f"sequences[{sequence_index}].sequence_id"
        )
        if sequence_id in sequence_ids:
            raise ValueError(f"duplicate event sequence_id {sequence_id!r}")
        sequence_ids.add(sequence_id)
        raw_rows = sequence["rows"]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"event sequence {sequence_id!r} rows must be a non-empty array")
        rows = []
        for row_index, raw_row in enumerate(raw_rows):
            label = f"sequences[{sequence_index}].rows[{row_index}]"
            row = _object(
                raw_row,
                name=label,
                allowed={
                    "question_id",
                    "clip_id",
                    "bank_row",
                    "reveal_ticks_after_prior_strike",
                    "next_strike_ticks_after_prior_strike",
                    "available",
                    "unavailable_reason",
                },
                required={
                    "question_id",
                    "clip_id",
                    "bank_row",
                    "reveal_ticks_after_prior_strike",
                    "next_strike_ticks_after_prior_strike",
                },
            )
            question = _nonempty_string(row["question_id"], name=f"{label}.question_id")
            if len(question) != 16 or question.lower() != question or any(
                ch not in "0123456789abcdef" for ch in question
            ):
                raise ValueError(f"{label}.question_id must be a lowercase blake2b64 hex id")
            clip_id = _integer(row["clip_id"], name=f"{label}.clip_id", minimum=0)
            bank_row = _integer(row["bank_row"], name=f"{label}.bank_row", minimum=0)
            reveal = _integer(
                row["reveal_ticks_after_prior_strike"],
                name=f"{label}.reveal_ticks_after_prior_strike",
                minimum=1,
            )
            deadline = _integer(
                row["next_strike_ticks_after_prior_strike"],
                name=f"{label}.next_strike_ticks_after_prior_strike",
                minimum=2,
            )
            if deadline <= reveal:
                raise ValueError(
                    f"{label} deadline must be strictly after reveal, got {reveal}/{deadline}"
                )
            available = row.get("available", True)
            if not isinstance(available, bool):
                raise ValueError(f"{label}.available must be boolean")
            reason = row.get("unavailable_reason")
            if available:
                if reason not in (None, ""):
                    raise ValueError(
                        f"{label}.unavailable_reason is only valid when available=false"
                    )
                reason = None
            else:
                reason = _nonempty_string(reason, name=f"{label}.unavailable_reason")
            rows.append(
                EventScheduleRow(
                    question_id=question,
                    clip_id=clip_id,
                    bank_row=bank_row,
                    reveal_ticks_after_prior_strike=reveal,
                    next_strike_ticks_after_prior_strike=deadline,
                    available=available,
                    unavailable_reason=reason,
                )
            )
        sequences.append(EventScheduleSequence(sequence_id=sequence_id, rows=tuple(rows)))

    return EventSchedule(
        schema_version=schema,
        schedule_id=schedule_id,
        policy_rate_hz=policy_rate,
        sequences=tuple(sequences),
        source_path=str(source),
        source_sha256=actual,
        source_bytes=len(payload),
    )


class EventTimingScheduler:
    """Vectorized fail-closed post-strike timing ledger.

    Environments are deterministically assigned to sequences by ``env_id % sequence_count``.
    Rows never repeat within an episode.  A true episode reset restarts that environment's same
    immutable sequence and returns it to the unarmed state.
    """

    def __init__(
        self,
        schedule: EventSchedule,
        num_envs: int,
        device: str | torch.device,
        sequence_indices: Sequence[int] | torch.Tensor | None = None,
    ):
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or int(num_envs) <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.schedule = schedule
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        sequence_count = len(schedule.sequences)
        max_rows = max(len(sequence.rows) for sequence in schedule.sequences)

        row_index = torch.full((sequence_count, max_rows), -1, dtype=torch.long)
        clip_id = torch.zeros((sequence_count, max_rows), dtype=torch.long)
        bank_row = torch.zeros((sequence_count, max_rows), dtype=torch.long)
        reveal = torch.zeros((sequence_count, max_rows), dtype=torch.long)
        deadline = torch.zeros((sequence_count, max_rows), dtype=torch.long)
        available = torch.zeros((sequence_count, max_rows), dtype=torch.bool)
        lengths = []
        flat_rows = []
        for sequence_index, sequence in enumerate(schedule.sequences):
            lengths.append(len(sequence.rows))
            for cursor, row in enumerate(sequence.rows):
                flat_index = len(flat_rows)
                flat_rows.append(row)
                row_index[sequence_index, cursor] = flat_index
                clip_id[sequence_index, cursor] = row.clip_id
                bank_row[sequence_index, cursor] = row.bank_row
                reveal[sequence_index, cursor] = row.reveal_ticks_after_prior_strike
                deadline[sequence_index, cursor] = row.next_strike_ticks_after_prior_strike
                available[sequence_index, cursor] = row.available
        self.flat_rows = tuple(flat_rows)
        self._row_index = row_index.to(self.device)
        self._clip_id = clip_id.to(self.device)
        self._bank_row = bank_row.to(self.device)
        self._reveal_offset = reveal.to(self.device)
        self._deadline_offset = deadline.to(self.device)
        self._available = available.to(self.device)
        self._sequence_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)

        if sequence_indices is None:
            assigned = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            assigned = assigned.remainder(sequence_count)
        else:
            raw = torch.as_tensor(sequence_indices, device=self.device)
            if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
                raise ValueError("sequence_indices must use an integer dtype")
            assigned = raw.to(dtype=torch.long).reshape(-1)
            if len(assigned) != self.num_envs:
                raise ValueError("sequence_indices must contain one entry per environment")
            if torch.any(assigned < 0) or torch.any(assigned >= sequence_count):
                raise ValueError("sequence_indices contains an out-of-range sequence")
        self.sequence_index = assigned

        shape = (self.num_envs,)
        self.current_tick = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.origin_tick = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.reveal_tick = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.deadline_tick = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.cursor = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.armed = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.exhausted = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.row_revealed = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.row_installed = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.row_unavailable = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.row_infeasible = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.row_late_reveal = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.deadline_due = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.event_just_installed = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.event_just_unavailable = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.event_just_infeasible = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.deadline_just_due = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.opportunities_consumed = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.last_consumed_deadline_tick = torch.full(
            shape, -1, dtype=torch.long, device=self.device
        )

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        raw = torch.as_tensor(env_ids, device=self.device)
        if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
            raise ValueError("event scheduler env_ids must use an integer dtype")
        ids = raw.to(dtype=torch.long).reshape(-1)
        if len(torch.unique(ids)) != len(ids) or torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError("event scheduler env_ids must be unique and in range")
        return ids

    def _current(self, table: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        if ids is None:
            sequence = self.sequence_index
            cursor = self.cursor
        else:
            sequence = self.sequence_index[ids]
            cursor = self.cursor[ids]
        safe_cursor = cursor.clamp(min=0, max=table.shape[1] - 1)
        return table[sequence, safe_cursor]

    @property
    def current_schedule_row(self) -> torch.Tensor:
        return self._current(self._row_index)

    @property
    def current_clip_id(self) -> torch.Tensor:
        return self._current(self._clip_id)

    @property
    def current_bank_row(self) -> torch.Tensor:
        return self._current(self._bank_row)

    @property
    def deadline_ticks_remaining(self) -> torch.Tensor:
        return self.deadline_tick - self.current_tick

    @property
    def exact_strike_allowed(self) -> torch.Tensor:
        """Mask stale targets out of unavailable/infeasible scheduled opportunities."""

        return (~self.armed) | self.row_installed

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = self._env_ids(env_ids)
        if len(ids) == 0:
            return
        for tensor, value in (
            (self.current_tick, 0),
            (self.origin_tick, 0),
            (self.reveal_tick, 0),
            (self.deadline_tick, 0),
            (self.cursor, 0),
            (self.opportunities_consumed, 0),
            (self.last_consumed_deadline_tick, -1),
        ):
            tensor[ids] = value
        for tensor in (
            self.armed,
            self.exhausted,
            self.row_revealed,
            self.row_installed,
            self.row_unavailable,
            self.row_infeasible,
            self.row_late_reveal,
            self.deadline_due,
            self.event_just_installed,
            self.event_just_unavailable,
            self.event_just_infeasible,
            self.deadline_just_due,
        ):
            tensor[ids] = False

    def _set_row_deadlines(self, ids: torch.Tensor, origins: torch.Tensor) -> None:
        self.origin_tick[ids] = origins
        self.reveal_tick[ids] = origins + self._current(self._reveal_offset, ids)
        self.deadline_tick[ids] = origins + self._current(self._deadline_offset, ids)

    def record_exact_strike(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Arm unarmed environments at an explicitly accepted exact-strike opportunity."""

        ids = self._env_ids(env_ids)
        if len(ids) == 0:
            return ids
        accepted = ids[~self.armed[ids] & ~self.exhausted[ids]]
        if len(accepted) == 0:
            return accepted
        self.cursor[accepted] = 0
        self.armed[accepted] = True
        self.row_revealed[accepted] = False
        self.row_installed[accepted] = False
        self.row_unavailable[accepted] = False
        self.row_infeasible[accepted] = False
        self.row_late_reveal[accepted] = False
        self._set_row_deadlines(accepted, self.current_tick[accepted])
        return accepted

    def advance(self, native_strike_ticks_by_clip: Sequence[int] | torch.Tensor) -> EventStep:
        """Advance one policy tick and emit reveals/deadlines without moving any deadline."""

        if bool(self.deadline_due.any()):
            pending = torch.where(self.deadline_due)[0].tolist()
            raise RuntimeError(
                "event deadlines must be finalized after racket metrics and before the next "
                f"scheduler advance; pending envs={pending[:16]}"
            )
        raw_native = torch.as_tensor(native_strike_ticks_by_clip, device=self.device)
        if raw_native.dtype == torch.bool or raw_native.is_floating_point() or raw_native.is_complex():
            raise ValueError("native_strike_ticks_by_clip must use an integer dtype")
        native = raw_native.to(dtype=torch.long).reshape(-1)
        if len(native) == 0 or torch.any(native <= 0):
            raise ValueError("native_strike_ticks_by_clip must be a non-empty positive vector")
        max_clip = max(row.clip_id for row in self.flat_rows)
        if max_clip >= len(native):
            raise ValueError(
                f"event schedule references clip {max_clip}, native timing has {len(native)} clips"
            )

        self.current_tick.add_(1)
        self.event_just_installed.zero_()
        self.event_just_unavailable.zero_()
        self.event_just_infeasible.zero_()
        self.deadline_just_due.zero_()

        active = self.armed & ~self.exhausted
        reveal_due = active & ~self.row_revealed & (self.current_tick >= self.reveal_tick)
        if bool(reveal_due.any()):
            ids = torch.where(reveal_due)[0]
            available = self._current(self._available, ids)
            clip = self._current(self._clip_id, ids)
            reveal_offset = self._current(self._reveal_offset, ids)
            deadline_offset = self._current(self._deadline_offset, ids)
            notice = deadline_offset - reveal_offset
            native_ticks = native[clip]
            late = self.current_tick[ids] != self.reveal_tick[ids]
            unavailable = ~available
            infeasible = available & (late | (native_ticks > notice))
            install = available & ~infeasible

            self.row_revealed[ids] = True
            self.row_unavailable[ids] = unavailable
            self.row_infeasible[ids] = infeasible
            self.row_late_reveal[ids] = late
            self.row_installed[ids] = install
            self.event_just_unavailable[ids] = unavailable
            self.event_just_infeasible[ids] = infeasible
            self.event_just_installed[ids] = install

        deadline = active & (self.current_tick >= self.deadline_tick)
        self.deadline_due[deadline] = True
        self.deadline_just_due[deadline] = True

        install_ids = torch.where(self.event_just_installed)[0]
        unavailable_ids = torch.where(self.event_just_unavailable)[0]
        infeasible_ids = torch.where(self.event_just_infeasible)[0]
        deadline_ids = torch.where(self.deadline_just_due)[0]
        if len(install_ids) > 0:
            reveal_offset = self._current(self._reveal_offset, install_ids)
            deadline_offset = self._current(self._deadline_offset, install_ids)
            clip = self._current(self._clip_id, install_ids)
            hold = deadline_offset - reveal_offset - native[clip]
            if torch.any(hold < 0):
                raise RuntimeError("internal event scheduler error: feasible install has negative hold")
        else:
            hold = torch.empty(0, dtype=torch.long, device=self.device)
        return EventStep(
            install_env_ids=install_ids,
            install_schedule_rows=self._current(self._row_index, install_ids),
            install_clip_ids=self._current(self._clip_id, install_ids),
            install_bank_rows=self._current(self._bank_row, install_ids),
            install_hold_steps=hold,
            unavailable_env_ids=unavailable_ids,
            infeasible_env_ids=infeasible_ids,
            deadline_env_ids=deadline_ids,
        )

    def finalize_deadlines(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Consume due opportunities regardless of hit/miss and advance from fixed deadlines."""

        ids = self._env_ids(env_ids)
        due = ids[self.deadline_due[ids]]
        if len(due) == 0:
            return due
        old_deadline = self.deadline_tick[due].clone()
        self.last_consumed_deadline_tick[due] = old_deadline
        self.opportunities_consumed[due] += 1
        self.deadline_due[due] = False
        self.row_installed[due] = False
        self.cursor[due] += 1
        lengths = self._sequence_lengths[self.sequence_index[due]]
        exhausted = self.cursor[due] >= lengths
        exhausted_ids = due[exhausted]
        next_ids = due[~exhausted]
        self.exhausted[exhausted_ids] = True
        if len(next_ids) > 0:
            self.row_revealed[next_ids] = False
            self.row_unavailable[next_ids] = False
            self.row_infeasible[next_ids] = False
            self.row_late_reveal[next_ids] = False
            # Critical invariant: the next row is based on the immutable scheduled deadline even
            # if finalize was called late or the previous opportunity was missed/infeasible.
            self._set_row_deadlines(next_ids, old_deadline[~exhausted])
        return due
