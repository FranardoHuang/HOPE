"""Versioned, backend-neutral pre-task wait schedule for ActionBall.

This module is deliberately not wired into an environment.  It defines the
portable control-plane contract needed by a future ``RESET_WAIT -> TASK_ACTIVE``
runtime:

* a deterministic integer wait in policy ticks, derived only from
  ``(schedule, env_id, reset_generation)``;
* explicit WAIT/TASK validity semantics; and
* a self-sealed checkpoint highwater that rejects generation reuse or gaps.

The counter-based sampler never reads Python, NumPy, Torch, or simulator RNG
state.  A and C therefore share the same schedule by sharing one canonical
schedule SHA and the same ``env_id/reset_generation`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping


WAIT_SCHEDULE_SCHEMA_VERSION = 1
WAIT_SCHEDULE_KIND = "action_ball_pre_task_wait_schedule"
WAIT_ASSIGNMENT_KIND = "action_ball_pre_task_wait_assignment"
WAIT_HIGHWATER_SCHEMA_VERSION = 1
WAIT_HIGHWATER_KIND = "action_ball_pre_task_wait_highwater"
COUNTER_ALGORITHM = "sha256_rejection_u64_v1"
WAIT_DISTRIBUTION = "uniform_integer_policy_ticks_inclusive"
MAX_INT64 = (1 << 63) - 1
_U64_CARDINALITY = 1 << 64


def _plain_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a plain integer in [{minimum}, {maximum}]")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be exact lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ActionBallTaskWaitSchedule:
    """Content-addressed uniform wait schedule in integer policy ticks.

    ``required_active_ticks`` reserves enough episode horizon after the latest
    possible task start.  It is part of the schedule identity so a launch
    cannot increase the wait range without also proving the active task still
    fits in the episode.
    """

    seed: int
    min_wait_ticks: int
    max_wait_ticks: int
    episode_horizon_ticks: int
    required_active_ticks: int

    def __post_init__(self) -> None:
        _plain_int(self.seed, name="seed", minimum=0, maximum=MAX_INT64)
        _plain_int(
            self.min_wait_ticks,
            name="min_wait_ticks",
            minimum=1,
            maximum=MAX_INT64,
        )
        _plain_int(
            self.max_wait_ticks,
            name="max_wait_ticks",
            minimum=self.min_wait_ticks,
            maximum=MAX_INT64,
        )
        _plain_int(
            self.episode_horizon_ticks,
            name="episode_horizon_ticks",
            minimum=2,
            maximum=MAX_INT64,
        )
        _plain_int(
            self.required_active_ticks,
            name="required_active_ticks",
            minimum=1,
            maximum=MAX_INT64,
        )
        if self.max_wait_ticks + self.required_active_ticks > self.episode_horizon_ticks:
            raise ValueError(
                "max_wait_ticks + required_active_ticks must fit within "
                "episode_horizon_ticks"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": WAIT_SCHEDULE_SCHEMA_VERSION,
            "kind": WAIT_SCHEDULE_KIND,
            "counter_algorithm": COUNTER_ALGORITHM,
            "distribution": WAIT_DISTRIBUTION,
            "unit": "policy_tick",
            "seed": self.seed,
            "min_wait_ticks": self.min_wait_ticks,
            "max_wait_ticks": self.max_wait_ticks,
            "episode_horizon_ticks": self.episode_horizon_ticks,
            "required_active_ticks": self.required_active_ticks,
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        return {**payload, "canonical_sha256": self.canonical_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "ActionBallTaskWaitSchedule":
        if not isinstance(value, Mapping):
            raise ValueError("wait schedule must be a mapping")
        expected = {
            "schema_version",
            "kind",
            "counter_algorithm",
            "distribution",
            "unit",
            "seed",
            "min_wait_ticks",
            "max_wait_ticks",
            "episode_horizon_ticks",
            "required_active_ticks",
            "canonical_sha256",
        }
        if set(value) != expected:
            raise ValueError("wait schedule has unexpected or missing fields")
        if (
            value["schema_version"] != WAIT_SCHEDULE_SCHEMA_VERSION
            or value["kind"] != WAIT_SCHEDULE_KIND
            or value["counter_algorithm"] != COUNTER_ALGORITHM
            or value["distribution"] != WAIT_DISTRIBUTION
            or value["unit"] != "policy_tick"
        ):
            raise ValueError("wait schedule fixed semantics differ")
        result = cls(
            seed=value["seed"],
            min_wait_ticks=value["min_wait_ticks"],
            max_wait_ticks=value["max_wait_ticks"],
            episode_horizon_ticks=value["episode_horizon_ticks"],
            required_active_ticks=value["required_active_ticks"],
        )
        declared = _exact_sha256(
            value["canonical_sha256"], name="schedule canonical_sha256"
        )
        if declared != result.canonical_sha256:
            raise ValueError("wait schedule canonical SHA mismatch")
        return result

    def assignment(self, *, env_id: int, reset_generation: int) -> "TaskWaitAssignment":
        env_id = _plain_int(env_id, name="env_id", minimum=0, maximum=MAX_INT64)
        reset_generation = _plain_int(
            reset_generation,
            name="reset_generation",
            minimum=1,
            maximum=MAX_INT64,
        )
        span = self.max_wait_ticks - self.min_wait_ticks + 1
        rejection_limit = _U64_CARDINALITY - (_U64_CARDINALITY % span)
        rejection_round = 0
        while True:
            counter_payload = {
                "algorithm": COUNTER_ALGORITHM,
                "schedule_canonical_sha256": self.canonical_sha256,
                "env_id": env_id,
                "reset_generation": reset_generation,
                "rejection_round": rejection_round,
            }
            word = int.from_bytes(
                hashlib.sha256(_canonical_json_bytes(counter_payload)).digest()[:8],
                byteorder="big",
                signed=False,
            )
            if word < rejection_limit:
                wait_ticks = self.min_wait_ticks + (word % span)
                return TaskWaitAssignment(
                    schedule_canonical_sha256=self.canonical_sha256,
                    env_id=env_id,
                    reset_generation=reset_generation,
                    wait_ticks=wait_ticks,
                    rejection_round=rejection_round,
                )
            rejection_round += 1


@dataclass(frozen=True)
class TaskWaitAssignment:
    """One deterministic schedule row for one environment reset generation."""

    schedule_canonical_sha256: str
    env_id: int
    reset_generation: int
    wait_ticks: int
    rejection_round: int

    def __post_init__(self) -> None:
        _exact_sha256(
            self.schedule_canonical_sha256,
            name="assignment schedule_canonical_sha256",
        )
        _plain_int(self.env_id, name="env_id", minimum=0, maximum=MAX_INT64)
        _plain_int(
            self.reset_generation,
            name="reset_generation",
            minimum=1,
            maximum=MAX_INT64,
        )
        _plain_int(
            self.wait_ticks,
            name="wait_ticks",
            minimum=1,
            maximum=MAX_INT64,
        )
        _plain_int(
            self.rejection_round,
            name="rejection_round",
            minimum=0,
            maximum=MAX_INT64,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": WAIT_SCHEDULE_SCHEMA_VERSION,
            "kind": WAIT_ASSIGNMENT_KIND,
            "schedule_canonical_sha256": self.schedule_canonical_sha256,
            "env_id": self.env_id,
            "reset_generation": self.reset_generation,
            "wait_ticks": self.wait_ticks,
            "rejection_round": self.rejection_round,
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        return {**payload, "canonical_sha256": self.canonical_sha256}


@dataclass(frozen=True)
class WaitTaskValidity:
    """Actor/critic/reward validity at one elapsed episode tick."""

    phase: str
    wait_active: bool
    task_active: bool
    task_fields_valid: bool
    ball_fields_valid: bool
    clocks_valid: bool
    task_started_this_tick: bool
    wait_remaining_ticks: int
    task_age_ticks: int

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "wait_active": self.wait_active,
            "task_active": self.task_active,
            "task_fields_valid": self.task_fields_valid,
            "ball_fields_valid": self.ball_fields_valid,
            "clocks_valid": self.clocks_valid,
            "task_started_this_tick": self.task_started_this_tick,
            "wait_remaining_ticks": self.wait_remaining_ticks,
            "task_age_ticks": self.task_age_ticks,
        }


def wait_task_validity(
    schedule: ActionBallTaskWaitSchedule,
    assignment: TaskWaitAssignment,
    *,
    elapsed_ticks: int,
) -> WaitTaskValidity:
    """Return fail-closed field validity for one WAIT/TASK timeline tick.

    At ``elapsed_ticks == wait_ticks`` the task becomes active atomically at
    task age zero.  Before that boundary, task, ball, and task-clock fields are
    all invalid; callers must mask their numeric payloads.
    """

    if not isinstance(schedule, ActionBallTaskWaitSchedule):
        raise TypeError("schedule must be ActionBallTaskWaitSchedule")
    if not isinstance(assignment, TaskWaitAssignment):
        raise TypeError("assignment must be TaskWaitAssignment")
    elapsed_ticks = _plain_int(
        elapsed_ticks,
        name="elapsed_ticks",
        minimum=0,
        maximum=schedule.episode_horizon_ticks,
    )
    expected = schedule.assignment(
        env_id=assignment.env_id,
        reset_generation=assignment.reset_generation,
    )
    if assignment != expected:
        raise ValueError("wait assignment differs from its canonical schedule row")
    task_active = elapsed_ticks >= assignment.wait_ticks
    task_age = max(elapsed_ticks - assignment.wait_ticks, 0)
    wait_remaining = max(assignment.wait_ticks - elapsed_ticks, 0)
    return WaitTaskValidity(
        phase="TASK" if task_active else "WAIT",
        wait_active=not task_active,
        task_active=task_active,
        task_fields_valid=task_active,
        ball_fields_valid=task_active,
        clocks_valid=task_active,
        task_started_this_tick=(elapsed_ticks == assignment.wait_ticks),
        wait_remaining_ticks=wait_remaining,
        task_age_ticks=task_age,
    )


class ActionBallTaskWaitHighwater:
    """Replay-protected, self-sealed reset-generation checkpoint state.

    Sampling is stateless; this object exists only to prove that each env's
    generations were consumed exactly once and without gaps.  Restoring its
    state cannot perturb assignment values because those are keyed directly by
    ``env_id/reset_generation``.
    """

    def __init__(self, schedule: ActionBallTaskWaitSchedule) -> None:
        if not isinstance(schedule, ActionBallTaskWaitSchedule):
            raise TypeError("schedule must be ActionBallTaskWaitSchedule")
        self.schedule = schedule
        self._highwater_by_env: dict[int, int] = {}

    def record(self, *, env_id: int, reset_generation: int) -> TaskWaitAssignment:
        env_id = _plain_int(env_id, name="env_id", minimum=0, maximum=MAX_INT64)
        reset_generation = _plain_int(
            reset_generation,
            name="reset_generation",
            minimum=1,
            maximum=MAX_INT64,
        )
        expected = self._highwater_by_env.get(env_id, 0) + 1
        if reset_generation != expected:
            raise ValueError(
                "wait reset generation must advance exactly once: "
                f"env_id={env_id} expected={expected} actual={reset_generation}"
            )
        assignment = self.schedule.assignment(
            env_id=env_id, reset_generation=reset_generation
        )
        self._highwater_by_env[env_id] = reset_generation
        return assignment

    def assert_recorded(self, assignment: TaskWaitAssignment) -> None:
        if not isinstance(assignment, TaskWaitAssignment):
            raise TypeError("assignment must be TaskWaitAssignment")
        expected = self.schedule.assignment(
            env_id=assignment.env_id,
            reset_generation=assignment.reset_generation,
        )
        if assignment != expected:
            raise ValueError("recorded wait assignment is not canonical")
        highwater = self._highwater_by_env.get(assignment.env_id, 0)
        if not 1 <= assignment.reset_generation <= highwater:
            raise ValueError("wait assignment generation is above checkpoint highwater")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": WAIT_HIGHWATER_SCHEMA_VERSION,
            "kind": WAIT_HIGHWATER_KIND,
            "schedule_canonical_sha256": self.schedule.canonical_sha256,
            "highwater_by_env": [
                [env_id, generation]
                for env_id, generation in sorted(self._highwater_by_env.items())
            ],
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    def state_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        return {**payload, "canonical_sha256": self.canonical_sha256}

    @classmethod
    def from_state_dict(
        cls,
        schedule: ActionBallTaskWaitSchedule,
        state: object,
    ) -> "ActionBallTaskWaitHighwater":
        if not isinstance(state, Mapping):
            raise ValueError("wait highwater state must be a mapping")
        expected_keys = {
            "schema_version",
            "kind",
            "schedule_canonical_sha256",
            "highwater_by_env",
            "canonical_sha256",
        }
        if set(state) != expected_keys:
            raise ValueError("wait highwater state has unexpected or missing fields")
        if (
            state["schema_version"] != WAIT_HIGHWATER_SCHEMA_VERSION
            or state["kind"] != WAIT_HIGHWATER_KIND
        ):
            raise ValueError("wait highwater fixed semantics differ")
        schedule_sha = _exact_sha256(
            state["schedule_canonical_sha256"],
            name="highwater schedule_canonical_sha256",
        )
        if schedule_sha != schedule.canonical_sha256:
            raise ValueError("wait highwater belongs to a different schedule")
        rows = state["highwater_by_env"]
        if not isinstance(rows, list):
            raise ValueError("wait highwater_by_env must be a list")
        parsed: dict[int, int] = {}
        previous_env = -1
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("wait highwater row must be [env_id, generation]")
            env_id = _plain_int(
                row[0], name="highwater env_id", minimum=0, maximum=MAX_INT64
            )
            generation = _plain_int(
                row[1],
                name="highwater reset_generation",
                minimum=1,
                maximum=MAX_INT64,
            )
            if env_id <= previous_env:
                raise ValueError("wait highwater rows must be strictly env-sorted")
            previous_env = env_id
            parsed[env_id] = generation
        result = cls(schedule)
        result._highwater_by_env = parsed
        declared = _exact_sha256(
            state["canonical_sha256"], name="highwater canonical_sha256"
        )
        if declared != result.canonical_sha256:
            raise ValueError("wait highwater canonical SHA mismatch")
        return result
