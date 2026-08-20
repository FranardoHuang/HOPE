from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import pytest


_UTILS = (
    Path(__file__).parents[1]
    / "source/whole_body_tracking/whole_body_tracking/utils"
)


def _load_source(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _UTILS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_rsl = types.ModuleType("rsl_rl")
_rsl_runners = types.ModuleType("rsl_rl.runners")
_rsl_on_policy = types.ModuleType("rsl_rl.runners.on_policy_runner")
_rsl_on_policy.OnPolicyRunner = type("OnPolicyRunner", (), {})
_prior_modules = {
    name: sys.modules.get(name)
    for name in ("rsl_rl", "rsl_rl.runners", "rsl_rl.runners.on_policy_runner")
}
try:
    sys.modules["rsl_rl"] = _rsl
    sys.modules["rsl_rl.runners"] = _rsl_runners
    sys.modules["rsl_rl.runners.on_policy_runner"] = _rsl_on_policy
    adapter = _load_source(
        "action_ball_full_mdp_rsl3_adapter_direct",
        "action_ball_full_mdp_rsl3_adapter.py",
    )
finally:
    for _name, _prior in _prior_modules.items():
        if _prior is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _prior

durable_wal = _load_source(
    "action_ball_full_mdp_durable_wal_direct",
    "action_ball_full_mdp_durable_wal.py",
)


@dataclass(frozen=True)
class _Frontier:
    update_index: int
    completed_environment_steps: int
    operation_sequence: int = 1
    drain_sequence: int = 1
    start_commit: int = 0
    end_commit: int = 0
    shot_slot_capacity: int = 1


@dataclass(frozen=True)
class _Settlement:
    transactions: int = 0
    due_rows: int = 0
    selected_rows: int = 0
    accepted: int = 0
    censored: int = 0
    rejected: int = 0
    deferred: int = 0
    not_ready: int = 0


@dataclass(frozen=True)
class _Commits:
    motion_committed_rows: int = 0
    racket_committed_rows: int = 0
    r05_committed_rows: int = 0


@dataclass(frozen=True)
class _Lifecycle:
    playback_started_rows: int = 0
    closed_unplayed_rows: int = 0
    physical_launch_rows: int = 0
    outcome_settled_rows: int = 0
    payment_recorded_rows: int = 0
    retired_rows: int = 0
    terminal_shot_rows: int = 0


@dataclass(frozen=True)
class _Faults:
    attributed_fault_rows: int = 0


@dataclass(frozen=True)
class _Continuation:
    active_before: int = 0
    active_after: int = 0
    awaiting_playback_after: int = 0
    awaiting_outcome_after: int = 0
    awaiting_payment_after: int = 0


class _Milestone:
    def as_json(self, names):
        assert names == ("reward",)
        return {"reward_terms": [{"term": "reward"}]}


@dataclass(frozen=True)
class _Summary:
    frontier: _Frontier
    settlement: _Settlement = _Settlement()
    reveal_commit: _Commits = _Commits()
    lifecycle: _Lifecycle = _Lifecycle()
    owner_faults: _Faults = _Faults()
    continuation: _Continuation = _Continuation()
    action_opportunities: tuple = ()
    completed_shots: tuple = ()
    terminal_shots: tuple = ()
    terminal_resets: tuple = ()
    milestone: object = _Milestone()


class _Tensor:
    """Dependency-light tensor algebra used only by the direct adapter test."""

    def __init__(self, values, shape, dtype):
        self._values = tuple(values)
        self.shape = tuple(shape)
        self.dtype = f"torch.{dtype}"
        expected = math.prod(self.shape) if self.shape else 1
        assert len(self._values) == expected

    def _binary(self, other, operation, *, dtype="bool"):
        values = other._values if isinstance(other, _Tensor) else (other,) * len(self._values)
        assert len(values) == len(self._values)
        return _Tensor(
            (operation(left, right) for left, right in zip(self._values, values)),
            self.shape,
            dtype,
        )

    def eq(self, other):
        return self._binary(other, lambda left, right: left == right)

    def ge(self, other):
        return self._binary(other, lambda left, right: left >= right)

    def gt(self, other):
        return self._binary(other, lambda left, right: left > right)

    def le(self, other):
        return self._binary(other, lambda left, right: left <= right)

    def __or__(self, other):
        return self._binary(other, lambda left, right: bool(left) or bool(right))

    def all(self):
        return _Tensor((all(self._values),), (), "bool")

    def any(self, dim=None):
        if dim is None:
            return _Tensor((any(self._values),), (), "bool")
        assert dim == 1 and len(self.shape) == 2
        rows, columns = self.shape
        return _Tensor(
            (
                any(self._values[row * columns : (row + 1) * columns])
                for row in range(rows)
            ),
            (rows,),
            "bool",
        )

    def sum(self):
        return _Tensor((sum(self._values),), (), "int64")

    def amin(self):
        return _Tensor((min(self._values),), (), "float64")

    def isfinite(self):
        return _Tensor((math.isfinite(value) for value in self._values), self.shape, "bool")

    def item(self):
        assert self.shape == ()
        return self._values[0]


def _filled(value, shape, dtype):
    return _Tensor((value,) * math.prod(shape), shape, dtype)


def _compact_snapshot(
    *,
    num_envs,
    policy_steps,
    first_sequence,
    consume_sequence,
    capacity=2,
    actual_edge=False,
    nan_gap=False,
    incomplete=False,
):
    env_shape = (num_envs,)
    joint_shape = (num_envs, 1)
    complete_values = [policy_steps] * num_envs
    incomplete_values = [0] * num_envs
    if incomplete:
        complete_values[0] -= 1
        incomplete_values[0] = 1
    actual_values = [0] * num_envs
    lower_values = [1.0] * num_envs
    upper_values = [1.0] * num_envs
    actual_latch = [False] * num_envs
    if actual_edge:
        actual_values[0] = 1
        lower_values[0] = -0.125
        actual_latch[0] = True
    if nan_gap:
        lower_values[0] = float("nan")
    return {
        "schema_version": 1,
        "enabled": True,
        "diagnostic_compact_evidence": True,
        "diagnostic_first_policy_step_sequence": first_sequence,
        "diagnostic_last_policy_step_sequence": first_sequence + policy_steps - 1,
        "since_last_consume": {
            "consume_sequence": consume_sequence,
            "has_data": True,
            "identity_bound_policy_step_count": 0,
            "policy_step_count": _filled(policy_steps, env_shape, "int64"),
            "complete_policy_step_count": _Tensor(complete_values, env_shape, "int64"),
            "incomplete_policy_step_count": _Tensor(incomplete_values, env_shape, "int64"),
            "apply_readback_count": _filled(policy_steps * 4, env_shape, "int64"),
            "post_readback_count": _filled(policy_steps, env_shape, "int64"),
            "timestamp_invariant_pass_count": _filled(policy_steps, env_shape, "int64"),
            "hard_crossing_latch": _filled(False, env_shape, "bool"),
            "actual_hard_edge_latch": _Tensor(actual_latch, env_shape, "bool"),
            "qdes_joint_count": _filled(0, joint_shape, "int64"),
            "policy_crossing_joint_count": _filled(0, joint_shape, "int64"),
            "substep_hard_crossing_joint_count": _filled(0, joint_shape, "int64"),
            "actual_hard_edge_joint_count": _Tensor(actual_values, joint_shape, "int64"),
            "minimum_hard_lower_gap": _Tensor(lower_values, joint_shape, "float32"),
            "minimum_hard_upper_gap": _Tensor(upper_values, joint_shape, "float32"),
        },
        "terminal_archives": (),
        "identity_bound_policy_steps": (),
        "policy_step_summary_capacity": capacity,
        "policy_step_summary_used": 0,
        "policy_step_summary_payload_bytes": 0,
        "policy_step_summary_overflow_latch": False,
        "policy_step_summary_overflow_count": 0,
        "terminal_archive_capacity": capacity,
        "terminal_archive_used": 0,
        "terminal_archive_payload_bytes": 0,
        "terminal_archive_overflow_latch": False,
        "terminal_archive_overflow_count": 0,
    }


class _Epoch:
    def __init__(self, num_envs, shot_slot_capacity=1):
        self.num_envs = num_envs
        self.shot_slot_capacity = shot_slot_capacity


class ClampedJointPositionAction(ABC):
    _joint_safety_diagnostic_compact_evidence = True
    _pre_apply_guard_decimation = 4

    def __init__(self, events):
        self.events = events
        self.snapshot = None
        self.pending = None
        self.acknowledged = 0

    def prepare_joint_safety_ledger_consume(self):
        if self.pending is not None or self.snapshot is None:
            raise RuntimeError("joint-safety evidence is already frozen or absent")
        self.pending = ("safety", self.acknowledged)
        self.events.append("safety_prepare")
        return self.pending, self.snapshot

    def acknowledge_joint_safety_ledger(self, token):
        if token != self.pending:
            raise RuntimeError("joint-safety token differs")
        self.events.append("safety_ack")
        self.pending = None
        self.acknowledged += 1


ClampedJointPositionAction.__module__ = (
    "whole_body_tracking.tasks.tracking.mdp.hope_actions"
)
_ActionTerm = ClampedJointPositionAction


class _ForeignActionTerm(ClampedJointPositionAction):
    def prepare_joint_safety_ledger_consume(self):
        return super().prepare_joint_safety_ledger_consume()

    def acknowledge_joint_safety_ledger(self, token):
        return super().acknowledge_joint_safety_ledger(token)


class _ActionManager:
    def __init__(self, term):
        self.term = term

    def get_term(self, name):
        assert name == "joint_pos"
        return self.term


class _Owner:
    def __init__(self, env, lease, num_envs):
        self.full_mdp_runtime_env = env
        self.full_mdp_runtime_lease = lease
        self.epoch_owner = _Epoch(num_envs)
        self.events = []
        self.active = None
        self.summary = None
        self.completed_environment_steps = None
        self.poisoned = False

    def require_healthy(self):
        if self.poisoned:
            raise RuntimeError("owner is poisoned")
        self.events.append("healthy")

    def prepare_pre_optimizer_ppo_boundary(
        self, *, update_index, completed_environment_steps
    ):
        self.events.append(("prepare", update_index, completed_environment_steps))
        self.completed_environment_steps = completed_environment_steps
        self.active = object()
        return self.active

    def mark_optimizer_returned(self, boundary, *, update_index):
        assert boundary is self.active
        self.events.append(("mark", update_index))

    def prepare_post_update_summary(self, boundary, *, update_index):
        assert boundary is self.active
        self.events.append(("summary", update_index))
        self.summary = _Summary(
            _Frontier(
                update_index=update_index,
                completed_environment_steps=self.completed_environment_steps,
                operation_sequence=update_index + 1,
                drain_sequence=update_index + 1,
            )
        )
        return self.summary

    def acknowledge_post_update(self, boundary, summary, *, update_index):
        assert boundary is self.active and summary is self.summary
        self.events.append(("ack", update_index))
        self.active = None
        return summary

    def _record_durable_epoch_ack_span(self, summary, **kwargs):
        assert summary is self.summary and self.active is None
        self.events.append(("latch", kwargs))

    def poison_optimizer_boundary(self, boundary, *, update_index, reason):
        self.events.append(("poison", boundary, update_index, reason))
        self.poisoned = True


class _Env:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.unwrapped = self
        self.action_ball_full_mdp_runtime_lease = object()
        self.owner = _Owner(self, self.action_ball_full_mdp_runtime_lease, num_envs)
        self.action_term = _ActionTerm(self.owner.events)
        self.action_manager = _ActionManager(self.action_term)
        self.action_term.snapshot = _compact_snapshot(
            num_envs=num_envs,
            policy_steps=24,
            first_sequence=0,
            consume_sequence=0,
        )

    def action_ball_full_mdp_ppo_drain_owner(self, lease):
        assert lease is self.action_ball_full_mdp_runtime_lease
        return self.owner


@pytest.fixture
def _fake_imports(monkeypatch):
    runtime = types.SimpleNamespace(
        ActionBallFullMdpLeanRuntimeOwner=_Owner,
        ActionEpochPpoBoundarySummary=_Summary,
        EpochDrainFrontier=_Frontier,
        drain_v2=types.SimpleNamespace(
            SHOT_LIFECYCLE_FLAGS=(
                "reveal_committed",
                "playback_started",
                "motion_closed",
                "physical_launched",
                "outcome_settled",
                "payment_recorded",
            )
        ),
    )

    def import_module(name):
        if name.endswith("action_ball_full_mdp_lean_runtime"):
            return runtime
        if name.endswith("action_ball_full_mdp_lean_rewards"):
            return types.SimpleNamespace(MANAGER_NAMES=("reward",))
        if name.endswith("hope_actions"):
            module = types.ModuleType(name)
            module.ClampedJointPositionAction = ClampedJointPositionAction
            return module
        if name.endswith("action_ball_full_mdp_durable_wal"):
            return durable_wal
        raise AssertionError(f"unexpected adapter import: {name}")

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)


def test_update_orders_optimizer_between_prepare_and_durable_ack(
    tmp_path, _fake_imports, capsys, monkeypatch
):
    # IsaacLab's ManagerTermBase inherits ABC, so the real action class uses
    # ABCMeta rather than the literal ``type`` metaclass.  The adapter must pin
    # the class object and live instance, not reject that legitimate metaclass.
    assert isinstance(ClampedJointPositionAction, type)
    assert type(ClampedJointPositionAction) is not type
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )

    def update():
        env.owner.events.append("update")
        return {"loss": 1.0}

    append = boundary._append
    append_count = 0

    def logged_append(line):
        nonlocal append_count
        env.owner.events.append("wal_pending" if append_count == 0 else "wal_epoch_ack")
        append_count += 1
        return append(line)

    monkeypatch.setattr(boundary, "_append", logged_append)

    assert boundary.update(
        update, update_index=0, completed_environment_steps=48
    ) == {"loss": 1.0}
    assert [event if isinstance(event, str) else event[0] for event in env.owner.events] == [
        "healthy",
        "healthy",
        "prepare",
        "safety_prepare",
        "update",
        "mark",
        "summary",
        "wal_pending",
        "ack",
        "wal_epoch_ack",
        "latch",
        "safety_ack",
    ]
    rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "action_ball_epoch_durable_pending_v2",
        "action_ball_epoch_durable_ack_v2",
    ]
    assert rows[0]["pending_ack_telemetry"]["diagnostic_unauthorized"] is True
    assert rows[0]["pending_ack_telemetry"]["schema_version"] == 11
    assert rows[0]["pending_ack_telemetry"]["kind"] == (
        "action_ball_epoch_optimizer_update_ack_telemetry_v11"
    )
    assert rows[0]["pending_ack_telemetry"]["joint_safety"] == {
        "event": "hope_joint_safety_diagnostic_compact_update",
        "schema_version": 1,
        "status": "diagnostic_compact_prepared_before_optimizer",
        "ppo_update": 0,
        "consume_sequence": 0,
        "num_envs": 2,
        "policy_step_count": 24,
        "first_policy_step_sequence": 0,
        "last_policy_step_sequence": 23,
        "counter_totals": {
            "qdes_joint_count": 0,
            "policy_crossing_joint_count": 0,
            "substep_hard_crossing_joint_count": 0,
            "actual_hard_edge_joint_count": 0,
        },
        "minimum_hard_gap_rad": 1.0,
        "terminal_archive_count": 0,
        "identity_bound_policy_step_count": 0,
        "formal_authority": False,
    }
    assert env.action_term.pending is None
    assert env.action_term.acknowledged == 1
    output = capsys.readouterr().out.splitlines()
    assert [line.split("=", 1)[0] for line in output] == [
        "HOPE_JOINT_SAFETY_UPDATE_JSON",
        "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON",
    ]
    safety_receipt = json.loads(output[0].split("=", 1)[1])
    assert safety_receipt["status"] == (
        "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
    )
    assert {
        name: safety_receipt[name]
        for name in (
            "ppo_update",
            "consume_sequence",
            "completed_environment_steps",
            "epoch_operation_sequence",
            "epoch_drain_sequence",
            "epoch_commit_start",
            "epoch_commit_end",
        )
    } == {
        "ppo_update": 0,
        "consume_sequence": 0,
        "completed_environment_steps": 48,
        "epoch_operation_sequence": 1,
        "epoch_drain_sequence": 1,
        "epoch_commit_start": 0,
        "epoch_commit_end": 0,
    }


def test_optimizer_exception_poisoned_without_wal_or_ack(tmp_path, _fake_imports):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )

    def fail():
        env.owner.events.append("update")
        raise ValueError("optimizer failed")

    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(fail, update_index=0, completed_environment_steps=48)
    assert boundary._path.read_bytes() == b""
    names = [event if isinstance(event, str) else event[0] for event in env.owner.events]
    assert names == [
        "healthy", "healthy", "prepare", "safety_prepare", "update", "poison"
    ]
    assert env.action_term.pending is not None
    assert env.action_term.acknowledged == 0


def test_low_reward_optimizer_success_is_not_an_ack_gate(
    tmp_path, _fake_imports, capsys
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    result = boundary.update(
        lambda: {"mean_reward": -1.0e30, "mean_loss": 1.0e30},
        update_index=0,
        completed_environment_steps=48,
    )
    assert result == {"mean_reward": -1.0e30, "mean_loss": 1.0e30}
    assert env.owner.poisoned is False
    names = [
        event if isinstance(event, str) else event[0]
        for event in env.owner.events
    ]
    assert names[-5:] == ["mark", "summary", "ack", "latch", "safety_ack"]
    assert [
        json.loads(line)["kind"]
        for line in boundary._path.read_text().splitlines()
    ] == [
        "action_ball_epoch_durable_pending_v2",
        "action_ball_epoch_durable_ack_v2",
    ]
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" in capsys.readouterr().out


def test_mark_optimizer_returned_failure_is_sticky_and_never_acks(
    tmp_path, _fake_imports, capsys, monkeypatch
):
    env = _Env()

    def fail_mark(self, _boundary, *, update_index):
        del update_index
        self.events.append("mark_failed")
        raise RuntimeError("injected optimizer-return mark failure")

    monkeypatch.setattr(_Owner, "mark_optimizer_returned", fail_mark)
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    with pytest.raises(RuntimeError, match="retry forbidden") as caught:
        boundary.update(
            lambda: env.owner.events.append("update"),
            update_index=0,
            completed_environment_steps=48,
        )
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "optimizer-return mark failure" in str(caught.value.__cause__)
    assert env.owner.poisoned is True
    names = [
        event if isinstance(event, str) else event[0]
        for event in env.owner.events
    ]
    assert names[-3:] == ["update", "mark_failed", "poison"]
    assert "summary" not in names
    assert "ack" not in names
    assert boundary._path.read_bytes() == b""
    assert env.action_term.pending is not None
    assert env.action_term.acknowledged == 0
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: None,
            update_index=0,
            completed_environment_steps=48,
        )


def test_pending_fsync_failure_precedes_destructive_owner_ack(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    monkeypatch.setattr(
        boundary, "_append", lambda _line: (_ for _ in ()).throw(OSError("fsync"))
    )
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(lambda: {}, update_index=0, completed_environment_steps=48)
    names = [event if isinstance(event, str) else event[0] for event in env.owner.events]
    assert "ack" not in names
    assert "safety_ack" not in names
    assert names[-1] == "poison"
    assert env.action_term.pending is not None


def test_compact_small_capacity_survives_five_n4096_updates(
    tmp_path, _fake_imports, capsys
):
    env = _Env(num_envs=4096)
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    for update_index in range(5):
        env.action_term.snapshot = _compact_snapshot(
            num_envs=4096,
            policy_steps=24,
            first_sequence=update_index * 24,
            consume_sequence=update_index,
            capacity=2,
        )
        result = boundary.update(
            lambda update_index=update_index: {"update": update_index},
            update_index=update_index,
            completed_environment_steps=(update_index + 1) * 4096 * 24,
        )
        assert result == {"update": update_index}
        assert env.action_term.pending is None
    rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
    assert len(rows) == 10
    assert [row["kind"] for row in rows] == [
        kind
        for _ in range(5)
        for kind in (
            "action_ball_epoch_durable_pending_v2",
            "action_ball_epoch_durable_ack_v2",
        )
    ]
    pending = rows[::2]
    assert [row["pending_ack_telemetry"]["ppo_update"] for row in pending] == list(
        range(5)
    )
    assert [
        row["pending_ack_telemetry"]["completed_environment_steps"]
        for row in pending
    ] == [(index + 1) * 4096 * 24 for index in range(5)]
    assert env.action_term.acknowledged == 5
    output = capsys.readouterr().out.splitlines()
    receipts = [
        json.loads(line.split("=", 1)[1])
        for line in output
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    assert len(receipts) == 5
    assert [receipt["ppo_update"] for receipt in receipts] == list(range(5))
    assert [receipt["consume_sequence"] for receipt in receipts] == list(range(5))
    assert all(
        receipt["status"]
        == "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
        for receipt in receipts
    )


def test_finite_nonzero_actual_hard_edge_is_consumed(tmp_path, _fake_imports):
    env = _Env()
    env.action_term.snapshot = _compact_snapshot(
        num_envs=2,
        policy_steps=24,
        first_sequence=0,
        consume_sequence=0,
        actual_edge=True,
    )
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    assert boundary.update(
        lambda: "finite-terminal-sample",
        update_index=0,
        completed_environment_steps=48,
    ) == "finite-terminal-sample"
    pending = json.loads(boundary._path.read_text().splitlines()[0])
    safety = pending["pending_ack_telemetry"]["joint_safety"]
    assert safety["counter_totals"]["actual_hard_edge_joint_count"] == 1
    assert safety["minimum_hard_gap_rad"] == -0.125
    assert env.action_term.acknowledged == 1
    assert env.action_term.pending is None


@pytest.mark.parametrize("malformation", ["nan", "incomplete"])
def test_malformed_compact_snapshot_freezes_and_prevents_optimizer(
    tmp_path, _fake_imports, malformation
):
    env = _Env()
    env.action_term.snapshot = _compact_snapshot(
        num_envs=2,
        policy_steps=24,
        first_sequence=0,
        consume_sequence=0,
        nan_gap=malformation == "nan",
        incomplete=malformation == "incomplete",
    )
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    optimizer_calls = []
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )
    assert optimizer_calls == []
    assert boundary._path.read_bytes() == b""
    assert env.action_term.pending is not None
    assert env.action_term.acknowledged == 0
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )
    assert optimizer_calls == []


@pytest.mark.parametrize(
    ("field", "shape", "dtype"),
    [
        ("policy_step_count", (2,), "uint8"),
        ("actual_hard_edge_joint_count", (2, 1), "int32"),
        ("minimum_hard_lower_gap", (2, 1), "float64"),
    ],
)
def test_compact_snapshot_rejects_nonproducer_dtypes_before_optimizer(
    tmp_path, _fake_imports, field, shape, dtype
):
    env = _Env()
    env.action_term.snapshot["since_last_consume"][field] = _filled(
        0.0 if dtype.startswith("float") else 0,
        shape,
        dtype,
    )
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    optimizer_calls = []
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )
    assert optimizer_calls == []
    assert boundary._path.read_bytes() == b""
    assert env.action_term.pending is not None
    assert env.action_term.acknowledged == 0


@pytest.mark.parametrize("run_mode", ["formal", "legacy"])
def test_runner_rejects_nonlean_mode_before_any_safety_or_wal(
    tmp_path, _fake_imports, run_mode
):
    env = _Env()
    with pytest.raises(RuntimeError, match="only fresh single_action_lean"):
        adapter.ActionBallFullMdpRsl3Runner(
            env,
            {},
            str(tmp_path),
            action_ball_full_mdp_runtime_owner=env.owner,
            action_ball_full_mdp_run_mode=run_mode,
        )
    assert env.owner.events == []
    assert env.action_term.pending is None
    assert list(tmp_path.iterdir()) == []


def test_runner_save_uses_explicit_nonresumable_snapshot_name_and_metadata(
    tmp_path, monkeypatch
):
    calls = []

    class _Tensor:
        pass

    class _Finite:
        def all(self):
            return self

        def item(self):
            return True

    fake_torch = types.SimpleNamespace(
        Tensor=_Tensor,
        isfinite=lambda _value: _Finite(),
        load=lambda _stream, **_kwargs: {
            "model_state_dict": {"weight": _Tensor()},
            "optimizer_state_dict": {
                "state": {0: {"step": _Tensor()}},
                "param_groups": [{"params": [0], "lr": 1.0e-3}],
            },
            "iter": 1000,
            "infos": calls[-1][1],
        },
    )
    real_import_module = adapter.importlib.import_module

    def _import_module(name):
        if name == "torch":
            return fake_torch
        return real_import_module(name)

    monkeypatch.setattr(adapter.importlib, "import_module", _import_module)

    def _base_save(_self, path, infos=None):
        calls.append((path, infos))
        Path(path).write_bytes(b"unit-fixture-snapshot")

    monkeypatch.setattr(adapter.OnPolicyRunner, "save", _base_save, raising=False)
    runner = object.__new__(adapter.ActionBallFullMdpRsl3Runner)
    requested = tmp_path / "model_1000.pt"
    runner.save(str(requested), infos={"upstream": "kept"})

    assert calls == [
        (
            str(tmp_path / "model_1000.diagnostic_nonresumable.pt"),
            {
                "upstream": "kept",
                "action_ball_full_mdp_snapshot_kind": (
                    "policy_optimizer_diagnostic_nonresumable_v1"
                ),
                "checkpoint_authority": False,
                "resume_authority": False,
            },
        )
    ]
    snapshot = tmp_path / "model_1000.diagnostic_nonresumable.pt"
    receipt_path = tmp_path / (
        "model_1000.diagnostic_nonresumable.pt.receipt.json"
    )
    assert json.loads(receipt_path.read_text()) == {
        "schema_version": 1,
        "kind": "action_ball_full_mdp_diagnostic_snapshot_receipt_v1",
        "snapshot_name": snapshot.name,
        "learning_iteration": 1000,
        "snapshot_size_bytes": snapshot.stat().st_size,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "payload_kind": "policy_optimizer_diagnostic_nonresumable_v1",
        "model_tensor_count": 1,
        "optimizer_tensor_count": 1,
        "all_tensors_finite": True,
        "checkpoint_authority": False,
        "resume_authority": False,
    }
    with pytest.raises(RuntimeError, match="forbids checkpoint load/resume"):
        runner.load(str(calls[0][0]))


def test_snapshot_receipt_rejects_unparseable_payload_before_sidecar(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "model_7.diagnostic_nonresumable.pt"
    snapshot.write_bytes(b"not-a-torch-payload")
    fake_torch = types.SimpleNamespace(
        load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unparseable")
        )
    )
    real_import_module = adapter.importlib.import_module
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import_module(name),
    )
    with pytest.raises(RuntimeError, match="unparseable"):
        adapter._write_snapshot_receipt(
            snapshot,
            learning_iteration=7,
            required_infos={
                "action_ball_full_mdp_snapshot_kind": (
                    "policy_optimizer_diagnostic_nonresumable_v1"
                ),
                "checkpoint_authority": False,
                "resume_authority": False,
            },
        )
    assert not snapshot.with_name(snapshot.name + ".receipt.json").exists()


def test_adapter_rejects_foreign_exact_looking_compact_action_term(
    tmp_path, _fake_imports
):
    env = _Env()
    foreign = _ForeignActionTerm(env.owner.events)
    foreign.snapshot = env.action_term.snapshot
    env.action_term = foreign
    env.action_manager.term = foreign
    with pytest.raises(
        RuntimeError,
        match=r"compact joint-safety action producer: joint_pos_type=",
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )
    assert foreign.pending is None
    assert list(tmp_path.iterdir()) == []


def test_adapter_names_disabled_live_compact_evidence_before_wal(
    tmp_path, _fake_imports
):
    env = _Env()
    env.action_term._joint_safety_diagnostic_compact_evidence = False
    with pytest.raises(
        RuntimeError,
        match=r"compact joint-safety action producer: compact_evidence$",
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )
    assert list(tmp_path.iterdir()) == []


def test_owner_ack_failure_preserves_frozen_safety_generation(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    monkeypatch.setattr(
        boundary,
        "_ack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("owner ack")),
    )
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(lambda: {}, update_index=0, completed_environment_steps=48)
    rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "action_ball_epoch_durable_pending_v2"
    ]
    assert env.action_term.pending is not None
    assert env.action_term.acknowledged == 0
    assert env.owner.poisoned is True


def test_safety_ack_is_after_two_wal_rows_and_owner_latch(
    tmp_path, _fake_imports, monkeypatch, capsys
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )

    def reject_safety_ack(_token):
        names = [
            event if isinstance(event, str) else event[0]
            for event in env.owner.events
        ]
        assert names[-2:] == ["ack", "latch"]
        rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
        assert [row["kind"] for row in rows] == [
            "action_ball_epoch_durable_pending_v2",
            "action_ball_epoch_durable_ack_v2",
        ]
        raise RuntimeError("safety ack")

    monkeypatch.setattr(boundary, "_safety_ack", reject_safety_ack)
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(lambda: {}, update_index=0, completed_environment_steps=48)
    assert env.action_term.pending is not None
    assert env.owner.poisoned is True
    assert "HOPE_JOINT_SAFETY_UPDATE_JSON=" not in capsys.readouterr().out


def test_epoch_ack_fsync_failure_preserves_frozen_safety_generation(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    append = boundary._append
    append_calls = 0

    def fail_second_append(line):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 2:
            raise OSError("epoch ack fsync")
        return append(line)

    monkeypatch.setattr(boundary, "_append", fail_second_append)
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(lambda: {}, update_index=0, completed_environment_steps=48)
    rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "action_ball_epoch_durable_pending_v2"
    ]
    names = [event if isinstance(event, str) else event[0] for event in env.owner.events]
    assert "ack" in names
    assert "latch" not in names
    assert "safety_ack" not in names
    assert env.action_term.pending is not None
    assert env.owner.poisoned is True


def test_owner_latch_failure_preserves_frozen_safety_generation(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    monkeypatch.setattr(
        boundary,
        "_latch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("latch")),
    )
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(lambda: {}, update_index=0, completed_environment_steps=48)
    rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "action_ball_epoch_durable_pending_v2",
        "action_ball_epoch_durable_ack_v2",
    ]
    assert "safety_ack" not in env.owner.events
    assert env.action_term.pending is not None
    assert env.owner.poisoned is True
