from __future__ import annotations

from dataclasses import dataclass
import json
import types

import pytest

from whole_body_tracking.utils import action_ball_full_mdp_rsl3_adapter as adapter


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


class _Epoch:
    num_envs = 2
    shot_slot_capacity = 1


class _Owner:
    def __init__(self, env, lease):
        self.full_mdp_runtime_env = env
        self.full_mdp_runtime_lease = lease
        self.epoch_owner = _Epoch()
        self.events = []
        self.active = None
        self.summary = None

    def require_healthy(self):
        self.events.append("healthy")

    def prepare_pre_optimizer_ppo_boundary(
        self, *, update_index, completed_environment_steps
    ):
        self.events.append(("prepare", update_index, completed_environment_steps))
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
                completed_environment_steps=(update_index + 1) * 48,
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


class _Env:
    num_envs = 2

    def __init__(self):
        self.unwrapped = self
        self.action_ball_full_mdp_runtime_lease = object()
        self.owner = _Owner(self, self.action_ball_full_mdp_runtime_lease)

    def action_ball_full_mdp_ppo_drain_owner(self, lease):
        assert lease is self.action_ball_full_mdp_runtime_lease
        return self.owner


@pytest.fixture
def _fake_imports(monkeypatch):
    real_import = adapter.importlib.import_module
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
        return real_import(name)

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)


def test_update_orders_optimizer_between_prepare_and_durable_ack(
    tmp_path, _fake_imports, capsys
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )

    def update():
        env.owner.events.append("update")
        return {"loss": 1.0}

    assert boundary.update(
        update, update_index=0, completed_environment_steps=48
    ) == {"loss": 1.0}
    assert [event if isinstance(event, str) else event[0] for event in env.owner.events] == [
        "healthy", "healthy", "prepare", "update", "mark", "summary", "ack", "latch"
    ]
    rows = [json.loads(line) for line in boundary._path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == [
        "action_ball_epoch_durable_pending_v2",
        "action_ball_epoch_durable_ack_v2",
    ]
    assert rows[0]["pending_ack_telemetry"]["diagnostic_unauthorized"] is True
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" in capsys.readouterr().out


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
    assert names == ["healthy", "healthy", "prepare", "update", "poison"]


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
    assert names[-1] == "poison"
