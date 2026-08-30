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


_REAL_IMPORT_MODULE = importlib.import_module
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
    due_terminal_overlap_rows: int = 0


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

    def assert_joint_safety_ledger_consume_idle(self):
        if self.pending is not None:
            raise RuntimeError(
                "joint-safety ledger has a prepared consume awaiting acknowledgement"
            )

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


class RacketTargetCommand:
    def __init__(self):
        # FullMDP performs one canonical genesis command compute before the
        # first H=24 rollout, so update zero must drain H+1 rows.
        self.pending_rows = 25
        self.pending_hold_rows = 25
        self.materialize_calls = 0
        self.events = None
        self.materialize_failure = None
        self.assert_failure = None

    def _action_ball_full_mdp_deferred_exact_metrics_enabled(self):
        return True

    def stage_exact_metric_rows(self, count):
        if (
            type(count) is not int
            or count < 1
            or self.pending_rows != 0
            or self.pending_hold_rows != 0
        ):
            raise RuntimeError("fake exact metric row staging differs")
        self.pending_rows = count
        self.pending_hold_rows = count

    def materialize_action_ball_diagnostic_metrics_for_report(
        self, *, expected_full_mdp_exact_row_counts=None
    ):
        if self.events is not None:
            self.events.append("metric_materialize")
        if self.materialize_failure is not None:
            raise self.materialize_failure
        if expected_full_mdp_exact_row_counts is not None:
            if self.pending_hold_rows != self.pending_rows:
                raise RuntimeError(
                    "full-MDP exact/hold metric rollout row counts differ"
                )
            if self.pending_rows not in expected_full_mdp_exact_row_counts:
                raise RuntimeError("full-MDP exact metric rollout row count differs")
            if self.pending_hold_rows not in expected_full_mdp_exact_row_counts:
                raise RuntimeError(
                    "full-MDP hold/recovery metric rollout row count differs"
                )
        self.materialize_calls += 1
        self.pending_rows = 0
        self.pending_hold_rows = 0

    def assert_action_ball_diagnostic_metrics_materialized_for_report(self):
        if self.events is not None:
            self.events.append("metric_assert")
        if self.assert_failure is not None:
            raise self.assert_failure
        if self.pending_rows:
            raise RuntimeError("full-MDP exact metrics are pending on device")
        if self.pending_hold_rows:
            raise RuntimeError(
                "full-MDP hold/recovery metrics are pending on device"
            )


RacketTargetCommand.__module__ = (
    "whole_body_tracking.tasks.tracking.mdp.hope_commands"
)


class _ForeignRacketTargetCommand(RacketTargetCommand):
    pass


class _CommandManager:
    def __init__(self, term):
        self.term = term

    def get_term(self, name):
        assert name == "racket_target"
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

    def require_optimizer_boundary_idle(self):
        self.require_healthy()
        if self.active is not None:
            raise RuntimeError("owner optimizer boundary is active")

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
        self.observation_facts = _v3_observation_facts()
        self.action_ball_full_mdp_runtime_lease = object()
        self.owner = _Owner(self, self.action_ball_full_mdp_runtime_lease, num_envs)
        self.action_term = _ActionTerm(self.owner.events)
        self.action_manager = _ActionManager(self.action_term)
        self.command_term = RacketTargetCommand()
        self.command_manager = _CommandManager(self.command_term)
        self.owner._racket = self.command_term
        self.action_term.snapshot = _compact_snapshot(
            num_envs=num_envs,
            policy_steps=24,
            first_sequence=0,
            consume_sequence=0,
        )

    def action_ball_full_mdp_ppo_drain_owner(self, lease):
        assert lease is self.action_ball_full_mdp_runtime_lease
        return self.owner


def _v3_observation_facts(**overrides):
    facts = {
        "actor_obs_contract": "action_ball_full_mdp_semantic_actor_v3",
        "actor_obs_mode": "action_ball_full_mdp",
        "actor_obs_total_dim": 215,
        "actor_obs_term_names": ["action_epoch"],
        "actor_obs_term_dims": [215],
        "critic_obs_contract": "action_ball_full_mdp_semantic_critic_v3",
        "critic_obs_total_dim": 231,
        "critic_obs_term_names": ["action_epoch"],
        "critic_obs_term_dims": [231],
        "fresh_full_mdp_observation_kind": (
            "action_ball_full_mdp_semantic_observation_v3"
        ),
        "fresh_full_mdp_diagnostic_unauthorized": True,
        "fresh_full_mdp_launch_authorized": False,
        "fresh_full_mdp_no_capacity_receipt_or_sha_authority": True,
    }
    facts.update(overrides)
    return facts


def _v3_snapshot_identity(*, sha256="a" * 64, **overrides):
    identity = {
        "action_ball_full_mdp_snapshot_kind": (
            "policy_optimizer_diagnostic_nonresumable_v2"
        ),
        "fresh_full_mdp_observation_kind": (
            "action_ball_full_mdp_semantic_observation_v3"
        ),
        "actor_obs_contract": "action_ball_full_mdp_semantic_actor_v3",
        "actor_obs_total_dim": 215,
        "critic_obs_contract": "action_ball_full_mdp_semantic_critic_v3",
        "critic_obs_total_dim": 231,
        "training_contract_schema_version": 3,
        "training_contract_sha256": sha256,
        "diagnostic_unauthorized": True,
        "checkpoint_authority": False,
        "resume_authority": False,
    }
    identity.update(overrides)
    return identity


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
    def import_module(name, package=None):
        resolved = (
            importlib.util.resolve_name(name, package)
            if name.startswith(".")
            else name
        )
        if resolved == "torch" or resolved.startswith("torch."):
            return _REAL_IMPORT_MODULE(name, package)
        if resolved == (
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_runtime"
        ):
            return runtime
        if resolved == (
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_observation_cfg"
        ):
            return types.SimpleNamespace(
                installed_observation_facts=lambda env: env.observation_facts
            )
        if resolved == (
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_rewards"
        ):
            return types.SimpleNamespace(MANAGER_NAMES=("reward",))
        if resolved == "whole_body_tracking.tasks.tracking.mdp.hope_actions":
            module = types.ModuleType(resolved)
            module.ClampedJointPositionAction = ClampedJointPositionAction
            return module
        if resolved == "whole_body_tracking.tasks.tracking.mdp.hope_commands":
            module = types.ModuleType(resolved)
            module.RacketTargetCommand = RacketTargetCommand
            return module
        if resolved == (
            "whole_body_tracking.utils.action_ball_full_mdp_durable_wal"
        ):
            return durable_wal
        raise AssertionError(f"unexpected adapter import: {resolved}")

    monkeypatch.setattr(
        adapter,
        "importlib",
        types.SimpleNamespace(import_module=import_module),
    )
    return runtime


def test_schedule_telemetry_uses_existing_public_due_plus_terminal_overlap(
    _fake_imports,
):
    summary = _Summary(
        frontier=_Frontier(
            update_index=3,
            completed_environment_steps=96,
            due_terminal_overlap_rows=2,
        ),
        settlement=_Settlement(
            transactions=1,
            due_rows=5,
            accepted=1,
            censored=1,
            rejected=1,
            deferred=2,
        ),
    )

    telemetry = adapter._telemetry(summary, _fake_imports)

    assert telemetry["d05_due_rows"] == 5
    assert telemetry["d05_due_terminal_overlap_rows"] == 2
    assert telemetry["d05_scheduled_due_rows"] == 7
    assert "d05_public_due_rows" not in telemetry


def test_compact_telemetry_keeps_exact_action_side_denominators_and_bounded_samples():
    def opportunity(**overrides):
        values = {
            "action_uid": 11,
            "action_slot": 0,
            "stroke_family": "backhand",
            "attribution_valid": True,
            "selected": True,
            "accepted": True,
            "censored": False,
            "rejected": False,
            "deferred": False,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)
    strata = adapter._opportunity_strata(
        (
            opportunity(),
            opportunity(selected=False, accepted=False, deferred=True),
            opportunity(
                action_uid=12,
                action_slot=1,
                stroke_family="forehand",
                selected=False,
                accepted=False,
                rejected=True,
            ),
        )
    )
    assert strata == [
        {
            "action_uid": 11,
            "action_slot": 0,
            "stroke_family": "backhand",
            "action_attribution_valid": True,
            "opportunity_rows": 2,
            "selected_rows": 1,
            "accepted_rows": 1,
            "censored_rows": 0,
            "rejected_rows": 0,
            "deferred_rows": 1,
        },
        {
            "action_uid": 12,
            "action_slot": 1,
            "stroke_family": "forehand",
            "action_attribution_valid": True,
            "opportunity_rows": 1,
            "selected_rows": 0,
            "accepted_rows": 0,
            "censored_rows": 0,
            "rejected_rows": 1,
            "deferred_rows": 0,
        },
    ]
    sample = adapter._bounded_rows(
        tuple(range(9)), limit=4, projector=lambda value: {"value": value}
    )
    assert sample == {
        "row_count": 9,
        "sample_limit": 4,
        "sample_rows": [
            {"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}
        ],
        "dropped_row_count": 5,
    }


def test_compact_shot_strata_preserve_lifecycle_and_outcome_histograms():
    def evidence(**overrides):
        values = {
            "lifecycle_bits": 0b101101,
            "r03_valid_bits": 3,
            "physical_valid_bits": 7,
            "r06_valid_bits": 5,
            "r06_outcome_code": 2,
            "r06_predicate_bits": 9,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def shot(**overrides):
        values = {
            "action_uid": 11,
            "action_slot": 0,
            "stroke_family": "backhand",
            "attribution_valid": True,
            "evidence": evidence(),
            "motion_close_reason": 4,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)
    strata = adapter._shot_strata(
        (shot(), shot(evidence=evidence(r06_outcome_code=3))),
        lifecycle_flags=(
            "reveal_committed",
            "playback_started",
            "motion_closed",
            "physical_launched",
            "outcome_settled",
            "payment_recorded",
        ),
    )
    assert len(strata) == 1
    row = strata[0]
    assert row["shot_rows"] == 2
    assert row["lifecycle_flag_counts"] == {
        "reveal_committed": 2,
        "playback_started": 0,
        "motion_closed": 2,
        "physical_launched": 2,
        "outcome_settled": 0,
        "payment_recorded": 2,
    }
    assert row["r06_outcome_code_counts"] == {"2": 1, "3": 1}
    assert row["motion_close_reason_counts"] == {"4": 2}


def test_fake_imports_preserve_package_aware_torch_lazy_import(
    _fake_imports,
):
    assert importlib.import_module is _REAL_IMPORT_MODULE
    assert adapter.importlib.import_module is not _REAL_IMPORT_MODULE


def test_create_wal_fsyncs_file_child_directory_and_parent(tmp_path, monkeypatch):
    real_open = adapter.os.open
    real_fsync = adapter.os.fsync
    fd_paths = {}
    synced_paths = []

    def tracked_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        synced_paths.append(fd_paths.get(fd))
        return real_fsync(fd)

    monkeypatch.setattr(adapter.os, "open", tracked_open)
    monkeypatch.setattr(adapter.os, "fsync", tracked_fsync)

    path, _identity, _segment = adapter.ActionBallFullMdpRsl3Adapter._create_wal(
        str(tmp_path)
    )

    assert path.parent == tmp_path / "action_ball_epoch_durable_wal"
    assert synced_paths == [path, path.parent, tmp_path]


def test_create_wal_parent_fsync_failure_is_fail_closed(tmp_path, monkeypatch):
    real_fsync = adapter.os.fsync
    calls = 0

    def fail_parent_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("parent fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(adapter.os, "fsync", fail_parent_fsync)

    with pytest.raises(OSError, match="parent fsync failed"):
        adapter.ActionBallFullMdpRsl3Adapter._create_wal(str(tmp_path))
    assert calls == 3
    torch = pytest.importorskip("torch")
    assert adapter.importlib.import_module(".optim", "torch") is torch.optim
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    optimizer.zero_grad()
    parameter.square().sum().backward()
    optimizer.step()
    assert int(optimizer.state[parameter]["step"].item()) == 1
    assert importlib.import_module is _REAL_IMPORT_MODULE


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
    env.command_term.events = env.owner.events

    def update():
        assert env.command_term.pending_rows == 0
        assert env.command_term.pending_hold_rows == 0
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

    owner_ack = boundary._ack

    def command_ack(*args, **kwargs):
        result = owner_ack(*args, **kwargs)
        assert result is None
        return None

    monkeypatch.setattr(boundary, "_ack", command_ack)

    assert boundary.update(
        update, update_index=0, completed_environment_steps=48
    ) == {"loss": 1.0}
    assert env.command_term.materialize_calls == 1
    assert [event if isinstance(event, str) else event[0] for event in env.owner.events] == [
        "healthy",
        "healthy",
        "prepare",
        "metric_materialize",
        "metric_assert",
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
    assert rows[0]["pending_ack_telemetry"]["schema_version"] == 13
    assert rows[0]["pending_ack_telemetry"]["kind"] == (
        "action_ball_epoch_optimizer_update_ack_telemetry_v13"
    )
    assert {
        key: rows[0]["pending_ack_telemetry"][key]
        for key in (
            "d05_scheduled_due_rows",
            "d05_due_terminal_overlap_rows",
        )
    } == {
        "d05_scheduled_due_rows": 0,
        "d05_due_terminal_overlap_rows": 0,
    }
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
    assert env.owner.active is None
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


def test_command_metric_drain_failure_poisons_before_optimizer(
    tmp_path, _fake_imports
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    env.command_term.events = env.owner.events
    env.command_term.materialize_failure = RuntimeError("metric drain failed")
    optimizer_calls = []
    with pytest.raises(RuntimeError, match="optimizer boundary failed"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )

    assert optimizer_calls == []
    assert env.owner.poisoned is True
    assert env.owner.active is not None
    poison = [event for event in env.owner.events if isinstance(event, tuple) and event[0] == "poison"]
    assert len(poison) == 1
    assert poison[0][1] is env.owner.active
    assert boundary._last_update == -1
    names = [
        event if isinstance(event, str) else event[0]
        for event in env.owner.events
    ]
    assert "metric_materialize" in names
    assert not {
        "metric_assert",
        "safety_prepare",
        "update",
        "mark",
        "summary",
        "wal_pending",
        "ack",
        "wal_epoch_ack",
        "latch",
        "safety_ack",
    }.intersection(names)
    assert boundary._path.read_bytes() == b""
    events_after_failure = list(env.owner.events)
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )
    assert optimizer_calls == []
    assert env.owner.events == events_after_failure
    assert boundary._path.read_bytes() == b""


def test_command_metric_failure_is_sticky_when_owner_poison_hook_fails(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    env.command_term.events = env.owner.events
    env.command_term.materialize_failure = RuntimeError("metric drain failed")
    optimizer_calls = []
    monkeypatch.setattr(
        boundary,
        "_poison",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("owner poison hook failed")
        ),
    )

    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )

    assert env.owner.poisoned is False
    assert boundary._failure_reason is not None
    assert optimizer_calls == []
    assert boundary._path.read_bytes() == b""
    events_after_failure = list(env.owner.events)
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.assert_snapshot_boundary_clean()
    assert env.owner.events == events_after_failure
    assert optimizer_calls == []
    assert boundary._path.read_bytes() == b""


def test_first_update_requires_genesis_plus_rollout_metric_rows(
    tmp_path, _fake_imports
):
    env = _Env()
    env.command_term.pending_rows = 24
    env.command_term.pending_hold_rows = 24
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    optimizer_calls = []

    with pytest.raises(RuntimeError, match="optimizer boundary failed"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )

    assert optimizer_calls == []
    assert env.owner.poisoned is True
    assert env.owner.active is not None
    assert boundary._last_update == -1


def test_first_update_hold_metric_span_failure_poisons_before_optimizer(
    tmp_path, _fake_imports
):
    env = _Env()
    env.command_term.pending_hold_rows = 24
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    optimizer_calls = []

    with pytest.raises(RuntimeError, match="optimizer boundary failed"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )

    assert optimizer_calls == []
    assert env.owner.poisoned is True
    assert env.owner.active is not None
    assert boundary._last_update == -1


@pytest.mark.parametrize("pending_rows", [0, 23, 25])
def test_later_update_requires_exactly_one_rollout_of_metric_rows(
    tmp_path, _fake_imports, pending_rows
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    optimizer_calls = []
    boundary.update(
        lambda: optimizer_calls.append(0),
        update_index=0,
        completed_environment_steps=48,
    )
    env.action_term.snapshot = _compact_snapshot(
        num_envs=2,
        policy_steps=24,
        first_sequence=24,
        consume_sequence=1,
    )
    if pending_rows:
        env.command_term.stage_exact_metric_rows(pending_rows)

    with pytest.raises(RuntimeError, match="optimizer boundary failed"):
        boundary.update(
            lambda: optimizer_calls.append(1),
            update_index=1,
            completed_environment_steps=96,
        )

    assert optimizer_calls == [0]
    assert env.owner.poisoned is True
    assert env.owner.active is not None
    assert boundary._last_update == 0


def test_command_metric_assert_failure_poisons_before_optimizer(
    tmp_path, _fake_imports
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    env.command_term.events = env.owner.events
    env.command_term.assert_failure = RuntimeError(
        "metric tape remained pending"
    )
    optimizer_calls = []
    with pytest.raises(RuntimeError, match="optimizer boundary failed"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )

    assert optimizer_calls == []
    assert env.owner.poisoned is True
    assert env.owner.active is not None
    poison = [event for event in env.owner.events if isinstance(event, tuple) and event[0] == "poison"]
    assert len(poison) == 1
    assert poison[0][1] is env.owner.active
    assert boundary._last_update == -1
    names = [
        event if isinstance(event, str) else event[0]
        for event in env.owner.events
    ]
    assert names.index("metric_materialize") < names.index("metric_assert")
    assert not {
        "safety_prepare",
        "update",
        "mark",
        "summary",
        "wal_pending",
        "ack",
        "wal_epoch_ack",
        "latch",
        "safety_ack",
    }.intersection(names)
    assert boundary._path.read_bytes() == b""
    events_after_failure = list(env.owner.events)
    with pytest.raises(RuntimeError, match="retry forbidden"):
        boundary.update(
            lambda: optimizer_calls.append(True),
            update_index=0,
            completed_environment_steps=48,
        )
    assert optimizer_calls == []
    assert env.owner.events == events_after_failure
    assert boundary._path.read_bytes() == b""


def test_snapshot_guard_rejects_active_optimizer_boundary(
    tmp_path, _fake_imports
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    env.command_term.pending_rows = 0
    env.command_term.pending_hold_rows = 0
    boundary._safety_pending = {"prepared": True}

    with pytest.raises(RuntimeError, match="active optimizer boundary"):
        boundary.assert_snapshot_boundary_clean()


def test_snapshot_guard_rejects_before_owner_prepare(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    prepare = boundary._prepare
    guard_errors = []

    def reentrant_prepare(**kwargs):
        with pytest.raises(RuntimeError, match="active optimizer boundary") as exc:
            boundary.assert_snapshot_boundary_clean()
        guard_errors.append(str(exc.value))
        return prepare(**kwargs)

    monkeypatch.setattr(boundary, "_prepare", reentrant_prepare)
    boundary.update(
        lambda: {"loss": 1.0},
        update_index=0,
        completed_environment_steps=48,
    )

    assert len(guard_errors) == 1
    assert boundary._update_in_progress is False


@pytest.mark.parametrize(
    ("dirty_state", "error"),
    (
        ("owner_poisoned", "owner is poisoned"),
        ("owner_active", "optimizer boundary is active"),
        ("safety_pending", "prepared consume awaiting"),
        ("metric_pending", "metrics are pending"),
        ("hold_metric_pending", "hold/recovery metrics are pending"),
    ),
)
def test_runner_save_real_adapter_rejects_every_dirty_boundary_before_write(
    tmp_path, _fake_imports, monkeypatch, dirty_state, error
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    if dirty_state not in ("metric_pending", "hold_metric_pending"):
        env.command_term.pending_rows = 0
        env.command_term.pending_hold_rows = 0
    elif dirty_state == "metric_pending":
        env.command_term.pending_hold_rows = 0
    else:
        env.command_term.pending_rows = 0
    if dirty_state == "owner_poisoned":
        env.owner.poisoned = True
    elif dirty_state == "owner_active":
        boundary._prepare(update_index=0, completed_environment_steps=48)
    elif dirty_state == "safety_pending":
        env.action_term.prepare_joint_safety_ledger_consume()
    elif dirty_state not in ("metric_pending", "hold_metric_pending"):
        raise AssertionError("unknown dirty snapshot state")

    base_save_calls = []
    monkeypatch.setattr(
        adapter.OnPolicyRunner,
        "save",
        lambda *_args, **_kwargs: base_save_calls.append(True),
        raising=False,
    )
    runner = object.__new__(adapter.ActionBallFullMdpRsl3Runner)
    runner._full_mdp_adapter = boundary
    requested = tmp_path / "model_1000.pt"

    with pytest.raises(RuntimeError, match=error):
        runner.save(str(requested))

    assert base_save_calls == []
    assert not requested.exists()
    assert not (tmp_path / "model_1000.diagnostic_nonresumable.pt").exists()
    assert not (
        tmp_path / "model_1000.diagnostic_nonresumable.pt.receipt.json"
    ).exists()


def test_runner_alg_update_executes_real_optimizer_inside_boundary_and_returns_result(
    tmp_path, _fake_imports, monkeypatch
):
    torch = pytest.importorskip("torch")
    env = _Env()
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    storage_sentinel = object()
    result_sentinel = object()

    class _Algorithm:
        def __init__(self):
            self.storage = storage_sentinel

        def update(self):
            assert self.storage is storage_sentinel
            env.owner.events.append("optimizer_begin")
            optimizer.zero_grad()
            loss = parameter.square().sum()
            loss.backward()
            optimizer.step()
            env.owner.events.append("optimizer_end")
            return result_sentinel

    def base_init(self, base_env, train_cfg, log_dir, device):
        del train_cfg, log_dir, device
        self.env = base_env
        self.alg = _Algorithm()
        self.is_distributed = False
        self.num_steps_per_env = 24

    monkeypatch.setattr(adapter.OnPolicyRunner, "__init__", base_init, raising=False)
    runner = adapter.ActionBallFullMdpRsl3Runner(
        env,
        {},
        str(tmp_path),
        training_contract_schema_version=3,
        training_contract_sha256="a" * 64,
        action_ball_full_mdp_runtime_owner=env.owner,
        action_ball_full_mdp_run_mode="single_action_lean",
    )
    boundary = runner._full_mdp_adapter
    append = boundary._append
    append_count = 0

    def logged_append(line):
        nonlocal append_count
        env.owner.events.append(
            "wal_pending" if append_count == 0 else "wal_epoch_ack"
        )
        append_count += 1
        return append(line)

    monkeypatch.setattr(boundary, "_append", logged_append)
    before = parameter.detach().clone()

    assert runner.alg.update() is result_sentinel
    assert not torch.equal(parameter.detach(), before)
    state = optimizer.state[parameter]
    assert int(state["step"].item()) == 1
    assert torch.count_nonzero(state["exp_avg"]).item() == 1
    names = [
        event if isinstance(event, str) else event[0]
        for event in env.owner.events
    ]
    assert names == [
        "healthy",
        "healthy",
        "prepare",
        "safety_prepare",
        "optimizer_begin",
        "optimizer_end",
        "mark",
        "summary",
        "wal_pending",
        "ack",
        "wal_epoch_ack",
        "latch",
        "safety_ack",
    ]
    assert boundary._last_update == 0
    assert env.owner.active is None
    assert env.action_term.pending is None
    assert env.action_term.acknowledged == 1


@pytest.mark.parametrize("failure_mode", ["short_write", "broken_pipe"])
def test_stdout_delivery_failure_cannot_poison_committed_training_transaction(
    tmp_path, _fake_imports, monkeypatch, failure_mode
):
    env = _Env()
    boundary = adapter.ActionBallFullMdpRsl3Adapter(
        env=env, owner=env.owner, log_dir=str(tmp_path)
    )
    stdout_markers = []
    stderr_rows = []

    class FailingStdout:
        def write(self, value):
            stdout_markers.append(value)
            if failure_mode == "broken_pipe":
                raise BrokenPipeError("collector closed")
            return len(value) - 1

        def flush(self):
            raise AssertionError("failed stdout writes must not be flushed")

    class RecordingStderr:
        def write(self, value):
            stderr_rows.append(value)
            return len(value)

        def flush(self):
            return None

    monkeypatch.setattr(
        adapter,
        "sys",
        types.SimpleNamespace(stdout=FailingStdout(), stderr=RecordingStderr()),
    )

    assert boundary.update(
        lambda: {"loss": 1.0},
        update_index=0,
        completed_environment_steps=48,
    ) == {"loss": 1.0}
    env.action_term.snapshot = _compact_snapshot(
        num_envs=2,
        policy_steps=24,
        first_sequence=24,
        consume_sequence=1,
    )
    env.command_term.stage_exact_metric_rows(24)
    assert boundary.update(
        lambda: {"loss": 0.5},
        update_index=1,
        completed_environment_steps=96,
    ) == {"loss": 0.5}

    assert env.owner.poisoned is False
    assert env.owner.active is None
    assert env.action_term.pending is None
    assert env.action_term.acknowledged == 2
    assert boundary._last_update == 1
    assert [row.split("=", 1)[0] for row in stdout_markers] == [
        "HOPE_JOINT_SAFETY_UPDATE_JSON",
        "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON",
    ] * 2
    warnings = [
        json.loads(row.split("=", 1)[1])
        for row in stderr_rows
        if row.startswith("HOPE_NONAUTHORITATIVE_STDOUT_WARNING_JSON=")
    ]
    assert [warning["marker"] for warning in warnings] == [
        "HOPE_JOINT_SAFETY_UPDATE_JSON",
        "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON",
    ] * 2
    assert {warning["failure"] for warning in warnings} == {
        "short_write" if failure_mode == "short_write" else "builtins.BrokenPipeError"
    }
    for warning in warnings:
        assert warning["stdout_authoritative"] is False
        if warning["marker"] == "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON":
            assert warning["durable_wal_authoritative"] is True
            assert warning["durable_scope"] == "action_epoch"
            assert warning["training_transaction"] == "epoch_ack_committed"
        else:
            assert warning["durable_wal_authoritative"] is False
            assert warning["durable_scope"] == "none"
            assert (
                warning["training_transaction"]
                == "completed_in_process_not_durable"
            )
    forensic = durable_wal.read_forensic_committed_frontier(
        boundary._path,
        expected_rank=0,
        expected_segment_id=boundary._segment,
    )
    assert forensic["durable_epoch_ack_count"] == 2
    assert forensic["pending_without_epoch_ack"] is False
    assert forensic["committed_frontier"] == {
        "ppo_update": 1,
        "completed_environment_steps": 96,
        "epoch_operation_sequence": 2,
        "epoch_drain_sequence": 2,
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
        if update_index > 0:
            env.command_term.stage_exact_metric_rows(24)
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


@pytest.mark.parametrize(
    "authority_kwargs",
    [
        {"training_contract_lineage_exact": True},
        {"training_launch_claim_sha256": "a" * 64},
        {"require_exact_resume_state": True},
    ],
)
def test_runner_rejects_resume_or_lineage_authority_before_any_side_effect(
    tmp_path, _fake_imports, authority_kwargs
):
    env = _Env()
    with pytest.raises(RuntimeError, match="only fresh single_action_lean"):
        adapter.ActionBallFullMdpRsl3Runner(
            env,
            {},
            str(tmp_path),
            action_ball_full_mdp_runtime_owner=env.owner,
            action_ball_full_mdp_run_mode="single_action_lean",
            **authority_kwargs,
        )
    assert env.owner.events == []
    assert env.action_term.pending is None
    assert list(tmp_path.iterdir()) == []


def _install_fake_runner_base(monkeypatch, calls):
    class _Algorithm:
        def update(self):
            return None

    def _base_init(self, env, train_cfg, log_dir, device):
        calls.append((env, train_cfg, log_dir, device))
        self.env = env
        self.alg = _Algorithm()
        self.is_distributed = False
        self.num_steps_per_env = 48

    monkeypatch.setattr(
        adapter.OnPolicyRunner,
        "__init__",
        _base_init,
        raising=False,
    )


def test_runner_binds_v3_observation_and_contract_before_base_init(
    tmp_path, _fake_imports, monkeypatch
):
    env = _Env()
    calls = []
    _install_fake_runner_base(monkeypatch, calls)

    runner = adapter.ActionBallFullMdpRsl3Runner(
        env,
        {},
        str(tmp_path),
        training_contract_schema_version=3,
        training_contract_sha256="a" * 64,
        action_ball_full_mdp_runtime_owner=env.owner,
        action_ball_full_mdp_run_mode="single_action_lean",
    )

    assert calls == [(env, {}, str(tmp_path), "cpu")]
    assert runner.training_contract_schema_version == 3
    assert runner.training_contract_sha256 == "a" * 64
    assert runner._full_mdp_observation_identity == {
        "fresh_full_mdp_observation_kind": (
            "action_ball_full_mdp_semantic_observation_v3"
        ),
        "actor_obs_contract": "action_ball_full_mdp_semantic_actor_v3",
        "actor_obs_total_dim": 215,
        "critic_obs_contract": "action_ball_full_mdp_semantic_critic_v3",
        "critic_obs_total_dim": 231,
    }
    assert not hasattr(runner, "training_contract_lineage_exact")
    assert not hasattr(runner, "training_launch_claim_sha256")
    assert not hasattr(runner, "require_exact_resume_state")


@pytest.mark.parametrize(
    ("schema", "sha256", "error"),
    [
        (2, "a" * 64, "schema differs"),
        (True, "a" * 64, "schema differs"),
        (3, "A" * 64, "SHA differs"),
        (3, "a" * 63, "SHA differs"),
    ],
)
def test_runner_rejects_non_v3_contract_identity_before_base_init(
    tmp_path, _fake_imports, monkeypatch, schema, sha256, error
):
    env = _Env()
    calls = []
    _install_fake_runner_base(monkeypatch, calls)

    with pytest.raises(RuntimeError, match=error):
        adapter.ActionBallFullMdpRsl3Runner(
            env,
            {},
            str(tmp_path),
            training_contract_schema_version=schema,
            training_contract_sha256=sha256,
            action_ball_full_mdp_runtime_owner=env.owner,
            action_ball_full_mdp_run_mode="single_action_lean",
        )

    assert calls == []
    assert env.owner.events == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_obs_contract", "action_ball_full_mdp_action_epoch_v1"),
        ("actor_obs_total_dim", 229),
        ("critic_obs_contract", "action_ball_full_mdp_action_epoch_critic_v1"),
        ("critic_obs_total_dim", 399),
        (
            "fresh_full_mdp_observation_kind",
            "action_ball_full_mdp_action_epoch_observation_v1",
        ),
    ],
)
def test_runner_rejects_v1_observation_identity_before_base_init(
    tmp_path, _fake_imports, monkeypatch, field, value
):
    env = _Env()
    env.observation_facts[field] = value
    calls = []
    _install_fake_runner_base(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="semantic observation identity differs"):
        adapter.ActionBallFullMdpRsl3Runner(
            env,
            {},
            str(tmp_path),
            training_contract_schema_version=3,
            training_contract_sha256="a" * 64,
            action_ball_full_mdp_runtime_owner=env.owner,
            action_ball_full_mdp_run_mode="single_action_lean",
        )

    assert calls == []
    assert env.owner.events == []
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
    runner.training_contract_schema_version = 3
    runner.training_contract_sha256 = "a" * 64
    runner._full_mdp_observation_identity = {
        key: value
        for key, value in _v3_snapshot_identity().items()
        if key
        in {
            "fresh_full_mdp_observation_kind",
            "actor_obs_contract",
            "actor_obs_total_dim",
            "critic_obs_contract",
            "critic_obs_total_dim",
        }
    }
    requested = tmp_path / "model_1000.pt"
    with pytest.raises(RuntimeError, match="forbids caller infos"):
        runner.save(
            str(requested),
            infos={"training_contract_lineage_exact": True},
        )
    assert calls == []
    assert not requested.exists()

    def poisoned_owner():
        raise RuntimeError("owner is poisoned")

    runner._full_mdp_adapter = types.SimpleNamespace(
        assert_snapshot_boundary_clean=poisoned_owner
    )
    with pytest.raises(RuntimeError, match="owner is poisoned"):
        runner.save(str(requested))
    assert calls == []
    assert not requested.exists()

    def pending_metrics():
        raise RuntimeError("full-MDP exact metrics are pending on device")

    runner._full_mdp_adapter = types.SimpleNamespace(
        assert_snapshot_boundary_clean=pending_metrics
    )
    with pytest.raises(RuntimeError, match="metrics are pending"):
        runner.save(str(requested))
    assert calls == []
    assert not requested.exists()

    runner._full_mdp_adapter = types.SimpleNamespace(
        assert_snapshot_boundary_clean=lambda: None
    )

    runner.save(str(requested))

    assert calls == [
        (
            str(tmp_path / "model_1000.diagnostic_nonresumable.pt"),
            _v3_snapshot_identity(),
        )
    ]
    snapshot = tmp_path / "model_1000.diagnostic_nonresumable.pt"
    receipt_path = tmp_path / (
        "model_1000.diagnostic_nonresumable.pt.receipt.json"
    )
    assert json.loads(receipt_path.read_text()) == {
        "schema_version": 2,
        "kind": "action_ball_full_mdp_diagnostic_snapshot_receipt_v2",
        "snapshot_name": snapshot.name,
        "learning_iteration": 1000,
        "snapshot_size_bytes": snapshot.stat().st_size,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "model_tensor_count": 1,
        "optimizer_tensor_count": 1,
        "all_tensors_finite": True,
        **_v3_snapshot_identity(),
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
            required_infos=_v3_snapshot_identity(),
        )
    assert not snapshot.with_name(snapshot.name + ".receipt.json").exists()


def test_snapshot_receipt_rejects_v1_identity_before_read(tmp_path):
    snapshot = tmp_path / "model_7.diagnostic_nonresumable.pt"
    snapshot.write_bytes(b"old-v1-snapshot")
    old_identity = _v3_snapshot_identity(
        action_ball_full_mdp_snapshot_kind=(
            "policy_optimizer_diagnostic_nonresumable_v1"
        ),
        fresh_full_mdp_observation_kind=(
            "action_ball_full_mdp_action_epoch_observation_v1"
        ),
        actor_obs_contract="action_ball_full_mdp_action_epoch_v1",
        actor_obs_total_dim=229,
        critic_obs_contract="action_ball_full_mdp_action_epoch_critic_v1",
        critic_obs_total_dim=399,
        training_contract_schema_version=1,
    )

    with pytest.raises(RuntimeError, match="training-contract schema differs"):
        adapter._write_snapshot_receipt(
            snapshot,
            learning_iteration=7,
            required_infos=old_identity,
        )

    assert not snapshot.with_name(snapshot.name + ".receipt.json").exists()


def test_snapshot_receipt_rejects_semantic_v2_identity_before_read(tmp_path):
    snapshot = tmp_path / "model_7.diagnostic_nonresumable.pt"
    snapshot.write_bytes(b"old-semantic-v2-snapshot")
    old_identity = _v3_snapshot_identity(
        fresh_full_mdp_observation_kind=(
            "action_ball_full_mdp_semantic_observation_v2"
        ),
        actor_obs_contract="action_ball_full_mdp_semantic_actor_v2",
        actor_obs_total_dim=203,
        critic_obs_contract="action_ball_full_mdp_semantic_critic_v2",
        critic_obs_total_dim=219,
    )

    with pytest.raises(RuntimeError, match="snapshot identity differs"):
        adapter._write_snapshot_receipt(
            snapshot,
            learning_iteration=7,
            required_infos=old_identity,
        )

    assert not snapshot.with_name(snapshot.name + ".receipt.json").exists()


def test_snapshot_receipt_rejects_v1_payload_metadata(tmp_path, monkeypatch):
    snapshot = tmp_path / "model_7.diagnostic_nonresumable.pt"
    snapshot.write_bytes(b"old-v1-snapshot")

    class _Tensor:
        pass

    class _Finite:
        def all(self):
            return self

        def item(self):
            return True

    old_infos = {
        "action_ball_full_mdp_snapshot_kind": (
            "policy_optimizer_diagnostic_nonresumable_v1"
        ),
        "checkpoint_authority": False,
        "resume_authority": False,
    }
    fake_torch = types.SimpleNamespace(
        Tensor=_Tensor,
        isfinite=lambda _value: _Finite(),
        load=lambda _stream, **_kwargs: {
            "model_state_dict": {"weight": _Tensor()},
            "optimizer_state_dict": {"state": {0: {"step": _Tensor()}}},
            "iter": 7,
            "infos": old_infos,
        },
    )
    real_import_module = adapter.importlib.import_module
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import_module(name),
    )

    with pytest.raises(RuntimeError, match="snapshot metadata differs"):
        adapter._write_snapshot_receipt(
            snapshot,
            learning_iteration=7,
            required_infos=_v3_snapshot_identity(),
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


def test_adapter_rejects_foreign_same_type_command_term_before_wal(
    tmp_path, _fake_imports
):
    env = _Env()
    env.command_manager.term = RacketTargetCommand()

    with pytest.raises(
        RuntimeError,
        match=r"deferred-metric command producer: runtime_owner_racket_identity",
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )

    assert list(tmp_path.iterdir()) == []


def test_adapter_rejects_foreign_command_subclass_even_when_graph_identity_matches(
    tmp_path, _fake_imports
):
    env = _Env()
    foreign = _ForeignRacketTargetCommand()
    env.command_term = foreign
    env.command_manager.term = foreign
    env.owner._racket = foreign

    with pytest.raises(
        RuntimeError,
        match=r"deferred-metric command producer: racket_target_type=",
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )

    assert list(tmp_path.iterdir()) == []


def test_adapter_rejects_instance_shadowed_command_predicate_before_wal(
    tmp_path, _fake_imports
):
    env = _Env()
    env.command_term._action_ball_full_mdp_deferred_exact_metrics_enabled = (
        lambda: True
    )

    with pytest.raises(
        RuntimeError,
        match=r"deferred-metric command producer: deferred_exact_metrics",
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "method_name",
    (
        "materialize_action_ball_diagnostic_metrics_for_report",
        "assert_action_ball_diagnostic_metrics_materialized_for_report",
    ),
)
def test_adapter_rejects_instance_shadowed_command_metric_methods_before_wal(
    tmp_path, _fake_imports, method_name
):
    env = _Env()
    setattr(env.command_term, method_name, lambda *args, **kwargs: None)

    with pytest.raises(
        RuntimeError,
        match="FullMDP owner lacks exact bound method " + method_name,
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )

    assert list(tmp_path.iterdir()) == []


def test_adapter_rejects_instance_shadowed_owner_idle_guard_before_wal(
    tmp_path, _fake_imports
):
    env = _Env()
    env.owner.require_optimizer_boundary_idle = lambda: None

    with pytest.raises(
        RuntimeError,
        match=(
            "FullMDP owner lacks exact bound method "
            "require_optimizer_boundary_idle"
        ),
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )

    assert list(tmp_path.iterdir()) == []


def test_adapter_rejects_instance_shadowed_safety_idle_guard_before_wal(
    tmp_path, _fake_imports
):
    env = _Env()
    env.action_term.assert_joint_safety_ledger_consume_idle = lambda: None

    with pytest.raises(
        RuntimeError,
        match=(
            "FullMDP owner lacks exact bound method "
            "assert_joint_safety_ledger_consume_idle"
        ),
    ):
        adapter.ActionBallFullMdpRsl3Adapter(
            env=env, owner=env.owner, log_dir=str(tmp_path)
        )

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
