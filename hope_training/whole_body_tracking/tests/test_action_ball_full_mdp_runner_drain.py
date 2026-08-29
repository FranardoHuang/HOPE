"""Real RSL-runner callpoint tests for the fresh full-MDP PPO drain."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import errno
import builtins
import pickle
import stat
import tempfile
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from _action_ball_runner_test_harness import (
    _Env,
    _load_runner_module,
)
from test_joint_limit_safety import (
    _action_and_env,
    _finish_guarded_policy_step,
)


@pytest.fixture(scope="module")
def runner_module():
    focused_names = (
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch",
        "action_ball_full_mdp_epoch",
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_rewards",
        "action_ball_full_mdp_lean_rewards",
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_runtime",
        "_runner_drain_lean_runtime",
    )
    focused_prior = {name: sys.modules.get(name) for name in focused_names}
    module, saved = _load_runner_module()
    module.MotionOnPolicyRunner._bind_joint_safety_action_term = (
        lambda self, *, required: (
            self.env.unwrapped.action_manager.get_term("joint_pos")
            if required
            else None
        )
    )
    try:
        yield module
    finally:
        for name, previous in reversed(tuple(saved.items())):
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, previous in focused_prior.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _diagnostic_runner(
    runner_module, owner, events, *, env=None, log_dir=None
):
    """Exact lean owner harness; it is not evidence of Isaac integration."""

    env = (
        getattr(owner, "full_mdp_runtime_env", None) or _Env()
        if env is None
        else env
    )
    env._drain_owner = owner
    owned_log_dir = None
    if log_dir is None:
        owned_log_dir = tempfile.TemporaryDirectory(
            prefix="full-mdp-diagnostic-n2-no-save-test-"
        )
        log_dir = owned_log_dir.name
    runner = runner_module.MotionOnPolicyRunner(
        env,
        {"num_steps_per_env": 2, "leave_storage_full": False},
        log_dir=str(log_dir),
        training_contract_schema_version=1,
        training_contract_sha256=hashlib.sha256(
            b"diagnostic-contract"
        ).hexdigest(),
        training_contract_lineage_exact=False,
        require_exact_resume_state=True,
        action_ball_full_mdp_runtime_owner=owner,
        action_ball_full_mdp_run_mode="single_action_lean",
    )
    runner._test_owned_diagnostic_log_dir = owned_log_dir
    original_update = runner.alg.update

    def optimizer():
        events.append("optimizer")
        return original_update()

    runner.alg.update = optimizer
    runner.env.unwrapped.command_manager = SimpleNamespace(
        active_terms=("full_mdp",),
        get_term=lambda _name: SimpleNamespace(
            on_rollout_end=lambda step: events.append(("rollout_end", step))
        ),
    )
    runner_module.MotionOnPolicyRunner._notify_command_terms_rollout_end = (
        lambda self, step: self.env.unwrapped.command_manager.get_term(
            "full_mdp"
        ).on_rollout_end(step)
    )
    return runner, env


def _load_focused_module_once(
    *,
    canonical_name,
    source_path,
    standalone_name=None,
):
    """Load one source module without replacing either existing identity."""

    canonical = sys.modules.get(canonical_name)
    standalone = (
        None
        if standalone_name is None
        else sys.modules.get(standalone_name)
    )
    if canonical is not None and standalone is not None:
        if canonical is not standalone:
            # Another focused test may have loaded the same source through its
            # standalone key during collection.  Canonical identity owns the
            # production ABI; alias it only for this module-scoped fixture and
            # restore the prior namespace when the fixture exits.
            sys.modules[standalone_name] = canonical
        return canonical
    module = canonical if canonical is not None else standalone
    if module is None:
        spec = importlib.util.spec_from_file_location(
            canonical_name, source_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[canonical_name] = module
        if standalone_name is not None:
            sys.modules[standalone_name] = module
        spec.loader.exec_module(module)
    else:
        sys.modules.setdefault(canonical_name, module)
        if standalone_name is not None:
            sys.modules.setdefault(standalone_name, module)

    assert sys.modules[canonical_name] is module
    if standalone_name is not None:
        assert sys.modules[standalone_name] is module
    return module


def _load_lean_runtime_module(runner_module):
    mdp = (
        Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    source = Path(__file__).resolve().parents[1] / "source/whole_body_tracking"
    for path in (str(source), str(mdp)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Canonical-first source loading avoids the Isaac package initializer and
    # preserves any epoch/reward/runtime identities established by another
    # focused test.  Their standalone names are aliases to those same objects,
    # never second executions of the source files.
    package = "whole_body_tracking.tasks.tracking.mdp"
    import action_ball_full_mdp_selected_reset  # noqa: F401
    import action_ball_full_mdp_drain_summary  # noqa: F401
    import action_ball_full_mdp_action_strata  # noqa: F401
    epoch = _load_focused_module_once(
        canonical_name=f"{package}.action_ball_full_mdp_epoch",
        standalone_name="action_ball_full_mdp_epoch",
        source_path=mdp / "action_ball_full_mdp_epoch.py",
    )
    _load_focused_module_once(
        canonical_name=f"{package}.action_ball_full_mdp_lean_rewards",
        standalone_name="action_ball_full_mdp_lean_rewards",
        source_path=mdp / "action_ball_full_mdp_lean_rewards.py",
    )

    wal_name = (
        "whole_body_tracking.utils.action_ball_full_mdp_durable_wal"
    )
    _load_focused_module_once(
        canonical_name=wal_name,
        source_path=(
            mdp.parents[2]
            / "utils/action_ball_full_mdp_durable_wal.py"
        ),
    )
    lean = _load_focused_module_once(
        canonical_name=f"{package}.action_ball_full_mdp_lean_runtime",
        standalone_name="_runner_drain_lean_runtime",
        source_path=mdp / "action_ball_full_mdp_lean_runtime.py",
    )
    runner_module._action_ball_full_mdp_lean_runtime_module = lambda: lean
    return lean, epoch


def test_lean_loader_preserves_preinstalled_canonical_reward_identity(
    runner_module,
):
    canonical_name = (
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_rewards"
    )
    _load_lean_runtime_module(runner_module)
    before = sys.modules[canonical_name]
    _load_lean_runtime_module(runner_module)
    assert sys.modules[canonical_name] is before
    assert sys.modules["action_ball_full_mdp_lean_rewards"] is before


def test_lean_loader_is_module_identity_idempotent(runner_module):
    first_lean, first_epoch = _load_lean_runtime_module(runner_module)
    first_rewards = sys.modules[
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_rewards"
    ]
    second_lean, second_epoch = _load_lean_runtime_module(runner_module)
    assert second_lean is first_lean
    assert second_epoch is first_epoch
    assert sys.modules[
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_rewards"
    ] is first_rewards
    assert sys.modules["action_ball_full_mdp_lean_rewards"] is first_rewards


def test_lean_loader_preserves_reward_manager_callable_pickle_identity(
    runner_module,
):
    _load_lean_runtime_module(runner_module)
    rewards = sys.modules[
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_rewards"
    ]
    function = rewards.racket_position
    assert pickle.loads(pickle.dumps(function)) is function


def _lean_owner(
    runner_module,
    events,
    *,
    device="cpu",
    num_envs=2,
    shot_slot_capacity=1,
):
    lean, epoch = _load_lean_runtime_module(runner_module)
    env = _Env(obs_mode="action_ball_full_mdp", target_mode="action_ball_full_mdp")
    env.device = device
    env.num_envs = num_envs
    epoch_owner = epoch.ActionEpochOwner(
        num_envs=num_envs,
        device=device,
        shot_slot_capacity=shot_slot_capacity,
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(num_envs, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(
            num_envs, dtype=torch.int64, device=device
        ),
    )
    rewards = sys.modules["action_ball_full_mdp_lean_rewards"]
    graph = rewards.LeanActionEpochRewardGraph(epoch_owner=epoch_owner)
    owner = lean.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=env.action_ball_full_mdp_runtime_lease,
        epoch_owner=epoch_owner,
        reward_graph=graph,
        r05_runtime=object(),
        motion=object(),
        racket=object(),
        physical_ball=object(),
        r06_landing_outcome=object(),
        r03_strike_fact=object(),
        r07_recovery=object(),
    )
    env._drain_owner = owner
    env._action_ball_full_mdp_lean_reward_graph = graph
    action, _action_env, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.1,
        runtime_step_dt=0.1,
        target_mode="action_ball_full_mdp",
        action_ball_diagnostic_unauthorized=True,
        diagnostic_compact_evidence=True,
    )
    env.cfg.commands.racket_target.action_ball_diagnostic_unauthorized = True
    env.action_manager = SimpleNamespace(
        get_term=lambda name: action if name == "joint_pos" else None
    )
    original_step = env.step

    def step_with_joint_safety(actions):
        action.process_actions(actions.detach().to(device="cpu"))
        _finish_guarded_policy_step(action, asset)
        return original_step(actions)

    env.step = step_with_joint_safety
    env._test_joint_safety_action = action
    env._test_joint_safety_asset = asset
    del events
    return owner, env, lean, epoch


def _completed_shot(lean, **changes):
    lifecycle_bits = lean.drain_v2.SHOT_LIFECYCLE_BITS
    shot = lean.drain_v2.CompletedActionEpochShot(
        env_row=1,
        slot_index=0,
        reset_generation=3,
        ball_generation=4,
        action_uid=5,
        action_slot=0,
        shot_index=6,
        task_identity=7,
        outcome_identity=8,
        ball_identity=9,
        target_x_m=0.321,
        target_y_m=-0.123,
        motion_close_reason=1,
        settlement_step=41,
        payment_step=42,
        retirement_step=43,
        stroke_family="backhand",
        action_attribution_valid=True,
        evidence=lean.drain_v2.ActionEpochShotEvidence(
            lifecycle_bits=sum(
                lifecycle_bits[name]
                for name in (
                    "reveal_committed", "playback_started", "motion_closed",
                    "physical_launched", "outcome_settled", "payment_recorded",
                )
            ),
            r03_valid_bits=0,
            r03_source_step=-1,
            physical_valid_bits=0,
            physical_actor_pair_contact_source_step=-1,
            r06_valid_bits=7,
            r06_outcome_code=1,
            r06_predicate_bits=0,
            r07_valid_bits=0,
            r07_qualified_source_step=-1,
            r07_first_ready_source_step=-1,
        ),
    )
    return replace(shot, **changes)


def _action_opportunity(lean, **changes):
    row = lean.drain_v2.D05ActionOpportunity(
        env_row=1, slot_index=0, action_uid=5, action_slot=0,
        stroke_family="backhand", attribution_valid=True, selected=True,
        accepted=True, censored=False, rejected=False, deferred=False,
    )
    return replace(row, **changes)


def _terminal_shot(lean, **changes):
    shot = lean.drain_v2.TerminalActionEpochShot(
        env_row=1, slot_index=0, reset_generation=3, ball_generation=4,
        action_uid=5, action_slot=0, shot_index=6, task_identity=7,
        outcome_identity=8, ball_identity=9, target_x_m=0.0,
        target_y_m=0.0, motion_close_reason=-1, settlement_step=-1,
        payment_step=-1, stroke_family="backhand",
        action_attribution_valid=True,
        evidence=lean.drain_v2.ActionEpochShotEvidence(
            lifecycle_bits=lean.drain_v2.SHOT_LIFECYCLE_BITS["reveal_committed"],
            r03_valid_bits=0, r03_source_step=-1, physical_valid_bits=0,
            physical_actor_pair_contact_source_step=-1, r06_valid_bits=-1,
            r06_outcome_code=-1, r06_predicate_bits=0, r07_valid_bits=0,
            r07_qualified_source_step=-1, r07_first_ready_source_step=-1,
        ),
        reset_generation_after=4, reset_common_step=50,
        reset_episode_tick=11, reset_reason_bits=2,
    )
    return replace(shot, **changes)


def _terminal_reset(lean, **changes):
    reset = lean.drain_v2.ResetTelemetry(
        env_row=1,
        reset_generation=3,
        common_step=40,
        episode_tick=7,
        reason_bits=1 | 4,
    )
    return replace(reset, **changes)


def _prepare_pending_epoch_summary(runner_module, owner, runner):
    runner._action_ball_full_mdp_preflight_durable_wal()
    boundary = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=4
    )
    owner.mark_optimizer_returned(boundary, update_index=0)
    summary = owner.prepare_post_update_summary(boundary, update_index=0)
    active = runner_module._ActionBallFullMdpRunnerDrainChronology(
        boundary, 0, 4, True, False, 0, 0, None, False
    )
    runner._action_ball_full_mdp_active_drain_chronology = active
    return boundary, summary, active


def _durable_wal_path(runner):
    return (
        Path(runner.log_dir)
        / "action_ball_epoch_durable_wal"
        / f"rank_{runner._joint_safety_rank():04d}.jsonl"
    )


def _durable_wal_rows(runner):
    path = _durable_wal_path(runner)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _durable_wal_scan(runner):
    module = sys.modules[
        "whole_body_tracking.utils.action_ball_full_mdp_durable_wal"
    ]
    return module.read_forensic_committed_frontier(
        _durable_wal_path(runner),
        expected_rank=runner._joint_safety_rank(),
        expected_segment_id=(
            runner._action_ball_full_mdp_durable_wal_segment_id
        ),
    )


@pytest.mark.parametrize("num_envs", (1, 2, 64))
def test_single_action_lean_runner_accepts_positive_n(
    runner_module, num_envs
):
    owner, env, _lean, _epoch = _lean_owner(
        runner_module, [], num_envs=num_envs
    )
    runner, resolved = _diagnostic_runner(
        runner_module, owner, [], env=env
    )
    assert resolved.num_envs == num_envs
    assert runner._action_ball_full_mdp_run_mode == "single_action_lean"


def test_single_action_lean_forbids_invalid_n_r10_or_capsule(
    runner_module,
):
    owner, env, _lean, _epoch = _lean_owner(runner_module, [])
    env.num_envs = True
    with pytest.raises(RuntimeError, match="positive exact-int"):
        runner_module.MotionOnPolicyRunner(
            env,
            {"num_steps_per_env": 2},
            action_ball_full_mdp_runtime_owner=owner,
            action_ball_full_mdp_run_mode="single_action_lean",
        )
    env.num_envs = 2
    with pytest.raises(RuntimeError, match="forbids the R10"):
        runner_module.MotionOnPolicyRunner(
            env,
            {"num_steps_per_env": 2},
            action_ball_full_mdp_runtime_owner=owner,
            action_ball_full_mdp_run_mode="single_action_lean",
            action_ball_r10_checkpoint_adapter=object(),
        )
    with pytest.raises(RuntimeError, match="forbids a cold-restore"):
        runner_module.MotionOnPolicyRunner(
            env,
            {"num_steps_per_env": 2},
            action_ball_full_mdp_runtime_owner=owner,
            action_ball_full_mdp_run_mode="single_action_lean",
            action_ball_r10_cold_restore_capsule=object(),
        )


def test_ordinary_runner_without_full_mdp_owner_never_enters_r10(
    runner_module, tmp_path
):
    env = _Env()
    env.command_manager = SimpleNamespace(
        get_term=lambda _name: SimpleNamespace(on_rollout_end=lambda _step: None)
    )
    runner = runner_module.MotionOnPolicyRunner(
        env,
        {"num_steps_per_env": 2, "leave_storage_full": False},
        log_dir=str(tmp_path),
        training_contract_schema_version=1,
        training_contract_sha256=hashlib.sha256(
            b"ordinary-no-r10-contract"
        ).hexdigest(),
        training_contract_lineage_exact=True,
    )
    runner.learn(1)
    assert env.getter_calls == 2
    assert env.reset_calls == 0
    assert env.step_calls == 2
    assert not hasattr(runner, "_action_ball_r10_last_boundary_handoff") or (
        runner._action_ball_r10_last_boundary_handoff is None
    )


def test_diagnostic_ack_returns_one_typed_summary_and_one_marker(
    runner_module, capsys
):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    assert not hasattr(owner, "require_owned_runner_frontier_projection")
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    runner.learn(1)
    lines = capsys.readouterr().out.splitlines()
    markers = [
        line.split("=", 1)[1]
        for line in lines
        if line.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=")
    ]
    assert len(markers) == 1
    payload = json.loads(markers[0])
    assert payload["ppo_update"] == 0
    assert payload["d05_transactions"] == 0
    assert payload["d05_selected_rows"] == 0
    chronology = runner._action_ball_full_mdp_last_drain_chronology
    assert type(chronology.projection) is lean.ActionEpochPpoBoundarySummary
    assert chronology.projection.frontier.start_commit == 0
    assert chronology.projection.frontier.end_commit == 1
    assert owner._operation_sequence == owner._drain_sequence == 1
    safety = next(
        json.loads(line.split("=", 1)[1])
        for line in lines
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    )
    localization = safety["actual_hard_edge_localization"]
    assert localization["kind"] == (
        "joint_safety_actual_hard_edge_localization_v1"
    )
    assert localization["schema_version"] == 1
    assert localization["joint_order"] == ["j0", "j1"]
    assert localization["side_order"] == ["lower", "upper"]
    assert localization["rows"] == []


def test_diagnostic_wal_is_durable_before_destructive_ack_and_stdout(
    runner_module, tmp_path, capsys
):
    events = []
    owner, _env, _lean, _epoch = _lean_owner(runner_module, events)
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, log_dir=tmp_path
    )
    runner.rank = 3

    original_validate = runner._action_ball_full_mdp_prepare_epoch_ack_record
    original_encode = runner._action_ball_full_mdp_encode_durable_pending_ack
    original_append = runner._action_ball_full_mdp_append_durable_wal
    original_ack = runner._action_ball_full_mdp_ack_callback
    original_consume = runner._action_ball_full_mdp_consume_epoch_ack_summary

    def validate(*args, **kwargs):
        events.append("validate")
        return original_validate(*args, **kwargs)

    def encode(*args, **kwargs):
        events.append("encode")
        return original_encode(*args, **kwargs)

    def append(*args, **kwargs):
        result = original_append(*args, **kwargs)
        events.append(
            "pending_fsynced" if len(events) < 5 else "epoch_ack_fsynced"
        )
        return result

    def acknowledge(*args, **kwargs):
        events.append("destructive_ack")
        assert events[-2] == "pending_fsynced"
        return original_ack(*args, **kwargs)

    def consume(*args, **kwargs):
        events.append("chronology_and_stdout")
        assert events[-2] == "epoch_ack_fsynced"
        return original_consume(*args, **kwargs)

    runner._action_ball_full_mdp_prepare_epoch_ack_record = validate
    runner._action_ball_full_mdp_encode_durable_pending_ack = encode
    runner._action_ball_full_mdp_append_durable_wal = append
    runner._action_ball_full_mdp_ack_callback = acknowledge
    runner._action_ball_full_mdp_consume_epoch_ack_summary = consume
    runner.learn(1)

    assert events == [
        "optimizer",
        "validate",
        "encode",
        "pending_fsynced",
        "destructive_ack",
        "epoch_ack_fsynced",
        "chronology_and_stdout",
        ("rollout_end", 0),
    ]
    rows = _durable_wal_rows(runner)
    assert len(rows) == 2
    assert rows[0]["status"] == (
        "optimizer_succeeded_durable_pending_destructive_ack"
    )
    assert rows[0]["rank"] == 3
    assert rows[0]["ppo_update"] == 0
    stdout_payload = next(
        json.loads(line.split("=", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=")
    )
    assert rows[0]["pending_ack_telemetry"] == stdout_payload
    assert rows[1]["kind"] == "action_ball_epoch_durable_ack_v2"
    assert rows[1]["status"] == "destructive_epoch_ack_durable"
    assert rows[1]["record_key"] != rows[0]["record_key"]
    assert rows[1]["pending_record_key"] == rows[0]["record_key"]
    assert rows[1]["pending_byte_start"] == 0
    assert rows[1]["pending_byte_end"] > 0
    scan = _durable_wal_scan(runner)
    assert scan["durable_epoch_ack_count"] == 1
    assert scan["pending_without_epoch_ack"] is False
    assert scan["committed_frontier"]["ppo_update"] == 0


def test_diagnostic_first_wal_install_fsyncs_file_leaf_and_parent_in_order(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    runner.rank = 12
    parent_stat = os.stat(tmp_path)
    calls = []
    original_fsync = runner_module.os.fsync

    def traced_fsync(fd):
        value = os.fstat(fd)
        if stat.S_ISREG(value.st_mode):
            calls.append("file")
        elif (value.st_dev, value.st_ino) == (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            calls.append("parent_log_dir")
        else:
            calls.append("leaf_directory")
        return original_fsync(fd)

    monkeypatch.setattr(runner_module.os, "fsync", traced_fsync)
    runner._action_ball_full_mdp_preflight_durable_wal()
    assert calls == ["file", "leaf_directory", "parent_log_dir"]


def test_diagnostic_wal_rejects_symlink_log_dir_before_optimizer(
    runner_module, tmp_path
):
    real_log_dir = tmp_path / "real"
    real_log_dir.mkdir()
    linked_log_dir = tmp_path / "linked"
    linked_log_dir.symlink_to(real_log_dir, target_is_directory=True)
    events = []
    owner, _env, _lean, _epoch = _lean_owner(runner_module, events)
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, log_dir=linked_log_dir
    )

    with pytest.raises(RuntimeError, match="pre-optimizer drain failed"):
        runner.learn(1)
    assert events.count("optimizer") == 0
    assert owner.poisoned is True


@pytest.mark.parametrize(
    "failure",
    ("validation", "encode", "write", "short_write", "flush", "fsync", "close"),
)
def test_diagnostic_durable_wal_failures_keep_epoch_pending_and_poison(
    runner_module, tmp_path, monkeypatch, capsys, failure
):
    events = []
    owner, _env, _lean, epoch = _lean_owner(runner_module, events)
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, log_dir=tmp_path
    )
    runner.rank = 4
    runner._action_ball_full_mdp_preflight_durable_wal()

    if failure == "validation":
        runner._action_ball_full_mdp_prepare_epoch_ack_record = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected validation failure")
            )
        )
    elif failure == "encode":
        runner._action_ball_full_mdp_encode_durable_pending_ack = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("injected encode failure")
            )
        )
    elif failure == "fsync":
        monkeypatch.setattr(
            runner_module.os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(
                OSError(errno.ENOSPC, "injected fsync failure")
            ),
        )
    else:
        original_open = builtins.open

        class FailingWalHandle:
            def __init__(self, path, mode, buffering):
                self._handle = original_open(path, mode, buffering=buffering)

            def fileno(self):
                return self._handle.fileno()

            def write(self, value):
                if failure == "write":
                    raise OSError(errno.EIO, "injected write failure")
                if failure == "short_write":
                    return self._handle.write(value[:-1])
                return self._handle.write(value)

            def flush(self):
                if failure == "flush":
                    raise OSError(errno.ENOSPC, "injected flush failure")
                return self._handle.flush()

            def close(self):
                self._handle.close()
                if failure == "close":
                    raise OSError(errno.EIO, "injected close failure")

        monkeypatch.setattr(
            runner_module,
            "open",
            lambda path, mode, buffering=0: FailingWalHandle(
                path, mode, buffering
            ),
            raising=False,
        )

    with pytest.raises((OSError, RuntimeError, ValueError)):
        runner.learn(1)
    assert events.count("optimizer") == 1
    assert owner.poisoned is True
    assert runner._action_ball_full_mdp_boundary_poisoned is True
    assert owner._active_post_update_summary is not None
    assert owner.epoch_owner.drain_frontier == 0
    assert runner._action_ball_full_mdp_boundary_active is True
    with pytest.raises(
        RuntimeError, match="checkpoint/save|checkpoint is forbidden"
    ):
        runner._action_ball_full_mdp_require_checkpointable()
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.learn(1)
    assert events.count("optimizer") == 1
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out


def test_diagnostic_wal_primary_error_is_not_masked_by_close_failure(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    runner._action_ball_full_mdp_preflight_durable_wal()
    original_open = builtins.open

    class DoubleFailHandle:
        def __init__(self, path, mode, buffering):
            self._handle = original_open(path, mode, buffering=buffering)

        def fileno(self):
            return self._handle.fileno()

        def write(self, _value):
            raise OSError(errno.EIO, "primary-write")

        def close(self):
            self._handle.close()
            raise OSError(errno.ENOSPC, "secondary-close")

    monkeypatch.setattr(
        runner_module,
        "open",
        lambda path, mode, buffering=0: DoubleFailHandle(
            path, mode, buffering
        ),
        raising=False,
    )
    with pytest.raises(OSError, match="primary-write"):
        runner.learn(1)
    assert owner.poisoned is True
    assert owner._active_post_update_summary is not None


def _install_failing_cleanup_profiler(monkeypatch):
    from types import ModuleType

    module_name = "whole_body_tracking.utils.action_ball_update_profiler"
    profiler_module = ModuleType(module_name)
    monkeypatch.setitem(
        sys.modules,
        module_name,
        profiler_module,
    )
    monkeypatch.setenv("HOPE_ACTION_BALL_UPDATE_PROFILE", "1")
    monkeypatch.setattr(
        profiler_module,
        "parse_action_ball_update_profile_request",
        lambda _environment: True,
        raising=False,
    )

    class FailingCleanupProfiler:
        def __init__(self):
            self.close_calls = 0

        def emit_update(self, **_kwargs):
            return None

        def close(self):
            self.close_calls += 1
            raise OSError(errno.ENOSPC, "injected profiler close failure")

    profiler = FailingCleanupProfiler()
    monkeypatch.setattr(
        profiler_module,
        "install_diagnostic_action_ball_update_profiler",
        lambda *_args, **_kwargs: profiler,
        raising=False,
    )
    return profiler


def test_diagnostic_wal_eio_remains_primary_when_profiler_close_fails(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    runner._action_ball_full_mdp_preflight_durable_wal()
    runner._action_ball_diagnostic_unauthorized = lambda: True
    runner.disable_logs = False
    runner.rank = 0
    profiler = _install_failing_cleanup_profiler(monkeypatch)
    original_open = builtins.open

    class WriteFailHandle:
        def __init__(self, path, mode, buffering):
            self._handle = original_open(path, mode, buffering=buffering)

        def fileno(self):
            return self._handle.fileno()

        def write(self, _value):
            raise OSError(errno.EIO, "primary WAL write EIO")

        def close(self):
            self._handle.close()

    monkeypatch.setattr(
        runner_module,
        "open",
        lambda path, mode, buffering=0: WriteFailHandle(
            path, mode, buffering
        ),
        raising=False,
    )
    with pytest.raises(OSError, match="primary WAL write EIO"):
        runner.learn(1)
    assert profiler.close_calls == 1
    assert runner._learn_cleanup_secondary_failures == (
        (
            "close_action_ball_update_profiler",
            "builtins.OSError: [Errno 28] injected profiler close failure",
        ),
    )
    assert owner.poisoned is True
    assert owner._active_post_update_summary is not None


def test_diagnostic_cleanup_raises_first_failure_when_learn_has_no_primary(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    runner._action_ball_diagnostic_unauthorized = lambda: True
    runner.disable_logs = False
    runner.rank = 0
    profiler = _install_failing_cleanup_profiler(monkeypatch)
    with pytest.raises(OSError, match="injected profiler close failure"):
        runner.learn(1)
    assert profiler.close_calls == 1
    assert runner._learn_cleanup_secondary_failures[0][0] == (
        "close_action_ball_update_profiler"
    )


def test_full_mdp_compact_joint_safety_predicate_does_not_widen_auxiliaries(
    runner_module,
):
    owner, env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [], env=env)

    assert runner._full_mdp_diagnostic_joint_safety_compact_evidence() is True
    assert runner._diagnostic_joint_safety_compact_update_evidence() is True
    assert runner._diagnostic_joint_safety_compact_evidence() is False
    assert runner._consume_actual_joint_forbidden_diagnostic(0) is None
    assert runner._consume_push_velocity_diagnostic_update(0) is None

def test_diagnostic_full_mdp_joint_safety_drains_once_per_update_and_keeps_actual_edge(
    runner_module, capsys
):
    owner, env, _lean, _epoch = _lean_owner(runner_module, [])
    action = env._test_joint_safety_action
    asset = env._test_joint_safety_asset
    calls = []
    original_prepare = action.prepare_joint_safety_ledger_consume
    original_ack = action.acknowledge_joint_safety_ledger

    def prepare():
        calls.append("prepare")
        return original_prepare()

    def acknowledge(token):
        calls.append("ack")
        return original_ack(token)

    action.prepare_joint_safety_ledger_consume = prepare
    action.acknowledge_joint_safety_ledger = acknowledge
    asset.data.joint_pos[0, 0] = 1.21
    runner, _ = _diagnostic_runner(runner_module, owner, [], env=env)

    runner.learn(2)

    assert calls == ["prepare", "ack", "prepare", "ack"]
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["since_last_consume"]["has_data"] is False
    assert snapshot["policy_step_summary_used"] == 0
    assert snapshot["terminal_archive_used"] == 0
    lines = capsys.readouterr().out.splitlines()
    safety = [
        json.loads(line.split("=", 1)[1])
        for line in lines
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    epoch = [
        json.loads(line.split("=", 1)[1])
        for line in lines
        if line.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=")
    ]
    assert [record["ppo_update"] for record in safety] == [0, 1]
    assert [record["ppo_update"] for record in epoch] == [0, 1]
    assert all(
        record["counter_totals"]["actual_hard_edge_events"] > 0
        for record in safety
    )
    for record in safety:
        assert record["schema_version"] == 1
        assert record["formal_authority"] is False
        localization = record["actual_hard_edge_localization"]
        assert localization["kind"] == (
            "joint_safety_actual_hard_edge_localization_v1"
        )
        assert localization["joint_order"] == ["j0", "j1"]
        assert localization["side_order"] == ["lower", "upper"]
        assert localization["rows"] == [
            {
                "env_row": 0,
                "joint_index": 0,
                "joint_name": "j0",
                "side": "upper",
                "joint_readback_count_either_side": 10,
                "minimum_signed_hard_gap_rad": pytest.approx(-0.01),
            }
        ]
        assert record["minimum_hard_gap_rad"] == pytest.approx(-0.01)
    assert all(
        record["terminal_reset_reason_joint_qdes_forbidden_count"] == 0
        for record in epoch
    )
    assert not any(
        line.startswith("HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON=")
        or line.startswith("HOPE_PUSH_VELOCITY_DIAGNOSTIC_UPDATE_JSON=")
        for line in lines
    )


@pytest.mark.parametrize("malformed", ["joint_order", "count_gap"])
def test_diagnostic_full_mdp_joint_safety_localization_fails_before_optimizer(
    runner_module, malformed
):
    events = []
    owner, env, _lean, _epoch = _lean_owner(runner_module, events)
    action = env._test_joint_safety_action
    if malformed == "joint_order":
        action._joint_names = ["j0", "j0"]
        expected = "joint-name order"
    else:
        original_prepare = action.prepare_joint_safety_ledger_consume

        def malformed_prepare():
            token, snapshot = original_prepare()
            snapshot = dict(snapshot)
            since = dict(snapshot["since_last_consume"])
            counts = since["actual_hard_edge_joint_count"].clone()
            latch = since["actual_hard_edge_latch"].clone()
            counts[0, 0] = 1
            latch[0] = True
            since["actual_hard_edge_joint_count"] = counts
            since["actual_hard_edge_latch"] = latch
            snapshot["since_last_consume"] = since
            return token, snapshot

        action.prepare_joint_safety_ledger_consume = malformed_prepare
        expected = "actual_hard_edge_count_gap_equivalence"
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, env=env
    )

    with pytest.raises(RuntimeError, match=expected):
        runner.learn(1)

    assert "optimizer" not in events
    assert runner._joint_safety_pending_prepared is not None
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"][
            "has_data"
        ]
        is True
    )
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.process_actions(torch.zeros(2, 2))


@pytest.mark.parametrize("failure", ["optimizer", "joint_safety_ack"])
def test_diagnostic_full_mdp_joint_safety_failure_never_clears_frozen_evidence(
    runner_module, capsys, failure
):
    owner, env, _lean, _epoch = _lean_owner(runner_module, [])
    action = env._test_joint_safety_action
    env._test_joint_safety_asset.data.joint_pos[0, 0] = 1.21
    runner, _ = _diagnostic_runner(runner_module, owner, [], env=env)

    if failure == "optimizer":
        runner.alg.update = lambda: (_ for _ in ()).throw(
            RuntimeError("injected optimizer failure")
        )
    else:
        action.acknowledge_joint_safety_ledger = lambda _token: (
            (_ for _ in ()).throw(RuntimeError("injected safety ACK failure"))
        )

    with pytest.raises(RuntimeError, match="injected"):
        runner.learn(1)

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["since_last_consume"]["has_data"] is True
    assert runner._joint_safety_pending_prepared is not None
    rows = runner._joint_safety_pending_prepared["record"][
        "actual_hard_edge_localization"
    ]["rows"]
    assert rows[0]["env_row"] == 0
    assert rows[0]["joint_name"] == "j0"
    assert rows[0]["side"] == "upper"
    assert rows[0]["joint_readback_count_either_side"] == 10
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.process_actions(torch.zeros(2, 2))
    assert owner.poisoned is True
    assert "HOPE_JOINT_SAFETY_UPDATE_JSON=" not in capsys.readouterr().out


def test_diagnostic_marker_flattens_typed_completed_shots(
    runner_module, capsys
):
    from types import MethodType

    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    original_prepare = owner.prepare_post_update_summary
    shot = _completed_shot(lean)

    def prepare_with_completed(self, boundary, *, update_index):
        assert self is owner
        summary = original_prepare(boundary, update_index=update_index)
        completed = replace(
            summary,
            lifecycle=replace(summary.lifecycle, retired_rows=1),
            completed_shots=(shot,),
        )
        self._active_post_update_summary = completed
        return completed

    owner.prepare_post_update_summary = MethodType(
        prepare_with_completed, owner
    )
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    runner.learn(1)
    payload = next(
        json.loads(line.split("=", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=")
    )
    assert payload["schema_version"] == 10
    assert payload["milestone"]["schema_version"] == 8
    reward_terms = payload["milestone"]["reward_terms"]
    rewards = sys.modules[
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_rewards"
    ]
    assert tuple(row["term"] for row in reward_terms) == rewards.MANAGER_NAMES
    playback = payload["milestone"]["paddle_motion_prior_playback"]
    assert tuple(row["term"] for row in playback["terms"]) == (
        rewards.PADDLE_MOTION_PRIOR_NAMES
    )
    assert all(row["playback_count"] == 0 for row in playback["terms"])
    assert len(payload["completed_shots"]) == 1
    completed = payload["completed_shots"][0]
    assert completed["action_uid"] == shot.action_uid
    assert completed["target_x_m"] == pytest.approx(shot.target_x_m)
    assert completed["retirement_step"] == shot.retirement_step
    assert completed["evidence"]["r06_valid_bits"] == 7
    assert completed["evidence"]["contact_face"] == {
        "availability": "not_produced"
    }
    assert "receipt" not in payload


def test_diagnostic_marker_preserves_prelaunch_terminal_partial_without_false_zero(
    runner_module,
):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    _boundary, summary, _active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )
    shot = _terminal_shot(lean)
    reset = lean.drain_v2.ResetTelemetry(1, 4, 50, 11, 2)
    typed = replace(
        summary,
        lifecycle=replace(summary.lifecycle, terminal_shot_rows=1),
        terminal_shots=(shot,),
        terminal_resets=(reset,),
    )
    record = runner._action_ball_full_mdp_prepare_epoch_ack_record(
        typed, update_index=0, completed_environment_steps=4
    )
    row = record["terminal_shots"][0]
    assert row["target_x_m"] is None and row["target_y_m"] is None
    assert row["evidence"]["contact_face"] == {"availability": "not_produced"}
    assert row["evidence"]["recovery_horizon"] == {"availability": "not_produced"}
    with pytest.raises(RuntimeError, match="terminal-shot count"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            replace(typed, terminal_shots=()),
            update_index=0, completed_environment_steps=4,
        )
    with pytest.raises(RuntimeError, match="closure identity"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            replace(
                typed,
                lifecycle=replace(typed.lifecycle, retired_rows=1),
                completed_shots=(_completed_shot(lean),),
            ),
            update_index=0, completed_environment_steps=4,
        )


def test_diagnostic_marker_serializes_typed_action_opportunities(runner_module):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    _boundary, summary, _active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )
    row = _action_opportunity(lean)
    settlement = replace(
        summary.settlement, due_rows=1, selected_rows=1, accepted=1
    )
    typed = replace(
        summary, settlement=settlement, action_opportunities=(row,)
    )
    record = runner._action_ball_full_mdp_prepare_epoch_ack_record(
        typed,
        update_index=0, completed_environment_steps=4,
    )
    assert record["action_opportunities"] == [
        {name: getattr(row, name) for name in row.__dataclass_fields__}
    ]
    with pytest.raises(RuntimeError, match="action-opportunity telemetry values"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            replace(typed, action_opportunities=(
                _action_opportunity(
                    lean, stroke_family="unknown", attribution_valid=True
                ),
            )),
            update_index=0, completed_environment_steps=4,
        )
    class _StrSubclass(str):
        pass

    with pytest.raises(RuntimeError, match="action-opportunity telemetry values"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            replace(typed, action_opportunities=(
                _action_opportunity(lean, stroke_family=_StrSubclass("backhand")),
            )),
            update_index=0, completed_environment_steps=4,
        )
    with pytest.raises(RuntimeError, match="opportunity settlement differs"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            replace(summary, action_opportunities=(row,)),
            update_index=0, completed_environment_steps=4,
        )
    for field in (
        "due_rows", "selected_rows", "accepted", "censored", "rejected", "deferred"
    ):
        with pytest.raises(RuntimeError, match="opportunity settlement differs"):
            runner._action_ball_full_mdp_prepare_epoch_ack_record(
                replace(typed, settlement=replace(
                    settlement, **{field: 1 - getattr(settlement, field)}
                )),
                update_index=0, completed_environment_steps=4,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", 8), ("kind", "action_ball_epoch_ppo_boundary_summary_v8")),
)
def test_epoch_ack_rejects_drain_schema_before_wal_or_ack(
    runner_module, field, value
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    _boundary, summary, active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )
    wal_before = _durable_wal_path(runner).read_bytes()
    mutant = replace(
        summary, frontier=replace(summary.frontier, **{field: value})
    )
    with pytest.raises(RuntimeError, match="frontier differs"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            mutant, update_index=0, completed_environment_steps=4,
        )
    assert _durable_wal_path(runner).read_bytes() == wal_before
    assert active.post_update_acknowledged is False


def test_epoch_ack_rejects_nested_count_types_and_completed_retirement(
    runner_module
):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    _boundary, summary, _active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )
    for field in (
        "due_rows", "selected_rows", "accepted", "censored", "rejected", "deferred"
    ):
        for invalid in (True, -1):
            with pytest.raises(RuntimeError, match="opportunity settlement differs"):
                runner._action_ball_full_mdp_prepare_epoch_ack_record(
                    replace(summary, settlement=replace(
                        summary.settlement, **{field: invalid}
                    )),
                    update_index=0, completed_environment_steps=4,
                )
    for field, invalid in (
        ("transactions", True), ("transactions", -1), ("not_ready", -7),
        ("not_ready", 1),
    ):
        with pytest.raises(RuntimeError, match="opportunity settlement differs"):
            runner._action_ball_full_mdp_prepare_epoch_ack_record(
                replace(summary, settlement=replace(
                    summary.settlement, **{field: invalid}
                )),
                update_index=0, completed_environment_steps=4,
            )
    for invalid in (True, -1):
        with pytest.raises(RuntimeError, match="completed-shot retirement differs"):
            runner._action_ball_full_mdp_prepare_epoch_ack_record(
                replace(summary, lifecycle=replace(
                    summary.lifecycle, retired_rows=invalid
                )),
                update_index=0, completed_environment_steps=4,
            )
    with pytest.raises(RuntimeError, match="completed-shot retirement differs"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            replace(summary, completed_shots=(_completed_shot(lean),)),
            update_index=0, completed_environment_steps=4,
        )
    strata = sys.modules["action_ball_full_mdp_action_strata"]
    with pytest.raises(ValueError, match="catalog differs"):
        strata.ActionStrokeFamilyCatalog((1,), (True,))


def test_diagnostic_marker_preserves_each_terminal_reset_and_derives_counts(
    runner_module, capsys
):
    from types import MethodType

    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    original_prepare = owner.prepare_post_update_summary
    resets = (
        _terminal_reset(lean),
        _terminal_reset(
            lean,
            reset_generation=4,
            common_step=41,
            episode_tick=1,
            reason_bits=4 | 16,
        ),
    )

    def prepare_with_resets(self, boundary, *, update_index):
        assert self is owner
        summary = original_prepare(boundary, update_index=update_index)
        completed = replace(summary, terminal_resets=resets)
        self._active_post_update_summary = completed
        return completed

    owner.prepare_post_update_summary = MethodType(
        prepare_with_resets, owner
    )
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    runner.learn(1)
    payload = next(
        json.loads(line.split("=", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=")
    )
    assert payload["terminal_resets"] == [
        {name: getattr(reset, name) for name in reset.__dataclass_fields__}
        for reset in resets
    ]
    assert payload["terminal_reset_rows"] == 2
    assert payload["terminal_reset_reason_time_out_count"] == 1
    assert payload["terminal_reset_reason_base_fell_tilt_count"] == 0
    assert payload["terminal_reset_reason_base_too_low_count"] == 2
    assert payload["terminal_reset_reason_joint_qdes_forbidden_count"] == 0
    assert payload["terminal_reset_reason_robot_hit_table_count"] == 1


@pytest.mark.parametrize(
    "changes",
    (
        {"env_row": True},
        {"common_step": 0},
        {"episode_tick": 0},
        {"reason_bits": 0},
        {"reason_bits": 32},
    ),
)
def test_diagnostic_terminal_reset_abi_fails_before_telemetry_mutation(
    runner_module, capsys, changes
):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    boundary, summary, active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )
    malformed = replace(
        summary, terminal_resets=(_terminal_reset(lean, **changes),)
    )
    with pytest.raises(RuntimeError, match="terminal-reset telemetry"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            malformed, update_index=0, completed_environment_steps=4
        )
    assert runner._action_ball_full_mdp_active_drain_chronology is active
    assert runner._action_ball_full_mdp_last_drain_chronology is None
    assert active.telemetry_emitted is False
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutation",
    (
        "list",
        "foreign_row",
        "row_subclass",
        "tensor",
        "int_subclass",
        "bool",
        "float_subclass",
        "nan",
        "inf",
        "env_oob",
        "invalid_key",
        "invalid_reason",
        "payment_rollback",
        "retirement_rollback",
    ),
)
def test_diagnostic_completed_shot_abi_fails_before_telemetry_state_mutation(
    runner_module, capsys, mutation
):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    boundary, summary, active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )
    shot = _completed_shot(lean)

    class _IntSubclass(int):
        pass

    class _FloatSubclass(float):
        pass

    if mutation == "list":
        malformed = replace(summary, completed_shots=[shot])
    elif mutation == "foreign_row":
        malformed = replace(summary, completed_shots=(object(),))
    elif mutation == "row_subclass":
        class _ShotSubclass(type(shot)):
            pass

        subclass = _ShotSubclass(
            **{
                name: getattr(shot, name)
                for name in shot.__dataclass_fields__
            }
        )
        malformed = replace(summary, completed_shots=(subclass,))
    else:
        changes = {
            "tensor": {"target_x_m": torch.tensor(0.321)},
            "int_subclass": {"action_uid": _IntSubclass(5)},
            "bool": {"env_row": True},
            "float_subclass": {"target_x_m": _FloatSubclass(0.321)},
            "nan": {"target_x_m": float("nan")},
            "inf": {"target_y_m": float("inf")},
            "env_oob": {"env_row": 2},
            "invalid_key": {"action_uid": 0},
            "invalid_reason": {"motion_close_reason": 3},
            "payment_rollback": {"settlement_step": 43},
            "retirement_rollback": {"retirement_step": 41},
        }[mutation]
        malformed = replace(
            summary, completed_shots=(_completed_shot(lean, **changes),)
        )

    with pytest.raises(RuntimeError, match="shot telemetry"):
        runner._action_ball_full_mdp_prepare_epoch_ack_record(
            malformed, update_index=0, completed_environment_steps=4
        )
    assert runner._action_ball_full_mdp_active_drain_chronology is active
    assert runner._action_ball_full_mdp_last_drain_chronology is None
    assert active.telemetry_emitted is False
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out

    record = runner._action_ball_full_mdp_prepare_epoch_ack_record(
        summary, update_index=0, completed_environment_steps=4
    )
    ack_json, _wal_line = (
        runner._action_ball_full_mdp_encode_durable_pending_ack(
            record, update_index=0, completed_environment_steps=4
        )
    )
    owner.acknowledge_post_update(boundary, summary, update_index=0)
    record = runner._action_ball_full_mdp_consume_epoch_ack_summary(
        summary,
        update_index=0,
        completed_environment_steps=4,
        canonical_ack_json=ack_json,
    )
    assert record["completed_shots"] == []
    assert capsys.readouterr().out.count(
        "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON="
    ) == 1


def test_diagnostic_consumer_uses_construction_frozen_capacity_without_receipt(
    runner_module, capsys
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    assert runner._action_ball_full_mdp_lean_shot_slot_capacity == 1
    assert "receipt" not in inspect.signature(
        runner._action_ball_full_mdp_consume_epoch_ack_summary
    ).parameters
    boundary, summary, _active = _prepare_pending_epoch_summary(
        runner_module, owner, runner
    )

    class _NoCapacityRequery:
        @property
        def shot_slot_capacity(self):
            raise AssertionError("ACK consumer requeried ActionEpoch capacity")

    runner._action_ball_full_mdp_lean_epoch_owner_identity = _NoCapacityRequery()
    record = runner._action_ball_full_mdp_prepare_epoch_ack_record(
        summary, update_index=0, completed_environment_steps=4
    )
    ack_json, _wal_line = (
        runner._action_ball_full_mdp_encode_durable_pending_ack(
            record, update_index=0, completed_environment_steps=4
        )
    )
    owner.acknowledge_post_update(boundary, summary, update_index=0)
    record = runner._action_ball_full_mdp_consume_epoch_ack_summary(
        summary,
        update_index=0,
        completed_environment_steps=4,
        canonical_ack_json=ack_json,
    )
    assert record["shot_slot_capacity"] == 1
    assert record["completed_shots"] == []
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" in capsys.readouterr().out


def test_diagnostic_constructor_rejects_multi_slot_before_stepping(
    runner_module,
):
    owner, env, _lean, _epoch = _lean_owner(
        runner_module, [], shot_slot_capacity=2
    )
    with pytest.raises(RuntimeError, match="shot capacity differs"):
        _diagnostic_runner(runner_module, owner, [], env=env)
    assert env.step_calls == 0


def test_diagnostic_update_zero_and_later_share_general_zero_or_many_abi(
    runner_module, capsys
):
    owner, _env, lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    runner.learn(2)
    markers = [
        json.loads(line.split("=", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=")
    ]
    assert [row["ppo_update"] for row in markers] == [0, 1]
    assert [row["d05_transactions"] for row in markers] == [0, 0]
    assert owner._operation_sequence == owner._drain_sequence == 2
    chronology = runner._action_ball_full_mdp_last_drain_chronology
    assert type(chronology.projection) is lean.ActionEpochPpoBoundarySummary
    assert chronology.projection.frontier.operation_sequence == 2
    assert chronology.projection.frontier.start_commit == 1
    assert chronology.projection.frontier.end_commit == 1


def test_diagnostic_prepare_materializes_the_frozen_suffix_exactly_once(
    runner_module,
):
    from types import MethodType

    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    epoch_owner = owner.epoch_owner
    original = epoch_owner.materialize_drain
    calls = []

    def counted(self, *, start, end):
        assert self is epoch_owner
        calls.append((start, end))
        materialized = original(start=start, end=end)
        assert materialized.row_fault_bits.device.type == "cpu"
        assert materialized.row_fault_bits.dtype == torch.int64
        assert not bool(materialized.row_fault_bits.ne(0).any())
        assert all(
            value.device.type == "cpu" and value.is_contiguous()
            for entry in materialized.entries
            for value in entry.delta.values
        )
        return materialized

    epoch_owner.materialize_drain = MethodType(counted, epoch_owner)
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    runner.learn(1)
    assert calls == [(0, 1)]


def test_diagnostic_runner_cannot_consume_the_same_ack_summary_twice(
    runner_module, capsys
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    runner.learn(1)
    capsys.readouterr()
    chronology = runner._action_ball_full_mdp_last_drain_chronology
    with pytest.raises(RuntimeError, match="foreign or stale"):
        runner._action_ball_full_mdp_consume_epoch_ack_summary(
            chronology.projection,
            update_index=chronology.update_index,
            completed_environment_steps=chronology.completed_environment_steps,
            canonical_ack_json=b"{}",
        )
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out


def test_diagnostic_ack_failure_emits_no_marker_and_poison_is_sticky(
    runner_module, capsys
):
    from types import MethodType

    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])

    def fail_ack(self, boundary, summary, *, update_index):
        del self, boundary, summary, update_index
        raise RuntimeError("injected typed ACK failure")

    owner.acknowledge_post_update = MethodType(fail_ack, owner)
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    with pytest.raises(RuntimeError, match="injected typed ACK failure"):
        runner.learn(1)
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out
    assert owner.poisoned is True


def test_diagnostic_fsynced_wal_survives_destructive_ack_failure(
    runner_module, tmp_path, capsys
):
    from types import MethodType

    owner, _env, _lean, epoch = _lean_owner(runner_module, [])

    def fail_ack(self, boundary, summary, *, update_index):
        assert self._active_post_update_summary is summary
        del boundary, update_index
        raise RuntimeError("injected ACK after fsync")

    owner.acknowledge_post_update = MethodType(fail_ack, owner)
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="injected ACK after fsync"):
        runner.learn(1)
    rows = _durable_wal_rows(runner)
    assert [row["ppo_update"] for row in rows] == [0]
    assert rows[0]["status"] == (
        "optimizer_succeeded_durable_pending_destructive_ack"
    )
    assert owner.epoch_owner.drain_frontier == 0
    assert owner._active_post_update_summary is not None
    assert owner.poisoned is True
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", ("write", "fsync"))
def test_diagnostic_epoch_ack_append_failure_after_owner_ack_is_fail_stop(
    runner_module, tmp_path, capsys, monkeypatch, failure
):
    events = []
    owner, _env, _lean, _epoch = _lean_owner(runner_module, events)
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, log_dir=tmp_path
    )
    runner._action_ball_full_mdp_preflight_durable_wal()
    original_append = runner._action_ball_full_mdp_append_durable_wal
    if failure == "write":
        calls = 0

        def fail_epoch_ack_append(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError(errno.EIO, "injected EPOCH_ACK append failure")
            return original_append(*args, **kwargs)

        runner._action_ball_full_mdp_append_durable_wal = (
            fail_epoch_ack_append
        )
    else:
        original_fsync = runner_module.os.fsync
        regular_fsyncs = 0

        def fail_epoch_ack_fsync(fd):
            nonlocal regular_fsyncs
            if stat.S_ISREG(os.fstat(fd).st_mode):
                regular_fsyncs += 1
                if regular_fsyncs == 2:
                    raise OSError(
                        errno.EIO, "injected EPOCH_ACK append failure"
                    )
            return original_fsync(fd)

        monkeypatch.setattr(runner_module.os, "fsync", fail_epoch_ack_fsync)
    with pytest.raises(OSError, match="injected EPOCH_ACK append failure"):
        runner.learn(1)
    rows = _durable_wal_rows(runner)
    assert len(rows) == 1
    assert rows[0]["kind"] == "action_ball_epoch_durable_pending_v2"
    assert owner.epoch_owner.drain_frontier == 1
    assert owner._active_post_update_summary is None
    assert owner.poisoned is True
    assert runner._action_ball_full_mdp_boundary_poisoned is True
    assert events.count("optimizer") == 1
    scan = _durable_wal_scan(runner)
    assert scan["pending_without_epoch_ack"] is True
    assert scan["committed_frontier"] is None
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.learn(1)
    assert events.count("optimizer") == 1


def test_diagnostic_stdout_broken_pipe_cannot_erase_durable_wal(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    original_stdout = runner_module.sys.stdout

    class BrokenAckStdout:
        def write(self, value):
            if value.startswith("HOPE_ACTION_EPOCH_UPDATE_ACK_JSON="):
                raise BrokenPipeError("injected stdout failure")
            return original_stdout.write(value)

        def flush(self):
            return original_stdout.flush()

        def __getattr__(self, name):
            return getattr(original_stdout, name)

    monkeypatch.setattr(runner_module.sys, "stdout", BrokenAckStdout())
    with pytest.raises(BrokenPipeError, match="injected stdout failure"):
        runner.learn(1)
    rows = _durable_wal_rows(runner)
    assert [row["ppo_update"] for row in rows] == [0, 0]
    assert [row["kind"] for row in rows] == [
        "action_ball_epoch_durable_pending_v2",
        "action_ball_epoch_durable_ack_v2",
    ]
    assert _durable_wal_scan(runner)["committed_frontier"]["ppo_update"] == 0
    assert owner.epoch_owner.drain_frontier == 1
    assert owner._active_post_update_summary is None
    assert runner._action_ball_full_mdp_boundary_poisoned is True
    chronology = runner._action_ball_full_mdp_last_drain_chronology
    assert chronology.post_update_acknowledged is True
    assert chronology.telemetry_emitted is False


def test_diagnostic_two_updates_append_distinct_pending_and_epoch_ack_rows(
    runner_module, tmp_path
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    runner.rank = 6
    runner.learn(2)
    rows = _durable_wal_rows(runner)
    assert [row["ppo_update"] for row in rows] == [0, 0, 1, 1]
    pending = rows[::2]
    acknowledged = rows[1::2]
    assert [row["pending_ack_telemetry"]["epoch_operation_sequence"] for row in pending] == [1, 2]
    assert all(
        row["status"]
        == "optimizer_succeeded_durable_pending_destructive_ack"
        for row in pending
    )
    assert all(row["status"] == "destructive_epoch_ack_durable" for row in acknowledged)
    scan = _durable_wal_scan(runner)
    assert scan["durable_epoch_ack_count"] == 2
    assert scan["committed_frontier"]["ppo_update"] == 1


@pytest.mark.parametrize("mutation", ("unlink", "replace"))
def test_diagnostic_update_one_revalidates_wal_inode_before_optimizer(
    runner_module, tmp_path, mutation
):
    events = []
    owner, _env, _lean, _epoch = _lean_owner(runner_module, events)
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, log_dir=tmp_path
    )
    runner.rank = 9
    path = _durable_wal_path(runner)
    original_rollout_end = runner._notify_command_terms_rollout_end

    def mutate_after_update(step):
        original_rollout_end(step)
        if step == 0:
            assert path.exists()
            path.unlink()
            if mutation == "replace":
                path.write_bytes(b"")

    runner._notify_command_terms_rollout_end = mutate_after_update

    with pytest.raises((FileNotFoundError, RuntimeError)) as caught:
        runner.learn(2)
    assert "pre-optimizer drain failed" in str(caught.value)
    assert events.count("optimizer") == 1
    assert owner.poisoned is True
    assert owner._active_post_update_summary is None
    chronology = runner._action_ball_full_mdp_last_drain_chronology
    assert chronology.update_index == 0
    assert chronology.post_update_acknowledged is True
    assert chronology.telemetry_emitted is True


@pytest.mark.parametrize(
    "preexisting",
    (b"", b'{"partial":1}', b'{"old_complete_line":true}\n'),
)
def test_diagnostic_fresh_namespace_rejects_any_preexisting_wal(
    runner_module, tmp_path, preexisting
):
    events = []
    owner, _env, _lean, _epoch = _lean_owner(runner_module, events)
    runner, _ = _diagnostic_runner(
        runner_module, owner, events, log_dir=tmp_path
    )
    runner.rank = 10
    path = _durable_wal_path(runner)
    path.parent.mkdir(parents=True)
    path.write_bytes(preexisting)

    with pytest.raises(RuntimeError, match="pre-optimizer drain failed") as caught:
        runner.learn(1)
    assert isinstance(caught.value.__cause__, FileExistsError)
    assert events.count("optimizer") == 0
    assert owner.poisoned is True
    assert owner._active_post_update_summary is None
    assert path.read_bytes() == preexisting


def test_diagnostic_malformed_pending_summary_stops_before_wal_or_ack(
    runner_module, tmp_path, capsys
):
    from types import MethodType

    owner, _env, lean, epoch = _lean_owner(runner_module, [])
    original_prepare = owner.prepare_post_update_summary

    def malformed_prepare(self, boundary, *, update_index):
        summary = original_prepare(boundary, update_index=update_index)
        malformed = replace(
            summary,
            terminal_resets=(
                _terminal_reset(lean, reason_bits=0),
            ),
        )
        self._active_post_update_summary = malformed
        return malformed

    owner.prepare_post_update_summary = MethodType(malformed_prepare, owner)
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="terminal-reset telemetry"):
        runner.learn(1)
    assert _durable_wal_rows(runner) == []
    assert owner.epoch_owner.drain_frontier == 0
    assert owner._active_post_update_summary is not None
    assert owner.poisoned is True
    assert "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON=" not in capsys.readouterr().out


def test_diagnostic_checkpoint_and_save_hold_before_filesystem_write(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    runner, _ = _diagnostic_runner(runner_module, owner, [])
    checkpoint = tmp_path / "forbidden.pt"
    calls = []
    original = type(owner)._capture_private_carry_for_save

    def counted(self):
        calls.append("preflight")
        return original(self)

    monkeypatch.setattr(type(owner), "_capture_private_carry_for_save", counted)
    for _ in range(2):
        with pytest.raises(
            RuntimeError, match="forbids checkpoint/save.*mandatory roles"
        ):
            runner.save(str(checkpoint))
    assert calls == ["preflight", "preflight"]
    assert not checkpoint.exists()


def test_diagnostic_control_checkpoint_holds_before_any_directory_creation(
    runner_module, tmp_path, monkeypatch
):
    owner, _env, _lean, _epoch = _lean_owner(runner_module, [])
    log_dir = tmp_path / "never-created-log-dir"
    runner, _ = _diagnostic_runner(
        runner_module, owner, [], log_dir=log_dir
    )
    calls = []
    original = type(owner)._capture_private_carry_for_save

    def counted(self):
        calls.append("preflight")
        assert not log_dir.exists()
        return original(self)

    monkeypatch.setattr(type(owner), "_capture_private_carry_for_save", counted)
    with pytest.raises(
        RuntimeError, match="forbids checkpoint/save.*mandatory roles"
    ):
        runner._action_ball_control_checkpoint(
            step=0, purpose="policy_snapshot", request_seq=0
        )
    assert calls == ["preflight"]
    assert not log_dir.exists()
    assert not (log_dir / "curriculum_control").exists()
    assert not (
        log_dir / "curriculum_control"
        / "update_00000000000000000000_policy_snapshot_request_00000000000000000000"
    ).exists()


def test_diagnostic_rejects_foreign_owner_before_observation_or_sampling(
    runner_module,
):
    owner = object()
    env = _Env(obs_mode="action_ball_full_mdp", target_mode="action_ball_full_mdp")
    env._drain_owner = owner
    with pytest.raises(RuntimeError, match="exact code-owned lean runtime owner"):
        runner_module.MotionOnPolicyRunner(
            env,
            {"num_steps_per_env": 2},
            action_ball_full_mdp_runtime_owner=owner,
            action_ball_full_mdp_run_mode="single_action_lean",
        )

def test_diagnostic_one_update_gpu2_three_stage_chronology(runner_module):
    if not runner_module.torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    events = []
    owner, env, _lean, _epoch = _lean_owner(
        runner_module, events, device="cuda:0"
    )
    runner, _env = _diagnostic_runner(runner_module, owner, events, env=env)
    device = runner_module.torch.device("cuda:0")
    runner.device = device
    runner.alg.policy.to(device)
    runner.obs_normalizer.to(device)
    runner.privileged_obs_normalizer.to(device)
    runner.alg.policy.memory_a.hidden_states = runner_module.torch.zeros(
        1, runner.env.num_envs, 3, device=device
    )
    runner.alg.policy.memory_c.hidden_states = runner_module.torch.zeros(
        1, runner.env.num_envs, 3, device=device
    )

    runner.learn(1)

    assert runner.alg.policy.weight.device.type == "cuda"
    assert events == ["optimizer", ("rollout_end", 0)]


def test_diagnostic_joint_safety_localization_uses_the_existing_cuda_pack(
    runner_module,
):
    if not runner_module.torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = runner_module.torch.device("cuda:0")
    num_envs = 2
    joint_count = 2
    counts = runner_module.torch.tensor(
        [[3, 0], [0, 2]], dtype=runner_module.torch.int64, device=device
    )
    minimum_lower = runner_module.torch.full(
        (num_envs, joint_count), 0.2, device=device
    )
    minimum_upper = runner_module.torch.full(
        (num_envs, joint_count), 0.3, device=device
    )
    minimum_upper[0, 0] = -2.3e-5
    minimum_lower[1, 1] = -1.0e-5
    minimum_upper[1, 1] = -2.0e-5
    env_counts = runner_module.torch.full(
        (num_envs,), 2, dtype=runner_module.torch.int64, device=device
    )
    zeros_env = runner_module.torch.zeros_like(env_counts)
    zeros_joint = runner_module.torch.zeros_like(counts)
    actual_latch = runner_module.torch.tensor(
        [True, True], dtype=runner_module.torch.bool, device=device
    )
    snapshot = {
        "enabled": True,
        "diagnostic_compact_evidence": True,
        "terminal_archives": (),
        "identity_bound_policy_steps": (),
        "policy_step_summary_used": 0,
        "policy_step_summary_overflow_latch": False,
        "policy_step_summary_overflow_count": 0,
        "terminal_archive_used": 0,
        "terminal_archive_overflow_latch": False,
        "terminal_archive_overflow_count": 0,
        "diagnostic_first_policy_step_sequence": 0,
        "diagnostic_last_policy_step_sequence": 1,
        "since_last_consume": {
            "has_data": True,
            "consume_sequence": 0,
            "policy_step_count": env_counts,
            "complete_policy_step_count": env_counts.clone(),
            "incomplete_policy_step_count": zeros_env,
            "apply_readback_count": env_counts * 4,
            "post_readback_count": env_counts.clone(),
            "timestamp_invariant_pass_count": env_counts.clone(),
            "qdes_joint_count": zeros_joint,
            "policy_crossing_joint_count": counts.clone(),
            "substep_hard_crossing_joint_count": counts.clone(),
            "actual_hard_edge_joint_count": counts,
            "minimum_hard_lower_gap": minimum_lower,
            "minimum_hard_upper_gap": minimum_upper,
            "hard_crossing_latch": actual_latch.clone(),
            "actual_hard_edge_latch": actual_latch,
        },
    }

    class CudaCompactTerm:
        _joint_safety_diagnostic_compact_evidence = True
        _pre_apply_guard_decimation = 4
        _pre_apply_guard_physics_dt_s = 0.005
        _pre_apply_guard_margin_rad = 0.0
        _pre_apply_guard_margin_fraction = 0.05
        _pre_apply_guard_brake_mode = "velocity_horizon_v1"
        _joint_ids = slice(None)
        _joint_names = ["j0", "j1"]

        def __init__(self):
            self._processed_actions = runner_module.torch.zeros(
                num_envs, joint_count, device=device
            )
            limits = runner_module.torch.tensor(
                [[-1.0, 1.0], [-1.2, 1.2]], device=device
            ).repeat(num_envs, 1, 1)
            self._asset = SimpleNamespace(
                data=SimpleNamespace(joint_pos_limits=limits)
            )
            self.token = object()
            self.acknowledged = False

        def prepare_joint_safety_ledger_consume(self):
            return self.token, snapshot

        def acknowledge_joint_safety_ledger(self, token):
            assert token is self.token
            self.acknowledged = True

    term = CudaCompactTerm()
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda _name: term)
        )
    )
    runner.num_steps_per_env = 2
    runner._diagnostic_joint_safety_compact_update_evidence = lambda: True

    prepared = runner._prepare_diagnostic_joint_safety_update(
        0, expected_action_term=term
    )
    localization = prepared["record"]["actual_hard_edge_localization"]
    assert localization["rows"] == [
        {
            "env_row": 0,
            "joint_index": 0,
            "joint_name": "j0",
            "side": "upper",
            "joint_readback_count_either_side": 3,
            "minimum_signed_hard_gap_rad": pytest.approx(-2.3e-5),
        },
        {
            "env_row": 1,
            "joint_index": 1,
            "joint_name": "j1",
            "side": "lower",
            "joint_readback_count_either_side": 2,
            "minimum_signed_hard_gap_rad": pytest.approx(-1.0e-5),
        },
        {
            "env_row": 1,
            "joint_index": 1,
            "joint_name": "j1",
            "side": "upper",
            "joint_readback_count_either_side": 2,
            "minimum_signed_hard_gap_rad": pytest.approx(-2.0e-5),
        },
    ]
    runner._commit_diagnostic_joint_safety_update(prepared)
    assert term.acknowledged is True
