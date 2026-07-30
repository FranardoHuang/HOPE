"""Focused contract test for diagnostic ActionBall joint-safety draining."""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from test_joint_limit_safety import (
    _finish_guarded_policy_step,
    _two_step_cross_reset_action_ball_ledger,
)
from test_training_launch_claim import (
    _load_contract_module,
    _load_runner_module,
)


def test_diagnostic_action_ball_drains_joint_safety_without_reward_authority(
    monkeypatch, tmp_path, capsys
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger(
        diagnostic_compact_evidence=True
    )

    class ForbiddenRewardLedger:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError(
                "diagnostic ActionBall must not construct a formal Reward ledger"
            )

    forbidden_reward_module = types.ModuleType(
        "whole_body_tracking.utils.effective_reward_recipe"
    )
    forbidden_reward_module.EffectiveRewardActivationLedger = (
        ForbiddenRewardLedger
    )
    forbidden_reward_module.ActionBoundRewardEvidenceLedger = (
        ForbiddenRewardLedger
    )
    forbidden_reward_module.canonical_effective_reward_activation_json = (
        lambda record: json.dumps(record, sort_keys=True)
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        forbidden_reward_module,
    )

    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def one_update(_self, **_kwargs):
        _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", one_update, raising=False)
    optimizer_calls = []
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(
        unwrapped=env,
        step=lambda *_args, **_kwargs: None,
    )
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0
    runner.current_learning_iteration = 169
    runner.alg = types.SimpleNamespace(
        update=lambda: optimizer_calls.append("optimizer")
    )
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False
    runner._service_action_ball_frozen_evaluation = lambda _step: False

    assert runner._effective_reward_activation_task_kind() is None
    before = action.joint_safety_ledger_snapshot()
    assert before["since_last_consume"]["has_data"] is True
    assert before["record_count"] == 0
    assert before["q"].numel() == 0
    assert before["identity_bound_policy_steps"] == ()
    assert before["policy_step_summary_used"] == 0

    runner.learn(num_learning_iterations=1)

    assert optimizer_calls == ["optimizer"]
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is False
    )
    assert runner._joint_safety_consumed_step == 169

    # A second rollout must continue the producer sequence after the first
    # aggregate was acknowledged and cleared.
    asset = env.scene["robot"]
    for _ in range(2):
        action.process_actions(torch.zeros(2, 2))
        _finish_guarded_policy_step(action, asset)
    runner.current_learning_iteration = 170
    runner.learn(num_learning_iterations=1)
    assert optimizer_calls == ["optimizer", "optimizer"]
    assert runner._joint_safety_consumed_step == 170
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is False
    )

    # If PPO fails, the compact aggregate remains frozen and unacknowledged;
    # later environment mutation must fail instead of silently dropping it.
    for _ in range(2):
        action.process_actions(torch.zeros(2, 2))
        _finish_guarded_policy_step(action, asset)
    runner.current_learning_iteration = 171

    def failed_optimizer():
        raise RuntimeError("synthetic optimizer failure")

    runner.alg.update = failed_optimizer
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        runner.learn(num_learning_iterations=1)
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is True
    )
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.process_actions(torch.zeros(2, 2))

    output_lines = capsys.readouterr().out.splitlines()
    joint_lines = [
        line
        for line in output_lines
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    assert len(joint_lines) == 2
    records = [
        json.loads(line.split("=", 1)[1]) for line in joint_lines
    ]
    assert [record["ppo_update"] for record in records] == [169, 170]
    assert records[1]["consume_sequence"] == records[0]["consume_sequence"] + 1
    assert (
        records[1]["first_policy_step_sequence"]
        == records[0]["last_policy_step_sequence"] + 1
    )
    for record in records:
        assert record["status"] == (
            "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
        )
        assert record["formal_authority"] is False
        assert record["identity_bound_policy_step_count"] == 0
        assert record["terminal_archive_count"] == 0
    assert not (tmp_path / "joint_safety_ledgers").exists()
    assert not any("REWARD_ACTIVATION" in line for line in output_lines)
