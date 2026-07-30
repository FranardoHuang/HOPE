"""Focused contract test for diagnostic ActionBall joint-safety draining."""

from __future__ import annotations

import json
import sys
import types

from test_joint_limit_safety import (
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
    action, env = _two_step_cross_reset_action_ball_ledger()
    env.cfg.commands.racket_target.action_ball_diagnostic_unauthorized = True

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
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is True
    )

    runner.learn(num_learning_iterations=1)

    assert optimizer_calls == ["optimizer"]
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is False
    )
    assert runner._joint_safety_consumed_step == 169
    output_lines = capsys.readouterr().out.splitlines()
    joint_lines = [
        line
        for line in output_lines
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    assert len(joint_lines) == 1
    record = json.loads(joint_lines[0].split("=", 1)[1])
    assert record["ppo_update"] == 169
    assert record["status"] == "optimizer_committed_and_ledger_acknowledged"
    assert not any("REWARD_ACTIVATION" in line for line in output_lines)
