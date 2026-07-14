from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_rewards.py"
)


def _load_rewards_module():
    dependency_name = "whole_body_tracking.tasks.tracking.mdp.hope_commands"
    previous = sys.modules.get(dependency_name)
    stub = types.ModuleType(dependency_name)
    stub.RacketTargetCommand = object
    stub.face_tracking_pair = lambda cmd: (cmd.racket_normal_w, cmd.target_normal_cmd)
    sys.modules[dependency_name] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "base_decel_activation_hope_rewards_under_test", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            del sys.modules[dependency_name]
        else:
            sys.modules[dependency_name] = previous


R = _load_rewards_module()


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "racket_target"
        return self.command


def _case(*, common_step_counter=0, std=1.0):
    motion = types.SimpleNamespace(
        in_hold=torch.tensor([True, False, False, False])
    )
    command = types.SimpleNamespace(
        racket_target_pos_w=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
        racket_pos_w=torch.zeros(4, 3),
        robot=types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_lin_vel_w=torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.25, 0.0, 0.0],
                    ]
                )
            )
        ),
        pre_strike=torch.tensor([True, True, False, True]),
        _motion=lambda: motion,
    )
    env = types.SimpleNamespace(
        command_manager=_CommandManager(command),
        common_step_counter=common_step_counter,
        reward_buf=torch.tensor([9.0, 8.0, 7.0, 6.0]),
    )
    params = {"v_gain": 1.0, "v_max": 2.0, "std": std}
    return env, command, params


def _expected(command, params):
    planar_err = torch.norm(
        command.racket_target_pos_w[:, :2] - command.racket_pos_w[:, :2], dim=-1
    )
    v_des = (params["v_gain"] * planar_err).clamp(0.0, params["v_max"])
    v_base = torch.norm(command.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_base - v_des) / params["std"] ** 2)
    eligible = command.pre_strike & ~command._motion().in_hold
    return raw, eligible, raw * eligible.float()


def test_reward_execution_and_activation_share_the_exact_kernel_and_gate():
    env, command, params = _case()
    raw, eligible, expected_reward = _expected(command, params)

    actual_reward = R.base_decel_tracking(env, "racket_target", **params)
    counters = R.consume_base_decel_activation_counters(env, "racket_target")

    assert torch.equal(actual_reward, expected_reward)
    assert counters["base_decel_eligible_sample_count"].item() == int(
        eligible.sum()
    )
    assert torch.equal(
        counters["base_decel_raw_kernel_sum"], expected_reward.sum()
    )
    assert counters["base_decel_raw_kernel_nonzero_sample_count"].item() == int(
        (eligible & torch.isfinite(raw) & raw.gt(0)).sum()
    )
    assert all(value.ndim == 0 for value in counters.values())
    assert counters["base_decel_eligible_sample_count"].dtype == torch.long
    assert counters["base_decel_raw_kernel_nonzero_sample_count"].dtype == torch.long
    assert counters["base_decel_raw_kernel_sum"].dtype == expected_reward.dtype


def test_reward_stage_probe_measures_weight_zero_control_and_returns_exact_zero():
    env, command, params = _case()
    command_before = {
        "target": command.racket_target_pos_w.clone(),
        "racket": command.racket_pos_w.clone(),
        "velocity": command.robot.data.root_lin_vel_w.clone(),
        "pre_strike": command.pre_strike.clone(),
        "hold": command._motion().in_hold.clone(),
    }
    reward_buf_before = env.reward_buf.clone()
    rng_before = torch.random.get_rng_state().clone()

    # The control's real base_decel term has weight zero and is skipped.  The separate probe has
    # manager weight 1.0, but its function output is identically zero after recording raw evidence.
    probe_reward = R.base_decel_activation_probe(env, "racket_target", **params)
    counters = R.consume_base_decel_activation_counters(env, "racket_target")

    assert torch.equal(probe_reward, torch.zeros_like(probe_reward))
    assert counters["base_decel_eligible_sample_count"].item() == 2
    assert counters["base_decel_raw_kernel_nonzero_sample_count"].item() == 2
    assert counters["base_decel_raw_kernel_sum"].item() > 0.0
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(env.reward_buf, reward_buf_before)
    assert torch.equal(command.racket_target_pos_w, command_before["target"])
    assert torch.equal(command.racket_pos_w, command_before["racket"])
    assert torch.equal(command.robot.data.root_lin_vel_w, command_before["velocity"])
    assert torch.equal(command.pre_strike, command_before["pre_strike"])
    assert torch.equal(command._motion().in_hold, command_before["hold"])


def test_probe_and_treatment_reward_are_idempotent_within_one_reward_stage():
    env, _command, params = _case(common_step_counter=17)
    _raw, eligible, expected_reward = _expected(_command, params)

    probe_reward = R.base_decel_activation_probe(env, "racket_target", **params)
    reward = R.base_decel_tracking(env, "racket_target", **params)
    first = R.consume_base_decel_activation_counters(env, "racket_target")
    assert torch.equal(probe_reward, torch.zeros_like(probe_reward))
    assert torch.equal(reward, expected_reward)
    assert first["base_decel_eligible_sample_count"].item() == int(eligible.sum())
    assert torch.equal(first["base_decel_raw_kernel_sum"], expected_reward.sum())

    # Consuming does not forget the last step: a late duplicate observation cannot leak into the
    # next PPO update's window.
    R.base_decel_activation_probe(env, "racket_target", **params)
    same_step = R.consume_base_decel_activation_counters(env, "racket_target")
    assert all(value.item() == 0 for value in same_step.values())

    env.common_step_counter += 1
    R.base_decel_activation_probe(env, "racket_target", **params)
    R.base_decel_tracking(env, "racket_target", **params)
    next_step = R.consume_base_decel_activation_counters(env, "racket_target")
    assert next_step["base_decel_eligible_sample_count"].item() == int(
        eligible.sum()
    )
    assert torch.equal(next_step["base_decel_raw_kernel_sum"], expected_reward.sum())


def test_control_and_treatment_measure_the_same_pre_command_reward_phase():
    """Encode Isaac 2.1's reward -> reset -> command order for this paired ledger."""

    def _reward_then_command(*, treatment):
        env, command, params = _case(common_step_counter=31)
        _raw, _eligible, expected_reward = _expected(command, params)

        # RewardManager stage: the probe is active in both arms; only treatment also owns the real
        # shaping term.  No reset/command mutation occurs between RewardTerms.
        total_reward = R.base_decel_activation_probe(
            env, "racket_target", **params
        )
        if treatment:
            total_reward = total_reward + R.base_decel_tracking(
                env, "racket_target", **params
            )
        counters = R.consume_base_decel_activation_counters(env, "racket_target")

        # Isaac reset/command stage happens later and may replace the target.  It must not write a
        # second, next-state observation into this reward-stage ledger.
        command.racket_target_pos_w.add_(10.0)
        return counters, total_reward, expected_reward

    control, control_total, expected = _reward_then_command(treatment=False)
    treatment, treatment_total, treatment_expected = _reward_then_command(
        treatment=True
    )

    assert all(torch.equal(control[name], treatment[name]) for name in control)
    assert torch.equal(control_total, torch.zeros_like(control_total))
    assert torch.equal(treatment_total, treatment_expected)
    assert torch.equal(expected, treatment_expected)


def test_same_step_parameter_mismatch_fails_instead_of_attesting_another_kernel():
    env, _command, params = _case(common_step_counter=4)
    R.observe_base_decel_activation(env, "racket_target", **params)

    with pytest.raises(RuntimeError, match="different v_gain/v_max/std"):
        R.base_decel_tracking(
            env,
            "racket_target",
            v_gain=params["v_gain"] + 0.1,
            v_max=params["v_max"],
            std=params["std"],
        )


def test_counts_accumulate_across_steps_until_one_update_consume():
    env, command, params = _case(common_step_counter=100)
    raw, eligible, expected_reward = _expected(command, params)

    for _ in range(3):
        R.observe_base_decel_activation(env, "racket_target", **params)
        env.common_step_counter += 1
    counters = R.consume_base_decel_activation_counters(env, "racket_target")

    assert counters["base_decel_eligible_sample_count"].item() == 3 * int(
        eligible.sum()
    )
    assert torch.equal(
        counters["base_decel_raw_kernel_sum"], 3 * expected_reward.sum()
    )
    assert counters["base_decel_raw_kernel_nonzero_sample_count"].item() == 3 * int(
        (eligible & torch.isfinite(raw) & raw.gt(0)).sum()
    )


def test_nonzero_numerator_exposes_kernel_underflow_without_losing_denominator():
    env, _command, params = _case(std=1.0e-6)
    R.observe_base_decel_activation(env, "racket_target", **params)
    counters = R.consume_base_decel_activation_counters(env, "racket_target")

    assert counters["base_decel_eligible_sample_count"].item() == 2
    assert counters["base_decel_raw_kernel_sum"].item() == 0.0
    assert counters["base_decel_raw_kernel_nonzero_sample_count"].item() == 0


def test_nonfinite_raw_kernel_is_preserved_in_sum_but_not_counted_as_nonzero():
    env, _command, params = _case(std=float("nan"))
    R.observe_base_decel_activation(env, "racket_target", **params)
    counters = R.consume_base_decel_activation_counters(env, "racket_target")

    assert counters["base_decel_eligible_sample_count"].item() == 2
    assert torch.isnan(counters["base_decel_raw_kernel_sum"])
    assert counters["base_decel_raw_kernel_nonzero_sample_count"].item() == 0


def test_consume_before_any_execution_returns_scalar_zeros_and_resets_exactly():
    env, _command, _params = _case()
    first = R.consume_base_decel_activation_counters(env, "racket_target")
    second = R.consume_base_decel_activation_counters(env, "racket_target")

    assert set(first) == {
        "base_decel_eligible_sample_count",
        "base_decel_raw_kernel_sum",
        "base_decel_raw_kernel_nonzero_sample_count",
    }
    assert all(value.ndim == 0 and value.item() == 0 for value in first.values())
    assert all(value.ndim == 0 and value.item() == 0 for value in second.values())
