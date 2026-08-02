"""Host-only guards for the live RSL-RL normalization/std ABI."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
)
CONTRACT = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _FakeOnPolicyRunner:
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        del init_at_random_ep_len
        for _ in range(int(num_learning_iterations)):
            self.alg.update()


def _load_runner(monkeypatch):
    contract_spec = importlib.util.spec_from_file_location(
        "runtime_guard_contract", CONTRACT
    )
    contract = importlib.util.module_from_spec(contract_spec)
    assert contract_spec.loader is not None
    contract_spec.loader.exec_module(contract)

    fake_rsl = _module("rsl_rl")
    fake_rsl.__path__ = []
    fake_runners = _module("rsl_rl.runners")
    fake_runners.__path__ = []
    fake_wbt = _module("whole_body_tracking")
    fake_wbt.__path__ = []
    fake_utils = _module("whole_body_tracking.utils")
    fake_utils.__path__ = []
    fake_isaaclab = _module("isaaclab_rl")
    fake_isaaclab.__path__ = []
    modules = {
        "rsl_rl": fake_rsl,
        "rsl_rl.env": _module("rsl_rl.env", VecEnv=object),
        "rsl_rl.runners": fake_runners,
        "rsl_rl.runners.on_policy_runner": _module(
            "rsl_rl.runners.on_policy_runner",
            OnPolicyRunner=_FakeOnPolicyRunner,
        ),
        "isaaclab_rl": fake_isaaclab,
        "isaaclab_rl.rsl_rl": _module(
            "isaaclab_rl.rsl_rl",
            export_policy_as_onnx=lambda *args, **kwargs: None,
        ),
        "whole_body_tracking": fake_wbt,
        "whole_body_tracking.utils": fake_utils,
        "whole_body_tracking.utils.exporter": _module(
            "whole_body_tracking.utils.exporter",
            attach_onnx_metadata=lambda *args, **kwargs: None,
            export_motion_policy_as_onnx=lambda *args, **kwargs: False,
            is_empirical_normalizer=lambda value: (
                value is not None
                and not isinstance(value, torch.nn.Identity)
            ),
        ),
        "whole_body_tracking.utils.training_contract": contract,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "runner_runtime_guards_under_test", RUNNER
    )
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture()
def runner_module(monkeypatch):
    return _load_runner(monkeypatch)


class _EmpiricalNormalizer(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, width))
        self.register_buffer("_var", torch.ones(1, width))
        self.register_buffer("_std", torch.ones(1, width))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))


class _Policy(torch.nn.Module):
    def __init__(self, values, *, noise_std_type: str):
        super().__init__()
        self.noise_std_type = noise_std_type
        tensor = torch.tensor(values, dtype=torch.float32)
        if noise_std_type == "scalar":
            self.std = torch.nn.Parameter(tensor)
        elif noise_std_type == "log":
            self.log_std = torch.nn.Parameter(tensor.log())
        else:  # pragma: no cover - test construction guard
            raise ValueError(noise_std_type)


def _runner(runner_module, *, policy=None, empirical=False):
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.empirical_normalization = empirical
    runner.obs_normalizer = (
        _EmpiricalNormalizer(4) if empirical else torch.nn.Identity()
    )
    runner.privileged_obs_normalizer = (
        _EmpiricalNormalizer(6) if empirical else torch.nn.Identity()
    )
    runner.rank = 0
    if policy is not None:
        runner.alg = SimpleNamespace(
            policy=policy,
            learning_rate=1.0e-3,
            optimizer=SimpleNamespace(param_groups=[{"lr": 1.0e-3}]),
        )
    return runner


def test_empirical_normalizers_are_live_finite_and_shape_consistent(
    runner_module,
):
    runner = _runner(runner_module, empirical=True)
    binding = runner._validate_training_normalizers()
    assert binding["empirical_normalization"] is True
    assert binding["normalizers"]["actor"]["enabled"] is True
    assert binding["normalizers"]["actor"]["state_shapes"] == {
        "_mean": [1, 4],
        "_var": [1, 4],
        "_std": [1, 4],
        "count": [],
    }
    assert binding["normalizers"]["critic"]["state_shapes"]["_mean"] == [1, 6]


def test_stage1_v2_actor_normalizer_accepts_exact_225_width(runner_module):
    runner = _runner(runner_module, empirical=True)
    runner.obs_normalizer = _EmpiricalNormalizer(225)
    runner.privileged_obs_normalizer = _EmpiricalNormalizer(318)

    binding = runner._validate_training_normalizers()

    assert binding["normalizers"]["actor"]["state_shapes"]["_mean"] == [1, 225]
    assert binding["normalizers"]["actor"]["state_shapes"]["_std"] == [1, 225]
    assert binding["normalizers"]["critic"]["state_shapes"]["_mean"] == [1, 318]


def test_stage1_v2_obs_mode_selects_strict_exact_resume_without_claiming_v1(
    runner_module,
):
    runner = _runner(runner_module, empirical=False)
    racket = SimpleNamespace(target_mode="reference_perturbed")
    cfg = SimpleNamespace(
        obs_mode="stage1_natural_clip_paddle_world",
        commands=SimpleNamespace(racket_target=racket),
    )
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))

    assert runner._strict_exact_resume_target_mode() == "stage1_natural_clip"

    # Historical 170-D checkpoints remain a separate v1 contract.  Merely
    # selecting their old mode must not relabel them as the fresh 225-D ABI.
    cfg.obs_mode = "stage1_natural_clip"
    assert runner._strict_exact_resume_target_mode() is None


@pytest.mark.parametrize("moment_name", ["var", "std"])
def test_normalizer_guard_accepts_supported_state_key_variants(
    runner_module, moment_name
):
    class AlternateNormalizer(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.register_buffer("mean", torch.zeros(width))
            self.register_buffer(moment_name, torch.ones(width))
            self.register_buffer("count", torch.tensor(3, dtype=torch.long))

    runner = _runner(runner_module, empirical=True)
    runner.obs_normalizer = AlternateNormalizer(4)
    runner.privileged_obs_normalizer = AlternateNormalizer(6)
    binding = runner._validate_training_normalizers()
    semantic = binding["normalizers"]["actor"]["semantic_buffers"]
    assert semantic["mean"] == "mean"
    assert semantic[moment_name] == moment_name
    assert semantic["count"] == "count"


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_empirical_normalization_rejects_absent_or_identity(
    runner_module, role
):
    runner = _runner(runner_module, empirical=True)
    attribute = (
        "obs_normalizer"
        if role == "actor"
        else "privileged_obs_normalizer"
    )
    setattr(runner, attribute, torch.nn.Identity())
    with pytest.raises(RuntimeError, match=f"empirical {role}.*no-op"):
        runner._validate_training_normalizers()


def test_disabled_normalization_rejects_a_live_transform(runner_module):
    runner = _runner(runner_module, empirical=False)
    runner.actor_obs_normalizer = _EmpiricalNormalizer(4)
    with pytest.raises(RuntimeError, match="disabled actor.*live"):
        runner._validate_training_normalizers()


@pytest.mark.parametrize("defect", ["shape", "nan", "negative_std", "count"])
def test_empirical_normalizer_state_defects_fail_loud(runner_module, defect):
    runner = _runner(runner_module, empirical=True)
    normalizer = runner.obs_normalizer
    if defect == "shape":
        normalizer._std = torch.ones(1, 3)
    elif defect == "nan":
        normalizer._mean[0] = float("nan")
    elif defect == "negative_std":
        normalizer._std[0] = -1.0
    else:
        normalizer.count = torch.zeros(2)
    with pytest.raises(RuntimeError, match="empirical actor"):
        runner._validate_training_normalizers()


@pytest.mark.parametrize("noise_std_type", ["scalar", "log"])
def test_policy_std_snapshot_reports_realized_values_without_mutation(
    runner_module, noise_std_type
):
    policy = _Policy([0.02, 0.04, 0.08], noise_std_type=noise_std_type)
    runner = _runner(runner_module, policy=policy)
    before = {
        key: value.detach().clone()
        for key, value in policy.state_dict().items()
    }
    record = runner._policy_std_snapshot(ppo_update=7)
    assert record["noise_std_type"] == noise_std_type
    assert record["ppo_update"] == 7
    assert record["policy_std_min"] == pytest.approx(0.02)
    assert record["policy_std_mean"] == pytest.approx(0.14 / 3.0)
    assert record["policy_std_max"] == pytest.approx(0.08)
    assert record["learning_rate"] == 1.0e-3
    assert record["learning_rate_at_floor"] is False
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("bad", [0.0, -0.01, float("nan"), float("inf")])
def test_scalar_policy_std_must_be_finite_and_positive(runner_module, bad):
    runner = _runner(
        runner_module,
        policy=_Policy([0.02, 0.03], noise_std_type="scalar"),
    )
    with torch.no_grad():
        runner.alg.policy.std[0] = bad
    with pytest.raises(RuntimeError, match="realized policy std"):
        runner._policy_std_snapshot(ppo_update=1)


def test_log_policy_std_underflow_is_rejected(runner_module):
    runner = _runner(
        runner_module,
        policy=_Policy([0.02, 0.03], noise_std_type="log"),
    )
    with torch.no_grad():
        runner.alg.policy.log_std[0] = -1000.0
    with pytest.raises(RuntimeError, match="strictly positive"):
        runner._policy_std_snapshot(ppo_update=1)


def test_policy_std_json_contains_exact_lr_floor_record(runner_module, capsys):
    runner = _runner(
        runner_module,
        policy=_Policy([0.02, 0.03], noise_std_type="scalar"),
    )
    runner.alg.learning_rate = 1.0e-5
    runner.alg.optimizer.param_groups[0]["lr"] = 1.0e-5
    record = runner._emit_policy_std_update(ppo_update=11)
    line = capsys.readouterr().out.strip()
    assert line.startswith("HOPE_POLICY_STD_UPDATE_JSON=")
    decoded = json.loads(line.split("=", 1)[1])
    assert decoded == record
    assert decoded["learning_rate"] == 1.0e-5
    assert decoded["learning_rate_at_floor"] is True


def test_runtime_abi_receipt_records_distribution_origin_and_capabilities(
    runner_module, monkeypatch, capsys
):
    runner = _runner(
        runner_module,
        policy=_Policy([0.02, 0.03], noise_std_type="scalar"),
    )
    binding = runner._validate_training_normalizers()
    monkeypatch.setattr(
        runner_module.MotionOnPolicyRunner,
        "_rsl_rl_distribution_identity",
        staticmethod(
            lambda: {
                "distributions": [{"name": "rsl-rl-lib", "version": "3.1.2"}],
                "package_origin": "/runtime/rsl_rl/__init__.py",
                "runner_module": "rsl_rl.runners.on_policy_runner",
                "runner_origin": "/runtime/rsl_rl/runners/on_policy_runner.py",
            }
        ),
    )
    record = runner._emit_rsl_rl_runtime_abi(
        normalizer_binding=binding
    )
    line = capsys.readouterr().out.strip()
    assert json.loads(line.split("=", 1)[1]) == record
    assert record["runtime"]["distributions"][0]["version"] == "3.1.2"
    assert record["capabilities"]["positive_realized_policy_std_guard"] is True


def test_learn_emits_one_post_optimizer_scalar_std_record_per_update(
    runner_module, capsys
):
    policy = _Policy([0.02, 0.03], noise_std_type="scalar")
    runner = _runner(runner_module, policy=policy, empirical=False)
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            cfg=SimpleNamespace(commands=None),
            command_manager=None,
        ),
        reset=lambda: None,
    )
    runner.current_learning_iteration = 4
    runner.num_steps_per_env = 24
    runner.disable_logs = True
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False
    optimizer_calls = []

    def optimizer_update():
        optimizer_calls.append(len(optimizer_calls))
        with torch.no_grad():
            policy.std.add_(0.001)

    runner.alg.update = optimizer_update
    runner.learn(num_learning_iterations=2)

    lines = capsys.readouterr().out.splitlines()
    records = [
        json.loads(line.split("=", 1)[1])
        for line in lines
        if line.startswith("HOPE_POLICY_STD_UPDATE_JSON=")
    ]
    assert optimizer_calls == [0, 1]
    assert [record["ppo_update"] for record in records] == [4, 5]
    assert records[0]["policy_std_min"] == pytest.approx(0.021)
    assert records[1]["policy_std_min"] == pytest.approx(0.022)
    torch.testing.assert_close(
        policy.std.detach(), torch.tensor([0.022, 0.032])
    )


def _economy_activation(samples: int) -> dict:
    return {
        "event": "hope_effective_reward_activation_update",
        "task_kind": "action_ball",
        "ppo_update": 0,
        "environment_step_count": 24,
        "num_envs": 4096,
        "observed_sample_count": samples,
        "recipe_sha256": "a" * 64,
        "total_weighted_reward_sum": 3.0,
        "reward_cache_contract": {
            "total_reward_closure": "validated",
            "max_abs_error": 0.0,
        },
        "terms": [
            {
                "name": f"term_{index:02d}",
                "observed_sample_count": samples,
                "raw_sum": 1.0,
                "weighted_sum": 0.1,
                "raw_recomposition_max_abs_error": 0.0,
            }
            for index in range(30)
        ],
    }


def test_reward_ppo_economy_rollout_snapshot_covers_required_fields(
    runner_module,
):
    samples = 4096 * 24
    raw_advantage = torch.linspace(-2.0, 2.0, samples).reshape(24, 4096, 1)
    normalized = (
        raw_advantage - raw_advantage.mean()
    ) / (raw_advantage.std() + 1.0e-8)
    runner = _runner(runner_module, empirical=False)
    runner.alg = SimpleNamespace(
        storage=SimpleNamespace(
            rewards=torch.ones(24, 4096, 1),
            returns=raw_advantage + 0.5,
            values=torch.full((24, 4096, 1), 0.5),
            advantages=normalized,
        )
    )
    snapshot = runner._prepare_reward_ppo_economy_rollout(
        activation=_economy_activation(samples), ppo_update=0
    )
    assert set(snapshot) == {"reward", "advantage"}
    assert snapshot["reward"]["reward_manager_total_sum"] == 3.0
    assert snapshot["reward"]["return_std"] > 0.0
    assert math.isfinite(snapshot["reward"]["explained_variance"])
    assert snapshot["reward"]["pre_advantage_reward_semantics"] == (
        "ppo_storage_reward_after_timeout_bootstrap"
    )
    assert len(snapshot["reward"]["per_term_raw_sum"]) == 30
    assert set(snapshot["reward"]["per_term_eligible_denominator"].values()) == {
        samples
    }
    post = snapshot["advantage"]["post_normalization_mean_std_min_max"]
    assert post["mean"] == pytest.approx(0.0, abs=5.0e-5)
    assert post["std"] == pytest.approx(1.0, abs=5.0e-5)


def test_reward_ppo_economy_gate_is_explicit_and_exact_4096x24(
    runner_module, monkeypatch
):
    runner = _runner(runner_module, empirical=False)
    runner.num_steps_per_env = 24
    runner.rank = 0
    runner.alg = SimpleNamespace(
        storage=SimpleNamespace(
            training_type="rl",
            num_envs=4096,
            num_transitions_per_env=24,
        )
    )
    runner._action_ball_diagnostic_unauthorized = lambda: True
    monkeypatch.setenv("HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE", "1")
    assert runner._reward_ppo_economy_gate_requested() is True

    runner.alg.storage.num_envs = 1
    with pytest.raises(RuntimeError, match="exact 4096x24"):
        runner._reward_ppo_economy_gate_requested()


def test_reward_ppo_economy_observes_real_gradient_clip_without_replacing_it(
    runner_module,
):
    class EconomyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(2, 2)
            self.critic = torch.nn.Linear(2, 1)
            self.log_std = torch.nn.Parameter(torch.zeros(2))

    policy = EconomyPolicy()
    runner = _runner(runner_module, empirical=False)
    runner.alg = SimpleNamespace(
        policy=policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        max_grad_norm=1.0,
    )
    calls = []

    def update():
        for _ in range(20):
            policy.zero_grad(set_to_none=True)
            loss = (
                policy.actor(torch.ones(3, 2)).sum()
                + policy.critic(torch.ones(3, 2)).sum()
                + policy.log_std.sum()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            calls.append(1)
        return {"surrogate": 0.0, "value_function": 0.0, "entropy": 0.0}

    result, gradient = runner._run_reward_ppo_economy_optimizer(update)
    assert len(calls) == 20
    assert result["surrogate"] == 0.0
    assert gradient["optimizer_minibatch_count"] == 20
    assert gradient["pre_clip_actor_mean_parameter_grad_norm"] > 0.0
    assert gradient["pre_clip_critic_parameter_grad_norm"] > 0.0
    assert gradient["pre_clip_std_parameter_grad_norm"] > 0.0
    assert gradient["post_clip_total_grad_norm"] <= 1.0 + 1.0e-5
    assert gradient[
        "post_clip_total_grad_norm_distribution"
    ]["max"] <= 1.0 + 1.0e-5
    assert gradient["pre_clip_total_grad_norm_distribution"]["mean"] == (
        gradient["pre_clip_total_grad_norm"]
    )
    assert 0.0 <= gradient["clip_factor_distribution"]["min"] <= 1.0
