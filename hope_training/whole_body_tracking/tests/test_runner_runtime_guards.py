"""Host-only guards for the live RSL-RL normalization/std ABI."""

from __future__ import annotations

import importlib.util
import inspect
import io
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
        self.forward_calls = 0

    def forward(self, values):
        self.forward_calls += 1
        if values.ndim != 2 or values.shape[1] != self._mean.shape[1]:
            raise RuntimeError("test normalizer width mismatch")
        if self.training and values.shape[0] > 0:
            detached = values.detach()
            batch_count = float(detached.shape[0])
            batch_mean = detached.mean(dim=0, keepdim=True)
            batch_var = detached.var(dim=0, unbiased=False, keepdim=True)
            old_count = float(self.count.item())
            total_count = old_count + batch_count
            delta = batch_mean - self._mean
            combined_mean = self._mean + delta * (batch_count / total_count)
            combined_m2 = (
                self._var * old_count
                + batch_var * batch_count
                + delta.square() * old_count * batch_count / total_count
            )
            with torch.no_grad():
                self._mean.copy_(combined_mean)
                self._var.copy_(combined_m2 / total_count)
                self._std.copy_(torch.sqrt(self._var))
                self.count.fill_(int(total_count))
        return (values - self._mean) / (self._std + 1.0e-2)


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


@pytest.mark.parametrize(
    ("preflight_attribute", "receipt_key", "actor_identity", "critic_identity"),
    [
        (
            "action_ball_a211_trainability_preflight",
            "a211_trainability",
            "action_ball_a211_actor_norm_v2",
            "action_ball_a211_critic_norm_v1",
        ),
        (
            "action_ball_c211_trainability_preflight",
            "c211_trainability",
            "action_ball_c211_actor_norm_v2",
            "action_ball_c211_critic_norm_v1",
        ),
    ],
)
def test_a211_c211_normalizers_are_fresh_211_319_and_separately_identified(
    runner_module,
    preflight_attribute,
    receipt_key,
    actor_identity,
    critic_identity,
):
    runner = _runner(runner_module, empirical=True)
    runner.obs_normalizer = _EmpiricalNormalizer(211)
    runner.privileged_obs_normalizer = _EmpiricalNormalizer(319)
    runner.action_ball_a211_trainability_preflight = None
    runner.action_ball_c211_trainability_preflight = None
    setattr(runner, preflight_attribute, {"actor_width": 211, "critic_width": 319})

    binding = runner._validate_training_normalizers()

    assert binding[receipt_key] == {"actor_width": 211, "critic_width": 319}
    assert binding["normalizers"]["actor"]["contract_identity"] == actor_identity
    assert binding["normalizers"]["critic"]["contract_identity"] == critic_identity


def _fresh_action_ball_211_runner(runner_module, preflight_attribute):
    runner = _runner(runner_module, empirical=True)
    runner.obs_normalizer = _EmpiricalNormalizer(211)
    runner.privileged_obs_normalizer = _EmpiricalNormalizer(319)
    runner.action_ball_a211_trainability_preflight = None
    runner.action_ball_c211_trainability_preflight = None
    setattr(
        runner,
        preflight_attribute,
        {"actor_width": 211, "critic_width": 319},
    )
    runner.device = "cpu"
    runner.privileged_obs_type = "critic"
    runner._install_action_ball_211_wait_normalizer_masks()
    return runner


def _active_211_rows(width, mask_start, mask_stop):
    rows = torch.zeros(4, width, dtype=torch.float32)
    row_scale = torch.arange(1, 5, dtype=torch.float32).unsqueeze(1)
    feature_scale = torch.arange(
        1, mask_stop - mask_start + 1, dtype=torch.float32
    ).unsqueeze(0)
    rows[:, mask_start:mask_stop] = row_scale * feature_scale
    rows[:, -1] = 1.0
    return rows


def _wait_211_rows(width, mask_start, mask_stop):
    rows = torch.full((2, width), 7.0, dtype=torch.float32)
    rows[:, mask_start:mask_stop] = 0.0
    rows[:, -1] = 0.0
    return rows


def _assert_exact_zero(tensor, start, stop):
    region = tensor[..., start:stop]
    assert torch.equal(region, torch.zeros_like(region))


def test_action_ball_211_initial_normalization_is_wired_and_scope_is_leaf_only(
    runner_module,
):
    learn_source = inspect.getsource(runner_module.MotionOnPolicyRunner.learn)
    patch = learn_source.index(
        "self.env.get_observations = get_normalized_initial_observations"
    )
    base_learn = learn_source.index("super().learn(")
    restore = learn_source.index(
        "self.env.get_observations = original_get_observations"
    )
    assert patch < base_learn < restore
    assert "self._normalize_action_ball_211_initial_observations(" in learn_source

    legacy = _runner(runner_module, empirical=True)
    actor = legacy.obs_normalizer
    critic = legacy.privileged_obs_normalizer
    legacy._install_action_ball_211_wait_normalizer_masks()
    assert legacy.obs_normalizer is actor
    assert legacy.privileged_obs_normalizer is critic
    assert len(actor._forward_hooks) == 0
    assert len(critic._forward_hooks) == 0


@pytest.mark.parametrize(
    "preflight_attribute",
    [
        "action_ball_a211_trainability_preflight",
        "action_ball_c211_trainability_preflight",
    ],
)
def test_action_ball_211_wait_mask_uses_raw_validity_for_initial_next_and_bootstrap(
    runner_module, preflight_attribute
):
    runner = _fresh_action_ball_211_runner(
        runner_module, preflight_attribute
    )
    actor_normalizer = runner.obs_normalizer
    critic_normalizer = runner.privileged_obs_normalizer

    # Warm the exact live moments with TASK_ACTIVE rows first.  A later raw
    # zero would therefore become nonzero without the post-normalization mask.
    active_actor = _active_211_rows(211, 197, 210)
    active_critic = _active_211_rows(319, 305, 318)
    actor_normalizer(active_actor)
    critic_normalizer(active_critic)
    assert actor_normalizer.count.item() == 4
    assert critic_normalizer.count.item() == 4

    raw_wait_actor = _wait_211_rows(211, 197, 210)
    raw_wait_critic = _wait_211_rows(319, 305, 318)
    raw_extras = {
        "observations": {"critic": raw_wait_critic},
        "sentinel": "preserved",
    }

    # This is the one initial getter call patched ahead of the first alg.act
    # (and hence ahead of the first RolloutStorage insert).
    initial_actor, initial_extras = (
        runner._normalize_action_ball_211_initial_observations(
            (raw_wait_actor, raw_extras)
        )
    )
    initial_critic = initial_extras["observations"]["critic"]
    assert actor_normalizer.forward_calls == 2
    assert critic_normalizer.forward_calls == 2
    assert actor_normalizer.count.item() == 6
    assert critic_normalizer.count.item() == 6
    _assert_exact_zero(initial_actor, 197, 210)
    _assert_exact_zero(initial_critic, 305, 318)
    assert torch.count_nonzero(initial_actor[:, -1]).item() == 2
    assert torch.count_nonzero(initial_critic[:, -1]).item() == 2
    assert initial_extras["sentinel"] == "preserved"
    assert raw_extras["observations"]["critic"] is raw_wait_critic

    # Upstream next-observation calls hit these same modules.  alg.act stores
    # the previous pair, and compute_returns reuses the final normalized
    # critic tensor as bootstrap without another transform.
    storage = [(initial_actor, initial_critic)]
    next_actor = actor_normalizer(raw_wait_actor)
    next_critic = critic_normalizer(raw_wait_critic)
    storage.append((next_actor, next_critic))
    bootstrap_critic = next_critic
    assert actor_normalizer.forward_calls == 3
    assert critic_normalizer.forward_calls == 3
    for stored_actor, stored_critic in storage:
        _assert_exact_zero(stored_actor, 197, 210)
        _assert_exact_zero(stored_critic, 305, 318)
    assert bootstrap_critic is next_critic
    _assert_exact_zero(bootstrap_critic, 305, 318)

    # TASK_ACTIVE is not inferred from the normalized last column and must not
    # be cleared even when the probe is far from the accumulated mean.
    actor_normalizer.eval()
    critic_normalizer.eval()
    active_actor_probe = active_actor[:1].clone()
    active_actor_probe[:, 197:210] += 100.0
    active_critic_probe = active_critic[:1].clone()
    active_critic_probe[:, 305:318] += 100.0
    expected_actor = (
        active_actor_probe - actor_normalizer._mean
    ) / (actor_normalizer._std + 1.0e-2)
    expected_critic = (
        active_critic_probe - critic_normalizer._mean
    ) / (critic_normalizer._std + 1.0e-2)
    actual_actor = actor_normalizer(active_actor_probe)
    actual_critic = critic_normalizer(active_critic_probe)
    torch.testing.assert_close(actual_actor, expected_actor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_critic, expected_critic, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(actual_actor[:, 197:210]).item() == 13
    assert torch.count_nonzero(actual_critic[:, 305:318]).item() == 13


@pytest.mark.parametrize(
    "preflight_attribute",
    [
        "action_ball_a211_trainability_preflight",
        "action_ball_c211_trainability_preflight",
    ],
)
def test_action_ball_211_wait_hook_preserves_checkpoint_state_and_formal_restore(
    runner_module, preflight_attribute
):
    source = _fresh_action_ball_211_runner(
        runner_module, preflight_attribute
    )
    source_actor = source.obs_normalizer
    source_critic = source.privileged_obs_normalizer
    source_actor(_active_211_rows(211, 197, 210))
    source_critic(_active_211_rows(319, 305, 318))

    # Idempotent installation keeps the concrete objects and the exact RSL-RL
    # state keys; no wrapper prefix may enter checkpoints or frozen-eval hashes.
    source._install_action_ball_211_wait_normalizer_masks()
    assert source.obs_normalizer is source_actor
    assert source.privileged_obs_normalizer is source_critic
    assert tuple(source_actor.state_dict()) == ("_mean", "_var", "_std", "count")
    assert tuple(source_critic.state_dict()) == ("_mean", "_var", "_std", "count")
    assert len(source_actor._forward_hooks) == 1
    assert len(source_critic._forward_hooks) == 1

    source_policy = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.Adam(source_policy.parameters(), lr=1.0e-3)
    payload = {
        "model_state_dict": source_policy.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
        "obs_norm_state_dict": source_actor.state_dict(),
        "privileged_obs_norm_state_dict": source_critic.state_dict(),
        "iter": 7,
        "infos": {"sentinel": "roundtrip"},
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored_payload = torch.load(stream, map_location="cpu", weights_only=False)

    restored = _fresh_action_ball_211_runner(
        runner_module, preflight_attribute
    )
    restored_actor = restored.obs_normalizer
    restored_critic = restored.privileged_obs_normalizer
    restored_policy = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(
        restored_policy.parameters(), lr=1.0e-3
    )
    restored.alg = SimpleNamespace(
        policy=restored_policy,
        optimizer=restored_optimizer,
    )
    infos = restored._apply_formal_preloaded_checkpoint(
        restored_payload,
        load_optimizer=True,
        prefix="test ActionBall211 checkpoint",
    )
    assert infos == {"sentinel": "roundtrip"}
    assert restored.current_learning_iteration == 7
    assert restored.obs_normalizer is restored_actor
    assert restored.privileged_obs_normalizer is restored_critic
    for key, value in source_actor.state_dict().items():
        torch.testing.assert_close(restored_actor.state_dict()[key], value)
    for key, value in source_critic.state_dict().items():
        torch.testing.assert_close(restored_critic.state_dict()[key], value)

    restored_actor.eval()
    restored_critic.eval()
    normalized_wait_actor = restored_actor(_wait_211_rows(211, 197, 210))
    normalized_wait_critic = restored_critic(_wait_211_rows(319, 305, 318))
    _assert_exact_zero(normalized_wait_actor, 197, 210)
    _assert_exact_zero(normalized_wait_critic, 305, 318)


def test_a211_normalizer_rejects_legacy_225_actor_width(runner_module):
    runner = _runner(runner_module, empirical=True)
    runner.obs_normalizer = _EmpiricalNormalizer(225)
    runner.privileged_obs_normalizer = _EmpiricalNormalizer(319)
    runner.action_ball_a211_trainability_preflight = {"actor_width": 211}
    runner.action_ball_c211_trainability_preflight = None

    with pytest.raises(RuntimeError, match="A211 actor normalizer"):
        runner._validate_training_normalizers()


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
