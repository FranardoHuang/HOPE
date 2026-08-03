"""Dependency-free fail-closed tests for the trainable A225 leaf."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK_YAML = (
    ROOT / "cfg/task/HOPEPingPongActionBallA225VendorV2N1Learnability.yaml"
)
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    / "action_ball_225_trainability.py"
)
SPEC = importlib.util.spec_from_file_location("a225_trainability_under_test", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _cfg(*, mode: str = M.A225_ACTOR_CONTRACT, critic=True):
    return SimpleNamespace(
        obs_mode=mode,
        action_ball_225_construction_only=False,
        action_ball_225_trainability_contract=M.A225_TRAINABILITY_CONTRACT,
        critic_obs_contract=M.A225_CRITIC_CONTRACT,
        observations=SimpleNamespace(critic=object() if critic else None),
    )


def _runtime(cfg=None):
    cfg = _cfg() if cfg is None else cfg
    manager = SimpleNamespace(
        active_terms={
            "policy": [name for name, _dim in M.A225_ACTOR_LAYOUT],
            "critic": [name for name, _dim in M.A225_CRITIC_LAYOUT],
        },
        group_obs_term_dim={
            "policy": [(dim,) for _name, dim in M.A225_ACTOR_LAYOUT],
            "critic": [(dim,) for _name, dim in M.A225_CRITIC_LAYOUT],
        },
        group_obs_dim={
            "policy": (M.A225_ACTOR_WIDTH,),
            "critic": (M.A225_CRITIC_WIDTH,),
        },
    )
    runtime = SimpleNamespace(cfg=cfg, observation_manager=manager)
    runtime.unwrapped = runtime
    return runtime


def _wrapped(runtime=None):
    runtime = _runtime() if runtime is None else runtime
    return SimpleNamespace(
        unwrapped=runtime,
        num_obs=M.A225_ACTOR_WIDTH,
        num_privileged_obs=M.A225_CRITIC_WIDTH,
    )


def test_cfg_guard_allows_only_dedicated_a225_and_always_refuses_c225():
    M.validate_action_ball_225_cfg_trainability(_cfg(), entrypoint="test")
    with pytest.raises(RuntimeError, match="construction-only"):
        M.validate_action_ball_225_cfg_trainability(
            _cfg(mode="action_ball_c225"), entrypoint="test"
        )
    with pytest.raises(RuntimeError, match="symmetric actor fallback"):
        M.validate_action_ball_225_cfg_trainability(
            _cfg(critic=False), entrypoint="test"
        )


def test_runtime_and_wrapper_require_exact_actor_and_privileged_critic_abis():
    runtime = _runtime()
    facts = M.validate_action_ball_225_runtime(runtime)
    assert facts["actor_width"] == 225
    assert facts["critic_width"] == 318
    assert facts["fresh_normalizers_required"] is True

    runtime.observation_manager.active_terms["critic"][-1] = "wrong_clock"
    with pytest.raises(RuntimeError, match="critic runtime ABI mismatch"):
        M.validate_action_ball_225_runtime(runtime)

    wrapped = _wrapped()
    wrapped.num_privileged_obs = 225
    with pytest.raises(RuntimeError, match="real 318-D privileged critic"):
        M.validate_action_ball_225_wrapped_env(wrapped)


def test_runner_requires_asymmetric_networks_and_distinct_fresh_normalizers():
    wrapped = _wrapped()
    actor_norm = object()
    critic_norm = object()
    policy = SimpleNamespace(num_actor_obs=225, num_critic_obs=318)
    runner = SimpleNamespace(
        env=wrapped,
        alg=SimpleNamespace(policy=policy),
        empirical_normalization=True,
        _resolve_runtime_normalizer=lambda role: (
            ("obs_normalizer", actor_norm, ())
            if role == "actor"
            else ("privileged_obs_normalizer", critic_norm, ())
        ),
    )
    facts = M.validate_action_ball_225_runner(runner)
    assert facts["runner_actor_width"] == 225
    assert facts["runner_critic_width"] == 318

    policy.num_critic_obs = 225
    with pytest.raises(RuntimeError, match="runner network ABI mismatch"):
        M.validate_action_ball_225_runner(runner)
    policy.num_critic_obs = 318
    runner._resolve_runtime_normalizer = lambda role: (role, actor_norm, ())
    with pytest.raises(RuntimeError, match="distinct objects"):
        M.validate_action_ball_225_runner(runner)

    runner.empirical_normalization = False
    with pytest.raises(RuntimeError, match="fresh empirical"):
        M.validate_action_ball_225_runner(runner)


def test_a225_resolved_cfg_inherits_vendor_v2_adaptive_sigma_contract():
    import yaml

    raw_task = yaml.safe_load(TASK_YAML.read_text(encoding="utf-8"))
    for key in (
        "adaptive_sigma",
        "adaptive_sigma_monotonic",
        "adaptive_sigma_normal",
    ):
        assert key not in raw_task["racket"]

    if importlib.util.find_spec("hydra") is None:
        return
    import hydra

    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((ROOT / "cfg").resolve()),
    ):
        task = hydra.compose(
            config_name="train",
            overrides=[
                "task=HOPEPingPongActionBallA225VendorV2N1Learnability"
            ],
        ).task

    assert task.racket.adaptive_sigma is True
    assert task.racket.adaptive_sigma_monotonic is True
    assert task.racket.adaptive_sigma_normal is True
    assert task.racket.adaptive_sigma_source == "ball_exact_strike"
    assert task.racket.sigma_pos_max == pytest.approx(0.50)
    assert task.racket.sigma_vel_max == pytest.approx(3.0)
    assert task.racket.sigma_normal_max == pytest.approx(2.10)
