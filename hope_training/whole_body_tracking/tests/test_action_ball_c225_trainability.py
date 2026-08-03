"""Dependency-free fail-closed tests for the trainable C225 leaf."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    / "action_ball_c225_trainability.py"
)
ENV_CFG_PATH = (
    MODULE_PATH.parent / "config/agibot_a3/hope_env_cfg.py"
)
REGISTRY_PATH = ENV_CFG_PATH.with_name("__init__.py")
TASK_YAML = (
    ROOT / "cfg/task/HOPEPingPongActionBallC225VendorV2N1Learnability.yaml"
)
SPEC = importlib.util.spec_from_file_location("c225_trainability_under_test", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _cfg(*, critic=True):
    return SimpleNamespace(
        obs_mode=M.C225_ACTOR_CONTRACT,
        action_ball_225_construction_only=False,
        action_ball_225_trainability_contract=M.C225_TRAINABILITY_CONTRACT,
        critic_obs_contract=M.C225_CRITIC_CONTRACT,
        observations=SimpleNamespace(critic=object() if critic else None),
    )


def _runtime(cfg=None):
    cfg = _cfg() if cfg is None else cfg
    manager = SimpleNamespace(
        active_terms={
            "policy": [name for name, _dim in M.C225_ACTOR_LAYOUT],
            "critic": [name for name, _dim in M.C225_CRITIC_LAYOUT],
        },
        group_obs_term_dim={
            "policy": [(dim,) for _name, dim in M.C225_ACTOR_LAYOUT],
            "critic": [(dim,) for _name, dim in M.C225_CRITIC_LAYOUT],
        },
        group_obs_dim={
            "policy": (M.C225_ACTOR_WIDTH,),
            "critic": (M.C225_CRITIC_WIDTH,),
        },
    )
    runtime = SimpleNamespace(cfg=cfg, observation_manager=manager)
    runtime.unwrapped = runtime
    return runtime


def _wrapped(runtime=None):
    runtime = _runtime() if runtime is None else runtime
    return SimpleNamespace(
        unwrapped=runtime,
        num_obs=M.C225_ACTOR_WIDTH,
        num_privileged_obs=M.C225_CRITIC_WIDTH,
    )


def test_c225_exact_actor_and_independent_critic_abis_have_no_contact_target():
    assert M.C225_ACTOR_WIDTH == 225
    assert M.C225_CRITIC_WIDTH == 318
    assert M.C225_ACTOR_LAYOUT[10:13] == (
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
    )
    assert M.C225_CRITIC_LAYOUT[11:14] == M.C225_ACTOR_LAYOUT[10:13]
    actor_offset = sum(dim for _name, dim in M.C225_ACTOR_LAYOUT[:10])
    assert actor_offset == 212
    for layout in (M.C225_ACTOR_LAYOUT, M.C225_CRITIC_LAYOUT):
        names = " ".join(name for name, _dim in layout).lower()
        assert "desired_contact" not in names
        assert "contact_face" not in names
        assert "table_midpoint" not in names


def test_c225_cfg_runtime_wrapper_and_runner_fail_closed_on_fallbacks():
    M.validate_action_ball_c225_cfg_trainability(_cfg(), entrypoint="test")
    with pytest.raises(RuntimeError, match="construction-only"):
        wrong = _cfg()
        wrong.action_ball_225_trainability_contract = "action_ball_a225_fixed_question_learnability_v1"
        M.validate_action_ball_c225_cfg_trainability(wrong, entrypoint="test")
    with pytest.raises(RuntimeError, match="symmetric actor fallback"):
        M.validate_action_ball_c225_cfg_trainability(_cfg(critic=False), entrypoint="test")

    facts = M.validate_action_ball_c225_runtime(_runtime())
    assert facts["actor_width"] == 225
    assert facts["critic_width"] == 318
    assert facts["contact_target_absent"] is True

    runtime = _runtime()
    runtime.observation_manager.active_terms["critic"][11] = (
        "task_desired_contact_position_heading"
    )
    with pytest.raises(RuntimeError, match="critic runtime ABI mismatch"):
        M.validate_action_ball_c225_runtime(runtime)

    wrapped = _wrapped()
    wrapped.num_privileged_obs = 225
    with pytest.raises(RuntimeError, match="real 318-D privileged critic"):
        M.validate_action_ball_c225_wrapped_env(wrapped)

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
    assert M.validate_action_ball_c225_runner(runner)["runner_critic_width"] == 318
    runner._resolve_runtime_normalizer = lambda role: (role, actor_norm, ())
    with pytest.raises(RuntimeError, match="distinct objects"):
        M.validate_action_ball_c225_runner(runner)


def test_c225_trainable_cfg_gym_and_yaml_are_dedicated_and_dense_outcome_enabled():
    env_source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(env_source, filename=str(ENV_CFG_PATH))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    critic_outer = classes["HOPEActionBallC225TrainableObservationsCfg"]
    critic = next(
        node for node in critic_outer.body if isinstance(node, ast.ClassDef)
    )
    critic_tail = tuple(
        target.id
        for child in critic.body
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call)
        for target in child.targets
        if isinstance(target, ast.Name)
    )
    assert critic_tail == tuple(name for name, _dim in M.C225_CRITIC_LAYOUT[10:])
    assert "desired_contact" not in ast.get_source_segment(env_source, critic_outer)

    leaf_source = ast.get_source_segment(
        env_source,
        classes["HOPEPingPongActionBallC225LearnabilityAgibotA3EnvCfg"],
    )
    assert leaf_source is not None
    assert "C225_TRAINABILITY_CONTRACT" in leaf_source
    assert "C225_CRITIC_CONTRACT" in leaf_source
    assert "HOPEActionBallC225TrainableObservationsCfg" in leaf_source

    registry = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "HOPE-PingPong-ActionBall-C225Learnability-AgibotA3-v0" in registry
    assert "HOPEPingPongActionBallC225LearnabilityAgibotA3EnvCfg" in registry

    import yaml

    raw_task = yaml.safe_load(TASK_YAML.read_text(encoding="utf-8"))
    assert raw_task["actor_obs_contract"] == "action_ball_c225"
    assert raw_task["racket"]["action_ball_target_recipe"] == "outcome_dense_only"
    assert raw_task["racket"]["action_ball_target_validity_mask"] == [False, False, False]
    vendor_v2 = yaml.safe_load(
        (ROOT / "cfg/task/HOPEPingPongActionBallA3VendorV2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert vendor_v2["rewards"]["strike_capture_bonus_weight"] == pytest.approx(25.0)
    assert vendor_v2["rewards"]["virtual_pass_net_weight"] == pytest.approx(20.0)
    assert vendor_v2["rewards"]["virtual_landing_dense_weight"] == pytest.approx(20.0)
    assert vendor_v2["rewards"]["virtual_landing_weight"] == pytest.approx(500.0)

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
                "task=HOPEPingPongActionBallC225VendorV2N1Learnability"
            ],
        ).task
    assert task.actor_obs_contract == "action_ball_c225"
    assert task.gym_task == "HOPE-PingPong-ActionBall-C225Learnability-AgibotA3-v0"
    assert task.racket.action_ball_target_source == "immutable_tape"
    assert task.racket.action_ball_target_recipe == "outcome_dense_only"
    assert list(task.racket.action_ball_target_validity_mask) == [False, False, False]
    assert task.racket.action_ball_target_observation_noise is False
    # C has no desired-contact Reward, but it is not sparse-only: these
    # achieved-contact/post-contact forward-outcome terms remain non-zero.
    assert task.rewards.strike_capture_bonus_weight == pytest.approx(25.0)
    assert task.rewards.virtual_pass_net_weight == pytest.approx(20.0)
    assert task.rewards.virtual_landing_dense_weight == pytest.approx(20.0)
    assert task.rewards.virtual_landing_weight == pytest.approx(500.0)
