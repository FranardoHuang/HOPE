"""Dependency-free fail-closed tests for the fresh trainable C211 leaf."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    / "action_ball_c211_trainability.py"
)
ENV_CFG_PATH = (
    MODULE_PATH.parent / "config/agibot_a3/hope_env_cfg.py"
)
REGISTRY_PATH = ENV_CFG_PATH.with_name("__init__.py")
TASK_YAML = (
    ROOT / "cfg/task/HOPEPingPongActionBallC211VendorV2N1Learnability.yaml"
)
COMMAND_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "hope_commands.py"
)
SPEC = importlib.util.spec_from_file_location("c211_trainability_under_test", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _cfg(*, critic=True):
    return SimpleNamespace(
        obs_mode=M.C211_ACTOR_CONTRACT,
        action_ball_211_construction_only=False,
        action_ball_211_trainability_contract=M.C211_TRAINABILITY_CONTRACT,
        critic_obs_contract=M.C211_CRITIC_CONTRACT,
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(
                action_ball_task_wait_enabled=True,
                action_ball_task_wait_policy_dt_s=0.02,
                action_ball_task_wait_seed=20260804,
                action_ball_task_wait_min_wait_ticks=5,
                action_ball_task_wait_max_wait_ticks=25,
                action_ball_task_wait_episode_horizon_ticks=500,
                action_ball_task_wait_required_active_ticks=200,
                action_ball_target_source="direct_ball",
                action_ball_reuse_exact_question_until_semantics_change=False,
                action_ball_initial_center_single_question=True,
                action_ball_target_recipe="outcome_dense_only",
                action_ball_target_validity_mask=[False, False, False],
            )
        ),
        observations=SimpleNamespace(critic=object() if critic else None),
    )


def _runtime(cfg=None):
    cfg = _cfg() if cfg is None else cfg
    manager = SimpleNamespace(
        active_terms={
            "policy": [name for name, _dim in M.C211_ACTOR_LAYOUT],
            "critic": [name for name, _dim in M.C211_CRITIC_LAYOUT],
        },
        group_obs_term_dim={
            "policy": [(dim,) for _name, dim in M.C211_ACTOR_LAYOUT],
            "critic": [(dim,) for _name, dim in M.C211_CRITIC_LAYOUT],
        },
        group_obs_dim={
            "policy": (M.C211_ACTOR_WIDTH,),
            "critic": (M.C211_CRITIC_WIDTH,),
        },
    )
    runtime = SimpleNamespace(cfg=cfg, observation_manager=manager)
    runtime.unwrapped = runtime
    return runtime


def _wrapped(runtime=None):
    runtime = _runtime() if runtime is None else runtime
    return SimpleNamespace(
        unwrapped=runtime,
        num_obs=M.C211_ACTOR_WIDTH,
        num_privileged_obs=M.C211_CRITIC_WIDTH,
    )


def test_c211_exact_actor_and_independent_critic_abis_have_no_contact_target():
    assert M.C211_ACTOR_WIDTH == 211
    assert M.C211_CRITIC_WIDTH == 319
    assert M.C211_ACTOR_LAYOUT[10:13] == (
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
    )
    assert M.C211_CRITIC_LAYOUT[11:14] == M.C211_ACTOR_LAYOUT[10:13]
    actor_offset = sum(dim for _name, dim in M.C211_ACTOR_LAYOUT[:10])
    assert actor_offset == 197
    for layout in (M.C211_ACTOR_LAYOUT, M.C211_CRITIC_LAYOUT):
        names = " ".join(name for name, _dim in layout).lower()
        assert "desired_contact" not in names
        assert "contact_face" not in names
        assert "table_midpoint" not in names
        assert layout[-1] == ("task_valid", 1)
    assert "teacher_base_now_world" not in M.layout_names(M.C211_ACTOR_LAYOUT)


def test_c211_cfg_runtime_wrapper_and_runner_fail_closed_on_fallbacks():
    M.validate_action_ball_c211_cfg_trainability(_cfg(), entrypoint="test")
    with pytest.raises(RuntimeError, match="construction-only"):
        wrong = _cfg()
        wrong.action_ball_211_trainability_contract = "action_ball_a211_fixed_question_learnability_v2"
        M.validate_action_ball_c211_cfg_trainability(wrong, entrypoint="test")
    with pytest.raises(RuntimeError, match="symmetric actor fallback"):
        M.validate_action_ball_c211_cfg_trainability(_cfg(critic=False), entrypoint="test")

    for attribute, bad_value in (
        ("action_ball_target_source", "online_solver"),
        ("action_ball_reuse_exact_question_until_semantics_change", True),
        ("action_ball_initial_center_single_question", False),
        ("action_ball_target_recipe", "current_lm"),
        ("action_ball_target_validity_mask", [False, True, False]),
    ):
        wrong = _cfg()
        setattr(wrong.commands.racket_target, attribute, bad_value)
        with pytest.raises(RuntimeError, match=attribute):
            M.validate_action_ball_c211_cfg_trainability(wrong, entrypoint="test")

    facts = M.validate_action_ball_c211_runtime(_runtime())
    assert facts["actor_width"] == 211
    assert facts["critic_width"] == 319
    assert facts["contact_target_absent"] is True
    source = facts["question_source_contract"]
    assert source["family"] == "C211"
    assert source["question_sampler"]["sampler_runs_every_reset"] is True
    assert source["question_sampler"][
        "curriculum_domain_levels_consulted_every_reset"
    ] is True
    provider = source["target_provider"]
    assert provider == {
        "source": "direct_ball",
        "desired_contact_inverse": False,
        "exact_question_answer_cache": {"enabled": False},
        "online_inverse_solves_per_reset": 0,
        "online_inverse_solves_per_step": 0,
    }
    assert facts["c225_reward_contract"] == M.c211_reward_contract_facts()
    assert facts["c225_reward_contract"]["landing"][
        "observed_physical_landing_available"
    ] is False

    runtime = _runtime()
    runtime.observation_manager.active_terms["critic"][11] = (
        "task_desired_contact_position_heading"
    )
    with pytest.raises(RuntimeError, match="critic runtime ABI mismatch"):
        M.validate_action_ball_c211_runtime(runtime)

    wrapped = _wrapped()
    wrapped.num_privileged_obs = 211
    with pytest.raises(RuntimeError, match="real 319-D privileged critic"):
        M.validate_action_ball_c211_wrapped_env(wrapped)

    wrapped = _wrapped()
    actor_norm = object()
    critic_norm = object()
    policy = SimpleNamespace(num_actor_obs=211, num_critic_obs=319)
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
    assert M.validate_action_ball_c211_runner(runner)["runner_critic_width"] == 319
    runner._resolve_runtime_normalizer = lambda role: (role, actor_norm, ())
    with pytest.raises(RuntimeError, match="distinct objects"):
        M.validate_action_ball_c211_runner(runner)


def test_c211_trainable_cfg_gym_and_yaml_are_dedicated_and_dense_outcome_enabled():
    env_source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(env_source, filename=str(ENV_CFG_PATH))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    critic_outer = classes["HOPEActionBallC211TrainableObservationsCfg"]
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
    assert critic_tail == tuple(name for name, _dim in M.C211_CRITIC_LAYOUT[10:])
    assert "desired_contact" not in ast.get_source_segment(env_source, critic_outer)

    leaf_source = ast.get_source_segment(
        env_source,
        classes["HOPEPingPongActionBallC211LearnabilityAgibotA3EnvCfg"],
    )
    assert leaf_source is not None
    assert "C211_TRAINABILITY_CONTRACT" in leaf_source
    assert "C211_CRITIC_CONTRACT" in leaf_source
    assert "HOPEActionBallC211TrainableObservationsCfg" in leaf_source

    c_reward_source = ast.get_source_segment(
        env_source, classes["HOPEActionBallC211RewardsCfg"]
    )
    assert c_reward_source is not None
    assert "c225_strike_ball_paddle_center_proximity" in c_reward_source
    assert "c225_landing_outcome_actual_contact" in c_reward_source
    assert "weight=240.0" in c_reward_source
    c_base_source = ast.get_source_segment(
        env_source, classes["HOPEPingPongActionBallC211AgibotA3EnvCfg"]
    )
    assert "HOPEActionBallC211RewardsCfg" in c_base_source

    registry = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "HOPE-PingPong-ActionBall-C211Learnability-AgibotA3-v0" in registry
    assert "HOPEPingPongActionBallC211LearnabilityAgibotA3EnvCfg" in registry

    import yaml

    raw_task = yaml.safe_load(TASK_YAML.read_text(encoding="utf-8"))
    assert raw_task["actor_obs_contract"] == "action_ball_c211"
    assert raw_task["racket"]["action_ball_target_recipe"] == "outcome_dense_only"
    assert raw_task["racket"]["action_ball_target_validity_mask"] == [False, False, False]
    c_rewards = raw_task["rewards"]
    for name in (
        "racket_position_coarse_weight",
        "racket_velocity_coarse_weight",
        "racket_normal_coarse_weight",
        "racket_position_weight",
        "racket_velocity_weight",
        "racket_normal_weight",
        "racket_position_precision_weight",
        "racket_velocity_precision_weight",
        "racket_normal_precision_weight",
        "virtual_pass_net_weight",
        "virtual_landing_dense_weight",
    ):
        assert c_rewards[name] == 0.0
    assert c_rewards["strike_capture_bonus_weight"] == 0.0
    assert c_rewards["virtual_landing_weight"] == pytest.approx(700.0)
    assert c_rewards["virtual_landing_base_frac"] == pytest.approx(0.6)

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
                "task=HOPEPingPongActionBallC211VendorV2N1Learnability"
            ],
        ).task
    assert task.actor_obs_contract == "action_ball_c211"
    assert task.gym_task == "HOPE-PingPong-ActionBall-C211Learnability-AgibotA3-v0"
    assert task.racket.action_ball_target_source == "direct_ball"
    assert (
        task.racket.action_ball_reuse_exact_question_until_semantics_change
        is False
    )
    assert task.racket.action_ball_target_recipe == "outcome_dense_only"
    assert list(task.racket.action_ball_target_validity_mask) == [False, False, False]
    assert task.racket.action_ball_target_observation_noise is False
    # C has one fixed-time miss bridge (owned by its reward class) and one
    # contact-gated landing hierarchy.  Independent hit/pass/dense terms and
    # all desired-contact terms stay off after composition.
    assert task.rewards.strike_capture_bonus_weight == 0.0
    assert task.rewards.virtual_pass_net_weight == 0.0
    assert task.rewards.virtual_landing_dense_weight == 0.0
    assert task.rewards.virtual_landing_weight == pytest.approx(700.0)
    assert task.rewards.virtual_landing_base_frac == pytest.approx(0.6)
    assert task.rewards.racket_position_weight == 0.0
    assert task.rewards.racket_velocity_weight == 0.0
    assert task.rewards.racket_normal_weight == 0.0


def test_c211_validity_000_skips_unused_velocity_and_face_metric_kernels():
    """Keep the C hot path cheap without dropping its position progress norm."""

    source = COMMAND_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(COMMAND_PATH))
    command_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    update_metrics = next(
        node
        for node in command_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_metrics"
    )
    segment = ast.get_source_segment(source, update_metrics)
    assert segment is not None
    assert "racket_pos_err_vec = self.racket_pos_w - self.racket_target_pos_w" in segment
    assert "pos_err = torch.norm(racket_pos_err_vec, dim=-1)" in segment

    validity_branches = {
        node.test.id: ast.get_source_segment(source, node)
        for node in ast.walk(update_metrics)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id in {"target_vel_valid", "target_face_valid"}
    }
    assert set(validity_branches) == {"target_vel_valid", "target_face_valid"}
    velocity_branch = validity_branches["target_vel_valid"]
    face_branch = validity_branches["target_face_valid"]
    assert velocity_branch is not None and face_branch is not None
    assert "vel_err = torch.norm" in velocity_branch
    assert "vel_err = torch.zeros_like(pos_err)" in velocity_branch
    assert "face_tracking_pair(self)" in face_branch
    assert "normal_err_rad = torch.acos" in face_branch
    assert "normal_err_rad = torch.zeros_like(pos_err)" in face_branch


def test_c211_post_dt_reward_economics_have_margin_and_a_nonzero_tail():
    contract = M.c211_reward_contract_facts()
    assert contract["identity"] == "action_ball_c211_achieved_outcome_reward_v3"
    assert contract["task_valid_required"] is True
    bridge = contract["strike_bridge"]
    economics = contract["economics"]
    assert bridge["weight"] == 240.0
    assert bridge["std_m"] == 0.15
    assert (
        economics["task_valid_swing_mimic_discounted_cap"]
        < economics["strike_bridge_discounted_at_contact"]
        < economics["legal_landing_discounted_at_contact_min"]
    )

    amplitude = bridge["weight"] * economics["policy_dt_s"]
    sigma = bridge["std_m"]
    expected = {
        0.0: (4.8, 0.0),
        0.075: (3.84, -20.48),
        0.15: (2.4, -16.0),
        0.30: (0.96, -5.12),
        0.45: (0.48, -1.92),
        0.90: (0.12972972972972974, -0.2804967129291454),
    }
    for distance, (expected_income, expected_gradient) in expected.items():
        ratio = distance / sigma
        income = amplitude / (1.0 + ratio * ratio)
        gradient = -(2.0 * amplitude / sigma) * ratio / math.pow(
            1.0 + ratio * ratio, 2
        )
        assert income == pytest.approx(expected_income)
        assert gradient == pytest.approx(expected_gradient)
    assert expected[0.90][0] > 0.0
    for kernel in (0.0, 0.1, 0.5, 1.0):
        legal_quality = 0.6 + 0.4 * kernel
        off_table_quality = 0.5 * kernel
        assert off_table_quality <= 0.5 * legal_quality
