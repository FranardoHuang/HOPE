"""Dependency-free fail-closed tests for the fresh trainable A211 leaf."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/train.py"
TASK_YAML = (
    ROOT / "cfg/task/HOPEPingPongActionBallA211VendorV2N1Learnability.yaml"
)
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    / "action_ball_a211_trainability.py"
)
SPEC = importlib.util.spec_from_file_location("a211_trainability_under_test", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _cfg(*, mode: str = M.A211_ACTOR_CONTRACT, critic=True):
    return SimpleNamespace(
        obs_mode=mode,
        action_ball_211_construction_only=False,
        action_ball_211_trainability_contract=M.A211_TRAINABILITY_CONTRACT,
        critic_obs_contract=M.A211_CRITIC_CONTRACT,
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(
                action_ball_task_wait_enabled=True,
                action_ball_task_wait_policy_dt_s=0.02,
                action_ball_task_wait_seed=20260804,
                action_ball_task_wait_min_wait_ticks=5,
                action_ball_task_wait_max_wait_ticks=25,
                action_ball_task_wait_episode_horizon_ticks=500,
                action_ball_task_wait_required_active_ticks=200,
                action_ball_target_source="online_solver",
                action_ball_reuse_exact_question_until_semantics_change=True,
                action_ball_initial_center_single_question=True,
                action_ball_target_recipe="current_lm",
                action_ball_target_validity_mask=[True, True, True],
            )
        ),
        observations=SimpleNamespace(critic=object() if critic else None),
    )


def _runtime(cfg=None):
    cfg = _cfg() if cfg is None else cfg
    manager = SimpleNamespace(
        active_terms={
            "policy": [name for name, _dim in M.A211_ACTOR_LAYOUT],
            "critic": [name for name, _dim in M.A211_CRITIC_LAYOUT],
        },
        group_obs_term_dim={
            "policy": [(dim,) for _name, dim in M.A211_ACTOR_LAYOUT],
            "critic": [(dim,) for _name, dim in M.A211_CRITIC_LAYOUT],
        },
        group_obs_dim={
            "policy": (M.A211_ACTOR_WIDTH,),
            "critic": (M.A211_CRITIC_WIDTH,),
        },
    )
    runtime = SimpleNamespace(cfg=cfg, observation_manager=manager)
    runtime.unwrapped = runtime
    return runtime


def _wrapped(runtime=None):
    runtime = _runtime() if runtime is None else runtime
    return SimpleNamespace(
        unwrapped=runtime,
        num_obs=M.A211_ACTOR_WIDTH,
        num_privileged_obs=M.A211_CRITIC_WIDTH,
    )


def test_cfg_guard_allows_only_dedicated_a211_and_refuses_legacy_abis():
    M.validate_action_ball_211_cfg_trainability(_cfg(), entrypoint="test")
    wrong_wait = _cfg()
    wrong_wait.commands.racket_target.action_ball_task_wait_min_wait_ticks = 4
    with pytest.raises(RuntimeError, match="min_wait_ticks"):
        M.validate_action_ball_211_cfg_trainability(wrong_wait, entrypoint="test")
    for legacy in ("action_ball_a225", "action_ball_c225", "action_ball_a210"):
        with pytest.raises(RuntimeError, match="not consumable"):
            M.validate_action_ball_211_cfg_trainability(
                _cfg(mode=legacy), entrypoint="test"
            )
    with pytest.raises(RuntimeError, match="symmetric actor fallback"):
        M.validate_action_ball_211_cfg_trainability(
            _cfg(critic=False), entrypoint="test"
        )

    for attribute, bad_value in (
        ("action_ball_target_source", "direct_ball"),
        ("action_ball_reuse_exact_question_until_semantics_change", False),
        ("action_ball_initial_center_single_question", False),
        ("action_ball_target_recipe", "outcome_dense_only"),
        ("action_ball_target_validity_mask", [True, False, True]),
    ):
        wrong = _cfg()
        setattr(wrong.commands.racket_target, attribute, bad_value)
        with pytest.raises(RuntimeError, match=attribute):
            M.validate_action_ball_211_cfg_trainability(wrong, entrypoint="test")


def test_validity_mask_accepts_the_tuple_the_runtime_actually_holds():
    """人话:Hydra / configclass 会把序列固化成 tuple。

    这道门原来拿 list 直接 `==` tuple,于是完全合法的 (True, True, True) 被判成漂移 ——
    A211/C211 四格流水线的第一站(materialize)就死在这个假阴性上,一次都没跑起来。
    现在 tuple 与 list 都认,但要求逐元素是**显式 bool**、长度一致,比原来更严。
    """

    tupled = _cfg()
    tupled.commands.racket_target.action_ball_target_validity_mask = (
        True,
        True,
        True,
    )
    M.validate_action_ball_211_cfg_trainability(tupled, entrypoint="test")

    for bad_value in (
        (True, False, True),          # 值不对
        (True, True),                 # 长度不对
        (True, True, True, True),     # 长度不对
        (1, 1, 1),                    # 不是显式 bool(原来的 `==` 会放行!)
        "TTT",                        # 不是序列容器
        None,
    ):
        wrong = _cfg()
        wrong.commands.racket_target.action_ball_target_validity_mask = bad_value
        with pytest.raises(RuntimeError, match="action_ball_target_validity_mask"):
            M.validate_action_ball_211_cfg_trainability(wrong, entrypoint="test")


def test_runtime_and_wrapper_require_exact_actor_and_privileged_critic_abis():
    runtime = _runtime()
    facts = M.validate_action_ball_211_runtime(runtime)
    assert facts["actor_width"] == 211
    assert facts["critic_width"] == 319
    assert facts["fresh_normalizers_required"] is True
    assert facts["task_wait_contract"] == M.action_ball_211_wait_contract_facts()
    source = facts["question_source_contract"]
    assert source == M.action_ball_211_question_source_contract_facts(
        family="A211"
    )
    assert source["family"] == "A211"
    sampler = source["question_sampler"]
    assert sampler["curriculum_domain_levels_consulted_every_reset"] is True
    assert sampler["sampler_runs_every_reset"] is True
    assert sampler["sampler_rng_reused_by_target_provider"] is False
    provider = source["target_provider"]
    assert provider["source"] == "online_solver"
    cache = provider["exact_question_answer_cache"]
    assert cache["cold_first_distinct_question_inverse_solve_calls"] == 1
    assert cache["same_batch_identical_question_inverse_solve_calls"] == 0
    assert cache["later_identical_question_inverse_solve_calls"] == 0
    assert cache["changed_question_inverse_solve_calls"] == 1
    assert M.A211_ACTOR_LAYOUT[-1] == ("task_valid", 1)
    assert M.A211_CRITIC_LAYOUT[-1] == ("task_valid", 1)
    assert "teacher_base_now_world" not in M.layout_names(M.A211_ACTOR_LAYOUT)

    runtime.observation_manager.active_terms["critic"][-1] = "wrong_clock"
    with pytest.raises(RuntimeError, match="critic runtime ABI mismatch"):
        M.validate_action_ball_211_runtime(runtime)

    wrapped = _wrapped()
    wrapped.num_privileged_obs = 211
    with pytest.raises(RuntimeError, match="real 319-D privileged critic"):
        M.validate_action_ball_211_wrapped_env(wrapped)


def test_runner_requires_asymmetric_networks_and_distinct_fresh_normalizers():
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
    facts = M.validate_action_ball_211_runner(runner)
    assert facts["runner_actor_width"] == 211
    assert facts["runner_critic_width"] == 319

    policy.num_critic_obs = 211
    with pytest.raises(RuntimeError, match="runner network ABI mismatch"):
        M.validate_action_ball_211_runner(runner)
    policy.num_critic_obs = 319
    runner._resolve_runtime_normalizer = lambda role: (role, actor_norm, ())
    with pytest.raises(RuntimeError, match="distinct objects"):
        M.validate_action_ball_211_runner(runner)

    runner.empirical_normalization = False
    with pytest.raises(RuntimeError, match="fresh empirical"):
        M.validate_action_ball_211_runner(runner)


def test_a211_resolved_cfg_disables_controller_but_keeps_fixed_fine_widths():
    import yaml

    raw_task = yaml.safe_load(TASK_YAML.read_text(encoding="utf-8"))
    assert raw_task["racket"]["adaptive_sigma"] is False
    assert raw_task["racket"]["adaptive_sigma_monotonic"] is False
    assert raw_task["racket"]["adaptive_sigma_normal"] is False

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
                "task=HOPEPingPongActionBallA211VendorV2N1Learnability"
            ],
        ).task

    assert task.racket.adaptive_sigma is False
    assert task.racket.adaptive_sigma_monotonic is False
    assert task.racket.adaptive_sigma_normal is False
    assert task.racket.adaptive_sigma_source == "ball_exact_strike"
    assert task.racket.sigma_pos_max == pytest.approx(0.50)
    assert task.racket.sigma_vel_max == pytest.approx(3.0)
    assert task.racket.sigma_normal_max == pytest.approx(2.10)


def test_train_finalizer_accepts_only_the_current_a211_c211_v2_markers():
    """Prevent the launcher/runtime ABI version from drifting past train.py again."""

    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert '"action_ball_a211_fixed_question_learnability_v2"' in source
    assert '"action_ball_c211_fixed_midpoint_learnability_v2"' in source
    assert '"action_ball_a211_fixed_question_learnability_v1"' not in source
    assert '"action_ball_c211_fixed_midpoint_learnability_v1"' not in source
