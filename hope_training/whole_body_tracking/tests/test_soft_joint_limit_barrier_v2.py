"""Focused CPU tests for fresh ActionBall's non-grazeable soft-limit barrier v2.

The legacy q_des barrier remains covered by ``test_qdes_limit_barrier.py``.  This file pins the
fresh ActionBall-only v2 math, the distinct processed-q_des / actual-q activation channels, the
adopted dose, and the generic-death/table-specific invariants.
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
import torch
import yaml

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _NS, _Term, _apply_legacy_v1, _make_env_cfg, train_mod


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_env_cfg.py"
)
ACTION_BALL_YAML = ROOT / "cfg/task/HOPEPingPongActionBall.yaml"
JOINTS = list(hope_rewards_mod._A3_RUNTIME_JOINT_ORDER)
# 2026-08-05 层级对齐(exp §5.6 第 9 条):两条 soft-limit v2 通道的带宽一起 0.08 -> 0.05
#(q_des 的 qdes_limit_barrier_margin_frac 与 actual-q 的 joint_limit_margin_frac)。
MARGIN = 0.05
FLOOR = 0.25


def _env(num_envs: int = 2):
    limits = torch.stack(
        (
            torch.full((num_envs, 31), -1.0),
            torch.full((num_envs, 31), 1.0),
        ),
        dim=-1,
    )
    data = types.SimpleNamespace(
        joint_names=list(JOINTS),
        soft_joint_pos_limits=limits,
        default_joint_pos=torch.zeros(num_envs, 31),
        joint_pos=torch.zeros(num_envs, 31),
    )
    asset = types.SimpleNamespace(data=data)
    action = types.SimpleNamespace(
        processed_actions=torch.zeros(num_envs, 31),
        _asset=asset,
        _joint_names=list(JOINTS),
        _joint_ids=slice(None),
    )
    env = types.SimpleNamespace(
        common_step_counter=41,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
        scene={"robot": asset},
    )
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    return env, action, data, asset_cfg


def test_v2_is_zero_outside_band_and_any_positive_intrusion_pays_floor():
    env, action, _, _ = _env(2)
    action.processed_actions[0, 0] = 0.83  # upper soft-band edge is q=0.84
    action.processed_actions[1, 0] = 0.840001
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert values[0].item() == 0.0
    assert values[1].item() >= FLOOR


def test_v2_ramp_is_monotone_bounded_and_sum_aggregated():
    env, action, _, _ = _env(4)
    for row, q in enumerate((0.840001, 0.88, 0.92, 1.0)):
        action.processed_actions[row, 3] = q
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert FLOOR <= values[0].item() < values[1].item() < values[2].item() < values[3].item()
    assert values[3].item() == pytest.approx(1.0, abs=1.0e-6)

    env2, action2, _, _ = _env(1)
    action2.processed_actions[0, [2, 5, 19]] = 1.0
    summed = hope_rewards_mod.qdes_limit_barrier_v2(env2)
    assert summed.item() == pytest.approx(3.0, abs=1.0e-6)
    action2.processed_actions[:] = 20.0
    bounded = hope_rewards_mod.qdes_limit_barrier_v2(env2)
    assert bounded.item() == pytest.approx(31.0, abs=1.0e-5)


def test_v2_stance_exemption_is_free_but_moving_toward_limit_is_not():
    env, action, data, _ = _env(2)
    # d(default) = 0.03, therefore m_eff = 0.025 after the 0.005 breathing gap.
    data.default_joint_pos[:, 5] = -0.94
    action.processed_actions[0, 5] = -0.94
    action.processed_actions[1, 5] = -0.950001
    values = hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert values[0].item() == 0.0
    assert values[1].item() >= FLOOR


def test_qdes_and_actual_q_activate_as_distinct_objectives():
    env, action, data, asset_cfg = _env(2)
    action.processed_actions[0, 7] = 0.90
    data.joint_pos[1, 9] = -0.90
    qdes = hope_rewards_mod.qdes_limit_barrier_v2(env)
    actual = hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg)
    assert qdes[0].item() > 0.0 and qdes[1].item() == 0.0
    assert actual[0].item() == 0.0 and actual[1].item() > 0.0


def test_v2_probes_are_zero_and_ledger_keeps_qdes_actual_separate():
    env, action, data, asset_cfg = _env(2)
    action.processed_actions[0, 1] = 0.90
    data.joint_pos[1, 2] = -0.90
    assert torch.equal(
        hope_rewards_mod.qdes_limit_barrier_v2_probe(env), torch.zeros(2)
    )
    hope_rewards_mod.qdes_limit_barrier_v2(env)
    assert torch.equal(
        hope_rewards_mod.actual_joint_limit_barrier_v2_probe(env, asset_cfg),
        torch.zeros(2),
    )
    hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg)

    counters = hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)
    assert counters["qdes_observed_sample_count"].item() == 2
    assert counters["actual_observed_sample_count"].item() == 2
    assert counters["qdes_intrusion_joint_count"].item() == 1
    assert counters["actual_intrusion_joint_count"].item() == 1
    assert counters["qdes_reward_enabled_sample_count"].item() == 2
    assert counters["actual_reward_enabled_sample_count"].item() == 2
    assert counters["qdes_barrier_value_sum"].item() > 0.0
    assert counters["actual_barrier_value_sum"].item() > 0.0
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(
            env
        ).values()
    )


def test_v2_activation_requires_step_identity_and_both_channels():
    env, _, _, _ = _env(1)
    del env.common_step_counter
    with pytest.raises(RuntimeError, match="common_step_counter"):
        hope_rewards_mod.qdes_limit_barrier_v2(env)

    env, _, _, _ = _env(1)
    hope_rewards_mod.qdes_limit_barrier_v2(env)
    with pytest.raises(RuntimeError, match="both qdes and actual channels"):
        hope_rewards_mod.consume_qdes_limit_barrier_activation_counters(env)


def test_v2_activation_state_corruption_fails_closed():
    env, _, _, _ = _env(1)
    setattr(
        env,
        hope_rewards_mod._QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR,
        {"observed_sample_count": torch.zeros((), dtype=torch.long)},
    )
    with pytest.raises(RuntimeError, match="schema mismatch"):
        hope_rewards_mod.qdes_limit_barrier_v2(env)


@pytest.mark.parametrize("source", ("qdes", "actual"))
@pytest.mark.parametrize(
    "field,value,match",
    (
        ("margin_frac", 0.0, r"\(0, 0.5\)"),
        ("margin_frac", 0.5, r"\(0, 0.5\)"),
        ("penalty_floor", 0.0, r"\(0, 1\)"),
        ("penalty_floor", 1.0, r"\(0, 1\)"),
        ("penalty_floor", math.nan, r"\(0, 1\)"),
    ),
)
def test_v2_invalid_scalars_fail_closed(source, field, value, match):
    env, _, _, asset_cfg = _env(1)
    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        if source == "qdes":
            hope_rewards_mod.qdes_limit_barrier_v2(env, **kwargs)
        else:
            hope_rewards_mod.actual_joint_limit_barrier_v2(env, asset_cfg, **kwargs)


def test_action_ball_config_uses_v2_callables_and_separate_probe_terms():
    source = ENV_CFG.read_text(encoding="utf-8")
    start = source.index("class HOPEActionBallRewardsCfg")
    end = source.index("class HOPEActionBallTerminationsCfg", start)
    block = source[start:end]
    assert "func=mdp.qdes_limit_barrier_v2," in block
    assert "func=mdp.qdes_limit_barrier_v2_probe," in block
    assert "func=mdp.actual_joint_limit_barrier_v2," in block
    assert "func=mdp.actual_joint_limit_barrier_v2_probe," in block
    assert block.count('"penalty_floor": 0.25') == 4
    assert block.count("weight=-5.0") == 3  # qdes, actual-q, projection
    for term_name in (
        "qdes_limit_barrier_probe",
        "actual_joint_limit_barrier_probe",
    ):
        term_start = block.index(f"{term_name} = RewTerm(")
        term_end = block.index("\n    )", term_start)
        assert "weight=1.0" in block[term_start:term_end]


def test_adopted_scale_and_generic_death_invariants_are_pinned():
    task = yaml.safe_load(ACTION_BALL_YAML.read_text(encoding="utf-8"))
    rewards = task["rewards"]
    assert rewards["qdes_limit_barrier_weight"] == pytest.approx(-5.0)
    assert rewards["joint_limit_weight"] == pytest.approx(-5.0)
    # 2026-08-05 带宽对齐(exp §5.6 第 9 条):两条通道一起 0.08 -> 0.05。v2 硬合同要求
    # q_des / actual-q 通道逐字段同权同带宽,所以这里断言的是"相等"本身,不只是数值。
    assert rewards["qdes_limit_barrier_margin_frac"] == pytest.approx(MARGIN)
    assert rewards["joint_limit_margin_frac"] == pytest.approx(MARGIN)
    assert (
        rewards["joint_limit_margin_frac"] == rewards["qdes_limit_barrier_margin_frac"]
    )
    assert rewards["joint_limit_weight"] == rewards["qdes_limit_barrier_weight"]
    # 2026-08-05 层级对齐(exp §5.6 第 7 条):death -300.0 -> -10.0(post-dt -6.0 -> -0.2)。
    # 权威在 HOPEPingPongActionBall.yaml:151;本文件读那份 yaml,是它的算术副本。
    assert rewards["death_penalty_weight"] == pytest.approx(-10.0)
    assert rewards["table_hit_penalty_weight"] == pytest.approx(0.0)

    policy_dt = 0.02
    floor_step_per_joint_channel = abs(-5.0 * policy_dt * FLOOR)
    full_step_per_joint_channel = abs(-5.0 * policy_dt)
    max_two_channel_step = 31 * 2 * full_step_per_joint_channel
    hard_death = abs(-10.0 * policy_dt)
    landing_max = 500.0 * policy_dt
    assert floor_step_per_joint_channel == pytest.approx(0.025)
    assert full_step_per_joint_channel == pytest.approx(0.10)
    assert max_two_channel_step == pytest.approx(6.2)
    # 对齐后 death 不再压过 soft 限位通道:6.0 -> 0.2,与 max_two_channel_step 6.2 的量级
    # 关系整个翻转(原本 death 约等于两通道满额,现在只有它的 3%)。这正是 §5.6 想要的次序。
    assert hard_death == pytest.approx(0.2)
    assert landing_max == pytest.approx(10.0)
    assert 50 * floor_step_per_joint_channel == pytest.approx(1.25)


# --------------------------------------------------------------------------------------------- #
# train.py override translation: the actual-q band travels with the q_des band
# --------------------------------------------------------------------------------------------- #
_ACTUAL_PARAMS = {
    "margin_frac": 0.08,
    "penalty_floor": 0.25,
    "expected_joint_count": 31,
}


def _actual_band_env_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.joint_limit = _Term(
        weight=-5.0,
        params={"asset_cfg": _NS(name="robot", joint_ids=slice(None)), **_ACTUAL_PARAMS},
    )
    cfg.rewards.actual_joint_limit_barrier_probe = _Term(
        weight=1.0,
        params={"asset_cfg": _NS(name="robot", joint_ids=slice(None)), **_ACTUAL_PARAMS},
    )
    return cfg


def _apply_actual_band(task, cfg=None):
    cfg = cfg if cfg is not None else _actual_band_env_cfg()
    return cfg, _apply_legacy_v1(cfg, task)


def test_actual_band_override_moves_term_and_probe_together():
    cfg, applied = _apply_actual_band(
        {"rewards": {"joint_limit_weight": -5.0, "joint_limit_margin_frac": MARGIN}}
    )
    term = cfg.rewards.joint_limit
    probe = cfg.rewards.actual_joint_limit_barrier_probe
    assert term.params["margin_frac"] == pytest.approx(MARGIN)
    assert probe.params["margin_frac"] == pytest.approx(MARGIN)
    assert probe.weight == pytest.approx(1.0)
    assert term.weight == pytest.approx(-5.0)
    assert f"rewards.joint_limit.params.margin_frac={MARGIN}" in applied
    assert any(
        marker.startswith("rewards.actual_joint_limit_barrier_probe=")
        for marker in applied
    )


def test_actual_band_without_explicit_weight_is_refused():
    with pytest.raises(train_mod._OverrideError, match="joint_limit_weight"):
        _apply_actual_band({"rewards": {"joint_limit_margin_frac": MARGIN}})


@pytest.mark.parametrize("margin", [0.0, 0.5, -0.1, float("nan"), True, "bad"])
def test_invalid_actual_band_override_is_refused(margin):
    with pytest.raises(train_mod._OverrideError, match=r"\(0, 0.5\)"):
        _apply_actual_band(
            {
                "rewards": {
                    "joint_limit_weight": -5.0,
                    "joint_limit_margin_frac": margin,
                }
            }
        )


def test_invalid_actual_band_does_not_partially_mutate_either_term():
    cfg = _actual_band_env_cfg()
    term = cfg.rewards.joint_limit
    probe = cfg.rewards.actual_joint_limit_barrier_probe
    before = (dict(term.params), probe.weight, dict(probe.params))
    with pytest.raises(train_mod._OverrideError):
        _apply_actual_band(
            {
                "rewards": {
                    "joint_limit_weight": -5.0,
                    "joint_limit_margin_frac": 0.5,
                }
            },
            cfg=cfg,
        )
    assert (dict(term.params), probe.weight, dict(probe.params)) == before


def test_reward_keys_whitelist_carries_both_actual_band_keys_once():
    keys = [key for key in train_mod._REWARD_KEYS if key.startswith("joint_limit")]
    assert sorted(keys) == ["joint_limit_margin_frac", "joint_limit_weight"]
    with pytest.raises(train_mod._OverrideError, match="joint_limit_margin_farc"):
        _apply_actual_band({"rewards": {"joint_limit_margin_farc": MARGIN}})
