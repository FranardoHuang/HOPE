"""mjlab-ported action second-difference smoothing (action_acc_l2) — host-only unit tests.

Pinned here (all CPU, isaaclab stubbed via test_reward_flags_mdp; 结构照 test_foot_contact_shaping):

* action_acc_l2 math — 公式手算 ||a_t - 2a_{t-1} + a_{t-2}||²(逐关节二阶差分平方求和);
  匀速斜坡/恒定动作 = 精确 0(掉头才罚,迈步不罚——那是 action_rate 的地盘);有效位 False
  的 env 精确 0。
* ClampedJointPositionAction raw 历史缓冲 — a_{t-1}/a_{t-2} 自存(isaaclab ActionManager
  只有 prev_action);reset 清零 + 有效位清 False -> 复位后前两步不计费(episode 边界永远
  造不出虚构的"掉头"罚),第三步起按真实历史计费;部分 env reset 只清该 env。
* fail-loud 面:动作项缺 raw 历史缓冲、历史形状不对、有效位 dtype/形状不对。
* train.py 覆盖层往返:action_acc_weight 进 _REWARD_KEYS 白名单;weight 必须 finite 且
  <= 0(显式 0 = 对照);bool/NaN/正数/字符串拒收;默认路径(不写键)对 cfg 零改动 =
  字节等价;与 action_rate_weight 同臂并存(集成升级波的用法)。
* hope_env_cfg 源码级守卫:默认 weight=0.0、action_name=joint_pos。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_action_acc_smoothing.py -q
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_actions_mod, hope_rewards_mod
from test_reward_flags_overrides import _NS, _Term, _make_env_cfg, train_mod

ROOT = Path(__file__).resolve().parents[1]
ENV_CFG_SRC = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------------------------- #
def _acc_env(term):
    """env stand-in: action_manager.get_term(name) -> 动作项。"""

    return types.SimpleNamespace(
        action_manager=types.SimpleNamespace(get_term=lambda name: term)
    )


def _acc_term(current, previous, before_previous, valid):
    return types.SimpleNamespace(
        raw_actions=current,
        prev_raw_actions=previous,
        prev_prev_raw_actions=before_previous,
        raw_action_history_valid=valid,
    )


def _real_action(n=2, joints=2, clamp=False):
    """真 ClampedJointPositionAction(isaaclab 用 test_reward_flags_mdp 的 stub)。"""

    names = [f"j{i}" for i in range(joints)]
    limits = torch.stack(
        (torch.full((n, joints), -10.0), torch.full((n, joints), 10.0)), dim=-1
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=torch.zeros(n, joints),
            soft_joint_pos_limits=limits,
        )
    )
    cfg = types.SimpleNamespace(
        asset_name="robot", scale=1.0, use_default_offset=False, clamp=clamp
    )
    env = types.SimpleNamespace(scene={"robot": asset}, num_envs=n, device="cpu")
    return hope_actions_mod.ClampedJointPositionAction(cfg, env)


# --------------------------------------------------------------------------------------------- #
# action_acc_l2 math (hand-computed)
# --------------------------------------------------------------------------------------------- #
def test_action_acc_formula_hand_computed():
    # env0: d = a_t - 2a_{t-1} + a_{t-2} = [0.25, -0.25] -> 0.0625 + 0.0625 = 0.125
    # env1: 匀速斜坡(每步 +0.1 / +0.1)-> 二阶差分精确 0(迈步不罚,掉头才罚)
    term = _acc_term(
        torch.tensor([[1.0, -2.0], [0.3, 0.1]]),
        torch.tensor([[0.5, -1.0], [0.2, 0.0]]),
        torch.tensor([[0.25, -0.25], [0.1, -0.1]]),
        torch.tensor([True, True]),
    )
    value = hope_rewards_mod.action_acc_l2(_acc_env(term))
    assert value.tolist() == pytest.approx([0.125, 0.0], abs=1e-6)


def test_action_acc_invalid_history_pays_exactly_zero():
    # 有效位 False 的 env,哪怕历史缓冲里是垃圾数,也必须精确 0(复位后前两步的语义)。
    term = _acc_term(
        torch.tensor([[5.0, -5.0], [0.3, 0.1]]),
        torch.tensor([[99.0, 99.0], [0.2, 0.0]]),
        torch.tensor([[-99.0, 99.0], [0.1, -0.1]]),
        torch.tensor([False, True]),
    )
    value = hope_rewards_mod.action_acc_l2(_acc_env(term))
    assert value[0].item() == 0.0
    assert value[1].item() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------------- #
# 真动作项全链路:历史搬运、reset 清零、复位后前两步免费
# --------------------------------------------------------------------------------------------- #
def test_action_acc_through_real_action_term_with_reset():
    action = _real_action()
    env = _acc_env(action)

    a1 = torch.tensor([[0.1, -0.1], [1.0, 0.5]])
    a2 = torch.tensor([[0.3, 0.1], [1.0, 0.5]])
    a3 = torch.tensor([[0.2, 0.4], [1.0, 0.5]])

    # 第 1/2 步:历史不齐,双 env 都不计费
    action.process_actions(a1)
    assert action.raw_action_history_valid.tolist() == [False, False]
    assert hope_rewards_mod.action_acc_l2(env).tolist() == [0.0, 0.0]
    action.process_actions(a2)
    assert action.raw_action_history_valid.tolist() == [False, False]
    assert hope_rewards_mod.action_acc_l2(env).tolist() == [0.0, 0.0]

    # 第 3 步:env0 d = a3 - 2a2 + a1 = [-0.3, 0.1] -> 0.10;env1 恒定动作 -> 0
    action.process_actions(a3)
    assert action.raw_action_history_valid.tolist() == [True, True]
    assert torch.equal(action.prev_raw_actions, a2)
    assert torch.equal(action.prev_prev_raw_actions, a1)
    value = hope_rewards_mod.action_acc_l2(env)
    assert value.tolist() == pytest.approx([0.10, 0.0], abs=1e-6)

    # env0 reset:raw 历史清零 + 有效位清 False;env1 不受影响
    action.reset(env_ids=torch.tensor([0]))
    assert action.prev_raw_actions[0].tolist() == [0.0, 0.0]
    assert action.prev_prev_raw_actions[0].tolist() == [0.0, 0.0]
    assert torch.equal(action.prev_raw_actions[1], a2[1])

    # 复位后第 1 步:env0 免费;env1 d = a4 - 2a3 + a2 = [-0.2, 0.0] -> 0.04
    a4 = torch.tensor([[0.4, 0.0], [0.8, 0.5]])
    action.process_actions(a4)
    value = hope_rewards_mod.action_acc_l2(env)
    assert value.tolist() == pytest.approx([0.0, 0.04], abs=1e-6)

    # 复位后第 2 步:env0 仍免费;env1 d = a5 - 2a4 + a3 = [0.2, 0.0] -> 0.04
    a5 = torch.tensor([[0.1, 0.2], [0.8, 0.5]])
    action.process_actions(a5)
    value = hope_rewards_mod.action_acc_l2(env)
    assert value.tolist() == pytest.approx([0.0, 0.04], abs=1e-6)

    # 复位后第 3 步:env0 用复位后的真实历史计费
    # d = a6 - 2a5 + a4 = [0.0-0.2+0.4, -0.2-0.4+0.0] = [0.2, -0.6] -> 0.04 + 0.36 = 0.40
    a6 = torch.tensor([[0.0, -0.2], [0.8, 0.5]])
    action.process_actions(a6)
    value = hope_rewards_mod.action_acc_l2(env)
    assert value.tolist() == pytest.approx([0.40, 0.0], abs=1e-6)


def test_full_reset_clears_every_env():
    action = _real_action()
    env = _acc_env(action)
    for step in range(3):
        action.process_actions(torch.rand(2, 2))
    assert action.raw_action_history_valid.tolist() == [True, True]
    action.reset()  # env_ids=None = 全量 reset
    assert action.raw_action_history_valid.tolist() == [False, False]
    assert action.prev_raw_actions.abs().sum().item() == 0.0
    assert action.prev_prev_raw_actions.abs().sum().item() == 0.0
    action.process_actions(torch.rand(2, 2))
    assert hope_rewards_mod.action_acc_l2(env).tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------------------------- #
# fail-loud surfaces
# --------------------------------------------------------------------------------------------- #
def test_action_acc_fail_loud_surfaces():
    good = torch.zeros(2, 3)
    # 动作项没有自存历史(裸 JointPositionAction)-> 必须炸,不能静默算错
    bare = types.SimpleNamespace(raw_actions=good, prev_raw_actions=good)
    with pytest.raises(RuntimeError, match="raw-action history"):
        hope_rewards_mod.action_acc_l2(_acc_env(bare))
    # 历史形状不对
    term = _acc_term(good, torch.zeros(2, 2), good, torch.zeros(2, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="raw-action history"):
        hope_rewards_mod.action_acc_l2(_acc_env(term))
    # 有效位 dtype 不对(float 而非 bool)
    term = _acc_term(good, good, good, torch.zeros(2))
    with pytest.raises(RuntimeError, match="validity mask"):
        hope_rewards_mod.action_acc_l2(_acc_env(term))
    # 有效位形状不对
    term = _acc_term(good, good, good, torch.zeros(3, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="validity mask"):
        hope_rewards_mod.action_acc_l2(_acc_env(term))


# --------------------------------------------------------------------------------------------- #
# train.py override translation (action_acc_weight)
# --------------------------------------------------------------------------------------------- #
_ACC_PARAMS = {"action_name": "joint_pos"}


def _acc_env_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.action_acc_l2 = _Term(weight=0.0, params=dict(_ACC_PARAMS))
    return cfg


def _apply_acc(task, cfg=None):
    cfg = cfg if cfg is not None else _acc_env_cfg()
    applied = train_mod._apply_task_overrides(cfg, task, clip_name=None)
    return cfg, applied


def test_action_acc_key_is_whitelisted():
    assert "action_acc_weight" in train_mod._REWARD_KEYS


def test_action_acc_weight_roundtrip():
    cfg, applied = _apply_acc({"rewards": {"action_acc_weight": -0.05}})
    assert cfg.rewards.action_acc_l2.weight == -0.05
    assert "rewards.action_acc_l2.weight=-0.05" in applied


def test_action_acc_explicit_zero_weight_is_a_measured_control():
    cfg, applied = _apply_acc({"rewards": {"action_acc_weight": 0.0}})
    assert cfg.rewards.action_acc_l2.weight == 0.0
    assert "rewards.action_acc_l2.weight=0.0" in applied


def test_action_acc_coexists_with_action_rate():
    # 集成升级波的用法:-0.2 一阶 + -0.05 二阶同臂并存
    cfg, applied = _apply_acc({"rewards": {
        "action_rate_weight": -0.2,
        "action_acc_weight": -0.05,
    }})
    assert cfg.rewards.action_rate_l2.weight == -0.2
    assert cfg.rewards.action_acc_l2.weight == -0.05
    assert "rewards.action_rate_l2.weight=-0.2" in applied
    assert "rewards.action_acc_l2.weight=-0.05" in applied


@pytest.mark.parametrize("bad", [0.1, 1.0, True, float("nan"), float("inf"), "x"])
def test_action_acc_weight_must_be_finite_nonpositive(bad):
    with pytest.raises(train_mod._OverrideError, match="finite and <= 0"):
        _apply_acc({"rewards": {"action_acc_weight": bad}})


def test_action_acc_default_path_is_byte_identical():
    cfg, applied = _apply_acc({"rewards": {"racket_position_weight": 14.0}})
    assert cfg.rewards.action_acc_l2.weight == 0.0
    assert cfg.rewards.action_acc_l2.params == _ACC_PARAMS
    assert not [line for line in applied if "action_acc" in line]


# --------------------------------------------------------------------------------------------- #
# cfg-source guards (default stays OFF, joint_pos action)
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_default_is_off_and_targets_joint_pos():
    block = re.search(r"action_acc_l2 = RewTerm\((.*?)\n    \)", ENV_CFG_SRC, re.DOTALL)
    assert block and "weight=0.0" in block.group(1)
    assert '"action_name": "joint_pos"' in block.group(1)
    assert "func=mdp.action_acc_l2" in block.group(1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
