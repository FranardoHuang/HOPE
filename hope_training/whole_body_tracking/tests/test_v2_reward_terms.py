"""v2 蓝图新 reward 项(upright_exp / hit_unstable_support / strike_capture_bonus)— host-only unit tests.

Pinned here (all CPU, isaaclab stubbed via test_reward_flags_mdp):

* upright_exp math — mjlab 收入型站正:exp(-||projected_gravity_xy||² / std²);完全直立 = 1.0,
  已知倾斜手算对上 exp(-x/std²),最大倾斜(||g_xy||=1)仍 > 0(有界 (0,1],天然 alive bonus);
  std 可调且只改衰减速度不改 0 倾斜处的 1.0。
* hit_unstable_support math — PACE 单条击球稳定:indicator(触地脚数 <= 1) x strike_window,
  取值 {0.0, 1.0}:窗内双脚触地 = 0、窗内单脚 = 1、窗内双脚腾空 = 1、窗外怎么站都 = 0;
  触地判据 = 单脚接触力范数 > 10.0 N(与 sensor force_threshold / foot_clearance /
  hope_commands in_contact 同判据同阈值,边界 9.9/10.1 N 两侧验证)。
* 两个函数的 fail-loud 面:std 非法(0/负/NaN/bool)、projected_gravity_b 缺失或形状不对;
  脚数不是 2、net_forces_w 缺失或形状不对、strike_window 缺失/不是 bool/形状不对。
* strike_capture_bonus math — L2 击中层 one-shot(redesign §3.5):cmd.vb_fired True 的
  env 付 1.0、False 付 0.0(一拍一发的锁存由 RacketTargetCommand._vb_evaluate 负责,函数
  只是读门);cmd 缺 vb_fired 属性(非 virtual-ball 谱系误接线)fail-loud,与 virtual_*
  三项的 _cmd 读法同路。cfg 声明:HOPEVirtualBallRewardsCfg 以 weight=0.0 待命
  (reward_pack=v2 翻译层直写名义 850,probe 校准后冻结 prereg)。
* cfg 声明(B1 波接力):两项已在 HOPEDeployParityRewardsCfg(strike_upright 所在谱系)以
  weight=0.0 声明 = IsaacLab 直接跳过 = 默认路径字节等价;参数逐字冻结由源码级守卫钉住,
  tracking_env_cfg 基类谱系仍然干净。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_v2_reward_terms.py -q
"""

from __future__ import annotations

import math
import re
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_rewards_mod

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------------------------- #
class _Scene:
    """env.scene stand-in: .sensors dict + scene[name] articulation lookup."""

    def __init__(self, sensors=None, entities=None):
        self.sensors = dict(sensors or {})
        self._entities = dict(entities or {})

    def __getitem__(self, name):
        return self._entities[name]


def _upright_env(*, n=2, projected_gravity=...):
    """upright_exp fake: robot articulation with data.projected_gravity_b."""

    if projected_gravity is ...:
        projected_gravity = torch.zeros(n, 3)
        projected_gravity[:, 2] = -1.0  # 完全直立:重力全在 -z,xy 分量为 0
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(projected_gravity_b=projected_gravity)
    )
    env = types.SimpleNamespace(scene=_Scene(entities={"robot": robot}))
    return env, projected_gravity


def _support_env(
    *,
    n=2,
    feet_ids=(1, 3),
    bodies=4,
    forces=None,
    window=None,
):
    """hit_unstable_support fake: contact sensor forces + command strike_window."""

    if forces is None:
        forces = torch.zeros(n, bodies, 3)
    if window is None:
        window = torch.zeros(n, dtype=torch.bool)
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=forces))
    cmd = types.SimpleNamespace(strike_window=window)
    env = types.SimpleNamespace(
        scene=_Scene(sensors={"contact_forces": sensor}),
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd),
    )
    cfg = types.SimpleNamespace(name="contact_forces", body_ids=list(feet_ids))
    return env, cfg, forces, window, cmd


# --------------------------------------------------------------------------------------------- #
# upright_exp math (hand-computed)
# --------------------------------------------------------------------------------------------- #
def test_upright_exp_is_one_at_zero_tilt():
    env, _ = _upright_env()
    value = hope_rewards_mod.upright_exp(env)
    assert value.tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_upright_exp_matches_hand_computed_kernel():
    # env0 直立;env1 倾斜 g_xy=(0.3, 0.4) -> ||g_xy||² = 0.25 -> exp(-0.25/0.2)(默认 std²=0.2)
    g = torch.tensor([[0.0, 0.0, -1.0], [0.3, 0.4, -math.sqrt(0.75)]])
    env, _ = _upright_env(projected_gravity=g)
    value = hope_rewards_mod.upright_exp(env)
    assert value.tolist() == pytest.approx([1.0, math.exp(-1.25)], abs=1e-6)


def test_upright_exp_std_only_changes_decay_rate():
    g = torch.tensor([[0.3, 0.4, -math.sqrt(0.75)], [0.0, 0.0, -1.0]])
    env, _ = _upright_env(projected_gravity=g)
    v_default = hope_rewards_mod.upright_exp(env)  # std² = 0.2
    v_wide = hope_rewards_mod.upright_exp(env, std=1.0)  # std² = 1.0
    assert v_default[0].item() == pytest.approx(math.exp(-0.25 / 0.2), abs=1e-6)
    assert v_wide[0].item() == pytest.approx(math.exp(-0.25), abs=1e-6)
    # 0 倾斜处永远发满额 1.0,std 只改歪了之后掉钱多快
    assert v_default[1].item() == pytest.approx(1.0, abs=1e-6)
    assert v_wide[1].item() == pytest.approx(1.0, abs=1e-6)


def test_upright_exp_is_bounded_positive_even_lying_flat():
    # 躺平(||g_xy|| = 1,倾角 90°)也仍是正收入 exp(-1/0.2) —— 有界 (0,1],不会变成罚
    g = torch.tensor([[1.0, 0.0, 0.0]])
    env, _ = _upright_env(projected_gravity=g)
    value = hope_rewards_mod.upright_exp(env)
    assert value[0].item() == pytest.approx(math.exp(-5.0), abs=1e-9)
    assert value[0].item() > 0.0


@pytest.mark.parametrize("bad_std", [0.0, -0.5, float("nan"), float("inf"), True, "x"])
def test_upright_exp_fail_loud_on_bad_std(bad_std):
    env, _ = _upright_env()
    with pytest.raises(ValueError):
        hope_rewards_mod.upright_exp(env, std=bad_std)


def test_upright_exp_fail_loud_on_missing_or_misshaped_gravity():
    env, _ = _upright_env(projected_gravity=None)
    with pytest.raises(RuntimeError, match="projected_gravity_b"):
        hope_rewards_mod.upright_exp(env)
    env2, _ = _upright_env(projected_gravity=torch.zeros(2, 2))  # 缺 z 列
    with pytest.raises(RuntimeError, match="projected_gravity_b"):
        hope_rewards_mod.upright_exp(env2)
    env3, _ = _upright_env(projected_gravity=torch.zeros(3))  # 缺 env 维
    with pytest.raises(RuntimeError, match="projected_gravity_b"):
        hope_rewards_mod.upright_exp(env3)


# --------------------------------------------------------------------------------------------- #
# hit_unstable_support math (hand-computed)
# --------------------------------------------------------------------------------------------- #
def test_two_feet_in_contact_inside_window_is_free():
    env, cfg, forces, window, _ = _support_env()
    window[0] = True
    forces[0, 1, 2] = 200.0  # 左脚踩实
    forces[0, 3, 2] = 200.0  # 右脚踩实
    value = hope_rewards_mod.hit_unstable_support(env, cfg)
    assert value.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_one_foot_inside_window_pays_exactly_one():
    env, cfg, forces, window, _ = _support_env()
    window[0] = True
    forces[0, 1, 2] = 200.0  # 只有左脚触地
    value = hope_rewards_mod.hit_unstable_support(env, cfg)
    assert value.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)


def test_airborne_inside_window_pays_exactly_one():
    env, cfg, forces, window, _ = _support_env()
    window[1] = True  # 双脚腾空(力全 0)+ 在窗内 -> 1
    value = hope_rewards_mod.hit_unstable_support(env, cfg)
    assert value.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_one_foot_outside_window_is_free():
    env, cfg, forces, window, _ = _support_env()
    # 窗外(window 全 False)怎么站都免费——这条只管击球那一瞬
    forces[0, 1, 2] = 200.0  # env0 单脚
    value = hope_rewards_mod.hit_unstable_support(env, cfg)
    assert value.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_contact_threshold_matches_sensor_10n():
    env, cfg, forces, window, _ = _support_env()
    window[0] = True
    forces[0, 1, 2] = 200.0  # 左脚踩实
    forces[0, 3, 2] = 9.9  # 右脚 < 10 N = 不算触地(与 sensor force_threshold 同判据)-> 单脚 -> 1
    assert hope_rewards_mod.hit_unstable_support(env, cfg)[0].item() == pytest.approx(
        1.0, abs=1e-6
    )
    forces[0, 3, 2] = 10.1  # > 10 N = 触地 -> 双脚 -> 0
    assert hope_rewards_mod.hit_unstable_support(env, cfg)[0].item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_values_are_binary_indicator():
    env, cfg, forces, window, _ = _support_env(n=4)
    window[:] = torch.tensor([True, True, True, False])
    forces[0, 1, 2] = 500.0  # 单脚,力再大也只是 1(指示器,不按力大小计价)
    forces[1, 1, 2] = 50.0
    forces[1, 3, 2] = 50.0  # 双脚
    # env2 腾空;env3 窗外
    value = hope_rewards_mod.hit_unstable_support(env, cfg)
    assert value.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0], abs=1e-6)
    assert set(value.tolist()) <= {0.0, 1.0}


def test_support_fail_loud_surfaces():
    # 脚数不是 2
    env, _, _, _, _ = _support_env()
    bad = types.SimpleNamespace(name="contact_forces", body_ids=[1])
    with pytest.raises(RuntimeError, match="exactly the two A3 feet"):
        hope_rewards_mod.hit_unstable_support(env, bad)
    # net_forces_w 缺失
    env2, cfg2, _, _, _ = _support_env()
    env2.scene.sensors["contact_forces"].data.net_forces_w = None
    with pytest.raises(RuntimeError, match="net_forces_w"):
        hope_rewards_mod.hit_unstable_support(env2, cfg2)
    # net_forces_w 形状不对(缺 xyz 维)
    env3, cfg3, _, _, _ = _support_env()
    env3.scene.sensors["contact_forces"].data.net_forces_w = torch.zeros(2, 4)
    with pytest.raises(RuntimeError, match="net_forces_w"):
        hope_rewards_mod.hit_unstable_support(env3, cfg3)
    # strike_window 缺失
    env4, cfg4, _, _, cmd4 = _support_env()
    cmd4.strike_window = None
    with pytest.raises(RuntimeError, match="strike_window"):
        hope_rewards_mod.hit_unstable_support(env4, cfg4)
    # strike_window 不是 bool(浮点掩码会静默把门变成加权,必须炸)
    env5, cfg5, _, _, cmd5 = _support_env()
    cmd5.strike_window = torch.zeros(2)
    with pytest.raises(RuntimeError, match="strike_window"):
        hope_rewards_mod.hit_unstable_support(env5, cfg5)
    # strike_window env 数对不上
    env6, cfg6, _, _, cmd6 = _support_env()
    cmd6.strike_window = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="strike_window"):
        hope_rewards_mod.hit_unstable_support(env6, cfg6)


# --------------------------------------------------------------------------------------------- #
# strike_capture_bonus — L2 击中层 one-shot(redesign §3.5;判据 = 现成 vb_fired capture 门)
# --------------------------------------------------------------------------------------------- #
def _capture_env(vb_fired):
    """strike_capture_bonus fake:command term 只带 vb_fired(仿 virtual_* 的 _cmd 读法)。"""

    cmd = types.SimpleNamespace(vb_fired=vb_fired)
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd)
    )
    return env


def test_strike_capture_bonus_pays_one_on_fired_envs_only():
    env = _capture_env(torch.tensor([True, False, True]))
    value = hope_rewards_mod.strike_capture_bonus(env, "racket_target")
    assert value.tolist() == pytest.approx([1.0, 0.0, 1.0], abs=1e-6)
    assert value.dtype == torch.float32  # RewardManager 端要的是 float,不是 bool


def test_strike_capture_bonus_all_quiet_is_all_zero():
    env = _capture_env(torch.zeros(4, dtype=torch.bool))
    value = hope_rewards_mod.strike_capture_bonus(env, "racket_target")
    assert value.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_strike_capture_bonus_missing_vb_fired_fails_loud():
    # 非 virtual-ball 谱系误接线(cmd 根本没有 vb_fired)必须当场炸,绝不静默付 0。
    cmd = types.SimpleNamespace()
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd)
    )
    with pytest.raises(AttributeError, match="vb_fired"):
        hope_rewards_mod.strike_capture_bonus(env, "racket_target")


def test_strike_capture_bonus_declared_default_off_in_virtual_ball_cfg():
    import re

    cfg_dir = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    env_src = (cfg_dir / "config/agibot_a3/hope_env_cfg.py").read_text(encoding="utf-8")
    decl = re.search(r"strike_capture_bonus = RewTerm\((.*?)\)\n", env_src, re.DOTALL)
    assert decl and "func=mdp.strike_capture_bonus" in decl.group(1)
    assert "weight=0.0" in decl.group(1)  # weight 0 = skipped = 默认(v1 兜底)逐字节等价
    assert '"command_name": "racket_target"' in decl.group(1)
    # 声明位置:virtual_pass_net 所在的 HOPEVirtualBallRewardsCfg(vb_fired 只有该谱系有)
    class_start = env_src.index("class HOPEVirtualBallRewardsCfg")
    next_class = env_src.index("\nclass ", class_start + 1)
    assert class_start < decl.start() < next_class


# --------------------------------------------------------------------------------------------- #
# cfg-source guard(B1 波接力):两项已在 HOPEDeployParityRewardsCfg 以 weight=0.0 声明
# (= IsaacLab 直接跳过 = 默认路径字节等价);参数逐字冻结,tracking_env_cfg 仍然干净。
# --------------------------------------------------------------------------------------------- #
def test_v2_terms_declared_default_off_in_deploy_parity_cfg():
    import re

    cfg_dir = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
    env_src = (cfg_dir / "config/agibot_a3/hope_env_cfg.py").read_text(encoding="utf-8")
    up = re.search(r"upright_exp = RewTerm\((.*?)\n    \)", env_src, re.DOTALL)
    hit = re.search(
        r"hit_unstable_support = RewTerm\((.*?)\n    \)", env_src, re.DOTALL
    )
    assert up and "func=mdp.upright_exp" in up.group(1)
    assert "weight=0.0" in up.group(1)  # weight 0 = skipped = 默认逐字节等价
    assert '"std": math.sqrt(0.2)' in up.group(1)  # A2 函数默认值逐字冻结
    assert hit and "func=mdp.hit_unstable_support" in hit.group(1)
    assert "weight=0.0" in hit.group(1)
    assert hit.group(1).count("A3_FEET_BODIES") == 1  # 两脚 sensor 名单,foot_soft_landing 同款
    assert '"command_name": "racket_target"' in hit.group(1)
    # 声明位置:strike_upright 所在的 HOPEDeployParityRewardsCfg(所有在跑波次的 reward 谱系)
    class_start = env_src.index("class HOPEDeployParityRewardsCfg")
    class_end = env_src.index("class HOPEDeployParityTerminationsCfg")
    assert class_start < up.start() < class_end
    assert class_start < hit.start() < class_end
    # tracking_env_cfg(基类谱系)仍然不引用这两项
    tracking_src = (cfg_dir / "tracking_env_cfg.py").read_text(encoding="utf-8")
    assert "upright_exp" not in tracking_src
    assert "hit_unstable_support" not in tracking_src


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------------------------- #
# virtual_landing legal_base(v2.2:上台组唯一保留项)— 手算
# --------------------------------------------------------------------------------------------- #
def _vb_env(n=2, sigma=1.0):
    cmd = types.SimpleNamespace(
        vb_landing_xy=torch.zeros(n, 2),
        _vb_target_xy=torch.tensor([2.555, 0.0]),
        vb_landing_valid=torch.zeros(n, dtype=torch.bool),
        vb_net_clear=torch.zeros(n, dtype=torch.bool),
        vb_on_opponent=torch.zeros(n, dtype=torch.bool),
        vb_depth_ok=torch.zeros(n, dtype=torch.bool),
        vb_fired=torch.zeros(n, dtype=torch.bool),
        cfg=types.SimpleNamespace(vb_landing_sigma=sigma),
    )
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd)
    )
    return env, cmd


def test_legal_base_pays_floor_anywhere_on_table_and_kernel_at_center():
    env, cmd = _vb_env(2)
    for t_ in (cmd.vb_landing_valid, cmd.vb_net_clear, cmd.vb_on_opponent, cmd.vb_fired):
        t_[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 0.0])   # 正中 -> 1.0
    cmd.vb_landing_xy[1] = torch.tensor([2.555, 1.0])   # 距台心 1.0(近台角) -> 0.6+0.4e^-1
    v = hope_rewards_mod.virtual_landing(env, "racket_target", mode="legal_base")
    assert v[0].item() == pytest.approx(1.0, abs=1e-6)
    assert v[1].item() == pytest.approx(0.6 + 0.4 * math.exp(-1.0), abs=1e-6)


def test_virtual_landing_uses_each_envs_installed_action_ball_target():
    env, cmd = _vb_env(2)
    cmd._vb_target_xy_per_env = torch.tensor(
        [[2.555, 0.0], [2.555, 1.0]]
    )
    cmd.vb_landing_xy.copy_(cmd._vb_target_xy_per_env)
    for flag in (
        cmd.vb_landing_valid,
        cmd.vb_net_clear,
        cmd.vb_on_opponent,
        cmd.vb_fired,
    ):
        flag[:] = True
    value = hope_rewards_mod.virtual_landing(
        env,
        "racket_target",
        mode="legal_base",
    )
    # Env 1 is one metre from the legacy global target but exactly at its
    # installed per-env aim; both rows must therefore receive the full kernel.
    assert value.tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_counter_rally_quality_reward_is_one_staged_total_without_legacy_double_score():
    env, cmd = _vb_env(2)
    cmd._counter_rally_enabled = True
    cmd._counter_rally_reward_terms = torch.tensor(
        [
            [0.11, 0.12, 0.13, 0.14, 0.61],
            [0.21, 0.22, 0.23, 0.24, 0.83],
        ],
        dtype=torch.float32,
    )
    cmd._vb_target_xy_per_env = torch.tensor(
        [[2.1, -0.4], [2.9, 0.5]],
        dtype=torch.float32,
    )
    cmd.vb_fired[:] = True
    # Make every legacy coarse-scoring signal disagree with the staged score.
    cmd.vb_landing_xy[:] = torch.tensor(
        [[99.0, 99.0], [-99.0, -99.0]]
    )
    cmd.vb_landing_valid[:] = False
    cmd.vb_net_clear[:] = False
    cmd.vb_on_opponent[:] = False
    cmd.vb_depth_ok[:] = False
    cmd.vb_net_z = torch.tensor([100.0, -100.0])
    cmd.vb_topspin = torch.tensor([1000.0, 2000.0])

    landing = hope_rewards_mod.virtual_landing(
        env,
        "racket_target",
        mode="legal_base",
    )
    pass_net = hope_rewards_mod.virtual_pass_net(
        env,
        "racket_target",
    )
    spin = hope_rewards_mod.virtual_spin(env, "racket_target")
    assert landing.tolist() == pytest.approx([0.61, 0.83], abs=1e-6)
    assert pass_net.tolist() == [0.0, 0.0]
    assert spin.tolist() == [0.0, 0.0]
    assert (landing + pass_net + spin).tolist() == pytest.approx(
        [0.61, 0.83],
        abs=1e-6,
    )

    # The climb alias also reads the same one-shot staged total. Cache clearing
    # in _vb_evaluate is what makes non-contact steps zero, not a second scorer.
    cmd.vb_fired[:] = False
    climb = hope_rewards_mod.virtual_landing(env, "racket_target")
    assert climb.tolist() == pytest.approx([0.61, 0.83], abs=1e-6)


def test_counter_rally_staged_reward_cache_shape_fails_loud():
    env, cmd = _vb_env(2)
    cmd._counter_rally_enabled = True
    cmd._counter_rally_reward_terms = torch.zeros(2, 4)
    with pytest.raises(RuntimeError, match=r"shape \[num_envs,5\]"):
        hope_rewards_mod.virtual_landing(
            env,
            "racket_target",
            mode="legal_base",
        )


def test_legal_base_gate_is_a_prerequisite_not_a_bonus():
    env, cmd = _vb_env(1)
    cmd.vb_landing_valid[:] = True
    cmd.vb_on_opponent[:] = True
    cmd.vb_fired[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 0.0])
    cmd.vb_net_clear[:] = False  # 没过网:落点再准也一分不发(先决条件语义)
    v = hope_rewards_mod.virtual_landing(env, "racket_target", mode="legal_base")
    assert v[0].item() == 0.0
    cmd.vb_net_clear[:] = True
    cmd.vb_fired[:] = False      # capture 没触发同样零
    v = hope_rewards_mod.virtual_landing(env, "racket_target", mode="legal_base")
    assert v[0].item() == 0.0


def test_climb_mode_stays_byte_identical_v1():
    env, cmd = _vb_env(1)
    cmd.vb_landing_valid[:] = True
    cmd.vb_fired[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 1.0])  # kernel = e^-1,无奖金
    v = hope_rewards_mod.virtual_landing(env, "racket_target")  # 默认 climb
    assert v[0].item() == pytest.approx(math.exp(-1.0), abs=1e-6)


def test_legal_base_fail_loud_surfaces():
    env, _ = _vb_env(1)
    with pytest.raises(ValueError, match="base_frac"):
        hope_rewards_mod.virtual_landing(env, "racket_target", mode="legal_base", base_frac=1.5)
    with pytest.raises(ValueError, match="mode"):
        hope_rewards_mod.virtual_landing(env, "racket_target", mode="v3")


# --------------------------------------------------------------------------------------------- #
# action_rate_l2_clamped —— **2026-08-08 起是退役件**(没有任何现役配方启用它)。
# 下面这组保留是因为"退役 != 静默删除":它把"为什么退役"的算术钉在可执行的断言里,
# 谁想把它复活,得先让这几条红给他看。现役一阶平滑的测试在本节末尾"现役形状"那一组。
# --------------------------------------------------------------------------------------------- #
def test_action_rate_clamped_matches_builtin_below_and_caps_above():
    mgr = types.SimpleNamespace(
        action=torch.tensor([[1.0, 2.0], [10.0, 0.0]]),
        prev_action=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    )
    env = types.SimpleNamespace(action_manager=mgr)
    v = hope_rewards_mod.action_rate_l2_clamped(env, value_clamp=9.0)
    assert v[0].item() == pytest.approx(5.0, abs=1e-6)   # 1+4 < 9:同内置
    assert v[1].item() == pytest.approx(9.0, abs=1e-6)   # 100 -> 封顶 9
    with pytest.raises(ValueError, match="value_clamp"):
        hope_rewards_mod.action_rate_l2_clamped(env, value_clamp=0.0)


# --------------------------------------------------------------------------------------------- #
# 封顶档位是不是**落在真实工作区里**——上面那条只证明代码按 min() 算对了,它对
# "档位选错、于是整条曲线都在天花板上"零意见。这一组把参照工作区写死成数字。
#
# 参照工作区来自唯一已知能打球的实现 build_1(BerkeleyPingPong/hope_wbc,dongc_1):
#   * iter 4    action_rate = -0.1262/步(post-dt),weight -0.1 x dt 0.02
#                 -> ||Δa||² = 0.1262 / 0.002 = 63.1  (= 2 x 31 x std² ,std≈1.0)
#   * 收敛(21896 iter) action_rate = -0.0216~-0.0241/步 -> ||Δa||² = 10.8~12.05
# 也就是说 build_1 从开局到收敛,||Δa||² 一路 63 -> 11,**从没进过 9.0 以下**。
# 我们现役 value_clamp = 9.0 因此在整条谱系上恒饱和(s15r1 实测 raw_sum = 98304 x 9)。
# --------------------------------------------------------------------------------------------- #
BUILD1_ACTION_RATE_SQ_EARLY = 63.1        # iter 4
BUILD1_ACTION_RATE_SQ_CONVERGED = 10.8    # 21896 iter,取两跑里更低的那个
BUILD1_ACTION_RATE_DOSE_EARLY = -0.1262   # 每步 post-dt,830xw9hy@4(i4dxpbwy@4 = -0.1264)
BUILD1_ACTION_RATE_DOSE_CONVERGED = -0.0216  # 每步 post-dt,收敛(另一跑 -0.0241)
#: 退役的封顶档位。2026-08-08 之后**不再是现役值** —— 保留为"复活它要先跨过的那道坎"。
RETIRED_ACTION_RATE_VALUE_CLAMP = 9.0
#: 现役一阶平滑的权重:上游 isaaclab ``action_rate_l2``,无封顶,-0.1。
#: 真源 = train.py :: _REWARD_PACK_V2_DIRECT 的 ("action_rate_l2", -0.1)。
LIVE_ACTION_RATE_L2_WEIGHT = -0.1
POLICY_DT_S = 0.02
ACTION_DIM = 31


def _action_rate_env(per_joint_step, n_joints=31):
    """构造一个 ||Δa||² = n_joints * per_joint_step² 的两步动作序列。"""

    prev = torch.zeros(1, n_joints)
    cur = torch.full((1, n_joints), per_joint_step)
    mgr = types.SimpleNamespace(action=cur, prev_action=prev)
    return types.SimpleNamespace(action_manager=mgr)


def _action_rate_value(sq_norm, clamp, n_joints=31):
    env = _action_rate_env(math.sqrt(sq_norm / n_joints), n_joints)
    return hope_rewards_mod.action_rate_l2_clamped(env, value_clamp=clamp)[0].item()


def test_action_rate_is_deterministic_on_the_same_sequence():
    """同一个序列喂两次 -> 逐位相同。响应性测试的对照组,先排除随机性。"""

    for clamp in (RETIRED_ACTION_RATE_VALUE_CLAMP, 128.0):
        first = _action_rate_value(BUILD1_ACTION_RATE_SQ_EARLY, clamp)
        second = _action_rate_value(BUILD1_ACTION_RATE_SQ_EARLY, clamp)
        assert first == second


def test_retired_clamp_was_dead_across_build1s_entire_operating_range():
    """退役理由本身:9.0 那一档下,开局与收敛这两个差 5.8 倍的动作序列吐**同一个数**。

    这就是 s15r1 里那项焊死的机制,也是 2026-08-08 把它退役的全部理由。断言故意写成
    "必须相等" —— 谁把 9.0 复活并改成别的数,这条会红,提醒他把参照数字一起更新。
    """

    early = _action_rate_value(
        BUILD1_ACTION_RATE_SQ_EARLY, RETIRED_ACTION_RATE_VALUE_CLAMP
    )
    converged = _action_rate_value(
        BUILD1_ACTION_RATE_SQ_CONVERGED, RETIRED_ACTION_RATE_VALUE_CLAMP
    )
    assert early == converged == pytest.approx(
        RETIRED_ACTION_RATE_VALUE_CLAMP, abs=1e-6
    )
    assert BUILD1_ACTION_RATE_SQ_CONVERGED > RETIRED_ACTION_RATE_VALUE_CLAMP


def test_a_clamp_above_the_operating_range_restores_responsiveness():
    """误拦的不再拦:把档位抬到工作区之上,两个不同序列必须给出不同的数,且顺序正确。"""

    clamp = 128.0  # > build_1 开局的 63.1
    early = _action_rate_value(BUILD1_ACTION_RATE_SQ_EARLY, clamp)
    converged = _action_rate_value(BUILD1_ACTION_RATE_SQ_CONVERGED, clamp)
    assert early != converged
    assert early == pytest.approx(BUILD1_ACTION_RATE_SQ_EARLY, rel=1e-6)
    assert converged == pytest.approx(BUILD1_ACTION_RATE_SQ_CONVERGED, rel=1e-6)
    assert early > converged  # 抖得越狠罚得越多 —— 这正是被封顶抹掉的那个梯度


def test_clamp_still_caps_a_genuine_outlier():
    """粗一档就过不了:抬档位不等于取消封顶,离谱的一帧仍然被削平。"""

    clamp = 128.0
    assert _action_rate_value(4.0e4, clamp) == pytest.approx(clamp, abs=1e-6)


# --------------------------------------------------------------------------------------------- #
# 现役形状(2026-08-08 Franco 裁定:一阶平滑照开源对齐)
#
# 现役项 = 上游 isaaclab ``mdp.action_rate_l2`` = ``sum((a_t − a_{t−1})²)``,
# raw 动作、SUM over 31 维、**无上界**、不做相位门控。四家逐字同形:
#   * 我们自己的上游 BeyondMimic  tasks/tracking/tracking_env_cfg.py:237  weight −1e-1
#   * mjlab-tracking            src/mjlab/tasks/tracking/tracking_env_cfg.py  weight −1e-1
#   * unitree_rl_lab-mimic      tasks/mimic/robots/g1_29dof/*/tracking_env_cfg.py  weight −1e-1
#   (unitree 步行臂 −0.05;智元 AMP parkour 的 −1e-3 活在 discriminator 收入经济里,勿抄。)
#
# 这一组是"改完后该项必须随策略变化"的变异测试。它不测封顶版 —— 封顶版已经退役。
# --------------------------------------------------------------------------------------------- #
def _upstream_action_rate_l2(current, previous):
    """上游 ``action_rate_l2`` 的逐字复刻,只用来当参照实现(不是被测对象)。

    isaaclab 2.1.0 ``envs/mdp/rewards.py:245-247``::

        return torch.sum(torch.square(env.action_manager.action
                                      - env.action_manager.prev_action), dim=1)
    """

    return torch.sum(torch.square(current - previous), dim=1)


def _sq_norm_sequence(sq_norm, n_joints=ACTION_DIM):
    prev = torch.zeros(1, n_joints)
    cur = torch.full((1, n_joints), math.sqrt(sq_norm / n_joints))
    return cur, prev


def test_live_action_rate_shape_responds_to_the_policy():
    """该动的要动:开局与收敛这两个差 5.8 倍的序列,现役形状必须给出**不同**的数。

    这正是封顶版抹掉的那个梯度 —— 同样两个输入在 9.0 档下是同一个数(上面那条断言)。
    """

    early = _upstream_action_rate_l2(*_sq_norm_sequence(BUILD1_ACTION_RATE_SQ_EARLY))
    converged = _upstream_action_rate_l2(
        *_sq_norm_sequence(BUILD1_ACTION_RATE_SQ_CONVERGED)
    )
    assert early.item() != converged.item()
    assert early.item() > converged.item()   # 抖得越狠罚得越多
    assert early.item() == pytest.approx(BUILD1_ACTION_RATE_SQ_EARLY, rel=1e-6)
    assert converged.item() == pytest.approx(BUILD1_ACTION_RATE_SQ_CONVERGED, rel=1e-6)


def test_live_action_rate_shape_is_deterministic_on_the_same_sequence():
    """对照组:同一个序列喂两次必须逐位相同 —— 上面那条的"不同"才有意义。"""

    cur, prev = _sq_norm_sequence(BUILD1_ACTION_RATE_SQ_EARLY)
    assert (
        _upstream_action_rate_l2(cur, prev).item()
        == _upstream_action_rate_l2(cur, prev).item()
    )


def test_live_action_rate_shape_has_no_ceiling():
    """误拦的不再拦:没有任何输入会把它压平 —— 4e4 的离谱一帧照实付 4e4。

    粗一档就过不了:任何人重新引入哪怕 128.0 的封顶,这条立刻红。
    """

    huge = _upstream_action_rate_l2(*_sq_norm_sequence(4.0e4))
    assert huge.item() == pytest.approx(4.0e4, rel=1e-5)
    # 线性无上界:输入翻 4 倍,输出就翻 4 倍(封顶会让这个比值塌成 1)
    base = _upstream_action_rate_l2(*_sq_norm_sequence(1.0e4))
    assert huge.item() / base.item() == pytest.approx(4.0, rel=1e-5)


def test_live_weight_reproduces_build1s_measured_dose_at_matched_policy_level():
    """权重不是抄的号码,是同等策略水平下的交叉验证。

    我们 u0--u4 实测 policy_std_mean ≈ 1.001、动作 31 维、rsl_rl 每步独立采样
    ⇒ E‖Δa‖² = 2 × 31 × σ² = 62.1;build_1(同底盘、同 31 维、σ 同为 1.0)在 iter 4
    实测每步 −0.1262 / −0.1264 ⇒ 反推 ‖Δa‖² = 63.1。两条独立路径差 1.6%。
    """

    analytic_sq_norm = 2.0 * ACTION_DIM * (1.001 ** 2)
    assert analytic_sq_norm == pytest.approx(BUILD1_ACTION_RATE_SQ_EARLY, rel=0.02)

    predicted_dose = LIVE_ACTION_RATE_L2_WEIGHT * BUILD1_ACTION_RATE_SQ_EARLY * POLICY_DT_S
    assert predicted_dose == pytest.approx(BUILD1_ACTION_RATE_DOSE_EARLY, abs=1e-6)

    converged_dose = (
        LIVE_ACTION_RATE_L2_WEIGHT * BUILD1_ACTION_RATE_SQ_CONVERGED * POLICY_DT_S
    )
    assert converged_dose == pytest.approx(BUILD1_ACTION_RATE_DOSE_CONVERGED, abs=1e-6)
    # 衰减 5 倍以上 —— 这才是 build_1 |负|/正 穿过 1.0 的引擎,封顶版把它焊死了。
    assert BUILD1_ACTION_RATE_DOSE_EARLY / BUILD1_ACTION_RATE_DOSE_CONVERGED > 5.0


def test_repo_does_not_shadow_the_upstream_action_rate_l2():
    """"照开源对齐"的最强形式:我们连第二份实现都不写。

    ``mdp/__init__.py`` 先 ``from isaaclab.envs.mdp import *``,再 ``from .*rewards import *``。
    只要我们包里任何一个模块 ``def action_rate_l2``,它就会**静默盖掉**上游那条 —— 那正是
    "指纹对上、语义早漂"的典型入口。这条扫描源码而不是问对象,所以 isaaclab 被 stub 掉也有效。
    """

    mdp_dir = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    )
    offenders = [
        path.name
        for path in sorted(mdp_dir.glob("*.py"))
        if re.search(r"^def action_rate_l2\s*\(", path.read_text(), re.MULTILINE)
    ]
    assert offenders == [], (
        f"{offenders} 定义了 action_rate_l2,会盖掉上游那条"
    )


# --------------------------------------------------------------------------------------------- #
# virtual_landing 延付制(v2.2 修订:重生刷分漏洞的解)— 逐步推演
# --------------------------------------------------------------------------------------------- #
def _vb_env_with_clock(n=1):
    env, cmd = _vb_env(n)
    cmd._age = torch.zeros(n)
    cmd._same = torch.ones(n, dtype=torch.bool)
    cmd.post_strike_age_and_same_attempt = lambda: (cmd._age, cmd._same)
    return env, cmd


def _step(env, cmd):
    return hope_rewards_mod.virtual_landing(
        env, "racket_target", mode="legal_base", settle_delay_s=0.24
    )


def test_deferred_prize_pays_only_after_surviving_the_delay():
    env, cmd = _vb_env_with_clock()
    for t_ in (cmd.vb_landing_valid, cmd.vb_net_clear, cmd.vb_on_opponent):
        t_[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 0.0])  # 满分落点
    cmd.vb_fired[:] = True; cmd._age[:] = 0.0
    assert _step(env, cmd)[0].item() == 0.0            # 触球步:挂账不发
    cmd.vb_fired[:] = False; cmd._age[:] = 0.12
    assert _step(env, cmd)[0].item() == 0.0            # 6 步:未到期
    cmd._age[:] = 0.24
    assert _step(env, cmd)[0].item() == pytest.approx(1.0, abs=1e-6)  # 到期存活:全额
    cmd._age[:] = 0.26
    assert _step(env, cmd)[0].item() == 0.0            # 只发一次


def test_deferred_prize_is_forfeited_on_death():
    env, cmd = _vb_env_with_clock()
    for t_ in (cmd.vb_landing_valid, cmd.vb_net_clear, cmd.vb_on_opponent):
        t_[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 0.0])
    cmd.vb_fired[:] = True; cmd._age[:] = 0.0
    _step(env, cmd)
    cmd.vb_fired[:] = False
    cmd._same[:] = False                                # 摔死/重置:attempt 终结
    assert _step(env, cmd)[0].item() == 0.0             # 没收
    cmd._same[:] = True; cmd._age[:] = 0.02             # 重生新 attempt(未触球)
    assert _step(env, cmd)[0].item() == 0.0             # 旧奖不复活
    cmd._age[:] = 0.30
    assert _step(env, cmd)[0].item() == 0.0


def test_deferred_prize_new_fire_overwrites_pending():
    env, cmd = _vb_env_with_clock()
    for t_ in (cmd.vb_landing_valid, cmd.vb_net_clear, cmd.vb_on_opponent):
        t_[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 1.0])  # 第一板台角:0.6+0.4e^-1
    cmd.vb_fired[:] = True; cmd._age[:] = 0.0
    _step(env, cmd)
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 0.0])  # 第二板满分,覆盖挂账
    cmd._age[:] = 0.0
    _step(env, cmd)
    cmd.vb_fired[:] = False; cmd._age[:] = 0.24
    assert _step(env, cmd)[0].item() == pytest.approx(1.0, abs=1e-6)


def test_zero_delay_keeps_instant_semantics():
    env, cmd = _vb_env(1)
    for t_ in (cmd.vb_landing_valid, cmd.vb_net_clear, cmd.vb_on_opponent, cmd.vb_fired):
        t_[:] = True
    cmd.vb_landing_xy[0] = torch.tensor([2.555, 0.0])
    v = hope_rewards_mod.virtual_landing(env, "racket_target", mode="legal_base")
    assert v[0].item() == pytest.approx(1.0, abs=1e-6)
