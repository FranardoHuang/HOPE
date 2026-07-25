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
# action_rate_l2_clamped(v2 值封顶;fresh 自杀区间的解)— 手算
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
