"""mjlab-ported foot-contact shaping + 击球窗下肢模仿衰减 — host-only unit tests.

Pinned here (all CPU, isaaclab stubbed via test_reward_flags_mdp):

* foot_soft_landing math — first-contact-only 计费、history 峰值(decimation 下冲击尖峰在旧
  子步也要抓到)、拉扯负 z clamp 到 0、超阈按阈值归一、单脚封顶 3.0、双脚求和;量纲结论:
  输出是【无量纲超阈倍数】不是牛顿 —— mjlab -1e-5/N 的等效剂量 = -1e-5 x threshold(默认
  300 N -> -3e-3),波配置别抄 -0.1 也别抄 -1e-4。
* foot_clearance math — 触地脚(>10 N,与 sensor force_threshold 同值)完全免费;摆动脚付
  clamp(目标高-脚高, min=0) x 水平速度(2026-07-25 起单侧:只罚抬脚不足,多抬免费——旧双侧
  |·| 会在凸包顶把脚往下按,粗糙地形上帮倒忙);左右脚名单必须同序同名否则 fail-loud。
* 两个函数的 fail-loud 面:脚数不是 2、sensor 缺 compute_first_contact(track_air_time 没开)、
  history 形状不对、张量缺失、阈值/目标高非法。
* train.py 覆盖层往返:foot_* 四键进 _REWARD_KEYS 白名单与 null-删参表;param-without-weight
  拒收;weight 必须 finite 且 <= 0(显式 0 = 对照);参数必须 finite 且 > 0;v1 兜底路径
  (reward_pack=v1、不写其他键)对 cfg 零改动 = 字节等价(2026-07-25 默认翻转后,真·默认
  路径 = v2 展开,由 test_reward_flags_overrides JOB1 区块专测;本套件经 _apply_legacy_v1
  钉 v1 维持 legacy 断言)。
* lower_body_imitation_scale_in_window — Wave-B 信封(配了必须显式给 B1/B2 两个 weight);
  写进 term.params 且探针 params 跟随;函数级:默认 1.0 原样返回,k != 1 只把 WIDE 击球窗内
  的 env 乘 k,台账/签名记未衰减原值;窗口形状/dtype 不对 fail-loud。
* hope_env_cfg / tracking_env_cfg 源码级守卫:默认 weight=0.0、阈值 300 N、目标高 0.08 m、
  contact sensor track_air_time=True 且 history_length >= 1(foot_soft_landing 的前提)。

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_foot_contact_shaping.py -q
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _NS, _Term, _apply_legacy_v1, _make_env_cfg, train_mod

ROOT = Path(__file__).resolve().parents[1]
ENV_CFG_SRC = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
).read_text(encoding="utf-8")
TRACKING_CFG_SRC = (
    ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
).read_text(encoding="utf-8")


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


def _landing_env(
    *,
    n=2,
    hist=3,
    bodies=4,
    feet_ids=(1, 3),
    first_contact=None,
    forces_hist=None,
    with_first_contact_fn=True,
):
    """foot_soft_landing fake: sensor with first-contact mask + force history."""

    if forces_hist is None:
        forces_hist = torch.zeros(n, hist, bodies, 3)
    if first_contact is None:
        first_contact = torch.zeros(n, bodies, dtype=torch.bool)

    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(net_forces_w_history=forces_hist),
    )
    if with_first_contact_fn:
        sensor.compute_first_contact = lambda dt: first_contact
    env = types.SimpleNamespace(
        step_dt=0.02,
        scene=_Scene(sensors={"contact_forces": sensor}),
    )
    cfg = _NS(name="contact_forces", body_ids=list(feet_ids))
    return env, cfg, forces_hist, first_contact


def _clearance_env(
    *,
    n=2,
    sensor_ids=(0, 1),
    asset_ids=(2, 5),
    names=("left_ankle_roll", "right_ankle_roll"),
    asset_names=None,
    forces=None,
    pos=None,
    vel=None,
):
    """foot_clearance fake: sensor contact forces + articulation foot pos/vel."""

    n_sensor_bodies = max(sensor_ids) + 1
    n_asset_bodies = max(asset_ids) + 1
    if forces is None:
        forces = torch.zeros(n, n_sensor_bodies, 3)
    if pos is None:
        pos = torch.zeros(n, n_asset_bodies, 3)
    if vel is None:
        vel = torch.zeros(n, n_asset_bodies, 3)
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=forces))
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(body_pos_w=pos, body_lin_vel_w=vel)
    )
    env = types.SimpleNamespace(
        scene=_Scene(sensors={"contact_forces": sensor}, entities={"robot": asset})
    )
    sensor_cfg = _NS(
        name="contact_forces", body_ids=list(sensor_ids), body_names=list(names)
    )
    asset_cfg = _NS(
        name="robot",
        body_ids=list(asset_ids),
        body_names=list(asset_names if asset_names is not None else names),
    )
    return env, sensor_cfg, asset_cfg, forces, pos, vel


# --------------------------------------------------------------------------------------------- #
# foot_soft_landing math (hand-computed)
# --------------------------------------------------------------------------------------------- #
def test_soft_landing_pays_only_on_first_contact_excess():
    env, cfg, forces, fc = _landing_env()
    # env0 foot0 (body 1): first contact, peak 450 N -> excess (450-300)/300 = 0.5
    fc[0, 1] = True
    forces[0, 0, 1, 2] = 450.0
    # env0 foot1 (body 3): NOT first contact, huge support force -> free (那是体重不是冲击)
    forces[0, :, 3, 2] = 9999.0
    # env1 both feet first contact but below threshold -> free
    fc[1, 1] = True
    fc[1, 3] = True
    forces[1, 0, 1, 2] = 100.0
    forces[1, 0, 3, 2] = 250.0
    value = hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=300.0)
    assert value.tolist() == pytest.approx([0.5, 0.0], abs=1e-6)


def test_soft_landing_takes_peak_across_history_not_current_frame():
    env, cfg, forces, fc = _landing_env()
    fc[0, 1] = True
    forces[0, 0, 1, 2] = 50.0  # 当前子步很小
    forces[0, 2, 1, 2] = 600.0  # 冲击尖峰在旧 history 子步 -> (600-300)/300 = 1.0
    value = hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=300.0)
    assert value.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)


def test_soft_landing_clamps_pull_forces_and_caps_per_foot():
    env, cfg, forces, fc = _landing_env()
    # env0 foot0: 拉扯(负 z)不算冲击
    fc[0, 1] = True
    forces[0, :, 1, 2] = -500.0
    # env1 双脚暴力落地:3000 N -> 9 倍超阈,单脚封顶 3.0;双脚求和 = 6.0
    fc[1, 1] = True
    fc[1, 3] = True
    forces[1, 0, 1, 2] = 3000.0
    forces[1, 1, 3, 2] = 3000.0
    value = hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=300.0)
    assert value.tolist() == pytest.approx([0.0, 6.0], abs=1e-6)


def test_soft_landing_output_is_dimensionless_not_newton():
    # 量纲守卫:同一冲击、阈值翻倍 -> 输出按 (peak-thr)/thr 缩,不是原始牛顿数。
    env, cfg, forces, fc = _landing_env()
    fc[0, 1] = True
    forces[0, 0, 1, 2] = 900.0
    v300 = hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=300.0)
    v450 = hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=450.0)
    assert v300[0].item() == pytest.approx(2.0, abs=1e-6)  # (900-300)/300
    assert v450[0].item() == pytest.approx(1.0, abs=1e-6)  # (900-450)/450


def test_soft_landing_fail_loud_surfaces():
    env, cfg, _, _ = _landing_env()
    with pytest.raises(ValueError):
        hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=0.0)
    with pytest.raises(ValueError):
        hope_rewards_mod.foot_soft_landing(env, cfg, force_threshold_n=float("nan"))
    # 脚数不是 2
    bad = _NS(name="contact_forces", body_ids=[1])
    with pytest.raises(RuntimeError, match="exactly the two A3 feet"):
        hope_rewards_mod.foot_soft_landing(env, bad, force_threshold_n=300.0)
    # sensor 没开 track_air_time -> 没有 compute_first_contact
    env2, cfg2, _, _ = _landing_env(with_first_contact_fn=False)
    with pytest.raises(RuntimeError, match="track_air_time"):
        hope_rewards_mod.foot_soft_landing(env2, cfg2, force_threshold_n=300.0)
    # history 形状不对(缺 hist 维)
    env3, cfg3, _, _ = _landing_env()
    env3.scene.sensors["contact_forces"].data.net_forces_w_history = torch.zeros(2, 4, 3)
    with pytest.raises(RuntimeError, match="net_forces_w_history"):
        hope_rewards_mod.foot_soft_landing(env3, cfg3, force_threshold_n=300.0)


# --------------------------------------------------------------------------------------------- #
# foot_clearance math (hand-computed)
# --------------------------------------------------------------------------------------------- #
def test_clearance_charges_swing_feet_only():
    env, s_cfg, a_cfg, forces, pos, vel = _clearance_env()
    # env0 foot0: 触地(50 N)、高度乱飘 -> 免费(那是 foot_slip/foot_drag 的地盘)
    forces[0, 0, 2] = 50.0
    pos[0, 2, 2] = 0.50
    vel[0, 2, :2] = torch.tensor([2.0, 0.0])
    # env0 foot1: 摆动、z=0.02、xy 速度 (0.6, 0.8) -> clamp(0.08-0.02) x 1.0 = 0.06
    pos[0, 5, 2] = 0.02
    vel[0, 5, :2] = torch.tensor([0.6, 0.8])
    # env1 foot0: 摆动、z=0.30 已高于目标、xy 速度 (3,4)=5 -> 单侧化后多抬免费(0.0;
    # 2026-07-25 前双侧 |0.30-0.08| x 5 = 1.1 会把高处的脚往下按,粗糙地形上帮倒忙)
    pos[1, 2, 2] = 0.30
    vel[1, 2, :2] = torch.tensor([3.0, 4.0])
    # env1 foot1: 摆动但原地不动 -> 免费(慢慢挪几乎免费)
    pos[1, 5, 2] = 0.50
    value = hope_rewards_mod.foot_clearance(env, s_cfg, a_cfg, target_m=0.08)
    assert value.tolist() == pytest.approx([0.06, 0.0], abs=1e-6)


def test_clearance_contact_threshold_matches_sensor_10n():
    env, s_cfg, a_cfg, forces, pos, vel = _clearance_env()
    pos[0, 2, 2] = 0.02  # 低于目标 0.08 -> 摆动时欠 0.06
    vel[0, 2, :2] = torch.tensor([1.0, 0.0])
    forces[0, 0, 2] = 9.9  # < 10 N = 仍算摆动(与 sensor force_threshold / in_contact 同判据)
    assert hope_rewards_mod.foot_clearance(env, s_cfg, a_cfg, target_m=0.08)[
        0
    ].item() == pytest.approx(0.06, abs=1e-6)
    forces[0, 0, 2] = 10.1  # > 10 N = 触地,免费
    assert hope_rewards_mod.foot_clearance(env, s_cfg, a_cfg, target_m=0.08)[
        0
    ].item() == pytest.approx(0.0, abs=1e-6)


def test_clearance_fail_loud_surfaces():
    env, s_cfg, a_cfg, *_ = _clearance_env()
    with pytest.raises(ValueError):
        hope_rewards_mod.foot_clearance(env, s_cfg, a_cfg, target_m=0.0)
    # 左右脚名单不同序:左脚接触配右脚速度会静默算错,必须炸
    env2, s_cfg2, a_cfg2, *_ = _clearance_env(
        asset_names=("right_ankle_roll", "left_ankle_roll")
    )
    with pytest.raises(RuntimeError, match="SAME two feet"):
        hope_rewards_mod.foot_clearance(env2, s_cfg2, a_cfg2, target_m=0.08)
    # 张量缺失
    env3, s_cfg3, a_cfg3, *_ = _clearance_env()
    env3.scene.sensors["contact_forces"].data.net_forces_w = None
    with pytest.raises(RuntimeError, match="foot_clearance requires"):
        hope_rewards_mod.foot_clearance(env3, s_cfg3, a_cfg3, target_m=0.08)


# --------------------------------------------------------------------------------------------- #
# train.py override translation (foot keys)
# --------------------------------------------------------------------------------------------- #
_SOFT_PARAMS = {"sensor_cfg": "SENTINEL_SENSOR", "force_threshold_n": 300.0}
_CLEAR_PARAMS = {
    "sensor_cfg": "SENTINEL_SENSOR",
    "asset_cfg": "SENTINEL_ASSET",
    "target_m": 0.08,
}


def _foot_env_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.foot_soft_landing = _Term(weight=0.0, params=dict(_SOFT_PARAMS))
    cfg.rewards.foot_clearance = _Term(weight=0.0, params=dict(_CLEAR_PARAMS))
    return cfg


def _apply_foot(task, cfg=None):
    # 2026-07-25 默认翻转后本套件仍测 legacy 翻译行为:钉 v1 + 滤 v1 记账行,原断言原样成立
    # (默认路径 = v2 展开,由 test_reward_flags_overrides JOB1 区块专测)。
    cfg = cfg if cfg is not None else _foot_env_cfg()
    applied = _apply_legacy_v1(cfg, task)
    return cfg, applied


def test_foot_keys_are_whitelisted_and_null_removable():
    for key in (
        "foot_soft_landing_weight",
        "foot_soft_landing_force_threshold_n",
        "foot_clearance_weight",
        "foot_clearance_target_m",
        "foot_slip_sq_weight",
        "foot_drag_weight",
        "lower_body_imitation_scale_in_window",
    ):
        assert key in train_mod._REWARD_KEYS
    assert train_mod._REWARD_NULL_REMOVABLE_PARAMS["foot_soft_landing_force_threshold_n"] == (
        "foot_soft_landing",
        "force_threshold_n",
    )
    assert train_mod._REWARD_NULL_REMOVABLE_PARAMS["foot_clearance_target_m"] == (
        "foot_clearance",
        "target_m",
    )


def test_foot_weight_and_param_roundtrip():
    cfg, applied = _apply_foot({"rewards": {
        "foot_soft_landing_weight": -3e-3,
        "foot_soft_landing_force_threshold_n": 350.0,
        "foot_clearance_weight": -0.05,
        "foot_clearance_target_m": 0.15,
    }})
    assert cfg.rewards.foot_soft_landing.weight == -3e-3
    assert cfg.rewards.foot_soft_landing.params["force_threshold_n"] == 350.0
    assert cfg.rewards.foot_soft_landing.params["sensor_cfg"] == "SENTINEL_SENSOR"
    assert cfg.rewards.foot_clearance.weight == -0.05
    assert cfg.rewards.foot_clearance.params["target_m"] == 0.15
    assert "rewards.foot_soft_landing.weight=-0.003" in applied
    assert "rewards.foot_clearance.weight=-0.05" in applied
    assert any("force_threshold_n=350.0" in line for line in applied)
    assert any("target_m=0.15" in line for line in applied)


def test_foot_explicit_zero_weight_is_a_measured_control():
    cfg, applied = _apply_foot({"rewards": {"foot_soft_landing_weight": 0.0}})
    assert cfg.rewards.foot_soft_landing.weight == 0.0
    assert "rewards.foot_soft_landing.weight=0.0" in applied


def test_foot_param_without_weight_refused():
    with pytest.raises(train_mod._OverrideError, match="requires foot_soft_landing_weight"):
        _apply_foot({"rewards": {"foot_soft_landing_force_threshold_n": 350.0}})
    with pytest.raises(train_mod._OverrideError, match="requires foot_clearance_weight"):
        _apply_foot({"rewards": {"foot_clearance_target_m": 0.15}})


@pytest.mark.parametrize("bad", [0.1, 1.0, True, float("nan"), float("inf"), "x"])
def test_foot_weight_must_be_finite_nonpositive(bad):
    with pytest.raises(train_mod._OverrideError, match="finite and <= 0"):
        _apply_foot({"rewards": {"foot_soft_landing_weight": bad}})


@pytest.mark.parametrize("bad", [0.0, -1.0, True, float("nan"), "x"])
def test_foot_params_must_be_finite_positive(bad):
    with pytest.raises(train_mod._OverrideError, match="finite and > 0"):
        _apply_foot({"rewards": {
            "foot_soft_landing_weight": -1e-3,
            "foot_soft_landing_force_threshold_n": bad,
        }})
    with pytest.raises(train_mod._OverrideError, match="finite and > 0"):
        _apply_foot({"rewards": {
            "foot_clearance_weight": -1e-3,
            "foot_clearance_target_m": bad,
        }})


def test_foot_v1_baseline_path_is_byte_identical():
    cfg, _ = _apply_foot({"rewards": {"racket_position_weight": 14.0}})
    assert cfg.rewards.foot_soft_landing.weight == 0.0
    assert cfg.rewards.foot_soft_landing.params == _SOFT_PARAMS
    assert cfg.rewards.foot_clearance.weight == 0.0
    assert cfg.rewards.foot_clearance.params == _CLEAR_PARAMS


def test_foot_yaml_null_removes_inherited_param():
    cfg, applied = _apply_foot({"rewards": {"foot_soft_landing_force_threshold_n": None}})
    assert "force_threshold_n" not in cfg.rewards.foot_soft_landing.params
    assert applied == [
        "rewards.foot_soft_landing.params.force_threshold_n=<removed by YAML null>"
    ]


# --------------------------------------------------------------------------------------------- #
# foot_slip_sq / foot_drag 剂量键(2026-07-22:此前源码常开 -1.0/-0.5、CLI 够不着;
# penlight 减负臂与将来消融需要显式剂量)
# --------------------------------------------------------------------------------------------- #
def _slip_env_cfg():
    cfg = _foot_env_cfg()
    cfg.rewards.foot_slip_sq = _Term(weight=-1.0, params={"command_name": "racket_target"})
    cfg.rewards.foot_drag = _Term(weight=-0.5, params={"command_name": "racket_target"})
    return cfg


def test_foot_slip_and_drag_weight_roundtrip():
    cfg = _slip_env_cfg()
    cfg, applied = _apply_foot(
        {"rewards": {"foot_slip_sq_weight": -0.33, "foot_drag_weight": -0.17}},
        cfg=cfg,
    )
    assert cfg.rewards.foot_slip_sq.weight == -0.33
    assert cfg.rewards.foot_drag.weight == -0.17
    assert "rewards.foot_slip_sq.weight=-0.33" in applied
    assert "rewards.foot_drag.weight=-0.17" in applied


def test_foot_slip_and_drag_defaults_untouched():
    cfg = _slip_env_cfg()
    cfg, _ = _apply_foot({"rewards": {"racket_position_weight": 14.0}}, cfg=cfg)
    assert cfg.rewards.foot_slip_sq.weight == -1.0  # 源码常开默认,不写键=不动
    assert cfg.rewards.foot_drag.weight == -0.5


# --------------------------------------------------------------------------------------------- #
# lower_body_imitation_scale_in_window — override translation
# --------------------------------------------------------------------------------------------- #
def test_scale_in_window_requires_wave_b_envelope():
    with pytest.raises(train_mod._OverrideError, match="Wave-B B0/B1/B2 requires both"):
        _apply_foot({"rewards": {"lower_body_imitation_scale_in_window": 0.25}})


def test_scale_in_window_writes_term_and_probe_params():
    cfg, applied = _apply_foot({"rewards": {
        "lower_body_pose_imitation_weight": 1.0,
        "lower_body_stability_bundle_weight": 0.0,
        "lower_body_imitation_scale_in_window": 0.25,
    }})
    pose = cfg.rewards.lower_body_pose_imitation
    probe = cfg.rewards.lower_body_pose_imitation_probe
    assert pose.params["scale_in_window"] == 0.25
    assert probe.params == pose.params  # 探针 params 全量跟随(clear + update)
    assert probe.weight == 1.0
    assert "rewards.lower_body_pose_imitation.params.scale_in_window=0.25" in applied


@pytest.mark.parametrize("bad", [-0.1, True, float("nan"), "x"])
def test_scale_in_window_rejects_bad_values(bad):
    with pytest.raises(train_mod._OverrideError):
        _apply_foot({"rewards": {
            "lower_body_pose_imitation_weight": 1.0,
            "lower_body_stability_bundle_weight": 0.0,
            "lower_body_imitation_scale_in_window": bad,
        }})


def test_scale_in_window_zero_means_full_mute_and_is_legal():
    cfg, _ = _apply_foot({"rewards": {
        "lower_body_pose_imitation_weight": 1.0,
        "lower_body_stability_bundle_weight": 0.0,
        "lower_body_imitation_scale_in_window": 0.0,
    }})
    assert cfg.rewards.lower_body_pose_imitation.params["scale_in_window"] == 0.0


# --------------------------------------------------------------------------------------------- #
# lower_body_pose_imitation scale_in_window — function-level windowing
# --------------------------------------------------------------------------------------------- #
def _patched_pose_values(monkeypatch, values, window):
    """Monkeypatch kernel + ledger; return the ledger capture list."""

    recorded = []
    zeros = torch.zeros_like(values)

    def fake_values(env, racket, motion, std, pre, post):
        return values, zeros.bool(), zeros, zeros

    def fake_record(env, vals, eligible, joint_error, reference_motion, *, reward_active, signature):
        recorded.append({"values": vals, "reward_active": reward_active, "signature": signature})

    monkeypatch.setattr(hope_rewards_mod, "_lower_body_pose_values", fake_values)
    monkeypatch.setattr(
        hope_rewards_mod, "_record_lower_body_pose_activation", fake_record
    )
    cmd = types.SimpleNamespace(strike_window_wide=window, strike_window=None)
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd)
    )
    return env, recorded


def test_scale_in_window_default_is_byte_identical(monkeypatch):
    values = torch.tensor([0.7, 0.3, 0.9])
    window = torch.tensor([True, False, True])
    env, recorded = _patched_pose_values(monkeypatch, values, window)
    out = hope_rewards_mod.lower_body_pose_imitation(env)
    assert out is values  # 默认 1.0:原张量原样返回,连乘法都不做
    assert recorded[-1]["signature"][-1] == 1.0


def test_scale_in_window_scales_only_wide_window_envs(monkeypatch):
    values = torch.tensor([0.8, 0.4, 1.0])
    window = torch.tensor([True, False, True])
    env, recorded = _patched_pose_values(monkeypatch, values, window)
    out = hope_rewards_mod.lower_body_pose_imitation(env, scale_in_window=0.25)
    assert out.tolist() == pytest.approx([0.2, 0.4, 0.25], abs=1e-6)
    # 台账记的是未衰减原值(对账用),签名带上系数(探针/reward 参数漂移会炸)
    assert torch.equal(recorded[-1]["values"], values)
    assert recorded[-1]["signature"][-1] == 0.25


def test_scale_in_window_zero_mutes_window_completely(monkeypatch):
    values = torch.tensor([0.8, 0.4])
    window = torch.tensor([True, False])
    env, _ = _patched_pose_values(monkeypatch, values, window)
    out = hope_rewards_mod.lower_body_pose_imitation(env, scale_in_window=0.0)
    assert out.tolist() == pytest.approx([0.0, 0.4], abs=1e-6)


def test_scale_in_window_requires_aligned_bool_window(monkeypatch):
    values = torch.tensor([0.8, 0.4])
    env, _ = _patched_pose_values(monkeypatch, values, torch.tensor([1.0, 0.0]))
    with pytest.raises(RuntimeError, match="strike-window mask"):
        hope_rewards_mod.lower_body_pose_imitation(env, scale_in_window=0.5)
    env2, _ = _patched_pose_values(monkeypatch, values, torch.tensor([True]))
    with pytest.raises(RuntimeError, match="strike-window mask"):
        hope_rewards_mod.lower_body_pose_imitation(env2, scale_in_window=0.5)


def test_probe_returns_zeros_and_logs_unscaled(monkeypatch):
    values = torch.tensor([0.8, 0.4])
    window = torch.tensor([True, True])
    env, recorded = _patched_pose_values(monkeypatch, values, window)
    out = hope_rewards_mod.lower_body_pose_imitation_probe(env, scale_in_window=0.25)
    assert out.tolist() == [0.0, 0.0]
    assert torch.equal(recorded[-1]["values"], values)
    assert recorded[-1]["reward_active"] is False
    assert recorded[-1]["signature"][-1] == 0.25


def test_wide_window_falls_back_to_strike_window(monkeypatch):
    # 1c 不拆窗时 strike_window_wide 为 None -> 用 strike_window(与 _window_wide 一致)
    values = torch.tensor([0.8, 0.4])
    env, _ = _patched_pose_values(monkeypatch, values, None)
    cmd = env.command_manager.get_term("racket_target")
    cmd.strike_window = torch.tensor([True, False])
    out = hope_rewards_mod.lower_body_pose_imitation(env, scale_in_window=0.5)
    assert out.tolist() == pytest.approx([0.4, 0.4], abs=1e-6)


# --------------------------------------------------------------------------------------------- #
# cfg-source guards (defaults stay OFF; sensor prerequisites hold)
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_defaults_are_off_and_match_documented_constants():
    soft = re.search(
        r"foot_soft_landing = RewTerm\((.*?)\n    \)", ENV_CFG_SRC, re.DOTALL
    )
    clear = re.search(r"foot_clearance = RewTerm\((.*?)\n    \)", ENV_CFG_SRC, re.DOTALL)
    assert soft and "weight=0.0" in soft.group(1)
    assert '"force_threshold_n": 300.0' in soft.group(1)
    assert clear and "weight=0.0" in clear.group(1)
    assert '"target_m": 0.08' in clear.group(1)
    assert soft.group(1).count("A3_FEET_BODIES") == 1
    assert clear.group(1).count("A3_FEET_BODIES") == 2  # sensor + articulation 同名单


def test_contact_sensor_prerequisites_for_soft_landing():
    sensor = re.search(r"contact_forces = ContactSensorCfg\((.*?)\)\n", TRACKING_CFG_SRC, re.DOTALL)
    assert sensor is not None
    assert "track_air_time=True" in sensor.group(1)  # compute_first_contact 的前提
    hist = re.search(r"history_length=(\d+)", sensor.group(1))
    assert hist and int(hist.group(1)) >= 1  # net_forces_w_history 的前提
    assert "force_threshold=10.0" in sensor.group(1)  # foot_clearance 的 10 N 判据同源


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
