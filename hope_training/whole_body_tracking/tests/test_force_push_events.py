"""F 轴间隔持续力推(task.force_push.*)— 事件函数 + 覆盖翻译 + 合同测试,NO Isaac。

人话:这波要训"每隔几秒朝水平随机方向推底座一把恒力"的抗扰平衡,与速度推档位
(p02/p035/p05/p08)按同冲量对表:Δv_equiv = force_n × duration_s / m_robot,机器人总质量
运行时读 articulation 写进合同。默认必须全关:所有在跑矩阵格没有 task.force_push 键,
events.force_push / events.force_push_sweep 保持双 None、合同不写 force_push_event 块、
字节逐位不变。开了以后由单一来源的校验/装配(training_contract.force_push_event_block)
生成"施力事件 + 每控制步到期清扫事件"这一对,以及 schema-3 force_push_event 合同块。

Covers (mock 模式照 test_push_robot_events.py:train.py 直接 import;env cfg 是
plain-namespace 假件;training_contract.py / hope_push_events.py 按文件路径加载):

* 事件函数:默认无感(无账本 = 严格 no-op)、触发施力方向水平且模长 = force_n(含非单位
  四元数的 body 系旋转往返)、到期后力为零、多 env 独立到期、到期前重触发续期、非法参数 /
  body 缺失重名 / 抢账本 fail-closed;
* 覆盖翻译:absent == 逐字节 no-op;enable=false 干净路径 + 上膛参数拒绝;enable=true 造出
  施力 + 清扫两个 EventTerm(interval / params / 每控制步清扫间隔全部逐位核对);非法参数
  fail-loud(缺键、坏区间、非整数控制步的 duration、字符串/int 冒充浮点);
* 合同绑定(_force_push_event_contract):运行时质量 + Δv_equiv 记录、半接线拒绝(含
  "有施力没清扫 = 力永远清不掉")、走样事件形状拒绝;
* schema-3 校验(_validate_force_push_event_contract):JSON roundtrip 通过,篡改块
  (含 application_point 标成 COM 的 Yikang V9 反例)一律 raise;接进
  validate_schema3_contract_structure。

Run:  <python> -m pytest hope_training/whole_body_tracking/tests/test_force_push_events.py -q
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import train as train_mod  # noqa: E402  (hydra/omegaconf only at import time)
from test_reward_flags_overrides import _apply_legacy_v1  # noqa: E402  (v1 兜底钉扣,见下)

_TC_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)
_TC_SPEC = importlib.util.spec_from_file_location("force_push_tc_under_test", _TC_PATH)
TC = importlib.util.module_from_spec(_TC_SPEC)
_TC_SPEC.loader.exec_module(TC)

_EV_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_push_events.py"
)
_EV_SPEC = importlib.util.spec_from_file_location("hope_push_events_under_test", _EV_PATH)
EV = importlib.util.module_from_spec(_EV_SPEC)
_EV_SPEC.loader.exec_module(EV)

_HOPE_ENV_CFG_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)

_CONTROL_DT = 0.005 * 4  # tracking_env_cfg: sim.dt=0.005, decimation=4 -> 50 Hz control


# --------------------------------------------------------------------------------------------- #
# fakes: minimal asset/env surface the event functions touch (mock 模式照 test_push_robot_events)
# --------------------------------------------------------------------------------------------- #
class _NS(types.SimpleNamespace):
    pass


class _FakeAsset:
    """Articulation stand-in: find_bodies + set_external_force_and_torque + body quats."""

    def __init__(self, num_envs, body_names=("pelvis_link", "left_leg", "right_leg")):
        self.device = "cpu"
        self._body_names = list(body_names)
        quat = torch.zeros(num_envs, len(self._body_names), 4)
        quat[:, :, 0] = 1.0  # identity (w, x, y, z)
        self.data = _NS(body_quat_w=quat)
        self.applied_forces = None
        self.applied_torques = None
        self.applied_body_ids = None
        self.call_count = 0

    def find_bodies(self, name, preserve_order=True):
        ids = [i for i, n in enumerate(self._body_names) if n == name]
        if not ids:  # Isaac 的 resolve_matching_names 对无匹配也 raise
            raise ValueError(f"no body matches {name!r}")
        return ids, [self._body_names[i] for i in ids]

    def set_external_force_and_torque(self, forces, torques, body_ids=None, env_ids=None):
        self.call_count += 1
        self.applied_forces = forces.clone()
        self.applied_torques = torques.clone()
        self.applied_body_ids = None if body_ids is None else list(body_ids)


class _FakeEnv:
    def __init__(self, num_envs=4, body_names=("pelvis_link", "left_leg", "right_leg")):
        self.num_envs = num_envs
        self.scene = {"robot": _FakeAsset(num_envs, body_names=body_names)}
        self.common_step_counter = 0


# --------------------------------------------------------------------------------------------- #
# event function: trigger applies a horizontal constant force of exact magnitude
# --------------------------------------------------------------------------------------------- #
def test_trigger_applies_horizontal_force_with_exact_magnitude():
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=4)
    env.common_step_counter = 100
    EV.push_by_applying_wrench(env, torch.tensor([0, 2]), force_n=60.0, duration_steps=15)
    asset = env.scene["robot"]
    assert asset.applied_body_ids == [0]  # pelvis_link resolves to body 0
    forces = asset.applied_forces
    assert forces.shape == (4, 1, 3)
    for env_id in (0, 2):  # identity quat -> body frame == world frame
        f = forces[env_id, 0]
        assert float(f[2]) == pytest.approx(0.0, abs=1e-6)  # 水平:z 分量 0
        assert float(torch.linalg.norm(f)) == pytest.approx(60.0, rel=1e-6)  # 模长 = force_n
    for env_id in (1, 3):  # 没被抽中的 env 一牛都不挨
        assert torch.all(forces[env_id] == 0.0)
    assert torch.all(asset.applied_torques == 0.0)  # 无力矩
    state = getattr(env, EV.FORCE_PUSH_STATE_ATTR)
    assert state["expiry_step"].tolist() == [115, -1, 115, -1]


def test_trigger_rotated_body_frame_force_is_world_horizontal():
    """Isaac Lab 2.1 外力是 body 系:歪着的 pelvis 也必须收到"世界系水平"的力。"""
    torch.manual_seed(1)
    env = _FakeEnv(num_envs=2)
    asset = env.scene["robot"]
    half = math.sqrt(0.5)
    asset.data.body_quat_w[1, 0] = torch.tensor([half, half, 0.0, 0.0])  # 90 deg roll
    EV.push_by_applying_wrench(env, torch.tensor([1]), force_n=45.0, duration_steps=15)
    force_b = asset.applied_forces[1, 0].unsqueeze(0)
    assert float(torch.linalg.norm(force_b)) == pytest.approx(45.0, rel=1e-5)
    force_w = EV._quat_rotate_wxyz(asset.data.body_quat_w[1, 0].unsqueeze(0), force_b)[0]
    assert float(force_w[2]) == pytest.approx(0.0, abs=1e-4)  # 转回世界系仍是水平力


def test_expiry_clears_force_and_sweep_goes_quiet():
    torch.manual_seed(2)
    env = _FakeEnv(num_envs=3)
    env.common_step_counter = 100
    EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)
    asset = env.scene["robot"]
    env.common_step_counter = 114  # expiry=115,还没到
    EV.sweep_expired_force_pushes(env)
    assert float(torch.linalg.norm(asset.applied_forces[0, 0])) == pytest.approx(60.0, rel=1e-6)
    env.common_step_counter = 115  # 到期:力必须真的清零
    EV.sweep_expired_force_pushes(env)
    assert torch.all(asset.applied_forces == 0.0)
    state = getattr(env, EV.FORCE_PUSH_STATE_ATTR)
    assert state["applied_any"] is False
    calls_after_clear = asset.call_count
    env.common_step_counter = 116  # 清干净之后清扫是严格 no-op(不跟别的外力写者抢)
    EV.sweep_expired_force_pushes(env)
    assert asset.call_count == calls_after_clear


def test_multi_env_independent_expiry():
    torch.manual_seed(3)
    env = _FakeEnv(num_envs=4)
    asset = env.scene["robot"]
    env.common_step_counter = 100
    EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)
    env.common_step_counter = 105
    EV.push_by_applying_wrench(env, torch.tensor([1]), force_n=60.0, duration_steps=15)
    env.common_step_counter = 115  # env0 到期(115),env1 还有 5 步(120)
    EV.sweep_expired_force_pushes(env)
    assert torch.all(asset.applied_forces[0] == 0.0)
    assert float(torch.linalg.norm(asset.applied_forces[1, 0])) == pytest.approx(60.0, rel=1e-6)
    env.common_step_counter = 120
    EV.sweep_expired_force_pushes(env)
    assert torch.all(asset.applied_forces == 0.0)


def test_retrigger_before_expiry_renews_the_clock():
    torch.manual_seed(4)
    env = _FakeEnv(num_envs=2)
    env.common_step_counter = 100
    EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)
    env.common_step_counter = 110
    EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)
    env.common_step_counter = 115  # 老到期步:重触发已把到期续到 125
    EV.sweep_expired_force_pushes(env)
    asset = env.scene["robot"]
    assert float(torch.linalg.norm(asset.applied_forces[0, 0])) == pytest.approx(60.0, rel=1e-6)
    env.common_step_counter = 125
    EV.sweep_expired_force_pushes(env)
    assert torch.all(asset.applied_forces == 0.0)


def test_sweep_without_state_is_total_noop():
    env = _FakeEnv(num_envs=2)
    EV.sweep_expired_force_pushes(env)  # 默认全关:没有账本,一个字节都不动
    assert env.scene["robot"].call_count == 0
    assert not hasattr(env, EV.FORCE_PUSH_STATE_ATTR)


@pytest.mark.parametrize(
    "bad_kwargs, match",
    [
        ({"force_n": 0.0}, "force_n"),
        ({"force_n": -60.0}, "force_n"),
        ({"force_n": float("nan")}, "force_n"),
        ({"force_n": True}, "force_n"),
        ({"duration_steps": 0}, "duration_steps"),
        ({"duration_steps": True}, "duration_steps"),
        ({"duration_steps": 1.5}, "duration_steps"),
        ({"body_name": ""}, "body_name"),
        ({"body_name": 123}, "body_name"),
    ],
)
def test_event_function_illegal_params_fail_closed(bad_kwargs, match):
    env = _FakeEnv()
    kwargs = {"force_n": 60.0, "duration_steps": 15, "body_name": "pelvis_link"}
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        EV.push_by_applying_wrench(env, torch.tensor([0]), **kwargs)


def test_missing_body_fails_closed():
    env = _FakeEnv(body_names=("torso_link", "left_leg"))
    with pytest.raises(RuntimeError, match="not found"):
        EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)


def test_duplicate_body_fails_closed():
    env = _FakeEnv(body_names=("pelvis_link", "pelvis_link"))
    with pytest.raises(RuntimeError, match="exactly one"):
        EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)


def test_competing_state_writer_refused():
    env = _FakeEnv()
    setattr(env, EV.FORCE_PUSH_STATE_ATTR, {"schema_version": 1, "body_id": 2})
    with pytest.raises(RuntimeError, match="competing"):
        EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)


def test_non_integer_step_counter_fails_closed():
    env = _FakeEnv()
    env.common_step_counter = 3.5
    with pytest.raises(RuntimeError, match="common_step_counter"):
        EV.push_by_applying_wrench(env, torch.tensor([0]), force_n=60.0, duration_steps=15)


# --------------------------------------------------------------------------------------------- #
# fakes: the minimal env-cfg surface the force-push override + contract binder touch
# --------------------------------------------------------------------------------------------- #
class _FakeEventTerm:
    """isaaclab.managers.EventTermCfg stand-in: captures the exact constructor kwargs."""

    def __init__(self, func=None, mode=None, interval_range_s=None, params=None):
        self.func = func
        self.mode = mode
        self.interval_range_s = interval_range_s
        self.params = dict(params or {})


def _fake_push_by_applying_wrench(*args, **kwargs):
    raise AssertionError("cfg-level tests must never call the event function")


def _fake_sweep_expired_force_pushes(*args, **kwargs):
    raise AssertionError("cfg-level tests must never call the sweep function")


# 合同绑定按函数 __name__ 认事件(fail-closed),假函数必须顶真名。
_fake_push_by_applying_wrench.__name__ = "push_by_applying_wrench"
_fake_sweep_expired_force_pushes.__name__ = "sweep_expired_force_pushes"


def _make_env_cfg():
    """DeployParity 形状的假 env cfg:只搭 force_push 覆盖/合同路径摸到的面(仿 push 测试)。"""
    racket_target = _NS(
        question_bank="", face_command=False,
        station_anchor_offset_xy=(0.0, 0.0),
        track_envelope_violation=False,
    )
    motion = _NS(speed_scale_range=(1.0, 1.0))
    return _NS(
        rewards=_NS(),
        commands=_NS(motion=motion, racket_target=racket_target),
        observations=_NS(policy=_NS(), critic=_NS()),
        terminations=_NS(),
        face_command_obs=False,
        station_obs=False,
        scene=_NS(env_spacing=2.5),
        sim=_NS(dt=0.005),
        decimation=4,
        events=_NS(push_robot=None, force_push=None, force_push_sweep=None),
        push=_NS(
            enable=False,
            interval_range_s=(5.0, 15.0),
            vel_xy_mps=0.0,
            ang_vel_radps=0.0,
            ang_axes="none",
        ),
        force_push=_NS(
            enable=False,
            interval_range_s=(5.0, 15.0),
            force_n=0.0,
            duration_s=0.30,
        ),
    )


def _make_runtime_env(mass_total=30.0, num_bodies=3, num_envs=2):
    """合同绑定用的假 runtime env:articulation 名义质量表(逐 body 均分)。"""
    per_body = mass_total / num_bodies
    return _NS(
        scene={
            "robot": _NS(data=_NS(default_mass=torch.full((num_envs, num_bodies), per_body)))
        }
    )


def _stub_modules(monkeypatch):
    """force_push 覆盖分支里的惰性 import:isaaclab.managers.EventTermCfg、mdp 的两个事件函数、
    whole_body_tracking.utils.training_contract(真模块,按文件路径加载的 TC)。"""
    fake_managers = types.ModuleType("isaaclab.managers")
    fake_managers.EventTermCfg = _FakeEventTerm
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.managers = fake_managers

    fake_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.push_by_applying_wrench = _fake_push_by_applying_wrench
    fake_mdp.sweep_expired_force_pushes = _fake_sweep_expired_force_pushes
    fake_tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    fake_tracking.mdp = fake_mdp
    fake_tasks = types.ModuleType("whole_body_tracking.tasks")
    fake_tasks.tracking = fake_tracking
    fake_utils = types.ModuleType("whole_body_tracking.utils")
    fake_utils.training_contract = TC
    fake_root = types.ModuleType("whole_body_tracking")
    fake_root.tasks = fake_tasks
    fake_root.utils = fake_utils
    for name, module in (
        ("isaaclab", fake_isaaclab),
        ("isaaclab.managers", fake_managers),
        ("whole_body_tracking", fake_root),
        ("whole_body_tracking.tasks", fake_tasks),
        ("whole_body_tracking.tasks.tracking", fake_tracking),
        ("whole_body_tracking.tasks.tracking.mdp", fake_mdp),
        ("whole_body_tracking.utils", fake_utils),
        ("whole_body_tracking.utils.training_contract", TC),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return fake_mdp


def _apply(task, env_cfg=None):
    # 2026-07-25 默认翻转后本套件仍测 legacy 翻译行为:钉 v1 + 滤 v1 记账行,原断言原样成立。
    env_cfg = env_cfg if env_cfg is not None else _make_env_cfg()
    applied = _apply_legacy_v1(env_cfg, task)
    return env_cfg, applied


_FULL_RECIPE = {
    "enable": True,
    "interval_range_s": [5.0, 15.0],
    "force_n": 60.0,
    "duration_s": 0.3,
}


def _recipe(**overrides):
    recipe = dict(_FULL_RECIPE)
    recipe.update(overrides)
    return recipe


# --------------------------------------------------------------------------------------------- #
# default path: running matrix cells have NO task.force_push key — byte-for-byte no-op
# --------------------------------------------------------------------------------------------- #
def test_absent_task_force_push_is_total_noop(monkeypatch):
    _stub_modules(monkeypatch)
    for task in ({}, {"racket": {}}):
        env_cfg, applied = _apply(task)
        assert env_cfg.events.force_push is None
        assert env_cfg.events.force_push_sweep is None
        assert env_cfg.force_push.enable is False
        assert not any("force_push" in marker for marker in applied)
        assert train_mod._force_push_event_contract(env_cfg, _make_runtime_env()) is None


def test_force_push_whitelist_exact():
    assert set(train_mod._FORCE_PUSH_KEYS) == {
        "enable", "interval_range_s", "force_n", "duration_s",
    }


def test_unknown_force_push_key_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="task.force_push"):
        _apply({"force_push": {"enable": True, "force": 60.0}})


def test_non_mapping_force_push_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="mapping"):
        _apply({"force_push": "yes"})


def test_enable_missing_raises():
    with pytest.raises(train_mod._OverrideError, match="enable=true\\|false"):
        _apply({"force_push": {"force_n": 60.0}})


# --------------------------------------------------------------------------------------------- #
# enable=false: clean marker; dormant loaded fields refused (关着的开关不许挂上膛参数)
# --------------------------------------------------------------------------------------------- #
def test_disabled_clean_keeps_none_and_logs_marker():
    env_cfg, applied = _apply({"force_push": {"enable": False}})
    assert env_cfg.events.force_push is None
    assert env_cfg.events.force_push_sweep is None
    assert env_cfg.force_push.enable is False
    assert "force_push.enable=False (historical no-force-push path)" in applied


@pytest.mark.parametrize(
    "dormant",
    [
        {"force_n": 60.0},
        {"duration_s": 0.3},
        {"interval_range_s": [5.0, 15.0]},
    ],
)
def test_disabled_with_dormant_fields_raises(dormant):
    with pytest.raises(train_mod._OverrideError, match="dormant"):
        _apply({"force_push": {"enable": False, **dormant}})


# --------------------------------------------------------------------------------------------- #
# enable=true: the force + sweeper EventTerm pair, byte-exact
# --------------------------------------------------------------------------------------------- #
def test_enabled_builds_exact_term_pair(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, applied = _apply({"force_push": _recipe()})
    term = env_cfg.events.force_push
    assert isinstance(term, _FakeEventTerm)
    assert term.func.__name__ == "push_by_applying_wrench"
    assert term.mode == "interval"
    assert term.interval_range_s == (5.0, 15.0)
    assert term.params == {
        "force_n": 60.0,
        "duration_steps": 15,  # 0.30 s @ 50 Hz 控制频率
        "body_name": "pelvis_link",
    }
    sweep = env_cfg.events.force_push_sweep
    assert isinstance(sweep, _FakeEventTerm)
    assert sweep.func.__name__ == "sweep_expired_force_pushes"
    assert sweep.mode == "interval"
    assert sweep.interval_range_s == (_CONTROL_DT, _CONTROL_DT)  # 每个控制步清扫一次
    assert sweep.params == {}
    # 描述性旗标组同步(cfg 字段保持诚实)+ 人话发射日志行
    assert env_cfg.force_push.enable is True
    assert env_cfg.force_push.force_n == 60.0
    assert env_cfg.force_push.duration_s == 0.3
    assert any("F-axis force push" in marker for marker in applied)


def test_enabled_incomplete_recipe_raises(monkeypatch):
    _stub_modules(monkeypatch)
    for dropped in ("interval_range_s", "force_n", "duration_s"):
        recipe = _recipe()
        recipe.pop(dropped)
        with pytest.raises(train_mod._OverrideError, match="missing"):
            _apply({"force_push": recipe})


def test_enabled_without_events_attr_fails_loud(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg = _make_env_cfg()
    del env_cfg.events
    with pytest.raises(train_mod._OverrideError, match="events.force_push"):
        _apply({"force_push": _recipe()}, env_cfg)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"force_n": -60.0}, "> 0"),                              # 负力
        ({"force_n": 0.0}, "> 0"),                                # 零力还开着
        ({"duration_s": 0.0}, "> 0"),                             # 零时长
        ({"duration_s": 0.31}, "whole number of control"),        # 不是整数个控制步
        ({"interval_range_s": [15.0, 5.0]}, "lo <= hi"),          # 区间倒置
        ({"interval_range_s": [0.0, 5.0]}, "> 0"),                # 非正下界
        ({"interval_range_s": [5.0]}, "pair"),                    # 长度不对
        ({"interval_range_s": 5.0}, "pair"),                      # 不是序列
        ({"force_n": "60"}, "exact finite float"),                # 字符串冒充浮点
        ({"force_n": 60}, "exact finite float"),                  # int 冒充浮点(仓库纪律)
        ({"duration_s": 1}, "exact finite float"),                # int 冒充浮点
        ({"enable": "maybe"}, "explicit boolean"),                # 含糊布尔
    ],
)
def test_enabled_illegal_params_fail_loud(monkeypatch, bad, match):
    _stub_modules(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match=match):
        _apply({"force_push": _recipe(**bad)})


# --------------------------------------------------------------------------------------------- #
# hard contract binding (train.py _force_push_event_contract)
# --------------------------------------------------------------------------------------------- #
def test_contract_block_binds_runtime_mass_and_delta_v(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, _ = _apply({"force_push": _recipe()})
    block = train_mod._force_push_event_contract(env_cfg, _make_runtime_env(mass_total=30.0))
    assert set(block) == {
        "schema_version", "enabled", "func", "mode", "interval_range_s",
        "force_n", "duration_s", "duration_steps", "control_dt_s",
        "body_name", "application_point", "robot_mass_kg", "delta_v_equiv_mps",
    }
    assert block["schema_version"] == 1 and block["enabled"] is True
    assert block["func"] == "push_by_applying_wrench" and block["mode"] == "interval"
    assert block["interval_range_s"] == [5.0, 15.0]
    assert block["force_n"] == 60.0 and block["duration_s"] == 0.3
    assert block["duration_steps"] == 15 and block["control_dt_s"] == _CONTROL_DT
    assert block["body_name"] == "pelvis_link"
    # Yikang V9 教训:施在 link 原点就诚实写 origin,不许标 COM
    assert block["application_point"] == "pelvis_link_origin"
    assert block["robot_mass_kg"] == 30.0
    assert block["delta_v_equiv_mps"] == 60.0 * 0.3 / 30.0  # Δv_equiv = F·Δt/m ≈ 0.6 m/s

def test_contract_refuses_half_wired_states(monkeypatch):
    _stub_modules(monkeypatch)
    runtime = _make_runtime_env()
    # 旗标开了但事件没挂上
    env_cfg = _make_env_cfg()
    env_cfg.force_push.enable = True
    with pytest.raises(RuntimeError, match="half-wired"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 事件挂着但旗标是关的
    env_cfg, _ = _apply({"force_push": _recipe()})
    env_cfg.force_push.enable = False
    with pytest.raises(RuntimeError, match="half-wired"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 有施力没清扫 = 力永远清不掉,必须拒绝
    env_cfg, _ = _apply({"force_push": _recipe()})
    env_cfg.events.force_push_sweep = None
    with pytest.raises(RuntimeError, match="never clear"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 只有清扫没有施力也是半接线
    env_cfg = _make_env_cfg()
    env_cfg.events.force_push_sweep = _FakeEventTerm(
        func=_fake_sweep_expired_force_pushes, mode="interval",
        interval_range_s=(_CONTROL_DT, _CONTROL_DT), params={},
    )
    with pytest.raises(RuntimeError, match="half-wired"):
        train_mod._force_push_event_contract(env_cfg, runtime)


def test_contract_refuses_alien_term_shapes(monkeypatch):
    _stub_modules(monkeypatch)
    runtime = _make_runtime_env()

    def _enabled_cfg():
        env_cfg, _ = _apply({"force_push": _recipe()})
        return env_cfg

    # 别的函数名
    env_cfg = _enabled_cfg()
    env_cfg.events.force_push.func = "push_by_setting_velocity"
    with pytest.raises(RuntimeError, match="push_by_applying_wrench"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 非 interval 模式
    env_cfg = _enabled_cfg()
    env_cfg.events.force_push.mode = "reset"
    with pytest.raises(RuntimeError, match="interval"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # params 键面走样
    env_cfg = _enabled_cfg()
    env_cfg.events.force_push.params["extra"] = 1.0
    with pytest.raises(RuntimeError, match="exactly"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 清扫间隔不是每个控制步
    env_cfg = _enabled_cfg()
    env_cfg.events.force_push_sweep.interval_range_s = (0.04, 0.04)
    with pytest.raises(RuntimeError, match="every control step"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # duration_steps 与旗标 duration_s 漂移
    env_cfg = _enabled_cfg()
    env_cfg.events.force_push.params["duration_steps"] = 14
    with pytest.raises(RuntimeError, match="disagrees"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 旗标 force_n 与事件漂移
    env_cfg = _enabled_cfg()
    env_cfg.force_push.force_n = 61.0
    with pytest.raises(RuntimeError, match="disagrees"):
        train_mod._force_push_event_contract(env_cfg, runtime)
    # 施力 body 走样
    env_cfg = _enabled_cfg()
    env_cfg.events.force_push.params["body_name"] = "torso_link"
    with pytest.raises(RuntimeError, match="disagrees"):
        train_mod._force_push_event_contract(env_cfg, runtime)


def test_contract_requires_runtime_mass_table(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, _ = _apply({"force_push": _recipe()})
    no_mass = _NS(scene={"robot": _NS(data=_NS(default_mass=None))})
    with pytest.raises(RuntimeError, match="default_mass"):
        train_mod._force_push_event_contract(env_cfg, no_mass)


def test_hard_contract_assembly_wires_the_block():
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert "_force_push_event_contract(env_cfg, env)" in source
    assert '"force_push_event"' in source


# --------------------------------------------------------------------------------------------- #
# training_contract pure helpers (single assembly source)
# --------------------------------------------------------------------------------------------- #
def test_tc_disabled_returns_none_and_rejects_loaded_force():
    assert (
        TC.force_push_event_block(
            enable=False, interval_range_s=None, force_n=None,
            duration_s=None, control_dt_s=None,
        )
        is None
    )
    assert (
        TC.force_push_event_block(
            enable=False, interval_range_s=(5.0, 15.0), force_n=0.0,
            duration_s=0.30, control_dt_s=0.02,
        )
        is None
    )
    with pytest.raises(ValueError, match="nonzero"):
        TC.force_push_event_block(
            enable=False, interval_range_s=None, force_n=60.0,
            duration_s=None, control_dt_s=None,
        )
    with pytest.raises(ValueError, match="explicit boolean"):
        TC.force_push_event_block(
            enable="true", interval_range_s=None, force_n=None,
            duration_s=None, control_dt_s=None,
        )


def test_tc_assembly_values_and_step_conversion():
    block = TC.force_push_event_block(
        enable=True, interval_range_s=(5.0, 15.0), force_n=60.0,
        duration_s=0.3, control_dt_s=_CONTROL_DT,
    )
    assert set(block) == set(TC._FORCE_PUSH_ASSEMBLY_KEYS)
    assert block["duration_steps"] == 15  # 0.30 s = 15 个控制步 @ 50 Hz
    assert block["body_name"] == "pelvis_link"
    assert block["application_point"] == "pelvis_link_origin"
    with pytest.raises(ValueError, match="whole number of control"):
        TC.force_push_event_block(
            enable=True, interval_range_s=(5.0, 15.0), force_n=60.0,
            duration_s=0.31, control_dt_s=_CONTROL_DT,
        )


def test_tc_runtime_mass_binding():
    block = TC.force_push_event_block(
        enable=True, interval_range_s=(5.0, 15.0), force_n=60.0,
        duration_s=0.3, control_dt_s=_CONTROL_DT,
    )
    full = TC.bind_force_push_runtime_mass(block, robot_mass_kg=30.0)
    assert set(full) == set(TC._FORCE_PUSH_EVENT_KEYS)
    assert full["robot_mass_kg"] == 30.0
    assert full["delta_v_equiv_mps"] == 60.0 * 0.3 / 30.0
    with pytest.raises(ValueError, match="> 0"):
        TC.bind_force_push_runtime_mass(block, robot_mass_kg=0.0)
    with pytest.raises(ValueError, match="canonical assembly block"):
        TC.bind_force_push_runtime_mass({"schema_version": 1}, robot_mass_kg=30.0)


# --------------------------------------------------------------------------------------------- #
# schema-3 validator + JSON roundtrip
# --------------------------------------------------------------------------------------------- #
def _valid_block(**overrides):
    block = TC.bind_force_push_runtime_mass(
        TC.force_push_event_block(
            enable=True, interval_range_s=(5.0, 15.0), force_n=60.0,
            duration_s=0.3, control_dt_s=_CONTROL_DT,
        ),
        robot_mass_kg=30.0,
    )
    block.update(overrides)
    return block


def test_validator_absent_block_passes_and_null_block_raises():
    TC._validate_force_push_event_contract({})  # 没这个块 = 关(历史合同全部如此)
    with pytest.raises(ValueError, match="omitted"):
        TC._validate_force_push_event_contract({"force_push_event": None})


def test_validator_json_roundtrip_passes_bit_exact():
    block = _valid_block()
    loaded = json.loads(json.dumps({"force_push_event": block}))
    TC._validate_force_push_event_contract(loaded)
    assert loaded["force_push_event"] == block  # 一层内容 SHA 依赖字节稳定的 JSON 往返


@pytest.mark.parametrize(
    "tamper, match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"enabled": False}, "omitting the block"),
        ({"func": "push_by_setting_velocity"}, "push_by_applying_wrench"),
        ({"mode": "reset"}, "interval"),
        ({"body_name": "torso_link"}, "body_name"),
        ({"application_point": "pelvis_com"}, "Yikang V9"),      # link 原点不许标 COM
        ({"duration_steps": 14}, "internally inconsistent"),     # 步数改了但 duration 没改
        ({"control_dt_s": 0.01}, "internally inconsistent"),     # dt 改了但步数没改
        ({"force_n": 61.0}, "internally inconsistent"),          # 力改了但 Δv 没改
        ({"delta_v_equiv_mps": 0.7}, "internally inconsistent"), # Δv 必须 = F·Δt/m 重算
        ({"robot_mass_kg": -30.0}, "is invalid"),                # 非法质量本体
        ({"interval_range_s": [15.0, 5.0]}, "is invalid"),       # 倒置区间(非法参数本体)
    ],
)
def test_validator_rejects_tampered_blocks(tamper, match):
    with pytest.raises(ValueError, match=match):
        TC._validate_force_push_event_contract({"force_push_event": _valid_block(**tamper)})


def test_validator_rejects_wrong_key_sets():
    block = _valid_block()
    extra = dict(block, surprise=1.0)
    with pytest.raises(ValueError, match="force_push_event"):
        TC._validate_force_push_event_contract({"force_push_event": extra})
    short = dict(block)
    short.pop("robot_mass_kg")
    with pytest.raises(ValueError, match="force_push_event"):
        TC._validate_force_push_event_contract({"force_push_event": short})


def test_validator_wired_into_schema3_structure():
    source = inspect.getsource(TC.validate_schema3_contract_structure)
    assert "_validate_force_push_event_contract(contract)" in source


# --------------------------------------------------------------------------------------------- #
# hope_env_cfg declaration (source-text regression, the push-robot test pattern)
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_declares_default_off_force_push_group():
    source = _HOPE_ENV_CFG_PATH.read_text()
    # 事件对默认双 None(逐字节 no-op)
    assert "force_push = None" in source
    assert "force_push_sweep = None" in source
    # 默认关的旗标组,四个字段逐字冻结
    assert "class HOPEForcePushCfg" in source
    assert "enable: bool = False" in source
    assert "interval_range_s: tuple[float, float] = (5.0, 15.0)" in source
    assert "force_n: float = 0.0" in source
    assert "duration_s: float = 0.30" in source
    # env cfg 挂旗标组 + __post_init__ 消费(cfg 直启路径)
    assert "force_push: HOPEForcePushCfg = HOPEForcePushCfg()" in source
    assert "apply_force_push_event(self)" in source
    # builder 与 train.py 同款事件对:同函数、interval 模式、共用同一纯装配来源
    builder = source[source.index("def apply_force_push_event") :]
    builder = builder[: builder.index("\n##")]
    assert "force_push_event_block" in builder
    assert "func=mdp.push_by_applying_wrench" in builder
    assert "func=mdp.sweep_expired_force_pushes" in builder
    assert 'mode="interval"' in builder


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
