"""Wave-P random base push (task.push.*) — override translation + contract tests, NO Isaac.

人话:这波要训"随机推机器人一把"的抗扰平衡(PACE ±0.2 m/s 每 5-15 s;BeyondMimic ±0.5 m/s +
rpy 角速度每 1-3 s)。默认必须全关:所有 24 个在跑矩阵格没有 task.push 键,events.push_robot
保持 None、合同不写 push 块、字节逐位不变。开了以后由一套单一来源的校验/装配
(training_contract.push_robot_event_block)生成 interval push 事件与 schema-3 push_robot_event
合同块。

Covers (train.py is imported directly — hydra/omegaconf only at import time; the env cfg is a
plain-namespace fake, the level _apply_task_overrides operates on; training_contract.py is loaded
by file path, the standalone pattern of test_training_contract_schema3.py):

* absent task.push == byte-for-byte no-op (push_robot None, no applied marker, no contract block);
* enable=false clean path + dormant-field rejection (enable=false 但带非零幅度必须炸);
* enable=true builds the EventTerm with the exact interval / symmetric ±v box for all three
  ang_axes recipes (none / yaw / rpy) and the fast 1-3 s interval;
* fail-loud illegal params (unknown key, missing enable, incomplete recipe, bad interval,
  negative/typed-wrong amplitudes, contradictory axes combos, zero-everything);
* hard-contract block binding (_push_robot_event_contract): exact keys, half-wired refusal,
  alien term-shape refusal, absent-when-disabled;
* schema-3 validator (_validate_push_robot_event_contract): JSON roundtrip passes, tampered /
  incomplete / disabled-spelling blocks raise; wired into validate_schema3_contract_structure.

Run:  <python> -m pytest hope_training/whole_body_tracking/tests/test_push_robot_events.py -q
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

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
_TC_SPEC = importlib.util.spec_from_file_location("push_tc_under_test", _TC_PATH)
TC = importlib.util.module_from_spec(_TC_SPEC)
_TC_SPEC.loader.exec_module(TC)

_HOPE_ENV_CFG_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)


# --------------------------------------------------------------------------------------------- #
# fakes: the minimal env-cfg surface the push override + contract binder touch
# --------------------------------------------------------------------------------------------- #
class _NS(types.SimpleNamespace):
    pass


class _FakeEventTerm:
    """isaaclab.managers.EventTermCfg stand-in: captures the exact constructor kwargs."""

    def __init__(self, func=None, mode=None, interval_range_s=None, params=None):
        self.func = func
        self.mode = mode
        self.interval_range_s = interval_range_s
        self.params = dict(params or {})


def _fake_push_by_setting_velocity(*args, **kwargs):
    raise AssertionError("cfg-level tests must never call the event function")


# 合同绑定按函数 __name__ 认事件(fail-closed),假函数必须顶真名。
_fake_push_by_setting_velocity.__name__ = "push_by_setting_velocity"


def _make_env_cfg():
    """DeployParity 形状的假 env cfg:只搭 push 覆盖/合同路径摸到的面(仿 station 测试)。"""
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
        actions=_NS(
            joint_pos=_NS(
                clamp=True,
                control_step_action_delay_min=0,
                control_step_action_delay_max=0,
            )
        ),
        events=_NS(push_robot=None),
        push=_NS(
            enable=False,
            recipe="legacy_v1",
            interval_range_s=(5.0, 15.0),
            vel_xy_mps=0.0,
            ang_vel_radps=0.0,
            ang_axes="none",
            velocity_range=None,
        ),
    )


def _stub_modules(monkeypatch):
    """push 覆盖分支里的惰性 import:isaaclab.managers.EventTermCfg、mdp.push_by_setting_velocity、
    whole_body_tracking.utils.training_contract(真模块,按文件路径加载的 TC)。"""
    fake_managers = types.ModuleType("isaaclab.managers")
    fake_managers.EventTermCfg = _FakeEventTerm
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.managers = fake_managers

    fake_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.push_by_setting_velocity = _fake_push_by_setting_velocity
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
    "vel_xy_mps": 0.35,
    "ang_vel_radps": 0.0,
    "ang_axes": "none",
}


def _recipe(**overrides):
    recipe = dict(_FULL_RECIPE)
    recipe.update(overrides)
    return recipe


_VENDOR_AXIS_BOX = {
    "x": [-0.25, 0.25],
    "y": [-0.25, 0.25],
    "z": [-0.1, 0.1],
    "roll": [-0.26, 0.26],
    "pitch": [-0.26, 0.26],
    "yaw": [-0.39, 0.39],
}


def _axis_box_recipe(**overrides):
    recipe = {
        "enable": True,
        "recipe": "axis_box_6d_v2",
        "interval_range_s": [5.0, 15.0],
        "velocity_range": dict(_VENDOR_AXIS_BOX),
    }
    recipe.update(overrides)
    return recipe


# --------------------------------------------------------------------------------------------- #
# default path: the 24 running matrix cells have NO task.push key — byte-for-byte no-op
# --------------------------------------------------------------------------------------------- #
def test_absent_task_push_is_total_noop(monkeypatch):
    _stub_modules(monkeypatch)
    for task in ({}, {"racket": {}}):
        env_cfg, applied = _apply(task)
        assert env_cfg.events.push_robot is None
        assert env_cfg.push.enable is False
        assert not any("push" in marker for marker in applied)
        assert train_mod._push_robot_event_contract(env_cfg) is None


def test_push_whitelist_exact():
    assert set(train_mod._PUSH_KEYS) == {
        "enable", "recipe", "interval_range_s", "vel_xy_mps", "ang_vel_radps",
        "ang_axes", "velocity_range",
    }


def test_unknown_push_key_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="task.push"):
        _apply({"push": {"enable": True, "vel_xy": 0.2}})


def test_non_mapping_push_fails_loud():
    with pytest.raises(train_mod._OverrideError, match="mapping"):
        _apply({"push": "yes"})


def test_enable_missing_raises():
    with pytest.raises(train_mod._OverrideError, match="enable=true\\|false"):
        _apply({"push": {"vel_xy_mps": 0.35}})


# --------------------------------------------------------------------------------------------- #
# enable=false: clean marker; dormant loaded fields refused (关着的开关不许挂上膛参数)
# --------------------------------------------------------------------------------------------- #
def test_disabled_clean_keeps_none_and_logs_marker():
    env_cfg, applied = _apply({"push": {"enable": False}})
    assert env_cfg.events.push_robot is None
    assert env_cfg.push.enable is False
    assert "push.enable=False (historical no-push path)" in applied


@pytest.mark.parametrize(
    "dormant",
    [
        {"vel_xy_mps": 0.35},
        {"ang_vel_radps": 0.5},
        {"ang_axes": "yaw"},
        {"interval_range_s": [5.0, 15.0]},
    ],
)
def test_disabled_with_dormant_fields_raises(dormant):
    with pytest.raises(train_mod._OverrideError, match="dormant"):
        _apply({"push": {"enable": False, **dormant}})


# --------------------------------------------------------------------------------------------- #
# enable=true: EventTerm shape/amplitude/interval for the Wave-P arms
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("vel", [0.2, 0.35, 0.5])  # p02 / p035 / p05
def test_enabled_linear_push_builds_exact_term(monkeypatch, vel):
    _stub_modules(monkeypatch)
    env_cfg, applied = _apply({"push": _recipe(vel_xy_mps=vel)})
    term = env_cfg.events.push_robot
    assert isinstance(term, _FakeEventTerm)
    assert term.func.__name__ == "push_by_setting_velocity"
    assert term.mode == "interval"
    assert term.interval_range_s == (5.0, 15.0)
    assert term.params == {
        "velocity_range": {"x": (-vel, vel), "y": (-vel, vel)}
    }  # 纯线性推:没有任何角速度轴
    # 描述性旗标组同步(cfg 字段保持诚实)+ 人话发射日志行
    assert env_cfg.push.enable is True
    assert env_cfg.push.vel_xy_mps == vel
    assert env_cfg.push.ang_axes == "none"
    assert any("Wave-P random base push" in marker for marker in applied)


def test_enabled_yaw_axes(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, _ = _apply(
        {"push": _recipe(ang_vel_radps=0.5, ang_axes="yaw")}
    )
    velocity_range = env_cfg.events.push_robot.params["velocity_range"]
    assert velocity_range == {
        "x": (-0.35, 0.35), "y": (-0.35, 0.35), "yaw": (-0.5, 0.5),
    }
    assert env_cfg.push.ang_axes == "yaw"


def test_enabled_rpy_axes(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, _ = _apply(
        {"push": _recipe(ang_vel_radps=0.5, ang_axes="rpy")}
    )
    velocity_range = env_cfg.events.push_robot.params["velocity_range"]
    assert velocity_range == {
        "x": (-0.35, 0.35), "y": (-0.35, 0.35),
        "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
    }


def test_enabled_fast_interval(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, _ = _apply({"push": _recipe(interval_range_s=[1.0, 3.0])})
    assert env_cfg.events.push_robot.interval_range_s == (1.0, 3.0)
    assert env_cfg.push.interval_range_s == (1.0, 3.0)


def test_axis_box_v2_builds_exact_vendor_six_axis_term_and_contract(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, applied = _apply({"push": _axis_box_recipe()})
    term = env_cfg.events.push_robot
    assert term.func.__name__ == "push_by_setting_velocity"
    assert term.mode == "interval"
    assert term.interval_range_s == (5.0, 15.0)
    expected_range = {
        axis: (float(pair[0]), float(pair[1]))
        for axis, pair in _VENDOR_AXIS_BOX.items()
    }
    assert term.params == {"velocity_range": expected_range}
    assert env_cfg.push.recipe == "axis_box_6d_v2"
    assert env_cfg.push.velocity_range == expected_range
    assert any("recipe=axis_box_6d_v2" in marker for marker in applied)

    block = train_mod._push_robot_event_contract(env_cfg)
    assert block == {
        "schema_version": 2,
        "enabled": True,
        "semantics": "symmetric_6d_velocity_delta",
        "func": "push_by_setting_velocity",
        "mode": "interval",
        "interval_range_s": [5.0, 15.0],
        "velocity_range": _VENDOR_AXIS_BOX,
    }


@pytest.mark.parametrize(
    "recipe, match",
    [
        (
            _axis_box_recipe(vel_xy_mps=0.25),
            "v1/v2 push spellings may not be mixed",
        ),
        (
            {**_recipe(), "recipe": "axis_box_6d_v2", "velocity_range": _VENDOR_AXIS_BOX},
            "v1/v2 push spellings may not be mixed",
        ),
        (
            {**_recipe(), "recipe": "legacy_v1", "velocity_range": _VENDOR_AXIS_BOX},
            "v1/v2 push spellings may not be mixed",
        ),
    ],
)
def test_task_override_rejects_mixed_v1_v2_spellings(monkeypatch, recipe, match):
    _stub_modules(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match=match):
        _apply({"push": recipe})


def test_task_actions_compose_vendor_control_step_delay():
    env_cfg, applied = _apply(
        {
            "actions": {
                "control_step_action_delay_min": 0,
                "control_step_action_delay_max": 2,
            }
        }
    )
    assert env_cfg.actions.joint_pos.control_step_action_delay_min == 0
    assert env_cfg.actions.joint_pos.control_step_action_delay_max == 2
    assert any("[0,2] policy/control steps" in marker for marker in applied)


@pytest.mark.parametrize(
    "actions, match",
    [
        ({"control_step_action_delay_min": 0}, "requires both"),
        (
            {
                "control_step_action_delay_min": 2,
                "control_step_action_delay_max": 1,
            },
            "0 <= min <= max",
        ),
        (
            {
                "control_step_action_delay_min": False,
                "control_step_action_delay_max": 2,
            },
            "exact integer",
        ),
    ],
)
def test_task_actions_reject_invalid_control_step_delay(actions, match):
    with pytest.raises(train_mod._OverrideError, match=match):
        _apply({"actions": actions})


def test_enabled_incomplete_recipe_raises(monkeypatch):
    _stub_modules(monkeypatch)
    for dropped in ("interval_range_s", "vel_xy_mps", "ang_vel_radps", "ang_axes"):
        recipe = _recipe()
        recipe.pop(dropped)
        with pytest.raises(train_mod._OverrideError, match="missing"):
            _apply({"push": recipe})


def test_enabled_without_events_attr_fails_loud(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg = _make_env_cfg()
    del env_cfg.events
    with pytest.raises(train_mod._OverrideError, match="events.push_robot"):
        _apply({"push": _recipe()}, env_cfg)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"vel_xy_mps": -0.2}, ">= 0"),                             # 负幅度
        ({"ang_vel_radps": -0.5}, ">= 0"),                          # 负角速度
        ({"ang_axes": "pitch"}, "ang_axes"),                        # 不认识的轴组合
        ({"ang_vel_radps": 0.5}, "ang_axes='none'"),                # none + 非零角速度矛盾
        ({"ang_vel_radps": 0.0, "ang_axes": "yaw"}, "requires ang_vel_radps > 0"),
        ({"vel_xy_mps": 0.0}, "must push something"),               # 全零还开着
        ({"interval_range_s": [15.0, 5.0]}, "lo <= hi"),            # 区间倒置
        ({"interval_range_s": [0.0, 5.0]}, "> 0"),                  # 非正下界
        ({"interval_range_s": [5.0]}, "pair"),                      # 长度不对
        ({"interval_range_s": 5.0}, "pair"),                        # 不是序列
        ({"vel_xy_mps": "0.35"}, "exact finite float"),             # 字符串冒充浮点
        ({"vel_xy_mps": 1}, "exact finite float"),                  # int 冒充浮点(仓库纪律)
        ({"enable": "maybe"}, "explicit boolean"),                  # 含糊布尔
    ],
)
def test_enabled_illegal_params_fail_loud(monkeypatch, bad, match):
    _stub_modules(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match=match):
        _apply({"push": _recipe(**bad)})


# --------------------------------------------------------------------------------------------- #
# hard contract binding (train.py _push_robot_event_contract)
# --------------------------------------------------------------------------------------------- #
def test_contract_block_binds_the_enabled_term(monkeypatch):
    _stub_modules(monkeypatch)
    env_cfg, _ = _apply({"push": _recipe(ang_vel_radps=0.5, ang_axes="yaw")})
    block = train_mod._push_robot_event_contract(env_cfg)
    assert set(block) == {
        "schema_version", "enabled", "func", "mode", "interval_range_s",
        "vel_xy_mps", "ang_vel_radps", "ang_axes", "velocity_range",
    }
    assert block["schema_version"] == 1 and block["enabled"] is True
    assert block["func"] == "push_by_setting_velocity" and block["mode"] == "interval"
    assert block["interval_range_s"] == [5.0, 15.0]
    assert block["vel_xy_mps"] == 0.35 and block["ang_vel_radps"] == 0.5
    assert block["ang_axes"] == "yaw"
    assert block["velocity_range"] == {
        "x": [-0.35, 0.35], "y": [-0.35, 0.35], "yaw": [-0.5, 0.5],
    }


def test_contract_refuses_half_wired_states(monkeypatch):
    _stub_modules(monkeypatch)
    # 旗标开了但事件没挂上
    env_cfg = _make_env_cfg()
    env_cfg.push.enable = True
    with pytest.raises(RuntimeError, match="half-wired"):
        train_mod._push_robot_event_contract(env_cfg)
    # 事件挂着但旗标是关的
    env_cfg, _ = _apply({"push": _recipe()})
    env_cfg.push.enable = False
    with pytest.raises(RuntimeError, match="half-wired"):
        train_mod._push_robot_event_contract(env_cfg)


def test_contract_refuses_alien_term_shapes(monkeypatch):
    _stub_modules(monkeypatch)

    def _term(velocity_range, mode="interval", func=_fake_push_by_setting_velocity):
        return _FakeEventTerm(
            func=func, mode=mode, interval_range_s=(5.0, 15.0),
            params={"velocity_range": velocity_range},
        )

    base = {"x": (-0.35, 0.35), "y": (-0.35, 0.35)}
    cases = [
        ({**base, "z": (-0.2, 0.2)}, "do not match"),                 # 基类 BeyondMimic 带 z 的箱子
        ({**base, "roll": (-0.5, 0.5)}, "do not match"),              # 只有 roll 不是合法组合
        ({"x": (-0.35, 0.35), "y": (-0.2, 0.2)}, "must match"),       # x/y 幅度不等
        ({"x": (0.0, 0.35), "y": (-0.35, 0.35)}, "symmetric"),        # 不对称箱子
        ({"x": (-0.35, 0.35)}, "push x and y"),                       # 缺 y
    ]
    for velocity_range, match in cases:
        env_cfg = _make_env_cfg()
        env_cfg.push.enable = True
        env_cfg.events.push_robot = _term(velocity_range)
        with pytest.raises(RuntimeError, match=match):
            train_mod._push_robot_event_contract(env_cfg)
    # 非 interval 模式 / 别的函数名也拒绝
    env_cfg = _make_env_cfg()
    env_cfg.push.enable = True
    env_cfg.events.push_robot = _term(base, mode="reset")
    with pytest.raises(RuntimeError, match="interval"):
        train_mod._push_robot_event_contract(env_cfg)
    env_cfg.events.push_robot = _term(base, func="some_other_event")
    with pytest.raises(RuntimeError, match="push_by_setting_velocity"):
        train_mod._push_robot_event_contract(env_cfg)


def test_hard_contract_assembly_wires_the_block():
    source = inspect.getsource(train_mod._build_training_hard_contract)
    assert "_push_robot_event_contract(env_cfg)" in source
    assert '"push_robot_event"' in source


# --------------------------------------------------------------------------------------------- #
# training_contract pure helpers (single assembly source)
# --------------------------------------------------------------------------------------------- #
def test_tc_disabled_returns_none_and_rejects_loaded_fields():
    assert (
        TC.push_robot_event_block(
            enable=False, interval_range_s=None, vel_xy_mps=None,
            ang_vel_radps=None, ang_axes=None,
        )
        is None
    )
    assert (
        TC.push_robot_event_block(
            enable=False, interval_range_s=None, vel_xy_mps=0.0,
            ang_vel_radps=0.0, ang_axes="none",
        )
        is None
    )
    with pytest.raises(ValueError, match="nonzero"):
        TC.push_robot_event_block(
            enable=False, interval_range_s=None, vel_xy_mps=0.35,
            ang_vel_radps=None, ang_axes=None,
        )
    with pytest.raises(ValueError, match="ang_axes"):
        TC.push_robot_event_block(
            enable=False, interval_range_s=None, vel_xy_mps=None,
            ang_vel_radps=None, ang_axes="yaw",
        )
    with pytest.raises(ValueError, match="explicit boolean"):
        TC.push_robot_event_block(
            enable="true", interval_range_s=None, vel_xy_mps=None,
            ang_vel_radps=None, ang_axes=None,
        )


def test_tc_assembly_values_per_axes_recipe():
    p05 = TC.push_robot_event_block(
        enable=True, interval_range_s=(5.0, 15.0), vel_xy_mps=0.5,
        ang_vel_radps=0.0, ang_axes="none",
    )
    assert p05["velocity_range"] == {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}
    yaw = TC.push_robot_event_block(
        enable=True, interval_range_s=(5.0, 15.0), vel_xy_mps=0.35,
        ang_vel_radps=0.5, ang_axes="yaw",
    )
    assert yaw["velocity_range"]["yaw"] == [-0.5, 0.5]
    assert "roll" not in yaw["velocity_range"] and "pitch" not in yaw["velocity_range"]
    rpy = TC.push_robot_event_block(
        enable=True, interval_range_s=(1.0, 3.0), vel_xy_mps=0.35,
        ang_vel_radps=0.5, ang_axes="rpy",
    )
    assert rpy["interval_range_s"] == [1.0, 3.0]
    for axis in ("roll", "pitch", "yaw"):
        assert rpy["velocity_range"][axis] == [-0.5, 0.5]
    assert set(rpy) == set(TC._PUSH_ROBOT_EVENT_KEYS)


def test_tc_axis_box_v2_json_roundtrip_and_validator():
    block = TC.push_robot_axis_box_event_block(
        enable=True,
        interval_range_s=(5.0, 15.0),
        velocity_range=_VENDOR_AXIS_BOX,
    )
    assert block["schema_version"] == 2
    assert block["velocity_range"] == _VENDOR_AXIS_BOX
    loaded = json.loads(json.dumps({"push_robot_event": block}))
    TC._validate_push_robot_event_contract(loaded)
    assert loaded["push_robot_event"] == block


@pytest.mark.parametrize(
    "velocity_range, match",
    [
        (
            {key: value for key, value in _VENDOR_AXIS_BOX.items() if key != "z"},
            "exactly",
        ),
        ({**_VENDOR_AXIS_BOX, "surprise": [-0.1, 0.1]}, "extra"),
        ({**_VENDOR_AXIS_BOX, "yaw": [-0.2, 0.39]}, "symmetric"),
        ({axis: [0.0, 0.0] for axis in _VENDOR_AXIS_BOX}, "all six"),
        ({**_VENDOR_AXIS_BOX, "x": [float("nan"), 0.25]}, "finite"),
    ],
)
def test_tc_axis_box_v2_rejects_invalid_boxes(velocity_range, match):
    with pytest.raises(ValueError, match=match):
        TC.push_robot_axis_box_event_block(
            enable=True,
            interval_range_s=(5.0, 15.0),
            velocity_range=velocity_range,
        )


def test_tc_axis_box_v2_rejects_combined_exclusive():
    with pytest.raises(ValueError, match="cannot use combined_exclusive"):
        TC.push_robot_axis_box_event_block(
            enable=True,
            interval_range_s=(5.0, 15.0),
            velocity_range=_VENDOR_AXIS_BOX,
            combined_exclusive=True,
        )


@pytest.mark.parametrize(
    "tamper, match",
    [
        ({"semantics": "other"}, "semantics"),
        ({"interval_range_s": [15.0, 5.0]}, "invalid"),
        ({"velocity_range": {**_VENDOR_AXIS_BOX, "yaw": [-0.2, 0.39]}}, "invalid"),
        ({"enabled": False}, "enabled"),
    ],
)
def test_tc_axis_box_v2_validator_rejects_tampering(tamper, match):
    block = TC.push_robot_axis_box_event_block(
        enable=True,
        interval_range_s=(5.0, 15.0),
        velocity_range=_VENDOR_AXIS_BOX,
    )
    block.update(tamper)
    with pytest.raises(ValueError, match=match):
        TC._validate_push_robot_event_contract({"push_robot_event": block})


# --------------------------------------------------------------------------------------------- #
# schema-3 validator + JSON roundtrip
# --------------------------------------------------------------------------------------------- #
def _valid_block(**overrides):
    block = TC.push_robot_event_block(
        enable=True, interval_range_s=(5.0, 15.0), vel_xy_mps=0.35,
        ang_vel_radps=0.5, ang_axes="rpy",
    )
    block.update(overrides)
    return block


def test_validator_absent_block_passes_and_null_block_raises():
    TC._validate_push_robot_event_contract({})  # 没这个块 = 关(历史合同全部如此)
    with pytest.raises(ValueError, match="omitted"):
        TC._validate_push_robot_event_contract({"push_robot_event": None})


def test_validator_json_roundtrip_passes_bit_exact():
    block = _valid_block()
    loaded = json.loads(json.dumps({"push_robot_event": block}))
    TC._validate_push_robot_event_contract(loaded)
    assert loaded["push_robot_event"] == block  # 一层内容 SHA 依赖字节稳定的 JSON 往返


@pytest.mark.parametrize(
    "tamper, match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"enabled": False}, "omitting the block"),
        ({"func": "apply_external_force_torque"}, "push_by_setting_velocity"),
        ({"mode": "reset"}, "interval"),
        ({"vel_xy_mps": 0.5}, "internally inconsistent"),        # 幅度改了但箱子没改
        ({"ang_axes": "yaw"}, "internally inconsistent"),        # 轴组合改了但箱子没改
        ({"interval_range_s": [15.0, 5.0]}, "is invalid"),       # 倒置区间(非法参数本体)
        (
            {"velocity_range": {"x": [-0.35, 0.35], "y": [-0.35, 0.35]}},
            "internally inconsistent",
        ),
    ],
)
def test_validator_rejects_tampered_blocks(tamper, match):
    with pytest.raises(ValueError, match=match):
        TC._validate_push_robot_event_contract({"push_robot_event": _valid_block(**tamper)})


def test_validator_rejects_wrong_key_sets():
    block = _valid_block()
    extra = dict(block, surprise=1.0)
    with pytest.raises(ValueError, match="push_robot_event"):
        TC._validate_push_robot_event_contract({"push_robot_event": extra})
    short = dict(block)
    short.pop("velocity_range")
    with pytest.raises(ValueError, match="push_robot_event"):
        TC._validate_push_robot_event_contract({"push_robot_event": short})


def test_validator_wired_into_schema3_structure():
    source = inspect.getsource(TC.validate_schema3_contract_structure)
    assert "_validate_push_robot_event_contract(contract)" in source


# --------------------------------------------------------------------------------------------- #
# hope_env_cfg declaration (source-text regression, the settle-debt test pattern)
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_declares_default_off_push_group_and_keeps_no_push_default():
    source = _HOPE_ENV_CFG_PATH.read_text()
    # HITTER 对齐的历史默认必须原地保留:HOPEEventCfg.push_robot = None
    assert "push_robot = None" in source
    # 默认关的旗标组,五个字段逐字冻结
    assert "class HOPEPushRobotCfg" in source
    assert "enable: bool = False" in source
    assert "interval_range_s: tuple[float, float] = (5.0, 15.0)" in source
    assert "vel_xy_mps: float = 0.0" in source
    assert "ang_vel_radps: float = 0.0" in source
    assert 'ang_axes: str = "none"' in source
    # env cfg 挂旗标组 + __post_init__ 消费(cfg 直启路径)
    assert "push: HOPEPushRobotCfg = HOPEPushRobotCfg()" in source
    assert "apply_push_robot_event(self)" in source
    # builder 与 train.py 同款事件:同函数、interval 模式、共用同一纯装配来源
    builder = source[source.index("def apply_push_robot_event") :]
    builder = builder[: builder.index("\n##")]
    assert "push_robot_event_block" in builder
    assert "func=mdp.push_by_setting_velocity" in builder
    assert 'mode="interval"' in builder


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
