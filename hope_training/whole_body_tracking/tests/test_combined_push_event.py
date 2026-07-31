"""合并互斥推(push.combined_exclusive)— 抽签事件 + cfg 装配 + 合同测试,NO Isaac。

人话(Franco 2026-07-25:两种随机推合并 sample 防同时叠加):legacy 配方里速度推
(push_robot)与力推(force_push)是两个独立时钟的 interval 事件,可能同一瞬间一起砸下来
叠加;合并模式把两种推合并成【一个】interval 事件,每次触发按 force_prob 逐 env 抽签二选
一,同一次触发同一个 env 绝不两种都挨。默认必须全关:combined_exclusive=False 时两个
legacy 独立事件与合同块逐字节不变。

Covers(mock 模式照 test_force_push_events.py:training_contract.py / hope_push_events.py
按文件路径加载;cfg 三个 applier 用 ast 抽出真源码编译执行,fake EventTerm/mdp 注入):

* 抽签事件 push_combined_exclusive:mock 抽签器验证互斥(force/velocity 分支 env 集合是
  同一张 mask 的正反两半,交集恒空并集恒全;全力/全速度时另一分支绝不被调);真随机多次
  触发仍互斥;力分支真调 push_by_applying_wrench(账本/水平力核对);force_prob 出界、
  velocity_range 缺轴、抽签器形状走样一律 fail-loud;
* cfg 装配 apply_combined_push_event(+ 两个 legacy applier 的让路守卫):默认关 = 逐字节
  no-op;开 = 单事件 + 清扫对,legacy 三槽保持 None;合并 + legacy 独立事件同时在场
  fail-loud(combined+legacy-both);只上膛一个分支 / 时钟拼写不一致 / combined 关着还挂
  force_prob / events cfg 没声明槽位,一律 fail-loud;
* 合同 push_robot_event_block(v1 flag 语义):legacy 块逐字节不变(不带 combined 键);
  合并块装配 + bind_force_push_runtime_mass 同冲量记账(Δv_equiv = F·Δt/m);schema-3 校验
  合并块 JSON roundtrip 通过、篡改块一律 raise、legacy 拼写继续有效。

Run:  <python> -m pytest hope_training/whole_body_tracking/tests/test_combined_push_event.py -q
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

_TC_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)
_TC_SPEC = importlib.util.spec_from_file_location("combined_push_tc_under_test", _TC_PATH)
TC = importlib.util.module_from_spec(_TC_SPEC)
_TC_SPEC.loader.exec_module(TC)

_EV_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_push_events.py"
)
_EV_SPEC = importlib.util.spec_from_file_location("combined_push_ev_under_test", _EV_PATH)
EV = importlib.util.module_from_spec(_EV_SPEC)
_EV_SPEC.loader.exec_module(EV)

_HOPE_ENV_CFG_PATH = (
    Path(HERE).resolve().parents[0]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
)
_ENV_SRC = _HOPE_ENV_CFG_PATH.read_text(encoding="utf-8")

_CONTROL_DT = 0.005 * 4  # tracking_env_cfg: sim.dt=0.005, decimation=4 -> 50 Hz control


# --------------------------------------------------------------------------------------------- #
# fakes(仿 test_force_push_events):asset/env 给事件函数,EventTerm/mdp 给 cfg applier
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
        self.call_count = 0

    def find_bodies(self, name, preserve_order=True):
        ids = [i for i, n in enumerate(self._body_names) if n == name]
        if not ids:
            raise ValueError(f"no body matches {name!r}")
        return ids, [self._body_names[i] for i in ids]

    def set_external_force_and_torque(self, forces, torques, body_ids=None, env_ids=None):
        self.call_count += 1
        self.applied_forces = forces.clone()
        self.applied_torques = torques.clone()


class _FakeEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.scene = {"robot": _FakeAsset(num_envs)}
        self.common_step_counter = 0


_VEL_RANGE = {"x": (-0.35, 0.35), "y": (-0.35, 0.35)}


def _recorder(calls, name):
    def _record(env, env_ids, **kwargs):
        calls.append((name, env_ids.clone(), dict(kwargs)))

    _record.__name__ = name
    return _record


def _run_combined(monkeypatch, *, env=None, env_ids, mask=None, force_prob=0.5):
    """跑一次合并事件:两个分支都打桩成 recorder,可选固定抽签 mask。"""
    env = env or _FakeEnv(num_envs=8)
    calls = []
    monkeypatch.setattr(EV, "push_by_applying_wrench", _recorder(calls, "force"))
    monkeypatch.setattr(
        EV, "_velocity_push_delegate", lambda: _recorder(calls, "velocity")
    )
    if mask is not None:
        fixed = torch.as_tensor(mask, dtype=torch.bool)
        monkeypatch.setattr(
            EV, "_sample_force_branch_mask", lambda num, p, device: fixed
        )
    EV.push_combined_exclusive(
        env, env_ids, velocity_range=_VEL_RANGE,
        force_n=60.0, duration_steps=15, force_prob=force_prob,
    )
    return env, calls


# --------------------------------------------------------------------------------------------- #
# sampler exclusivity: force / velocity 分支是同一张 mask 的正反两半(mock RNG)
# --------------------------------------------------------------------------------------------- #
def test_sampler_mocked_mask_splits_exclusively(monkeypatch):
    _, calls = _run_combined(
        monkeypatch, env_ids=torch.tensor([0, 2, 5]), mask=[True, False, True]
    )
    assert [name for name, _, _ in calls] == ["force", "velocity"]
    force_ids = calls[0][1].tolist()
    velocity_ids = calls[1][1].tolist()
    assert force_ids == [0, 5] and velocity_ids == [2]
    # 互斥 + 全覆盖:同一次触发,一个 env 恰好挨一种推
    assert set(force_ids) & set(velocity_ids) == set()
    assert sorted(force_ids + velocity_ids) == [0, 2, 5]
    # 力分支参数原样直达 legacy 实现
    assert calls[0][2] == {"force_n": 60.0, "duration_steps": 15, "body_name": "pelvis_link"}
    assert calls[1][2] == {"velocity_range": _VEL_RANGE}


def test_sampler_all_force_never_touches_velocity_branch(monkeypatch):
    _, calls = _run_combined(
        monkeypatch, env_ids=torch.tensor([1, 3]), mask=[True, True]
    )
    assert [name for name, _, _ in calls] == ["force"]
    assert calls[0][1].tolist() == [1, 3]


def test_sampler_all_velocity_never_touches_force_branch(monkeypatch):
    _, calls = _run_combined(
        monkeypatch, env_ids=torch.tensor([1, 3]), mask=[False, False]
    )
    assert [name for name, _, _ in calls] == ["velocity"]
    assert calls[0][1].tolist() == [1, 3]


def test_sampler_real_rng_partition_is_exclusive_and_complete(monkeypatch):
    torch.manual_seed(0)
    env = _FakeEnv(num_envs=512)
    for _ in range(10):  # 多次触发,每次都必须是严格二划分
        _, calls = _run_combined(
            monkeypatch, env=env, env_ids=torch.arange(512), force_prob=0.5
        )
        by_branch = {name: ids.tolist() for name, ids, _ in calls}
        force_ids = by_branch.get("force", [])
        velocity_ids = by_branch.get("velocity", [])
        assert set(force_ids) & set(velocity_ids) == set()
        assert sorted(force_ids + velocity_ids) == list(range(512))


def test_sampler_none_env_ids_covers_all_envs(monkeypatch):
    _, calls = _run_combined(
        monkeypatch, env_ids=None, mask=[True, False, False, True, True, False, True, False]
    )
    combined = sorted(sum((ids.tolist() for _, ids, _ in calls), []))
    assert combined == list(range(8))


def test_sampler_empty_env_ids_is_noop(monkeypatch):
    _, calls = _run_combined(monkeypatch, env_ids=torch.tensor([], dtype=torch.long))
    assert calls == []


def test_sampler_force_branch_really_books_the_ledger(monkeypatch):
    """力分支不打桩:真调 push_by_applying_wrench,账本/水平力与 legacy 逐字节同源。"""
    torch.manual_seed(1)
    env = _FakeEnv(num_envs=4)
    env.common_step_counter = 100
    velocity_calls = []
    monkeypatch.setattr(
        EV, "_velocity_push_delegate", lambda: _recorder(velocity_calls, "velocity")
    )
    monkeypatch.setattr(
        EV, "_sample_force_branch_mask",
        lambda num, p, device: torch.tensor([True, False]),
    )
    EV.push_combined_exclusive(
        env, torch.tensor([0, 2]), velocity_range=_VEL_RANGE,
        force_n=60.0, duration_steps=15, force_prob=0.5,
    )
    asset = env.scene["robot"]
    f = asset.applied_forces[0, 0]
    assert float(f[2]) == pytest.approx(0.0, abs=1e-6)  # 水平
    assert float(torch.linalg.norm(f)) == pytest.approx(60.0, rel=1e-6)  # 模长 = force_n
    assert torch.all(asset.applied_forces[2] == 0.0)  # 速度分支 env 一牛都不挨
    state = getattr(env, EV.FORCE_PUSH_STATE_ATTR)
    assert state["expiry_step"].tolist() == [115, -1, -1, -1]
    assert [ids.tolist() for _, ids, _ in velocity_calls] == [[2]]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan"), True, "0.5"])
def test_sampler_force_prob_out_of_open_interval_fails_loud(bad):
    env = _FakeEnv()
    with pytest.raises(ValueError, match="force_prob"):
        EV.push_combined_exclusive(
            env, torch.tensor([0]), velocity_range=_VEL_RANGE,
            force_n=60.0, duration_steps=15, force_prob=bad,
        )


def test_sampler_velocity_range_must_carry_xy():
    env = _FakeEnv()
    with pytest.raises(ValueError, match="velocity_range"):
        EV.push_combined_exclusive(
            env, torch.tensor([0]), velocity_range={"x": (-0.35, 0.35)},
            force_n=60.0, duration_steps=15, force_prob=0.5,
        )


def test_sampler_mask_contract_enforced(monkeypatch):
    env = _FakeEnv()
    monkeypatch.setattr(
        EV, "_sample_force_branch_mask", lambda num, p, device: torch.zeros(num)
    )  # float mask 走样
    with pytest.raises(RuntimeError, match="bool mask"):
        EV.push_combined_exclusive(
            env, torch.tensor([0, 1]), velocity_range=_VEL_RANGE,
            force_n=60.0, duration_steps=15, force_prob=0.5,
        )


# --------------------------------------------------------------------------------------------- #
# cfg appliers(ast 抽真源码编译执行;fake EventTerm/mdp/training_contract 注入)
# --------------------------------------------------------------------------------------------- #
class _FakeEventTerm:
    """isaaclab.managers.EventTermCfg stand-in: captures the exact constructor kwargs."""

    def __init__(self, func=None, mode=None, interval_range_s=None, params=None):
        self.func = func
        self.mode = mode
        self.interval_range_s = interval_range_s
        self.params = dict(params or {})


def _named_stub(name):
    def _stub(*args, **kwargs):
        raise AssertionError("cfg-level tests must never call the event function")

    _stub.__name__ = name
    return _stub


def _load_appliers(monkeypatch):
    """把 hope_env_cfg.py 里的三个 applier 抽出来按真源码编译(不 import 整个 isaaclab 链)。"""
    wanted = {
        "apply_push_robot_event",
        "apply_force_push_event",
        "apply_combined_push_event",
    }
    tree = ast.parse(_ENV_SRC)
    fns = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {fn.name for fn in fns} == wanted
    fake_mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.push_by_setting_velocity = _named_stub("push_by_setting_velocity")
    fake_mdp.push_by_applying_wrench = _named_stub("push_by_applying_wrench")
    fake_mdp.sweep_expired_force_pushes = _named_stub("sweep_expired_force_pushes")
    fake_mdp.push_combined_exclusive = _named_stub("push_combined_exclusive")
    fake_utils = types.ModuleType("whole_body_tracking.utils")
    fake_utils.training_contract = TC
    fake_root = types.ModuleType("whole_body_tracking")
    fake_root.utils = fake_utils
    for name, module in (
        ("whole_body_tracking", fake_root),
        ("whole_body_tracking.utils", fake_utils),
        ("whole_body_tracking.utils.training_contract", TC),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    namespace = {"EventTerm": _FakeEventTerm, "mdp": fake_mdp}
    exec(
        compile(ast.Module(body=fns, type_ignores=[]), str(_HOPE_ENV_CFG_PATH), "exec"),
        namespace,
    )
    return namespace


def _make_env_cfg():
    return _NS(
        sim=_NS(dt=0.005),
        decimation=4,
        events=_NS(
            push_robot=None, force_push=None, force_push_sweep=None,
            combined_push=None, combined_push_sweep=None,
        ),
        push=_NS(
            enable=False, recipe="legacy_v1", interval_range_s=(5.0, 15.0),
            vel_xy_mps=0.0, ang_vel_radps=0.0, ang_axes="none",
            velocity_range=None,
            combined_exclusive=False, force_prob=0.5,
        ),
        force_push=_NS(
            enable=False, interval_range_s=(5.0, 15.0), force_n=0.0, duration_s=0.30,
        ),
    )


def _armed_cfg():
    cfg = _make_env_cfg()
    cfg.push.enable = True
    cfg.push.combined_exclusive = True
    cfg.push.vel_xy_mps = 0.35
    cfg.force_push.enable = True
    cfg.force_push.force_n = 60.0
    cfg.force_push.duration_s = 0.3
    return cfg


def _post_init_order(ns, cfg):
    """__post_init__ 的真实调用顺序:combined 先跑,legacy applier 随后让路。"""
    ns["apply_combined_push_event"](cfg)
    ns["apply_push_robot_event"](cfg)
    ns["apply_force_push_event"](cfg)


def test_apply_default_off_is_total_noop(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _make_env_cfg()
    _post_init_order(ns, cfg)
    for slot in (
        "push_robot", "force_push", "force_push_sweep",
        "combined_push", "combined_push_sweep",
    ):
        assert getattr(cfg.events, slot) is None


def test_apply_axis_box_v2_builds_one_exact_six_axis_velocity_event(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _make_env_cfg()
    cfg.push.enable = True
    cfg.push.recipe = "axis_box_6d_v2"
    cfg.push.velocity_range = {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
        "z": (-0.1, 0.1),
        "roll": (-0.26, 0.26),
        "pitch": (-0.26, 0.26),
        "yaw": (-0.39, 0.39),
    }
    _post_init_order(ns, cfg)
    assert cfg.events.push_robot.params == {
        "velocity_range": cfg.push.velocity_range
    }
    assert cfg.events.push_robot.interval_range_s == (5.0, 15.0)
    assert cfg.events.combined_push is None


def test_apply_axis_box_v2_rejects_combined_exclusive(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _make_env_cfg()
    cfg.push.enable = True
    cfg.push.recipe = "axis_box_6d_v2"
    cfg.push.velocity_range = {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
        "z": (-0.1, 0.1),
        "roll": (-0.26, 0.26),
        "pitch": (-0.26, 0.26),
        "yaw": (-0.39, 0.39),
    }
    cfg.push.combined_exclusive = True
    with pytest.raises(ValueError, match="cannot use combined_exclusive"):
        ns["apply_combined_push_event"](cfg)


def test_apply_combined_builds_single_event_pair_and_keeps_legacy_none(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _armed_cfg()
    _post_init_order(ns, cfg)
    term = cfg.events.combined_push
    assert isinstance(term, _FakeEventTerm)
    assert term.func.__name__ == "push_combined_exclusive"
    assert term.mode == "interval"
    assert term.interval_range_s == (5.0, 15.0)
    assert term.params == {
        "velocity_range": {"x": (-0.35, 0.35), "y": (-0.35, 0.35)},
        "force_n": 60.0,
        "duration_steps": 15,  # 0.30 s @ 50 Hz
        "force_prob": 0.5,
        "body_name": "pelvis_link",
    }
    sweep = cfg.events.combined_push_sweep
    assert isinstance(sweep, _FakeEventTerm)
    assert sweep.func.__name__ == "sweep_expired_force_pushes"
    assert sweep.interval_range_s == (_CONTROL_DT, _CONTROL_DT)
    assert sweep.params == {}
    # legacy 三槽必须保持 None:合并模式下 legacy applier 只让路,不装事件
    assert cfg.events.push_robot is None
    assert cfg.events.force_push is None
    assert cfg.events.force_push_sweep is None


def test_apply_combined_with_yaw_axes_carries_angular_box(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _armed_cfg()
    cfg.push.ang_vel_radps = 0.5
    cfg.push.ang_axes = "yaw"
    _post_init_order(ns, cfg)
    assert cfg.events.combined_push.params["velocity_range"] == {
        "x": (-0.35, 0.35), "y": (-0.35, 0.35), "yaw": (-0.5, 0.5),
    }


def test_apply_combined_plus_legacy_events_fails_loud(monkeypatch):
    ns = _load_appliers(monkeypatch)
    # combined + legacy-both:两个 legacy 独立事件都挂着,必须 fail-loud
    cfg = _armed_cfg()
    cfg.events.push_robot = _FakeEventTerm()
    cfg.events.force_push = _FakeEventTerm()
    with pytest.raises(ValueError, match="forbids the legacy independent"):
        ns["apply_combined_push_event"](cfg)
    # 单个 legacy 事件在场也一样炸(含清扫兜底槽)
    for slot in ("push_robot", "force_push", "force_push_sweep"):
        cfg = _armed_cfg()
        setattr(cfg.events, slot, _FakeEventTerm())
        with pytest.raises(ValueError, match="forbids the legacy independent"):
            ns["apply_combined_push_event"](cfg)


def test_apply_combined_requires_both_branches_armed(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _armed_cfg()
    cfg.force_push.enable = False
    with pytest.raises(ValueError, match="BOTH branch recipes"):
        ns["apply_combined_push_event"](cfg)
    cfg = _armed_cfg()
    cfg.push.enable = False
    with pytest.raises(ValueError, match="BOTH branch recipes"):
        ns["apply_combined_push_event"](cfg)


def test_apply_combined_interval_spelling_must_match(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _armed_cfg()
    cfg.force_push.interval_range_s = (1.0, 3.0)
    with pytest.raises(ValueError, match="ONE trigger clock"):
        ns["apply_combined_push_event"](cfg)


def test_apply_combined_off_with_loaded_force_prob_fails_loud(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _make_env_cfg()
    cfg.push.force_prob = 0.7  # 关着的开关不许挂上膛参数
    with pytest.raises(ValueError, match="force_prob"):
        ns["apply_combined_push_event"](cfg)


def test_apply_combined_requires_declared_event_slots(monkeypatch):
    ns = _load_appliers(monkeypatch)
    cfg = _armed_cfg()
    cfg.events = _NS(push_robot=None, force_push=None, force_push_sweep=None)
    with pytest.raises(ValueError, match="DECLARE"):
        ns["apply_combined_push_event"](cfg)


def test_legacy_appliers_refuse_combined_without_merged_event(monkeypatch):
    """combined 旗标开着但合并事件没装好(顺序反了/漏跑):legacy applier 必须炸,不许静默让路。"""
    ns = _load_appliers(monkeypatch)
    cfg = _armed_cfg()  # events.combined_push 还是 None
    with pytest.raises(ValueError, match="combined_push is not wired"):
        ns["apply_push_robot_event"](cfg)
    with pytest.raises(ValueError, match="combined_push is not wired"):
        ns["apply_force_push_event"](cfg)


def test_legacy_appliers_untouched_when_combined_off(monkeypatch):
    """默认 combined_exclusive=False:legacy 直启路径行为逐字节照旧(独立事件对照常装配)。"""
    ns = _load_appliers(monkeypatch)
    cfg = _make_env_cfg()
    cfg.push.enable = True
    cfg.push.vel_xy_mps = 0.35
    cfg.force_push.enable = True
    cfg.force_push.force_n = 60.0
    cfg.force_push.duration_s = 0.3
    _post_init_order(ns, cfg)
    assert cfg.events.combined_push is None
    assert cfg.events.combined_push_sweep is None
    assert cfg.events.push_robot.func.__name__ == "push_by_setting_velocity"
    assert cfg.events.push_robot.params == {
        "velocity_range": {"x": (-0.35, 0.35), "y": (-0.35, 0.35)}
    }
    assert cfg.events.force_push.func.__name__ == "push_by_applying_wrench"
    assert cfg.events.force_push_sweep.func.__name__ == "sweep_expired_force_pushes"


# --------------------------------------------------------------------------------------------- #
# training_contract:legacy 块逐字节不变;合并块装配/记账/schema-3 校验
# --------------------------------------------------------------------------------------------- #
def _legacy_block(**overrides):
    kwargs = dict(
        enable=True, interval_range_s=(5.0, 15.0), vel_xy_mps=0.35,
        ang_vel_radps=0.5, ang_axes="yaw",
    )
    kwargs.update(overrides)
    return TC.push_robot_event_block(**kwargs)


def _combined_kwargs(**overrides):
    kwargs = dict(
        enable=True, interval_range_s=(5.0, 15.0), vel_xy_mps=0.35,
        ang_vel_radps=0.0, ang_axes="none",
        combined_exclusive=True, force_prob=0.5, force_n=60.0,
        duration_s=0.3, control_dt_s=_CONTROL_DT,
    )
    kwargs.update(overrides)
    return kwargs


def test_tc_legacy_block_is_byte_identical_and_free_of_combined_keys():
    block = _legacy_block()
    assert set(block) == set(TC._PUSH_ROBOT_EVENT_KEYS)
    assert "combined_exclusive" not in block  # v1 flag 语义:legacy 拼写一个新键都不带
    assert block == {
        "schema_version": 1,
        "enabled": True,
        "func": "push_by_setting_velocity",
        "mode": "interval",
        "interval_range_s": [5.0, 15.0],
        "vel_xy_mps": 0.35,
        "ang_vel_radps": 0.5,
        "ang_axes": "yaw",
        "velocity_range": {
            "x": [-0.35, 0.35], "y": [-0.35, 0.35], "yaw": [-0.5, 0.5],
        },
    }
    # 显式 combined_exclusive=False 与缺省调用逐字节同款
    assert _legacy_block(combined_exclusive=False) == block


def test_tc_combined_assembly_happy_path():
    block = TC.push_robot_event_block(**_combined_kwargs())
    assert set(block) == set(TC._PUSH_COMBINED_ASSEMBLY_KEYS)
    assert block["combined_exclusive"] is True and block["enabled"] is True
    assert block["func"] == "push_combined_exclusive" and block["mode"] == "interval"
    assert block["force_prob"] == 0.5
    # 速度侧与 legacy 装配逐位一致
    assert block["velocity_range"] == {"x": [-0.35, 0.35], "y": [-0.35, 0.35]}
    # 力侧与 force_push_event_block 单一来源逐位一致(含整数控制步换算)
    force = TC.force_push_event_block(
        enable=True, interval_range_s=(5.0, 15.0), force_n=60.0,
        duration_s=0.3, control_dt_s=_CONTROL_DT,
    )
    for key in ("force_n", "duration_s", "duration_steps", "control_dt_s",
                "body_name", "application_point"):
        assert block[key] == force[key]
    assert block["duration_steps"] == 15
    assert block["application_point"] == "pelvis_link_origin"


def test_tc_combined_requires_enable_true():
    with pytest.raises(ValueError, match="requires enable=true"):
        TC.push_robot_event_block(**_combined_kwargs(enable=False))


def test_tc_combined_off_rejects_dormant_merged_fields():
    for field in ("force_prob", "force_n", "duration_s", "control_dt_s"):
        with pytest.raises(ValueError, match="combined_exclusive=false"):
            _legacy_block(**{field: 0.5})


def test_tc_combined_missing_fields_fail_loud():
    for field in ("force_prob", "force_n", "duration_s", "control_dt_s"):
        with pytest.raises(ValueError, match="missing"):
            TC.push_robot_event_block(**_combined_kwargs(**{field: None}))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 2.0, float("nan"), True])
def test_tc_combined_force_prob_open_interval(bad):
    with pytest.raises(ValueError, match="force_prob"):
        TC.push_robot_event_block(**_combined_kwargs(force_prob=bad))


def test_tc_combined_force_side_reuses_force_push_validation():
    with pytest.raises(ValueError, match="whole number of control"):
        TC.push_robot_event_block(**_combined_kwargs(duration_s=0.31))
    with pytest.raises(ValueError, match="> 0"):
        TC.push_robot_event_block(**_combined_kwargs(force_n=0.0))


def test_tc_combined_exclusive_must_be_explicit_bool():
    with pytest.raises(ValueError, match="explicit boolean"):
        TC.push_robot_event_block(**_combined_kwargs(combined_exclusive="yes"))


def test_tc_bind_runtime_mass_accepts_combined_and_books_delta_v():
    block = TC.push_robot_event_block(**_combined_kwargs())
    full = TC.bind_force_push_runtime_mass(block, robot_mass_kg=30.0)
    assert set(full) == set(TC._PUSH_COMBINED_EVENT_KEYS)
    assert full["robot_mass_kg"] == 30.0
    assert full["delta_v_equiv_mps"] == 60.0 * 0.3 / 30.0  # Δv_equiv = F·Δt/m(力分支单次)
    # legacy force 装配块照旧可绑;走样键面照旧拒绝
    force = TC.force_push_event_block(
        enable=True, interval_range_s=(5.0, 15.0), force_n=60.0,
        duration_s=0.3, control_dt_s=_CONTROL_DT,
    )
    assert set(TC.bind_force_push_runtime_mass(force, robot_mass_kg=30.0)) == set(
        TC._FORCE_PUSH_EVENT_KEYS
    )
    with pytest.raises(ValueError, match="canonical assembly block"):
        TC.bind_force_push_runtime_mass(dict(block, surprise=1.0), robot_mass_kg=30.0)


# --------------------------------------------------------------------------------------------- #
# schema-3 validator:合并块 roundtrip + 篡改拒收;legacy 拼写继续有效
# --------------------------------------------------------------------------------------------- #
def _valid_combined_block(**overrides):
    block = TC.bind_force_push_runtime_mass(
        TC.push_robot_event_block(**_combined_kwargs()), robot_mass_kg=30.0
    )
    block.update(overrides)
    return block


def test_validator_combined_json_roundtrip_passes_bit_exact():
    block = _valid_combined_block()
    loaded = json.loads(json.dumps({"push_robot_event": block}))
    TC._validate_push_robot_event_contract(loaded)
    assert loaded["push_robot_event"] == block


def test_validator_legacy_spelling_still_valid():
    loaded = json.loads(json.dumps({"push_robot_event": _legacy_block()}))
    TC._validate_push_robot_event_contract(loaded)  # legacy 块继续有效(v1 flag 语义)


@pytest.mark.parametrize(
    "tamper, match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"enabled": False}, "omitting the block"),
        ({"combined_exclusive": False}, "combined_exclusive must be true"),
        ({"func": "push_by_setting_velocity"}, "push_combined_exclusive"),
        ({"mode": "reset"}, "interval"),
        ({"body_name": "torso_link"}, "body_name"),
        ({"application_point": "pelvis_com"}, "Yikang V9"),
        ({"force_prob": 1.5}, "is invalid"),                        # 出界概率本体
        ({"vel_xy_mps": 0.5}, "internally inconsistent"),           # 幅度改了但箱子没改
        ({"duration_steps": 14}, "internally inconsistent"),        # 步数改了但 duration 没改
        ({"delta_v_equiv_mps": 0.9}, "internally inconsistent"),    # Δv 必须 F·Δt/m 重算
        ({"robot_mass_kg": -30.0}, "is invalid"),                   # 非法质量本体
        (
            {"velocity_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}},
            "internally inconsistent",
        ),
    ],
)
def test_validator_rejects_tampered_combined_blocks(tamper, match):
    with pytest.raises(ValueError, match=match):
        TC._validate_push_robot_event_contract(
            {"push_robot_event": _valid_combined_block(**tamper)}
        )


def test_validator_rejects_half_combined_key_sets():
    # 带 combined_exclusive 键就按合并拼写要求全键面:缺一个/多一个都拒收
    short = _valid_combined_block()
    short.pop("robot_mass_kg")
    with pytest.raises(ValueError, match="missing fields"):
        TC._validate_push_robot_event_contract({"push_robot_event": short})
    extra = _valid_combined_block(surprise=1.0)
    with pytest.raises(ValueError, match="unknown fields"):
        TC._validate_push_robot_event_contract({"push_robot_event": extra})
    # legacy 块偷挂 combined_exclusive=False 也不是合法拼写(合并关着 = 不带这个键)
    legacy_plus_flag = dict(_legacy_block(), combined_exclusive=False)
    with pytest.raises(ValueError, match="missing fields"):
        TC._validate_push_robot_event_contract({"push_robot_event": legacy_plus_flag})


# --------------------------------------------------------------------------------------------- #
# hope_env_cfg declaration (source-text regression, the push/force-push test pattern)
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_declares_combined_defaults_off():
    source = _ENV_SRC
    # 旗标默认关 + 抽签概率默认 0.5,逐字冻结
    assert "combined_exclusive: bool = False" in source
    assert "force_prob: float = 0.5" in source
    # 合并事件对默认双 None(逐字节 no-op)
    assert "combined_push = None" in source
    assert "combined_push_sweep = None" in source
    # __post_init__ 消费顺序:combined 先跑,legacy applier 随后让路
    assert "apply_combined_push_event(self)" in source
    assert source.index("apply_combined_push_event(self)") < source.index(
        "apply_push_robot_event(self)"
    )
    # builder 与合同同源:同函数、interval 模式、共用 push_robot_event_block 合并分支
    builder = source[source.index("def apply_combined_push_event") :]
    builder = builder[: builder.index("\n##")]
    assert "push_robot_event_block" in builder
    assert "combined_exclusive=True" in builder
    assert "func=mdp.push_combined_exclusive" in builder
    assert "func=mdp.sweep_expired_force_pushes" in builder
    assert 'mode="interval"' in builder


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
