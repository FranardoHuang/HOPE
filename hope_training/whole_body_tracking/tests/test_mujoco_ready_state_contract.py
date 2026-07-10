"""CPU-only regressions for cross-teacher MuJoCo ready-state identity."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "mujoco_eval_onnx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mujoco_ready_state_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()


def _snapshot(**overrides):
    values = dict(
        mode=M.FORMAL_READY_STATE_MODE,
        qpos=np.arange(8, dtype=np.float64),
        qvel=np.zeros(7),
        act=np.zeros(0),
        ctrl=np.zeros(2),
        last_action=np.zeros(2),
    )
    values.update(overrides)
    return M.ready_state_snapshot_contract(**values)


def test_ready_state_hash_is_deterministic_and_covers_hidden_actor_plant_channels():
    first = _snapshot()
    assert first == _snapshot()
    assert len(first["sha256"]) == 64

    for field in ("qpos", "qvel", "ctrl", "last_action"):
        changed = np.asarray({
            "qpos": np.arange(8, dtype=np.float64),
            "qvel": np.zeros(7),
            "ctrl": np.zeros(2),
            "last_action": np.zeros(2),
        }[field]).copy()
        changed[-1] += 1.0
        assert _snapshot(**{field: changed})["sha256"] != first["sha256"]
    assert _snapshot(time_s=0.01)["sha256"] != first["sha256"]
    assert _snapshot(qacc_warmstart=np.ones(7))["sha256"] != first["sha256"]
    assert _snapshot(mode=M.TEACHER_REFERENCE_READY_STATE_MODE)["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="finite"):
        _snapshot(qvel=np.array([np.nan]))


def test_ready_mode_defaults_formal_bank_to_common_stand_and_diagnostic_is_explicit():
    assert M.resolve_ready_state_mode(
        "auto", target_source="bank", deploy_faithful=False,
        allow_inexact_contract=False,
    ) == M.FORMAL_READY_STATE_MODE
    assert M.resolve_ready_state_mode(
        "auto", target_source="boxes", deploy_faithful=False,
        allow_inexact_contract=False,
    ) == M.TEACHER_REFERENCE_READY_STATE_MODE
    with pytest.raises(SystemExit, match="candidate-dependent"):
        M.resolve_ready_state_mode(
            "teacher-reference", target_source="bank", deploy_faithful=False,
            allow_inexact_contract=False,
        )
    assert M.resolve_ready_state_mode(
        "teacher-reference", target_source="bank", deploy_faithful=False,
        allow_inexact_contract=True,
    ) == M.TEACHER_REFERENCE_READY_STATE_MODE


def test_named_keyframe_reset_is_fail_closed_and_zeroes_dynamic_state_before_forward():
    source = inspect.getsource(M.MujocoRobot.reset_to_named_keyframe)
    reset = source.index("mj_resetDataKeyframe")
    qvel = source.index("self.data.qvel[:]")
    act = source.index("self.data.act[:]")
    ctrl = source.index("self.data.ctrl[:]")
    forward = source.index("mj_forward")
    assert reset < qvel < act < ctrl < forward

    fake_mj = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_KEY=1),
        mj_name2id=lambda *_args: -1,
    )
    fake_robot = SimpleNamespace(mj=fake_mj, model=object(), data=object())
    with pytest.raises(SystemExit, match="missing required named keyframe"):
        M.MujocoRobot.reset_to_named_keyframe(fake_robot, "stand")


def test_teacher_reference_contract_is_a_set_not_a_false_common_state():
    a = _snapshot(mode=M.TEACHER_REFERENCE_READY_STATE_MODE)
    b = _snapshot(
        mode=M.TEACHER_REFERENCE_READY_STATE_MODE,
        qpos=np.arange(8, dtype=np.float64) + 1.0,
    )
    aggregate = M.aggregate_teacher_reference_ready_contract([a, b])
    assert aggregate["mode"] == M.TEACHER_REFERENCE_READY_STATE_MODE
    assert [item["ready_state_sha256"] for item in aggregate["per_clip"]] == [
        a["sha256"], b["sha256"],
    ]
    assert aggregate["sha256"] not in (a["sha256"], b["sha256"])


def test_execution_contract_binds_actual_plant_and_ready_state():
    model = SimpleNamespace(
        dof_damping=np.array([1.0, 2.0]),
        dof_frictionloss=np.array([0.0, 0.0]),
        dof_armature=np.array([0.1, 0.2]),
        opt=SimpleNamespace(integrator=3),
    )
    robot = SimpleNamespace(
        model=model,
        vadr=np.array([0, 1]),
        ctrl_lo=np.array([-10.0, -20.0]),
        ctrl_hi=np.array([10.0, 20.0]),
        soft_jnt_lo=np.array([-1.0, -2.0]),
        soft_jnt_hi=np.array([1.0, 2.0]),
    )
    policy = SimpleNamespace(
        obs_dim=175,
        joint_names=("j0", "j1"),
        default_q=np.array([0.0, 0.1]),
        action_scale=np.array([0.25, 0.25]),
        kp=np.array([10.0, 20.0]),
        kd=np.array([1.0, 2.0]),
    )
    ready = _snapshot(last_action=np.zeros(2))

    def build():
        return M.build_evaluation_execution_contract(
            robot=robot, policy=policy, mjcf_sha256="a" * 64,
            evaluator_sha256="b" * 64, ready_state_contract=ready,
            sim_dt=0.005, decimation=4, pd_mode="implicit",
            passive_damping_mode="zero", frictionloss_mode="zero",
            qdes_clamp=True, one_question_reset=True,
            plant_semantics={"friction_proxy": False},
        )

    first = build()
    assert first == build()
    model.dof_armature[1] += 0.01
    assert build()["sha256"] != first["sha256"]
