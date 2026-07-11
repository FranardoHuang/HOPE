"""Host-only source-contract tests for scripts/view_a3_stand.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/view_a3_stand.py"
HEADER = (
    ROOT / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/a3_policy_parameters.hpp"
)
MJCF = (
    ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)


def _load():
    spec = importlib.util.spec_from_file_location("view_a3_stand_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_production_arrays_are_parsed_not_retyped_and_neck_is_explicitly_passive():
    module = _load()
    params = module.load_production_parameters(HEADER)
    assert len(params.default_angles) == len(params.kps) == len(params.kds) == 29
    assert params.kps[0] == 400.0
    assert params.kps[17] == 1500.0
    assert params.kps[20] == 2000.0
    assert params.kds[17] == 8.0
    assert module.PASSIVE_JOINTS == ("head_yaw_joint", "head_pitch_joint")
    assert params.source_sha256 == hashlib.sha256(HEADER.read_bytes()).hexdigest()

    # Exercise the controller contract without importing MuJoCo: model order may
    # interleave the passive neck, while production arrays remain in policy order.
    names = list(module.POLICY_JOINT_ORDER[:3]) + list(module.PASSIVE_JOINTS)
    names += list(module.POLICY_JOINT_ORDER[3:])

    class FakeJointType:
        mjJNT_HINGE = 3

    class FakeObjectType:
        mjOBJ_JOINT = 0

    class FakeMujoco:
        mjtJoint = FakeJointType
        mjtObj = FakeObjectType

        @staticmethod
        def mj_id2name(model, _object_type, joint_id):
            return model.names[joint_id]

    class FakeModel:
        def __init__(self):
            self.nu = len(names)
            self.names = names
            self.actuator_trnid = np.column_stack(
                (np.arange(self.nu), np.zeros(self.nu, dtype=int))
            )
            self.jnt_type = np.full(self.nu, FakeJointType.mjJNT_HINGE)
            self.jnt_qposadr = np.arange(self.nu)
            self.jnt_dofadr = np.arange(self.nu)
            self.actuator_ctrllimited = np.ones(self.nu, dtype=bool)
            self.actuator_ctrlrange = np.column_stack(
                (np.full(self.nu, -1.0e9), np.full(self.nu, 1.0e9))
            )

    model = FakeModel()
    controller = module.StandPD(FakeMujoco, model, params)

    class FakeData:
        qpos = np.zeros(len(names))
        qvel = np.zeros(len(names))
        ctrl = np.full(len(names), np.nan)

    data = FakeData()
    policy_index = {name: i for i, name in enumerate(module.POLICY_JOINT_ORDER)}
    for joint_id, name in enumerate(names):
        if name not in module.PASSIVE_JOINTS:
            data.qpos[joint_id] = params.default_angles[policy_index[name]] + 0.01
    controller(model, data)
    for actuator_id, name in enumerate(names):
        if name in module.PASSIVE_JOINTS:
            assert data.ctrl[actuator_id] == 0.0
        else:
            assert data.ctrl[actuator_id] == pytest.approx(-0.01 * params.kps[policy_index[name]])


def test_cpp_array_parser_fails_on_declared_count_mismatch_and_duplicate():
    module = _load()
    with pytest.raises(ValueError, match="declares 2 values but parsed 1"):
        module.parse_cpp_double_arrays("constexpr std::array<double, 2> a = {1.0};")
    with pytest.raises(ValueError, match=r"duplicate C\+\+ array"):
        module.parse_cpp_double_arrays(
            "constexpr std::array<double, 1> a = {1.0};\n"
            "constexpr std::array<double, 1> a = {2.0};"
        )
    with pytest.raises(ValueError, match="non-literal initializer token"):
        module.parse_cpp_double_arrays(
            "constexpr std::array<double, 1> a = {1.0, SOME_MACRO};"
        )


def test_help_and_identity_only_do_not_require_mujoco():
    help_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "diagnostic" in help_proc.stdout

    identity_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--identity-only"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert identity_proc.returncode == 0, identity_proc.stderr
    payload = json.loads(identity_proc.stdout.splitlines()[0])
    assert payload["claim_scope"] == "plain_mujoco_pd_stand_diagnostic_only"
    assert payload["gate3"] == "not_run"
    assert payload["hardware"] == "not_run"
    assert payload["sources"]["vendor_mjcf"]["sha256"] == hashlib.sha256(
        MJCF.read_bytes()
    ).hexdigest()
    assert payload["sources"]["production_parameter_header"]["sha256"] == hashlib.sha256(
        HEADER.read_bytes()
    ).hexdigest()
    assert payload["controller_contract"]["passive_joint_names"] == [
        "head_yaw_joint",
        "head_pitch_joint",
    ]

    nan_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--identity-only", "--max-z-drift", "nan"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert nan_proc.returncode != 0
    assert "must all be finite" in nan_proc.stderr


def test_bad_explicit_source_fails_loudly(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--identity-only",
            "--gain-header",
            str(tmp_path / "missing.hpp"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "FileNotFoundError" in proc.stderr
