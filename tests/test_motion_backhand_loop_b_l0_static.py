from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_motion_schema2_l0_static.py"
PLAN = ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260714.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("motion_schema2_l0_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L0 = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    return L0.json_loads_exact(PLAN.read_bytes(), "tracked plan")


def _write_exact_npz(path: Path) -> Path:
    frames = 151
    body_names = np.asarray(
        [
            line
            for line in (ROOT / "configs/a3_runtime_body_order.txt").read_text().splitlines()
            if line
        ]
    )
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=np.zeros((frames, 31), dtype=np.float32),
        joint_vel=np.zeros((frames, 31), dtype=np.float32),
        body_pos_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 32, 3), dtype=np.float32),
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=body_names,
    )
    return path


def test_tracked_static_source_gate_passes_without_runtime_inputs():
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prereg",
            str(PLAN),
            "--expected-prereg-sha256",
            _sha(PLAN),
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "source_exact=true" in run.stdout
    assert "runtime_audit=false" in run.stdout
    assert "no_write=true" in run.stdout


def test_plan_binds_exact_runtime_outputs_and_keeps_downstream_closed():
    plan, digest = L0.validate_plan(PLAN, _sha(PLAN))
    assert digest == _sha(PLAN)
    assert plan["exact_runtime_inputs"]["motion_npz"]["sha256"] == (
        "e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc"
    )
    assert plan["exact_runtime_inputs"]["materialization_report"]["sha256"] == (
        "4f5245937956290b3f623acbb588d99b346e5a1d874e55ee9caf010f2d75bc38"
    )
    assert plan["exact_runtime_inputs"]["consume_claim"]["sha256"] == (
        "76e7ff88fea39c13b45096edaad504b2570b3ce079acc96366b820a9c1295fb0"
    )
    assert plan["exact_runtime_inputs"]["consume_success"]["sha256"] == (
        "c0a25f2cba0e61bf0df7f63e6493948e16c5a3d3074f65091430f29e417f4f8b"
    )
    assert plan["authorization"]["l0_static_complete"] is False
    assert plan["authorization"]["vendor_l1_authorized"] is False
    assert plan["authorization"]["training_authorized"] is False
    assert plan["authorization"]["hardware_authorized"] is False


def test_duplicate_json_keys_and_nonfinite_constants_fail_closed():
    with pytest.raises(L0.L0ContractError, match="duplicate JSON key"):
        L0.json_loads_exact(b'{"a":1,"a":2}', "attack")
    with pytest.raises(L0.L0ContractError, match="non-finite JSON constant"):
        L0.json_loads_exact(b'{"a":NaN}', "attack")


def test_symlink_input_is_rejected(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(L0.L0ContractError, match="symlink component"):
        L0.ensure_regular_no_symlink(link, "attack")


def test_exact_npz_shape_time_order_finite_and_quaternion_contract(tmp_path):
    path = _write_exact_npz(tmp_path / "motion.npz")
    arrays = L0.load_npz_exact(path, _plan())
    assert arrays["joint_pos"].shape == (151, 31)
    assert arrays["body_pos_w"].shape == (151, 32, 3)
    assert float(arrays["_quaternion_max_norm_error"]) == 0.0

    nonfinite = tmp_path / "nonfinite.npz"
    _write_exact_npz(nonfinite)
    with np.load(nonfinite, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["joint_pos"] = payload["joint_pos"].copy()
    payload["joint_pos"][4, 2] = np.nan
    np.savez(nonfinite, **payload)
    with pytest.raises(L0.L0ContractError, match="NaN/Inf"):
        L0.load_npz_exact(nonfinite, _plan())

    bad_quat = tmp_path / "bad_quat.npz"
    _write_exact_npz(bad_quat)
    with np.load(bad_quat, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["body_quat_w"] = payload["body_quat_w"].copy()
    payload["body_quat_w"][0, 0, 0] = 0.9
    np.savez(bad_quat, **payload)
    with pytest.raises(L0.L0ContractError, match="quaternion max norm error"):
        L0.load_npz_exact(bad_quat, _plan())


def test_npz_rejects_unexpected_duplicate_members_and_velocity_drift(tmp_path):
    unexpected = _write_exact_npz(tmp_path / "unexpected.npz")
    with zipfile.ZipFile(unexpected, "a") as archive:
        archive.writestr("extra.npy", b"not-an-array")
    with pytest.raises(L0.L0ContractError, match="missing, duplicated or unexpected"):
        L0.load_npz_exact(unexpected, _plan())

    duplicate = _write_exact_npz(tmp_path / "duplicate.npz")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "a") as archive:
            archive.writestr("fps.npy", archive.read("fps.npy"))
    with pytest.raises(L0.L0ContractError, match="missing, duplicated or unexpected"):
        L0.load_npz_exact(duplicate, _plan())

    drift = _write_exact_npz(tmp_path / "drift.npz")
    with np.load(drift, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["joint_vel"] = payload["joint_vel"].copy()
    payload["joint_vel"][10, 2] = 0.1
    np.savez(drift, **payload)
    with pytest.raises(L0.L0ContractError, match="producer-exact gradient"):
        L0.load_npz_exact(drift, _plan())


def test_joint_range_uses_bound_limits_and_only_frozen_tolerance():
    q = np.array([[0.0, 1.000009]], dtype=np.float32)
    ranges = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    result = L0.evaluate_joint_ranges(q, ranges, ("a_joint", "b_joint"), 1.0e-5)
    assert result["max_excess_rad"] <= 1.0e-5
    q[0, 1] = 1.00002
    with pytest.raises(L0.L0ContractError, match="joint range excess"):
        L0.evaluate_joint_ranges(q, ranges, ("a_joint", "b_joint"), 1.0e-5)


def test_ground_clearance_uses_frozen_grounding_interval():
    result = L0.evaluate_ground_clearance(
        np.array([9.5e-6, 2.0e-3]),
        target_m=1.0e-5,
        maximum_m=1.0e-3,
        tolerance_m=5.0e-7,
    )
    assert result["minimum_frame"] == 0
    with pytest.raises(L0.L0ContractError, match="minimum ground clearance"):
        L0.evaluate_ground_clearance(
            np.array([9.4e-6, 2.0e-3]),
            target_m=1.0e-5,
            maximum_m=1.0e-3,
            tolerance_m=5.0e-7,
        )
    with pytest.raises(L0.L0ContractError, match="minimum ground clearance"):
        L0.evaluate_ground_clearance(
            np.array([1.0006e-3]),
            target_m=1.0e-5,
            maximum_m=1.0e-3,
            tolerance_m=5.0e-7,
        )


def test_certificate_publication_is_exclusive_and_rejects_symlink_parent(tmp_path):
    output = tmp_path / "certificate.json"
    L0.write_certificate_exclusive(output, {"schema_version": 1, "finite": True})
    assert json.loads(output.read_text()) == {"finite": True, "schema_version": 1}
    with pytest.raises(L0.L0ContractError, match="already exists"):
        L0.write_certificate_exclusive(output, {"schema_version": 1})

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(L0.L0ContractError, match="symlink component"):
        L0.write_certificate_exclusive(link / "attack.json", {"schema_version": 1})


def test_certificate_boundary_is_not_vendor_l1_or_training_pass():
    plan = _plan()
    assert "vendor_self_collision_or_racket_self_hit" in plan["explicit_non_claims"]
    assert "table_or_net_swept_clearance" in plan["explicit_non_claims"]
    assert "continuous_time_ground_clearance" in plan["explicit_non_claims"]
    assert "dynamics_balance_or_contact_stability" in plan["explicit_non_claims"]
    assert "RL_training_or_checkpoint_quality" in plan["explicit_non_claims"]
    assert "Gate3_or_hardware_safety" in plan["explicit_non_claims"]
