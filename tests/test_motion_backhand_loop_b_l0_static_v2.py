from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_motion_schema2_l0_static_v2.py"
PLAN = ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260715_v2.json"
V1_SCRIPT = ROOT / "scripts/audit_motion_schema2_l0_static.py"
V1_PLAN = ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260714.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("motion_schema2_l0_v2_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L0 = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    return json.loads(PLAN.read_text())


def test_v2_static_source_gate_inherits_exact_unchanged_v1():
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
    assert "v1_unchanged=true" in run.stdout
    plan, digest, v1 = L0.validate_plan(PLAN, _sha(PLAN))
    assert digest == _sha(PLAN)
    assert plan["frozen_v1"]["validator"]["sha256"] == _sha(V1_SCRIPT)
    assert plan["frozen_v1"]["preregistration"]["sha256"] == _sha(V1_PLAN)
    assert v1["l0_contract"]["joint_range_tolerance_rad"] == 1.0e-5


def test_v1_failure_is_recorded_without_claiming_certificate():
    plan = _plan()
    failure = plan["v1_failure_evidence"]
    assert failure["outcome"] == "fail_closed_before_certificate"
    assert failure["certificate_written"] is False
    assert failure["position"] == {
        "not_equal_components": 537,
        "max_abs": 1.1920929e-7,
    }
    assert failure["body_angular_velocity"] == {
        "not_equal_components": 2320,
        "max_abs": 5.9679151e-6,
    }
    assert plan["authorization"]["l0_static_complete"] is False
    assert plan["authorization"]["vendor_l1_authorized"] is False
    assert plan["authorization"]["training_authorized"] is False
    assert plan["authorization"]["hardware_authorized"] is False


def test_two_bin_pose_contract_accepts_projection_roundoff_and_rejects_larger_drift():
    reference = np.ones((1, 1, 3), dtype=np.float32)
    one_bin = np.nextafter(reference, np.float32(np.inf))
    accepted = L0.evaluate_ulp_scaled_replay(
        reference,
        one_bin,
        label="position",
        max_ulp_bins=2,
        one_unit_floor=True,
        max_abs_tolerance=5.0e-7,
    )
    assert accepted["not_byte_equal_components"] == 3
    assert accepted["max_budget_fraction"] <= 0.5

    too_far = reference.copy()
    for _ in range(4):
        too_far = np.nextafter(too_far, np.float32(np.inf))
    with pytest.raises(L0.L0ContractError, match="exceeds 2-bin"):
        L0.evaluate_ulp_scaled_replay(
            reference,
            too_far,
            label="position",
            max_ulp_bins=2,
            one_unit_floor=True,
            max_abs_tolerance=5.0e-7,
        )


def test_one_unit_floor_is_tight_and_fails_non_numeric_motion_drift():
    reference = np.zeros((1,), dtype=np.float32)
    candidate = np.array([2.0 * L0.FLOAT32_EPSILON], dtype=np.float32)
    L0.evaluate_ulp_scaled_replay(
        reference,
        candidate,
        label="near-zero component",
        max_ulp_bins=2,
        one_unit_floor=True,
        max_abs_tolerance=5.0e-7,
    )
    candidate[0] = np.float32(3.0 * L0.FLOAT32_EPSILON)
    with pytest.raises(L0.L0ContractError, match="near-zero component exceeds"):
        L0.evaluate_ulp_scaled_replay(
            reference,
            candidate,
            label="near-zero component",
            max_ulp_bins=2,
            one_unit_floor=True,
            max_abs_tolerance=5.0e-7,
        )


def test_quaternion_hemisphere_cannot_be_hidden_by_numeric_budget():
    pos = np.zeros((2, 1, 3), dtype=np.float32)
    quat = np.zeros((2, 1, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    contract = _plan()["numerical_replay_contract"]
    with pytest.raises(L0.L0ContractError, match="changed hemisphere"):
        L0.evaluate_pose_replay(pos, pos, quat, -quat, contract)


def test_com_reconstruction_uses_exact_local_inertial_offset():
    pos = np.zeros((3, 1, 3), dtype=np.float32)
    quat = np.zeros((3, 1, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    pos[:, 0, 0] = np.array([0.0, 0.02, 0.04], dtype=np.float32)
    ipos = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    com = L0.reconstruct_com_from_stored_pose(pos, quat, ipos)
    assert com.dtype == np.float32
    expected = (pos[:, 0].astype(np.float64) + ipos[0]).astype(np.float32)
    assert np.array_equal(com[:, 0], expected)


def test_com_velocity_bound_is_derived_from_float32_and_50hz_not_observed_failure():
    frames = 151
    com = np.zeros((frames, 1, 3), dtype=np.float32)
    com[:, 0, 0] = np.arange(frames, dtype=np.float32) * np.float32(0.001)
    stored = np.gradient(com, 0.02, axis=0).astype(np.float32)
    contract = _plan()["numerical_replay_contract"]["com_linear_velocity"]
    result = L0.evaluate_com_linear_velocity(
        stored,
        com,
        np.array([[0.0, 0.0, 0.1]], dtype=np.float64),
        dt=0.02,
        pose_ulp_bins=2,
        contract=contract,
    )
    assert result["max_abs_delta_mps"] == 0.0
    assert 0.0 < result["derived_abs_tolerance_mps"] < 2.0e-4

    drift = stored.copy()
    drift[20, 0, 0] += np.float32(3.0e-4)
    with pytest.raises(L0.L0ContractError, match="exceeds 50 Hz roundoff bound"):
        L0.evaluate_com_linear_velocity(
            drift,
            com,
            np.array([[0.0, 0.0, 0.1]], dtype=np.float64),
            dt=0.02,
            pose_ulp_bins=2,
            contract=contract,
        )


def test_angular_velocity_stays_byte_exact_from_stored_quaternion():
    quat = np.zeros((5, 1, 4), dtype=np.float32)
    quat[..., 0] = 1.0

    def so3_derivative(values, dt):
        assert dt == 0.02
        return np.zeros((values.shape[0], 3), dtype=np.float32)

    converter = SimpleNamespace(so3_derivative=so3_derivative)
    stored = np.zeros((5, 1, 3), dtype=np.float32)
    assert L0.evaluate_body_angular_velocity_exact(
        stored, quat, dt=0.02, converter=converter
    )["comparison"] == "byte_equal"
    stored[2, 0, 1] = np.float32(1.0e-6)
    with pytest.raises(L0.L0ContractError, match="not producer-exact"):
        L0.evaluate_body_angular_velocity_exact(
            stored, quat, dt=0.02, converter=converter
        )


def test_v2_cannot_loosen_joint_ground_support_or_safety_gates(tmp_path):
    plan = _plan()
    plan["inherited_hard_gates"]["joint_range_tolerance_rad"] = 2.0e-5
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(plan))
    with pytest.raises(L0.L0ContractError, match="weakened or changed"):
        L0.validate_plan(drifted, _sha(drifted))

    plan = _plan()
    plan["authorization"]["training_authorized"] = True
    drifted.write_text(json.dumps(plan))
    with pytest.raises(L0.L0ContractError, match="authorization changed"):
        L0.validate_plan(drifted, _sha(drifted))


def test_dry_run_full_path_never_publishes_certificate(tmp_path, monkeypatch, capsys):
    plan = _plan()
    v1 = L0.V1.read_json(V1_PLAN, "V1 plan")
    certificate = tmp_path / "must-remain-absent.json"
    plan["output_contract"]["certificate_path"] = str(certificate)
    calls = []
    monkeypatch.setattr(L0, "validate_plan", lambda path, digest: (plan, digest, v1))
    monkeypatch.setattr(
        L0,
        "build_certificate",
        lambda *args: calls.append(args) or {"status": "synthetic_read_only_pass"},
    )
    monkeypatch.setattr(
        L0.V1,
        "write_certificate_exclusive",
        lambda *args, **kwargs: pytest.fail("dry-run attempted certificate publication"),
    )
    assert L0.main(
        ["--prereg", str(PLAN), "--expected-prereg-sha256", "1" * 64, "dry-run"]
    ) == 0
    assert len(calls) == 1
    assert not certificate.exists()
    assert "certificate_written=false" in capsys.readouterr().out
