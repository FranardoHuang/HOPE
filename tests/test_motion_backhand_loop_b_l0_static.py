from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_motion_schema2_l0_static.py"
RUNNER_SCRIPT = ROOT / "scripts/run_motion_schema2_fk_consume_once.py"
PLAN = ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260714.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("motion_schema2_l0_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L0 = _load_module()


def _load_runner():
    spec = importlib.util.spec_from_file_location("motion_schema2_runner_portable_test", RUNNER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_runner()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    return L0.json_loads_exact(PLAN.read_bytes(), "tracked plan")


def _portable_lineage_fixture(tmp_path: Path):
    plan = _plan()
    activation_path = ROOT / plan["upstream_contracts"]["consume_activation"]["path"]
    receipt_path = ROOT / "configs/motion_backhand_loop_bc_schema2_fk_runtime_inspection_receipt_20260714.json"
    activation = L0.read_json(activation_path, "activation")
    receipt = L0.read_json(receipt_path, "receipt")
    canonical = Path(plan["upstream_contracts"]["consume_activation"]["path"])

    current_root = tmp_path / "new-clean-checkout"
    current_path = current_root / canonical
    current_path.parent.mkdir(parents=True)
    current_path.write_bytes(activation_path.read_bytes())
    for relative in (
        Path(plan["upstream_contracts"]["consume_runner"]["path"]),
        Path(plan["upstream_contracts"]["consume_source_gate_validator"]["path"]),
        Path(plan["upstream_contracts"]["runtime_body_order"]["path"]),
    ):
        destination = current_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=current_root, check=True)
    subprocess.run(["git", "add", "."], cwd=current_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "same activation bytes",
        ],
        cwd=current_root,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", "-q", "HEAD"], cwd=current_root, check=True)
    assert subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=current_root, text=True
    ) == ""
    current_meta = L0.binding(current_path)

    recorded_root = tmp_path / "old-consume-checkout"
    recorded_meta = {
        "path": str(recorded_root / canonical),
        "bytes": current_meta["bytes"],
        "sha256": current_meta["sha256"],
    }
    claim = {
        "activation": recorded_meta,
        "source_checkout": copy.deepcopy(activation["source_checkout"]),
    }
    return plan, activation, receipt, current_meta, claim, current_root, recorded_root


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


def test_dry_run_executes_full_audit_without_publishing_certificate(
    tmp_path, monkeypatch, capsys
):
    plan = _plan()
    certificate = tmp_path / "must-remain-absent.json"
    plan["output_contract"]["certificate_path"] = str(certificate)
    calls = []
    monkeypatch.setattr(
        L0,
        "validate_plan",
        lambda path, expected_sha: (plan, expected_sha),
    )
    monkeypatch.setattr(
        L0,
        "build_certificate",
        lambda received_plan, path, digest: calls.append(
            (received_plan, path, digest)
        )
        or {"status": "synthetic_full_read_only_pass"},
    )
    monkeypatch.setattr(
        L0,
        "write_certificate_exclusive",
        lambda *args, **kwargs: pytest.fail("dry-run attempted to publish a certificate"),
    )
    digest = "1" * 64
    assert L0.main(["--prereg", str(PLAN), "--expected-prereg-sha256", digest, "dry-run"]) == 0
    assert len(calls) == 1
    assert calls[0][0] is plan
    assert calls[0][2] == digest
    assert not certificate.exists()
    output = capsys.readouterr().out
    assert "runtime_audit=true" in output
    assert "certificate_written=false" in output
    assert "l0_static_complete=false" in output


def test_same_activation_bytes_validate_from_a_new_checkout_path(tmp_path):
    plan, activation, receipt, current_meta, claim, current_root, recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    gate = SimpleNamespace(REPO_ROOT=current_root)
    calls = []

    def validate_formal_result(
        received_activation,
        received_receipt,
        asset,
        received_meta,
        *,
        portable_source_context,
    ):
        names = R._expected_body_names(
            received_activation,
            portable_source_context=portable_source_context,
        )
        calls.append(
            (
                received_activation,
                received_receipt,
                asset,
                received_meta,
                gate.REPO_ROOT,
                portable_source_context,
            )
        )
        return {"portable": True, "body_count": len(names)}

    runner = SimpleNamespace(
        gate=gate,
        validate_formal_result=validate_formal_result,
        validate_detached_clean_checkout=R.validate_detached_clean_checkout,
        _validate_portable_source_context=R._validate_portable_source_context,
        HISTORICAL_RUNNER_BINDING=R.HISTORICAL_RUNNER_BINDING,
        HISTORICAL_SOURCE_GATE_VALIDATOR_BINDING=(
            R.HISTORICAL_SOURCE_GATE_VALIDATOR_BINDING
        ),
    )
    result = L0.validate_formal_result_portably(
        runner,
        activation,
        receipt,
        L0.ASSET_ID,
        current_meta,
        plan,
        claim,
        repo_root=current_root,
    )
    assert result == {"portable": True, "body_count": 32}
    assert len(calls) == 1
    assert calls[0][:5] == (
        activation,
        receipt,
        L0.ASSET_ID,
        claim["activation"],
        recorded_root,
    )
    assert calls[0][5]["recorded_source_checkout"] == activation["source_checkout"]
    assert not recorded_root.exists()
    assert gate.REPO_ROOT == current_root


def test_portable_current_body_order_drift_fails_closed(tmp_path):
    plan, activation, _receipt, _meta, _claim, current_root, _recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    body_order = current_root / plan["upstream_contracts"]["runtime_body_order"]["path"]
    body_order.write_text(body_order.read_text() + "drift\n")
    subprocess.run(["git", "add", "."], cwd=current_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "drift body order",
        ],
        cwd=current_root,
        check=True,
    )
    with pytest.raises(L0.L0ContractError, match="body order binding changed"):
        L0.build_portable_source_context(
            R,
            plan,
            activation["source_checkout"],
            repo_root=current_root,
        )


@pytest.mark.parametrize(
    "binding_name,error_fragment",
    [
        ("consume_runner", "current runner binding changed"),
        ("consume_source_gate_validator", "source validator binding changed"),
    ],
)
def test_portable_current_validator_code_drift_fails_closed(
    tmp_path, binding_name, error_fragment
):
    plan, activation, _receipt, _meta, _claim, current_root, _recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    target = current_root / plan["upstream_contracts"][binding_name]["path"]
    target.write_text(target.read_text() + "\n# drift\n")
    subprocess.run(["git", "add", "."], cwd=current_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", f"drift {binding_name}",
        ],
        cwd=current_root,
        check=True,
    )
    with pytest.raises(L0.L0ContractError, match=error_fragment):
        L0.build_portable_source_context(
            R,
            plan,
            activation["source_checkout"],
            repo_root=current_root,
        )


def test_portable_current_wrong_commit_fails_closed(tmp_path):
    plan, activation, _receipt, _meta, _claim, current_root, _recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    context = L0.build_portable_source_context(
        R,
        plan,
        activation["source_checkout"],
        repo_root=current_root,
    )
    context["current_checkout"]["commit"] = "0" * 40
    with pytest.raises(R.OneShotRunnerError, match="HEAD changed"):
        R._validate_portable_source_context(activation, context)


def test_portable_current_body_order_symlink_fails_closed(tmp_path):
    plan, activation, _receipt, _meta, _claim, current_root, _recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    body_order = current_root / plan["upstream_contracts"]["runtime_body_order"]["path"]
    body_order.unlink()
    body_order.symlink_to(ROOT / "configs/a3_runtime_body_order.txt")
    subprocess.run(["git", "add", "."], cwd=current_root, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "symlink body order",
        ],
        cwd=current_root,
        check=True,
    )
    with pytest.raises(L0.L0ContractError, match="symlink component"):
        L0.build_portable_source_context(
            R,
            plan,
            activation["source_checkout"],
            repo_root=current_root,
        )


def test_portable_activation_rejects_content_drift(tmp_path):
    plan, activation, receipt, current_meta, claim, current_root, _recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    claim["activation"]["sha256"] = "0" * 64
    with pytest.raises(L0.L0ContractError, match="content identity changed"):
        L0.portable_activation_context(
            plan,
            activation,
            receipt,
            current_meta,
            claim,
            repo_root=current_root,
        )


def test_portable_activation_rejects_wrong_bound_source_commit(tmp_path):
    plan, activation, receipt, current_meta, claim, current_root, _recorded_root = (
        _portable_lineage_fixture(tmp_path)
    )
    claim["source_checkout"]["commit"] = "0" * 40
    with pytest.raises(L0.L0ContractError, match="inspected source commit"):
        L0.portable_activation_context(
            plan,
            activation,
            receipt,
            current_meta,
            claim,
            repo_root=current_root,
        )


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
