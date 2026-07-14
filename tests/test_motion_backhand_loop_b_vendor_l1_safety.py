from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_motion_schema2_vendor_l1_safety.py"
PLAN = ROOT / "configs/motion_backhand_loop_b_vendor_l1_safety_prereg_20260715.json"
L0_PLAN = ROOT / "configs/motion_backhand_loop_b_l0_static_prereg_20260715_v2.json"
L0_VALIDATOR = ROOT / "scripts/audit_motion_schema2_l0_static_v2.py"
PHASE = ROOT / "scripts/screen_motion_gmr_phase_safety.py"
SELF_COLLISION = ROOT / "hope_training/whole_body_tracking/scripts/audit_self_collision.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L1 = _load(SCRIPT, "motion_vendor_l1_test")
PHASE_MODULE = _load(PHASE, "motion_phase_safety_dense_test")


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_static_gate_binds_exact_l0_certificate_npz_model_runtime_and_helpers():
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
    assert "continuous_time_claim=false" in run.stdout
    plan, digest, _ = L1.validate_plan(PLAN, _sha(PLAN))
    assert digest == _sha(PLAN)
    assert plan["frozen_l0"]["certificate"]["sha256"] == (
        "60c08185e15c80621063bcedc65b42b6b738a12caeb8fb4e40a4c197e7daafc6"
    )
    assert plan["exact_runtime_input"]["sha256"] == (
        "e2eb99e69f624250e37d012ebc2c7db53c4213a6c73e8cd232b92640051d28cc"
    )
    assert plan["a3_model"]["derived_closure"]["manifest_sha256"] == (
        "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de"
    )
    assert plan["runtime"]["packages"] == {"numpy": "2.5.0", "mujoco": "3.10.0"}
    for key, path in (
        ("dense_safety_tool", PHASE),
        ("self_collision_helper", SELF_COLLISION),
    ):
        assert plan["dependencies"][key]["sha256"] == _sha(path)
        assert plan["dependencies"][key]["bytes"] == path.stat().st_size
    assert plan["frozen_l0"]["preregistration"]["sha256"] == _sha(L0_PLAN)
    assert plan["frozen_l0"]["validator"]["sha256"] == _sha(L0_VALIDATOR)


def test_dense_sampling_reuses_shortest_arc_contract_and_is_finite_not_continuous():
    frames = 151
    payload = {
        "root_pos": np.column_stack(
            [np.arange(frames, dtype=np.float64), np.zeros(frames), np.ones(frames)]
        ),
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1)),
        "dof_pos": np.zeros((frames, 31), dtype=np.float64),
        "fps": np.array([50.0]),
    }
    dense, source_time = PHASE_MODULE.densify_payload(payload, 8)
    assert dense["root_pos"].shape == (1201, 3)
    assert dense["dof_pos"].shape == (1201, 31)
    assert source_time.shape == (1201,)
    assert np.array_equal(dense["root_pos"][0], payload["root_pos"][0])
    assert np.array_equal(dense["root_pos"][-1], payload["root_pos"][-1])
    assert float(np.asarray(dense["fps"]).reshape(-1)[0]) == 400.0
    contract = _plan()["safety_contract"]
    assert contract["interpolation"].endswith("joint_position_linear")
    assert contract["dense_sampling_is_continuous_time_certificate"] is False


def test_any_collision_or_clearance_failure_is_noncompensable_and_marks_both_endpoints():
    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    with pytest.raises(L1.VendorL1Error, match="self_collision=1"):
        L1.summarize_hard_failures(
            np.array([False, True, False]),
            np.zeros(3, dtype=bool),
            times,
            2,
            PHASE_MODULE.unsafe_source_mask,
        )
    with pytest.raises(L1.VendorL1Error, match="racket_clearance=1"):
        L1.summarize_hard_failures(
            np.zeros(3, dtype=bool),
            np.array([False, True, False]),
            times,
            2,
            PHASE_MODULE.unsafe_source_mask,
        )
    safe = L1.summarize_hard_failures(
        np.zeros(3, dtype=bool),
        np.zeros(3, dtype=bool),
        times,
        2,
        PHASE_MODULE.unsafe_source_mask,
    )
    assert safe == {
        "dangerous_dense_samples": 0,
        "self_collision_dense_samples": 0,
        "racket_clearance_dense_samples": 0,
        "unsafe_source_frames": 0,
        "unsafe_source_indices": [],
        "hard_fail_is_noncompensable": True,
    }
    marked = PHASE_MODULE.unsafe_source_mask(
        2, times, np.array([False, True, False])
    )
    assert marked.tolist() == [True, True]


def test_contract_cannot_claim_continuous_time_or_weaken_five_mm_gate(tmp_path):
    plan = _plan()
    plan["safety_contract"]["dense_sampling_is_continuous_time_certificate"] = True
    drifted = tmp_path / "continuous.json"
    drifted.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(L1.VendorL1Error, match="safety contract changed"):
        L1.validate_plan(drifted, _sha(drifted))

    plan = _plan()
    plan["safety_contract"]["hard_racket_body_clearance_m"] = 0.004
    drifted.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(L1.VendorL1Error, match="safety contract changed"):
        L1.validate_plan(drifted, _sha(drifted))


def test_only_b_is_authorized_and_all_downstream_gates_remain_false():
    plan = _plan()
    assert plan["asset_id"] == "franco_backhand_loop_b"
    assert "backhand_loop_c" not in json.dumps(plan)
    assert plan["authorization"] == {
        "source_gate_pass": True,
        "cpu_vendor_l1_audit_authorized_after_review": True,
        "l0_static_complete": True,
        "vendor_l1_complete": False,
        "table_net_authorized": False,
        "dynamics_authorized": False,
        "simulator_authorized": False,
        "training_authorized": False,
        "formal_motion_authorized": False,
        "hardware_authorized": False,
    }


def test_dry_run_executes_runtime_path_but_never_writes(monkeypatch, tmp_path, capsys):
    plan = _plan()
    output = tmp_path / "must-remain-absent.json"
    plan["output_contract"]["certificate_path"] = str(output)
    calls: list[tuple] = []
    monkeypatch.setattr(
        L1,
        "validate_plan",
        lambda *args: (plan, "a" * 64, {"synthetic": "l0-v1"}),
    )
    monkeypatch.setattr(
        L1,
        "build_certificate",
        lambda *args: calls.append(args) or {"status": "synthetic-pass"},
    )
    monkeypatch.setattr(
        L1,
        "write_exclusive",
        lambda *args: pytest.fail("dry-run attempted certificate publication"),
    )
    assert L1.main(
        ["--prereg", str(PLAN), "--expected-prereg-sha256", "a" * 64, "dry-run"]
    ) == 0
    assert len(calls) == 1
    assert not output.exists()
    assert "certificate_written=false" in capsys.readouterr().out


def test_certificate_write_is_no_clobber(tmp_path):
    output = tmp_path / "certificate.json"
    L1.write_exclusive(output, {"status": "first"})
    first = output.read_bytes()
    with pytest.raises(L1.VendorL1Error, match="already exists"):
        L1.write_exclusive(output, {"status": "second"})
    assert output.read_bytes() == first
