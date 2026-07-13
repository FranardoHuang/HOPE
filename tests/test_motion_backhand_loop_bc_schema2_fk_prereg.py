from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_motion_schema2_fk.py"
B_PLAN = ROOT / "configs/motion_backhand_loop_b_schema2_fk_prereg_20260714.json"
C_PLAN = ROOT / "configs/motion_backhand_loop_c_schema2_fk_prereg_20260714.json"
SHARED = ROOT / "configs/motion_backhand_loop_bc_schema2_fk_runtime_v1.json"
SPEC = importlib.util.spec_from_file_location("schema2_fk_prereg_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_pair():
    b = MODULE.validate_plan(B_PLAN, MODULE.sha256_file(B_PLAN))
    c = MODULE.validate_plan(C_PLAN, MODULE.sha256_file(C_PLAN))
    MODULE.validate_pair(b, c)
    return b, c


def test_checked_in_pair_is_exact_independent_and_source_only():
    b, c = _validated_pair()
    assert {b[0]["asset_id"], c[0]["asset_id"]} == set(MODULE.ASSET_IDS)
    assert b[0]["source_motion"]["sha256"] != c[0]["source_motion"]["sha256"]
    assert b[0]["source_structure"]["expected_output_frames"] == 151
    assert c[0]["source_structure"]["expected_output_frames"] == 163
    assert b[0]["output_contract"]["output_root"] != c[0]["output_contract"]["output_root"]
    assert b[2]["closure"] == {
        "algorithm": "sha256(canonical-json(sorted[{path,bytes,sha256}]))-v1",
        "file_count": 75,
        "total_bytes": 14127373,
        "manifest_sha256": "e0381752eab46013c08559b331abb261beaa88a207a3c2f1155ab00857b962de",
        "xml_file_count": 1,
        "include_reference_count": 0,
        "external_file_reference_count": 74,
        "unique_external_file_count": 74,
        "mesh_reference_count": 74,
    }
    for plan, _sha, _evidence in (b, c):
        assert plan["authorization"]["source_gate_pass"] is True
        assert plan["authorization"]["runtime_inspection_run"] is False
        assert plan["authorization"]["schema2_materialized"] is False
        assert plan["authorization"]["simulator_authorized"] is False
        assert plan["authorization"]["training_authorized"] is False
        assert plan["authorization"]["hardware_authorized"] is False


@pytest.mark.parametrize("primary,peer", [(B_PLAN, C_PLAN), (C_PLAN, B_PLAN)])
def test_static_cli_passes_without_private_assets_or_runtime(primary: Path, peer: Path):
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prereg", str(primary),
            "--expected-prereg-sha256", MODULE.sha256_file(primary),
            "--peer-prereg", str(peer),
            "--expected-peer-prereg-sha256", MODULE.sha256_file(peer),
            "--hope_frame", "off",
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "PASS static" in run.stdout
    assert "runtime_inspection=false" in run.stdout


def test_cli_cannot_enable_second_hope_rotation():
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prereg", str(B_PLAN),
            "--expected-prereg-sha256", MODULE.sha256_file(B_PLAN),
            "--peer-prereg", str(C_PLAN),
            "--expected-peer-prereg-sha256", MODULE.sha256_file(C_PLAN),
            "--hope_frame", "on",
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert "invalid choice" in run.stderr


def test_static_cli_rejects_runtime_donor_argument():
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prereg", str(B_PLAN),
            "--expected-prereg-sha256", MODULE.sha256_file(B_PLAN),
            "--peer-prereg", str(C_PLAN),
            "--expected-peer-prereg-sha256", MODULE.sha256_file(C_PLAN),
            "--hope_frame", "off",
            "--donor", "/does/not/exist.onnx",
            "static",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert "must not receive --donor" in run.stderr


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["frame"].update({"input_already_in_HOPE_frame": False}), "HOPE/root-frame"),
        (lambda value: value["temporal"].update({"output_fps": 49}), "temporal contract"),
        (
            lambda value: value["schema2_output"].update(
                {"body_lin_vel_w_point": "link_origin_from_MuJoCo_xpos_gradient"}
            ),
            "output semantics",
        ),
        (
            lambda value: value["vendor_mjcf_closure"]["derived_closure"].update(
                {"manifest_sha256": "0" * 64}
            ),
            "closure drifted",
        ),
        (
            lambda value: value["restricted_pickle"].update({"allowed_globals": ["builtins.eval"]}),
            "allowlist drifted",
        ),
        (
            lambda value: value["donor_metadata"].update({"source_onnx_sha256": "0" * 64}),
            "differs from metadata snapshot",
        ),
    ],
)
def test_shared_contract_mutations_fail_closed(mutation, match):
    shared = copy.deepcopy(_load(SHARED))
    mutation(shared)
    with pytest.raises(MODULE.Schema2ContractError, match=match):
        MODULE.validate_shared_document(shared)


def test_pair_rejects_overlapping_output_and_duplicate_asset():
    b, c = _validated_pair()
    overlap = copy.deepcopy(c)
    overlap[0]["output_contract"]["output_root"] = b[0]["output_contract"]["output_root"]
    with pytest.raises(MODULE.Schema2ContractError, match="overlap"):
        MODULE.validate_pair(b, overlap)
    duplicate = copy.deepcopy(c)
    duplicate[0]["asset_id"] = "franco_backhand_loop_b"
    with pytest.raises(MODULE.Schema2ContractError, match="exactly one B and one C"):
        MODULE.validate_pair(b, duplicate)


def test_plan_hash_drift_and_duplicate_json_keys_fail_before_use(tmp_path: Path):
    with pytest.raises(MODULE.Schema2ContractError, match="prereg SHA"):
        MODULE.validate_plan(B_PLAN, "0" * 64)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(MODULE.Schema2ContractError, match="duplicate JSON key"):
        MODULE.validate_plan(duplicate, MODULE.sha256_file(duplicate))


@pytest.mark.parametrize("frames,expected", [(91, 151), (98, 163)])
def test_resample_is_exact_30_to_50_without_hope_rotation(frames: int, expected: int):
    phase = np.arange(frames, dtype=np.float32)
    root_pos = np.stack((phase, phase * 2.0, phase * 0.0 + 1.0), axis=1)
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0  # xyzw identity
    dof = np.repeat(phase[:, None], 31, axis=1)
    out_pos, out_rot, out_dof = MODULE.resample_payload(
        {"fps": 30, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof}
    )
    assert out_pos.shape == (expected, 3)
    assert out_rot.shape == (expected, 4)
    assert out_dof.shape == (expected, 31)
    np.testing.assert_array_equal(out_pos[0], root_pos[0])
    np.testing.assert_array_equal(out_pos[-1], root_pos[-1])
    np.testing.assert_array_equal(out_rot, np.repeat([[1.0, 0.0, 0.0, 0.0]], expected, axis=0))
    assert np.isfinite(out_dof).all()


def test_donor_snapshot_is_honest_required_subset_not_runtime_receipt():
    b, _c = _validated_pair()
    snapshot = b[2]["donor_snapshot"]
    assert snapshot["source_onnx_sha256"] == (
        "0c428ddf9968b047acbe7bbd5a39069a8e661ab0421038ea3b635284deb7b155"
    )
    boundary = snapshot["honesty_boundary"]
    assert boundary["tracked_map_is_required_subset_not_full_custom_metadata_dump"] is True
    assert boundary["metadata_reextracted_from_exact_onnx_in_this_source_gate"] is False
    assert set(snapshot["required_custom_metadata_map"]) == {
        "joint_names", "articulation_joint_names", "action_joint_ids"
    }


def test_schema2_builder_uses_link_pose_but_com_velocity_without_mujoco_runtime():
    b, _c = _validated_pair()
    plan, _sha, evidence = b
    frames = plan["source_structure"]["frames"]
    phase = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    payload = {
        "fps": 30,
        "root_pos": np.stack((phase, phase * 0.0, phase * 0.0 + 1.0), axis=1),
        "root_rot": np.repeat([[0.0, 0.0, 0.0, 1.0]], frames, axis=0).astype(np.float32),
        "dof_pos": np.repeat(phase[:, None], 31, axis=1),
    }

    class _Model:
        nbody = 33

    class _FakeFK:
        model = _Model()

        def body_names(self):
            return ["world", *evidence["body_names"]]

        def fk_with_com(self, base_pos, base_rot, dof_by_name):
            pos = np.repeat(np.asarray(base_pos, dtype=np.float32)[None, :], 33, axis=0)
            quat = np.repeat(np.asarray(base_rot, dtype=np.float32)[None, :], 33, axis=0)
            # A time-varying link->COM offset makes the required point split observable.
            offset = float(dof_by_name[evidence["joint_contract"].target_names[0]])
            com = pos.copy()
            com[:, 1] += offset
            return pos, quat, com

    arrays = MODULE.build_schema2_arrays(
        plan, evidence, {"payload": payload, "fkm": _FakeFK()}
    )
    assert arrays["body_pos_point"].item() == "link_origin"
    assert arrays["body_lin_vel_point"].item() == "center_of_mass"
    assert arrays["kinematics_schema_version"].item() == 2
    link_fd = np.gradient(arrays["body_pos_w"], 1.0 / 50.0, axis=0)
    assert not np.allclose(arrays["body_lin_vel_w"], link_fd)
    assert np.max(np.abs(arrays["body_lin_vel_w"][:, :, 1])) > 0.0
