"""Dependency-light tests for the formal Stage-1 per-clip normal envelope."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/stage1_normal_envelope.py"
)
SPEC = importlib.util.spec_from_file_location("stage1_normal_envelope_under_test", MODULE_PATH)
NE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NE)


def _canonical_sha256(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_bank(path: Path, *, flip_backhand: bool = False) -> tuple[str, str, dict[str, np.ndarray]]:
    rows = {
        "forehand": np.asarray(
            [[0.82, 0.55, 0.16], [0.88, 0.43, 0.20], [0.78, 0.60, 0.17]],
            dtype=np.float64,
        ),
        "backhand": np.asarray(
            [[-0.75, 0.63, -0.20], [-0.83, 0.52, -0.19], [-0.70, 0.68, -0.22]],
            dtype=np.float64,
        ),
    }
    rows = {name: value / np.linalg.norm(value, axis=1)[:, None] for name, value in rows.items()}
    references = {
        "forehand": np.asarray([0.8, 0.58, 0.16], dtype=np.float64),
        "backhand": np.asarray([-0.76, 0.62, -0.19], dtype=np.float64),
    }
    references = {name: value / np.linalg.norm(value) for name, value in references.items()}
    if flip_backhand:
        poisoned = np.asarray([-0.1, -0.99, 0.0], dtype=np.float64)
        rows["backhand"][1] = poisoned / np.linalg.norm(poisoned)
    family = {
        "contract": "stage1-source-family-v2",
        "stage": "S1",
        "face_frame": "mount_plusY_A",
        "clip_order": ["forehand", "backhand"],
        "clips": {
            name: {"clip_normal": np.round(value, 12).tolist()}
            for name, value in references.items()
        },
    }
    family_sha = _canonical_sha256(family)
    metadata = {
        "schema_version": 3,
        "split": "train",
        "stage": "S1",
        "face_frame": "mount_plusY_A",
        "clip_order": ["forehand", "backhand"],
        "source_family_contract": family,
        "source_family_sha256": family_sha,
        "clips": {
            name: {
                "question_count": int(value.shape[0]),
                "clip_normal": np.round(references[name], 12).tolist(),
            }
            for name, value in rows.items()
        },
    }
    arrays = {
        "meta_json": np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )
    }
    for name in ("forehand", "backhand"):
        arrays[f"{name}/demanded_normal"] = rows[name]
        arrays[f"{name}/clip_normal"] = references[name]
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest(), family_sha, rows


def _decode_vectors(value: str) -> np.ndarray:
    return np.asarray(
        [[float(item) for item in vector.split(",")] for vector in value.split(";")],
        dtype=np.float64,
    )


def test_python_payload_matches_cpp_representative_digest():
    metadata = {
        "stage1_normal_envelope_schema_version": "1",
        "stage1_normal_envelope_frame": "world_table_frame0",
        "stage1_normal_envelope_face_convention": "mount_plusY_A",
        "stage1_normal_envelope_pairing": "shared_plus_y",
        "stage1_normal_envelope_algorithm": "per_clip_sign_preserving_spherical_mean_cap_v1",
        "stage1_normal_envelope_bank_row_unit_tolerance": "0.0002",
        "stage1_normal_envelope_runtime_unit_tolerance": "0.000001",
        "stage1_normal_envelope_runtime_dot_tolerance": "0.000001",
        "stage1_normal_envelope_clip_order": "forehand,backhand",
        "stage1_normal_envelope_mount_normal_sign_per_clip": "1,-1",
        "stage1_normal_envelope_centers": "0.8,0.6,0;-0.8,0.6,0",
        "stage1_normal_envelope_reference_normals": "0.8,0.6,0;-0.8,0.6,0",
        "stage1_normal_envelope_min_dots": "0.974278,0.972078",
        "stage1_normal_envelope_row_counts": "757,724",
        "stage1_normal_envelope_train_bank_sha256": (
            "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700"
        ),
        "stage1_normal_envelope_source_family_sha256": (
            "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5"
        ),
    }
    digest = hashlib.sha256(NE.normal_envelope_payload(metadata).encode()).hexdigest()
    assert digest == "d47b426c41df974fbdd83adb68cf563f87844958b3a4e28a7b3d80d64a7ddc88"


def test_envelope_is_per_clip_sign_preserving_and_content_bound(tmp_path: Path):
    bank = tmp_path / "questions_train.npz"
    bank_sha, family_sha, rows = _write_bank(bank)
    metadata = NE.derive_stage1_normal_envelope(
        bank,
        expected_train_bank_sha256=bank_sha,
        expected_source_family_sha256=family_sha,
        mount_normal_sign_per_clip=(1.0, -1.0),
    )
    assert metadata["stage1_normal_envelope_frame"] == "world_table_frame0"
    assert metadata["stage1_normal_envelope_face_convention"] == "mount_plusY_A"
    assert metadata["stage1_normal_envelope_pairing"] == "shared_plus_y"
    assert metadata["stage1_normal_envelope_clip_order"] == "forehand,backhand"
    assert metadata["stage1_normal_envelope_mount_normal_sign_per_clip"] == "1,-1"
    assert metadata["stage1_normal_envelope_row_counts"] == "3,3"
    assert metadata["stage1_normal_envelope_train_bank_sha256"] == bank_sha
    assert metadata["stage1_normal_envelope_source_family_sha256"] == family_sha
    payload_sha = hashlib.sha256(NE.normal_envelope_payload(metadata).encode()).hexdigest()
    assert metadata["stage1_normal_envelope_payload_sha256"] == payload_sha

    centers = _decode_vectors(metadata["stage1_normal_envelope_centers"])
    thresholds = np.asarray(
        [float(value) for value in metadata["stage1_normal_envelope_min_dots"].split(",")]
    )
    assert np.linalg.norm(centers, axis=1) == pytest.approx([1.0, 1.0], abs=1e-12)
    for clip, name in enumerate(("forehand", "backhand")):
        unit_rows = rows[name] / np.linalg.norm(rows[name], axis=1)[:, None]
        assert np.min(unit_rows @ centers[clip]) == pytest.approx(thresholds[clip], abs=1e-15)
    # A merely opponent-facing normal can still be outside the clip-specific train support.
    assert float(np.asarray([1.0, 0.0, 0.0]) @ centers[0]) < thresholds[0]
    # Raw-A forehand/backhand caps are not pooled. BH stays raw negative-X; multiplying by its
    # frozen striking-face sign produces the opponent-facing positive-X physical-B wire normal.
    assert centers[0, 0] > 0.0 and centers[1, 0] < 0.0
    assert (-1.0 * centers[1])[0] > 0.0


def test_envelope_rejects_cross_sign_rows_and_mismatched_bindings(tmp_path: Path):
    bank = tmp_path / "questions_train.npz"
    bank_sha, family_sha, _ = _write_bank(bank, flip_backhand=True)
    with pytest.raises(ValueError, match="opposite faces must never be averaged"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256=family_sha,
            mount_normal_sign_per_clip=(1.0, -1.0),
        )

    bank_sha, family_sha, _ = _write_bank(bank)
    with pytest.raises(ValueError, match="train-bank SHA disagrees"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256="0" * 64,
            expected_source_family_sha256=family_sha,
            mount_normal_sign_per_clip=(1.0, -1.0),
        )
    with pytest.raises(ValueError, match="source-family SHA disagrees"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256="1" * 64,
            mount_normal_sign_per_clip=(1.0, -1.0),
        )


def test_envelope_rejects_wrong_clip_order_and_non_unit_rows(tmp_path: Path):
    bank = tmp_path / "questions_train.npz"
    bank_sha, family_sha, _ = _write_bank(bank)
    with pytest.raises(ValueError, match="requires clip order"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256=family_sha,
            mount_normal_sign_per_clip=(1.0, -1.0),
            clip_order=("backhand", "forehand"),
        )

    with np.load(bank) as original:
        arrays = {key: np.asarray(original[key]) for key in original.files}
    arrays["forehand/demanded_normal"] = arrays["forehand/demanded_normal"].copy()
    arrays["forehand/demanded_normal"][0] *= 0.9
    np.savez(bank, **arrays)
    changed_sha = hashlib.sha256(bank.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unit tolerance"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=changed_sha,
            expected_source_family_sha256=family_sha,
            mount_normal_sign_per_clip=(1.0, -1.0),
        )

    bank_sha, family_sha, _ = _write_bank(bank)
    with np.load(bank) as original:
        arrays = {key: np.asarray(original[key]) for key in original.files}
    row = np.asarray([0.1, 0.98, 0.17], dtype=np.float64)
    arrays["backhand/demanded_normal"] = arrays["backhand/demanded_normal"].copy()
    arrays["backhand/demanded_normal"][0] = row / np.linalg.norm(row)
    np.savez(bank, **arrays)
    changed_sha = hashlib.sha256(bank.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="wire cannot represent"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=changed_sha,
            expected_source_family_sha256=family_sha,
            mount_normal_sign_per_clip=(1.0, -1.0),
        )

    bank_sha, family_sha, _ = _write_bank(bank)
    with pytest.raises(ValueError, match=r"requires mount_normal_sign_per_clip=\[\+1,-1\]"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256=family_sha,
            mount_normal_sign_per_clip=(1.0, 1.0),
        )


def test_both_export_paths_derive_from_exact_bank_and_runtime_gate_is_precommit():
    native = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"
    ).read_text(encoding="utf-8")
    standalone = (ROOT / "scripts/standalone_onnx_export.py").read_text(encoding="utf-8")
    for source in (native, standalone):
        assert "derive_stage1_normal_envelope(" in source
        assert "expected_train_bank_sha256=" in source
        assert "expected_source_family_sha256=" in source
        assert "mount_normal_sign_per_clip=checkpoint_mount_signs" in source
    assert "--train-bank" in standalone
    assert "load_question_bank(" in standalone
    assert "expected_split=\"train\"" in standalone
    assert "validate_runtime_motion_contract(" in standalone

    policy = (
        ROOT.parents[1]
        / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
        "a3_pingpong/pp_policy.hpp"
    ).read_text(encoding="utf-8")
    convert = "face_normal_raw_a_from_wire_b(eng_clip, normal_w)"
    gate = "face_normal_within_training_envelope(eng_clip, normal_raw_a_w)"
    commit = "planner_frozen_normal_w_ = normal_raw_a_w"
    assert convert in policy and gate in policy and commit in policy
    assert policy.index(convert) < policy.index(gate) < policy.index(commit)
    assert "planner_frozen_pos_w_ = pos_w" in policy
    assert "planner_frozen_vel_w_ = candidate_vel_w" in policy
    assert "face_command_out_of_train_envelope" in policy


def test_standalone_contract_import_is_isaac_free_subprocess():
    script = ROOT / "scripts/standalone_onnx_export.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--contract-import-smoke"],
        cwd=ROOT.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "whole_body_tracking_package_imported=false" in result.stdout
    assert "isaac_modules_imported=false" in result.stdout
    assert "onnx_torch_imported=false" in result.stdout
    assert "training_schema=3 envelope_schema=1" in result.stdout


def test_real_formal_bank_expectation_fixture_binds_backhand_a_to_b_sign():
    repo = ROOT.parents[1]
    fixture_path = repo / "configs/phase1_face179_real_bank_envelope_expectations_20260712.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    manifest = repo / fixture["source_asset_manifest"]["path"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == fixture["source_asset_manifest"][
        "sha256"
    ]
    assert fixture["train_bank"]["sha256"] == (
        "2da2bd1280c45944418d41fe5788d09d7c0ebb0ff7d34fa87c8dd0fcf16a0700"
    )
    assert fixture["train_bank"]["source_family_sha256"] == (
        "b21c161a0240893a4a469136c2d5298c2ecfa9f2b4a8c6fb9493b679f3728ad5"
    )
    restore_doc = repo / fixture["ignored_train_bank_restore"]["runbook"].split("#", 1)[0]
    assert restore_doc.is_file()
    assert fixture["ignored_train_bank_restore"]["pod1_path"].endswith(
        "/s1_v4rg_runtime_order_schema3_train.npz"
    )
    assert fixture["mount_normal_sign_per_clip"] == [1.0, -1.0]
    assert fixture["external_wire"]["normal_semantics"] == (
        "physical_striking_face_B_opponent_facing"
    )
    for name in ("forehand", "backhand"):
        clip = fixture["clips"][name]
        sign = clip["mount_normal_sign"]
        raw = np.asarray(clip["raw_A_x_range"], dtype=np.float64)
        physical = np.asarray(clip["physical_B_x_range"], dtype=np.float64)
        # Multiplication reverses interval order for BH; sort before exact comparison.
        assert np.sort(sign * raw).tolist() == pytest.approx(physical.tolist(), abs=1e-12)
        assert np.all(physical > fixture["external_wire"]["normal_x_strict_min"])
        assert clip["min_raw_A_row_dot_raw_A_reference"] > 0.0
        assert clip["expected_center_dot_raw_A_reference"] > 0.0
        assert 0.0 < clip["expected_cap_min_dot"] <= 1.0
    assert fixture["clips"]["backhand"]["raw_A_x_range"][1] < 0.0
    assert fixture["clips"]["backhand"]["physical_B_x_range"][0] > 0.0
    assert [fixture["clips"][name]["row_count"] for name in ("forehand", "backhand")] == [
        757,
        724,
    ]
    assert [
        fixture["clips"][name]["expected_cap_min_dot"]
        for name in ("forehand", "backhand")
    ] == pytest.approx([0.974278, 0.972078], abs=1e-12)
