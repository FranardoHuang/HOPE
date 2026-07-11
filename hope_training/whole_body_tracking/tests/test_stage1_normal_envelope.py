"""Dependency-light tests for the formal Stage-1 per-clip normal envelope."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

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
            [[0.75, -0.63, 0.20], [0.83, -0.52, 0.19], [0.70, -0.68, 0.22]],
            dtype=np.float64,
        ),
    }
    rows = {name: value / np.linalg.norm(value, axis=1)[:, None] for name, value in rows.items()}
    references = {
        "forehand": np.asarray([0.8, 0.58, 0.16], dtype=np.float64),
        "backhand": np.asarray([0.76, -0.62, 0.19], dtype=np.float64),
    }
    references = {name: value / np.linalg.norm(value) for name, value in references.items()}
    if flip_backhand:
        poisoned = np.asarray([0.1, 0.99, 0.0], dtype=np.float64)
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


def test_envelope_is_per_clip_sign_preserving_and_content_bound(tmp_path: Path):
    bank = tmp_path / "questions_train.npz"
    bank_sha, family_sha, rows = _write_bank(bank)
    metadata = NE.derive_stage1_normal_envelope(
        bank,
        expected_train_bank_sha256=bank_sha,
        expected_source_family_sha256=family_sha,
    )
    assert metadata["stage1_normal_envelope_frame"] == "world_table_frame0"
    assert metadata["stage1_normal_envelope_face_convention"] == "mount_plusY_A"
    assert metadata["stage1_normal_envelope_pairing"] == "shared_plus_y"
    assert metadata["stage1_normal_envelope_clip_order"] == "forehand,backhand"
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
    # Forehand and backhand caps are not pooled; their lateral signs remain distinct.
    assert centers[0, 1] > 0.0 and centers[1, 1] < 0.0


def test_envelope_rejects_cross_sign_rows_and_mismatched_bindings(tmp_path: Path):
    bank = tmp_path / "questions_train.npz"
    bank_sha, family_sha, _ = _write_bank(bank, flip_backhand=True)
    with pytest.raises(ValueError, match="opposite faces must never be averaged"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256=family_sha,
        )

    bank_sha, family_sha, _ = _write_bank(bank)
    with pytest.raises(ValueError, match="train-bank SHA disagrees"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256="0" * 64,
            expected_source_family_sha256=family_sha,
        )
    with pytest.raises(ValueError, match="source-family SHA disagrees"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256="1" * 64,
        )


def test_envelope_rejects_wrong_clip_order_and_non_unit_rows(tmp_path: Path):
    bank = tmp_path / "questions_train.npz"
    bank_sha, family_sha, _ = _write_bank(bank)
    with pytest.raises(ValueError, match="requires clip order"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=bank_sha,
            expected_source_family_sha256=family_sha,
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
        )

    bank_sha, family_sha, _ = _write_bank(bank)
    with np.load(bank) as original:
        arrays = {key: np.asarray(original[key]) for key in original.files}
    row = np.asarray([-0.1, 0.98, 0.17], dtype=np.float64)
    arrays["forehand/demanded_normal"] = arrays["forehand/demanded_normal"].copy()
    arrays["forehand/demanded_normal"][0] = row / np.linalg.norm(row)
    np.savez(bank, **arrays)
    changed_sha = hashlib.sha256(bank.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="wire cannot represent"):
        NE.derive_stage1_normal_envelope(
            bank,
            expected_train_bank_sha256=changed_sha,
            expected_source_family_sha256=family_sha,
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
    assert "--train-bank" in standalone
    assert "load_question_bank(" in standalone
    assert "expected_split=\"train\"" in standalone
    assert "validate_runtime_motion_contract(" in standalone

    policy = (
        ROOT.parents[1]
        / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include/"
        "a3_pingpong/pp_policy.hpp"
    ).read_text(encoding="utf-8")
    gate = "face_normal_within_training_envelope(eng_clip, normal_w)"
    commit = "planner_frozen_normal_w_ = normal_w"
    assert gate in policy and commit in policy
    assert policy.index(gate) < policy.index(commit)
    assert "face_command_out_of_train_envelope" in policy
