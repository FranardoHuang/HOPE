"""CPU contract tests for the exogenous post-swing teacher receipt."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "post_swing_teacher.py"
)
SPEC = importlib.util.spec_from_file_location("post_swing_teacher_contract_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ARTIFACT_KIND = MODULE.ARTIFACT_KIND
CAPTURE_CONTRACT = MODULE.CAPTURE_CONTRACT
PostSwingTeacherError = MODULE.PostSwingTeacherError
load_post_swing_teacher_states = MODULE.load_post_swing_teacher_states
sha256_file = MODULE.sha256_file


def _write_fixture(tmp_path: Path, *, count: int = 4, joints: int = 3):
    motion_paths = []
    for index in range(2):
        path = tmp_path / f"motion_{index}.npz"
        np.savez(path, marker=np.array([index], dtype=np.int64))
        motion_paths.append(path)
    joint_names = [f"joint_{index}" for index in range(joints)]
    root = np.zeros((count, 13), dtype=np.float32)
    root[:, 3] = 1.0
    root[:, 2] = np.linspace(0.9, 1.0, count, dtype=np.float32)
    payload = tmp_path / "teacher_states.npz"
    np.savez(
        payload,
        root_state_origin_relative=root,
        joint_pos=np.zeros((count, joints), dtype=np.float32),
        joint_vel=np.zeros((count, joints), dtype=np.float32),
    )
    receipt = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "capture_contract": dict(CAPTURE_CONTRACT),
        "teacher": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "training_contract_sha256": "3" * 64,
            "training_contract_schema_version": 3,
            "fresh_lineage": True,
        },
        "motion_clips": [
            {"index": index, "sha256": sha256_file(path)}
            for index, path in enumerate(motion_paths)
        ],
        "states": {
            "relative_path": payload.name,
            "sha256": sha256_file(payload),
            "count": count,
            "root_shape": [count, 13],
            "joint_pos_shape": [count, joints],
            "joint_vel_shape": [count, joints],
            "joint_names": joint_names,
        },
    }
    receipt_path = tmp_path / "teacher_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, motion_paths, joint_names, receipt


def _load(receipt_path, motion_paths, joint_names, *, min_fill=4, buffer_size=8):
    return load_post_swing_teacher_states(
        receipt_path,
        sha256_file(receipt_path),
        expected_motion_sha256=[sha256_file(path) for path in motion_paths],
        expected_joint_names=joint_names,
        min_fill=min_fill,
        buffer_size=buffer_size,
    )


def test_exact_receipt_loads_and_returns_hard_contract(tmp_path):
    receipt_path, motions, joints, _ = _write_fixture(tmp_path)
    loaded = _load(receipt_path, motions, joints)
    assert loaded.root_state_origin_relative.shape == (4, 13)
    assert loaded.joint_pos.shape == loaded.joint_vel.shape == (4, 3)
    assert loaded.hard_contract["receipt_sha256"] == sha256_file(receipt_path)
    assert loaded.hard_contract["capture_contract"]["event"] == "natural_clip_wrap"
    assert loaded.hard_contract["teacher"]["fresh_lineage"] is True


def test_receipt_rejects_byte_drift_and_wrong_runtime_bindings(tmp_path):
    receipt_path, motions, joints, _ = _write_fixture(tmp_path)
    with pytest.raises(PostSwingTeacherError, match="receipt byte SHA"):
        load_post_swing_teacher_states(
            receipt_path,
            "f" * 64,
            expected_motion_sha256=[sha256_file(path) for path in motions],
            expected_joint_names=joints,
            min_fill=4,
            buffer_size=8,
        )
    with pytest.raises(PostSwingTeacherError, match="runtime motion bytes"):
        load_post_swing_teacher_states(
            receipt_path,
            sha256_file(receipt_path),
            expected_motion_sha256=["f" * 64, sha256_file(motions[1])],
            expected_joint_names=joints,
            min_fill=4,
            buffer_size=8,
        )
    with pytest.raises(PostSwingTeacherError, match="joint_names/order"):
        _load(receipt_path, motions, list(reversed(joints)))


def test_receipt_rejects_non_natural_wrap_and_underfilled_payload(tmp_path):
    receipt_path, motions, joints, receipt = _write_fixture(tmp_path)
    receipt["capture_contract"]["event"] = "arbitrary_timeout"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PostSwingTeacherError, match="natural-clip-wrap"):
        _load(receipt_path, motions, joints)

    under = tmp_path / "under"
    under.mkdir()
    receipt_path, motions, joints, _ = _write_fixture(under, count=3)
    with pytest.raises(PostSwingTeacherError, match="below post_swing_min_fill"):
        _load(receipt_path, motions, joints, min_fill=4)


def test_receipt_rejects_payload_tamper_nonfinite_and_symlink(tmp_path):
    receipt_path, motions, joints, receipt = _write_fixture(tmp_path)
    payload = tmp_path / receipt["states"]["relative_path"]
    with payload.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(PostSwingTeacherError, match="payload byte SHA"):
        _load(receipt_path, motions, joints)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    receipt_path, motions, joints, receipt = _write_fixture(bad_dir)
    payload = bad_dir / receipt["states"]["relative_path"]
    root = np.zeros((4, 13), dtype=np.float32)
    root[:, 3] = 1.0
    root[0, 0] = np.nan
    np.savez(
        payload,
        root_state_origin_relative=root,
        joint_pos=np.zeros((4, 3), dtype=np.float32),
        joint_vel=np.zeros((4, 3), dtype=np.float32),
    )
    receipt["states"]["sha256"] = sha256_file(payload)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PostSwingTeacherError, match="NaN or Inf"):
        _load(receipt_path, motions, joints)

    link_dir = tmp_path / "link"
    link_dir.mkdir()
    receipt_path, motions, joints, receipt = _write_fixture(link_dir)
    target = link_dir / receipt["states"]["relative_path"]
    real = link_dir / "real.npz"
    target.rename(real)
    target.symlink_to(real.name)
    receipt["states"]["sha256"] = sha256_file(real)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PostSwingTeacherError, match="symlink"):
        _load(receipt_path, motions, joints)
