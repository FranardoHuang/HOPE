"""CPU contract tests for the exogenous post-swing teacher receipt."""

from __future__ import annotations

import importlib.util
import json
import io
from pathlib import Path
import sys
import zipfile

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
        "schema_version": 2,
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
            "velocity_limits": {
                "root_linear_norm_max_mps": 2.0,
                "root_angular_norm_max_radps": 4.0,
                "joint_abs_max_radps": [5.0] * joints,
            },
        },
        "attestation": {
            "schema_version": 1,
            "artifact_kind": MODULE.ATTESTATION_KIND,
            "capture_result_sha256": "4" * 64,
            "capture_result_relative_path": MODULE.CAPTURE_RESULT_NAME,
            "capture_claim_sha256": "a" * 64,
            "capture_claim_relative_path": MODULE.CAPTURE_CLAIM_NAME,
            "checkpoint": {
                "sha256": "2" * 64,
                "training_contract_schema_version": 3,
                "training_contract_sha256": "3" * 64,
                "training_contract_lineage_exact": True,
                "training_launch_claim_sha256": "5" * 64,
            },
            "hard_contract": {"sha256": "3" * 64, "schema_version": 3},
            "checkpoint_source": {
                "commit": "1" * 40,
                "launch_claim_content_sha256": "5" * 64,
            },
            "capture_source": {
                "commit": "6" * 40,
                "clean": True,
                "producer_source_sha256": "8" * 64,
                "attestor_source_sha256": "9" * 64,
            },
        },
    }
    capture_claim = {
        "schema_version": 1,
        "artifact_kind": MODULE.CAPTURE_CLAIM_KIND,
        "producer_source_sha256": "8" * 64,
        "runtime_hard_contract_sha256": "3" * 64,
        "target_count": count,
        "motion_clips": receipt["motion_clips"],
        "joint_names": joint_names,
        "exclusive_create": True,
    }
    claim_path = tmp_path / MODULE.CAPTURE_CLAIM_NAME
    claim_path.write_text(
        json.dumps(capture_claim, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_claim_sha256"] = sha256_file(claim_path)
    capture_result = {
        "schema_version": 2,
        "artifact_kind": MODULE.CAPTURE_RESULT_KIND,
        "capture_contract": dict(CAPTURE_CONTRACT),
        "evidence": {
            "producer_source_sha256": "8" * 64,
            "runtime_hard_contract_sha256": "3" * 64,
            "exclusive_claim_sha256": sha256_file(claim_path),
            "exclusive_claim_relative_path": MODULE.CAPTURE_CLAIM_NAME,
            "no_clobber": True,
        },
        "motion_clips": receipt["motion_clips"],
        "states": {
            key: value for key, value in receipt["states"].items() if key != "velocity_limits"
        },
    }
    capture_path = tmp_path / MODULE.CAPTURE_RESULT_NAME
    capture_path.write_text(
        json.dumps(capture_result, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_result_sha256"] = sha256_file(capture_path)
    receipt_path = tmp_path / "teacher_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, motion_paths, joint_names, receipt


def _rewrite_receipt_and_capture(receipt_path: Path, receipt: dict) -> None:
    claim_path = receipt_path.parent / receipt["attestation"]["capture_claim_relative_path"]
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["motion_clips"] = receipt["motion_clips"]
    claim["joint_names"] = receipt["states"]["joint_names"]
    claim["target_count"] = receipt["states"]["count"]
    claim_path.write_text(
        json.dumps(claim, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_claim_sha256"] = sha256_file(claim_path)
    capture_path = receipt_path.parent / receipt["attestation"]["capture_result_relative_path"]
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["evidence"]["exclusive_claim_sha256"] = sha256_file(claim_path)
    capture["motion_clips"] = receipt["motion_clips"]
    capture["states"] = {
        key: value for key, value in receipt["states"].items() if key != "velocity_limits"
    }
    capture_path.write_text(
        json.dumps(capture, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["attestation"]["capture_result_sha256"] = sha256_file(capture_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


def _load(receipt_path, motion_paths, joint_names, *, min_fill=4, buffer_size=8):
    return load_post_swing_teacher_states(
        receipt_path,
        sha256_file(receipt_path),
        expected_motion_sha256=[sha256_file(path) for path in motion_paths],
        expected_joint_names=joint_names,
        expected_joint_velocity_limits=[5.0] * len(joint_names),
        expected_root_linear_velocity_limit_mps=2.0,
        expected_root_angular_velocity_limit_radps=4.0,
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
            expected_joint_velocity_limits=[5.0] * len(joints),
            expected_root_linear_velocity_limit_mps=2.0,
            expected_root_angular_velocity_limit_radps=4.0,
            min_fill=4,
            buffer_size=8,
        )
    with pytest.raises(PostSwingTeacherError, match="runtime motion bytes"):
        load_post_swing_teacher_states(
            receipt_path,
            sha256_file(receipt_path),
            expected_motion_sha256=["f" * 64, sha256_file(motions[1])],
            expected_joint_names=joints,
            expected_joint_velocity_limits=[5.0] * len(joints),
            expected_root_linear_velocity_limit_mps=2.0,
            expected_root_angular_velocity_limit_radps=4.0,
            min_fill=4,
            buffer_size=8,
        )
    with pytest.raises(PostSwingTeacherError, match="joint_names/order"):
        _load(receipt_path, motions, list(reversed(joints)))


def test_receipt_rejects_non_natural_wrap_and_underfilled_payload(tmp_path):
    receipt_path, motions, joints, receipt = _write_fixture(tmp_path)
    receipt["capture_contract"]["event"] = "arbitrary_timeout"
    _rewrite_receipt_and_capture(receipt_path, receipt)
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
    _rewrite_receipt_and_capture(receipt_path, receipt)
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
    _rewrite_receipt_and_capture(receipt_path, receipt)
    with pytest.raises(PostSwingTeacherError, match="symlink"):
        _load(receipt_path, motions, joints)


def test_receipt_and_npz_are_parsed_from_the_single_hashed_byte_snapshot(tmp_path, monkeypatch):
    receipt_path, motions, joints, _ = _write_fixture(tmp_path)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("receipt must not be reopened through Path.read_text")
    ))
    original = np.load
    seen = []

    def checked(source, *args, **kwargs):
        seen.append(source)
        assert isinstance(source, io.BytesIO)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked)
    loaded = _load(receipt_path, motions, joints)
    assert loaded.root_state_origin_relative.shape == (4, 13)
    assert seen and all(isinstance(source, io.BytesIO) for source in seen)


def test_duplicate_npz_zip_key_and_json_type_coercions_are_rejected(tmp_path):
    receipt_path, motions, joints, receipt = _write_fixture(tmp_path)
    payload = tmp_path / receipt["states"]["relative_path"]
    root = np.zeros((4, 13), dtype=np.float32)
    root[:, 3] = 1.0

    def npy(value):
        out = io.BytesIO()
        np.save(out, value, allow_pickle=False)
        return out.getvalue()

    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("root_state_origin_relative.npy", npy(root))
        archive.writestr("joint_pos.npy", npy(np.zeros((4, 3), dtype=np.float32)))
        archive.writestr("joint_vel.npy", npy(np.zeros((4, 3), dtype=np.float32)))
        archive.writestr("joint_vel.npy", npy(np.ones((4, 3), dtype=np.float32)))
    receipt["states"]["sha256"] = sha256_file(payload)
    _rewrite_receipt_and_capture(receipt_path, receipt)
    with pytest.raises(PostSwingTeacherError, match="duplicate NPZ ZIP keys"):
        _load(receipt_path, motions, joints)

    typed = tmp_path / "typed"
    typed.mkdir()
    receipt_path, motions, joints, receipt = _write_fixture(typed)
    receipt["schema_version"] = True
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PostSwingTeacherError, match="unsupported.*schema"):
        _load(receipt_path, motions, joints)


def test_velocity_limits_bind_runtime_and_reject_unsafe_states(tmp_path):
    receipt_path, motions, joints, receipt = _write_fixture(tmp_path)
    with pytest.raises(PostSwingTeacherError, match="root velocity limits differ"):
        load_post_swing_teacher_states(
            receipt_path,
            sha256_file(receipt_path),
            expected_motion_sha256=[sha256_file(path) for path in motions],
            expected_joint_names=joints,
            expected_joint_velocity_limits=[5.0] * len(joints),
            expected_root_linear_velocity_limit_mps=1.0,
            expected_root_angular_velocity_limit_radps=4.0,
            min_fill=4,
            buffer_size=8,
        )

    payload = tmp_path / receipt["states"]["relative_path"]
    root = np.zeros((4, 13), dtype=np.float32)
    root[:, 3] = 1.0
    root[0, 7] = 2.1
    np.savez(
        payload,
        root_state_origin_relative=root,
        joint_pos=np.zeros((4, 3), dtype=np.float32),
        joint_vel=np.zeros((4, 3), dtype=np.float32),
    )
    receipt["states"]["sha256"] = sha256_file(payload)
    _rewrite_receipt_and_capture(receipt_path, receipt)
    with pytest.raises(PostSwingTeacherError, match="linear velocity exceeds"):
        _load(receipt_path, motions, joints)


def test_red_team_array_forgery_seam_is_removed_and_legacy_label_is_rejected(tmp_path):
    # Exact b886256 repro: an external caller could import a module-global object, feed arbitrary
    # finite arrays to a public writer, then self-report the expected callback string.  Neither
    # signing seam exists now; source-owned MotionCommand capture accepts only env ids/live tensors.
    assert not hasattr(MODULE, "NaturalWrapCaptureWriter")
    assert not hasattr(MODULE, "_NATURAL_WRAP_CAPABILITY")
    commands_source = MODULE_PATH.with_name("commands.py").read_text(encoding="utf-8")
    assert "NaturalWrapCaptureWriter" not in commands_source
    assert "_append_from_natural_wrap" not in commands_source
    assert "_NATURAL_WRAP_CAPABILITY" not in commands_source

    receipt_path, motions, joints, receipt = _write_fixture(tmp_path)
    result_path = tmp_path / MODULE.CAPTURE_RESULT_NAME
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema_version"] = 1
    result["producer"] = {
        "callback_method": "MotionCommand._capture_post_swing_states",
        "writer_source_sha256": "7" * 64,
        "callback_source_sha256": "8" * 64,
        "runtime_hard_contract_sha256": "3" * 64,
        "no_clobber": True,
    }
    result["callback_batches"] = 1
    del result["evidence"]
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    receipt["attestation"]["capture_result_sha256"] = sha256_file(result_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PostSwingTeacherError, match="keys differ"):
        _load(receipt_path, motions, joints)


def test_default_off_training_contract_extension_is_byte_equivalent():
    historical = {"schema_version": 3, "motion_post_swing_start_prob": 0.0}
    before = json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()
    replay = {
        "teacher_receipt": None,
        "teacher_distribution": "immutable",
        "require_ready_at_init": False,
        "fail_fast_first_reset": False,
        "first_reset_acceptance": {
            "min_adopted_count": 1,
            "min_adopted_fraction": 0.0,
            "selection_probability_abs_tolerance": 1.0,
            "require_state_readback": False,
        },
    }
    after_value = {**historical, **MODULE.training_contract_extension(replay)}
    after = json.dumps(after_value, sort_keys=True, separators=(",", ":")).encode()
    assert after == before
