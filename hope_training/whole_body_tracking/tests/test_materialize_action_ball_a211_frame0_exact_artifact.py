"""Host tests for the no-clobber A211 frame-0 artifact producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_action_ball_a211_frame0_exact_artifact.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_a211_frame0", SCRIPT)
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _motion(path: Path) -> dict[str, np.ndarray]:
    root_pos = np.asarray(
        [[[-0.125, 0.375, 0.8125], [1.0, 2.0, 3.0]],
         [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]],
        dtype=np.float32,
    )
    root_quat = np.asarray(
        [[[0.5, 0.5, -0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
         [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    joint_pos = np.arange(62, dtype=np.float32).reshape(2, 31) / np.float32(17.0)
    arrays = {
        "joint_pos": joint_pos,
        # Deliberately nonzero: the producer must not leak source velocities.
        "joint_vel": np.full((2, 31), 3.25, dtype=np.float32),
        "body_names": np.asarray(["pelvis_link", "torso_link"]),
        "body_pos_w": root_pos,
        "body_quat_w": root_quat,
        "body_lin_vel_w": np.full((2, 2, 3), 4.5, dtype=np.float32),
        "body_ang_vel_w": np.full((2, 2, 3), -2.75, dtype=np.float32),
        "kinematics_schema_version": np.asarray([2], dtype=np.int64),
        "body_pos_point": np.asarray("link_origin"),
        "body_lin_vel_point": np.asarray("center_of_mass"),
        "measured_racket_schema_version": np.asarray([4], dtype=np.int64),
    }
    np.savez(path, **arrays)
    return arrays


def _argv(root: Path, motion_sha: str, output: str) -> list[str]:
    timing = root / "timing.json"
    if not timing.exists():
        unsigned = {
            "schema_version": 5,
            "motion_sha256": motion_sha,
            "contact_time_step_s": 0.02,
            "pre_swing_wait_s": 0.7123799138976297,
        }
        document = {
            **unsigned,
            "canonical_sha256": materializer.canonical_sha256(unsigned),
        }
        timing.write_bytes(materializer.canonical_bytes(document) + b"\n")
    return [
        "--repo-root", str(root),
        "--action-id", "take_061_unit04_bh",
        "--motion-path", "motion.npz",
        "--expected-motion-sha256", motion_sha,
        "--timing-receipt-path", "timing.json",
        "--expected-timing-receipt-sha256", _sha(timing),
        "--output", output,
    ]


def test_exact_frame0_copy_zero_velocity_and_canonical_bindings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    motion = tmp_path / "motion.npz"
    arrays = _motion(motion)
    assert materializer.main(_argv(tmp_path, _sha(motion), "artifact.json")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "MATERIALIZED_POD_NOMINAL_HOLD_REQUIRED"
    assert result["diagnostic_unauthorized"] is True
    assert result["launch_authorized"] is False
    assert result["nominal_hold_receipt_created"] is False

    raw = (tmp_path / "artifact.json").read_bytes()
    artifact = json.loads(raw)
    assert raw == materializer.canonical_bytes(artifact) + b"\n"
    assert artifact["kind"] == materializer._L.FRAME0_EXACT_ARTIFACT_KIND
    assert artifact["source_kind"] == materializer._L.FRAME0_EXACT_SOURCE_KIND
    assert artifact["action_id"] == "take_061_unit04_bh"
    assert artifact["motion_sha256"] == _sha(motion)
    assert artifact["policy_dt_s"] == materializer._L.POLICY_DT_S
    assert artifact["schema_version"] == 2
    assert artifact["birth_horizon"]["required_policy_ticks"] == 62
    assert artifact["birth_horizon"]["pre_swing_wait_policy_ticks_ceil"] == 36
    assert "durability_policy_ticks_excluded" not in artifact["birth_horizon"]
    assert "durability_physics_substeps_excluded" not in artifact["birth_horizon"]
    assert artifact["timing_receipt"]["sha256"] == _sha(tmp_path / "timing.json")
    assert artifact["wait_schedule_canonical_sha256"] == (
        materializer._L.WAIT_SCHEDULE["canonical_sha256"]
    )
    assert artifact["diagnostic_unauthorized"] is True
    frame0 = artifact["frame0"]
    assert frame0["root_pos_w_m"] == arrays["body_pos_w"][0, 0].tolist()
    assert frame0["root_quat_wxyz"] == arrays["body_quat_w"][0, 0].tolist()
    assert frame0["joint_pos_rad"] == arrays["joint_pos"][0].tolist()
    assert len(frame0["joint_pos_rad"]) == 31
    assert frame0["root_lin_vel_w_mps"] == [0.0, 0.0, 0.0]
    assert frame0["root_ang_vel_w_radps"] == [0.0, 0.0, 0.0]
    assert frame0["joint_vel_radps"] == [0.0] * 31
    assert materializer.require_content_sha(artifact) == artifact["content_sha256"]


def test_payload_tamper_breaks_content_sha(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    _motion(motion)
    assert materializer.main(_argv(tmp_path, _sha(motion), "artifact.json")) == 0
    artifact = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))
    artifact["frame0"]["joint_pos_rad"][7] += 0.125
    with pytest.raises(materializer.MaterializationError, match="not reproducible"):
        materializer.require_content_sha(artifact)


def test_replay_is_byte_exact_and_binds_source_sha(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    _motion(motion)
    motion_sha = _sha(motion)
    assert materializer.main(_argv(tmp_path, motion_sha, "first.json")) == 0
    assert materializer.main(_argv(tmp_path, motion_sha, "second.json")) == 0
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert _sha(tmp_path / "first.json") == _sha(tmp_path / "second.json")

    assert materializer.main(_argv(tmp_path, "0" * 64, "bad.json")) == 2
    assert not (tmp_path / "bad.json").exists()


def test_no_clobber_preserves_existing_bytes(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    _motion(motion)
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"user-owned\n")
    assert materializer.main(_argv(tmp_path, _sha(motion), "artifact.json")) == 2
    assert destination.read_bytes() == b"user-owned\n"


def test_refuses_non_measured_or_malformed_frame0(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    arrays = _motion(motion)
    arrays["measured_racket_schema_version"] = np.asarray([3], dtype=np.int64)
    np.savez(motion, **arrays)
    assert materializer.main(_argv(tmp_path, _sha(motion), "bad-schema.json")) == 2
    assert not (tmp_path / "bad-schema.json").exists()

    arrays["measured_racket_schema_version"] = np.asarray([4], dtype=np.int64)
    arrays["joint_pos"] = arrays["joint_pos"][:, :-1]
    np.savez(motion, **arrays)
    assert materializer.main(_argv(tmp_path, _sha(motion), "bad-q.json")) == 2
    assert not (tmp_path / "bad-q.json").exists()
