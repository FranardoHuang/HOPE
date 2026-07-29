"""Contract tests for the grounded-upper qvel-only fast materializer.

The real exact-MuJoCo integration is intentionally run on a Pod.  These tests
cover the producer's fail-closed source contracts, the only allowed tensor
change, the bitwise racket invariant, and receipt-last/no-clobber publication.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_grounded_upper_motion.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "materialize_grounded_upper_motion", _SCRIPT
)
materializer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = materializer
_SPEC.loader.exec_module(materializer)


def _ready_fixture() -> dict[str, object]:
    return {
        "joint_pos": np.zeros(31, dtype=np.float64),
        "root_pos_w": np.asarray([0.1, -0.2, 0.9], dtype=np.float64),
        "root_quat_w": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }


def _motion_fixture(frames: int = 5) -> dict[str, np.ndarray]:
    q = np.zeros((frames, 31), dtype=np.float32)
    # One right-arm core coordinate moves and returns; endpoints stay ready.
    q[1:-1, materializer.RIGHT_ARM_JOINT_INDICES[0]] = np.asarray(
        [0.1, 0.2, 0.1], dtype=np.float32
    )[: frames - 2]
    qd = np.zeros_like(q)
    leg = np.asarray(materializer.LEG_JOINT_INDICES, dtype=np.int64)
    qd[1:-1, leg] = np.float32(0.25)
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    body_pos[:, 0] = np.asarray([0.1, -0.2, 0.9], dtype=np.float32)
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((frames, 32, 3), dtype=np.float32)
    body_ang = np.zeros((frames, 32, 3), dtype=np.float32)
    body_names = tuple(f"body_{index}" for index in range(32))
    return {
        "fps": np.asarray([50], dtype=np.int64),
        "joint_pos": q,
        "joint_vel": qd,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "kinematics_schema_version": np.asarray([2], dtype=np.int64),
        "body_pos_point": np.asarray("link_origin"),
        "body_lin_vel_point": np.asarray("center_of_mass"),
        "body_names": np.asarray(body_names),
    }


def test_qvel_repair_changes_only_leg_velocity_columns() -> None:
    motion = _motion_fixture()
    q_before = motion["joint_pos"].copy()
    qd_before = motion["joint_vel"].copy()
    solved = q_before[0].astype(np.float64)
    published = solved.copy()

    q_after, qd_after = materializer._repair_constant_grounded_leg_velocity(
        q_before,
        qd_before,
        solved,
        published,
    )

    leg = np.asarray(materializer.LEG_JOINT_INDICES, dtype=np.int64)
    nonleg = np.asarray(
        [index for index in range(31) if index not in set(leg.tolist())],
        dtype=np.int64,
    )
    assert np.array_equal(q_after, q_before)
    assert np.array_equal(qd_after[:, nonleg], qd_before[:, nonleg])
    assert np.count_nonzero(qd_after[:, leg]) == 0
    # Producer inputs remain immutable.
    assert np.array_equal(motion["joint_pos"], q_before)
    assert np.array_equal(motion["joint_vel"], qd_before)


@pytest.mark.parametrize("which", ["solved", "published"])
def test_qvel_repair_rejects_any_grounded_qpos_mismatch(which: str) -> None:
    motion = _motion_fixture()
    solved = motion["joint_pos"][0].astype(np.float64)
    published = solved.copy()
    target = solved if which == "solved" else published
    target[materializer.LEG_JOINT_INDICES[0]] = 0.125

    with pytest.raises(
        materializer.GroundedUpperMaterializationError,
        match="leg qpos",
    ):
        materializer._repair_constant_grounded_leg_velocity(
            motion["joint_pos"],
            motion["joint_vel"],
            solved,
            published,
        )


def test_upper_motion_validation_accepts_target_inconsistency_and_reports_it() -> None:
    motion = _motion_fixture()
    report = materializer._validate_motion(
        motion,
        ready_v1=_ready_fixture(),
        strike_frame=2,
    )
    assert report["frames"] == 5
    assert report["strike_frame"] == 2
    assert sum(
        row["nonzero_samples"] for row in report["leg_velocity_before"]
    ) == 12 * 3
    assert [row["index"] for row in report["leg_velocity_before"]] == list(
        materializer.LEG_JOINT_INDICES
    )
    assert [row["name"] for row in report["leg_velocity_before"]] == list(
        materializer.LEG_JOINT_NAMES
    )


def test_upper_motion_validation_rejects_moving_leg_qpos() -> None:
    motion = _motion_fixture()
    motion["joint_pos"][2, materializer.LEG_JOINT_INDICES[0]] = 0.01
    with pytest.raises(
        materializer.GroundedUpperMaterializationError,
        match="already-constant 12-leg qpos",
    ):
        materializer._validate_motion(
            motion,
            ready_v1=_ready_fixture(),
            strike_frame=2,
        )


def test_upper_motion_validation_rejects_nonstationary_root() -> None:
    motion = _motion_fixture()
    motion["body_pos_w"][2, 0, 0] += 0.01
    with pytest.raises(
        materializer.GroundedUpperMaterializationError,
        match="root pose must be bitwise constant",
    ):
        materializer._validate_motion(
            motion,
            ready_v1=_ready_fixture(),
            strike_frame=2,
        )


def _site_trace() -> materializer.SiteTrace:
    return materializer.SiteTrace(
        position_w=np.arange(15, dtype=np.float64).reshape(5, 3),
        rotation_w=np.arange(45, dtype=np.float64).reshape(5, 9),
        linear_velocity_w=np.arange(15, dtype=np.float64).reshape(5, 3) / 10.0,
        angular_velocity_w=np.arange(15, dtype=np.float64).reshape(5, 3) / 20.0,
    )


def test_site_proof_requires_all_four_channels_bitwise_equal() -> None:
    before = _site_trace()
    after = _site_trace()
    proof = materializer._assert_site_trace_bitwise_equal(before, after)
    assert set(proof) == {
        "position_w",
        "rotation_w",
        "linear_velocity_w",
        "angular_velocity_w",
    }
    assert all(row["bitwise_equal"] is True for row in proof.values())
    assert all(row["maximum_abs_delta"] == 0.0 for row in proof.values())

    changed = _site_trace()
    changed.linear_velocity_w[3, 1] = np.nextafter(
        changed.linear_velocity_w[3, 1], np.inf
    )
    with pytest.raises(
        materializer.GroundedUpperMaterializationError,
        match="linear_velocity_w is not bitwise preserved",
    ):
        materializer._assert_site_trace_bitwise_equal(before, changed)


def test_publish_is_no_clobber_receipt_last_and_unauthorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    original = materializer._exclusive_write_at

    def record(directory_fd: int, filename: str, payload: bytes) -> None:
        writes.append(filename)
        original(directory_fd, filename, payload)

    monkeypatch.setattr(materializer, "_exclusive_write_at", record)
    receipt = {
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "outputs": {"motion_sha256": "1" * 64},
    }
    output = tmp_path / "bundle"
    result = materializer._publish_bundle(
        output_directory=output,
        motion_filename="motion.npz",
        motion_payload=b"motion",
        schema_manifest_payload=b"manifest",
        schema_report_payload=b"report",
        receipt=receipt,
    )
    assert writes == [
        "motion.npz",
        materializer.SCHEMA2_MANIFEST_FILENAME,
        materializer.SCHEMA2_REPORT_FILENAME,
        materializer.RECEIPT_FILENAME,
    ]
    assert result.motion.read_bytes() == b"motion"
    loaded = json.loads(result.receipt.read_text(encoding="ascii"))
    assert loaded["authorization"] == receipt["authorization"]
    assert len(loaded["receipt_payload_sha256"]) == 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materializer._publish_bundle(
            output_directory=output,
            motion_filename="motion.npz",
            motion_payload=b"motion",
            schema_manifest_payload=b"manifest",
            schema_report_payload=b"report",
            receipt=receipt,
        )


def test_publish_rejects_any_authorization_true(tmp_path: Path) -> None:
    receipt = {
        "authorization": {
            "training_authorized": True,
            "deployment_authorized": False,
            "hardware_authorized": False,
        }
    }
    with pytest.raises(
        materializer.GroundedUpperMaterializationError,
        match="deny all authorization",
    ):
        materializer._publish_bundle(
            output_directory=tmp_path / "forbidden",
            motion_filename="motion.npz",
            motion_payload=b"motion",
            schema_manifest_payload=b"manifest",
            schema_report_payload=b"report",
            receipt=receipt,
        )


def test_cli_requires_every_identity_pin() -> None:
    parser = materializer._parser()
    required = {
        action.dest
        for action in parser._actions
        if getattr(action, "required", False)
    }
    assert required == {
        "input_motion",
        "expected_input_sha256",
        "canonical_ready_v1",
        "expected_canonical_ready_v1_sha256",
        "grounded_reference_candidate",
        "expected_grounded_reference_candidate_sha256",
        "grounded_reference_receipt",
        "expected_grounded_reference_receipt_sha256",
        "body_order",
        "expected_body_order_sha256",
        "strike_frame",
        "output_dir",
    }
