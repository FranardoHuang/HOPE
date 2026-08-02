"""CPU-only tests for the measured-racket mechanical admission audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_measured_racket_mechanical_admission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_measured_racket_mechanical_admission", SCRIPT
)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

JOINTS = ("j0_joint", "j1_joint")
FPS = 10
TOY_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <joint name="j0_joint" type="revolute">
    <limit lower="-1" upper="1" velocity="100" effort="10"/>
  </joint>
  <joint name="j1_joint" type="revolute">
    <limit lower="-2" upper="2" velocity="100" effort="20"/>
  </joint>
</robot>
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def urdf(tmp_path: Path) -> Path:
    path = tmp_path / "toy.urdf"
    path.write_text(TOY_URDF)
    return path


def _write_clip(
    path: Path,
    q: np.ndarray,
    dq: np.ndarray,
    uid: str = "Take_001_unit00_BH",
) -> Path:
    np.savez(
        path,
        fps=np.asarray([FPS], dtype=np.int64),
        joint_pos=np.asarray(q, dtype=np.float32),
        joint_vel=np.asarray(dq, dtype=np.float32),
        measured_racket_uid=np.asarray(uid),
        measured_racket_schema_version=np.asarray([4], dtype=np.int64),
        measured_racket_retarget_admitted=np.asarray([1], dtype=np.int64),
        measured_racket_joint_order_contract_id=np.asarray("toy-contract"),
    )
    return path


def _clean_clip(path: Path, uid: str = "Take_001_unit00_BH") -> Path:
    q = np.zeros((6, 2), dtype=np.float64)
    q[:, 0] = np.arange(6) * 0.001
    dq = np.zeros_like(q)
    dq[:, 0] = 0.01
    return _write_clip(path, q, dq, uid=uid)


def test_clean_kinematics_remain_mechanically_unknown(urdf: Path, tmp_path: Path):
    path = _clean_clip(tmp_path / "hope_Take_001_unit00_BH.npz")
    row = audit.audit_action(
        path, audit.parse_urdf_limits(urdf), joint_names=JOINTS
    )

    assert row["kinematic_limit_verdict"] == "PASS"
    assert row["mechanical_verdict"] == "UNKNOWN"
    assert row["mechanical_admitted"] is False
    assert row["diagnostic_unauthorized"] is True
    assert row["denominators"] == {
        "joint_position_samples": 12,
        "stored_velocity_samples": 12,
        "finite_difference_velocity_samples": 10,
        "finite_difference_acceleration_samples": 8,
    }
    assert row["checks"]["joint_position_margin"]["status"] == "PASS"
    assert row["checks"]["joint_velocity"]["status"] == "PASS"
    assert row["checks"]["finite_difference_joint_acceleration"]["status"] == "UNKNOWN"
    assert row["checks"]["torque_speed"] == {
        "status": "UNKNOWN",
        "reason": "authoritative_torque_speed_curves_and_joint_torque_trajectory_unavailable",
        "urdf_effort_velocity_rectangle_accepted": False,
    }
    assert "authoritative_torque_speed_curves_unavailable" in row["reasons"]
    assert "per_frame_inverse_dynamics_joint_torque_unavailable" in row["reasons"]


def test_position_and_both_velocity_sources_fail_with_exact_reasons(
    urdf: Path, tmp_path: Path
):
    q = np.zeros((5, 2), dtype=np.float64)
    q[2, 0] = 1.2  # outside [-1, 1], and creates FD speed 12 rad/s
    dq = np.zeros_like(q)
    dq[3, 1] = 101.0
    path = _write_clip(tmp_path / "hope_Take_001_unit00_BH.npz", q, dq)

    row = audit.audit_action(
        path, audit.parse_urdf_limits(urdf), joint_names=JOINTS
    )

    assert row["mechanical_verdict"] == "FAIL"
    assert row["checks"]["joint_position_margin"]["violating_samples"] == 1
    assert row["checks"]["joint_velocity"]["stored_violating_samples"] == 1
    # 1.2 rad / 0.1 s = 12 rad/s, below the toy URDF's 100 rad/s limit.
    assert row["checks"]["joint_velocity"]["finite_difference_violating_samples"] == 0
    assert "joint_position_limit_violation" in row["reasons"]
    assert "stored_joint_velocity_limit_violation" in row["reasons"]

    # Make the same position jump legal in position but illegal in FD velocity.
    q[:, 0] = 0.0
    q[2, 0] = 0.9
    limits = dict(audit.parse_urdf_limits(urdf))
    limits["j0_joint"] = audit.JointLimit(-1.0, 1.0, 5.0, 10.0)
    path = _write_clip(tmp_path / "hope_Take_001_unit00_BH.npz", q, np.zeros_like(q))
    row = audit.audit_action(path, limits, joint_names=JOINTS)
    assert row["checks"]["joint_velocity"]["finite_difference_violating_samples"] == 2
    assert "finite_difference_joint_velocity_limit_violation" in row["reasons"]


def test_explicit_acceleration_limit_is_checked_but_does_not_close_torque_speed(
    urdf: Path, tmp_path: Path
):
    # q second difference at the middle sample: 0.01 rad * 10^2 = 1 rad/s^2.
    q = np.asarray([[0, 0], [0, 0], [0.01, 0], [0.03, 0], [0.06, 0]], dtype=float)
    path = _write_clip(
        tmp_path / "hope_Take_001_unit00_BH.npz", q, np.zeros_like(q)
    )
    limits = audit.parse_urdf_limits(urdf)

    failed = audit.audit_action(
        path,
        limits,
        joint_names=JOINTS,
        acceleration_limits_rad_s2=np.asarray([0.5, 0.5]),
    )
    assert failed["checks"]["finite_difference_joint_acceleration"]["status"] == "FAIL"
    assert "finite_difference_joint_acceleration_limit_violation" in failed["reasons"]

    passed_acceleration = audit.audit_action(
        path,
        limits,
        joint_names=JOINTS,
        acceleration_limits_rad_s2=np.asarray([2.0, 2.0]),
    )
    assert passed_acceleration["checks"]["finite_difference_joint_acceleration"]["status"] == "PASS"
    assert passed_acceleration["mechanical_verdict"] == "UNKNOWN"
    assert passed_acceleration["mechanical_admitted"] is False


def test_bank_receipt_denominators_sha_and_cli_exit_unknown(
    urdf: Path, tmp_path: Path
):
    bank = tmp_path / "bank"
    bank.mkdir()
    first = _clean_clip(bank / "hope_Take_001_unit00_BH.npz")
    second = _clean_clip(
        bank / "hope_Take_002_unit00_FH.npz", uid="Take_002_unit00_FH"
    )
    receipt = {
        "kind": "test_measured_racket_bank",
        "denominators": {"materialized_npz": 2},
        "actions": [
            {"uid": "Take_001_unit00_BH", "file": first.name, "sha256": _sha256(first)},
            {"uid": "Take_002_unit00_FH", "file": second.name, "sha256": _sha256(second)},
        ],
    }
    (bank / "BANK_IMPORT_RECEIPT.json").write_text(json.dumps(receipt))

    report = audit.audit_bank(bank, urdf, joint_names=JOINTS)
    assert report["overall_verdict"] == "UNKNOWN"
    assert report["exit_code"] == 1
    assert report["diagnostic_unauthorized"] is True
    assert report["denominators"]["actions_expected"] == 2
    assert report["denominators"]["actions_audited"] == 2
    assert report["denominators"]["actions_mechanically_admitted"] == 0
    assert report["aggregate"]["per_side"]["BH"]["actions"] == 1
    assert report["aggregate"]["per_side"]["FH"]["actions"] == 1
    assert report["aggregate"]["urdf_effort_velocity_rectangle_accepted"] is False

    exit_code = audit.main(
        [
            "--bank",
            str(bank),
            "--urdf",
            str(urdf),
            "--joint-names",
            ",".join(JOINTS),
            "--quiet",
        ]
    )
    assert exit_code == 1


def test_receipt_sha_mismatch_fails_before_auditing(urdf: Path, tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    clip = _clean_clip(bank / "hope_Take_001_unit00_BH.npz")
    receipt = {
        "kind": "test_measured_racket_bank",
        "denominators": {"materialized_npz": 1},
        "actions": [{"uid": "Take_001_unit00_BH", "file": clip.name, "sha256": "0" * 64}],
    }
    (bank / "BANK_IMPORT_RECEIPT.json").write_text(json.dumps(receipt))

    assert audit.main(
        [
            "--bank",
            str(bank),
            "--urdf",
            str(urdf),
            "--joint-names",
            ",".join(JOINTS),
            "--quiet",
        ]
    ) == 2


def test_local_chingmu73_bank_is_auditable_without_promotion():
    repo = Path(__file__).resolve().parents[3]
    bank = repo / "assets/motions/chingmu73_measured_v4_20260803"
    if not bank.is_dir():
        pytest.skip("local measured-racket bank is an ignored/restored asset")
    urdf = repo / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"

    report = audit.audit_bank(bank, urdf)
    assert report["denominators"]["actions_expected"] == 73
    assert report["denominators"]["actions_audited"] == 73
    assert len(report["actions"]) == 73
    assert report["denominators"]["actions_mechanically_admitted"] == 0
    assert report["diagnostic_unauthorized"] is True
    assert all(row["diagnostic_unauthorized"] for row in report["actions"])
    assert all(
        row["denominators"]["finite_difference_acceleration_samples"]
        == max(row["frames"] - 2, 0) * 31
        for row in report["actions"]
    )
