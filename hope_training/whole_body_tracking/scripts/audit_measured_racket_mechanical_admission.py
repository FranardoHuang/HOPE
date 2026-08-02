#!/usr/bin/env python3
"""Fail-closed mechanical-admission audit for measured-racket motion banks.

This audit deliberately answers a narrower question than racket retarget/FK:
"can the stored joint trajectory be admitted mechanically?"  It checks every
action and every sample against the exact URDF position and velocity limits,
and computes a second-order finite-difference acceleration trace.  It does NOT
promote kinematic alignment into mechanical evidence.

The URDF contains independent effort and no-load velocity limits.  Those two
numbers do not define the actuator's torque-speed curve.  This script therefore
never treats their rectangle as an admissible operating envelope.  In addition,
the measured-racket NPZ has no inverse-dynamics joint-torque trajectory.  The
torque-speed tier consequently remains UNKNOWN and mechanical admission remains
false.  Missing authoritative acceleration limits likewise remain UNKNOWN.

The JSON report is suitable for selecting a simulation-only diagnostic motion:
it contains per-action denominators, exact reasons, worst joints/frames, and an
unconditional ``diagnostic_unauthorized=true`` marker.  It does not authorize
training, promotion, deployment, or hardware execution.

Usage::

    python scripts/audit_measured_racket_mechanical_admission.py \
      --bank assets/motions/chingmu73_measured_v4_20260803 \
      --output-json /tmp/chingmu73_mechanical.json

Exit codes: 0 only if mechanically admitted (currently impossible by design),
1 for an UNKNOWN-only fail-closed result, and 2 for any observed hard failure.
Dependencies: NumPy plus the Python standard library; no Isaac/MuJoCo/Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree

import numpy as np


ISAAC_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

EXPECTED_MEASURED_RACKET_SCHEMA = 4
POSITION_TOLERANCE_RAD = 1.0e-6
VELOCITY_TOLERANCE_RAD_S = 1.0e-6
EXPECTED_JOINT_ORDER_CONTRACT_ID = "a3-gmr-dof-pos-to-runtime-articulation-v1"


@dataclass(frozen=True)
class JointLimit:
    lower_rad: float
    upper_rad: float
    velocity_rad_s: float
    urdf_effort_nm: Optional[float]


class AuditInputError(ValueError):
    """The audit inputs are malformed or do not bind the claimed identity."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_urdf() -> Path:
    return _repo_root() / "agi/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_urdf_limits(path: Path) -> Dict[str, JointLimit]:
    """Load finite position/velocity limits without inventing torque-speed data."""
    root = ElementTree.parse(str(path)).getroot()
    result: Dict[str, JointLimit] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if not name or joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        velocity = limit.get("velocity")
        if lower is None or upper is None or velocity is None:
            continue
        effort = limit.get("effort")
        parsed = JointLimit(
            lower_rad=float(lower),
            upper_rad=float(upper),
            velocity_rad_s=float(velocity),
            urdf_effort_nm=None if effort is None else float(effort),
        )
        if not (
            np.isfinite(parsed.lower_rad)
            and np.isfinite(parsed.upper_rad)
            and np.isfinite(parsed.velocity_rad_s)
            and parsed.lower_rad < parsed.upper_rad
            and parsed.velocity_rad_s > 0.0
        ):
            raise AuditInputError("invalid URDF limits for %s" % name)
        result[name] = parsed
    return result


def _ordered_limits(
    limits: Mapping[str, JointLimit], joint_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise AuditInputError("URDF lacks joint limits: %s" % ", ".join(missing))
    lower = np.asarray([limits[name].lower_rad for name in joint_names], dtype=np.float64)
    upper = np.asarray([limits[name].upper_rad for name in joint_names], dtype=np.float64)
    velocity = np.asarray(
        [limits[name].velocity_rad_s for name in joint_names], dtype=np.float64
    )
    return lower, upper, velocity


def _scalar(array: np.ndarray, key: str):
    flat = np.asarray(array).reshape(-1)
    if flat.size != 1:
        raise AuditInputError("%s must contain exactly one value" % key)
    return flat[0].item()


def _uid_from_stem(stem: str) -> str:
    return stem[5:] if stem.startswith("hope_") else stem


def _worst_2d(values: np.ndarray) -> Tuple[int, int, float]:
    if values.size == 0:
        return -1, -1, 0.0
    flat = int(np.nanargmax(values))
    frame, joint = np.unravel_index(flat, values.shape)
    return int(frame), int(joint), float(values[frame, joint])


def _load_acceleration_limits(
    path: Optional[Path], joint_names: Sequence[str]
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    if path is None:
        return None, None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("limits_rad_s2"), dict):
        raise AuditInputError(
            "acceleration-limit JSON must contain a limits_rad_s2 object"
        )
    authority = raw.get("authority")
    if not isinstance(authority, str) or not authority.strip():
        raise AuditInputError("acceleration-limit JSON requires a non-empty authority")
    missing = [name for name in joint_names if name not in raw["limits_rad_s2"]]
    if missing:
        raise AuditInputError(
            "acceleration-limit JSON lacks joints: %s" % ", ".join(missing)
        )
    values = np.asarray(
        [raw["limits_rad_s2"][name] for name in joint_names], dtype=np.float64
    )
    if values.shape != (len(joint_names),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise AuditInputError("acceleration limits must be finite and strictly positive")
    identity = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "authority": authority,
    }
    return values, identity


def _empty_failed_action(path: Path, reason: str) -> dict:
    return {
        "uid": _uid_from_stem(path.stem),
        "file": path.name,
        "sha256": _sha256(path),
        "side": "UNKNOWN",
        "frames": 0,
        "fps": None,
        "denominators": {
            "joint_position_samples": 0,
            "stored_velocity_samples": 0,
            "finite_difference_velocity_samples": 0,
            "finite_difference_acceleration_samples": 0,
        },
        "checks": {"input_identity": {"status": "FAIL", "reason": reason}},
        "kinematic_limit_verdict": "FAIL",
        "mechanical_verdict": "FAIL",
        "mechanical_admitted": False,
        "diagnostic_unauthorized": True,
        "reasons": [reason],
    }


def audit_action(
    path: Path,
    limits: Mapping[str, JointLimit],
    joint_names: Sequence[str] = ISAAC_JOINT_NAMES,
    acceleration_limits_rad_s2: Optional[np.ndarray] = None,
) -> dict:
    """Audit one measured-racket action and return a JSON-serializable row."""
    try:
        with np.load(str(path), allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        q = np.asarray(arrays["joint_pos"], dtype=np.float64)
        dq_stored = np.asarray(arrays["joint_vel"], dtype=np.float64)
        fps = float(_scalar(arrays["fps"], "fps"))
        uid = str(_scalar(arrays["measured_racket_uid"], "measured_racket_uid"))
        schema = int(
            _scalar(arrays["measured_racket_schema_version"], "measured_racket_schema_version")
        )
        admitted = int(
            _scalar(arrays["measured_racket_retarget_admitted"], "measured_racket_retarget_admitted")
        )
        order_id = str(
            _scalar(
                arrays["measured_racket_joint_order_contract_id"],
                "measured_racket_joint_order_contract_id",
            )
        )
        if q.ndim != 2 or q.shape[1] != len(joint_names) or q.shape[0] < 3:
            raise AuditInputError(
                "joint_pos must be [T,%d] with T>=3" % len(joint_names)
            )
        if dq_stored.shape != q.shape:
            raise AuditInputError("joint_vel shape must equal joint_pos shape")
        if not (np.isfinite(q).all() and np.isfinite(dq_stored).all() and np.isfinite(fps)):
            raise AuditInputError("joint arrays and fps must be finite")
        if fps <= 0:
            raise AuditInputError("fps must be strictly positive")
        if schema != EXPECTED_MEASURED_RACKET_SCHEMA:
            raise AuditInputError(
                "measured_racket_schema_version=%d, expected %d"
                % (schema, EXPECTED_MEASURED_RACKET_SCHEMA)
            )
        if admitted != 1:
            raise AuditInputError("measured_racket_retarget_admitted is not 1")
        if len(joint_names) == len(ISAAC_JOINT_NAMES) and order_id != EXPECTED_JOINT_ORDER_CONTRACT_ID:
            raise AuditInputError("unexpected measured-racket joint-order contract")
    except Exception as exc:
        return _empty_failed_action(path, "input_identity_invalid:%s" % exc)

    lower, upper, velocity_limits = _ordered_limits(limits, joint_names)
    dt = 1.0 / fps
    dq_fd = np.diff(q, axis=0) / dt
    ddq_fd = np.diff(q, n=2, axis=0) / (dt * dt)

    lower_margin = q - lower[None, :]
    upper_margin = upper[None, :] - q
    signed_margin = np.minimum(lower_margin, upper_margin)
    range_rad = upper - lower
    normalized_margin = signed_margin / range_rad[None, :]
    position_violation = signed_margin < -POSITION_TOLERANCE_RAD
    min_flat = int(np.argmin(signed_margin))
    min_frame, min_joint = np.unravel_index(min_flat, signed_margin.shape)

    stored_ratio = np.abs(dq_stored) / velocity_limits[None, :]
    fd_ratio = np.abs(dq_fd) / velocity_limits[None, :]
    stored_over = np.abs(dq_stored) > velocity_limits[None, :] + VELOCITY_TOLERANCE_RAD_S
    fd_over = np.abs(dq_fd) > velocity_limits[None, :] + VELOCITY_TOLERANCE_RAD_S
    stored_frame, stored_joint, stored_peak_ratio = _worst_2d(stored_ratio)
    fd_frame, fd_joint, fd_peak_ratio = _worst_2d(fd_ratio)
    acc_abs = np.abs(ddq_fd)
    acc_frame, acc_joint, acc_peak = _worst_2d(acc_abs)

    reasons: List[str] = []
    position_status = "PASS"
    if position_violation.any():
        position_status = "FAIL"
        reasons.append("joint_position_limit_violation")
    velocity_status = "PASS"
    if stored_over.any():
        velocity_status = "FAIL"
        reasons.append("stored_joint_velocity_limit_violation")
    if fd_over.any():
        velocity_status = "FAIL"
        reasons.append("finite_difference_joint_velocity_limit_violation")

    acceleration_check: dict
    if acceleration_limits_rad_s2 is None:
        acceleration_check = {
            "status": "UNKNOWN",
            "reason": "authoritative_joint_acceleration_limits_unavailable",
            "peak_abs_rad_s2": acc_peak,
            "worst_joint": joint_names[acc_joint],
            "worst_frame": acc_frame + 1,
            "violating_samples": None,
        }
        reasons.append("authoritative_joint_acceleration_limits_unavailable")
    else:
        acceleration_limits_rad_s2 = np.asarray(acceleration_limits_rad_s2, dtype=np.float64)
        if acceleration_limits_rad_s2.shape != (len(joint_names),):
            raise AuditInputError("acceleration-limit vector shape mismatch")
        acc_ratio = acc_abs / acceleration_limits_rad_s2[None, :]
        acc_over = acc_abs > acceleration_limits_rad_s2[None, :] + 1.0e-6
        ratio_frame, ratio_joint, ratio_peak = _worst_2d(acc_ratio)
        acceleration_check = {
            "status": "FAIL" if acc_over.any() else "PASS",
            "limit_source": "explicit_authoritative_file",
            "peak_ratio": ratio_peak,
            "peak_abs_rad_s2": float(acc_abs[ratio_frame, ratio_joint]),
            "worst_joint": joint_names[ratio_joint],
            "worst_frame": ratio_frame + 1,
            "violating_samples": int(acc_over.sum()),
        }
        if acc_over.any():
            reasons.append("finite_difference_joint_acceleration_limit_violation")

    # An URDF effort scalar and a no-load velocity scalar are not a torque-speed
    # curve.  Also, no inverse-dynamics torque trace is stored in this bank.
    torque_speed_check = {
        "status": "UNKNOWN",
        "reason": "authoritative_torque_speed_curves_and_joint_torque_trajectory_unavailable",
        "urdf_effort_velocity_rectangle_accepted": False,
    }
    reasons.append("authoritative_torque_speed_curves_unavailable")
    reasons.append("per_frame_inverse_dynamics_joint_torque_unavailable")

    hard_failure = (
        position_status == "FAIL"
        or velocity_status == "FAIL"
        or acceleration_check["status"] == "FAIL"
    )
    kinematic_verdict = "FAIL" if hard_failure else "PASS"
    mechanical_verdict = "FAIL" if hard_failure else "UNKNOWN"
    uid_from_filename = _uid_from_stem(path.stem)
    if uid != uid_from_filename:
        reasons.insert(0, "uid_filename_mismatch")
        kinematic_verdict = "FAIL"
        mechanical_verdict = "FAIL"

    return {
        "uid": uid,
        "file": path.name,
        "sha256": _sha256(path),
        "side": "FH" if uid.endswith("_FH") else "BH" if uid.endswith("_BH") else "UNKNOWN",
        "frames": int(q.shape[0]),
        "fps": fps,
        "denominators": {
            "joint_position_samples": int(q.size),
            "stored_velocity_samples": int(dq_stored.size),
            "finite_difference_velocity_samples": int(dq_fd.size),
            "finite_difference_acceleration_samples": int(ddq_fd.size),
        },
        "checks": {
            "input_identity": {
                "status": "FAIL" if uid != uid_from_filename else "PASS",
                "measured_racket_schema_version": schema,
                "retarget_admitted": bool(admitted),
                "joint_order_contract_id": order_id,
            },
            "joint_position_margin": {
                "status": position_status,
                "minimum_signed_margin_rad": float(signed_margin[min_frame, min_joint]),
                "minimum_normalized_margin_fraction": float(np.min(normalized_margin)),
                "worst_joint": joint_names[int(min_joint)],
                "worst_frame": int(min_frame),
                "violating_samples": int(position_violation.sum()),
            },
            "joint_velocity": {
                "status": velocity_status,
                "stored_peak_ratio": stored_peak_ratio,
                "stored_worst_joint": joint_names[stored_joint],
                "stored_worst_frame": stored_frame,
                "stored_violating_samples": int(stored_over.sum()),
                "finite_difference_peak_ratio": fd_peak_ratio,
                "finite_difference_worst_joint": joint_names[fd_joint],
                "finite_difference_worst_interval_start": fd_frame,
                "finite_difference_violating_samples": int(fd_over.sum()),
            },
            "finite_difference_joint_acceleration": acceleration_check,
            "torque_speed": torque_speed_check,
        },
        "kinematic_limit_verdict": kinematic_verdict,
        "mechanical_verdict": mechanical_verdict,
        "mechanical_admitted": False,
        "diagnostic_unauthorized": True,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _load_bank_paths(bank: Path) -> Tuple[List[Path], Optional[dict], List[str]]:
    if bank.is_file() and bank.suffix == ".npz":
        return [bank.resolve()], None, []
    if not bank.is_dir():
        raise AuditInputError("--bank must be an NPZ or directory")

    receipt_path = bank / "BANK_IMPORT_RECEIPT.json"
    identity_reasons: List[str] = []
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        actions = receipt.get("actions")
        if not isinstance(actions, list):
            raise AuditInputError("BANK_IMPORT_RECEIPT actions must be a list")
        paths = []
        for row in actions:
            if not isinstance(row, dict) or not isinstance(row.get("file"), str):
                raise AuditInputError("invalid action row in BANK_IMPORT_RECEIPT")
            path = (bank / row["file"]).resolve()
            if not path.is_file():
                raise AuditInputError("receipt action file missing: %s" % path)
            claimed_sha = row.get("sha256")
            if not isinstance(claimed_sha, str) or _sha256(path) != claimed_sha:
                raise AuditInputError("receipt action SHA mismatch: %s" % path)
            paths.append(path)
        extras = sorted(
            {path.resolve() for path in bank.glob("hope_*.npz")} - set(paths)
        )
        if extras:
            identity_reasons.append("unreceipted_npz_files_present")
        return paths, receipt, identity_reasons

    paths = sorted(path.resolve() for path in bank.glob("hope_*.npz"))
    if not paths:
        raise AuditInputError("bank contains no hope_*.npz files")
    identity_reasons.append("bank_import_receipt_unavailable")
    return paths, None, identity_reasons


def audit_bank(
    bank: Path,
    urdf: Path,
    joint_names: Sequence[str] = ISAAC_JOINT_NAMES,
    acceleration_limits_path: Optional[Path] = None,
) -> dict:
    paths, receipt, bank_identity_reasons = _load_bank_paths(bank.resolve())
    limits = parse_urdf_limits(urdf.resolve())
    acceleration_limits, acceleration_identity = _load_acceleration_limits(
        acceleration_limits_path, joint_names
    )
    actions = [
        audit_action(
            path,
            limits,
            joint_names=joint_names,
            acceleration_limits_rad_s2=acceleration_limits,
        )
        for path in paths
    ]

    status_counts = {
        status: sum(row["mechanical_verdict"] == status for row in actions)
        for status in ("PASS", "UNKNOWN", "FAIL")
    }
    per_side = {}
    for side in ("FH", "BH", "UNKNOWN"):
        rows = [row for row in actions if row["side"] == side]
        if rows:
            per_side[side] = {
                "actions": len(rows),
                "kinematic_limit_pass": sum(
                    row["kinematic_limit_verdict"] == "PASS" for row in rows
                ),
                "mechanical_unknown": sum(
                    row["mechanical_verdict"] == "UNKNOWN" for row in rows
                ),
                "mechanical_fail": sum(
                    row["mechanical_verdict"] == "FAIL" for row in rows
                ),
            }

    any_hard_failure = bool(status_counts["FAIL"] or bank_identity_reasons)
    overall_verdict = "FAIL" if any_hard_failure else "UNKNOWN"
    report = {
        "kind": "measured_racket_mechanical_admission_audit_v1",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_unauthorized": True,
        "authorization": {
            "mechanical_admission": False,
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "overall_verdict": overall_verdict,
        "exit_code": 2 if any_hard_failure else 1,
        "sources": {
            "bank": str(bank.resolve()),
            "bank_import_receipt": None
            if receipt is None
            else {
                "path": str((bank.resolve() / "BANK_IMPORT_RECEIPT.json")),
                "sha256": _sha256(bank.resolve() / "BANK_IMPORT_RECEIPT.json"),
                "kind": receipt.get("kind"),
            },
            "urdf": {"path": str(urdf.resolve()), "sha256": _sha256(urdf.resolve())},
            "acceleration_limits": acceleration_identity,
            "torque_speed_curves": None,
            "per_frame_inverse_dynamics_joint_torque": None,
        },
        "denominators": {
            "actions_expected": None
            if receipt is None
            else receipt.get("denominators", {}).get("materialized_npz"),
            "actions_audited": len(actions),
            "actions_with_kinematic_limit_pass": sum(
                row["kinematic_limit_verdict"] == "PASS" for row in actions
            ),
            "actions_mechanically_admitted": 0,
            "total_frames": sum(row["frames"] for row in actions),
            "joint_position_samples": sum(
                row["denominators"]["joint_position_samples"] for row in actions
            ),
            "stored_velocity_samples": sum(
                row["denominators"]["stored_velocity_samples"] for row in actions
            ),
            "finite_difference_velocity_samples": sum(
                row["denominators"]["finite_difference_velocity_samples"] for row in actions
            ),
            "finite_difference_acceleration_samples": sum(
                row["denominators"]["finite_difference_acceleration_samples"]
                for row in actions
            ),
        },
        "aggregate": {
            "mechanical_verdict_counts": status_counts,
            "per_side": per_side,
            "bank_identity_reasons": bank_identity_reasons,
            "unclosed_authority": [
                "authoritative_joint_acceleration_limits"
                if acceleration_limits is None
                else None,
                "authoritative_joint_torque_speed_curves",
                "per_frame_inverse_dynamics_joint_torque",
            ],
            "urdf_effort_velocity_rectangle_accepted": False,
        },
        "actions": actions,
    }
    report["aggregate"]["unclosed_authority"] = [
        item for item in report["aggregate"]["unclosed_authority"] if item is not None
    ]
    return report


def _parse_joint_names(value: Optional[str]) -> Tuple[str, ...]:
    if value is None:
        return ISAAC_JOINT_NAMES
    path = Path(value)
    if path.is_file():
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or len(set(names)) != len(names):
        raise AuditInputError("joint names must be a non-empty unique sequence")
    return tuple(names)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bank", type=Path, required=True, help="measured-racket bank directory or one NPZ")
    parser.add_argument("--urdf", type=Path, default=_default_urdf(), help="exact A3 URDF")
    parser.add_argument(
        "--joint-names",
        default=None,
        help="testing/alternate-robot override: comma list or one-name-per-line file",
    )
    parser.add_argument(
        "--acceleration-limits",
        type=Path,
        default=None,
        help="optional authoritative JSON with authority and limits_rad_s2; torque-speed still unclosed",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = audit_bank(
            args.bank,
            args.urdf,
            joint_names=_parse_joint_names(args.joint_names),
            acceleration_limits_path=args.acceleration_limits,
        )
    except Exception as exc:
        report = {
            "kind": "measured_racket_mechanical_admission_audit_v1",
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostic_unauthorized": True,
            "authorization": {
                "mechanical_admission": False,
                "training": False,
                "promotion": False,
                "deployment": False,
                "hardware": False,
            },
            "overall_verdict": "FAIL",
            "exit_code": 2,
            "fatal_error": str(exc),
            "denominators": {"actions_audited": 0, "actions_mechanically_admitted": 0},
            "actions": [],
        }

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")
    if not args.quiet:
        print(payload)
    print(
        "[measured-racket-mechanical] verdict=%s actions=%s/%s admitted=0 diagnostic_unauthorized=true"
        % (
            report.get("overall_verdict"),
            report.get("denominators", {}).get("actions_audited", 0),
            report.get("denominators", {}).get("actions_expected"),
        ),
        file=sys.stderr,
    )
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())
