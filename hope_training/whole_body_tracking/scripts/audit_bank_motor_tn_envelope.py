#!/usr/bin/env python3
"""Test a measured-racket bank against the A3 motor torque-speed envelope.

Mechanical admission has been blocked on three reasons that all say the same thing -- the data
needed to judge a motion did not exist:

* ``authoritative_torque_speed_curves_unavailable``
* ``per_frame_inverse_dynamics_joint_torque_unavailable``
* ``authoritative_joint_acceleration_limits_unavailable``

The 2026-08-04 vendor motor package supplies the first. This tool supplies the second by running
MuJoCo inverse dynamics over the bank, and then tests the two together: at every frame, for every
joint, is the actual (speed, torque) pair inside that motor's envelope?

Two things this deliberately does NOT do. It does not compare the training config's independent
effort and velocity limits against the curve -- those are independent limits whose rectangle corner
necessarily sits outside a coupled envelope, so that comparison is meaningless. And it does not
judge the six parallel joints (waist roll/pitch and both ankle roll/pitch): they are 2x2 parallel
mechanisms whose joint-space torque maps to motor space through a pose-dependent Jacobian the
vendor package does not supply. Those are reported as UNKNOWN, never as PASS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ENVELOPE_MANIFEST = REPO_ROOT / "configs" / "a3_motor_tn_envelope_v1.json"
ENVELOPE_CSV = REPO_ROOT / "configs" / "a3_motor_tn" / "current_conservative_tn_envelope.csv"
JOINT_MAP_CSV = REPO_ROOT / "configs" / "a3_motor_tn" / "a3_joint_motor_mapping.csv"
RUNTIME_JOINT_ORDER = REPO_ROOT / "configs" / "a3_runtime_articulation_joint_order.txt"
# The bank stores joint_pos/joint_vel as a bare 31-wide array and names the ordering only by
# contract id, so the names come from that contract rather than from the file.
EXPECTED_JOINT_ORDER_CONTRACT_ID = "a3-gmr-dof-pos-to-runtime-articulation-v1"


class EnvelopeError(RuntimeError):
    """Raised when an input violates the recorded contract."""


def load_envelope() -> dict[str, dict[str, float]]:
    """Three control points per family: zero-speed plateau, plateau end, zero torque."""
    families: dict[str, list[tuple[float, float]]] = {}
    with ENVELOPE_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            families.setdefault(row["family"], []).append(
                (float(row["speed_rad_s"]), float(row["torque_nm"]))
            )
    out = {}
    for family, points in families.items():
        points.sort()
        if len(points) != 3:
            raise EnvelopeError(f"{family} does not have exactly three control points: {points}")
        plateau_torque = points[0][1]
        plateau_end = points[1][0]
        zero_torque = points[2][0]
        if not (points[1][1] == plateau_torque and points[2][1] == 0.0):
            raise EnvelopeError(f"{family} control points are not the documented plateau/derate shape")
        if not (0.0 < plateau_end < zero_torque):
            raise EnvelopeError(f"{family} breakpoints are not ordered: {plateau_end}, {zero_torque}")
        out[family] = {
            "plateau_torque_nm": plateau_torque,
            "plateau_end_rad_s": plateau_end,
            "zero_torque_rad_s": zero_torque,
        }
    return out


def torque_limit(envelope: dict[str, float], speed_rad_s: np.ndarray) -> np.ndarray:
    """Symmetric two-segment envelope, queried on |speed|."""
    v = np.abs(np.asarray(speed_rad_s, dtype=np.float64))
    t0 = envelope["plateau_torque_nm"]
    vb = envelope["plateau_end_rad_s"]
    vz = envelope["zero_torque_rad_s"]
    limit = np.full_like(v, t0)
    derate = (v > vb) & (v < vz)
    limit[derate] = t0 * (vz - v[derate]) / (vz - vb)
    limit[v >= vz] = 0.0
    return limit


def load_joint_map() -> dict[str, dict[str, str]]:
    with JOINT_MAP_CSV.open(encoding="utf-8") as handle:
        return {row["joint_name"]: row for row in csv.DictReader(handle)}


def inverse_dynamics(mjcf: Path, joint_names: list[str], qpos: np.ndarray, qvel: np.ndarray, fps: float):
    """Per-frame joint torque from MuJoCo inverse dynamics.

    Root state is not stored in the bank, so the floating base is held at the keyframe pose. That
    biases absolute torques -- it is why the caller must treat a PASS here as necessary, not
    sufficient -- but it does not affect which joints dominate, and it is the same assumption on
    every plant so plant-to-plant comparison stays fair.  Contacts are disabled: the bank stores no
    root state, so the feet would otherwise be driven through the floor and the contact forces
    would swamp the very quantity being measured.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EnvelopeError("this audit requires MuJoCo") from exc

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    # The MJCF carries a floor and the stand keyframe rests on it.  Injecting bank joint angles
    # frame by frame drives the feet through that floor, and the resulting contact forces land in
    # qfrc_inverse -- which is how a 6 N*m wrist ended up reading 41,934 N*m on the first attempt.
    # Motor sizing wants the inertial and gravity terms, so contacts are disabled outright rather
    # than hoped away.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    qadr = {}
    vadr = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise EnvelopeError(f"MJCF lacks joint {name!r}")
        qadr[name] = int(model.jnt_qposadr[jid])
        vadr[name] = int(model.jnt_dofadr[jid])

    frames = qpos.shape[0]
    dt = 1.0 / float(fps)
    qacc = np.zeros_like(qvel)
    if frames >= 3:
        qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2.0 * dt)
        qacc[0] = (qvel[1] - qvel[0]) / dt
        qacc[-1] = (qvel[-1] - qvel[-2]) / dt

    torque = np.zeros((frames, len(joint_names)), dtype=np.float64)
    for f in range(frames):
        for i, name in enumerate(joint_names):
            data.qpos[qadr[name]] = qpos[f, i]
            data.qvel[vadr[name]] = qvel[f, i]
        # Order matters and cost a wrong answer once: mj_forward RECOMPUTES qacc from the
        # dynamics, so the target acceleration has to be written after it and before mj_inverse,
        # or the audit silently measures the passive motion instead of the commanded one.
        mujoco.mj_forward(model, data)
        for i, name in enumerate(joint_names):
            data.qacc[vadr[name]] = qacc[f, i]
        mujoco.mj_inverse(model, data)
        for i, name in enumerate(joint_names):
            torque[f, i] = data.qfrc_inverse[vadr[name]]
    return torque, qacc


def audit_clip(path: Path, mjcf: Path, envelope, joint_map) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    contract = str(np.asarray(payload["measured_racket_joint_order_contract_id"]).ravel()[0])
    if contract != EXPECTED_JOINT_ORDER_CONTRACT_ID:
        raise EnvelopeError(
            f"{path.name} declares joint order contract {contract!r}, expected "
            f"{EXPECTED_JOINT_ORDER_CONTRACT_ID!r}"
        )
    names = [
        line.strip()
        for line in RUNTIME_JOINT_ORDER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(names) != int(np.asarray(payload["joint_pos"]).shape[1]):
        raise EnvelopeError(
            f"{path.name} has {np.asarray(payload['joint_pos']).shape[1]} joint columns but the "
            f"runtime order names {len(names)}"
        )
    qpos = np.asarray(payload["joint_pos"], dtype=np.float64)
    qvel = np.asarray(payload["joint_vel"], dtype=np.float64)
    fps = float(np.asarray(payload["fps"]).ravel()[0])

    torque, qacc = inverse_dynamics(mjcf, names, qpos, qvel, fps)

    per_joint = {}
    worst = {"joint": None, "ratio": 0.0, "frame": -1}
    unknown = []
    for i, name in enumerate(names):
        row = joint_map.get(name)
        if row is None:
            unknown.append({"joint": name, "reason": "not_in_vendor_joint_motor_map"})
            continue
        if row["topology"] != "serial":
            unknown.append(
                {"joint": name, "reason": "parallel_mechanism_needs_pose_dependent_jacobian",
                 "pair": row["parallel_pair"]}
            )
            continue
        limit = torque_limit(envelope[row["motor_family"]], qvel[:, i])
        magnitude = np.abs(torque[:, i])
        # A zero limit past the zero-torque speed cannot be expressed as a ratio; flag separately.
        past_zero = limit <= 0.0
        ratio = np.where(past_zero, np.inf, magnitude / np.maximum(limit, 1e-12))
        peak = int(np.argmax(ratio))
        per_joint[name] = {
            "motor_family": row["motor_family"],
            "max_ratio": float(ratio[peak]) if np.isfinite(ratio[peak]) else None,
            "max_ratio_frame": peak,
            "frames_over_envelope": int(np.sum(ratio > 1.0)),
            "frames_past_zero_torque_speed": int(np.sum(past_zero)),
            "peak_abs_torque_nm": float(magnitude.max()),
            "peak_abs_speed_rad_s": float(np.abs(qvel[:, i]).max()),
        }
        if np.isfinite(ratio[peak]) and ratio[peak] > worst["ratio"]:
            worst = {"joint": name, "ratio": float(ratio[peak]), "frame": peak}

    over = {k: v for k, v in per_joint.items() if v["frames_over_envelope"] > 0}
    return {
        "file": path.name,
        "frames": int(qpos.shape[0]),
        "fps": fps,
        "serial_joints_checked": len(per_joint),
        "unknown_joints": unknown,
        "joints_over_envelope": sorted(over),
        "worst": worst,
        "per_joint": per_joint,
        "verdict": "FAIL" if over else ("UNKNOWN" if unknown else "PASS"),
        "verdict_note": (
            "PASS here means every SERIAL joint stayed inside its motor envelope under a "
            "pinned-base inverse dynamics estimate; the parallel joints are not judged, so this is "
            "necessary but not sufficient for mechanical admission"
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="audit only the first N clips")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        envelope = load_envelope()
        joint_map = load_joint_map()
        clips = sorted(args.bank.glob("*.npz")) if args.bank.is_dir() else [args.bank]
        if args.limit:
            clips = clips[: args.limit]
        results = [audit_clip(p, args.mjcf, envelope, joint_map) for p in clips]
        counts: dict[str, int] = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        over_joints: dict[str, int] = {}
        for r in results:
            for j in r["joints_over_envelope"]:
                over_joints[j] = over_joints.get(j, 0) + 1
        report = {
            "kind": "a3_motor_tn_envelope_bank_audit_v1",
            "bank": str(args.bank),
            "mjcf": str(args.mjcf),
            "clips": len(results),
            "verdict_counts": counts,
            "clips_with_any_joint_over_envelope": sum(1 for r in results if r["joints_over_envelope"]),
            "joint_over_envelope_clip_counts": dict(sorted(over_joints.items(), key=lambda kv: -kv[1])),
            "parallel_joints_not_judged": sorted(
                {u["joint"] for r in results for u in r["unknown_joints"]}
            ),
            "actions": results,
        }
        if args.output_json:
            args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "actions"}, ensure_ascii=False))
        return 0
    except (EnvelopeError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
