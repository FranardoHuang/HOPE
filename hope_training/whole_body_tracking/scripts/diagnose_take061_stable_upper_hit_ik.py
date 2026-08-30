#!/usr/bin/env python3
"""Offline stable-support hit-window IK for the Take061 N=1 diagnostic.

This is deliberately not a motion-bank compiler or a training consumer.  It answers one
counterfactual: while the accepted dynamic-ready root and every non-waist/right-arm joint remain
fixed, can the existing A3 paddle site reproduce the measured hit-window position, signed face,
long axis and point velocity without leaving the *executable* qdes envelope?

The implementation reuses the canonical full-phase solver's plant geometry, joint set, paddle
axes, URDF velocity limits and SE(3) target semantics.  The output is a diagnostic JSON/NPZ pair;
no production path reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_SOLVER_PATH = Path(__file__).with_name("solve_chingmu_canonical_racket_full_phase.py")
_SOLVER_SPEC = importlib.util.spec_from_file_location(
    "_hope_canonical_racket_full_phase_for_stable_hit", _SOLVER_PATH
)
if _SOLVER_SPEC is None or _SOLVER_SPEC.loader is None:
    raise RuntimeError("cannot load canonical full-phase solver")
_SOLVER = importlib.util.module_from_spec(_SOLVER_SPEC)
_SOLVER_SPEC.loader.exec_module(_SOLVER)
OPTIMIZED_JOINTS = _SOLVER.OPTIMIZED_JOINTS
ROBOT_BUTT_TO_BLADE_AXIS_LOCAL = _SOLVER.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
SITE_NAME = _SOLVER.SITE_NAME
load_urdf_motion_limits = _SOLVER.load_urdf_motion_limits


STRICT_ENVELOPE_EPS_RAD = 1.0e-5


class DiagnosticError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_executable_bounds(
    lower: np.ndarray, upper: np.ndarray, *, epsilon_rad: float = STRICT_ENVELOPE_EPS_RAD
) -> tuple[np.ndarray, np.ndarray]:
    """Return an open-envelope approximation; never solve exactly on a projector edge."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (
        lower.ndim != 1
        or lower.shape != upper.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not math.isfinite(epsilon_rad)
        or epsilon_rad <= 0.0
        or np.any(upper - lower <= 2.0 * epsilon_rad)
    ):
        raise DiagnosticError("invalid executable qdes envelope")
    return lower + epsilon_rad, upper - epsilon_rad


def minimum_jerk_duration_s(
    start: np.ndarray, goal: np.ndarray, velocity_limit_rad_s: np.ndarray
) -> tuple[float, np.ndarray]:
    """Minimum duration for a quintic smoothstep under per-joint velocity limits.

    The derivative of ``10u^3-15u^4+6u^5`` peaks at 1.875, hence this is stricter than the
    usual ``abs(delta)/vmax`` estimate and directly matches the diagnostic bridge interpolation.
    """

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    velocity = np.asarray(velocity_limit_rad_s, dtype=np.float64)
    if start.shape != goal.shape or start.shape != velocity.shape or np.any(velocity <= 0.0):
        raise DiagnosticError("invalid minimum-jerk duration inputs")
    per_joint = 1.875 * np.abs(goal - start) / velocity
    return float(np.max(per_joint)), per_joint


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _load_ready(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    names = tuple(document["runtime_plant"]["joint_names"])
    ready = np.asarray(document["physical_ready"]["joint_pos_rad"], dtype=np.float64)
    root_pos = np.asarray(document["physical_ready"]["root_pos_w_m"], dtype=np.float64)
    root_wxyz = np.asarray(document["physical_ready"]["root_quat_wxyz"], dtype=np.float64)
    lower = np.asarray(document["runtime_plant"]["executed_qdes_lower_rad"], dtype=np.float64)
    upper = np.asarray(document["runtime_plant"]["executed_qdes_upper_rad"], dtype=np.float64)
    if len(names) != 31 or any(value.shape != (31,) for value in (ready, lower, upper)):
        raise DiagnosticError("dynamic-ready runtime joint contract is not 31-DoF")
    if root_pos.shape != (3,) or root_wxyz.shape != (4,):
        raise DiagnosticError("dynamic-ready root contract is malformed")
    strict_lower, strict_upper = strict_executable_bounds(lower, upper)
    outside = np.flatnonzero((ready < strict_lower) | (ready > strict_upper))
    if len(outside):
        detail = {names[i]: float(ready[i]) for i in outside}
        raise DiagnosticError(f"stable ready itself is outside strict executable bounds: {detail}")
    return {
        "document": document,
        "names": names,
        "ready": ready,
        "root_pos": root_pos,
        "root_wxyz": root_wxyz,
        "lower": strict_lower,
        "upper": strict_upper,
    }


def solve(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mujoco
        from scipy.optimize import minimize
    except ImportError as exc:
        raise DiagnosticError("MuJoCo Python bindings and SciPy are required") from exc

    ready = _load_ready(args.dynamic_ready)
    names = ready["names"]
    name_to_runtime = {name: i for i, name in enumerate(names)}
    optimized_runtime = np.asarray([name_to_runtime[name] for name in OPTIMIZED_JOINTS])
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    if any(value < 0 for value in joint_ids):
        raise DiagnosticError("model does not contain the dynamic-ready runtime joint order")
    qpos_adrs = np.asarray([model.jnt_qposadr[value] for value in joint_ids], dtype=np.int64)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise DiagnosticError(f"model lacks {SITE_NAME}")

    qbase = np.asarray(model.qpos0, dtype=np.float64).copy()
    qbase[:3] = ready["root_pos"]
    # MuJoCo free-joint quaternions are wxyz, matching the ready artifact.
    qbase[3:7] = ready["root_wxyz"]
    qbase[qpos_adrs] = ready["ready"]
    opt_qpos_adrs = qpos_adrs[optimized_runtime]
    lower = ready["lower"][optimized_runtime]
    upper = ready["upper"][optimized_runtime]
    _, _, urdf_velocity = load_urdf_motion_limits(args.urdf, OPTIMIZED_JOINTS)
    velocity_limit = urdf_velocity * float(args.velocity_fraction)

    with np.load(args.motion, allow_pickle=False) as motion:
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        target_pos = np.asarray(motion["measured_racket_site_pos_w"], dtype=np.float64)
        target_face = np.asarray(motion["measured_racket_normal_w"], dtype=np.float64)
        target_long = np.asarray(motion["measured_racket_long_axis_w"], dtype=np.float64)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        mount_sign = float(np.asarray(motion["measured_racket_robot_mount_normal_sign"]).reshape(-1)[0])
    hit = int(args.hit_frame)
    frames = np.asarray([hit - 1, hit, hit + 1], dtype=np.int64)
    if hit <= 0 or hit + 1 >= len(joint_pos) or joint_pos.shape[1] != 31:
        raise DiagnosticError("hit frame lacks a complete three-frame measured window")

    def fk(arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = qbase.copy()
        q[opt_qpos_adrs] = arm
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        return (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
            np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
        )

    raw_style = np.clip(joint_pos[frames][:, optimized_runtime], lower, upper)
    x0 = raw_style.reshape(-1)
    bounds = list(zip(np.tile(lower, 3), np.tile(upper, 3)))

    def evaluate(flat: np.ndarray):
        arms = flat.reshape(3, -1)
        sites, faces, longs = [], [], []
        for arm in arms:
            site, rotation = fk(arm)
            sites.append(site)
            faces.append(rotation[:, 1] * mount_sign)
            longs.append(rotation @ ROBOT_BUTT_TO_BLADE_AXIS_LOCAL)
        sites = np.asarray(sites)
        faces = np.asarray(faces)
        longs = np.asarray(longs)
        velocity = (sites[2] - sites[0]) * (0.5 * fps)
        target_velocity = (target_pos[hit + 1] - target_pos[hit - 1]) * (0.5 * fps)
        return arms, sites, faces, longs, velocity, target_velocity

    def cost(flat: np.ndarray) -> float:
        arms, sites, faces, longs, velocity, target_velocity = evaluate(flat)
        # Position/face are the primary hard-to-recover contact geometry; velocity and style are
        # lower-weight tie breakers, as in the reviewed canonical full-phase solver.
        value = float(args.position_weight) * float(np.sum((sites - target_pos[frames]) ** 2))
        value += float(args.face_weight) * float(
            np.sum(1.0 - np.sum(faces * target_face[frames], axis=1))
        )
        value += float(args.long_axis_weight) * float(
            np.sum(1.0 - np.sum(longs * target_long[frames], axis=1))
        )
        value += float(args.velocity_weight) * float(np.sum((velocity - target_velocity) ** 2))
        value += 0.2 * float(np.sum((arms - raw_style) ** 2))
        value += 0.05 * float(np.sum(np.diff(arms, axis=0) ** 2))
        return value

    result = minimize(
        cost, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(args.maxiter), "maxls": 50}
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise DiagnosticError("hit-window optimizer produced non-finite output")
    arms, sites, faces, longs, velocity, target_velocity = evaluate(result.x)
    hit_arm = arms[1]
    duration, duration_by_joint = minimum_jerk_duration_s(
        ready["ready"][optimized_runtime], hit_arm, velocity_limit
    )
    strict_margin = np.minimum(hit_arm - lower, upper - hit_arm)
    report = {
        "kind": "take061_stable_upper_hit_ik_diagnostic_v1",
        "non_claims": ["not a production motion", "not training admission", "not deployment evidence"],
        "inputs": {
            "dynamic_ready": {"path": str(args.dynamic_ready), "sha256": _sha256(args.dynamic_ready)},
            "motion": {"path": str(args.motion), "sha256": _sha256(args.motion)},
            "model": {"path": str(args.model), "sha256": _sha256(args.model)},
            "urdf": {"path": str(args.urdf), "sha256": _sha256(args.urdf)},
        },
        "reuse_graph": {
            "stable_support": "dynamic_ready.physical_ready root + all non-optimized joints",
            "kinematics": "canonical A3 MuJoCo right_racket site",
            "targets": "motion_kinematics_contract-backed measured_racket_* arrays",
            "limits": "dynamic-ready executable qdes envelope intersected with strict epsilon",
            "timing": "URDF per-joint velocity x fraction; quintic minimum-jerk peak derivative",
        },
        "optimized_joints": list(OPTIMIZED_JOINTS),
        "fixed_joint_count": 21,
        "hit_frame": hit,
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.fun),
            "weights": {
                "position": float(args.position_weight),
                "face": float(args.face_weight),
                "long_axis": float(args.long_axis_weight),
                "velocity": float(args.velocity_weight),
            },
        },
        "residual": {
            "hit_position_m": float(np.linalg.norm(sites[1] - target_pos[hit])),
            "hit_face_deg": _angle_deg(faces[1], target_face[hit]),
            "hit_long_axis_deg": _angle_deg(longs[1], target_long[hit]),
            "hit_velocity_mps": float(np.linalg.norm(velocity - target_velocity)),
            "actual_velocity_mps": velocity.tolist(),
            "target_velocity_mps": target_velocity.tolist(),
        },
        "executable_envelope": {
            "strict_epsilon_rad": STRICT_ENVELOPE_EPS_RAD,
            "minimum_hit_margin_to_strict_bound_rad": float(np.min(strict_margin)),
            "minimum_hit_margin_to_projector_edge_rad": float(
                np.min(strict_margin) + STRICT_ENVELOPE_EPS_RAD
            ),
            "ready_waist_roll_rad": float(ready["ready"][name_to_runtime["waist_roll_joint"]]),
            "upper_waist_roll_rad": float(ready["upper"][name_to_runtime["waist_roll_joint"]]),
        },
        "preparation": {
            "minimum_jerk_duration_s": duration,
            "limiting_joint": OPTIMIZED_JOINTS[int(np.argmax(duration_by_joint))],
            "velocity_fraction": float(args.velocity_fraction),
        },
        "table_keepout": {"status": "NOT_MEASURED", "reason": "robot-only canonical plant has no table assembly"},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_json.with_suffix(".npz"),
        ready_qdes=ready["ready"].astype(np.float32),
        optimized_runtime_indices=optimized_runtime,
        hit_window_frames=frames,
        hit_window_optimized_qdes=arms.astype(np.float32),
        hit_window_site_w=sites.astype(np.float32),
    )
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-ready", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hit-frame", type=int, default=48)
    parser.add_argument("--velocity-fraction", type=float, default=0.90)
    parser.add_argument("--position-weight", type=float, default=1.0e6)
    parser.add_argument("--face-weight", type=float, default=1.0e4)
    parser.add_argument("--long-axis-weight", type=float, default=2.0e3)
    parser.add_argument("--velocity-weight", type=float, default=0.02)
    parser.add_argument("--maxiter", type=int, default=1000)
    return parser


def main() -> int:
    report = solve(_parser().parse_args())
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
