#!/usr/bin/env python3
"""Produce or explicitly reject one stable-support Take061 plant trajectory.

This is an offline N=1 diagnostic producer.  It does not modify the canonical
v4 retarget, the schema-2 materializer, or any FullMDP runtime.  Stable root,
feet and lower body are fixed to the accepted dynamic-ready artifact.  The
right arm is opened first, then waist+right-arm only if necessary.  The
canonical paddle path is a soft prior outside the contact window; contact
position, velocity and signed face are hard post-solve predicates.

Every candidate is evaluated through the actual MuJoCo plant with inverse
dynamics.  The executable target is
``qdes_ff = q_ref + (tau_required + Kd*qdot_ref) / Kp``.  A candidate that
cannot satisfy qdes, torque, support, table or contact predicates is emitted as
a typed reject artifact rather than being silently admitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
_HIT_IK_PATH = HERE / "diagnose_take061_stable_upper_hit_ik.py"
_SPEC = importlib.util.spec_from_file_location("_take061_hit_ik_reuse", _HIT_IK_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load stable-upper IK reuse module")
_HIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HIT)

_SOLVER = _HIT._SOLVER
SITE_NAME = _HIT.SITE_NAME
ROBOT_BUTT_TO_BLADE_AXIS_LOCAL = _HIT.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL

SEMANTIC_KIND = "take061_stable_support_plant_feasible_retarget_v1"
RIGHT_ARM_JOINTS = tuple(name for name in _HIT.OPTIMIZED_JOINTS if not name.startswith("waist_"))
WAIST_RIGHT_ARM_JOINTS = _HIT.OPTIMIZED_JOINTS
CONTACT_RADIUS = 1
STRICT_EPS_RAD = 1.0e-5
TABLE_CLEARANCE_M = 0.020
NUMERIC_TOL_M = 2.0e-8


class ProducerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def plant_compensated_qdes(
    q_ref: np.ndarray,
    qdot_ref: np.ndarray,
    tau_required: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (
        q_ref, qdot_ref, tau_required, kp, kd
    ))
    if any(value.shape != arrays[0].shape for value in arrays[1:]) or np.any(arrays[3] <= 0):
        raise ProducerError("invalid plant-compensated qdes inputs")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ProducerError("non-finite plant-compensated qdes input")
    return arrays[0] + (arrays[2] + arrays[4] * arrays[1]) / arrays[3]


def trajectory_derivatives(q: np.ndarray, fps: float, timewarp: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or len(q) < 3 or fps <= 0 or timewarp < 1 or not np.isfinite(q).all():
        raise ProducerError("invalid trajectory derivative input")
    dt = float(timewarp) / float(fps)
    return np.gradient(q, dt, axis=0), np.gradient(np.gradient(q, dt, axis=0), dt, axis=0)


def minimum_jerk_samples(start: np.ndarray, goal: np.ndarray, duration_s: float, dt: float):
    if duration_s <= 0 or dt <= 0:
        raise ProducerError("minimum-jerk timing must be positive")
    # Three samples are the minimum required for the acceleration receipt.
    steps = max(3, int(math.ceil(duration_s / dt)) + 1)
    t = np.linspace(0.0, duration_s, steps)
    u = t / duration_s
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration_s
    sdd = (60 * u - 180 * u**2 + 120 * u**3) / (duration_s**2)
    delta = np.asarray(goal) - np.asarray(start)
    return (
        np.asarray(start)[None, :] + s[:, None] * delta[None, :],
        sd[:, None] * delta[None, :],
        sdd[:, None] * delta[None, :],
    )


def choose_first_admitted(stages: list[dict[str, Any]]) -> int | None:
    for index, stage in enumerate(stages):
        if bool(stage.get("admitted", False)):
            return index
    return None


def contact_velocity_error(
    sites: np.ndarray,
    target_pos: np.ndarray,
    fps: float,
    timewarp: float,
    contact_frames: np.ndarray,
) -> float:
    """Compare the time-warped candidate velocity against the fixed task velocity."""
    actual = np.gradient(np.asarray(sites), float(timewarp) / float(fps), axis=0)
    target = np.gradient(np.asarray(target_pos), 1.0 / float(fps), axis=0)
    return float(np.max(np.linalg.norm(actual[contact_frames] - target[contact_frames], axis=1)))


def _angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)))


def _load_model(mujoco, path: Path):
    if path.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(str(path))
    return mujoco.MjModel.from_xml_path(str(path))


def _resolve_name(mujoco, model, object_type, canonical: str) -> int:
    direct = int(mujoco.mj_name2id(model, object_type, canonical))
    if direct >= 0:
        return direct
    count_by_type = {
        int(mujoco.mjtObj.mjOBJ_JOINT): int(model.njnt),
        int(mujoco.mjtObj.mjOBJ_SITE): int(model.nsite),
    }
    matches = []
    for index in range(count_by_type[int(object_type)]):
        name = mujoco.mj_id2name(model, object_type, index) or ""
        if name.endswith("/" + canonical):
            matches.append(index)
    if len(matches) != 1:
        raise ProducerError(
            "compiled plant name closure is not unique for " + repr(canonical)
        )
    return int(matches[0])


def _runtime_mapping(mujoco, model, names: tuple[str, ...]):
    joint_ids = np.asarray([
        _resolve_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
    ], dtype=np.int64)
    qadr = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    dadr = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free) < 1:
        raise ProducerError("plant lacks robot free root")
    root = int(free[0])
    return joint_ids, qadr, dadr, int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])


def _fk(mujoco, model, data, qbase, qadr, q, site_id):
    data.qpos[:] = qbase
    data.qpos[qadr] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return (
        np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
        np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def _solve_stage(
    *, mujoco, minimize, model, data, qbase, qadr, site_id, ready, motion_q,
    target_pos, target_face, target_long, fps, mount_sign, joint_names,
    lower, upper, velocity_limit, hit,
) -> dict[str, Any]:
    name_to_runtime = {name: i for i, name in enumerate(ready["names"])}
    opt = np.asarray([name_to_runtime[name] for name in joint_names], dtype=np.int64)
    q = np.tile(ready["ready"], (len(motion_q), 1))
    q[:, opt] = np.clip(motion_q[:, opt], lower[opt], upper[opt])
    order = [hit]
    for distance in range(1, len(q)):
        if hit - distance >= 0:
            order.append(hit - distance)
        if hit + distance < len(q):
            order.append(hit + distance)
    solved = np.zeros(len(q), dtype=bool)

    def one(frame: int):
        neighbor = None
        if frame < hit and solved[frame + 1]:
            neighbor = q[frame + 1, opt]
        elif frame > hit and solved[frame - 1]:
            neighbor = q[frame - 1, opt]
        lo = lower[opt].copy()
        hi = upper[opt].copy()
        if neighbor is not None:
            max_step = velocity_limit[opt] / fps
            lo = np.maximum(lo, neighbor - max_step)
            hi = np.minimum(hi, neighbor + max_step)
        initial = np.clip(q[frame, opt], lo, hi)
        contact = abs(frame - hit) <= CONTACT_RADIUS

        def objective(value):
            row = q[frame].copy()
            row[opt] = value
            site, rotation = _fk(mujoco, model, data, qbase, qadr, row, site_id)
            face = rotation[:, 1] * mount_sign
            long_axis = rotation @ ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
            wp = 1.0e6 if contact else 2.0e3
            wf = 2.0e4 if contact else 100.0
            wl = 1.0e3 if contact else 20.0
            style = motion_q[frame, opt]
            value_cost = wp * np.sum((site - target_pos[frame]) ** 2)
            value_cost += wf * (1.0 - np.dot(face, target_face[frame]))
            value_cost += wl * (1.0 - np.dot(long_axis, target_long[frame]))
            value_cost += 0.2 * np.sum((value - style) ** 2)
            if neighbor is not None:
                value_cost += 0.05 * np.sum((value - neighbor) ** 2)
            return float(value_cost)

        result = minimize(
            objective, initial, method="L-BFGS-B", bounds=list(zip(lo, hi)),
            options={"maxiter": 250 if contact else 80, "maxls": 40},
        )
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise ProducerError(f"optimizer non-finite at frame {frame}")
        q[frame, opt] = np.asarray(result.x)
        solved[frame] = True

    for frame in order:
        one(frame)

    sites, faces, longs = [], [], []
    for row in q:
        site, rotation = _fk(mujoco, model, data, qbase, qadr, row, site_id)
        sites.append(site)
        faces.append(rotation[:, 1] * mount_sign)
        longs.append(rotation @ ROBOT_BUTT_TO_BLADE_AXIS_LOCAL)
    sites = np.asarray(sites)
    faces = np.asarray(faces)
    longs = np.asarray(longs)
    velocities = np.gradient(sites, 1.0 / fps, axis=0)
    target_vel = np.gradient(target_pos, 1.0 / fps, axis=0)
    contact_frames = np.arange(hit - CONTACT_RADIUS, hit + CONTACT_RADIUS + 1)
    residual = {
        "contact_position_max_m": float(np.max(np.linalg.norm(sites[contact_frames] - target_pos[contact_frames], axis=1))),
        "contact_velocity_max_mps": float(np.max(np.linalg.norm(velocities[contact_frames] - target_vel[contact_frames], axis=1))),
        "contact_face_max_deg": float(np.max(_angle_deg(faces[contact_frames], target_face[contact_frames]))),
        "contact_long_axis_max_deg": float(np.max(_angle_deg(longs[contact_frames], target_long[contact_frames]))),
        "full_position_p95_m": float(np.percentile(np.linalg.norm(sites - target_pos, axis=1), 95)),
    }
    return {"joint_names": list(joint_names), "q_ref": q, "sites": sites, "faces": faces, "longs": longs, "residual": residual}


def _plant_eval(
    *, mujoco, model, qbase, qadr, dadr, root_dadr, q_ref, fps, timewarp,
    kp, kd, effort, qdes_lower, qdes_upper, table_geom_ids, robot_geom_ids,
):
    data = mujoco.MjData(model)
    inverse_model = copy.copy(model)
    inverse_model.opt.disableflags = int(inverse_model.opt.disableflags) | int(
        mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    )
    inverse_data = mujoco.MjData(inverse_model)
    qd, qdd = trajectory_derivatives(q_ref, fps, timewarp)
    tau_rows = []
    qdes_rows = []
    root_wrench = []
    min_table = float("inf")
    first_table = None
    support_counts = {"left": 0, "right": 0}
    for frame, (q, dq, ddq) in enumerate(zip(q_ref, qd, qdd)):
        data.qpos[:] = qbase
        data.qpos[qadr] = q
        data.qvel[:] = 0.0
        data.qvel[dadr] = dq
        mujoco.mj_forward(model, data)
        frame_support = {"left": False, "right": False}
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            for geom_id in (int(contact.geom1), int(contact.geom2)):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                lowered = geom_name.lower()
                if "left" in lowered and ("foot" in lowered or "ankle" in lowered):
                    frame_support["left"] = True
                if "right" in lowered and ("foot" in lowered or "ankle" in lowered):
                    frame_support["right"] = True
        for side in frame_support:
            support_counts[side] += int(frame_support[side])
        for robot_gid in robot_geom_ids:
            for table_gid in table_geom_ids:
                distance = float(mujoco.mj_geomDistance(
                    model, data, int(robot_gid), int(table_gid), TABLE_CLEARANCE_M,
                    np.zeros(6, dtype=np.float64),
                ))
                if distance < min_table:
                    min_table = distance
                    first_table = {
                        "frame": int(frame),
                        "robot_geom_id": int(robot_gid),
                        "robot_geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(robot_gid)),
                        "table_geom_id": int(table_gid),
                        "table_geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(table_gid)),
                    }
        inverse_data.qpos[:] = qbase
        inverse_data.qpos[qadr] = q
        inverse_data.qvel[:] = 0.0
        inverse_data.qvel[dadr] = dq
        inverse_data.qacc[:] = 0.0
        inverse_data.qacc[dadr] = ddq
        mujoco.mj_inverse(inverse_model, inverse_data)
        tau = np.asarray(inverse_data.qfrc_inverse[dadr], dtype=np.float64).copy()
        tau_rows.append(tau)
        root_wrench.append(np.asarray(
            inverse_data.qfrc_inverse[root_dadr:root_dadr + 6], dtype=np.float64
        ).copy())
        qdes_rows.append(plant_compensated_qdes(q, dq, tau, kp, kd))
    tau = np.asarray(tau_rows)
    qdes = np.asarray(qdes_rows)
    root_wrench = np.asarray(root_wrench)
    qdes_margin = np.minimum(qdes - qdes_lower[None, :], qdes_upper[None, :] - qdes)
    torque_margin = effort[None, :] - np.abs(tau)
    return {
        "qdot": qd, "qdd": qdd, "tau": tau, "qdes_ff": qdes,
        "qdes_margin_min": float(np.min(qdes_margin)),
        "torque_margin_min": float(np.min(torque_margin)),
        "torque_ratio_max": float(np.max(np.abs(tau) / effort[None, :])),
        "root_wrench_abs_max": float(np.max(np.abs(root_wrench))),
        "table_distance_min_m": float(min_table),
        "table_witness": first_table,
        "bilateral_support_frame_fraction": float(
            min(support_counts.values()) / max(1, len(q_ref))
        ),
        "support_frame_counts": support_counts,
    }


def solve(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mujoco
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ProducerError("MuJoCo and SciPy are required") from exc
    ready = _HIT._load_ready(args.dynamic_ready)
    names = ready["names"]
    document = ready["document"]["runtime_plant"]
    model = _load_model(mujoco, args.model)
    data = mujoco.MjData(model)
    _joint_ids, qadr, dadr, root_qadr, root_dadr = _runtime_mapping(mujoco, model, names)
    site_id = _resolve_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    qbase = np.asarray(model.qpos0, dtype=np.float64).copy()
    qbase[root_qadr:root_qadr + 3] = ready["root_pos"]
    qbase[root_qadr + 3:root_qadr + 7] = ready["root_wxyz"]
    qbase[qadr] = ready["ready"]
    with np.load(args.motion, allow_pickle=False) as motion:
        motion_q = np.asarray(motion["joint_pos"], dtype=np.float64)
        target_pos = np.asarray(motion["measured_racket_site_pos_w"], dtype=np.float64)
        target_face = np.asarray(motion["measured_racket_normal_w"], dtype=np.float64)
        target_long = np.asarray(motion["measured_racket_long_axis_w"], dtype=np.float64)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        mount_sign = float(np.asarray(motion["measured_racket_robot_mount_normal_sign"]).reshape(-1)[0])
    if motion_q.shape != (len(target_pos), 31) or len(motion_q) < args.hit_frame + 2:
        raise ProducerError("motion schema/length differs")
    # The dynamic-ready runtime plant is the content-bound 0807 authority.  Do
    # not silently combine it with the older repository URDF merely to obtain
    # a second set of limits.
    velocity_limit = np.asarray(document["joint_velocity_limits"], dtype=np.float64)
    stages = []
    for stage_names in (RIGHT_ARM_JOINTS, WAIST_RIGHT_ARM_JOINTS):
        stage = _solve_stage(
            mujoco=mujoco, minimize=minimize, model=model, data=data, qbase=qbase,
            qadr=qadr, site_id=site_id, ready=ready, motion_q=motion_q,
            target_pos=target_pos, target_face=target_face, target_long=target_long,
            fps=fps, mount_sign=mount_sign, joint_names=stage_names,
            lower=ready["lower"], upper=ready["upper"],
            velocity_limit=velocity_limit, hit=args.hit_frame,
        )
        r = stage["residual"]
        stage["contact_predicates"] = {
            "position": r["contact_position_max_m"] <= args.contact_position_max_m,
            "velocity": r["contact_velocity_max_mps"] <= args.contact_velocity_max_mps,
            "signed_face": r["contact_face_max_deg"] <= args.contact_face_max_deg,
        }
        stage["kinematic_contact_admitted"] = all(stage["contact_predicates"].values())
        stages.append(stage)
        if stage["kinematic_contact_admitted"]:
            break

    kp = np.asarray(document["joint_stiffness"], dtype=np.float64)
    kd = np.asarray(document["joint_damping"], dtype=np.float64)
    effort = np.asarray(document["joint_effort_limits"], dtype=np.float64)
    qdes_lower = np.asarray(document["executed_qdes_lower_rad"], dtype=np.float64) + STRICT_EPS_RAD
    qdes_upper = np.asarray(document["executed_qdes_upper_rad"], dtype=np.float64) - STRICT_EPS_RAD
    table_geom_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("court_table_top", "court_net")
    ], dtype=np.int64)
    table_geom_ids = table_geom_ids[table_geom_ids >= 0]
    if len(table_geom_ids) == 0:
        raise ProducerError("plant has no authoritative table geometry")
    robot_geom_ids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith("court_") and "ball" not in name.lower() and int(model.geom_bodyid[gid]) != 0:
            robot_geom_ids.append(gid)
    if not robot_geom_ids:
        raise ProducerError("plant has no robot geometry")

    for stage in stages:
        stage["plant_candidates"] = []
        for timewarp in np.arange(1.0, args.max_timewarp + 0.5 * args.timewarp_step, args.timewarp_step):
            plant = _plant_eval(
                mujoco=mujoco, model=model, qbase=qbase, qadr=qadr, dadr=dadr,
                root_dadr=root_dadr, q_ref=stage["q_ref"], fps=fps,
                timewarp=float(timewarp), kp=kp, kd=kd, effort=effort,
                qdes_lower=qdes_lower, qdes_upper=qdes_upper,
                table_geom_ids=table_geom_ids, robot_geom_ids=robot_geom_ids,
            )
            contact_frames = np.arange(
                args.hit_frame - CONTACT_RADIUS, args.hit_frame + CONTACT_RADIUS + 1
            )
            contact_velocity_scaled = contact_velocity_error(
                stage["sites"], target_pos, fps, float(timewarp), contact_frames
            )
            plant["timewarp"] = float(timewarp)
            plant["contact_velocity_error_mps"] = contact_velocity_scaled
            plant["predicates"] = {
                "qdes_margin": plant["qdes_margin_min"] >= 0.0,
                "torque_margin": plant["torque_margin_min"] >= 0.0,
                "table_clearance": plant["table_distance_min_m"] > TABLE_CLEARANCE_M - NUMERIC_TOL_M,
                "contact_velocity_preserved": contact_velocity_scaled <= args.contact_velocity_max_mps,
                "bilateral_support": plant["bilateral_support_frame_fraction"] >= 1.0,
            }
            stage["plant_candidates"].append(plant)
            if stage["kinematic_contact_admitted"] and all(plant["predicates"].values()):
                break
        selected_plant = next((p for p in stage["plant_candidates"] if all(p["predicates"].values())), None)
        stage["selected_plant"] = selected_plant
        stage["admitted"] = bool(stage["kinematic_contact_admitted"] and selected_plant is not None)

    selected_index = choose_first_admitted(stages)
    selected = stages[selected_index] if selected_index is not None else stages[-1]
    reject_reasons = []
    if selected_index is None:
        if not any(stage["kinematic_contact_admitted"] for stage in stages):
            reject_reasons.append("CONTACT_WINDOW_KINEMATIC_UNREACHABLE")
        for predicate in (
            "qdes_margin", "torque_margin", "table_clearance",
            "contact_velocity_preserved", "bilateral_support",
        ):
            if not any(any(c["predicates"].get(predicate, False) for c in stage["plant_candidates"]) for stage in stages):
                reject_reasons.append(predicate.upper() + "_UNSATISFIED")

    # Preparation is independently searched against the same plant equations.
    prep = {"status": "NOT_EVALUATED"}
    goal = selected["q_ref"][0]
    for duration in np.arange(args.prep_step_s, args.max_prep_s + 0.5 * args.prep_step_s, args.prep_step_s):
        pq, pqd, pqdd = minimum_jerk_samples(ready["ready"], goal, float(duration), 1.0 / fps)
        # Reuse plant evaluation by treating sampled prep as its own trajectory.
        # Its finite-difference derivatives match the analytic samples closely;
        # store analytic maxima as an independent timing receipt.
        plant = _plant_eval(
            mujoco=mujoco, model=model, qbase=qbase, qadr=qadr, dadr=dadr,
            root_dadr=root_dadr, q_ref=pq, fps=fps, timewarp=1.0,
            kp=kp, kd=kd, effort=effort, qdes_lower=qdes_lower,
            qdes_upper=qdes_upper, table_geom_ids=table_geom_ids,
            robot_geom_ids=robot_geom_ids,
        )
        ok = (
            plant["qdes_margin_min"] >= 0
            and plant["torque_margin_min"] >= 0
            and plant["table_distance_min_m"] > TABLE_CLEARANCE_M - NUMERIC_TOL_M
            and plant["bilateral_support_frame_fraction"] >= 1.0
        )
        if ok:
            prep = {
                "status": "ADMITTED", "duration_s": float(duration),
                "analytic_qdot_abs_max": float(np.max(np.abs(pqd))),
                "analytic_qdd_abs_max": float(np.max(np.abs(pqdd))),
                "plant": plant,
            }
            break
    if prep["status"] != "ADMITTED":
        reject_reasons.append("PREPARATION_PLANT_FEASIBILITY_UNSATISFIED")

    admitted = selected_index is not None and prep["status"] == "ADMITTED"
    report_stages = []
    for stage in stages:
        stage_report = {
            key: value for key, value in stage.items()
            if key not in ("q_ref", "sites", "faces", "longs", "plant_candidates", "selected_plant")
        }
        stage_report.update({
            "plant_candidates": [
                {key: value for key, value in candidate.items() if key not in ("qdot", "qdd", "tau", "qdes_ff")}
                for candidate in stage["plant_candidates"]
            ],
            "selected_plant": None if stage["selected_plant"] is None else {
                key: value for key, value in stage["selected_plant"].items()
                if key not in ("qdot", "qdd", "tau", "qdes_ff")
            },
        })
        report_stages.append(stage_report)
    report = {
        "schema_version": 1,
        "kind": SEMANTIC_KIND,
        "action_uid": "Take_061_unit04_BH",
        "diagnostic_unauthorized": True,
        "training_authorized": False,
        "admitted": bool(admitted),
        "typed_reject_reasons": sorted(set(reject_reasons)),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (("dynamic_ready", args.dynamic_ready), ("motion", args.motion), ("plant", args.model))
        },
        "reuse_graph": {
            "stable_support": "dynamic_ready.physical_ready root/lower/feet",
            "kinematics_and_limits": "canonical full-phase right_racket, joint set, URDF limits",
            "motion_schema": "schema2 measured_racket site/signed-face/long-axis arrays",
            "plant": "MuJoCo inverse dynamics and compiled actual table/robot geoms",
        },
        "plant_receipt": {
            "mujoco_version": getattr(mujoco, "__version__", "UNKNOWN"),
            "disableflags": int(model.opt.disableflags),
            "nativeccd_disabled": bool(
                int(model.opt.disableflags)
                & int(getattr(mujoco.mjtDisableBit, "mjDSBL_NATIVECCD", 0))
            ),
            "table_clearance_m": TABLE_CLEARANCE_M,
            "table_geom_ids": [int(value) for value in table_geom_ids],
            "table_geom_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(value))
                for value in table_geom_ids
            ],
            "robot_geom_count": len(robot_geom_ids),
            "narrowphase": "compiled_mujoco_mj_geomDistance_nativeccd",
        },
        "contact_hard_thresholds": {
            "position_max_m": args.contact_position_max_m,
            "velocity_max_mps": args.contact_velocity_max_mps,
            "signed_face_max_deg": args.contact_face_max_deg,
            "long_axis": "soft_prior_only",
        },
        "stages": report_stages,
        "selected_stage_index": selected_index,
        "preparation": {
            key: value for key, value in prep.items() if key != "plant"
        },
        "non_claims": ["not production motion", "not policy input", "not deployment evidence"],
    }
    if prep.get("plant") is not None:
        report["preparation"]["plant"] = {
            key: value for key, value in prep["plant"].items()
            if key not in ("qdot", "qdd", "tau", "qdes_ff")
        }
    arrays = {
        "q_ref": selected["q_ref"].astype(np.float32),
        "racket_site": selected["sites"].astype(np.float32),
        "racket_face": selected["faces"].astype(np.float32),
        "racket_long": selected["longs"].astype(np.float32),
    }
    if selected.get("selected_plant") is not None:
        for key in ("qdot", "qdd", "tau", "qdes_ff"):
            arrays[key] = selected["selected_plant"][key].astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    report_payload = _json_bytes(report)
    report["artifact_payloads"] = {
        "npz_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "report_preidentity_sha256": hashlib.sha256(report_payload).hexdigest(),
    }
    final_report = _json_bytes(report)
    _write_no_replace(args.output_npz, buffer.getvalue())
    _write_no_replace(args.output_report, final_report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dynamic-ready", type=Path, required=True)
    result.add_argument("--motion", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--output-report", type=Path, required=True)
    result.add_argument("--output-npz", type=Path, required=True)
    result.add_argument("--hit-frame", type=int, default=48)
    result.add_argument("--contact-position-max-m", type=float, default=0.02)
    result.add_argument("--contact-velocity-max-mps", type=float, default=0.25)
    result.add_argument("--contact-face-max-deg", type=float, default=10.0)
    result.add_argument("--max-timewarp", type=float, default=2.0)
    result.add_argument("--timewarp-step", type=float, default=0.1)
    result.add_argument("--max-prep-s", type=float, default=2.0)
    result.add_argument("--prep-step-s", type=float, default=0.02)
    return result


def main() -> int:
    report = solve(parser().parse_args())
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
