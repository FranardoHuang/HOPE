#!/usr/bin/env python3
"""Materialize one honest slow-motion Take061 exact-face diagnostic identity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
MDP = HERE.parent / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
KIND = "take061_slow_block_exact_face_phase4_v1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P3 = _load("_take061_phase3_for_phase4", HERE / "solve_take061_joint_ball_phase3.py")
P1 = P3.P1
GEOMETRY = _load("_take061_exact_geometry_phase4", MDP / "racket_contact_geometry.py")


def solve(args):
    import mujoco
    from scipy.spatial.transform import Rotation

    source = json.loads(args.phase3_report.read_text())
    P3._require_seed_valid("phase3", source)
    P3._require_ball_physics_lineage(
        "phase3", source.get("selected", {}).get("solver"), args.ball_physics
    )
    with np.load(args.phase3_npz, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    ready = P1._HIT._load_ready(args.dynamic_ready)
    model = P1._load_model(mujoco, args.model)
    data = mujoco.MjData(model)
    _jids, qadr, dadr, root_qadr, root_dadr = P1._runtime_mapping(
        mujoco, model, ready["names"]
    )
    site_id = P1._resolve_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, P1.SITE_NAME)
    qbase = np.asarray(model.qpos0, np.float64).copy()
    qbase[root_qadr : root_qadr + 3] = ready["root_pos"]
    qbase[root_qadr + 3 : root_qadr + 7] = ready["root_wxyz"]
    qbase[qadr] = ready["ready"]
    q_ref = np.asarray(arrays["q_ref"], np.float64).copy()
    yaw_index = ready["names"].index("waist_yaw_joint")
    taper = np.clip(
        1.0 - np.abs(np.arange(len(q_ref)) - int(args.hit_frame))
        / float(args.contact_bias_window_frames), 0.0, 1.0
    )
    q_ref[:, yaw_index] += float(args.waist_yaw_contact_bias_rad) * taper
    rotations, sites = [], []
    for q in q_ref:
        site_value, rotation = P1._fk(mujoco, model, data, qbase, qadr, q, site_id)
        sites.append(site_value)
        rotations.append(rotation)
    rotations = np.asarray(rotations)
    sites = np.asarray(sites)
    hit = int(args.hit_frame)
    xyzw = Rotation.from_matrix(rotations[hit]).as_quat()
    reference_quat = tuple(float(v) for v in (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
    reference_long_axis = rotations[hit] @ P1.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL
    dt = float(source["selected"]["timewarp"]) / float(args.fps)
    reference_omega = tuple(float(v) for v in Rotation.from_matrix(
        rotations[hit + 1] @ rotations[hit - 1].T
    ).as_rotvec() / (2.0 * dt))
    site_velocity_path = np.gradient(sites, dt, axis=0)
    raw_face_path = rotations[:, :, 1]
    site = sites[hit]
    site_velocity = site_velocity_path[hit]
    raw_face = raw_face_path[hit]
    ball_center = site + np.asarray(GEOMETRY.quat_rotate_wxyz(
        reference_quat, GEOMETRY.ball_center_from_site_local(args.mount_normal_sign)
    ))
    face_offset_local = GEOMETRY.face_center_from_site_local(args.mount_normal_sign)
    face_offset_w = GEOMETRY.quat_rotate_wxyz(reference_quat, face_offset_local)
    center = {
        "site_w_m": ball_center.tolist(),
        "velocity_w_mps": site_velocity.tolist(),
        "signed_face_w": raw_face.tolist(),
    }
    incoming = np.asarray(source["selected"]["incoming_ball_center"]["incoming_velocity_w_mps"])
    aim = np.asarray(source["selected"]["incoming_ball_center"]["landing_aim_w_xy_m"])
    replay = P3._solver_replay(center, incoming, aim, args)
    if not replay.get("solver_admitted"):
        raise P1.ProducerError("corrected ball center was rejected by continuous solver")
    exact = GEOMETRY.solve_exact_face_contact(
        ball_contact_w_m=ball_center.tolist(),
        racket_face_center_velocity_w_mps=replay["racket_velocity_w_mps"],
        solved_raw_a_normal_w=replay["signed_face_w"],
        mount_normal_sign=args.mount_normal_sign,
        reference_racket_quat_wxyz=reference_quat,
        reference_racket_angular_velocity_w_radps=reference_omega,
        reference_racket_site_speed_mps=float(np.linalg.norm(site_velocity)),
        teacher_rate_min=args.teacher_rate_min,
        teacher_rate_max=args.teacher_rate_max,
    )
    import torch
    cq, vb = P3.P2._load_fixed_solver_math()
    prm = vb.load_venue_params(str(args.ball_physics))
    v_plus, w_plus = cq.predict_paddle_contact(
        torch.tensor([incoming], dtype=torch.float32),
        torch.tensor([replay["racket_velocity_w_mps"]], dtype=torch.float32),
        torch.tensor([replay["signed_face_w"]], dtype=torch.float32)
        * float(args.mount_normal_sign),
        torch.zeros(1, 3, dtype=torch.float32), prm,
    )
    landing = vb.coarse_landing(
        torch.tensor([ball_center], dtype=torch.float32), v_plus, w_plus, prm,
        surface_z=args.surface_z, net_x=args.net_x, h=0.01, n_steps=100,
    )
    position_error = float(np.linalg.norm(np.asarray(exact.racket_site_target_w_m) - site))
    velocity_error = float(np.linalg.norm(np.asarray(exact.racket_site_velocity_w_mps) - site_velocity))
    runtime = ready["document"]["runtime_plant"]
    table_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("court_table_top", "court_net")
    ], np.int64)
    robot_ids = [
        gid for gid in range(model.ngeom)
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith("court_")
        and "ball" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        and int(model.geom_bodyid[gid]) != 0
    ]
    plant_full = P1._plant_eval(
        mujoco=mujoco, model=model, qbase=qbase, qadr=qadr, dadr=dadr,
        root_dadr=root_dadr, q_ref=q_ref, fps=float(args.fps),
        timewarp=float(source["selected"]["timewarp"]),
        kp=np.asarray(runtime["joint_stiffness"], np.float64),
        kd=np.asarray(runtime["joint_damping"], np.float64),
        effort=np.asarray(runtime["joint_effort_limits"], np.float64),
        qdes_lower=np.asarray(runtime["executed_qdes_lower_rad"], np.float64) + P1.STRICT_EPS_RAD,
        qdes_upper=np.asarray(runtime["executed_qdes_upper_rad"], np.float64) - P1.STRICT_EPS_RAD,
        table_geom_ids=table_ids, robot_geom_ids=robot_ids,
    )
    plant = {key: value for key, value in plant_full.items() if key not in ("qdot", "qdd", "tau", "qdes_ff")}
    robust = bool(
        plant["qdes_margin_min"] >= args.robust_qdes_margin_rad
        and replay["velocity_error_mps"] <= args.robust_velocity_error_mps
        and replay["face_error_deg"] <= args.robust_face_error_deg
        and position_error <= args.robust_exact_site_position_error_m
        and velocity_error <= args.robust_exact_site_velocity_error_mps
        and plant["table_distance_min_m"] >= args.table_clearance_m
        and plant["torque_margin_min"] > 0.0
        and plant["bilateral_support_frame_fraction"] >= 1.0
        and bool(landing["land_valid"][0])
        and bool(landing["net_valid"][0])
        and float(landing["net_z"][0]) > args.net_top_z
    )
    arrays.update({
        "q_ref": q_ref.astype(np.float32),
        "qdot": plant_full["qdot"].astype(np.float32),
        "qdd": plant_full["qdd"].astype(np.float32),
        "tau": plant_full["tau"].astype(np.float32),
        "qdes_ff": plant_full["qdes_ff"].astype(np.float32),
        "racket_site": sites.astype(np.float32),
        "racket_velocity": site_velocity_path.astype(np.float32),
        "racket_face": raw_face_path.astype(np.float32),
        "racket_quat_wxyz": np.asarray(reference_quat, np.float32),
        "racket_omega_w_radps": np.asarray(reference_omega, np.float32),
        "racket_long_axis_w": np.asarray(reference_long_axis, np.float32),
        "ball_center_w_m": ball_center.astype(np.float32),
        "exact_racket_site_target_w_m": np.asarray(exact.racket_site_target_w_m, np.float32),
        "exact_racket_site_velocity_w_mps": np.asarray(exact.racket_site_velocity_w_mps, np.float32),
        "exact_racket_command_quat_wxyz": np.asarray(exact.racket_command_quat_wxyz, np.float32),
    })
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
    identity = hashlib.sha256(buffer.getvalue()).hexdigest()
    report = {
        "schema_version": 1, "kind": KIND, "diagnostic_unauthorized": True,
        "new_action_identity_sha256": identity,
        "seed_valid": True,
        "matched": bool(replay.get("matched", False)),
        "admitted": robust,
        "exact_face_admitted": True, "robust_curriculum_center": robust,
        "typed_reject_reasons": [] if robust else ["ROBUST_MARGIN_TARGETS_NOT_MET"],
        "continuous_solver": replay,
        "action_ball": {
            "incoming_velocity_w_mps": incoming.tolist(),
            "incoming_spin_w_radps": [0.0, 0.0, 0.0],
            "landing_aim_w_xy_m": aim.tolist(),
            "analytic_landing_w_xy_m": landing["land_xy"][0].tolist(),
            "analytic_landing_valid": bool(landing["land_valid"][0]),
            "analytic_net_crossing_valid": bool(landing["net_valid"][0]),
            "analytic_net_crossing_z_m": float(landing["net_z"][0]),
            "analytic_landing_error_m": float(np.linalg.norm(
                np.asarray(landing["land_xy"][0].tolist()) - aim
            )),
        },
        "exact_face": {
            "geometry_source_sha256": exact.geometry_source_sha256,
            "ball_radius_m": GEOMETRY.BALL_RADIUS_M,
            "mount_normal_sign": exact.mount_normal_sign,
            "ball_center_w_m": ball_center.tolist(),
            "reference_racket_quat_wxyz": list(reference_quat),
            "reference_racket_angular_velocity_w_radps": list(reference_omega),
            "reference_racket_long_axis_w": reference_long_axis.tolist(),
            "face_center_from_site_local_m": list(face_offset_local),
            "face_center_from_site_w_m": list(face_offset_w),
            "ball_center_from_site_local_m": list(
                GEOMETRY.ball_center_from_site_local(args.mount_normal_sign)
            ),
            "racket_command_quat_wxyz": list(exact.racket_command_quat_wxyz),
            "racket_command_angular_velocity_w_radps": list(exact.racket_command_angular_velocity_w_radps),
            "racket_site_target_w_m": list(exact.racket_site_target_w_m),
            "racket_site_velocity_w_mps": list(exact.racket_site_velocity_w_mps),
            "teacher_rate": exact.teacher_rate,
            "site_position_error_m": position_error,
            "site_velocity_error_mps": velocity_error,
        },
        "plant": plant,
        "retarget": {
            "waist_yaw_contact_bias_rad": float(args.waist_yaw_contact_bias_rad),
            "contact_bias_window_frames": int(args.contact_bias_window_frames),
        },
        "timing": source["timing"],
        "artifact_payloads": {"npz_sha256": identity},
        "inputs": {name: {"path": str(path), "sha256": P1._sha256(path)} for name, path in (
            ("phase3_report", args.phase3_report), ("phase3_npz", args.phase3_npz),
            ("dynamic_ready", args.dynamic_ready), ("plant", args.model),
            ("geometry", MDP / "racket_contact_geometry.py"),
            ("ball_physics", args.ball_physics),
        )},
        "non_claims": [
            "not the old Take061 action identity", "not a runtime fixed tape",
            "not policy, promotion, or deployment evidence",
        ],
    }
    P1._write_no_replace(args.output_npz, buffer.getvalue())
    P1._write_no_replace(args.output_report, P1._json_bytes(report))
    return report


def parser():
    p = argparse.ArgumentParser()
    for name in ("phase3-report", "phase3-npz", "dynamic-ready", "model", "output-report", "output-npz"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ball-physics", type=Path, default=Path("configs/ball_physics_venue.yaml"))
    p.add_argument("--hit-frame", type=int, default=48); p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--mount-normal-sign", type=int, default=1)
    p.add_argument("--waist-yaw-contact-bias-rad", type=float, default=0.0)
    p.add_argument("--contact-bias-window-frames", type=int, default=21)
    p.add_argument("--teacher-rate-min", type=float, default=0.6); p.add_argument("--teacher-rate-max", type=float, default=1.0)
    p.add_argument("--velocity-tolerance-mps", type=float, default=0.10); p.add_argument("--face-tolerance-deg", type=float, default=10.0)
    p.add_argument("--robust-qdes-margin-rad", type=float, default=0.02)
    p.add_argument("--robust-velocity-error-mps", type=float, default=0.08)
    p.add_argument("--robust-face-error-deg", type=float, default=8.0)
    p.add_argument("--robust-exact-site-position-error-m", type=float, default=0.005)
    p.add_argument("--robust-exact-site-velocity-error-mps", type=float, default=0.08)
    p.add_argument("--table-clearance-m", type=float, default=0.02)
    p.add_argument("--surface-z", type=float, default=0.78); p.add_argument("--net-x", type=float, default=1.87); p.add_argument("--net-top-z", type=float, default=0.9325)
    return p


def main():
    report = solve(parser().parse_args()); print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["robust_curriculum_center"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
