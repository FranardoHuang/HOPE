#!/usr/bin/env python3
"""Jointly retarget one executable Take061 centre and its fixed-solver ball."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import types

import numpy as np


HERE = Path(__file__).resolve().parent


def _module(label, filename):
    spec = importlib.util.spec_from_file_location(label, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


P1 = _module("_take061_phase1", "solve_take061_stable_support_plant_feasible.py")
P2 = _module("_take061_phase2", "solve_take061_feasible_center_phase2.py")
KIND = "take061_joint_ball_feasible_center_phase3_v1"


def _require_ball_physics_lineage(label, solver_report, ball_physics):
    """Reject a phase chain that silently changes its fitted ball model."""
    try:
        declared_sha = solver_report["solver_sources"]["ball_physics_sha256"]
    except (KeyError, TypeError) as exc:
        raise P1.ProducerError(
            f"{label} does not declare solver_sources.ball_physics_sha256"
        ) from exc
    actual_sha = P1._sha256(ball_physics)
    if declared_sha != actual_sha:
        raise P1.ProducerError(
            f"{label} ball physics SHA mismatch: declared {declared_sha}, "
            f"requested {actual_sha} ({ball_physics})"
        )


def _solver_replay(center, incoming, aim, args):
    import torch

    cq, vb = P2._load_fixed_solver_math()
    velocity = torch.tensor([center["velocity_w_mps"]], dtype=torch.float32)
    direction = velocity / torch.linalg.norm(velocity, dim=1, keepdim=True)
    protos = types.SimpleNamespace(
        v_hat_b=direction,
        speed_min=torch.tensor([0.1]),
        speed_max=torch.tensor([2.0]),
        face_sign=torch.tensor([1.0]),
    )
    output = cq.solve_proposals(
        torch.tensor([0]),
        torch.tensor([center["site_w_m"]], dtype=torch.float32),
        torch.tensor([incoming], dtype=torch.float32),
        torch.zeros(1, 3),
        torch.tensor([aim], dtype=torch.float32),
        torch.tensor([center["signed_face_w"]], dtype=torch.float32),
        protos=protos,
        base_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        prm=vb.load_venue_params(str(args.ball_physics)),
        surface_z=args.surface_z, net_x=args.net_x, net_top_z=args.net_top_z,
        cfg=cq.ContinuousQuestionCfg(
            fixed_direction=True, n_iters=12, max_redraw_rounds=1, speed_budget=2.0
        ),
        h=0.01, n_steps=100,
    )
    if not bool(output.ok[0]):
        return {"solver_admitted": False, "reason_counts": output.reason_counts}
    achieved_v = velocity[0]
    achieved_n = torch.tensor(center["signed_face_w"])
    solved_n = output.n_racket[0]
    face = torch.rad2deg(torch.acos(torch.clamp(
        torch.dot(achieved_n / torch.linalg.norm(achieved_n), solved_n / torch.linalg.norm(solved_n)),
        -1, 1,
    )))
    velocity_error = torch.linalg.norm(achieved_v - output.v_racket[0])
    return {
        "solver_admitted": True,
        "matched": bool(
            velocity_error <= args.velocity_tolerance_mps
            and face <= args.face_tolerance_deg
        ),
        "racket_velocity_w_mps": output.v_racket[0].tolist(),
        "signed_face_w": output.n_racket[0].tolist(),
        "residual_m": float(output.resid_m[0]),
        "velocity_error_mps": float(velocity_error),
        "face_error_deg": float(face),
        "reason_counts": output.reason_counts,
        "solver_sources": {
            "continuous_questions_sha256": P1._sha256(Path(cq.__file__)),
            "virtual_ball_sha256": P1._sha256(Path(vb.__file__)),
            "ball_physics_sha256": P1._sha256(args.ball_physics),
        },
    }


def solve(args):
    import mujoco
    from scipy.optimize import minimize

    ready = P1._HIT._load_ready(args.dynamic_ready)
    runtime = ready["document"]["runtime_plant"]
    model = P1._load_model(mujoco, args.model)
    data = mujoco.MjData(model)
    _ids, qadr, dadr, root_qadr, root_dadr = P1._runtime_mapping(
        mujoco, model, ready["names"]
    )
    site_id = P1._resolve_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, P1.SITE_NAME)
    qbase = np.asarray(model.qpos0, np.float64).copy()
    qbase[root_qadr : root_qadr + 3] = ready["root_pos"]
    qbase[root_qadr + 3 : root_qadr + 7] = ready["root_wxyz"]
    qbase[qadr] = ready["ready"]
    with np.load(args.phase2_npz, allow_pickle=False) as artifact:
        q_seed = np.asarray(artifact["q_ref"], np.float64)
        sites = np.asarray(artifact["racket_site"], np.float64)
        faces = np.asarray(artifact["racket_face"], np.float64)
    phase2 = json.loads(args.phase2_report.read_text())
    inverse = phase2["fixed_action_solver_inverse"]
    _require_ball_physics_lineage("phase2", inverse, args.ball_physics)
    desired_velocity = np.asarray(inverse["solver_racket_velocity_w_mps"], np.float64)
    desired_face = np.asarray(inverse["solver_signed_face_w"], np.float64)
    incoming = np.asarray(inverse["incoming_velocity_w_mps"], np.float64)
    aim = np.asarray(inverse["landing_aim_w_xy_m"], np.float64)
    with np.load(args.motion, allow_pickle=False) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        longs = np.asarray(motion["measured_racket_long_axis_w"], np.float64)
        mount_sign = float(np.asarray(
            motion["measured_racket_robot_mount_normal_sign"]
        ).reshape(-1)[0])
    target_pos = sites.copy()
    target_face = faces.copy()
    effective_dt = float(phase2["selected"]["timewarp"]) / fps
    for frame in range(args.hit_frame - 1, args.hit_frame + 2):
        target_pos[frame] = sites[args.hit_frame] + desired_velocity * float(
            args.contact_velocity_gain
        ) * (
            (frame - args.hit_frame) * effective_dt
        )
        target_face[frame] = desired_face
    upper_names = tuple(
        name for name in ready["names"]
        if not any(token in name for token in ("hip_", "knee_", "ankle_"))
    )
    stage = P1._solve_stage(
        mujoco=mujoco, minimize=minimize, model=model, data=data, qbase=qbase,
        qadr=qadr, site_id=site_id, ready=ready, motion_q=q_seed,
        target_pos=target_pos, target_face=target_face, target_long=longs,
        fps=fps / float(phase2["selected"]["timewarp"]), mount_sign=mount_sign,
        joint_names=upper_names, lower=ready["lower"], upper=ready["upper"],
        velocity_limit=np.asarray(runtime["joint_velocity_limits"], np.float64),
        hit=args.hit_frame,
    )
    kp = np.asarray(runtime["joint_stiffness"], np.float64)
    kd = np.asarray(runtime["joint_damping"], np.float64)
    effort = np.asarray(runtime["joint_effort_limits"], np.float64)
    qdes_lo = np.asarray(runtime["executed_qdes_lower_rad"], np.float64) + P1.STRICT_EPS_RAD
    qdes_hi = np.asarray(runtime["executed_qdes_upper_rad"], np.float64) - P1.STRICT_EPS_RAD
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
    wrist_index = ready["names"].index("right_wrist_yaw_joint")
    frame_index = np.arange(q_seed.shape[0], dtype=np.float64)
    distance = np.abs(frame_index - float(args.hit_frame))
    wrist_taper = np.clip(
        1.0 - distance / max(2.0, float(args.wrist_bias_window_frames)), 0.0, 1.0
    )
    candidates = []
    for alpha in args.alphas:
        q_alpha = q_seed + float(alpha) * (stage["q_ref"] - q_seed)
        for wrist_bias in args.wrist_inward_biases:
            q = q_alpha.copy()
            q[:, wrist_index] += float(wrist_bias) * wrist_taper
            q[:, wrist_index] = np.clip(
                q[:, wrist_index], ready["lower"][wrist_index], ready["upper"][wrist_index]
            )
            for timewarp in args.timewarps:
                plant = P1._plant_eval(
                    mujoco=mujoco, model=model, qbase=qbase, qadr=qadr, dadr=dadr,
                    root_dadr=root_dadr, q_ref=q, fps=fps, timewarp=timewarp,
                    kp=kp, kd=kd, effort=effort, qdes_lower=qdes_lo,
                    qdes_upper=qdes_hi, table_geom_ids=table_ids,
                    robot_geom_ids=robot_ids,
                )
                path_site, path_face, path_long = P2._fk_path(
                    mujoco, model, qbase, qadr, site_id, q
                )
                path_face *= mount_sign
                velocity = np.gradient(path_site, float(timewarp) / fps, axis=0)
                center = {
                    "site_w_m": path_site[args.hit_frame].tolist(),
                    "velocity_w_mps": velocity[args.hit_frame].tolist(),
                    "signed_face_w": path_face[args.hit_frame].tolist(),
                }
                replay = _solver_replay(center, incoming, aim, args)
                mechanical = bool(
                    plant["qdes_margin_min"] > 0 and plant["torque_margin_min"] > 0
                    and plant["bilateral_support_frame_fraction"] >= 1.0
                    and plant["table_distance_min_m"] >= P1.TABLE_CLEARANCE_M - P1.NUMERIC_TOL_M
                )
                candidates.append({
                    "alpha": float(alpha), "timewarp": float(timewarp),
                    "wrist_inward_bias_rad": float(wrist_bias),
                    "q_ref": q, "site": path_site, "face": path_face,
                    "velocity": velocity, "long": path_long, "plant": plant,
                    "center": center, "solver": replay,
                    "admitted": bool(mechanical and replay.get("matched", False)),
                })
    chosen = min(candidates, key=lambda row: (
        not row["admitted"],
        max(0.0, -row["plant"]["qdes_margin_min"]),
        max(0.0, -row["plant"]["torque_margin_min"]),
        row["solver"].get("face_error_deg", 999.0),
        row["solver"].get("velocity_error_mps", 999.0),
    ))

    def plant_summary(value):
        return {k: v for k, v in value.items() if k not in ("qdot", "qdd", "tau", "qdes_ff")}

    report = {
        "schema_version": 1, "kind": KIND,
        "diagnostic_unauthorized": True,
        "admitted": chosen["admitted"],
        "typed_reject_reasons": [] if chosen["admitted"] else [
            "NO_JOINT_MECHANICAL_AND_FIXED_SOLVER_MATCH_IN_REGISTERED_SEARCH"
        ],
        "selected": {
            "alpha": chosen["alpha"], "timewarp": chosen["timewarp"],
            "wrist_inward_bias_rad": chosen["wrist_inward_bias_rad"],
            "center": chosen["center"], "plant": plant_summary(chosen["plant"]),
            "solver": chosen["solver"],
            "incoming_ball_center": {
                "contact_w_m": chosen["center"]["site_w_m"],
                "incoming_velocity_w_mps": incoming.tolist(),
                "spin_w_radps": [0.0, 0.0, 0.0],
                "landing_aim_w_xy_m": aim.tolist(),
                "support": {
                    "contact_w_m_lower": chosen["center"]["site_w_m"],
                    "contact_w_m_upper": chosen["center"]["site_w_m"],
                    "incoming_velocity_w_mps_lower": incoming.tolist(),
                    "incoming_velocity_w_mps_upper": incoming.tolist(),
                    "spin_w_radps_lower": [0.0, 0.0, 0.0],
                    "spin_w_radps_upper": [0.0, 0.0, 0.0],
                    "landing_aim_w_xy_m_lower": aim.tolist(),
                    "landing_aim_w_xy_m_upper": aim.tolist(),
                },
            },
        },
        "timing": {
            "t_hit_s": args.reference_t_hit_s * chosen["timewarp"],
            "t_cycle_s": args.reference_t_cycle_s * chosen["timewarp"],
        },
        "contact_velocity_gain": float(args.contact_velocity_gain),
        "candidate_count": len(candidates),
        "closest_candidates": [
            {
                "alpha": row["alpha"], "timewarp": row["timewarp"],
                "wrist_inward_bias_rad": row["wrist_inward_bias_rad"],
                "qdes_margin_min_rad": row["plant"]["qdes_margin_min"],
                "torque_margin_min_nm": row["plant"]["torque_margin_min"],
                "table_distance_min_m": row["plant"]["table_distance_min_m"],
                "velocity_error_mps": row["solver"].get("velocity_error_mps"),
                "face_error_deg": row["solver"].get("face_error_deg"),
                "admitted": row["admitted"],
            }
            for row in sorted(candidates, key=lambda row: (
                not row["admitted"],
                max(0.0, -row["plant"]["qdes_margin_min"]),
                max(0.0, -row["plant"]["torque_margin_min"]),
                row["solver"].get("face_error_deg", 999.0),
                row["solver"].get("velocity_error_mps", 999.0),
            ))[:12]
        ],
        "inputs": {
            name: {"path": str(path), "sha256": P1._sha256(path)}
            for name, path in (
                ("dynamic_ready", args.dynamic_ready), ("motion", args.motion),
                ("plant", args.model), ("phase2_npz", args.phase2_npz),
                ("phase2_report", args.phase2_report),
                ("ball_physics", args.ball_physics),
            )
        },
    }
    arrays = {
        "q_ref": chosen["q_ref"].astype(np.float32),
        "qdot": chosen["plant"]["qdot"].astype(np.float32),
        "qdd": chosen["plant"]["qdd"].astype(np.float32),
        "tau": chosen["plant"]["tau"].astype(np.float32),
        "qdes_ff": chosen["plant"]["qdes_ff"].astype(np.float32),
        "racket_site": chosen["site"].astype(np.float32),
        "racket_velocity": chosen["velocity"].astype(np.float32),
        "racket_face": chosen["face"].astype(np.float32),
    }
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
    report["artifact_payloads"] = {"npz_sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
    P1._write_no_replace(args.output_npz, buffer.getvalue())
    P1._write_no_replace(args.output_report, P1._json_bytes(report))
    return report


def parser():
    p = argparse.ArgumentParser()
    for name in ("dynamic-ready", "motion", "model", "phase2-npz", "phase2-report", "output-report", "output-npz"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ball-physics", type=Path, default=Path("configs/ball_physics_venue.yaml"))
    p.add_argument("--hit-frame", type=int, default=48)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.4, 0.5, 0.65, 0.75, 1.0])
    p.add_argument("--timewarps", type=float, nargs="+", default=[4.0, 4.5, 5.0])
    p.add_argument("--wrist-inward-biases", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06])
    p.add_argument("--wrist-bias-window-frames", type=int, default=21)
    p.add_argument("--velocity-tolerance-mps", type=float, default=0.10)
    p.add_argument("--face-tolerance-deg", type=float, default=10.0)
    p.add_argument("--surface-z", type=float, default=0.78)
    p.add_argument("--net-x", type=float, default=1.87)
    p.add_argument("--net-top-z", type=float, default=0.9325)
    p.add_argument("--reference-t-hit-s", type=float, default=0.96)
    p.add_argument("--reference-t-cycle-s", type=float, default=1.12)
    p.add_argument("--contact-velocity-gain", type=float, default=1.0)
    return p


def main():
    report = solve(parser().parse_args())
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
