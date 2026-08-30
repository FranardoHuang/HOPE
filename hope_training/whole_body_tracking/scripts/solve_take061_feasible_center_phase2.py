#!/usr/bin/env python3
"""Co-design a smooth executable Take061 contact centre from a Phase-1 seed.

This deterministic offline diagnostic deliberately stops before publication to
the shared fixed-action solver.  It searches smooth, stable-support trajectories
whose *achieved* contact site/velocity/signed face define the new action centre;
the old canonical SE(3), long axis and style are ranking priors only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_phase1", HERE / "solve_take061_stable_support_plant_feasible.py"
)
P1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P1)

KIND = "take061_stable_support_feasible_center_phase2_v1"


def _phase2_velocity_matched(velocity_error_mps, tolerance_mps):
    """Phase2 matches the executable centre to solver velocity only.

    The measured teacher face and the solver face encode different intents and
    have no specification requiring equality.  Their angle remains a recorded
    diagnostic; Phase3/4 own solver-face retarget and final admission.
    """
    return bool(
        np.isfinite(velocity_error_mps)
        and np.isfinite(tolerance_mps)
        and velocity_error_mps <= tolerance_mps
    )


def _load_fixed_solver_math():
    """Load the shipped solver modules by path without importing IsaacLab."""
    root = HERE.parents[0] / "source" / "whole_body_tracking" / "whole_body_tracking"
    mdp = root / "tasks" / "tracking" / "mdp"
    package = "_take061_phase2_mdp"
    module = types.ModuleType(package)
    module.__path__ = [str(mdp)]
    sys.modules[package] = module
    loaded = {}
    for name in (
        "racket_contact_geometry", "virtual_ball", "strike_spec_torch",
        "stroke_adapt_torch", "continuous_questions",
    ):
        spec = importlib.util.spec_from_file_location(
            package + "." + name, mdp / (name + ".py")
        )
        child = importlib.util.module_from_spec(spec)
        sys.modules[package + "." + name] = child
        spec.loader.exec_module(child)
        loaded[name] = child
    return loaded["continuous_questions"], loaded["virtual_ball"]


def _fixed_solver_inverse_grid(center, args):
    import torch

    cq, vb = _load_fixed_solver_math()
    rows = []
    for speed in args.ball_speeds:
        for y in args.ball_direction_y:
            for z in args.ball_direction_z:
                direction = np.asarray([-1.0, y, z], np.float64)
                direction /= np.linalg.norm(direction)
                for aim_x in args.landing_aim_x:
                    for aim_y in args.landing_aim_y:
                        rows.append((float(speed), direction, float(aim_x), float(aim_y)))
    n = len(rows)
    dtype = torch.float32
    contact = torch.tensor([center["site_w_m"]] * n, dtype=dtype)
    incoming = torch.tensor(
        [(speed * direction).tolist() for speed, direction, _, _ in rows], dtype=dtype
    )
    spin = torch.zeros(n, 3, dtype=dtype)
    aim = torch.tensor([[x, y] for _, _, x, y in rows], dtype=dtype)
    feasible_velocity = torch.tensor(center["velocity_w_mps"], dtype=dtype)
    direction = feasible_velocity / torch.linalg.norm(feasible_velocity)
    protos = types.SimpleNamespace(
        v_hat_b=direction[None].expand(n, 3).clone(),
        speed_min=torch.full((n,), 0.1, dtype=dtype),
        speed_max=torch.full((n,), 2.0, dtype=dtype),
        face_sign=torch.ones(n, dtype=dtype),
    )
    base_quat = torch.zeros(n, 4, dtype=dtype)
    base_quat[:, 0] = 1.0
    reference_face = torch.tensor([center["signed_face_w"]] * n, dtype=dtype)
    cfg = cq.ContinuousQuestionCfg(
        fixed_direction=True, n_iters=12, max_redraw_rounds=1, speed_budget=2.0
    )
    prm = vb.load_venue_params(str(args.ball_physics))
    solved = cq.solve_proposals(
        torch.zeros(n, dtype=torch.long), contact, incoming, spin, aim,
        reference_face, protos=protos, base_quat=base_quat, prm=prm,
        surface_z=args.surface_z, net_x=args.net_x, net_top_z=args.net_top_z,
        cfg=cfg, h=0.01, n_steps=100,
    )
    unit_face = solved.n_racket / torch.linalg.norm(solved.n_racket, dim=1, keepdim=True)
    target_face = reference_face / torch.linalg.norm(reference_face, dim=1, keepdim=True)
    face_deg = torch.rad2deg(torch.acos(torch.clamp((unit_face * target_face).sum(1), -1, 1)))
    velocity_error = torch.linalg.norm(solved.v_racket - feasible_velocity, dim=1)
    # Teacher face and solver face encode different targets.  Do not use their
    # angle either to select or admit a Phase2 velocity seed.
    score = velocity_error
    score[~solved.ok] = float("inf")
    best = int(torch.argmin(score))
    if not bool(torch.isfinite(score[best])):
        return {
            "matched": False,
            "velocity_matched": False,
            "admitted_count": 0,
            "face_error_admission_gate": False,
            "reason_counts": solved.reason_counts,
        }
    speed, incoming_direction, aim_x, aim_y = rows[best]
    velocity_matched = _phase2_velocity_matched(
        float(velocity_error[best]), args.solver_velocity_tolerance_mps
    )
    return {
        "matched": velocity_matched,
        "velocity_matched": velocity_matched,
        "match_basis": "mechanical_center_velocity_only",
        "admitted_count": int(solved.ok.sum()),
        "proposal_count": n,
        "incoming_speed_mps": speed,
        "incoming_direction_w": incoming_direction.tolist(),
        "incoming_velocity_w_mps": incoming[best].tolist(),
        "landing_aim_w_xy_m": [aim_x, aim_y],
        "solver_racket_velocity_w_mps": solved.v_racket[best].tolist(),
        "solver_signed_face_w": solved.n_racket[best].tolist(),
        "solver_residual_m": float(solved.resid_m[best]),
        "velocity_error_from_feasible_center_mps": float(velocity_error[best]),
        "face_error_from_feasible_center_deg": float(face_deg[best]),
        "velocity_tolerance_mps": args.solver_velocity_tolerance_mps,
        "face_tolerance_deg": args.solver_face_tolerance_deg,
        "face_error_admission_gate": False,
        "face_error_role": "diagnostic_only_teacher_and_solver_faces_need_not_match",
        "reason_counts": solved.reason_counts,
        "solver_sources": {
            "continuous_questions_sha256": P1._sha256(Path(cq.__file__)),
            "virtual_ball_sha256": P1._sha256(Path(vb.__file__)),
            "ball_physics_sha256": P1._sha256(args.ball_physics),
        },
    }


def _solver_seed_valid(solver_inverse):
    """Return whether Phase3 has one finite, solver-admitted retarget seed.

    A seed need not already match the Phase2 racket centre: Phase3 exists to
    retarget that mechanically executable centre toward this solver solution.
    """
    if not isinstance(solver_inverse, dict):
        return False
    try:
        if int(solver_inverse["admitted_count"]) <= 0:
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_shapes = {
        "solver_racket_velocity_w_mps": (3,),
        "solver_signed_face_w": (3,),
        "incoming_velocity_w_mps": (3,),
        "landing_aim_w_xy_m": (2,),
    }
    for name, shape in expected_shapes.items():
        try:
            value = np.asarray(solver_inverse[name], np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        if value.shape != shape or not bool(np.all(np.isfinite(value))):
            return False
    return True


def _smooth_with_contact_taper(q, ready, hit, window, savgol_filter):
    smooth = savgol_filter(q, window_length=window, polyorder=3, axis=0, mode="interp")
    # Far from contact, favour the dynamically stable ready posture.  At the
    # three-frame contact window retain the optimized kinematics exactly.
    distance = np.abs(np.arange(len(q)) - int(hit))
    contact_weight = np.clip(1.0 - (distance - 1.0) / max(1.0, window), 0.0, 1.0)
    style_weight = 0.25 * (1.0 - contact_weight)
    out = (
        contact_weight[:, None] * q
        + (1.0 - contact_weight)[:, None] * smooth
    )
    out = (1.0 - style_weight[:, None]) * out + style_weight[:, None] * ready[None, :]
    out[hit - 1 : hit + 2] = q[hit - 1 : hit + 2]
    return out


def _fk_path(mujoco, model, qbase, qadr, site_id, q):
    data = mujoco.MjData(model)
    sites, faces, longs = [], [], []
    for row in q:
        site, rotation = P1._fk(mujoco, model, data, qbase, qadr, row, site_id)
        sites.append(site)
        faces.append(rotation[:, 1])
        longs.append(rotation @ P1.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL)
    return np.asarray(sites), np.asarray(faces), np.asarray(longs)


def solve(args):
    import mujoco
    from scipy.signal import savgol_filter

    ready = P1._HIT._load_ready(args.dynamic_ready)
    document = ready["document"]["runtime_plant"]
    model = P1._load_model(mujoco, args.model)
    _jids, qadr, dadr, root_qadr, root_dadr = P1._runtime_mapping(
        mujoco, model, ready["names"]
    )
    site_id = P1._resolve_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, P1.SITE_NAME)
    qbase = np.asarray(model.qpos0, np.float64).copy()
    qbase[root_qadr : root_qadr + 3] = ready["root_pos"]
    qbase[root_qadr + 3 : root_qadr + 7] = ready["root_wxyz"]
    qbase[qadr] = ready["ready"]
    with np.load(args.phase1_npz, allow_pickle=False) as seed:
        q_seed = np.asarray(seed["q_ref"], np.float64)
    with np.load(args.motion, allow_pickle=False) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        old_site = np.asarray(motion["measured_racket_site_pos_w"], np.float64)
        old_face = np.asarray(motion["measured_racket_normal_w"], np.float64)
        old_long = np.asarray(motion["measured_racket_long_axis_w"], np.float64)
        mount_sign = float(
            np.asarray(motion["measured_racket_robot_mount_normal_sign"]).reshape(-1)[0]
        )
    if q_seed.shape != (len(old_site), len(ready["names"])):
        raise P1.ProducerError("Phase-1 seed shape differs from authoritative motion/runtime")

    kp = np.asarray(document["joint_stiffness"], np.float64)
    kd = np.asarray(document["joint_damping"], np.float64)
    effort = np.asarray(document["joint_effort_limits"], np.float64)
    qdes_lo = np.asarray(document["executed_qdes_lower_rad"], np.float64) + P1.STRICT_EPS_RAD
    qdes_hi = np.asarray(document["executed_qdes_upper_rad"], np.float64) - P1.STRICT_EPS_RAD
    table_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("court_table_top", "court_net")
    ], np.int64)
    robot_ids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith("court_") and "ball" not in name.lower() and int(model.geom_bodyid[gid]) != 0:
            robot_ids.append(gid)

    candidates = []
    wrist_index = ready["names"].index("right_wrist_yaw_joint")
    for window in args.windows:
        if window % 2 != 1 or window < 5:
            raise P1.ProducerError("smoothing windows must be odd and >=5")
        base_q = _smooth_with_contact_taper(
            q_seed, ready["ready"], args.hit_frame, window, savgol_filter
        )
        distance = np.abs(np.arange(len(base_q)) - int(args.hit_frame))
        inward_taper = np.clip(1.0 - distance / max(2.0, float(window)), 0.0, 1.0)
        for wrist_bias in args.wrist_inward_biases:
            q = base_q.copy()
            q[:, wrist_index] += float(wrist_bias) * inward_taper
            q[:, wrist_index] = np.clip(
                q[:, wrist_index], ready["lower"][wrist_index], ready["upper"][wrist_index]
            )
            sites, faces, longs = _fk_path(mujoco, model, qbase, qadr, site_id, q)
            faces *= mount_sign
            for timewarp in args.timewarps:
                plant = P1._plant_eval(
                    mujoco=mujoco, model=model, qbase=qbase, qadr=qadr, dadr=dadr,
                    root_dadr=root_dadr, q_ref=q, fps=fps, timewarp=timewarp,
                    kp=kp, kd=kd, effort=effort, qdes_lower=qdes_lo,
                    qdes_upper=qdes_hi, table_geom_ids=table_ids,
                    robot_geom_ids=robot_ids,
                )
                velocity = np.gradient(sites, float(timewarp) / fps, axis=0)
                hit = args.hit_frame
                row = {
                    "window": int(window),
                    "wrist_inward_bias_rad": float(wrist_bias),
                    "timewarp": float(timewarp),
                    "q_ref": q,
                    "sites": sites,
                    "faces": faces,
                    "longs": longs,
                    "velocity": velocity,
                    "plant": plant,
                    "center": {
                        "site_w_m": sites[hit].tolist(),
                        "velocity_w_mps": velocity[hit].tolist(),
                        "signed_face_w": faces[hit].tolist(),
                        "canonical_site_error_m": float(np.linalg.norm(sites[hit] - old_site[hit])),
                        "canonical_face_error_deg": float(P1._angle_deg(faces[hit], old_face[hit])),
                        "canonical_long_error_deg": float(P1._angle_deg(longs[hit], old_long[hit])),
                    },
                }
                row["mechanically_admitted"] = bool(
                    plant["qdes_margin_min"] >= 0
                    and plant["torque_margin_min"] >= 0
                    and plant["table_distance_min_m"] >= P1.TABLE_CLEARANCE_M - P1.NUMERIC_TOL_M
                    and plant["bilateral_support_frame_fraction"] >= 1.0
                )
                candidates.append(row)

    admitted = [row for row in candidates if row["mechanically_admitted"]]
    chosen = min(
        admitted or candidates,
        key=lambda row: (
            not row["mechanically_admitted"],
            max(0.0, -row["plant"]["qdes_margin_min"]),
            max(0.0, -row["plant"]["torque_margin_min"]),
            row["center"]["canonical_site_error_m"],
        ),
    )
    solver_inverse = None
    if chosen["mechanically_admitted"]:
        solver_inverse = _fixed_solver_inverse_grid(chosen["center"], args)
    reject = []
    if not chosen["mechanically_admitted"]:
        reject.append("NO_MECHANICALLY_EXECUTABLE_CENTER")
    elif not solver_inverse["velocity_matched"]:
        reject.append("FIXED_ACTION_SOLVER_VELOCITY_HAS_NO_MATCH_IN_REGISTERED_SEARCH")
    seed_valid = bool(
        chosen["mechanically_admitted"] and _solver_seed_valid(solver_inverse)
    )

    def plant_summary(plant):
        return {
            key: value for key, value in plant.items()
            if key not in ("qdot", "qdd", "tau", "qdes_ff")
        }

    report = {
        "schema_version": 1,
        "kind": KIND,
        "diagnostic_unauthorized": True,
        "seed_valid": seed_valid,
        "seed_typed_reject_reasons": [] if seed_valid else [
            "NO_MECHANICALLY_EXECUTABLE_SOLVER_ADMITTED_RETARGET_SEED"
        ],
        "matched": bool(
            solver_inverse is not None and solver_inverse.get("matched", False)
        ),
        "admitted": bool(
            chosen["mechanically_admitted"]
            and solver_inverse is not None
            and solver_inverse["velocity_matched"]
        ),
        "mechanically_admitted": bool(chosen["mechanically_admitted"]),
        "typed_reject_reasons": reject,
        "selected": {
            "window": chosen["window"],
            "wrist_inward_bias_rad": chosen["wrist_inward_bias_rad"],
            "timewarp": chosen["timewarp"],
            "center": chosen["center"],
            "plant": plant_summary(chosen["plant"]),
        },
        "fixed_action_solver_inverse": solver_inverse,
        "timing": {
            "reference_t_hit_s": args.reference_t_hit_s,
            "reference_t_cycle_s": args.reference_t_cycle_s,
            "selected_timewarp": chosen["timewarp"],
            "retarget_t_hit_s": args.reference_t_hit_s * chosen["timewarp"],
            "retarget_t_cycle_s": args.reference_t_cycle_s * chosen["timewarp"],
        },
        "candidate_summaries": [
            {
                "window": row["window"], "timewarp": row["timewarp"],
                "wrist_inward_bias_rad": row["wrist_inward_bias_rad"],
                "mechanically_admitted": row["mechanically_admitted"],
                "center": row["center"], "plant": plant_summary(row["plant"]),
            }
            for row in candidates
        ],
        "inputs": {
            name: {"path": str(path), "sha256": P1._sha256(path)}
            for name, path in (
                ("dynamic_ready", args.dynamic_ready), ("motion", args.motion),
                ("plant", args.model), ("phase1_seed", args.phase1_npz),
                ("ball_physics", args.ball_physics),
            )
        },
        "non_claims": [
            "not a production motion", "not a shared-solver task",
            "not training or deployment evidence",
        ],
    }
    arrays = {
        "q_ref": chosen["q_ref"].astype(np.float32),
        "qdot": chosen["plant"]["qdot"].astype(np.float32),
        "qdd": chosen["plant"]["qdd"].astype(np.float32),
        "tau": chosen["plant"]["tau"].astype(np.float32),
        "qdes_ff": chosen["plant"]["qdes_ff"].astype(np.float32),
        "racket_site": chosen["sites"].astype(np.float32),
        "racket_velocity": chosen["velocity"].astype(np.float32),
        "racket_face": chosen["faces"].astype(np.float32),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    report["artifact_payloads"] = {
        "npz_sha256": hashlib.sha256(buffer.getvalue()).hexdigest()
    }
    P1._write_no_replace(args.output_npz, buffer.getvalue())
    P1._write_no_replace(args.output_report, P1._json_bytes(report))
    return report


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--dynamic-ready", type=Path, required=True)
    result.add_argument("--motion", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--phase1-npz", type=Path, required=True)
    result.add_argument("--output-report", type=Path, required=True)
    result.add_argument("--output-npz", type=Path, required=True)
    result.add_argument("--hit-frame", type=int, default=48)
    result.add_argument("--windows", type=int, nargs="+", default=[5, 7, 9, 11, 15, 21])
    result.add_argument("--timewarps", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75, 2.0])
    result.add_argument(
        "--wrist-inward-biases", type=float, nargs="+",
        default=[0.0, 0.02, 0.04, 0.06, 0.08],
    )
    result.add_argument("--ball-physics", type=Path, default=Path("configs/ball_physics_venue.yaml"))
    result.add_argument("--ball-speeds", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0, 6.0])
    result.add_argument("--ball-direction-y", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    result.add_argument("--ball-direction-z", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    result.add_argument("--landing-aim-x", type=float, nargs="+", default=[2.1, 2.5, 3.0])
    result.add_argument("--landing-aim-y", type=float, nargs="+", default=[-0.6, 0.0, 0.6])
    result.add_argument("--solver-velocity-tolerance-mps", type=float, default=0.10)
    result.add_argument("--solver-face-tolerance-deg", type=float, default=10.0)
    result.add_argument("--surface-z", type=float, default=0.78)
    result.add_argument("--net-x", type=float, default=1.87)
    result.add_argument("--net-top-z", type=float, default=0.9325)
    result.add_argument("--reference-t-hit-s", type=float, default=0.96)
    result.add_argument("--reference-t-cycle-s", type=float, default=1.12)
    return result


def main():
    report = solve(parser().parse_args())
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
