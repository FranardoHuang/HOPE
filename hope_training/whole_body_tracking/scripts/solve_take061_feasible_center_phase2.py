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
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_phase1", HERE / "solve_take061_stable_support_plant_feasible.py"
)
P1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P1)

KIND = "take061_stable_support_feasible_center_phase2_v1"


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
    # This phase may establish a mechanically executable action centre.  Ball
    # inversion remains fail-closed until the exact shared solver replays it.
    reject = []
    if not chosen["mechanically_admitted"]:
        reject.append("NO_MECHANICALLY_EXECUTABLE_CENTER")
    reject.append("FIXED_ACTION_BALL_INVERSION_NOT_YET_REPLAYED")

    def plant_summary(plant):
        return {
            key: value for key, value in plant.items()
            if key not in ("qdot", "qdd", "tau", "qdes_ff")
        }

    report = {
        "schema_version": 1,
        "kind": KIND,
        "diagnostic_unauthorized": True,
        "admitted": False,
        "mechanically_admitted": bool(chosen["mechanically_admitted"]),
        "typed_reject_reasons": reject,
        "selected": {
            "window": chosen["window"],
            "wrist_inward_bias_rad": chosen["wrist_inward_bias_rad"],
            "timewarp": chosen["timewarp"],
            "center": chosen["center"],
            "plant": plant_summary(chosen["plant"]),
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
    return result


def main():
    report = solve(parser().parse_args())
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
