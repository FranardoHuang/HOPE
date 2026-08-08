#!/usr/bin/env python3
"""击球帧运动学证书 —— 不跑 rollout,不跑策略,不算动力学。

一句话:在老师自己指定的那一帧(strike frame)上,把老师的拍子状态摆出来,
问它和题目要求的接触状态差多少。

两张证书,都不经过任何 rollout:
  证书 1(够不够得到):老师击球帧的"选中胶面接触时的球心"离题目要求的接触点多远。
                      再拆成【法向】和【切向】——切向超过拍面内切圆就是打空。
  证书 2(打不打得进):把老师那一帧真实的拍面法向 + 拍面速度,连同题目的来球速度,
                      直接喂进仓库自己的接触冲量律 + 飞行 RK4,看过不过网、落哪儿。

复用(不自造第二套):
  - materialize_n1_contact_training_bundle._motion_state   击球帧状态提取(FK 已在 npz 里)
  - racket_contact_geometry                                 拍面几何/球心偏置
  - virtual_return_scorer                                   接触冲量律 + 飞行事件(Isaac 判分的 numpy 规范)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--motion-root", required=True,
                    help="覆盖 manifest 里的 motion 目录(用来跑 0807 盘)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    mdp = repo / ("hope_training/whole_body_tracking/source/whole_body_tracking/"
                  "whole_body_tracking/tasks/tracking/mdp")
    scripts = repo / "hope_training/whole_body_tracking/scripts"

    geom = _load("racket_contact_geometry", mdp / "racket_contact_geometry.py")
    bundle = _load("n1bundle", scripts / "materialize_n1_contact_training_bundle.py")
    scorer_mod = _load("virtual_return_scorer", scripts / "virtual_return_scorer.py")

    params = scorer_mod.load_venue_params(str(repo / "configs/ball_physics_venue.yaml"))

    # 训练用的虚拟台摆位(hope_commands._cq_planes 的三个面,别再推导一遍)
    NEAR_X = 0.5
    SURFACE_Z = 0.76
    TABLE_LEN = 2.74
    NET_X_TABLE = 1.37
    NET_HEIGHT = 0.1525
    HALF_W = 1.525 / 2.0
    contact_plane_z = SURFACE_Z + params.ball_radius
    net_x = NEAR_X + NET_X_TABLE
    far_x = NEAR_X + TABLE_LEN
    net_clear_z = SURFACE_Z + NET_HEIGHT + params.ball_radius

    CAPTURE_RADIUS = 0.095          # vb_capture_radius
    MIN_APPROACH = 0.3              # vb_min_approach_speed
    UN_MIN, UN_MAX = 1.4, 7.2       # PADDLE_NORMAL_SPEED_MIN/MAX_MPS
    TANGENTIAL_OK = geom.SAFE_BALL_CENTER_TANGENTIAL_RADIUS_M

    manifest = json.loads(Path(args.manifest).read_text())
    aim = manifest["landing_aim"]["center_w_xy_m"]
    motion_root = Path(args.motion_root).resolve()

    # 对照组:mocap 量到的真实出球(人自己那一拍打出去的球)。
    # 它跟机器人无关,用来把"老师的拍面复刻得对不对"和"题目本身有没有解"分开。
    src = json.loads((motion_root / "SOURCE_MANIFEST.json").read_text())
    units = {u["uid"].lower(): u for u in src["units"]}

    rows = []
    for action in manifest["actions"]:
        aid = action["action_id"]
        npz = motion_root / Path(action["motion_path"]).name
        row = {"action_id": aid, "family": action["family"]}
        try:
            with np.load(npz, allow_pickle=False) as arch:
                frame_count = int(np.asarray(arch["body_pos_w"]).shape[0])
            strike_frame = int(round(float(action["strike_phase"]) * (frame_count - 1)))
            state = bundle._motion_state(
                motion_path=npz, action=action, geometry=geom,
                scope="full", strike_frame=strike_frame,
            )
        except Exception as exc:                      # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        prof = action["ball_profile"]
        demanded_b = np.asarray(prof["contact_offset_center_b_yaw_m"], dtype=np.float64)
        teacher_ball_b = np.asarray(state["ball_contact_center_b_yaw_m"], dtype=np.float64)
        teacher_face_b = np.asarray(state["face_center_b_yaw_m"], dtype=np.float64)
        teacher_site_b = np.asarray(state["site_b_yaw_m"], dtype=np.float64)
        n_b = np.asarray(state["physical_normal_b"], dtype=np.float64)

        # ---------- 证书 1:够不够得到 ----------
        delta = demanded_b - teacher_ball_b
        dist = float(np.linalg.norm(delta))
        normal_off = float(np.dot(delta, n_b))
        tangential_off = float(np.linalg.norm(delta - normal_off * n_b))
        # 题目要求的球心相对老师【拍面中心】的偏置(打空判据要用拍面中心,不是球心)
        d_face = demanded_b - teacher_face_b
        face_normal_off = float(np.dot(d_face, n_b))
        face_tangential_off = float(np.linalg.norm(d_face - face_normal_off * n_b))

        # ---------- 全程最近接近(旁证,不是判据) ----------
        # 重跑一遍所有帧的 ball_contact_center,只取最小距离
        with np.load(npz, allow_pickle=False) as arch:
            names = tuple(str(v) for v in arch["body_names"])
            pos = np.asarray(arch["body_pos_w"], dtype=np.float64)
            quat = np.asarray(arch["body_quat_w"], dtype=np.float64)
        wi = names.index(bundle.RACKET_WRIST_BODY)
        ri = names.index(bundle.ROOT_BODY)
        rot = bundle._quat_to_rotation(quat[:, wi])
        site_all = pos[:, wi] + np.einsum("tij,j->ti", rot,
                                          np.asarray(geom.RACKET_SITE_OFFSET_WRIST_M))
        sign = int(action["mount_normal_sign"])
        face_all = site_all + np.einsum(
            "tij,j->ti", rot, np.asarray(geom.face_center_from_site_local(sign)))
        nrm_all = float(sign) * rot[:, :, 1]
        ball_all_w = face_all + float(geom.BALL_RADIUS_M) * nrm_all
        ball_all_b = bundle._to_ready_b_yaw(
            ball_all_w, ready_root_w=np.asarray(state["ready_root_w_m"]),
            ready_yaw_rad=float(state["ready_yaw_rad"]))
        d_all = np.linalg.norm(ball_all_b - demanded_b[None, :], axis=1)
        best_frame = int(np.argmin(d_all))

        # ---------- 证书 2:打不打得进 ----------
        speed = float(prof["incoming_speed_center_mps"])
        v_in_b = speed * np.asarray(prof["incoming_direction_center_b_yaw"], dtype=np.float64)
        spin_b = float(prof["spin_magnitude_center_radps"]) * np.asarray(
            prof["spin_direction_center_b_yaw"], dtype=np.float64)
        v_r_b = float(state["face_speed_mps"]) * np.asarray(
            state["face_velocity_hat_b"], dtype=np.float64)

        oriented = scorer_mod.orient_normal(n_b, v_in_b, v_r_b)
        approach = float(np.dot(v_r_b, oriented))
        u = v_in_b + np.cross(spin_b, -params.ball_radius * oriented) - v_r_b
        u_n = abs(float(np.dot(u, oriented)))

        # 世界系:机器人站在 base_spawn_center,朝向 = 该 clip 的 canonical ready yaw
        base_yaw = float(state["ready_yaw_rad"])
        spawn = np.asarray(prof["base_spawn_center_w_xy_m"], dtype=np.float64)
        root_w = np.array([spawn[0], spawn[1],
                           float(np.asarray(state["ready_root_w_m"])[2])])
        c, s = math.cos(base_yaw), math.sin(base_yaw)

        def to_w(v):
            return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

        contact_w = root_w + to_w(demanded_b)
        v_out_b, w_out_b = scorer_mod.predict_paddle_contact(
            v_in_b, v_r_b, n_b, spin_b, params)
        ev = scorer_mod.coarse_flight_events(
            contact_w, to_w(v_out_b), to_w(w_out_b), params,
            contact_plane_z=contact_plane_z, net_x=net_x, h=0.01, n_steps=100)

        in_bounds = bool(ev.landing_valid
                         and ev.landing_xy[0] > net_x and ev.landing_xy[0] <= far_x
                         and abs(ev.landing_xy[1]) <= HALF_W)
        net_ok = bool(ev.net_crossed and ev.net_z > net_clear_z)

        # ---------- 对照组:mocap 真实出球 ----------
        unit = units.get(aid)
        meas = {}
        if unit is not None:
            yaw0 = math.radians(float(unit["yaw_before_deg"]))
            cy, sy = math.cos(-yaw0), math.sin(-yaw0)
            vo = np.asarray(unit["v_out_fit_hope_ms"], dtype=np.float64)
            vo_b = np.array([cy * vo[0] - sy * vo[1], sy * vo[0] + cy * vo[1], vo[2]])
            ev_m = scorer_mod.coarse_flight_events(
                contact_w, to_w(vo_b), np.zeros(3), params,
                contact_plane_z=contact_plane_z, net_x=net_x, h=0.01, n_steps=100)
            m_in = bool(ev_m.landing_valid and ev_m.landing_xy[0] > net_x
                        and ev_m.landing_xy[0] <= far_x
                        and abs(ev_m.landing_xy[1]) <= HALF_W)
            meas = {
                "measured_out_speed_mps": float(np.linalg.norm(vo_b)),
                "measured_vs_predicted_out_deg": float(math.degrees(math.acos(max(-1.0, min(
                    1.0, float(np.dot(vo_b, v_out_b) / max(1e-12,
                        np.linalg.norm(vo_b) * np.linalg.norm(v_out_b)))))))),
                "measured_net_clear": bool(ev_m.net_crossed and ev_m.net_z > net_clear_z),
                "measured_landing_xy": [float(ev_m.landing_xy[0]), float(ev_m.landing_xy[1])],
                "measured_landing_in_bounds": m_in,
                "measured_landing_err_from_aim_m": float(np.linalg.norm(
                    np.asarray(ev_m.landing_xy) - np.asarray(aim)))
                    if ev_m.landing_valid else None,
                "measured_racket_site_speed_mps": float(
                    action["reference_racket_site_speed_mps"]),
            }

        row.update({
            "strike_frame": int(state["contact_frame"]),
            "frame_count": int(state["frame_count"]),
            "t_hit_s": float(state["motion_t_hit_s"]),
            "mount_normal_sign": sign,
            # 证书 1
            "reach_dist_m": dist,
            "reach_normal_off_m": normal_off,
            "reach_tangential_off_m": tangential_off,
            "face_center_to_demanded_tangential_m": face_tangential_off,
            "face_center_to_demanded_normal_m": face_normal_off,
            "within_capture_radius_0p095": dist < CAPTURE_RADIUS,
            "within_alignment_0p03": dist <= 0.03,
            "on_blade_tangential": face_tangential_off <= TANGENTIAL_OK,
            # 旁证:全程最近接近
            "traj_min_dist_m": float(d_all[best_frame]),
            "traj_min_frame": best_frame,
            "traj_min_frame_minus_strike": best_frame - int(state["contact_frame"]),
            # 证书 2
            "face_speed_mps": float(state["face_speed_mps"]),
            "approach_mps": approach,
            "u_n_mps": u_n,
            "approach_ok": approach > MIN_APPROACH,
            "u_n_in_envelope": UN_MIN <= u_n <= UN_MAX,
            "out_speed_mps": float(np.linalg.norm(v_out_b)),
            "net_crossed": bool(ev.net_crossed),
            "net_z_m": float(ev.net_z),
            "net_clear": net_ok,
            "landing_valid": bool(ev.landing_valid),
            "landing_xy": [float(ev.landing_xy[0]), float(ev.landing_xy[1])],
            "landing_in_bounds": in_bounds,
            "landing_err_from_aim_m": float(np.linalg.norm(
                np.asarray(ev.landing_xy) - np.asarray(aim))) if ev.landing_valid else None,
            "legal_return": bool(dist < CAPTURE_RADIUS and approach > MIN_APPROACH
                                 and net_ok and in_bounds),
        })
        row.update(meas)
        rows.append(row)

    ok = [r for r in rows if "error" not in r]
    def q(key, sel=None):
        vals = sorted(float(r[key]) for r in ok if r.get(key) is not None
                      and (sel is None or sel(r)))
        if not vals:
            return None
        def p(f):
            return vals[min(len(vals) - 1, int(round(f * (len(vals) - 1))))]
        return {"n": len(vals), "min": vals[0], "p25": p(.25), "median": p(.5),
                "p75": p(.75), "p95": p(.95), "max": vals[-1],
                "mean": sum(vals) / len(vals)}

    summary = {
        "label": args.label,
        "manifest": str(args.manifest),
        "motion_root": str(motion_root),
        "actions_total": len(rows),
        "actions_ok": len(ok),
        "errors": [r for r in rows if "error" in r],
        "thresholds": {
            "capture_radius_m": CAPTURE_RADIUS,
            "alignment_receipt_m": 0.03,
            "safe_tangential_radius_m": TANGENTIAL_OK,
            "min_approach_mps": MIN_APPROACH,
            "u_n_envelope_mps": [UN_MIN, UN_MAX],
        },
        "dist": {
            "reach_dist_m": q("reach_dist_m"),
            "reach_tangential_off_m": q("reach_tangential_off_m"),
            "reach_normal_off_m": q("reach_normal_off_m"),
            "face_center_to_demanded_tangential_m": q("face_center_to_demanded_tangential_m"),
            "traj_min_dist_m": q("traj_min_dist_m"),
            "traj_min_frame_minus_strike": q("traj_min_frame_minus_strike"),
            "face_speed_mps": q("face_speed_mps"),
            "approach_mps": q("approach_mps"),
            "u_n_mps": q("u_n_mps"),
            "landing_err_from_aim_m": q("landing_err_from_aim_m"),
            "measured_out_speed_mps": q("measured_out_speed_mps"),
            "measured_vs_predicted_out_deg": q("measured_vs_predicted_out_deg"),
            "measured_landing_err_from_aim_m": q("measured_landing_err_from_aim_m"),
            "measured_racket_site_speed_mps": q("measured_racket_site_speed_mps"),
        },
        "counts": {
            k: sum(1 for r in ok if r.get(k)) for k in (
                "within_capture_radius_0p095", "within_alignment_0p03",
                "on_blade_tangential", "approach_ok", "u_n_in_envelope",
                "net_crossed", "net_clear", "landing_valid",
                "landing_in_bounds", "legal_return",
                "measured_net_clear", "measured_landing_in_bounds")
        },
        "rows": rows,
    }
    Path(args.json_out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
