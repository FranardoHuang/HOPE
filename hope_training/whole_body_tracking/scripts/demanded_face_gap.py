#!/usr/bin/env python3
"""题目要求的拍面 vs 老师自己的拍面 —— 差多少度。

只用闭式解算器 strike_spec_analytic.solve_analytic(题目 -> 拍子该摆成什么样),
不跑 rollout、不跑策略。现役 A211 走的是 LM 求解器(current_lm),这里用闭式解
做全库分布;take_061 的现役 LM 数(38.92 deg)是直接从活收据读出来的,两者互为佐证。
"""
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


repo = Path(sys.argv[1]).resolve()
reach = json.load(open(sys.argv[2]))
out_path = sys.argv[3]
mdp = repo / ("hope_training/whole_body_tracking/source/whole_body_tracking/"
              "whole_body_tracking/tasks/tracking/mdp")
scripts = repo / "hope_training/whole_body_tracking/scripts"
geom = _load("racket_contact_geometry", mdp / "racket_contact_geometry.py")
bundle = _load("n1bundle", scripts / "materialize_n1_contact_training_bundle.py")
sc = _load("virtual_return_scorer", scripts / "virtual_return_scorer.py")
an = _load("strike_spec_analytic", mdp / "strike_spec_analytic.py")

prm = sc.load_venue_params(str(repo / "configs/ball_physics_venue.yaml"))
manifest = json.loads((repo / "configs/action_ball_chingmu73_measured_v4_f10_20260803.json").read_text())
aim = manifest["landing_aim"]["center_w_xy_m"]
acts = {a["action_id"]: a for a in manifest["actions"]}
rows = {r["action_id"]: r for r in reach["rows"]}
mroot = Path(reach["motion_root"])

SURF_Z, NEAR_X = 0.76, 0.5
plane_z = SURF_Z + prm.ball_radius
net_x = NEAR_X + 1.37
net_top = SURF_Z + 0.1525 + prm.ball_radius

recs = []
for aid, a in acts.items():
    r = rows[aid]
    prof = a["ball_profile"]
    npz = mroot / Path(a["motion_path"]).name
    st = bundle._motion_state(motion_path=npz, action=a, geometry=geom,
                              scope="full", strike_frame=r["strike_frame"])
    yaw = float(st["ready_yaw_rad"])
    c, s = math.cos(yaw), math.sin(yaw)
    tow = lambda v: np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
    spawn = np.asarray(prof["base_spawn_center_w_xy_m"], float)
    root = np.array([spawn[0], spawn[1], float(np.asarray(st["ready_root_w_m"])[2])])
    contact_w = root + tow(np.asarray(prof["contact_offset_center_b_yaw_m"], float))
    v_in_w = tow(float(prof["incoming_speed_center_mps"])
                 * np.asarray(prof["incoming_direction_center_b_yaw"], float))
    w_in_w = np.zeros(3)
    n_teacher_w = tow(np.asarray(st["physical_normal_b"], float))
    v_teacher_w = tow(float(st["face_speed_mps"]) * np.asarray(st["face_velocity_hat_b"], float))

    T = lambda x: torch.tensor(np.asarray(x, float)[None, :], dtype=torch.float64)
    best = None
    for t_f in (0.50, 0.58, 0.66, 0.74, 0.82, 0.90):
        try:
            out = an.solve_analytic(T(contact_w), T(v_in_w), T(w_in_w),
                                    T(aim)[:, :2], prm, plane_z, net_x,
                                    t_flight=t_f, pin="min_speed",
                                    speed_budget=10.0, net_top_z=net_top)
        except Exception:
            continue
        if not bool(out["ok"][0]):
            continue
        spd = float(out["speed"][0])
        if best is None or spd < best[0]:
            best = (spd, np.asarray(out["n"][0].numpy(), float),
                    np.asarray(out["v_r"][0].numpy(), float), t_f,
                    str(out["reason"][0]) if "reason" in out else "")
    rec = {"action_id": aid, "family": a["family"]}
    if best is None:
        rec["solver_ok"] = False
    else:
        spd, n_d, v_d, t_f, _ = best
        ang = lambda p, q: math.degrees(math.acos(max(-1.0, min(1.0, float(
            np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))))))
        rec.update({
            "solver_ok": True, "t_flight_s": t_f,
            "demanded_face_vs_teacher_deg": ang(n_d, n_teacher_w),
            "demanded_vel_dir_vs_teacher_deg": ang(v_d, v_teacher_w),
            "demanded_speed_mps": spd,
            "teacher_face_speed_mps": float(st["face_speed_mps"]),
            "speed_ratio_demanded_over_teacher": spd / float(st["face_speed_mps"]),
        })
    recs.append(rec)

ok = [r for r in recs if r.get("solver_ok")]
def q(k):
    v = sorted(r[k] for r in ok)
    if not v:
        return None
    p = lambda f: v[min(len(v) - 1, int(round(f * (len(v) - 1))))]
    return {"n": len(v), "min": v[0], "p25": p(.25), "median": p(.5),
            "p75": p(.75), "p95": p(.95), "max": v[-1]}

summary = {"solved": len(ok), "total": len(recs), "aim_w_xy_m": aim,
           "dist": {k: q(k) for k in ("demanded_face_vs_teacher_deg",
                                      "demanded_vel_dir_vs_teacher_deg",
                                      "demanded_speed_mps",
                                      "speed_ratio_demanded_over_teacher")},
           "face_gap_gt_15deg": sum(1 for r in ok if r["demanded_face_vs_teacher_deg"] > 15.0),
           "rows": recs}
Path(out_path).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
