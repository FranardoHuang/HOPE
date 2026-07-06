"""Ball-only trajectory prediction validation gate.

This answers the deploy-side question: if no opponent-racket data is available,
how well can we predict the rest of the ball trajectory after observing only a
short post-hit ball track?

For each racket strike with a first-bounce ground truth (observed bounce p_c or a
terminal-window reconstruction), score state estimates built from ball-only
windows ending at:

  post30 / post60 / post100 / post150 ms after contact
  net+20ms: 20 ms after the observed x=0 net/centerline crossing

For every scored prediction, integrate the full flight and report landing,
trajectory-position error, trajectory-velocity error, net-crossing height/time,
and spin speed/direction stability. The model intentionally ignores paddle data.

Usage:
  python trajectory_prediction_gate.py --yaml ../../configs/ball_physics_venue.yaml
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from ballcore import G_NOM, R_BALL, TAKES, load_take, smooth_vel, spin_from_quats
from predict_check import terminal_window_gt
from stage1_segments import prop_state, window_fit
from stage2_fits import in_split
from validate_stage4 import (
    TABLE_HALF_L,
    TABLE_HALF_W,
    integrate_to_table,
    load_yaml_constants,
    pair_strike_bounce,
)

ANA = paths.ANALYSIS
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.abspath(os.path.join(HERE, "..", "..", "configs", "ball_physics_venue.yaml"))
NET_HEIGHT_M = 0.1525


def rk4_step(p, v, w, kd, km, dt):
    g = np.array([0.0, 0.0, -G_NOM])

    def acc(vv):
        return g - kd * np.linalg.norm(vv) * vv + km * np.cross(w, vv)

    a1 = acc(v)
    v2 = v + 0.5 * dt * a1; a2 = acc(v2)
    v3 = v + 0.5 * dt * a2; a3 = acc(v3)
    v4 = v + dt * a3; a4 = acc(v4)
    p_new = p + (dt / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
    v_new = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return p_new, v_new


def integrate_at_times(p0, v0, w, kd, km, t_rel, max_step=0.002):
    """Return predicted positions/velocities at sorted relative times."""
    t_rel = np.asarray(t_rel, float)
    order = np.argsort(t_rel)
    p = np.asarray(p0, float).copy()
    v = np.asarray(v0, float).copy()
    w = np.asarray(w, float)
    t_cur = 0.0
    P = np.empty((len(t_rel), 3))
    V = np.empty((len(t_rel), 3))
    for oi in order:
        target = float(t_rel[oi])
        while t_cur + 1e-12 < target:
            dt = min(max_step, target - t_cur)
            p, v = rk4_step(p, v, w, kd, km, dt)
            t_cur += dt
        P[oi] = p
        V[oi] = v
    return P, V


def integrate_to_x_plane(p0, v0, w, kd, km, x_plane=0.0, t_max=2.0, dt=0.001):
    p = np.asarray(p0, float).copy()
    v = np.asarray(v0, float).copy()
    w = np.asarray(w, float)
    t = 0.0
    s0 = p[0] - x_plane
    while t < t_max:
        p_new, v_new = rk4_step(p, v, w, kd, km, dt)
        s1 = p_new[0] - x_plane
        if (s0 == 0.0 and t > 0.0) or (s0 * s1 <= 0.0 and abs(s0 - s1) > 1e-12):
            f = abs(s0) / max(abs(s0 - s1), 1e-12)
            return p + f * (p_new - p), v + f * (v_new - v), t + f * dt
        p, v, s0, t = p_new, v_new, s1, t + dt
    return None, None, None


def contiguous_runs(idxs):
    if len(idxs) == 0:
        return []
    cuts = np.where(np.diff(idxs) > 1)[0] + 1
    return np.split(idxs, cuts)


def best_observation_run(take, t_start, t_end, min_frames=8, min_duration=0.018):
    t = take["t"]
    pos = take["ball_pos_t_m"].astype(float)
    ok = take["ball_present"].astype(bool) & np.isfinite(pos[:, 0])
    idx = np.where(ok & (t >= t_start) & (t <= t_end))[0]
    candidates = []
    for run in contiguous_runs(idx):
        if len(run) < min_frames:
            continue
        dur = float(t[run[-1]] - t[run[0]])
        if dur < min_duration:
            continue
        candidates.append(run)
    if not candidates:
        return None
    # Use the latest complete run available by this deadline.
    return max(candidates, key=lambda r: (take["t"][r[-1]], len(r)))


def state_from_ball_window(take, t_start, t_end, kd, km, min_frames=8, min_duration=0.018):
    run = best_observation_run(
        take, t_start, t_end, min_frames=min_frames, min_duration=min_duration)
    if run is None:
        return None
    t = take["t"]
    pos = take["ball_pos_t_m"].astype(float)
    quat = take["ball_quat_xyzw"].astype(float)
    _, _, w_rob = spin_from_quats(quat[run], float(take["rate"]), R_table=take["table_R"])
    w = w_rob if np.isfinite(w_rob).all() else np.zeros(3)
    fit = window_fit(t[run], pos[run], w, kd=kd, km=km)
    t_pred = float(t[run[-1]])
    p, v = prop_state(fit, t_pred)
    return dict(
        p=p, v=v, w=w, t=t_pred, t0=float(t[run[0]]),
        n_frames=int(len(run)), obs_ms=float((t[run[-1]] - t[run[0]]) * 1e3),
        fit_rms_mm=float(fit["rms"] * 1e3),
    )


def observed_x_crossing(take, t0, t1, x_plane=0.0):
    t = take["t"]
    pos = take["ball_pos_t_m"].astype(float)
    ok = take["ball_present"].astype(bool) & np.isfinite(pos[:, 0])
    idx = np.where(ok & (t >= t0) & (t <= t1))[0]
    for run in contiguous_runs(idx):
        if len(run) < 2:
            continue
        p = pos[run]
        x = p[:, 0] - x_plane
        for j in range(len(run) - 1):
            if x[j] == 0.0 or x[j] * x[j + 1] <= 0.0:
                denom = x[j] - x[j + 1]
                f = 0.0 if abs(denom) < 1e-12 else x[j] / denom
                return dict(t=float(t[run[j]] + f * (t[run[j + 1]] - t[run[j]])),
                            p=p[j] + f * (p[j + 1] - p[j]))
    return None


def observed_future_states(take, t0, t1):
    t = take["t"]
    pos = take["ball_pos_t_m"].astype(float)
    ok = take["ball_present"].astype(bool) & np.isfinite(pos[:, 0])
    idx = np.where(ok & (t >= t0) & (t <= t1))[0]
    times, positions, velocities = [], [], []
    for run in contiguous_runs(idx):
        if len(run) < 9:
            continue
        sp, sv = smooth_vel(t[run], pos[run])
        times.append(t[run])
        positions.append(sp)
        velocities.append(sv)
    if not times:
        return None
    return dict(t=np.concatenate(times),
                p=np.vstack(positions),
                v=np.vstack(velocities))


def future_spin_estimate(take, t0, t1):
    t = take["t"]
    quat = take["ball_quat_xyzw"].astype(float)
    pos = take["ball_pos_t_m"].astype(float)
    ok = take["ball_present"].astype(bool) & np.isfinite(pos[:, 0]) & np.isfinite(quat[:, 0])
    idx = np.where(ok & (t >= t0) & (t <= t1))[0]
    best = None
    for run in contiguous_runs(idx):
        if len(run) >= 8 and (best is None or len(run) > len(best)):
            best = run
    if best is None:
        return None
    _, _, w = spin_from_quats(quat[best], float(take["rate"]), R_table=take["table_R"])
    return w if np.isfinite(w).all() else None


def angle_deg(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def build_ground_truth(strikes, bounces, kd, km, surface_z):
    pairs = pair_strike_bounce(strikes, bounces, kd, km, surface_z)
    paired_keys = {(pr["strike"]["take"], pr["strike"]["t_c"]) for pr in pairs}
    takes_cache = {}
    gt_records = [
        dict(strike=pr["strike"], gt_xy=np.array(pr["bounce"]["p_c"])[:2],
             t_land=pr["bounce"]["t_c"], gt_source="observed_bounce")
        for pr in pairs
    ]

    events_by_take = {}
    for s in strikes:
        events_by_take.setdefault(s["take"], []).append(s["t_c"])
    for b in bounces:
        events_by_take.setdefault(b["take"], []).append(b["t_c"])

    for s in strikes:
        if (s["take"], s["t_c"]) in paired_keys:
            continue
        if s["take"] not in takes_cache:
            takes_cache[s["take"]] = load_take(s["take"])
        take = takes_cache[s["take"]]
        gt = terminal_window_gt(take, take["table_R"], s["t_c"], float(take["rate"]),
                                kd, km, surface_z)
        if gt is None:
            continue
        if any(s["t_c"] + 0.01 < te < gt["t_land"] - 0.01
               for te in events_by_take.get(s["take"], [])):
            continue
        w_meas = np.array(s["w_out"], float) if s["spin_out_ok"] else np.zeros(3)
        p_l, _, t_l = integrate_to_table(s["ball_p"], s["v_out"], w_meas, kd, km, surface_z)
        if p_l is None:
            continue
        if (np.linalg.norm(p_l[:2] - gt["p_land"][:2]) > 0.40
                or abs((s["t_c"] + t_l) - gt["t_land"]) > 0.12):
            continue
        gt_records.append(dict(strike=s, gt_xy=gt["p_land"][:2],
                               t_land=gt["t_land"], gt_source="terminal_window"))
    return gt_records


def score_prediction(take, rec, mode, state, kd, km, surface_z):
    s = rec["strike"]
    gt_xy = np.asarray(rec["gt_xy"], float)
    t_land = float(rec["t_land"])
    p_l, v_l, t_l = integrate_to_table(state["p"], state["v"], state["w"], kd, km, surface_z)
    if p_l is None:
        return None
    row = dict(
        mode=mode, take=s["take"], t_strike=s["t_c"], t_pred=state["t"],
        ms_after_strike=float((state["t"] - s["t_c"]) * 1e3),
        obs_ms=state["obs_ms"], n_frames=state["n_frames"],
        fit_rms_mm=state["fit_rms_mm"], gt_source=rec["gt_source"],
        spin_revs=float(np.linalg.norm(state["w"]) / (2 * np.pi)),
        pred_landing=[float(x) for x in p_l[:2]],
        gt_landing=[float(x) for x in gt_xy],
        landing_err_m=float(np.linalg.norm(p_l[:2] - gt_xy)),
        landing_dt_ms=float(((state["t"] + t_l) - t_land) * 1e3),
        pred_landing_speed=float(np.linalg.norm(v_l)),
    )
    on_gt = bool(abs(gt_xy[0]) < TABLE_HALF_L and abs(gt_xy[1]) < TABLE_HALF_W)
    on_pred = bool(abs(p_l[0]) < TABLE_HALF_L and abs(p_l[1]) < TABLE_HALF_W)
    row["onoff_ok"] = bool(on_gt == on_pred)

    obs = observed_future_states(take, state["t"], min(t_land - 0.015, state["t"] + t_l))
    if obs is not None and len(obs["t"]):
        p_pred, v_pred = integrate_at_times(state["p"], state["v"], state["w"], kd, km,
                                            obs["t"] - state["t"])
        pe = np.linalg.norm(p_pred - obs["p"], axis=1)
        ve = np.linalg.norm(v_pred - obs["v"], axis=1)
        row.update(
            traj_n=int(len(pe)),
            traj_pos_med_mm=float(np.median(pe) * 1e3),
            traj_pos_p95_mm=float(np.percentile(pe, 95) * 1e3),
            traj_vel_med_mps=float(np.median(ve)),
            traj_vel_p95_mps=float(np.percentile(ve, 95)),
        )

    obs_net = observed_x_crossing(take, state["t"], t_land)
    p_net, v_net, t_net = integrate_to_x_plane(
        state["p"], state["v"], state["w"], kd, km, x_plane=0.0,
        t_max=max(0.05, t_land - state["t"] + 0.2))
    if obs_net is not None and p_net is not None and t_net <= t_l + 1e-6:
        row.update(
            net_obs_t=float(obs_net["t"]),
            net_pred_t=float(state["t"] + t_net),
            net_height_err_mm=float((p_net[2] - obs_net["p"][2]) * 1e3),
            net_dt_ms=float(((state["t"] + t_net) - obs_net["t"]) * 1e3),
            net_pred_center_clearance_m=float(p_net[2] - (surface_z + NET_HEIGHT_M + R_BALL)),
            net_pred_speed=float(np.linalg.norm(v_net)),
        )

    w_future = future_spin_estimate(take, state["t"], min(t_land - 0.015, state["t"] + 0.25))
    if w_future is not None:
        row["spin_speed_err_revs"] = float(
            abs(np.linalg.norm(state["w"]) - np.linalg.norm(w_future)) / (2 * np.pi))
        ang = angle_deg(state["w"], w_future)
        if ang is not None:
            row["spin_axis_err_deg"] = ang
    return row


def summarize(rows, mode):
    rr = [r for r in rows if r["mode"] == mode]
    out = dict(n=len(rr))
    if not rr:
        return out

    def med_p90(key):
        vals = np.array([r[key] for r in rr if key in r and np.isfinite(r[key])], float)
        if not len(vals):
            return None
        return dict(n=int(len(vals)), median=float(np.median(vals)),
                    p90=float(np.percentile(vals, 90)))

    out.update(
        ms_after_strike=med_p90("ms_after_strike"),
        obs_ms=med_p90("obs_ms"),
        landing_err_m=med_p90("landing_err_m"),
        landing_dt_ms=med_p90("landing_dt_ms"),
        onoff_acc_pct=float(np.mean([r["onoff_ok"] for r in rr]) * 100.0),
        traj_pos_med_mm=med_p90("traj_pos_med_mm"),
        traj_pos_p95_mm=med_p90("traj_pos_p95_mm"),
        traj_vel_med_mps=med_p90("traj_vel_med_mps"),
        net_height_err_mm=med_p90("net_height_err_mm"),
        net_dt_ms=med_p90("net_dt_ms"),
        net_pred_center_clearance_m=med_p90("net_pred_center_clearance_m"),
        spin_speed_err_revs=med_p90("spin_speed_err_revs"),
        spin_axis_err_deg=med_p90("spin_axis_err_deg"),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--out", default=os.path.join(ANA, "fits", "trajectory_prediction_gate.json"))
    ap.add_argument("--post-ms", default="30,60,100,150",
                    help="comma-separated post-strike observation deadlines")
    args = ap.parse_args()

    Y = load_yaml_constants(args.yaml)
    kd, km = Y["flight.k_d"], Y["flight.k_m"]
    meta = json.load(open(os.path.join(ANA, "segments", "meta.json")))
    surface_z = meta["surface_z"]
    bounces = json.load(open(os.path.join(ANA, "segments", "bounces.json")))
    strikes = json.load(open(os.path.join(ANA, "segments", "strikes.json")))
    gt_records = build_ground_truth(strikes, bounces, kd, km, surface_z)
    takes_cache = {}
    post_ms = [int(x) for x in args.post_ms.split(",") if x.strip()]
    modes = [f"post{m}ms" for m in post_ms] + ["net+20ms"]

    rows = []
    for rec in gt_records:
        s = rec["strike"]
        if not in_split(f'{s["take"]}:{s["t_c"]:.3f}', args.split):
            continue
        if s["fit_rms_mm"] > 8.0:
            continue
        if s["take"] not in takes_cache:
            takes_cache[s["take"]] = load_take(s["take"])
        take = takes_cache[s["take"]]
        t_land = float(rec["t_land"])
        t_start = s["t_c"] + 0.008

        for ms in post_ms:
            t_end = s["t_c"] + ms / 1000.0
            if t_end >= t_land - 0.020:
                continue
            min_frames = max(5, int(0.4 * ms / 1000.0 * take["rate"]))
            min_duration = 0.010 if ms <= 30 else 0.018
            state = state_from_ball_window(take, t_start, t_end, kd, km,
                                           min_frames=min_frames,
                                           min_duration=min_duration)
            if state is None:
                continue
            row = score_prediction(take, rec, f"post{ms}ms", state, kd, km, surface_z)
            if row is not None:
                rows.append(row)

        net = observed_x_crossing(take, s["t_c"] + 0.012, t_land)
        if net is not None:
            t_end = min(net["t"] + 0.020, t_land - 0.020)
            if t_end > t_start:
                state = state_from_ball_window(take, t_start, t_end, kd, km, min_frames=12)
                if state is not None:
                    row = score_prediction(take, rec, "net+20ms", state, kd, km, surface_z)
                    if row is not None:
                        rows.append(row)

    summaries = {m: summarize(rows, m) for m in modes}
    rep = dict(
        params_source=args.yaml,
        split=args.split,
        ground_truth=dict(
            n_records=len(gt_records),
            n_by_source={k: sum(1 for r in gt_records if r["gt_source"] == k)
                         for k in ("observed_bounce", "terminal_window")},
        ),
        model=dict(kd=kd, km=km, net_height_m=NET_HEIGHT_M),
        summaries=summaries,
        rows=rows,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {
        "post30ms": "tab:red", "post60ms": "tab:orange",
        "post100ms": "tab:blue", "post150ms": "tab:green",
        "net+20ms": "tab:purple",
    }
    for mode in modes:
        vals = np.sort([r["landing_err_m"] for r in rows if r["mode"] == mode])
        if len(vals):
            ax[0].plot(vals, np.arange(1, len(vals) + 1) / len(vals),
                       label=f"{mode} n={len(vals)} med={np.median(vals)*100:.1f}cm",
                       color=colors.get(mode))
    ax[0].axvline(0.10, color="k", ls=":", lw=1)
    ax[0].set_xlabel("landing error (m)")
    ax[0].set_ylabel("CDF")
    ax[0].set_title("landing prediction from ball-only windows")
    ax[0].legend(fontsize=8)
    for mode in modes:
        vals = np.sort([r["traj_pos_med_mm"] for r in rows if r["mode"] == mode and "traj_pos_med_mm" in r])
        if len(vals):
            ax[1].plot(vals, np.arange(1, len(vals) + 1) / len(vals),
                       label=mode, color=colors.get(mode))
    ax[1].set_xlabel("per-event median trajectory position error (mm)")
    ax[1].set_ylabel("CDF")
    ax[1].set_title("future trajectory positions")
    ax[1].legend(fontsize=8)
    for mode in modes:
        vals = np.array([r["net_height_err_mm"] for r in rows if r["mode"] == mode and "net_height_err_mm" in r])
        if len(vals):
            ax[2].scatter([mode] * len(vals), vals, s=10, alpha=0.6, color=colors.get(mode))
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("predicted - observed net height (mm)")
    ax[2].set_title("net/centerline height error")
    ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    png = args.out.replace(".json", ".png")
    fig.savefig(png, dpi=120)

    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    print(f"-> {args.out}\n-> {png}")


if __name__ == "__main__":
    main()
