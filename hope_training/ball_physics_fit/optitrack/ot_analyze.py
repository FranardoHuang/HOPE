"""
Stage-0-style QA + content statistics for an OptiTrack ball take, reusing
ballcore (arc extraction, parabola / RK4 shooting fits, contact detection).

Spin caveat: these takes have no orientation channel, so all fits are
SPIN-BLIND. Per-arc parabola g and the joint kd+g fit absorb Magnus (expect
|g| a few % high on topspin-heavy play); the venue |g|-9.81 <= 0.05 gate does
NOT apply. Contact-detector 'hit' events are unreliable at the ~5 mm
surface-wander noise (use the vx-reversal stroke count instead).

Auto table calibration: the surface height is found from the mode of bounce
local-minima (no table markers needed); bounce plane tilt / cloud extent and
axis alignment are reported from the bounce cloud.

Usage: python ot_analyze.py <take_npz> [--out-dir DIR] [--rally-gap 2.0]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ballcore as bc  # noqa: E402


def bounce_minima(t, pos, present, rate):
    """All local z-minima with a downward->upward vz reversal (no height gate);
    used to locate the table surface without prior calibration."""
    out = []
    for a0, b0 in bc.tracked_runs(t, present, rate):
        sl = slice(a0, b0 + 1)
        sp, sv = bc.smooth_vel(t[sl], pos[sl])
        z = sp[:, 2]
        for i in range(2, len(z) - 2):
            if (z[i] <= z[i - 1] and z[i] <= z[i + 1]
                    and sv[i - 1, 2] < -0.25 and sv[i + 1, 2] > 0.25):
                out.append((a0 + i, float(pos[a0 + i, 2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take_npz")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rally-gap", type=float, default=2.0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = args.out_dir or os.path.dirname(os.path.abspath(args.take_npz))
    os.makedirs(out, exist_ok=True)

    d = np.load(args.take_npz)
    t, pos = d["t"], d["ball_pos_t_m"].astype(float)
    present = d["ball_present"].astype(bool)
    rate = float(d["rate"])
    name = str(d["name"]) if "name" in d else "take"
    take = dict(t=t, ball_pos_t_m=pos, ball_quat_xyzw=np.full((len(t), 4), np.nan),
                ball_present=present, rate=rate, name=name)
    rep = dict(take=name, spin_channel=False)

    # ---- surface height from bounce-minima mode ----
    bm = bounce_minima(t, pos, present, rate)
    zb = np.array([z for _, z in bm])
    if len(zb) >= 10:
        hist, edges = np.histogram(zb, bins=np.arange(zb.min() - 0.05,
                                                      zb.max() + 0.05, 0.02))
        mode_c = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
        near = zb[np.abs(zb - mode_c) < 0.06]
        surface_z = float(np.median(near)) - bc.R_BALL
    else:
        surface_z = 0.0
    rep["surface_z_m"] = round(surface_z, 4)

    # ---- arcs ----
    arcs = bc.extract_arcs(take, min_frames=20, table_z=surface_z)
    pars = [bc.arc_parabola(a) for a in arcs]
    bmask = [bc.arc_is_ballistic(a, p) for a, p in zip(arcs, pars)]
    bal = [a for a, b in zip(arcs, bmask) if b]
    bal_p = [p for p, b in zip(pars, bmask) if b]
    rep["coverage_pct"] = round(float(present.mean()) * 100, 1)
    rep["arcs"] = dict(
        n=len(arcs), n_ballistic=len(bal),
        dur_total_s=round(sum(float(a["t"][-1] - a["t"][0]) for a in arcs), 1),
        parab_rms_med_mm=round(float(np.median([p["rms"] for p in bal_p])) * 1e3, 2)
        if bal_p else None,
        parab_g_med=round(float(np.median([p["g"] for p in bal_p])), 3)
        if bal_p else None,
        parab_g_iqr=[round(float(np.percentile([p["g"] for p in bal_p], q)), 3)
                     for q in (25, 75)] if bal_p else None,
    )

    # ---- joint kd+g shooting fit (spin-blind) ----
    cands = sorted(bal, key=lambda a: -(a["t"][-1] - a["t"][0]))[:60]
    cands = [a for a in cands if (a["t"][-1] - a["t"][0]) >= 0.2]
    if len(cands) >= 5:
        fit = bc.fit_arcs_global(cands, fit=("kd", "g"))
        g = fit["g_vec"]
        gmag = float(np.linalg.norm(g))
        rep["gravity_spinblind"] = dict(
            n_arcs=len(cands), g_mag=round(gmag, 3), kd=round(float(fit["kd"]), 4),
            tilt_vs_z_deg=round(float(np.degrees(
                np.arccos(np.clip(-g[2] / gmag, -1, 1)))), 2),
            shoot_rms_med_mm=round(float(np.median(fit["rms_per_arc"])) * 1e3, 2),
            caveat="Magnus absorbed into kd/g; venue +-0.05 gate not applicable")

    # ---- table bounces via detect_contacts at calibrated surface ----
    tb, bounce_pts, zmins = [], [], []
    for a0, b0 in bc.tracked_runs(t, present, rate):
        sl = slice(a0, b0 + 1)
        sp, sv = bc.smooth_vel(t[sl], pos[sl])
        for i, kind in bc.detect_contacts(t[sl], sp, sv, rate, table_z=surface_z):
            if kind == "table":
                tb.append(a0 + i)
                bounce_pts.append(pos[a0 + i])
                zwin = pos[max(a0 + i - 4, a0):a0 + i + 5, 2]
                zwin = zwin[np.isfinite(zwin)]
                if len(zwin):
                    zmins.append(float(zwin.min()))
    bounce_pts = np.asarray(bounce_pts).reshape(-1, 3)
    rep["bounces"] = dict(n_table=len(tb))
    if len(bounce_pts) >= 10:
        zarr = np.array(zmins)
        A = np.column_stack([bounce_pts[:, 0], bounce_pts[:, 1],
                             np.ones(len(bounce_pts))])
        coef, *_ = np.linalg.lstsq(A, zarr, rcond=None)
        nvec = np.array([-coef[0], -coef[1], 1.0])
        nvec /= np.linalg.norm(nvec)
        xy = bounce_pts[:, :2]
        c = xy.mean(0)
        _, _, Vt = np.linalg.svd(xy - c)
        # evaluate the plane at the cloud center (intercept-at-origin would add
        # a tilt-times-lever-arm artifact)
        z_at_c = float(coef[0] * c[0] + coef[1] * c[1] + coef[2])
        rep["bounce_plane"] = dict(
            tilt_deg=round(float(np.degrees(np.arccos(nvec[2]))), 3),
            z_at_center_vs_surface_mm=round(
                (z_at_c - surface_z - bc.R_BALL) * 1e3, 1))
        rep["bounce_cloud"] = dict(
            center=[round(float(v), 3) for v in c],
            x_pctl_5_95=[round(float(np.percentile(xy[:, 0], q)), 3) for q in (5, 95)],
            y_pctl_5_95=[round(float(np.percentile(xy[:, 1], q)), 3) for q in (5, 95)],
            principal_axis_deg_vs_x=round(float(np.degrees(
                np.arctan2(Vt[0, 1], Vt[0, 0]))), 1))

    # ---- rallies ----
    runs = bc.tracked_runs(t, present, rate)
    rallies, cur = [], None
    for a0, b0 in runs:
        if cur is None or t[a0] - t[cur[1]] > args.rally_gap:
            if cur:
                rallies.append(tuple(cur))
            cur = [a0, b0]
        else:
            cur[1] = b0
    if cur:
        rallies.append(tuple(cur))
    rd = [float(t[b] - t[a]) for a, b in rallies]
    rep["rallies"] = dict(n=len(rallies), dur_med_s=round(float(np.median(rd)), 1),
                          dur_max_s=round(float(np.max(rd)), 1),
                          total_active_s=round(float(np.sum(rd)), 1))

    # ---- strokes: vx sign reversal (robust vs contact-detector 'hit') ----
    strokes = []
    for a0, b0 in runs:
        sl = slice(a0, b0 + 1)
        sp, sv = bc.smooth_vel(t[sl], pos[sl])
        vx = sv[:, 0]
        n = len(vx)
        if n < 30:
            continue
        vpad = np.pad(vx, 4, mode="edge")
        vf = np.array([np.nanmedian(vpad[i:i + 9]) for i in range(n)])
        sg = np.sign(vf)
        for i in range(5, n - 5):
            if sg[i - 1] * sg[i] < 0:
                pre = np.nanmedian(vf[max(0, i - 18):i - 3])
                post = np.nanmedian(vf[i + 3:i + 18])
                if (np.isfinite(pre) and np.isfinite(post)
                        and abs(pre) > 1.0 and abs(post) > 1.0 and pre * post < 0):
                    strokes.append(a0 + i)
    strokes = [s for k, s in enumerate(sorted(strokes))
               if k == 0 or (s - sorted(strokes)[k - 1]) / rate > 0.15]
    iv = np.diff(np.array(strokes)) / rate if len(strokes) > 1 else np.array([])
    iv = iv[iv < 5]
    rep["strokes"] = dict(n=len(strokes),
                          interval_med_s=round(float(np.median(iv)), 2)
                          if len(iv) else None,
                          note="occlusion undercounts; contact 'hit' events "
                               "unusable at this noise level")

    # ---- speed ----
    spd = []
    for a0, b0 in runs:
        sl = slice(a0, b0 + 1)
        _, sv = bc.smooth_vel(t[sl], pos[sl])
        spd.append(np.linalg.norm(sv, axis=1))
    spd = np.concatenate(spd) if spd else np.zeros(1)
    spd = spd[np.isfinite(spd)]
    rep["speed_mps"] = {q: round(float(np.percentile(spd, p)), 2)
                        for q, p in (("med", 50), ("p90", 90), ("p99", 99))}

    # ---- plots ----
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for k, lbl in ((2, "z"), (0, "x"), (1, "y")):
            a = ax[[2, 0, 1].index(k)]
            a.plot(t, pos[:, k], lw=0.3)
            a.set_ylabel(f"{lbl} (m)")
        ax[0].axhline(surface_z + bc.R_BALL, color="r", lw=0.5)
        ax[0].set_title(f"{name} ball trajectory")
        ax[2].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_overview.png"), dpi=110)
        plt.close(fig)
        if len(bounce_pts) >= 10:
            fig, a = plt.subplots(figsize=(8, 6))
            a.scatter(bounce_pts[:, 0], bounce_pts[:, 1], s=6, alpha=0.6)
            a.set_aspect("equal")
            a.set_xlabel("x (m)")
            a.set_ylabel("y (m)")
            a.set_title(f"{name} table-bounce points")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"{name}_bounce_xy.png"), dpi=110)
            plt.close(fig)
        if bal_p:
            fig, axs = plt.subplots(1, 3, figsize=(14, 4))
            axs[0].hist([p["g"] for p in bal_p], bins=30)
            axs[0].axvline(9.81, color="r")
            axs[0].set_title("per-arc parabola g")
            axs[1].hist([p["rms"] * 1e3 for p in bal_p], bins=30)
            axs[1].set_title("arc rms (mm)")
            axs[2].hist(spd, bins=60)
            axs[2].set_yscale("log")
            axs[2].set_title("speed (m/s)")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"{name}_qa_hists.png"), dpi=110)
            plt.close(fig)

    with open(os.path.join(out, f"{name}_analysis.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
