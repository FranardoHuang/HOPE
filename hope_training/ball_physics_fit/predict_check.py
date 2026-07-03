"""
Two-horizon landing-prediction check — the deploy-style accuracy probe.

For every racket strike that has an OBSERVED first bounce (continuity-gated
pairing, ground truth = the bounce contact point p_c from bounces.json):

  H0  at-contact, through-paddle : measured pre-contact ball + racket state ->
      paddle contact model -> flight model -> predicted landing.
      (what the planner/reward will do the instant the ball leaves the racket)
  H1  at-contact, measured-out   : measured post-contact ball state -> flight
      model -> landing. Isolates flight-model + state noise from paddle-model error.
  H2  near-landing (~100 ms lead): ball state re-fit on a short window ending
      ~30 ms before touchdown -> flight model -> landing.
      (the late-refinement prediction; should be the tightest)

Reports per-horizon planar landing error + timing error, per-take breakdown,
and writes analysis/fits/predict_check.json + predict_check.png.

Usage: python predict_check.py [--yaml configs/ball_physics_venue.yaml]
                               [--paddle-e exp|const|lin] [--split all]
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from ballcore import TAKES, load_take, spin_from_quats, R_BALL
from stage1_segments import window_fit
from stage2_fits import in_split
from validate_stage4 import (load_yaml_constants, integrate_to_table, paddle_outgoing,
                             pair_strike_bounce, ANA)


def h2_state(take, Rt, t_b, rate, kd, km, lead_end_s=0.017, lead_start_s=0.150):
    """Ball state shortly before touchdown: shooting fit on the window
    [t_b - lead_start, t_b - lead_end] (ends before the contact exclusion zone)."""
    t = take["t"]
    pos = take["ball_pos_t_m"].astype(float)
    quat = take["ball_quat_xyzw"].astype(float)
    ok = take["ball_present"].astype(bool)
    m = ok & (t >= t_b - lead_start_s) & (t <= t_b - lead_end_s)
    if m.sum() < 15:
        return None
    _, _, w_rob = spin_from_quats(quat[m], rate, R_table=Rt)
    w = w_rob if np.isfinite(w_rob).all() else np.zeros(3)
    f = window_fit(t[m], pos[m], w, kd=kd, km=km)
    t_end = float(t[m][-1])
    return dict(fit=f, w=w, t_end=t_end, lead_s=float(t_b - t_end))


def terminal_window_gt(take, Rt, t_strike, rate, kd, km, surface_z,
                       max_extrap_s=0.20, max_horizon_s=2.0):
    """Landing ground truth for strikes with NO recorded bounce: take the LAST
    tracked descending samples of the post-strike ball path and extrapolate the
    short remaining distance to the surface (same machinery as H2; the bounce
    extractor misses ~2/3 of first bounces because the post-bounce arc is
    occluded/short, but the incoming arc's terminal window is usually there).
    Returns dict(p_land(3), t_land, lead_s) or None."""
    t = take["t"]
    pos = take["ball_pos_t_m"].astype(float)
    quat = take["ball_quat_xyzw"].astype(float)
    ok = take["ball_present"].astype(bool) & np.isfinite(pos[:, 0])
    zc = surface_z + R_BALL
    lo = np.searchsorted(t, t_strike + 0.03)
    hi = np.searchsorted(t, t_strike + max_horizon_s)
    idx = np.where(ok[lo:hi])[0] + lo
    if len(idx) < 20:
        return None
    # i_end = the FIRST approach to the surface after the strike (z within 30 mm of
    # the contact height). If the track never gets that low, the first bounce is not
    # observable here — reject rather than risk latching onto a later arc.
    near = idx[pos[idx, 2] < zc + 0.030]
    if not len(near):
        return None
    i_end = near[0]
    # terminal fit window: last tracked frames strictly before i_end
    win = idx[(idx < i_end) & (t[idx] >= t[i_end] - 0.160)]
    if len(win) < 15:
        return None
    _, _, w_rob = spin_from_quats(quat[win], rate, R_table=Rt)
    w = w_rob if np.isfinite(w_rob).all() else np.zeros(3)
    f = window_fit(t[win], pos[win], w, kd=kd, km=km)
    v_end = np.polyfit(t[win] - t[win][0], pos[win], 1)[0]
    if v_end[2] > -0.3:                  # must be descending toward the table
        return None
    pL, _, tL = integrate_to_table(f["p0"], f["v0"], w, kd, km, surface_z, t_max=0.6)
    if pL is None:
        return None
    lead = float(f["t0"] + tL - t[win][-1])
    if lead > max_extrap_s:              # only trust SHORT extrapolations
        return None
    return dict(p_land=pL, t_land=float(f["t0"] + tL), lead_s=lead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--paddle-e", default="exp", choices=["const", "exp", "lin"])
    ap.add_argument("--fits", default=os.path.join(ANA, "fits", "stage2_fits_all.json"))
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--out", default=os.path.join(ANA, "fits", "predict_check.json"))
    args = ap.parse_args()

    if args.yaml:
        Y = load_yaml_constants(args.yaml)
        kd, km = Y["flight.k_d"], Y["flight.k_m"]
        pd_ = dict(mode=args.paddle_e, e_eff=Y["contact.paddle.e_eff"],
                   a_t=Y["contact.paddle.a_t"], b_t=Y["contact.paddle.b_t"],
                   mu=Y["contact.paddle.mu_safety"],
                   g1=Y.get("contact.paddle.e_exp_g1"), g2=Y.get("contact.paddle.e_exp_g2"),
                   a=Y.get("contact.paddle.e_lin_a"), b=Y.get("contact.paddle.e_lin_b"))
        src = f"yaml (paddle e={args.paddle_e})"
    else:
        F = json.load(open(args.fits))
        kd, km = F["kd"]["kd"], F["km"]["km"]
        p = F["paddle"]
        pd_ = dict(mode="const", e_eff=p["e_eff"], a_t=p["a_t"], b_t=p["b_t"], mu=p["mu"])
        src = f"fits:{os.path.basename(args.fits)}"

    meta = json.load(open(os.path.join(ANA, "segments", "meta.json")))
    surface_z = meta["surface_z"]
    bounces = json.load(open(os.path.join(ANA, "segments", "bounces.json")))
    strikes = json.load(open(os.path.join(ANA, "segments", "strikes.json")))

    pairs = pair_strike_bounce(strikes, bounces, kd, km, surface_z)
    paired_keys = {(pr["strike"]["take"], pr["strike"]["t_c"]) for pr in pairs}
    takes_cache = {}

    # ground-truth records: observed bounce p_c where paired, else terminal-window
    # extrapolation of the post-strike arc (recovers the ~2/3 of first bounces the
    # bounce extractor misses; lead <= 200 ms so it is H2-grade, mm-cm accurate)
    gt_records = [dict(strike=pr["strike"],
                       gt_xy=np.array(pr["bounce"]["p_c"])[:2],
                       t_land=pr["bounce"]["t_c"], gt_source="observed_bounce")
                  for pr in pairs]
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
        # identity gates: the GT must belong to THIS strike's ballistic segment.
        # (a) no other recorded contact between strike and touchdown
        if any(s["t_c"] + 0.01 < te < gt["t_land"] - 0.01
               for te in events_by_take.get(s["take"], [])):
            continue
        # (b) loose continuity with the measured out-state (same gates family as
        #     pair_strike_bounce; identity check, not an accuracy tune)
        w_meas = np.array(s["w_out"], float) if s["spin_out_ok"] else np.zeros(3)
        pL1, _, tL1 = integrate_to_table(s["ball_p"], s["v_out"], w_meas, kd, km, surface_z)
        if pL1 is None:
            continue
        if (np.linalg.norm(pL1[:2] - gt["p_land"][:2]) > 0.40
                or abs((s["t_c"] + tL1) - gt["t_land"]) > 0.12):
            continue
        gt_records.append(dict(strike=s, gt_xy=gt["p_land"][:2],
                               t_land=gt["t_land"], gt_source="terminal_window"))

    rows = []
    for rec in gt_records:
        s = rec["strike"]
        if not in_split(f'{s["take"]}:{s["t_c"]:.3f}', args.split):
            continue
        if s["fit_rms_mm"] > 8.0 or s["pad_fit_rms_mm"] > 15.0 or not s["spin_in_ok"]:
            continue
        gt_xy = rec["gt_xy"]
        t_land = rec["t_land"]
        row = dict(take=s["take"], t_strike=s["t_c"], t_land=t_land,
                   u_n=s["u_n"], gt=[float(g) for g in gt_xy],
                   gt_source=rec["gt_source"])

        # H0: through-paddle
        vp, wp = paddle_outgoing(s, pd_)
        pL, _, tL = integrate_to_table(s["ball_p"], vp, wp, kd, km, surface_z)
        if pL is not None:
            row["h0_err_m"] = float(np.linalg.norm(pL[:2] - gt_xy))
            row["h0_dt_ms"] = float(((s["t_c"] + tL) - t_land) * 1e3)

        # H1: measured out-state
        w_meas = np.array(s["w_out"], float) if s["spin_out_ok"] else np.zeros(3)
        pL1, _, tL1 = integrate_to_table(s["ball_p"], s["v_out"], w_meas, kd, km, surface_z)
        if pL1 is not None:
            row["h1_err_m"] = float(np.linalg.norm(pL1[:2] - gt_xy))
            row["h1_dt_ms"] = float(((s["t_c"] + tL1) - t_land) * 1e3)

        # H2: near-landing refinement
        if s["take"] not in takes_cache:
            takes_cache[s["take"]] = load_take(s["take"])
        take = takes_cache[s["take"]]
        st = h2_state(take, take["table_R"], t_land, float(take["rate"]), kd, km)
        if st is not None:
            f = st["fit"]
            pL2, _, tL2 = integrate_to_table(f["p0"], f["v0"], st["w"], kd, km, surface_z)
            if pL2 is not None:
                row["h2_err_m"] = float(np.linalg.norm(pL2[:2] - gt_xy))
                row["h2_dt_ms"] = float(((f["t0"] + tL2) - t_land) * 1e3)
                row["h2_lead_ms"] = float(st["lead_s"] * 1e3)
        rows.append(row)

    def stats(key):
        v = np.array([r[key] for r in rows if key in r])
        if not len(v):
            return dict(n=0)
        return dict(n=int(len(v)), median=float(np.median(v)),
                    p90=float(np.percentile(v, 90)), mean=float(v.mean()))

    rep = dict(params_source=src, split=args.split,
               n_pairs=len(rows),
               n_by_gt_source={k: sum(1 for r in rows if r["gt_source"] == k)
                               for k in ("observed_bounce", "terminal_window")},
               h0_through_paddle=stats("h0_err_m"),
               h1_measured_out=stats("h1_err_m"),
               h2_near_landing=stats("h2_err_m"),
               h0_timing_ms=stats("h0_dt_ms"), h1_timing_ms=stats("h1_dt_ms"),
               h2_timing_ms=stats("h2_dt_ms"),
               per_take={tk: dict(
                   n=len([r for r in rows if r["take"] == tk]),
                   h0=stats_take(rows, tk, "h0_err_m"),
                   h1=stats_take(rows, tk, "h1_err_m"),
                   h2=stats_take(rows, tk, "h2_err_m"))
                   for tk in sorted({r["take"] for r in rows})},
               rows=rows)
    json.dump(rep, open(args.out, "w"), indent=1)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for key, lbl, c in (("h0_err_m", "H0 at-contact (through paddle)", "tab:red"),
                        ("h1_err_m", "H1 at-contact (measured out)", "tab:orange"),
                        ("h2_err_m", "H2 near-landing (~100ms lead)", "tab:green")):
        v = np.sort([r[key] for r in rows if key in r])
        if len(v):
            ax[0].plot(v, np.arange(1, len(v) + 1) / len(v), label=f"{lbl} (med {np.median(v)*100:.0f} cm)", color=c)
    ax[0].axvline(0.10, color="k", ls=":", lw=1, label="0.10 m target")
    ax[0].set_xlabel("planar landing error (m)"); ax[0].set_ylabel("CDF")
    ax[0].set_xlim(0, 0.8); ax[0].legend(fontsize=8); ax[0].set_title("prediction error by horizon")
    for r in rows:
        if "h0_err_m" in r:
            ax[1].plot([r["gt"][0]], [r["gt"][1]], "k.", ms=4)
    ax[1].add_patch(plt.Rectangle((-1.37, -0.7625), 2.74, 1.525, fill=False, color="b", lw=1))
    ax[1].set_title("observed landings (table frame)"); ax[1].set_aspect("equal")
    ax[1].set_xlabel("X length (m)"); ax[1].set_ylabel("Y width (m)")
    un = [r["u_n"] for r in rows if "h0_err_m" in r]
    e0 = [r["h0_err_m"] for r in rows if "h0_err_m" in r]
    ax[2].scatter(un, e0, s=14, c="tab:red")
    ax[2].set_xlabel("|u_n| at racket (m/s)"); ax[2].set_ylabel("H0 landing error (m)")
    ax[2].set_title("through-paddle error vs contact speed")
    fig.tight_layout()
    png = args.out.replace(".json", ".png")
    fig.savefig(png, dpi=110)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "per_take")}, indent=1))
    print("per-take:", json.dumps(rep["per_take"], indent=1))
    print(f"-> {args.out}\n-> {png}")


def stats_take(rows, tk, key):
    v = [r[key] for r in rows if r["take"] == tk and key in r]
    return dict(n=len(v), median=float(np.median(v)) if v else None)


if __name__ == "__main__":
    main()
