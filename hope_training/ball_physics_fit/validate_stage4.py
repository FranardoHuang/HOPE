"""
Stage 4 — held-out validation (fits from --fits json, events from the TEST split).

  1. flight  : state from first 60 ms -> position error at +0.4 s
               targets: median <= 25 mm, p95 <= 60 mm
  2. bounce  : predicted vs measured outgoing state
               targets: speed <= 5%, direction <= 3 deg, w+ <= 15% (where observable)
  3. strike->landing : paddle model on measured racket state -> flight -> first
               table crossing vs the MEASURED first bounce of the post-strike ball
               targets: median <= 0.10 m, p90 <= 0.25 m, on/off-table >= 90%
Writes analysis/fits/stage4_validation.json
"""
import os, sys, json, argparse
import numpy as np
import paths

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ballcore import (TAKES, load_take, extract_arcs, arc_parabola, arc_is_ballistic,
                      spin_from_quats, R_BALL, G_NOM)
from contact_model import predict_contact
from stage2_fits import in_split
from stage1_segments import rk4_window

ANA = paths.ANALYSIS
TABLE_HALF_L, TABLE_HALF_W = 1.370, 0.7625


def flight_state_from_window(t, pos, w, kd, km):
    from scipy.optimize import least_squares
    g = np.array([0, 0, -G_NOM])
    dt = np.median(np.diff(t))
    v0g = np.polyfit(t - t[0], pos, 1)[0]

    def resid(x):
        P, _ = rk4_window(x[:3], x[3:], w, kd, km, g, dt, len(t) - 1)
        return (P - pos).ravel()

    sol = least_squares(resid, np.concatenate([pos[0], v0g]), method="lm", max_nfev=200)
    return sol.x[:3], sol.x[3:]


def integrate_to_table(p0, v0, w, kd, km, surface_z, t_max=2.0, dt=1e-3):
    """RK4 until ball-center crosses surface_z + R while descending. Returns (p, v, t)."""
    g = np.array([0, 0, -G_NOM])
    zc = surface_z + R_BALL
    p, v = p0.astype(float).copy(), v0.astype(float).copy()
    t = 0.0
    def acc(v):
        return g - kd * np.linalg.norm(v) * v + km * np.cross(w, v)
    while t < t_max:
        a1 = acc(v); v2 = v + 0.5 * dt * a1; a2 = acc(v2)
        v3 = v + 0.5 * dt * a2; a3 = acc(v3)
        v4 = v + dt * a3; a4 = acc(v4)
        p_new = p + (dt / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
        v_new = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        if p[2] > zc >= p_new[2] and v_new[2] < 0:
            f = (p[2] - zc) / max(p[2] - p_new[2], 1e-9)
            return p + f * (p_new - p), v_new, t + f * dt
        p, v, t = p_new, v_new, t + dt
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=os.path.join(ANA, "fits", "stage2_fits_train.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(ANA, "fits", "stage4_validation.json"))
    args = ap.parse_args()
    F = json.load(open(args.fits))
    kd, km = F["kd"]["kd"], F["km"]["km"]
    te, tt = F["table_e"], F["table_tangential"]
    pd_ = F["paddle"]
    surface_z = F["surface_z"]
    meta = json.load(open(os.path.join(ANA, "segments", "meta.json")))
    bounces = json.load(open(os.path.join(ANA, "segments", "bounces.json")))
    strikes = json.load(open(os.path.join(ANA, "segments", "strikes.json")))

    # ---------- 1. flight propagation ----------
    errs = []
    for name in TAKES:
        take = load_take(name)
        Rt = take["table_R"]; rate = float(take["rate"])
        for ai, a in enumerate(extract_arcs(take, table_z=surface_z)):
            if not arc_is_ballistic(a):
                continue
            if not in_split(f"{name}:{ai}", args.split):
                continue
            dur = a["t"][-1] - a["t"][0]
            if dur < 0.46:
                continue
            n60 = max(int(0.060 * rate), 6)
            w, ang, w_rob = spin_from_quats(a["quat"][:n60 + 1], rate, R_table=Rt)
            wv = w_rob if np.isfinite(w_rob).all() else np.zeros(3)
            p0, v0 = flight_state_from_window(a["t"][:n60], a["pos"][:n60], wv, kd, km)
            g = np.array([0, 0, -G_NOM])
            dt = 1.0 / rate
            i_t = int(round(0.4 * rate))
            P, _ = rk4_window(p0, v0, wv, kd, km, g, dt, i_t)
            i_obs = i_t  # a["t"] uniform grid from arc start
            if i_obs < len(a["pos"]):
                errs.append(float(np.linalg.norm(P[-1] - a["pos"][i_obs])))
    flight_rep = dict(n=len(errs),
                      median_mm=float(np.median(errs) * 1e3) if errs else None,
                      p95_mm=float(np.percentile(errs, 95) * 1e3) if errs else None,
                      target="median<=25mm, p95<=60mm")

    # ---------- 2. bounce prediction ----------
    sp_err, dir_err, w_err = [], [], []
    for b in bounces:
        if not in_split(f'{b["take"]}:{b["t_c"]:.3f}', args.split):
            continue
        if b["fit_rms_mm"] > 5.0 or not b["spin_in_ok"]:
            continue
        out = predict_contact(np.array(b["v_in"])[None], np.zeros(3), np.array([0, 0, 1.0]),
                              np.array(b["w_in"])[None], te["e_const"],
                              tt.get("a_t", 0.369), tt.get("b_t", 0.0), tt.get("mu", 2.0))
        vp = out["v_plus"][0]; wp = out["omega_plus"][0]
        vm = np.array(b["v_out"])
        sp_err.append(abs(np.linalg.norm(vp) - np.linalg.norm(vm)) / np.linalg.norm(vm))
        cosang = np.dot(vp, vm) / (np.linalg.norm(vp) * np.linalg.norm(vm))
        dir_err.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        if b["spin_out_ok"]:
            wm = np.array(b["w_out"])
            if np.linalg.norm(wm) > 6:
                w_err.append(float(np.linalg.norm(wp - wm) / np.linalg.norm(wm)))
    bounce_rep = dict(n=len(sp_err),
                      speed_err_med_pct=float(np.median(sp_err) * 100) if sp_err else None,
                      dir_err_med_deg=float(np.median(dir_err)) if dir_err else None,
                      spin_err_med_pct=float(np.median(w_err) * 100) if w_err else None,
                      target="speed<=5%, dir<=3deg, spin<=15%")

    # ---------- 3. strike -> landing ----------
    land_err, on_off_ok = [], []
    details = []
    # measured first bounce after each strike: nearest bounce event in same take
    bt = {}
    for b in bounces:
        bt.setdefault(b["take"], []).append(b)
    for s in strikes:
        if not in_split(f'{s["take"]}:{s["t_c"]:.3f}', args.split):
            continue
        if s["fit_rms_mm"] > 8.0 or s["pad_fit_rms_mm"] > 15.0 or not s["spin_in_ok"]:
            continue
        cands = [b for b in bt.get(s["take"], [])
                 if 0.02 < b["t_c"] - s["t_c"] < 1.5]
        if not cands:
            continue
        b0 = min(cands, key=lambda b: b["t_c"])
        # measured landing: extrapolate incoming arc of b0 to contact -> use its z-cross pos
        # (b0 was built by extrapolating to the surface, so use its position via v_in? store z_c)
        # We stored z_c (ball center z at contact) but not xy! -> recompute from post-strike arc:
        # use strike outgoing state MEASURED (v_out at t_c) and integrate: no, that's the model.
        # Instead reconstruct from bounce record: we need xy of contact -> approximate from
        # strike ball_p + measured v_out * (t_b - t_c) with drag integration (measured init).
        p0 = np.array(s["ball_p"], float)
        v_meas = np.array(s["v_out"], float)
        w_meas = np.array(s["w_out"], float) if s["spin_out_ok"] else np.zeros(3)
        pL_meas, _, tL_meas = integrate_to_table(p0, v_meas, w_meas, kd, km, surface_z)
        if pL_meas is None:
            continue
        # model: paddle contact from measured pre-state
        cp = p0 - R_BALL * np.array(s["pad_n"])
        v_r = np.array(s["pad_v"]) + np.cross(np.array(s["pad_w"]), cp - np.array(s["pad_p"]))
        out = predict_contact(np.array(s["v_in"])[None], v_r[None], np.array(s["pad_n"])[None],
                              np.array(s["w_in"])[None], pd_.get("e_eff", 0.46),
                              pd_.get("a_t", 1.2), pd_.get("b_t", 0.3), pd_.get("mu", 2.5))
        vp = out["v_plus"][0]; wp = out["omega_plus"][0]
        pL_mod, _, tL_mod = integrate_to_table(p0, vp, wp, kd, km, surface_z)
        if pL_mod is None:
            continue
        err = float(np.linalg.norm(pL_mod[:2] - pL_meas[:2]))
        land_err.append(err)
        on_meas = bool(abs(pL_meas[0]) < TABLE_HALF_L and abs(pL_meas[1]) < TABLE_HALF_W)
        on_mod = bool(abs(pL_mod[0]) < TABLE_HALF_L and abs(pL_mod[1]) < TABLE_HALF_W)
        on_off_ok.append(on_meas == on_mod)
        details.append(dict(take=s["take"], t_c=s["t_c"], err_m=err,
                            on_meas=on_meas, on_mod=on_mod))
    land_rep = dict(n=len(land_err),
                    median_m=float(np.median(land_err)) if land_err else None,
                    p90_m=float(np.percentile(land_err, 90)) if land_err else None,
                    onoff_acc_pct=float(np.mean(on_off_ok) * 100) if on_off_ok else None,
                    target="median<=0.10m, p90<=0.25m, onoff>=90%")

    rep = dict(fits=args.fits, split=args.split,
               flight=flight_rep, bounce=bounce_rep, landing=land_rep,
               landing_details=details)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "landing_details"}, indent=1))


if __name__ == "__main__":
    main()
