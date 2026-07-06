"""Audit strike-factor coverage and effect sizes.

This is a companion to the F1-F8 falsification battery. It answers the practical
model-scope questions that are not single assumptions:

  - how much the observed spin changes the predicted landing on this dataset
  - whether front/back racket side can be tested from the recorded labels
  - coverage for contact position, incidence angle, and racket velocity
  - how much error remains in the rigid instantaneous paddle-contact model
  - where net/full-trajectory prediction is handled

The script does not refit constants. It scores the shipped yaml against the same
observed/terminal-window landing ground truth used by predict_check.py.

Usage:
  python strike_factor_audit.py --yaml ../../configs/ball_physics_venue.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from ballcore import R_BALL
from stage2_fits import in_split
from trajectory_prediction_gate import build_ground_truth
from validate_stage4 import (
    TABLE_HALF_L,
    TABLE_HALF_W,
    integrate_to_table,
    load_yaml_constants,
    paddle_outgoing,
)

ANA = paths.ANALYSIS
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.abspath(os.path.join(HERE, "..", "..", "configs", "ball_physics_venue.yaml"))


def pct(vals, qs=(10, 50, 90)):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if len(vals) == 0:
        return dict(n=0)
    out = {f"p{int(q)}": float(np.percentile(vals, q)) for q in qs}
    out.update(n=int(len(vals)), min=float(vals.min()), max=float(vals.max()))
    return out


def med_p90(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if len(vals) == 0:
        return dict(n=0)
    return dict(n=int(len(vals)), median=float(np.median(vals)),
                p90=float(np.percentile(vals, 90)))


def spearman(rows, x_key, y_key):
    x = np.asarray([r[x_key] for r in rows if x_key in r and y_key in r
                    and np.isfinite(r[x_key]) and np.isfinite(r[y_key])], float)
    y = np.asarray([r[y_key] for r in rows if x_key in r and y_key in r
                    and np.isfinite(r[x_key]) and np.isfinite(r[y_key])], float)
    if len(x) < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return dict(n=int(len(x)), rho=None, p=None)
    res = stats.spearmanr(x, y)
    return dict(n=int(len(x)), rho=float(res.statistic), p=float(res.pvalue))


def tercile_bins(rows, x_key, y_key):
    vals = np.asarray([r[x_key] for r in rows if x_key in r and y_key in r
                       and np.isfinite(r[x_key]) and np.isfinite(r[y_key])], float)
    if len(vals) < 12:
        return []
    q1, q2 = np.percentile(vals, [33.333, 66.667])
    bins = [(-np.inf, q1, "low"), (q1, q2, "mid"), (q2, np.inf, "high")]
    out = []
    for lo, hi, label in bins:
        sub = [r for r in rows if x_key in r and y_key in r
               and np.isfinite(r[x_key]) and np.isfinite(r[y_key])
               and lo < r[x_key] <= hi]
        if not sub:
            continue
        out.append(dict(
            bin=label,
            n=len(sub),
            x_range=[float(min(r[x_key] for r in sub)), float(max(r[x_key] for r in sub))],
            y_median=float(np.median([r[y_key] for r in sub])),
            y_p90=float(np.percentile([r[y_key] for r in sub], 90)),
        ))
    return out


def paddle_params_from_yaml(vals, paddle_e):
    return dict(
        mode=paddle_e,
        e_eff=vals["contact.paddle.e_eff"],
        a_t=vals["contact.paddle.a_t"],
        b_t=vals["contact.paddle.b_t"],
        mu=vals["contact.paddle.mu_safety"],
        g1=vals.get("contact.paddle.e_exp_g1"),
        g2=vals.get("contact.paddle.e_exp_g2"),
        a=vals.get("contact.paddle.e_lin_a"),
        b=vals.get("contact.paddle.e_lin_b"),
    )


def contact_point_features(s):
    n = np.asarray(s["pad_n"], float)
    pad_p = np.asarray(s["pad_p"], float)
    ball_p = np.asarray(s["ball_p"], float)
    pad_v = np.asarray(s["pad_v"], float)
    cp = ball_p - R_BALL * n
    r = cp - pad_p
    normal_gap = float(np.dot(r, n))
    tangent_vec = r - normal_gap * n
    pad_vn = float(np.dot(pad_v, n))
    pad_vt = pad_v - pad_vn * n
    un = float(s["u_n"])
    ut = float(s["u_t"])
    theta = float(np.degrees(np.arctan2(ut, max(un, 1e-9))))
    return dict(
        contact_offset_m=float(np.linalg.norm(tangent_vec)),
        contact_normal_gap_m=normal_gap,
        incidence_theta_deg=theta,
        ut_over_un=float(ut / max(un, 1e-9)),
        pad_speed_mps=float(np.linalg.norm(pad_v)),
        pad_speed_normal_mps=pad_vn,
        pad_speed_tangent_mps=float(np.linalg.norm(pad_vt)),
        pad_speed_x_mps=float(pad_v[0]),
        pad_speed_y_mps=float(pad_v[1]),
        pad_speed_z_mps=float(pad_v[2]),
    )


def score_rows(gt_records, kd, km, pd, surface_z, split):
    rows = []
    for rec in gt_records:
        s = rec["strike"]
        if not in_split(f'{s["take"]}:{s["t_c"]:.3f}', split):
            continue
        if s["fit_rms_mm"] > 8.0 or s["pad_fit_rms_mm"] > 15.0 or not s["spin_in_ok"]:
            continue
        gt_xy = np.asarray(rec["gt_xy"], float)
        n = np.asarray(s["pad_n"], float)
        w_meas = np.asarray(s["w_out"], float) if s["spin_out_ok"] else np.zeros(3)
        p_h1, _, t_h1 = integrate_to_table(s["ball_p"], s["v_out"], w_meas, kd, km, surface_z)
        p_zero, _, t_zero = integrate_to_table(s["ball_p"], s["v_out"], np.zeros(3), kd, km, surface_z)
        vp, wp = paddle_outgoing(s, pd)
        p_h0, _, t_h0 = integrate_to_table(s["ball_p"], vp, wp, kd, km, surface_z)
        if p_h1 is None or p_zero is None or p_h0 is None:
            continue

        dv = vp - np.asarray(s["v_out"], float)
        dvn = float(np.dot(dv, n))
        dvt = dv - dvn * n
        row = dict(
            take=s["take"],
            paddle=s["paddle"],
            t_strike=float(s["t_c"]),
            gt_source=rec["gt_source"],
            u_n=float(s["u_n"]),
            u_t=float(s["u_t"]),
            spin_out_revs=float(np.linalg.norm(w_meas) / (2 * np.pi)),
            h0_landing_err_m=float(np.linalg.norm(p_h0[:2] - gt_xy)),
            h1_landing_err_m=float(np.linalg.norm(p_h1[:2] - gt_xy)),
            h0_minus_h1_landing_m=float(np.linalg.norm(p_h0[:2] - p_h1[:2])),
            spin_landing_shift_m=float(np.linalg.norm(p_zero[:2] - p_h1[:2])),
            spin_landing_delta_err_m=float(
                np.linalg.norm(p_zero[:2] - gt_xy) - np.linalg.norm(p_h1[:2] - gt_xy)),
            zero_spin_dt_vs_spin_ms=float((t_zero - t_h1) * 1e3),
            contact_dv_mps=float(np.linalg.norm(dv)),
            contact_dv_normal_mps=abs(dvn),
            contact_dv_tangent_mps=float(np.linalg.norm(dvt)),
            contact_dw_revs=float(
                np.linalg.norm(wp - w_meas) / (2 * np.pi)) if s["spin_out_ok"] else None,
            onoff_ok=bool((abs(gt_xy[0]) < TABLE_HALF_L and abs(gt_xy[1]) < TABLE_HALF_W)
                          == (abs(p_h0[0]) < TABLE_HALF_L and abs(p_h0[1]) < TABLE_HALF_W)),
        )
        row.update(contact_point_features(s))
        rows.append(row)
    return rows


def summarize(rows, yaml_vals):
    by_paddle = {}
    for paddle in sorted({r["paddle"] for r in rows}):
        sub = [r for r in rows if r["paddle"] == paddle]
        by_paddle[paddle] = dict(
            n=len(sub),
            h0_landing_err_m=med_p90([r["h0_landing_err_m"] for r in sub]),
            h1_landing_err_m=med_p90([r["h1_landing_err_m"] for r in sub]),
            u_n=pct([r["u_n"] for r in sub]),
        )

    feature_keys = [
        "u_n", "u_t", "ut_over_un", "incidence_theta_deg", "contact_offset_m",
        "contact_normal_gap_m", "pad_speed_mps", "pad_speed_normal_mps",
        "pad_speed_tangent_mps", "pad_speed_x_mps", "pad_speed_y_mps",
        "pad_speed_z_mps", "spin_out_revs",
    ]
    coverage = {k: pct([r[k] for r in rows]) for k in feature_keys}
    correlations = {
        k: spearman(rows, k, "h0_landing_err_m")
        for k in ("spin_out_revs", "u_n", "u_t", "ut_over_un", "incidence_theta_deg",
                  "contact_offset_m", "pad_speed_normal_mps", "pad_speed_tangent_mps")
    }
    binned = {
        k: tercile_bins(rows, k, "h0_landing_err_m")
        for k in ("spin_out_revs", "u_n", "incidence_theta_deg", "contact_offset_m",
                  "pad_speed_normal_mps", "pad_speed_tangent_mps")
    }
    return dict(
        n_scored=len(rows),
        n_by_gt_source={k: sum(1 for r in rows if r["gt_source"] == k)
                        for k in ("observed_bounce", "terminal_window")},
        n_by_paddle={k: sum(1 for r in rows if r["paddle"] == k)
                     for k in sorted({r["paddle"] for r in rows})},
        coverage=coverage,
        effects=dict(
            spin_landing_shift_m=med_p90([r["spin_landing_shift_m"] for r in rows]),
            spin_landing_delta_err_m=med_p90([abs(r["spin_landing_delta_err_m"]) for r in rows]),
            h0_landing_err_m=med_p90([r["h0_landing_err_m"] for r in rows]),
            h1_landing_err_m=med_p90([r["h1_landing_err_m"] for r in rows]),
            h0_minus_h1_landing_m=med_p90([r["h0_minus_h1_landing_m"] for r in rows]),
            contact_dv_mps=med_p90([r["contact_dv_mps"] for r in rows]),
            contact_dv_normal_mps=med_p90([r["contact_dv_normal_mps"] for r in rows]),
            contact_dv_tangent_mps=med_p90([r["contact_dv_tangent_mps"] for r in rows]),
            contact_dw_revs=med_p90([r["contact_dw_revs"] for r in rows]),
            onoff_acc_pct=float(np.mean([r["onoff_ok"] for r in rows]) * 100.0) if rows else None,
        ),
        pair_restitution=dict(
            table_e_eff=yaml_vals.get("contact.table.e_eff"),
            paddle_e_eff_const=yaml_vals.get("contact.paddle.e_eff"),
            paddle_e_exp_g1=yaml_vals.get("contact.paddle.e_exp_g1"),
            paddle_e_exp_g2=yaml_vals.get("contact.paddle.e_exp_g2"),
            note="These are effective pair coefficients for the measured ball+table and ball+paddle pair.",
        ),
        model_scope=dict(
            front_back_side="not identifiable: strikes record paddle id (p1/p2) and oriented face normal, but no rubber-side/front-back label",
            contact_position="proxy only: contact offset is relative to the paddle marker centroid, not a calibrated blade-local sweet-spot frame",
            net_height="covered by trajectory_prediction_gate.py, not duplicated here",
            ball_only_post_hit="covered by trajectory_prediction_gate.py post30/post60/post100/post150/net+20 modes",
            spin_output="trajectory_prediction_gate.py scores speed and axis stability; current rig spin axis is noisy",
        ),
        by_paddle=by_paddle,
        correlations_vs_h0_landing=correlations,
        binned_h0_landing=binned,
    )


def plot(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    specs = [
        ("spin_out_revs", "spin_landing_shift_m", "spin out (rev/s)", "landing shift if spin ignored (m)"),
        ("u_n", "h0_landing_err_m", "|u_n| normal speed (m/s)", "through-paddle landing error (m)"),
        ("incidence_theta_deg", "h0_landing_err_m", "incidence angle atan(u_t/u_n) (deg)", "through-paddle landing error (m)"),
        ("contact_offset_m", "h0_landing_err_m", "contact offset proxy (m)", "through-paddle landing error (m)"),
        ("pad_speed_normal_mps", "h0_landing_err_m", "paddle normal speed (m/s)", "through-paddle landing error (m)"),
        ("pad_speed_tangent_mps", "h0_landing_err_m", "paddle tangent speed (m/s)", "through-paddle landing error (m)"),
    ]
    colors = {"p1": "tab:blue", "p2": "tab:orange"}
    for axis, (xk, yk, xl, yl) in zip(ax.ravel(), specs):
        for paddle in sorted({r["paddle"] for r in rows}):
            sub = [r for r in rows if r["paddle"] == paddle]
            axis.scatter([r[xk] for r in sub], [r[yk] for r in sub],
                         s=18, alpha=0.65, label=paddle, color=colors.get(paddle))
        axis.set_xlabel(xl)
        axis.set_ylabel(yl)
        axis.grid(alpha=0.2)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--paddle-e", default="exp", choices=["const", "exp", "lin"])
    ap.add_argument("--out", default=os.path.join(ANA, "fits", "strike_factor_audit.json"))
    args = ap.parse_args()

    vals = load_yaml_constants(args.yaml)
    kd, km = vals["flight.k_d"], vals["flight.k_m"]
    pd = paddle_params_from_yaml(vals, args.paddle_e)
    meta = json.load(open(os.path.join(ANA, "segments", "meta.json")))
    bounces = json.load(open(os.path.join(ANA, "segments", "bounces.json")))
    strikes = json.load(open(os.path.join(ANA, "segments", "strikes.json")))
    gt_records = build_ground_truth(strikes, bounces, kd, km, meta["surface_z"])
    rows = score_rows(gt_records, kd, km, pd, meta["surface_z"], args.split)
    rep = dict(
        params_source=args.yaml,
        split=args.split,
        paddle_e_mode=args.paddle_e,
        ground_truth_records=len(gt_records),
        summary=summarize(rows, vals),
        rows=rows,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    png = args.out.replace(".json", ".png")
    plot(rows, png)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    print(f"-> {args.out}\n-> {png}")


if __name__ == "__main__":
    main()
