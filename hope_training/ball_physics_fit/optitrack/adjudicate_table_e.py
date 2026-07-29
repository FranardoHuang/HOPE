"""
Table restitution from a PURE-BOUNCE take, by inter-bounce FLIGHT TIME.

Why this estimator: the rally takes give e = 0.9443 from the stage-1
reconstruction (two-sided shooting fits meeting at an estimated contact instant
t_c), against the 2026-07-03 venue fit's 0.9215 on the SAME competition table.
The live suspicion is a residual t_c bias - the reconstructed contact height sits
+3.2 mm above geometric, and e rises with that offset across terciles
(0.938 / 0.944 / 0.966).

For a ball bouncing repeatedly on the table with nothing touching it,
    T_k = 2 * v_z_out(k) / g          (no drag)
    v_z_in(k+1) = -v_z_out(k)
    => e(k+1) = v_out(k+1)/v_in(k+1) = T_{k+1} / T_k
so e comes from BOUNCE TIMES ALONE. A constant error in t_c cancels to first
order and no velocity extrapolation is involved at all. Rally takes cannot
support it (measured: of 155 bounce intervals, 86 have a racket strike between
and 52 span a tracking gap -> 0 usable triples), which is why the pure-bounce
take matters.

Drag makes the real flight shorter than 2v/g, so the raw ratio is corrected by
numerically inverting T -> v_z with the fitted k_d.

Selection depends ONLY on validity (single clean z-minimum, ball tracked
throughout, no strike, plausible duration) and NEVER on the estimated ratio -
an earlier attempt filtered pairs to T2 <= 1.05*T1 and that biased e downward.

Usage: python adjudicate_table_e.py [--take chuntan] [--kd 0.1253]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from ballcore import load_take, R_BALL, G_NOM, smooth_vel, spin_from_quats


def flight_time_to_vz(T, kd, vx, g=G_NOM, iters=60):
    """Invert flight time -> launch vertical speed, WITH quadratic drag.

    Integrates the coupled 2-D problem (drag depends on total speed, so the
    horizontal speed matters) and bisects on v_z0 until the up-down flight time
    matches T.
    """
    def flight_time(vz0):
        dt = 2e-4
        z, vz, vxx = 0.0, vz0, vx
        t = 0.0
        while t < 5.0:
            s = np.hypot(vxx, vz)
            az = -g - kd * s * vz
            ax = -kd * s * vxx
            vz += az * dt
            vxx += ax * dt
            z += vz * dt
            t += dt
            if z <= 0.0 and t > 1e-3:
                return t
        return np.nan

    lo, hi = 0.05, 12.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tm = flight_time(mid)
        if not np.isfinite(tm):
            hi = mid
        elif tm < T:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def find_bounces(t, pos, present, rate, z_contact, band=0.030):
    """Bounce frames: local z-minima near the contact height with a vz reversal.

    Returns a list of (index, t_min_parabola) where the time is refined by
    fitting a parabola to z around the minimum (sub-frame, and independent of
    any contact model).
    """
    z = np.where(present, pos[:, 2], np.nan)
    out = []
    for i in range(3, len(z) - 3):
        w = z[i - 3:i + 4]
        if not np.isfinite(w).all():
            continue
        if z[i] > z_contact + band:
            continue
        if not (z[i] <= z[i - 1] and z[i] <= z[i + 1]
                and z[i - 2] > z[i - 1] and z[i + 2] > z[i + 1]):
            continue
        # sub-frame minimum by parabola through the 5 points around i
        k = np.arange(i - 2, i + 3)
        c = np.polyfit(t[k] - t[i], z[k], 2)
        if c[0] <= 0:
            continue
        t_min = t[i] - c[1] / (2 * c[0])
        if abs(t_min - t[i]) > 2.0 / rate:
            continue
        out.append((i, float(t_min), float(np.polyval(c, t_min - t[i]))))
    # de-duplicate minima closer than 40 ms
    ded = []
    for rec in out:
        if ded and rec[1] - ded[-1][1] < 0.040:
            continue
        ded.append(rec)
    return ded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="chuntan")
    ap.add_argument("--kd", type=float, default=0.1253)
    ap.add_argument("--surface-z", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tk = load_take(args.take)
    rate = float(tk["rate"])
    t = tk["t"]
    pos = tk["ball_pos_t_m"].astype(float)
    present = tk["ball_present"].astype(bool)
    quat = tk["ball_quat_xyzw"].astype(float)

    # surface from the bounce minima themselves (median z_min - R)
    z_guess = args.surface_z
    if z_guess is None:
        b0 = find_bounces(t, pos, present, rate, z_contact=0.05, band=0.060)
        if not b0:
            raise SystemExit("no bounce candidates to calibrate the surface")
        z_guess = float(np.median([r[2] for r in b0])) - R_BALL
    z_contact = z_guess + R_BALL
    b = find_bounces(t, pos, present, rate, z_contact=z_contact)
    print(f"{args.take}: surface_z = {z_guess*1e3:.2f} mm, "
          f"contact height = {z_contact*1e3:.2f} mm, bounces found = {len(b)}")

    sp, sv = smooth_vel(t, np.where(present[:, None], pos, np.nan))
    rows = []
    for j in range(len(b) - 2):
        i0, t0, _ = b[j]
        i1, t1, _ = b[j + 1]
        i2, t2, _ = b[j + 2]
        T1, T2 = t1 - t0, t2 - t1
        if not (0.08 < T1 < 1.5 and 0.08 < T2 < 1.5):
            continue
        # validity only: ball tracked through both flights, exactly one z-max each
        seg1 = slice(i0 + 2, i1 - 1)
        seg2 = slice(i1 + 2, i2 - 1)
        if present[seg1].mean() < 0.7 or present[seg2].mean() < 0.7:
            continue
        z1 = np.where(present[seg1], pos[seg1, 2], np.nan)
        z2 = np.where(present[seg2], pos[seg2, 2], np.nan)
        if np.nanmax(z1) < z_contact + 0.02 or np.nanmax(z2) < z_contact + 0.02:
            continue
        vx1 = float(np.nanmedian(np.linalg.norm(sv[seg1][:, :2], axis=1)))
        vx2 = float(np.nanmedian(np.linalg.norm(sv[seg2][:, :2], axis=1)))
        # spin around the middle bounce, if the orientation channel solved there
        w_in = w_out = None
        qa = quat[max(i1 - 40, 0):i1 - 2]
        qb = quat[i1 + 2:i1 + 40]
        if np.isfinite(qa).all(1).sum() > 8:
            _, _, w = spin_from_quats(qa, rate, R_table=tk["table_R"])
            w_in = w if np.isfinite(w).all() else None
        if np.isfinite(qb).all(1).sum() > 8:
            _, _, w = spin_from_quats(qb, rate, R_table=tk["table_R"])
            w_out = w if np.isfinite(w).all() else None

        e_raw = T2 / T1
        vz1 = flight_time_to_vz(T1, args.kd, vx1)
        vz2 = flight_time_to_vz(T2, args.kd, vx2)
        e_drag = vz2 / vz1
        rows.append(dict(t_c=t1, T1=T1, T2=T2, e_raw=e_raw, e_drag=e_drag,
                         vz_in=vz1, vz_out=vz2, vx=vx1,
                         w_in_rev=(float(np.linalg.norm(w_in) / 2 / np.pi)
                                   if w_in is not None else None),
                         w_out_rev=(float(np.linalg.norm(w_out) / 2 / np.pi)
                                    if w_out is not None else None)))

    if not rows:
        raise SystemExit("no usable bounce triples")
    er = np.array([r["e_raw"] for r in rows])
    ed = np.array([r["e_drag"] for r in rows])
    vn = np.array([r["vz_in"] for r in rows])
    rng = np.random.default_rng(0)

    def boot(x):
        return [float(np.percentile([np.median(x[rng.integers(0, len(x), len(x))])
                                    for _ in range(4000)], p)) for p in (2.5, 97.5)]

    print(f"\nusable triples: {len(rows)}   v_n range {vn.min():.2f}-{vn.max():.2f} m/s")
    print(f"  e (raw T2/T1)      median {np.median(er):.4f}  CI95 "
          f"{[round(v,4) for v in boot(er)]}")
    print(f"  e (drag-corrected) median {np.median(ed):.4f}  CI95 "
          f"{[round(v,4) for v in boot(ed)]}")
    print(f"  drag correction shifts e by {np.median(ed)-np.median(er):+.4f}")
    A = np.vstack([np.ones_like(vn), vn]).T
    c, *_ = np.linalg.lstsq(A, ed, rcond=None)
    print(f"  speed law: e = {c[0]:.4f} {c[1]:+.4f} * v_n   "
          f"-> at ITTF drop 2.43 m/s: {c[0]+c[1]*2.43:.4f}")
    ws = [r["w_in_rev"] for r in rows if r["w_in_rev"] is not None]
    if ws:
        print(f"  spin before the bounce (n={len(ws)}): median "
              f"{np.median(ws):.2f} rev/s, p90 {np.percentile(ws,90):.2f}")
    print(f"\n  REFERENCE  rally-take stage-1 estimate 0.9443 CI [0.9379, 0.9538]")
    print(f"             venue 2026-07-03 fit          0.9215")
    print(f"             ITTF band at drop speed       0.876 - 0.931")

    if args.out:
        json.dump(dict(take=args.take, kd=args.kd, surface_z=z_guess,
                       n_bounces=len(b), n_triples=len(rows),
                       e_raw_median=float(np.median(er)), e_raw_ci=boot(er),
                       e_drag_median=float(np.median(ed)), e_drag_ci=boot(ed),
                       speed_law=[float(c[0]), float(c[1])],
                       e_at_ittf_drop=float(c[0] + c[1] * 2.43),
                       rows=rows), open(args.out, "w"), indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
