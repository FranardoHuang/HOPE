"""
Is the mocap ball a RIGID marker constellation (=> spin is measurable) or points
wandering on a coated surface (=> orientation does not exist)?

The naive test - scatter of pairwise distances indexed BY COLUMN - is invalid
here, because Motive relabels markers on a small fast-spinning ball, so column
i is not the same physical marker from frame to frame. On the 2026-07-30 takes
that test read 6.2-6.9 mm; the permutation-invariant version (sorted distance
spectrum) reads 2.2-2.8 mm, i.e. most of the apparent non-rigidity was label
churn. But 2.2-2.8 mm is ALSO what 8 points re-drawn uniformly on a 20 mm
sphere give (Monte-Carlo null: 2.48 mm), so the spectrum alone cannot decide.

The decisive test is to actually SOLVE the pose with unknown correspondence and
look at what comes out:
  - a rigid constellation admits ONE permutation+rotation per frame that fits
    every marker to the measurement-noise floor, and yields an angular velocity
    that is temporally COHERENT (smooth in free flight, jumping at bounces);
  - wandering points admit no such fit, and give white-noise "rotation".

Both are quantified here against a matched synthetic control, so the verdict is
a comparison of like with like rather than an eyeballed threshold.

Usage: python ball_rigidity_probe.py <take.c3d> [--max-frames 4000]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otc3d import probe, iter_chunks, detect_unit_scale, group_labeled_columns

R_BALL = 0.020


def kabsch(A, B):
    """Rotation taking centred A onto centred B."""
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def descriptor(P):
    """Rotation- and permutation-invariant per-point descriptor: the sorted
    vector of distances from each point to all the others."""
    d = np.linalg.norm(P[:, None] - P[None, :], axis=2)
    return np.sort(d, axis=1)[:, 1:]


def match_and_fit(P, T, icp_iters=3):
    """Align point set P (k,3) to template T (n,3), correspondence unknown.

    Stage 1: Hungarian assignment on the rotation-invariant descriptor.
    Stage 2: a few ICP rounds re-assigning by nearest neighbour after rotation.
    Returns (R, rms, perm) where perm[j] is the template index for P[j].
    """
    Pc = P - P.mean(0)
    Tc = T - T.mean(0)
    dp, dt = descriptor(Pc), descriptor(Tc)
    m = min(dp.shape[1], dt.shape[1])
    cost = np.linalg.norm(dp[:, None, :m] - dt[None, :, :m], axis=2)
    ri, ci = linear_sum_assignment(cost)
    perm = ci
    R = kabsch(Tc[perm], Pc[ri])
    for _ in range(icp_iters):
        # re-assign by nearest neighbour in the aligned frame
        cost2 = np.linalg.norm(Pc[:, None, :] - (Tc @ R.T)[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(cost2)
        perm = ci
        R = kabsch(Tc[perm], Pc[ri])
    resid = Pc[ri] - Tc[perm] @ R.T
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    return R, rms, perm


def build_template(frames, iters=4):
    """Iteratively refine a marker template from full-visibility frames."""
    T = frames[0] - frames[0].mean(0)
    for _ in range(iters):
        acc = np.zeros_like(T)
        cnt = 0
        for P in frames:
            R, rms, perm = match_and_fit(P, T)
            Pc = P - P.mean(0)
            # rotate the observation back into template space, in template order
            back = Pc @ R
            inv = np.empty(len(perm), int)
            inv[perm] = np.arange(len(perm))
            acc += back[inv]
            cnt += 1
        T = acc / cnt
        T -= T.mean(0)
    return T


def load_ball(path, asset="A", max_frames=None):
    geom = probe(path)
    unit, _ = detect_unit_scale(geom)
    groups, _ = group_labeled_columns(geom["labels"])
    cols = np.array(groups[asset], int)
    nf = geom["data_nf"]
    X = np.full((nf, len(cols), 4), np.nan, np.float32)
    for s, pts in iter_chunks(geom, chunk=8192):
        X[s:s + len(pts)] = pts[:, cols, :]
    ok = (X[:, :, 3] >= 0) & np.isfinite(X[:, :, 0])
    P = X[:, :, :3].astype(float) * unit
    return P, ok, float(geom["rate"])


def synthetic_control(T, n_frames, sigma, rate, omega, rng):
    """Rigid control: the SAME template, spun at `omega`, with `sigma` noise."""
    out = []
    for k in range(n_frames):
        R = Rot.from_rotvec(omega * k / rate).as_matrix()
        out.append(T @ R.T + rng.normal(0, sigma, T.shape))
    return out


def wandering_control(n_frames, sigma, rng, n=8, R=R_BALL):
    """Null control: points re-drawn uniformly on the sphere every frame."""
    out = []
    for _ in range(n_frames):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        out.append(v * R + rng.normal(0, sigma, (n, 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c3d")
    ap.add_argument("--asset", default="A")
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P, ok, rate = load_ball(args.c3d, args.asset)
    n_mk = P.shape[1]
    full = np.where(ok.sum(1) == n_mk)[0]
    print(f"{os.path.basename(args.c3d)}: {len(full)} full-visibility frames "
          f"({n_mk} markers) @ {rate} Hz")
    if len(full) < 100:
        raise SystemExit("not enough full-visibility frames")
    sel = full if len(full) <= args.max_frames else \
        full[np.linspace(0, len(full) - 1, args.max_frames).astype(int)]
    frames = [P[k] for k in sel]

    T = build_template(frames[:400])
    radii = np.linalg.norm(T, axis=1)
    print(f"template radii (mm): {np.round(radii * 1e3, 2)}  "
          f"-> mean {radii.mean()*1e3:.2f}, spread {np.ptp(radii)*1e3:.2f}")

    rms = np.array([match_and_fit(Pk, T)[1] for Pk in frames])
    print(f"OBSERVED  match rms: median {np.median(rms)*1e3:.2f} mm, "
          f"p90 {np.percentile(rms, 90)*1e3:.2f} mm")

    # controls, matched in frame count
    rng = np.random.default_rng(0)
    sigma = float(np.median(rms)) / np.sqrt(2)     # rough per-marker noise
    n_ctl = min(len(frames), 1500)
    rig = synthetic_control(T, n_ctl, sigma, rate, np.array([0, 60.0, 0]), rng)
    rms_rig = np.array([match_and_fit(Pk, T)[1] for Pk in rig])
    wan = wandering_control(n_ctl, sigma, rng, n=n_mk)
    Tw = build_template(wan[:200])
    rms_wan = np.array([match_and_fit(Pk, Tw)[1] for Pk in wan])
    print(f"CONTROL rigid (sigma={sigma*1e3:.2f} mm): median "
          f"{np.median(rms_rig)*1e3:.2f} mm")
    print(f"CONTROL wandering            : median "
          f"{np.median(rms_wan)*1e3:.2f} mm")

    # ---- temporal coherence of the recovered rotation ----
    # consecutive FULL frames only, so the correspondence is solved the same way
    Rs, idx = {}, []
    for k in sel:
        Rk, r, _ = match_and_fit(P[k], T)
        Rs[k] = Rk
        idx.append(k)
    idx = np.array(idx)
    step = np.diff(idx)
    adj = np.where(step == 1)[0]
    print(f"adjacent full-frame pairs: {len(adj)}")
    if len(adj) > 20:
        w = []
        for j in adj:
            a, b = idx[j], idx[j + 1]
            dR = Rs[b] @ Rs[a].T
            w.append(Rot.from_matrix(dR).as_rotvec() * rate)
        w = np.array(w)
        mag = np.linalg.norm(w, axis=1)
        print(f"per-frame |omega|: median {np.median(mag):.1f} rad/s "
              f"({np.median(mag)/2/np.pi:.1f} rev/s), p90 {np.percentile(mag,90):.1f}")
        # coherence: correlation between consecutive omega estimates. A rigid body
        # spinning smoothly gives high correlation; white noise gives ~0.
        run = []
        for j in range(len(adj) - 1):
            if idx[adj[j] + 1] == idx[adj[j + 1]]:
                run.append((w[j], w[j + 1]))
        if len(run) > 20:
            A = np.array([r[0] for r in run]); B = np.array([r[1] for r in run])
            cc = [float(np.corrcoef(A[:, i], B[:, i])[0, 1]) for i in range(3)]
            print(f"consecutive-omega correlation per axis: "
                  f"{[round(c,3) for c in cc]}   (n={len(run)})")
            print("  rigid + smooth rotation -> high positive; wandering -> ~0")
        # same statistic on the controls
        for label, ctl in (("rigid", rig), ("wandering", wan)):
            Rc = [match_and_fit(Pk, T if label == "rigid" else Tw)[0] for Pk in ctl]
            wc = np.array([Rot.from_matrix(Rc[i + 1] @ Rc[i].T).as_rotvec() * rate
                           for i in range(len(Rc) - 1)])
            cc = [float(np.corrcoef(wc[:-1, i], wc[1:, i])[0, 1]) for i in range(3)]
            print(f"  CONTROL {label:9s} consecutive-omega corr: "
                  f"{[round(c,3) for c in cc]}, |omega| med "
                  f"{np.median(np.linalg.norm(wc,axis=1)):.1f} rad/s")

    if args.out:
        json.dump(dict(take=os.path.basename(args.c3d), n_full=len(full),
                       rms_median_mm=float(np.median(rms) * 1e3),
                       rms_rigid_control_mm=float(np.median(rms_rig) * 1e3),
                       rms_wandering_control_mm=float(np.median(rms_wan) * 1e3),
                       template_radii_mm=[float(r * 1e3) for r in radii]),
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
