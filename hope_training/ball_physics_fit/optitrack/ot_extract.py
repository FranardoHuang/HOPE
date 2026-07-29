"""
Pass 2: ball-center trajectory -> canonical take npz (ballcore-compatible keys).

Center estimator: CENTROID of the visible ball-asset points. The coated ball's
markers are reconstruction points wandering on the sphere surface (validated on
take 0721: centroid beats fixed-radius sphere fitting on arc-parabola RMS
because sphere fits need >=4 points and fragment coverage; both carry the same
~5 mm wander noise). No usable orientation -> ball_quat_xyzw is NaN.

Gap fill: frames where the asset solve dropped (<min-markers) are filled from
Unlabeled_* points near a ballistic prediction from neighbors (forward+backward
sweep), then micro-gaps are linearly interpolated. A speed-spike guard rejects
fill/interp frames implying impossible velocity.

Usage: python ot_extract.py <take.c3d> <out_dir>   # out_dir must hold scan.npz
Options: --min-markers 3 --attach-r 0.045 --coast-ms 100 --max-gap 5 --vmax 30
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otc3d import probe, iter_chunks, detect_unit_scale


def sweep_fill(pos, filled, starts, s_xyz, rate, attach_r, coast_s, direction):
    """One directional pass: coast a ballistic prediction across solve gaps and
    attach unlabeled clusters. Mutates pos/filled in place."""
    nf = len(pos)
    rng = range(nf) if direction > 0 else range(nf - 1, -1, -1)
    last_p = last_v = last_i = None
    for i in rng:
        if filled[i] >= 0 and np.isfinite(pos[i, 0]):
            if last_p is not None and abs(i - last_i) <= 3:
                last_v = (pos[i] - last_p) / ((i - last_i) / rate)
            else:
                last_v = None
            last_p, last_i = pos[i].copy(), i
            continue
        if last_p is None:
            continue
        dt = (i - last_i) / rate
        if abs(dt) > coast_s:
            continue
        pred = last_p + (last_v * dt if last_v is not None else 0.0)
        if last_v is not None:
            pred = pred + 0.5 * np.array([0.0, 0.0, -9.81]) * dt * dt
        a, b = starts[i], starts[i + 1]
        if a == b:
            continue
        pts = s_xyz[a:b]
        sel = np.linalg.norm(pts - pred, axis=1) < \
            (attach_r if last_v is not None else 2 * attach_r)
        if not sel.any():
            continue
        pos[i] = pts[sel].mean(0)
        filled[i] = 1
        last_p, last_i = pos[i].copy(), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c3d")
    ap.add_argument("out_dir")
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--attach-r", type=float, default=0.045)
    ap.add_argument("--coast-ms", type=float, default=100.0)
    ap.add_argument("--max-gap", type=int, default=5)
    ap.add_argument("--vmax", type=float, default=30.0,
                    help="reject fill/interp frames implying speed above this (m/s)")
    ap.add_argument("--ref-excl-r", type=float, default=0.25)
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    scan = np.load(os.path.join(args.out_dir, "scan.npz"))
    ball_cols = scan["ball_cols"]
    ball_pts = scan["ball_pts"].astype(np.float64)
    refs = scan["static_ref_pos"]
    geom = probe(args.c3d)
    unit, _ = detect_unit_scale(geom)
    nf, rate = geom["data_nf"], geom["rate"]
    name = os.path.basename(args.c3d).replace(".c3d", "").replace(" ", "_")

    # ---- sparse non-ball points (streamed) ----
    mask_other = np.ones(geom["npts"], bool)
    mask_other[ball_cols] = False
    fr_l, xyz_l = [], []
    t0 = time.time()

    def prog(e, n):
        print(f"  sparse {e}/{n}  {time.time() - t0:.0f}s", flush=True)

    for s, pts in iter_chunks(geom, chunk=args.chunk, progress=prog):
        ok = (pts[:, :, 3] >= 0) & mask_other[None, :]
        fi, ci = np.where(ok)
        fr_l.append((s + fi).astype(np.int64))
        xyz_l.append(pts[fi, ci, :3].astype(np.float64) * unit)
    s_fr = np.concatenate(fr_l)
    s_xyz = np.concatenate(xyz_l)
    for rp in refs:
        keep = np.linalg.norm(s_xyz - rp, axis=1) >= args.ref_excl_r
        s_fr, s_xyz = s_fr[keep], s_xyz[keep]
    order = np.argsort(s_fr, kind="stable")
    s_fr, s_xyz = s_fr[order], s_xyz[order]
    starts = np.searchsorted(s_fr, np.arange(nf + 1))
    print(f"sparse non-ball points kept: {len(s_fr)}")

    # ---- centroid + fill + interp ----
    ok_b = np.isfinite(ball_pts[:, :, 0]) & (ball_pts[:, :, 3] >= 0)
    nmk = ok_b.sum(1)
    pos = np.full((nf, 3), np.nan)
    solved = nmk >= args.min_markers
    xyz_b = np.where(ok_b[:, :, None], ball_pts[:, :, :3], np.nan)
    pos[solved] = np.nanmean(xyz_b[solved], axis=1)
    filled = np.full(nf, -1, np.int8)
    filled[solved] = 0
    cov0 = solved.mean()
    sweep_fill(pos, filled, starts, s_xyz, rate, args.attach_r,
               args.coast_ms / 1e3, +1)
    sweep_fill(pos, filled, starts, s_xyz, rate, args.attach_r,
               args.coast_ms / 1e3, -1)

    # speed-spike guard on fill frames
    def spike_reject():
        n_rej = 0
        while True:
            have = filled >= 0
            idx = np.where(have)[0]
            v = np.linalg.norm(np.diff(pos[idx], axis=0), axis=1) / \
                (np.diff(idx) / rate)
            bad = np.where(v > args.vmax)[0]
            rejected = False
            for k in bad:
                for j in (idx[k], idx[k + 1]):
                    if filled[j] == 1:      # only drop fill frames, never solves
                        filled[j] = -1
                        pos[j] = np.nan
                        n_rej += 1
                        rejected = True
                        break
            if not rejected:
                return n_rej

    n_rej = spike_reject()

    interp = np.zeros(nf, bool)
    idx = np.where(filled >= 0)[0]
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        if 1 < b - a <= args.max_gap + 1:
            w = np.arange(a + 1, b)
            f = (w - a) / (b - a)
            pos[w] = pos[a] * (1 - f[:, None]) + pos[b] * f[:, None]
            interp[w] = True
    present = np.isfinite(pos[:, 0])

    t_axis = np.arange(nf) / rate
    np.savez_compressed(
        os.path.join(args.out_dir, f"{name}.npz"),
        t=t_axis, rate=rate, name=name,
        ball_pos_t_m=pos.astype(np.float32), ball_present=present,
        ball_quat_xyzw=np.full((nf, 4), np.nan, np.float32),
        fill_source=filled, interp_mask=interp, n_markers=nmk.astype(np.int8),
    )
    man = dict(
        take=name, rate_hz=rate, frames=int(nf), duration_s=round(nf / rate, 1),
        truncated=geom["truncated"], unit_to_m=unit,
        coverage_solved_pct=round(float(cov0) * 100, 1),
        coverage_filled_pct=round(float((filled >= 0).mean()) * 100, 1),
        coverage_final_pct=round(float(present.mean()) * 100, 1),
        n_fill=int((filled == 1).sum()), n_interp=int(interp.sum()),
        n_spike_rejected=int(n_rej),
        params=dict(min_markers=args.min_markers, attach_r=args.attach_r,
                    coast_ms=args.coast_ms, max_gap=args.max_gap, vmax=args.vmax),
    )
    with open(os.path.join(args.out_dir, f"{name}_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
