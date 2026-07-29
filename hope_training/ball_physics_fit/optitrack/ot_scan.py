"""
Pass 1 over an OptiTrack ball-take C3D: per-column statistics + role detection.

Streams the whole data section once (memory O(npts), any file size) and writes
<out_dir>/scan.npz + scan_manifest.json with:
  - per-column: n_valid, first/last valid frame, z range, max inter-frame speed
  - ball asset columns (auto-detected labeled rigid-body group, or --ball-asset)
  - static reference markers (long-lived, tight bounding box) to exclude later
  - truncation / unit / rate report

Usage: python ot_scan.py <take.c3d> <out_dir> [--ball-asset NAME] [--chunk N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otc3d import probe, iter_chunks, detect_unit_scale, group_labeled_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c3d")
    ap.add_argument("out_dir")
    ap.add_argument("--ball-asset", default=None,
                    help="labeled asset name for the ball (default: auto = "
                         "the labeled marker group with the most columns)")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    geom = probe(args.c3d)
    unit, p95 = detect_unit_scale(geom)
    groups, _ = group_labeled_columns(geom["labels"])
    if args.ball_asset:
        if args.ball_asset not in groups:
            sys.exit(f"asset {args.ball_asset!r} not in labeled groups {list(groups)}")
        ball_asset = args.ball_asset
    else:
        if not groups:
            sys.exit("no labeled marker groups found; pass --ball-asset or check export")
        ball_asset = max(groups, key=lambda a: len(groups[a]))
    ball_cols = np.array(groups[ball_asset], int)

    npts, nf, rate = geom["npts"], geom["data_nf"], geom["rate"]
    print(f"{os.path.basename(args.c3d)}: {npts} cols x {nf} frames @ {rate} Hz "
          f"({nf / rate:.1f}s), unit->m x{unit}, ball asset {ball_asset!r} "
          f"({len(ball_cols)} markers)"
          + (f"  [TRUNCATED: header claims {geom['header_nf']} frames, "
             f"{nf / geom['header_nf'] * 100:.1f}% present]" if geom["truncated"] else ""))

    n_valid = np.zeros(npts, np.int64)
    first_v = np.full(npts, -1, np.int64)
    last_v = np.full(npts, -1, np.int64)
    zmin = np.full(npts, np.inf)
    zmax = np.full(npts, -np.inf)
    max_sp = np.zeros(npts)                      # m per frame
    sum_xyz = np.zeros((npts, 3))
    ball_pts = np.full((nf, len(ball_cols), 4), np.nan, np.float32)
    prev_xyz = prev_ok = None
    t0 = time.time()

    def prog(e, n):
        print(f"  scan {e}/{n} frames  {time.time() - t0:.0f}s", flush=True)

    for s, pts in iter_chunks(geom, chunk=args.chunk, progress=prog):
        e = s + len(pts)
        ok = pts[:, :, 3] >= 0
        xyz = pts[:, :, :3].astype(np.float64) * unit
        n_valid += ok.sum(0)
        anyok = ok.any(0)
        fidx = ok.argmax(0)
        newfirst = (first_v < 0) & anyok
        first_v[newfirst] = s + fidx[newfirst]
        lidx = (len(pts) - 1) - ok[::-1].argmax(0)
        last_v[anyok] = s + lidx[anyok]
        z = np.where(ok, xyz[:, :, 2], np.nan)
        with np.errstate(invalid="ignore"):
            zmin = np.fmin(zmin, np.nanmin(np.where(np.isfinite(z), z, np.inf), 0))
            zmax = np.fmax(zmax, np.nanmax(np.where(np.isfinite(z), z, -np.inf), 0))
        sum_xyz += np.where(ok[:, :, None], xyz, 0).sum(0)
        x_all = xyz if prev_xyz is None else np.concatenate([prev_xyz[None], xyz], 0)
        o_all = ok if prev_ok is None else np.concatenate([prev_ok[None], ok], 0)
        d = np.linalg.norm(np.diff(np.where(o_all[:, :, None], x_all, np.nan), axis=0),
                           axis=2)
        if len(d):
            max_sp = np.fmax(max_sp, np.nan_to_num(np.fmax.reduce(
                np.where(np.isfinite(d), d, 0.0), axis=0)))
        prev_xyz, prev_ok = xyz[-1], ok[-1]
        bp = pts[:, ball_cols, :].astype(np.float32)
        bp[..., :3] *= unit
        ball_pts[s:e] = bp

    mean_xyz = sum_xyz / np.maximum(n_valid, 1)[:, None]
    # static references: long-lived, tight bbox, not the ball
    dur = np.where(n_valid > 0, (last_v - first_v + 1) / rate, 0)
    is_ball = np.zeros(npts, bool)
    is_ball[ball_cols] = True
    static = (~is_ball) & (dur > 10) & (n_valid > 10 * rate) & \
             ((zmax - zmin) < 0.10) & (max_sp * rate < 8.0)
    # cluster static medians within 0.3 m
    refs = []
    for i in np.where(static)[0]:
        p = mean_xyz[i]
        for r in refs:
            if np.linalg.norm(p - r["pos"]) < 0.3:
                r["cols"].append(int(i))
                break
        else:
            refs.append(dict(pos=p.copy(), cols=[int(i)]))

    np.savez_compressed(
        os.path.join(args.out_dir, "scan.npz"),
        n_valid=n_valid, first_v=first_v, last_v=last_v, zmin=zmin, zmax=zmax,
        max_sp=max_sp, mean_xyz=mean_xyz, ball_pts=ball_pts, ball_cols=ball_cols,
        static_ref_pos=np.array([r["pos"] for r in refs]).reshape(-1, 3),
    )
    ball_ok = np.isfinite(ball_pts[:, :, 0]) & (ball_pts[:, :, 3] >= 0)
    nmk = ball_ok.sum(1)
    man = dict(
        take=os.path.basename(args.c3d), rate_hz=rate,
        frames_in_file=int(nf), frames_in_header=int(geom["header_nf"]),
        truncated=geom["truncated"],
        duration_s=round(nf / rate, 1), unit_to_m=unit, coord_p95=round(p95, 1),
        n_cols=int(npts), n_cols_active=int((n_valid > 0).sum()),
        ball_asset=ball_asset, ball_cols=[int(c) for c in ball_cols],
        ball_marker_count_hist={int(k): int(v) for k, v in
                                zip(*np.unique(nmk, return_counts=True))},
        ball_ge3_pct=round(float((nmk >= 3).mean()) * 100, 1),
        static_refs=[dict(pos=[round(float(x), 3) for x in r["pos"]],
                          n_cols=len(r["cols"])) for r in refs],
        scan_seconds=round(time.time() - t0),
    )
    with open(os.path.join(args.out_dir, "scan_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(json.dumps(man, ensure_ascii=False))


if __name__ == "__main__":
    main()
