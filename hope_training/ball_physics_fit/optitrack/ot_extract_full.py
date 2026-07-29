"""
Pass 1+2 for a FULL OptiTrack (Motive) take: ball + both rackets + table.

The 2026-07-30 session finally exports every asset the fit needs:
    A     : coated ball, 8 wandering surface reconstruction points (NOT rigid)
    PPP1  : racket 1, 8-marker rigid body
    PPP2  : racket 2, 8-marker rigid body
    PPT   : table, 6 static markers = 4 corners + 2 net posts

That makes it a superset of take 0721 (ball only -> ot_extract.py) and a
sibling of the 2026-07-03 Avatar-Pro venue exports (extract_canonical.py), so
this writes the SAME canonical npz schema ballcore/stage1/stage2 consume.
Differences from the venue extractor, all forced by the export format:

  - streams the file (otc3d.iter_chunks); a 2.3 GB / 3255-column export cannot
    go through c3d.Reader.read_frames() in reasonable time
  - units auto-detected (Motive omits POINT:UNITS); this session is METERS
  - ball ORIENTATION needs a permutation-robust solve. Motive relabels markers
    on a small spinning ball, so column identity is meaningless frame to frame;
    ball_orientation.solve_orientation_chained recovers rotation by matching
    CONSECUTIVE clouds (where the step is ~1.5 deg and the correspondence is
    unambiguous) and chaining. Ball CENTRE comes from a fixed-radius sphere fit
    (>=4 points) with centroid fallback at 3.  Pass --no-spin to skip the
    orientation solve.
  - the racket template is built by robust averaging over many full-visibility
    frames, not from a single first frame (marker noise here is ~3-5 mm).

Output: <out_dir>/<take>.npz + <take>_manifest.json

Usage: python ot_extract_full.py <take.c3d> <out_dir> [--center {sphere,centroid}]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otc3d import probe, iter_chunks, detect_unit_scale, group_labeled_columns

R_BALL = 0.020
ITTF_LEN, ITTF_WID = 2.740, 1.525
NET_HEIGHT = 0.1525
TRIM_S = 1.0            # cut this much from each end of the take
BALL_ASSET = "A"
RACKETS = ("PPP1", "PPP2")
TABLE_ASSET = "PPT"


# ---------------------------------------------------------------- streaming

def read_assets(path, assets, chunk=1024, verbose=True, required=()):
    """Stream the take once; return {asset: (nf, M, 3) meters, NaN where absent}.

    Assets in `assets` but not in the export are skipped with a warning; only
    `required` ones are fatal. A pure-bounce take carries no racket at all, and
    a session may name its rackets differently, so refusing to read anything
    unless all four assets are present would be wrong.
    """
    geom = probe(path)
    unit, p95 = detect_unit_scale(geom)
    groups, unlabeled = group_labeled_columns(geom["labels"])
    missing_req = [a for a in required if a not in groups]
    if missing_req:
        raise SystemExit(f"{path}: REQUIRED assets {missing_req} not in export "
                         f"(labeled groups present: {sorted(groups)})")
    absent = [a for a in assets if a not in groups]
    if absent:
        print(f"  note: assets {absent} absent from this export "
              f"(present: {sorted(groups)}) - skipping their channels")
    cols = {a: np.array(groups[a], int) for a in assets if a in groups}
    nf = geom["data_nf"]
    out = {a: np.full((nf, len(c), 3), np.nan, np.float32) for a, c in cols.items()}
    t0 = time.time()

    def prog(e, n):
        if verbose:
            print(f"  read {e}/{n} frames  {time.time() - t0:.0f}s", flush=True)

    for s, pts in iter_chunks(geom, chunk=chunk, progress=prog):
        e = s + len(pts)
        for a, c in cols.items():
            sub = pts[:, c, :]
            ok = sub[:, :, 3] >= 0
            xyz = sub[:, :, :3] * unit
            out[a][s:e] = np.where(ok[:, :, None], xyz, np.nan)
    return out, geom, unit, p95, len(unlabeled)


# ------------------------------------------------------------- table frame

def table_frame(tab_m):
    """Canonical table frame from the 6 static PPT markers.

    Net posts are separated by height (they stand ~0.15 m above the surface),
    the remaining 4 markers are the table corners. Returns origin = centroid of
    the 4 corner markers, R with columns [X=length, Y=width, Z=up] and a QA dict.
    """
    med = np.nanmedian(tab_m, axis=0)                       # (6,3)
    if not np.isfinite(med).all():
        raise SystemExit("table markers never simultaneously visible")
    order = np.argsort(med[:, 2])
    corners, posts = med[order[:4]], med[order[4:]]
    center = corners.mean(0)
    Q = corners - center
    _, _, Vt = np.linalg.svd(Q)
    ez = Vt[2]
    ez = ez if ez[2] > 0 else -ez
    Qp = Q - np.outer(Q @ ez, ez)
    evals, evecs = np.linalg.eigh(Qp.T @ Qp)
    ex = evecs[:, 2]                                        # longest in-plane axis
    ex = ex - (ex @ ez) * ez
    ex /= np.linalg.norm(ex)
    if ex[np.argmax(np.abs(ex))] < 0:                       # deterministic sign
        ex = -ex
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])
    ct = (corners - center) @ R
    qa = dict(
        length_ext_m=float(np.ptp(ct[:, 0])),
        width_ext_m=float(np.ptp(ct[:, 1])),
        corner_plane_rms_mm=float(np.sqrt((ct[:, 2] ** 2).mean()) * 1e3),
        tilt_vs_world_deg=float(np.degrees(np.arccos(np.clip(ez[2], -1, 1)))),
        corners_t_m=ct.tolist(),
        posts_t_m=((posts - center) @ R).tolist(),
        net_x_t_m=float(np.mean(((posts - center) @ R)[:, 0])),
        net_post_height_above_corner_plane_m=float(np.mean(((posts - center) @ R)[:, 2])),
        static_rms_mm=float(np.nanmean(np.nanstd(tab_m, axis=0)) * 1e3),
    )
    return center, R, corners, posts, qa


# ------------------------------------------------------------- ball centre

def sphere_center_fixed_R(P, R=R_BALL, iters=12):
    """Least-squares centre of points constrained to a sphere of KNOWN radius R.

    Gauss-Newton on f_i = |p_i - c| - R.  P: (k,3) with k >= 4 (k=3 is solvable
    but has a two-fold ambiguity, so callers use the centroid there).
    Returns (centre, rms_residual).
    """
    c = P.mean(0)
    for _ in range(iters):
        d = P - c
        n = np.linalg.norm(d, axis=1)
        n = np.maximum(n, 1e-9)
        u = d / n[:, None]
        r = n - R                       # residual
        # J = -u  ->  solve (u^T u) dc = u^T r
        dc, *_ = np.linalg.lstsq(u, r, rcond=None)
        c = c + dc
        if np.linalg.norm(dc) < 1e-9:
            break
    d = np.linalg.norm(P - c, axis=1) - R
    return c, float(np.sqrt((d ** 2).mean()))


def ball_center_series(ball_m, mode="sphere"):
    """Per-frame ball centre. Returns (pos (nf,3), present, resid_rms, n_markers)."""
    nf, M, _ = ball_m.shape
    valid = np.isfinite(ball_m[:, :, 0])
    nval = valid.sum(1)
    pos = np.full((nf, 3), np.nan)
    resid = np.full(nf, np.nan)
    cent = np.where(nval[:, None] > 0,
                    np.nansum(np.nan_to_num(ball_m), axis=1) / np.maximum(nval, 1)[:, None],
                    np.nan)
    if mode == "centroid":
        ok = nval >= 3
        pos[ok] = cent[ok]
        return pos, ok, resid, nval
    for k in np.where(nval >= 4)[0]:
        c, rr = sphere_center_fixed_R(ball_m[k, valid[k]].astype(float))
        pos[k], resid[k] = c, rr
    three = nval == 3
    pos[three] = cent[three]                     # fallback: centroid of 3
    present = nval >= 3
    return pos, present, resid, nval


# ------------------------------------------------------------ racket poses

def kabsch(A, B):
    """Rotation taking A onto B (both centred), as a matrix."""
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def robust_template(mk, max_frames=400, iters=3):
    """Marker template (M,3) centred, averaged over full-visibility frames.

    Motive marker noise on this rig is 2-5 mm, so a single seed frame (the venue
    extractor's choice) bakes that noise into every downstream normal. Iterate:
    align each full frame to the current template, average, repeat.
    """
    nf, M, _ = mk.shape
    full = np.where(np.isfinite(mk[:, :, 0]).all(1))[0]
    if len(full) < 10:
        return None, None, len(full)
    sel = full[np.linspace(0, len(full) - 1, min(max_frames, len(full))).astype(int)]
    F = mk[sel].astype(float)
    F = F - F.mean(1, keepdims=True)
    tpl = F[0].copy()
    for _ in range(iters):
        acc = np.zeros_like(tpl)
        for k in range(len(F)):
            acc += F[k] @ kabsch(tpl, F[k])          # rotate frame back onto tpl
        tpl = acc / len(F)
        tpl -= tpl.mean(0)
    # spread of each marker about the template after alignment
    err = np.zeros((len(F), M))
    for k in range(len(F)):
        err[k] = np.linalg.norm(F[k] @ kabsch(tpl, F[k]) - tpl, axis=1)
    return tpl, err.mean(0), len(full)


def rigid_pose(mk, tpl, min_markers=4, rms_gate=0.020):
    """Per-frame centroid + quaternion (template->world) by subset Kabsch."""
    nf, M, _ = mk.shape
    valid = np.isfinite(mk[:, :, 0])
    nval = valid.sum(1)
    cent = np.full((nf, 3), np.nan)
    quat = np.full((nf, 4), np.nan)
    rms = np.full(nf, np.nan)
    for k in np.where(nval >= min_markers)[0]:
        sel = valid[k]
        P = mk[k, sel].astype(float)
        c = P.mean(0)
        A = tpl[sel] - tpl[sel].mean(0)
        B = P - c
        Rk = kabsch(A, B)
        r = float(np.sqrt(((B - A @ Rk.T) ** 2).sum(1).mean()))
        if r > rms_gate:
            continue
        rms[k] = r
        quat[k] = Rot.from_matrix(Rk).as_quat()
        # centroid of the FULL template mapped to world (robust to visible subset)
        cent[k] = c - Rk @ tpl[sel].mean(0)
    return np.isfinite(quat).all(1), cent, quat, rms


def face_normal_local(tpl):
    """Blade-face normal in template coordinates + planarity of the marker set."""
    _, S, Vt = np.linalg.svd(tpl - tpl.mean(0))
    n = Vt[2]
    return n, float(np.abs((tpl - tpl.mean(0)) @ n).max()), (S / S[0]).tolist()


# ------------------------------------------------- blank-frame gap repair

def blank_frames(raw):
    """Frames where NO asset holds a single marker = empty frames in the export.

    This session drops roughly every 11th frame WHOLESALE: the ball, both
    rackets AND the static table markers vanish together. The table is bolted
    down and permanently visible, so a frame that loses it is not a tracking
    failure, it is a frame the exporter wrote empty. Real sample rate is
    therefore ~rate*10/11 on an unchanged rate-Hz time grid.
    """
    n = next(iter(raw.values())).shape[0]
    any_marker = np.zeros(n, bool)
    for mk in raw.values():
        any_marker |= np.isfinite(mk[:, :, 0]).any(1)
    return ~any_marker


def _slerp_fill(quat, present_old, present_new, t):
    """Slerp orientation onto frames that fill_short_gaps just repaired."""
    q = quat.copy()
    new = present_new & ~present_old
    if not new.any():
        return q
    src = np.where(present_old)[0]
    if len(src) < 2:
        return q
    from scipy.spatial.transform import Slerp
    tgt = np.where(new & (t >= t[src[0]]) & (t <= t[src[-1]]))[0]
    if len(tgt) == 0:
        return q
    sl = Slerp(t[src], Rot.from_quat(quat[src]))
    q[tgt] = sl(t[tgt]).as_quat()
    return q


def fill_short_gaps(pos, present, t, max_gap, allow, win=6):
    """Local-polynomial fill of short holes. Returns (pos, present, filled_mask).

    Only gaps of <= max_gap frames whose indices are ALL in `allow` are filled,
    from up to `win` real samples on each side (quadratic per axis, blended
    across the gap). At 360 Hz a 1-3 frame hole is 2.8-8.3 ms, over which a
    ballistic arc is quadratic to far below the 3 mm position noise -- but a
    hole landing on a bounce straddles a velocity discontinuity, so filled
    samples are MARKED and stage 1's contact exclusion zones drop them.
    """
    pos = pos.copy()
    present = present.copy()
    filled = np.zeros(len(present), bool)
    idx = np.where(~present)[0]
    if len(idx) == 0:
        return pos, present, filled
    breaks = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, breaks + 1)
    for grp in groups:
        a, b = grp[0], grp[-1]
        if len(grp) > max_gap or a == 0 or b == len(present) - 1:
            continue
        if not allow[grp].all():
            continue
        left = np.where(present[:a])[0][-win:]
        right = np.where(present[b + 1:])[0][:win] + b + 1
        if len(left) < 3 or len(right) < 3:
            continue
        both = np.concatenate([left, right])
        t0 = t[a]
        A = np.vstack([np.ones(len(both)), t[both] - t0, (t[both] - t0) ** 2]).T
        coef, *_ = np.linalg.lstsq(A, pos[both], rcond=None)
        tg = t[grp] - t0
        pos[grp] = np.vstack([np.ones(len(tg)), tg, tg ** 2]).T @ coef
        present[grp] = True
        filled[grp] = True
    return pos, present, filled


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c3d")
    ap.add_argument("out_dir")
    ap.add_argument("--center", choices=["sphere", "centroid"], default="sphere")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--no-spin", action="store_true",
                    help="skip the orientation solve (emit all-NaN quaternions)")
    ap.add_argument("--max-fill-gap", type=int, default=3,
                    help="repair holes up to this many frames (0 disables)")
    ap.add_argument("--fill-blank-only", action="store_true", default=True,
                    help="only repair holes caused by EMPTY export frames")
    ap.add_argument("--fill-any-gap", dest="fill_blank_only", action="store_false",
                    help="also repair genuine short occlusions")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(args.c3d).replace(".c3d", "")

    assets = (BALL_ASSET,) + RACKETS + (TABLE_ASSET,)
    t_start = time.time()
    raw, geom, unit, p95, n_unlab = read_assets(args.c3d, assets, chunk=args.chunk,
                                                required=(BALL_ASSET, TABLE_ASSET))
    rate = float(geom["rate"])
    nf = geom["data_nf"]
    t = np.arange(nf) / rate

    center, Rt, corners, posts, tqa = table_frame(raw[TABLE_ASSET])

    ball_w, ball_present, ball_resid, ball_nmk = ball_center_series(
        raw[BALL_ASSET], mode=args.center)

    # ---- ball ORIENTATION (spin) -------------------------------------------
    # The ball IS a rigid marker constellation on this session; see
    # ball_orientation.py for why the two obvious rigidity tests said otherwise.
    ori = None
    if not args.no_spin:
        from ball_orientation import solve_orientation_chained
        okm = np.isfinite(raw[BALL_ASSET][:, :, 0])
        ori = solve_orientation_chained(raw[BALL_ASSET].astype(float), okm, rate)
        # NaN the first frame of every run so a run boundary never produces a
        # spurious consecutive-quaternion difference downstream
        rid = ori["run_id"]
        starts = np.where((rid >= 0) & (np.r_[True, rid[1:] != rid[:-1]]))[0]
        ori["quat"][starts] = np.nan

    blank = blank_frames(raw)
    n_blank_raw = int(blank.sum())
    if args.max_fill_gap > 0:
        allow_blank = blank if args.fill_blank_only else np.ones(nf, bool)
        ball_w, ball_present, ball_filled = fill_short_gaps(
            ball_w, ball_present, t, args.max_fill_gap, allow_blank)
    else:
        ball_filled = np.zeros(nf, bool)
    ball_t = (ball_w - center) @ Rt

    paddles = {}
    for pn, asset in zip(("p1", "p2"), RACKETS):
        if asset not in raw:
            continue
        tpl, mk_spread, n_full = robust_template(raw[asset])
        if tpl is None:
            print(f"  WARNING: {asset} never fully visible ({n_full} frames) - skipped")
            continue
        present, cw, quat, rms = rigid_pose(raw[asset], tpl)
        if args.max_fill_gap > 0:
            cw, present_f, _ = fill_short_gaps(cw, present, t, args.max_fill_gap,
                                               blank if args.fill_blank_only
                                               else np.ones(nf, bool))
            quat = _slerp_fill(quat, present, present_f, t)
            present = present_f
        n0, planarity, svals = face_normal_local(tpl)
        nrm_w = np.full((nf, 3), np.nan)
        g = np.isfinite(quat).all(1)
        nrm_w[g] = np.einsum("kij,j->ki", Rot.from_quat(quat[g]).as_matrix(), n0)
        paddles[pn] = dict(asset=asset, present=present, cent_w=cw, quat=quat, rms=rms,
                           normal_w=nrm_w, template=tpl, planarity_m=planarity,
                           marker_spread_m=mk_spread, svals=svals, n_full=n_full)

    # ---- trim leading/trailing seconds and untracked head/tail ----
    keep = np.ones(nf, bool)
    ntr = int(round(TRIM_S * rate))
    keep[:ntr] = False
    keep[nf - ntr:] = False
    idx = np.where(ball_present & keep)[0]
    if len(idx) == 0:
        raise SystemExit("no tracked ball frames survive trimming")
    a, b = idx[0], idx[-1]
    sl = slice(a, b + 1)

    # ---- below-table cut (ball only; time axis preserved) ----
    below = np.isfinite(ball_t[:, 2]) & (ball_t[:, 2] < tqa["corner_plane_z_cut"]) \
        if "corner_plane_z_cut" in tqa else np.zeros(nf, bool)
    ball_valid = ball_present & ~below

    bpos_w = ball_w.copy(); bpos_w[~ball_valid] = np.nan
    bpos_t = ball_t.copy(); bpos_t[~ball_valid] = np.nan

    out = dict(
        t=t[sl], frame=np.arange(a, b + 1).astype(np.int32), rate=rate,
        ball_present=ball_valid[sl], ball_below_table=below[sl],
        ball_interpolated=ball_filled[sl], frame_blank_in_export=blank[sl],
        ball_pos_w_m=bpos_w[sl].astype(np.float32),
        ball_pos_t_m=bpos_t[sl].astype(np.float32),
        ball_quat_xyzw=(ori["quat"][sl] if ori is not None
                        else np.full((b - a + 1, 4), np.nan)).astype(np.float32),
        ball_spin_step_rms_m=(ori["step_rms"][sl] if ori is not None
                              else np.full(b - a + 1, np.nan)).astype(np.float32),
        ball_spin_run_id=(ori["run_id"][sl] if ori is not None
                          else np.full(b - a + 1, -1)).astype(np.int32),
        ball_sphere_resid_m=ball_resid[sl].astype(np.float32),
        ball_n_markers=ball_nmk[sl].astype(np.int8),
        ball_markers_w_m=raw[BALL_ASSET][sl].astype(np.float32),
        ball_sphere_radius_m=R_BALL,
        table_center_w_m=center, table_R=Rt,
        table_corners_w_m=corners, table_posts_w_m=posts,
        table_length_ext_m=tqa["length_ext_m"], table_width_ext_m=tqa["width_ext_m"],
    )
    for pn, P in paddles.items():
        out.update({
            f"{pn}_present": P["present"][sl],
            f"{pn}_pos_w_m": P["cent_w"][sl].astype(np.float32),
            f"{pn}_pos_t_m": ((P["cent_w"] - center) @ Rt)[sl].astype(np.float32),
            f"{pn}_quat_xyzw": P["quat"][sl].astype(np.float32),
            f"{pn}_normal_t": (P["normal_w"] @ Rt)[sl].astype(np.float32),
            f"{pn}_kabsch_rms_m": P["rms"][sl].astype(np.float32),
            f"{pn}_template_m": P["template"].astype(np.float32),
        })
    np.savez_compressed(os.path.join(args.out_dir, f"{name}.npz"), **out)

    # ------------------------------------------------------------ manifest
    pk = ball_valid[sl]
    gaps, s = [], None
    for i, v in enumerate(pk):
        if not v and s is None:
            s = i
        if v and s is not None:
            gaps.append((i - s) / rate * 1e3); s = None
    if s is not None:
        gaps.append((len(pk) - s) / rate * 1e3)
    d = np.linalg.norm(np.diff(bpos_w[sl], axis=0), axis=1)
    man = dict(
        take=name, source=os.path.abspath(args.c3d), rate_hz=rate,
        n_frames_in_file=int(nf), n_frames_in_header=int(geom["header_nf"]),
        truncated=bool(geom["truncated"]),
        truncation_note=("header/data differ by "
                         f"{geom['header_nf'] - geom['data_nf']} frame(s)"),
        unit_to_m=unit, coord_p95=round(p95, 2), n_unlabeled_cols=n_unlab,
        n_point_cols=int(geom["npts"]),
        kept_frames=[int(a), int(b)], kept_dur_s=round(float(t[b] - t[a]), 2),
        center_mode=args.center,
        ball_tracked_pct=round(float(np.mean(pk)) * 100, 2),
        ball_marker_hist={int(k): int(v) for k, v in
                          zip(*np.unique(ball_nmk[sl], return_counts=True))},
        ball_sphere_resid_median_mm=round(float(np.nanmedian(ball_resid[sl])) * 1e3, 3),
        export_blank_frames=n_blank_raw,
        export_blank_pct=round(n_blank_raw / nf * 100, 2),
        effective_sample_rate_hz=round(rate * (1 - n_blank_raw / nf), 1),
        spin_channel=bool(ori is not None),
        spin_solved_pct=(round(float(np.isfinite(ori["quat"][sl]).all(1).mean()) * 100, 2)
                         if ori is not None else 0.0),
        spin_step_rms_median_mm=(round(float(np.nanmedian(ori["step_rms"][sl])) * 1e3, 3)
                                 if ori is not None else None),
        spin_marker_radius_mm=(round(ori["radius_m"] * 1e3, 2)
                               if ori is not None else None),
        ball_interpolated_frames=int(ball_filled[sl].sum()),
        ball_interpolated_pct=round(float(ball_filled[sl].mean()) * 100, 2),
        max_fill_gap_frames=args.max_fill_gap,
        fill_blank_frames_only=bool(args.fill_blank_only),
        frozen_consecutive_positions=int(np.nansum(d < 1e-9)),
        n_gaps=len(gaps),
        gap_ms_median=round(float(np.median(gaps)), 1) if gaps else 0,
        gap_ms_p90=round(float(np.percentile(gaps, 90)), 1) if gaps else 0,
        gap_ms_max=round(float(np.max(gaps)), 1) if gaps else 0,
        table=dict(
            length_ext_m=round(tqa["length_ext_m"], 4),
            width_ext_m=round(tqa["width_ext_m"], 4),
            length_err_mm=round((tqa["length_ext_m"] - ITTF_LEN) * 1e3, 1),
            width_err_mm=round((tqa["width_ext_m"] - ITTF_WID) * 1e3, 1),
            corner_plane_rms_mm=round(tqa["corner_plane_rms_mm"], 2),
            tilt_vs_world_deg=round(tqa["tilt_vs_world_deg"], 3),
            static_rms_mm=round(tqa["static_rms_mm"], 3),
            net_x_in_table_frame_m=round(tqa["net_x_t_m"], 4),
            net_post_z_above_corner_plane_m=round(
                tqa["net_post_height_above_corner_plane_m"], 4),
            corners_t_m=[[round(v, 4) for v in c] for c in tqa["corners_t_m"]],
            posts_t_m=[[round(v, 4) for v in c] for c in tqa["posts_t_m"]],
            center_w_m=[round(float(v), 4) for v in center],
        ),
        extract_seconds=round(time.time() - t_start),
    )
    for pn, P in paddles.items():
        man[f"{pn}_asset"] = P["asset"]
        man[f"{pn}_tracked_pct"] = round(float(np.mean(P["present"][sl])) * 100, 2)
        man[f"{pn}_kabsch_rms_median_mm"] = round(
            float(np.nanmedian(P["rms"][sl])) * 1e3, 3)
        man[f"{pn}_planarity_mm"] = round(P["planarity_m"] * 1e3, 2)
        man[f"{pn}_template_singular_ratios"] = [round(v, 4) for v in P["svals"]]
        man[f"{pn}_marker_spread_mm"] = [round(float(v) * 1e3, 2)
                                         for v in P["marker_spread_m"]]
        man[f"{pn}_full_visibility_frames"] = int(P["n_full"])
    with open(os.path.join(args.out_dir, f"{name}_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(json.dumps(man, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
