#!/usr/bin/env python3
"""Trim a BeyondMimic motion .npz to a strike-centered window (SMASH-scale experiment).

SMASH's strike segments are ~1.08 s (0.54 s each side of contact); ours are 2.6-2.8 s with the
strike off-center (forehand frame 65/139, backhand 44/132). A tighter strike-centered clip gives
more swings per episode and less imitation of irrelevant lead-in/follow-through. This tool cuts a
symmetric window around the given strike frame and reports the NEW strike phase (which becomes
exactly (strike-start)/(len-1), typically 0.5 for a symmetric cut).

Usage:
  python scripts/trim_motion_clip.py --in fh.npz --out fh_trim.npz --strike-frame 65 --half-window-s 0.7
"""

from __future__ import annotations

import argparse

import numpy as np

TIME_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strike-frame", type=int, required=True,
                   help="hand-annotated contact frame in the SOURCE clip (see cfg/strike_annotations.yaml)")
    p.add_argument("--half-window-s", type=float, default=0.7,
                   help="seconds kept on EACH side of the strike (SMASH uses 0.54)")
    args = p.parse_args()

    data = dict(np.load(args.inp))
    fps = int(np.array(data["fps"]).reshape(-1)[0])
    n = data[TIME_KEYS[0]].shape[0]
    half = int(round(args.half_window_s * fps))
    lo = max(0, args.strike_frame - half)
    hi = min(n, args.strike_frame + half + 1)
    if lo == 0 or hi == n:
        print(f"[trim] WARNING: window clipped by clip bounds (lo={lo}, hi={hi}, n={n}) — "
              f"strike phase will be off-center")

    for k in TIME_KEYS:
        assert data[k].shape[0] == n, f"{k} length {data[k].shape[0]} != {n}"
        data[k] = data[k][lo:hi]

    new_len = hi - lo
    new_phase = (args.strike_frame - lo) / (new_len - 1)
    np.savez(args.out, **data)
    print(f"[trim] {args.inp}: {n} frames (strike {args.strike_frame}, phase {args.strike_frame/(n-1):.3f})")
    print(f"[trim] -> {args.out}: {new_len} frames (strike {args.strike_frame - lo}, "
          f"NEW strike phase {new_phase:.3f}) @ {fps} Hz = {new_len/fps:.2f} s")
    print(f"[trim] pass to training:  racket.strike_phase_per_clip=[...] with {new_phase:.3f} for this clip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
