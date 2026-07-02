"""Validate the per-clip racket target-velocity sampling BEFORE training (no Isaac needed).

Reads `vel_range_per_clip` (and the shared `vel_*_range` fallback) straight from
cfg/task/HOPEPingPong.yaml, samples many targets exactly as RacketTargetCommand._sample_targets_uniform
would (independent uniform boxes per axis), and prints the forehand/backhand target-SPEED distribution.

Purpose: confirm the config produces the intended speeds (forehand mean ~2.7 m/s, backhand mean ~2.0 m/s)
WITHOUT launching a training run. This is a pure sampling check — it does not touch the policy, the env,
rewards, or observations.

    python scripts/check_perclip_vel_sampling.py            # uses HOPEPingPong.yaml
    python scripts/check_perclip_vel_sampling.py --n 500000 --seed 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml


def _box_speed_stats(rng, n, x_r, y_r, z_r):
    vx = rng.uniform(*x_r, n)
    vy = rng.uniform(*y_r, n)
    vz = rng.uniform(*z_r, n)
    spd = np.sqrt(vx * vx + vy * vy + vz * vz)
    return dict(mean=spd.mean(), std=spd.std(),
                p10=np.percentile(spd, 10), p50=np.percentile(spd, 50), p90=np.percentile(spd, 90),
                min=spd.min(), max=spd.max())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    wbt = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default=os.path.join(wbt, "cfg/task/HOPEPingPong.yaml"))
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)
    rk = cfg["racket"]
    shared = (rk["vel_x_range"], rk["vel_y_range"], rk["vel_z_range"])
    per_clip = rk.get("vel_range_per_clip")

    rng = np.random.default_rng(args.seed)
    print(f"[check] yaml = {args.yaml}")
    print(f"[check] shared box (fallback): x={shared[0]} y={shared[1]} z={shared[2]}")
    if per_clip is None:
        print("[check] vel_range_per_clip ABSENT -> both clips use the shared box (old behavior).")
        clips = {"forehand(shared)": shared, "backhand(shared)": shared}
    else:
        print("[check] vel_range_per_clip ENABLED:")
        clips = {}
        for name in ("forehand", "backhand"):
            axes = per_clip[name]
            box = (axes["x"], axes["y"], axes["z"])
            clips[name] = box
            print(f"[check]   {name}: x={box[0]} y={box[1]} z={box[2]}")

    print("\n" + "=" * 78)
    print(f"{'clip':18s}{'mean':>9s}{'std':>9s}{'p10':>9s}{'p50':>9s}{'p90':>9s}{'min':>9s}{'max':>9s}")
    print("-" * 78)
    stats = {}
    for name, box in clips.items():
        s = _box_speed_stats(rng, args.n, *box)
        stats[name] = s
        print(f"{name:18s}" + "".join(f"{s[k]:9.3f}" for k in ("mean", "std", "p10", "p50", "p90", "min", "max")))
    print("=" * 78)

    # Sanity checks against the design intent (forehand ~2.7, backhand ~2.0).
    fh = next((v for k, v in stats.items() if k.startswith("forehand")), None)
    bh = next((v for k, v in stats.items() if k.startswith("backhand")), None)
    if fh and bh:
        print(f"\nforehand mean speed = {fh['mean']:.3f} m/s   (intent ~2.7)")
        print(f"backhand mean speed = {bh['mean']:.3f} m/s   (intent ~2.0)")
        print(f"forehand - backhand mean = {fh['mean'] - bh['mean']:.3f} m/s "
              f"({'OK: backhand is slower' if bh['mean'] < fh['mean'] else 'WARNING: backhand not slower'})")


if __name__ == "__main__":
    main()
