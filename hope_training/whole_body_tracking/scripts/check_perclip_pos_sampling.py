"""Validate the per-clip racket target-POSITION sampling BEFORE training (no Isaac needed).

Reads `pos_range_per_clip` (and the shared `pos_x_range` / `racket_pos_y_abs_range` / `pos_z_range`
fallback) straight from cfg/task/HOPEPingPong.yaml, samples many targets exactly as
RacketTargetCommand._sample_targets_uniform would, and prints the forehand/backhand target-POSITION
distribution plus physical-plausibility checks (below table surface, below pelvis, distance from the
measured reference strike point).

Purpose: confirm the config places the targets where the imitated swing actually strikes
(forehand z~0.84, backhand z~1.22 at strike_phase 0.50) WITHOUT launching a training run. Pure
sampling check — does not touch the policy, env, rewards, or observations. Mirrors
`scripts/check_perclip_vel_sampling.py`.

    python scripts/check_perclip_pos_sampling.py            # uses HOPEPingPong.yaml
    python scripts/check_perclip_pos_sampling.py --n 500000 --seed 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

# Frame fact (tasks/table_tennis/geometry.py): table surface is 0.76 m above the floor (training z=0=floor).
TABLE_H = 0.76
PELVIS_Z = 0.93
# Measured reference racket strike points (scripts/strike_phase.py): where the imitated swing's racket is.
REF_STRIKE = {"forehand": np.array([0.603, -0.323, 0.840]), "backhand": np.array([0.582, 0.282, 1.218])}


def _sample_shared(rng, n, clip, rk):
    """Replicate the shared-box branch: x fixed range, |y| with per-clip sign, z range."""
    x = rng.uniform(*rk["pos_x_range"], n)
    ymag = rng.uniform(*rk["racket_pos_y_abs_range"], n)
    fh_sign = -1.0 if bool(rk.get("forehand_on_negative_y", True)) else 1.0
    sign = fh_sign if clip == "forehand" else -fh_sign
    y = sign * ymag
    z = rng.uniform(*rk["pos_z_range"], n)
    return np.stack([x, y, z], 1)


def _sample_box(rng, n, box):
    """Replicate the per-clip branch: signed x/y/z box added to origin (origin=0 here)."""
    x = rng.uniform(*box["x"], n)
    y = rng.uniform(*box["y"], n)
    z = rng.uniform(*box["z"], n)
    return np.stack([x, y, z], 1)


def _stats(name, P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ref = REF_STRIKE[name]
    far = np.linalg.norm(P - ref, axis=1) > 0.25
    print(f"\n--- {name} (n={len(P)}) ---")
    print(f"  x: mean {x.mean():.3f}  range [{x.min():.3f}, {x.max():.3f}]   (ref strike x={ref[0]:.2f})")
    print(f"  y: mean {y.mean():+.3f} range [{y.min():+.3f}, {y.max():+.3f}]  (ref strike y={ref[1]:+.2f})")
    print(f"  z: mean {z.mean():.3f}  range [{z.min():.3f}, {z.max():.3f}]   (ref strike z={ref[2]:.2f})")
    print(f"  below table surface (z<{TABLE_H}) : {(z < TABLE_H).mean() * 100:5.1f}%")
    print(f"  below pelvis (z<{PELVIS_Z})        : {(z < PELVIS_Z).mean() * 100:5.1f}%")
    print(f"  |y| outside table (>0.7625)       : {(np.abs(y) > 0.7625).mean() * 100:5.1f}%")
    print(f"  FAR from reference strike (>0.25m): {far.mean() * 100:5.1f}%  (small = targets match the swing)")


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
    per_clip = rk.get("pos_range_per_clip")
    rng = np.random.default_rng(args.seed)

    print(f"[check] yaml = {args.yaml}")
    print(f"[check] shared box (fallback): x={rk['pos_x_range']} |y|={rk['racket_pos_y_abs_range']} z={rk['pos_z_range']}")
    if per_clip is None:
        print("[check] pos_range_per_clip ABSENT -> both clips use the shared box (old behavior).")
        for name in ("forehand", "backhand"):
            _stats(name, _sample_shared(rng, args.n, name, rk))
    else:
        print("[check] pos_range_per_clip ENABLED:")
        for name in ("forehand", "backhand"):
            box = per_clip[name]
            print(f"        {name}: x={box['x']} y={box['y']} z={box['z']}")
            _stats(name, _sample_box(rng, args.n, box))
    print("\n[check] GOAL: 'FAR from reference strike' small (targets reachable by the imitated swing),")
    print("        'below table surface' ~0%, and backhand z mean near its reference z=1.22.")


if __name__ == "__main__":
    main()
