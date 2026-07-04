"""Check HOPE motion orientation and target/reference alignment without Isaac.

This is the quick gate before launching a long training run:

1. Frame-0 pelvis yaw should be near 0 deg, so the robot faces HOPE +X.
2. Racket velocity at the configured strike phase should be +X-dominant.
3. ``reference_perturbed`` target mode should start exactly at each teacher
   clip's strike position/velocity. If ``uniform`` is used, the script checks
   the configured box centers against the same reference states.

Example:

    python scripts/check_motion_target_alignment.py \
      --clip forehand:artifacts/hope_forehand_hopex/motion.npz \
      --clip backhand:artifacts/hope_backhand_hopex/motion.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hope_frame_utils import PELVIS_BODY, RACKET_BODY, yaw_of_wxyz  # noqa: E402


DEFAULT_CLIPS = {
    "forehand": os.path.join(WBT, "../../hope_training/motions/preprocessed/hope_forehand_hopex.npz"),
    "backhand": os.path.join(WBT, "../../hope_training/motions/preprocessed/hope_backhand_hopex.npz"),
}


def _resolve(path: str) -> str:
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(WBT, path)


def _center_box(box: dict[str, list[float]]) -> np.ndarray:
    return np.array(
        [
            0.5 * (float(box["x"][0]) + float(box["x"][1])),
            0.5 * (float(box["y"][0]) + float(box["y"][1])),
            0.5 * (float(box["z"][0]) + float(box["z"][1])),
        ],
        dtype=np.float64,
    )


def _phase_map(cfg: dict) -> dict[str, float]:
    rk = cfg["racket"]
    phases = rk.get("strike_phase_per_clip") or []
    names = ("forehand", "backhand")
    if len(phases) == len(names):
        return {name: float(ph) for name, ph in zip(names, phases)}
    return {name: float(rk["strike_phase"]) for name in names}


def _clip_specs(args) -> dict[str, str]:
    if not args.clip:
        return DEFAULT_CLIPS
    out = {}
    for spec in args.clip:
        name, sep, path = spec.partition(":")
        if not sep:
            raise SystemExit(f"--clip must be name:path, got {spec!r}")
        out[name] = _resolve(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default=os.path.join(WBT, "cfg/task/HOPEPingPong.yaml"))
    ap.add_argument("--clip", action="append", default=None, help="name:path.npz, repeatable")
    ap.add_argument("--yaw-thresh-deg", type=float, default=5.0)
    ap.add_argument("--pos-center-thresh", type=float, default=0.03)
    ap.add_argument("--vel-center-thresh", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)
    rk = cfg["racket"]
    phases = _phase_map(cfg)
    clips = _clip_specs(args)
    target_mode = str(rk.get("target_mode", "uniform"))

    print(f"[check] yaml={args.yaml}")
    print(f"[check] target_mode={target_mode}")
    failed = False

    pos_boxes = rk.get("pos_range_per_clip") or {}
    vel_boxes = rk.get("vel_range_per_clip") or {}

    for name, path in clips.items():
        d = np.load(path)
        pos = d["body_pos_w"].astype(np.float64)
        quat = d["body_quat_w"].astype(np.float64)
        vel = d["body_lin_vel_w"].astype(np.float64)
        phase = phases.get(name, float(rk["strike_phase"]))
        frame = int(round(phase * (pos.shape[0] - 1)))
        yaw_deg = float(np.degrees(yaw_of_wxyz(quat[0, PELVIS_BODY])))
        ref_pos = pos[frame, RACKET_BODY]
        ref_vel = vel[frame, RACKET_BODY]
        speed = float(np.linalg.norm(ref_vel))
        x_dominant = abs(ref_vel[0]) >= abs(ref_vel[1])
        forward = ref_vel[0] > 0.0

        print(f"\n--- {name}: {path} ---")
        print(f"  T={pos.shape[0]} strike_phase={phase:.3f} frame={frame}")
        print(f"  frame0_yaw_deg={yaw_deg:+.2f}")
        print(f"  ref_pos=({ref_pos[0]:+.3f},{ref_pos[1]:+.3f},{ref_pos[2]:+.3f})")
        print(f"  ref_vel=({ref_vel[0]:+.3f},{ref_vel[1]:+.3f},{ref_vel[2]:+.3f}) |v|={speed:.3f}")

        if abs(yaw_deg) > args.yaw_thresh_deg:
            print(f"  FAIL yaw: expected |yaw| <= {args.yaw_thresh_deg:.1f} deg")
            failed = True
        if not (x_dominant and forward):
            print("  FAIL velocity direction: expected +X-dominant strike velocity")
            failed = True

        if target_mode == "reference_perturbed":
            print("  target_center_pos = ref_pos exactly by construction")
            print("  target_center_vel = ref_vel * ref_vel_scale exactly by construction")
            print(
                "  long_run_half_extents: "
                f"pos={tuple(rk.get('ref_perturb_pos', (0.0, 0.0, 0.0)))} "
                f"vel={tuple(rk.get('ref_perturb_vel', (0.0, 0.0, 0.0)))} "
                f"normal={float(rk.get('ref_perturb_normal', 0.0)):.3f}"
            )
        else:
            if name in pos_boxes:
                pc = _center_box(pos_boxes[name])
                pe = float(np.linalg.norm(pc - ref_pos))
                print(f"  uniform_pos_center=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:+.3f}) err={pe:.3f}")
                if pe > args.pos_center_thresh:
                    print(f"  FAIL pos center: err > {args.pos_center_thresh:.3f} m")
                    failed = True
            if name in vel_boxes:
                vc = _center_box(vel_boxes[name])
                ve = float(np.linalg.norm(vc - ref_vel))
                print(f"  uniform_vel_center=({vc[0]:+.3f},{vc[1]:+.3f},{vc[2]:+.3f}) err={ve:.3f}")
                if ve > args.vel_center_thresh:
                    print(f"  FAIL vel center: err > {args.vel_center_thresh:.3f} m/s")
                    failed = True

    print("\n[check] " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
