"""Check HOPE motion orientation and target/reference alignment without Isaac.

This is the quick gate before launching a long training run:

1. Frame-0 pelvis yaw should be near 0 deg, so the robot faces HOPE +X.
2. The URDF racket control-point velocity at the configured strike phase should
   be +X-dominant.  The control point is the ``pingpang_red_Link`` origin, not
   the bare wrist origin.
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
from racket_geometry_contract import RACKET_SITE_OFFSET_WRIST_M, rigid_point_velocity  # noqa: E402


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


def _quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to world<-body rotation matrices."""

    q = np.asarray(quat, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)
    w, x, y, z = (q[..., i] for i in range(4))
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[..., 0, 1] = 2.0 * (x * y - w * z)
    out[..., 0, 2] = 2.0 * (x * z + w * y)
    out[..., 1, 0] = 2.0 * (x * y + w * z)
    out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[..., 1, 2] = 2.0 * (y * z - w * x)
    out[..., 2, 0] = 2.0 * (x * z - w * y)
    out[..., 2, 1] = 2.0 * (y * z + w * x)
    out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def racket_control_point_series(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_link_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    *,
    wrist_body: int = RACKET_BODY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return position and analytic velocity of the same URDF racket point.

    For a rigid point fixed at wrist-local ``r``, the only correct world-frame
    velocity is ``v_point = v_link_origin + omega x (R r)``.  The third input
    therefore MUST be a link-origin velocity (Isaac ``body_link_lin_vel_w``),
    not the misleading legacy ``body_lin_vel_w`` COM velocity.  Keeping this
    helper explicit makes accidental COM/link mixing fail code review.
    """

    pos = np.asarray(body_pos_w, dtype=np.float64)
    quat = np.asarray(body_quat_w, dtype=np.float64)
    lin = np.asarray(body_link_lin_vel_w, dtype=np.float64)
    ang = np.asarray(body_ang_vel_w, dtype=np.float64)
    if not (pos.ndim == quat.ndim == lin.ndim == ang.ndim == 3):
        raise ValueError("motion body arrays must all have shape (T, B, C)")
    if not (pos.shape[-1] == lin.shape[-1] == ang.shape[-1] == 3 and quat.shape[-1] == 4):
        raise ValueError("expected xyz position/velocity and wxyz quaternion arrays")
    if any(a.shape[:2] != pos.shape[:2] for a in (quat, lin, ang)):
        raise ValueError("motion body arrays disagree on (T, B)")
    if wrist_body < 0 or wrist_body >= pos.shape[1]:
        raise IndexError(f"wrist body index {wrist_body} out of range for {pos.shape[1]} bodies")

    rot = _quat_to_rot_wxyz(quat[:, wrist_body])
    offset_w = np.einsum("tij,j->ti", rot, RACKET_SITE_OFFSET_WRIST_M)
    point_pos = pos[:, wrist_body] + offset_w
    point_vel = rigid_point_velocity(lin[:, wrist_body], ang[:, wrist_body], offset_w)
    return point_pos, point_vel


def clean_point_velocity(
    point_pos_w: np.ndarray, frame: int, fps: float, window: int
) -> np.ndarray:
    """Training-parity centered difference of the racket control-point path."""

    p = np.asarray(point_pos_w, dtype=np.float64)
    f = int(frame)
    W = max(1, int(window))
    if p.ndim != 2 or p.shape[1] != 3 or not (0 <= f < len(p)):
        raise ValueError(f"point path/frame invalid: shape={p.shape}, frame={f}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    lo, hi = max(0, f - W), min(len(p) - 1, f + W)
    if hi == lo:
        raise ValueError("clip is too short to differentiate the racket point")
    # Strike frames are interior in accepted clips, so this is exactly the
    # RacketTargetCommand denominator 2*W*step_dt.  Use the actual clamped span
    # at an edge instead of silently scaling a one-sided diagnostic down.
    return (p[hi] - p[lo]) * float(fps) / float(hi - lo)


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default=os.path.join(WBT, "cfg/task/HOPEPingPong.yaml"))
    ap.add_argument("--clip", action="append", default=None, help="name:path.npz, repeatable")
    ap.add_argument("--yaw-thresh-deg", type=float, default=5.0)
    ap.add_argument("--pos-center-thresh", type=float, default=0.03)
    ap.add_argument("--vel-center-thresh", type=float, default=0.15)
    args = ap.parse_args(argv)

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
        if "body_ang_vel_w" not in d.files:
            raise SystemExit(
                f"{path}: body_ang_vel_w is required to reconstruct the racket control-point velocity"
            )
        ang = d["body_ang_vel_w"].astype(np.float64)
        phase = phases.get(name, float(rk["strike_phase"]))
        frame = int(round(phase * (pos.shape[0] - 1)))
        yaw_deg = float(np.degrees(yaw_of_wxyz(quat[0, PELVIS_BODY])))
        # Motion NPZ kinematics schema 2 intentionally stores link-origin position but
        # COM-point velocity, so it cannot supply the wrist-origin velocity
        # required by rigid-point FK without the MJCF COM offset.  The formal
        # target path already uses a centered difference of the same site
        # position; derive the gate from that identical, point-consistent path.
        control_pos, _ = racket_control_point_series(pos, quat, np.zeros_like(pos), ang)
        ref_pos = control_pos[frame]
        clean_enabled = bool(rk.get("clean_reference_strike_velocity", False))
        clean_window = int(rk.get("clean_strike_vel_window", 2))
        ref_vel_clean = clean_point_velocity(
            control_pos, frame, float(np.asarray(d["fps"]).reshape(-1)[0]), clean_window
        )
        if not clean_enabled:
            raise SystemExit(
                f"{path}: clean_reference_strike_velocity=false cannot be certified by this gate: "
                "motion body_lin_vel_w is a COM-point channel, while the controlled racket point "
                "is the URDF site. Enable the point-consistent clean target or add an explicit "
                "MJCF COM->link conversion; refusing to mix points."
            )
        ref_vel = ref_vel_clean
        speed = float(np.linalg.norm(ref_vel))
        x_dominant = abs(ref_vel[0]) >= abs(ref_vel[1])
        forward = ref_vel[0] > 0.0

        print(f"\n--- {name}: {path} ---")
        print(f"  T={pos.shape[0]} strike_phase={phase:.3f} frame={frame}")
        print(f"  frame0_yaw_deg={yaw_deg:+.2f}")
        print(
            f"  control_point=pingpang_red_Link origin; "
            f"ref_pos=({ref_pos[0]:+.3f},{ref_pos[1]:+.3f},{ref_pos[2]:+.3f})"
        )
        vel_source = f"clean +-{clean_window}-frame FD"
        print(
            f"  ref_vel[{vel_source}]=({ref_vel[0]:+.3f},{ref_vel[1]:+.3f},{ref_vel[2]:+.3f}) "
            f"|v|={speed:.3f}"
        )

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
