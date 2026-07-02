"""Re-ground a retargeted motion clip into the HOPE table world frame (forward = +X).

The v2 clips were grounded facing world +Y (~82-86 deg): the robot's standing heading and its
swing/strike velocity point along +Y, which is the table WIDTH (lateral) in the HOPE frame
(geometry.py: +X = table length / toward opponent, Y = width, robot faces +X). This mis-orients the
world-frame racket-velocity command by ~90 deg relative to the deploy planner / table.

This script applies a single global yaw rotation about world +Z so the clip's FRAME-0 pelvis heading
becomes exactly +X (the robot stands facing the opponent; the torso still rotates naturally during the
swing). It rotates every world-frame array and leaves the local joint data untouched:

    body_pos_w      rotated about the pelvis frame-0 XY pivot (XY), z unchanged   [includes root = body 0]
    body_quat_w     pre-multiplied by the yaw quaternion (world-frame reorientation)
    body_lin_vel_w  rotated as a world vector
    body_ang_vel_w  rotated as a world vector
    joint_pos       UNCHANGED (local joint space is invariant to a world yaw)
    joint_vel       UNCHANGED
    fps             UNCHANGED

Originals are never overwritten; the corrected clip is written to a separate path.

    python scripts/reground_hope_frame.py \
        --in  artifacts/hope_forehand:v2/motion.npz \
        --out artifacts/hope_forehand_hopex/motion.npz
    python scripts/reground_hope_frame.py \
        --in  artifacts/hope_backhand:v2/motion.npz \
        --out artifacts/hope_backhand_hopex/motion.npz \
        --strike-phase 0.33
"""
from __future__ import annotations

import argparse
import os

import numpy as np

PELVIS = 0
RACKET = 31


def yaw_of(q_wxyz: np.ndarray) -> float:
    """Heading (yaw about world +Z) of a wxyz quaternion. 0 = +X, +pi/2 = +Y."""
    w, x, y, z = q_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2 for wxyz quats, broadcast over leading dims."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def reground(inp: str, out: str, theta_deg: float | None, strike_phase: float) -> None:
    d = np.load(inp)
    data = {k: d[k].copy() for k in d.files}
    P = data["body_pos_w"].astype(np.float64)         # (T, B, 3)
    Q = data["body_quat_w"].astype(np.float64)        # (T, B, 4) wxyz
    T = P.shape[0]

    yaw0 = yaw_of(Q[0, PELVIS])
    theta = np.radians(theta_deg) if theta_deg is not None else -yaw0   # default: heading -> +X
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s], [s, c]])                  # 2x2 XY rotation
    pivot = P[0, PELVIS, :2].copy()                   # rotate about the pelvis frame-0 XY (keep stance)
    qz = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])  # yaw quat wxyz

    def rot_xy(arr):                                  # positions: rotate XY about pivot, keep z
        out_a = arr.copy()
        xy = arr[..., :2] - pivot
        out_a[..., 0] = pivot[0] + xy[..., 0] * c - xy[..., 1] * s
        out_a[..., 1] = pivot[1] + xy[..., 0] * s + xy[..., 1] * c
        return out_a

    def rot_vec(arr):                                 # world vectors: rotate XY, keep z (no pivot)
        out_a = arr.copy()
        x = arr[..., 0].copy(); y = arr[..., 1].copy()
        out_a[..., 0] = x * c - y * s
        out_a[..., 1] = x * s + y * c
        return out_a

    data["body_pos_w"] = rot_xy(P).astype(d["body_pos_w"].dtype)
    qz_b = np.broadcast_to(qz, Q.shape)
    data["body_quat_w"] = quat_mul(qz_b, Q).astype(d["body_quat_w"].dtype)
    if "body_lin_vel_w" in data:
        data["body_lin_vel_w"] = rot_vec(data["body_lin_vel_w"].astype(np.float64)).astype(d["body_lin_vel_w"].dtype)
    if "body_ang_vel_w" in data:
        data["body_ang_vel_w"] = rot_vec(data["body_ang_vel_w"].astype(np.float64)).astype(d["body_ang_vel_w"].dtype)
    # joint_pos / joint_vel / fps: UNCHANGED (already copied verbatim).

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **data)

    # ---- report ----
    f = int(round(strike_phase * (T - 1)))
    rp_old, rv_old = P[f, RACKET], d["body_lin_vel_w"][f, RACKET].astype(np.float64)
    rp_new = data["body_pos_w"][f, RACKET].astype(np.float64)
    rv_new = data["body_lin_vel_w"][f, RACKET].astype(np.float64)
    ynew = yaw_of(data["body_quat_w"][0, PELVIS].astype(np.float64))
    print(f"\n=== {os.path.basename(os.path.dirname(inp))} -> {out} ===")
    print(f"  frame-0 pelvis yaw: {np.degrees(yaw0):+7.2f} deg  ->  {np.degrees(ynew):+7.2f} deg   (theta = {np.degrees(theta):+7.2f})")
    print(f"  strike frame {f}/{T}  (phase {strike_phase})")
    print(f"    racket pos  OLD (+Y) ({rp_old[0]:+.3f},{rp_old[1]:+.3f},{rp_old[2]:+.3f})  ->  NEW (+X) ({rp_new[0]:+.3f},{rp_new[1]:+.3f},{rp_new[2]:+.3f})")
    print(f"    racket vel  OLD (+Y) ({rv_old[0]:+.3f},{rv_old[1]:+.3f},{rv_old[2]:+.3f})  ->  NEW (+X) ({rv_new[0]:+.3f},{rv_new[1]:+.3f},{rv_new[2]:+.3f})")
    print(f"    |v| preserved: {np.linalg.norm(rv_old):.4f} -> {np.linalg.norm(rv_new):.4f}")
    print(f"    dominant horizontal vel axis: {'+X' if abs(rv_new[0]) >= abs(rv_new[1]) else '+Y'} "
          f"(|vx|={abs(rv_new[0]):.3f} vs |vy|={abs(rv_new[1]):.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--theta-deg", type=float, default=None,
                    help="explicit yaw (deg) to apply; default = -(frame-0 pelvis yaw) so heading -> +X")
    ap.add_argument("--strike-phase", type=float, default=0.47, help="phase used only for the report")
    args = ap.parse_args()
    reground(args.inp, args.out, args.theta_deg, args.strike_phase)


if __name__ == "__main__":
    main()
