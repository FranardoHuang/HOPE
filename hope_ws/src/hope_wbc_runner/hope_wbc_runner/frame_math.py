"""Quaternion / frame math (numpy, scalar-first w,x,y,z).

Ported VERBATIM from hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py
so the deployed observation is bit-for-bit the same convention the policy was
trained and sim2sim-validated with (IsaacLab + MuJoCo both use w,x,y,z). Do not
"improve" these — they must match training exactly.
"""

import math

import numpy as np


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (w * w + x * x + y * y + z * z)


def mat_from_quat(q):
    """3x3 rotation matrix R (world<-body) from quaternion (w,x,y,z)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate(q, v):
    return mat_from_quat(q) @ v


def quat_rotate_inverse(q, v):
    return mat_from_quat(q).T @ v


def yaw_quat(q):
    """Quaternion of the yaw-only (Z) rotation of q."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def subtract_frame_transforms(t01, q01, t02, q02):
    """Pose of frame 2 expressed in frame 1. Returns (t12, q12)."""
    q10 = quat_inv(q01)
    t12 = quat_rotate(q10, t02 - t01)
    q12 = quat_mul(q10, q02)
    return t12, q12


def projected_gravity_body(base_quat_w):
    """Gravity unit vector in the body frame: R_body^T @ [0,0,-1]."""
    return quat_rotate_inverse(base_quat_w, np.array([0.0, 0.0, -1.0]))
