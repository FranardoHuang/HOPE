"""Utilities for normalizing retargeted motions into the HOPE table frame.

HOPE table-tennis training assumes the robot stands on the P1 side facing world
+X, and the return swing velocity is +X-dominant. GVHMR/GMR clips recovered from
ordinary videos can be globally yawed, so this module rotates saved world-frame
motion arrays while leaving local joint trajectories unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PELVIS_BODY = 0
RACKET_BODY = 31


@dataclass(frozen=True)
class HopeFrameReport:
    """Summary of the yaw transform applied to a motion clip."""

    yaw_before_rad: float
    yaw_after_rad: float
    theta_rad: float
    pivot_xy: tuple[float, float]


def yaw_of_wxyz(q_wxyz: np.ndarray) -> float:
    """Return heading yaw about world +Z for a wxyz quaternion. 0 means +X."""

    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2`` for wxyz quaternions, broadcast over leading dims."""

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def rotate_motion_to_hope_x(
    data: dict[str, np.ndarray],
    *,
    theta_deg: float | None = None,
    pelvis_body: int = PELVIS_BODY,
) -> tuple[dict[str, np.ndarray], HopeFrameReport]:
    """Rotate a motion dict so frame-0 pelvis heading faces HOPE +X.

    The input/output dict follows the BeyondMimic motion ``.npz`` contract. The
    following world-frame arrays are rotated when present:

    - ``body_pos_w`` about the frame-0 pelvis XY pivot, preserving Z
    - ``body_quat_w`` by pre-multiplying a world-yaw quaternion
    - ``body_lin_vel_w`` and ``body_ang_vel_w`` as world vectors

    Joint arrays and metadata are copied unchanged.
    """

    if "body_pos_w" not in data or "body_quat_w" not in data:
        raise KeyError("motion data must contain body_pos_w and body_quat_w")

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    pos = np.asarray(out["body_pos_w"], dtype=np.float64)
    quat = np.asarray(out["body_quat_w"], dtype=np.float64)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (T, B, 3), got {pos.shape}")
    if quat.ndim != 3 or quat.shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape (T, B, 4), got {quat.shape}")
    if pelvis_body < 0 or pelvis_body >= pos.shape[1]:
        raise IndexError(f"pelvis_body index {pelvis_body} out of range for {pos.shape[1]} bodies")

    yaw_before = yaw_of_wxyz(quat[0, pelvis_body])
    theta = np.radians(theta_deg) if theta_deg is not None else -yaw_before
    c, s = float(np.cos(theta)), float(np.sin(theta))
    pivot = pos[0, pelvis_body, :2].copy()
    qz = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float64)

    def rot_xy(arr: np.ndarray) -> np.ndarray:
        arr64 = np.asarray(arr, dtype=np.float64)
        rotated = arr64.copy()
        xy = arr64[..., :2] - pivot
        rotated[..., 0] = pivot[0] + xy[..., 0] * c - xy[..., 1] * s
        rotated[..., 1] = pivot[1] + xy[..., 0] * s + xy[..., 1] * c
        return rotated

    def rot_vec(arr: np.ndarray) -> np.ndarray:
        arr64 = np.asarray(arr, dtype=np.float64)
        rotated = arr64.copy()
        x = arr64[..., 0].copy()
        y = arr64[..., 1].copy()
        rotated[..., 0] = x * c - y * s
        rotated[..., 1] = x * s + y * c
        return rotated

    out["body_pos_w"] = rot_xy(out["body_pos_w"]).astype(data["body_pos_w"].dtype, copy=False)

    qz_b = np.broadcast_to(qz, quat.shape)
    quat_rot = quat_mul_wxyz(qz_b, quat)
    quat_rot /= np.maximum(np.linalg.norm(quat_rot, axis=-1, keepdims=True), 1e-8)
    out["body_quat_w"] = quat_rot.astype(data["body_quat_w"].dtype, copy=False)

    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        if key in out:
            out[key] = rot_vec(out[key]).astype(data[key].dtype, copy=False)

    yaw_after = yaw_of_wxyz(out["body_quat_w"][0, pelvis_body].astype(np.float64))
    report = HopeFrameReport(
        yaw_before_rad=float(yaw_before),
        yaw_after_rad=float(yaw_after),
        theta_rad=float(theta),
        pivot_xy=(float(pivot[0]), float(pivot[1])),
    )
    return out, report
