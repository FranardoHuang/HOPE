"""Sim-to-real frame alignment: HOPE table frame -> policy (env-origin) frame.

Pure numpy, ROS-free, unit-testable (mirrors the planner.py / node.py split).

THE TWO "WORLD"S
----------------
* HOPE table frame (the real planner's + mocap relay's ``world``):
    origin = P1 near-side LEFT table corner, ON the table SURFACE.
    +x toward the opponent (P2), +y left (table spans y 0..-1.525),
    +z up with z=0 at the table surface -> the FLOOR is z = -0.76.
* Policy frame (what model_15200 / model_9000 were trained in):
    origin = the ground point under the robot's base_link at episode start,
    +x = robot-forward at start, +y left, +z = height above the FLOOR.

A RacketCommand from the real hope_planner (frame_id="world") and the /P1/pose
mocap base pose are in the TABLE frame; every quantity inside the 180-D obs
(base_pos_w, torso_pos_w, racket target pos/vel, base_target_xy) must be in the
POLICY frame. This module is the single place that conversion happens.

The transform is rigid, yaw-only (the floor is level):
    p_policy = Rz(-yaw0) @ (p_table - origin_table)
    v_policy = Rz(-yaw0) @ v_table
with origin_table = [x0, y0, floor_z] (floor_z = -0.76: the policy z-origin is
the FLOOR, 0.76 m below the table-frame z-origin) and yaw0 = the robot's boot
heading in the table frame (0.0 when the robot is placed facing the table, +x).

YAW OBSERVABILITY (read before trusting)
----------------------------------------
The arena mocap provides POSITION ONLY for the robot rigid body (see
mocap/HOPE_Motion_Capture_System_..._Setup.md): yaw is NOT measured. Two
consequences, both handled here:
  * ``yaw0`` (boot heading in the table frame) must come from a PLACEMENT
    CONVENTION (put the robot down facing +x/opponent) or a manual measurement
    -- it cannot be auto-calibrated from mocap.
  * the pelvis-IMU yaw is unreferenced and drifts boot-to-boot, so the policy's
    in-episode yaw is IMU-relative. ``ImuYawAligner`` captures the IMU yaw while
    the robot stands still at boot and re-zeroes it, so IMU yaw == policy yaw
    under the same placement convention. (Same fix as the C++ deploy runner's
    yaw-align, commit d17c: "add yaw-align detect to avoid imu yaw drifting".)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .frame_math import quat_mul, quat_rotate_inverse, yaw_quat

TABLE_FLOOR_Z = -0.76   # floor height in the HOPE table frame (table surface = 0)


def _quat_z(yaw: float) -> np.ndarray:
    """Quaternion (w,x,y,z) of a pure z rotation by ``yaw`` radians."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class TableToPolicy:
    """Rigid yaw-only transform from the HOPE table frame to the policy frame.

    ``origin_xy_table`` is the robot's boot base_link ground point (XY, table
    frame); ``yaw_table`` its boot heading (radians, table frame). Both come
    from config, and the XY can be refined once live mocap samples arrive
    (see :class:`OriginCapture`).
    """

    origin_xy_table: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.7625]))
    yaw_table: float = 0.0
    floor_z_table: float = TABLE_FLOOR_Z
    origin_source: str = "config"        # "config" | "mocap_capture"

    def __post_init__(self):
        self.origin_xy_table = np.asarray(self.origin_xy_table, dtype=float)

    @property
    def _origin(self) -> np.ndarray:
        return np.array([self.origin_xy_table[0], self.origin_xy_table[1], self.floor_z_table])

    def pos(self, p_table: np.ndarray) -> np.ndarray:
        """Point in table frame -> point in policy frame."""
        return _rz(self.yaw_table).T @ (np.asarray(p_table, dtype=float) - self._origin)

    def vec(self, v_table: np.ndarray) -> np.ndarray:
        """Free vector (velocity, normal) in table frame -> policy frame."""
        return _rz(self.yaw_table).T @ np.asarray(v_table, dtype=float)

    def quat(self, q_table: np.ndarray) -> np.ndarray:
        """Orientation (w,x,y,z) in table frame -> policy frame."""
        return quat_mul(_quat_z(-self.yaw_table), np.asarray(q_table, dtype=float))

    def set_origin_xy(self, xy_table: np.ndarray, source: str) -> None:
        self.origin_xy_table = np.asarray(xy_table, dtype=float).copy()
        self.origin_source = source


class OriginCapture:
    """Average the first N mocap base positions to refine the policy origin.

    The policy origin is the ground point under base_link at boot. Averaging a
    short stationary window rejects mocap jitter. Call :meth:`push` with each
    raw /P1/pose position (TABLE frame); returns the captured XY once done.
    """

    def __init__(self, n_samples: int = 50):
        self.n_samples = max(int(n_samples), 1)
        self._buf: list[np.ndarray] = []
        self.done = False
        self.result_xy: np.ndarray | None = None

    def push(self, p_table: np.ndarray) -> np.ndarray | None:
        if self.done:
            return None
        self._buf.append(np.asarray(p_table[:2], dtype=float).copy())
        if len(self._buf) >= self.n_samples:
            self.result_xy = np.mean(np.asarray(self._buf), axis=0)
            self.done = True
            return self.result_xy
        return None


class ImuYawAligner:
    """Re-zero the (unreferenced, drifting) pelvis-IMU yaw at boot.

    Captures the mean IMU yaw over the first N samples while the robot stands
    still, then :meth:`correct` composes every subsequent IMU quaternion with
    Rz(boot_yaw_policy - captured_yaw) so the corrected yaw equals the robot's
    heading in the POLICY frame (0 at boot by construction: policy +x is the
    robot's boot forward). Roll/pitch pass through untouched (gravity-observable
    -> already correct).
    """

    def __init__(self, n_samples: int = 50, boot_yaw_policy: float = 0.0):
        self.n_samples = max(int(n_samples), 1)
        self.boot_yaw_policy = float(boot_yaw_policy)
        self._yaw_sin = 0.0
        self._yaw_cos = 0.0
        self._count = 0
        self.offset: float | None = None    # radians to ADD to the IMU yaw

    @staticmethod
    def _yaw_of(q: np.ndarray) -> float:
        w, x, y, z = q
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @property
    def done(self) -> bool:
        return self.offset is not None

    def push(self, q_imu: np.ndarray) -> None:
        if self.done:
            return
        yaw = self._yaw_of(np.asarray(q_imu, dtype=float))
        # circular mean (robust across the +/-pi wrap)
        self._yaw_sin += math.sin(yaw)
        self._yaw_cos += math.cos(yaw)
        self._count += 1
        if self._count >= self.n_samples:
            captured = math.atan2(self._yaw_sin, self._yaw_cos)
            self.offset = self.boot_yaw_policy - captured

    def correct(self, q_imu: np.ndarray) -> np.ndarray:
        """IMU quaternion -> policy-frame base quaternion (yaw re-zeroed)."""
        if self.offset is None:
            return np.asarray(q_imu, dtype=float)
        return quat_mul(_quat_z(self.offset), np.asarray(q_imu, dtype=float))


@dataclass
class TargetGate:
    """Reachability gate on the BASE-RELATIVE racket target (deploy safety).

    The real planner publishes wherever the ball crosses its hitting plane; it
    knows nothing about the policy's trained target box. Driving the policy far
    out-of-distribution is how deploy falls happen, so out-of-box targets are
    rejected (the runner stands instead of lunging). Defaults are the UNION of
    the model_15200 and model_9000 validated boxes plus a small margin -- set
    them per-model from config for a tight gate.
    """

    x_range: tuple = (0.20, 0.90)
    y_abs_max: float = 0.85
    z_range: tuple = (0.55, 1.40)
    speed_max: float = 3.5
    enabled: bool = True

    def check(self, pos_gate: np.ndarray, vel: np.ndarray) -> tuple[bool, str]:
        """Return (ok, reason). ``pos_gate`` = [x_base_rel, y_base_rel, z_above_floor]:
        XY in the yaw-heading base frame (x fwd, y left) but z ABSOLUTE height above
        the floor (policy frame) — the training boxes are env-frame z, and a
        base-relative z would shift with the (possibly nominal) pelvis estimate."""
        if not self.enabled:
            return True, ""
        x, y, z = (float(v) for v in pos_gate)
        if not (self.x_range[0] <= x <= self.x_range[1]):
            return False, f"x={x:.2f} outside [{self.x_range[0]:.2f},{self.x_range[1]:.2f}]"
        if abs(y) > self.y_abs_max:
            return False, f"|y|={abs(y):.2f} > {self.y_abs_max:.2f}"
        if not (self.z_range[0] <= z <= self.z_range[1]):
            return False, f"z={z:.2f} outside [{self.z_range[0]:.2f},{self.z_range[1]:.2f}]"
        speed = float(np.linalg.norm(vel))
        if speed > self.speed_max:
            return False, f"|v|={speed:.2f} > {self.speed_max:.2f}"
        return True, ""


def base_relative_target(target_pos_policy: np.ndarray, base_pos_policy: np.ndarray,
                         base_quat_policy: np.ndarray) -> np.ndarray:
    """Racket target in the yaw-heading base frame (the obs-10 convention)."""
    yq = yaw_quat(np.asarray(base_quat_policy, dtype=float))
    return quat_rotate_inverse(
        yq, np.asarray(target_pos_policy, dtype=float) - np.asarray(base_pos_policy, dtype=float))
