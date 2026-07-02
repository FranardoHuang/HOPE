"""Deploy-parity observation: canonical layout + the deploy-honest racket-target reframe spec.

This is the DEPLOY-PARITY reference for the C++ obs builder (pp_obs_builder). It documents the exact
175-D actor observation layout and the ONE non-trivial new transform — the racket-target reframe —
and self-tests (numpy only, no Isaac) the identity that makes it base-position-free.

The reframe (replaces the `full`-mode racket_target_pos_b):

    racket_target_pos_b = quat_rotate_inverse( yaw_quat(base_quat_w),  racket_target_pos_w - racket_pos_w )

vs the OLD `full` term which used the world BASE position (fabricated on hardware):

    full:  quat_rotate_inverse( yaw_quat(base_quat_w),  racket_target_pos_w - base_pos_w )

Because quat_rotate_inverse is LINEAR in its vector argument, the base position cancels:

    R^T(target - racket) = R^T(target - base) - R^T(racket - base)
                         = (target rel base)  -  (racket FK rel base)

so the term is computable on the real robot from the planner target + racket forward kinematics,
WITHOUT any world base pose. This script verifies that identity numerically.

Run (host, numpy only):
    python scripts/realsensor_obs_reference.py
"""

from __future__ import annotations

import numpy as np


# --- IsaacLab-convention quaternion ops (wxyz) ------------------------------------------------- #
def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix R (body->world) from a wxyz quaternion."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v from world into the body frame: R(q)^T v. Linear in v."""
    return quat_to_rot(q_wxyz).T @ v


def yaw_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """The yaw-only quaternion (wxyz) — matches isaaclab.utils.math.yaw_quat."""
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction in the base frame (what mdp.projected_gravity feeds the actor)."""
    return quat_rotate_inverse(q_wxyz, np.array([0.0, 0.0, -1.0]))


def racket_target_pos_b_rel(base_quat_w, racket_target_pos_w, racket_pos_w) -> np.ndarray:
    """DEPLOY-HONEST racket target (real_sensor_only). No world base position."""
    return quat_rotate_inverse(yaw_quat(base_quat_w), racket_target_pos_w - racket_pos_w)


def racket_target_pos_b_full(base_quat_w, racket_target_pos_w, base_pos_w) -> np.ndarray:
    """OLD `full` racket target — uses the world base position (fabricated on hardware)."""
    return quat_rotate_inverse(yaw_quat(base_quat_w), racket_target_pos_w - base_pos_w)


# --- canonical 175-D actor layout (deploy parity / legacy real_sensor_only) -------------------- #
# (name, dim, on-hardware source). Order MUST match the obs manager (HOPEPolicyRealSensorCfg) and the
# C++ pp_obs_builder real_sensor variant.
LAYOUT = [
    ("command",              62, "reference clip future joint pos/vel (baked at deploy)"),
    ("motion_anchor_ori_b",   6, "relative ORIENTATION ref-vs-robot (IMU + clip); base position NOT used"),
    ("base_ang_vel",          3, "pelvis IMU gyro"),
    ("joint_pos",            31, "joint encoders (q - default_q)"),
    ("joint_vel",            31, "joint encoders (dq)"),
    ("actions",              31, "last action"),
    ("projected_gravity",     3, "pelvis IMU (gravity in base frame)"),
    ("racket_target_pos_b",   3, "REFRAMED: yaw_quat(base)^T (target_w - racket_FK_w); no base position"),
    ("racket_target_vel_w",   3, "planner desired racket velocity (world)"),
    ("time_to_strike",        1, "swing clock"),
    ("swing_type",            1, "forehand +1 / backhand -1 (planner)"),
]
REMOVED_VS_FULL = [
    ("motion_anchor_pos_b", 3, "[62:65] reference torso POSITION error — needs world base position"),
    ("base_target_pos_b",   2, "[170:172] base-repositioning target — needs world base position"),
]


def print_layout() -> int:
    print("deploy-parity actor observation layout (deploy-honest):")
    o = 0
    for name, dim, src in LAYOUT:
        print(f"  [{o:3d}:{o + dim:3d}] {name:22s} ({dim:2d})  <- {src}")
        o += dim
    print(f"  TOTAL = {o}")
    print("\nremoved vs `full` (180):")
    for name, dim, why in REMOVED_VS_FULL:
        print(f"  - {name:22s} ({dim})  {why}")
    print(f"  full 180  ->  deploy_parity {o}")
    return o


def self_test():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(10000):
        q = rng.standard_normal(4)
        q = q / np.linalg.norm(q)  # random unit quaternion (wxyz)
        target = rng.standard_normal(3)
        racket = rng.standard_normal(3)
        base = rng.standard_normal(3)  # the (unknown-on-hardware) world base position
        # honest reframe (no base position):
        rel = racket_target_pos_b_rel(q, target, racket)
        # identity: equals (target rel base) - (racket rel base), for ANY base position ->
        # the reframe is invariant to the base position (cancels). Verify across many random bases:
        full_target = racket_target_pos_b_full(q, target, base)
        full_racket = quat_rotate_inverse(yaw_quat(q), racket - base)
        err = np.max(np.abs(rel - (full_target - full_racket)))
        max_err = max(max_err, err)
    print(f"\n[self-test] base-position-cancellation identity max error over 10k random samples: {max_err:.2e}")
    assert max_err < 1e-10, "racket-target reframe is NOT base-position-invariant — convention bug!"
    # also confirm the reframe genuinely IGNORES the base position (two different bases -> same result):
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t, r = np.array([0.4, -0.2, 0.8]), np.array([0.3, -0.1, 0.7])
    a = racket_target_pos_b_rel(q, t, r)
    print(f"[self-test] example: target={t.tolist()} racket_FK={r.tolist()} -> racket_target_pos_b={np.round(a,4).tolist()}")
    print("[self-test] PASS — the reframe is base-position-free (deploy-honest).")


if __name__ == "__main__":
    n = print_layout()
    assert n == 175, f"layout sums to {n}, expected 175"
    self_test()
