"""Racket forward kinematics for the 175-D deploy_parity observation.

``racket_pos_pelvis(q_isaac_31)`` returns the racket (pingpang_red_Link)
position expressed in the PELVIS frame, from the 10-revolute-joint chain
pelvis_link -> right_wrist_yaw_Link plus the fixed pingpang mount offset.

World position is then:  base_pos_w + R(base_quat_w) @ racket_pos_pelvis(q).

Verbatim port of hope_training/whole_body_tracking/scripts/racket_fk_ref.py
(the ground-truth reference, validated against Isaac's pingpang_red_Link world
pos to < 1e-4 m; the C++ pp_racket_fk.hpp is a port of the same file). Do not
"improve" the numbers — they are the URDF joint origins.
"""

from __future__ import annotations

import numpy as np


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rot_rpy(r, p, y):
    """URDF fixed-axis roll-pitch-yaw = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


_AXIS = {"x": _rot_x, "y": _rot_y, "z": _rot_z}

# The 10-joint chain: (isaac q-index, origin_xyz, origin_rpy, axis)
_CHAIN = [
    (2,  (0.0, 0.0, 0.0),                                                (0.0, 0.0, 0.0),                 "z"),  # waist_yaw
    (5,  (0.0, 0.0, 0.0),                                                (0.0, 0.0, 0.0),                 "x"),  # waist_roll
    (8,  (-0.0199999999998296, 0.0, 0.00500000000000012),                (0.0, 0.0, 0.0),                 "y"),  # waist_pitch
    (13, (0.03030000000207, -0.148224512812557, 0.283817906204843),      (-0.0872664625997163, 0.0, 0.0), "y"),  # r_shoulder_pitch
    (18, (0.0, -0.0589999999982601, 0.0),                                (0.0872664625997163, 0.0, 0.0),  "x"),  # r_shoulder_roll
    (22, (0.0, -0.0100000000003387, -0.147499999999977),                 (0.0, 0.0, 0.0),                 "z"),  # r_shoulder_yaw
    (24, (0.00999999997356363, 0.0, -0.132999999990091),                 (0.0, 0.0, 0.0),                 "y"),  # r_elbow
    (26, (0.129, 0.0, -0.00999999999980283),                             (0.0, 0.0, 0.0),                 "x"),  # r_wrist_roll
    (28, (0.0600000000000003, 0.0, 0.0),                                 (0.0, 0.0, 0.0),                 "y"),  # r_wrist_pitch
    (30, (0.046, 0.0, 0.0),                                              (0.0, 0.0, 0.0),                 "z"),  # r_wrist_yaw
]

# Fixed mount offset: pingpang_red_Link origin relative to right_wrist_yaw_Link
# (right_hand_pingpang_joint is identity, so this is applied at wrist_yaw frame).
_MOUNT = np.array([0.21021, 0.032078, 0.032036], dtype=np.float64)


def racket_pos_pelvis(q_isaac_31) -> np.ndarray:
    """Racket position in the pelvis frame for a 31-vec of Isaac-order joint angles."""
    q = np.asarray(q_isaac_31, dtype=np.float64).reshape(-1)
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    for idx, xyz, rpy, axis in _CHAIN:
        # T_i = Translate(origin_xyz) * Rot_rpy(origin_rpy) * Rot_axis(q_i)
        Ri = _rot_rpy(rpy[0], rpy[1], rpy[2]) @ _AXIS[axis](float(q[idx]))
        t = t + R @ np.array(xyz, dtype=np.float64)
        R = R @ Ri
    return t + R @ _MOUNT
