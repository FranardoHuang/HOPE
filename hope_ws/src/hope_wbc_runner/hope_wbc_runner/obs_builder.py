"""180-D observation builder for model_15200 (pure numpy, ROS-free, testable).

Builds the EXACT observation the policy was trained + sim2sim-validated with,
in the verified order (see mujoco_eval_onnx.py):

  idx  size  field
   1   62    command            = cat(ref_joint_pos[31], ref_joint_vel[31])
   2    3    motion_anchor_pos_b  ref torso pos in the ROBOT torso (anchor) frame
   3    6    motion_anchor_ori_b  6D rot rep (first 2 cols of R) of ref torso ori in robot torso frame
   4    3    base_ang_vel         pelvis gyro, PELVIS BODY frame (IMU)
   5   31    joint_pos            q - default_joint_pos     (Isaac articulation order)
   6   31    joint_vel            qdot                       (Isaac articulation order)
   7   31    actions              last raw policy action
   8    3    projected_gravity    gravity unit vec, PELVIS BODY frame (IMU)
   9    2    base_target_pos_b    desired base XY, yaw-heading base frame
  10    3    racket_target_pos_b  desired racket pos, yaw-heading base frame
  11    3    racket_target_vel_w  desired racket velocity, WORLD frame
  12    1    time_to_strike       seconds until strike
  13    1    swing_type           +1 forehand / -1 backhand
                                                              total = 180

FRAME NOTE (sim2real): the policy's "world" is the robot-ground / env-origin
frame (origin under the robot, +x robot-forward, +y left, +z up = height above
floor). planner_imitate's ``base_link`` targets are already in THIS frame, so the
runner treats the incoming RacketCommand position/velocity as ``*_w`` directly.
The anchor-position term needs the robot torso WORLD pose; on a real robot with
no global localisation this is the known sim2real gap (see README, "open items").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frame_math import (
    mat_from_quat,
    projected_gravity_body,
    quat_rotate_inverse,
    subtract_frame_transforms,
    yaw_quat,
)

ANCHOR_TRACKED_IDX = 7   # torso_Link in the 14-body tracked order
OBS_DIM = 180
N_JOINTS = 31


@dataclass
class RobotState:
    """Robot state needed to build the obs (all world poses in the policy frame)."""

    base_pos_w: np.ndarray            # (3) pelvis world position
    base_quat_w: np.ndarray           # (4) pelvis world quat (w,x,y,z)
    torso_pos_w: np.ndarray           # (3) torso/anchor world position
    torso_quat_w: np.ndarray          # (4) torso/anchor world quat
    base_ang_vel_b: np.ndarray        # (3) pelvis gyro in pelvis BODY frame
    q: np.ndarray                     # (31) joint pos, Isaac order
    qd: np.ndarray                    # (31) joint vel, Isaac order


@dataclass
class RacketTarget:
    """Planner target, already in the policy world frame (see FRAME NOTE)."""

    pos_w: np.ndarray                 # (3) racket target position
    vel_w: np.ndarray                 # (3) racket target velocity
    swing_sign: float                 # +1 forehand / -1 backhand
    time_to_strike: float             # seconds
    base_target_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))  # desired base XY (world)


def build_obs(refs: dict, state: RobotState, target: RacketTarget,
              last_action: np.ndarray, default_q: np.ndarray) -> np.ndarray:
    """Assemble the 180-D observation. ``refs`` are the ONNX side-outputs at the
    current time_step (keys: joint_pos[31], joint_vel[31], body_pos_w[14,3],
    body_quat_w[14,4])."""
    # 1. command (62)
    command = np.concatenate([np.asarray(refs["joint_pos"]), np.asarray(refs["joint_vel"])])

    # 2/3. ref anchor (torso) expressed in robot anchor (torso) frame
    ref_anchor_pos_w = np.asarray(refs["body_pos_w"])[ANCHOR_TRACKED_IDX]
    ref_anchor_quat_w = np.asarray(refs["body_quat_w"])[ANCHOR_TRACKED_IDX]
    pos_b, ori_q = subtract_frame_transforms(
        state.torso_pos_w, state.torso_quat_w, ref_anchor_pos_w, ref_anchor_quat_w)
    ori_b6 = mat_from_quat(ori_q)[:, :2].reshape(-1)   # [R00,R01,R10,R11,R20,R21]

    # 4. base angular velocity (3), body frame
    base_ang_vel = np.asarray(state.base_ang_vel_b)

    # 5/6. joint pos/vel (31 each)
    joint_pos_rel = np.asarray(state.q) - np.asarray(default_q)
    joint_vel_rel = np.asarray(state.qd)

    # 7. last action (31)
    last_action = np.asarray(last_action)

    # 8. projected gravity (3), body frame
    proj_grav = projected_gravity_body(state.base_quat_w)

    # 9. base target XY in yaw-heading base frame (2)
    yq = yaw_quat(state.base_quat_w)
    delta_base = np.array([target.base_target_xy[0] - state.base_pos_w[0],
                           target.base_target_xy[1] - state.base_pos_w[1], 0.0])
    base_tgt_b = quat_rotate_inverse(yq, delta_base)[:2]

    # 10. racket target pos in yaw-heading base frame (3)
    racket_tgt_b = quat_rotate_inverse(yq, np.asarray(target.pos_w) - np.asarray(state.base_pos_w))

    # 11. racket target velocity, world (3)
    racket_vel_w = np.asarray(target.vel_w)

    # 12/13. time_to_strike (1), swing_type (1)
    tts = np.array([target.time_to_strike])
    swing = np.array([target.swing_sign])

    obs = np.concatenate([
        command, pos_b, ori_b6, base_ang_vel, joint_pos_rel, joint_vel_rel,
        last_action, proj_grav, base_tgt_b, racket_tgt_b, racket_vel_w, tts, swing,
    ]).astype(np.float64)
    assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape} != {OBS_DIM}"
    return obs


def synthetic_state_from_refs(refs: dict, default_q: np.ndarray) -> RobotState:
    """A 'perfectly-tracking' robot state derived from the reference frame.

    For dry-run with NO robot/sim connected: the robot is assumed to exactly
    track the reference (q = ref joint pos, base/torso at the ref root/anchor
    pose, zero measured gyro). Lets the full pipeline run standalone and produces
    a deterministic, representative action to log. NOT physically real - use real
    state subscriptions in shadow/hardware modes.
    """
    bp = np.asarray(refs["body_pos_w"])
    bq = np.asarray(refs["body_quat_w"])
    root_pos, root_quat = bp[0], bq[0]              # pelvis
    anchor_pos, anchor_quat = bp[ANCHOR_TRACKED_IDX], bq[ANCHOR_TRACKED_IDX]
    return RobotState(
        base_pos_w=root_pos.copy(), base_quat_w=root_quat.copy(),
        torso_pos_w=anchor_pos.copy(), torso_quat_w=anchor_quat.copy(),
        base_ang_vel_b=np.zeros(3),
        q=np.asarray(refs["joint_pos"]).copy(), qd=np.asarray(refs["joint_vel"]).copy(),
    )
