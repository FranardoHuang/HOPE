"""HOPE racket-target observation terms.

These wrap :class:`RacketTargetCommand`. The actor (policy) group should use only the *desired*
quantities the planner provides at deploy time (HITTER actor observation, Table I):

* :func:`racket_target_pos_b`  — desired racket position relative to base (3)
* :func:`racket_target_vel_w`  — desired racket velocity in world frame (3)
* :func:`time_to_strike`       — time remaining until strike (1)
* :func:`base_target_pos_b`    — desired base XY position relative to base (2)
* :func:`base_position_table`  — base root position relative to table-surface center (3)
* :func:`base_orientation_table_6d` — full base orientation in the table frame (6)
* :func:`base_lin_vel_heading` — root rigid-body COM linear velocity in the yaw-heading frame (3)
* :func:`station_anchor_err_b` — world station anchor minus current base XY, base frame (2;
  R10c station_obs flag, appended after the face channel = 179 -> 181)

The desired racket *normal* and the *actual* racket state are privileged/critic-only or used by
the reward; they are intentionally NOT in the HITTER actor observation (the racket is never sensed
on hardware). :func:`swing_type` is provided for a unified forehand+backhand policy variant; the
HOPE default trains separate policies and does not need it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, quat_rotate_inverse, yaw_quat

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand, face_tracking_pair
from whole_body_tracking.tasks.tracking.mdp.stage1_question_bank import face_command_obs_vector

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


# --- R-a actor leg-reference masking (reward_staged_design 2026-07-08 §⑥) ------------------- #
_LEG_JOINT_EXPR = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
_N_LEG_JOINTS = 12  # 2 x (hip pitch/roll/yaw + knee + ankle pitch/roll); loud error if not


def _leg_mask_indices(cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve + cache the leg-joint articulation indices for the actor command mask.

    RUNTIME-DERIVED via ``robot.find_joints`` — never hardcoded: the command layout is the Isaac
    articulation (BFS-interleaved) joint order, and a wrong index table would be a policy-killing
    experiment (design R-a risk row). The resolved names/indices are printed ONCE to the launch
    log; pre-registration reads that printout, not any table. Raises loudly unless exactly 12 leg
    joints resolve.
    """
    cached = getattr(cmd, "_actor_leg_mask_ids", None)
    if cached is not None:
        return cached
    ids, names = cmd.robot.find_joints(_LEG_JOINT_EXPR)
    ids = [int(i) for i in ids]
    if len(ids) != _N_LEG_JOINTS:
        raise RuntimeError(
            f"generated_commands_actor_leg_masked: expected {_N_LEG_JOINTS} leg joints from "
            f"find_joints({_LEG_JOINT_EXPR}), got {len(ids)}: {list(zip(ids, names))} — refusing "
            "to mask a wrong dim set (policy-killing if miscounted)."
        )
    n_joints = cmd.joint_pos.shape[1]
    pos_ids = torch.tensor(ids, dtype=torch.long, device=cmd.device)
    vel_ids = pos_ids + n_joints
    print(
        "[actor_leg_ref_mask] R-a ACTIVE — actor command leg dims -> default stand + zero vel "
        f"(critic untouched). {len(ids)} leg joints (articulation idx: name): "
        + ", ".join(f"{i}: {n}" for i, n in zip(ids, names))
        + f" | masked command dims pos={pos_ids.tolist()} vel={vel_ids.tolist()} "
        f"(command dim = {2 * n_joints})",
        flush=True,
    )
    cmd._actor_leg_mask_ids = (pos_ids, vel_ids)
    return cmd._actor_leg_mask_ids


def generated_commands_actor_leg_masked(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """R-a: the MOTION command (62-D ``cat([joint_pos, joint_vel])``) with the 24 LEG dims fed the
    DEFAULT STAND joint positions + ZERO velocities — the actor stops seeing the leg reference
    (HITTER's critic-only reference, observation side), while pos/ori/vel of the upper body keep
    the swing style. 人话:actor 眼里腿参考=站姿常数,critic 照旧全看。Same output shape/order as
    ``mdp.generated_commands`` (zero obs-contract cost); wired by train.py's task.actor_leg_ref_mask
    override, which swaps ONLY ``observations.policy.command.func`` (the critic keeps
    ``generated_commands``). Zero-OOD: during the pre-swing hold the command already equals
    stand-pose + zero-vel for every joint, so the masked values are a distribution the policy sees
    constantly. The leg indices are runtime-derived and printed (see ``_leg_mask_indices``)."""
    cmd = env.command_manager.get_term(command_name)  # MotionCommand
    pos_ids, vel_ids = _leg_mask_indices(cmd)
    out = torch.cat([cmd.joint_pos, cmd.joint_vel], dim=1)  # fresh tensor; safe to write in place
    out[:, pos_ids] = cmd.robot.data.default_joint_pos[:, pos_ids]
    out[:, vel_ids] = 0.0
    return out


# --- actor (policy) observations: desired targets only ------------------------------------ #
def racket_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket pos rel-base (yaw frame). PRIVILEGED — uses world base position (`full` mode)."""
    return _cmd(env, command_name).racket_target_pos_b()


def racket_target_pos_rel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket pos relative to the CURRENT racket (FK), yaw frame. DEPLOY-HONEST (no world
    base position; see :meth:`RacketTargetCommand.racket_target_pos_b_rel`). Used by the deploy-parity
    actor contract (legacy task name: `real_sensor_only`). A1: reads the ACTOR-visible target view
    (delayed/jittered when target latency is on; the live tensor otherwise)."""
    return _cmd(env, command_name).racket_target_pos_b_rel()


def racket_target_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket velocity, world frame. ACTOR term — A1: reads the ACTOR-visible view
    (delayed/jittered when target latency is on; the live tensor otherwise, byte-identical).
    The critic uses :func:`racket_target_vel_w_live`."""
    return _cmd(env, command_name).actor_racket_target_vel_w()


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """TRUE live time remaining until strike (s), used by the privileged critic/default actor.

    The explicit atomic planner-tuple training modes wire the policy term to
    :func:`actor_time_to_strike` in ``train.py`` while leaving this live critic source untouched.
    """
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def actor_time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actor-visible planner TTS: live, source-timestamp compensated, or stale negative control."""
    return _cmd(env, command_name).actor_time_to_strike().unsqueeze(-1)


def base_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).base_target_pos_b()


def _base_position_table_from_tensors(
    base_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    *,
    table_near_x: float,
    table_surface_z: float,
    table_length: float,
) -> torch.Tensor:
    """Return base-root XYZ relative to the env-local table-surface center."""

    table_center = base_pos_w.new_tensor(
        (
            float(table_near_x) + 0.5 * float(table_length),
            0.0,
            float(table_surface_z),
        )
    )
    return base_pos_w - env_origins - table_center


def _base_orientation_table_6d_from_quat(
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return table/world-aligned base orientation as R[:, :2], row-major.

    The six values are ``R00,R01,R10,R11,R20,R21``, exactly the flattening
    convention already used by ``motion_anchor_ori_b``.  This continuous
    representation carries roll, pitch and yaw without Euler discontinuities
    or quaternion sign ambiguity.
    """

    matrix = matrix_from_quat(base_quat_w)
    return matrix[..., :2].reshape(matrix.shape[0], -1)


def base_position_table(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Absolute base XYZ in the calibrated table frame.

    Simulation reads the root pose directly.  Deployment supplies the same
    quantity from the calibrated mocap base pose; task target positions remain
    in their existing relative channels.
    """

    from whole_body_tracking.tasks.table_tennis import geometry as table_geometry

    command = _cmd(env, command_name)
    return _base_position_table_from_tensors(
        command.base_pos_w,
        env.scene.env_origins,
        table_near_x=float(command.cfg.vb_table_near_x),
        table_surface_z=float(command.cfg.vb_table_surface_z),
        table_length=float(table_geometry.TABLE_LENGTH),
    )


def base_orientation_table_6d(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Absolute base roll/pitch/yaw information in the calibrated table frame.

    The table axes are aligned with the ActionBall simulation world.  The
    deploy producer must use the calibrated mocap orientation in those same
    axes, not an engage-relative IMU yaw.
    """

    return _base_orientation_table_6d_from_quat(
        _cmd(env, command_name).base_quat_w
    )


def base_lin_vel_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Root rigid-body COM linear velocity in the base yaw-heading frame.

    This matches the existing relative base/racket task frame while keeping
    world vertical independent of transient roll/pitch.  The deployment
    producer is a causal estimator with OptiTrack position as its absolute
    anchor and optional IMU-accelerometer propagation.  It must match this
    exact COM point, including the calibrated marker-to-root/COM offset and
    angular-velocity cross-offset term, before rotating world velocity by the
    inverse OptiTrack yaw.
    """

    command = _cmd(env, command_name)
    return quat_rotate_inverse(
        yaw_quat(command.base_quat_w),
        command.robot.data.root_lin_vel_w,
    )


def station_anchor_err_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """R10c 站位锚误差(2)——station_obs 旗标追加在 actor 观测尾部(179→181)。人话:世界系
    常数锚点(出生点)减去当前 base XY,旋进 base 系;不小心漂移了这两个数自己变大,策略因此
    始终有世界系位置基准(franco 07-09:"就算不需要移动,它也是一个锚")。语义 = jiayi Hitter
    的 base_target_pos_b(+2 站位通道)同构物,数学共用 RacketTargetCommand._target_xy_err_b;
    区别只在目标是 reset 常数而非采样站位。锚点常存在(buffer 无条件初始化),旗标关 = 不进
    观测,契约逐位不变。"""
    return _cmd(env, command_name).station_anchor_err_b()


def swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Forehand (+1) / backhand (-1). Only needed for a unified (single) policy. A1: delayed with
    the target when latency is on (the flag rides the same planner->runner message as the target)."""
    return _cmd(env, command_name).actor_swing_sign().unsqueeze(-1)


def action_one_hot(
    env: ManagerBasedRLEnv,
    command_name: str,
    expected_actions: int,
) -> torch.Tensor:
    """Actor-visible local action identity for an arbitrary-size motion bank.

    ``swing_type`` is only a forehand/backhand family feature.  Two forehand
    motions therefore receive the same sign even though they have different
    reference trajectories.  A task-first policy must be able to distinguish
    those actions when the requested task is otherwise identical, so it gets a
    categorical one-hot indexed by the *loaded motion bank's* local clip slot.

    This is intentionally not the stable planner ``action_uid``.  A UID is an
    opaque registry identity and feeding its numeric value to a neural network
    would invent an ordinal relationship.  The action catalog maps UID to this
    dense local slot before inference.
    """

    expected = int(expected_actions)
    if expected <= 0:
        raise ValueError(f"expected_actions must be positive, got {expected_actions!r}")
    command = _cmd(env, command_name)
    motion = command._motion()
    actual = int(motion.motion.num_segments) if motion._multiseg else 1
    if actual != expected:
        raise RuntimeError(
            "task-first action observation contract mismatch: "
            f"expected {expected} loaded action(s), got {actual}"
        )
    if motion._multiseg:
        clip = motion.clip_id
    else:
        clip = torch.zeros(command.num_envs, dtype=torch.long, device=command.device)
    if bool(((clip < 0) | (clip >= actual)).any()):
        raise RuntimeError(
            f"motion clip_id is outside the loaded action bank [0,{actual}): "
            f"min={int(clip.min())}, max={int(clip.max())}"
        )
    return F.one_hot(clip.to(dtype=torch.long), num_classes=actual).to(dtype=torch.float32)


def racket_target_normal_cmd(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Stage-1 face-command channel, 4-D per env: [DEMANDED face normal (3, world frame, question
    bank / StrikeSpec n), rho placeholder (1, zero-filled)]. rho is the S3 spin-lane scalar,
    reserved now so the layout matches the frozen contract-day 175 -> 179 decision and no ladder
    retrain is needed later. NOT in the frozen 175-D contract — only wired into the actor when
    racket.face_command_obs is enabled. Normal is zeros when the question bank is off (the buffer
    always exists)."""
    # The demanded normal rides the same planner message as target position/velocity and side.
    # Reading the actor view is load-bearing when A1 delay/dropout is enabled: the former live read
    # paired question N+1's face with question N's delayed/held position and velocity.
    return face_command_obs_vector(_cmd(env, command_name).actor_target_normal_cmd())


# --- HITTER Table-I exact actor terms (hitter_pure contract, 2026-07-07) ------------------- #
# World-frame vectors + the explicit base forward vector e_base,x, exactly as the paper's actor
# observation. NOT pre-rotated into the yaw-heading frame: the policy learns the rotation itself
# (this is what lets it correct its facing toward the table — the heading-frame formulation loses
# the yaw error entirely once the reference-orientation term is removed from the actor).
def base_forward_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Base forward unit vector e_base,x (world xy, 2). Deploy: IMU + yaw-align-at-engage."""
    return _cmd(env, command_name).base_forward_xy()


def base_target_delta_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target base position p̂_base,xy − p_base,xy (world frame, 2). Deploy: planner station −
    mocap base position (same world frame; no rotation)."""
    return _cmd(env, command_name).base_target_delta_xy_w()


def racket_target_rel_base(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target racket position relative to the base (world frame, 3; HITTER §V-B-1). Deploy:
    planner racket target − mocap base position. A1: actor-visible (delayed/jittered) view."""
    return _cmd(env, command_name).racket_target_rel_base_w()


# --- privileged (critic) observations: desired normal + actual racket state --------------- #
def racket_target_vel_w_live(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """TRUE live desired racket velocity (world). CRITIC/privileged term: the asymmetric critic
    keeps the undegraded target even when the actor's view is delayed/jittered (A1). Identical to
    :func:`racket_target_vel_w` when the A1 knobs are off."""
    return _cmd(env, command_name).racket_target_vel_w


def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward-consistent desired face normal for the privileged critic.

    In a face-command bank run this is the demanded +Y/A-frame normal; otherwise it is the
    historical clip/reference target. The width stays 3-D, so actor-tail warm starts do not need a
    critic resize, but the value function no longer misses the random command it is asked to value.
    """
    return face_tracking_pair(_cmd(env, command_name))[1]


def racket_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket position relative to base (FK). Privileged — not sensed on hardware."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_pos_w - cmd.base_pos_w)


def racket_lin_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_lin_vel_w


def racket_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward-consistent actual face normal (FK), world frame, privileged.

    This selects raw +Y in a face-command run and the historical per-clip signed face otherwise,
    matching :func:`racket_target_normal_w` without changing the critic observation width.
    """
    return face_tracking_pair(_cmd(env, command_name))[0]


def episode_time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in the episode (seconds). HITTER critic privileged input."""
    # IsaacLab 2.1 calls observation terms once during ObservationManager._prepare_terms (dimension
    # probe) BEFORE ManagerBasedRLEnv allocates episode_length_buf — fall back to zeros there.
    buf = getattr(env, "episode_length_buf", None)
    if buf is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    left = (env.max_episode_length - buf).float() * env.step_dt
    return left.unsqueeze(-1)
