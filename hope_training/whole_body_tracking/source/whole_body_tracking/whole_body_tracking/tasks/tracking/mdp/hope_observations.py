"""HOPE racket-target observation terms.

These wrap :class:`RacketTargetCommand`. The actor (policy) group should use only the *desired*
quantities the planner provides at deploy time (HITTER actor observation, Table I):

* :func:`racket_target_pos_b`  — desired racket position relative to base (3)
* :func:`racket_target_vel_w`  — desired racket velocity in world frame (3)
* :func:`racket_target_vel_heading` — desired racket velocity in the base yaw-heading frame (3)
* :func:`time_to_strike`       — time remaining until strike (1)
* :func:`time_to_teacher_start_s` — time until the frozen teacher leaves its ready frame (1)
* :func:`base_target_pos_b`    — desired base XY position relative to base (2)
* :func:`base_position_table`  — base root position relative to table-surface center (3)
* :func:`base_orientation_table_6d` — full base orientation in the table frame (6)
* :func:`base_lin_vel_heading` — root rigid-body COM linear velocity in the yaw-heading frame (3)
* :func:`station_anchor_err_b` — world station anchor minus current base XY, base frame (2;
  R10c station_obs flag, appended after the face channel = 179 -> 181)

In the historical HITTER actor layouts, the desired racket *normal* and actual racket state are
critic/reward-only.  The fresh Stage-1 paddle-world-v2 contract below deliberately differs: its
actual official-site tuple is deterministic FK from deploy-available joint/base state (not an
external racket sensor), and it pairs that tuple with teacher-now and teacher/contact previews.
That contract has neither :func:`swing_type` nor an action one-hot; physical future contact state
disambiguates the selected motion without a categorical identity shortcut.
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


def racket_target_vel_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Actor-visible desired racket velocity in the base yaw-heading frame.

    ActionBall expresses the racket-position residual in this same frame.  The
    transform therefore keeps position and its demanded velocity covariant
    under a global/table yaw change.  It deliberately consumes the delayed
    actor tuple rather than the live critic/reward target.
    """

    command = _cmd(env, command_name)
    return quat_rotate_inverse(
        yaw_quat(command.base_quat_w),
        command.actor_racket_target_vel_w(),
    )


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """TRUE live time remaining until strike (s), used by the privileged critic/default actor.

    The explicit atomic planner-tuple training modes wire the policy term to
    :func:`actor_time_to_strike` in ``train.py`` while leaving this live critic source untouched.
    """
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def actor_time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actor-visible planner TTS: live, source-timestamp compensated, or stale negative control."""
    return _cmd(env, command_name).actor_time_to_strike().unsqueeze(-1)


def time_to_teacher_start_s(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Exact live countdown until ActionBall teacher playback starts."""

    return _cmd(
        env, command_name
    ).actor_time_to_teacher_start_s().unsqueeze(-1)


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


# --- Stage-1 natural-clip paddle-world v2 ------------------------------------------------- #
def _stage1_exact_matrix(
    value: torch.Tensor, *, num_envs: int, width: int, name: str
) -> torch.Tensor:
    """Fail closed on a producer that would silently mutate the fixed actor ABI."""

    expected = (int(num_envs), int(width))
    if value.shape != expected:
        raise ValueError(
            f"Stage-1 {name} has shape {tuple(value.shape)}, expected {expected}"
        )
    torch._assert_async(torch.isfinite(value).all())
    return value


def _stage1_pack_base_state_world(
    position_w: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
) -> torch.Tensor:
    """Pack one causal ``position + orientation-6D + linear/angular velocity`` row.

    The helper is shared by achieved and teacher producers so the two adjacent 15-D terms cannot
    drift to different component orders.  Positions must already use canonical HOPE-world origin;
    velocities use the same world axes.
    """

    batch_shape = position_w.shape[:-1]
    expected = {
        "position_w": (*batch_shape, 3),
        "quaternion_wxyz": (*batch_shape, 4),
        "linear_velocity_w": (*batch_shape, 3),
        "angular_velocity_w": (*batch_shape, 3),
    }
    values = {
        "position_w": position_w,
        "quaternion_wxyz": quaternion_wxyz,
        "linear_velocity_w": linear_velocity_w,
        "angular_velocity_w": angular_velocity_w,
    }
    for name, value in values.items():
        if value.shape != expected[name]:
            raise ValueError(
                f"Stage-1 {name} has shape {tuple(value.shape)}, expected {expected[name]}"
            )
        if value.device != position_w.device or value.dtype != position_w.dtype:
            raise ValueError("Stage-1 base-state tensors must share device and dtype")
    orientation_6d = _base_orientation_table_6d_from_quat(quaternion_wxyz)
    result = torch.cat(
        (position_w, orientation_6d, linear_velocity_w, angular_velocity_w),
        dim=-1,
    )
    torch._assert_async(torch.isfinite(result).all())
    return result


def _stage1_env_position_to_hope_world(
    command: RacketTargetCommand,
    env_origins_w: torch.Tensor,
    position_w: torch.Tensor,
) -> torch.Tensor:
    """Convert replicated tracking-env positions into the canonical HOPE venue frame.

    The tracking scene is expressed in ``a3_robot_origin_ground_z0``: the virtual table's near
    edge is at ``cfg.vb_table_near_x``, its centre line is ``y=0``, and its surface is at
    ``cfg.vb_table_surface_z``.  Canonical HOPE world instead starts at the near-left table-surface
    corner, so the table centre is ``[1.37, -TABLE_WIDTH/2, 0]``.  Merely subtracting Isaac's
    per-environment replication origin would leave the actor in the robot/floor frame and would
    violate the versioned observation contract.
    """

    if position_w.shape[-1:] not in ((2,), (3,)):
        raise ValueError(
            "Stage-1 HOPE-world position must end in two or three components, "
            f"got shape {tuple(position_w.shape)}"
        )
    width = position_w.shape[-1]
    expected_origin_shape = (*position_w.shape[:-1], 3)
    if env_origins_w.shape != expected_origin_shape:
        raise ValueError(
            "Stage-1 env origins do not match the position batch: "
            f"got {tuple(env_origins_w.shape)}, expected {expected_origin_shape}"
        )
    if not hasattr(command, "_vb_half_w"):
        raise RuntimeError(
            "Stage-1 HOPE-world bridge requires the command's resolved table half-width"
        )
    translation_values = (
        float(command.cfg.vb_table_near_x),
        float(command._vb_half_w),
        float(command.cfg.vb_table_surface_z),
    )[:width]
    translation = position_w.new_tensor(translation_values)
    result = position_w - env_origins_w[..., :width] - translation
    torch._assert_async(torch.isfinite(result).all())
    return result


def _stage1_reference_vector_in_aligned_world(
    vector_w: torch.Tensor,
    raw_reference_quat_wxyz: torch.Tensor,
    aligned_reference_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Apply the MotionCommand's yaw alignment to a reference world vector.

    ``MotionCommand.body_*_vel_w`` retains the clip's raw world axes while
    ``body_*_relative_w`` contains the teacher pose after per-environment alignment.  Deriving the
    relative rotation from those two quaternions prevents teacher base pose and teacher base twist
    from silently using different frames.
    """

    if vector_w.shape[-1:] != (3,):
        raise ValueError("Stage-1 aligned reference vector must end in three components")
    if raw_reference_quat_wxyz.shape != (*vector_w.shape[:-1], 4):
        raise ValueError("Stage-1 raw reference quaternion shape does not match vector batch")
    if aligned_reference_quat_wxyz.shape != (*vector_w.shape[:-1], 4):
        raise ValueError("Stage-1 aligned reference quaternion shape does not match vector batch")
    raw_rotation = matrix_from_quat(raw_reference_quat_wxyz)
    aligned_rotation = matrix_from_quat(aligned_reference_quat_wxyz)
    alignment = torch.matmul(aligned_rotation, raw_rotation.transpose(-1, -2))
    result = torch.matmul(alignment, vector_w.unsqueeze(-1)).squeeze(-1)
    torch._assert_async(torch.isfinite(result).all())
    return result


def _stage1_pack_racket_state_heading(
    base_position_w: torch.Tensor,
    base_quaternion_wxyz: torch.Tensor,
    site_position_w: torch.Tensor,
    site_linear_velocity_w: torch.Tensor,
    site_signed_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Pack ``site position + absolute linear velocity + signed normal`` in base heading.

    Position uses the *current actual base* as origin.  Velocity is the absolute world point
    velocity merely expressed in heading axes; it intentionally does not subtract base velocity,
    because the ball-contact task depends on absolute paddle speed.  The same transform is used by
    achieved-now, teacher-now, teacher-at-hit and desired-at-contact.
    """

    batch_shape = base_position_w.shape[:-1]
    if base_position_w.shape != (*batch_shape, 3):
        raise ValueError("Stage-1 base position must end in three components")
    if base_quaternion_wxyz.shape != (*batch_shape, 4):
        raise ValueError("Stage-1 base quaternion must match the base-position batch")
    for name, value in (
        ("site_position_w", site_position_w),
        ("site_linear_velocity_w", site_linear_velocity_w),
        ("site_signed_normal_w", site_signed_normal_w),
    ):
        if value.shape != (*batch_shape, 3):
            raise ValueError(
                f"Stage-1 {name} has shape {tuple(value.shape)}, expected {(*batch_shape, 3)}"
            )
        if value.device != base_position_w.device or value.dtype != base_position_w.dtype:
            raise ValueError("Stage-1 racket-state tensors must share device and dtype")
    heading = yaw_quat(base_quaternion_wxyz)
    position_heading = quat_rotate_inverse(
        heading, site_position_w - base_position_w
    )
    velocity_heading = quat_rotate_inverse(heading, site_linear_velocity_w)
    normal_heading = quat_rotate_inverse(heading, site_signed_normal_w)
    normal_norm = torch.linalg.vector_norm(normal_heading, dim=-1, keepdim=True)
    torch._assert_async(
        (
            torch.isfinite(normal_norm[..., 0])
            & (normal_norm[..., 0] > 1.0e-12)
        ).all()
    )
    normal_heading = normal_heading / normal_norm.clamp_min(1.0e-12)
    result = torch.cat(
        (position_heading, velocity_heading, normal_heading), dim=-1
    )
    torch._assert_async(torch.isfinite(result).all())
    return result


def _stage1_motion_and_command(
    env: ManagerBasedRLEnv, command_name: str
) -> tuple[RacketTargetCommand, object]:
    command = _cmd(env, command_name)
    motion = command._motion()
    body_names = tuple(str(name) for name in motion.cfg.body_names)
    robot_body_names = tuple(str(name) for name in command.robot.body_names)
    if not body_names or not robot_body_names or body_names[0] != robot_body_names[0]:
        raise RuntimeError(
            "Stage-1 paddle-world observation requires the first tracked body to be the "
            "articulation root/base"
        )
    return command, motion


def stage1_base_state_world(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Current A3 base state in canonical HOPE world (near-table-corner origin)."""

    command, _motion = _stage1_motion_and_command(env, command_name)
    origins = env.scene.env_origins
    robot = command.robot
    return _stage1_pack_base_state_world(
        _stage1_env_position_to_hope_world(command, origins, robot.data.root_pos_w),
        robot.data.root_quat_w,
        robot.data.root_lin_vel_w,
        robot.data.root_ang_vel_w,
    )


def stage1_teacher_base_state_now_world(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Aligned current teacher root state in the exact same 15-D world layout."""

    command, motion = _stage1_motion_and_command(env, command_name)
    origins = env.scene.env_origins
    aligned_quat = motion.body_quat_relative_w[:, 0]
    raw_quat = motion.body_quat_w[:, 0]
    aligned_linear_velocity = _stage1_reference_vector_in_aligned_world(
        motion.body_lin_vel_w[:, 0], raw_quat, aligned_quat
    )
    aligned_angular_velocity = _stage1_reference_vector_in_aligned_world(
        motion.body_ang_vel_w[:, 0], raw_quat, aligned_quat
    )
    return _stage1_pack_base_state_world(
        _stage1_env_position_to_hope_world(
            command, origins, motion.body_pos_relative_w[:, 0]
        ),
        aligned_quat,
        aligned_linear_velocity,
        aligned_angular_velocity,
    )


def stage1_joint_pos_rel(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command, _motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        command.robot.data.joint_pos - command.robot.data.default_joint_pos,
        num_envs=env.num_envs,
        width=31,
        name="joint_pos",
    )


def stage1_teacher_joint_pos_rel(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command, motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        motion.joint_pos - command.robot.data.default_joint_pos,
        num_envs=env.num_envs,
        width=31,
        name="teacher_joint_pos_rel",
    )


def stage1_joint_vel(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command, _motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        command.robot.data.joint_vel,
        num_envs=env.num_envs,
        width=31,
        name="joint_vel",
    )


def stage1_teacher_joint_vel(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    _command, motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        motion.joint_vel,
        num_envs=env.num_envs,
        width=31,
        name="teacher_joint_vel",
    )


def stage1_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous normalized actor output, matching Isaac Lab's ``last_action`` term."""

    return _stage1_exact_matrix(
        env.action_manager.action,
        num_envs=env.num_envs,
        width=31,
        name="actions",
    )


def stage1_racket_site_achieved_now_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command, _motion = _stage1_motion_and_command(env, command_name)
    robot = command.robot
    return _stage1_pack_racket_state_heading(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command.racket_pos_w,
        command.racket_lin_vel_w,
        command.racket_normal_w,
    )


def stage1_racket_site_teacher_now_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
        stage1_aligned_clip_site_target_now,
    )

    command, _motion = _stage1_motion_and_command(env, command_name)
    robot = command.robot
    position, normal, velocity = stage1_aligned_clip_site_target_now(command)
    return _stage1_pack_racket_state_heading(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        position,
        velocity,
        normal,
    )


def stage1_racket_site_teacher_at_reference_hit_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
        stage1_aligned_clip_site_target_at_reference_hit,
    )

    command, _motion = _stage1_motion_and_command(env, command_name)
    robot = command.robot
    position, normal, velocity = (
        stage1_aligned_clip_site_target_at_reference_hit(command)
    )
    return _stage1_pack_racket_state_heading(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        position,
        velocity,
        normal,
    )


def stage1_racket_contact_desired_at_t_hit_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Stage-1 contact demand: an explicit copy of, not an alias for, teacher-at-hit.

    A later ball-conditioned contract may replace this producer while retaining the physical tuple
    meaning.  This Stage-1 contract has no ball target and therefore fails closed to the exact
    clip-derived contact state instead of inventing zeros or reading legacy planner buffers.
    """

    from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
        stage1_aligned_clip_site_target_at_reference_hit,
    )

    command, _motion = _stage1_motion_and_command(env, command_name)
    robot = command.robot
    position, normal, velocity = (
        stage1_aligned_clip_site_target_at_reference_hit(command)
    )
    return _stage1_pack_racket_state_heading(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        position,
        velocity,
        normal,
    ).clone()


def stage1_base_target_position_world_xy(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command, _motion = _stage1_motion_and_command(env, command_name)
    origins = env.scene.env_origins
    return _stage1_env_position_to_hope_world(
        command, origins, command.base_target_pos_w
    )


def stage1_time_to_contact_s(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Signed Stage-1 contact clock derived from the selected clip's hit landmark."""

    command, _motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        command.time_to_strike.unsqueeze(-1),
        num_envs=env.num_envs,
        width=1,
        name="time_to_contact_s",
    )


def stage1_time_to_teacher_start_s(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Countdown until the clip teacher leaves its current ready hold.

    Stage 1 has no ActionBall task receipt.  Its wait clock is the MotionCommand-owned number of
    future frozen-reference control steps, converted to seconds.  The active VendorV2 recipe has
    zero hold, but this explicit producer keeps the column meaningful if a later same-ABI stage
    enables a ready wait without pretending an ActionBall task was bound.
    """

    _command, motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        motion.teacher_start_wait_remaining_s.unsqueeze(-1),
        num_envs=env.num_envs,
        width=1,
        name="time_to_teacher_start_s",
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


def racket_target_normal_cmd_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """ActionBall face command in the base yaw-heading frame plus unchanged rho.

    The fixed action's raw-A face normal is selected before this observation is
    built.  Rotating it here changes only the actor representation; solver,
    reward, contact physics and the planner wire remain in the canonical
    table/world frame.  The reserved scalar ``rho`` is frame invariant.
    """

    command = _cmd(env, command_name)
    raw = face_command_obs_vector(command.actor_target_normal_cmd())
    normal_heading = quat_rotate_inverse(
        yaw_quat(command.base_quat_w),
        raw[:, :3],
    )
    return torch.cat((normal_heading, raw[:, 3:4]), dim=-1)


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
