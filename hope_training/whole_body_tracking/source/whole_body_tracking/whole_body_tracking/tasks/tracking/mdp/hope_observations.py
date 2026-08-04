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


def _target_component_or_zero(
    command: RacketTargetCommand,
    component: str,
    value: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed-question validity mask at the observation boundary.

    This final mask is intentionally downstream of every relative/heading transform.  An invalid
    absolute position is represented as zero inside the fixed-width transport, but ``0 - base`` or
    ``0 - racket`` is not an invalid relative target: it leaks robot state into a column that must
    be exactly zero.  Keeping the last mask here makes actor and critic observation producers safe
    even if an upstream command accessor already masked its world-frame value.
    """

    validity = getattr(command, "action_ball_target_component_valid", None)
    # Legacy/source-level command doubles predate the ActionBall validity contract and are complete
    # targets by definition.  Production RacketTargetCommand always owns the method.
    if validity is not None and not validity(component):
        return torch.zeros_like(value)
    return value


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
    command = _cmd(env, command_name)
    return _target_component_or_zero(
        command, "position", command.racket_target_pos_b()
    )


def racket_target_pos_rel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket pos relative to the CURRENT racket (FK), yaw frame. DEPLOY-HONEST (no world
    base position; see :meth:`RacketTargetCommand.racket_target_pos_b_rel`). Used by the deploy-parity
    actor contract (legacy task name: `real_sensor_only`). A1: reads the ACTOR-visible target view
    (delayed/jittered when target latency is on; the live tensor otherwise)."""
    command = _cmd(env, command_name)
    return _target_component_or_zero(
        command, "position", command.racket_target_pos_b_rel()
    )


def racket_target_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket velocity, world frame. ACTOR term — A1: reads the ACTOR-visible view
    (delayed/jittered when target latency is on; the live tensor otherwise, byte-identical).
    The critic uses :func:`racket_target_vel_w_live`."""
    command = _cmd(env, command_name)
    return _target_component_or_zero(
        command, "velocity", command.actor_racket_target_vel_w()
    )


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
    value = quat_rotate_inverse(
        yaw_quat(command.base_quat_w),
        command.actor_racket_target_vel_w(),
    )
    return _target_component_or_zero(command, "velocity", value)


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


def _action_ball_pack_base_pose_lin_vel_world(
    position_w: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    linear_velocity_w: torch.Tensor,
) -> torch.Tensor:
    """Pack the deploy-source-homogeneous 12-D ActionBall base/localizer row."""

    batch_shape = position_w.shape[:-1]
    expected = {
        "position_w": (*batch_shape, 3),
        "quaternion_wxyz": (*batch_shape, 4),
        "linear_velocity_w": (*batch_shape, 3),
    }
    values = {
        "position_w": position_w,
        "quaternion_wxyz": quaternion_wxyz,
        "linear_velocity_w": linear_velocity_w,
    }
    for name, value in values.items():
        if value.shape != expected[name]:
            raise ValueError(
                f"ActionBall {name} has shape {tuple(value.shape)}, expected {expected[name]}"
            )
        if value.device != position_w.device or value.dtype != position_w.dtype:
            raise ValueError("ActionBall base-pose tensors must share device and dtype")
    orientation_6d = _base_orientation_table_6d_from_quat(quaternion_wxyz)
    result = torch.cat((position_w, orientation_6d, linear_velocity_w), dim=-1)
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


_ACTION_BALL_225_SNAPSHOT_SOURCES = {
    "a211": (
        "racket_target_pos_w",
        "racket_target_vel_w",
        "racket_target_normal_w",
    ),
    "c211": (
        "_action_ball_ball_contact_target_w",
        "vb_vel_in_w",
        "vb_spin_in_w",
    ),
}


def _action_ball_225_identity(command: RacketTargetCommand) -> torch.Tensor:
    """Return the installed task identity used to guard one policy snapshot."""

    names = (
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "_action_ball_action_uid",
        "_action_ball_action_slot",
        "_action_ball_attempt_action",
    )
    values = []
    for name in names:
        value = getattr(command, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.dtype.is_floating_point
            or value.dtype == torch.bool
        ):
            raise RuntimeError(
                f"ActionBall 225 snapshot requires integer command field {name}[N]"
            )
        values.append(value)
    batch_size = int(values[0].shape[0])
    if any(tuple(value.shape) != (batch_size,) for value in values):
        raise RuntimeError("ActionBall 225 identity tensors have inconsistent shapes")
    if any(
        value.device != values[0].device or value.dtype != values[0].dtype
        for value in values[1:]
    ):
        raise RuntimeError("ActionBall 225 identity tensors must share dtype/device")
    return torch.stack(values, dim=-1)


def _action_ball_225_source_values(
    command: RacketTargetCommand, snapshot_kind: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    names = _ACTION_BALL_225_SNAPSHOT_SOURCES.get(snapshot_kind)
    if names is None:
        raise RuntimeError(f"unknown ActionBall 225 snapshot kind {snapshot_kind!r}")
    values = tuple(getattr(command, name, None) for name in names)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise RuntimeError(
            f"ActionBall {snapshot_kind} source packet is not installed"
        )
    return values


def _action_ball_225_assert(condition: torch.Tensor) -> None:
    """Torch-2.0-compatible device assertion (message argument was added later)."""

    torch._assert_async(condition)


def _action_ball_225_is_construction_probe(env: ManagerBasedRLEnv) -> bool:
    """Identify ObservationManager's pre-reset term-shape probe explicitly."""

    # ManagerBasedRLEnv assigns ``observation_manager`` only after its
    # constructor has finished probing every ObsTerm.  A real reset/step has a
    # live manager even when ``common_step_counter`` is still zero.
    return getattr(env, "observation_manager", None) is None


def _action_ball_225_install_token(
    command: RacketTargetCommand, batch_size: int
) -> tuple:
    """Return receipt-owned host identity without reading a device scalar."""

    receipts = getattr(command, "_action_ball_task_by_env", None)
    if not isinstance(receipts, list) or len(receipts) != batch_size:
        raise RuntimeError(
            "ActionBall 225 snapshot requires one host task receipt per environment"
        )
    rows = []
    for env_id, receipt in enumerate(receipts):
        if receipt is None:
            # RESET_WAIT deliberately owns no task receipt yet.  Preserve the
            # empty host row in the transaction token so a mid-observation
            # install still fails closed instead of mixing WAIT/TASK bytes.
            rows.append(None)
            continue
        row = (
            getattr(receipt, "env_id", None),
            getattr(receipt, "reset_generation", None),
            getattr(receipt, "swing_generation", None),
            getattr(receipt, "action_uid", None),
            getattr(receipt, "action_slot", None),
        )
        if any(type(value) is not int for value in row) or row[0] != env_id:
            raise RuntimeError("ActionBall 225 host task receipt identity is invalid")
        rows.append(row)
    return tuple(rows)


def _action_ball_211_task_valid(command: RacketTargetCommand) -> torch.Tensor:
    """Return the runtime-owned atomic TASK_ACTIVE validity mask."""

    value = getattr(command, "_action_ball_task_valid", None)
    base_position = getattr(
        getattr(getattr(command, "robot", None), "data", None),
        "root_pos_w",
        None,
    )
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype != torch.bool
        or not isinstance(base_position, torch.Tensor)
        or tuple(base_position.shape) != (int(value.shape[0]), 3)
        or value.device != base_position.device
    ):
        raise RuntimeError(
            "ActionBall 211 observation requires runtime-owned bool "
            "_action_ball_task_valid[N] on the robot device"
        )
    return value


def _action_ball_211_mask_task_value(
    command: RacketTargetCommand, value: torch.Tensor
) -> torch.Tensor:
    """Apply TASK_ACTIVE at the final observation boundary."""

    valid = _action_ball_211_task_valid(command)
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or int(value.shape[0]) != int(valid.shape[0])
        or value.device != valid.device
        or not value.dtype.is_floating_point
    ):
        raise RuntimeError("ActionBall 211 task observation payload is invalid")
    return torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))


def _action_ball_225_snapshot_heading(
    env: ManagerBasedRLEnv,
    command_name: str,
    snapshot_kind: str,
    component_index: int,
) -> torch.Tensor:
    """Run one ordered, generation-bound 9-D A211/C211 producer transaction.

    Position (component zero) starts the transaction and freezes all nine
    scalars once.  Velocity and face/spin only return slices from that frozen
    result.  Their host receipt identity and step must still match; an
    interleaved install fails instead of rebuilding and mixing generations.
    No CUDA scalar is extracted and no fixed-midpoint landing target is read.
    """

    if component_index not in (0, 1, 2):
        raise RuntimeError("ActionBall 225 component index must be 0, 1, or 2")
    token = getattr(env, "common_step_counter", None)
    if type(token) is not int or token < 0:
        raise RuntimeError(
            "ActionBall 225 observation requires a non-negative integer "
            "env.common_step_counter"
        )
    command = _cmd(env, command_name)
    ensure_runtime = getattr(command, "_ensure_action_ball_runtime_initialized", None)
    if not callable(ensure_runtime):
        raise RuntimeError("ActionBall 225 observation requires ActionBall runtime")
    ensure_runtime()
    task_valid = _action_ball_211_task_valid(command)
    if token == 0 and _action_ball_225_is_construction_probe(env):
        construction_identity = _action_ball_225_identity(command)
        construction_active = getattr(command, "_action_ball_attempt_active", None)
        if (
            isinstance(construction_active, torch.Tensor)
            and construction_active.dtype == torch.bool
            and tuple(construction_active.shape)
            == (int(construction_identity.shape[0]),)
            and bool((~construction_active).all())
        ):
            expected = torch.tensor(
                (0, -1, -1, -1, -1),
                dtype=construction_identity.dtype,
                device=construction_identity.device,
            ).expand_as(construction_identity)
            if not bool((construction_identity == expected).all()):
                raise RuntimeError(
                    "ActionBall 225 inactive construction identity is not pristine"
                )
            base_position = getattr(
                getattr(command.robot, "data", None), "root_pos_w", None
            )
            if not isinstance(base_position, torch.Tensor) or tuple(
                base_position.shape
            ) != (int(construction_identity.shape[0]), 3):
                raise RuntimeError(
                    "ActionBall 225 construction base pose is unavailable"
                )
            # This exact construction-phase, pristine-only zero is not a task
            # recipe and is never cached.  A real reset has a live
            # observation_manager; an installed task at token zero is active.
            # Both must take the authoritative path below.
            return torch.zeros(
                int(construction_identity.shape[0]),
                9,
                dtype=base_position.dtype,
                device=base_position.device,
            )
    cache = getattr(command, "_action_ball_225_observation_cache", None)
    if cache is None:
        cache = {}
        setattr(command, "_action_ball_225_observation_cache", cache)
    if type(cache) is not dict:
        raise RuntimeError("ActionBall 225 observation cache has invalid type")
    cached = cache.get(snapshot_kind)
    if component_index != 0:
        if not isinstance(cached, dict):
            raise RuntimeError("ActionBall 225 producer transaction was not started")
        if cached.get("token") != token:
            raise RuntimeError("ActionBall 225 policy tick changed within transaction")
        if cached.get("next_component") != component_index:
            raise RuntimeError("ActionBall 225 producer transaction order is invalid")
        identity = cached.get("install_token")
        if not isinstance(identity, tuple):
            raise RuntimeError("ActionBall 225 cached install identity is missing")
        if _action_ball_225_install_token(command, len(identity)) != identity:
            raise RuntimeError(
                "ActionBall 225 install identity changed within producer transaction"
            )
        result = cached.get("result")
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("ActionBall 225 cached result is missing")
        cached_valid = cached.get("task_valid")
        if not isinstance(cached_valid, torch.Tensor):
            raise RuntimeError("ActionBall 211 cached task validity is missing")
        _action_ball_225_assert((task_valid == cached_valid).all())
        cached["next_component"] = component_index + 1
        return result

    if snapshot_kind == "a211":
        component_valid = getattr(
            command, "action_ball_target_component_valid", None
        )
        if not callable(component_valid) or not all(
            component_valid(component)
            for component in ("position", "velocity", "face")
        ):
            raise RuntimeError(
                "ActionBall A211 requires a complete valid task-derived p/v/face tuple"
            )

    identity_before = _action_ball_225_identity(command).clone()
    install_token_before = _action_ball_225_install_token(
        command, int(identity_before.shape[0])
    )
    active_before = getattr(command, "_action_ball_attempt_active", None)
    if (
        not isinstance(active_before, torch.Tensor)
        or active_before.dtype != torch.bool
        or tuple(active_before.shape) != (int(identity_before.shape[0]),)
        or active_before.device != identity_before.device
    ):
        raise RuntimeError("ActionBall 225 snapshot requires bool attempt-active[N]")
    active_before = active_before.clone()
    task_valid_before = task_valid.clone()

    robot = getattr(command, "robot", None)
    robot_data = getattr(robot, "data", None)
    base_position = getattr(robot_data, "root_pos_w", None)
    base_quaternion = getattr(robot_data, "root_quat_w", None)
    if not isinstance(base_position, torch.Tensor) or not isinstance(
        base_quaternion, torch.Tensor
    ):
        raise RuntimeError("ActionBall 225 snapshot requires current robot base pose")
    base_position = base_position.clone()
    base_quaternion = base_quaternion.clone()
    source_values = tuple(
        value.clone()
        for value in _action_ball_225_source_values(command, snapshot_kind)
    )
    identity_after = _action_ball_225_identity(command).clone()
    install_token_after = _action_ball_225_install_token(
        command, int(identity_after.shape[0])
    )
    active_after = getattr(command, "_action_ball_attempt_active").clone()
    task_valid_after = _action_ball_211_task_valid(command).clone()
    token_after = getattr(env, "common_step_counter", None)

    batch_size = int(identity_before.shape[0])
    float_tensors = (base_position, base_quaternion, *source_values)
    expected_shapes = (
        (batch_size, 3),
        (batch_size, 4),
        (batch_size, 3),
        (batch_size, 3),
        (batch_size, 3),
    )
    if token_after != token:
        raise RuntimeError("ActionBall 225 policy tick changed during snapshot")
    if install_token_after != install_token_before:
        raise RuntimeError("ActionBall 225 install changed during snapshot")
    for value, shape in zip(float_tensors, expected_shapes):
        if (
            tuple(value.shape) != shape
            or not value.dtype.is_floating_point
            or value.device != base_position.device
            or value.dtype != base_position.dtype
        ):
            raise RuntimeError(
                "ActionBall 225 snapshot payload has wrong shape/dtype/device"
            )
    if identity_before.device != base_position.device:
        raise RuntimeError("ActionBall 225 identity and payload devices differ")
    _action_ball_225_assert(
        (identity_before == identity_after).all()
        & (active_before == active_after).all()
        & (task_valid_before == task_valid_after).all()
    )
    # RESET_WAIT keeps the immutable attempt installed while concealing its
    # fields.  TASK_ACTIVE must imply an installed attempt, but WAIT is allowed
    # to have attempt_active=True and task_valid=False.
    _action_ball_225_assert(((~task_valid_before) | active_before).all())
    _action_ball_225_assert(
        (
            (~task_valid_before)
            | (
                (identity_before[:, 0] >= 0)
                & (identity_before[:, 1] >= 0)
                & (identity_before[:, 2] >= 0)
                & (identity_before[:, 3] >= 0)
                & (identity_before[:, 4] == identity_before[:, 3])
            )
        ).all()
    )
    _action_ball_225_assert(torch.isfinite(base_position).all())
    _action_ball_225_assert(torch.isfinite(base_quaternion).all())
    for value in source_values:
        _action_ball_225_assert(
            ((~task_valid_before) | torch.isfinite(value).all(dim=-1)).all()
        )
    quaternion_norm = torch.linalg.vector_norm(base_quaternion, dim=-1)
    _action_ball_225_assert(
        (torch.abs(quaternion_norm - 1.0) <= 1.0e-5).all()
    )
    if snapshot_kind == "a211":
        face_norm = torch.linalg.vector_norm(source_values[2], dim=-1)
        _action_ball_225_assert(
            ((~task_valid_before) | (torch.abs(face_norm - 1.0) <= 1.0e-5)).all()
        )

    heading = yaw_quat(base_quaternion)
    result = torch.cat(
        (
            quat_rotate_inverse(heading, source_values[0] - base_position),
            quat_rotate_inverse(heading, source_values[1]),
            quat_rotate_inverse(heading, source_values[2]),
        ),
        dim=-1,
    )
    if tuple(result.shape) != (batch_size, 9):
        raise RuntimeError("ActionBall 225 task snapshot did not produce [N,9]")
    _action_ball_225_assert(
        ((~task_valid_before) | torch.isfinite(result).all(dim=-1)).all()
    )
    result = torch.where(
        task_valid_before.unsqueeze(-1), result, torch.zeros_like(result)
    )
    cache[snapshot_kind] = {
        "token": token,
        "install_token": install_token_before,
        "next_component": 1,
        "identity": identity_before,
        "active": active_before,
        "task_valid": task_valid_before,
        "base_position": base_position,
        "base_quaternion": base_quaternion,
        "source_values": source_values,
        "result": result,
    }
    return result


def action_ball_a211_task_desired_contact_position_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    return _action_ball_225_snapshot_heading(env, command_name, "a211", 0)[:, 0:3]


def action_ball_a211_task_desired_contact_velocity_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    return _action_ball_225_snapshot_heading(env, command_name, "a211", 1)[:, 3:6]


def action_ball_a211_task_desired_contact_face_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    return _action_ball_225_snapshot_heading(env, command_name, "a211", 2)[:, 6:9]


def action_ball_c211_incoming_ball_contact_position_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    return _action_ball_225_snapshot_heading(env, command_name, "c211", 0)[:, 0:3]


def action_ball_c211_incoming_ball_contact_velocity_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    return _action_ball_225_snapshot_heading(env, command_name, "c211", 1)[:, 3:6]


def action_ball_c211_incoming_ball_contact_spin_heading(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    return _action_ball_225_snapshot_heading(env, command_name, "c211", 2)[:, 6:9]


def action_ball_211_base_target_position_world_xy(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command = _cmd(env, command_name)
    return _action_ball_211_mask_task_value(
        command, stage1_base_target_position_world_xy(env, command_name)
    )


def action_ball_211_time_to_contact(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command = _cmd(env, command_name)
    return _action_ball_211_mask_task_value(
        command, time_to_strike(env, command_name)
    )


def action_ball_211_time_to_teacher_start(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command = _cmd(env, command_name)
    return _action_ball_211_mask_task_value(
        command, time_to_teacher_start_s(env, command_name)
    )


def action_ball_task_valid(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    command = _cmd(env, command_name)
    valid = _action_ball_211_task_valid(command)
    dtype = command.robot.data.root_pos_w.dtype
    return valid.to(dtype=dtype).unsqueeze(-1)


def _stage1_motion_and_command(
    env: ManagerBasedRLEnv, command_name: str
) -> tuple[RacketTargetCommand, object]:
    command = _cmd(env, command_name)
    motion = command._motion()
    if bool(
        getattr(motion, "action_ball_diagnostic_split_ready_teacher", False)
    ):
        capture = getattr(
            motion, "_capture_action_ball_safe_ready_reference", None
        )
        if callable(capture):
            capture()
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


def action_ball_actual_base_pose_lin_vel_world(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Actual base position/orientation/linear velocity in canonical HOPE world.

    Angular velocity is deliberately not packed here: A211/C211 obtain the one
    actor-visible copy from the pelvis/body-frame gyro term below.
    """

    command, _motion = _stage1_motion_and_command(env, command_name)
    robot = command.robot
    return _action_ball_pack_base_pose_lin_vel_world(
        _stage1_env_position_to_hope_world(
            command, env.scene.env_origins, robot.data.root_pos_w
        ),
        robot.data.root_quat_w,
        robot.data.root_lin_vel_w,
    )


def action_ball_base_ang_vel_body(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Pelvis/root angular velocity in the robot body frame, i.e. the IMU gyro ABI."""

    command, _motion = _stage1_motion_and_command(env, command_name)
    return _stage1_exact_matrix(
        command.robot.data.root_ang_vel_b,
        num_envs=env.num_envs,
        width=3,
        name="action_ball_base_ang_vel_body",
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
    command = _cmd(env, command_name)
    value = face_command_obs_vector(command.actor_target_normal_cmd())
    return _target_component_or_zero(command, "face", value)


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
    value = torch.cat((normal_heading, raw[:, 3:4]), dim=-1)
    return _target_component_or_zero(command, "face", value)


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
    command = _cmd(env, command_name)
    # ``racket_target_rel_base_w`` historically consumed an already-masked absolute target, so a
    # 000 recipe produced ``0 - base_pos``.  The observation boundary owns the final fixed-width
    # contract: transform first, then erase the entire relative column when position is undefined.
    value = command.racket_target_rel_base_w()
    return _target_component_or_zero(command, "position", value)


# --- privileged (critic) observations: desired normal + actual racket state --------------- #
def racket_target_vel_w_live(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """TRUE live desired racket velocity (world). CRITIC/privileged term: the asymmetric critic
    keeps the undegraded target even when the actor's view is delayed/jittered (A1). Identical to
    :func:`racket_target_vel_w` when the A1 knobs are off."""
    command = _cmd(env, command_name)
    value = command.racket_target_vel_w
    return _target_component_or_zero(command, "velocity", value)


def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward-consistent desired face normal for the privileged critic.

    In a face-command bank run this is the demanded +Y/A-frame normal; otherwise it is the
    historical clip/reference target. The width stays 3-D, so actor-tail warm starts do not need a
    critic resize, but the value function no longer misses the random command it is asked to value.
    """
    command = _cmd(env, command_name)
    value = face_tracking_pair(command)[1]
    return _target_component_or_zero(command, "face", value)


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
