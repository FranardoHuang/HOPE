"""HOPE goal-tracking reward terms (HITTER r_goal).

These implement the racket/base target tracking rewards on top of the BeyondMimic imitation
reward (``r_imitation``, the ``motion_*`` terms already in ``rewards.py``) and the regularization
reward (``r_regularization``, ``action_rate_l2`` / ``joint_torques_l2`` / contact penalties).

Activation timing follows HITTER: the base-position reward is active **before** the strike; the
racket position/velocity/normal rewards are active only in a **short window around** the strike.
Because a ``RewardTermCfg`` weight is constant, the time gating is applied *inside* each term by
multiplying the exponential kernel by the command's ``pre_strike`` / ``strike_window`` mask.

The exponential kernel form (``exp(-error/std**2)``) mirrors the BeyondMimic motion-tracking
rewards. HITTER does not publish reward weights or kernel forms, so the weights in the env config
are HOPE choices to be tuned, not paper-sourced values.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand, face_tracking_pair
from whole_body_tracking.tasks.tracking.mdp import racket_contact_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def _dbg_log(cmd: RacketTargetCommand, name: str, raw: torch.Tensor, mask: torch.Tensor) -> None:
    """Log the current pre-mask kernel and its actual post-mask value.

    No-op unless ``cmd.cfg.debug_reward_logging`` is set.  The old implementation updated both
    tensors only where ``mask`` was true, making ``dbg_*_gated`` identically equal to
    ``dbg_*_raw`` and unable to reveal how often the gate removed income.  These diagnostics now
    mirror the reward expression on every step: ``gated = raw * mask``.
    """
    if not cmd.cfg.debug_reward_logging:
        return
    cmd.metrics[f"dbg_{name}_raw"] = raw
    cmd.metrics[f"dbg_{name}_gated"] = raw * mask.float()


def action_ball_task_valid_mask(cmd: RacketTargetCommand) -> torch.Tensor:
    """Return the public task/reward eligibility bit for each environment.

    Historical environments do not install RESET_WAIT and therefore retain an
    all-true mask.  A211/C211 install the bool tensor on the command owner; a
    malformed public owner is a contract error rather than a permissive
    fallback.  The private wait countdown is deliberately not read here.
    """

    task_valid = getattr(cmd, "_action_ball_task_valid", None)
    if task_valid is None:
        template = getattr(cmd, "pre_strike", None)
        if template is None:
            template = getattr(cmd, "strike_window", None)
        if template is None:
            raise RuntimeError(
                "reward eligibility requires pre_strike or strike_window"
            )
        return torch.ones_like(template, dtype=torch.bool)
    template = getattr(cmd, "pre_strike", None)
    if template is None:
        raise RuntimeError(
            "ActionBall task_valid requires the command pre_strike batch"
        )
    if (
        not isinstance(task_valid, torch.Tensor)
        or task_valid.dtype != torch.bool
        or task_valid.shape != template.shape
        or task_valid.device != template.device
    ):
        raise RuntimeError(
            "ActionBall task_valid must be a bool tensor matching the command batch"
        )
    return task_valid


def _stage1_split_ready_wait_mask(cmd: RacketTargetCommand) -> torch.Tensor:
    """Return the one public mask that selects the hidden-WAIT teacher."""

    motion = cmd._motion()
    if not bool(
        getattr(motion, "action_ball_diagnostic_split_ready_teacher", False)
    ):
        return torch.zeros_like(motion.in_hold, dtype=torch.bool)
    # Rewards and paddle observations may be the first reset-time consumers;
    # do not depend on a preceding Motion command update or observation term.
    capture = getattr(
        motion, "_capture_action_ball_safe_ready_reference", None
    )
    if callable(capture):
        capture()
    task_valid = action_ball_task_valid_mask(cmd)
    bound = getattr(motion, "_action_ball_public_task_valid", None)
    owned = getattr(cmd, "_action_ball_task_valid", None)
    if bound is None or bound is not owned:
        raise RuntimeError(
            "split-ready teacher requires Motion and Racket to share one "
            "task_valid tensor"
        )
    return ~task_valid


def _window_pos(cmd: RacketTargetCommand) -> torch.Tensor:
    """1c TIGHT window for the position channel (== strike_window unless racket.strike_window_pos_s)."""
    win = getattr(cmd, "strike_window_pos", None)
    return (cmd.strike_window if win is None else win) & action_ball_task_valid_mask(cmd)


def _window_wide(cmd: RacketTargetCommand) -> torch.Tensor:
    """1c WIDE window for the normal/velocity channels (== strike_window unless racket.strike_window_wide_s)."""
    win = getattr(cmd, "strike_window_wide", None)
    return (cmd.strike_window if win is None else win) & action_ball_task_valid_mask(cmd)


def _stage1_quat_normalize(quat: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize a WXYZ quaternion without moving a device scalar to the host."""

    if quat.ndim < 1 or quat.shape[-1] != 4:
        raise ValueError(f"{name} must end in four WXYZ components, got {tuple(quat.shape)}")
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    valid = torch.isfinite(quat).all(dim=-1) & torch.isfinite(norm[..., 0]) & (norm[..., 0] > 1.0e-12)
    torch._assert_async(valid.all())
    return quat / norm.clamp_min(1.0e-12)


def _stage1_quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """WXYZ quaternion product, implemented locally so the pure helper has no Isaac dependency."""

    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _stage1_quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate ``vector`` by unit WXYZ ``quat`` using only Torch tensor operations."""

    xyz = quat[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _stage1_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return the yaw-only WXYZ quaternion corresponding to ``quat``."""

    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def stage1_clip_site_target_from_aligned_body_pose(
    previous_body_pos_w: torch.Tensor,
    previous_body_quat_wxyz: torch.Tensor,
    current_body_pos_w: torch.Tensor,
    current_body_quat_wxyz: torch.Tensor,
    next_body_pos_w: torch.Tensor,
    next_body_quat_wxyz: torch.Tensor,
    *,
    mount_offset_body: torch.Tensor,
    mount_quat_wxyz: torch.Tensor,
    normal_axis: int,
    normal_sign: torch.Tensor | float,
    central_difference_span_s: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the Stage-1 teacher's official racket-site state from aligned clip poses.

    All body poses are already in the *same aligned world frame*.  The helper applies the fixed
    body->official-site transform at each of the previous/current/next samples, obtains the signed
    physical face normal from the current site orientation, and differentiates the **site point**
    positions.  It intentionally never reads ``body_lin_vel_w``: that channel is a COM-point
    velocity in the pinned Isaac runtime and is not the derivative of the controlled racket site.

    ``central_difference_span_s`` is the elapsed reference time from the previous to the next
    sample (normally two 50-Hz frames, shortened only at a clip boundary).  This function contains
    no ball, inverse solver, LM, environment mutation, or sampling and is suitable for CPU tests and
    batched GPU reward evaluation.
    """

    positions = (previous_body_pos_w, current_body_pos_w, next_body_pos_w)
    quaternions = (previous_body_quat_wxyz, current_body_quat_wxyz, next_body_quat_wxyz)
    batch_shape = current_body_pos_w.shape[:-1]
    for index, value in enumerate(positions):
        if value.shape != (*batch_shape, 3):
            raise ValueError(
                f"Stage-1 body position {index} has shape {tuple(value.shape)}, "
                f"expected {(*batch_shape, 3)}"
            )
        if value.device != current_body_pos_w.device or value.dtype != current_body_pos_w.dtype:
            raise ValueError("Stage-1 body positions must share device and dtype")
    for index, value in enumerate(quaternions):
        if value.shape != (*batch_shape, 4):
            raise ValueError(
                f"Stage-1 body quaternion {index} has shape {tuple(value.shape)}, "
                f"expected {(*batch_shape, 4)}"
            )
        if value.device != current_body_pos_w.device or value.dtype != current_body_pos_w.dtype:
            raise ValueError("Stage-1 body quaternions must share the position device and dtype")
    if type(normal_axis) is not int or normal_axis not in (0, 1, 2):
        raise ValueError(f"normal_axis must be a plain integer in [0, 2], got {normal_axis!r}")

    offset = torch.as_tensor(
        mount_offset_body, device=current_body_pos_w.device, dtype=current_body_pos_w.dtype
    )
    mount_quat = torch.as_tensor(
        mount_quat_wxyz, device=current_body_pos_w.device, dtype=current_body_pos_w.dtype
    )
    try:
        offset = torch.broadcast_to(offset, (*batch_shape, 3))
        mount_quat = torch.broadcast_to(mount_quat, (*batch_shape, 4))
    except RuntimeError as exc:
        raise ValueError("Stage-1 mount offset/quaternion cannot broadcast to the body batch") from exc

    body_quats = tuple(
        _stage1_quat_normalize(value, name=f"body_quat[{index}]")
        for index, value in enumerate(quaternions)
    )
    mount_quat = _stage1_quat_normalize(mount_quat, name="mount_quat")

    def _site(pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        site_pos = pos + _stage1_quat_apply(quat, offset)
        site_quat = _stage1_quat_normalize(
            _stage1_quat_mul(quat, mount_quat), name="site_quat"
        )
        return site_pos, site_quat

    previous_site, _ = _site(previous_body_pos_w, body_quats[0])
    current_site, current_site_quat = _site(current_body_pos_w, body_quats[1])
    next_site, _ = _site(next_body_pos_w, body_quats[2])

    axis = torch.zeros((*batch_shape, 3), device=current_site.device, dtype=current_site.dtype)
    axis[..., normal_axis] = 1.0
    raw_normal = _stage1_quat_apply(current_site_quat, axis)
    sign = torch.as_tensor(normal_sign, device=current_site.device, dtype=current_site.dtype)
    try:
        sign = torch.broadcast_to(sign, batch_shape)
    except RuntimeError as exc:
        raise ValueError("Stage-1 face sign cannot broadcast to the body batch") from exc
    torch._assert_async((torch.isfinite(sign) & (torch.abs(sign) == 1.0)).all())
    normal = raw_normal * sign.unsqueeze(-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-12)

    span = torch.as_tensor(
        central_difference_span_s, device=current_site.device, dtype=current_site.dtype
    )
    try:
        span = torch.broadcast_to(span, batch_shape)
    except RuntimeError as exc:
        raise ValueError("Stage-1 central-difference span cannot broadcast to the body batch") from exc
    torch._assert_async((torch.isfinite(span) & (span > 0.0)).all())
    velocity = (next_site - previous_site) / span.unsqueeze(-1)

    finite = (
        torch.isfinite(current_site).all(dim=-1)
        & torch.isfinite(normal).all(dim=-1)
        & torch.isfinite(velocity).all(dim=-1)
    )
    torch._assert_async(finite.all())
    return current_site, normal, velocity


def stage1_clip_site_target_from_aligned_measured_racket(
    previous_site_pos_w: torch.Tensor,
    current_site_pos_w: torch.Tensor,
    next_site_pos_w: torch.Tensor,
    current_signed_normal_w: torch.Tensor,
    *,
    central_difference_span_s: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and differentiate an already aligned physical-paddle teacher channel."""

    batch_shape = current_site_pos_w.shape[:-1]
    for name, value in (
        ("previous_site_pos_w", previous_site_pos_w),
        ("current_site_pos_w", current_site_pos_w),
        ("next_site_pos_w", next_site_pos_w),
        ("current_signed_normal_w", current_signed_normal_w),
    ):
        if value.shape != (*batch_shape, 3):
            raise ValueError(
                f"Stage-1 measured {name} has shape {tuple(value.shape)}, "
                f"expected {(*batch_shape, 3)}"
            )
        if value.device != current_site_pos_w.device or value.dtype != current_site_pos_w.dtype:
            raise ValueError("Stage-1 measured racket tensors must share device and dtype")
    normal_norm = torch.linalg.vector_norm(current_signed_normal_w, dim=-1, keepdim=True)
    torch._assert_async((torch.isfinite(normal_norm) & (normal_norm > 1.0e-12)).all())
    normal = current_signed_normal_w / normal_norm
    span = torch.as_tensor(
        central_difference_span_s,
        device=current_site_pos_w.device,
        dtype=current_site_pos_w.dtype,
    )
    try:
        span = torch.broadcast_to(span, batch_shape)
    except RuntimeError as exc:
        raise ValueError("Stage-1 measured central-difference span cannot broadcast") from exc
    torch._assert_async((torch.isfinite(span) & (span > 0.0)).all())
    velocity = (next_site_pos_w - previous_site_pos_w) / span.unsqueeze(-1)
    finite = (
        torch.isfinite(current_site_pos_w).all(dim=-1)
        & torch.isfinite(normal).all(dim=-1)
        & torch.isfinite(velocity).all(dim=-1)
    )
    torch._assert_async(finite.all())
    return current_site_pos_w, normal, velocity


def _stage1_split_ready_safe_racket_tuple(
    cmd: RacketTargetCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the official paddle tuple from Motion's physical safe-ready body.

    The returned order is site position, signed face, site-point velocity and
    butt-to-blade long axis.  This path intentionally reads no measured motion
    row.  Position/orientation come from the same aligned safe-ready body
    snapshot used by whole-body mimic, and a frozen snapshot has literal-zero
    point velocity.
    """

    motion = cmd._motion()
    if not bool(
        getattr(motion, "action_ball_diagnostic_split_ready_teacher", False)
    ):
        raise RuntimeError("safe-ready racket tuple requested outside split-ready")

    source_index = (
        int(cmd._racket_body_index)
        if cmd._racket_mode == "body"
        else int(cmd._wrist_body_index)
    )
    robot_body_names = tuple(motion.robot.body_names)
    tracked_body_names = tuple(motion.cfg.body_names)
    if source_index < 0 or source_index >= len(robot_body_names):
        raise RuntimeError("split-ready racket source body index is invalid")
    source_name = robot_body_names[source_index]
    try:
        local_index = tracked_body_names.index(source_name)
    except ValueError as exc:
        raise RuntimeError(
            "split-ready racket source body is absent from the physical "
            "safe-ready tracking tuple"
        ) from exc

    body_pos = motion.body_pos_relative_w
    body_quat = motion.body_quat_relative_w
    expected_pos = (cmd.num_envs, len(tracked_body_names), 3)
    expected_quat = (cmd.num_envs, len(tracked_body_names), 4)
    if body_pos.shape != expected_pos or body_quat.shape != expected_quat:
        raise ValueError(
            "split-ready aligned body tuple changed shape: "
            f"{tuple(body_pos.shape)}/{tuple(body_quat.shape)} vs "
            f"{expected_pos}/{expected_quat}"
        )
    site_pos = body_pos[:, local_index]
    site_quat = _stage1_quat_normalize(
        body_quat[:, local_index], name="safe_ready_racket_body_quat"
    )
    if cmd._racket_mode != "body":
        site_pos = site_pos + _stage1_quat_apply(site_quat, cmd._mount_offset)
        site_quat = _stage1_quat_normalize(
            _stage1_quat_mul(site_quat, cmd._mount_quat),
            name="safe_ready_racket_site_quat",
        )

    axis = torch.zeros_like(site_pos)
    axis[:, int(cmd.cfg.mount_normal_axis)] = 1.0
    signed_face = _stage1_quat_apply(site_quat, axis)
    use_per_clip_sign = bool(cmd.cfg.mount_normal_sign_per_clip) or (
        str(getattr(cmd.cfg, "motion_teacher_racket_source", "robot_fk"))
        == "measured_channel"
    )
    if use_per_clip_sign:
        signs = cmd._mount_signs_cfg(int(motion.motion.num_segments))
        sign_table = torch.as_tensor(
            signs, device=cmd.device, dtype=signed_face.dtype
        )
        sign = (
            sign_table[motion.clip_id]
            if bool(getattr(motion, "_multiseg", False))
            else sign_table[0].expand(cmd.num_envs)
        )
    else:
        sign = torch.full(
            (cmd.num_envs,),
            float(cmd.cfg.mount_normal_sign),
            device=cmd.device,
            dtype=signed_face.dtype,
        )
    signed_face = signed_face * sign[:, None]

    local_long_axis = torch.as_tensor(
        racket_contact_geometry.RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
        device=cmd.device,
        dtype=site_pos.dtype,
    ).expand(cmd.num_envs, 3)
    long_axis = _stage1_quat_apply(site_quat, local_long_axis)
    long_axis = long_axis / torch.linalg.vector_norm(
        long_axis, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    point_velocity = torch.zeros_like(site_pos)
    finite = (
        torch.isfinite(site_pos).all(dim=-1)
        & torch.isfinite(signed_face).all(dim=-1)
        & torch.isfinite(long_axis).all(dim=-1)
    )
    unit = (
        (torch.linalg.vector_norm(signed_face, dim=-1) - 1.0).abs()
        <= 1.0e-5
    ) & (
        (torch.linalg.vector_norm(long_axis, dim=-1) - 1.0).abs()
        <= 1.0e-5
    )
    torch._assert_async((finite & unit).all())
    return site_pos, signed_face, point_velocity, long_axis


def _stage1_select_split_ready_site_target(
    cmd: RacketTargetCommand,
    measured: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Atomically select safe-ready or measured paddle p/face/point-v."""

    wait = _stage1_split_ready_wait_mask(cmd)
    if not bool(
        getattr(
            cmd._motion(), "action_ball_diagnostic_split_ready_teacher", False
        )
    ):
        return measured
    safe = _stage1_split_ready_safe_racket_tuple(cmd)
    return tuple(
        torch.where(wait[:, None], safe_value, measured_value)
        for safe_value, measured_value in zip(safe[:3], measured)
    )


def _stage1_aligned_clip_site_target_at_steps(
    cmd: RacketTargetCommand, current: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the aligned official-site teacher state at explicit reference steps.

    ``current`` is an absolute MotionLoader row per environment.  Keeping the alignment and
    official-site reconstruction in this one helper makes the phase-continuous reward target and
    the nominal contact-time observation share the same frame, mount, face-sign and point-velocity
    semantics.  No ball/task target is read here.
    """

    motion = cmd._motion()
    loader = motion.motion
    required = ("_body_pos_w", "_body_quat_w", "seg_start", "seg_len")
    missing = [name for name in required if not hasattr(loader, name)]
    if missing:
        raise AttributeError(
            "Stage-1 clip-site reward requires MotionLoader full-body reference poses and segment "
            f"bounds; missing {missing}"
        )
    current = current.to(device=cmd.device, dtype=torch.long)
    if current.shape != (cmd.num_envs,):
        raise ValueError(
            f"Stage-1 motion steps must have shape ({cmd.num_envs},), got {tuple(current.shape)}"
        )
    if bool(getattr(motion, "_multiseg", False)):
        clips = motion.clip_id.to(device=cmd.device, dtype=torch.long)
        starts = loader.seg_start[clips]
        ends = starts + loader.seg_len[clips] - 1
    else:
        starts = torch.zeros_like(current)
        ends = torch.full_like(current, int(loader.time_step_total) - 1)
    previous = torch.maximum(current - 1, starts)
    following = torch.minimum(current + 1, ends)
    torch._assert_async((following > previous).all())

    # A held reference is a stationary teacher even though its neighboring source rows encode the
    # future swing.  In particular, split-ready reset leaves ``speed_scale == 0`` until the first
    # command-manager compute, while rewards/observations are evaluated before that compute.  Do
    # not divide the source-row span by zero and do not clamp it to an epsilon (which would invent
    # an enormous moving target).  Use a harmless unit rate for the geometric differentiation and
    # overwrite the held point velocity with literal zero, matching MotionCommand's joint/body
    # velocity semantics.  Unheld rows remain byte-for-byte on the prior span formula.
    held = motion.in_hold.to(
        device=cmd.device, dtype=torch.bool
    ) | _stage1_split_ready_wait_mask(cmd)
    speed = motion.speed_scale.to(device=cmd.device)
    if held.shape != (cmd.num_envs,) or speed.shape != (cmd.num_envs,):
        raise ValueError(
            "Stage-1 hold/speed masks must have one row per environment, got "
            f"{tuple(held.shape)}/{tuple(speed.shape)}"
        )
    torch._assert_async(
        (torch.isfinite(speed) & (speed >= 0.0) & (held | (speed > 0.0))).all()
    )

    teacher_source = str(getattr(cmd.cfg, "motion_teacher_racket_source", "robot_fk"))
    if teacher_source not in ("robot_fk", "measured_channel"):
        raise ValueError(
            "motion_teacher_racket_source must be 'robot_fk' or 'measured_channel', got "
            f"{teacher_source!r}"
        )
    source_index = (
        int(cmd._racket_body_index)
        if cmd._racket_mode == "body"
        else int(cmd._wrist_body_index)
    )
    # ``motion_anchor_body_index`` is local to the configured tracked-body view,
    # while the private MotionLoader tensors below are indexed by the full raw
    # articulation order.  ``robot_anchor_body_index`` is the already-resolved
    # raw index (A3 local 7 -> raw 9) and avoids a per-step GPU ``.item()`` sync.
    anchor_index = int(motion.robot_anchor_body_index)
    body_pos = loader._body_pos_w
    body_quat = loader._body_quat_w
    for name, value, tail in (
        ("_body_pos_w", body_pos, (3,)),
        ("_body_quat_w", body_quat, (4,)),
    ):
        if value.ndim != 3 or value.shape[-1:] != tail:
            raise ValueError(f"MotionLoader {name} has invalid shape {tuple(value.shape)}")
        if source_index < 0 or source_index >= value.shape[1] or anchor_index < 0 or anchor_index >= value.shape[1]:
            raise IndexError(
                f"Stage-1 source/anchor body indexes {source_index}/{anchor_index} exceed "
                f"MotionLoader {name} body count {value.shape[1]}"
            )

    origins = cmd._env.scene.env_origins
    if origins.shape != (cmd.num_envs, 3):
        raise ValueError(
            f"Stage-1 env origins must have shape ({cmd.num_envs}, 3), got {tuple(origins.shape)}"
        )
    if origins.device != body_pos.device or origins.dtype != body_pos.dtype:
        raise ValueError("Stage-1 env origins must share MotionLoader body position device and dtype")
    # MotionCommand.body_pos_w adds env_origins before applying its anchor-relative alignment.
    # Adding the same origin to both the source body and reference anchor below makes the x/y
    # subtraction cancel while preserving a non-zero terrain/origin z exactly like the upstream
    # body_pos_relative_w formula.
    ref_anchor_pos = body_pos[current, anchor_index] + origins
    ref_anchor_quat = _stage1_quat_normalize(
        body_quat[current, anchor_index], name="reference_anchor_quat"
    )
    robot_anchor_pos = motion.robot_anchor_pos_w
    robot_anchor_quat = _stage1_quat_normalize(
        motion.robot_anchor_quat_w, name="robot_anchor_quat"
    )
    ref_anchor_inv = ref_anchor_quat.clone()
    ref_anchor_inv[..., 1:] *= -1.0
    delta_quat = _stage1_yaw_quat(_stage1_quat_mul(robot_anchor_quat, ref_anchor_inv))
    delta_pos = robot_anchor_pos.clone()
    delta_pos[..., 2] = ref_anchor_pos[..., 2]

    def _aligned(step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_pos = body_pos[step, source_index] + origins
        raw_quat = _stage1_quat_normalize(body_quat[step, source_index], name="reference_body_quat")
        aligned_pos = delta_pos + _stage1_quat_apply(delta_quat, raw_pos - ref_anchor_pos)
        aligned_quat = _stage1_quat_normalize(
            _stage1_quat_mul(delta_quat, raw_quat), name="aligned_reference_body_quat"
        )
        return aligned_pos, aligned_quat

    if teacher_source == "measured_channel":
        measured_pos = getattr(loader, "_measured_racket_site_pos_w", None)
        measured_normal = getattr(loader, "_measured_racket_normal_w", None)
        if (
            not bool(getattr(loader, "measured_racket_available", False))
            or measured_pos is None
            or measured_normal is None
        ):
            raise RuntimeError(
                "motion_teacher_racket_source='measured_channel' requires every motion NPZ to "
                "carry the complete measured-racket schema; FK fallback is intentionally forbidden"
            )
        expected_pos = (int(loader.time_step_total), 3)
        if measured_pos.shape != expected_pos or measured_normal.shape != expected_pos:
            raise ValueError(
                "MotionLoader measured-racket tensors changed shape: "
                f"{tuple(measured_pos.shape)}/{tuple(measured_normal.shape)} vs {expected_pos}"
            )

        def _aligned_measured_pos(step: torch.Tensor) -> torch.Tensor:
            raw = measured_pos[step] + origins
            return delta_pos + _stage1_quat_apply(delta_quat, raw - ref_anchor_pos)

        measured_previous = _aligned_measured_pos(previous)
        measured_current = _aligned_measured_pos(current)
        measured_next = _aligned_measured_pos(following)
        measured_normal_current = _stage1_quat_apply(
            delta_quat, measured_normal[current]
        )
        typed_speed = speed.to(dtype=measured_current.dtype)
        safe_speed = torch.where(held, torch.ones_like(typed_speed), typed_speed)
        frame_span = (following - previous).to(dtype=measured_current.dtype)
        span_s = frame_span * float(cmd._env.step_dt) / safe_speed
        position, normal, velocity = stage1_clip_site_target_from_aligned_measured_racket(
            measured_previous,
            measured_current,
            measured_next,
            measured_normal_current,
            central_difference_span_s=span_s,
        )
        return position, normal, torch.where(held[:, None], torch.zeros_like(velocity), velocity)

    previous_pos, previous_quat = _aligned(previous)
    current_pos, current_quat = _aligned(current)
    next_pos, next_quat = _aligned(following)
    if cmd._racket_mode == "body":
        mount_offset = torch.zeros_like(current_pos)
        mount_quat = torch.zeros(
            (cmd.num_envs, 4), device=current_pos.device, dtype=current_pos.dtype
        )
        mount_quat[:, 0] = 1.0
    else:
        mount_offset = cmd._mount_offset
        mount_quat = cmd._mount_quat

    if cmd.cfg.mount_normal_sign_per_clip:
        signs = cmd._mount_signs_cfg(int(loader.num_segments))
        sign_table = torch.as_tensor(signs, device=cmd.device, dtype=current_pos.dtype)
        sign = sign_table[motion.clip_id] if bool(getattr(motion, "_multiseg", False)) else sign_table[0]
    else:
        sign = float(cmd.cfg.mount_normal_sign)
    typed_speed = speed.to(dtype=current_pos.dtype)
    safe_speed = torch.where(held, torch.ones_like(typed_speed), typed_speed)
    frame_span = (following - previous).to(dtype=current_pos.dtype)
    span_s = frame_span * float(cmd._env.step_dt) / safe_speed
    position, normal, velocity = stage1_clip_site_target_from_aligned_body_pose(
        previous_pos,
        previous_quat,
        current_pos,
        current_quat,
        next_pos,
        next_quat,
        mount_offset_body=mount_offset,
        mount_quat_wxyz=mount_quat,
        normal_axis=int(cmd.cfg.mount_normal_axis),
        normal_sign=sign,
        central_difference_span_s=span_s,
    )
    return position, normal, torch.where(held[:, None], torch.zeros_like(velocity), velocity)


def _stage1_aligned_clip_site_target(
    cmd: RacketTargetCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize one cached, phase-continuous teacher site target for this policy step."""

    token = getattr(getattr(cmd, "_env", None), "common_step_counter", None)
    cached = getattr(cmd, "_stage1_clip_site_target_cache", None)
    if type(token) is int and cached is not None and cached[0] == token:
        measured = cached[1]
    else:
        current = cmd._motion()._pose_reference_steps()
        measured = _stage1_aligned_clip_site_target_at_steps(cmd, current)
        if type(token) is int:
            cmd._stage1_clip_site_target_cache = (token, measured)
    # Cache only the measured producer.  task_valid may reveal within the same
    # public control token, so caching the selected value would make one
    # consumer observe WAIT while another observes TASK_ACTIVE.
    return _stage1_select_split_ready_site_target(cmd, measured)


def _stage1_aligned_clip_measured_long_axis_target(
    cmd: RacketTargetCommand,
) -> torch.Tensor:
    """Return the aligned measured butt-to-blade axis at the current teacher phase."""

    if str(getattr(cmd.cfg, "motion_teacher_racket_source", "robot_fk")) != "measured_channel":
        raise RuntimeError(
            "racket long-axis imitation requires motion_teacher_racket_source='measured_channel'"
        )
    token = getattr(getattr(cmd, "_env", None), "common_step_counter", None)
    cached = getattr(cmd, "_stage1_clip_long_axis_target_cache", None)
    if type(token) is int and cached is not None and cached[0] == token:
        return cached[1]
    motion = cmd._motion()
    loader = motion.motion
    measured_long_axis = getattr(loader, "_measured_racket_long_axis_w", None)
    if (
        not bool(getattr(loader, "measured_racket_available", False))
        or measured_long_axis is None
        or measured_long_axis.shape != (int(loader.time_step_total), 3)
    ):
        raise RuntimeError(
            "measured-channel long-axis reward requires one schema-3 axis row per motion frame"
        )
    current = motion._pose_reference_steps().to(device=cmd.device, dtype=torch.long)
    if current.shape != (cmd.num_envs,):
        raise ValueError("measured long-axis reference steps changed shape")
    body_quat = loader._body_quat_w
    anchor_index = int(motion.robot_anchor_body_index)
    if (
        body_quat.ndim != 3
        or body_quat.shape[-1] != 4
        or not 0 <= anchor_index < body_quat.shape[1]
    ):
        raise ValueError("MotionLoader anchor quaternion contract changed")
    ref_anchor_quat = _stage1_quat_normalize(
        body_quat[current, anchor_index], name="reference_anchor_quat"
    )
    robot_anchor_quat = _stage1_quat_normalize(
        motion.robot_anchor_quat_w, name="robot_anchor_quat"
    )
    ref_anchor_inv = ref_anchor_quat.clone()
    ref_anchor_inv[..., 1:] *= -1.0
    delta_quat = _stage1_yaw_quat(
        _stage1_quat_mul(robot_anchor_quat, ref_anchor_inv)
    )
    result = _stage1_quat_apply(delta_quat, measured_long_axis[current])
    norm = torch.linalg.vector_norm(result, dim=-1, keepdim=True)
    torch._assert_async((torch.isfinite(norm) & (norm > 1.0e-12)).all())
    result = result / norm
    if type(token) is int:
        cmd._stage1_clip_long_axis_target_cache = (token, result)
    return result


def _stage1_aligned_clip_long_axis_target(
    cmd: RacketTargetCommand,
) -> torch.Tensor:
    """Atomically select safe-ready or measured butt-to-blade axis."""

    measured = _stage1_aligned_clip_measured_long_axis_target(cmd)
    if not bool(
        getattr(
            cmd._motion(), "action_ball_diagnostic_split_ready_teacher", False
        )
    ):
        return measured
    wait = _stage1_split_ready_wait_mask(cmd)
    safe_long_axis = _stage1_split_ready_safe_racket_tuple(cmd)[3]
    return torch.where(wait[:, None], safe_long_axis, measured)


def stage1_aligned_clip_site_target_now(
    cmd: RacketTargetCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public full-phase teacher state at the motion command's current reference step.

    Tuple order is ``(position, signed normal, point velocity)``.  Observation producers may reorder
    the three blocks at concatenation time, but must not independently reconstruct their geometry.
    """

    return _stage1_aligned_clip_site_target(cmd)


def stage1_aligned_clip_site_target_at_reference_hit(
    cmd: RacketTargetCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the clip teacher's nominal contact-time paddle state for every environment.

    The configured per-clip strike phase is converted to an absolute reference row, then passed
    through the exact same alignment/site/face/central-difference path as the full-phase reward.
    This is a deterministic clip-derived observation producer: it never reads a sampled ball target,
    a planner result, or the actor's current paddle state.
    """

    token = getattr(getattr(cmd, "_env", None), "common_step_counter", None)
    cached = getattr(cmd, "_stage1_clip_site_reference_hit_cache", None)
    if type(token) is int and cached is not None and cached[0] == token:
        measured = cached[1]
    else:
        env_ids = torch.arange(
            cmd.num_envs, device=cmd.device, dtype=torch.long
        )
        # One row authority: this is the exact producer used by the live strike clock, including its
        # single-clip Python-double and multi-clip cached-tensor rounding behavior.  Reimplementing the
        # phase arithmetic here would let a half-frame boundary put the observation and reward windows
        # on adjacent reference rows.
        hit_steps = cmd._strike_steps_for_envs(env_ids).to(
            device=cmd.device, dtype=torch.long
        )
        if hit_steps.shape != (cmd.num_envs,):
            raise ValueError("Stage-1 reference-hit rows changed shape")
        measured = _stage1_aligned_clip_site_target_at_steps(cmd, hit_steps)
        if type(token) is int:
            cmd._stage1_clip_site_reference_hit_cache = (token, measured)
    return _stage1_select_split_ready_site_target(cmd, measured)


def stage1_clip_racket_tracking_errors(
    cmd: RacketTargetCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (site-position m, signed-normal rad, site-velocity m/s) Stage-1 errors.

    This is also the intentionally small interface the monotonic adaptive-sigma controller consumes
    over the full clip.  Merely renaming reward terms is insufficient: the legacy driver compares
    against ball-conditioned ``racket_target_*`` buffers at exact strike.
    """

    target_pos, target_normal, target_velocity = _stage1_aligned_clip_site_target(cmd)
    pos_error = torch.linalg.vector_norm(cmd.racket_pos_w - target_pos, dim=-1)
    normal_cos = torch.sum(cmd.racket_normal_w * target_normal, dim=-1).clamp(-1.0, 1.0)
    normal_error = torch.acos(normal_cos)
    velocity_error = torch.linalg.vector_norm(cmd.racket_lin_vel_w - target_velocity, dim=-1)
    return pos_error, normal_error, velocity_error


def stage1_clip_racket_position_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the natural clip's aligned official racket site over the full clip."""

    cmd = _cmd(env, command_name)
    target_pos, _, _ = _stage1_aligned_clip_site_target(cmd)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos), dim=-1)
    raw = torch.exp(-error / float(std) ** 2)
    full_phase = torch.ones_like(cmd.strike_window, dtype=torch.bool)
    _dbg_log(cmd, "stage1_clip_racket_pos", raw, full_phase)
    return raw


def stage1_clip_racket_normal_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track the natural clip's signed physical paddle face over the full clip."""

    cmd = _cmd(env, command_name)
    _, target_normal, _ = _stage1_aligned_clip_site_target(cmd)
    cos_angle = torch.sum(cmd.racket_normal_w * target_normal, dim=-1).clamp(-1.0, 1.0)
    raw = torch.exp(-torch.square(torch.acos(cos_angle)) / float(std) ** 2)
    full_phase = torch.ones_like(cmd.strike_window, dtype=torch.bool)
    _dbg_log(cmd, "stage1_clip_racket_normal", raw, full_phase)
    return raw


def stage1_clip_racket_velocity_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track point-consistent clip site velocity over the full clip."""

    cmd = _cmd(env, command_name)
    _, _, target_velocity = _stage1_aligned_clip_site_target(cmd)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - target_velocity), dim=-1)
    raw = torch.exp(-error / float(std) ** 2)
    full_phase = torch.ones_like(cmd.strike_window, dtype=torch.bool)
    _dbg_log(cmd, "stage1_clip_racket_vel", raw, full_phase)
    return raw


def stage1_clip_racket_position_coarse_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Fixed broad full-clip position kernel; ``std=0.70 m`` covers the cold-start envelope."""

    cmd = _cmd(env, command_name)
    target_pos, _, _ = _stage1_aligned_clip_site_target(cmd)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos), dim=-1)
    raw = torch.exp(-error / float(std) ** 2)
    full_phase = torch.ones_like(cmd.strike_window, dtype=torch.bool)
    _dbg_log(cmd, "stage1_clip_racket_pos_coarse", raw, full_phase)
    return raw


def stage1_clip_racket_velocity_coarse_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Fixed broad full-clip velocity kernel; ``std=4 m/s`` covers the cold-start envelope."""

    cmd = _cmd(env, command_name)
    _, _, target_velocity = _stage1_aligned_clip_site_target(cmd)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - target_velocity), dim=-1)
    raw = torch.exp(-error / float(std) ** 2)
    full_phase = torch.ones_like(cmd.strike_window, dtype=torch.bool)
    _dbg_log(cmd, "stage1_clip_racket_vel_coarse", raw, full_phase)
    return raw


def stage1_clip_racket_normal_coarse_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Fixed broad full-clip face kernel; ``std=pi rad`` covers the cold-start envelope."""

    cmd = _cmd(env, command_name)
    _, target_normal, _ = _stage1_aligned_clip_site_target(cmd)
    cos_angle = torch.sum(cmd.racket_normal_w * target_normal, dim=-1).clamp(-1.0, 1.0)
    raw = torch.exp(-torch.square(torch.acos(cos_angle)) / float(std) ** 2)
    full_phase = torch.ones_like(cmd.strike_window, dtype=torch.bool)
    _dbg_log(cmd, "stage1_clip_racket_normal_coarse", raw, full_phase)
    return raw


def motion_racket_position_tracking_cauchy(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    scale_in_strike_window: float = 1.0,
) -> torch.Tensor:
    """Measured-paddle position imitation, optionally attenuated at ball contact."""

    cmd = _cmd(env, command_name)
    target_pos, _, _ = _stage1_aligned_clip_site_target(cmd)
    error = torch.linalg.vector_norm(cmd.racket_pos_w - target_pos, dim=-1)
    raw = _cauchy_tracking_kernel(error, std)
    scale = (~_window_wide(cmd)).float() + _window_wide(cmd).float() * float(
        scale_in_strike_window
    )
    return raw * scale


def motion_racket_velocity_tracking_cauchy(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    scale_in_strike_window: float = 1.0,
) -> torch.Tensor:
    """Measured-paddle site-velocity imitation, optionally attenuated at ball contact."""

    cmd = _cmd(env, command_name)
    _, _, target_velocity = _stage1_aligned_clip_site_target(cmd)
    error = torch.linalg.vector_norm(cmd.racket_lin_vel_w - target_velocity, dim=-1)
    raw = _cauchy_tracking_kernel(error, std)
    scale = (~_window_wide(cmd)).float() + _window_wide(cmd).float() * float(
        scale_in_strike_window
    )
    return raw * scale


def motion_racket_normal_tracking_cauchy(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    scale_in_strike_window: float = 1.0,
) -> torch.Tensor:
    """Measured physical-face imitation, optionally attenuated at ball contact."""

    cmd = _cmd(env, command_name)
    _, target_normal, _ = _stage1_aligned_clip_site_target(cmd)
    cosine = torch.sum(cmd.racket_normal_w * target_normal, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cosine)
    raw = _cauchy_tracking_kernel(error, std)
    scale = (~_window_wide(cmd)).float() + _window_wide(cmd).float() * float(
        scale_in_strike_window
    )
    return raw * scale


def motion_racket_long_axis_tracking_cauchy(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    scale_in_strike_window: float = 1.0,
) -> torch.Tensor:
    """Measured butt-to-blade imitation; closes wrist twist left invisible by face normal."""

    cmd = _cmd(env, command_name)
    target_long_axis = _stage1_aligned_clip_long_axis_target(cmd)
    cosine = torch.sum(cmd.racket_long_axis_w * target_long_axis, dim=-1).clamp(-1.0, 1.0)
    raw = _cauchy_tracking_kernel(torch.acos(cosine), std)
    scale = (~_window_wide(cmd)).float() + _window_wide(cmd).float() * float(
        scale_in_strike_window
    )
    return raw * scale


def stage1_clip_racket_position_precision_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Fixed narrow contact-position bonus in the tight strike window."""

    cmd = _cmd(env, command_name)
    target_pos, _, _ = _stage1_aligned_clip_site_target(cmd)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos), dim=-1)
    raw = torch.exp(-error / float(std) ** 2)
    win = _window_pos(cmd)
    _dbg_log(cmd, "stage1_clip_racket_pos_precision", raw, win)
    return raw * win.float()


def stage1_clip_racket_velocity_precision_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Fixed narrow contact-velocity bonus in the wide strike window."""

    cmd = _cmd(env, command_name)
    _, _, target_velocity = _stage1_aligned_clip_site_target(cmd)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - target_velocity), dim=-1)
    raw = torch.exp(-error / float(std) ** 2)
    win = _window_wide(cmd)
    _dbg_log(cmd, "stage1_clip_racket_vel_precision", raw, win)
    return raw * win.float()


def stage1_clip_racket_normal_precision_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Fixed narrow signed-face bonus in the wide strike window."""

    cmd = _cmd(env, command_name)
    _, target_normal, _ = _stage1_aligned_clip_site_target(cmd)
    cos_angle = torch.sum(cmd.racket_normal_w * target_normal, dim=-1).clamp(-1.0, 1.0)
    raw = torch.exp(-torch.square(torch.acos(cos_angle)) / float(std) ** 2)
    win = _window_wide(cmd)
    _dbg_log(cmd, "stage1_clip_racket_normal_precision", raw, win)
    return raw * win.float()


def _pos_gate(cmd: RacketTargetCommand, pos_gate_radius: float | None) -> torch.Tensor | float:
    """Proximity power-gate (reward_staged_design §② C2a): sigmoid((r_gate - pos_err)/0.05) with
    pos_err = ||racket_FK - target||. ~0 when the paddle cannot reach the target (no face/velocity
    money AND no face/velocity gradient noise while out of reach), ~1 once inside the gate; smooth
    so there is no bang-bang flicker at the gate edge. 人话:拍子够得着球才开始付拍面/拍速的钱。
    ``None`` (the default of every caller) returns 1.0 — byte-identical baseline."""
    if pos_gate_radius is None:
        return 1.0
    if not _target_component_valid(cmd, "position"):
        return 1.0
    pos_err = torch.norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    return torch.sigmoid((float(pos_gate_radius) - pos_err) / 0.05)


def _target_component_valid(cmd: RacketTargetCommand, component: str) -> bool:
    """Run-constant fixed-question ablation mask; legacy commands are fully valid."""

    accessor = getattr(cmd, "action_ball_target_component_valid", None)
    return True if accessor is None else bool(accessor(component))


def _target_position_now(cmd: RacketTargetCommand) -> torch.Tensor:
    """Contact point trajectory without leaking an invalid desired velocity channel."""

    if _target_component_valid(cmd, "velocity"):
        return (
            cmd.racket_target_pos_w
            - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
        )
    return cmd.racket_target_pos_w


def _pos_kernel_raw(cmd: RacketTargetCommand, std: float) -> torch.Tensor:
    """UNGATED swing-through position kernel (shared by racket_position / racket_strike_success)."""
    if not _target_component_valid(cmd, "position"):
        return torch.zeros_like(cmd.time_to_strike)
    target_pos_now = _target_position_now(cmd)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    return torch.exp(-error / std**2)


def _vel_kernel_raw(cmd: RacketTargetCommand, std: float) -> torch.Tensor:
    """UNGATED velocity kernel (shared by racket_velocity / racket_strike_success)."""
    if not _target_component_valid(cmd, "velocity"):
        return torch.zeros_like(cmd.time_to_strike)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def _face_pair(cmd: RacketTargetCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """(measured, target) face normals for EVERY face-channel term — the single source of the
    face frame. Any new face reward/penalty MUST read through here, never pick buffers itself.

    face_command=True (question-bank demanded normals): the bank is a +Y-calibration-frame ("A"
    convention) product — gen_stage1_questions.py sign-aligns every demanded normal to the RAW +Y
    clip face and has no notion of the striking-face sign table. So the measured side must be the
    raw +Y axis (``racket_normal_raw_w``), NOT the striking-face-signed ``racket_normal_w``.
    Pairing the signed normal against the bank target was the M3c/M2f 单翻病 (2026-07-09 病因定案):
    on sign=-1 (backhand) clips the reward optimum sat ~180° from the physically correct face and
    both arms converged to a ~34° systematic face error. A-vs-A is bitwise identical to flipping
    both sides (dot(-a,-b) == dot(a,b)), so nothing is lost: the mount sign table keeps serving
    the metric / reference / diagnostic channels (B convention) untouched.

    face_command=False: unchanged clip-reference pairing (signed vs signed — both sides carry the
    same per-clip sign, so this path is flip-invariant and byte-identical to the baseline).
    """
    return face_tracking_pair(cmd)


def _normal_kernel_raw(cmd: RacketTargetCommand, std: float) -> torch.Tensor:
    """UNGATED face-normal kernel (shared by racket_normal / racket_strike_success).

    Stage-1 face command: the reference is the DEMANDED (inverse-solved, question-bank) normal
    instead of the clip-locked reference normal. face_command=False keeps the old tensor read —
    byte-identical baseline. racket_strike_success re-anchors through this helper automatically.
    The (measured, target) pair comes from ``_face_pair`` — see its docstring for the frame rules.
    """
    if not _target_component_valid(cmd, "face"):
        return torch.zeros_like(cmd.time_to_strike)
    measured, target_normal = _face_pair(cmd)
    cos_ang = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    return torch.exp(-(angle**2) / std**2)


def racket_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket center position near strike using the target's swing-through trajectory.
    Gated by the TIGHT position window (1c): contact must be precise; == strike_window by default."""
    cmd = _cmd(env, command_name)
    raw = _pos_kernel_raw(cmd, std)
    win = _window_pos(cmd)
    _dbg_log(cmd, "racket_pos", raw, win)
    return raw * win.float()


def racket_position_coarse_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Coarse companion to :func:`racket_position_tracking_exp`.

    This deliberately shares the exact swing-through target and TIGHT strike window with the
    precision channel.  Only ``std`` and the independently configured reward weight differ.  A
    separate term keeps the 7.5 cm precision objective intact while allowing a wider kernel to
    rank the currently observed 20--70 cm strike-window misses.
    """
    cmd = _cmd(env, command_name)
    raw = _pos_kernel_raw(cmd, std)
    win = _window_pos(cmd)
    _dbg_log(cmd, "racket_pos_coarse", raw, win)
    return raw * win.float()


def _cauchy_tracking_kernel(error: torch.Tensor, std: float) -> torch.Tensor:
    """Polynomial-tail tracking kernel used by the far-error ActionBall channels.

    ``exp(-(e/std)^2)`` is an excellent precision reward but its gradient is numerically absent
    for the 20--70 cm / 1--3 m/s errors observed at cold start.  ``1/(1+(e/std)^2)`` has the same
    unique optimum and half-height at ``e == std`` while retaining a finite, correctly directed
    gradient at every finite non-zero error.  The helper accepts an error magnitude (not a squared
    error) so position, velocity, and angular channels share exactly one landscape contract.
    """

    scale = float(std)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Cauchy tracking std must be finite and positive, got {std!r}")
    return torch.reciprocal(1.0 + torch.square(error / scale))


def racket_position_coarse_tracking_cauchy(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Wide strike-window position shaping whose far-error gradient does not exponentially die."""

    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "position"):
        return torch.zeros_like(cmd.time_to_strike)
    target = _target_position_now(cmd)
    error = torch.linalg.vector_norm(cmd.racket_pos_w - target, dim=-1)
    raw = _cauchy_tracking_kernel(error, std)
    win = _window_pos(cmd)
    _dbg_log(cmd, "racket_pos_coarse", raw, win)
    return raw * win.float()


def racket_velocity_coarse_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Gaussian broad velocity companion retained for explicit comparison arms."""

    cmd = _cmd(env, command_name)
    raw = _vel_kernel_raw(cmd, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_vel_coarse", raw, win)
    return raw * win.float() * _pos_gate(cmd, None)


def racket_normal_coarse_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Gaussian broad signed-face companion retained for explicit comparison arms."""

    cmd = _cmd(env, command_name)
    raw = _normal_kernel_raw(cmd, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_normal_coarse", raw, win)
    return raw * win.float() * _pos_gate(cmd, None)


def racket_velocity_coarse_tracking_cauchy(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Wide strike-window site-velocity shaping with polynomial rather than exponential tails."""

    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "velocity"):
        return torch.zeros_like(cmd.time_to_strike)
    error = torch.linalg.vector_norm(cmd.racket_lin_vel_w - cmd.racket_target_vel_w, dim=-1)
    raw = _cauchy_tracking_kernel(error, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_vel_coarse", raw, win)
    return raw * win.float() * _pos_gate(cmd, None)


def racket_normal_coarse_tracking_cauchy(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Wide strike-window signed-face shaping with a polynomial-tail angular landscape."""

    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "face"):
        return torch.zeros_like(cmd.time_to_strike)
    measured, target_normal = _face_pair(cmd)
    cosine = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cosine)
    raw = _cauchy_tracking_kernel(error, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_normal_coarse", raw, win)
    return raw * win.float() * _pos_gate(cmd, None)


def racket_position_tracking_static_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Ablation B: track the strike POINT itself (no swing-through), decoupling position from timing/velocity.

    Identical gating to ``racket_position_tracking_exp`` but compares against the bare ``racket_target_pos_w``
    instead of the moving swing-through point ``target - vel*t_to_strike``. Over a ±0.15 s window the
    swing-through point sweeps up to ~0.9 m at a 6 m/s target, so the standard term mostly rewards being on
    the moving line (timing/velocity); this variant gives a clean "get the paddle to the point" signal for
    early stable positioning. Select via ``rewards.racket_position_static: true`` in the task YAML.
    """
    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "position"):
        return torch.zeros_like(cmd.time_to_strike)
    error = torch.sum(torch.square(cmd.racket_pos_w - cmd.racket_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    win = _window_pos(cmd)
    _dbg_log(cmd, "racket_pos", raw, win)
    return raw * win.float()


def racket_velocity_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, pos_gate_radius: float | None = None
) -> torch.Tensor:
    """Track racket linear velocity near the strike time (FK actual vs desired, world frame).
    Gated by the WIDE window (1c; == strike_window by default) and, when rewards.face_gate_by_pos
    is on, by the proximity power-gate (see ``_pos_gate``)."""
    cmd = _cmd(env, command_name)
    raw = _vel_kernel_raw(cmd, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_vel", raw, win)
    return raw * win.float() * _pos_gate(cmd, pos_gate_radius)


def racket_normal_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, pos_gate_radius: float | None = None
) -> torch.Tensor:
    """Track racket face-normal orientation near the strike time. ``std`` is in radians.
    Gated by the WIDE window (1c; == strike_window by default) and, when rewards.face_gate_by_pos
    is on, by the proximity power-gate (see ``_pos_gate``)."""
    cmd = _cmd(env, command_name)
    raw = _normal_kernel_raw(cmd, std)
    win = _window_wide(cmd)
    _dbg_log(cmd, "racket_normal", raw, win)
    return raw * win.float() * _pos_gate(cmd, pos_gate_radius)


def base_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track desired base XY position before the strike (encourages repositioning footwork)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    eligible = cmd.pre_strike & action_ball_task_valid_mask(cmd)
    _dbg_log(cmd, "base", raw, eligible)
    return raw * eligible.float()


def post_strike_brake(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """POSITIVE braking reward through the FOLLOW-THROUGH (2026-07-07 continuous-rally upgrade).

    Deploy P7 failure mode: the walk-and-strike lunge carries base momentum past the strike; with
    nothing positive active in the tts<0 segment (every goal term is pre_strike/strike_window gated)
    the policy has no incentive to arrest it, and over consecutive swings the displacement
    accumulates until a swing starts from an untrained stance and falls. This term pays
    ``exp(-(|v_base_xy|/std)^2)`` ONLY in the follow-through window::

        (~pre_strike) & (~strike_window)

    i.e. from strike-window EXIT (tts < -strike_window_s) to the clip wrap — it can never touch the
    strike itself (the swing's through-speed is strike_window-protected), and on the wrap step tts
    snaps positive for the next swing so the window closes exactly at the wrap. During a post-wrap
    HOLD, ``pre_strike`` is True (the hold freezes tts positive at the windup value), so braking
    there is ``hold_ready``'s job (stillness x planted feet), not this term's. The window length is
    clip-clocked (not policy-controllable), so the bounded positive income cannot be farmed by
    prolonging it. Deliberately NO position target here: pulling toward any station mid-follow-
    through fights the swing's natural momentum sink — position homing is ``base_position``'s job
    once the next station appears at the wrap.
    """
    cmd = _cmd(env, command_name)
    v_xy = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_xy / std))
    gate = (~cmd.pre_strike) & (~cmd.strike_window)
    _dbg_log(cmd, "post_strike_brake", raw, gate)
    return raw * gate.float()


def pre_strike_foot_slip(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize horizontal foot speed WHILE the foot is in contact, BEFORE the strike only.

    The robot was sliding/leaning to reach far racket targets while the base reward pinned it near spawn
    (foot_slip_speed high, foot_contact_frac low). This term teaches it to plant its feet and stabilize
    during the approach. It is gated by ``pre_strike`` ONLY (not ``strike_window``), so the strike swing's
    footwork is untouched. ``foot_slip_in_contact`` (sum over feet of horizontal speed * in_contact) is
    precomputed by the RacketTargetCommand each step (0 if the feet/contact sensor cannot be resolved).
    Returns a positive magnitude; the RewTerm weight is negative.
    """
    cmd = _cmd(env, command_name)
    return cmd.foot_slip_in_contact * cmd.pre_strike.float()


# ============================================================================================== #
# Footwork-to-strike (BASE-FREE). The legs move because moving the body REDUCES the racket->target
# distance (racket_progress), not because they track a base target. Footwork is penalized for being
# BAD (slip / drag / violent / unstable at strike), NOT for stepping — the feet are free to move.
# ============================================================================================== #
def racket_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """DENSE pre-strike reward for reducing the racket->target distance (prev - current, clamped). This
    is the base-free driver of whole-body footwork: the legs/waist/arms all get credit for moving the
    racket closer to the target, with NO base-position target. Gated to pre_strike (approach phase); the
    strike swing itself is scored by the racket pos/vel/normal terms. Positive when approaching; RewTerm
    weight is POSITIVE."""
    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "position"):
        return torch.zeros_like(cmd.racket_progress)
    eligible = cmd.pre_strike & action_ball_task_valid_mask(cmd)
    return cmd.racket_progress * eligible.float()


def hold_ready(
    env: ManagerBasedRLEnv, command_name: str, std: float, reach: float = 0.65, reach_mode: str = "racket"
) -> torch.Tensor:
    """POSITIVE ready-stance reward during the pre-swing HOLD (the between-swing recovery phase).

    HITTER's balance recovery comes from a positive "prepare for the next target" signal (its pre-strike
    base-position reward), not from balance penalties. In the base-free deploy-parity design the hold
    phase (reference frozen at the next swing's first frame) already pulls the UPPER body to the ready
    pose via imitation, but the legs/base get zero positive signal — only penalties. This term fills that
    gap without a base-position target (deploy-honest: everything here is proprioceptive in spirit —
    stillness + planted feet): ``exp(-(|v_base|^2 + |w_base|^2)/std^2) * feet_contact_frac``, gated to
    the motion command's ``in_hold`` mask. Rewards arriving at the next windup calm, upright-by-stillness
    and with both feet planted — i.e. finishing the previous swing in a recoverable state.

    ``reach`` gate: stillness is only the CORRECT ready action when the robot is already where it can
    strike from. Without the gate this term pays ~weight/step for planted stillness, which out-earns the
    telescoping racket_progress for stepping during the hold — i.e. it would teach freeze-then-rush
    exactly when wide target boxes need footwork. Two gate modes (``reach_mode``):

    * ``"racket"`` (legacy default, base-free tasks): ``racket_target_distance < reach`` — the 3D
      FK-blade->target distance. CAVEAT (2026-07-05 footwork audit): this gate is NOT
      station-selective — the blade distance is arm-pose-controllable (arm imitation is swing-only,
      so reaching toward the target during the hold is reward-free), and for near-side targets it is
      SMALLER at the wrong station than at the correct one, inverting the settle income exactly where
      a step is required. Keep it only for base-free tasks that have no meaningful station.
    * ``"station"`` (HITTER footwork tasks): ``|base_xy − base_target_xy| < reach`` — the planar
      base->commanded-station error. Station-selective by construction and not arm-gameable: far
      station -> the term is silent (base_position/racket_progress drive the step, untaxed);
      arrived -> the stillness income switches on (move to the stance, THEN settle, then swing).

    Zero outside the hold (the swing itself is untouched) and a safe no-op if the motion command has
    no hold state. RewTerm weight is POSITIVE.
    """
    cmd = _cmd(env, command_name)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    if in_hold is None:
        return torch.zeros(cmd.num_envs, device=cmd.device)
    data = cmd.robot.data
    motion_sq = torch.sum(torch.square(data.root_lin_vel_w), dim=-1) + torch.sum(
        torch.square(data.root_ang_vel_w), dim=-1
    )
    raw = torch.exp(-motion_sq / std**2) * cmd.feet_contact_frac
    if reach_mode == "station":
        station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w, dim=-1)
        near = (station_err < reach).float()
    elif reach_mode == "racket":
        near = (cmd.racket_target_distance < reach).float()
    else:
        raise ValueError(f"hold_ready: unknown reach_mode '{reach_mode}' (expected 'racket' or 'station')")
    return raw * near * in_hold.float()


def hold_heading(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.6
) -> torch.Tensor:
    """Reward re-squaring to world +x during a recovery hold.

    A yawed stand-start distribution supplies the missing recovery states; this term is
    deliberately zero outside ``in_hold`` so it cannot reshape the strike itself.
    """
    if not math.isfinite(float(std)) or float(std) <= 0.0:
        raise ValueError(f"hold_heading std must be finite and > 0, got {std!r}")
    cmd = _cmd(env, command_name)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    if in_hold is None:
        return torch.zeros(cmd.num_envs, device=cmd.device)
    q = cmd.base_quat_w  # scalar-first (w, x, y, z)
    forward_x = 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    forward_y = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    yaw = torch.atan2(forward_y, forward_x)
    return torch.exp(-torch.square(yaw) / std**2) * in_hold.float()


_BASE_DECEL_ACTIVATION_ATTR = "_hope_base_decel_activation_counters"
_BASE_DECEL_LAST_STEP_ATTR = "_hope_base_decel_activation_last_step"
_BASE_DECEL_LAST_SIGNATURE_ATTR = "_hope_base_decel_activation_last_signature"
_BASE_DECEL_ELIGIBLE_COUNT = "base_decel_eligible_sample_count"
_BASE_DECEL_RAW_KERNEL_SUM = "base_decel_raw_kernel_sum"
_BASE_DECEL_RAW_NONZERO_COUNT = "base_decel_raw_kernel_nonzero_sample_count"


def _base_decel_values(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_gain: float,
    v_max: float,
    std: float,
) -> tuple[RacketTargetCommand, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact base-decel command, raw kernel, eligibility mask, and reward.

    Both the actual RewardTerm and the weight-independent activation observer below use this
    function.  Keeping the arithmetic in one place prevents a zero-weight control from being
    measured with a mask or kernel that differs from the treatment's real reward execution.
    """

    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "position"):
        zero = torch.zeros_like(cmd.time_to_strike)
        return cmd, zero, torch.zeros_like(cmd.pre_strike), zero
    planar_err = torch.norm(
        cmd.racket_target_pos_w[:, :2] - cmd.racket_pos_w[:, :2], dim=-1
    )
    v_des = (v_gain * planar_err).clamp(0.0, v_max)
    v_base = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    raw = torch.exp(-torch.square(v_base - v_des) / std**2)
    in_hold = getattr(cmd._motion(), "in_hold", None)
    eligible = cmd.pre_strike if in_hold is None else (cmd.pre_strike & ~in_hold)
    eligible = eligible & action_ball_task_valid_mask(cmd)
    reward = raw * eligible.float()
    return cmd, raw, eligible, reward


def _base_decel_counter_state(
    cmd: RacketTargetCommand, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Lazily allocate device scalar counters without changing command configuration."""

    state = getattr(cmd, _BASE_DECEL_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _BASE_DECEL_ELIGIBLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _BASE_DECEL_RAW_KERNEL_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _BASE_DECEL_RAW_NONZERO_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
        }
        setattr(cmd, _BASE_DECEL_ACTIVATION_ATTR, state)
    return state


def _base_decel_step_token(env: ManagerBasedRLEnv) -> int | None:
    """Return Isaac's host-side simulator-step token when the environment exposes it.

    The real manager environment owns an integer ``common_step_counter``.  Synthetic callers that
    do not expose it still get correct reward arithmetic, but cannot request same-step idempotence.
    Avoiding ``int(tensor)`` here is deliberate: activation accounting must never add a CUDA sync.
    """

    token = getattr(env, "common_step_counter", None)
    return token if type(token) is int else None


def _record_base_decel_activation(
    env: ManagerBasedRLEnv,
    cmd: RacketTargetCommand,
    raw: torch.Tensor,
    eligible: torch.Tensor,
    reward: torch.Tensor,
    *,
    signature: tuple[float, float, float],
) -> None:
    """Accumulate one unique simulator step of activation evidence.

    A future unconditional step hook must call :func:`observe_base_decel_activation` for both the
    zero-weight control and the treatment.  The treatment's RewardManager also calls
    :func:`base_decel_tracking`; the shared ``common_step_counter`` token makes those two calls
    idempotent.  A parameter mismatch in the same step fails loudly instead of letting the hook
    attest a different kernel than the reward used.
    """

    token = _base_decel_step_token(env)
    if token is not None:
        previous_token = getattr(cmd, _BASE_DECEL_LAST_STEP_ATTR, None)
        if previous_token == token:
            previous_signature = getattr(cmd, _BASE_DECEL_LAST_SIGNATURE_ATTR, None)
            if previous_signature != signature:
                raise RuntimeError(
                    "base-decel activation observer and RewardTerm used different "
                    "v_gain/v_max/std values in the same simulator step"
                )
            return

    state = _base_decel_counter_state(cmd, raw)
    state[_BASE_DECEL_ELIGIBLE_COUNT].add_(
        eligible.detach().sum(dtype=torch.long)
    )
    # ``reward`` is the exact raw kernel after the real ``pre_strike & ~in_hold`` gate.  Summing
    # this tensor (rather than recomputing or indexing ``raw``) preserves NaN/Inf evidence and the
    # treatment's executed arithmetic exactly.
    state[_BASE_DECEL_RAW_KERNEL_SUM].add_(reward.detach().sum())
    raw_detached = raw.detach()
    state[_BASE_DECEL_RAW_NONZERO_COUNT].add_(
        (
            eligible.detach()
            & torch.isfinite(raw_detached)
            & raw_detached.gt(0)
        ).sum(dtype=torch.long)
    )
    if token is not None:
        setattr(cmd, _BASE_DECEL_LAST_STEP_ATTR, token)
        setattr(cmd, _BASE_DECEL_LAST_SIGNATURE_ATTR, signature)


def observe_base_decel_activation(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_gain: float = 2.0,
    v_max: float = 1.6,
    std: float = 0.4,
) -> None:
    """Measure base-decel activation without applying any reward.

    Both arms call this observer from :func:`base_decel_activation_probe`, a nonzero-weight
    instrumentation RewardTerm.  Keeping the observer in RewardManager is essential: Isaac 2.1
    computes reward before reset/command updates, so a command-stage observer would measure the
    next command state while the treatment's real RewardTerm measured the previous state.  The
    probe performs no random draw or simulator write and shares the treatment's exact kernel/mask.
    If the treatment RewardTerm also runs in that reward stage, ``env.common_step_counter`` makes
    accounting idempotent.
    """

    cmd, raw, eligible, reward = _base_decel_values(
        env, command_name, v_gain, v_max, std
    )
    _record_base_decel_activation(
        env,
        cmd,
        raw,
        eligible,
        reward,
        signature=(float(v_gain), float(v_max), float(std)),
    )


def base_decel_activation_probe(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_gain: float = 2.0,
    v_max: float = 1.6,
    std: float = 0.4,
) -> torch.Tensor:
    """Reward-stage activation probe whose weighted reward contribution is identically zero.

    The experiment translator gives this term a nonzero manager weight in *both* control and
    treatment so IsaacLab cannot optimize it away.  Returning an environment-sized zero tensor
    means even a weight of ``1.0`` changes neither per-term reward nor total reward.  The treatment's
    later :func:`base_decel_tracking` call sees the same state and deduplicates on the shared step
    token.
    """

    observe_base_decel_activation(env, command_name, v_gain, v_max, std)
    cmd = _cmd(env, command_name)
    return torch.zeros_like(cmd.racket_target_pos_w[:, 0])


def consume_base_decel_activation_counters(
    env: ManagerBasedRLEnv, command_name: str
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's base-decel activation device scalars.

    The last simulator-step token intentionally survives the reset.  A logger that consumes after
    rollout and then observes the same step cannot accidentally charge that step to the next PPO
    update.  A second consume therefore returns exact scalar zeros.
    """

    cmd = _cmd(env, command_name)
    state = _base_decel_counter_state(cmd, cmd.racket_target_pos_w)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    # RewardManager may lazily create these counters while Isaac is stepping under
    # ``torch.inference_mode()``.  PyTorch deliberately refuses to mutate such an inference
    # tensor from the logger's normal-mode context, even though cloning it for the snapshot is
    # allowed.  Reset in inference mode as well; this changes only the private accounting
    # scalars and preserves their device/dtype and the last-step deduplication token.
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


def base_decel_tracking(
    env: ManagerBasedRLEnv, command_name: str, v_gain: float = 2.0, v_max: float = 1.6, std: float = 0.4
) -> torch.Tensor:
    """P2.4 PACE-style smooth-deceleration shaping: track a pseudo base-velocity command that decays
    with the remaining planar racket->target error (G08: the robot rushes far targets reactively, with
    no deceleration profile, and arrives too hot to strike).

    PACE's remedy is a velocity command proportional to the remaining position error, so the DESIRED
    speed goes to ~0 exactly at arrival. Deploy-parity constraint: the 175-D actor obs contract is
    FROZEN, so this is a REWARD-side term only — nothing new is observed; the kernel reuses the task's
    own error measure (the planar racket->target distance, frame-invariant, no world base position):

        v_des = clamp(v_gain * ||(racket_target_xy - racket_xy)||, 0, v_max)
        reward = exp(-(||v_base_xy|| - v_des)^2 / std^2)

    Far target -> v_des saturates at v_max and the term pays for MOVING (it cooperates with
    racket_progress instead of taxing the approach); as the strike stance is reached v_des -> 0 and the
    term pays for a CALM base — a smooth taper instead of the bang-bang rush-then-slam. Gated to
    active ``pre_strike`` motion but explicitly OFF during the frozen pre-swing hold: ``hold_ready``
    owns hold stillness, while paying ``base_decel``'s nonzero target speed there asks the base to move
    and creates a contradictory objective. The strike swing and post-strike recovery are untouched
    (post-strike the distance to the OLD swung-through target would otherwise command a bogus speed-up). Base velocity
    is the WORLD planar root velocity (same source as hold_ready); v_gain [1/s] is the P-gain of the
    pseudo velocity command, v_max [m/s] its cap, std [m/s] the kernel width. RewTerm weight is
    POSITIVE; default weight 0.0 = OFF (flag-gated via task.rewards.base_decel_weight)."""
    cmd, raw, eligible, reward = _base_decel_values(
        env, command_name, v_gain, v_max, std
    )
    _record_base_decel_activation(
        env,
        cmd,
        raw,
        eligible,
        reward,
        signature=(float(v_gain), float(v_max), float(std)),
    )
    return reward


_QDOT_ACTIVATION_ATTR = "_hope_qdot_hinge_activation_counters"
_QDOT_OBSERVED_STEP_ATTR = "_hope_qdot_hinge_observed_step"
_QDOT_ACTIVE_STEP_ATTR = "_hope_qdot_hinge_active_step"
_QDOT_OBSERVED_COUNT = "observed_sample_count"
_QDOT_ACTIVE_COUNT = "hinge_active_sample_count"
_QDOT_EXCESS_COUNT = "excess_sample_count"
_QDOT_EXCESS_SQUARE_SUM = "normalized_excess_square_sum"


def _joint_velocity_limit_hinge_values(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin: float = 0.85,
    expected_joint_count: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env hinge value and the per-env above-margin mask.

    The first returned tensor is non-negative; the :class:`RewardTermCfg` weight must therefore
    be non-positive.  Unlike ``action_rate_l2``, this term measures the robot's *realized* joint
    velocity against the actual articulation limits, in the exact runtime articulation order::

        sum_j(relu(abs(qd_j) / qd_limit_j - margin) ** 2)

    SUM 聚合(Franco 2026-07-25 裁定,安全家族统一口径,同 qbar 的理由:mean 是稀释器,
    单关节尖叫被 ÷31 摊平——安全项恰恰要它最响)。违规几个关节就罚几份;历史 mean 剂量
    (-5.0/-2.5/-1.0)换算到 SUM 语义须 ÷31(单关节同深度违规同罚时等价),复活旧臂前重调。

    This source gate is deliberately A3-specific: all 31 runtime joints must be selected in
    identity order.  Missing/reordered joints and zero/non-finite limits fail closed rather than
    silently changing the denominator.  The limit tensor is read on every call: Isaac may update
    articulation limits at runtime, so caching the first value would cease to measure the actual
    plant.  CUDA validity assertions stay asynchronous rather than forcing a host sync per step.
    """
    if type(expected_joint_count) is not int or expected_joint_count != 31:
        raise ValueError(
            "joint_velocity_limit_hinge requires the exact 31-joint A3 runtime order"
        )
    if isinstance(margin, bool):
        raise ValueError("joint_velocity_limit_hinge margin must be finite and in (0, 1)")
    margin = float(margin)
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("joint_velocity_limit_hinge margin must be finite and in (0, 1)")

    asset = env.scene[asset_cfg.name]
    data = asset.data
    joint_names = list(getattr(data, "joint_names", getattr(asset, "joint_names", ())))
    if (
        len(joint_names) != expected_joint_count
        or any(not str(name) for name in joint_names)
        or len(set(joint_names)) != expected_joint_count
    ):
        raise RuntimeError(
            "joint_velocity_limit_hinge requires exactly 31 unique runtime joint names"
        )

    raw_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if isinstance(raw_ids, slice):
        joint_ids = list(range(expected_joint_count))[raw_ids]
    else:
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        joint_ids = [int(value) for value in raw_ids]
    if joint_ids != list(range(expected_joint_count)):
        raise RuntimeError(
            "joint_velocity_limit_hinge requires identity 31-joint articulation order, "
            f"got joint_ids={joint_ids}"
        )

    joint_vel = data.joint_vel
    if joint_vel.ndim != 2 or tuple(joint_vel.shape)[1] != expected_joint_count:
        raise RuntimeError(
            "joint_velocity_limit_hinge requires joint_vel shaped [num_envs, 31], "
            f"got {tuple(joint_vel.shape)}"
        )

    runtime_limits = data.joint_vel_limits
    if runtime_limits.ndim == 1:
        if tuple(runtime_limits.shape) != (expected_joint_count,):
            raise RuntimeError(
                "joint_velocity_limit_hinge requires 31 articulation velocity limits, "
                f"got {tuple(runtime_limits.shape)}"
            )
        limits = runtime_limits
        limits_valid = torch.all(torch.isfinite(limits) & (limits > 0.0))
    elif runtime_limits.ndim == 2:
        if (
            tuple(runtime_limits.shape)[0] < 1
            or tuple(runtime_limits.shape)[1] != expected_joint_count
            or tuple(runtime_limits.shape)[0] not in (1, tuple(joint_vel.shape)[0])
        ):
            raise RuntimeError(
                "joint_velocity_limit_hinge requires joint_vel_limits shaped [31] or "
                f"[num_envs, 31], got {tuple(runtime_limits.shape)}"
            )
        limits = runtime_limits
        limits_valid = torch.all(torch.isfinite(limits) & (limits > 0.0))
        if tuple(runtime_limits.shape)[0] > 1:
            limits_valid = limits_valid & torch.all(
                runtime_limits == runtime_limits[0].unsqueeze(0)
            )
    else:
        raise RuntimeError(
            "joint_velocity_limit_hinge requires joint_vel_limits shaped [31] or "
            f"[num_envs, 31], got {tuple(runtime_limits.shape)}"
        )

    # CPU tests and CPU VecEnv paths get the precise diagnostic immediately.  On CUDA,
    # torch._assert_async fails the training stream without a per-step host synchronization.
    if limits_valid.device.type == "cpu":
        if not bool(limits_valid):
            raise RuntimeError(
                "joint_velocity_limit_hinge requires finite, positive, identical "
                "articulation velocity limits in every environment"
            )
    else:
        torch._assert_async(limits_valid)

    normalized_excess = torch.relu(torch.abs(joint_vel) / limits - margin)
    # SUM 不平均(2026-07-25):违规几个关节就罚几份,单关节尖叫不被 ÷31 稀释。
    squared_sum = torch.sum(torch.square(normalized_excess), dim=-1)
    return squared_sum, torch.any(normalized_excess > 0.0, dim=-1)


def _qdot_activation_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _QDOT_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _QDOT_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDOT_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDOT_EXCESS_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDOT_EXCESS_SQUARE_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
        }
        setattr(env, _QDOT_ACTIVATION_ATTR, state)
    return state


def _record_joint_velocity_limit_hinge_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    excess_mask: torch.Tensor,
    *,
    hinge_active: bool,
) -> None:
    """Book one simulator step, deduplicating probe and real RewardTerm calls."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _qdot_activation_counter_state(env, values)
    if token is None or getattr(env, _QDOT_OBSERVED_STEP_ATTR, None) != token:
        state[_QDOT_OBSERVED_COUNT].add_(values.numel())
        state[_QDOT_EXCESS_COUNT].add_(
            excess_mask.detach().sum(dtype=torch.long)
        )
        state[_QDOT_EXCESS_SQUARE_SUM].add_(values.detach().sum())
        if token is not None:
            setattr(env, _QDOT_OBSERVED_STEP_ATTR, token)
    if hinge_active and (
        token is None or getattr(env, _QDOT_ACTIVE_STEP_ATTR, None) != token
    ):
        state[_QDOT_ACTIVE_COUNT].add_(values.numel())
        if token is not None:
            setattr(env, _QDOT_ACTIVE_STEP_ATTR, token)


def joint_velocity_limit_hinge_probe(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin: float = 0.85,
    expected_joint_count: int = 31,
) -> torch.Tensor:
    """Observe realized qdot without changing the scalar reward.

    A nonzero manager weight is safe because this function returns exact zeros.  The probe and the
    actual hinge share the validation/math helper and a simulator-step token, so treatment samples
    are observed once while hinge-active samples are booked separately.
    """

    values, excess_mask = _joint_velocity_limit_hinge_values(
        env, asset_cfg, margin, expected_joint_count
    )
    _record_joint_velocity_limit_hinge_activation(
        env, values, excess_mask, hinge_active=False
    )
    return torch.zeros_like(values)


def joint_velocity_limit_hinge(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin: float = 0.85,
    expected_joint_count: int = 31,
) -> torch.Tensor:
    """Penalize and attest the normalized joint-speed tail near articulation limits."""

    values, excess_mask = _joint_velocity_limit_hinge_values(
        env, asset_cfg, margin, expected_joint_count
    )
    _record_joint_velocity_limit_hinge_activation(
        env, values, excess_mask, hinge_active=True
    )
    return values


def consume_joint_velocity_limit_hinge_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's qdot observer/activation ledger."""

    template = env.scene["robot"].data.joint_vel
    state = _qdot_activation_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


# The deploy-space recovery smoother is deliberately narrower than action_rate_l2: it observes the
# affine-transformed, q_des-clamped target, only on the three waist + twelve leg joints, and only in
# the same swing's post-contact recovery window.  Keep the semantic joint set explicit so a renamed,
# missing, or accidentally arm-inclusive action contract fails closed instead of changing the mean.
_PROCESSED_QDES_RECOVERY_JOINT_NAMES = frozenset(
    {
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    }
)
_PROCESSED_QDES_CONTROL_DT_S = 0.02
_PROCESSED_QDES_ACTIVATION_ATTR = "_hope_processed_qdes_slew_activation_counters"
_PROCESSED_QDES_OBSERVED_STEP_ATTR = "_hope_processed_qdes_slew_observed_step"
_PROCESSED_QDES_ACTIVE_STEP_ATTR = "_hope_processed_qdes_slew_active_step"
_PROCESSED_QDES_SIGNATURE_ATTR = "_hope_processed_qdes_slew_signature"
_PROCESSED_QDES_OBSERVED_COUNT = "observed_sample_count"
_PROCESSED_QDES_VALID_COUNT = "previous_qdes_valid_sample_count"
_PROCESSED_QDES_INVALID_COUNT = "previous_qdes_invalid_first_step_sample_count"
_PROCESSED_QDES_ELIGIBLE_COUNT = "recovery_eligible_sample_count"
_PROCESSED_QDES_ACTIVE_COUNT = "reward_enabled_eligible_sample_count"
_PROCESSED_QDES_TAIL_ACTIVE_COUNT = "tail_active_sample_count"
_PROCESSED_QDES_EXCESS_JOINT_COUNT = "above_margin_joint_count"
_PROCESSED_QDES_TAIL_SUM = "gated_tail_value_sum"


def _processed_qdes_slew_hinge_values(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "racket_target",
    margin: float = 0.85,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gated tail value, eligibility, history validity, and per-joint excess.

    For each selected joint, ``u = abs(q_des[t] - q_des[t-1]) / (qdot_limit * 0.02)`` and::

        tail = 1 - exp(-square(relu(u - margin) / (1 - margin)))

    The environment reward is the SUM over the exact 15-joint waist/leg set(Franco
    2026-07-25 裁定,安全家族统一 SUM 口径:mean 把单关节猛拧 ÷15 稀释;违规几个关节
    就罚几份,单关节 tail 有界 →1。历史 mean 剂量(-0.25/-1.0)换算 SUM 语义须 ÷15,
    复活旧臂前重调)。Reset-invalid
    history and samples outside the same-attempt recovery window return exact zero.
    """

    if action_name != "joint_pos":
        raise ValueError("processed_qdes_slew_hinge action_name must be exactly 'joint_pos'")
    if command_name != "racket_target":
        raise ValueError(
            "processed_qdes_slew_hinge command_name must be exactly 'racket_target'"
        )
    if isinstance(margin, bool):
        raise ValueError("processed_qdes_slew_hinge margin must be finite and in (0, 1)")
    margin = float(margin)
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("processed_qdes_slew_hinge margin must be finite and in (0, 1)")
    if isinstance(recovery_start_s, bool) or isinstance(recovery_end_s, bool):
        raise ValueError(
            "processed_qdes_slew_hinge recovery window must be finite and satisfy 0 <= start < end"
        )
    recovery_start_s = float(recovery_start_s)
    recovery_end_s = float(recovery_end_s)
    if (
        not math.isfinite(recovery_start_s)
        or not math.isfinite(recovery_end_s)
        or recovery_start_s < 0.0
        or recovery_start_s >= recovery_end_s
    ):
        raise ValueError(
            "processed_qdes_slew_hinge recovery window must be finite and satisfy 0 <= start < end"
        )
    raw_control_dt = getattr(env, "step_dt", None)
    if (
        isinstance(raw_control_dt, bool)
        or not isinstance(raw_control_dt, (int, float))
        or not math.isfinite(float(raw_control_dt))
        or not math.isclose(
            float(raw_control_dt),
            _PROCESSED_QDES_CONTROL_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires env.step_dt == 0.02 s control time"
        )

    action = env.action_manager.get_term(action_name)
    processed = getattr(action, "processed_actions", None)
    previous = getattr(action, "previous_processed_qdes", None)
    previous_valid = getattr(action, "previous_processed_qdes_valid", None)
    if (
        not torch.is_tensor(processed)
        or not torch.is_tensor(previous)
        or processed.ndim != 2
        or previous.shape != processed.shape
        or not torch.is_tensor(previous_valid)
        or previous_valid.dtype != torch.bool
        or tuple(previous_valid.shape) != (tuple(processed.shape)[0],)
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires reset-aware processed q_des history from joint_pos"
        )

    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    runtime_names = list(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    action_names = list(getattr(action, "_joint_names", ()))
    joint_count = len(runtime_names)
    if (
        joint_count != 31
        or len(set(runtime_names)) != joint_count
        or action_names != runtime_names
        or tuple(processed.shape)[1] != joint_count
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires identity 31-joint A3 action/articulation order"
        )
    raw_joint_ids = getattr(action, "_joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        action_joint_ids = list(range(joint_count))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        action_joint_ids = [int(value) for value in raw_joint_ids]
    if action_joint_ids != list(range(joint_count)):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires identity 31-joint A3 action/articulation order"
        )
    selected = [
        index
        for index, name in enumerate(runtime_names)
        if name in _PROCESSED_QDES_RECOVERY_JOINT_NAMES
    ]
    if (
        len(selected) != 15
        or {runtime_names[index] for index in selected}
        != _PROCESSED_QDES_RECOVERY_JOINT_NAMES
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires the exact 15 A3 waist/leg joints"
        )

    runtime_limits = getattr(data, "joint_vel_limits", None)
    if not torch.is_tensor(runtime_limits):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires runtime articulation joint velocity limits"
        )
    if runtime_limits.ndim == 1 and tuple(runtime_limits.shape) == (joint_count,):
        selected_limits = runtime_limits[selected]
    elif (
        runtime_limits.ndim == 2
        and tuple(runtime_limits.shape)[1] == joint_count
        and tuple(runtime_limits.shape)[0] in (1, tuple(processed.shape)[0])
    ):
        selected_limits = runtime_limits[:, selected]
    else:
        raise RuntimeError(
            "processed_qdes_slew_hinge requires joint_vel_limits shaped [31], [1,31], or [num_envs,31]"
        )
    limits_valid = torch.all(torch.isfinite(selected_limits) & selected_limits.gt(0.0))
    if limits_valid.device.type == "cpu":
        if not bool(limits_valid):
            raise RuntimeError(
                "processed_qdes_slew_hinge requires finite positive waist/leg velocity limits"
            )
    else:
        torch._assert_async(limits_valid)

    current_selected = processed[:, selected]
    previous_selected = previous[:, selected]
    normalized = torch.abs(current_selected - previous_selected) / (
        selected_limits * _PROCESSED_QDES_CONTROL_DT_S
    )
    normalized_excess = torch.relu(normalized - margin)
    per_joint_tail = 1.0 - torch.exp(
        -torch.square(normalized_excess / (1.0 - margin))
    )
    # SUM 不平均(2026-07-25):违规几个关节就罚几份;单关节 tail 有界,总值 ∈ [0, 15)。
    raw_value = torch.sum(per_joint_tail, dim=-1)

    cmd = _cmd(env, command_name)
    clock = getattr(cmd, "post_strike_age_and_same_attempt", None)
    if not callable(clock):
        raise RuntimeError(
            "processed_qdes_slew_hinge requires racket_target's same-attempt post-strike clock"
        )
    age_s, same_attempt = clock()
    if (
        not torch.is_tensor(age_s)
        or not torch.is_tensor(same_attempt)
        or tuple(age_s.shape) != tuple(previous_valid.shape)
        or tuple(same_attempt.shape) != tuple(previous_valid.shape)
        or same_attempt.dtype != torch.bool
    ):
        raise RuntimeError(
            "processed_qdes_slew_hinge received an invalid same-attempt post-strike clock"
        )
    in_recovery = (
        same_attempt
        & age_s.ge(recovery_start_s)
        & age_s.le(recovery_end_s)
    )
    eligible = previous_valid & in_recovery
    value = torch.where(eligible, raw_value, torch.zeros_like(raw_value))
    excess = eligible.unsqueeze(-1) & normalized_excess.gt(0.0)
    return value, eligible, previous_valid, excess


def _processed_qdes_slew_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _PROCESSED_QDES_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _PROCESSED_QDES_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_VALID_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_INVALID_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_ELIGIBLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_TAIL_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_EXCESS_JOINT_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _PROCESSED_QDES_TAIL_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
        }
        setattr(env, _PROCESSED_QDES_ACTIVATION_ATTR, state)
    return state


def _record_processed_qdes_slew_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    previous_valid: torch.Tensor,
    excess: torch.Tensor,
    *,
    hinge_active: bool,
    signature: tuple[str, str, float, float, float],
) -> None:
    """Book one simulator step, idempotently sharing the probe and real term."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _processed_qdes_slew_counter_state(env, values)
    already_observed = (
        token is not None
        and getattr(env, _PROCESSED_QDES_OBSERVED_STEP_ATTR, None) == token
    )
    if already_observed:
        if getattr(env, _PROCESSED_QDES_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "processed q_des slew probe and RewardTerm used different parameters in one step"
            )
    else:
        state[_PROCESSED_QDES_OBSERVED_COUNT].add_(values.numel())
        state[_PROCESSED_QDES_VALID_COUNT].add_(
            previous_valid.detach().sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_INVALID_COUNT].add_(
            (~previous_valid.detach()).sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_ELIGIBLE_COUNT].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_TAIL_ACTIVE_COUNT].add_(
            torch.any(excess.detach(), dim=-1).sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_EXCESS_JOINT_COUNT].add_(
            excess.detach().sum(dtype=torch.long)
        )
        state[_PROCESSED_QDES_TAIL_SUM].add_(values.detach().sum())
        if token is not None:
            setattr(env, _PROCESSED_QDES_OBSERVED_STEP_ATTR, token)
            setattr(env, _PROCESSED_QDES_SIGNATURE_ATTR, signature)
    if hinge_active and (
        token is None or getattr(env, _PROCESSED_QDES_ACTIVE_STEP_ATTR, None) != token
    ):
        state[_PROCESSED_QDES_ACTIVE_COUNT].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _PROCESSED_QDES_ACTIVE_STEP_ATTR, token)


def processed_qdes_slew_hinge_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "racket_target",
    margin: float = 0.85,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Measure processed-q_des recovery slew while contributing exact zero reward."""

    values, eligible, previous_valid, excess = _processed_qdes_slew_hinge_values(
        env, action_name, command_name, margin, recovery_start_s, recovery_end_s
    )
    _record_processed_qdes_slew_activation(
        env,
        values,
        eligible,
        previous_valid,
        excess,
        hinge_active=False,
        signature=(
            action_name,
            command_name,
            float(margin),
            float(recovery_start_s),
            float(recovery_end_s),
        ),
    )
    return torch.zeros_like(values)


def processed_qdes_slew_hinge(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "racket_target",
    margin: float = 0.85,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Penalize deploy-space waist/leg q_des slew only during same-swing recovery."""

    values, eligible, previous_valid, excess = _processed_qdes_slew_hinge_values(
        env, action_name, command_name, margin, recovery_start_s, recovery_end_s
    )
    _record_processed_qdes_slew_activation(
        env,
        values,
        eligible,
        previous_valid,
        excess,
        hinge_active=True,
        signature=(
            action_name,
            command_name,
            float(margin),
            float(recovery_start_s),
            float(recovery_end_s),
        ),
    )
    return values


def consume_processed_qdes_slew_hinge_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's processed-q_des recovery ledger."""

    action = env.action_manager.get_term("joint_pos")
    template = action.processed_actions[:, 0]
    state = _processed_qdes_slew_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


# Wave-Q qdes limit barrier ------------------------------------------------------------------- #
#
# All-joint deploy-space q_des position-limit barrier.  Unlike processed_qdes_slew_hinge (a rate
# hinge on a 15-joint subset inside the recovery window), this term charges every one of the 31
# joints, every control step, as soon as the commanded target enters the margin band next to a
# position limit.  人话:哪个关节的目标角贴到限位边上就罚哪个,全身 31 个关节全程盯着,
# 不挑"最狠的几个"。
_QDES_LIMIT_BARRIER_JOINT_COUNT = 31
# 站姿豁免(2026-07-25)的两个归一化常量:有效罚带收窄到"默认站姿减 EPS 呼吸间隙"处,
# 收窄后的带宽若连 FLOOR 都不到(站姿本身贴死软限位)= 建模错误,fail loud。
_QDES_LIMIT_BARRIER_STANCE_EPS = 0.005
_QDES_LIMIT_BARRIER_MARGIN_FLOOR = 0.005
_QDES_LIMIT_BARRIER_ACTIVATION_ATTR = "_hope_qdes_limit_barrier_activation_counters"
_QDES_LIMIT_BARRIER_OBSERVED_STEP_ATTR = "_hope_qdes_limit_barrier_observed_step"
_QDES_LIMIT_BARRIER_ACTIVE_STEP_ATTR = "_hope_qdes_limit_barrier_active_step"
_QDES_LIMIT_BARRIER_SIGNATURE_ATTR = "_hope_qdes_limit_barrier_signature"
_QDES_LIMIT_BARRIER_OBSERVED_COUNT = "observed_sample_count"
_QDES_LIMIT_BARRIER_ABOVE_MARGIN_JOINT_COUNT = "above_margin_joint_count"
_QDES_LIMIT_BARRIER_ABOVE_MARGIN_SAMPLE_COUNT = "above_margin_sample_count"
_QDES_LIMIT_BARRIER_MAX_INTRUSION = "max_intrusion_depth_frac"
_QDES_LIMIT_BARRIER_VALUE_SUM = "barrier_value_sum"
_QDES_LIMIT_BARRIER_ACTIVE_COUNT = "reward_enabled_sample_count"


def _qdes_limit_barrier_values(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-env barrier value and the per-joint intrusion depth.

    For EVERY joint j of the 31-joint A3 articulation (no top-k, no subset)::

        d_j = min(q_des_j - lo_j, hi_j - q_des_j) / (hi_j - lo_j)     # normalized limit distance
        m_j = min(margin_frac, d_default_j - STANCE_EPS)               # stance-exempt margin (07-25)
        t_j = relu(m_j - d_j) / m_j                                    # intrusion depth in [0, 1+]
        value = sum_j(1 - exp(-t_j^2))                                 # per-joint bounded tails, summed

    SUM aggregation (Franco 2026-07-21 裁定): mean 是稀释器——单关节违规被 ÷31,top-k 的
    "挑几个狠罚"其实就是在补这个稀释;去 top-k 的正当性来自 sum:违规几个关节就罚几份,
    每个关节的 tail 有界 (t→∞ 时 →1),满违规单关节 tail≈1,权重 -0.65 下每步约 -0.65×dt.
    Dense: charged on every control step in every phase (no strike/recovery gate) — 对标 Jiayi
    v14 的全程 barrier.  q_des is the SAME processed (affine + clamp) target the PD controller
    receives, and lo/hi are the SAME soft_joint_pos_limits the deploy-parity clamp uses.

    站姿豁免 m_j(2026-07-25):肩 roll 这类硬限位不对称的关节(内收硬停只有 -5°、外展
    +150°),0.9 软限位系数与 0.08 罚带都按全跨度等比缩放,会把设计站姿直接圈进罚带——
    实测双肩 roll 在 ready 外展 0.12 rad 处 d=0.0296<0.08:常驻罚 0.656/步、梯度持续把
    双肩往外推着跟模仿拔河、above_margin_joint_count 垫 2 的地板。收窄到站姿之外后:
    站姿零罚零梯度,从站姿再向限位靠近照罚;d_default >= margin_frac + EPS 的关节
    m_j == margin_frac,数学逐字节不变(31 关节里 29 个)。站姿距软限位近到连
    MARGIN_FLOOR 的带宽都留不出 = 建模错误,fail loud 而不是静默豁免。
    """

    if action_name != "joint_pos":
        raise ValueError("qdes_limit_barrier action_name must be exactly 'joint_pos'")
    if isinstance(margin_frac, bool):
        raise ValueError(
            "qdes_limit_barrier margin_frac must be finite and in (0, 0.5)"
        )
    margin_frac = float(margin_frac)
    if not math.isfinite(margin_frac) or not 0.0 < margin_frac < 0.5:
        raise ValueError(
            "qdes_limit_barrier margin_frac must be finite and in (0, 0.5)"
        )

    action = env.action_manager.get_term(action_name)
    processed = getattr(action, "processed_actions", None)
    if not torch.is_tensor(processed) or processed.ndim != 2:
        raise RuntimeError(
            "qdes_limit_barrier requires processed q_des targets from joint_pos"
        )

    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    runtime_names = list(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    action_names = list(getattr(action, "_joint_names", ()))
    joint_count = len(runtime_names)
    if (
        joint_count != _QDES_LIMIT_BARRIER_JOINT_COUNT
        or len(set(runtime_names)) != joint_count
        or action_names != runtime_names
        or tuple(processed.shape)[1] != joint_count
    ):
        raise RuntimeError(
            "qdes_limit_barrier requires identity 31-joint A3 action/articulation order"
        )
    raw_joint_ids = getattr(action, "_joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        action_joint_ids = list(range(joint_count))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        action_joint_ids = [int(value) for value in raw_joint_ids]
    if action_joint_ids != list(range(joint_count)):
        raise RuntimeError(
            "qdes_limit_barrier requires identity 31-joint A3 action/articulation order"
        )

    limits = getattr(data, "soft_joint_pos_limits", None)
    if not torch.is_tensor(limits):
        raise RuntimeError(
            "qdes_limit_barrier requires runtime articulation soft joint position limits"
        )
    if limits.ndim == 2 and tuple(limits.shape) == (joint_count, 2):
        lower = limits[:, 0]
        upper = limits[:, 1]
    elif (
        limits.ndim == 3
        and tuple(limits.shape)[1] == joint_count
        and tuple(limits.shape)[2] == 2
        and tuple(limits.shape)[0] in (1, tuple(processed.shape)[0])
    ):
        lower = limits[:, :, 0]
        upper = limits[:, :, 1]
    else:
        raise RuntimeError(
            "qdes_limit_barrier requires soft_joint_pos_limits shaped [31,2], [1,31,2], or [num_envs,31,2]"
        )
    span = upper - lower
    limits_valid = torch.all(
        torch.isfinite(lower) & torch.isfinite(upper) & span.gt(0.0)
    )
    if limits_valid.device.type == "cpu":
        if not bool(limits_valid):
            raise RuntimeError(
                "qdes_limit_barrier requires finite joint position limits with lo < hi"
            )
    else:
        torch._assert_async(limits_valid)

    default_q = getattr(data, "default_joint_pos", None)
    if not torch.is_tensor(default_q):
        raise RuntimeError(
            "qdes_limit_barrier requires articulation default_joint_pos "
            "(the stance-exempt margin needs the designed stance)"
        )
    d_default = torch.minimum(default_q - lower, upper - default_q) / span
    margin_eff = torch.clamp(
        d_default - _QDES_LIMIT_BARRIER_STANCE_EPS, max=margin_frac
    )
    stance_valid = torch.all(margin_eff.gt(_QDES_LIMIT_BARRIER_MARGIN_FLOOR))
    if stance_valid.device.type == "cpu":
        if not bool(stance_valid):
            raise RuntimeError(
                "qdes_limit_barrier: a default-stance joint sits within EPS+FLOOR of its "
                "soft limit — fix the limits or the stance instead of silencing the barrier"
            )
    else:
        torch._assert_async(stance_valid)

    distance = torch.minimum(processed - lower, upper - processed) / span
    intrusion = torch.relu(margin_eff - distance) / margin_eff
    value = torch.sum(1.0 - torch.exp(-torch.square(intrusion)), dim=-1)
    return value, intrusion


def _qdes_limit_barrier_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _QDES_LIMIT_BARRIER_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _QDES_LIMIT_BARRIER_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_LIMIT_BARRIER_ABOVE_MARGIN_JOINT_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_LIMIT_BARRIER_ABOVE_MARGIN_SAMPLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_LIMIT_BARRIER_MAX_INTRUSION: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _QDES_LIMIT_BARRIER_VALUE_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _QDES_LIMIT_BARRIER_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
        }
        setattr(env, _QDES_LIMIT_BARRIER_ACTIVATION_ATTR, state)
    return state


def _record_qdes_limit_barrier_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    intrusion: torch.Tensor,
    *,
    barrier_active: bool,
    signature: tuple[str, float],
) -> None:
    """Book one simulator step, idempotently sharing the probe and real term."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _qdes_limit_barrier_counter_state(env, values)
    already_observed = (
        token is not None
        and getattr(env, _QDES_LIMIT_BARRIER_OBSERVED_STEP_ATTR, None) == token
    )
    if already_observed:
        if getattr(env, _QDES_LIMIT_BARRIER_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "qdes limit barrier probe and RewardTerm used different parameters in one step"
            )
    else:
        above = intrusion.detach().gt(0.0)
        state[_QDES_LIMIT_BARRIER_OBSERVED_COUNT].add_(values.numel())
        state[_QDES_LIMIT_BARRIER_ABOVE_MARGIN_JOINT_COUNT].add_(
            above.sum(dtype=torch.long)
        )
        state[_QDES_LIMIT_BARRIER_ABOVE_MARGIN_SAMPLE_COUNT].add_(
            torch.any(above, dim=-1).sum(dtype=torch.long)
        )
        state[_QDES_LIMIT_BARRIER_MAX_INTRUSION].copy_(
            torch.maximum(
                state[_QDES_LIMIT_BARRIER_MAX_INTRUSION], intrusion.detach().max()
            )
        )
        state[_QDES_LIMIT_BARRIER_VALUE_SUM].add_(values.detach().sum())
        if token is not None:
            setattr(env, _QDES_LIMIT_BARRIER_OBSERVED_STEP_ATTR, token)
            setattr(env, _QDES_LIMIT_BARRIER_SIGNATURE_ATTR, signature)
    if barrier_active and (
        token is None
        or getattr(env, _QDES_LIMIT_BARRIER_ACTIVE_STEP_ATTR, None) != token
    ):
        state[_QDES_LIMIT_BARRIER_ACTIVE_COUNT].add_(values.numel())
        if token is not None:
            setattr(env, _QDES_LIMIT_BARRIER_ACTIVE_STEP_ATTR, token)


def qdes_limit_barrier_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
) -> torch.Tensor:
    """Measure the q_des limit barrier while contributing exact zero reward.

    Same design provenance as :func:`qdes_limit_barrier` — Jiayi V14 全关节 top-k qdes barrier
    思想,去 top-k 重做(Franco 指示),-0.65/margin 0.08 取自 v14 起点.  人话:零权重探针,
    只记账(贴限位的关节数、最大侵入深度),不改任何奖励。
    """

    values, intrusion = _qdes_limit_barrier_values(env, action_name, margin_frac)
    _record_qdes_limit_barrier_activation(
        env,
        values,
        intrusion,
        barrier_active=False,
        signature=(action_name, float(margin_frac)),
    )
    return torch.zeros_like(values)


def qdes_limit_barrier(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
) -> torch.Tensor:
    """Penalize processed q_des targets inside the margin band next to a position limit.

    Jiayi V14 全关节 top-k qdes barrier 思想,去 top-k 重做(Franco 指示):全部 31 个关节直接罚,
    不做 top-k、不做子集;-0.65/margin 0.08 取自 v14 起点.  Per-joint bounded tails are SUMMED,
    not averaged (Franco: mean 把单关节违规 ÷31 稀释掉;sum 让违规几个关节就罚几份,这也是
    去 top-k 的正当性所在;满违规单关节 tail≈1 → 每步约 -0.65×dt).  Dense — charged on every
    control step in every phase, matching V14's whole-episode barrier.  人话:目标角贴到限位
    边缘就开始扣钱,越贴越狠,几个关节贴就扣几份,全身关节一视同仁,全程有效。
    2026-07-25 站姿豁免:设计站姿本身贴限的关节(双肩 roll)罚带收窄到站姿之外——站着
    不动零罚,往限位再靠才扣钱;详见 _qdes_limit_barrier_values。
    """

    values, intrusion = _qdes_limit_barrier_values(env, action_name, margin_frac)
    _record_qdes_limit_barrier_activation(
        env,
        values,
        intrusion,
        barrier_active=True,
        signature=(action_name, float(margin_frac)),
    )
    return values


# ActionBall soft-limit barrier v2 ------------------------------------------------------------- #
#
# V1 above has a zero-valued quadratic tail at the margin edge.  That is a poor safety objective:
# a policy can keep one or more joints an arbitrarily small distance inside the forbidden margin
# while paying an arbitrarily small price.  Fresh ActionBall uses these explicitly versioned terms
# instead.  Historical task IDs retain v1 unless their reward config opts into the v2 callables.
_SOFT_LIMIT_BARRIER_V2_SCHEMA_VERSION = 2
_SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR = 0.25
_SOFT_LIMIT_BARRIER_V2_SHAPE_RATE = 4.0
_QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR = "_hope_qdes_limit_barrier_v2_activation_counters"
_QDES_LIMIT_BARRIER_V2_OBSERVED_STEP_ATTR = "_hope_qdes_limit_barrier_v2_observed_step"
_QDES_LIMIT_BARRIER_V2_ACTIVE_STEP_ATTR = "_hope_qdes_limit_barrier_v2_active_step"
_QDES_LIMIT_BARRIER_V2_SIGNATURE_ATTR = "_hope_qdes_limit_barrier_v2_signature"
_ACTUAL_LIMIT_BARRIER_V2_ACTIVATION_ATTR = "_hope_actual_limit_barrier_v2_activation_counters"
_ACTUAL_LIMIT_BARRIER_V2_OBSERVED_STEP_ATTR = "_hope_actual_limit_barrier_v2_observed_step"
_ACTUAL_LIMIT_BARRIER_V2_ACTIVE_STEP_ATTR = "_hope_actual_limit_barrier_v2_active_step"
_ACTUAL_LIMIT_BARRIER_V2_SIGNATURE_ATTR = "_hope_actual_limit_barrier_v2_signature"
_SOFT_LIMIT_BARRIER_V2_OBSERVED_COUNT = "observed_sample_count"
_SOFT_LIMIT_BARRIER_V2_INTRUSION_JOINT_COUNT = "intrusion_joint_count"
_SOFT_LIMIT_BARRIER_V2_INTRUSION_SAMPLE_COUNT = "intrusion_sample_count"
_SOFT_LIMIT_BARRIER_V2_MAX_INTRUSION = "max_intrusion_depth_frac"
_SOFT_LIMIT_BARRIER_V2_VALUE_SUM = "barrier_value_sum"
_SOFT_LIMIT_BARRIER_V2_ACTIVE_COUNT = "reward_enabled_sample_count"

# ActionBall finite-q_des projection ---------------------------------------------------------- #
#
# PPO still samples and stores its ordinary unbounded Gaussian action.  The action term maps the
# resulting affine q_des to the nearest point in the already-validated drive target envelope.
# This separate term charges the distance between those two points; it must not be folded into the
# processed-q_des soft barrier above because a plant-state brake can replace ``processed_actions``
# after the nominal projection has been computed.
_QDES_PROJECTION_SCHEMA_VERSION = 1
_QDES_PROJECTION_DEFAULT_SHAPE_RATE = 4.0
_QDES_PROJECTION_ACTIVATION_ATTR = "_hope_qdes_projection_activation_counters"
_QDES_PROJECTION_OBSERVED_STEP_ATTR = "_hope_qdes_projection_observed_step"
_QDES_PROJECTION_SIGNATURE_ATTR = "_hope_qdes_projection_signature"
_QDES_PROJECTION_OBSERVED_COUNT = "observed_sample_count"
_QDES_PROJECTION_SAMPLE_COUNT = "projection_sample_count"
_QDES_PROJECTION_NONFINITE_SAMPLE_COUNT = "nonfinite_sample_count"
_QDES_PROJECTION_JOINT_COUNT = "projection_joint_count"
_QDES_PROJECTION_LOWER_JOINT_COUNT = "lower_projection_joint_count"
_QDES_PROJECTION_UPPER_JOINT_COUNT = "upper_projection_joint_count"
_QDES_PROJECTION_DISTANCE_SUM = "normalized_projection_distance_sum"
_QDES_PROJECTION_DISTANCE_MAX = "normalized_projection_distance_max"
_QDES_PROJECTION_VALUE_SUM = "penalty_value_sum"
_QDES_PROJECTION_MAX_DISTANCE = "max_normalized_projection_distance"


def _validate_soft_limit_barrier_v2_scalars(
    margin_frac: float,
    penalty_floor: float,
    *,
    term_name: str,
) -> tuple[float, float]:
    if isinstance(margin_frac, bool):
        raise ValueError(f"{term_name} margin_frac must be finite and in (0, 0.5)")
    if isinstance(penalty_floor, bool):
        raise ValueError(f"{term_name} penalty_floor must be finite and in (0, 1)")
    margin_frac = float(margin_frac)
    penalty_floor = float(penalty_floor)
    if not math.isfinite(margin_frac) or not 0.0 < margin_frac < 0.5:
        raise ValueError(f"{term_name} margin_frac must be finite and in (0, 0.5)")
    if not math.isfinite(penalty_floor) or not 0.0 < penalty_floor < 1.0:
        raise ValueError(f"{term_name} penalty_floor must be finite and in (0, 1)")
    return margin_frac, penalty_floor


def _soft_limit_barrier_v2_kernel(
    positions: torch.Tensor,
    limits: torch.Tensor,
    default_joint_pos: torch.Tensor,
    *,
    margin_frac: float,
    penalty_floor: float,
    term_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sum barrier, normalized per-joint intrusion)`` for one position source.

    For every joint, using the stance-exempt margin from qbar v1::

        d       = min(q-lo, hi-q) / (hi-lo)
        m_eff   = min(margin_frac, d(default_q) - 0.005)
        u       = relu(m_eff-d) / m_eff
        ramp(u) = (1-exp(-4*clamp(u,0,1))) / (1-exp(-4))
        b(u)    = 0                              if u == 0
                  floor + (1-floor)*ramp(u)      if u > 0
        value   = sum_j b(u_j)

    The discontinuous positive floor is intentional: there is no sequence of positive
    intrusions whose charge tends to zero.  Above the edge, the term is monotone and capped at
    one per joint.  SUM aggregation keeps a single unsafe joint undiluted and exposes multi-joint
    exploitation linearly.
    """

    margin_frac, penalty_floor = _validate_soft_limit_barrier_v2_scalars(
        margin_frac, penalty_floor, term_name=term_name
    )
    if (
        not torch.is_tensor(positions)
        or positions.ndim != 2
        or tuple(positions.shape)[1] != _QDES_LIMIT_BARRIER_JOINT_COUNT
    ):
        raise RuntimeError(f"{term_name} requires joint positions shaped [num_envs, 31]")
    num_envs = tuple(positions.shape)[0]
    if not torch.is_tensor(limits):
        raise RuntimeError(f"{term_name} requires runtime articulation soft joint position limits")
    if limits.ndim == 2 and tuple(limits.shape) == (_QDES_LIMIT_BARRIER_JOINT_COUNT, 2):
        lower = limits[:, 0]
        upper = limits[:, 1]
    elif (
        limits.ndim == 3
        and tuple(limits.shape)[1:] == (_QDES_LIMIT_BARRIER_JOINT_COUNT, 2)
        and tuple(limits.shape)[0] in (1, num_envs)
    ):
        lower = limits[:, :, 0]
        upper = limits[:, :, 1]
    else:
        raise RuntimeError(
            f"{term_name} requires soft_joint_pos_limits shaped [31,2], "
            "[1,31,2], or [num_envs,31,2]"
        )
    span = upper - lower
    valid_limits = torch.all(torch.isfinite(lower) & torch.isfinite(upper) & span.gt(0.0))
    if valid_limits.device.type == "cpu":
        if not bool(valid_limits):
            raise RuntimeError(
                f"{term_name} requires finite joint position limits with lo < hi"
            )
    else:
        torch._assert_async(valid_limits)

    if not torch.is_tensor(default_joint_pos):
        raise RuntimeError(
            f"{term_name} requires articulation default_joint_pos for stance exemption"
        )
    if default_joint_pos.ndim == 1:
        if tuple(default_joint_pos.shape) != (_QDES_LIMIT_BARRIER_JOINT_COUNT,):
            raise RuntimeError(
                f"{term_name} requires default_joint_pos shaped [31], [1,31], "
                "or [num_envs,31]"
            )
        default_q = default_joint_pos
    elif (
        default_joint_pos.ndim == 2
        and tuple(default_joint_pos.shape)[1] == _QDES_LIMIT_BARRIER_JOINT_COUNT
        and tuple(default_joint_pos.shape)[0] in (1, num_envs)
    ):
        default_q = default_joint_pos
    else:
        raise RuntimeError(
            f"{term_name} requires default_joint_pos shaped [31], [1,31], "
            "or [num_envs,31]"
        )
    default_valid = torch.all(torch.isfinite(default_q))
    if default_valid.device.type == "cpu":
        if not bool(default_valid):
            raise RuntimeError(f"{term_name} requires finite default_joint_pos")
    else:
        torch._assert_async(default_valid)

    d_default = torch.minimum(default_q - lower, upper - default_q) / span
    margin_eff = torch.clamp(
        d_default - _QDES_LIMIT_BARRIER_STANCE_EPS, max=margin_frac
    )
    stance_valid = torch.all(margin_eff.gt(_QDES_LIMIT_BARRIER_MARGIN_FLOOR))
    if stance_valid.device.type == "cpu":
        if not bool(stance_valid):
            raise RuntimeError(
                f"{term_name}: a default-stance joint sits within EPS+FLOOR of its "
                "soft limit — fix the limits or stance instead of silencing the barrier"
            )
    else:
        torch._assert_async(stance_valid)

    distance = torch.minimum(positions - lower, upper - positions) / span
    intrusion = torch.relu(margin_eff - distance) / margin_eff
    unit_depth = torch.clamp(intrusion, max=1.0)
    denominator = -math.expm1(-_SOFT_LIMIT_BARRIER_V2_SHAPE_RATE)
    ramp = -torch.expm1(-_SOFT_LIMIT_BARRIER_V2_SHAPE_RATE * unit_depth) / denominator
    per_joint = torch.where(
        intrusion > 0.0,
        penalty_floor + (1.0 - penalty_floor) * ramp,
        torch.zeros_like(ramp),
    )
    return torch.sum(per_joint, dim=-1), intrusion


def _qdes_limit_barrier_v2_values(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
    penalty_floor: float = _SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_name != "joint_pos":
        raise ValueError("qdes_limit_barrier_v2 action_name must be exactly 'joint_pos'")
    action = env.action_manager.get_term(action_name)
    processed = getattr(action, "processed_actions", None)
    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    runtime_names = list(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    action_names = list(getattr(action, "_joint_names", ()))
    if (
        len(runtime_names) != _QDES_LIMIT_BARRIER_JOINT_COUNT
        or len(set(runtime_names)) != _QDES_LIMIT_BARRIER_JOINT_COUNT
        or action_names != runtime_names
    ):
        raise RuntimeError(
            "qdes_limit_barrier_v2 requires identity 31-joint A3 action/articulation order"
        )
    raw_joint_ids = getattr(action, "_joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        action_joint_ids = list(range(_QDES_LIMIT_BARRIER_JOINT_COUNT))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        action_joint_ids = [int(value) for value in raw_joint_ids]
    if action_joint_ids != list(range(_QDES_LIMIT_BARRIER_JOINT_COUNT)):
        raise RuntimeError(
            "qdes_limit_barrier_v2 requires identity 31-joint A3 action/articulation order"
        )
    return _soft_limit_barrier_v2_kernel(
        processed,
        getattr(data, "soft_joint_pos_limits", None),
        getattr(data, "default_joint_pos", None),
        margin_frac=margin_frac,
        penalty_floor=penalty_floor,
        term_name="qdes_limit_barrier_v2",
    )


def _actual_joint_limit_barrier_v2_values(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin_frac: float = 0.08,
    penalty_floor: float = _SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR,
    expected_joint_count: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(expected_joint_count) is not int or expected_joint_count != 31:
        raise ValueError(
            "actual_joint_limit_barrier_v2 requires the exact 31-joint A3 runtime order"
        )
    asset = env.scene[asset_cfg.name]
    data = asset.data
    runtime_names = list(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    if (
        len(runtime_names) != expected_joint_count
        or len(set(runtime_names)) != expected_joint_count
        or any(not str(name) for name in runtime_names)
    ):
        raise RuntimeError(
            "actual_joint_limit_barrier_v2 requires exactly 31 unique runtime joint names"
        )
    raw_joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if isinstance(raw_joint_ids, slice):
        joint_ids = list(range(expected_joint_count))[raw_joint_ids]
    else:
        if hasattr(raw_joint_ids, "tolist"):
            raw_joint_ids = raw_joint_ids.tolist()
        joint_ids = [int(value) for value in raw_joint_ids]
    if joint_ids != list(range(expected_joint_count)):
        raise RuntimeError(
            "actual_joint_limit_barrier_v2 requires identity 31-joint articulation order"
        )
    return _soft_limit_barrier_v2_kernel(
        getattr(data, "joint_pos", None),
        getattr(data, "soft_joint_pos_limits", None),
        getattr(data, "default_joint_pos", None),
        margin_frac=margin_frac,
        penalty_floor=penalty_floor,
        term_name="actual_joint_limit_barrier_v2",
    )


def _soft_limit_barrier_v2_counter_state(
    env: ManagerBasedRLEnv,
    attr_name: str,
    template: torch.Tensor,
) -> dict[str, torch.Tensor]:
    expected_count_names = {
        _SOFT_LIMIT_BARRIER_V2_OBSERVED_COUNT,
        _SOFT_LIMIT_BARRIER_V2_INTRUSION_JOINT_COUNT,
        _SOFT_LIMIT_BARRIER_V2_INTRUSION_SAMPLE_COUNT,
        _SOFT_LIMIT_BARRIER_V2_ACTIVE_COUNT,
    }
    expected_float_names = {
        _SOFT_LIMIT_BARRIER_V2_MAX_INTRUSION,
        _SOFT_LIMIT_BARRIER_V2_VALUE_SUM,
    }
    expected_names = expected_count_names | expected_float_names
    state = getattr(env, attr_name, None)
    if state is None:
        state = {
            _SOFT_LIMIT_BARRIER_V2_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _SOFT_LIMIT_BARRIER_V2_INTRUSION_JOINT_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _SOFT_LIMIT_BARRIER_V2_INTRUSION_SAMPLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _SOFT_LIMIT_BARRIER_V2_MAX_INTRUSION: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _SOFT_LIMIT_BARRIER_V2_VALUE_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _SOFT_LIMIT_BARRIER_V2_ACTIVE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
        }
        setattr(env, attr_name, state)
    if not isinstance(state, dict) or set(state) != expected_names:
        raise RuntimeError(
            "soft-limit barrier v2 activation state has a schema mismatch"
        )
    for name, value in state.items():
        expected_dtype = torch.long if name in expected_count_names else template.dtype
        if (
            not torch.is_tensor(value)
            or value.ndim != 0
            or value.device != template.device
            or value.dtype != expected_dtype
        ):
            raise RuntimeError(
                "soft-limit barrier v2 activation state has a dtype/device/shape mismatch"
            )
    return state


def _record_soft_limit_barrier_v2_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    intrusion: torch.Tensor,
    *,
    source: str,
    barrier_active: bool,
    signature: tuple,
) -> None:
    if source == "qdes":
        state_attr = _QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR
        observed_attr = _QDES_LIMIT_BARRIER_V2_OBSERVED_STEP_ATTR
        active_attr = _QDES_LIMIT_BARRIER_V2_ACTIVE_STEP_ATTR
        signature_attr = _QDES_LIMIT_BARRIER_V2_SIGNATURE_ATTR
    elif source == "actual":
        state_attr = _ACTUAL_LIMIT_BARRIER_V2_ACTIVATION_ATTR
        observed_attr = _ACTUAL_LIMIT_BARRIER_V2_OBSERVED_STEP_ATTR
        active_attr = _ACTUAL_LIMIT_BARRIER_V2_ACTIVE_STEP_ATTR
        signature_attr = _ACTUAL_LIMIT_BARRIER_V2_SIGNATURE_ATTR
    else:
        raise ValueError(f"unknown soft-limit barrier v2 source {source!r}")

    token = getattr(env, "common_step_counter", None)
    if type(token) is not int or token < 0:
        raise RuntimeError(
            "soft-limit barrier v2 activation requires a nonnegative plain "
            "common_step_counter"
        )
    state = _soft_limit_barrier_v2_counter_state(env, state_attr, values)
    already_observed = getattr(env, observed_attr, None) == token
    if already_observed:
        if getattr(env, signature_attr, None) != signature:
            raise RuntimeError(
                f"{source} soft-limit barrier v2 probe and RewardTerm used "
                "different parameters in one step"
            )
    else:
        active = intrusion.detach().gt(0.0)
        state[_SOFT_LIMIT_BARRIER_V2_OBSERVED_COUNT].add_(values.numel())
        state[_SOFT_LIMIT_BARRIER_V2_INTRUSION_JOINT_COUNT].add_(
            active.sum(dtype=torch.long)
        )
        state[_SOFT_LIMIT_BARRIER_V2_INTRUSION_SAMPLE_COUNT].add_(
            torch.any(active, dim=-1).sum(dtype=torch.long)
        )
        state[_SOFT_LIMIT_BARRIER_V2_MAX_INTRUSION].copy_(
            torch.maximum(
                state[_SOFT_LIMIT_BARRIER_V2_MAX_INTRUSION],
                intrusion.detach().max(),
            )
        )
        state[_SOFT_LIMIT_BARRIER_V2_VALUE_SUM].add_(values.detach().sum())
        setattr(env, observed_attr, token)
        setattr(env, signature_attr, signature)
    if barrier_active and getattr(env, active_attr, None) != token:
        state[_SOFT_LIMIT_BARRIER_V2_ACTIVE_COUNT].add_(values.numel())
        setattr(env, active_attr, token)


def qdes_limit_barrier_v2_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
    penalty_floor: float = _SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR,
) -> torch.Tensor:
    """Measure the floor-bearing processed-q_des barrier while returning exact zero."""

    values, intrusion = _qdes_limit_barrier_v2_values(
        env, action_name, margin_frac, penalty_floor
    )
    _record_soft_limit_barrier_v2_activation(
        env,
        values,
        intrusion,
        source="qdes",
        barrier_active=False,
        signature=(action_name, float(margin_frac), float(penalty_floor)),
    )
    return torch.zeros_like(values)


def qdes_limit_barrier_v2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    margin_frac: float = 0.08,
    penalty_floor: float = _SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR,
) -> torch.Tensor:
    """Penalize every processed q_des soft-band intrusion with a nonzero per-joint floor."""

    values, intrusion = _qdes_limit_barrier_v2_values(
        env, action_name, margin_frac, penalty_floor
    )
    _record_soft_limit_barrier_v2_activation(
        env,
        values,
        intrusion,
        source="qdes",
        barrier_active=True,
        signature=(action_name, float(margin_frac), float(penalty_floor)),
    )
    return values


def _validate_qdes_projection_shape_rate(shape_rate: float) -> float:
    if isinstance(shape_rate, bool):
        raise ValueError("qdes_projection_penalty shape_rate must be finite and > 0")
    try:
        shape_rate = float(shape_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "qdes_projection_penalty shape_rate must be finite and > 0"
        ) from exc
    if not math.isfinite(shape_rate) or shape_rate <= 0.0:
        raise ValueError("qdes_projection_penalty shape_rate must be finite and > 0")
    return shape_rate


def _qdes_projection_penalty_values(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    shape_rate: float = _QDES_PROJECTION_DEFAULT_SHAPE_RATE,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return bounded projection cost and its per-joint causal evidence.

    For each joint, ``d = abs(pre_clamp_qdes - nominal_projection) / envelope_span`` and
    ``cost = 1 - exp(-shape_rate * d)``.  Thus an in-envelope request is exactly zero, every
    finite additional overshoot is strictly more expensive, and an arbitrarily large proposal
    stays bounded below one per joint.  Non-finite proposals receive a finite near-maximal cost
    here and remain terminal in the DoneTerm.
    """

    if action_name != "joint_pos":
        raise ValueError(
            "qdes_projection_penalty action_name must be exactly 'joint_pos'"
        )
    shape_rate = _validate_qdes_projection_shape_rate(shape_rate)
    action = env.action_manager.get_term(action_name)
    projection_enabled = getattr(
        action, "finite_preclamp_qdes_projection_enabled", False
    )
    if type(projection_enabled) is not bool or not projection_enabled:
        raise RuntimeError(
            "qdes_projection_penalty requires the ActionBall finite-q_des projection mode"
        )

    pre_qdes = getattr(action, "pre_clamp_qdes", None)
    projected = getattr(action, "nominal_projected_qdes", None)
    span = getattr(action, "nominal_projection_span", None)
    pre_valid = getattr(action, "pre_clamp_qdes_valid", None)
    projected_valid = getattr(action, "nominal_projected_qdes_valid", None)
    expected_shape = (int(env.num_envs), _QDES_LIMIT_BARRIER_JOINT_COUNT)
    for name, tensor in (
        ("pre_clamp_qdes", pre_qdes),
        ("nominal_projected_qdes", projected),
        ("nominal_projection_span", span),
    ):
        if (
            not torch.is_tensor(tensor)
            or tuple(tensor.shape) != expected_shape
            or tensor.dtype not in (torch.float32, torch.float64)
        ):
            raise RuntimeError(
                f"qdes_projection_penalty requires {name} shaped "
                f"[num_envs, {_QDES_LIMIT_BARRIER_JOINT_COUNT}] in floating point"
            )
    if pre_qdes.device != projected.device or pre_qdes.device != span.device:
        raise RuntimeError(
            "qdes_projection_penalty requires request, projection and span on one device"
        )
    if pre_qdes.dtype != projected.dtype or pre_qdes.dtype != span.dtype:
        raise RuntimeError(
            "qdes_projection_penalty requires request, projection and span in one dtype"
        )
    for name, valid in (
        ("pre_clamp_qdes_valid", pre_valid),
        ("nominal_projected_qdes_valid", projected_valid),
    ):
        if (
            not torch.is_tensor(valid)
            or valid.dtype != torch.bool
            or tuple(valid.shape) != (expected_shape[0],)
            or valid.device != pre_qdes.device
        ):
            raise RuntimeError(
                f"qdes_projection_penalty requires {name} as same-device bool [num_envs]"
            )

    valid = pre_valid & projected_valid
    structural_valid = torch.all(
        torch.isfinite(projected) & torch.isfinite(span) & span.gt(0.0), dim=1
    )
    invalid_active = torch.any(valid & ~structural_valid)
    if invalid_active.device.type == "cpu":
        if bool(invalid_active):
            raise RuntimeError(
                "qdes_projection_penalty received a valid row with invalid projection/span"
            )
    else:
        torch._assert_async(~invalid_active)

    # Invalid/reset rows are exact zero.  A non-finite raw proposal uses one full envelope span as
    # a finite surrogate distance; its true safety consequence is independently preserved by the
    # terminal term.
    safe_projected = torch.where(
        structural_valid.unsqueeze(-1), projected, torch.zeros_like(projected)
    )
    safe_span = torch.where(
        structural_valid.unsqueeze(-1), span, torch.ones_like(span)
    )
    nonfinite_joint = ~torch.isfinite(pre_qdes)
    safe_pre_qdes = torch.where(
        nonfinite_joint, safe_projected + safe_span, pre_qdes
    )
    distance = torch.abs(safe_pre_qdes - safe_projected) / safe_span
    per_joint = -torch.expm1(-shape_rate * distance)
    valid_joint = valid.unsqueeze(-1)
    per_joint = torch.where(valid_joint, per_joint, torch.zeros_like(per_joint))
    distance = torch.where(valid_joint, distance, torch.zeros_like(distance))
    projected_joint = valid_joint & (nonfinite_joint | distance.gt(0.0))
    finite_joint = valid_joint & ~nonfinite_joint
    lower_projected_joint = projected_joint & finite_joint & pre_qdes.lt(projected)
    upper_projected_joint = projected_joint & finite_joint & pre_qdes.gt(projected)
    values = torch.sum(per_joint, dim=-1)
    nonfinite_sample = valid & torch.any(nonfinite_joint, dim=-1)
    return (
        values,
        per_joint,
        distance,
        projected_joint,
        lower_projected_joint,
        upper_projected_joint,
        nonfinite_sample,
    )


def _qdes_projection_counter_state(
    env: ManagerBasedRLEnv,
    template: torch.Tensor,
) -> dict[str, torch.Tensor]:
    expected_names = {
        _QDES_PROJECTION_OBSERVED_COUNT,
        _QDES_PROJECTION_SAMPLE_COUNT,
        _QDES_PROJECTION_NONFINITE_SAMPLE_COUNT,
        _QDES_PROJECTION_JOINT_COUNT,
        _QDES_PROJECTION_LOWER_JOINT_COUNT,
        _QDES_PROJECTION_UPPER_JOINT_COUNT,
        _QDES_PROJECTION_DISTANCE_SUM,
        _QDES_PROJECTION_DISTANCE_MAX,
        _QDES_PROJECTION_VALUE_SUM,
        _QDES_PROJECTION_MAX_DISTANCE,
    }
    count_names = {
        _QDES_PROJECTION_OBSERVED_COUNT,
        _QDES_PROJECTION_SAMPLE_COUNT,
        _QDES_PROJECTION_NONFINITE_SAMPLE_COUNT,
    }
    vector_count_names = {
        _QDES_PROJECTION_JOINT_COUNT,
        _QDES_PROJECTION_LOWER_JOINT_COUNT,
        _QDES_PROJECTION_UPPER_JOINT_COUNT,
    }
    vector_float_names = {
        _QDES_PROJECTION_DISTANCE_SUM,
        _QDES_PROJECTION_DISTANCE_MAX,
    }
    state = getattr(env, _QDES_PROJECTION_ACTIVATION_ATTR, None)
    if state is None:
        state = {
            _QDES_PROJECTION_OBSERVED_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_PROJECTION_SAMPLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_PROJECTION_NONFINITE_SAMPLE_COUNT: torch.zeros(
                (), dtype=torch.long, device=template.device
            ),
            _QDES_PROJECTION_JOINT_COUNT: torch.zeros(
                (_QDES_LIMIT_BARRIER_JOINT_COUNT,),
                dtype=torch.long,
                device=template.device,
            ),
            _QDES_PROJECTION_LOWER_JOINT_COUNT: torch.zeros(
                (_QDES_LIMIT_BARRIER_JOINT_COUNT,),
                dtype=torch.long,
                device=template.device,
            ),
            _QDES_PROJECTION_UPPER_JOINT_COUNT: torch.zeros(
                (_QDES_LIMIT_BARRIER_JOINT_COUNT,),
                dtype=torch.long,
                device=template.device,
            ),
            _QDES_PROJECTION_DISTANCE_SUM: torch.zeros(
                (_QDES_LIMIT_BARRIER_JOINT_COUNT,),
                dtype=template.dtype,
                device=template.device,
            ),
            _QDES_PROJECTION_DISTANCE_MAX: torch.zeros(
                (_QDES_LIMIT_BARRIER_JOINT_COUNT,),
                dtype=template.dtype,
                device=template.device,
            ),
            _QDES_PROJECTION_VALUE_SUM: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
            _QDES_PROJECTION_MAX_DISTANCE: torch.zeros(
                (), dtype=template.dtype, device=template.device
            ),
        }
        setattr(env, _QDES_PROJECTION_ACTIVATION_ATTR, state)
    if not isinstance(state, dict) or set(state) != expected_names:
        raise RuntimeError("qdes projection activation state has a schema mismatch")
    for name, value in state.items():
        expected_dtype = (
            torch.long
            if name in count_names or name in vector_count_names
            else template.dtype
        )
        expected_shape = (
            (_QDES_LIMIT_BARRIER_JOINT_COUNT,)
            if name in vector_count_names or name in vector_float_names
            else ()
        )
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != expected_shape
            or value.device != template.device
            or value.dtype != expected_dtype
        ):
            raise RuntimeError(
                "qdes projection activation state has a dtype/device/shape mismatch"
            )
    return state


def _record_qdes_projection_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    distance: torch.Tensor,
    projected_joint: torch.Tensor,
    lower_projected_joint: torch.Tensor,
    upper_projected_joint: torch.Tensor,
    nonfinite_sample: torch.Tensor,
    *,
    signature: tuple,
) -> None:
    token = getattr(env, "common_step_counter", None)
    if type(token) is not int or token < 0:
        raise RuntimeError(
            "qdes projection activation requires a nonnegative plain common_step_counter"
        )
    state = _qdes_projection_counter_state(env, values)
    if getattr(env, _QDES_PROJECTION_OBSERVED_STEP_ATTR, None) == token:
        if getattr(env, _QDES_PROJECTION_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "qdes projection RewardTerm used different parameters in one step"
            )
        return

    active_sample = torch.any(projected_joint.detach(), dim=-1)
    state[_QDES_PROJECTION_OBSERVED_COUNT].add_(values.numel())
    state[_QDES_PROJECTION_SAMPLE_COUNT].add_(
        active_sample.sum(dtype=torch.long)
    )
    state[_QDES_PROJECTION_NONFINITE_SAMPLE_COUNT].add_(
        nonfinite_sample.detach().sum(dtype=torch.long)
    )
    state[_QDES_PROJECTION_JOINT_COUNT].add_(
        projected_joint.detach().sum(dim=0, dtype=torch.long)
    )
    state[_QDES_PROJECTION_LOWER_JOINT_COUNT].add_(
        lower_projected_joint.detach().sum(dim=0, dtype=torch.long)
    )
    state[_QDES_PROJECTION_UPPER_JOINT_COUNT].add_(
        upper_projected_joint.detach().sum(dim=0, dtype=torch.long)
    )
    state[_QDES_PROJECTION_DISTANCE_SUM].add_(
        torch.where(
            projected_joint.detach(),
            distance.detach(),
            torch.zeros_like(distance),
        ).sum(dim=0)
    )
    state[_QDES_PROJECTION_DISTANCE_MAX].copy_(
        torch.maximum(
            state[_QDES_PROJECTION_DISTANCE_MAX],
            distance.detach().max(dim=0).values,
        )
    )
    state[_QDES_PROJECTION_VALUE_SUM].add_(values.detach().sum())
    state[_QDES_PROJECTION_MAX_DISTANCE].copy_(
        torch.maximum(
            state[_QDES_PROJECTION_MAX_DISTANCE],
            distance.detach().max(),
        )
    )
    setattr(env, _QDES_PROJECTION_OBSERVED_STEP_ATTR, token)
    setattr(env, _QDES_PROJECTION_SIGNATURE_ATTR, signature)


def qdes_projection_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    shape_rate: float = _QDES_PROJECTION_DEFAULT_SHAPE_RATE,
    objective_weight: float | None = None,
) -> torch.Tensor:
    """Penalize projection distance and always expose the unweighted causal dose.

    The production default leaves ``objective_weight`` absent and lets the
    RewardManager apply the configured term weight.  A reviewed ablation sets
    the manager weight to minus one and supplies its non-positive objective weight
    here.  In particular, explicit zero still executes this callable, records
    the same unweighted counters, and returns exact zero reward.
    """

    shape_rate = _validate_qdes_projection_shape_rate(shape_rate)
    if objective_weight is not None:
        if type(objective_weight) not in (int, float):
            raise ValueError(
                "qdes_projection_penalty objective_weight must be an exact "
                "finite int/float in [-5.0, 0.0]"
            )
        objective_weight = float(objective_weight)
        if (
            not math.isfinite(objective_weight)
            or objective_weight < -5.0
            or objective_weight > 0.0
        ):
            raise ValueError(
                "qdes_projection_penalty objective_weight must be an exact "
                "finite int/float in [-5.0, 0.0]"
            )
    (
        values,
        _,
        distance,
        projected_joint,
        lower_projected_joint,
        upper_projected_joint,
        nonfinite_sample,
    ) = (
        _qdes_projection_penalty_values(env, action_name, shape_rate)
    )
    _record_qdes_projection_activation(
        env,
        values,
        distance,
        projected_joint,
        lower_projected_joint,
        upper_projected_joint,
        nonfinite_sample,
        signature=(action_name, shape_rate),
    )
    if objective_weight is None:
        return values
    # RewardManager contributes ``manager_weight * callable_value``.  The
    # explicit-ablation manager weight is -1, so this nonnegative magnitude
    # preserves the requested non-positive objective dose exactly.
    return values * (-objective_weight)


def actual_joint_limit_barrier_v2_probe(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin_frac: float = 0.08,
    penalty_floor: float = _SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR,
    expected_joint_count: int = 31,
) -> torch.Tensor:
    """Measure the floor-bearing realized-joint barrier while returning exact zero."""

    values, intrusion = _actual_joint_limit_barrier_v2_values(
        env, asset_cfg, margin_frac, penalty_floor, expected_joint_count
    )
    _record_soft_limit_barrier_v2_activation(
        env,
        values,
        intrusion,
        source="actual",
        barrier_active=False,
        signature=(float(margin_frac), float(penalty_floor), expected_joint_count),
    )
    return torch.zeros_like(values)


def actual_joint_limit_barrier_v2(
    env: ManagerBasedRLEnv,
    asset_cfg,
    margin_frac: float = 0.08,
    penalty_floor: float = _SOFT_LIMIT_BARRIER_V2_DEFAULT_FLOOR,
    expected_joint_count: int = 31,
) -> torch.Tensor:
    """Penalize actual-q soft-band intrusion separately from the commanded-q_des term."""

    values, intrusion = _actual_joint_limit_barrier_v2_values(
        env, asset_cfg, margin_frac, penalty_floor, expected_joint_count
    )
    _record_soft_limit_barrier_v2_activation(
        env,
        values,
        intrusion,
        source="actual",
        barrier_active=True,
        signature=(float(margin_frac), float(penalty_floor), expected_joint_count),
    )
    return values


def consume_qdes_limit_barrier_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update's q_des/actual limit-barrier ledger.

    The reset MUST run under ``torch.inference_mode()``: the counters are created during rollout
    collection (inference mode), and an in-place ``zero_()`` on an inference tensor outside
    InferenceMode raises at the first PPO log flush.

    Historical v1 runs retain their exact unprefixed keys.  A v2 ActionBall run emits explicit
    ``qdes_*`` and ``actual_*`` keys so command and realized-state charges cannot be mistaken for
    one another or hidden as an aggregate.
    """

    action = env.action_manager.get_term("joint_pos")
    template = action.processed_actions[:, 0]
    v2_states = []
    for prefix, attr_name in (
        ("qdes", _QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR),
        ("actual", _ACTUAL_LIMIT_BARRIER_V2_ACTIVATION_ATTR),
    ):
        state = getattr(env, attr_name, None)
        if state is not None:
            v2_states.append((prefix, state))
    if v2_states:
        if len(v2_states) != 2:
            raise RuntimeError(
                "soft-limit barrier v2 counter consume requires both qdes and actual channels"
            )
        v2_states = [
            (
                prefix,
                _soft_limit_barrier_v2_counter_state(env, attr_name, template),
            )
            for prefix, attr_name in (
                ("qdes", _QDES_LIMIT_BARRIER_V2_ACTIVATION_ATTR),
                ("actual", _ACTUAL_LIMIT_BARRIER_V2_ACTIVATION_ATTR),
            )
        ]
        snapshot = {
            f"{prefix}_{name}": value.detach().clone()
            for prefix, state in v2_states
            for name, value in state.items()
        }
        projection_state = getattr(env, _QDES_PROJECTION_ACTIVATION_ATTR, None)
        if projection_state is not None:
            projection_state = _qdes_projection_counter_state(
                env, template
            )
            joint_counts = projection_state[_QDES_PROJECTION_JOINT_COUNT]
            distance_sums = projection_state[_QDES_PROJECTION_DISTANCE_SUM]
            distance_maxima = projection_state[_QDES_PROJECTION_DISTANCE_MAX]
            observed_count = projection_state[_QDES_PROJECTION_OBSERVED_COUNT]
            for name, value in projection_state.items():
                if value.ndim == 0:
                    snapshot[f"projection_{name}"] = value.detach().clone()
            snapshot["projection_saturation_sample_step_ratio"] = torch.where(
                observed_count.gt(0),
                projection_state[_QDES_PROJECTION_SAMPLE_COUNT]
                / torch.clamp(observed_count, min=1),
                torch.zeros_like(distance_sums.sum()),
            ).detach().clone()
            total_count = joint_counts.sum()
            snapshot["projection_mean_normalized_projection_distance"] = (
                torch.where(
                    total_count.gt(0),
                    distance_sums.sum() / torch.clamp(total_count, min=1),
                    torch.zeros_like(distance_sums.sum()),
                ).detach().clone()
            )
            for joint_index in range(_QDES_LIMIT_BARRIER_JOINT_COUNT):
                count = joint_counts[joint_index]
                prefix = f"projection_joint_{joint_index:02d}"
                snapshot[f"{prefix}_trigger_count"] = count.detach().clone()
                # "Saturation" means the nominal projected q_des is exactly on the corresponding
                # lower/upper target-envelope clamp edge.  Counts are environment-policy-steps;
                # dividing by observed samples makes the launch target (for example 0.5%) direct.
                snapshot[
                    f"{prefix}_saturation_env_step_count"
                ] = count.detach().clone()
                snapshot[f"{prefix}_saturation_env_step_ratio"] = torch.where(
                    observed_count.gt(0),
                    count / torch.clamp(observed_count, min=1),
                    torch.zeros_like(distance_sums[joint_index]),
                ).detach().clone()
                snapshot[f"{prefix}_lower_saturation_env_step_count"] = projection_state[
                    _QDES_PROJECTION_LOWER_JOINT_COUNT
                ][joint_index].detach().clone()
                snapshot[f"{prefix}_upper_saturation_env_step_count"] = projection_state[
                    _QDES_PROJECTION_UPPER_JOINT_COUNT
                ][joint_index].detach().clone()
                snapshot[f"{prefix}_mean_normalized_excess"] = torch.where(
                    count.gt(0),
                    distance_sums[joint_index] / torch.clamp(count, min=1),
                    torch.zeros_like(distance_sums[joint_index]),
                ).detach().clone()
                snapshot[f"{prefix}_max_normalized_excess"] = distance_maxima[
                    joint_index
                ].detach().clone()
        with torch.inference_mode():
            for _, state in v2_states:
                for value in state.values():
                    value.zero_()
            if projection_state is not None:
                for value in projection_state.values():
                    value.zero_()
        return snapshot

    state = _qdes_limit_barrier_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


# Wave-B lower-body diagnostics --------------------------------------------------------------- #
#
# Both mechanisms below are intentionally default-off and share one phase gate: a bounded
# pre-contact support interval plus the existing reset/wrap/reveal-aware post-strike clock.  The
# clock is armed by the phase-aligned strike opportunity, not by racket contact or task success,
# so a failed attempt remains in the sample instead of being silently selected away.
_A3_RUNTIME_JOINT_ORDER = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_LOWER_BODY_LEG_JOINT_NAMES = frozenset(_A3_RUNTIME_JOINT_ORDER[-12:])
_LOWER_BODY_FOOT_BODY_NAMES = ("left_ankle_roll_Link", "right_ankle_roll_Link")

_LOWER_BODY_POSE_COUNTER_ATTR = "_hope_lower_body_pose_activation_counters"
_LOWER_BODY_POSE_OBSERVED_STEP_ATTR = "_hope_lower_body_pose_observed_step"
_LOWER_BODY_POSE_ACTIVE_STEP_ATTR = "_hope_lower_body_pose_active_step"
_LOWER_BODY_POSE_SIGNATURE_ATTR = "_hope_lower_body_pose_signature"
_LOWER_BODY_BUNDLE_COUNTER_ATTR = "_hope_lower_body_bundle_activation_counters"
_LOWER_BODY_BUNDLE_OBSERVED_STEP_ATTR = "_hope_lower_body_bundle_observed_step"
_LOWER_BODY_BUNDLE_ACTIVE_STEP_ATTR = "_hope_lower_body_bundle_active_step"
_LOWER_BODY_BUNDLE_SIGNATURE_ATTR = "_hope_lower_body_bundle_signature"


def _finite_scalar(value, *, name: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return parsed


def _lower_body_support_gate(
    env: ManagerBasedRLEnv,
    racket_command_name: str,
    motion_command_name: str,
    support_pre_s: float,
    support_post_s: float,
) -> torch.Tensor:
    """Inclusive support gate without success-conditioned sample selection."""

    if racket_command_name != "racket_target" or motion_command_name != "motion":
        raise ValueError(
            "lower-body support rewards require racket_target and motion command names"
        )
    support_pre_s = _finite_scalar(
        support_pre_s, name="lower-body support_pre_s", nonnegative=True
    )
    support_post_s = _finite_scalar(
        support_post_s, name="lower-body support_post_s", nonnegative=True
    )
    racket = _cmd(env, racket_command_name)
    motion = env.command_manager.get_term(motion_command_name)
    time_to_strike = getattr(racket, "time_to_strike", None)
    pre_strike = getattr(racket, "pre_strike", None)
    in_hold = getattr(motion, "in_hold", None)
    clock = getattr(racket, "post_strike_age_and_same_attempt", None)
    if (
        not torch.is_tensor(time_to_strike)
        or not torch.is_tensor(pre_strike)
        or pre_strike.dtype != torch.bool
        or not torch.is_tensor(in_hold)
        or in_hold.dtype != torch.bool
        or not callable(clock)
    ):
        raise RuntimeError(
            "lower-body support rewards require live TTS/pre-strike, motion hold, and "
            "same-attempt post-strike clock tensors"
        )
    age_s, same_attempt = clock()
    expected_shape = tuple(time_to_strike.shape)
    if (
        time_to_strike.ndim != 1
        or tuple(pre_strike.shape) != expected_shape
        or tuple(in_hold.shape) != expected_shape
        or not torch.is_tensor(age_s)
        or tuple(age_s.shape) != expected_shape
        or not torch.is_tensor(same_attempt)
        or tuple(same_attempt.shape) != expected_shape
        or same_attempt.dtype != torch.bool
    ):
        raise RuntimeError("lower-body support reward phase tensors must be aligned per environment")

    pre_support = (
        (~in_hold)
        & pre_strike
        & time_to_strike.ge(0.0)
        & time_to_strike.le(support_pre_s)
    )
    post_support = same_attempt & age_s.ge(0.0) & age_s.le(support_post_s)
    return pre_support | post_support


def _lower_body_runtime_tensors(
    env: ManagerBasedRLEnv,
    motion_command_name: str,
    *,
    require_motion_reference: bool,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[int]]:
    """Resolve the exact current-main A3 reference/runtime order or fail closed."""

    if motion_command_name != "motion":
        raise ValueError("lower-body rewards require motion_command_name='motion'")
    robot = env.scene["robot"]
    data = robot.data
    runtime_names = tuple(
        str(name)
        for name in getattr(data, "joint_names", getattr(robot, "joint_names", ()))
    )
    # The live Isaac articulation enumerates joints breadth-first, not in the
    # deploy-runtime order; require the exact A3 name set (31, unique) and select
    # leg indices by name below, matching processed_qdes_slew_hinge's discipline.
    if (
        len(runtime_names) != len(_A3_RUNTIME_JOINT_ORDER)
        or len(set(runtime_names)) != len(runtime_names)
        or set(runtime_names) != set(_A3_RUNTIME_JOINT_ORDER)
    ):
        raise RuntimeError(
            "lower-body rewards require the exact 31-joint A3 runtime articulation order"
        )
    q = getattr(data, "joint_pos", None)
    qd = getattr(data, "joint_vel", None)
    default_q = getattr(data, "default_joint_pos", None)
    motion = env.command_manager.get_term(motion_command_name)
    if getattr(motion, "robot", None) is not robot:
        raise RuntimeError("lower-body rewards require motion and reward to reference the same robot")
    reference = None
    expected_width = len(_A3_RUNTIME_JOINT_ORDER)
    if (
        not torch.is_tensor(q)
        or q.ndim != 2
        or q.shape[1] != expected_width
        or not torch.is_tensor(qd)
        or tuple(qd.shape) != tuple(q.shape)
        or not torch.is_tensor(default_q)
        or tuple(default_q.shape) not in (tuple(q.shape), (1, expected_width))
    ):
        raise RuntimeError(
            "lower-body rewards require aligned [env,31] runtime/default tensors"
        )
    if require_motion_reference:
        reference = getattr(motion, "joint_pos", None)
        loaded_reference = getattr(getattr(motion, "motion", None), "joint_pos", None)
        if (
            not torch.is_tensor(reference)
            or tuple(reference.shape) != tuple(q.shape)
            or not torch.is_tensor(loaded_reference)
            or loaded_reference.ndim != 2
            or loaded_reference.shape[1] != expected_width
        ):
            raise RuntimeError(
                "lower-body pose imitation requires an aligned [env,31] current motion "
                "tensor and a 31-column loaded motion reference"
            )
    leg_ids = [
        index for index, name in enumerate(runtime_names) if name in _LOWER_BODY_LEG_JOINT_NAMES
    ]
    if (
        len(leg_ids) != 12
        or {runtime_names[index] for index in leg_ids} != _LOWER_BODY_LEG_JOINT_NAMES
    ):
        raise RuntimeError("lower-body rewards require exactly 12 leg joints")
    return robot, q, qd, default_q, reference, leg_ids


def _lower_body_pose_values(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    std: float = 0.35,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bounded 12-leg pose imitation and diagnostic raw magnitudes."""

    std = _finite_scalar(std, name="lower_body_pose_imitation std", positive=True)
    _, q, _, default_q, reference, leg_ids = _lower_body_runtime_tensors(
        env, motion_command_name, require_motion_reference=True
    )
    eligible = _lower_body_support_gate(
        env,
        racket_command_name,
        motion_command_name,
        support_pre_s,
        support_post_s,
    )
    delta = q[:, leg_ids] - reference[:, leg_ids]
    error_sq_mean = torch.mean(torch.square(delta), dim=-1)
    kernel = torch.exp(-error_sq_mean / (std * std))
    value = torch.where(eligible, kernel, torch.zeros_like(kernel))
    reference_motion_l1 = torch.mean(
        torch.abs(reference[:, leg_ids] - default_q[:, leg_ids]), dim=-1
    )
    return value, eligible, torch.mean(torch.abs(delta), dim=-1), reference_motion_l1


def _lower_body_pose_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _LOWER_BODY_POSE_COUNTER_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "support_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "reward_enabled_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "gated_kernel_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_joint_abs_error_mean_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_reference_motion_l1_mean_sum": torch.zeros((), dtype=template.dtype, device=template.device),
        }
        setattr(env, _LOWER_BODY_POSE_COUNTER_ATTR, state)
    return state


def _record_lower_body_pose_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    joint_error: torch.Tensor,
    reference_motion: torch.Tensor,
    *,
    reward_active: bool,
    signature: tuple,
) -> None:
    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _lower_body_pose_counter_state(env, values)
    already = token is not None and getattr(env, _LOWER_BODY_POSE_OBSERVED_STEP_ATTR, None) == token
    if already:
        if getattr(env, _LOWER_BODY_POSE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError("lower-body pose probe and reward used different parameters in one step")
    else:
        mask = eligible.detach()
        state["observed_sample_count"].add_(values.numel())
        state["support_eligible_sample_count"].add_(mask.sum(dtype=torch.long))
        state["gated_kernel_sum"].add_(values.detach().sum())
        state["gated_joint_abs_error_mean_sum"].add_((joint_error.detach() * mask).sum())
        state["gated_reference_motion_l1_mean_sum"].add_((reference_motion.detach() * mask).sum())
        if token is not None:
            setattr(env, _LOWER_BODY_POSE_OBSERVED_STEP_ATTR, token)
            setattr(env, _LOWER_BODY_POSE_SIGNATURE_ATTR, signature)
    if reward_active and (
        token is None or getattr(env, _LOWER_BODY_POSE_ACTIVE_STEP_ATTR, None) != token
    ):
        state["reward_enabled_eligible_sample_count"].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _LOWER_BODY_POSE_ACTIVE_STEP_ATTR, token)


def lower_body_pose_imitation_probe(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    std: float = 0.35,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
    scale_in_window: float = 1.0,
) -> torch.Tensor:
    # 探针只记账、恒返回 0;台账记的是【未衰减】的 kernel 原值(scale_in_window 不进台账),
    # 这样开了击球窗衰减的臂也能拿探针数字和 reward 曲线对账出衰减吃掉了多少。
    scale_in_window = _finite_scalar(
        scale_in_window, name="lower_body_pose_imitation scale_in_window", nonnegative=True
    )
    values, eligible, joint_error, reference_motion = _lower_body_pose_values(
        env, racket_command_name, motion_command_name, std, support_pre_s, support_post_s
    )
    _record_lower_body_pose_activation(
        env,
        values,
        eligible,
        joint_error,
        reference_motion,
        reward_active=False,
        signature=(
            racket_command_name,
            motion_command_name,
            float(std),
            float(support_pre_s),
            float(support_post_s),
            float(scale_in_window),
        ),
    )
    return torch.zeros_like(values)


def lower_body_pose_imitation(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    std: float = 0.35,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
    scale_in_window: float = 1.0,
) -> torch.Tensor:
    """Positive v4rg 12-leg pose imitation only in the same swing's support window.

    ``scale_in_window``(默认 1.0 = 字节等价):击球窗内把本项贡献乘以该系数——下肢模仿别在
    触球那一瞬和击球奖励抢话筒(和上半身 motion_scale_in_window/V2 同一个思想、同一个窗:
    racket 命令的 WIDE strike window,1c 拆窗时用 strike_window_wide,否则就是 strike_window)。
    台账/探针记的仍是未衰减原值,见 probe 注释。
    """

    scale_in_window = _finite_scalar(
        scale_in_window, name="lower_body_pose_imitation scale_in_window", nonnegative=True
    )
    values, eligible, joint_error, reference_motion = _lower_body_pose_values(
        env, racket_command_name, motion_command_name, std, support_pre_s, support_post_s
    )
    _record_lower_body_pose_activation(
        env,
        values,
        eligible,
        joint_error,
        reference_motion,
        reward_active=True,
        signature=(
            racket_command_name,
            motion_command_name,
            float(std),
            float(support_pre_s),
            float(support_post_s),
            float(scale_in_window),
        ),
    )
    if scale_in_window == 1.0:
        return values
    window = _window_wide(_cmd(env, racket_command_name))
    if (
        not torch.is_tensor(window)
        or window.dtype != torch.bool
        or tuple(window.shape) != tuple(values.shape)
    ):
        raise RuntimeError(
            "lower_body_pose_imitation scale_in_window requires an aligned per-env bool "
            "strike-window mask on the racket command"
        )
    return torch.where(window, values * scale_in_window, values)


def _lower_body_stability_bundle_values(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    min_stance_width_m: float = 0.22,
    stance_scale_m: float = 0.05,
    leg_velocity_margin_radps: float = 1.0,
    leg_velocity_scale_radps: float = 0.5,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a bounded, reference-free physical support bundle.

    This deliberately does not repeat motion-reference foot orientation, slip, base-upright, or
    dense all-joint velocity costs.  It charges only (1) collapse/crossover below an absolute
    physical support width and (2) a 12-leg realized-qdot tail above a free margin.
    """

    min_stance_width_m = _finite_scalar(
        min_stance_width_m, name="lower_body_stability min_stance_width_m", positive=True
    )
    stance_scale_m = _finite_scalar(
        stance_scale_m, name="lower_body_stability stance_scale_m", positive=True
    )
    leg_velocity_margin_radps = _finite_scalar(
        leg_velocity_margin_radps,
        name="lower_body_stability leg_velocity_margin_radps",
        nonnegative=True,
    )
    leg_velocity_scale_radps = _finite_scalar(
        leg_velocity_scale_radps,
        name="lower_body_stability leg_velocity_scale_radps",
        positive=True,
    )
    robot, q, qd, _, _, leg_ids = _lower_body_runtime_tensors(
        env, motion_command_name, require_motion_reference=False
    )
    eligible = _lower_body_support_gate(
        env,
        racket_command_name,
        motion_command_name,
        support_pre_s,
        support_post_s,
    )

    body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    if (
        len(body_names) == 0
        or len(set(body_names)) != len(body_names)
        or any(name not in body_names for name in _LOWER_BODY_FOOT_BODY_NAMES)
    ):
        raise RuntimeError("lower-body stability requires unique left/right A3 ankle-roll bodies")
    foot_ids = [body_names.index(name) for name in _LOWER_BODY_FOOT_BODY_NAMES]
    body_pos_w = getattr(robot.data, "body_pos_w", None)
    if (
        not torch.is_tensor(body_pos_w)
        or body_pos_w.ndim != 3
        or body_pos_w.shape[0] != q.shape[0]
        or body_pos_w.shape[1] != len(body_names)
        or body_pos_w.shape[2] != 3
    ):
        raise RuntimeError("lower-body stability requires runtime body_pos_w shaped [env,body,3]")
    base_quat = getattr(_cmd(env, racket_command_name), "base_quat_w", None)
    if not torch.is_tensor(base_quat) or tuple(base_quat.shape) != (q.shape[0], 4):
        raise RuntimeError("lower-body stability requires per-environment base quaternion wxyz")
    quat = base_quat / torch.norm(base_quat, dim=-1, keepdim=True).clamp(min=1.0e-12)
    w, x, y, z = quat.unbind(dim=-1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    left_minus_right = body_pos_w[:, foot_ids[0], :2] - body_pos_w[:, foot_ids[1], :2]
    signed_width = -sin_yaw * left_minus_right[:, 0] + cos_yaw * left_minus_right[:, 1]
    stance_excess = torch.relu(min_stance_width_m - signed_width) / stance_scale_m
    stance_tail = 1.0 - torch.exp(-torch.square(stance_excess))

    leg_velocity_excess = torch.relu(
        torch.abs(qd[:, leg_ids]) - leg_velocity_margin_radps
    ) / leg_velocity_scale_radps
    leg_velocity_tail = torch.mean(
        1.0 - torch.exp(-torch.square(leg_velocity_excess)), dim=-1
    )
    raw_bundle = (stance_tail + leg_velocity_tail) / 2.0
    value = torch.where(eligible, raw_bundle, torch.zeros_like(raw_bundle))
    return value, eligible, stance_tail, leg_velocity_tail, signed_width


def _lower_body_bundle_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _LOWER_BODY_BUNDLE_COUNTER_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "support_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "reward_enabled_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "narrow_or_crossed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "gated_bundle_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_stance_tail_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_leg_velocity_tail_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "gated_signed_stance_width_m_sum": torch.zeros((), dtype=template.dtype, device=template.device),
        }
        setattr(env, _LOWER_BODY_BUNDLE_COUNTER_ATTR, state)
    return state


def _record_lower_body_bundle_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    stance_tail: torch.Tensor,
    velocity_tail: torch.Tensor,
    signed_width: torch.Tensor,
    *,
    min_stance_width_m: float,
    reward_active: bool,
    signature: tuple,
) -> None:
    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _lower_body_bundle_counter_state(env, values)
    already = token is not None and getattr(env, _LOWER_BODY_BUNDLE_OBSERVED_STEP_ATTR, None) == token
    if already:
        if getattr(env, _LOWER_BODY_BUNDLE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError("lower-body stability probe and reward used different parameters in one step")
    else:
        mask = eligible.detach()
        state["observed_sample_count"].add_(values.numel())
        state["support_eligible_sample_count"].add_(mask.sum(dtype=torch.long))
        state["narrow_or_crossed_sample_count"].add_(
            (mask & signed_width.detach().lt(min_stance_width_m)).sum(dtype=torch.long)
        )
        state["gated_bundle_sum"].add_(values.detach().sum())
        state["gated_stance_tail_sum"].add_((stance_tail.detach() * mask).sum())
        state["gated_leg_velocity_tail_sum"].add_((velocity_tail.detach() * mask).sum())
        state["gated_signed_stance_width_m_sum"].add_((signed_width.detach() * mask).sum())
        if token is not None:
            setattr(env, _LOWER_BODY_BUNDLE_OBSERVED_STEP_ATTR, token)
            setattr(env, _LOWER_BODY_BUNDLE_SIGNATURE_ATTR, signature)
    if reward_active and (
        token is None or getattr(env, _LOWER_BODY_BUNDLE_ACTIVE_STEP_ATTR, None) != token
    ):
        state["reward_enabled_eligible_sample_count"].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _LOWER_BODY_BUNDLE_ACTIVE_STEP_ATTR, token)


def lower_body_stability_bundle_probe(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    min_stance_width_m: float = 0.22,
    stance_scale_m: float = 0.05,
    leg_velocity_margin_radps: float = 1.0,
    leg_velocity_scale_radps: float = 0.5,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> torch.Tensor:
    args = (
        racket_command_name,
        motion_command_name,
        min_stance_width_m,
        stance_scale_m,
        leg_velocity_margin_radps,
        leg_velocity_scale_radps,
        support_pre_s,
        support_post_s,
    )
    values, eligible, stance, velocity, width = _lower_body_stability_bundle_values(
        env, *args
    )
    signature = tuple(float(value) if index >= 2 else value for index, value in enumerate(args))
    _record_lower_body_bundle_activation(
        env,
        values,
        eligible,
        stance,
        velocity,
        width,
        min_stance_width_m=float(min_stance_width_m),
        reward_active=False,
        signature=signature,
    )
    return torch.zeros_like(values)


def lower_body_stability_bundle(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    min_stance_width_m: float = 0.22,
    stance_scale_m: float = 0.05,
    leg_velocity_margin_radps: float = 1.0,
    leg_velocity_scale_radps: float = 0.5,
    support_pre_s: float = 0.30,
    support_post_s: float = 0.40,
) -> torch.Tensor:
    """Negative-weight B2 bundle: fixed support-width floor plus realized leg-qdot tail."""

    args = (
        racket_command_name,
        motion_command_name,
        min_stance_width_m,
        stance_scale_m,
        leg_velocity_margin_radps,
        leg_velocity_scale_radps,
        support_pre_s,
        support_post_s,
    )
    values, eligible, stance, velocity, width = _lower_body_stability_bundle_values(
        env, *args
    )
    signature = tuple(float(value) if index >= 2 else value for index, value in enumerate(args))
    _record_lower_body_bundle_activation(
        env,
        values,
        eligible,
        stance,
        velocity,
        width,
        min_stance_width_m=float(min_stance_width_m),
        reward_active=True,
        signature=signature,
    )
    return values


def consume_lower_body_wave_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot/reset both Wave-B ledgers exactly once per PPO update."""

    template = env.scene["robot"].data.joint_pos[:, 0]
    pose = _lower_body_pose_counter_state(env, template)
    bundle = _lower_body_bundle_counter_state(env, template)
    snapshot = {
        **{f"pose/{name}": value.detach().clone() for name, value in pose.items()},
        **{f"bundle/{name}": value.detach().clone() for name, value in bundle.items()},
    }
    # inference_mode (not no_grad): the counters are inference tensors created inside
    # env.step, and in-place resets outside InferenceMode are rejected by torch.
    with torch.inference_mode():
        for value in (*pose.values(), *bundle.values()):
            value.zero_()
    return snapshot


# S1 post-swing settle debts ------------------------------------------------------------------- #
#
# Idea credit: Jiayi's V13 post-swing debts (an unmerged branch).  This is a clean re-derivation on
# main, NOT a copy: the five debt channels keep the V13 intent (quiet the base, stand upright and
# tall, plant the feet after the swing), while every margin/scale below is re-fixed by this repo's
# conventions — the V13 branch's unvalidated numbers are deliberately not reused.  The activation
# window deliberately REUSES the processed_qdes_slew_hinge recovery-window mechanism (the racket
# command's reset/wrap/reveal-aware same-attempt post-strike clock) instead of inventing a second
# clock, so both recovery mechanisms agree on what "the same swing's 0.20..1.55 s" means; a reset
# invalidates the attempt through that clock and the term returns exact zero.
# 人话:挥完拍 0.2–1.55 秒内要"稳稳站好"——身体别乱晃、别歪、别蹲矮、脚别滑;每项有免费额度,
# 超出的部分按有界"债务尾巴"扣钱,重置后的无效历史一分不扣。
_POST_SWING_SETTLE_FOOT_BODY_NAMES = _LOWER_BODY_FOOT_BODY_NAMES
# A3 stand-keyframe pelvis height: robots/agibot_a3.py init_state pos z = 1.0684 m (itself read
# from the vendor MuJoCo stand keyframe).  Kept as an explicit reward parameter so the training
# contract records the exact nominal height the run trained with.
_POST_SWING_SETTLE_NOMINAL_ROOT_Z_M = 1.0684
_POST_SWING_SETTLE_COUNTER_ATTR = "_hope_post_swing_settle_activation_counters"
_POST_SWING_SETTLE_OBSERVED_STEP_ATTR = "_hope_post_swing_settle_observed_step"
_POST_SWING_SETTLE_ACTIVE_STEP_ATTR = "_hope_post_swing_settle_active_step"
_POST_SWING_SETTLE_SIGNATURE_ATTR = "_hope_post_swing_settle_signature"
_POST_SWING_SETTLE_COMPONENT_NAMES = (
    "base_quiet_lin",
    "base_quiet_ang",
    "tilt_debt",
    "root_height_debt",
    "settle_foot_slip",
)


def _post_swing_settle_tail(
    magnitude: torch.Tensor, margin: float, scale: float
) -> torch.Tensor:
    """Bounded debt tail ``1 - exp(-square(relu(x - margin) / scale))`` in [0, 1)."""

    return 1.0 - torch.exp(-torch.square(torch.relu(magnitude - margin) / scale))


def _post_swing_settle_debt_values(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    base_lin_margin_mps: float = 0.30,
    base_lin_scale_mps: float = 0.20,
    base_ang_margin_radps: float = 0.50,
    base_ang_scale_radps: float = 0.30,
    tilt_margin_rad: float = 0.10,
    tilt_scale_rad: float = 0.10,
    nominal_root_z_m: float = _POST_SWING_SETTLE_NOMINAL_ROOT_Z_M,
    root_height_deadband_m: float = 0.05,
    root_height_scale_m: float = 0.05,
    foot_slip_margin_mps: float = 0.05,
    foot_slip_scale_mps: float = 0.10,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the gated five-debt settle value, eligibility, and per-component tails [env, 5].

    Component magnitudes (each turned into a bounded tail by ``_post_swing_settle_tail``):
    ``||root_lin_vel_w||``, ``||root_ang_vel_w||``, ``asin(||projected_gravity_b_xy||)`` (the same
    tilt quantity strike/prestrike upright terms read as a norm), ``relu(nominal_root_z -
    deadband - root_z)`` (deadband is the margin), and the two ankle-roll links' mean horizontal
    speed (the same bodies/tensors the foot-slip metrics read).  Samples outside the same-attempt
    recovery window — including everything after a reset until the next armed strike — are exact 0.
    """

    if racket_command_name != "racket_target":
        raise ValueError(
            "post_swing_settle_debt racket_command_name must be exactly 'racket_target'"
        )
    if motion_command_name != "motion":
        raise ValueError(
            "post_swing_settle_debt motion_command_name must be exactly 'motion'"
        )
    base_lin_margin_mps = _finite_scalar(
        base_lin_margin_mps, name="post_swing_settle_debt base_lin_margin_mps", nonnegative=True
    )
    base_lin_scale_mps = _finite_scalar(
        base_lin_scale_mps, name="post_swing_settle_debt base_lin_scale_mps", positive=True
    )
    base_ang_margin_radps = _finite_scalar(
        base_ang_margin_radps, name="post_swing_settle_debt base_ang_margin_radps", nonnegative=True
    )
    base_ang_scale_radps = _finite_scalar(
        base_ang_scale_radps, name="post_swing_settle_debt base_ang_scale_radps", positive=True
    )
    tilt_margin_rad = _finite_scalar(
        tilt_margin_rad, name="post_swing_settle_debt tilt_margin_rad", nonnegative=True
    )
    tilt_scale_rad = _finite_scalar(
        tilt_scale_rad, name="post_swing_settle_debt tilt_scale_rad", positive=True
    )
    nominal_root_z_m = _finite_scalar(
        nominal_root_z_m, name="post_swing_settle_debt nominal_root_z_m", positive=True
    )
    root_height_deadband_m = _finite_scalar(
        root_height_deadband_m,
        name="post_swing_settle_debt root_height_deadband_m",
        nonnegative=True,
    )
    root_height_scale_m = _finite_scalar(
        root_height_scale_m, name="post_swing_settle_debt root_height_scale_m", positive=True
    )
    foot_slip_margin_mps = _finite_scalar(
        foot_slip_margin_mps, name="post_swing_settle_debt foot_slip_margin_mps", nonnegative=True
    )
    foot_slip_scale_mps = _finite_scalar(
        foot_slip_scale_mps, name="post_swing_settle_debt foot_slip_scale_mps", positive=True
    )
    if isinstance(recovery_start_s, bool) or isinstance(recovery_end_s, bool):
        raise ValueError(
            "post_swing_settle_debt recovery window must be finite and satisfy 0 <= start < end"
        )
    recovery_start_s = float(recovery_start_s)
    recovery_end_s = float(recovery_end_s)
    if (
        not math.isfinite(recovery_start_s)
        or not math.isfinite(recovery_end_s)
        or recovery_start_s < 0.0
        or recovery_start_s >= recovery_end_s
    ):
        raise ValueError(
            "post_swing_settle_debt recovery window must be finite and satisfy 0 <= start < end"
        )

    robot = env.scene["robot"]
    data = robot.data
    motion = env.command_manager.get_term(motion_command_name)
    if getattr(motion, "robot", None) is not robot:
        raise RuntimeError(
            "post_swing_settle_debt requires motion and reward to reference the same robot"
        )
    root_lin_vel = getattr(data, "root_lin_vel_w", None)
    root_ang_vel = getattr(data, "root_ang_vel_w", None)
    root_pos = getattr(data, "root_pos_w", None)
    projected_gravity = getattr(data, "projected_gravity_b", None)
    if (
        not torch.is_tensor(root_lin_vel)
        or root_lin_vel.ndim != 2
        or root_lin_vel.shape[1] != 3
        or not torch.is_tensor(root_ang_vel)
        or tuple(root_ang_vel.shape) != tuple(root_lin_vel.shape)
        or not torch.is_tensor(root_pos)
        or tuple(root_pos.shape) != tuple(root_lin_vel.shape)
        or not torch.is_tensor(projected_gravity)
        or tuple(projected_gravity.shape) != tuple(root_lin_vel.shape)
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires aligned [env,3] root velocity/position/"
            "projected-gravity tensors"
        )
    num_envs = root_lin_vel.shape[0]

    body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    if (
        len(body_names) == 0
        or len(set(body_names)) != len(body_names)
        or any(name not in body_names for name in _POST_SWING_SETTLE_FOOT_BODY_NAMES)
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires unique left/right A3 ankle-roll bodies"
        )
    foot_ids = [body_names.index(name) for name in _POST_SWING_SETTLE_FOOT_BODY_NAMES]
    body_lin_vel_w = getattr(data, "body_lin_vel_w", None)
    if (
        not torch.is_tensor(body_lin_vel_w)
        or body_lin_vel_w.ndim != 3
        or body_lin_vel_w.shape[0] != num_envs
        or body_lin_vel_w.shape[1] != len(body_names)
        or body_lin_vel_w.shape[2] != 3
    ):
        raise RuntimeError(
            "post_swing_settle_debt requires runtime body_lin_vel_w shaped [env,body,3]"
        )

    base_lin_speed = torch.norm(root_lin_vel, dim=-1)
    base_ang_speed = torch.norm(root_ang_vel, dim=-1)
    tilt_rad = torch.asin(
        torch.norm(projected_gravity[:, :2], dim=-1).clamp(max=1.0)
    )
    root_height_debt_m = torch.relu(
        nominal_root_z_m - root_height_deadband_m - root_pos[:, 2]
    )
    foot_speed_xy = torch.norm(body_lin_vel_w[:, foot_ids, :2], dim=-1).mean(dim=-1)
    tails = torch.stack(
        (
            _post_swing_settle_tail(base_lin_speed, base_lin_margin_mps, base_lin_scale_mps),
            _post_swing_settle_tail(base_ang_speed, base_ang_margin_radps, base_ang_scale_radps),
            _post_swing_settle_tail(tilt_rad, tilt_margin_rad, tilt_scale_rad),
            # The deadband already played the margin's role above, so the tail's margin is 0.
            _post_swing_settle_tail(root_height_debt_m, 0.0, root_height_scale_m),
            _post_swing_settle_tail(foot_speed_xy, foot_slip_margin_mps, foot_slip_scale_mps),
        ),
        dim=-1,
    )
    raw_value = torch.mean(tails, dim=-1)

    # Same-attempt recovery-window gate: identical mechanism (and default window) to
    # processed_qdes_slew_hinge — one shared clock, never a second bookkeeping of "post swing".
    cmd = _cmd(env, racket_command_name)
    clock = getattr(cmd, "post_strike_age_and_same_attempt", None)
    if not callable(clock):
        raise RuntimeError(
            "post_swing_settle_debt requires racket_target's same-attempt post-strike clock"
        )
    age_s, same_attempt = clock()
    if (
        not torch.is_tensor(age_s)
        or not torch.is_tensor(same_attempt)
        or tuple(age_s.shape) != (num_envs,)
        or tuple(same_attempt.shape) != (num_envs,)
        or same_attempt.dtype != torch.bool
    ):
        raise RuntimeError(
            "post_swing_settle_debt received an invalid same-attempt post-strike clock"
        )
    eligible = (
        same_attempt
        & age_s.ge(recovery_start_s)
        & age_s.le(recovery_end_s)
    )
    value = torch.where(eligible, raw_value, torch.zeros_like(raw_value))
    return value, eligible, tails


def _post_swing_settle_counter_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _POST_SWING_SETTLE_COUNTER_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "recovery_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "reward_enabled_eligible_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "gated_debt_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            **{
                f"gated_{name}_tail_sum": torch.zeros(
                    (), dtype=template.dtype, device=template.device
                )
                for name in _POST_SWING_SETTLE_COMPONENT_NAMES
            },
        }
        setattr(env, _POST_SWING_SETTLE_COUNTER_ATTR, state)
    return state


def _record_post_swing_settle_activation(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
    eligible: torch.Tensor,
    tails: torch.Tensor,
    *,
    reward_active: bool,
    signature: tuple,
) -> None:
    """Book one simulator step, idempotently sharing the probe and real term."""

    token = getattr(env, "common_step_counter", None)
    token = token if type(token) is int else None
    state = _post_swing_settle_counter_state(env, values)
    already = (
        token is not None
        and getattr(env, _POST_SWING_SETTLE_OBSERVED_STEP_ATTR, None) == token
    )
    if already:
        if getattr(env, _POST_SWING_SETTLE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError(
                "post-swing settle probe and RewardTerm used different parameters in one step"
            )
    else:
        mask = eligible.detach()
        state["observed_sample_count"].add_(values.numel())
        state["recovery_eligible_sample_count"].add_(mask.sum(dtype=torch.long))
        state["gated_debt_sum"].add_(values.detach().sum())
        gated_tails = tails.detach() * mask.unsqueeze(-1)
        for index, name in enumerate(_POST_SWING_SETTLE_COMPONENT_NAMES):
            state[f"gated_{name}_tail_sum"].add_(gated_tails[:, index].sum())
        if token is not None:
            setattr(env, _POST_SWING_SETTLE_OBSERVED_STEP_ATTR, token)
            setattr(env, _POST_SWING_SETTLE_SIGNATURE_ATTR, signature)
    if reward_active and (
        token is None or getattr(env, _POST_SWING_SETTLE_ACTIVE_STEP_ATTR, None) != token
    ):
        state["reward_enabled_eligible_sample_count"].add_(
            eligible.detach().sum(dtype=torch.long)
        )
        if token is not None:
            setattr(env, _POST_SWING_SETTLE_ACTIVE_STEP_ATTR, token)


def _post_swing_settle_signature(args: tuple) -> tuple:
    return tuple(
        float(value) if index >= 2 else value for index, value in enumerate(args)
    )


def post_swing_settle_debt_probe(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    base_lin_margin_mps: float = 0.30,
    base_lin_scale_mps: float = 0.20,
    base_ang_margin_radps: float = 0.50,
    base_ang_scale_radps: float = 0.30,
    tilt_margin_rad: float = 0.10,
    tilt_scale_rad: float = 0.10,
    nominal_root_z_m: float = _POST_SWING_SETTLE_NOMINAL_ROOT_Z_M,
    root_height_deadband_m: float = 0.05,
    root_height_scale_m: float = 0.05,
    foot_slip_margin_mps: float = 0.05,
    foot_slip_scale_mps: float = 0.10,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Measure the settle debts while contributing exact zero reward."""

    args = (
        racket_command_name,
        motion_command_name,
        base_lin_margin_mps,
        base_lin_scale_mps,
        base_ang_margin_radps,
        base_ang_scale_radps,
        tilt_margin_rad,
        tilt_scale_rad,
        nominal_root_z_m,
        root_height_deadband_m,
        root_height_scale_m,
        foot_slip_margin_mps,
        foot_slip_scale_mps,
        recovery_start_s,
        recovery_end_s,
    )
    values, eligible, tails = _post_swing_settle_debt_values(env, *args)
    _record_post_swing_settle_activation(
        env,
        values,
        eligible,
        tails,
        reward_active=False,
        signature=_post_swing_settle_signature(args),
    )
    return torch.zeros_like(values)


def post_swing_settle_debt(
    env: ManagerBasedRLEnv,
    racket_command_name: str = "racket_target",
    motion_command_name: str = "motion",
    base_lin_margin_mps: float = 0.30,
    base_lin_scale_mps: float = 0.20,
    base_ang_margin_radps: float = 0.50,
    base_ang_scale_radps: float = 0.30,
    tilt_margin_rad: float = 0.10,
    tilt_scale_rad: float = 0.10,
    nominal_root_z_m: float = _POST_SWING_SETTLE_NOMINAL_ROOT_Z_M,
    root_height_deadband_m: float = 0.05,
    root_height_scale_m: float = 0.05,
    foot_slip_margin_mps: float = 0.05,
    foot_slip_scale_mps: float = 0.10,
    recovery_start_s: float = 0.20,
    recovery_end_s: float = 1.55,
) -> torch.Tensor:
    """Negative-weight S1 bundle: five bounded settle debts in the same-swing recovery window."""

    args = (
        racket_command_name,
        motion_command_name,
        base_lin_margin_mps,
        base_lin_scale_mps,
        base_ang_margin_radps,
        base_ang_scale_radps,
        tilt_margin_rad,
        tilt_scale_rad,
        nominal_root_z_m,
        root_height_deadband_m,
        root_height_scale_m,
        foot_slip_margin_mps,
        foot_slip_scale_mps,
        recovery_start_s,
        recovery_end_s,
    )
    values, eligible, tails = _post_swing_settle_debt_values(env, *args)
    _record_post_swing_settle_activation(
        env,
        values,
        eligible,
        tails,
        reward_active=True,
        signature=_post_swing_settle_signature(args),
    )
    return values


def consume_post_swing_settle_debt_activation_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot/reset one PPO update's post-swing settle ledger."""

    template = env.scene["robot"].data.root_pos_w[:, 0]
    state = _post_swing_settle_counter_state(env, template)
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    # inference_mode (not no_grad): see consume_lower_body_wave_activation_counters.
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


def racket_strike_success(
    env: ManagerBasedRLEnv, command_name: str, std_pos: float, std_vel: float, std_normal: float
) -> torch.Tensor:
    """MULTIPLICATIVE strike success R_pos * R_vel * R_normal, gated to the LEGACY strike window
    (``strike_window_s``). Unlike the additive racket terms (which give partial credit for getting only
    position OR velocity right), the product is high ONLY when position AND velocity AND normal are all
    good at once — a true hit. RewTerm weight is POSITIVE.

    1c split windows deliberately do NOT narrow this term (R3b forensics 2026-07-08): this is a BONUS
    channel (reward_staged_design §D: landing/net/spin/success 不动——验证奖金不是引导), and the vrr
    scoring gates (exact_strike, vb capture) are window-independent too. The first 1c implementation
    reused the internally window-gated kernels, so the product's support collapsed to the window
    INTERSECTION = the ±0.02 s tight position window (3 frames @50 Hz instead of 13): the true-hit
    bonus lost ~17x income (R3b 0.0021 vs R1b 0.0350) exactly when the tight window had already cut the
    dense position money, the policy farmed the wide vel/normal channels instead of contact, and the
    forehand missed the vb capture gate every swing (hit rate 0, return rate ~0). The product now uses
    the UNGATED kernels x the legacy window: byte-identical when the windows are not split (bool win:
    win^3 == win), and under split windows contact precision is still enforced by the position kernel
    itself + the vb capture gate — not by shrinking the bonus support.
    The proximity power-gate is deliberately NOT passed down here: success is already multiplicative
    (the design keeps the big money on the ungated product)."""
    cmd = _cmd(env, command_name)
    valid = (
        _target_component_valid(cmd, "position"),
        _target_component_valid(cmd, "velocity"),
        _target_component_valid(cmd, "face"),
    )
    if not any(valid):
        return torch.zeros_like(cmd.time_to_strike)
    pos = _pos_kernel_raw(cmd, std_pos)
    vel = _vel_kernel_raw(cmd, std_vel)
    normal = _normal_kernel_raw(cmd, std_normal)
    # Invalid factors are neutral in this multiplicative bonus.  B1/B2 therefore earn a
    # position+face success bonus without any hidden desired-velocity dependence; C earns none.
    raw = (
        (pos if valid[0] else torch.ones_like(pos))
        * (vel if valid[1] else torch.ones_like(vel))
        * (normal if valid[2] else torch.ones_like(normal))
    )
    return raw * (cmd.strike_window & action_ball_task_valid_mask(cmd)).float()


def racket_guidance(env: ManagerBasedRLEnv, command_name: str, d_max: float = 0.5) -> torch.Tensor:
    """Constant guidance penalty toward the racket target (reward_staged_design 2026-07-08 §② B2):
    ``min(||racket_FK - target||, d_max)``, paid every pre-strike AND in-window step (union: from
    swing start through the strike window; the post-strike follow-through is untouched). This is
    the "挥拍到指定位置" gradient that exists even when the paddle is far outside every exp
    kernel's responsive band (the exp-starvation antidote); ``min(·, d_max)`` caps the burden so a
    far target can never drown the imitation signal (risk ⑤-1). Returns a POSITIVE magnitude —
    the RewTerm weight is NEGATIVE (set via rewards.racket_guidance_weight; cfg default 0.0 = off,
    the term is skipped). 人话:挥不到球也天天有"往哪挥"的工资单,小而恒。"""
    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "position"):
        return torch.zeros_like(cmd.time_to_strike)
    dist = torch.norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    active = (cmd.pre_strike | cmd.strike_window) & action_ball_task_valid_mask(cmd)
    return dist.clamp(max=float(d_max)) * active.float()


def racket_face_guidance(
    env: ManagerBasedRLEnv, command_name: str, theta_max: float = 1.5707963
) -> torch.Tensor:
    """Constant FACE-ANGLE guidance penalty (2026-07-10, M3c 死区解药): ``min(angle, theta_max)``
    between the achieved mount normal and the demanded face normal, paid every pre-strike AND
    in-window step (same active mask as ``racket_guidance``). The exp face kernel has ~zero
    gradient beyond ~3·std (M3c 卡在 33°、v5syn 反手起步 ~53° 都在死区里) — this linear term is
    the face-channel twin of the position guidance: a small constant "which way to turn the
    blade" wage that never starves. The (measured, target) pair comes from ``_face_pair`` — the
    SAME frame the exp kernel uses. It must, or the two face terms fight: the original inline pick
    read the sign-flipped ``racket_normal_w`` against the A-frame bank target, so on sign=-1
    (backhand) clips this linear term pulled the blade toward the WRONG face with live gradient —
    worse than the dead exp kernel it was meant to rescue (2026-07-09 病因定案 + R9u/M3d-live
    止损). Returns POSITIVE radians — the RewTerm weight is NEGATIVE
    (rewards.racket_face_guidance_weight; cfg default 0.0 = term skipped, byte-identical).
    NOTE theta_max defaults to pi/2: rescues starting deeper than 90° (M3b-type 116° dead-zone
    starts) must pass theta_max=pi, or the clamp zeroes the gradient exactly where it is needed.
    HOT-RESTART note (M3c/M2f-type checkpoints onto the fixed frame): with weight != 0 this term
    steps once at restart (pre-fix backhand read ~146°, clamped to the pi/2 constant with zero
    gradient; post-fix it reads the true ~0.6 rad with live gradient — a one-off reward-level
    shift of ~+0.98*|weight| per active step). Watch value_loss/KL and the new
    face_cmd_normal_error_deg metric for the first few hundred iterations.
    人话:拍面反了 90° 时 exp 核一分钱梯度都不给,这里每一度都扣一点——把反面的拍子一路拉回来。"""
    cmd = _cmd(env, command_name)
    if not _target_component_valid(cmd, "face"):
        return torch.zeros_like(cmd.time_to_strike)
    measured, target_normal = _face_pair(cmd)
    cos_ang = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    active = (cmd.pre_strike | cmd.strike_window) & action_ball_task_valid_mask(cmd)
    return angle.clamp(max=float(theta_max)) * active.float()


def racket_face_conditional_guidance(
    env: ManagerBasedRLEnv,
    command_name: str,
    theta_free: float = 0.262,
    theta_max: float = math.pi,
    pos_full: float = 0.075,
    pos_zero: float = 0.095,
    vel_full: float = 0.5,
    vel_zero: float = 1.0,
) -> torch.Tensor:
    """Fixed-budget signed-face correction that cannot reward abandoning readiness.

    The always-on linear face penalty helped angle while degrading position, velocity and completion.
    This term isolates the proposed remedy: within the wide strike window it spends one fixed cost
    budget, then converts that cost from a readiness deficit into signed-face error as the current
    swing-through position and velocity enter compact margins.  The face gradient is exactly zero
    outside either outer readiness margin; inside the corresponding acceptance margin the readiness
    gate is one, with a linear transition between the two:

    ``g(e; full, zero) = clamp((zero - e) / (zero - full), 0, 1)``

    ``face = clamp((face_angle - theta_free) / (theta_max - theta_free), 0, 1)``

    ``ready = g(pos_err) * g(vel_err)``

    ``penalty = window * (1 - ready * (1 - face))``

    Thus an unready strike-window state pays the fixed maximum cost, while a ready state pays only
    its face-error fraction.  Since ``d penalty / d ready = -(1 - face) <= 0``, improving position or
    velocity readiness can never increase the cost.  This avoids the inverse incentive in the naive
    ``ready * face`` formulation, where a negatively weighted policy could escape the face penalty by
    deliberately becoming less ready.  The term can still add readiness-aligned shaping; that is part
    of the integrated mechanism being ablated, not a claim of a face-only scalar reward.

    The defaults bind existing task contracts rather than adding tunable thresholds: 7.5 cm is the
    exact position-pass threshold, 9.5 cm is the virtual-ball capture radius, 0.5 m/s is the exact
    velocity-pass threshold, and 1.0 m/s is twice that tolerance.  The angular free band is 15 degrees.
    Therefore the return is dimensionless in ``[0, 1]`` and the non-positive RewTerm weight is the exact
    maximum per-strike-window-step penalty budget.  It reads the same ``_face_pair`` as every other
    face reward.

    This is task shaping, never a safety credit: it is non-positive after weighting and changes no
    termination, collision, joint/torque/qdot limit, observation, action, or plant contract.  Safety
    regressions remain non-compensable experiment failures.
    """
    values = {
        "theta_free": theta_free,
        "theta_max": theta_max,
        "pos_full": pos_full,
        "pos_zero": pos_zero,
        "vel_full": vel_full,
        "vel_zero": vel_zero,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError(f"racket_face_conditional_guidance requires finite bounds, got {values}")
    if not 0.0 <= float(theta_free) < float(theta_max) <= math.pi:
        raise ValueError(
            "racket_face_conditional_guidance requires 0 <= theta_free < theta_max <= pi"
        )
    if not 0.0 < float(pos_full) < float(pos_zero):
        raise ValueError("racket_face_conditional_guidance requires 0 < pos_full < pos_zero")
    if not 0.0 < float(vel_full) < float(vel_zero):
        raise ValueError("racket_face_conditional_guidance requires 0 < vel_full < vel_zero")

    cmd = _cmd(env, command_name)
    if not all(
        _target_component_valid(cmd, component)
        for component in ("position", "velocity", "face")
    ):
        return torch.zeros_like(cmd.time_to_strike)
    measured, target_normal = _face_pair(cmd)
    cos_ang = torch.sum(measured * target_normal, dim=-1).clamp(-1.0, 1.0)
    # atan2(sin, cos) keeps the near-180-degree gradient finite; acos has an infinite derivative
    # at +/-1 and can turn an otherwise-zero compact gate into 0*NaN during backpropagation.
    sin_ang = torch.linalg.vector_norm(torch.cross(measured, target_normal, dim=-1), dim=-1)
    angle = torch.atan2(sin_ang, cos_ang)
    face_fraction = ((angle - float(theta_free)) / (float(theta_max) - float(theta_free))).clamp(
        0.0, 1.0
    )

    target_pos_now = _target_position_now(cmd)
    pos_err = torch.norm(cmd.racket_pos_w - target_pos_now, dim=-1)
    vel_err = torch.norm(cmd.racket_lin_vel_w - cmd.racket_target_vel_w, dim=-1)
    pos_gate = ((float(pos_zero) - pos_err) / (float(pos_zero) - float(pos_full))).clamp(0.0, 1.0)
    vel_gate = ((float(vel_zero) - vel_err) / (float(vel_zero) - float(vel_full))).clamp(0.0, 1.0)
    readiness = pos_gate * vel_gate
    event_window = _window_wide(cmd).float()
    active_face_gate = readiness * event_window

    # Spend a fixed budget in the exogenous strike-time window, then grant relief only when readiness
    # and face correctness coexist.  For every face_fraction in [0,1], greater readiness weakly lowers
    # this cost; leaving the readiness gate can never be a way to evade the penalty.
    penalty_fraction = event_window * (1.0 - readiness * (1.0 - face_fraction))

    # Directional observability for +200 screening.  These metrics exist only when the default-off
    # term is evaluated, so a purported treatment whose gate/reward never executes is visible.
    cmd.metrics["face_conditional_guidance_gate"] = active_face_gate
    cmd.metrics["face_conditional_guidance_error_fraction"] = face_fraction
    cmd.metrics["face_conditional_guidance_cost_fraction"] = penalty_fraction
    return penalty_fraction


def tracking_envelope_violation(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str],
    ignore_hold: bool = False,
) -> torch.Tensor:
    """R-b envelope-as-penalty (reward_staged_design 2026-07-08 §⑥): per-step indicator of the
    tracking-envelope violation that used to TERMINATE the episode — the union of the two removed
    terminations, with the SAME z-only expressions (terminations.bad_anchor_pos_z_only |
    bad_motion_body_pos_z_only over the feet+wrists list). Returns 1.0 while violating, else 0.0;
    the RewTerm weight is NEGATIVE (terminations.envelope_penalty_weight, e.g. -1.0 => -0.02/step
    @50 Hz), so standing in the violation zone costs money instead of ending the episode.
    ``command_name`` is the MOTION command ("motion"). 人话:跟丢参考不再判死,改成站在违规区里
    每秒扣钱。Weight 0.0 (cfg default) = term skipped, byte-identical."""
    from whole_body_tracking.tasks.tracking.mdp.terminations import (
        bad_anchor_pos_z_only,
        bad_motion_body_pos_z_only,
    )

    viol = bad_anchor_pos_z_only(env, command_name, threshold) | bad_motion_body_pos_z_only(
        env, command_name, threshold, body_names
    )
    if ignore_hold:
        viol = _ignore_hold(env.command_manager.get_term(command_name), viol, True)
    return viol.float()


# ============================================================================================== #
# Tier-1 VIRTUAL-BALL outcome terms (rewardDesign.md). One-shot: non-zero ONLY on the exact-strike
# step of envs that passed the capture gate (cmd.vb_fired, set by RacketTargetCommand._vb_evaluate
# from the venue-fitted contact + coarse landing rollout). All are inert (all-zero) unless
# commands.racket_target.virtual_ball is enabled. Anti-farming gates follow the adversarial
# verification (verify_tier1-reward-soundness.md (c)):
#   1. the in-bounds bonus requires landing depth > net_x + vb_min_landing_depth (dink guard),
#   2. the capture gate requires a minimum paddle approach speed (phantom-block guard, in _vb_evaluate),
#   3. the pass_net CLEAR BONUS pays only for shots that also land legally (net-without-landing
#      guard); its height KERNEL is deliberately ungated shaping — see virtual_pass_net docstring.
# ============================================================================================== #
def action_rate_l2_clamped(env: ManagerBasedRLEnv, value_clamp: float = 9.0) -> torch.Tensor:
    """一阶动作平滑罚的值封顶版(v2,fresh 臂自杀区间的解)。

    人话:与 isaaclab 内置 action_rate_l2 同式 ‖a_t−a_{t−1}‖²,但每 env 封顶 value_clamp。
    fresh 随机策略一步能把 31 维动作甩出 ‖Δa‖²≈60+,×(−0.2) 就是每步 −12+——比模仿收入
    上限大一个量级,早期净流为负 → 摔死最优。封顶后 raw 加权幅度 = 0.2×9 = 1.8；
    Isaac RewardManager 再乘 policy dt=0.02，所以进入 env reward 的单步最坏值是 0.036，
    不得把 raw 幅度误写成每步收入。clamp 内梯度原样、clamp 外为零(超大偏差是噪声,
    不需要按比例更狠)。档位 9.0 来自冻结表(预算上限与实测 p95 取小)。RewTerm weight
    用负数;默认 weight=0 = 项被跳过,字节等价。v1 的无封顶 action_rate_l2 照旧存在,
    v2 包把它归零并启用本项——静态、value 平稳,不是 schedule。
    """
    clamp = float(value_clamp)
    if not math.isfinite(clamp) or clamp <= 0.0:
        raise ValueError("action_rate_l2_clamped value_clamp must be finite and positive")
    mgr = env.action_manager
    diff = mgr.action - mgr.prev_action
    return torch.sum(torch.square(diff), dim=1).clamp(max=clamp)


def strike_capture_bonus(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """L2 击中层的一次性奖金(v2 记账 redesign §3.5):capture 门过的那一步付 1,其余恒 0。

    人话:每挥拍最多发一次的"击中大奖"——判据就是现成的 vb_fired 捕获门(exact-strike
    一步 & 拍面正确半球 & 位置误差 < vb_capture_radius & 沿法向进拍速 > 下限;一拍锁存,
    同一拍不会重复发)。"打没打上"语义上不可再分,所以它是全栈唯一的大额 one-shot
    (方差论证 redesign §2 B5:质量类信号都走窗口 dense,唯独击中判据允许 one-shot)。
    与 virtual_* 三项的区别:那三项还要 rollout 结果(过网/落台/旋转),本项只认"击中",
    给采集期一个不依赖出球质量的里程碑。RewTerm weight 用正数;量级 B 按 §2 定权公式
    (名义 ~850,probe 校准后冻结进 prereg);默认 weight=0 = 项被跳过,字节等价。
    """
    cmd = _cmd(env, command_name)
    fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
    return fired.float()


def virtual_pass_net(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Net-height shaping at the virtual net-plane crossing + fully-gated clear bonus.

    The Gaussian kernel on (net-crossing height - (net_top + margin)) pays for ANY shot that
    reaches the net plane inside the rollout horizon (v0 ``pass_net_margin`` semantics): it is the
    CLIMB gradient that teaches a flat-hitting policy to angle shots upward. Gating it on a legal
    landing (this term's original verify (c)4 reading) starved training completely — the E-champion
    warm-start crosses the net legally on only ~0.2% of strikes, so 2.5k iterations of vb_warmE14k3
    paid exactly zero virtual reward (2026-07-03 incident). The farming surface is bounded: the
    kernel requires an actual net-plane crossing, maxes only at the correct height, and is worth at
    most 1/swing; anti-farming gates stay in full on the +0.5 clear bonus here and on the
    landing/spin terms. RewTerm weight POSITIVE.
    """
    cmd = _cmd(env, command_name)
    if bool(getattr(cmd, "_counter_rally_enabled", False)):
        # Exact N=1 uses the single staged, objective-bound rally reward.
        # Keeping this legacy quality term active would double-score the same
        # post-paddle trajectory under a different coarse integrator.
        return torch.zeros_like(cmd.vb_net_z)
    target_z = cmd._vb_net_top_z + float(cmd.cfg.vb_net_margin)
    err = cmd.vb_net_z - target_z
    kernel = torch.exp(-(err**2) / float(cmd.cfg.vb_net_sigma) ** 2)
    legal = cmd.vb_net_clear & cmd.vb_landing_valid & cmd.vb_on_opponent
    raw = kernel * cmd.vb_net_crossed.float() + 0.5 * legal.float()
    fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
    return raw * fired.float()


def virtual_landing_dense_actual_contact(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Small landing-distance gradient after a valid achieved paddle contact.

    Unlike ``virtual_landing(mode='legal_base')`` this term does not require an already-legal
    return.  It pays only when the swept selected-rubber contact produced a finite achieved flight
    and that flight reached a valid landing plane; misses and hypothetical target trajectories get
    exactly zero.  Its weight is deliberately kept far below the legal-table prize.
    """

    cmd = _cmd(env, command_name)
    target_xy = getattr(cmd, "_vb_target_xy_per_env", None)
    if target_xy is None:
        target_xy = cmd._vb_target_xy.unsqueeze(0)
    if (
        target_xy.ndim != 2
        or target_xy.shape[1:] != cmd.vb_landing_xy.shape[1:]
        or target_xy.shape[0] not in (1, cmd.vb_landing_xy.shape[0])
    ):
        raise RuntimeError(
            "virtual_landing_dense_actual_contact target buffer must be "
            "broadcastable as [1,2] or match [num_envs,2] landing positions"
        )
    dist2 = torch.sum(torch.square(cmd.vb_landing_xy - target_xy), dim=-1)
    kernel = torch.exp(-dist2 / float(cmd.cfg.vb_landing_sigma) ** 2)
    fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
    return kernel * cmd.vb_landing_valid.float() * fired.float()


_LANDING_PRIZE_PENDING_ATTR = "_hope_landing_prize_pending"
_LANDING_PRIZE_ARMED_ATTR = "_hope_landing_prize_armed"


def virtual_landing(
    env: ManagerBasedRLEnv,
    command_name: str,
    mode: str = "climb",
    base_frac: float = 0.6,
    settle_delay_s: float = 0.0,
) -> torch.Tensor:
    """Landing-accuracy kernel + fully-gated in-bounds bonus (v0 ``landing_in_opponent_half``).

    CLIMB-PHASE shape (2026-07-04): the Gaussian kernel on ||landing_xy - target_xy|| pays for any
    landing inside the rollout horizon — NOT gated on net clearance. The E-warm-started policy
    lands ~1.9 m short of the target and reaches the net plane on only a few % of strikes, so both
    net-gated terms stayed ~zero for 5k+ iterations (vb_warmE14k3/4); this kernel is the dense
    bottom rung that pays for hitting DEEPER. Net-farming risk is bounded: the rollout has no net
    collider, so the kernel is smooth through the net plane with its single max AT the target —
    drilling the net base (err ~0.75 m) always pays less than clearing and landing deeper. The
    +1.0 bonus keeps the full gate: net clearance AND on-opponent AND depth past
    net_x + vb_min_landing_depth (verify (c)1 dink guard). Re-tighten (restore the net_clear gate
    on the kernel, sigma back toward 0.3) once virtual_net_clear_rate is healthy. RewTerm weight
    POSITIVE.
    """
    cmd = _cmd(env, command_name)
    target_xy = getattr(cmd, "_vb_target_xy_per_env", None)
    if target_xy is None:
        target_xy = cmd._vb_target_xy.unsqueeze(0)
    if (
        target_xy.ndim != 2
        or target_xy.shape[1:] != cmd.vb_landing_xy.shape[1:]
        or target_xy.shape[0] not in (1, cmd.vb_landing_xy.shape[0])
    ):
        raise RuntimeError(
            "virtual_landing target buffer must be broadcastable as "
            "[1,2] or match [num_envs,2] landing positions"
        )
    dist2 = torch.sum(
        torch.square(cmd.vb_landing_xy - target_xy), dim=-1
    )
    kernel = torch.exp(-dist2 / float(cmd.cfg.vb_landing_sigma) ** 2)
    if mode == "legal_base":
        # v2.2(Franco 07-25 裁定):上台组唯一保留项。"过网+落在对面台"是【先决条件】
        # (gate),不是单独给钱的项;门内任意落点都有可观底薪 base_frac,加 (1-base) 的
        # 中心核当梯度(台角距台心 ~1.0 m,σ=1.0 下台内最低 ~base+0.4e⁻¹≈0.75)——
        # "台上任意位置可观、梯度合理、量级有界 [0,1]"。旧 depth 奖金删除(底薪替代;
        # 吸血小球落点离台心远,自动只拿底薪档)。07-03 饿死史的新缓解:质量三核已是
        # 60/45/35+σ 自适应(不是当年 4/0.5/0.5),probe 必须盯 legal rate 起飞。
        base = float(base_frac)
        if not 0.0 < base < 1.0:
            raise ValueError("virtual_landing base_frac must be in (0, 1)")
        if bool(getattr(cmd, "_counter_rally_enabled", False)):
            terms = getattr(cmd, "_counter_rally_reward_terms", None)
            if (
                not isinstance(terms, torch.Tensor)
                or terms.ndim != 2
                or terms.shape != (cmd.vb_fired.numel(), 5)
            ):
                raise RuntimeError(
                    "counter-rally reward cache must have shape [num_envs,5]"
                )
            raw = terms[:, 4]
        else:
            legal = (
                cmd.vb_landing_valid
                & cmd.vb_net_clear
                & cmd.vb_on_opponent
            )
            raw = legal.float() * (base + (1.0 - base) * kernel)
        delay = float(settle_delay_s)
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError("virtual_landing settle_delay_s must be finite and >= 0")
        if delay == 0.0:
            fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
            return raw * fired.float()
        # 延付制(2026-07-26,对照臂 3k 迭代实测抓到的重生刷分漏洞的解):大奖不在触球步
        # 立发,而是【触球后 settle_delay_s 内同一 attempt 存活】才发;死亡/重置/换题没收。
        # 人话:上台且站得住才算数——RSI 重生每次都出生在参考挥拍中段,立发制下"借参考
        # 动量打一板→摔死→重生再打"的回合越短收益率越高(实测回合 18→7 步、摔倒×10);
        # 延付把死亡从"结算加速器"变回"没收器",且不误伤学站阶段(没打上本就无奖可没收)。
        # 工程约束:planner 重定时下随挥 ~21 步即 wrap(wrap 亦终结 attempt),delay 必须
        # 显著小于该窗(默认包用 0.24 s = 12 步);probe 盯 landing 实付率验证不被 wrap 误没收。
        clock = getattr(cmd, "post_strike_age_and_same_attempt", None)
        if clock is None:
            raise RuntimeError(
                "virtual_landing settle_delay_s>0 requires the command term's "
                "post_strike_age_and_same_attempt clock"
            )
        age_s, same_attempt = clock()
        pending = getattr(env, _LANDING_PRIZE_PENDING_ATTR, None)
        if pending is None:
            pending = torch.zeros_like(raw)
            armed = torch.zeros_like(cmd.vb_fired)
            setattr(env, _LANDING_PRIZE_PENDING_ATTR, pending)
            setattr(env, _LANDING_PRIZE_ARMED_ATTR, armed)
        else:
            armed = getattr(env, _LANDING_PRIZE_ARMED_ATTR)
        # attempt 终结(死/重置/换题)→ 没收;新触球 → 覆盖挂账;到期且存活 → 发放并解挂
        armed_new = armed & same_attempt
        fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
        pending_new = torch.where(fired, raw, pending)
        armed_new = armed_new | fired
        pay = armed_new & same_attempt & (age_s >= delay)
        out = pending_new * pay.float()
        armed_new = armed_new & ~pay
        pending.copy_(pending_new)
        armed.copy_(armed_new)
        return out
    if mode != "climb":
        raise ValueError(
            "virtual_landing mode must be 'climb' (v1 byte-identical) or 'legal_base' (v2.2)"
        )
    if bool(getattr(cmd, "_counter_rally_enabled", False)):
        terms = getattr(cmd, "_counter_rally_reward_terms", None)
        if (
            not isinstance(terms, torch.Tensor)
            or terms.ndim != 2
            or terms.shape != (cmd.vb_fired.numel(), 5)
        ):
            raise RuntimeError(
                "counter-rally reward cache must have shape [num_envs,5]"
            )
        return terms[:, 4] * action_ball_task_valid_mask(cmd).float()
    bonus = (cmd.vb_landing_valid & cmd.vb_net_clear & cmd.vb_on_opponent & cmd.vb_depth_ok).float()
    raw = kernel * cmd.vb_landing_valid.float() + bonus
    fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
    return raw * fired.float()


def virtual_spin(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Outgoing-topspin reward (Ace's ws-term), only for shots that land legally.

    ``clamp(topspin / vb_spin_ref, 0, 1)`` where topspin is omega_plus projected on z_hat x d_hat
    of the outgoing direction; gated on a valid net-clearing in-bounds landing so brushing wild
    swipes that miss the table cannot farm spin. RewTerm weight POSITIVE (ramp toward parity with
    landing per the Ace precedent once the wiring is validated).
    """
    cmd = _cmd(env, command_name)
    if bool(getattr(cmd, "_counter_rally_enabled", False)):
        return torch.zeros_like(cmd.vb_topspin)
    legal = cmd.vb_landing_valid & cmd.vb_net_clear & cmd.vb_on_opponent
    if getattr(cmd.cfg, "vb_spin_mode", "topspin") == "minimize":
        # Stage-1 placement-first semantics (franco 2026-07-04): the BEST shot kills the incoming
        # spin — reward small outgoing |omega|, not topspin generation (which is ball quality and
        # deliberately unrewarded in stage 1).
        kernel = torch.exp(-(cmd.vb_spin_out_norm / float(cmd.cfg.vb_spin_min_sigma)) ** 2)
        raw = kernel * legal.float()
    else:
        raw = (cmd.vb_topspin / float(cmd.cfg.vb_spin_ref)).clamp(0.0, 1.0) * legal.float()
    fired = cmd.vb_fired & action_ball_task_valid_mask(cmd)
    return raw * fired.float()


# --- footwork penalties (feet may STEP; we only punish BAD foot behaviour) --------------------- #
def foot_slip_sq(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot slip while in contact: sum over feet of contact * ||foot_xy_velocity||² (always on).
    A planted/landing foot should not skate. Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_slip_sq


def foot_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize excessive/violent foot velocity: sum over feet of ||foot_velocity||². Lets the foot step
    but discourages flailing. Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_vel_sq


def foot_drag(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot dragging: lateral foot speed while the foot is near the ground (skimming instead of
    lifting cleanly to step). Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_drag


# --- mjlab-ported foot-contact shaping (soft landing / swing clearance; DEFAULT OFF) ----------- #
# 与上面近亲们的分工(同一只脚,四种坏行为各罚各的,不重复计费):
#   * foot_slip_sq / foot_drag(常开):触地脚还在水平蹭/滑 —— 管切向速度;
#   * strike_foot_vel(击球窗):击球瞬间脚在动 —— 管击球稳定;
#   * foot_soft_landing(本组新增):落地那一步法向冲击力超标 —— 管"踩得多重",mjlab
#     soft_landing 思想;唯一读接触力大小的项;
#   * foot_clearance(本组新增):摆动相(不接触)脚又低又快地扫 —— 管"抬脚高度",mjlab
#     foot_clearance 思想;唯一管腾空脚高度的项,给"允许跨步"臂用,站立击球默认不开。
# mjlab 原版这两项都挂"速度指令门"(站立指令时自动关);我们的站立击球场景没有速度指令、
# 落地/抬脚行为全程都要管,所以刻意【不搬那个门】——项本身常开,默认由 weight=0 关断。
def foot_soft_landing(
    env: ManagerBasedRLEnv, sensor_cfg, force_threshold_n: float = 300.0
) -> torch.Tensor:
    """落地冲击惩罚(mjlab soft_landing 思想):first-contact 步的法向峰值力超阈部分,有界。

    人话:脚刚落地的那一个控制步,如果地面顶脚的竖直力比阈值大,超出多少罚多少(按阈值归一,
    并封顶),教会它轻轻放脚而不是砸下去。只在 first-contact 步计费——之后站着的支撑力再大也
    不管(那是体重,不是冲击)。A3 整机 58.2 kg(urdf mass.txt 求和)-> 静态重力约 571 N,双脚
    分摊每脚约 285 N;默认阈值 300 N ≈ "单脚超过静态支撑就算砸"。RewTerm weight 用负数;
    默认 weight=0 = 项被 RewardManager 跳过,字节等价。
    """
    threshold = _finite_scalar(
        force_threshold_n, name="foot_soft_landing force_threshold_n", positive=True
    )
    sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if not isinstance(body_ids, (list, tuple)) or len(body_ids) != 2:
        raise RuntimeError(
            "foot_soft_landing requires sensor_cfg resolved to exactly the two A3 feet"
        )
    body_ids = list(body_ids)
    first_contact_fn = getattr(sensor, "compute_first_contact", None)
    if not callable(first_contact_fn):
        raise RuntimeError(
            "foot_soft_landing requires a ContactSensor with track_air_time=True "
            "(compute_first_contact unavailable)"
        )
    # first_contact: 本控制步内"空中 -> 接触"的脚(需要 sensor cfg track_air_time=True,
    # tracking_env_cfg 的 contact_forces 已开)。
    first_contact = first_contact_fn(env.step_dt)[:, body_ids]
    forces_hist = getattr(sensor.data, "net_forces_w_history", None)
    if (
        not torch.is_tensor(forces_hist)
        or forces_hist.ndim != 4
        or forces_hist.shape[-1] != 3
    ):
        raise RuntimeError(
            "foot_soft_landing requires net_forces_w_history [env,hist,body,3] "
            "(ContactSensorCfg.history_length >= 1)"
        )
    if first_contact.dtype != torch.bool or tuple(first_contact.shape) != (
        forces_hist.shape[0],
        2,
    ):
        raise RuntimeError("foot_soft_landing first-contact mask must be [env,2] bool")
    # 法向 = 世界 z(平地场景);取 history 里的峰值——decimation 下冲击尖峰只存在于某个
    # 物理子步,只看当前帧会漏掉真正的最大冲击。拉扯(负 z)不算冲击,先 clamp 到 0。
    normal_peak = forces_hist[:, :, body_ids, 2].clamp(min=0.0).amax(dim=1)  # (E,2)
    # 有界惩罚(clip 防爆):超阈部分按阈值归一,单脚封顶 3.0(默认阈值下即 >=1200 N,
    # 约 2 倍整机重量,再大也同罚——一次暴力落地不该贡献天文数字梯度)。
    excess_frac = ((normal_peak - threshold) / threshold).clamp(min=0.0, max=3.0)
    return (excess_frac * first_contact.float()).sum(dim=-1)


def foot_clearance(
    env: ManagerBasedRLEnv, sensor_cfg, asset_cfg, target_m: float = 0.08
) -> torch.Tensor:
    """摆动相抬脚不足惩罚(mjlab foot_clearance 思想,2026-07-25 起单侧):非接触脚
    clamp(目标高-脚高, min=0) x 水平速度。

    人话:脚在空中横着移动时,抬得比"最低离地高度"低多少、移动越快,罚得越多——快速迈步
    必须把脚抬够,贴地扫着走(绊倒前兆)最贵;抬得比目标高不罚。(2026-07-25 前是双侧
    |脚高-目标高|:与本项自述的"最低离地要求"意图相反,还会在凸包顶上主动把脚往下按——
    粗糙地形上恰好帮倒忙,combo_fresh 类 rough 臂首当其冲;单侧化后多抬永远免费。)
    慢慢挪或已达标则几乎免费。触地的脚完全不管(那是 foot_slip_sq/foot_drag 的地盘)。
    注意"脚高"用的是 ankle_roll link 原点的世界 z,不是相对脚下地面的净空:站立贴地时
    原点约 0.07 m(见 foot_drag 的 Phase A 注释),默认 0.08 m 即鞋底约 1 cm 的最低离地
    要求。粗糙地形(HfRandomUniform 是绝对抬升带 [lo,hi],不是 ±抖动)下高包处的有效
    净空要求会缩水最多 hi,要保底应把 target_m 调到 hi + 0.07 + 期望鞋底净空;真要它
    抬腿跨步应往 0.15 m 以上调。RewTerm weight 用负数;默认 weight=0 = 项被跳过,字节等价。
    """
    target = _finite_scalar(target_m, name="foot_clearance target_m", positive=True)
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    sensor_ids = getattr(sensor_cfg, "body_ids", None)
    asset_ids = getattr(asset_cfg, "body_ids", None)
    sensor_names = list(getattr(sensor_cfg, "body_names", None) or ())
    asset_names = list(getattr(asset_cfg, "body_names", None) or ())
    # 接触状态(sensor 索引)和运动学(articulation 索引)必须逐脚配对:两份 cfg 解析出的
    # body 名单必须一模一样(含顺序),否则左脚的接触配右脚的速度,静默算错。
    if (
        not isinstance(sensor_ids, (list, tuple))
        or len(sensor_ids) != 2
        or not isinstance(asset_ids, (list, tuple))
        or len(asset_ids) != 2
        or len(sensor_names) != 2
        or sensor_names != asset_names
    ):
        raise RuntimeError(
            "foot_clearance requires sensor_cfg/asset_cfg resolved to the SAME two feet "
            "in the same order (per-foot contact state must pair with per-foot kinematics)"
        )
    sensor_ids = list(sensor_ids)
    asset_ids = list(asset_ids)
    forces = getattr(sensor.data, "net_forces_w", None)
    pos = getattr(asset.data, "body_pos_w", None)
    vel = getattr(asset.data, "body_lin_vel_w", None)
    if not torch.is_tensor(forces) or not torch.is_tensor(pos) or not torch.is_tensor(vel):
        raise RuntimeError(
            "foot_clearance requires net contact forces and foot position/velocity tensors"
        )
    # 10 N = sensor 的 force_threshold,与 hope_commands 的 in_contact 判据一致。
    in_contact = torch.norm(forces[:, sensor_ids, :], dim=-1) > 10.0  # (E,2)
    foot_z = pos[:, asset_ids, 2]  # (E,2)
    xy_speed = torch.norm(vel[:, asset_ids, :2], dim=-1)  # (E,2)
    swing = (~in_contact).float()
    # 单侧:只罚抬脚不足(target - foot_z 的正部),多抬不罚——"最低离地要求"语义。
    return (swing * (target - foot_z).clamp(min=0.0) * xy_speed).sum(dim=-1)


# --- mjlab-ported action second-difference smoothing (action_acc_l2; DEFAULT OFF) -------------- #
def action_acc_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos", value_clamp: float | None = None
) -> torch.Tensor:
    """动作二阶差分惩罚(mjlab action_acc_l2 思想):||a_t - 2·a_{t-1} + a_{t-2}||²。

    人话:action_rate 罚"步子迈多大"(一阶差分),这条罚"方向掉头多猛"(二阶差分)——高频抖
    (chatter)恰恰是一阶小、二阶大的信号,是平滑轴上的正交新轴。a 用 raw 动作(actor 归一化
    输出,与 isaaclab action_rate_l2 同源同量纲);a_{t-2} isaaclab 不存,由
    ClampedJointPositionAction 的自存缓冲提供——reset 清零且带有效位,复位后前两步历史不齐,
    这两步不计费(episode 边界永远造不出虚构的"掉头"罚)。剂量注意:二阶差分量纲大于一阶,
    起步取 action_rate 档位的 1/5~1/2(mjlab 采纳文档 §4),别按一阶惯用值抄。
    Positive magnitude; RewTerm weight 用负数;默认 weight=0 = 项被 RewardManager 跳过,
    字节等价。
    """
    term = env.action_manager.get_term(action_name)
    current = getattr(term, "raw_actions", None)
    previous = getattr(term, "prev_raw_actions", None)
    before_previous = getattr(term, "prev_prev_raw_actions", None)
    if (
        not torch.is_tensor(current)
        or current.ndim != 2
        or not torch.is_tensor(previous)
        or previous.shape != current.shape
        or not torch.is_tensor(before_previous)
        or before_previous.shape != current.shape
    ):
        raise RuntimeError(
            "action_acc_l2 requires ClampedJointPositionAction raw-action history "
            "(raw_actions / prev_raw_actions / prev_prev_raw_actions, same [env,joint] shape)"
        )
    valid = getattr(term, "raw_action_history_valid", None)
    if (
        not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != (tuple(current.shape)[0],)
    ):
        raise RuntimeError(
            "action_acc_l2 requires the per-env raw-action history validity mask "
            "(reset-aware [env] bool; the first two post-reset steps must be free)"
        )
    second_diff = current - 2.0 * previous + before_previous
    out = torch.sum(torch.square(second_diff), dim=-1) * valid.float()
    if value_clamp is not None:
        clamp = float(value_clamp)
        if not math.isfinite(clamp) or clamp <= 0.0:
            raise ValueError("action_acc_l2 value_clamp must be finite and positive")
        out = out.clamp(max=clamp)  # 值封顶(v2,fresh 自杀区间;同 action_rate_l2_clamped 理由)
    return out


_ACTION_ACC_PROBE_STATE_ATTR = "_hope_action_acc_jerk_probe_counters"
_ACTION_ACC_PROBE_STEP_ATTR = "_hope_action_acc_jerk_probe_step"
_ACTION_ACC_PROBE_SIGNATURE_ATTR = "_hope_action_acc_jerk_probe_signature"


def _action_acc_probe_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _ACTION_ACC_PROBE_STATE_ATTR, None)
    if state is None:
        state = {
            "observed_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "history_valid_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "nonfinite_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "above_clamp_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "raw_jerk_square_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "clamped_jerk_square_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "raw_jerk_square_max": torch.zeros((), dtype=template.dtype, device=template.device),
        }
        setattr(env, _ACTION_ACC_PROBE_STATE_ATTR, state)
    return state


def action_acc_jerk_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    value_clamp: float = 36.0,
) -> torch.Tensor:
    """Measure action jerk at RewardManager time while contributing exactly zero reward.

    The explicit probe flag installs this term with manager weight one so RewardManager cannot
    optimize it away.  The function itself always returns zeros.  It keeps raw and v2-clamped
    magnitudes in device scalar counters and never copies a per-environment tensor to the host.
    """

    clamp = float(value_clamp)
    if not math.isfinite(clamp) or clamp <= 0.0:
        raise ValueError("action_acc_jerk_probe value_clamp must be finite and positive")
    term = env.action_manager.get_term(action_name)
    raw = action_acc_l2(env, action_name=action_name, value_clamp=None)
    valid = term.raw_action_history_valid
    token = getattr(env, "common_step_counter", None)
    signature = (str(action_name), clamp)
    if type(token) is int and getattr(env, _ACTION_ACC_PROBE_STEP_ATTR, None) == token:
        if getattr(env, _ACTION_ACC_PROBE_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError("action_acc_jerk_probe parameters changed within one simulator step")
        return torch.zeros_like(raw)

    state = _action_acc_probe_state(env, raw)
    finite_valid = valid & torch.isfinite(raw)
    safe_raw = torch.where(finite_valid, raw, torch.zeros_like(raw))
    state["observed_sample_count"].add_(raw.numel())
    state["history_valid_sample_count"].add_(valid.sum(dtype=torch.long))
    state["nonfinite_sample_count"].add_((valid & ~torch.isfinite(raw)).sum(dtype=torch.long))
    state["above_clamp_sample_count"].add_((finite_valid & raw.gt(clamp)).sum(dtype=torch.long))
    state["raw_jerk_square_sum"].add_(safe_raw.sum())
    state["clamped_jerk_square_sum"].add_(safe_raw.clamp(max=clamp).sum())
    state["raw_jerk_square_max"].copy_(
        torch.maximum(state["raw_jerk_square_max"], safe_raw.max())
    )
    if type(token) is int:
        setattr(env, _ACTION_ACC_PROBE_STEP_ATTR, token)
        setattr(env, _ACTION_ACC_PROBE_SIGNATURE_ATTR, signature)
    return torch.zeros_like(raw)


def consume_action_acc_jerk_probe_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot and reset one PPO update of explicit jerk-probe device scalars."""

    action = env.action_manager.get_term("joint_pos")
    template = getattr(action, "raw_actions", None)
    if not torch.is_tensor(template) or template.ndim != 2:
        raise RuntimeError("action_acc_jerk_probe consumer requires joint_pos raw actions")
    state = _action_acc_probe_state(env, template[:, 0])
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


def arm_overreach(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Anti-arm-only: penalize solving the target by maxing the arm out — fraction of ARM joints within
    10% of a position limit. Encourages using the body/legs to bring the target into a comfortable arm
    range instead of stretching. Positive in [0,1]; RewTerm weight is negative."""
    return _cmd(env, command_name).arm_overreach_frac


def prestrike_waist_twist(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Anti twist-instead-of-step: penalize |waist_yaw|+|waist_roll| deviation from neutral BEFORE the
    strike. Widening the racket-target box alone did NOT force footwork — the policy just rotated its
    torso (waist yaw/roll) to face a lateral target while its feet stayed planted (arm_overreach stayed
    ~0, legs frozen). This term makes that twist costly during the approach, so getting behind a far
    target requires STEPPING. Gated by ``pre_strike`` ONLY (the strike swing's rotation is untouched) and
    ``waist_pitch`` is excluded (that is the swing wind-up / lean, not a lateral-reach cheat). Returns a
    positive magnitude (radians); the RewTerm weight is negative."""
    cmd = _cmd(env, command_name)
    return cmd.waist_twist * cmd.pre_strike.float()


# --- strike-window stability (penalize wobble/bob/skate AT the hit; gated to the strike window) - #
def strike_proj_grav_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize base tilt (||projected_gravity_xy||) DURING the strike window — be upright at the hit."""
    cmd = _cmd(env, command_name)
    return cmd.proj_grav_xy * cmd.strike_window.float()


def strike_base_ang_vel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize base roll/pitch rate (||base_ang_vel_xy||) DURING the strike window."""
    cmd = _cmd(env, command_name)
    return cmd.base_ang_vel_xy_norm * cmd.strike_window.float()


def prestrike_proj_grav_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sim2real balance shaping (CHANGE 3): penalize base/torso forward TILT (||projected_gravity_xy||, a
    POSITION quantity) DURING the approach (pre_strike). Together with the existing strike-window
    ``strike_upright`` this keeps the CoM over the support base THROUGH the whole swing — the forward
    pitch-over is exactly the AGI-MuJoCo failure mode. Deliberately NOT an angular-velocity penalty: a
    base-ang-vel penalty is anti-correlated with swing power and is gameable; projected-gravity tilt is a
    pose, so it does not fight the swing. Gated by pre_strike ONLY (the strike window is covered by
    strike_upright). Positive magnitude; the RewTerm weight is NEGATIVE."""
    cmd = _cmd(env, command_name)
    return cmd.proj_grav_xy * cmd.pre_strike.float()


def strike_foot_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot motion (sum ||foot_velocity||²) DURING the strike window — plant for the hit."""
    cmd = _cmd(env, command_name)
    return cmd.foot_vel_sq * cmd.strike_window.float()


def strike_vertical_bob(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize vertical base velocity (|base_lin_vel_z|) DURING the strike window — no bob at the hit."""
    cmd = _cmd(env, command_name)
    return cmd.vertical_speed * cmd.strike_window.float()


# --- v2 收入型平衡 + PACE 单条击球稳定(reward_redesign_20260725 §1.4/§3;DEFAULT OFF) ------- #
# 与近亲们的分工(替代关系,不是叠加关系;消融臂开新项时应同时把旧项关掉):
#   * upright_exp:替代税型 upright 罚(罚倾斜的 L2)——mjlab 收入型,站得正每步发钱,
#     有界 (0,1],顺带兼任 alive bonus(解 fresh 臂早期"活着净亏、摔死更划算");
#   * hit_unstable_support:替代 v1 的 strike 四件套站稳包(strike_upright / strike_ang_vel /
#     strike_foot_vel / strike_vbob)——PACE 只有这一条"击球窗内单脚/无支撑就罚",
#     0/1 指示器,姿态细节交给全身模仿,不再逐项管倾斜/角速度/脚速/起伏。
# 两条都是纯新增,本波不进任何 cfg;后续波次以 weight=0 声明(默认路径字节等价)。
def upright_exp(env: ManagerBasedRLEnv, std: float = math.sqrt(0.2)) -> torch.Tensor:
    """站正收入(mjlab 收入型 upright,v2 蓝图 §1.4):exp(-||projected_gravity_xy||² / std²)。

    人话:站得越正每步发的钱越多——完全直立 = 1.0,越歪按指数掉钱但永远 > 0,dense 每步
    都发。收入型替代税型 upright 罚:梯度方向一样,但收入侧天然为正,天然就是 alive bonus,
    不用另加 alive 项。倾斜量 ||projected_gravity_xy|| = sin(倾角);默认 std=sqrt(0.2) 即
    倾斜量 ~0.45(倾角 ~27°)时收入掉到 1/e。RewTerm weight 用【正数】。
    """
    std = _finite_scalar(std, name="upright_exp std", positive=True)
    projected_gravity = getattr(env.scene["robot"].data, "projected_gravity_b", None)
    if (
        not torch.is_tensor(projected_gravity)
        or projected_gravity.ndim != 2
        or projected_gravity.shape[1] != 3
    ):
        raise RuntimeError(
            "upright_exp requires robot.data.projected_gravity_b shaped [env,3]"
        )
    tilt_sq = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    return torch.exp(-tilt_sq / (std * std))


def hit_unstable_support(
    env: ManagerBasedRLEnv, sensor_cfg, command_name: str = "racket_target"
) -> torch.Tensor:
    """击球窗内支撑不足惩罚(PACE hit_unstable_support 思想):
    indicator(触地脚数 <= 1) x 击球窗,每 env 取值 {0.0, 1.0}。

    人话:触球那一瞬必须双脚踩实——击球窗内单脚站或双脚腾空,这一步记 1;窗外、或窗内
    双脚都触地,记 0。只管"支撑够不够"这一个事实,不管歪不歪/晃不晃(那些交给全身模仿
    和 upright_exp)。触地判据 = 单脚接触力范数 > 10.0 N(与 sensor force_threshold /
    foot_clearance / hope_commands in_contact 同判据同阈值)。RewTerm weight 用负数。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if not isinstance(body_ids, (list, tuple)) or len(body_ids) != 2:
        raise RuntimeError(
            "hit_unstable_support requires sensor_cfg resolved to exactly the two A3 feet"
        )
    body_ids = list(body_ids)
    forces = getattr(sensor.data, "net_forces_w", None)
    if not torch.is_tensor(forces) or forces.ndim != 3 or forces.shape[-1] != 3:
        raise RuntimeError(
            "hit_unstable_support requires sensor net_forces_w shaped [env,body,3]"
        )
    # 10 N = sensor 的 force_threshold,与 foot_clearance / hope_commands 的 in_contact 同判据。
    in_contact = torch.norm(forces[:, body_ids, :], dim=-1) > 10.0  # (E,2)
    unstable = (in_contact.sum(dim=-1) <= 1).float()  # 单脚或无支撑
    window = getattr(_cmd(env, command_name), "strike_window", None)
    if (
        not torch.is_tensor(window)
        or window.dtype != torch.bool
        or tuple(window.shape) != (forces.shape[0],)
    ):
        raise RuntimeError(
            "hit_unstable_support requires the command term's [env] bool strike_window mask"
        )
    return unstable * window.float()


# ============================================================================================== #
# Sim2real: torque-saturation penalty (CHANGE 2). Discourage the policy from demanding torque the
# EXPLICIT clipped-PD motor cannot deliver. Under IdealPDActuatorCfg the model computes the pre-clip
# effort (kp*(q_des-q)+kd*(-qd)) and clips it to ±effort_limit; the ratio |computed| / effort_limit >1
# is exactly the over-demand that lags on the real robot. Penalizing the mean over-limit fraction over
# the arm + waist joints teaches a swing that lives inside the torque envelope (the elbow was measured
# at ~6.7x its 24 Nm limit in the failing trace). Uses ``data.computed_torque`` (Isaac copies each
# actuator's PRE-clip computed_effort into it) and ``data.joint_effort_limits`` (the per-joint sim
# limit written from effort_limit_sim). This term is a hardware-envelope claim: when enabled, missing
# indices/data or invalid limits MUST stop the run rather than reporting a counterfeit zero saturation.
# ============================================================================================== #
_TORQUE_SAT_JOINT_EXPR = [".*shoulder.*", ".*elbow.*", ".*wrist.*", "waist_.*_joint"]
_ACTION_BALL_HARD_SAFETY_TERMINATIONS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
_STAGE1_OBJECT_FREE_HARD_SAFETY_TERMINATIONS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
)


def action_ball_safety_terminated(
    env: ManagerBasedRLEnv,
    term_names: tuple[str, ...] = _ACTION_BALL_HARD_SAFETY_TERMINATIONS,
) -> torch.Tensor:
    """Return the exact ActionBall hard-safety termination union.

    Unlike Isaac Lab's generic ``is_terminated``, this deliberately excludes
    reference-consistency envelopes such as ``anchor_pos``, ``anchor_ori`` and
    ``ee_body_pos``.  Those envelopes may still reset an episode, but they are
    teacher-consistency events rather than table/fall/hard-limit safety events
    and must not silently inherit the -72 hard-safety charge.
    """

    names = tuple(term_names)
    if names != _ACTION_BALL_HARD_SAFETY_TERMINATIONS:
        raise RuntimeError(
            "action_ball_safety_terminated requires the exact ordered hard-safety "
            f"union {_ACTION_BALL_HARD_SAFETY_TERMINATIONS!r}, got {names!r}"
        )
    manager = getattr(env, "termination_manager", None)
    active = tuple(getattr(manager, "active_terms", ()))
    missing = [name for name in names if name not in active]
    if missing:
        raise RuntimeError(
            "action_ball_safety_terminated is missing active termination terms "
            f"{missing!r}; active terms are {active!r}"
        )
    masks = [manager.get_term(name) for name in names]
    first = masks[0]
    if (
        not torch.is_tensor(first)
        or first.dtype != torch.bool
        or first.ndim != 1
    ):
        raise RuntimeError(
            "action_ball_safety_terminated requires one [env] bool mask per term"
        )
    union = torch.zeros_like(first)
    for name, mask in zip(names, masks):
        if (
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(first.shape)
            or mask.device != first.device
        ):
            raise RuntimeError(
                "action_ball_safety_terminated term "
                f"{name!r} is not a same-device [env] bool mask"
            )
        union |= mask
    return union.float()


def stage1_object_free_safety_terminated(
    env: ManagerBasedRLEnv,
    term_names: tuple[str, ...] = _STAGE1_OBJECT_FREE_HARD_SAFETY_TERMINATIONS,
) -> torch.Tensor:
    """Return the exact hard-safety union for the object-free motion prior.

    Stage 1 has no table frame or task object, so its terminal price covers only
    absolute fall/height and raw command/plant joint safety.  Reference-consistency
    envelopes still reset without receiving the hard-safety charge.  ActionBall's
    table-inclusive union remains a separate, unchanged contract.
    """

    names = tuple(term_names)
    if names != _STAGE1_OBJECT_FREE_HARD_SAFETY_TERMINATIONS:
        raise RuntimeError(
            "stage1_object_free_safety_terminated requires the exact ordered "
            "hard-safety union "
            f"{_STAGE1_OBJECT_FREE_HARD_SAFETY_TERMINATIONS!r}, got {names!r}"
        )
    manager = getattr(env, "termination_manager", None)
    active = tuple(getattr(manager, "active_terms", ()))
    missing = [name for name in names if name not in active]
    if missing:
        raise RuntimeError(
            "stage1_object_free_safety_terminated is missing active termination "
            f"terms {missing!r}; active terms are {active!r}"
        )
    masks = [manager.get_term(name) for name in names]
    first = masks[0]
    if (
        not torch.is_tensor(first)
        or first.dtype != torch.bool
        or first.ndim != 1
    ):
        raise RuntimeError(
            "stage1_object_free_safety_terminated requires one [env] bool mask per term"
        )
    union = torch.zeros_like(first)
    for name, mask in zip(names, masks):
        if (
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(first.shape)
            or mask.device != first.device
        ):
            raise RuntimeError(
                "stage1_object_free_safety_terminated term "
                f"{name!r} is not a same-device [env] bool mask"
            )
        union |= mask
    return union.float()


def _torque_sat_joint_idx(env: ManagerBasedRLEnv, command_name: str):
    """Resolve+cache the arm+waist joint indices on the command term (once)."""
    cmd = _cmd(env, command_name)
    idx = getattr(cmd, "_torque_sat_joint_idx", None)
    if idx is None:
        try:
            idx = list(cmd.robot.find_joints(_TORQUE_SAT_JOINT_EXPR)[0])
        except Exception as exc:
            raise RuntimeError(
                "arm_torque_saturation could not resolve shoulder/elbow/wrist/waist joints"
            ) from exc
        if not idx:
            raise RuntimeError(
                "arm_torque_saturation resolved zero shoulder/elbow/wrist/waist joints"
            )
        if len(idx) != len(set(idx)):
            raise RuntimeError(
                f"arm_torque_saturation resolved duplicate joint indices: {idx}"
            )
        cmd._torque_sat_joint_idx = idx
    return cmd, idx


def _runtime_joint_ids(value, joint_count: int) -> list[int]:
    if isinstance(value, slice):
        return list(range(joint_count))[value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


_IMPLICIT_PD_PROXY_STATE_ATTR = "_hope_implicit_pd_effort_proxy_counters"
_IMPLICIT_PD_PROXY_STEP_ATTR = "_hope_implicit_pd_effort_proxy_step"
_IMPLICIT_PD_PROXY_SIGNATURE_ATTR = "_hope_implicit_pd_effort_proxy_signature"


def _require_all_implicit_action_backend(action, joint_ids: list[int], joint_count: int) -> None:
    """Prove every measured action joint is owned by one implicit actuator."""

    asset = getattr(action, "_asset", None)
    actuators = getattr(asset, "actuators", None)
    if not isinstance(actuators, dict) or not actuators:
        raise RuntimeError("implicit PD effort proxy cannot prove actuator ownership")
    ownership: dict[int, bool] = {}
    for group_name, actuator in actuators.items():
        if not hasattr(actuator, "joint_indices") or not hasattr(actuator, "is_implicit_model"):
            raise RuntimeError(
                "implicit PD effort proxy actuator group "
                f"{group_name!r} lacks joint_indices/is_implicit_model"
            )
        for joint_id in _runtime_joint_ids(actuator.joint_indices, joint_count):
            if joint_id in ownership:
                raise RuntimeError(
                    "implicit PD effort proxy found duplicate actuator ownership for joint "
                    f"{joint_id}"
                )
            ownership[joint_id] = bool(actuator.is_implicit_model)
    missing = [joint_id for joint_id in joint_ids if joint_id not in ownership]
    explicit = [joint_id for joint_id in joint_ids if ownership.get(joint_id) is False]
    if missing or explicit:
        raise RuntimeError(
            "implicit PD effort proxy requires complete implicit-actuator ownership; "
            f"missing={missing!r} explicit={explicit!r}"
        )


def _implicit_pd_proxy_state(
    env: ManagerBasedRLEnv, template: torch.Tensor
) -> dict[str, torch.Tensor]:
    state = getattr(env, _IMPLICIT_PD_PROXY_STATE_ATTR, None)
    if state is None:
        state = {
            "observed_joint_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "valid_joint_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "invalid_joint_sample_count": torch.zeros((), dtype=torch.long, device=template.device),
            "above_soft_ratio_joint_count": torch.zeros((), dtype=torch.long, device=template.device),
            "above_limit_joint_count": torch.zeros((), dtype=torch.long, device=template.device),
            "utilization_ratio_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "excess_over_limit_ratio_sum": torch.zeros((), dtype=template.dtype, device=template.device),
            "peak_utilization_ratio": torch.zeros((), dtype=template.dtype, device=template.device),
        }
        setattr(env, _IMPLICIT_PD_PROXY_STATE_ATTR, state)
    return state


def implicit_pd_post_step_effort_proxy_probe(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    soft_limit_ratio: float = 0.9,
) -> torch.Tensor:
    """Observe an analytic implicit-PD effort-demand proxy and return zero reward.

    The proxy is exactly ``kp_live*(q_des_sent-q_post)-kd_live*qdot_post`` using the final target
    dispatched by the action term and the live, possibly randomized PhysX drive gains.  It is a
    *post-policy-step analytic demand proxy*: PhysX does not expose implicit-drive applied torque,
    and this observer is blind to larger transients within the four physics substeps.  Therefore
    these counters must never be described as actual torque, actuator clipping, or a substep peak.
    The explicit flag is the only installation path; the function contributes identically zero
    reward and performs no simulator write.
    """

    ratio_threshold = float(soft_limit_ratio)
    if not math.isfinite(ratio_threshold) or not 0.0 < ratio_threshold < 1.0:
        raise ValueError("implicit PD effort proxy soft_limit_ratio must be in (0,1)")
    action = env.action_manager.get_term(action_name)
    q_des = getattr(action, "processed_actions", None)
    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    joint_names = getattr(data, "joint_names", None) or getattr(asset, "joint_names", ())
    joint_count = len(joint_names)
    if not torch.is_tensor(q_des) or q_des.ndim != 2 or joint_count <= 0:
        raise RuntimeError("implicit PD effort proxy requires processed q_des and joint metadata")
    joint_ids = _runtime_joint_ids(getattr(action, "_joint_ids", slice(None)), joint_count)
    if len(joint_ids) != q_des.shape[1] or len(set(joint_ids)) != len(joint_ids):
        raise RuntimeError("implicit PD effort proxy requires one unique joint id per action column")
    if not getattr(action, "_implicit_pd_effort_proxy_backend_checked", False):
        _require_all_implicit_action_backend(action, joint_ids, joint_count)
        action._implicit_pd_effort_proxy_backend_checked = True

    tensors = {
        "joint_pos": getattr(data, "joint_pos", None),
        "joint_vel": getattr(data, "joint_vel", None),
        "joint_stiffness": getattr(data, "joint_stiffness", None),
        "joint_damping": getattr(data, "joint_damping", None),
        "joint_effort_limits": getattr(data, "joint_effort_limits", None),
    }
    selected: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != q_des.shape[0]:
            raise RuntimeError(f"implicit PD effort proxy requires {name} as [env,joint]")
        selected[name] = value[:, joint_ids]
        if (
            selected[name].shape != q_des.shape
            or selected[name].device != q_des.device
            or selected[name].dtype != q_des.dtype
        ):
            raise RuntimeError(
                f"implicit PD effort proxy {name} does not match processed q_des shape/device/dtype"
            )

    token = getattr(env, "common_step_counter", None)
    signature = (str(action_name), ratio_threshold, tuple(joint_ids))
    if type(token) is int and getattr(env, _IMPLICIT_PD_PROXY_STEP_ATTR, None) == token:
        if getattr(env, _IMPLICIT_PD_PROXY_SIGNATURE_ATTR, None) != signature:
            raise RuntimeError("implicit PD effort proxy parameters changed within one simulator step")
        return torch.zeros_like(q_des[:, 0])

    kp = selected["joint_stiffness"]
    kd = selected["joint_damping"]
    limits = selected["joint_effort_limits"]
    valid = (
        torch.isfinite(q_des)
        & torch.isfinite(selected["joint_pos"])
        & torch.isfinite(selected["joint_vel"])
        & torch.isfinite(kp)
        & torch.isfinite(kd)
        & torch.isfinite(limits)
        & kp.gt(0.0)
        & kd.ge(0.0)
        & limits.gt(0.0)
    )
    demand = kp * (q_des - selected["joint_pos"]) - kd * selected["joint_vel"]
    utilization = torch.abs(demand) / limits
    safe = torch.where(valid & torch.isfinite(utilization), utilization, torch.zeros_like(utilization))
    valid = valid & torch.isfinite(utilization)
    state = _implicit_pd_proxy_state(env, q_des[:, 0])
    state["observed_joint_sample_count"].add_(utilization.numel())
    state["valid_joint_sample_count"].add_(valid.sum(dtype=torch.long))
    state["invalid_joint_sample_count"].add_((~valid).sum(dtype=torch.long))
    state["above_soft_ratio_joint_count"].add_(
        (valid & utilization.gt(ratio_threshold)).sum(dtype=torch.long)
    )
    state["above_limit_joint_count"].add_((valid & utilization.gt(1.0)).sum(dtype=torch.long))
    state["utilization_ratio_sum"].add_(safe.sum())
    state["excess_over_limit_ratio_sum"].add_((safe - 1.0).clamp(min=0.0).sum())
    state["peak_utilization_ratio"].copy_(
        torch.maximum(state["peak_utilization_ratio"], safe.max())
    )
    if type(token) is int:
        setattr(env, _IMPLICIT_PD_PROXY_STEP_ATTR, token)
        setattr(env, _IMPLICIT_PD_PROXY_SIGNATURE_ATTR, signature)
    return torch.zeros_like(q_des[:, 0])


def consume_implicit_pd_post_step_effort_proxy_counters(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Snapshot/reset the explicit analytic proxy's per-update device scalars."""

    action = env.action_manager.get_term("joint_pos")
    q_des = getattr(action, "processed_actions", None)
    if not torch.is_tensor(q_des) or q_des.ndim != 2:
        raise RuntimeError("implicit PD effort proxy consumer requires processed q_des")
    state = _implicit_pd_proxy_state(env, q_des[:, 0])
    snapshot = {name: value.detach().clone() for name, value in state.items()}
    with torch.inference_mode():
        for value in state.values():
            value.zero_()
    return snapshot


def _require_explicit_torque_saturation_backend(cmd, idx: list[int]) -> None:
    """Prove every measured joint belongs to an explicit actuator group."""

    robot = cmd.robot
    actuators = getattr(robot, "actuators", None)
    joint_count = len(
        getattr(getattr(robot, "data", None), "joint_names", ())
        or getattr(robot, "joint_names", ())
    )
    if not isinstance(actuators, dict) or not actuators or joint_count <= 0:
        raise RuntimeError(
            "arm_torque_saturation cannot prove the runtime actuator backend"
        )
    ownership: dict[int, bool] = {}
    for group_name, actuator in actuators.items():
        if not hasattr(actuator, "joint_indices") or not hasattr(
            actuator, "is_implicit_model"
        ):
            raise RuntimeError(
                "arm_torque_saturation actuator group "
                f"{group_name!r} lacks joint_indices/is_implicit_model"
            )
        for joint_id in _runtime_joint_ids(actuator.joint_indices, joint_count):
            if joint_id in ownership:
                raise RuntimeError(
                    "arm_torque_saturation found duplicate actuator ownership "
                    f"for joint {joint_id}"
                )
            ownership[joint_id] = bool(actuator.is_implicit_model)
    missing = [joint_id for joint_id in idx if joint_id not in ownership]
    if missing:
        raise RuntimeError(
            "arm_torque_saturation cannot resolve actuator ownership for joints "
            f"{missing!r}"
        )
    implicit = [joint_id for joint_id in idx if ownership[joint_id]]
    if implicit:
        raise RuntimeError(
            "arm_torque_saturation is disabled for ImplicitActuator joints "
            f"{implicit!r}: computed_torque is not a proven explicit pre-clip demand"
        )


def arm_torque_saturation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Mean over-limit fraction of the COMPUTED (pre-clip) effort over the arm + waist joints:
    ``mean_j relu(|computed_torque_j| / effort_limit_j - 1)``. 0 when every arm/waist joint is inside its
    torque envelope; grows as the swing demands un-deliverable torque (the explicit-PD saturation that
    tips the free base in AGI's MuJoCo). Positive magnitude; the RewTerm weight is NEGATIVE."""
    cmd, idx = _torque_sat_joint_idx(env, command_name)
    if not getattr(cmd, "_torque_sat_backend_checked", False):
        _require_explicit_torque_saturation_backend(cmd, idx)
        cmd._torque_sat_backend_checked = True
    data = cmd.robot.data
    tau = getattr(data, "computed_torque", None)
    lim = getattr(data, "joint_effort_limits", None)
    if tau is None or lim is None:
        raise RuntimeError(
            "arm_torque_saturation requires robot.data.computed_torque (pre-clip) and "
            "robot.data.joint_effort_limits; the active actuator backend exposes neither/both "
            "incorrectly"
        )
    if tau.ndim != 2 or lim.ndim != 2 or tau.shape != lim.shape:
        raise RuntimeError(
            f"arm_torque_saturation expected matching [env,joint] tensors, got "
            f"computed_torque={tuple(tau.shape)}, limits={tuple(lim.shape)}"
        )
    if max(idx) >= tau.shape[1]:
        raise RuntimeError(
            f"arm_torque_saturation joint index {max(idx)} exceeds tensor width {tau.shape[1]}"
        )
    tau_a = torch.abs(tau[:, idx])
    lim_a = lim[:, idx]
    # The boolean checks synchronize a CUDA stream, so run the mechanism/data contract once,
    # not at every 50-Hz reward evaluation. Later non-finite torques still propagate into the
    # reward (and PPO's normal non-finite guard) rather than being converted into a fake zero.
    if not getattr(cmd, "_torque_sat_contract_checked", False):
        if not bool(torch.isfinite(tau_a).all()) or not bool(torch.isfinite(lim_a).all()):
            raise RuntimeError("arm_torque_saturation received non-finite torque/effort-limit data")
        if bool((lim_a <= 0.0).any()):
            raise RuntimeError("arm_torque_saturation requires strictly positive effort limits")
        cmd._torque_sat_contract_checked = True
    over = (tau_a / lim_a - 1.0).clamp(min=0.0)  # relu(ratio - 1): the un-deliverable fraction
    frac = over.mean(dim=-1)
    cmd.metrics["arm_torque_sat_frac"] = frac  # watch-metric: should fall toward 0 during fine-tune
    return frac

def _motion_imitation_eligible(command) -> torch.Tensor:
    """Keep legacy swing-only masking while admitting split-ready transition rows."""

    eligible = getattr(command, "imitation_eligible", None)
    if eligible is None:
        return ~command.in_hold.bool()
    return eligible.bool()


def motion_body_pos_swing_only(env, command_name: str, std: float, body_names=None,
                               window_scale: float = 1.0, window_command_name: str | None = None):
    """motion_relative_body_position_error_exp gated to ~in_hold (2026-07-05): during
    hold the joint reference is the default STAND (commands.joint_pos) while the frozen
    body refs still show clip frame 0's crouch — un-gated, the two imitation pulls
    fight and the policy settles into the splayed-feet crouch-stand. Swing-only.
    window_scale/window_command_name: V2 in-window imitation yield, forwarded to the base
    func (see rewards._apply_window_scale); defaults = no-op."""
    from .rewards import motion_relative_body_position_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_relative_body_position_error_exp(env, command_name, std, body_names,
                                                window_scale, window_command_name)
    return torch.where(_motion_imitation_eligible(cmd), r, torch.zeros_like(r))


def motion_body_ori_swing_only(env, command_name: str, std: float, body_names=None,
                               window_scale: float = 1.0, window_command_name: str | None = None):
    """See motion_body_pos_swing_only."""
    from .rewards import motion_relative_body_orientation_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_relative_body_orientation_error_exp(env, command_name, std, body_names,
                                                   window_scale, window_command_name)
    return torch.where(_motion_imitation_eligible(cmd), r, torch.zeros_like(r))


def motion_body_lin_vel_swing_only(env, command_name: str, std: float, body_names=None,
                                   window_scale: float = 1.0,
                                   window_command_name: str | None = None):
    """Body linear-velocity imitation with no income during a recovery hold.

    ``MotionCommand.body_lin_vel_w`` correctly exposes a stationary (zero-velocity)
    reference while held.  Paying the ordinary velocity kernel for that reference is still
    wrong for HitterPure rally recovery, though: it rewards *remaining still* while
    ``hold_heading`` asks the base/waist to turn back toward the table.  The video teacher is
    an imitation prior for the swing, not a hold controller, so the whole term is silent in
    hold just like the position/orientation terms above.
    """
    from .rewards import motion_global_body_linear_velocity_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_global_body_linear_velocity_error_exp(
        env, command_name, std, body_names, window_scale, window_command_name
    )
    return torch.where(_motion_imitation_eligible(cmd), r, torch.zeros_like(r))


def motion_body_ang_vel_swing_only(env, command_name: str, std: float, body_names=None,
                                   window_scale: float = 1.0,
                                   window_command_name: str | None = None):
    """Angular-velocity counterpart of :func:`motion_body_lin_vel_swing_only`."""
    from .rewards import motion_global_body_angular_velocity_error_exp
    cmd = env.command_manager.get_term(command_name)
    r = motion_global_body_angular_velocity_error_exp(
        env, command_name, std, body_names, window_scale, window_command_name
    )
    return torch.where(_motion_imitation_eligible(cmd), r, torch.zeros_like(r))


def _ignore_hold(command, value: torch.Tensor, ignore_hold: bool) -> torch.Tensor:
    """Mask a reference-relative termination during hold, failing loud on a bad command.

    The absolute fall guards remain separate termination terms.  This helper is deliberately
    not a permissive ``getattr(..., False)`` fallback: configuring ``ignore_hold=True`` on a
    command without an ``in_hold`` contract would silently reintroduce the reset-time bug.
    """
    if not ignore_hold:
        return value
    if not hasattr(command, "in_hold"):
        raise RuntimeError("ignore_hold=True requires the command to expose an in_hold mask")
    return value & ~command.in_hold.bool()


def _action_ball_reference_terminations_mask(
    env,
    *,
    reason: str | None = None,
    raw_verdict: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Per-env curriculum gate for the reference-consistency terminations.

    Franco 2026-07-28 third ruling (sole default behavior in curriculum
    mode): once an env's action has expanded past the center phase
    (phase >= marginal), the teacher-consistency envelopes anchor_pos /
    anchor_ori / ee_body_pos stop terminating that env; the absolute
    fall / table / joint guards are separate terms and always stay on.
    Outside action-ball runs the racket term does not expose the gate and
    every other task keeps the exact old verdicts.
    """

    manager = getattr(env, "command_manager", None)
    get_term = getattr(manager, "get_term", None)
    if not callable(get_term):
        return None
    try:
        racket = get_term("racket_target")
    except Exception:
        return None
    getter = getattr(
        racket, "action_ball_reference_terminations_enabled", None
    )
    if getter is None:
        return None
    # Default ``phase_gated`` takes the historical hot path: no recorder call,
    # no counters, and no new tensor work.  The explicit treatment reuses this
    # already-resolved command term to publish the raw mask before gating.
    if (
        reason is not None
        and getattr(
            getattr(racket, "cfg", None),
            "reference_guard_mode",
            "phase_gated",
        )
        == "metrics_only"
    ):
        if raw_verdict is None:
            raise RuntimeError(
                "metrics-only reference guard requires the raw verdict"
            )
        recorder = getattr(
            racket, "action_ball_record_reference_guard_raw", None
        )
        if not callable(recorder):
            raise RuntimeError(
                "metrics-only reference guard has no raw recorder"
            )
        recorder(reason, raw_verdict)
    return getter()


def _gate_reference_termination(
    env,
    verdict: torch.Tensor,
    *,
    reason: str | None = None,
) -> torch.Tensor:
    mask = _action_ball_reference_terminations_mask(
        env, reason=reason, raw_verdict=verdict
    )
    if mask is None:
        return verdict
    return verdict & mask


def bad_anchor_pos_z_only_hold_aware(
    env, command_name: str, threshold: float, ignore_hold: bool = False
) -> torch.Tensor:
    """Reference torso-height envelope with an explicit held-RSI exclusion."""
    from .terminations import bad_anchor_pos_z_only
    command = env.command_manager.get_term(command_name)
    return _gate_reference_termination(
        env,
        _ignore_hold(
            command,
            bad_anchor_pos_z_only(env, command_name, threshold),
            ignore_hold,
        ),
        reason="anchor_pos",
    )


def bad_anchor_ori_hold_aware(
    env, asset_cfg, command_name: str, threshold: float, ignore_hold: bool = False
) -> torch.Tensor:
    """Reference orientation envelope with an explicit held-RSI exclusion."""
    from .terminations import bad_anchor_ori
    command = env.command_manager.get_term(command_name)
    return _gate_reference_termination(
        env,
        _ignore_hold(
            command,
            bad_anchor_ori(env, asset_cfg, command_name, threshold),
            ignore_hold,
        ),
        reason="anchor_ori",
    )


def bad_motion_body_pos_z_only_hold_aware(
    env, command_name: str, threshold: float, body_names=None,
    ignore_hold: bool = False,
) -> torch.Tensor:
    """Reference body-height envelope with an explicit held-RSI exclusion."""
    from .terminations import bad_motion_body_pos_z_only
    command = env.command_manager.get_term(command_name)
    return _gate_reference_termination(
        env,
        _ignore_hold(
            command,
            bad_motion_body_pos_z_only(env, command_name, threshold, body_names),
            ignore_hold,
        ),
        reason="ee_body_pos",
    )

def foot_orientation_discipline(env, command_name: str, asset_cfg, hold_gate: bool = False):
    """L1 deviation of the foot-orientation joints (hip yaw/roll, ankle roll) from the
    REFERENCE joint positions — hold-aware via commands.joint_pos (default stand during
    hold, clip footwork during swings). 2026-07-05: with no joint-level imitation in
    the stack these DOF were reward-free, and the policy twisted the feet to
    -1.13/+0.90 rad during swings/side-switches vs a reference envelope of ±0.41
    (Gate 2.5 diag) — the 'weird foot placement' at strike/switch. Use a NEGATIVE
    weight (penalty); keep it small so it disciplines feet without taxing the lunge.
    When ``hold_gate`` is true, the term is zero during the recovery hold: otherwise the
    square-stand joint reference penalizes the hip-yaw motion needed to re-square the base.
    """
    cmd = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    ref = cmd.joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(q - ref), dim=1)
    if hold_gate:
        penalty = torch.where(cmd.in_hold, torch.zeros_like(penalty), penalty)
    return penalty
