"""Engine-neutral tensor ABI for the semantic ActionEpoch observation.

Only column order and pure tensor transforms live here. Isaac and MuJoCo own
their live numerical producers; this module defines no owner, registry,
receipt, gate, lifecycle journal, or fallback value.

V2/V3's table frame is the current fixed, axis-aligned ActionBall table frame.
A rotatable table requires a new contract with a real table-pose producer.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple


# Legacy V1 remains executable until both engines atomically cut over to V2.
# It is a compatibility ABI, not a source for successor V2 fields.
TASK_F32_WIDTH = 45
OWNER_FACT_F32_WIDTH = 32
REWARD_CONSUMER_COUNT = 14
EPOCH_IDLE_PHASE_INDEX = 0

ACTOR_CONTRACT_V1 = "action_ball_full_mdp_action_epoch_v1"
CRITIC_CONTRACT_V1 = "action_ball_full_mdp_action_epoch_critic_v1"
OBSERVATION_KIND_V1 = "action_ball_full_mdp_action_epoch_observation_v1"

DIRECT_FIELD_LAYOUT_V1 = (
    ("projected_gravity_b", 3),
    ("base_ang_vel_b", 3),
    ("joint_pos_rel", 31),
    ("joint_vel_rel", 31),
    ("last_action", 31),
    ("teacher_joint_pos_rel", 31),
    ("teacher_joint_vel_rel", 31),
)

ACTOR_LAYOUT_V1 = DIRECT_FIELD_LAYOUT_V1 + (
    ("motion_phase_one_hot", 5),
    ("epoch_task_f32", TASK_F32_WIDTH),
    ("epoch_clock_remaining_s", 5),
    ("epoch_phase_one_hot", 10),
    ("epoch_task_valid", 1),
    ("epoch_selected", 1),
    ("epoch_launch_succeeded", 1),
)

CRITIC_EXTENSION_LAYOUT_V1 = (
    ("physical_r03_r06_r07_fact_present", 4),
    ("physical_r03_r06_r07_fact_age_s", 4),
    ("physical_r03_r06_r07_fact_f32", 4 * OWNER_FACT_F32_WIDTH),
    ("physical_r03_r06_r07_fault_present", 4),
    ("reward_cycle_open", 1),
    ("reward_cycle_fault_present", 1),
    ("reward_due", REWARD_CONSUMER_COUNT),
    ("reward_paid", REWARD_CONSUMER_COUNT),
)

ACTOR_WIDTH_V1 = sum(width for _, width in ACTOR_LAYOUT_V1)
CRITIC_WIDTH_V1 = ACTOR_WIDTH_V1 + sum(
    width for _, width in CRITIC_EXTENSION_LAYOUT_V1
)


# Successor V2 is semantic and compact; it does not reinterpret V1 columns.
ACTOR_CONTRACT_V2 = "action_ball_full_mdp_semantic_actor_v2"
CRITIC_CONTRACT_V2 = "action_ball_full_mdp_semantic_critic_v2"
OBSERVATION_KIND_V2 = "action_ball_full_mdp_semantic_observation_v2"

COMMON_ACTOR_LAYOUT_V2 = (
    ("projected_gravity_b", 3),
    ("base_ang_vel_b", 3),
    ("base_position_table", 3),
    ("base_heading_table_xy", 2),
    ("base_com_lin_vel_heading", 3),
    ("joint_pos_rel", 31),
    ("joint_vel", 31),
    ("last_action", 31),
    ("teacher_joint_pos_rel", 31),
    ("teacher_joint_vel", 31),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b6", 6),
    ("motion_phase_one_hot", 5),
)

ACTOR_TASK_LAYOUT_V2 = (
    ("racket_target_pos_error_heading", 3),
    ("racket_target_vel_error_heading", 3),
    ("racket_target_normal_error_heading", 3),
    ("base_goal_error_heading_xy", 2),
    ("time_to_contact_s", 1),
    ("time_to_teacher_start_s", 1),
    ("time_to_next_opportunity_s", 1),
    ("epoch_learning_phase_one_hot", 5),
    ("task_valid", 1),
)

ACTOR_LAYOUT_V2 = COMMON_ACTOR_LAYOUT_V2 + ACTOR_TASK_LAYOUT_V2

CRITIC_EXTENSION_LAYOUT_V2 = (
    ("episode_time_remaining_s", 1),
    ("live_ball_center_rel_root_heading", 3),
    ("live_ball_lin_vel_heading", 3),
    ("live_ball_ang_vel_heading", 3),
    ("selected_rubber_contact_latched", 1),
    ("net_crossed_latched", 1),
    ("net_clear_latched", 1),
    ("foot_supported_lr", 2),
    ("cadence_ready_dwell_fraction", 1),
)

COMMON_ACTOR_WIDTH_V2 = sum(width for _, width in COMMON_ACTOR_LAYOUT_V2)
ACTOR_WIDTH_V2 = sum(width for _, width in ACTOR_LAYOUT_V2)
CRITIC_EXTENSION_WIDTH_V2 = sum(
    width for _, width in CRITIC_EXTENSION_LAYOUT_V2
)
CRITIC_WIDTH_V2 = ACTOR_WIDTH_V2 + CRITIC_EXTENSION_WIDTH_V2


# V3 preserves every V2 column and inserts only the four full-phase measured
# paddle residuals that make the actor observe the state used by the live
# imitation reward.  Ball-task residuals remain a distinct, task-masked tail.
ACTOR_CONTRACT_V3 = "action_ball_full_mdp_semantic_actor_v3"
CRITIC_CONTRACT_V3 = "action_ball_full_mdp_semantic_critic_v3"
OBSERVATION_KIND_V3 = "action_ball_full_mdp_semantic_observation_v3"

MOTION_RACKET_RESIDUAL_LAYOUT_V3 = (
    ("motion_racket_pos_error_heading", 3),
    ("motion_racket_vel_error_heading", 3),
    ("motion_racket_signed_normal_error_heading", 3),
    ("motion_racket_long_axis_error_heading", 3),
)

COMMON_ACTOR_LAYOUT_V3 = (
    COMMON_ACTOR_LAYOUT_V2 + MOTION_RACKET_RESIDUAL_LAYOUT_V3
)
ACTOR_TASK_LAYOUT_V3 = ACTOR_TASK_LAYOUT_V2
ACTOR_LAYOUT_V3 = COMMON_ACTOR_LAYOUT_V3 + ACTOR_TASK_LAYOUT_V3
CRITIC_EXTENSION_LAYOUT_V3 = CRITIC_EXTENSION_LAYOUT_V2

COMMON_ACTOR_WIDTH_V3 = sum(width for _, width in COMMON_ACTOR_LAYOUT_V3)
ACTOR_WIDTH_V3 = sum(width for _, width in ACTOR_LAYOUT_V3)
CRITIC_EXTENSION_WIDTH_V3 = sum(
    width for _, width in CRITIC_EXTENSION_LAYOUT_V3
)
CRITIC_WIDTH_V3 = ACTOR_WIDTH_V3 + CRITIC_EXTENSION_WIDTH_V3

# Static nondimensionalization is part of the V2 ABI.  These immutable host
# tuples are shared by both engines; V1 remains raw and never consumes them.
ACTOR_SCALE_BY_FIELD_V2 = (
    ("projected_gravity_b", 1.0),
    ("base_ang_vel_b", 0.25),
    ("base_position_table", 1.0),
    ("base_heading_table_xy", 1.0),
    ("base_com_lin_vel_heading", 0.5),
    ("joint_pos_rel", 1.0),
    ("joint_vel", 0.05),
    ("last_action", 1.0),
    ("teacher_joint_pos_rel", 1.0),
    ("teacher_joint_vel", 0.05),
    ("motion_anchor_pos_b", 10.0 / 3.0),
    ("motion_anchor_ori_b6", 1.0),
    ("motion_phase_one_hot", 1.0),
    ("racket_target_pos_error_heading", 5.0),
    ("racket_target_vel_error_heading", 1.0),
    ("racket_target_normal_error_heading", 2.0),
    ("base_goal_error_heading_xy", 5.0),
    ("time_to_contact_s", 1.0 / 2.42),
    ("time_to_teacher_start_s", 1.0),
    ("time_to_next_opportunity_s", 1.0 / 5.86),
    ("epoch_learning_phase_one_hot", 1.0),
    ("task_valid", 1.0),
)

CRITIC_EXTENSION_SCALE_BY_FIELD_V2 = (
    ("episode_time_remaining_s", 1.0 / 30.0),
    ("live_ball_center_rel_root_heading", 1.0),
    ("live_ball_lin_vel_heading", 0.1),
    ("live_ball_ang_vel_heading", 1.0 / 60.0),
    ("selected_rubber_contact_latched", 1.0),
    ("net_crossed_latched", 1.0),
    ("net_clear_latched", 1.0),
    ("foot_supported_lr", 1.0),
    ("cadence_ready_dwell_fraction", 1.0),
)

MOTION_RACKET_RESIDUAL_SCALE_BY_FIELD_V3 = (
    ("motion_racket_pos_error_heading", 5.0),
    ("motion_racket_vel_error_heading", 1.0),
    ("motion_racket_signed_normal_error_heading", 2.0),
    ("motion_racket_long_axis_error_heading", 2.0),
)

ACTOR_SCALE_BY_FIELD_V3 = (
    ACTOR_SCALE_BY_FIELD_V2[: len(COMMON_ACTOR_LAYOUT_V2)]
    + MOTION_RACKET_RESIDUAL_SCALE_BY_FIELD_V3
    + ACTOR_SCALE_BY_FIELD_V2[len(COMMON_ACTOR_LAYOUT_V2) :]
)
CRITIC_EXTENSION_SCALE_BY_FIELD_V3 = CRITIC_EXTENSION_SCALE_BY_FIELD_V2


def _expand_layout_scale(layout, scale_by_field):
    if tuple(name for name, _ in layout) != tuple(
        name for name, _ in scale_by_field
    ):
        raise RuntimeError("observation scale fields differ from layout")
    return tuple(
        multiplier
        for (_, width), (_, multiplier) in zip(layout, scale_by_field)
        for _ in range(width)
    )


ACTOR_SCALE_FLAT_V2 = _expand_layout_scale(
    ACTOR_LAYOUT_V2, ACTOR_SCALE_BY_FIELD_V2
)
CRITIC_EXTENSION_SCALE_FLAT_V2 = _expand_layout_scale(
    CRITIC_EXTENSION_LAYOUT_V2, CRITIC_EXTENSION_SCALE_BY_FIELD_V2
)
ACTOR_SCALE_FLAT_V3 = _expand_layout_scale(
    ACTOR_LAYOUT_V3, ACTOR_SCALE_BY_FIELD_V3
)
CRITIC_EXTENSION_SCALE_FLAT_V3 = _expand_layout_scale(
    CRITIC_EXTENSION_LAYOUT_V3, CRITIC_EXTENSION_SCALE_BY_FIELD_V3
)


def concatenate_layout_rows(
    layout: Sequence[Tuple[str, int]], rows: Mapping[str, object]
):
    """Concatenate tensors in the shared, named column order."""

    import torch

    return torch.cat(tuple(rows[name] for name, _ in layout), dim=1)


def normalize_quat_wxyz(quaternion):
    """Normalize a ``[...,4]`` wxyz quaternion without a silent identity pad."""

    import torch

    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    return quaternion / norm


def quat_conjugate_wxyz(quaternion):
    """Return the conjugate of wxyz quaternions."""

    import torch

    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_multiply_wxyz(left, right):
    """Hamilton product for broadcast-compatible wxyz quaternions."""

    import torch

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_rotate_wxyz(quaternion, vector):
    """Rotate vectors by normalized wxyz quaternions."""

    import torch

    q = normalize_quat_wxyz(quaternion)
    xyz = q[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + q[..., :1] * twice_cross + torch.cross(
        xyz, twice_cross, dim=-1
    )


def quat_rotate_inverse_wxyz(quaternion, vector):
    """Rotate world vectors into the quaternion's local frame."""

    q = normalize_quat_wxyz(quaternion)
    return quat_rotate_wxyz(quat_conjugate_wxyz(q), vector)


def heading_xy_from_quat_wxyz(quaternion):
    """Return unit ``[cos(yaw), sin(yaw)]`` from a wxyz quaternion.

    Yaw is undefined when the base X axis is vertical.  Use the deterministic
    canonical heading ``[1, 0]`` for only those singular rows.  A fallen base
    is a learnable terminal state, not a process-level observation fault; valid
    peers must therefore remain observable without poisoning the CUDA stream.
    """

    import torch

    q = normalize_quat_wxyz(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    projected = torch.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (w * z + x * y)),
        dim=-1,
    )
    norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    singular = norm <= 1.0e-6
    safe_norm = torch.where(singular, torch.ones_like(norm), norm)
    normalized = projected / safe_norm
    canonical = torch.cat((torch.ones_like(norm), torch.zeros_like(norm)), dim=-1)
    return torch.where(singular, canonical, normalized)


def rotate_world_to_heading_xy(heading_xy, vector):
    """Apply a precomputed inverse-yaw transform while preserving world Z."""

    import torch

    cosine, sine = heading_xy.unbind(dim=-1)
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        (cosine * x + sine * y, -sine * x + cosine * y, z), dim=-1
    )


def rotate_world_to_heading(quaternion, vector):
    """Rotate world-axis vectors by inverse yaw while preserving world Z."""

    return rotate_world_to_heading_xy(
        heading_xy_from_quat_wxyz(quaternion), vector
    )


def rotation_6d_from_quat_wxyz(quaternion):
    """Return the first two rotation-matrix columns in row-major order."""

    import torch

    q = normalize_quat_wxyz(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )


def relative_pose_6d(
    parent_position, parent_quaternion, child_position, child_quaternion
):
    """Express a child pose in its parent's frame as position plus orientation-6D."""

    parent_q = normalize_quat_wxyz(parent_quaternion)
    child_q = normalize_quat_wxyz(child_quaternion)
    position = quat_rotate_inverse_wxyz(
        parent_q, child_position - parent_position
    )
    orientation = quat_multiply_wxyz(quat_conjugate_wxyz(parent_q), child_q)
    return position, rotation_6d_from_quat_wxyz(orientation)


__all__ = [
    "TASK_F32_WIDTH",
    "OWNER_FACT_F32_WIDTH",
    "REWARD_CONSUMER_COUNT",
    "EPOCH_IDLE_PHASE_INDEX",
    "ACTOR_CONTRACT_V1",
    "CRITIC_CONTRACT_V1",
    "OBSERVATION_KIND_V1",
    "DIRECT_FIELD_LAYOUT_V1",
    "ACTOR_LAYOUT_V1",
    "CRITIC_EXTENSION_LAYOUT_V1",
    "ACTOR_WIDTH_V1",
    "CRITIC_WIDTH_V1",
    "ACTOR_CONTRACT_V2",
    "CRITIC_CONTRACT_V2",
    "OBSERVATION_KIND_V2",
    "COMMON_ACTOR_LAYOUT_V2",
    "ACTOR_TASK_LAYOUT_V2",
    "ACTOR_LAYOUT_V2",
    "CRITIC_EXTENSION_LAYOUT_V2",
    "COMMON_ACTOR_WIDTH_V2",
    "ACTOR_WIDTH_V2",
    "CRITIC_EXTENSION_WIDTH_V2",
    "CRITIC_WIDTH_V2",
    "ACTOR_SCALE_BY_FIELD_V2",
    "CRITIC_EXTENSION_SCALE_BY_FIELD_V2",
    "ACTOR_SCALE_FLAT_V2",
    "CRITIC_EXTENSION_SCALE_FLAT_V2",
    "ACTOR_CONTRACT_V3",
    "CRITIC_CONTRACT_V3",
    "OBSERVATION_KIND_V3",
    "MOTION_RACKET_RESIDUAL_LAYOUT_V3",
    "COMMON_ACTOR_LAYOUT_V3",
    "ACTOR_TASK_LAYOUT_V3",
    "ACTOR_LAYOUT_V3",
    "CRITIC_EXTENSION_LAYOUT_V3",
    "COMMON_ACTOR_WIDTH_V3",
    "ACTOR_WIDTH_V3",
    "CRITIC_EXTENSION_WIDTH_V3",
    "CRITIC_WIDTH_V3",
    "MOTION_RACKET_RESIDUAL_SCALE_BY_FIELD_V3",
    "ACTOR_SCALE_BY_FIELD_V3",
    "CRITIC_EXTENSION_SCALE_BY_FIELD_V3",
    "ACTOR_SCALE_FLAT_V3",
    "CRITIC_EXTENSION_SCALE_FLAT_V3",
    "concatenate_layout_rows",
    "normalize_quat_wxyz",
    "quat_conjugate_wxyz",
    "quat_multiply_wxyz",
    "quat_rotate_wxyz",
    "quat_rotate_inverse_wxyz",
    "heading_xy_from_quat_wxyz",
    "rotate_world_to_heading_xy",
    "rotate_world_to_heading",
    "rotation_6d_from_quat_wxyz",
    "relative_pose_6d",
]
