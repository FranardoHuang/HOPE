"""Pure helpers for the opt-in FullMDP frozen-teacher diagnostic.

This module does not construct a question, move a ball, classify a contact, or
drive the plant.  Those authorities remain in ``FullMdpInitialWaitVecEnv``.
It only turns the environment's current Motion-owned reference into the same
normalized action ABI consumed by the live MuJoCo plant and defines the small
content hashes used by the diagnostic receipt.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


TEACHER_REPLAY_SCHEMA_VERSION = 1
TEACHER_REPLAY_DEFAULT_STEPS = 180
TEACHER_REPLAY_NUM_ENVS = 1
TEACHER_REPLAY_ACTION_DELAY_STEPS = 0


@dataclass(frozen=True)
class TeacherReplayPreStep:
    """The exact Motion-owned state consumed to construct one request."""

    task_valid: object
    teacher_frame: object
    motion_phase: object
    teacher_qdes: object
    reveal_tick: object
    pre_swing_wait_s: object


def capture_teacher_replay_pre_step(env) -> TeacherReplayPreStep:
    """Clone the input state before physics can advance or reset the row."""

    return TeacherReplayPreStep(
        task_valid=env._epoch_task_valid.clone(),
        teacher_frame=env._full_a_teacher_frame.clone(),
        motion_phase=env._full_a_motion_phase_code.clone(),
        teacher_qdes=env._full_a_teacher_joint_pos.clone(),
        reveal_tick=env._epoch_clock_ticks[:, 0].clone(),
        pre_swing_wait_s=env._full_a_pre_swing_wait_s.clone(),
    )


def contact_capture_boundary(
    *, transition_start_step: int, capture_boundary: str,
    physics_substep_index, decimation: int,
) -> dict:
    """Name a contact patch boundary without conflating final forward with a step."""

    if (
        type(transition_start_step) is not int
        or transition_start_step < 0
        or type(decimation) is not int
        or decimation <= 0
    ):
        raise ValueError("contact patch transition boundary differs")
    if capture_boundary == "physics_substep_poststate":
        if (
            type(physics_substep_index) is not int
            or physics_substep_index < 1
            or physics_substep_index >= decimation
        ):
            raise ValueError("contact patch physics substep differs")
        # This callback's indices are 1..decimation-1 and expose post-states
        # 1..decimation-1; index zero has no derived-state forward yet.
        completed = physics_substep_index
    elif capture_boundary == "post_forward_final":
        if physics_substep_index is not None:
            raise ValueError("final-forward contact patch has a substep index")
        completed = decimation
    else:
        raise ValueError("contact patch capture boundary differs")
    return {
        "transition_start_step": transition_start_step,
        "capture_boundary": capture_boundary,
        "physics_substep_index": physics_substep_index,
        "completed_physics_substeps": completed,
    }


def validate_contact_patch_shot(
    *, contact_patch: dict, question_reset_generation: int,
    question_f32_sha256: str,
) -> None:
    """Reject a captured patch from any later reset generation or question."""

    if not contact_patch.get("present", False):
        return
    if (
        contact_patch.get("reset_generation") != question_reset_generation
        or contact_patch.get("question_f32_sha256") != question_f32_sha256
    ):
        raise RuntimeError("contact patch belongs to a different shot")


def remaining_teacher_frozen_steps(
    *, torch, common_step: int, reveal_tick, pre_swing_wait_s, step_dt: float
):
    """Return Motion's integer frame-zero hold counter at one actor boundary.

    The wait contract is defined on policy boundaries.  Convert the complete
    wait to a tick count first, then subtract elapsed integer ticks.  Subtracting
    float32 seconds before ``ceil`` can turn the exact decimal schedule
    ``(0.10 - 0.04) / 0.02 == 3`` into ``3.0000002`` and silently add a tick.

    A configured tick boundary can itself be represented just above the exact
    decimal in a floating dtype.  Reconstruct the one canonical encoding from
    the integer tick candidate and ``step_dt`` in float64, then cast it back to
    the input dtype.  Only an exact equality to that encoding is a boundary;
    every neighbouring or off-grid value keeps strict upward rounding.
    """

    if type(common_step) is not int or common_step < 0:
        raise ValueError("teacher replay common step differs")
    if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError("teacher replay policy dt differs")
    if (
        reveal_tick.ndim != 1
        or tuple(pre_swing_wait_s.shape) != tuple(reveal_tick.shape)
        or reveal_tick.dtype != torch.long
        or not pre_swing_wait_s.dtype.is_floating_point
    ):
        raise ValueError("teacher replay frozen-step tensors differ")

    # Construct the canonical encoded boundary in float64 before casting it
    # back to the input dtype.  In particular, doing ``5 * float32(0.02)`` in
    # float32 can round to the lower neighbour of ``float32(0.10)``.  There is
    # deliberately no tolerance: nextafter neighbours are real off-grid waits.
    wait_s_f64 = pre_swing_wait_s.to(dtype=torch.float64)
    wait_ratio = wait_s_f64 / float(step_dt)
    integer_tick_candidate = torch.round(wait_ratio).to(torch.long)
    candidate_tick_f64 = integer_tick_candidate.to(dtype=torch.float64)
    canonical_boundary_encoded = (
        candidate_tick_f64 * float(step_dt)
    ).to(dtype=pre_swing_wait_s.dtype)
    on_encoded_tick_boundary = pre_swing_wait_s == canonical_boundary_encoded
    total_wait_ticks = torch.ceil(
        torch.where(on_encoded_tick_boundary, candidate_tick_f64, wait_ratio)
    ).to(torch.long)
    elapsed_ticks = torch.clamp(common_step - reveal_tick, min=0)
    frozen = torch.clamp(total_wait_ticks - elapsed_ticks, min=0)
    torch._assert_async(torch.isfinite(pre_swing_wait_s).all())
    torch._assert_async(torch.all(frozen >= 0))
    return frozen


def frozen_teacher_qdes(
    *,
    torch,
    task_valid,
    hold_qdes,
    previous_qdes,
    teacher_qdes,
    frozen_steps,
    bridge,
):
    """Return WAIT hold or the exact split-ready diagnostic bridge command."""

    shape = tuple(teacher_qdes.shape)
    if (
        teacher_qdes.ndim != 2
        or tuple(hold_qdes.shape) != shape
        or tuple(previous_qdes.shape) != shape
        or tuple(task_valid.shape) != (shape[0],)
        or task_valid.dtype != torch.bool
        or tuple(frozen_steps.shape) != (shape[0],)
        or frozen_steps.dtype != torch.long
    ):
        raise ValueError("teacher replay q_des tensors differ")
    bridged = bridge(
        torch=torch,
        previous_qdes=previous_qdes,
        frame0_qdes=teacher_qdes,
        frozen_steps=frozen_steps,
    )
    command = torch.where(task_valid[:, None], bridged, hold_qdes)
    torch._assert_async(torch.isfinite(command).all())
    return command


def decode_teacher_qdes_to_action(*, torch, qdes, action_offset, action_scale):
    """Invert only the live affine decoder; the environment still executes it."""

    if (
        qdes.ndim != 2
        or action_offset.ndim != 1
        or action_scale.ndim != 1
        or qdes.shape[1] != action_offset.shape[0]
        or tuple(action_scale.shape) != tuple(action_offset.shape)
    ):
        raise ValueError("teacher replay action decoder tensors differ")
    torch._assert_async(torch.isfinite(qdes).all())
    torch._assert_async(torch.isfinite(action_offset).all())
    torch._assert_async(torch.isfinite(action_scale).all())
    torch._assert_async(torch.all(action_scale != 0.0))
    action = (qdes - action_offset.unsqueeze(0)) / action_scale.unsqueeze(0)
    torch._assert_async(torch.isfinite(action).all())
    return action


def tensor_f32_sha256(value) -> str:
    """Hash one tensor as canonical contiguous little-endian float32 bytes."""

    import numpy as np

    array = value.detach().cpu().numpy().astype("<f4", copy=False)
    array = np.ascontiguousarray(array)
    if not np.isfinite(array).all():
        raise ValueError("teacher replay hash operand is non-finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


__all__ = [
    "TEACHER_REPLAY_SCHEMA_VERSION",
    "TEACHER_REPLAY_DEFAULT_STEPS",
    "TEACHER_REPLAY_NUM_ENVS",
    "TEACHER_REPLAY_ACTION_DELAY_STEPS",
    "TeacherReplayPreStep",
    "capture_teacher_replay_pre_step",
    "contact_capture_boundary",
    "validate_contact_patch_shot",
    "remaining_teacher_frozen_steps",
    "frozen_teacher_qdes",
    "decode_teacher_qdes_to_action",
    "tensor_f32_sha256",
]
