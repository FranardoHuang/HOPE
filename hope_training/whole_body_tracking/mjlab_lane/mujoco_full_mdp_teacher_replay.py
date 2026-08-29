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


TEACHER_REPLAY_SCHEMA_VERSION = 1
TEACHER_REPLAY_DEFAULT_STEPS = 180
TEACHER_REPLAY_NUM_ENVS = 1
TEACHER_REPLAY_ACTION_DELAY_STEPS = 0


def remaining_teacher_frozen_steps(
    *, torch, common_step: int, reveal_tick, pre_swing_wait_s, step_dt: float
):
    """Return Motion's integer frame-zero hold counter at one actor boundary."""

    if type(common_step) is not int or common_step < 0:
        raise ValueError("teacher replay common step differs")
    if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError("teacher replay policy dt differs")
    if (
        reveal_tick.ndim != 1
        or tuple(pre_swing_wait_s.shape) != tuple(reveal_tick.shape)
        or reveal_tick.dtype != torch.long
    ):
        raise ValueError("teacher replay frozen-step tensors differ")
    elapsed = torch.clamp(
        (common_step - reveal_tick).to(dtype=pre_swing_wait_s.dtype)
        * float(step_dt),
        min=0.0,
    )
    remaining = torch.clamp(pre_swing_wait_s - elapsed, min=0.0)
    frozen = torch.ceil(remaining / float(step_dt) - 1.0e-12).to(torch.long)
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
    "remaining_teacher_frozen_steps",
    "frozen_teacher_qdes",
    "decode_teacher_qdes_to_action",
    "tensor_f32_sha256",
]
