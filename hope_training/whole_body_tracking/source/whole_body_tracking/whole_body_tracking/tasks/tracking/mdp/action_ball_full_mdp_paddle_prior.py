"""Pure measured-paddle error and fixed coarse-plus-precision kernels.

This module owns no simulator state, manager, observation, gate, or telemetry.
Both FullMDP engines pass their existing same-clock teacher and achieved
paddle tensors through these functions so reward and evidence share one small
mathematical contract without reconstructing either producer.
"""

from __future__ import annotations

import math

import torch


PADDLE_ERROR_NAMES = ("position", "velocity", "signed_face", "long_axis")
PADDLE_ERROR_UNITS = ("m", "m_per_s", "rad", "rad")


def tracking_errors(
    achieved_position: torch.Tensor,
    achieved_velocity: torch.Tensor,
    achieved_signed_face: torch.Tensor,
    achieved_long_axis: torch.Tensor,
    teacher_position: torch.Tensor,
    teacher_velocity: torch.Tensor,
    teacher_signed_face: torch.Tensor,
    teacher_long_axis: torch.Tensor,
) -> torch.Tensor:
    """Return exact ``[N,4]`` teacher-achieved physical errors."""

    tensors = (
        achieved_position,
        achieved_velocity,
        achieved_signed_face,
        achieved_long_axis,
        teacher_position,
        teacher_velocity,
        teacher_signed_face,
        teacher_long_axis,
    )
    reference = achieved_position
    if (
        type(reference) is not torch.Tensor
        or reference.ndim != 2
        or reference.shape[1] != 3
        or reference.dtype not in (torch.float32, torch.float64)
        or any(
            type(value) is not torch.Tensor
            or tuple(value.shape) != tuple(reference.shape)
            or value.device != reference.device
            or value.dtype != reference.dtype
            for value in tensors[1:]
        )
    ):
        raise ValueError("paddle teacher/achieved tensors must exactly match [N,3]")
    return torch.stack(
        (
            torch.linalg.vector_norm(
                achieved_position - teacher_position, dim=1
            ),
            torch.linalg.vector_norm(
                achieved_velocity - teacher_velocity, dim=1
            ),
            torch.acos(
                torch.sum(
                    achieved_signed_face * teacher_signed_face, dim=1
                ).clamp(-1.0, 1.0)
            ),
            torch.acos(
                torch.sum(achieved_long_axis * teacher_long_axis, dim=1).clamp(
                    -1.0, 1.0
                )
            ),
        ),
        dim=1,
    )


def coarse_precision_kernel(
    error: torch.Tensor, *, precision_std: float, coarse_std: float
) -> torch.Tensor:
    """Return the fixed 50/50 precision-exp plus coarse-Cauchy kernel."""

    precision = float(precision_std)
    coarse = float(coarse_std)
    if (
        type(error) is not torch.Tensor
        or error.dtype not in (torch.float32, torch.float64)
        or not math.isfinite(precision)
        or precision <= 0.0
        or not math.isfinite(coarse)
        or coarse <= precision
    ):
        raise ValueError("paddle composite kernel arguments differ")
    precision_kernel = torch.exp(-torch.square(error / precision))
    coarse_kernel = torch.reciprocal(1.0 + torch.square(error / coarse))
    return 0.5 * (precision_kernel + coarse_kernel)


def kernels(
    errors: torch.Tensor,
    *,
    precision_stds: torch.Tensor,
    coarse_stds: torch.Tensor,
) -> torch.Tensor:
    """Vectorized four-channel form used by the MuJoCo reward graph."""

    if (
        type(errors) is not torch.Tensor
        or errors.ndim != 2
        or errors.shape[1] != len(PADDLE_ERROR_NAMES)
        or errors.dtype not in (torch.float32, torch.float64)
        or type(precision_stds) is not torch.Tensor
        or type(coarse_stds) is not torch.Tensor
        or tuple(precision_stds.shape) != (len(PADDLE_ERROR_NAMES),)
        or tuple(coarse_stds.shape) != tuple(precision_stds.shape)
        or precision_stds.device != errors.device
        or coarse_stds.device != errors.device
        or precision_stds.dtype != errors.dtype
        or coarse_stds.dtype != errors.dtype
    ):
        raise ValueError("paddle vectorized kernel tensors differ")
    # The immutable host specs are validated while the environment builds
    # these tensors.  Rechecking their values here would add four GPU
    # reductions to every control step without strengthening the contract.
    return 0.5 * (
        torch.exp(-torch.square(errors / precision_stds))
        + torch.reciprocal(1.0 + torch.square(errors / coarse_stds))
    )


__all__ = [
    "PADDLE_ERROR_NAMES",
    "PADDLE_ERROR_UNITS",
    "tracking_errors",
    "coarse_precision_kernel",
    "kernels",
]
