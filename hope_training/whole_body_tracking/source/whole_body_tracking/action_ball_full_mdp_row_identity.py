"""Portable row-wise identity for one ActionBall shot.

The eight tensors are the business identity.  Journal ordinals, engine storage
slots, settlement/payment times and provenance hashes deliberately stay out of
this type.  The leading tensor dimension is the environment row, so an
explicit ``env_id`` would duplicate identity already carried by layout.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch


class ActionEpochShotKeyError(RuntimeError):
    """A shot-key tensor has an invalid structural ABI."""


@dataclass(frozen=True)
class ActionEpochShotKey:
    """Aligned int64 tensors identifying one shot per addressed row/cell."""

    reset_generation: torch.Tensor
    ball_generation: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    shot_index: torch.Tensor
    task_identity: torch.Tensor
    outcome_identity: torch.Tensor
    ball_identity: torch.Tensor

    def clone(self) -> "ActionEpochShotKey":
        return ActionEpochShotKey(
            **{field.name: getattr(self, field.name).clone() for field in fields(self)}
        )


def empty_action_epoch_shot_key(
    shape: Sequence[int], *, device: torch.device | str
) -> ActionEpochShotKey:
    """Return a fresh, structurally valid, wholly unoccupied key image."""

    exact_shape = _exact_shape(shape)
    exact_device = torch.device(device)
    values = {
        field.name: torch.full(
            exact_shape, -1, dtype=torch.int64, device=exact_device
        )
        for field in fields(ActionEpochShotKey)
    }
    return ActionEpochShotKey(**values)


def require_action_epoch_shot_key(
    value: object,
    *,
    shape: Sequence[int],
    device: torch.device | str,
    label: str,
) -> ActionEpochShotKey:
    """Validate only the typed tensor ABI; do not invent a business verdict."""

    if type(value) is not ActionEpochShotKey:
        raise ActionEpochShotKeyError(label + " must be an exact ActionEpochShotKey")
    exact_shape = _exact_shape(shape)
    exact_device = torch.device(device)
    actual_shape, actual_device = _require_internal_abi(value, label=label)
    if actual_shape != exact_shape or actual_device != exact_device:
        raise ActionEpochShotKeyError(label + " has an unexpected shape or device")
    return value


def action_epoch_shot_key_valid(value: ActionEpochShotKey) -> torch.Tensor:
    """Return the device-resident occupied/valid predicate for each key cell."""

    _require_internal_abi(value, label="shot_key")
    return (
        value.reset_generation.ge(0)
        & value.ball_generation.ge(0)
        & value.action_uid.gt(0)
        & value.action_slot.ge(0)
        & value.shot_index.gt(0)
        & value.task_identity.gt(0)
        & value.outcome_identity.gt(0)
        & value.ball_identity.gt(0)
    )


def action_epoch_shot_key_equal(
    left: ActionEpochShotKey, right: ActionEpochShotKey
) -> torch.Tensor:
    """Return elementwise full-key equality without host synchronization."""

    left_shape, left_device = _require_internal_abi(left, label="left")
    right_shape, right_device = _require_internal_abi(right, label="right")
    if left_shape != right_shape or left_device != right_device:
        raise ActionEpochShotKeyError("shot keys have different shapes or devices")
    result = left.reset_generation.eq(right.reset_generation)
    for field in fields(ActionEpochShotKey)[1:]:
        result = result & getattr(left, field.name).eq(getattr(right, field.name))
    return result


def _require_internal_abi(
    value: object, *, label: str
) -> tuple[tuple[int, ...], torch.device]:
    if type(value) is not ActionEpochShotKey:
        raise ActionEpochShotKeyError(label + " must be an exact ActionEpochShotKey")
    first = value.reset_generation
    if type(first) is not torch.Tensor or not first.shape:
        raise ActionEpochShotKeyError(label + " has no tensor row layout")
    shape = tuple(first.shape)
    device = first.device
    occupied_storage_ranges: list[tuple[int, int, int, str]] = []
    for field in fields(ActionEpochShotKey):
        tensor = getattr(value, field.name)
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype is not torch.int64
            or tensor.device != device
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ActionEpochShotKeyError(
                label + "." + field.name + " has an invalid tensor ABI"
            )
        storage_pointer = tensor.untyped_storage().data_ptr()
        byte_start = tensor.storage_offset() * tensor.element_size()
        byte_end = byte_start + tensor.numel() * tensor.element_size()
        for prior_pointer, prior_start, prior_end, prior_name in occupied_storage_ranges:
            if (
                storage_pointer == prior_pointer
                and byte_start < prior_end
                and prior_start < byte_end
            ):
                raise ActionEpochShotKeyError(
                    label
                    + "."
                    + field.name
                    + " overlaps storage with "
                    + label
                    + "."
                    + prior_name
                )
        occupied_storage_ranges.append(
            (storage_pointer, byte_start, byte_end, field.name)
        )
    return shape, device


def _exact_shape(value: Sequence[int]) -> tuple[int, ...]:
    if type(value) not in (tuple, list) or not value:
        raise ActionEpochShotKeyError("shot-key shape must be a non-empty sequence")
    shape = tuple(value)
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ActionEpochShotKeyError("shot-key shape dimensions must be positive ints")
    return shape


__all__ = [
    "ActionEpochShotKey",
    "ActionEpochShotKeyError",
    "action_epoch_shot_key_equal",
    "action_epoch_shot_key_valid",
    "empty_action_epoch_shot_key",
    "require_action_epoch_shot_key",
]
