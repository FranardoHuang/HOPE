#!/usr/bin/env python3
"""Construction-owned Device-R05 target-profile authority.

Callers provide semantic source material, never a ``DeviceProfileProjection``
or a caller-authored content digest.  This module freezes the source material,
reconstructs the canonical C03 profile, derives the D05 binding, owns the
device tensor, and mints the only receipt accepted by the authority.

The authority is production-shaped but is not yet consumed by the production
runtime factory.  Consequently this module does not authorize a diagnostic or
training launch by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import struct
from typing import NoReturn, Sequence, Tuple

import torch

try:  # Focused tests import sibling modules directly from the source folder.
    import action_ball_continuous_runtime_transaction_device as _r05
    import action_ball_continuous_target_sampler as _c03
except ImportError:  # Installed-package import.
    from . import action_ball_continuous_runtime_transaction_device as _r05
    from . import action_ball_continuous_target_sampler as _c03


INTEGRATION_STATUS = "production_authority_constructible_factory_not_consuming"
PRODUCTION_INTEGRATED = False
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

_SPEC_CONSTRUCTION_KEY = object()
_OWNER_CONSTRUCTION_KEY = object()
_LANDING_COMPONENTS = ("landing_x_m", "landing_y_m")
_MAX_SUPPORT = (1 << 31) - 1


class DeviceProfileAuthorityError(RuntimeError):
    """The target-profile source, construction request, or receipt is invalid."""


class DeviceProfileAuthorityConflictError(DeviceProfileAuthorityError):
    """A foreign or caller-forged receipt was presented to the owner."""


@dataclass(frozen=True, slots=True, init=False)
class FrozenDeviceTargetProfileSpec:
    """Immutable code-owned semantic source for one finite target support.

    The spec deliberately contains no profile, semantic, or binding digest.
    ``frame_binding_sha256`` is an independently reviewed frame-authority root,
    not a digest manufactured to make this profile pass.  Per-cell semantic
    roots and the profile/binding digests are derived only during authority
    construction.
    """

    frame_id: str
    frame_binding_sha256: str
    cell_ids: Tuple[str, ...]
    targets_xy_f32: Tuple[Tuple[float, float], ...]

    def __init__(
        self,
        construction_key: object,
        *,
        frame_id: str,
        frame_binding_sha256: str,
        cell_ids: Tuple[str, ...],
        targets_xy_f32: Tuple[Tuple[float, float], ...],
    ) -> None:
        if construction_key is not _SPEC_CONSTRUCTION_KEY:
            raise TypeError(
                "Frozen device target profile specs require the freeze factory"
            )
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "frame_binding_sha256", frame_binding_sha256)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "targets_xy_f32", targets_xy_f32)


class DeviceProfileReceipt:
    """Opaque same-process identity minted by one exact profile owner."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("device profile receipts are owner-issued")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("device profile receipts are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("device profile receipts cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("device profile receipts cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("device profile receipts cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("device profile receipts cannot be serialized")


@dataclass(frozen=True, slots=True)
class _OwnedProfileRecord:
    profile_sha256: str
    profile_binding_sha256: str
    cell_ids: Tuple[str, ...]
    semantic_sha256s: Tuple[str, ...]
    targets_xy_m: torch.Tensor


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise DeviceProfileAuthorityError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DeviceProfileAuthorityError(
            f"{label} must be a lowercase SHA-256 semantic root"
        )
    return value


def _float32(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DeviceProfileAuthorityError(f"{label} must be a finite number")
    clean = float(value)
    if not math.isfinite(clean):
        raise DeviceProfileAuthorityError(f"{label} must be finite")
    try:
        quantized = struct.unpack(">f", struct.pack(">f", clean))[0]
    except (OverflowError, struct.error) as exc:
        raise DeviceProfileAuthorityError(
            f"{label} must be finite in runtime float32"
        ) from exc
    if not math.isfinite(quantized):
        raise DeviceProfileAuthorityError(
            f"{label} must be finite in runtime float32"
        )
    return 0.0 if quantized == 0.0 else quantized


def freeze_device_target_profile_spec(
    *,
    frame_id: str,
    frame_binding_sha256: str,
    cell_ids: Sequence[str],
    targets_xy_m: Sequence[Sequence[Real]],
) -> FrozenDeviceTargetProfileSpec:
    """Clone and freeze reviewed semantic material without accepting digests.

    Lists are accepted as materialization input but no list, row, or tensor is
    retained.  Values are quantized to the exact runtime binary32 domain here;
    authority construction independently reconstructs the canonical profile
    and owns the resulting device tensor.
    """

    clean_frame_id = _require_text(frame_id, label="frame_id")
    clean_frame_root = _require_sha256(
        frame_binding_sha256, label="frame_binding_sha256"
    )
    if not isinstance(cell_ids, (tuple, list)):
        raise DeviceProfileAuthorityError("cell_ids must be a finite sequence")
    clean_cell_ids = tuple(
        _require_text(cell_id, label=f"cell_ids[{index}]")
        for index, cell_id in enumerate(tuple(cell_ids))
    )
    if not (2 <= len(clean_cell_ids) <= _MAX_SUPPORT):
        raise DeviceProfileAuthorityError(
            "target support must contain at least two cells"
        )
    if len(set(clean_cell_ids)) != len(clean_cell_ids):
        raise DeviceProfileAuthorityError("cell_ids must be unique")
    if not isinstance(targets_xy_m, (tuple, list)):
        raise DeviceProfileAuthorityError(
            "targets_xy_m must be a finite row sequence"
        )
    source_rows = tuple(targets_xy_m)
    if len(source_rows) != len(clean_cell_ids):
        raise DeviceProfileAuthorityError(
            "targets_xy_m row count must equal the cell count"
        )
    clean_rows = []
    for row_index, row in enumerate(source_rows):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise DeviceProfileAuthorityError(
                f"targets_xy_m[{row_index}] must have shape (2,)"
            )
        clean_rows.append(
            (
                _float32(row[0], label=f"targets_xy_m[{row_index}][0]"),
                _float32(row[1], label=f"targets_xy_m[{row_index}][1]"),
            )
        )
    frozen_rows = tuple(clean_rows)
    if len(set(frozen_rows)) != len(frozen_rows):
        raise DeviceProfileAuthorityError(
            "different cell IDs must name unique runtime float32 targets"
        )
    return FrozenDeviceTargetProfileSpec(
        _SPEC_CONSTRUCTION_KEY,
        frame_id=clean_frame_id,
        frame_binding_sha256=clean_frame_root,
        cell_ids=clean_cell_ids,
        targets_xy_f32=frozen_rows,
    )


def _binding_sha256(
    *,
    profile_sha256: str,
    cell_ids: Tuple[str, ...],
    semantic_sha256s: Tuple[str, ...],
    targets_xy_f32: Tuple[Tuple[float, float], ...],
) -> str:
    binding = hashlib.sha256()
    binding.update(profile_sha256.encode("ascii"))
    for cell_id, semantic_sha256 in zip(cell_ids, semantic_sha256s):
        cell_id_bytes = cell_id.encode("utf-8")
        binding.update(len(cell_id_bytes).to_bytes(8, "big"))
        binding.update(cell_id_bytes)
        binding.update(bytes.fromhex(semantic_sha256))
    for row in targets_xy_f32:
        for value in row:
            binding.update(struct.pack(">f", value))
    return binding.hexdigest()


def _materialize_record(
    spec: FrozenDeviceTargetProfileSpec,
    *,
    device: torch.device,
    expected_support_size: int,
) -> _OwnedProfileRecord:
    if type(spec) is not FrozenDeviceTargetProfileSpec:
        raise DeviceProfileAuthorityError(
            "construction requires one FrozenDeviceTargetProfileSpec"
        )
    if (
        type(device) is not torch.device
        or device.type not in ("cpu", "cuda")
        or (device.type == "cuda" and device.index is None)
    ):
        raise DeviceProfileAuthorityError(
            "device must be an exact CPU or indexed CUDA torch.device"
        )
    if (
        type(expected_support_size) is not int
        or expected_support_size < 2
        or expected_support_size > _MAX_SUPPORT
    ):
        raise DeviceProfileAuthorityError(
            "expected_support_size must be an exact supported int"
        )

    # Reconstruct from immutable semantic material.  No supplied profile or
    # semantic digest participates in this derivation.
    try:
        portable = _c03.ContinuousTargetProfile(
            frame_id=spec.frame_id,
            frame_binding_sha256=spec.frame_binding_sha256,
            runtime_dtype=_c03.RUNTIME_DTYPE,
            quantization_contract=_c03.QUANTIZATION_CONTRACT,
            components=_LANDING_COMPONENTS,
            cells=tuple(
                _c03.TargetCell(cell_id=cell_id, target=target)
                for cell_id, target in zip(spec.cell_ids, spec.targets_xy_f32)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DeviceProfileAuthorityError(
            "frozen target profile semantic material is invalid"
        ) from exc
    if len(portable.cells) != expected_support_size:
        raise DeviceProfileAuthorityError(
            "expected_support_size differs from the frozen profile"
        )
    cell_ids = tuple(cell.cell_id for cell in portable.cells)
    semantic_sha256s = tuple(
        portable.semantic_sha256(cell) for cell in portable.cells
    )
    targets = tuple(
        (float(cell.target[0]), float(cell.target[1])) for cell in portable.cells
    )
    binding = _binding_sha256(
        profile_sha256=portable.profile_sha256,
        cell_ids=cell_ids,
        semantic_sha256s=semantic_sha256s,
        targets_xy_f32=targets,
    )
    try:
        targets_owned = torch.tensor(
            targets, dtype=torch.float32, device=device
        ).contiguous()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DeviceProfileAuthorityError(
            "target profile could not be materialized on the requested device"
        ) from exc
    if (
        tuple(targets_owned.shape) != (expected_support_size, 2)
        or targets_owned.dtype is not torch.float32
        or targets_owned.device != device
        or not targets_owned.is_contiguous()
        or not bool(torch.all(torch.isfinite(targets_owned)))
    ):
        raise DeviceProfileAuthorityError(
            "owned target tensor device, shape, dtype, or finiteness differs"
        )
    return _OwnedProfileRecord(
        profile_sha256=portable.profile_sha256,
        profile_binding_sha256=binding,
        cell_ids=cell_ids,
        semantic_sha256s=semantic_sha256s,
        targets_xy_m=targets_owned.clone(),
    )


class DeviceProfileAuthorityOwner:
    """Exact owner of one immutable target support and its receipt identity."""

    __slots__ = ("__record", "__receipt")

    def __init__(
        self,
        construction_key: object,
        *,
        record: _OwnedProfileRecord,
    ) -> None:
        if construction_key is not _OWNER_CONSTRUCTION_KEY:
            raise TypeError("device profile authorities require the constructor")
        self.__record = record
        self.__receipt = object.__new__(DeviceProfileReceipt)

    def __copy__(self) -> NoReturn:
        raise TypeError("device profile authorities cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("device profile authorities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("device profile authorities cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("device profile authorities cannot be serialized")

    def _issued_receipt(self) -> DeviceProfileReceipt:
        """Return the construction receipt without exposing record mutation."""

        return self.__receipt

    def require_owned_r05_profile(
        self, receipt: object
    ) -> _r05.DeviceProfileProjection:
        """Return a detached exact D05 projection for this owner's receipt."""

        if type(receipt) is not DeviceProfileReceipt or receipt is not self.__receipt:
            raise DeviceProfileAuthorityConflictError(
                "device profile receipt is foreign or caller-forged"
            )
        record = self.__record
        return _r05.DeviceProfileProjection(
            profile_sha256=record.profile_sha256,
            profile_binding_sha256=record.profile_binding_sha256,
            cell_ids=record.cell_ids,
            semantic_sha256s=record.semantic_sha256s,
            targets_xy_m=record.targets_xy_m.clone(),
        )


def construct_device_profile_authority(
    spec: FrozenDeviceTargetProfileSpec,
    *,
    device: torch.device,
    expected_support_size: int,
) -> tuple[DeviceProfileAuthorityOwner, DeviceProfileReceipt]:
    """Construct one production-shaped owner and mint its opaque receipt."""

    record = _materialize_record(
        spec,
        device=device,
        expected_support_size=expected_support_size,
    )
    owner = DeviceProfileAuthorityOwner(_OWNER_CONSTRUCTION_KEY, record=record)
    return owner, owner._issued_receipt()


__all__ = [
    "DIAGNOSTIC_UNAUTHORIZED",
    "DeviceProfileAuthorityConflictError",
    "DeviceProfileAuthorityError",
    "DeviceProfileAuthorityOwner",
    "DeviceProfileReceipt",
    "FrozenDeviceTargetProfileSpec",
    "INTEGRATION_STATUS",
    "LAUNCH_AUTHORIZED",
    "PRODUCTION_INTEGRATED",
    "RUNTIME_INTEGRATED",
    "construct_device_profile_authority",
    "freeze_device_target_profile_spec",
]
