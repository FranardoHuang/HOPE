#!/usr/bin/env python3
"""Content-bound protected-window digests for canonical motion files.

Contract: docs/interfaces/motion_preprocessing_contract.md §5.1.  Every formal
source and output motion must carry a digest that content-binds its strike
window slice across the six timed schema-2 channels, so windows can never be
cross-spliced between files, scopes, motions, or builds.

The SHA-256 input stream is exactly:

    ASCII domain + NUL
    + little-endian uint64 header-byte-length + canonical-JSON header bytes
    + per channel (fixed order):
        little-endian uint64 payload-byte-length + little-endian C-order bytes

The canonical header is UTF-8 JSON with sorted keys, no whitespace, and no
NaN/Inf.  It records role, the motion whole-file SHA, motion_id/scope, the six
channel names/dtypes/full-motion shapes/slice shapes, C-order, the frame-index
dtype (little-endian int64), and the complete integer index vector.

Source windows index every original frame of the inclusive protected span.
Output markers may land on fractional frames: the digest indexes the closed
integer range ``ceil(start)..floor(end)`` while the transformation receipt
stores the exact binary64 hex of both fractional endpoints — markers are never
rounded to the nearest frame.

This module never grants publication, training, or hardware capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DIGEST_DOMAIN = "canonical-protected-window-v1"
TRANSFORMATION_RECEIPT_TYPE = "canonical-window-transformation-receipt-v1"

_ROLES = ("source", "output")
_FRAME_INDEX_DTYPE = "<i8"

# (name, exact little-endian on-disk dtype, expected non-time shape)
WINDOW_CHANNELS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("joint_pos", "<f4", (31,)),
    ("joint_vel", "<f4", (31,)),
    ("body_pos_w", "<f4", (32, 3)),
    ("body_quat_w", "<f4", (32, 4)),
    ("body_lin_vel_w", "<f4", (32, 3)),
    ("body_ang_vel_w", "<f4", (32, 3)),
)


class ProtectedWindowError(ValueError):
    """Raised whenever a digest or receipt cannot be certified fail-closed."""


@dataclass(frozen=True)
class ProtectedWindowDigest:
    """One recomputable digest plus everything it binds."""

    role: str
    motion_id: str
    scope: str
    motion_sha256: str
    frame_indices: tuple[int, ...]
    header: Mapping[str, Any]
    digest_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "digest_domain": DIGEST_DOMAIN,
            "digest_sha256": self.digest_sha256,
            "role": self.role,
            "motion_id": self.motion_id,
            "scope": self.scope,
            "motion_sha256": self.motion_sha256,
            "frame_indices": list(self.frame_indices),
        }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ProtectedWindowError(f"header is not canonical JSON: {exc}") from exc


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or text != text.lower() or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtectedWindowError(f"{label} must be 64 lowercase hex digits")
    return text


def source_window_indices(span_inclusive: Sequence[int]) -> tuple[int, ...]:
    """Every original frame of the inclusive protected source span."""

    if len(span_inclusive) != 2:
        raise ProtectedWindowError("source span must have exactly two entries")
    start, end = (int(value) for value in span_inclusive)
    if not 0 <= start <= end:
        raise ProtectedWindowError(
            f"source span [{start}, {end}] is not a valid inclusive range"
        )
    return tuple(range(start, end + 1))


def output_window_indices(
    start_fractional_frame: float, end_fractional_frame: float
) -> tuple[tuple[int, ...], str, str]:
    """Closed integer range ceil(start)..floor(end) plus exact binary64 hex."""

    start = float(start_fractional_frame)
    end = float(end_fractional_frame)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ProtectedWindowError("fractional window markers must be finite")
    if start < 0.0 or end < start:
        raise ProtectedWindowError(
            f"fractional window [{start!r}, {end!r}] is not ordered"
        )
    first = math.ceil(start)
    last = math.floor(end)
    if last < first:
        raise ProtectedWindowError(
            f"fractional window [{start!r}, {end!r}] contains no integer frame"
        )
    return tuple(range(first, last + 1)), start.hex(), end.hex()


def _load_channels(
    motion_bytes: bytes, label: str
) -> dict[str, np.ndarray]:
    import io

    try:
        with np.load(io.BytesIO(motion_bytes), allow_pickle=False) as payload:
            missing = [
                name for name, _, _ in WINDOW_CHANNELS if name not in payload.files
            ]
            if missing:
                raise ProtectedWindowError(
                    f"{label} is missing timed channels: {missing}"
                )
            arrays = {
                name: payload[name] for name, _, _ in WINDOW_CHANNELS
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ProtectedWindowError):
            raise
        raise ProtectedWindowError(f"{label} is not a readable NPZ: {exc}") from exc
    return arrays


def compute_protected_window_digest(
    motion_path: str | Path,
    *,
    role: str,
    motion_id: str,
    scope: str,
    frame_indices: Sequence[int],
    expected_motion_sha256: str | None = None,
) -> ProtectedWindowDigest:
    """Compute the §5.1 digest from one byte snapshot of the whole file."""

    if role not in _ROLES:
        raise ProtectedWindowError(f"role must be one of {_ROLES}, got {role!r}")
    if not motion_id or not scope:
        raise ProtectedWindowError("motion_id and scope must be non-empty")
    indices = tuple(int(value) for value in frame_indices)
    if not indices:
        raise ProtectedWindowError("frame index vector may not be empty")
    if list(indices) != sorted(set(indices)):
        raise ProtectedWindowError("frame indices must be strictly increasing")
    if indices[0] < 0:
        raise ProtectedWindowError("frame indices must be non-negative")

    path = Path(motion_path)
    try:
        motion_bytes = path.read_bytes()
    except OSError as exc:
        raise ProtectedWindowError(f"cannot read motion file: {exc}") from exc
    motion_sha = hashlib.sha256(motion_bytes).hexdigest()
    if expected_motion_sha256 is not None and motion_sha != _require_sha256(
        expected_motion_sha256, "expected motion SHA-256"
    ):
        raise ProtectedWindowError(
            f"motion bytes drifted: expected {expected_motion_sha256}, "
            f"got {motion_sha}"
        )

    arrays = _load_channels(motion_bytes, f"{role} motion {motion_id}/{scope}")
    slices: list[bytes] = []
    channel_rows: list[dict[str, Any]] = []
    index_array = np.asarray(indices, dtype=_FRAME_INDEX_DTYPE)
    for name, dtype, tail_shape in WINDOW_CHANNELS:
        array = arrays[name]
        if array.dtype.str != dtype:
            raise ProtectedWindowError(
                f"channel {name} must be stored as {dtype}, got {array.dtype.str}"
            )
        if array.ndim != 1 + len(tail_shape) or array.shape[1:] != tail_shape:
            raise ProtectedWindowError(
                f"channel {name} has shape {array.shape}, expected (T, "
                f"{', '.join(str(v) for v in tail_shape)})"
            )
        if indices[-1] >= array.shape[0]:
            raise ProtectedWindowError(
                f"frame index {indices[-1]} leaves channel {name} with "
                f"{array.shape[0]} frames"
            )
        window = np.ascontiguousarray(array[index_array])
        if window.dtype.str != dtype:
            raise ProtectedWindowError(
                f"channel {name} slice changed dtype to {window.dtype.str}"
            )
        payload = window.tobytes(order="C")
        slices.append(payload)
        channel_rows.append(
            {
                "name": name,
                "dtype": dtype,
                "motion_shape": list(array.shape),
                "slice_shape": list(window.shape),
            }
        )

    header: dict[str, Any] = {
        "digest_domain": DIGEST_DOMAIN,
        "role": role,
        "motion_id": motion_id,
        "scope": scope,
        "motion_sha256": motion_sha,
        "channels": channel_rows,
        "byte_order": "little_endian_c_order",
        "frame_index_dtype": _FRAME_INDEX_DTYPE,
        "frame_indices": list(indices),
    }
    header_bytes = _canonical_json_bytes(header)

    digest = hashlib.sha256()
    digest.update(DIGEST_DOMAIN.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack("<Q", len(header_bytes)))
    digest.update(header_bytes)
    for payload in slices:
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)

    return ProtectedWindowDigest(
        role=role,
        motion_id=motion_id,
        scope=scope,
        motion_sha256=motion_sha,
        frame_indices=indices,
        header=header,
        digest_sha256=digest.hexdigest(),
    )


def build_transformation_receipt(
    *,
    motion_id: str,
    scope: str,
    source_digest: ProtectedWindowDigest,
    output_digest: ProtectedWindowDigest,
    source_span_inclusive: Sequence[int],
    output_window_start_fractional_frame: float,
    output_window_end_fractional_frame: float,
    entry_frame: int,
    exit_frame: int,
    marker_time_map: Mapping[str, Mapping[str, Any]],
    binding_sha256: Mapping[str, str],
    allowed_transforms: Sequence[str],
) -> dict[str, Any]:
    """Bind one source window and one output window to the same build."""

    for digest, expected_role in (
        (source_digest, "source"),
        (output_digest, "output"),
    ):
        if digest.role != expected_role:
            raise ProtectedWindowError(
                f"{expected_role} digest has role {digest.role!r}"
            )
        if digest.motion_id != motion_id or digest.scope != scope:
            raise ProtectedWindowError(
                "digest motion_id/scope disagree with the receipt identity"
            )
    if source_digest.digest_sha256 == output_digest.digest_sha256:
        raise ProtectedWindowError(
            "source and output window digests are identical; the compiled "
            "window must be rebuilt, not copied"
        )
    expected_source = source_window_indices(source_span_inclusive)
    if source_digest.frame_indices != expected_source:
        raise ProtectedWindowError(
            "source digest indices disagree with the protected span"
        )
    expected_output, start_hex, end_hex = output_window_indices(
        output_window_start_fractional_frame,
        output_window_end_fractional_frame,
    )
    if output_digest.frame_indices != expected_output:
        raise ProtectedWindowError(
            "output digest indices disagree with the fractional window"
        )
    required_bindings = {
        "recipe_sha256",
        "compiler_sha256",
        "mjcf_sha256",
        "urdf_sha256",
        "body_order_sha256",
    }
    missing = sorted(required_bindings - set(binding_sha256))
    if missing:
        raise ProtectedWindowError(f"receipt bindings are missing: {missing}")
    bindings = {
        str(name): _require_sha256(value, f"binding {name}")
        for name, value in sorted(binding_sha256.items())
    }
    transforms = [str(value) for value in allowed_transforms]
    if not transforms:
        raise ProtectedWindowError("allowed transform list may not be empty")
    if not isinstance(marker_time_map, Mapping) or not marker_time_map:
        raise ProtectedWindowError("marker time map may not be empty")

    receipt = {
        "receipt_type": TRANSFORMATION_RECEIPT_TYPE,
        "motion_id": motion_id,
        "scope": scope,
        "entry_frame": int(entry_frame),
        "exit_frame": int(exit_frame),
        "source": {
            "motion_sha256": source_digest.motion_sha256,
            "window_digest_sha256": source_digest.digest_sha256,
            "span_inclusive": [int(v) for v in source_span_inclusive],
            "frame_indices": list(source_digest.frame_indices),
        },
        "output": {
            "motion_sha256": output_digest.motion_sha256,
            "window_digest_sha256": output_digest.digest_sha256,
            "window_start_fractional_frame_hex": start_hex,
            "window_end_fractional_frame_hex": end_hex,
            "window_start_fractional_frame": float(
                output_window_start_fractional_frame
            ),
            "window_end_fractional_frame": float(
                output_window_end_fractional_frame
            ),
            "frame_indices": list(output_digest.frame_indices),
        },
        "marker_time_map": {
            str(name): dict(row) for name, row in marker_time_map.items()
        },
        "bindings_sha256": bindings,
        "allowed_transforms": transforms,
        "non_claims": [
            "no_behavior_window_certification",
            "no_training_deployment_or_hardware_authorization",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    ).hexdigest()
    return receipt


def verify_transformation_receipt(
    receipt: Mapping[str, Any],
    *,
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Recompute both digests from exact bytes and reject any cross-splice."""

    if receipt.get("receipt_type") != TRANSFORMATION_RECEIPT_TYPE:
        raise ProtectedWindowError("unknown transformation receipt type")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_receipt_sha = _require_sha256(
        receipt.get("receipt_sha256"), "receipt_sha256"
    )
    actual_receipt_sha = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    if actual_receipt_sha != expected_receipt_sha:
        raise ProtectedWindowError("transformation receipt bytes drifted")

    motion_id = str(receipt["motion_id"])
    scope = str(receipt["scope"])
    source_row = receipt["source"]
    output_row = receipt["output"]

    start_hex = str(output_row["window_start_fractional_frame_hex"])
    end_hex = str(output_row["window_end_fractional_frame_hex"])
    start = float.fromhex(start_hex)
    end = float.fromhex(end_hex)
    if float(output_row["window_start_fractional_frame"]) != start or float(
        output_row["window_end_fractional_frame"]
    ) != end:
        raise ProtectedWindowError(
            "fractional marker decimal values disagree with their exact hex"
        )

    recomputed_source = compute_protected_window_digest(
        source_path,
        role="source",
        motion_id=motion_id,
        scope=scope,
        frame_indices=source_window_indices(source_row["span_inclusive"]),
        expected_motion_sha256=str(source_row["motion_sha256"]),
    )
    output_indices, _, _ = output_window_indices(start, end)
    recomputed_output = compute_protected_window_digest(
        output_path,
        role="output",
        motion_id=motion_id,
        scope=scope,
        frame_indices=output_indices,
        expected_motion_sha256=str(output_row["motion_sha256"]),
    )
    if recomputed_source.digest_sha256 != str(source_row["window_digest_sha256"]):
        raise ProtectedWindowError(
            "source window digest does not reproduce from exact bytes"
        )
    if recomputed_output.digest_sha256 != str(output_row["window_digest_sha256"]):
        raise ProtectedWindowError(
            "output window digest does not reproduce from exact bytes"
        )
    if recomputed_source.digest_sha256 == recomputed_output.digest_sha256:
        raise ProtectedWindowError(
            "source and output digests are identical (cross-splice)"
        )
    return {
        "verdict": "WINDOW_DIGESTS_REPRODUCED",
        "receipt_sha256": expected_receipt_sha,
        "source_window_digest_sha256": recomputed_source.digest_sha256,
        "output_window_digest_sha256": recomputed_output.digest_sha256,
        "non_claims": [
            "no_behavior_window_certification",
            "no_training_deployment_or_hardware_authorization",
        ],
    }


__all__ = [
    "DIGEST_DOMAIN",
    "TRANSFORMATION_RECEIPT_TYPE",
    "WINDOW_CHANNELS",
    "ProtectedWindowDigest",
    "ProtectedWindowError",
    "build_transformation_receipt",
    "compute_protected_window_digest",
    "output_window_indices",
    "source_window_indices",
    "verify_transformation_receipt",
]
