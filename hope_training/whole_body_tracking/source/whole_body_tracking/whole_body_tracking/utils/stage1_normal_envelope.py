"""Content-bound per-clip demanded-normal envelope for formal Stage-1 exports.

This module is deliberately Isaac-free.  The native exporter calls it after the
runtime ``QuestionBank`` has passed the full schema-3 loader; the standalone
exporter first invokes that same loader and then calls this module.  Envelope
statistics are computed from the exact train-bank NPZ rows, never from an exam
bank, a filename, or a donor ONNX.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ENVELOPE_SCHEMA_VERSION = "1"
ENVELOPE_FRAME = "world_table_frame0"
ENVELOPE_FACE_CONVENTION = "mount_plusY_A"
ENVELOPE_PAIRING = "shared_plus_y"
ENVELOPE_ALGORITHM = "per_clip_sign_preserving_spherical_mean_cap_v1"
BANK_ROW_UNIT_TOLERANCE = "0.0002"
RUNTIME_UNIT_TOLERANCE = "0.000001"
RUNTIME_DOT_TOLERANCE = "0.000001"
FORMAL_CLIP_ORDER = ("forehand", "backhand")
FORMAL_MOUNT_NORMAL_SIGNS = (1.0, -1.0)

_PAYLOAD_KEYS = (
    "stage1_normal_envelope_schema_version",
    "stage1_normal_envelope_frame",
    "stage1_normal_envelope_face_convention",
    "stage1_normal_envelope_pairing",
    "stage1_normal_envelope_algorithm",
    "stage1_normal_envelope_bank_row_unit_tolerance",
    "stage1_normal_envelope_runtime_unit_tolerance",
    "stage1_normal_envelope_runtime_dot_tolerance",
    "stage1_normal_envelope_clip_order",
    "stage1_normal_envelope_mount_normal_sign_per_clip",
    "stage1_normal_envelope_centers",
    "stage1_normal_envelope_reference_normals",
    "stage1_normal_envelope_min_dots",
    "stage1_normal_envelope_row_counts",
    "stage1_normal_envelope_train_bank_sha256",
    "stage1_normal_envelope_source_family_sha256",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float(value: float) -> str:
    return format(float(value), ".17g")


def normal_envelope_payload(metadata: Mapping[str, str]) -> str:
    """Canonical byte payload hashed by both the Python exporter and C++ loader."""
    missing = [key for key in _PAYLOAD_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"normal-envelope payload is missing keys: {missing}")
    return "".join(f"{key}={metadata[key]}\n" for key in _PAYLOAD_KEYS)


def derive_stage1_normal_envelope(
    bank_path: str | Path,
    *,
    expected_train_bank_sha256: str,
    expected_source_family_sha256: str,
    mount_normal_sign_per_clip: Sequence[float],
    clip_order: Sequence[str] = FORMAL_CLIP_ORDER,
) -> dict[str, str]:
    """Derive a deterministic per-clip spherical cap from one exact train bank.

    Each demanded normal is normalized only after enforcing the schema-3 bank's
    2e-4 unit-length tolerance.  Rows must remain in the same open hemisphere as
    that clip's raw +Y/A-frame reference normal.  The cap axis is the normalized
    sum of only that clip's rows and its threshold is the minimum row dot.  Thus
    forehand/backhand and opposite-sign faces can never cancel through averaging.
    """
    path = Path(bank_path).resolve()
    if not path.is_file():
        raise ValueError(f"normal-envelope train bank does not exist: {path}")
    clip_order = tuple(str(name) for name in clip_order)
    if clip_order != FORMAL_CLIP_ORDER:
        raise ValueError(
            f"formal normal envelope requires clip order {FORMAL_CLIP_ORDER}, got {clip_order}"
        )
    try:
        mount_signs = tuple(float(value) for value in mount_normal_sign_per_clip)
    except (TypeError, ValueError) as exc:
        raise ValueError("formal normal envelope mount-normal sign table must be numeric") from exc
    if mount_signs != FORMAL_MOUNT_NORMAL_SIGNS:
        raise ValueError(
            "formal normal envelope requires mount_normal_sign_per_clip=[+1,-1], got "
            f"{mount_signs}"
        )
    before_sha = _sha256_file(path)
    if before_sha != expected_train_bank_sha256:
        raise ValueError(
            "normal-envelope train-bank SHA disagrees with the checkpoint/export contract: "
            f"actual={before_sha}, expected={expected_train_bank_sha256}"
        )
    if len(expected_source_family_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_source_family_sha256
    ):
        raise ValueError("normal-envelope source-family SHA must be lowercase hex64")

    with np.load(path) as data:
        if "meta_json" not in data:
            raise ValueError("normal-envelope source bank has no schema-3 meta_json")
        try:
            metadata = json.loads(
                bytes(np.asarray(data["meta_json"], dtype=np.uint8)).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("normal-envelope source bank has malformed meta_json") from exc
        required = {
            "schema_version": 3,
            "split": "train",
            "stage": "S1",
            "face_frame": ENVELOPE_FACE_CONVENTION,
        }
        mismatch = {
            key: (metadata.get(key), expected)
            for key, expected in required.items()
            if metadata.get(key) != expected
        }
        if mismatch:
            raise ValueError(f"normal-envelope source is not an exact S1 train bank: {mismatch}")
        if tuple(metadata.get("clip_order") or ()) != clip_order:
            raise ValueError(
                "normal-envelope bank clip order disagrees with the runner mapping: "
                f"{metadata.get('clip_order')!r}"
            )
        family = metadata.get("source_family_contract")
        family_sha = str(metadata.get("source_family_sha256", ""))
        if not isinstance(family, dict) or _canonical_sha256(family) != family_sha:
            raise ValueError("normal-envelope source-family contract/hash is inconsistent")
        family_header = {
            "contract": "stage1-source-family-v2",
            "stage": "S1",
            "face_frame": ENVELOPE_FACE_CONVENTION,
            "clip_order": list(clip_order),
        }
        family_mismatch = {
            key: (family.get(key), expected)
            for key, expected in family_header.items()
            if family.get(key) != expected
        }
        if family_mismatch:
            raise ValueError(
                f"normal-envelope source-family header is incompatible: {family_mismatch}"
            )
        if family_sha != expected_source_family_sha256:
            raise ValueError(
                "normal-envelope source-family SHA disagrees with the checkpoint/export contract"
            )

        centers: list[np.ndarray] = []
        references: list[np.ndarray] = []
        min_dots: list[float] = []
        counts: list[int] = []
        clips_meta = metadata.get("clips") or {}
        for clip, name in enumerate(clip_order):
            normal_key = f"{name}/demanded_normal"
            reference_key = f"{name}/clip_normal"
            if normal_key not in data or reference_key not in data:
                raise ValueError(
                    f"normal-envelope source clip {name!r} lacks demanded/reference normals"
                )
            rows = np.asarray(data[normal_key], dtype=np.float64)
            reference = np.asarray(data[reference_key], dtype=np.float64)
            if rows.ndim != 2 or rows.shape[1:] != (3,) or rows.shape[0] < 1:
                raise ValueError(
                    f"normal-envelope clip {name!r} rows must be non-empty (Q,3), got {rows.shape}"
                )
            if reference.shape != (3,) or not np.isfinite(rows).all() or not np.isfinite(reference).all():
                raise ValueError(f"normal-envelope clip {name!r} contains malformed/NaN/Inf normals")
            recorded_count = (clips_meta.get(name) or {}).get("question_count")
            if int(recorded_count if recorded_count is not None else -1) != rows.shape[0]:
                raise ValueError(
                    f"normal-envelope clip {name!r} question_count disagrees with row bytes"
                )
            row_norms = np.linalg.norm(rows, axis=1)
            reference_norm = float(np.linalg.norm(reference))
            unit_tol = float(BANK_ROW_UNIT_TOLERANCE)
            if not np.all(np.abs(row_norms - 1.0) <= unit_tol):
                raise ValueError(
                    f"normal-envelope clip {name!r} demanded rows exceed unit tolerance {unit_tol}"
                )
            if abs(reference_norm - 1.0) > unit_tol:
                raise ValueError(
                    f"normal-envelope clip {name!r} reference normal exceeds unit tolerance"
                )
            rounded_reference = np.round(reference, 12).tolist()
            recorded_reference = (clips_meta.get(name) or {}).get("clip_normal")
            family_reference = ((family.get("clips") or {}).get(name) or {}).get("clip_normal")
            if recorded_reference != rounded_reference or family_reference != rounded_reference:
                raise ValueError(
                    f"normal-envelope clip {name!r} reference array/provenance disagree"
                )
            unit_rows = rows / row_norms[:, None]
            unit_reference = reference / reference_norm
            # Bank/actor commands remain in raw mount +Y/A.  The external schema-2 wire is the
            # physical striking face B and must be opponent-facing +X.  Therefore the representable
            # condition is sign[clip] * n_A.x > 1e-6: FH raw A stays positive; BH raw A is negative
            # and becomes positive only after the -1 striking-face conversion.
            if np.any(mount_signs[clip] * unit_rows[:, 0] <= 1e-6):
                raise ValueError(
                    f"normal-envelope clip {name!r} contains a raw-A row the physical-B "
                    "schema-2 wire cannot represent (sign[clip] * raw_A.x must be > 1e-6)"
                )
            signed_dots = unit_rows @ unit_reference
            # Export and runtime share one open-hemisphere margin. Accepting a bank row at a
            # smaller positive dot than the C++ gate would create a training question the
            # deployed actor can never receive.
            reference_dot_min = float(RUNTIME_DOT_TOLERANCE)
            if np.any(signed_dots <= reference_dot_min):
                raise ValueError(
                    f"normal-envelope clip {name!r} is not inside the runtime raw +Y/A "
                    f"reference hemisphere (row dot must be > {RUNTIME_DOT_TOLERANCE}); "
                    "opposite faces must never be averaged; near-boundary rows are also rejected"
                )
            vector_sum = unit_rows.sum(axis=0)
            vector_sum_norm = float(np.linalg.norm(vector_sum))
            if not math.isfinite(vector_sum_norm) or vector_sum_norm <= 1e-12:
                raise ValueError(f"normal-envelope clip {name!r} has an undefined spherical mean")
            center = vector_sum / vector_sum_norm
            if float(center @ unit_reference) <= 0.0:
                raise ValueError(
                    f"normal-envelope clip {name!r} spherical mean flipped from its +Y/A reference"
                )
            min_dot = float(np.min(unit_rows @ center))
            if not math.isfinite(min_dot) or not (0.0 < min_dot <= 1.0 + 1e-12):
                raise ValueError(f"normal-envelope clip {name!r} produced an invalid cap threshold")
            centers.append(center)
            references.append(unit_reference)
            min_dots.append(min(min_dot, 1.0))
            counts.append(int(rows.shape[0]))

    after_sha = _sha256_file(path)
    if after_sha != before_sha:
        raise ValueError("normal-envelope train bank changed while it was being exported")

    def vectors(values: Sequence[np.ndarray]) -> str:
        return ";".join(",".join(_float(item) for item in row) for row in values)

    result = {
        "stage1_normal_envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "stage1_normal_envelope_frame": ENVELOPE_FRAME,
        "stage1_normal_envelope_face_convention": ENVELOPE_FACE_CONVENTION,
        "stage1_normal_envelope_pairing": ENVELOPE_PAIRING,
        "stage1_normal_envelope_algorithm": ENVELOPE_ALGORITHM,
        "stage1_normal_envelope_bank_row_unit_tolerance": BANK_ROW_UNIT_TOLERANCE,
        "stage1_normal_envelope_runtime_unit_tolerance": RUNTIME_UNIT_TOLERANCE,
        "stage1_normal_envelope_runtime_dot_tolerance": RUNTIME_DOT_TOLERANCE,
        "stage1_normal_envelope_clip_order": ",".join(clip_order),
        "stage1_normal_envelope_mount_normal_sign_per_clip": "1,-1",
        "stage1_normal_envelope_centers": vectors(centers),
        "stage1_normal_envelope_reference_normals": vectors(references),
        "stage1_normal_envelope_min_dots": ",".join(_float(value) for value in min_dots),
        "stage1_normal_envelope_row_counts": ",".join(str(value) for value in counts),
        "stage1_normal_envelope_train_bank_sha256": before_sha,
        "stage1_normal_envelope_source_family_sha256": family_sha,
    }
    result["stage1_normal_envelope_payload_sha256"] = hashlib.sha256(
        normal_envelope_payload(result).encode("utf-8")
    ).hexdigest()
    return result
