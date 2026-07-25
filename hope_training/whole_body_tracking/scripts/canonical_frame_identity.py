#!/usr/bin/env python3
"""Content-addressed temporal-frame identity receipts for motion sources.

This module proves a deliberately narrow claim: two independently materialized
motion NPZ files use the same ordered source timeline.  It does *not* claim that
the complete robot pose is identical.  In particular, a grip bake may re-solve
the right wrist while every explicitly declared invariant joint still matches
bit-for-bit at every source frame.

The receipt closes that distinction by recording:

* exact NPZ, provenance, and joint-order input hashes;
* one common content-addressed source PKL and resampling formula;
* explicit invariant joint names and indices;
* bitwise equality of ``joint_pos`` and ``joint_vel`` on those channels;
* a unique, same-index fingerprint for every frame;
* excluded-joint differences rather than silently ignoring them; and
* the fixed SE(2) relationship between invariant world-body trajectories.

Within each build or verification pass, every input is read into bytes exactly
once.  Hashing and parsing use those same bytes, avoiding a hash-then-reopen
time-of-check/time-of-use gap.  Publication performs a fresh exact rebuild,
then is evidence-only, atomic, and no-clobber.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


class FrameIdentityError(ValueError):
    """The requested frame-identity claim is not closed by exact evidence."""


FRAME_IDENTITY_SCHEMA_VERSION = 1
FRAME_IDENTITY_SPEC_SCHEMA_VERSION = 1
PUBLICATION_CLASS = "evidence_only_not_artifact_authorization"
MAPPING_ROLES = frozenset(
    {
        "ordinary_nominal_event_to_bound_recipe_source",
        "synthetic_face_construction_annotation_to_bound_recipe_source",
    }
)

_SHA256_HEX_LENGTH = 64
_NPZ_KEYS = frozenset(
    {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "kinematics_schema_version",
        "body_pos_point",
        "body_lin_vel_point",
        "body_names",
    }
)
_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "motion_id",
        "mapping_role",
        "inputs",
        "invariant_joints",
        "excluded_world_bodies",
        "world_residual_tolerances",
        "semantic_contract",
    }
)
_INPUT_KEYS = frozenset(
    {
        "event_npz",
        "bound_npz",
        "event_provenance",
        "bound_provenance",
        "event_authority",
        "joint_order",
        "joint_bijection",
    }
)
_FILE_BINDING_KEYS = frozenset({"path", "sha256"})
_INVARIANT_JOINT_KEYS = frozenset({"index", "name"})
_TOLERANCE_KEYS = frozenset(
    {
        "position_m",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
        "quaternion_l2_sign_invariant",
    }
)
_SEMANTIC_KEYS = frozenset(
    {
        "frame_identity_only",
        "observed_ball_contact",
        "behavior_authorized",
        "synthetic_behavior_nominal_claim",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "publication_class",
        "authorization",
        "motion_id",
        "mapping_role",
        "semantic_contract",
        "integrity_contract",
        "input_bindings",
        "event_authority_contract",
        "common_lineage",
        "npz_contract",
        "invariant_joint_contract",
        "event_mapping",
        "se2_world_contract",
        "non_claims",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "behavior_authorized",
        "hardware_authorized",
        "artifact_promotion_authorized",
    }
)
_WORLD_RESIDUAL_CEILINGS = MappingProxyType(
    {
        "position_m": 2.0e-7,
        "linear_velocity_m_s": 1.0e-5,
        "angular_velocity_rad_s": 2.0e-5,
        "quaternion_l2_sign_invariant": 2.0e-7,
    }
)
_ALLOWED_EXCLUDED_JOINT_INDICES = frozenset({26, 28, 30})


@dataclass(frozen=True)
class _BoundBytes:
    path: Path
    path_text: str
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class _MotionPayload:
    fps: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    body_names: tuple[str, ...]
    array_contract: Mapping[str, Any]


def _exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrameIdentityError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise FrameIdentityError(f"{label} keys must be strings")
    actual = frozenset(value)
    if actual != expected:
        raise FrameIdentityError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FrameIdentityError(
            f"{label} must be a non-empty string without edge whitespace"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if (
        len(text) != _SHA256_HEX_LENGTH
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise FrameIdentityError(
            f"{label} must be a 64-character lowercase SHA-256"
        )
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise FrameIdentityError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise FrameIdentityError(f"{label} must be >= {minimum}")
    return result


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise FrameIdentityError(f"{label} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FrameIdentityError(
            f"{label} must be a positive finite number"
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FrameIdentityError(f"{label} must be a positive finite number")
    return result


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FrameIdentityError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FrameIdentityError(
                    f"{label} contains non-finite JSON number {token!r}"
                )
            ),
        )
    except FrameIdentityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FrameIdentityError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FrameIdentityError(f"{label} must contain one JSON object")
    return value


def _read_bound_bytes(
    path_value: str | Path, expected_sha256: str, label: str
) -> _BoundBytes:
    expected = _sha256(expected_sha256, f"{label}.sha256")
    path_text = _string(str(path_value), f"{label}.path")
    path = Path(path_text).expanduser().resolve()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FrameIdentityError(f"cannot read {label} {path}: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise FrameIdentityError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return _BoundBytes(
        path=path,
        path_text=path_text,
        sha256=actual,
        payload=payload,
    )


def _scalar_integer(array: np.ndarray, label: str) -> int:
    raw = np.asarray(array)
    if raw.shape != (1,) or raw.dtype != np.dtype("<i8"):
        raise FrameIdentityError(
            f"{label} must have shape (1,) and dtype <i8"
        )
    return int(raw[0])


def _scalar_text(array: np.ndarray, label: str) -> str:
    raw = np.asarray(array)
    if raw.shape != () or raw.dtype.kind not in {"U", "S"}:
        raise FrameIdentityError(f"{label} must be one scalar string")
    return _string(str(raw.item()), label)


def _numeric_array(
    value: Any,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise FrameIdentityError(
            f"{label} shape must be {shape}, got {raw.shape}"
        )
    if raw.dtype != dtype:
        raise FrameIdentityError(
            f"{label} dtype must be {dtype.str}, got {raw.dtype.str}"
        )
    if not np.all(np.isfinite(raw)):
        raise FrameIdentityError(f"{label} contains NaN or infinity")
    return np.array(raw, copy=True, order="C")


def _load_motion_from_same_bytes(bound: _BoundBytes, label: str) -> _MotionPayload:
    try:
        with np.load(io.BytesIO(bound.payload), allow_pickle=False) as archive:
            if (
                len(archive.files) != len(_NPZ_KEYS)
                or frozenset(archive.files) != _NPZ_KEYS
            ):
                raise FrameIdentityError(
                    f"{label} NPZ keys changed; "
                    f"missing={sorted(_NPZ_KEYS - frozenset(archive.files))}, "
                    f"extra={sorted(frozenset(archive.files) - _NPZ_KEYS)}"
                )
            fps_raw = np.asarray(archive["fps"])
            fps = _scalar_integer(fps_raw, f"{label}.fps")
            if fps <= 0:
                raise FrameIdentityError(f"{label}.fps must be positive")
            schema_version_raw = np.asarray(
                archive["kinematics_schema_version"]
            )
            schema_version = _scalar_integer(
                schema_version_raw,
                f"{label}.kinematics_schema_version",
            )
            if schema_version != 2:
                raise FrameIdentityError(
                    f"{label}.kinematics_schema_version must equal 2"
                )
            joint_pos_raw = np.asarray(archive["joint_pos"])
            if joint_pos_raw.ndim != 2:
                raise FrameIdentityError(f"{label}.joint_pos must be 2-D")
            frame_count, joint_count = joint_pos_raw.shape
            body_pos_raw = np.asarray(archive["body_pos_w"])
            if body_pos_raw.ndim != 3 or body_pos_raw.shape[0] != frame_count:
                raise FrameIdentityError(
                    f"{label}.body_pos_w must have shape (frames,bodies,3)"
                )
            body_count = body_pos_raw.shape[1]
            joint_pos = _numeric_array(
                joint_pos_raw,
                (frame_count, joint_count),
                np.dtype("<f4"),
                f"{label}.joint_pos",
            )
            joint_vel = _numeric_array(
                archive["joint_vel"],
                (frame_count, joint_count),
                np.dtype("<f4"),
                f"{label}.joint_vel",
            )
            body_pos = _numeric_array(
                body_pos_raw,
                (frame_count, body_count, 3),
                np.dtype("<f4"),
                f"{label}.body_pos_w",
            )
            body_quat = _numeric_array(
                archive["body_quat_w"],
                (frame_count, body_count, 4),
                np.dtype("<f4"),
                f"{label}.body_quat_w",
            )
            body_lin = _numeric_array(
                archive["body_lin_vel_w"],
                (frame_count, body_count, 3),
                np.dtype("<f4"),
                f"{label}.body_lin_vel_w",
            )
            body_ang = _numeric_array(
                archive["body_ang_vel_w"],
                (frame_count, body_count, 3),
                np.dtype("<f4"),
                f"{label}.body_ang_vel_w",
            )
            body_names_raw = np.asarray(archive["body_names"])
            if body_names_raw.shape != (body_count,) or body_names_raw.dtype.kind not in {
                "U",
                "S",
            }:
                raise FrameIdentityError(
                    f"{label}.body_names must have shape ({body_count},)"
                )
            body_names = tuple(
                _string(str(item), f"{label}.body_names[{index}]")
                for index, item in enumerate(body_names_raw.tolist())
            )
            if len(set(body_names)) != len(body_names):
                raise FrameIdentityError(f"{label}.body_names are not unique")
            body_pos_point_raw = np.asarray(archive["body_pos_point"])
            body_pos_point = _scalar_text(
                body_pos_point_raw, f"{label}.body_pos_point"
            )
            body_lin_vel_point_raw = np.asarray(
                archive["body_lin_vel_point"]
            )
            body_lin_vel_point = _scalar_text(
                body_lin_vel_point_raw, f"{label}.body_lin_vel_point"
            )
    except FrameIdentityError:
        raise
    except (OSError, ValueError) as exc:
        raise FrameIdentityError(f"cannot parse {label} NPZ: {exc}") from exc

    contract = {
        "fps": {
            "shape": list(fps_raw.shape),
            "dtype": fps_raw.dtype.str,
            "value": fps,
        },
        "joint_pos": {"shape": list(joint_pos.shape), "dtype": joint_pos.dtype.str},
        "joint_vel": {"shape": list(joint_vel.shape), "dtype": joint_vel.dtype.str},
        "body_pos_w": {"shape": list(body_pos.shape), "dtype": body_pos.dtype.str},
        "body_quat_w": {
            "shape": list(body_quat.shape),
            "dtype": body_quat.dtype.str,
        },
        "body_lin_vel_w": {
            "shape": list(body_lin.shape),
            "dtype": body_lin.dtype.str,
        },
        "body_ang_vel_w": {
            "shape": list(body_ang.shape),
            "dtype": body_ang.dtype.str,
        },
        "kinematics_schema_version": {
            "shape": list(schema_version_raw.shape),
            "dtype": schema_version_raw.dtype.str,
            "value": schema_version,
        },
        "body_pos_point": {
            "shape": list(body_pos_point_raw.shape),
            "dtype": body_pos_point_raw.dtype.str,
            "value": body_pos_point,
        },
        "body_lin_vel_point": {
            "shape": list(body_lin_vel_point_raw.shape),
            "dtype": body_lin_vel_point_raw.dtype.str,
            "value": body_lin_vel_point,
        },
        "body_names": {
            "shape": list(body_names_raw.shape),
            "dtype": body_names_raw.dtype.str,
        },
    }
    return _MotionPayload(
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        body_names=body_names,
        array_contract=MappingProxyType(contract),
    )


def _load_joint_order(bound: _BoundBytes) -> tuple[str, ...]:
    try:
        text = bound.payload.decode("utf-8")
    except UnicodeError as exc:
        raise FrameIdentityError("joint order is not UTF-8") from exc
    names = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(names) != 31 or len(set(names)) != 31:
        raise FrameIdentityError(
            "joint order must contain exactly 31 unique non-comment names"
        )
    return names


def _canonical_names_sha256(names: Sequence[str]) -> str:
    payload = json.dumps(
        list(names), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_joint_bijection(
    bound: _BoundBytes,
    *,
    joint_order_sha256: str,
    joint_names: Sequence[str],
) -> None:
    value = _strict_json_bytes(bound.payload, "joint_bijection")
    if _integer(
        value.get("schema_version"), "joint_bijection.schema_version"
    ) != 1:
        raise FrameIdentityError("joint_bijection.schema_version must equal 1")
    if _integer(
        value.get("expected_joint_count"),
        "joint_bijection.expected_joint_count",
    ) != len(joint_names):
        raise FrameIdentityError(
            "joint_bijection expected_joint_count disagrees with joint order"
        )
    target = value.get("target_order")
    if not isinstance(target, Mapping):
        raise FrameIdentityError("joint_bijection.target_order must be an object")
    if target.get("name") != "runtime_articulation_joint_pos":
        raise FrameIdentityError(
            "joint_bijection target_order is not runtime articulation order"
        )
    if _sha256(
        target.get("file_sha256"),
        "joint_bijection.target_order.file_sha256",
    ) != joint_order_sha256:
        raise FrameIdentityError(
            "joint_bijection target-order file hash disagrees with bound joint order"
        )
    if _sha256(
        target.get("names_sha256"),
        "joint_bijection.target_order.names_sha256",
    ) != _canonical_names_sha256(joint_names):
        raise FrameIdentityError(
            "joint_bijection target-order names hash disagrees with parsed names"
        )


def _extract_lineage(
    provenance: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    try:
        source = provenance["source_pkl"]
        output = provenance["output"]
        mjcf = provenance["mjcf"]
        joint_contract = provenance["joint_contract"]
    except KeyError as exc:
        raise FrameIdentityError(
            f"{label} provenance omitted {exc.args[0]!r}"
        ) from exc
    if not isinstance(source, Mapping) or not isinstance(output, Mapping):
        raise FrameIdentityError(f"{label} source_pkl/output must be objects")
    if not isinstance(mjcf, Mapping):
        raise FrameIdentityError(f"{label}.mjcf must be an object")
    result = {
        "source_pkl_path": _string(
            source.get("path"), f"{label}.source_pkl.path"
        ),
        "source_pkl_sha256": _sha256(
            source.get("sha256"), f"{label}.source_pkl.sha256"
        ),
        "source_frames": _integer(
            source.get("frames"), f"{label}.source_pkl.frames", minimum=1
        ),
        "source_fps": _integer(
            source.get("fps"), f"{label}.source_pkl.fps", minimum=1
        ),
        "output_frames": _integer(
            output.get("frames"), f"{label}.output.frames", minimum=1
        ),
        "output_fps": _integer(
            output.get("fps"), f"{label}.output.fps", minimum=1
        ),
        "frame_formula": _string(
            output.get("frame_formula"), f"{label}.output.frame_formula"
        ),
        "mjcf_sha256": _sha256(
            mjcf.get("sha256"), f"{label}.mjcf.sha256"
        ),
        "joint_contract": _string(
            joint_contract, f"{label}.joint_contract"
        ),
    }
    expected_formula = (
        "round(((input_frames-1)/"
        f"{result['source_fps']})*{result['output_fps']})+1"
    )
    if result["frame_formula"] != expected_formula:
        raise FrameIdentityError(
            f"{label}.output.frame_formula must equal {expected_formula!r}"
        )
    expected_output_frames = (
        round(
            (result["source_frames"] - 1)
            / result["source_fps"]
            * result["output_fps"]
        )
        + 1
    )
    if result["output_frames"] != expected_output_frames:
        raise FrameIdentityError(
            f"{label}.output.frames disagrees with source frames/fps and "
            "the declared resampling formula"
        )
    return MappingProxyType(result)


def _extract_event_authority(
    authority: Mapping[str, Any],
    *,
    motion_id: str,
    mapping_role: str,
    event_npz_sha256: str,
    bound_npz_sha256: str,
) -> Mapping[str, Any]:
    if _integer(
        authority.get("schema_version"), "event_authority.schema_version"
    ) != 1:
        raise FrameIdentityError(
            "event_authority.schema_version must equal 1"
        )
    authority_id = _string(
        authority.get("authority_id"), "event_authority.authority_id"
    )
    review_status = _string(
        authority.get("review_status"), "event_authority.review_status"
    )
    scope = _string(authority.get("scope"), "event_authority.scope")
    authorization = _exact_keys(
        authority.get("authorization"),
        _AUTHORIZATION_KEYS,
        "event_authority.authorization",
    )
    if any(value is not False for value in authorization.values()):
        raise FrameIdentityError(
            "event authority contains a non-false authorization"
        )
    semantics = authority.get("semantic_contract")
    if not isinstance(semantics, Mapping):
        raise FrameIdentityError(
            "event_authority.semantic_contract must be an object"
        )
    if semantics.get("contact_truth_observed") is not False:
        raise FrameIdentityError(
            "event_authority semantic contract must deny observed contact"
        )
    if (
        semantics.get("unresolved_or_mismatched_source_binding_policy")
        != "fail_closed"
    ):
        raise FrameIdentityError(
            "event_authority source-binding policy must be fail_closed"
        )

    motions = authority.get("motions")
    if not isinstance(motions, list) or not motions:
        raise FrameIdentityError(
            "event_authority.motions must be a non-empty array"
        )
    indexed: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, raw_motion in enumerate(motions):
        if not isinstance(raw_motion, Mapping):
            raise FrameIdentityError(
                f"event_authority.motions[{index}] must be an object"
            )
        candidate_id = _string(
            raw_motion.get("motion_id"),
            f"event_authority.motions[{index}].motion_id",
        )
        if candidate_id in indexed:
            raise FrameIdentityError(
                f"event authority repeats motion_id {candidate_id!r}"
            )
        indexed[candidate_id] = (index, raw_motion)
    if motion_id not in indexed:
        raise FrameIdentityError(
            f"event authority has no motion_id {motion_id!r}"
        )
    motion_index, motion = indexed[motion_id]

    bound_source = motion.get("bound_recipe_source")
    nominal_event = motion.get("nominal_event")
    if not isinstance(bound_source, Mapping) or not isinstance(
        nominal_event, Mapping
    ):
        raise FrameIdentityError(
            "authority motion must contain bound_recipe_source and nominal_event"
        )
    authority_bound_sha = _sha256(
        bound_source.get("sha256"),
        "event_authority.bound_recipe_source.sha256",
    )
    if authority_bound_sha != bound_npz_sha256:
        raise FrameIdentityError(
            "event authority bound source SHA disagrees with bound NPZ input"
        )
    authority_event_sha = _sha256(
        nominal_event.get("event_source_sha256"),
        "event_authority.nominal_event.event_source_sha256",
    )
    if authority_event_sha != event_npz_sha256:
        raise FrameIdentityError(
            "event authority event source SHA disagrees with event NPZ input"
        )
    if nominal_event.get("contact_truth_observed") is not False:
        raise FrameIdentityError(
            "authority nominal event must deny observed ball contact"
        )
    semantic_kind = _string(
        nominal_event.get("semantic_kind"),
        "event_authority.nominal_event.semantic_kind",
    )
    is_synthetic_kind = "synthetic" in semantic_kind.lower()
    if (
        mapping_role
        == "ordinary_nominal_event_to_bound_recipe_source"
        and is_synthetic_kind
    ):
        raise FrameIdentityError(
            "ordinary mapping role cannot select a synthetic authority event"
        )
    if (
        mapping_role
        == "synthetic_face_construction_annotation_to_bound_recipe_source"
        and not is_synthetic_kind
    ):
        raise FrameIdentityError(
            "synthetic mapping role requires a synthetic authority event"
        )
    return MappingProxyType(
        {
            "authority_id": authority_id,
            "review_status": review_status,
            "scope": scope,
            "motion_index": motion_index,
            "motion_json_pointer": f"/motions/{motion_index}",
            "event_frame_json_pointer": (
                f"/motions/{motion_index}/nominal_event/frame"
            ),
            "event_source_frame": _integer(
                nominal_event.get("frame"),
                "event_authority.nominal_event.frame",
            ),
            "event_source_remote_path": _string(
                nominal_event.get("event_source_remote_path"),
                "event_authority.nominal_event.event_source_remote_path",
            ),
            "event_source_sha256": authority_event_sha,
            "bound_recipe_source_path": _string(
                bound_source.get("path"),
                "event_authority.bound_recipe_source.path",
            ),
            "bound_recipe_source_sha256": authority_bound_sha,
            "semantic_kind": semantic_kind,
            "contact_truth_observed": False,
            "evidence_json_pointer": _string(
                nominal_event.get("evidence_json_pointer"),
                "event_authority.nominal_event.evidence_json_pointer",
            ),
            "mapping_status_before_receipt": _string(
                nominal_event.get("mapping_to_bound_recipe_source"),
                (
                    "event_authority.nominal_event."
                    "mapping_to_bound_recipe_source"
                ),
            ),
        }
    )


def _se2_transform_from_provenance(
    provenance: Mapping[str, Any], label: str
) -> tuple[float, np.ndarray]:
    probe = provenance.get("se2_station_probe")
    yaw_deg = 0.0
    translation = np.zeros(3, dtype=np.float64)
    if probe is not None:
        if not isinstance(probe, Mapping):
            raise FrameIdentityError(f"{label}.se2_station_probe must be an object")
        try:
            yaw_deg = float(probe["yaw_deg"])
            translation_raw = np.asarray(
                probe["translation_w_m"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameIdentityError(
                f"{label}.se2_station_probe is invalid"
            ) from exc
        if (
            not math.isfinite(yaw_deg)
            or translation_raw.shape != (3,)
            or not np.all(np.isfinite(translation_raw))
        ):
            raise FrameIdentityError(
                f"{label}.se2_station_probe contains invalid numbers"
            )
        translation = translation_raw.copy()
    ground_shift = provenance.get("ground_z_shift_m")
    if ground_shift is not None:
        if isinstance(ground_shift, bool):
            raise FrameIdentityError(f"{label}.ground_z_shift_m is invalid")
        try:
            shift = float(ground_shift)
        except (TypeError, ValueError) as exc:
            raise FrameIdentityError(
                f"{label}.ground_z_shift_m is invalid"
            ) from exc
        if not math.isfinite(shift):
            raise FrameIdentityError(f"{label}.ground_z_shift_m is invalid")
        translation[2] += shift
    return yaw_deg, translation


def _rotation_z(yaw_deg: float) -> np.ndarray:
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_w = left[..., 0]
    left_v = left[..., 1:]
    right_w = right[..., 0]
    right_v = right[..., 1:]
    scalar = left_w * right_w - np.sum(left_v * right_v, axis=-1)
    vector = (
        left_w[..., None] * right_v
        + right_w[..., None] * left_v
        + np.cross(left_v, right_v)
    )
    return np.concatenate((scalar[..., None], vector), axis=-1)


def _update_array_digest(
    digest: Any, key: str, array: np.ndarray
) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(key.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            list(contiguous.shape), separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))


def _track_sha256(joint_pos: np.ndarray, joint_vel: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_array_digest(digest, "joint_pos", joint_pos)
    _update_array_digest(digest, "joint_vel", joint_vel)
    return digest.hexdigest()


def _frame_fingerprints(
    joint_pos: np.ndarray, joint_vel: np.ndarray
) -> tuple[str, ...]:
    fingerprints = []
    for frame in range(len(joint_pos)):
        digest = hashlib.sha256()
        _update_array_digest(digest, "joint_pos", joint_pos[frame])
        _update_array_digest(digest, "joint_vel", joint_vel[frame])
        fingerprints.append(digest.hexdigest())
    return tuple(fingerprints)


def _ordered_fingerprint_sha256(fingerprints: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for fingerprint in fingerprints:
        digest.update(bytes.fromhex(_sha256(fingerprint, "frame fingerprint")))
    return digest.hexdigest()


def _parse_invariant_joints(
    value: Any, joint_names: Sequence[str]
) -> tuple[tuple[int, str], ...]:
    if not isinstance(value, list) or not value:
        raise FrameIdentityError("invariant_joints must be a non-empty JSON array")
    rows: list[tuple[int, str]] = []
    for row_index, raw in enumerate(value):
        row = _exact_keys(
            raw, _INVARIANT_JOINT_KEYS, f"invariant_joints[{row_index}]"
        )
        index = _integer(
            row["index"], f"invariant_joints[{row_index}].index"
        )
        name = _string(row["name"], f"invariant_joints[{row_index}].name")
        if index >= len(joint_names):
            raise FrameIdentityError(
                f"invariant joint index {index} exceeds joint order"
            )
        if joint_names[index] != name:
            raise FrameIdentityError(
                f"invariant joint ({index},{name!r}) disagrees with joint order "
                f"name {joint_names[index]!r}"
            )
        rows.append((index, name))
    if rows != sorted(rows) or len({index for index, _ in rows}) != len(rows):
        raise FrameIdentityError(
            "invariant_joints must be unique and strictly index-sorted"
        )
    return tuple(rows)


def _parse_tolerances(value: Any) -> Mapping[str, float]:
    raw = _exact_keys(value, _TOLERANCE_KEYS, "world_residual_tolerances")
    parsed = {
        key: _positive_finite(
            raw[key], f"world_residual_tolerances.{key}"
        )
        for key in sorted(_TOLERANCE_KEYS)
    }
    excessive = {
        key: (parsed[key], _WORLD_RESIDUAL_CEILINGS[key])
        for key in parsed
        if parsed[key] > _WORLD_RESIDUAL_CEILINGS[key]
    }
    if excessive:
        raise FrameIdentityError(
            "world residual tolerances exceed fixed v1 ceilings: "
            f"{excessive}"
        )
    return MappingProxyType(parsed)


def _validate_semantic_contract(
    value: Any, mapping_role: str
) -> Mapping[str, bool]:
    raw = _exact_keys(value, _SEMANTIC_KEYS, "semantic_contract")
    expected = {
        "frame_identity_only": True,
        "observed_ball_contact": False,
        "behavior_authorized": False,
        "synthetic_behavior_nominal_claim": False,
    }
    for key, expected_value in expected.items():
        if not isinstance(raw[key], bool) or raw[key] is not expected_value:
            raise FrameIdentityError(
                f"semantic_contract.{key} must be exactly {expected_value}"
            )
    if mapping_role == "synthetic_face_construction_annotation_to_bound_recipe_source":
        # The same false value is deliberately checked above: construction
        # alignment may never become synthetic behavior truth by implication.
        pass
    return MappingProxyType(expected)


def _input_binding(raw: Any, label: str) -> tuple[str, str]:
    binding = _exact_keys(raw, _FILE_BINDING_KEYS, label)
    return (
        _string(binding["path"], f"{label}.path"),
        _sha256(binding["sha256"], f"{label}.sha256"),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_frame_identity_receipt(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build one immutable evidence-only receipt from an exact input spec."""

    raw = _exact_keys(spec, _SPEC_KEYS, "frame identity spec")
    if _integer(raw["schema_version"], "spec.schema_version") != (
        FRAME_IDENTITY_SPEC_SCHEMA_VERSION
    ):
        raise FrameIdentityError(
            f"spec.schema_version must equal {FRAME_IDENTITY_SPEC_SCHEMA_VERSION}"
        )
    motion_id = _string(raw["motion_id"], "spec.motion_id")
    mapping_role = _string(raw["mapping_role"], "spec.mapping_role")
    if mapping_role not in MAPPING_ROLES:
        raise FrameIdentityError(f"unsupported mapping_role {mapping_role!r}")
    semantic_contract = _validate_semantic_contract(
        raw["semantic_contract"], mapping_role
    )
    inputs = _exact_keys(raw["inputs"], _INPUT_KEYS, "spec.inputs")

    bindings: dict[str, _BoundBytes] = {}
    for name in sorted(_INPUT_KEYS):
        path_text, expected_sha = _input_binding(
            inputs[name], f"spec.inputs.{name}"
        )
        bindings[name] = _read_bound_bytes(path_text, expected_sha, name)

    event_authority = _strict_json_bytes(
        bindings["event_authority"].payload, "event_authority"
    )
    authority_contract = _extract_event_authority(
        event_authority,
        motion_id=motion_id,
        mapping_role=mapping_role,
        event_npz_sha256=bindings["event_npz"].sha256,
        bound_npz_sha256=bindings["bound_npz"].sha256,
    )
    event_frame = int(authority_contract["event_source_frame"])

    joint_names = _load_joint_order(bindings["joint_order"])
    _validate_joint_bijection(
        bindings["joint_bijection"],
        joint_order_sha256=bindings["joint_order"].sha256,
        joint_names=joint_names,
    )
    invariant_rows = _parse_invariant_joints(
        raw["invariant_joints"], joint_names
    )
    invariant_indices = np.asarray(
        [index for index, _ in invariant_rows], dtype=np.int64
    )
    excluded_indices = tuple(
        index for index in range(len(joint_names)) if index not in invariant_indices
    )
    excluded_index_set = frozenset(excluded_indices)
    if excluded_index_set not in {
        frozenset(),
        _ALLOWED_EXCLUDED_JOINT_INDICES,
    }:
        raise FrameIdentityError(
            "v1 invariant joints must cover all 31 runtime joints or all "
            "except right-wrist indices 26, 28, and 30"
        )
    tolerances = _parse_tolerances(raw["world_residual_tolerances"])
    excluded_world_raw = raw["excluded_world_bodies"]
    if (
        not isinstance(excluded_world_raw, list)
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in excluded_world_raw
        )
        or len(set(excluded_world_raw)) != len(excluded_world_raw)
    ):
        raise FrameIdentityError(
            "excluded_world_bodies must be a unique JSON string array"
        )
    excluded_world_bodies = tuple(excluded_world_raw)

    event_motion = _load_motion_from_same_bytes(
        bindings["event_npz"], "event_npz"
    )
    bound_motion = _load_motion_from_same_bytes(
        bindings["bound_npz"], "bound_npz"
    )
    if (
        event_motion.fps != bound_motion.fps
        or event_motion.joint_pos.shape != bound_motion.joint_pos.shape
        or event_motion.body_pos_w.shape != bound_motion.body_pos_w.shape
        or event_motion.body_names != bound_motion.body_names
        or event_motion.array_contract != bound_motion.array_contract
    ):
        raise FrameIdentityError(
            "event and bound NPZ schema, fps, shape, or body order differ"
        )
    frame_count, joint_count = event_motion.joint_pos.shape
    if joint_count != len(joint_names):
        raise FrameIdentityError(
            "joint order length does not match NPZ joint columns"
        )
    if event_frame >= frame_count:
        raise FrameIdentityError("event_source_frame exceeds NPZ frame count")

    event_provenance = _strict_json_bytes(
        bindings["event_provenance"].payload, "event_provenance"
    )
    bound_provenance = _strict_json_bytes(
        bindings["bound_provenance"].payload, "bound_provenance"
    )
    event_lineage = _extract_lineage(event_provenance, "event")
    bound_lineage = _extract_lineage(bound_provenance, "bound")
    if event_lineage != bound_lineage:
        raise FrameIdentityError(
            "event and bound provenance do not bind one exact source lineage"
        )
    if (
        int(event_lineage["output_frames"]) != frame_count
        or int(event_lineage["output_fps"]) != event_motion.fps
    ):
        raise FrameIdentityError(
            "provenance output frames/fps disagree with NPZ bytes"
        )

    event_invariant_pos = np.ascontiguousarray(
        event_motion.joint_pos[:, invariant_indices]
    )
    bound_invariant_pos = np.ascontiguousarray(
        bound_motion.joint_pos[:, invariant_indices]
    )
    event_invariant_vel = np.ascontiguousarray(
        event_motion.joint_vel[:, invariant_indices]
    )
    bound_invariant_vel = np.ascontiguousarray(
        bound_motion.joint_vel[:, invariant_indices]
    )
    if not np.array_equal(event_invariant_pos, bound_invariant_pos):
        raise FrameIdentityError(
            "invariant joint_pos is not bitwise equal over the full track"
        )
    if not np.array_equal(event_invariant_vel, bound_invariant_vel):
        raise FrameIdentityError(
            "invariant joint_vel is not bitwise equal over the full track"
        )
    event_track_sha = _track_sha256(
        event_invariant_pos, event_invariant_vel
    )
    bound_track_sha = _track_sha256(
        bound_invariant_pos, bound_invariant_vel
    )
    if event_track_sha != bound_track_sha:
        raise FrameIdentityError("equal invariant tracks produced unequal digests")

    event_fingerprints = _frame_fingerprints(
        event_invariant_pos, event_invariant_vel
    )
    bound_fingerprints = _frame_fingerprints(
        bound_invariant_pos, bound_invariant_vel
    )
    if len(set(event_fingerprints)) != frame_count:
        raise FrameIdentityError(
            "event invariant frame fingerprints are not all unique"
        )
    if len(set(bound_fingerprints)) != frame_count:
        raise FrameIdentityError(
            "bound invariant frame fingerprints are not all unique"
        )
    if event_fingerprints != bound_fingerprints:
        raise FrameIdentityError(
            "unique invariant frame map is not exact same-index identity"
        )

    excluded_differences = []
    for index in excluded_indices:
        pos_delta = (
            event_motion.joint_pos[:, index].astype(np.float64)
            - bound_motion.joint_pos[:, index].astype(np.float64)
        )
        vel_delta = (
            event_motion.joint_vel[:, index].astype(np.float64)
            - bound_motion.joint_vel[:, index].astype(np.float64)
        )
        excluded_differences.append(
            {
                "index": index,
                "name": joint_names[index],
                "joint_pos_max_abs_rad": float(np.max(np.abs(pos_delta))),
                "joint_vel_max_abs_rad_s": float(np.max(np.abs(vel_delta))),
                "event_joint_pos_delta_rad": float(pos_delta[event_frame]),
                "event_joint_vel_delta_rad_s": float(vel_delta[event_frame]),
            }
        )

    event_yaw, event_translation = _se2_transform_from_provenance(
        event_provenance, "event"
    )
    bound_yaw, bound_translation = _se2_transform_from_provenance(
        bound_provenance, "bound"
    )
    relative_yaw = bound_yaw - event_yaw
    relative_rotation = _rotation_z(relative_yaw)
    relative_translation = (
        bound_translation - relative_rotation @ event_translation
    )

    body_index = {
        name: index for index, name in enumerate(event_motion.body_names)
    }
    unknown_exclusions = sorted(set(excluded_world_bodies) - set(body_index))
    if unknown_exclusions:
        raise FrameIdentityError(
            f"excluded_world_bodies are absent: {unknown_exclusions}"
        )
    included_body_indices = np.asarray(
        [
            index
            for index, name in enumerate(event_motion.body_names)
            if name not in excluded_world_bodies
        ],
        dtype=np.int64,
    )
    if len(included_body_indices) == 0:
        raise FrameIdentityError("all world bodies were excluded")

    predicted_pos = (
        np.einsum(
            "ij,tbj->tbi",
            relative_rotation,
            event_motion.body_pos_w.astype(np.float64),
        )
        + relative_translation
    )
    predicted_lin = np.einsum(
        "ij,tbj->tbi",
        relative_rotation,
        event_motion.body_lin_vel_w.astype(np.float64),
    )
    predicted_ang = np.einsum(
        "ij,tbj->tbi",
        relative_rotation,
        event_motion.body_ang_vel_w.astype(np.float64),
    )
    half_angle = math.radians(relative_yaw) / 2.0
    relative_quaternion = np.asarray(
        [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
        dtype=np.float64,
    )
    event_quaternion = event_motion.body_quat_w.astype(np.float64)
    predicted_quaternion = _quaternion_multiply_wxyz(
        np.broadcast_to(relative_quaternion, event_quaternion.shape),
        event_quaternion,
    )
    bound_quaternion = bound_motion.body_quat_w.astype(np.float64)

    selected = included_body_indices
    position_residual = float(
        np.max(
            np.abs(
                predicted_pos[:, selected]
                - bound_motion.body_pos_w[:, selected].astype(np.float64)
            )
        )
    )
    linear_residual = float(
        np.max(
            np.abs(
                predicted_lin[:, selected]
                - bound_motion.body_lin_vel_w[:, selected].astype(np.float64)
            )
        )
    )
    angular_residual = float(
        np.max(
            np.abs(
                predicted_ang[:, selected]
                - bound_motion.body_ang_vel_w[:, selected].astype(np.float64)
            )
        )
    )
    quaternion_minus = np.linalg.norm(
        predicted_quaternion[:, selected] - bound_quaternion[:, selected],
        axis=-1,
    )
    quaternion_plus = np.linalg.norm(
        predicted_quaternion[:, selected] + bound_quaternion[:, selected],
        axis=-1,
    )
    quaternion_residual = float(
        np.max(np.minimum(quaternion_minus, quaternion_plus))
    )
    residuals = {
        "position_m": position_residual,
        "linear_velocity_m_s": linear_residual,
        "angular_velocity_rad_s": angular_residual,
        "quaternion_l2_sign_invariant": quaternion_residual,
    }
    failures = {
        key: (residuals[key], tolerances[key])
        for key in residuals
        if residuals[key] > tolerances[key]
    }
    if failures:
        raise FrameIdentityError(
            f"SE(2) world residuals exceed explicit tolerances: {failures}"
        )

    joint_names_sha = _canonical_names_sha256(joint_names)
    body_names_sha = _canonical_names_sha256(event_motion.body_names)
    ordered_fingerprint_sha = _ordered_fingerprint_sha256(
        event_fingerprints
    )
    input_receipts = {
        name: {
            "path": binding.path_text,
            "resolved_path": str(binding.path),
            "sha256": binding.sha256,
            "bytes_read_once_per_build_or_rebuild_pass": True,
        }
        for name, binding in sorted(bindings.items())
    }
    receipt = {
        "schema_version": FRAME_IDENTITY_SCHEMA_VERSION,
        "receipt_id": f"{motion_id}_frame_identity_v1",
        "publication_class": PUBLICATION_CLASS,
        "authorization": {
            "training_authorized": False,
            "behavior_authorized": False,
            "hardware_authorized": False,
            "artifact_promotion_authorized": False,
        },
        "motion_id": motion_id,
        "mapping_role": mapping_role,
        "semantic_contract": dict(semantic_contract),
        "integrity_contract": {
            "input_read": (
                "one_exact_byte_buffer_per_file_per_build_or_rebuild_pass_"
                "for_hash_and_parse"
            ),
            "npz_allow_pickle": False,
            "frame_map": "unique_invariant_fingerprint_identity_i_to_i",
            "world_residual_policy": "caller_may_tighten_but_not_exceed_v1_ceilings",
            "full_pose_identity_claimed": (
                len(excluded_indices) == 0
                and len(excluded_world_bodies) == 0
            ),
            "excluded_channels_are_reported": True,
        },
        "input_bindings": input_receipts,
        "event_authority_contract": {
            **dict(authority_contract),
            "input_sha256": bindings["event_authority"].sha256,
            "event_frame_derived_from_authority_not_spec": True,
        },
        "common_lineage": dict(event_lineage),
        "npz_contract": {
            "event_and_bound_contract_equal": True,
            "fps": event_motion.fps,
            "frame_count": frame_count,
            "joint_count": joint_count,
            "body_count": len(event_motion.body_names),
            "joint_order_names_sha256": joint_names_sha,
            "body_order_names_sha256": body_names_sha,
            "array_contract": dict(event_motion.array_contract),
        },
        "invariant_joint_contract": {
            "keys": ["joint_pos", "joint_vel"],
            "coverage_policy": (
                "all_31_runtime_joints_or_all_except_indices_26_28_30"
            ),
            "byte_order_and_dtype_preserved": True,
            "invariant_joints": [
                {"index": index, "name": name}
                for index, name in invariant_rows
            ],
            "excluded_joints": excluded_differences,
            "full_track_bitwise_equal": True,
            "invariant_track_payload_sha256": event_track_sha,
            "frame_fingerprint_contract": (
                "sha256(key_nul_dtype_nul_shape_json_nul_c_order_bytes)_"
                "for_joint_pos_then_joint_vel"
            ),
            "ordered_frame_fingerprints_contract": (
                "sha256(concatenated_32_byte_frame_fingerprint_digests)"
            ),
            "ordered_frame_fingerprints_sha256": ordered_fingerprint_sha,
            "event_unique_frame_count": frame_count,
            "bound_unique_frame_count": frame_count,
            "same_index_match_count": frame_count,
            "nonidentity_match_count": 0,
        },
        "event_mapping": {
            "event_source_frame": event_frame,
            "bound_source_frame": event_frame,
            "authority_frame_json_pointer": authority_contract[
                "event_frame_json_pointer"
            ],
            "frame_fingerprint_sha256": event_fingerprints[event_frame],
            "unique_in_event": True,
            "unique_in_bound": True,
            "identity_map": True,
        },
        "se2_world_contract": {
            "event_transform": {
                "yaw_deg": event_yaw,
                "translation_w_m": event_translation.tolist(),
            },
            "bound_transform": {
                "yaw_deg": bound_yaw,
                "translation_w_m": bound_translation.tolist(),
            },
            "event_to_bound": {
                "yaw_deg": relative_yaw,
                "translation_w_m": relative_translation.tolist(),
            },
            "excluded_world_bodies": list(excluded_world_bodies),
            "included_world_body_count": int(len(included_body_indices)),
            "residual_max": residuals,
            "tolerances": dict(tolerances),
            "fixed_v1_tolerance_ceilings": dict(
                _WORLD_RESIDUAL_CEILINGS
            ),
            "pass": True,
        },
        "non_claims": [
            "observed_ball_contact",
            "behavior_optimal_frame",
            "post_retime_returnability_window",
            "complete_pose_identity_when_any_joint_or_world_body_is_excluded",
            "grip_or_racket_kinematic_equivalence",
            "training_authorization",
            "artifact_promotion",
            "hardware_authorization",
        ],
    }
    return MappingProxyType(_json_safe(receipt))


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the sole on-disk JSON encoding used by this receipt contract."""

    return (
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_frame_identity_spec(
    spec_path: str | Path, *, expected_spec_sha256: str
) -> Mapping[str, Any]:
    """Read/hash/parse one spec from the same exact byte buffer."""

    bound = _read_bound_bytes(spec_path, expected_spec_sha256, "spec")
    return MappingProxyType(
        dict(_exact_keys(_strict_json_bytes(bound.payload, "spec"), _SPEC_KEYS, "spec"))
    )


def _rebuild_from_loaded_receipt(
    loaded: Mapping[str, Any],
) -> Mapping[str, Any]:
    if _integer(loaded["schema_version"], "receipt.schema_version") != (
        FRAME_IDENTITY_SCHEMA_VERSION
    ):
        raise FrameIdentityError(
            f"receipt.schema_version must equal {FRAME_IDENTITY_SCHEMA_VERSION}"
        )
    if loaded["publication_class"] != PUBLICATION_CLASS:
        raise FrameIdentityError("receipt publication_class changed")
    authorization = _exact_keys(
        loaded["authorization"],
        _AUTHORIZATION_KEYS,
        "receipt.authorization",
    )
    if any(value is not False for value in authorization.values()):
        raise FrameIdentityError("receipt contains a non-false authorization")

    input_bindings = loaded["input_bindings"]
    if not isinstance(input_bindings, Mapping):
        raise FrameIdentityError("receipt.input_bindings must be an object")
    inputs = {}
    for name in sorted(_INPUT_KEYS):
        try:
            row = input_bindings[name]
        except KeyError as exc:
            raise FrameIdentityError(
                f"receipt.input_bindings omitted {name!r}"
            ) from exc
        if not isinstance(row, Mapping):
            raise FrameIdentityError(
                f"receipt.input_bindings.{name} must be an object"
            )
        inputs[name] = {
            "path": row.get("path"),
            "sha256": row.get("sha256"),
        }
    invariant = loaded["invariant_joint_contract"]
    world = loaded["se2_world_contract"]
    if not isinstance(invariant, Mapping) or not isinstance(world, Mapping):
        raise FrameIdentityError("receipt contracts must be objects")
    try:
        rebuilt_spec = {
            "schema_version": FRAME_IDENTITY_SPEC_SCHEMA_VERSION,
            "motion_id": loaded["motion_id"],
            "mapping_role": loaded["mapping_role"],
            "inputs": inputs,
            "invariant_joints": invariant["invariant_joints"],
            "excluded_world_bodies": world["excluded_world_bodies"],
            "world_residual_tolerances": world["tolerances"],
            "semantic_contract": loaded["semantic_contract"],
        }
    except KeyError as exc:
        raise FrameIdentityError(
            f"receipt omitted rebuild field {exc.args[0]!r}"
        ) from exc
    return build_frame_identity_receipt(rebuilt_spec)


def write_frame_identity_receipt(
    receipt: Mapping[str, Any], output_path: str | Path
) -> str:
    """Atomically publish a receipt without replacing any existing path."""

    loaded = _exact_keys(receipt, _RECEIPT_KEYS, "receipt")
    payload = canonical_json_bytes(receipt)
    rebuilt = _rebuild_from_loaded_receipt(loaded)
    if canonical_json_bytes(rebuilt) != payload:
        raise FrameIdentityError(
            "receipt does not exactly equal a rebuild from its pinned inputs"
        )
    requested = Path(output_path).expanduser()
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    leaf = requested.name
    if not leaf or leaf in {".", ".."}:
        raise FrameIdentityError("receipt destination must name one file")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, directory_flags)
    temporary_name: str | None = None
    try:
        try:
            os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FrameIdentityError(
                f"receipt destination already exists: {parent / leaf}"
            )

        temporary_name = f".{leaf}.{secrets.token_hex(16)}.tmp"
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                leaf,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FrameIdentityError(
                f"receipt destination already exists: {parent / leaf}"
            ) from exc
        os.fsync(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    return sha256_bytes(payload)


def verify_frame_identity_receipt(
    receipt_path: str | Path, *, expected_receipt_sha256: str
) -> Mapping[str, Any]:
    """Rebuild a receipt from its pinned inputs and require exact equality."""

    bound = _read_bound_bytes(
        receipt_path, expected_receipt_sha256, "receipt"
    )
    loaded = _exact_keys(
        _strict_json_bytes(bound.payload, "receipt"),
        _RECEIPT_KEYS,
        "receipt",
    )
    rebuilt = _rebuild_from_loaded_receipt(loaded)
    if canonical_json_bytes(rebuilt) != bound.payload:
        raise FrameIdentityError(
            "receipt does not exactly equal a rebuild from its pinned inputs"
        )
    return MappingProxyType(dict(loaded))


def _load_spec_for_cli(path: str, expected_sha256: str) -> Mapping[str, Any]:
    if path == "-":
        payload = sys.stdin.buffer.read()
        expected = _sha256(expected_sha256, "spec.sha256")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise FrameIdentityError(
                f"spec SHA-256 mismatch: expected {expected}, got {actual}"
            )
        return MappingProxyType(
            dict(
                _exact_keys(
                    _strict_json_bytes(payload, "spec"), _SPEC_KEYS, "spec"
                )
            )
        )
    return load_frame_identity_spec(path, expected_spec_sha256=expected_sha256)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one no-clobber receipt")
    build.add_argument("--spec", required=True)
    build.add_argument("--spec-sha256", required=True)
    build.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify", help="rebuild and verify one receipt")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        spec = _load_spec_for_cli(args.spec, args.spec_sha256)
        receipt = build_frame_identity_receipt(spec)
        output_sha = write_frame_identity_receipt(receipt, args.output)
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).resolve()),
                    "sha256": output_sha,
                    "verdict": "PASS_FRAME_IDENTITY_EVIDENCE_ONLY",
                },
                sort_keys=True,
            )
        )
        return 0
    verified = verify_frame_identity_receipt(
        args.receipt, expected_receipt_sha256=args.receipt_sha256
    )
    print(
        json.dumps(
            {
                "receipt_id": verified["receipt_id"],
                "sha256": args.receipt_sha256,
                "verdict": "PASS_REBUILT_EXACTLY",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FRAME_IDENTITY_SCHEMA_VERSION",
    "FRAME_IDENTITY_SPEC_SCHEMA_VERSION",
    "FrameIdentityError",
    "MAPPING_ROLES",
    "PUBLICATION_CLASS",
    "build_frame_identity_receipt",
    "canonical_json_bytes",
    "load_frame_identity_spec",
    "main",
    "sha256_bytes",
    "verify_frame_identity_receipt",
    "write_frame_identity_receipt",
]
