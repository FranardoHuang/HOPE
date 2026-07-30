#!/usr/bin/env python3
"""Produce independent exact-MuJoCo evidence for a grounded neutral ready.

This tool is deliberately separate from ``canonical_ready_sidecar_mint.py``.
The minter only changes the container of an already-audited grounded state; it
does not prove which right-arm challenger was overlaid or measure racket-face
neutrality.  This producer closes that gap, fail-closed, by requiring and
content-binding all of the following:

* the minted strict nine-key ready NPZ;
* the exact AgiBot A3 grounded candidate NPZ and its sealed receipt
  (``candidate_id=G1`` is a candidate label, not a robot model);
* the existing 16-row lineage-pose receipt; and
* the unpublished right-arm challenger receipt.

Before MuJoCo is loaded, the ready must equal the grounded candidate, and the
grounded candidate's root and every non-leg joint (especially all seven
right-arm joints) must exactly equal the challenger selected by the supplied
challenger receipt.  A receipt assertion is not enough: the numeric arrays are
compared.

When identity closes, the tool reconstructs upper/full x four phases x BH/FH
from the content-pinned source recipe, reruns exact vendor-MuJoCo FK for every
target and for the ready, and measures signed spherical distance.  It publishes
the exact ``canonical-ready-face-neutrality-v1`` report consumed by
``canonical_motion_recipe.py`` only when every BH/FH pair differs by at most
five degrees.

Publication is no-clobber.  ``FACE_TARGET_SET.json`` is written first and
``FACE_NEUTRALITY_REPORT.json`` is written last.  Both artifacts deny training,
deployment, and hardware authorization.  Failure removes only files in the
newly-created output directory; pre-existing paths are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


TARGET_SET_FILENAME = "FACE_TARGET_SET.json"
REPORT_FILENAME = "FACE_NEUTRALITY_REPORT.json"
REPORT_TYPE = "canonical-ready-face-neutrality-v1"
ARTIFACT_CLASS = "independent_exact_fk_face_neutrality_evidence"
BACKEND_NAME = "exact_vendor_mujoco_fk"
FACE_NORMAL_CONVENTION = (
    "right_racket_site_local_plus_y_world_signed_face_normal_v1"
)
RACKET_SITE = "right_racket"
SCOPES = ("upper", "full")
PHASES = (
    "opportunity_start",
    "construction_donor_preferred",
    "nominal_event",
    "opportunity_end",
)
FACES = ("bh", "fh")
MAXIMUM_PAIR_ASYMMETRY_RAD = math.radians(5.0)
RIGHT_ARM_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEG_NAMES = (
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
READY_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "source_segment",
        "source_npz",
        "source_frame",
        "striking_joint_ids",
        "note",
    }
)
CANDIDATE_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "candidate_id",
        "receipt_sha256",
        "training_authorized",
        "hardware_authorized",
    }
)
FALSE_AUTHORIZATION = {
    "training_authorized": False,
    "deployment_authorized": False,
    "hardware_authorized": False,
}


class FaceNeutralityError(RuntimeError):
    """An input identity, exact-FK replay, or publication contract failed."""


@dataclass(frozen=True)
class FileSnapshot:
    """One immutable in-memory view of a repository file."""

    path: Path
    repo_path: str
    sha256: str
    payload: bytes

    def binding(self) -> Mapping[str, Any]:
        return {
            "path": self.repo_path,
            "bytes": len(self.payload),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ReadyState:
    """The exact state whose face neutrality is being certified."""

    joint_pos: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    striking_joint_ids: np.ndarray


@dataclass(frozen=True)
class ValidatedInputs:
    """Content snapshots and state retained after identity validation."""

    repo_root: Path
    ready: FileSnapshot
    candidate: FileSnapshot
    ground_receipt: FileSnapshot
    lineage_receipt: FileSnapshot
    challenger_receipt: FileSnapshot
    recipe: FileSnapshot
    phase_authority: FileSnapshot
    state: ReadyState
    ground: Mapping[str, Any]
    lineage: Mapping[str, Any]
    challenger: Mapping[str, Any]


@dataclass(frozen=True)
class TargetRow:
    """One freshly reconstructed exact-FK target."""

    scope: str
    phase: str
    face: str
    normal_w: np.ndarray
    target_sha256: str
    source_frame_index: int
    pose_content_sha256: str


@dataclass(frozen=True)
class ExactEvaluation:
    """Fresh exact-model FK output, independent of old numeric measurements."""

    mjcf_sha256: str
    compiled_model_sha256: str
    ready_normal_w: np.ndarray
    rows: Tuple[TargetRow, ...]


@dataclass(frozen=True)
class PublishedFaceEvidence:
    """Paths and identities of one no-clobber evidence publication."""

    directory: Path
    target_set: Path
    report: Path
    target_set_sha256: str
    report_sha256: str
    report_payload_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FaceNeutralityError(
            "evidence JSON is not finite/canonicalizable: {}".format(exc)
        ) from exc


def _sealed_json_bytes(
    value: Mapping[str, Any], payload_field: str
) -> Tuple[bytes, str]:
    detached = dict(value)
    detached.pop(payload_field, None)
    payload_sha = _sha256_bytes(_canonical_json_bytes(detached))
    sealed = dict(detached)
    sealed[payload_field] = payload_sha
    return _canonical_json_bytes(sealed), payload_sha


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise FaceNeutralityError(
                    "{} has duplicate JSON key {!r}".format(label, key)
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FaceNeutralityError(
                    "{} contains non-finite JSON number {!r}".format(label, token)
                )
            ),
        )
    except FaceNeutralityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FaceNeutralityError(
            "cannot parse {}: {}".format(label, exc)
        ) from exc
    if not isinstance(value, Mapping):
        raise FaceNeutralityError("{} must contain one JSON object".format(label))
    _canonical_json_bytes(value)
    return value


def _snapshot(repo_root: Path, path: Path, label: str) -> FileSnapshot:
    lexical = path if path.is_absolute() else repo_root / path
    try:
        if stat.S_ISLNK(lexical.lstat().st_mode):
            raise FaceNeutralityError(
                "{} may not be a symlink: {}".format(label, lexical)
            )
        resolved = lexical.resolve(strict=True)
        relative = resolved.relative_to(repo_root)
    except FaceNeutralityError:
        raise
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise FaceNeutralityError(
            "{} must be a regular file inside repo root: {}".format(label, lexical)
        ) from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(resolved), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FaceNeutralityError("{} is not a regular file".format(label))
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return FileSnapshot(
        path=resolved,
        repo_path=relative.as_posix(),
        sha256=_sha256_bytes(payload),
        payload=payload,
    )


def _finite_float64_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype != np.dtype("float64"):
        raise FaceNeutralityError(
            "{} must have dtype float64 and shape ({},), got {} {}".format(
                label, length, array.dtype, array.shape
            )
        )
    if not np.isfinite(array).all():
        raise FaceNeutralityError("{} contains NaN or infinity".format(label))
    return np.ascontiguousarray(array, dtype=np.float64)


def _json_vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FaceNeutralityError("{} is not numeric".format(label)) from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise FaceNeutralityError(
            "{} must be one finite {}-vector".format(label, length)
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _hash_array(digest: Any, label: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())


def _array_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "array", value)
    return digest.hexdigest()


def _neutral_array_sha256(value: Any) -> str:
    """Digest used by canonical_neutral_ready's contact evidence."""

    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise FaceNeutralityError("object arrays cannot enter target evidence")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().newbyteorder("<")
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _ready_state_sha256(state: ReadyState) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "joint_pos", state.joint_pos)
    _hash_array(digest, "root_pos_w", state.root_pos_w)
    _hash_array(digest, "root_quat_wxyz", state.root_quat_wxyz)
    return digest.hexdigest()


def _false_scalar(value: Any, label: str) -> None:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "b" or bool(array.item()):
        raise FaceNeutralityError("{} must be scalar false".format(label))


def _text_scalar(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise FaceNeutralityError("{} must be one text scalar".format(label))
    result = str(array.item())
    if not result:
        raise FaceNeutralityError("{} must not be empty".format(label))
    return result


def _authorization_false(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise FaceNeutralityError("{} must be an object".format(label))
    aliases = {
        "training_authorized": value.get("training_authorized"),
        "deployment_authorized": value.get(
            "deployment_authorized", value.get("deploy_authorized")
        ),
        "hardware_authorized": value.get("hardware_authorized"),
    }
    if aliases != FALSE_AUTHORIZATION:
        raise FaceNeutralityError(
            "{} must deny training, deployment, and hardware authorization".format(
                label
            )
        )


def _validate_ground_seals(receipt: Mapping[str, Any]) -> None:
    receipt_sha = receipt.get("receipt_payload_sha256")
    publication_sha = receipt.get("publication_payload_sha256")
    if not isinstance(receipt_sha, str) or not isinstance(publication_sha, str):
        raise FaceNeutralityError("ground receipt lacks both payload seals")
    unsigned_publication = dict(receipt)
    unsigned_publication.pop("publication_payload_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(unsigned_publication)) != publication_sha:
        raise FaceNeutralityError("ground receipt publication seal does not close")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("publication_payload_sha256", None)
    unsigned_receipt.pop("publication", None)
    unsigned_receipt.pop("receipt_payload_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(unsigned_receipt)) != receipt_sha:
        raise FaceNeutralityError("ground receipt payload seal does not close")


def _load_ready(snapshot: FileSnapshot) -> ReadyState:
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as payload:
            if frozenset(payload.files) != READY_KEYS:
                raise FaceNeutralityError(
                    "minted ready must have the strict nine-key schema"
                )
            joint_pos = _finite_float64_vector(
                payload["joint_pos"], 31, "ready joint_pos"
            )
            joint_vel = _finite_float64_vector(
                payload["joint_vel"], 31, "ready joint_vel"
            )
            root_pos = _finite_float64_vector(
                payload["root_pos_w"], 3, "ready root_pos_w"
            )
            root_quat = _finite_float64_vector(
                payload["root_quat_w"], 4, "ready root_quat_w"
            )
            source_segment = _text_scalar(
                payload["source_segment"], "ready source_segment"
            )
            _text_scalar(payload["source_npz"], "ready source_npz")
            _text_scalar(payload["note"], "ready note")
            source_frame = np.asarray(payload["source_frame"])
            striking_ids = np.asarray(payload["striking_joint_ids"])
    except FaceNeutralityError:
        raise
    except (OSError, ValueError) as exc:
        raise FaceNeutralityError(
            "cannot load minted ready: {}".format(exc)
        ) from exc
    if source_segment != "grounded_ready_v2_g1_neutral_arm":
        raise FaceNeutralityError("ready source_segment is not the grounded v2 identity")
    if source_frame.shape != () or int(source_frame.item()) != 0:
        raise FaceNeutralityError("ready source_frame must be integer zero")
    if (
        striking_ids.shape != (7,)
        or not np.issubdtype(striking_ids.dtype, np.integer)
        or len(set(int(value) for value in striking_ids)) != 7
    ):
        raise FaceNeutralityError("ready striking_joint_ids contract changed")
    striking_ids = np.ascontiguousarray(striking_ids, dtype=np.int64)
    if not np.array_equal(joint_vel, np.zeros(31, np.float64)):
        raise FaceNeutralityError("ready joint velocity must be exact zero")
    if abs(float(np.linalg.norm(root_quat)) - 1.0) > 1.0e-6:
        raise FaceNeutralityError("ready root quaternion is not unit length")
    return ReadyState(joint_pos, root_pos, root_quat, striking_ids)


def _load_candidate(
    snapshot: FileSnapshot, ground: Mapping[str, Any]
) -> ReadyState:
    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as payload:
            if frozenset(payload.files) != CANDIDATE_KEYS:
                raise FaceNeutralityError(
                    "grounded candidate key-set differs from exact A3 "
                    "candidate_id=G1 publication"
                )
            joint_pos = _finite_float64_vector(
                payload["joint_pos"], 31, "candidate joint_pos"
            )
            joint_vel = _finite_float64_vector(
                payload["joint_vel"], 31, "candidate joint_vel"
            )
            root_pos = _finite_float64_vector(
                payload["root_pos_w"], 3, "candidate root_pos_w"
            )
            root_quat = _finite_float64_vector(
                payload["root_quat_w"], 4, "candidate root_quat_w"
            )
            root_lin = _finite_float64_vector(
                payload["root_lin_vel_w"], 3, "candidate root_lin_vel_w"
            )
            root_ang = _finite_float64_vector(
                payload["root_ang_vel_w"], 3, "candidate root_ang_vel_w"
            )
            candidate_id = _text_scalar(payload["candidate_id"], "candidate_id")
            embedded_receipt_sha = _text_scalar(
                payload["receipt_sha256"], "candidate receipt_sha256"
            )
            _false_scalar(
                payload["training_authorized"], "candidate training_authorized"
            )
            _false_scalar(
                payload["hardware_authorized"], "candidate hardware_authorized"
            )
    except FaceNeutralityError:
        raise
    except (OSError, ValueError) as exc:
        raise FaceNeutralityError(
            "cannot load grounded candidate: {}".format(exc)
        ) from exc
    if candidate_id != "G1" or ground.get("candidate_id") != "G1":
        raise FaceNeutralityError(
            "grounded candidate must be exact AgiBot A3 candidate_id=G1"
        )
    if embedded_receipt_sha != ground.get("receipt_payload_sha256"):
        raise FaceNeutralityError(
            "candidate embedded receipt identity does not close"
        )
    if (
        not np.array_equal(joint_vel, np.zeros(31, np.float64))
        or not np.array_equal(root_lin, np.zeros(3, np.float64))
        or not np.array_equal(root_ang, np.zeros(3, np.float64))
    ):
        raise FaceNeutralityError("grounded candidate is not exact zero velocity")
    if abs(float(np.linalg.norm(root_quat)) - 1.0) > 1.0e-8:
        raise FaceNeutralityError("candidate root quaternion is not unit length")
    overlay = ground.get("source", {}).get("upper_overlay", {})
    ids = np.asarray(overlay.get("joint_indices", []), dtype=np.int64)
    return ReadyState(joint_pos, root_pos, root_quat, ids)


def _critical_ground_contract(
    ground: Mapping[str, Any],
    candidate: FileSnapshot,
    state: ReadyState,
) -> None:
    _validate_ground_seals(ground)
    if (
        ground.get("schema_version") != 1
        or ground.get("artifact_class")
        != "diagnostic_stationary_grounded_ready_candidate"
        or ground.get("verdict") != "PASS_STATIC_GROUNDED_READY_CANDIDATE"
    ):
        raise FaceNeutralityError(
            "ground receipt is not the exact A3 candidate_id=G1 PASS artifact"
        )
    _authorization_false(ground.get("authorization"), "ground authorization")
    gates = ground.get("gates")
    if (
        not isinstance(gates, Mapping)
        or len(gates) != 9
        or any(value != "PASS" for value in gates.values())
    ):
        raise FaceNeutralityError("all nine grounded static gates must be PASS")
    publication = ground.get("publication")
    if (
        not isinstance(publication, Mapping)
        or publication.get("candidate_filename") != candidate.path.name
        or publication.get("candidate_npz_sha256") != candidate.sha256
        or publication.get("completion_semantics")
        != "exclusive_directory_and_receipt_written_last"
    ):
        raise FaceNeutralityError("ground receipt does not bind candidate bytes")
    recorded = ground.get("candidate")
    if not isinstance(recorded, Mapping):
        raise FaceNeutralityError("ground receipt lacks candidate state")
    if (
        not np.array_equal(
            _json_vector(recorded.get("joint_pos"), 31, "ground joint_pos"),
            state.joint_pos,
        )
        or not np.array_equal(
            _json_vector(recorded.get("root_pos_w"), 3, "ground root_pos_w"),
            state.root_pos_w,
        )
        or not np.array_equal(
            _json_vector(
                recorded.get("root_quat_wxyz"), 4, "ground root_quat_wxyz"
            ),
            state.root_quat_wxyz,
        )
        or recorded.get("joint_pos_sha256") != _array_sha256(state.joint_pos)
    ):
        raise FaceNeutralityError("ground receipt state differs from candidate NPZ")
    model = ground.get("exact_model")
    if (
        not isinstance(model, Mapping)
        or model.get("exact_mujoco_backend") is not True
        or model.get("status") != "PASS_EXACT_MUJOCO"
    ):
        raise FaceNeutralityError("ground receipt lacks exact vendor MuJoCo identity")


def _critical_challenger_contract(
    challenger: Mapping[str, Any],
    ground: Mapping[str, Any],
    candidate_state: ReadyState,
) -> None:
    if (
        challenger.get("schema_version") != 1
        or challenger.get("artifact_class")
        != "diagnostic_face_neutral_ready_candidate"
        or challenger.get("verdict") != "INCOMPLETE_FAIL_CLOSED"
    ):
        raise FaceNeutralityError("challenger receipt identity changed")
    _authorization_false(
        challenger.get("authorization"), "challenger authorization"
    )
    gates = challenger.get("gates")
    required_pass = (
        "all_global_targets_exact_ik",
        "exact_model",
        "exact_site_normal_ik",
        "finite_global_optimizer_locus",
        "fixed_joints",
        "global_angular_minimax_bound",
        "input_contact_exact_fk",
        "joint_limits",
        "neutrality_threshold",
        "paired_face_and_site_content_contract",
        "source_ready_hash",
        "upstream_source_pose_reconstruction",
    )
    if not isinstance(gates, Mapping) or any(
        gates.get(key) != "PASS" for key in required_pass
    ):
        raise FaceNeutralityError("challenger exact-FK/neutrality gates do not close")
    joint_contract = challenger.get("joint_contract")
    selected = challenger.get("candidate")
    if not isinstance(joint_contract, Mapping) or not isinstance(selected, Mapping):
        raise FaceNeutralityError("challenger lacks selected joint contract")
    indices = np.asarray(
        joint_contract.get("active_joint_indices", []), dtype=np.int64
    )
    names = tuple(joint_contract.get("active_joint_names", ()))
    if (
        indices.shape != (7,)
        or names != RIGHT_ARM_NAMES
        or tuple(joint_contract.get("all_joint_names", ()))
        != tuple(ground.get("exact_model", {}).get("joint_order", ()))
    ):
        raise FaceNeutralityError("challenger right-arm/runtime joint contract changed")
    selected_joint_pos = _json_vector(
        selected.get("joint_pos"), 31, "challenger selected joint_pos"
    )
    if selected.get("joint_pos_sha256") != _neutral_array_sha256(
        selected_joint_pos
    ):
        raise FaceNeutralityError("challenger selected joint digest does not close")

    overlay = ground.get("source", {}).get("upper_overlay", {})
    overlay_ids = np.asarray(overlay.get("joint_indices", []), dtype=np.int64)
    overlay_names = tuple(overlay.get("joint_names", ()))
    if (
        overlay.get("applied") is not True
        or not np.array_equal(overlay_ids, indices)
        or overlay_names != names
        or overlay.get("root_preserved") is not True
        or overlay.get("lower_preserved") is not True
    ):
        raise FaceNeutralityError("ground receipt overlay contract changed")
    if overlay.get("input_joint_pos_sha256") != _array_sha256(
        selected_joint_pos
    ):
        raise FaceNeutralityError(
            "grounded overlay input digest does not match supplied challenger"
        )
    if overlay.get("copied_values_sha256") != _array_sha256(
        selected_joint_pos[indices]
    ):
        raise FaceNeutralityError(
            "grounded overlay copied-values digest does not match challenger"
        )
    if not np.array_equal(candidate_state.joint_pos[indices], selected_joint_pos[indices]):
        mismatch = [
            RIGHT_ARM_NAMES[offset]
            for offset, index in enumerate(indices)
            if candidate_state.joint_pos[index] != selected_joint_pos[index]
        ]
        raise FaceNeutralityError(
            "grounded ready right-arm differs from supplied challenger: {}".format(
                ", ".join(mismatch)
            )
        )

    runtime_names = tuple(ground["exact_model"]["joint_order"])
    leg_indices = {runtime_names.index(name) for name in LEG_NAMES}
    nonleg_indices = np.asarray(
        [index for index in range(31) if index not in leg_indices], dtype=np.int64
    )
    if not np.array_equal(
        candidate_state.joint_pos[nonleg_indices],
        selected_joint_pos[nonleg_indices],
    ):
        raise FaceNeutralityError(
            "grounded ready changed a non-leg joint outside the selected challenger"
        )
    selected_root_pos = _json_vector(
        selected.get("root_pos_w"), 3, "challenger root_pos_w"
    )
    selected_root_quat = _json_vector(
        selected.get("root_quat_w"), 4, "challenger root_quat_w"
    )
    if (
        not np.array_equal(candidate_state.root_pos_w, selected_root_pos)
        or not np.array_equal(candidate_state.root_quat_wxyz, selected_root_quat)
        or ground.get("source", {}).get("root_bitwise_preserved") is not True
        or ground.get("source", {}).get("nonleg_joint_values_bitwise_preserved")
        is not True
    ):
        raise FaceNeutralityError(
            "grounded ready root/non-leg preservation does not close"
        )


def _critical_lineage_contract(
    lineage: Mapping[str, Any],
    challenger: Mapping[str, Any],
    recipe: FileSnapshot,
    phase_authority: FileSnapshot,
) -> None:
    if (
        lineage.get("schema_version") != 1
        or lineage.get("builder_contract")
        != "canonical_block_lineage_pose_reconstruction_v2"
        or lineage.get("contact_row_count") != 16
        or len(lineage.get("rows", ())) != 16
    ):
        raise FaceNeutralityError("lineage receipt is not the exact 16-row contract")
    model = lineage.get("model_binding")
    challenger_model = challenger.get("model")
    if not isinstance(model, Mapping) or not isinstance(challenger_model, Mapping):
        raise FaceNeutralityError("lineage/challenger model identity is missing")
    for key in (
        "mjcf_sha256",
        "compiled_model_sha256",
        "backend_limits_sha256",
        "backend_model_contract_sha256",
    ):
        if model.get(key) != challenger_model.get(key):
            raise FaceNeutralityError(
                "lineage/challenger model identity differs at {}".format(key)
            )
    if (
        model.get("normal_convention") != FACE_NORMAL_CONVENTION
        or model.get("racket_site_name") != RACKET_SITE
    ):
        raise FaceNeutralityError("lineage face convention changed")
    bindings = lineage.get("file_bindings")
    if not isinstance(bindings, list):
        raise FaceNeutralityError("lineage receipt lacks file bindings")
    by_role = {}
    for row in bindings:
        if not isinstance(row, Mapping) or not isinstance(row.get("role"), str):
            raise FaceNeutralityError("lineage file binding is malformed")
        if row["role"] in by_role:
            raise FaceNeutralityError("lineage file role is duplicated")
        by_role[row["role"]] = row
    if (
        by_role.get("recipe", {}).get("sha256") != recipe.sha256
        or by_role.get("phase_authority", {}).get("sha256")
        != phase_authority.sha256
    ):
        raise FaceNeutralityError(
            "recipe or phase authority bytes differ from lineage receipt"
        )
    matrix = challenger.get("contact_matrix")
    if (
        not isinstance(matrix, Mapping)
        or matrix.get("input_sha256") != lineage.get("contact_matrix_sha256")
        or matrix.get("row_count") != 16
        or len(matrix.get("rows", ())) != 16
    ):
        raise FaceNeutralityError(
            "challenger target matrix does not bind the lineage receipt"
        )


def validate_inputs(
    *,
    repo_root: Path,
    ready_path: Path,
    candidate_path: Path,
    ground_receipt_path: Path,
    lineage_receipt_path: Path,
    challenger_receipt_path: Path,
    recipe_path: Path,
    phase_authority_path: Path,
) -> ValidatedInputs:
    """Snapshot and validate the complete pre-FK identity chain."""

    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise FaceNeutralityError("repo root is not a directory")
    ready = _snapshot(root, ready_path, "minted ready")
    candidate = _snapshot(root, candidate_path, "grounded candidate")
    ground_snapshot = _snapshot(root, ground_receipt_path, "ground receipt")
    lineage_snapshot = _snapshot(root, lineage_receipt_path, "lineage receipt")
    challenger_snapshot = _snapshot(
        root, challenger_receipt_path, "challenger receipt"
    )
    recipe = _snapshot(root, recipe_path, "canonical recipe")
    phase = _snapshot(root, phase_authority_path, "phase authority")
    ground = _strict_json(ground_snapshot.payload, "ground receipt")
    lineage = _strict_json(lineage_snapshot.payload, "lineage receipt")
    challenger = _strict_json(challenger_snapshot.payload, "challenger receipt")

    candidate_state = _load_candidate(candidate, ground)
    _critical_ground_contract(ground, candidate, candidate_state)
    ready_state = _load_ready(ready)
    if (
        not np.array_equal(ready_state.joint_pos, candidate_state.joint_pos)
        or not np.array_equal(ready_state.root_pos_w, candidate_state.root_pos_w)
        or not np.array_equal(
            ready_state.root_quat_wxyz, candidate_state.root_quat_wxyz
        )
        or not np.array_equal(
            ready_state.striking_joint_ids, candidate_state.striking_joint_ids
        )
    ):
        raise FaceNeutralityError(
            "minted ready state differs from grounded candidate state"
        )
    _critical_challenger_contract(
        challenger, ground, candidate_state
    )
    _critical_lineage_contract(lineage, challenger, recipe, phase)
    return ValidatedInputs(
        repo_root=root,
        ready=ready,
        candidate=candidate,
        ground_receipt=ground_snapshot,
        lineage_receipt=lineage_snapshot,
        challenger_receipt=challenger_snapshot,
        recipe=recipe,
        phase_authority=phase,
        state=ready_state,
        ground=ground,
        lineage=lineage,
        challenger=challenger,
    )


def _unit(value: Any, label: str) -> np.ndarray:
    vector = _json_vector(value, 3, label)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise FaceNeutralityError("{} is degenerate".format(label))
    unit = np.ascontiguousarray(vector / norm, dtype=np.float64)
    if abs(float(np.linalg.norm(unit)) - 1.0) > 1.0e-12:
        raise FaceNeutralityError("{} could not be normalized".format(label))
    return unit


def recompute_exact_evidence(validated: ValidatedInputs) -> ExactEvaluation:
    """Rerun source reconstruction and vendor-MuJoCo FK for ready and targets."""

    scripts = validated.repo_root / "hope_training/whole_body_tracking/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        import canonical_face_manifold as face
        import canonical_mujoco_dynamics_gate as dynamics_gate
        import canonical_neutral_ready as neutral
        import canonical_neutral_ready_cli as adapter
    except ImportError as exc:
        raise FaceNeutralityError(
            "exact vendor MuJoCo toolchain is unavailable: {}".format(exc)
        ) from exc

    model_row = validated.lineage["model_binding"]
    recipe_sha = validated.recipe.sha256
    try:
        recipe, _ = adapter._snapshot_recipe_inputs(
            validated.recipe.path,
            repo_root=validated.repo_root,
            expected_recipe_sha256=recipe_sha,
        )
        backend = face.MujocoRightRacketBackend(
            recipe.model_paths["mjcf"],
            dynamics_gate.RUNTIME_JOINT_NAMES,
            urdf_path=recipe.model_paths["urdf"],
        )
        actual_compiled = dynamics_gate.compiled_model_signature(backend.model)
        if actual_compiled != model_row["compiled_model_sha256"]:
            raise FaceNeutralityError(
                "fresh compiled MuJoCo model differs from bound identity"
            )
        binding = neutral.ExactModelBinding(
            mjcf_path=recipe.model_paths["mjcf"],
            expected_mjcf_sha256=model_row["mjcf_sha256"],
            expected_compiled_model_sha256=model_row[
                "compiled_model_sha256"
            ],
            urdf_path=recipe.model_paths["urdf"],
            expected_urdf_sha256=model_row["urdf_sha256"],
            expected_backend_limits_sha256=model_row[
                "backend_limits_sha256"
            ],
            expected_backend_model_contract_sha256=model_row[
                "backend_model_contract_sha256"
            ],
        )
        phase = adapter.load_block_phase_map_binding(
            validated.phase_authority.path,
            validated.phase_authority.sha256,
        )
        loaded = adapter.load_real_neutral_ready_inputs(
            validated.recipe.path,
            expected_recipe_sha256=recipe_sha,
            repo_root=validated.repo_root,
            backend=backend,
            model_binding=binding,
            phase_map_binding=phase,
        )
        _, ready_rotation = backend.site_pose(
            validated.state.joint_pos,
            validated.state.root_pos_w,
            validated.state.root_quat_wxyz,
        )
    except FaceNeutralityError:
        raise
    except Exception as exc:
        raise FaceNeutralityError(
            "exact target/ready FK replay failed: {}: {}".format(
                type(exc).__name__, exc
            )
        ) from exc

    fresh_proof_rows = loaded.contact_source_proof.receipt["rows"]
    old_proof_rows = validated.lineage["rows"]
    challenger_rows = validated.challenger["contact_matrix"]["rows"]
    if len(loaded.contacts) != 16:
        raise FaceNeutralityError("fresh target reconstruction did not yield 16 rows")
    output_rows = []
    for index, contact in enumerate(loaded.contacts):
        fresh = fresh_proof_rows[index]
        old = old_proof_rows[index]
        challenger_row = challenger_rows[index]
        expected_label = "{}:{}:{}".format(
            contact.scope, contact.phase, contact.face_name
        )
        if (
            fresh.get("label") != expected_label
            or old.get("label") != expected_label
            or challenger_row.get("label") != expected_label
        ):
            raise FaceNeutralityError(
                "target order/label changed at row {}".format(index)
            )
        for key in (
            "source_frame_index",
            "pose_content_sha256",
            "pair_contract_sha256",
            "joint_pos_sha256",
            "root_pos_w_sha256",
            "root_quat_w_sha256",
            "site_pos_w_sha256",
            "site_rotation_w_sha256",
            "signed_face_normal_w_sha256",
        ):
            if fresh.get(key) != old.get(key):
                raise FaceNeutralityError(
                    "fresh target FK differs from lineage row {} at {}".format(
                        expected_label, key
                    )
                )
        normal = _json_vector(
            contact.signed_face_normal_w,
            3,
            "fresh target {}".format(expected_label),
        )
        supplied_normal = _json_vector(
            challenger_row.get("signed_face_normal_w"),
            3,
            "challenger target {}".format(expected_label),
        )
        if (
            abs(float(np.linalg.norm(normal)) - 1.0) > 1.0e-10
            or abs(float(np.linalg.norm(supplied_normal)) - 1.0) > 1.0e-10
        ):
            raise FaceNeutralityError(
                "target normal is not unit for {}".format(expected_label)
            )
        if not np.array_equal(normal, supplied_normal):
            raise FaceNeutralityError(
                "fresh target normal differs from challenger row {}".format(
                    expected_label
                )
            )
        target_sha = _neutral_array_sha256(normal)
        if target_sha != fresh["signed_face_normal_w_sha256"]:
            raise FaceNeutralityError(
                "fresh target normal digest does not close for {}".format(
                    expected_label
                )
            )
        output_rows.append(
            TargetRow(
                scope=str(contact.scope),
                phase=str(contact.phase),
                face=str(contact.face_name),
                normal_w=normal,
                target_sha256=target_sha,
                source_frame_index=int(contact.source_frame_index),
                pose_content_sha256=str(contact.pose_content_sha256),
            )
        )
    ready_normal = _unit(
        np.asarray(ready_rotation, dtype=np.float64)[:, 1],
        "fresh ready face normal",
    )
    return ExactEvaluation(
        mjcf_sha256=str(model_row["mjcf_sha256"]),
        compiled_model_sha256=str(model_row["compiled_model_sha256"]),
        ready_normal_w=ready_normal,
        rows=tuple(output_rows),
    )


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        math.acos(
            max(-1.0, min(1.0, float(np.dot(_unit(left, "left"), _unit(right, "right")))))
        )
    )


def _validate_evaluation(
    validated: ValidatedInputs, evaluation: ExactEvaluation
) -> Tuple[Sequence[Mapping[str, Any]], float]:
    model = validated.ground["exact_model"]
    if (
        evaluation.mjcf_sha256 != model.get("mjcf_sha256")
        or evaluation.compiled_model_sha256
        != model.get("compiled_model_sha256")
    ):
        raise FaceNeutralityError(
            "exact evaluation model differs from grounded receipt"
        )
    if len(evaluation.rows) != 16:
        raise FaceNeutralityError("exact evaluation must contain 16 target rows")
    indexed = {}
    for row in evaluation.rows:
        key = (row.scope, row.phase, row.face)
        if key in indexed:
            raise FaceNeutralityError("exact evaluation duplicates {}".format(key))
        if (
            row.scope not in SCOPES
            or row.phase not in PHASES
            or row.face not in FACES
            or row.target_sha256 != _neutral_array_sha256(row.normal_w)
        ):
            raise FaceNeutralityError("exact evaluation row contract changed")
        indexed[key] = row
    expected = [
        (scope, phase, face_name)
        for scope in SCOPES
        for phase in PHASES
        for face_name in FACES
    ]
    if list(indexed) != expected:
        raise FaceNeutralityError("exact evaluation target order changed")
    ready_normal = _unit(evaluation.ready_normal_w, "ready face normal")
    rows = []
    maximum = 0.0
    for scope in SCOPES:
        for phase in PHASES:
            bh = indexed[(scope, phase, "bh")]
            fh = indexed[(scope, phase, "fh")]
            bh_distance = _angle(ready_normal, bh.normal_w)
            fh_distance = _angle(ready_normal, fh.normal_w)
            asymmetry = abs(bh_distance - fh_distance)
            maximum = max(maximum, asymmetry)
            rows.append(
                {
                    "scope": scope,
                    "phase": phase,
                    "bh_target_sha256": bh.target_sha256,
                    "fh_target_sha256": fh.target_sha256,
                    "bh_distance_rad": bh_distance,
                    "fh_distance_rad": fh_distance,
                    "absolute_asymmetry_rad": asymmetry,
                }
            )
    if maximum > MAXIMUM_PAIR_ASYMMETRY_RAD:
        raise FaceNeutralityError(
            "ready face pair asymmetry {:.9f} rad exceeds 5 degrees".format(maximum)
        )
    return rows, maximum


def build_artifacts(
    validated: ValidatedInputs,
    evaluation: ExactEvaluation,
    *,
    producer_snapshot: FileSnapshot,
    target_repo_path: str,
) -> Tuple[bytes, bytes, str]:
    """Build target-set bytes and recipe-compatible report bytes."""

    report_rows, maximum = _validate_evaluation(validated, evaluation)
    target_rows = [
        {
            "scope": row.scope,
            "phase": row.phase,
            "face": row.face,
            "source_frame_index": row.source_frame_index,
            "pose_content_sha256": row.pose_content_sha256,
            "target_normal_w": row.normal_w.tolist(),
            "target_normal_sha256": row.target_sha256,
        }
        for row in evaluation.rows
    ]
    target_set = {
        "schema_version": 1,
        "artifact_type": "canonical-ready-face-target-set-v1",
        "normal_convention": FACE_NORMAL_CONVENTION,
        "input_bindings": {
            "minted_ready": validated.ready.binding(),
            "grounded_candidate": validated.candidate.binding(),
            "grounded_receipt": validated.ground_receipt.binding(),
            "lineage_pose_receipt": validated.lineage_receipt.binding(),
            "right_arm_challenger_receipt": (
                validated.challenger_receipt.binding()
            ),
            "recipe": validated.recipe.binding(),
            "phase_authority": validated.phase_authority.binding(),
            "producer": producer_snapshot.binding(),
        },
        "identity_checks": {
            "ready_equals_grounded_candidate": True,
            "grounded_right_arm_equals_challenger": True,
            "grounded_root_equals_challenger": True,
            "grounded_nonleg_equals_challenger": True,
        },
        "model": {
            "mjcf_sha256": evaluation.mjcf_sha256,
            "compiled_model_sha256": evaluation.compiled_model_sha256,
            "racket_site": RACKET_SITE,
        },
        "ready": {
            "path": validated.ready.repo_path,
            "sha256": validated.ready.sha256,
            "state_sha256": _ready_state_sha256(validated.state),
            "fresh_face_normal_w": evaluation.ready_normal_w.tolist(),
            "fresh_face_normal_sha256": _neutral_array_sha256(
                evaluation.ready_normal_w
            ),
        },
        "rows": target_rows,
        "all_rows_exact_fk": True,
        "authorization": dict(FALSE_AUTHORIZATION),
        "non_claims": [
            "not a canonical-ready selection or human-adoption decision",
            "not a motion, connector, behavior, or closed-loop balance certificate",
            "not training, deployment, or hardware authorization",
        ],
    }
    target_payload, _ = _sealed_json_bytes(
        target_set, "target_set_payload_sha256"
    )
    target_sha = _sha256_bytes(target_payload)
    report = {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "artifact_class": ARTIFACT_CLASS,
        "producer": {
            "tool_path": producer_snapshot.repo_path,
            "tool_sha256": producer_snapshot.sha256,
            "independent_from_ready_minter": True,
            "backend": BACKEND_NAME,
        },
        "ready": {
            "path": validated.ready.repo_path,
            "sha256": validated.ready.sha256,
            "state_sha256": _ready_state_sha256(validated.state),
        },
        "model": {
            "mjcf_sha256": evaluation.mjcf_sha256,
            "compiled_model_sha256": evaluation.compiled_model_sha256,
            "racket_site": RACKET_SITE,
            "face_normal_convention": FACE_NORMAL_CONVENTION,
        },
        "evaluation": {
            "scopes": list(SCOPES),
            "phases": list(PHASES),
            "faces": list(FACES),
            "target_set_path": target_repo_path,
            "target_set_sha256": target_sha,
            "rows": list(report_rows),
            "maximum_pair_asymmetry_rad": maximum,
            "maximum_allowed_pair_asymmetry_rad": MAXIMUM_PAIR_ASYMMETRY_RAD,
            "all_rows_exact_fk": True,
        },
        "verdict": "PASS_FACE_NEUTRAL_READY",
        "authorization": dict(FALSE_AUTHORIZATION),
        "non_claims": [
            "not a canonical-ready selection or human-adoption decision",
            "not a grounded-static, trajectory, behavior, or deployment certificate",
            "not training, deployment, or hardware authorization",
        ],
    }
    report_payload, report_payload_sha = _sealed_json_bytes(
        report, "report_payload_sha256"
    )
    return target_payload, report_payload, report_payload_sha


def _exclusive_write_at(directory_fd: int, filename: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, mode=0o644, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise OSError("exclusive evidence write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(
    validated: ValidatedInputs,
    evaluation: ExactEvaluation,
    output_directory: Path,
    *,
    producer_path: Optional[Path] = None,
) -> PublishedFaceEvidence:
    """Publish target set then report into a newly-created directory."""

    output = output_directory
    if not output.is_absolute():
        output = Path.cwd() / output
    name = output.name
    if not name or name in {".", ".."}:
        raise FaceNeutralityError("output directory needs one final component")
    try:
        parent = output.parent.resolve(strict=True)
        parent.relative_to(validated.repo_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise FaceNeutralityError(
            "output parent must exist inside repo root"
        ) from exc
    output = parent / name
    target_path = output / TARGET_SET_FILENAME
    try:
        target_repo_path = target_path.relative_to(validated.repo_root).as_posix()
    except ValueError as exc:
        raise FaceNeutralityError("target set must be inside repo root") from exc
    actual_producer = (
        Path(__file__).resolve()
        if producer_path is None
        else producer_path.resolve(strict=True)
    )
    producer = _snapshot(
        validated.repo_root, actual_producer, "face-neutrality producer"
    )
    expected_scripts = (
        validated.repo_root / "hope_training/whole_body_tracking/scripts"
    ).resolve()
    try:
        producer.path.relative_to(expected_scripts)
    except ValueError as exc:
        raise FaceNeutralityError(
            "producer must live under the repository scripts root"
        ) from exc
    if producer.path.name == "canonical_ready_sidecar_mint.py":
        raise FaceNeutralityError("face producer may not be the ready minter")

    snapshots = (
        ("minted ready", validated.ready),
        ("grounded candidate", validated.candidate),
        ("ground receipt", validated.ground_receipt),
        ("lineage receipt", validated.lineage_receipt),
        ("challenger receipt", validated.challenger_receipt),
        ("recipe", validated.recipe),
        ("phase authority", validated.phase_authority),
    )
    for label, snapshot in snapshots:
        current = _snapshot(validated.repo_root, snapshot.path, label)
        if current.sha256 != snapshot.sha256 or current.payload != snapshot.payload:
            raise FaceNeutralityError("{} changed after validation".format(label))

    target_payload, report_payload, report_payload_sha = build_artifacts(
        validated,
        evaluation,
        producer_snapshot=producer,
        target_repo_path=target_repo_path,
    )
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(str(parent), parent_flags)
    output_fd = -1
    created = False
    try:
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            raise FileExistsError(
                "refusing to overwrite face evidence: {}".format(output)
            ) from None
        output_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(name, output_flags, dir_fd=parent_fd)
        _exclusive_write_at(output_fd, TARGET_SET_FILENAME, target_payload)
        # The recipe-visible verdict is the completion marker and is last.
        _exclusive_write_at(output_fd, REPORT_FILENAME, report_payload)
        os.fsync(output_fd)
        os.fsync(parent_fd)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode) or not os.path.samestat(
            entry, os.fstat(output_fd)
        ):
            raise FaceNeutralityError(
                "published evidence directory identity changed during write"
            )
        return PublishedFaceEvidence(
            directory=output,
            target_set=output / TARGET_SET_FILENAME,
            report=output / REPORT_FILENAME,
            target_set_sha256=_sha256_bytes(target_payload),
            report_sha256=_sha256_bytes(report_payload),
            report_payload_sha256=report_payload_sha,
        )
    except Exception:
        if output_fd >= 0:
            for filename in (REPORT_FILENAME, TARGET_SET_FILENAME):
                try:
                    os.unlink(filename, dir_fd=output_fd)
                except (FileNotFoundError, OSError):
                    pass
        if created:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ground-receipt", type=Path, required=True)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--challenger-receipt", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--phase-authority", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validated = validate_inputs(
            repo_root=args.repo_root,
            ready_path=args.ready,
            candidate_path=args.candidate,
            ground_receipt_path=args.ground_receipt,
            lineage_receipt_path=args.lineage_receipt,
            challenger_receipt_path=args.challenger_receipt,
            recipe_path=args.recipe,
            phase_authority_path=args.phase_authority,
        )
        evaluation = recompute_exact_evidence(validated)
        result = publish(validated, evaluation, args.output_directory)
    except (FaceNeutralityError, FileExistsError, OSError, ValueError) as exc:
        print("FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": "PASS_FACE_NEUTRAL_READY",
                "directory": str(result.directory),
                "target_set": str(result.target_set),
                "target_set_sha256": result.target_set_sha256,
                "report": str(result.report),
                "report_sha256": result.report_sha256,
                "report_payload_sha256": result.report_payload_sha256,
                "authorization": dict(FALSE_AUTHORIZATION),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
