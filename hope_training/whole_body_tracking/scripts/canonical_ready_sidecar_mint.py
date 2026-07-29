#!/usr/bin/env python3
"""Mint a strict nine-key canonical-ready sidecar from a pinned G1 bundle.

This is a host-only identity adapter.  It does not import MuJoCo, rerun
physics, select a canonical ready, or authorize training/hardware.  It accepts
the exact diagnostic bundle produced by ``canonical_grounded_ready.py`` and
fails closed unless:

* the candidate NPZ and receipt JSON match explicit caller-supplied SHA-256s;
* both receipt payload seals and the candidate/receipt cross-bindings close;
* the exact-MuJoCo G1 receipt reports every static-ground gate as ``PASS``;
* candidate and receipt both deny training/hardware authorization;
* the candidate is one finite 31-joint, unit-root, exact-zero-velocity state;
* receipt state arrays/digests are exact matches for the candidate bytes; and
* the G1 right-arm overlay identity matches the canonical seven-joint chain.

The output NPZ has exactly the nine keys consumed by the current canonical
recipe ready loader.  Its provenance is intentionally *not* a legacy donor
frame claim: ``source_segment`` names the grounded-ready construction and
``source_npz`` is the repository-relative candidate path.  The companion JSON
keeps ground and face identity separate.  In particular, a grounded-ready
receipt contains no face-neutrality proof, so the face identity remains
``NOT_PROVEN`` and an independent face report is required.

Publishing is no-clobber: a new output directory is created exclusively,
the NPZ is written with ``O_EXCL``, and the identity report is written last.
On failure, partial files and the newly-created directory are removed.
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
from typing import Any, Mapping, Sequence

import numpy as np

from canonical_face_manifold import RIGHT_STRIKE_CHAIN
from canonical_grounded_ready import RUNTIME_JOINT_NAMES


READY_FILENAME = "canonical_ready_v2_g1_neutral_arm.npz"
IDENTITY_REPORT_FILENAME = "IDENTITY_REPORT.json"
SOURCE_SEGMENT = "grounded_ready_v2_g1_neutral_arm"
TOOL_ID = "canonical_ready_sidecar_mint_v1"

_SHA256_LENGTH = 64
_CANDIDATE_KEYS = frozenset(
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
_READY_KEYS = frozenset(
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
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_class",
        "trust_scope",
        "candidate_id",
        "verdict",
        "candidate",
        "source",
        "exact_model",
        "foot_targets",
        "static_geometry",
        "static_ground_dynamics",
        "gates",
        "authorization",
        "selection",
        "non_claims",
        "config",
        "receipt_payload_sha256",
        "publication",
        "publication_payload_sha256",
    }
)
_RECEIPT_CANDIDATE_KEYS = frozenset(
    {
        "state_sha256",
        "joint_pos_sha256",
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_wxyz",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "zero_velocity_emitted",
    }
)
_GATE_KEYS = frozenset(
    {
        "exact_model_identity",
        "joint_limits",
        "foot_pose",
        "leg_to_foot_jacobian",
        "double_support",
        "sole_floor",
        "collision",
        "support_margin",
        "static_ground_dynamics",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "selected_as_canonical_ready",
        "automatic_G1_or_G2_adoption",
        "requires_outer_comparison_across_all_five_motions",
    }
)
_PUBLICATION_KEYS = frozenset(
    {
        "candidate_filename",
        "candidate_npz_sha256",
        "receipt_filename",
        "completion_semantics",
    }
)
_TRUST_SCOPE = {
    "value_class": "UNTRUSTED_DIAGNOSTIC_UNTIL_CONSTRUCTION_BOUND_PUBLICATION",
    "receipt_sha_semantics": "CONTENT_INTEGRITY_NOT_AUTHENTICATION",
    "trusted_compute_base": "LOADED_CODE_AND_INTERPRETER",
}
_EXPECTED_SOURCE_MODE = "G1_donor_root_flat_feet_leg12_continuation"
_NOTE = (
    "diagnostic identity promotion from exact G1 static-ground candidate; "
    "not donor-frame exact; face identity and training authorization remain external"
)


class ReadySidecarMintError(RuntimeError):
    """The input bundle or requested publication violates the mint contract."""


@dataclass(frozen=True)
class ValidatedReadyCandidate:
    """Exact state and identity retained after fail-closed input validation."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    candidate_path: Path
    candidate_repo_path: str
    candidate_sha256: str
    receipt_path: Path
    receipt_repo_path: str
    receipt_file_sha256: str
    receipt_payload_sha256: str
    publication_payload_sha256: str
    receipt: Mapping[str, Any]
    striking_joint_ids: np.ndarray


@dataclass(frozen=True)
class MintedReadySidecar:
    """Paths and byte identities of one exclusively published bundle."""

    directory: Path
    ready_npz: Path
    identity_report_json: Path
    ready_npz_sha256: str
    identity_report_json_sha256: str
    identity_report_payload_sha256: str


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReadySidecarMintError(f"{label} must be a 64-character lowercase SHA-256")
    return value


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
        raise ReadySidecarMintError(
            f"identity JSON is not finite/canonicalizable: {exc}"
        ) from exc


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReadySidecarMintError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReadySidecarMintError(
                    f"{label} contains non-finite JSON number {token!r}"
                )
            ),
        )
    except ReadySidecarMintError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReadySidecarMintError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadySidecarMintError(f"{label} must contain one JSON object")
    _canonical_json_bytes(value)
    return value


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadySidecarMintError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise ReadySidecarMintError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _resolve_repo_file(
    repo_root: Path,
    value: str | Path,
    label: str,
) -> tuple[Path, str]:
    raw = Path(value)
    lexical = raw if raw.is_absolute() else repo_root / raw
    try:
        if stat.S_ISLNK(lexical.lstat().st_mode):
            raise ReadySidecarMintError(f"{label} may not be a symlink: {lexical}")
        resolved = lexical.resolve(strict=True)
        relative = resolved.relative_to(repo_root)
    except ReadySidecarMintError:
        raise
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ReadySidecarMintError(
            f"{label} must be a regular file inside repo root: {lexical}"
        ) from exc
    if not resolved.is_file():
        raise ReadySidecarMintError(f"{label} is not a regular file: {resolved}")
    return resolved, relative.as_posix()


def _read_regular_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReadySidecarMintError(f"cannot open {label}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReadySidecarMintError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _finite_vector(
    value: Any,
    length: int,
    label: str,
    *,
    dtype: np.dtype[Any] = np.dtype("float64"),
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype != dtype:
        raise ReadySidecarMintError(
            f"{label} must have dtype {dtype} and shape ({length},), "
            f"got {array.dtype} {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ReadySidecarMintError(f"{label} contains NaN or infinity")
    return np.ascontiguousarray(array.copy())


def _scalar_text(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ReadySidecarMintError(f"{label} must be one scalar string")
    text = str(array.item())
    if not text or text.strip() != text:
        raise ReadySidecarMintError(f"{label} must be a non-empty trimmed string")
    return text


def _scalar_false(value: Any, label: str) -> None:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("bool") or bool(array.item()):
        raise ReadySidecarMintError(f"{label} must be one exact false boolean")


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


def _state_sha256(
    joint_pos: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "joint_pos", joint_pos)
    _hash_array(digest, "root_pos_w", root_pos_w)
    _hash_array(digest, "root_quat_wxyz", root_quat_wxyz)
    return digest.hexdigest()


def _receipt_vector(value: Any, expected: np.ndarray, label: str) -> None:
    try:
        actual = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReadySidecarMintError(f"{label} is not numeric") from exc
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.array_equal(actual, expected)
    ):
        raise ReadySidecarMintError(f"{label} differs from candidate NPZ")


def _validate_receipt_seals(receipt: Mapping[str, Any]) -> tuple[str, str]:
    payload_sha = _require_sha256(
        receipt.get("receipt_payload_sha256"), "receipt payload SHA-256"
    )
    publication_sha = _require_sha256(
        receipt.get("publication_payload_sha256"),
        "receipt publication payload SHA-256",
    )
    unsigned_publication = dict(receipt)
    unsigned_publication.pop("publication_payload_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(unsigned_publication)) != publication_sha:
        raise ReadySidecarMintError("receipt publication payload seal does not close")

    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("publication_payload_sha256", None)
    unsigned_receipt.pop("publication", None)
    unsigned_receipt.pop("receipt_payload_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(unsigned_receipt)) != payload_sha:
        raise ReadySidecarMintError(
            "receipt pre-publication payload seal does not close"
        )
    return payload_sha, publication_sha


def validate_ready_candidate_bundle(
    *,
    repo_root: str | Path,
    candidate_path: str | Path,
    expected_candidate_sha256: str,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
) -> ValidatedReadyCandidate:
    """Validate and return an immutable view of one exact G1 publication."""

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise ReadySidecarMintError(f"repo root is not a directory: {root}")
    candidate, candidate_repo_path = _resolve_repo_file(
        root, candidate_path, "candidate"
    )
    receipt_file, receipt_repo_path = _resolve_repo_file(root, receipt_path, "receipt")
    candidate_expected = _require_sha256(
        expected_candidate_sha256, "expected candidate SHA-256"
    )
    receipt_expected = _require_sha256(
        expected_receipt_sha256, "expected receipt SHA-256"
    )
    candidate_bytes = _read_regular_file(candidate, "candidate")
    receipt_bytes = _read_regular_file(receipt_file, "receipt")
    candidate_actual = _sha256_bytes(candidate_bytes)
    receipt_actual = _sha256_bytes(receipt_bytes)
    if candidate_actual != candidate_expected:
        raise ReadySidecarMintError(
            "candidate SHA-256 mismatch: "
            f"expected {candidate_expected}, got {candidate_actual}"
        )
    if receipt_actual != receipt_expected:
        raise ReadySidecarMintError(
            "receipt SHA-256 mismatch: "
            f"expected {receipt_expected}, got {receipt_actual}"
        )

    receipt = _strict_json_object(receipt_bytes, "grounded-ready receipt")
    _exact_keys(receipt, _RECEIPT_KEYS, "grounded-ready receipt")
    payload_sha, publication_sha = _validate_receipt_seals(receipt)

    try:
        with np.load(io.BytesIO(candidate_bytes), allow_pickle=False) as payload:
            if frozenset(payload.files) != _CANDIDATE_KEYS:
                raise ReadySidecarMintError(
                    "candidate NPZ keys changed; "
                    f"expected={sorted(_CANDIDATE_KEYS)}, got={sorted(payload.files)}"
                )
            joint_pos = _finite_vector(payload["joint_pos"], 31, "candidate joint_pos")
            joint_vel = _finite_vector(payload["joint_vel"], 31, "candidate joint_vel")
            root_pos = _finite_vector(payload["root_pos_w"], 3, "candidate root_pos_w")
            root_quat = _finite_vector(
                payload["root_quat_w"], 4, "candidate root_quat_w"
            )
            root_lin_vel = _finite_vector(
                payload["root_lin_vel_w"], 3, "candidate root_lin_vel_w"
            )
            root_ang_vel = _finite_vector(
                payload["root_ang_vel_w"], 3, "candidate root_ang_vel_w"
            )
            candidate_id = _scalar_text(payload["candidate_id"], "candidate_id")
            embedded_receipt_sha = _scalar_text(
                payload["receipt_sha256"], "candidate receipt_sha256"
            )
            _scalar_false(
                payload["training_authorized"], "candidate training_authorized"
            )
            _scalar_false(
                payload["hardware_authorized"], "candidate hardware_authorized"
            )
    except ReadySidecarMintError:
        raise
    except (OSError, ValueError) as exc:
        raise ReadySidecarMintError(f"cannot load candidate NPZ: {exc}") from exc

    zeros31 = np.zeros(31, dtype=np.float64)
    zeros3 = np.zeros(3, dtype=np.float64)
    if not np.array_equal(joint_vel, zeros31):
        raise ReadySidecarMintError("candidate joint_vel must be exact zero")
    if not np.array_equal(root_lin_vel, zeros3):
        raise ReadySidecarMintError("candidate root_lin_vel_w must be exact zero")
    if not np.array_equal(root_ang_vel, zeros3):
        raise ReadySidecarMintError("candidate root_ang_vel_w must be exact zero")
    quaternion_norm = float(np.linalg.norm(root_quat))
    if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ReadySidecarMintError(
            f"candidate root quaternion norm is {quaternion_norm}, expected 1"
        )
    if candidate_id != "G1" or receipt.get("candidate_id") != "G1":
        raise ReadySidecarMintError("only exact candidate_id='G1' is accepted")
    if embedded_receipt_sha != payload_sha:
        raise ReadySidecarMintError(
            "candidate embedded receipt SHA disagrees with receipt payload seal"
        )

    publication = _exact_keys(
        receipt.get("publication"), _PUBLICATION_KEYS, "receipt publication"
    )
    if (
        publication["candidate_filename"] != candidate.name
        or publication["candidate_npz_sha256"] != candidate_actual
        or publication["receipt_filename"] != receipt_file.name
        or publication["completion_semantics"]
        != "exclusive_directory_and_receipt_written_last"
    ):
        raise ReadySidecarMintError(
            "receipt publication does not bind the exact candidate/receipt files"
        )

    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_class")
        != "diagnostic_stationary_grounded_ready_candidate"
        or receipt.get("verdict") != "PASS_STATIC_GROUNDED_READY_CANDIDATE"
        or receipt.get("trust_scope") != _TRUST_SCOPE
    ):
        raise ReadySidecarMintError(
            "receipt schema, trust scope, or PASS verdict changed"
        )
    gates = _exact_keys(receipt.get("gates"), _GATE_KEYS, "receipt gates")
    if dict(gates) != {key: "PASS" for key in _GATE_KEYS}:
        raise ReadySidecarMintError("every exact G1 static-ground gate must be PASS")
    authorization = _exact_keys(
        receipt.get("authorization"),
        _AUTHORIZATION_KEYS,
        "receipt authorization",
    )
    if any(authorization[key] is not False for key in _AUTHORIZATION_KEYS):
        raise ReadySidecarMintError(
            "receipt must deny training, deployment, and hardware authorization"
        )
    selection = _exact_keys(
        receipt.get("selection"), _SELECTION_KEYS, "receipt selection"
    )
    if dict(selection) != {
        "selected_as_canonical_ready": False,
        "automatic_G1_or_G2_adoption": False,
        "requires_outer_comparison_across_all_five_motions": True,
    }:
        raise ReadySidecarMintError(
            "receipt selection must preserve the original non-adoption result"
        )

    exact_model = receipt.get("exact_model")
    if not isinstance(exact_model, Mapping):
        raise ReadySidecarMintError("receipt exact_model must be an object")
    if (
        exact_model.get("exact_mujoco_backend") is not True
        or exact_model.get("status") != "PASS_EXACT_MUJOCO"
        or tuple(exact_model.get("joint_order", ())) != RUNTIME_JOINT_NAMES
    ):
        raise ReadySidecarMintError(
            "receipt does not bind the exact MuJoCo backend/runtime joint order"
        )
    for key in (
        "mjcf_sha256",
        "compiled_model_sha256",
        "path_model_binding_sha256",
        "ground_model_binding_sha256",
        "joint_position_lower_sha256",
        "joint_position_upper_sha256",
    ):
        _require_sha256(exact_model.get(key), f"exact_model.{key}")

    static_geometry = receipt.get("static_geometry")
    static_dynamics = receipt.get("static_ground_dynamics")
    if (
        not isinstance(static_geometry, Mapping)
        or static_geometry.get("passed") is not True
        or not isinstance(static_dynamics, Mapping)
        or static_dynamics.get("feasible") is not True
        or static_dynamics.get("status") != "PASS_STATIC_GROUND_CONTACT_LP"
    ):
        raise ReadySidecarMintError(
            "receipt static geometry/dynamics aggregate must be exact PASS"
        )

    receipt_candidate = _exact_keys(
        receipt.get("candidate"),
        _RECEIPT_CANDIDATE_KEYS,
        "receipt candidate",
    )
    _receipt_vector(receipt_candidate["joint_pos"], joint_pos, "receipt joint_pos")
    _receipt_vector(receipt_candidate["joint_vel"], joint_vel, "receipt joint_vel")
    _receipt_vector(receipt_candidate["root_pos_w"], root_pos, "receipt root_pos_w")
    _receipt_vector(
        receipt_candidate["root_quat_wxyz"],
        root_quat,
        "receipt root_quat_wxyz",
    )
    _receipt_vector(
        receipt_candidate["root_lin_vel_w"],
        root_lin_vel,
        "receipt root_lin_vel_w",
    )
    _receipt_vector(
        receipt_candidate["root_ang_vel_w"],
        root_ang_vel,
        "receipt root_ang_vel_w",
    )
    if receipt_candidate["zero_velocity_emitted"] is not True:
        raise ReadySidecarMintError(
            "receipt candidate.zero_velocity_emitted must be true"
        )
    if receipt_candidate["joint_pos_sha256"] != _array_sha256(joint_pos):
        raise ReadySidecarMintError("receipt joint_pos_sha256 does not close")
    if receipt_candidate["state_sha256"] != _state_sha256(
        joint_pos, root_pos, root_quat
    ):
        raise ReadySidecarMintError("receipt candidate state_sha256 does not close")

    striking_ids = np.asarray(
        [RUNTIME_JOINT_NAMES.index(name) for name in RIGHT_STRIKE_CHAIN],
        dtype=np.int64,
    )
    source = receipt.get("source")
    if not isinstance(source, Mapping):
        raise ReadySidecarMintError("receipt source must be an object")
    overlay = source.get("upper_overlay")
    if (
        source.get("mode") != _EXPECTED_SOURCE_MODE
        or source.get("root_bitwise_preserved") is not True
        or source.get("nonleg_joint_values_bitwise_preserved") is not True
        or not isinstance(overlay, Mapping)
        or overlay.get("applied") is not True
        or tuple(overlay.get("joint_names", ())) != RIGHT_STRIKE_CHAIN
        or not np.array_equal(np.asarray(overlay.get("joint_indices")), striking_ids)
        or overlay.get("root_preserved") is not True
        or overlay.get("lower_preserved") is not True
    ):
        raise ReadySidecarMintError(
            "receipt is not the expected G1 plus seven-joint neutral-arm overlay"
        )
    _require_sha256(
        overlay.get("input_joint_pos_sha256"),
        "source upper_overlay.input_joint_pos_sha256",
    )
    if overlay.get("copied_values_sha256") != _array_sha256(joint_pos[striking_ids]):
        raise ReadySidecarMintError(
            "source upper-overlay copied-values digest does not close"
        )

    return ValidatedReadyCandidate(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos_w=root_pos,
        root_quat_wxyz=root_quat,
        candidate_path=candidate,
        candidate_repo_path=candidate_repo_path,
        candidate_sha256=candidate_actual,
        receipt_path=receipt_file,
        receipt_repo_path=receipt_repo_path,
        receipt_file_sha256=receipt_actual,
        receipt_payload_sha256=payload_sha,
        publication_payload_sha256=publication_sha,
        receipt=receipt,
        striking_joint_ids=striking_ids,
    )


def build_ready_npz_bytes(candidate: ValidatedReadyCandidate) -> bytes:
    """Return deterministic strict-nine-key ready bytes for a validated state."""

    output = io.BytesIO()
    np.savez(
        output,
        joint_pos=np.asarray(candidate.joint_pos, dtype=np.float64),
        joint_vel=np.asarray(candidate.joint_vel, dtype=np.float64),
        root_pos_w=np.asarray(candidate.root_pos_w, dtype=np.float64),
        root_quat_w=np.asarray(candidate.root_quat_wxyz, dtype=np.float64),
        source_segment=np.asarray(SOURCE_SEGMENT),
        source_npz=np.asarray(candidate.candidate_repo_path),
        source_frame=np.asarray(0, dtype=np.int64),
        striking_joint_ids=np.asarray(candidate.striking_joint_ids, dtype=np.int64),
        note=np.asarray(_NOTE),
    )
    payload = output.getvalue()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as ready:
            if frozenset(ready.files) != _READY_KEYS:
                raise ReadySidecarMintError("internal ready NPZ key-set mismatch")
            _finite_vector(ready["joint_pos"], 31, "minted joint_pos")
            minted_velocity = _finite_vector(ready["joint_vel"], 31, "minted joint_vel")
            _finite_vector(ready["root_pos_w"], 3, "minted root_pos_w")
            _finite_vector(ready["root_quat_w"], 4, "minted root_quat_w")
            if not np.array_equal(minted_velocity, np.zeros(31, np.float64)):
                raise ReadySidecarMintError("minted joint_vel is not exact zero")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ReadySidecarMintError):
            raise
        raise ReadySidecarMintError(
            f"cannot self-validate minted ready NPZ: {exc}"
        ) from exc
    return payload


def build_identity_report(
    candidate: ValidatedReadyCandidate,
    *,
    ready_npz_sha256: str,
) -> dict[str, Any]:
    """Build separate ground/face identity statements for the output sidecar."""

    ready_sha = _require_sha256(ready_npz_sha256, "ready NPZ SHA-256")
    receipt = candidate.receipt
    exact_model = receipt["exact_model"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "canonical-ready-sidecar-identity-v1",
        "tool_id": TOOL_ID,
        "artifact_class": "diagnostic_canonical_ready_sidecar",
        "source": {
            "candidate_path": candidate.candidate_repo_path,
            "candidate_npz_sha256": candidate.candidate_sha256,
            "receipt_path": candidate.receipt_repo_path,
            "receipt_json_sha256": candidate.receipt_file_sha256,
            "receipt_payload_sha256": candidate.receipt_payload_sha256,
            "publication_payload_sha256": candidate.publication_payload_sha256,
            "candidate_id": "G1",
            "source_segment": SOURCE_SEGMENT,
            "source_frame": 0,
        },
        "upstream_selection": dict(receipt["selection"]),
        "ready_state": {
            "joint_count": 31,
            "joint_pos_sha256": _array_sha256(candidate.joint_pos),
            "joint_vel_exact_zero": True,
            "root_velocity_exact_zero": True,
            "state_sha256": _state_sha256(
                candidate.joint_pos,
                candidate.root_pos_w,
                candidate.root_quat_wxyz,
            ),
            "root_quaternion_wxyz_norm": float(
                np.linalg.norm(candidate.root_quat_wxyz)
            ),
            "striking_joint_ids": candidate.striking_joint_ids.tolist(),
            "striking_joint_names": list(RIGHT_STRIKE_CHAIN),
        },
        "ground_identity": {
            "status": "PASS_BOUND_UPSTREAM_G1_STATIC_GROUND_RECEIPT",
            "physics_rerun_by_this_tool": False,
            "upstream_exact_mujoco_backend": True,
            "upstream_gates": dict(receipt["gates"]),
            "mjcf_sha256": exact_model["mjcf_sha256"],
            "compiled_model_sha256": exact_model["compiled_model_sha256"],
            "path_model_binding_sha256": exact_model["path_model_binding_sha256"],
            "ground_model_binding_sha256": exact_model["ground_model_binding_sha256"],
            "claim_scope": (
                "content-bound upstream identity and static-ground receipt only"
            ),
        },
        "face_identity": {
            "status": "NOT_PROVEN_BY_GROUNDED_READY_RECEIPT",
            "face_neutrality_proven": False,
            "external_face_identity_report_required": True,
            "claim_scope": (
                "right-arm overlay bytes are identified; face FK/neutrality is not"
            ),
        },
        "recipe_compatibility": {
            "strict_nine_key_ready_schema": True,
            "legacy_donor_frame_exact_contract": False,
            "required_recipe_provenance_mode": (
                "selected_static_grounded_ready_identity_v1"
            ),
            "identity_report_must_be_content_bound": True,
        },
        "output": {
            "ready_filename": READY_FILENAME,
            "ready_npz_sha256": ready_sha,
            "identity_report_filename": IDENTITY_REPORT_FILENAME,
            "completion_semantics": (
                "exclusive_directory_and_identity_report_written_last"
            ),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "not a canonical-ready selection or adoption decision",
            "not a face-neutrality certificate",
            "not a motion, connector, dynamics replay, or behavior certificate",
            "not training, deployment, or hardware authorization",
        ],
    }
    report["report_payload_sha256"] = _sha256_bytes(_canonical_json_bytes(report))
    return report


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
                raise OSError("exclusive artifact write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mint_ready_sidecar(
    candidate: ValidatedReadyCandidate,
    output_directory: str | Path,
) -> MintedReadySidecar:
    """Publish NPZ + identity report into one new exclusive directory."""

    output = Path(output_directory)
    if not output.is_absolute():
        output = Path.cwd() / output
    name = output.name
    if not name or name in {".", ".."}:
        raise ReadySidecarMintError(
            "output directory needs one concrete final component"
        )
    try:
        parent = output.parent.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReadySidecarMintError(
            f"output parent does not exist: {output.parent}"
        ) from exc
    if not parent.is_dir():
        raise ReadySidecarMintError(f"output parent is not a directory: {parent}")
    output = parent / name

    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    output_fd = -1
    created = False
    try:
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite canonical-ready sidecar bundle: {output}"
            ) from None
        output_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(name, output_flags, dir_fd=parent_fd)

        ready_payload = build_ready_npz_bytes(candidate)
        ready_sha = _sha256_bytes(ready_payload)
        _exclusive_write_at(output_fd, READY_FILENAME, ready_payload)

        report = build_identity_report(candidate, ready_npz_sha256=ready_sha)
        report_payload = _canonical_json_bytes(report)
        _exclusive_write_at(output_fd, IDENTITY_REPORT_FILENAME, report_payload)
        os.fsync(output_fd)
        os.fsync(parent_fd)

        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode) or not os.path.samestat(
            entry, os.fstat(output_fd)
        ):
            raise ReadySidecarMintError(
                "published directory identity changed during write"
            )
        return MintedReadySidecar(
            directory=output,
            ready_npz=output / READY_FILENAME,
            identity_report_json=output / IDENTITY_REPORT_FILENAME,
            ready_npz_sha256=ready_sha,
            identity_report_json_sha256=_sha256_bytes(report_payload),
            identity_report_payload_sha256=str(report["report_payload_sha256"]),
        )
    except Exception:
        if output_fd >= 0:
            for filename in (IDENTITY_REPORT_FILENAME, READY_FILENAME):
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mint a strict diagnostic canonical-ready sidecar from an exact, "
            "content-pinned G1 static-ground bundle."
        )
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        validated = validate_ready_candidate_bundle(
            repo_root=args.repo_root,
            candidate_path=args.candidate,
            expected_candidate_sha256=args.candidate_sha256,
            receipt_path=args.receipt,
            expected_receipt_sha256=args.receipt_sha256,
        )
        published = mint_ready_sidecar(validated, args.output_dir)
    except (ReadySidecarMintError, FileExistsError, OSError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": "PASS_DIAGNOSTIC_IDENTITY_MINT",
                "training_authorized": False,
                "hardware_authorized": False,
                "ready_npz": str(published.ready_npz),
                "ready_npz_sha256": published.ready_npz_sha256,
                "identity_report_json": str(published.identity_report_json),
                "identity_report_json_sha256": (published.identity_report_json_sha256),
                "identity_report_payload_sha256": (
                    published.identity_report_payload_sha256
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IDENTITY_REPORT_FILENAME",
    "MintedReadySidecar",
    "READY_FILENAME",
    "ReadySidecarMintError",
    "SOURCE_SEGMENT",
    "TOOL_ID",
    "ValidatedReadyCandidate",
    "build_identity_report",
    "build_ready_npz_bytes",
    "main",
    "mint_ready_sidecar",
    "validate_ready_candidate_bundle",
]
