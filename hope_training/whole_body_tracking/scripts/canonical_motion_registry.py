#!/usr/bin/env python3
"""Strict registry and runtime adapter for one canonical five-motion bank.

The canonical library has two independent banks: ``upper`` and ``full``.  Each
bank contains the same five motion identities in one fixed order, but every
entry binds its own scope-specific NPZ and provenance.  This module is the
single place that turns a registry into the four runtime lists that otherwise
drift independently:

* ``motion_file`` paths;
* ``clip_family_per_clip``;
* ``strike_phase_per_clip``; and
* ``mount_normal_sign_per_clip``.

The adapter also carries the explicit ``motion_ids`` selector keys and two
digests.  The registry digest binds the exact JSON bytes.  The alignment digest
binds the ordered semantic rows used to derive the runtime tables.

Loading is deliberately fail-closed.  JSON objects use exact keys, repository
paths cannot escape through ``..`` or symlinks, all referenced bytes are
SHA-256 bound, and motion NPZ files must satisfy the exact A3 schema-2 contract.
The canonical-ready file is itself content-addressed and its joint/root values
must exactly equal every NPZ endpoint.  Registry loading remains an
identity/schema gate; runtime adaptation separately requires a pinned registry,
the matching purpose metadata, and an opaque capability minted by the
code-rooted promotion-certificate verifier.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

import sys

sys.path.insert(0, str(_HERE))
from motion_kinematics_contract import (  # noqa: E402
    BODY_LIN_VEL_POINT,
    BODY_NAMES_KEY,
    BODY_POS_POINT,
    KINEMATICS_SCHEMA_VERSION,
    LIN_VEL_POINT_KEY,
    MIGRATION_SOURCE_POINT_KEY,
    MIGRATION_SOURCE_SHA256_KEY,
    MIGRATION_TOOL_KEY,
    POS_POINT_KEY,
    SCHEMA_KEY,
)
from mujoco_motion_player import RUNTIME_BODY_NAMES  # noqa: E402


def _load_local_motion_admission():
    """Ignore caller-populated import aliases and execute the sibling verifier."""

    module_name = "_hope_canonical_motion_admission_runtime"
    script = _HERE / "canonical_motion_admission.py"
    if not script.is_file():
        raise RuntimeError(f"canonical motion admission module is missing: {script}")
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot create canonical motion admission spec for {script}"
        )
    module = importlib.util.module_from_spec(spec)
    # Dataclass construction consults sys.modules while the module executes.
    # Deliberately replace, rather than trust, any caller-preloaded alias.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if Path(module.__file__).resolve() != script:
        raise RuntimeError("canonical motion admission resolved to a wrong file")
    return module


motion_admission = _load_local_motion_admission()


REGISTRY_SCHEMA_VERSION = 1
GENERIC_REGISTRY_SCHEMA_VERSION = 2
CANONICAL_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
SCOPES = ("upper", "full")
FAMILIES = ("forehand", "backhand")

TIME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
ACTIVE_METADATA_KEYS = (
    SCHEMA_KEY,
    POS_POINT_KEY,
    LIN_VEL_POINT_KEY,
    BODY_NAMES_KEY,
)
MIGRATION_KEYS = (
    MIGRATION_SOURCE_SHA256_KEY,
    MIGRATION_SOURCE_POINT_KEY,
    MIGRATION_TOOL_KEY,
)
BASE_KEYS = ("fps",) + TIME_KEYS + ACTIVE_METADATA_KEYS
ALLOWED_SCHEMA2_KEYSETS = (
    frozenset(BASE_KEYS),
    frozenset(BASE_KEYS + MIGRATION_KEYS),
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "bank_id",
        "scope",
        "canonical_ready_path",
        "canonical_ready_sha256",
        "canonical_ready_fk_path",
        "canonical_ready_fk_sha256",
        "entries",
    }
)
_TOP_LEVEL_KEYS_V2 = _TOP_LEVEL_KEYS | frozenset({"motion_ids"})
_ENTRY_KEYS = frozenset(
    {
        "motion_id",
        "scope",
        "variant",
        "npz_path",
        "npz_sha256",
        "frames",
        "fps",
        "family",
        "strike_marker_frame",
        "contact_opportunity_frames",
        "mount_normal_sign",
        "canonical_ready_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "build_manifest_path",
        "build_manifest_sha256",
        "applicability_manifest_path",
        "applicability_manifest_sha256",
        "evidence_level",
        "evidence_manifest_path",
        "evidence_manifest_sha256",
        "question_bank_path",
        "question_bank_sha256",
        "question_bank_schema_version",
        "training_config_path",
        "training_config_sha256",
        "training_config_schema_version",
        "onnx_model_path",
        "onnx_model_sha256",
        "onnx_model_schema_version",
        "onnx_metadata_path",
        "onnx_metadata_sha256",
        "onnx_metadata_schema_version",
        "adoption_manifest_path",
        "adoption_manifest_sha256",
        "publication_class",
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    }
)
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUATERNION_NORM_TOL = 2.0e-3
_READY_QUATERNION_NORM_TOL = 1.0e-6
_READY_FILE_KEYS = frozenset(
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
_READY_FK_FILE_KEYS = frozenset(
    {
        "canonical_ready_sha256",
        "body_names",
        "body_pos_w",
        "body_quat_w",
        "kinematics_contract_version",
    }
)
_READY_FK_CONTRACT_VERSION = 1
_ADOPTION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "motion_id",
        "scope",
        "variant",
        "npz_sha256",
        "strike_marker_frame",
        "contact_opportunity_frames",
        "mount_normal_sign",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "artifacts",
    }
)
_ADOPTION_ARTIFACT_NAMES = (
    "question_bank",
    "training_config",
    "onnx_model",
    "onnx_metadata",
)
_ADOPTION_ARTIFACT_BINDING_KEYS = frozenset({"sha256", "schema_version"})
_TRAINING_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "motion_id",
        "scope",
        "variant",
        "npz_sha256",
    }
)
_ONNX_METADATA_KEYS = frozenset(
    {
        "hope_metadata_schema_version",
        "motion_id",
        "scope",
        "variant",
        "npz_sha256",
        "strike_marker_frame",
        "contact_opportunity_frames",
        "mount_normal_sign",
        "canonical_ready_sha256",
        "canonical_ready_fk_sha256",
        "question_bank_sha256",
        "training_config_sha256",
        "onnx_model_sha256",
    }
)
_EVIDENCE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "motion_id",
        "scope",
        "variant",
        "npz_sha256",
        "highest_evidence_level",
        "certificates",
    }
)
_EVIDENCE_CERTIFICATE_REF_KEYS = frozenset(
    {"level", "path", "sha256", "status"}
)
_EVIDENCE_CERTIFICATE_KEYS = frozenset(
    {
        "schema_version",
        "level",
        "motion_id",
        "scope",
        "variant",
        "npz_sha256",
        "status",
    }
)
_APPLICABILITY_MANIFEST_KEYS = frozenset(
    {"schema_version", "motion_id", "scope", "variant", "npz_sha256", "domain"}
)
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = MappingProxyType(
    {
        "question_bank": frozenset({3}),
        "training_config": frozenset({1}),
        "onnx_model": frozenset({1}),
        "onnx_metadata": frozenset({2}),
    }
)
AUTHORIZATION_PURPOSES = ("training", "deployment", "hardware")
EVIDENCE_LEVELS = ("E0", "E1", "E2", "E3", "E4", "E5")
PUBLICATION_AUTHORIZATION = MappingProxyType(
    {
        "compiler_candidate": (False, False, False),
        "training_adopted": (True, False, False),
        "deployment_adopted": (True, True, False),
        "hardware_adopted": (True, True, True),
    }
)
PUBLICATION_CLASSES_BY_PURPOSE = MappingProxyType(
    {
        "training": frozenset(
            {"training_adopted", "deployment_adopted", "hardware_adopted"}
        ),
        "deployment": frozenset({"deployment_adopted", "hardware_adopted"}),
        "hardware": frozenset({"hardware_adopted"}),
    }
)
PUBLICATION_MIN_EVIDENCE = MappingProxyType(
    {
        "compiler_candidate": "E0",
        "training_adopted": "E2",
        "deployment_adopted": "E4",
        "hardware_adopted": "E5",
    }
)
PUBLICATION_REQUIRED_ARTIFACTS = MappingProxyType(
    {
        "compiler_candidate": frozenset(),
        "training_adopted": frozenset(
            {"question_bank", "training_config", "adoption_manifest"}
        ),
        "deployment_adopted": frozenset(
            {
                "question_bank",
                "training_config",
                "onnx_model",
                "onnx_metadata",
                "adoption_manifest",
            }
        ),
        "hardware_adopted": frozenset(
            {
                "question_bank",
                "training_config",
                "onnx_model",
                "onnx_metadata",
                "adoption_manifest",
            }
        ),
    }
)


class MotionRegistryError(ValueError):
    """The registry cannot be consumed without weakening its contract."""


@dataclass(frozen=True)
class MotionRegistryEntry:
    """One atomically bound motion row."""

    motion_id: str
    scope: str
    variant: str
    npz_path_text: str
    npz_path: Path
    npz_sha256: str
    frames: int
    fps: float
    family: str
    strike_marker_frame: int
    contact_opportunity_frames: tuple[int, int]
    mount_normal_sign: float
    canonical_ready_sha256: str
    source_manifest_path_text: str
    source_manifest_path: Path
    source_manifest_sha256: str
    build_manifest_path_text: str
    build_manifest_path: Path
    build_manifest_sha256: str
    applicability_manifest_path_text: str
    applicability_manifest_path: Path
    applicability_manifest_sha256: str
    evidence_level: str
    evidence_manifest_path_text: str
    evidence_manifest_path: Path
    evidence_manifest_sha256: str
    question_bank_path_text: str | None
    question_bank_path: Path | None
    question_bank_sha256: str | None
    question_bank_schema_version: int | None
    training_config_path_text: str | None
    training_config_path: Path | None
    training_config_sha256: str | None
    training_config_schema_version: int | None
    onnx_model_path_text: str | None
    onnx_model_path: Path | None
    onnx_model_sha256: str | None
    onnx_model_schema_version: int | None
    onnx_metadata_path_text: str | None
    onnx_metadata_path: Path | None
    onnx_metadata_sha256: str | None
    onnx_metadata_schema_version: int | None
    adoption_manifest_path_text: str | None
    adoption_manifest_path: Path | None
    adoption_manifest_sha256: str | None
    publication_class: str
    training_authorized: bool
    deployment_authorized: bool
    hardware_authorized: bool
    ready_runtime_sha256: str

    @property
    def strike_phase(self) -> float:
        """Runtime strike phase derived from the bound integer marker."""

        return float(self.strike_marker_frame) / float(self.frames - 1)


@dataclass(frozen=True)
class CanonicalMotionBankRegistry:
    """One validated scoped legacy-v1 or arbitrary-N v2 motion bank."""

    schema_version: int
    path: Path
    repo_root: Path
    registry_sha256: str
    bank_id: str
    scope: str
    canonical_ready_path_text: str
    canonical_ready_path: Path
    canonical_ready_sha256: str
    canonical_ready_fk_path_text: str
    canonical_ready_fk_path: Path
    canonical_ready_fk_sha256: str
    registry_digest_pinned: bool
    motion_ids: tuple[str, ...]
    entries: tuple[MotionRegistryEntry, ...]

    def entry(self, motion_id: str) -> MotionRegistryEntry:
        matches = tuple(row for row in self.entries if row.motion_id == motion_id)
        if len(matches) != 1:
            raise MotionRegistryError(
                f"bank has {len(matches)} entries named {motion_id!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class CanonicalRuntimeTables:
    """Aligned runtime, identity, provenance, authorization, and digest columns."""

    bank_id: str
    scope: str
    motion_file: tuple[str, ...]
    clip_family_per_clip: tuple[str, ...]
    strike_phase_per_clip: tuple[float, ...]
    contact_opportunity_frames_per_clip: tuple[tuple[int, int], ...]
    mount_normal_sign_per_clip: tuple[float, ...]
    motion_ids: tuple[str, ...]
    publication_class_per_clip: tuple[str, ...]
    source_manifest_path_per_clip: tuple[str, ...]
    source_manifest_sha256_per_clip: tuple[str, ...]
    build_manifest_path_per_clip: tuple[str, ...]
    build_manifest_sha256_per_clip: tuple[str, ...]
    applicability_manifest_path_per_clip: tuple[str, ...]
    applicability_manifest_sha256_per_clip: tuple[str, ...]
    evidence_level_per_clip: tuple[str, ...]
    evidence_manifest_path_per_clip: tuple[str, ...]
    evidence_manifest_sha256_per_clip: tuple[str, ...]
    question_bank_path_per_clip: tuple[str | None, ...]
    question_bank_sha256_per_clip: tuple[str | None, ...]
    question_bank_schema_version_per_clip: tuple[int | None, ...]
    training_config_path_per_clip: tuple[str | None, ...]
    training_config_sha256_per_clip: tuple[str | None, ...]
    training_config_schema_version_per_clip: tuple[int | None, ...]
    onnx_model_path_per_clip: tuple[str | None, ...]
    onnx_model_sha256_per_clip: tuple[str | None, ...]
    onnx_model_schema_version_per_clip: tuple[int | None, ...]
    onnx_metadata_path_per_clip: tuple[str | None, ...]
    onnx_metadata_sha256_per_clip: tuple[str | None, ...]
    onnx_metadata_schema_version_per_clip: tuple[int | None, ...]
    adoption_manifest_path_per_clip: tuple[str | None, ...]
    adoption_manifest_sha256_per_clip: tuple[str | None, ...]
    training_authorized_per_clip: tuple[bool, ...]
    deployment_authorized_per_clip: tuple[bool, ...]
    hardware_authorized_per_clip: tuple[bool, ...]
    npz_sha256_per_clip: tuple[str, ...]
    ready_runtime_sha256: str
    canonical_ready_path: str
    canonical_ready_sha256: str
    canonical_ready_fk_path: str
    canonical_ready_fk_sha256: str
    registry_sha256: str
    alignment_sha256: str
    authorization_purpose: str | None

    def as_python_loader_tables(self) -> Mapping[str, tuple[Any, ...]]:
        """Return immutable config-ready columns for an authorized runtime purpose.

        ``authorization_purpose=None`` is intentionally useful for identity/schema
        audits, but an audit result must not double as a runnable loader table.
        """

        if self.authorization_purpose is None:
            raise MotionRegistryError(
                "identity-only registry audit cannot export runtime loader tables; "
                "choose an explicit authorization_purpose"
            )

        return MappingProxyType(
            {
                "motion_file": self.motion_file,
                "clip_family_per_clip": self.clip_family_per_clip,
                "strike_phase_per_clip": self.strike_phase_per_clip,
                "contact_opportunity_frames_per_clip": (
                    self.contact_opportunity_frames_per_clip
                ),
                "mount_normal_sign_per_clip": self.mount_normal_sign_per_clip,
                "motion_ids": self.motion_ids,
                "publication_class_per_clip": self.publication_class_per_clip,
                "source_manifest_path_per_clip": (
                    self.source_manifest_path_per_clip
                ),
                "source_manifest_sha256_per_clip": (
                    self.source_manifest_sha256_per_clip
                ),
                "build_manifest_path_per_clip": (
                    self.build_manifest_path_per_clip
                ),
                "build_manifest_sha256_per_clip": (
                    self.build_manifest_sha256_per_clip
                ),
                "applicability_manifest_path_per_clip": (
                    self.applicability_manifest_path_per_clip
                ),
                "applicability_manifest_sha256_per_clip": (
                    self.applicability_manifest_sha256_per_clip
                ),
                "evidence_level_per_clip": self.evidence_level_per_clip,
                "evidence_manifest_path_per_clip": (
                    self.evidence_manifest_path_per_clip
                ),
                "evidence_manifest_sha256_per_clip": (
                    self.evidence_manifest_sha256_per_clip
                ),
                "question_bank_path_per_clip": self.question_bank_path_per_clip,
                "question_bank_sha256_per_clip": self.question_bank_sha256_per_clip,
                "question_bank_schema_version_per_clip": (
                    self.question_bank_schema_version_per_clip
                ),
                "training_config_path_per_clip": self.training_config_path_per_clip,
                "training_config_sha256_per_clip": (
                    self.training_config_sha256_per_clip
                ),
                "training_config_schema_version_per_clip": (
                    self.training_config_schema_version_per_clip
                ),
                "onnx_model_path_per_clip": self.onnx_model_path_per_clip,
                "onnx_model_sha256_per_clip": self.onnx_model_sha256_per_clip,
                "onnx_model_schema_version_per_clip": (
                    self.onnx_model_schema_version_per_clip
                ),
                "onnx_metadata_path_per_clip": self.onnx_metadata_path_per_clip,
                "onnx_metadata_sha256_per_clip": (
                    self.onnx_metadata_sha256_per_clip
                ),
                "onnx_metadata_schema_version_per_clip": (
                    self.onnx_metadata_schema_version_per_clip
                ),
                "adoption_manifest_path_per_clip": (
                    self.adoption_manifest_path_per_clip
                ),
                "adoption_manifest_sha256_per_clip": (
                    self.adoption_manifest_sha256_per_clip
                ),
                "training_authorized_per_clip": self.training_authorized_per_clip,
                "deployment_authorized_per_clip": self.deployment_authorized_per_clip,
                "hardware_authorized_per_clip": self.hardware_authorized_per_clip,
                "npz_sha256_per_clip": self.npz_sha256_per_clip,
            }
        )


@dataclass(frozen=True)
class _Schema2Summary:
    frames: int
    fps: float
    ready_joint_pos: np.ndarray
    ready_root_pos: np.ndarray
    ready_root_quat: np.ndarray
    ready_body_pos: np.ndarray
    ready_body_quat: np.ndarray


@dataclass(frozen=True)
class _ReadySummary:
    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray


@dataclass(frozen=True)
class _ReadyFkSummary:
    body_pos: np.ndarray
    body_quat: np.ndarray


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MotionRegistryError(
            f"{label} must be a 64-character lowercase SHA-256 digest"
        )
    return value


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise MotionRegistryError(
            f"{label} must be a lowercase identifier matching {_SLUG.pattern!r}"
        )
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MotionRegistryError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise MotionRegistryError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MotionRegistryError(f"{label} must be a finite positive number")
    return result


def _exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionRegistryError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise MotionRegistryError(
            f"{label} keys changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MotionRegistryError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise MotionRegistryError(f"JSON contains forbidden non-finite constant {value}")


def _json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MotionRegistryError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MotionRegistryError(f"{label} must contain one JSON object")
    return value


def _root_path(repo_root: os.PathLike[str] | str | None) -> Path:
    candidate = _REPO_ROOT if repo_root is None else Path(repo_root)
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise MotionRegistryError(f"cannot resolve repository root {candidate}: {exc}") from exc
    if not root.is_dir():
        raise MotionRegistryError(f"repository root is not a directory: {root}")
    return root


def _inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise MotionRegistryError(f"cannot resolve {label} {path}: {exc}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MotionRegistryError(f"{label} escapes repository root: {path}") from exc
    if not resolved.is_file():
        raise MotionRegistryError(f"{label} is not a regular file: {resolved}")
    return resolved


def _registry_path(
    path: os.PathLike[str] | str, repo_root: Path
) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else repo_root / raw
    return _inside_repo(candidate, repo_root, "registry")


def _bound_repo_file(value: Any, repo_root: Path, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise MotionRegistryError(f"{label} must be a non-empty repository-relative path")
    if "\\" in value:
        raise MotionRegistryError(f"{label} must use POSIX '/' separators")
    if value.startswith("/") or value.endswith("/") or "//" in value:
        raise MotionRegistryError(f"{label} must be a normalized repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise MotionRegistryError(f"{label} may not be absolute or contain '.'/'..'")
    if str(pure) != value:
        raise MotionRegistryError(f"{label} must be normalized, got {value!r}")
    return value, _inside_repo(repo_root.joinpath(*pure.parts), repo_root, label)


def _read_snapshot(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MotionRegistryError(f"cannot read {label} {path}: {exc}") from exc


def _bound_snapshot(
    path: Path, expected_sha256: str, label: str
) -> bytes:
    payload = _read_snapshot(path, label)
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise MotionRegistryError(
            f"{label} SHA-256 mismatch: registry={expected_sha256}, actual={actual}"
        )
    return payload


def _scalar_text(array: Any, label: str) -> str:
    raw = np.asarray(array)
    if raw.dtype.kind not in ("U", "S") or raw.size != 1:
        raise MotionRegistryError(f"schema-2 {label} must be one string scalar")
    item = raw.reshape(-1)[0]
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MotionRegistryError(
                f"schema-2 {label} is not UTF-8"
            ) from exc
    return str(item)


def _safe_npz_keys(payload: bytes, label: str) -> frozenset[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (
                not infos
                or len(names) != len(set(names))
                or any(
                    item.is_dir()
                    or item.flag_bits & 0x1
                    or "/" in item.filename
                    or "\\" in item.filename
                    or not item.filename.endswith(".npy")
                    for item in infos
                )
            ):
                raise MotionRegistryError(
                    f"{label} has duplicated, nested, encrypted, or non-NPY members"
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise MotionRegistryError(f"{label} is not a valid NPZ/ZIP: {exc}") from exc
    return frozenset(name[:-4] for name in names)


def _validate_npz_members(payload: bytes, label: str) -> frozenset[str]:
    keys = _safe_npz_keys(payload, label)
    if keys not in ALLOWED_SCHEMA2_KEYSETS:
        raise MotionRegistryError(
            f"{label} field set changed; got={sorted(keys)}, "
            "expected exact schema-2 11/14 fields"
        )
    return keys


def _ready_float64_vector(
    data: Any, key: str, length: int, label: str
) -> np.ndarray:
    raw = np.asarray(data[key])
    if raw.dtype != np.dtype(np.float64) or raw.shape != (length,):
        raise MotionRegistryError(
            f"{label} {key} must be exact float64 shape ({length},), "
            f"got {raw.dtype}/{raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise MotionRegistryError(f"{label} {key} contains NaN/Inf")
    return raw


def _validate_ready_snapshot(payload: bytes, label: str) -> _ReadySummary:
    keys = _safe_npz_keys(payload, label)
    if keys != _READY_FILE_KEYS:
        raise MotionRegistryError(
            f"{label} field set changed; got={sorted(keys)}, "
            f"expected={sorted(_READY_FILE_KEYS)}"
        )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise MotionRegistryError(f"cannot load {label}: {exc}") from exc
    with loaded as data:
        if frozenset(data.files) != keys:
            raise MotionRegistryError(f"{label} ZIP members and NumPy keys disagree")
        joint_pos = _ready_float64_vector(data, "joint_pos", 31, label)
        joint_vel = _ready_float64_vector(data, "joint_vel", 31, label)
        root_pos = _ready_float64_vector(data, "root_pos_w", 3, label)
        root_quat = _ready_float64_vector(data, "root_quat_w", 4, label)
        if np.count_nonzero(joint_vel) != 0:
            raise MotionRegistryError(f"{label} joint_vel must be exact zero")
        quaternion_norm = float(np.linalg.norm(root_quat))
        if not math.isclose(
            quaternion_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_READY_QUATERNION_NORM_TOL,
        ):
            raise MotionRegistryError(
                f"{label} root quaternion norm {quaternion_norm:.9g} is not unit"
            )
        for key in ("source_segment", "source_npz", "note"):
            if not _scalar_text(data[key], key):
                raise MotionRegistryError(f"{label} {key} must be non-empty")
        source_frame = np.asarray(data["source_frame"])
        if (
            source_frame.shape != ()
            or not np.issubdtype(source_frame.dtype, np.integer)
        ):
            raise MotionRegistryError(
                f"{label} source_frame must be one integer scalar"
            )
        striking_ids = np.asarray(data["striking_joint_ids"])
        if (
            striking_ids.shape != (7,)
            or not np.issubdtype(striking_ids.dtype, np.integer)
            or len(set(int(value) for value in striking_ids.tolist())) != 7
            or np.any((striking_ids < 0) | (striking_ids >= 31))
        ):
            raise MotionRegistryError(
                f"{label} striking_joint_ids must be seven unique indices in [0,31)"
            )
    return _ReadySummary(
        joint_pos=joint_pos.astype(np.float32),
        root_pos=root_pos.astype(np.float32),
        root_quat=root_quat.astype(np.float32),
    )


def _validate_ready_fk_snapshot(
    payload: bytes,
    *,
    canonical_ready_sha256: str,
    canonical_ready: _ReadySummary,
    label: str,
) -> _ReadyFkSummary:
    """Validate the independently content-addressed 32-body ready FK truth."""

    keys = _safe_npz_keys(payload, label)
    if keys != _READY_FK_FILE_KEYS:
        raise MotionRegistryError(
            f"{label} field set changed; got={sorted(keys)}, "
            f"expected={sorted(_READY_FK_FILE_KEYS)}"
        )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise MotionRegistryError(f"cannot load {label}: {exc}") from exc
    with loaded as data:
        if frozenset(data.files) != keys:
            raise MotionRegistryError(f"{label} ZIP members and NumPy keys disagree")
        bound_ready_sha = _scalar_text(
            data["canonical_ready_sha256"], "canonical_ready_sha256"
        )
        if bound_ready_sha != canonical_ready_sha256:
            raise MotionRegistryError(
                f"{label} does not bind the canonical ready SHA-256"
            )
        contract_version = np.asarray(data["kinematics_contract_version"])
        if (
            contract_version.dtype != np.dtype(np.int64)
            or contract_version.shape != (1,)
            or int(contract_version[0]) != _READY_FK_CONTRACT_VERSION
        ):
            raise MotionRegistryError(
                f"{label} kinematics_contract_version must be exact int64 "
                f"[{_READY_FK_CONTRACT_VERSION}]"
            )
        names_raw = np.asarray(data["body_names"])
        if names_raw.dtype.kind not in ("U", "S") or names_raw.shape != (32,):
            raise MotionRegistryError(
                f"{label} body_names must be one exact 32-string vector"
            )
        names = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in names_raw.tolist()
        )
        if names != tuple(RUNTIME_BODY_NAMES):
            raise MotionRegistryError(
                f"{label} body_names differs from A3 runtime order"
            )
        body_pos = _numeric_float32(data, "body_pos_w", (32, 3), label)
        body_quat = _numeric_float32(data, "body_quat_w", (32, 4), label)
        quaternion_error = float(
            np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0))
        )
        if quaternion_error > _QUATERNION_NORM_TOL:
            raise MotionRegistryError(
                f"{label} body quaternion norm error {quaternion_error:.3e} "
                f"exceeds {_QUATERNION_NORM_TOL:.3e}"
            )
        if not np.array_equal(body_pos[0], canonical_ready.root_pos):
            raise MotionRegistryError(
                f"{label} root body position differs from canonical ready"
            )
        if not np.array_equal(body_quat[0], canonical_ready.root_quat):
            raise MotionRegistryError(
                f"{label} root body quaternion differs from canonical ready"
            )
    return _ReadyFkSummary(
        body_pos=np.array(body_pos, copy=True),
        body_quat=np.array(body_quat, copy=True),
    )


def _numeric_float32(
    data: Any, key: str, shape: tuple[int, ...], label: str
) -> np.ndarray:
    raw = np.asarray(data[key])
    if raw.dtype != np.dtype(np.float32) or raw.shape != shape:
        raise MotionRegistryError(
            f"{label} {key} must be exact float32 shape {shape}, "
            f"got {raw.dtype}/{raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise MotionRegistryError(f"{label} {key} contains NaN/Inf")
    return raw


def _validate_schema2_snapshot(payload: bytes, label: str) -> _Schema2Summary:
    keys = _validate_npz_members(payload, label)
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise MotionRegistryError(f"cannot load {label}: {exc}") from exc
    with loaded as data:
        if frozenset(data.files) != keys:
            raise MotionRegistryError(f"{label} ZIP members and NumPy keys disagree")

        schema = np.asarray(data[SCHEMA_KEY])
        if (
            schema.dtype != np.dtype(np.int64)
            or schema.shape != (1,)
            or int(schema[0]) != KINEMATICS_SCHEMA_VERSION
        ):
            raise MotionRegistryError(
                f"{label} must contain exact int64 {SCHEMA_KEY}=[2]"
            )
        if _scalar_text(data[POS_POINT_KEY], POS_POINT_KEY) != BODY_POS_POINT:
            raise MotionRegistryError(
                f"{label} {POS_POINT_KEY} must be {BODY_POS_POINT!r}"
            )
        if (
            _scalar_text(data[LIN_VEL_POINT_KEY], LIN_VEL_POINT_KEY)
            != BODY_LIN_VEL_POINT
        ):
            raise MotionRegistryError(
                f"{label} {LIN_VEL_POINT_KEY} must be {BODY_LIN_VEL_POINT!r}"
            )
        body_names_raw = np.asarray(data[BODY_NAMES_KEY])
        if (
            body_names_raw.dtype.kind not in ("U", "S")
            or body_names_raw.shape != (32,)
        ):
            raise MotionRegistryError(
                f"{label} {BODY_NAMES_KEY} must be one exact 32-string vector"
            )
        body_names = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in body_names_raw.tolist()
        )
        if body_names != tuple(RUNTIME_BODY_NAMES):
            raise MotionRegistryError(
                f"{label} {BODY_NAMES_KEY} differs from A3 runtime order"
            )

        fps_raw = np.asarray(data["fps"])
        if (
            fps_raw.shape != (1,)
            or fps_raw.dtype.kind not in ("i", "u", "f")
        ):
            raise MotionRegistryError(f"{label} fps must be one numeric scalar")
        fps = float(fps_raw[0])
        if not math.isfinite(fps) or fps <= 0.0:
            raise MotionRegistryError(f"{label} fps must be finite and positive")

        joint_pos_raw = np.asarray(data["joint_pos"])
        if (
            joint_pos_raw.dtype != np.dtype(np.float32)
            or joint_pos_raw.ndim != 2
            or joint_pos_raw.shape[1] != 31
        ):
            raise MotionRegistryError(
                f"{label} joint_pos must be exact float32 shape (T,31)"
            )
        frames = int(joint_pos_raw.shape[0])
        if frames < 2:
            raise MotionRegistryError(f"{label} must contain at least two frames")

        joint_pos = _numeric_float32(data, "joint_pos", (frames, 31), label)
        joint_vel = _numeric_float32(data, "joint_vel", (frames, 31), label)
        body_pos = _numeric_float32(data, "body_pos_w", (frames, 32, 3), label)
        body_quat = _numeric_float32(data, "body_quat_w", (frames, 32, 4), label)
        body_lin = _numeric_float32(
            data, "body_lin_vel_w", (frames, 32, 3), label
        )
        body_ang = _numeric_float32(
            data, "body_ang_vel_w", (frames, 32, 3), label
        )

        quaternion_error = float(
            np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0))
        )
        if quaternion_error > _QUATERNION_NORM_TOL:
            raise MotionRegistryError(
                f"{label} body quaternion norm error {quaternion_error:.3e} "
                f"exceeds {_QUATERNION_NORM_TOL:.3e}"
            )
        if not np.array_equal(joint_pos[0], joint_pos[-1]):
            raise MotionRegistryError(
                f"{label} first/last joint pose is not the same canonical ready"
            )
        if not np.array_equal(body_pos[0], body_pos[-1]) or not np.array_equal(
            body_quat[0], body_quat[-1]
        ):
            raise MotionRegistryError(
                f"{label} first/last body pose is not the same canonical ready"
            )
        for key, array in (
            ("joint_vel", joint_vel),
            ("body_lin_vel_w", body_lin),
            ("body_ang_vel_w", body_ang),
        ):
            if np.count_nonzero(array[[0, -1]]) != 0:
                raise MotionRegistryError(
                    f"{label} {key} first/last frames must be exactly zero"
                )

        migration_present = tuple(key in keys for key in MIGRATION_KEYS)
        if any(migration_present) and not all(migration_present):
            raise MotionRegistryError(
                f"{label} migration provenance must be all-or-none"
            )
        if all(migration_present):
            migration_digest = _scalar_text(
                data[MIGRATION_SOURCE_SHA256_KEY],
                MIGRATION_SOURCE_SHA256_KEY,
            )
            if _SHA256.fullmatch(migration_digest.lower()) is None:
                raise MotionRegistryError(
                    f"{label} {MIGRATION_SOURCE_SHA256_KEY} is not SHA-256"
                )
            if not _scalar_text(
                data[MIGRATION_SOURCE_POINT_KEY], MIGRATION_SOURCE_POINT_KEY
            ):
                raise MotionRegistryError(
                    f"{label} {MIGRATION_SOURCE_POINT_KEY} must be non-empty"
                )
            if not _scalar_text(data[MIGRATION_TOOL_KEY], MIGRATION_TOOL_KEY):
                raise MotionRegistryError(
                    f"{label} {MIGRATION_TOOL_KEY} must be non-empty"
                )
    return _Schema2Summary(
        frames=frames,
        fps=fps,
        ready_joint_pos=np.array(joint_pos[0], copy=True),
        ready_root_pos=np.array(body_pos[0, 0], copy=True),
        ready_root_quat=np.array(body_quat[0, 0], copy=True),
        ready_body_pos=np.array(body_pos[0], copy=True),
        ready_body_quat=np.array(body_quat[0], copy=True),
    )


def _ready_runtime_sha256(
    joint_pos: np.ndarray, body_pos: np.ndarray, body_quat: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for label, array in (
        ("joint_pos", joint_pos),
        ("body_pos_w", body_pos),
        ("body_quat_w", body_quat),
    ):
        encoded = label.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _validate_ready_binding(
    summary: _Schema2Summary,
    ready: _ReadySummary,
    ready_fk: _ReadyFkSummary,
    label: str,
) -> str:
    for channel, actual, expected in (
        ("joint_pos", summary.ready_joint_pos, ready.joint_pos),
        ("root_pos_w", summary.ready_root_pos, ready.root_pos),
        ("root_quat_w", summary.ready_root_quat, ready.root_quat),
    ):
        if not np.array_equal(actual, expected):
            delta = np.abs(
                actual.astype(np.float64) - expected.astype(np.float64)
            )
            raise MotionRegistryError(
                f"{label} endpoint {channel} does not exactly equal the bound "
                f"canonical ready: mismatches={int(np.count_nonzero(actual != expected))} "
                f"max_abs={float(np.max(delta)):.9g}"
            )
    for channel, actual, expected in (
        ("body_pos_w", summary.ready_body_pos, ready_fk.body_pos),
        ("body_quat_w", summary.ready_body_quat, ready_fk.body_quat),
    ):
        if not np.array_equal(actual, expected):
            delta = np.abs(
                actual.astype(np.float64) - expected.astype(np.float64)
            )
            raise MotionRegistryError(
                f"{label} endpoint {channel} does not exactly equal the pinned "
                "canonical-ready FK truth: "
                f"mismatches={int(np.count_nonzero(actual != expected))} "
                f"max_abs={float(np.max(delta)):.9g}"
            )
    return _ready_runtime_sha256(
        ready.joint_pos, ready_fk.body_pos, ready_fk.body_quat
    )


def _manifest_snapshot(
    path: Path, sha256: str, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    payload = _bound_snapshot(path, sha256, label)
    return payload, _json_object(payload, label)


def _validate_build_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    npz_sha256: str,
    ready_sha256: str,
    label: str,
) -> None:
    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise MotionRegistryError(f"{label} must contain a hashes object")
    if hashes.get("output_npz_sha256") != npz_sha256:
        raise MotionRegistryError(
            f"{label} hashes.output_npz_sha256 does not bind the entry NPZ"
        )
    if hashes.get("ready_sha256") != ready_sha256:
        raise MotionRegistryError(
            f"{label} hashes.ready_sha256 does not bind canonical ready"
        )


def _parse_required_manifest(
    raw: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    repo_root: Path,
    label: str,
) -> tuple[str, Path, str, Mapping[str, Any]]:
    path_text, path = _bound_repo_file(raw[path_key], repo_root, f"{label}.{path_key}")
    sha256 = _sha256_text(raw[sha_key], f"{label}.{sha_key}")
    _, manifest = _manifest_snapshot(path, sha256, f"{label} {path_key}")
    return path_text, path, sha256, manifest


def _parse_optional_versioned_artifact(
    raw: Mapping[str, Any],
    *,
    artifact_name: str,
    repo_root: Path,
    label: str,
) -> tuple[str | None, Path | None, str | None, int | None]:
    path_key = f"{artifact_name}_path"
    sha_key = f"{artifact_name}_sha256"
    version_key = f"{artifact_name}_schema_version"
    values = (raw[path_key], raw[sha_key], raw[version_key])
    if values == (None, None, None):
        return None, None, None, None
    if any(value is None for value in values):
        raise MotionRegistryError(
            f"{label} {artifact_name} path/SHA/schema_version must be all-null "
            "or all-present"
        )
    path_text, path = _bound_repo_file(
        raw[path_key], repo_root, f"{label}.{path_key}"
    )
    sha256 = _sha256_text(raw[sha_key], f"{label}.{sha_key}")
    version = _plain_int(raw[version_key], f"{label}.{version_key}", minimum=1)
    if version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS[artifact_name]:
        raise MotionRegistryError(
            f"{label}.{version_key}={version} is unsupported; expected one of "
            f"{sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS[artifact_name])}"
        )
    _bound_snapshot(path, sha256, f"{label} {artifact_name}")
    return path_text, path, sha256, version


def _safe_artifact_npz_keys(payload: bytes, label: str) -> frozenset[str]:
    """Inspect a question-bank NPZ while allowing its intentional ``clip/key`` names."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if not infos or len(names) != len(set(names)):
                raise MotionRegistryError(
                    f"{label} has no members or duplicated members"
                )
            for item in infos:
                pure = PurePosixPath(item.filename)
                if (
                    item.is_dir()
                    or item.flag_bits & 0x1
                    or "\\" in item.filename
                    or not item.filename.endswith(".npy")
                    or pure.is_absolute()
                    or any(part in ("", ".", "..") for part in pure.parts)
                ):
                    raise MotionRegistryError(
                        f"{label} has unsafe/encrypted/non-NPY members"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise MotionRegistryError(f"{label} is not a valid NPZ/ZIP: {exc}") from exc
    return frozenset(name[:-4] for name in names)


def _validate_question_bank_artifact(
    path: Path,
    *,
    motion_id: str,
    npz_sha256: str,
    frames: int,
    strike_marker_frame: int,
    label: str,
) -> None:
    payload = _read_snapshot(path, label)
    keys = _safe_artifact_npz_keys(payload, label)
    required_arrays = (
        "contact_pos_env",
        "incoming_vel",
        "incoming_spin",
        "demanded_vel",
        "demanded_normal",
    )
    required_keys = {"meta_json"} | {
        f"{motion_id}/{name}" for name in required_arrays
    }
    missing = sorted(required_keys - keys)
    if missing:
        raise MotionRegistryError(
            f"{label} lacks schema-v3 motion rows {missing}"
        )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise MotionRegistryError(f"cannot load {label}: {exc}") from exc
    with loaded as data:
        meta_bytes = np.asarray(data["meta_json"])
        if meta_bytes.dtype != np.dtype(np.uint8) or meta_bytes.ndim != 1:
            raise MotionRegistryError(
                f"{label} meta_json must be one uint8 byte vector"
            )
        meta = _json_object(bytes(meta_bytes.tolist()), f"{label} meta_json")
        if meta.get("schema_version") != 3 or meta.get("split") != "train":
            raise MotionRegistryError(
                f"{label} must be a schema-v3 train question bank"
            )
        if motion_id not in tuple(meta.get("clip_order") or ()):
            raise MotionRegistryError(
                f"{label} clip_order does not contain {motion_id!r}"
            )
        clips = meta.get("clips")
        info = clips.get(motion_id) if isinstance(clips, Mapping) else None
        if not isinstance(info, Mapping):
            raise MotionRegistryError(
                f"{label} metadata lacks clip contract for {motion_id!r}"
            )
        allowed = info.get("motion_sha256_allowed")
        if allowed is None:
            allowed = [info.get("motion_sha256")]
        if (
            not isinstance(allowed, list)
            or npz_sha256 not in allowed
            or any(_SHA256.fullmatch(value or "") is None for value in allowed)
        ):
            raise MotionRegistryError(
                f"{label} does not authorize this motion NPZ SHA-256"
            )
        if (
            info.get("n_frames") != frames
            or info.get("anchor_frame") != strike_marker_frame
        ):
            raise MotionRegistryError(
                f"{label} frame/strike contract differs from the registry row"
            )
        contact = np.asarray(data[f"{motion_id}/contact_pos_env"])
        if (
            contact.dtype != np.dtype(np.float32)
            or contact.shape != (3,)
            or not np.isfinite(contact).all()
        ):
            raise MotionRegistryError(
                f"{label} contact_pos_env must be finite float32 shape (3,)"
            )
        question_count: int | None = None
        for name in required_arrays[1:]:
            array = np.asarray(data[f"{motion_id}/{name}"])
            if (
                array.dtype != np.dtype(np.float32)
                or array.ndim != 2
                or array.shape[0] < 1
                or array.shape[1] != 3
                or not np.isfinite(array).all()
            ):
                raise MotionRegistryError(
                    f"{label} {name} must be finite float32 shape (Q,3)"
                )
            if question_count is None:
                question_count = int(array.shape[0])
            elif int(array.shape[0]) != question_count:
                raise MotionRegistryError(
                    f"{label} question arrays disagree on Q"
                )


def _validate_training_config_artifact(
    path: Path,
    *,
    motion_id: str,
    scope: str,
    variant: str,
    npz_sha256: str,
    label: str,
) -> None:
    document = _exact_keys(
        _json_object(_read_snapshot(path, label), label),
        _TRAINING_CONFIG_KEYS,
        label,
    )
    expected = {
        "schema_version": 1,
        "contract": "canonical-motion-training-config-v1",
        "motion_id": motion_id,
        "scope": scope,
        "variant": variant,
        "npz_sha256": npz_sha256,
    }
    if document != expected:
        raise MotionRegistryError(
            f"{label} does not exactly bind the registry row"
        )


def _validate_onnx_model_artifact(path: Path, *, label: str) -> None:
    """Require the authoritative ONNX parser/checker before deployment adoption."""

    payload = _read_snapshot(path, label)
    try:
        import onnx  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as exc:
        raise MotionRegistryError(
            f"{label} cannot be authorized because the strict ONNX parser is "
            "unavailable"
        ) from exc
    loader = getattr(onnx, "load_model_from_string", None)
    checker = getattr(getattr(onnx, "checker", None), "check_model", None)
    if not callable(loader) or not callable(checker):
        raise MotionRegistryError(
            f"{label} cannot be authorized because onnx.load_model_from_string/"
            "onnx.checker.check_model are unavailable"
        )
    try:
        model = loader(payload)
        checker(model, full_check=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise MotionRegistryError(
            f"{label} fails strict ONNX parse/check: {exc}"
        ) from exc
    graph = getattr(model, "graph", None)
    if (
        graph is None
        or len(getattr(graph, "input", ())) < 1
        or len(getattr(graph, "output", ())) < 1
    ):
        raise MotionRegistryError(
            f"{label} must have at least one graph input and output"
        )


def _validate_onnx_metadata_artifact(
    path: Path,
    *,
    motion_id: str,
    scope: str,
    variant: str,
    npz_sha256: str,
    strike_marker_frame: int,
    contact_opportunity_frames: tuple[int, int],
    mount_normal_sign: float,
    canonical_ready_sha256: str,
    canonical_ready_fk_sha256: str,
    question_bank_sha256: str,
    training_config_sha256: str,
    onnx_model_sha256: str,
    label: str,
) -> None:
    document = _exact_keys(
        _json_object(_read_snapshot(path, label), label),
        _ONNX_METADATA_KEYS,
        label,
    )
    expected = {
        "hope_metadata_schema_version": 2,
        "motion_id": motion_id,
        "scope": scope,
        "variant": variant,
        "npz_sha256": npz_sha256,
        "strike_marker_frame": strike_marker_frame,
        "contact_opportunity_frames": list(contact_opportunity_frames),
        "mount_normal_sign": mount_normal_sign,
        "canonical_ready_sha256": canonical_ready_sha256,
        "canonical_ready_fk_sha256": canonical_ready_fk_sha256,
        "question_bank_sha256": question_bank_sha256,
        "training_config_sha256": training_config_sha256,
        "onnx_model_sha256": onnx_model_sha256,
    }
    if document != expected:
        raise MotionRegistryError(
            f"{label} does not exactly bind runtime identity/timing/artifacts"
        )


def _parse_and_validate_adoption_manifest(
    raw: Mapping[str, Any],
    *,
    motion_id: str,
    scope: str,
    variant: str,
    npz_sha256: str,
    strike_marker_frame: int,
    contact_opportunity_frames: tuple[int, int],
    mount_normal_sign: float,
    canonical_ready_sha256: str,
    canonical_ready_fk_sha256: str,
    artifacts: Mapping[str, tuple[str | None, int | None]],
    repo_root: Path,
    label: str,
) -> tuple[str | None, Path | None, str | None]:
    values = (raw["adoption_manifest_path"], raw["adoption_manifest_sha256"])
    if values == (None, None):
        return None, None, None
    if any(value is None for value in values):
        raise MotionRegistryError(
            f"{label} adoption manifest path/SHA must be both-null or both-present"
        )
    path_text, path = _bound_repo_file(
        raw["adoption_manifest_path"],
        repo_root,
        f"{label}.adoption_manifest_path",
    )
    sha256 = _sha256_text(
        raw["adoption_manifest_sha256"],
        f"{label}.adoption_manifest_sha256",
    )
    _, manifest = _manifest_snapshot(
        path, sha256, f"{label} adoption manifest"
    )
    manifest = _exact_keys(
        manifest, _ADOPTION_MANIFEST_KEYS, f"{label} adoption manifest"
    )
    expected_scalars = {
        "schema_version": 1,
        "motion_id": motion_id,
        "scope": scope,
        "variant": variant,
        "npz_sha256": npz_sha256,
        "strike_marker_frame": strike_marker_frame,
        "contact_opportunity_frames": list(contact_opportunity_frames),
        "mount_normal_sign": mount_normal_sign,
        "canonical_ready_sha256": canonical_ready_sha256,
        "canonical_ready_fk_sha256": canonical_ready_fk_sha256,
    }
    mismatched = tuple(
        key for key, expected in expected_scalars.items() if manifest[key] != expected
    )
    if mismatched:
        raise MotionRegistryError(
            f"{label} adoption manifest does not bind row identity/timing/ready: "
            f"{list(mismatched)}"
        )
    artifact_manifest = _exact_keys(
        manifest["artifacts"],
        frozenset(_ADOPTION_ARTIFACT_NAMES),
        f"{label} adoption manifest artifacts",
    )
    for artifact_name, (artifact_sha, artifact_version) in artifacts.items():
        actual = artifact_manifest[artifact_name]
        if artifact_sha is None:
            if actual is not None:
                raise MotionRegistryError(
                    f"{label} adoption manifest claims absent {artifact_name}"
                )
            continue
        binding = _exact_keys(
            actual,
            _ADOPTION_ARTIFACT_BINDING_KEYS,
            f"{label} adoption manifest {artifact_name}",
        )
        if (
            binding["sha256"] != artifact_sha
            or binding["schema_version"] != artifact_version
        ):
            raise MotionRegistryError(
                f"{label} adoption manifest does not bind {artifact_name} "
                "bytes/schema version"
            )
    return path_text, path, sha256


def _validate_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    motion_id: str,
    scope: str,
    variant: str,
    npz_sha256: str,
    evidence_level: str,
    repo_root: Path,
    label: str,
) -> None:
    manifest = _exact_keys(manifest, _EVIDENCE_MANIFEST_KEYS, label)
    expected_identity = {
        "schema_version": 1,
        "motion_id": motion_id,
        "scope": scope,
        "variant": variant,
        "npz_sha256": npz_sha256,
        "highest_evidence_level": evidence_level,
    }
    mismatched = tuple(
        key for key, expected in expected_identity.items() if manifest[key] != expected
    )
    if mismatched:
        raise MotionRegistryError(
            f"{label} does not bind exact motion lineage/level: {list(mismatched)}"
        )
    claimed_index = EVIDENCE_LEVELS.index(evidence_level)
    required_levels = EVIDENCE_LEVELS[1 : claimed_index + 1]
    certificates = manifest["certificates"]
    if not isinstance(certificates, list) or len(certificates) != len(
        required_levels
    ):
        raise MotionRegistryError(
            f"{label} must contain one passing certificate for every level "
            f"{list(required_levels)}"
        )
    for index, (certificate_raw, expected_level) in enumerate(
        zip(certificates, required_levels)
    ):
        ref_label = f"{label}.certificates[{index}]"
        ref = _exact_keys(
            certificate_raw, _EVIDENCE_CERTIFICATE_REF_KEYS, ref_label
        )
        if ref["level"] != expected_level or ref["status"] != "pass":
            raise MotionRegistryError(
                f"{ref_label} must be a passing {expected_level} certificate"
            )
        path_text, path = _bound_repo_file(
            ref["path"], repo_root, f"{ref_label}.path"
        )
        certificate_sha = _sha256_text(ref["sha256"], f"{ref_label}.sha256")
        _, certificate = _manifest_snapshot(
            path, certificate_sha, f"{ref_label} certificate"
        )
        certificate = _exact_keys(
            certificate, _EVIDENCE_CERTIFICATE_KEYS, f"{ref_label} certificate"
        )
        expected_certificate = {
            "schema_version": 1,
            "level": expected_level,
            "motion_id": motion_id,
            "scope": scope,
            "variant": variant,
            "npz_sha256": npz_sha256,
            "status": "pass",
        }
        if certificate != expected_certificate:
            raise MotionRegistryError(
                f"{ref_label} certificate does not bind exact motion lineage"
            )
        if path_text != ref["path"]:
            raise MotionRegistryError(f"{ref_label}.path normalization drift")


def _parse_entry(
    raw_value: Any,
    *,
    index: int,
    expected_motion_id: str,
    order_label: str,
    bank_scope: str,
    bank_ready_sha256: str,
    bank_ready: _ReadySummary,
    bank_ready_fk: _ReadyFkSummary,
    bank_ready_fk_sha256: str,
    repo_root: Path,
) -> MotionRegistryEntry:
    label = f"entries[{index}]"
    raw = _exact_keys(raw_value, _ENTRY_KEYS, label)
    motion_id = _slug(raw["motion_id"], f"{label}.motion_id")
    if motion_id != expected_motion_id:
        raise MotionRegistryError(
            f"{label}.motion_id must preserve {order_label}: "
            f"expected {expected_motion_id!r}, got {motion_id!r}"
        )
    scope = raw["scope"]
    if scope != bank_scope:
        raise MotionRegistryError(
            f"{label}.scope {scope!r} differs from bank scope {bank_scope!r}"
        )
    variant = _slug(raw["variant"], f"{label}.variant")

    npz_path_text, npz_path = _bound_repo_file(
        raw["npz_path"], repo_root, f"{label}.npz_path"
    )
    npz_sha256 = _sha256_text(raw["npz_sha256"], f"{label}.npz_sha256")
    npz_payload = _bound_snapshot(npz_path, npz_sha256, f"{label} NPZ")
    summary = _validate_schema2_snapshot(npz_payload, f"{label} NPZ")
    ready_runtime_sha256 = _validate_ready_binding(
        summary, bank_ready, bank_ready_fk, f"{label} NPZ"
    )

    frames = _plain_int(raw["frames"], f"{label}.frames", minimum=2)
    if frames != summary.frames:
        raise MotionRegistryError(
            f"{label}.frames={frames} differs from NPZ frames={summary.frames}"
        )
    fps = _positive_number(raw["fps"], f"{label}.fps")
    if not math.isclose(fps, summary.fps, rel_tol=0.0, abs_tol=1.0e-12):
        raise MotionRegistryError(
            f"{label}.fps={fps:.17g} differs from NPZ fps={summary.fps:.17g}"
        )

    family = raw["family"]
    if family not in FAMILIES:
        raise MotionRegistryError(
            f"{label}.family must be one of {FAMILIES}, got {family!r}"
        )
    marker = _plain_int(
        raw["strike_marker_frame"],
        f"{label}.strike_marker_frame",
    )
    opportunity = raw["contact_opportunity_frames"]
    if (
        not isinstance(opportunity, list)
        or len(opportunity) != 2
        or any(type(value) is not int for value in opportunity)
    ):
        raise MotionRegistryError(
            f"{label}.contact_opportunity_frames must be two integer inclusive frames"
        )
    opportunity_pair = (int(opportunity[0]), int(opportunity[1]))
    if not (
        0
        <= opportunity_pair[0]
        <= marker
        <= opportunity_pair[1]
        < frames
    ):
        raise MotionRegistryError(
            f"{label} requires 0 <= contact_start <= strike_marker <= "
            "contact_end < frames"
        )

    sign_value = raw["mount_normal_sign"]
    if type(sign_value) is not float or sign_value not in (-1.0, 1.0):
        raise MotionRegistryError(
            f"{label}.mount_normal_sign must be an explicit JSON float +1.0 or -1.0"
        )
    ready_sha256 = _sha256_text(
        raw["canonical_ready_sha256"],
        f"{label}.canonical_ready_sha256",
    )
    if ready_sha256 != bank_ready_sha256:
        raise MotionRegistryError(
            f"{label}.canonical_ready_sha256 differs from the bank common ready"
        )

    source_path_text, source_path = _bound_repo_file(
        raw["source_manifest_path"],
        repo_root,
        f"{label}.source_manifest_path",
    )
    source_sha256 = _sha256_text(
        raw["source_manifest_sha256"],
        f"{label}.source_manifest_sha256",
    )
    _manifest_snapshot(
        source_path,
        source_sha256,
        f"{label} source manifest",
    )

    build_path_text, build_path = _bound_repo_file(
        raw["build_manifest_path"],
        repo_root,
        f"{label}.build_manifest_path",
    )
    build_sha256 = _sha256_text(
        raw["build_manifest_sha256"],
        f"{label}.build_manifest_sha256",
    )
    _, build_manifest = _manifest_snapshot(
        build_path,
        build_sha256,
        f"{label} build manifest",
    )
    _validate_build_manifest_binding(
        build_manifest,
        npz_sha256=npz_sha256,
        ready_sha256=ready_sha256,
        label=f"{label} build manifest",
    )

    (
        applicability_path_text,
        applicability_path,
        applicability_sha256,
        applicability_manifest,
    ) = _parse_required_manifest(
        raw,
        path_key="applicability_manifest_path",
        sha_key="applicability_manifest_sha256",
        repo_root=repo_root,
        label=label,
    )
    applicability_manifest = _exact_keys(
        applicability_manifest,
        _APPLICABILITY_MANIFEST_KEYS,
        f"{label} applicability manifest",
    )
    if (
        applicability_manifest.get("schema_version") != 1
        or
        applicability_manifest.get("motion_id") != motion_id
        or applicability_manifest.get("scope") != scope
        or applicability_manifest.get("variant") != variant
        or applicability_manifest.get("npz_sha256") != npz_sha256
        or not isinstance(applicability_manifest.get("domain"), str)
        or not applicability_manifest["domain"]
    ):
        raise MotionRegistryError(
            f"{label} applicability manifest must bind motion_id/scope/variant/"
            "NPZ and one non-empty domain"
        )
    evidence_level = raw["evidence_level"]
    if evidence_level not in EVIDENCE_LEVELS:
        raise MotionRegistryError(
            f"{label}.evidence_level must be one of {EVIDENCE_LEVELS}, "
            f"got {evidence_level!r}"
        )
    (
        evidence_path_text,
        evidence_path,
        evidence_sha256,
        evidence_manifest,
    ) = _parse_required_manifest(
        raw,
        path_key="evidence_manifest_path",
        sha_key="evidence_manifest_sha256",
        repo_root=repo_root,
        label=label,
    )
    _validate_evidence_manifest(
        evidence_manifest,
        motion_id=motion_id,
        scope=str(scope),
        variant=variant,
        npz_sha256=npz_sha256,
        evidence_level=str(evidence_level),
        repo_root=repo_root,
        label=f"{label} evidence manifest",
    )
    (
        question_bank_path_text,
        question_bank_path,
        question_bank_sha256,
        question_bank_schema_version,
    ) = _parse_optional_versioned_artifact(
        raw,
        artifact_name="question_bank",
        repo_root=repo_root,
        label=label,
    )
    (
        training_config_path_text,
        training_config_path,
        training_config_sha256,
        training_config_schema_version,
    ) = _parse_optional_versioned_artifact(
        raw,
        artifact_name="training_config",
        repo_root=repo_root,
        label=label,
    )
    (
        onnx_model_path_text,
        onnx_model_path,
        onnx_model_sha256,
        onnx_model_schema_version,
    ) = _parse_optional_versioned_artifact(
        raw,
        artifact_name="onnx_model",
        repo_root=repo_root,
        label=label,
    )
    (
        onnx_metadata_path_text,
        onnx_metadata_path,
        onnx_metadata_sha256,
        onnx_metadata_schema_version,
    ) = _parse_optional_versioned_artifact(
        raw,
        artifact_name="onnx_metadata",
        repo_root=repo_root,
        label=label,
    )
    if question_bank_path is not None:
        _validate_question_bank_artifact(
            question_bank_path,
            motion_id=motion_id,
            npz_sha256=npz_sha256,
            frames=frames,
            strike_marker_frame=marker,
            label=f"{label} question_bank",
        )
    if training_config_path is not None:
        _validate_training_config_artifact(
            training_config_path,
            motion_id=motion_id,
            scope=str(scope),
            variant=variant,
            npz_sha256=npz_sha256,
            label=f"{label} training_config",
        )
    if onnx_metadata_path is not None:
        if (
            question_bank_sha256 is None
            or training_config_sha256 is None
            or onnx_model_sha256 is None
        ):
            raise MotionRegistryError(
                f"{label} ONNX metadata requires a bound question bank, "
                "training config, and ONNX model"
            )
        _validate_onnx_metadata_artifact(
            onnx_metadata_path,
            motion_id=motion_id,
            scope=str(scope),
            variant=variant,
            npz_sha256=npz_sha256,
            strike_marker_frame=marker,
            contact_opportunity_frames=opportunity_pair,
            mount_normal_sign=float(sign_value),
            canonical_ready_sha256=ready_sha256,
            canonical_ready_fk_sha256=bank_ready_fk_sha256,
            question_bank_sha256=question_bank_sha256,
            training_config_sha256=training_config_sha256,
            onnx_model_sha256=onnx_model_sha256,
            label=f"{label} onnx_metadata",
        )
    if onnx_model_path is not None and onnx_metadata_path is None:
        raise MotionRegistryError(
            f"{label} ONNX model requires bound ONNX metadata"
        )
    (
        adoption_manifest_path_text,
        adoption_manifest_path,
        adoption_manifest_sha256,
    ) = _parse_and_validate_adoption_manifest(
        raw,
        motion_id=motion_id,
        scope=str(scope),
        variant=variant,
        npz_sha256=npz_sha256,
        strike_marker_frame=marker,
        contact_opportunity_frames=opportunity_pair,
        mount_normal_sign=float(sign_value),
        canonical_ready_sha256=ready_sha256,
        canonical_ready_fk_sha256=bank_ready_fk_sha256,
        artifacts={
            "question_bank": (
                question_bank_sha256,
                question_bank_schema_version,
            ),
            "training_config": (
                training_config_sha256,
                training_config_schema_version,
            ),
            "onnx_model": (
                onnx_model_sha256,
                onnx_model_schema_version,
            ),
            "onnx_metadata": (
                onnx_metadata_sha256,
                onnx_metadata_schema_version,
            ),
        },
        repo_root=repo_root,
        label=label,
    )

    publication_class = _slug(
        raw["publication_class"], f"{label}.publication_class"
    )
    if publication_class not in PUBLICATION_AUTHORIZATION:
        raise MotionRegistryError(
            f"{label}.publication_class must be one of "
            f"{tuple(PUBLICATION_AUTHORIZATION)}, got {publication_class!r}"
        )
    flags: dict[str, bool] = {}
    for key in (
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        value = raw[key]
        if type(value) is not bool:
            raise MotionRegistryError(f"{label}.{key} must be a JSON boolean")
        flags[key] = value
    if flags["hardware_authorized"] and not flags["deployment_authorized"]:
        raise MotionRegistryError(
            f"{label} hardware authorization requires deployment authorization"
        )
    if flags["deployment_authorized"] and not flags["training_authorized"]:
        raise MotionRegistryError(
            f"{label} deployment authorization requires training authorization"
        )
    actual_authorization = (
        flags["training_authorized"],
        flags["deployment_authorized"],
        flags["hardware_authorized"],
    )
    expected_authorization = PUBLICATION_AUTHORIZATION[publication_class]
    if actual_authorization != expected_authorization:
        raise MotionRegistryError(
            f"{label} publication_class={publication_class!r} requires authorization "
            f"flags={expected_authorization}, got={actual_authorization}"
        )
    evidence_index = EVIDENCE_LEVELS.index(str(evidence_level))
    minimum_evidence = PUBLICATION_MIN_EVIDENCE[publication_class]
    if evidence_index < EVIDENCE_LEVELS.index(minimum_evidence):
        raise MotionRegistryError(
            f"{label} publication_class={publication_class!r} requires evidence "
            f">={minimum_evidence}, got={evidence_level}"
        )
    artifact_presence = {
        "question_bank": question_bank_path is not None,
        "training_config": training_config_path is not None,
        "onnx_model": onnx_model_path is not None,
        "onnx_metadata": onnx_metadata_path is not None,
        "adoption_manifest": adoption_manifest_path is not None,
    }
    missing_artifacts = sorted(
        name
        for name in PUBLICATION_REQUIRED_ARTIFACTS[publication_class]
        if not artifact_presence[name]
    )
    if missing_artifacts:
        raise MotionRegistryError(
            f"{label} publication_class={publication_class!r} lacks required "
            f"versioned artifacts {missing_artifacts}"
        )
    if publication_class in ("deployment_adopted", "hardware_adopted"):
        assert onnx_model_path is not None
        _validate_onnx_model_artifact(
            onnx_model_path,
            label=f"{label} onnx_model",
        )

    return MotionRegistryEntry(
        motion_id=motion_id,
        scope=scope,
        variant=variant,
        npz_path_text=npz_path_text,
        npz_path=npz_path,
        npz_sha256=npz_sha256,
        frames=frames,
        fps=fps,
        family=str(family),
        strike_marker_frame=marker,
        contact_opportunity_frames=opportunity_pair,
        mount_normal_sign=float(sign_value),
        canonical_ready_sha256=ready_sha256,
        source_manifest_path_text=source_path_text,
        source_manifest_path=source_path,
        source_manifest_sha256=source_sha256,
        build_manifest_path_text=build_path_text,
        build_manifest_path=build_path,
        build_manifest_sha256=build_sha256,
        applicability_manifest_path_text=applicability_path_text,
        applicability_manifest_path=applicability_path,
        applicability_manifest_sha256=applicability_sha256,
        evidence_level=str(evidence_level),
        evidence_manifest_path_text=evidence_path_text,
        evidence_manifest_path=evidence_path,
        evidence_manifest_sha256=evidence_sha256,
        question_bank_path_text=question_bank_path_text,
        question_bank_path=question_bank_path,
        question_bank_sha256=question_bank_sha256,
        question_bank_schema_version=question_bank_schema_version,
        training_config_path_text=training_config_path_text,
        training_config_path=training_config_path,
        training_config_sha256=training_config_sha256,
        training_config_schema_version=training_config_schema_version,
        onnx_model_path_text=onnx_model_path_text,
        onnx_model_path=onnx_model_path,
        onnx_model_sha256=onnx_model_sha256,
        onnx_model_schema_version=onnx_model_schema_version,
        onnx_metadata_path_text=onnx_metadata_path_text,
        onnx_metadata_path=onnx_metadata_path,
        onnx_metadata_sha256=onnx_metadata_sha256,
        onnx_metadata_schema_version=onnx_metadata_schema_version,
        adoption_manifest_path_text=adoption_manifest_path_text,
        adoption_manifest_path=adoption_manifest_path,
        adoption_manifest_sha256=adoption_manifest_sha256,
        publication_class=publication_class,
        training_authorized=flags["training_authorized"],
        deployment_authorized=flags["deployment_authorized"],
        hardware_authorized=flags["hardware_authorized"],
        ready_runtime_sha256=ready_runtime_sha256,
    )


def load_canonical_motion_bank_registry(
    path: os.PathLike[str] | str,
    *,
    repo_root: os.PathLike[str] | str | None = None,
    expected_registry_sha256: str | None = None,
) -> CanonicalMotionBankRegistry:
    """Load one exact content-addressed bank registry."""

    root = _root_path(repo_root)
    registry_path = _registry_path(path, root)
    payload = _read_snapshot(registry_path, "registry")
    registry_sha256 = _sha256_bytes(payload)
    if expected_registry_sha256 is not None:
        expected = _sha256_text(
            expected_registry_sha256, "expected_registry_sha256"
        )
        if registry_sha256 != expected:
            raise MotionRegistryError(
                "registry SHA-256 mismatch: "
                f"expected={expected}, actual={registry_sha256}"
            )
    document = _json_object(payload, "registry")
    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version not in (
        REGISTRY_SCHEMA_VERSION,
        GENERIC_REGISTRY_SCHEMA_VERSION,
    ):
        raise MotionRegistryError(
            "registry.schema_version must be exact integer "
            f"{REGISTRY_SCHEMA_VERSION} or {GENERIC_REGISTRY_SCHEMA_VERSION}"
        )
    raw = _exact_keys(
        document,
        (
            _TOP_LEVEL_KEYS
            if schema_version == REGISTRY_SCHEMA_VERSION
            else _TOP_LEVEL_KEYS_V2
        ),
        "registry",
    )
    if schema_version == REGISTRY_SCHEMA_VERSION:
        motion_ids = CANONICAL_MOTION_IDS
        order_label = "canonical order"
    else:
        raw_motion_ids = raw["motion_ids"]
        if not isinstance(raw_motion_ids, list) or not raw_motion_ids:
            raise MotionRegistryError(
                "registry.motion_ids must be a non-empty ordered array"
            )
        motion_ids = tuple(
            _slug(value, f"registry.motion_ids[{index}]")
            for index, value in enumerate(raw_motion_ids)
        )
        if len(set(motion_ids)) != len(motion_ids):
            raise MotionRegistryError(
                "registry.motion_ids must contain unique values"
            )
        order_label = "ordered motion_ids"
    bank_id = _slug(raw["bank_id"], "registry.bank_id")
    scope = raw["scope"]
    if scope not in SCOPES:
        raise MotionRegistryError(
            f"registry.scope must be one of {SCOPES}, got {scope!r}"
        )
    ready_sha256 = _sha256_text(
        raw["canonical_ready_sha256"],
        "registry.canonical_ready_sha256",
    )
    ready_path_text, ready_path = _bound_repo_file(
        raw["canonical_ready_path"],
        root,
        "registry.canonical_ready_path",
    )
    ready_payload = _bound_snapshot(
        ready_path,
        ready_sha256,
        "registry canonical ready",
    )
    ready = _validate_ready_snapshot(ready_payload, "registry canonical ready")
    ready_fk_sha256 = _sha256_text(
        raw["canonical_ready_fk_sha256"],
        "registry.canonical_ready_fk_sha256",
    )
    ready_fk_path_text, ready_fk_path = _bound_repo_file(
        raw["canonical_ready_fk_path"],
        root,
        "registry.canonical_ready_fk_path",
    )
    ready_fk_payload = _bound_snapshot(
        ready_fk_path,
        ready_fk_sha256,
        "registry canonical ready FK truth",
    )
    ready_fk = _validate_ready_fk_snapshot(
        ready_fk_payload,
        canonical_ready_sha256=ready_sha256,
        canonical_ready=ready,
        label="registry canonical ready FK truth",
    )
    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != len(
        motion_ids
    ):
        raise MotionRegistryError(
            f"registry.entries must contain exactly {len(motion_ids)} rows"
        )
    entries = tuple(
        _parse_entry(
            value,
            index=index,
            expected_motion_id=motion_ids[index],
            order_label=order_label,
            bank_scope=str(scope),
            bank_ready_sha256=ready_sha256,
            bank_ready=ready,
            bank_ready_fk=ready_fk,
            bank_ready_fk_sha256=ready_fk_sha256,
            repo_root=root,
        )
        for index, value in enumerate(raw_entries)
    )
    if tuple(row.motion_id for row in entries) != motion_ids:
        raise MotionRegistryError(
            "registry entries differ from the ordered motion_ids"
        )
    if len({row.motion_id for row in entries}) != len(motion_ids):
        raise MotionRegistryError(
            "registry must contain every motion ID exactly once"
        )
    if len({row.canonical_ready_sha256 for row in entries}) != 1:
        raise MotionRegistryError("registry entries do not share one canonical ready")
    if len({row.ready_runtime_sha256 for row in entries}) != 1:
        raise MotionRegistryError(
            "registry NPZ endpoints disagree on the full runtime ready body pose"
        )

    return CanonicalMotionBankRegistry(
        schema_version=int(schema_version),
        path=registry_path,
        repo_root=root,
        registry_sha256=registry_sha256,
        bank_id=bank_id,
        scope=str(scope),
        canonical_ready_path_text=ready_path_text,
        canonical_ready_path=ready_path,
        canonical_ready_sha256=ready_sha256,
        canonical_ready_fk_path_text=ready_fk_path_text,
        canonical_ready_fk_path=ready_fk_path,
        canonical_ready_fk_sha256=ready_fk_sha256,
        registry_digest_pinned=expected_registry_sha256 is not None,
        motion_ids=motion_ids,
        entries=entries,
    )


def _alignment_sha256(registry: CanonicalMotionBankRegistry) -> str:
    rows = [
        {
            "motion_id": row.motion_id,
            "scope": row.scope,
            "variant": row.variant,
            "npz_path": row.npz_path_text,
            "npz_sha256": row.npz_sha256,
            "frames": row.frames,
            "fps": format(row.fps, ".17g"),
            "family": row.family,
            "strike_marker_frame": row.strike_marker_frame,
            "strike_phase_rational": f"{row.strike_marker_frame}/{row.frames - 1}",
            "contact_opportunity_frames": list(row.contact_opportunity_frames),
            "mount_normal_sign": format(row.mount_normal_sign, ".1f"),
            "canonical_ready_sha256": row.canonical_ready_sha256,
            "ready_runtime_sha256": row.ready_runtime_sha256,
            "source_manifest_path": row.source_manifest_path_text,
            "source_manifest_sha256": row.source_manifest_sha256,
            "build_manifest_path": row.build_manifest_path_text,
            "build_manifest_sha256": row.build_manifest_sha256,
            "applicability_manifest_path": row.applicability_manifest_path_text,
            "applicability_manifest_sha256": (
                row.applicability_manifest_sha256
            ),
            "evidence_level": row.evidence_level,
            "evidence_manifest_path": row.evidence_manifest_path_text,
            "evidence_manifest_sha256": row.evidence_manifest_sha256,
            "question_bank_path": row.question_bank_path_text,
            "question_bank_sha256": row.question_bank_sha256,
            "question_bank_schema_version": row.question_bank_schema_version,
            "training_config_path": row.training_config_path_text,
            "training_config_sha256": row.training_config_sha256,
            "training_config_schema_version": row.training_config_schema_version,
            "onnx_model_path": row.onnx_model_path_text,
            "onnx_model_sha256": row.onnx_model_sha256,
            "onnx_model_schema_version": row.onnx_model_schema_version,
            "onnx_metadata_path": row.onnx_metadata_path_text,
            "onnx_metadata_sha256": row.onnx_metadata_sha256,
            "onnx_metadata_schema_version": row.onnx_metadata_schema_version,
            "adoption_manifest_path": row.adoption_manifest_path_text,
            "adoption_manifest_sha256": row.adoption_manifest_sha256,
            "publication_class": row.publication_class,
            "training_authorized": row.training_authorized,
            "deployment_authorized": row.deployment_authorized,
            "hardware_authorized": row.hardware_authorized,
        }
        for row in registry.entries
    ]
    document = {
        "schema_version": registry.schema_version,
        "registry_sha256": registry.registry_sha256,
        "bank_id": registry.bank_id,
        "scope": registry.scope,
        "canonical_ready_path": registry.canonical_ready_path_text,
        "canonical_ready_sha256": registry.canonical_ready_sha256,
        "canonical_ready_fk_path": registry.canonical_ready_fk_path_text,
        "canonical_ready_fk_sha256": registry.canonical_ready_fk_sha256,
        "rows": rows,
    }
    if registry.schema_version == GENERIC_REGISTRY_SCHEMA_VERSION:
        document["motion_ids"] = list(registry.motion_ids)
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _evidence_certificate_sha256(
    entry: MotionRegistryEntry,
) -> tuple[str, ...]:
    """Re-read one already-bound evidence bundle and return its receipt digests."""

    _, manifest = _manifest_snapshot(
        entry.evidence_manifest_path,
        entry.evidence_manifest_sha256,
        f"{entry.motion_id} evidence manifest",
    )
    certificates = manifest.get("certificates")
    if not isinstance(certificates, list):
        raise MotionRegistryError(
            f"{entry.motion_id} evidence manifest certificates changed"
        )
    result: list[str] = []
    for index, raw in enumerate(certificates):
        if not isinstance(raw, Mapping):
            raise MotionRegistryError(
                f"{entry.motion_id} evidence certificate ref[{index}] changed"
            )
        result.append(
            _sha256_text(
                raw.get("sha256"),
                f"{entry.motion_id} evidence certificate ref[{index}].sha256",
            )
        )
    return tuple(result)


def bank_promotion_binding(
    registry: CanonicalMotionBankRegistry,
    *,
    authorization_purpose: str,
) -> Any:
    """Derive the exact values an external code-rooted certificate must bind."""

    if authorization_purpose not in AUTHORIZATION_PURPOSES:
        raise MotionRegistryError(
            f"authorization_purpose must be one of {AUTHORIZATION_PURPOSES}"
        )
    if not registry.registry_digest_pinned:
        raise MotionRegistryError(
            "bank promotion binding requires expected_registry_sha256"
        )
    binding_type = (
        motion_admission.BankPromotionBinding
        if registry.schema_version == REGISTRY_SCHEMA_VERSION
        else motion_admission.GenericBankPromotionBinding
    )
    return binding_type(
        purpose=authorization_purpose,
        bank_id=registry.bank_id,
        scope=registry.scope,
        registry_sha256=registry.registry_sha256,
        alignment_sha256=_alignment_sha256(registry),
        motion_ids=tuple(row.motion_id for row in registry.entries),
        npz_sha256=tuple(row.npz_sha256 for row in registry.entries),
        canonical_ready_sha256=registry.canonical_ready_sha256,
        canonical_ready_fk_sha256=registry.canonical_ready_fk_sha256,
        build_manifest_sha256=tuple(
            row.build_manifest_sha256 for row in registry.entries
        ),
        evidence_levels=tuple(row.evidence_level for row in registry.entries),
        evidence_manifest_sha256=tuple(
            row.evidence_manifest_sha256 for row in registry.entries
        ),
        evidence_certificate_sha256=tuple(
            _evidence_certificate_sha256(row) for row in registry.entries
        ),
        question_bank_sha256=tuple(
            row.question_bank_sha256 for row in registry.entries
        ),
        training_config_sha256=tuple(
            row.training_config_sha256 for row in registry.entries
        ),
        onnx_model_sha256=tuple(
            row.onnx_model_sha256 for row in registry.entries
        ),
        onnx_metadata_sha256=tuple(
            row.onnx_metadata_sha256 for row in registry.entries
        ),
        adoption_manifest_sha256=tuple(
            row.adoption_manifest_sha256 for row in registry.entries
        ),
    )


def verify_registry_promotion_certificate(
    registry: CanonicalMotionBankRegistry,
    certificate_path: os.PathLike[str] | str,
    *,
    authorization_purpose: str,
) -> motion_admission.TrustedMotionAdmission:
    """Mint an opaque capability from one code-trusted promotion certificate."""

    binding = bank_promotion_binding(
        registry, authorization_purpose=authorization_purpose
    )
    try:
        return motion_admission.verify_bank_promotion_certificate(
            certificate_path,
            binding=binding,
            repo_root=registry.repo_root,
        )
    except motion_admission.MotionAdmissionError as exc:
        raise MotionRegistryError(f"motion admission rejected: {exc}") from exc


def adapt_registry_for_runtime(
    registry: CanonicalMotionBankRegistry,
    *,
    expected_alignment_sha256: str | None = None,
    expected_canonical_ready_sha256: str | None = None,
    expected_canonical_ready_fk_sha256: str | None = None,
    authorization_purpose: str | None = "training",
    admission: motion_admission.TrustedMotionAdmission | None = None,
) -> CanonicalRuntimeTables:
    """Atomically derive and cross-check all runtime columns from one registry.

    ``authorization_purpose=None`` is identity-only audit mode.  Every real
    runtime purpose additionally requires an opaque capability minted from a
    promotion-certificate digest present in the code-owned trust set.  Registry
    booleans, E-level labels, adoption manifests, and caller-provided hashes are
    necessary provenance but never mint that capability.
    """

    if not isinstance(registry, CanonicalMotionBankRegistry):
        raise MotionRegistryError(
            "registry must be a CanonicalMotionBankRegistry from the strict loader"
        )
    # Frozen dataclasses are value containers, not capabilities: callers can
    # construct or ``dataclasses.replace`` them.  Re-read the pinned bytes and
    # demand exact derived-object equality before any runtime authorization.
    reloaded = load_canonical_motion_bank_registry(
        registry.path,
        repo_root=registry.repo_root,
        expected_registry_sha256=(
            registry.registry_sha256 if registry.registry_digest_pinned else None
        ),
    )
    if reloaded != registry:
        raise MotionRegistryError(
            "registry object differs from a fresh strict parse of its bound bytes"
        )
    entries = registry.entries
    if authorization_purpose is not None:
        if (
            not isinstance(authorization_purpose, str)
            or authorization_purpose not in AUTHORIZATION_PURPOSES
        ):
            raise MotionRegistryError(
                "authorization_purpose must be one of "
                f"{AUTHORIZATION_PURPOSES} or None for identity-only audit"
            )
        if not registry.registry_digest_pinned:
            raise MotionRegistryError(
                "runtime adoption requires expected_registry_sha256; "
                "an unpinned registry is identity-audit only"
            )
        if expected_alignment_sha256 is None:
            raise MotionRegistryError(
                "runtime adoption requires expected_alignment_sha256; "
                "registry identity alone does not pin aligned runtime semantics"
            )
        if expected_canonical_ready_sha256 is None:
            raise MotionRegistryError(
                "runtime adoption requires expected_canonical_ready_sha256"
            )
        if expected_canonical_ready_fk_sha256 is None:
            raise MotionRegistryError(
                "runtime adoption requires expected_canonical_ready_fk_sha256"
            )
        expected_ready = _sha256_text(
            expected_canonical_ready_sha256,
            "expected_canonical_ready_sha256",
        )
        if registry.canonical_ready_sha256 != expected_ready:
            raise MotionRegistryError(
                "runtime canonical ready SHA-256 mismatch: "
                f"expected={expected_ready}, "
                f"actual={registry.canonical_ready_sha256}"
            )
        expected_ready_fk = _sha256_text(
            expected_canonical_ready_fk_sha256,
            "expected_canonical_ready_fk_sha256",
        )
        if registry.canonical_ready_fk_sha256 != expected_ready_fk:
            raise MotionRegistryError(
                "runtime canonical ready FK SHA-256 mismatch: "
                f"expected={expected_ready_fk}, "
                f"actual={registry.canonical_ready_fk_sha256}"
            )
        binding = bank_promotion_binding(
            registry, authorization_purpose=authorization_purpose
        )
        try:
            motion_admission.require_matching_admission(admission, binding)
        except motion_admission.MotionAdmissionError as exc:
            raise MotionRegistryError(
                f"{authorization_purpose} runtime adoption lacks trusted "
                f"admission: {exc}"
            ) from exc
        flag_name = f"{authorization_purpose}_authorized"
        unauthorized = tuple(
            row.motion_id for row in entries if not bool(getattr(row, flag_name))
        )
        if unauthorized:
            raise MotionRegistryError(
                f"{authorization_purpose} authorization is false for "
                f"{list(unauthorized)}"
            )
        invalid_publication = tuple(
            row.motion_id
            for row in entries
            if row.publication_class
            not in PUBLICATION_CLASSES_BY_PURPOSE[authorization_purpose]
        )
        if invalid_publication:
            raise MotionRegistryError(
                f"{authorization_purpose} runtime adoption rejects publication_class "
                f"for {list(invalid_publication)}"
            )
    tables = CanonicalRuntimeTables(
        bank_id=registry.bank_id,
        scope=registry.scope,
        motion_file=tuple(str(row.npz_path) for row in entries),
        clip_family_per_clip=tuple(row.family for row in entries),
        strike_phase_per_clip=tuple(row.strike_phase for row in entries),
        contact_opportunity_frames_per_clip=tuple(
            row.contact_opportunity_frames for row in entries
        ),
        mount_normal_sign_per_clip=tuple(
            row.mount_normal_sign for row in entries
        ),
        motion_ids=tuple(row.motion_id for row in entries),
        publication_class_per_clip=tuple(
            row.publication_class for row in entries
        ),
        source_manifest_path_per_clip=tuple(
            str(row.source_manifest_path) for row in entries
        ),
        source_manifest_sha256_per_clip=tuple(
            row.source_manifest_sha256 for row in entries
        ),
        build_manifest_path_per_clip=tuple(
            str(row.build_manifest_path) for row in entries
        ),
        build_manifest_sha256_per_clip=tuple(
            row.build_manifest_sha256 for row in entries
        ),
        applicability_manifest_path_per_clip=tuple(
            str(row.applicability_manifest_path) for row in entries
        ),
        applicability_manifest_sha256_per_clip=tuple(
            row.applicability_manifest_sha256 for row in entries
        ),
        evidence_level_per_clip=tuple(row.evidence_level for row in entries),
        evidence_manifest_path_per_clip=tuple(
            str(row.evidence_manifest_path) for row in entries
        ),
        evidence_manifest_sha256_per_clip=tuple(
            row.evidence_manifest_sha256 for row in entries
        ),
        question_bank_path_per_clip=tuple(
            None if row.question_bank_path is None else str(row.question_bank_path)
            for row in entries
        ),
        question_bank_sha256_per_clip=tuple(
            row.question_bank_sha256 for row in entries
        ),
        question_bank_schema_version_per_clip=tuple(
            row.question_bank_schema_version for row in entries
        ),
        training_config_path_per_clip=tuple(
            None
            if row.training_config_path is None
            else str(row.training_config_path)
            for row in entries
        ),
        training_config_sha256_per_clip=tuple(
            row.training_config_sha256 for row in entries
        ),
        training_config_schema_version_per_clip=tuple(
            row.training_config_schema_version for row in entries
        ),
        onnx_model_path_per_clip=tuple(
            None if row.onnx_model_path is None else str(row.onnx_model_path)
            for row in entries
        ),
        onnx_model_sha256_per_clip=tuple(
            row.onnx_model_sha256 for row in entries
        ),
        onnx_model_schema_version_per_clip=tuple(
            row.onnx_model_schema_version for row in entries
        ),
        onnx_metadata_path_per_clip=tuple(
            None if row.onnx_metadata_path is None else str(row.onnx_metadata_path)
            for row in entries
        ),
        onnx_metadata_sha256_per_clip=tuple(
            row.onnx_metadata_sha256 for row in entries
        ),
        onnx_metadata_schema_version_per_clip=tuple(
            row.onnx_metadata_schema_version for row in entries
        ),
        adoption_manifest_path_per_clip=tuple(
            None
            if row.adoption_manifest_path is None
            else str(row.adoption_manifest_path)
            for row in entries
        ),
        adoption_manifest_sha256_per_clip=tuple(
            row.adoption_manifest_sha256 for row in entries
        ),
        training_authorized_per_clip=tuple(
            row.training_authorized for row in entries
        ),
        deployment_authorized_per_clip=tuple(
            row.deployment_authorized for row in entries
        ),
        hardware_authorized_per_clip=tuple(
            row.hardware_authorized for row in entries
        ),
        npz_sha256_per_clip=tuple(row.npz_sha256 for row in entries),
        ready_runtime_sha256=entries[0].ready_runtime_sha256,
        canonical_ready_path=str(registry.canonical_ready_path),
        canonical_ready_sha256=registry.canonical_ready_sha256,
        canonical_ready_fk_path=str(registry.canonical_ready_fk_path),
        canonical_ready_fk_sha256=registry.canonical_ready_fk_sha256,
        registry_sha256=registry.registry_sha256,
        alignment_sha256=_alignment_sha256(registry),
        authorization_purpose=authorization_purpose,
    )
    columns = (
        tables.motion_file,
        tables.clip_family_per_clip,
        tables.strike_phase_per_clip,
        tables.contact_opportunity_frames_per_clip,
        tables.mount_normal_sign_per_clip,
        tables.motion_ids,
        tables.publication_class_per_clip,
        tables.source_manifest_path_per_clip,
        tables.source_manifest_sha256_per_clip,
        tables.build_manifest_path_per_clip,
        tables.build_manifest_sha256_per_clip,
        tables.applicability_manifest_path_per_clip,
        tables.applicability_manifest_sha256_per_clip,
        tables.evidence_level_per_clip,
        tables.evidence_manifest_path_per_clip,
        tables.evidence_manifest_sha256_per_clip,
        tables.question_bank_path_per_clip,
        tables.question_bank_sha256_per_clip,
        tables.question_bank_schema_version_per_clip,
        tables.training_config_path_per_clip,
        tables.training_config_sha256_per_clip,
        tables.training_config_schema_version_per_clip,
        tables.onnx_model_path_per_clip,
        tables.onnx_model_sha256_per_clip,
        tables.onnx_model_schema_version_per_clip,
        tables.onnx_metadata_path_per_clip,
        tables.onnx_metadata_sha256_per_clip,
        tables.onnx_metadata_schema_version_per_clip,
        tables.adoption_manifest_path_per_clip,
        tables.adoption_manifest_sha256_per_clip,
        tables.training_authorized_per_clip,
        tables.deployment_authorized_per_clip,
        tables.hardware_authorized_per_clip,
        tables.npz_sha256_per_clip,
    )
    expected_count = len(registry.motion_ids)
    if any(len(column) != expected_count for column in columns):
        raise MotionRegistryError(
            "runtime registry columns do not all match the ordered motion count"
        )
    if tables.motion_ids != registry.motion_ids:
        raise MotionRegistryError("runtime registry identity column changed order")
    if expected_alignment_sha256 is not None:
        expected = _sha256_text(
            expected_alignment_sha256, "expected_alignment_sha256"
        )
        if tables.alignment_sha256 != expected:
            raise MotionRegistryError(
                "runtime alignment SHA-256 mismatch: "
                f"expected={expected}, actual={tables.alignment_sha256}"
            )
    return tables


def load_runtime_tables(
    path: os.PathLike[str] | str,
    *,
    repo_root: os.PathLike[str] | str | None = None,
    expected_registry_sha256: str | None = None,
    expected_alignment_sha256: str | None = None,
    expected_canonical_ready_sha256: str | None = None,
    expected_canonical_ready_fk_sha256: str | None = None,
    authorization_purpose: str | None = "training",
    admission: motion_admission.TrustedMotionAdmission | None = None,
) -> CanonicalRuntimeTables:
    """One-call strict loader for the aligned, authorization-gated runtime lists."""

    registry = load_canonical_motion_bank_registry(
        path,
        repo_root=repo_root,
        expected_registry_sha256=expected_registry_sha256,
    )
    return adapt_registry_for_runtime(
        registry,
        expected_alignment_sha256=expected_alignment_sha256,
        expected_canonical_ready_sha256=expected_canonical_ready_sha256,
        expected_canonical_ready_fk_sha256=expected_canonical_ready_fk_sha256,
        authorization_purpose=authorization_purpose,
        admission=admission,
    )


def load_training_adopted_registry(
    path: os.PathLike[str] | str,
    promotion_certificate_path: os.PathLike[str] | str,
    *,
    repo_root: os.PathLike[str] | str | None,
    expected_registry_sha256: str,
    expected_alignment_sha256: str,
    expected_canonical_ready_sha256: str,
    expected_canonical_ready_fk_sha256: str,
    expected_promotion_certificate_sha256: str,
) -> tuple[CanonicalMotionBankRegistry, CanonicalRuntimeTables]:
    """Load one fully pinned training bank through the code-rooted trust chain.

    This is the narrow task-manifest-facing entry point.  It deliberately fixes
    the purpose to ``training`` and requires every independent pin, including
    the exact promotion-certificate bytes.  The returned registry preserves the
    ordered row identity needed for per-action reconciliation; the tables are
    independently reloaded and admission-checked by the runtime adapter.
    """

    expected_certificate = _sha256_text(
        expected_promotion_certificate_sha256,
        "expected_promotion_certificate_sha256",
    )
    registry = load_canonical_motion_bank_registry(
        path,
        repo_root=repo_root,
        expected_registry_sha256=expected_registry_sha256,
    )
    admission = verify_registry_promotion_certificate(
        registry,
        promotion_certificate_path,
        authorization_purpose="training",
    )
    if admission.certificate_sha256 != expected_certificate:
        raise MotionRegistryError(
            "promotion certificate SHA-256 mismatch: "
            f"expected={expected_certificate}, "
            f"actual={admission.certificate_sha256}"
        )
    tables = adapt_registry_for_runtime(
        registry,
        expected_alignment_sha256=expected_alignment_sha256,
        expected_canonical_ready_sha256=expected_canonical_ready_sha256,
        expected_canonical_ready_fk_sha256=(
            expected_canonical_ready_fk_sha256
        ),
        authorization_purpose="training",
        admission=admission,
    )
    return registry, tables
