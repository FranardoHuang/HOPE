#!/usr/bin/env python3
"""Strict loader for canonical motion-library recipe schema 2.

This module deliberately stops before geometry generation.  Its job is to make
the compiler inputs unambiguous:

* the recipe and every nested object use exact keys;
* every path is repository-relative and may not escape the repository;
* model, source-motion, and canonical-ready bytes match their bound SHA-256;
* motion sources are exact schema-2 files;
* the ready file has one exact, zero-velocity state;
* legacy ready donor claims are verified against bound donor source bytes;
* selected grounded-ready identities bind their candidate, ground receipt,
  minter report, independent face report, and explicit human adoption evidence;
* the exact marker-authority v2 path and SHA-256 are part of the recipe; and
* every source binding closes against that authority.

Recipe schema 2 intentionally has no per-motion protected span, source anchor,
or entry/exit override.  Those fields previously made a legacy stationary
scan seed look like a final certified window and hard-coded one retained core.
The compiler must instead consume the authority's distinct ge80 legacy seed,
ordinary nominal event (when one exists), synthetic construction annotation,
and explicit historical comparator.  A post-retime behavior rescan remains a
mandatory gate.

Passing this loader is an input-integrity gate only.  It does not authorize
training, hardware, motion geometry, dynamics, or a successful table-tennis
stroke.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from canonical_motion_markers import (
    MARKER_AUTHORITY_PROFILE_BY_PATH,
    MarkerSemantics,
    load_canonical_motion_markers,
)
from mujoco_motion_player import MotionClip, RUNTIME_BODY_NAMES, load_motion


class MotionRecipeError(ValueError):
    """The recipe or one of its content-addressed inputs is invalid."""


_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "library_id",
        "publication_class",
        "training_authorized",
        "hardware_authorized",
        "purpose",
        "frame_id",
        "marker_authority",
        "canonical_ready",
        "model_contract",
        "scope_contract",
        "time_law",
        "entry_exit_search",
        "motion_specs",
        "required_output_matrix",
        "post_build_gates",
    }
)
_MARKER_AUTHORITY_KEYS = frozenset({"path", "sha256"})
_READY_RECIPE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "donor_motion_id",
        "donor_source_frame",
        "donor_source_sha256",
        "endpoint_velocity_policy",
    }
)
_GROUNDED_READY_RECIPE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "provenance_mode",
        "candidate",
        "grounded_receipt",
        "minter_identity_report",
        "face_neutrality_report",
        "human_adoption_evidence",
        "endpoint_velocity_policy",
    }
)
_BOUND_FILE_KEYS = frozenset({"path", "sha256"})
_BOUND_PAYLOAD_FILE_KEYS = frozenset({"path", "sha256", "payload_sha256"})
_GROUNDED_READY_PROVENANCE_MODE = "selected_static_grounded_ready_identity_v1"
_GROUNDED_READY_SOURCE_SEGMENT = "grounded_ready_v2_g1_neutral_arm"
_ENDPOINT_VELOCITY_POLICY = "all_joint_root_body_velocities_exact_zero"
_MINTER_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "report_type",
        "tool_id",
        "artifact_class",
        "source",
        "upstream_selection",
        "ready_state",
        "ground_identity",
        "face_identity",
        "recipe_compatibility",
        "output",
        "authorization",
        "non_claims",
        "report_payload_sha256",
    }
)
_MINTER_SOURCE_KEYS = frozenset(
    {
        "candidate_path",
        "candidate_npz_sha256",
        "receipt_path",
        "receipt_json_sha256",
        "receipt_payload_sha256",
        "publication_payload_sha256",
        "candidate_id",
        "source_segment",
        "source_frame",
    }
)
_MINTER_READY_STATE_KEYS = frozenset(
    {
        "joint_count",
        "joint_pos_sha256",
        "joint_vel_exact_zero",
        "root_velocity_exact_zero",
        "state_sha256",
        "root_quaternion_wxyz_norm",
        "striking_joint_ids",
        "striking_joint_names",
    }
)
_MINTER_GROUND_KEYS = frozenset(
    {
        "status",
        "physics_rerun_by_this_tool",
        "upstream_exact_mujoco_backend",
        "upstream_gates",
        "mjcf_sha256",
        "compiled_model_sha256",
        "path_model_binding_sha256",
        "ground_model_binding_sha256",
        "claim_scope",
    }
)
_MINTER_FACE_KEYS = frozenset(
    {
        "status",
        "face_neutrality_proven",
        "external_face_identity_report_required",
        "claim_scope",
    }
)
_MINTER_COMPATIBILITY_KEYS = frozenset(
    {
        "strict_nine_key_ready_schema",
        "legacy_donor_frame_exact_contract",
        "required_recipe_provenance_mode",
        "identity_report_must_be_content_bound",
    }
)
_MINTER_OUTPUT_KEYS = frozenset(
    {
        "ready_filename",
        "ready_npz_sha256",
        "identity_report_filename",
        "completion_semantics",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    }
)
_UPSTREAM_SELECTION_KEYS = frozenset(
    {
        "selected_as_canonical_ready",
        "automatic_G1_or_G2_adoption",
        "requires_outer_comparison_across_all_five_motions",
    }
)
_FACE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "report_type",
        "artifact_class",
        "producer",
        "ready",
        "model",
        "evaluation",
        "verdict",
        "authorization",
        "non_claims",
        "report_payload_sha256",
    }
)
_FACE_PRODUCER_KEYS = frozenset(
    {
        "tool_path",
        "tool_sha256",
        "independent_from_ready_minter",
        "backend",
    }
)
_READY_EVIDENCE_KEYS = frozenset({"path", "sha256", "state_sha256"})
_FACE_MODEL_KEYS = frozenset(
    {
        "mjcf_sha256",
        "compiled_model_sha256",
        "racket_site",
        "face_normal_convention",
    }
)
_FACE_EVALUATION_KEYS = frozenset(
    {
        "scopes",
        "phases",
        "faces",
        "target_set_path",
        "target_set_sha256",
        "rows",
        "maximum_pair_asymmetry_rad",
        "maximum_allowed_pair_asymmetry_rad",
        "all_rows_exact_fk",
    }
)
_FACE_ROW_KEYS = frozenset(
    {
        "scope",
        "phase",
        "bh_target_sha256",
        "fh_target_sha256",
        "bh_distance_rad",
        "fh_distance_rad",
        "absolute_asymmetry_rad",
    }
)
_ADOPTION_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "evidence_type",
        "ready",
        "evidence_bindings",
        "decision",
        "authorization",
        "non_claims",
        "evidence_payload_sha256",
    }
)
_ADOPTION_BINDING_KEYS = frozenset(
    {
        "candidate_sha256",
        "grounded_receipt_sha256",
        "grounded_receipt_payload_sha256",
        "minter_identity_report_sha256",
        "minter_identity_report_payload_sha256",
        "face_neutrality_report_sha256",
        "face_neutrality_report_payload_sha256",
    }
)
_ADOPTION_DECISION_KEYS = frozenset(
    {
        "selected_as_canonical_ready",
        "decision_scope",
        "decision_maker_kind",
        "decision_maker",
        "decision_recorded_at_utc",
        "rationale",
    }
)
_MAX_READY_FACE_ASYMMETRY_RAD = math.radians(5.0)
_READY_FACE_PHASES = (
    "opportunity_start",
    "construction_donor_preferred",
    "nominal_event",
    "opportunity_end",
)
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
_MODEL_KEYS = frozenset(
    {
        "mjcf_path",
        "mjcf_sha256",
        "urdf_path",
        "urdf_sha256",
        "body_order_path",
        "body_order_sha256",
    }
)
_SCOPE_CONTRACT_KEYS = frozenset({"upper", "full"})
_UPPER_SCOPE_KEYS = frozenset(
    {"root", "lower_and_head", "pelvis_relative_rotation", "pelvis_translation"}
)
_FULL_SCOPE_KEYS = frozenset({"root", "joints"})
_TIME_LAW_KEYS = frozenset(
    {
        "fps",
        "joint_velocity_limit_fraction",
        "post_retime_behavior_opportunity_minimum_s",
        "legacy_seed_marker_policy",
        "kinematic_window_policy",
        "acceleration_policy",
        "window_acceleration_allowed_through_end",
        "window_acceleration_objective",
        "torque_claim",
    }
)
_ENTRY_SEARCH_KEYS = frozenset(
    {
        "mode",
        "legacy_ge80_halo_source_frames",
        "candidate_eligibility",
        "ranking_preference",
        "retained_source_prefix_required",
        "retained_source_suffix_required",
        "historical_adv2c3_role",
    }
)
_S0_OVERRIDE_KEYS = frozenset({"full"})
_S0_FULL_OVERRIDE_KEYS = frozenset({"grounding_policy", "maximum_grounding_offset_m"})
_MOTION_BASE_KEYS = frozenset(
    {
        "motion_id",
        "human_role",
        "source_path",
        "source_sha256",
        "scope_overrides",
    }
)
_MOTION_SYNTHETIC_EXTRA_KEYS = frozenset({"face_manifold"})
_FACE_KEYS = frozenset(
    {
        "mode",
        "active_joints",
        "site_position",
        "orientation",
        "single_axis_pi_overlay_forbidden",
    }
)
#: The canonical five, in order.  A library must still START with exactly
#: these, in exactly this order -- that is what stops an existing motion being
#: swapped out or reordered.  It may APPEND further motions after them.  The
#: old rule ("exactly five") conflated "do not touch the originals" with "this
#: repository may never hold a sixth stroke"; only the first half was ever the
#: point.
_REQUIRED_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)


def _ordered_prefix_error(actual: tuple[str, ...], label: str) -> str:
    return (
        f"{label} must begin with the canonical ordered five "
        f"{list(_REQUIRED_MOTION_IDS)} and may only append after them; got "
        f"{list(actual)}"
    )


def _check_ordered_prefix(actual: tuple[str, ...], label: str) -> None:
    if actual[: len(_REQUIRED_MOTION_IDS)] != _REQUIRED_MOTION_IDS:
        raise MotionRecipeError(_ordered_prefix_error(actual, label))
    if len(set(actual)) != len(actual):
        raise MotionRecipeError(f"{label} has duplicate motion ids: {list(actual)}")


_POST_BUILD_GATES = (
    "strict_schema2_and_shared_ready_digest",
    "exact_vendor_mujoco_fk_playback",
    "joint_position_velocity_and_plant_specific_torque_screen",
    "self_collision_body_racket_ground_table_net_scan",
    "post_retime_behavior_opportunity_rescan_per_scope",
    "stationary_behavior_and_recovery_exam_per_motion",
    "registry_consumer_export_deploy_contract",
)
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class ReadyState:
    path: Path
    sha256: str
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    source_segment: str
    source_frame: int


@dataclass(frozen=True)
class _ReadyProvenance:
    mode: str
    candidate_sha256: str | None = None
    grounded_receipt_sha256: str | None = None
    grounded_receipt_payload_sha256: str | None = None
    minter_identity_report_sha256: str | None = None
    minter_identity_report_payload_sha256: str | None = None
    face_neutrality_report_sha256: str | None = None
    face_neutrality_report_payload_sha256: str | None = None
    human_adoption_evidence_sha256: str | None = None
    human_adoption_evidence_payload_sha256: str | None = None
    grounded_mjcf_sha256: str | None = None
    grounded_compiled_model_sha256: str | None = None


@dataclass(frozen=True)
class MotionSource:
    motion_id: str
    human_role: str
    path: Path
    sha256: str
    clip: MotionClip
    face_manifold: Mapping[str, Any] | None
    scope_overrides: Mapping[str, Any]


@dataclass(frozen=True)
class CanonicalMotionRecipe:
    path: Path
    repo_root: Path
    raw: Mapping[str, Any]
    ready: ReadyState
    sources: tuple[MotionSource, ...]
    marker_semantics: MarkerSemantics
    marker_authority_path: Path
    marker_authority_sha256: str
    model_paths: Mapping[str, Path]
    model_hashes: Mapping[str, str]

    def source(self, motion_id: str) -> MotionSource:
        matches = [row for row in self.sources if row.motion_id == motion_id]
        if len(matches) != 1:
            raise MotionRecipeError(
                f"recipe has {len(matches)} sources named {motion_id!r}"
            )
        return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MotionRecipeError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MotionRecipeError(f"{label} contains non-finite JSON number {token!r}")
            ),
        )
    except MotionRecipeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MotionRecipeError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MotionRecipeError(f"{label} must contain one JSON object")
    return value


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MotionRecipeError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MotionRecipeError(f"{label} keys must be strings")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MotionRecipeError(
            f"{label} keys changed; missing={missing}, extra={extra}"
        )
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MotionRecipeError(
            f"{label} must be a non-empty string without edge whitespace"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if (
        len(text) != _SHA256_HEX_LENGTH
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise MotionRecipeError(f"{label} must be a 64-character lowercase SHA-256")
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MotionRecipeError(f"{label} must be an integer >= {minimum}")
    return value


def _repo_path(repo_root: Path, value: Any, label: str) -> Path:
    text = _nonempty_string(value, label)
    relative = Path(text)
    if relative.is_absolute():
        raise MotionRecipeError(f"{label} must be repository-relative")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MotionRecipeError(f"{label} escapes the repository") from exc
    if not resolved.is_file():
        raise MotionRecipeError(f"{label} does not exist: {resolved}")
    return resolved


def _check_bound_file(
    repo_root: Path,
    path_value: Any,
    sha_value: Any,
    label: str,
) -> tuple[Path, str]:
    path = _repo_path(repo_root, path_value, f"{label} path")
    expected = _sha256(sha_value, f"{label} sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise MotionRecipeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path, actual


def _finite_vector(
    value: Any,
    length: int,
    label: str,
    *,
    exact_dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,):
        raise MotionRecipeError(
            f"{label} must have shape ({length},), got {array.shape}"
        )
    if exact_dtype is not None and array.dtype != exact_dtype:
        raise MotionRecipeError(
            f"{label} dtype must be {exact_dtype}, got {array.dtype}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise MotionRecipeError(f"{label} must be real numeric")
    output = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(output)):
        raise MotionRecipeError(f"{label} contains NaN or infinity")
    return output.copy()


def _scalar_text(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise MotionRecipeError(f"{label} must be one scalar string")
    return _nonempty_string(str(array.item()), label)


def _load_ready(path: Path, expected_sha: str) -> ReadyState:
    if sha256_file(path) != expected_sha:
        raise MotionRecipeError("canonical ready bytes changed after hash verification")
    try:
        with np.load(path, allow_pickle=False) as payload:
            if frozenset(payload.files) != _READY_FILE_KEYS:
                raise MotionRecipeError(
                    "canonical ready file keys changed; "
                    f"expected={sorted(_READY_FILE_KEYS)}, got={sorted(payload.files)}"
                )
            joint_pos = _finite_vector(
                payload["joint_pos"],
                31,
                "ready joint_pos",
                exact_dtype=np.dtype("float64"),
            )
            joint_vel = _finite_vector(
                payload["joint_vel"],
                31,
                "ready joint_vel",
                exact_dtype=np.dtype("float64"),
            )
            root_pos = _finite_vector(
                payload["root_pos_w"],
                3,
                "ready root_pos_w",
                exact_dtype=np.dtype("float64"),
            )
            root_quat = _finite_vector(
                payload["root_quat_w"],
                4,
                "ready root_quat_w",
                exact_dtype=np.dtype("float64"),
            )
            source_segment = _scalar_text(
                payload["source_segment"], "ready source_segment"
            )
            source_frame_raw = np.asarray(payload["source_frame"])
            if source_frame_raw.shape != () or not np.issubdtype(
                source_frame_raw.dtype, np.integer
            ):
                raise MotionRecipeError("ready source_frame must be one integer scalar")
            source_frame = int(source_frame_raw.item())
            striking_ids = np.asarray(payload["striking_joint_ids"])
            if striking_ids.shape != (7,) or not np.issubdtype(
                striking_ids.dtype, np.integer
            ):
                raise MotionRecipeError(
                    "ready striking_joint_ids must have integer shape (7,)"
                )
            if len(set(int(index) for index in striking_ids)) != 7 or np.any(
                (striking_ids < 0) | (striking_ids >= 31)
            ):
                raise MotionRecipeError(
                    "ready striking_joint_ids are not seven unique joints"
                )
            _scalar_text(payload["source_npz"], "ready source_npz")
            _scalar_text(payload["note"], "ready note")
    except (OSError, ValueError) as exc:
        if isinstance(exc, MotionRecipeError):
            raise
        raise MotionRecipeError(f"cannot load canonical ready {path}: {exc}") from exc

    if not np.array_equal(joint_vel, np.zeros(31, dtype=np.float64)):
        raise MotionRecipeError("canonical ready joint velocity must be exact zero")
    quaternion_norm = float(np.linalg.norm(root_quat))
    # The ready bytes are an exact promotion of a float32 donor frame.  Do not
    # normalize them here (that would break donor equality); accept only the
    # float32-scale unit-quaternion residual that the source already carries.
    if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise MotionRecipeError(
            f"canonical ready root quaternion norm is {quaternion_norm}, expected 1"
        )
    return ReadyState(
        path=path,
        sha256=expected_sha,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos_w=root_pos,
        root_quat_wxyz=root_quat,
        source_segment=source_segment,
        source_frame=source_frame,
    )


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
        raise MotionRecipeError(f"cannot canonicalize evidence JSON: {exc}") from exc


def _load_bound_payload_json(
    repo_root: Path,
    raw_binding: Any,
    *,
    label: str,
    payload_field: str,
) -> tuple[Path, str, str, Mapping[str, Any]]:
    binding = _exact_keys(raw_binding, _BOUND_PAYLOAD_FILE_KEYS, label)
    path, file_sha = _check_bound_file(
        repo_root,
        binding["path"],
        binding["sha256"],
        label,
    )
    payload_sha = _sha256(binding["payload_sha256"], f"{label} payload_sha256")
    try:
        raw = _strict_json_bytes(path.read_bytes(), label)
    except OSError as exc:
        raise MotionRecipeError(f"cannot read {label}: {exc}") from exc
    observed = _sha256(raw.get(payload_field), f"{label}.{payload_field}")
    unsigned = dict(raw)
    unsigned.pop(payload_field, None)
    computed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if observed != payload_sha or computed != payload_sha:
        raise MotionRecipeError(
            f"{label} canonical payload SHA-256 does not close: "
            f"bound={payload_sha}, embedded={observed}, computed={computed}"
        )
    return path, file_sha, payload_sha, raw


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


def _ready_state_sha256(ready: ReadyState) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "joint_pos", ready.joint_pos)
    _hash_array(digest, "root_pos_w", ready.root_pos_w)
    _hash_array(digest, "root_quat_wxyz", ready.root_quat_wxyz)
    return digest.hexdigest()


def _require_false_authorization(value: Any, label: str) -> None:
    authorization = _exact_keys(value, _AUTHORIZATION_KEYS, label)
    if any(authorization[key] is not False for key in _AUTHORIZATION_KEYS):
        raise MotionRecipeError(
            f"{label} must deny training, deployment, and hardware authorization"
        )


def _require_nonempty_string_list(value: Any, label: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item or item.strip() != item
            for item in value
        )
    ):
        raise MotionRecipeError(f"{label} must be a non-empty string list")


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MotionRecipeError(f"{label} must be one finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise MotionRecipeError(f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise MotionRecipeError(f"{label} must be <= {maximum}")
    return number


def _validate_minter_identity_report(
    report: Mapping[str, Any],
    *,
    repo_root: Path,
    ready: ReadyState,
    candidate_path: Path,
    candidate_sha: str,
    receipt_path: Path,
    receipt_file_sha: str,
    receipt_payload_sha: str,
    validated_candidate: Any,
) -> None:
    _exact_keys(report, _MINTER_REPORT_KEYS, "minter identity report")
    if (
        report["schema_version"] != 1
        or report["report_type"] != "canonical-ready-sidecar-identity-v1"
        or report["tool_id"] != "canonical_ready_sidecar_mint_v1"
        or report["artifact_class"] != "diagnostic_canonical_ready_sidecar"
    ):
        raise MotionRecipeError("minter identity report schema/type changed")

    candidate_repo_path = candidate_path.relative_to(repo_root).as_posix()
    receipt_repo_path = receipt_path.relative_to(repo_root).as_posix()
    ready_repo_path = ready.path.relative_to(repo_root).as_posix()
    source = _exact_keys(
        report["source"], _MINTER_SOURCE_KEYS, "minter identity report source"
    )
    if source != {
        "candidate_path": candidate_repo_path,
        "candidate_npz_sha256": candidate_sha,
        "receipt_path": receipt_repo_path,
        "receipt_json_sha256": receipt_file_sha,
        "receipt_payload_sha256": receipt_payload_sha,
        "publication_payload_sha256": (validated_candidate.publication_payload_sha256),
        "candidate_id": "G1",
        "source_segment": _GROUNDED_READY_SOURCE_SEGMENT,
        "source_frame": 0,
    }:
        raise MotionRecipeError(
            "minter identity report source does not close against candidate/receipt"
        )

    selection = _exact_keys(
        report["upstream_selection"],
        _UPSTREAM_SELECTION_KEYS,
        "minter upstream_selection",
    )
    if dict(selection) != dict(validated_candidate.receipt["selection"]):
        raise MotionRecipeError(
            "minter identity report changed the upstream non-selection record"
        )
    if selection["selected_as_canonical_ready"] is not False:
        raise MotionRecipeError(
            "grounded receipt may not self-select a canonical ready"
        )

    state = _exact_keys(
        report["ready_state"],
        _MINTER_READY_STATE_KEYS,
        "minter ready_state",
    )
    if (
        state["joint_count"] != 31
        or state["joint_pos_sha256"] != _array_sha256(ready.joint_pos)
        or state["state_sha256"] != _ready_state_sha256(ready)
        or state["joint_vel_exact_zero"] is not True
        or state["root_velocity_exact_zero"] is not True
        or not math.isclose(
            _finite_number(
                state["root_quaternion_wxyz_norm"],
                "minter root quaternion norm",
            ),
            float(np.linalg.norm(ready.root_quat_wxyz)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise MotionRecipeError(
            "minter identity report ready-state identity does not close"
        )
    striking_ids = np.asarray(state["striking_joint_ids"])
    if (
        striking_ids.shape != (7,)
        or not np.issubdtype(striking_ids.dtype, np.integer)
        or not np.array_equal(striking_ids, validated_candidate.striking_joint_ids)
        or not isinstance(state["striking_joint_names"], list)
        or len(state["striking_joint_names"]) != 7
    ):
        raise MotionRecipeError("minter striking-joint identity changed")

    ground = _exact_keys(
        report["ground_identity"],
        _MINTER_GROUND_KEYS,
        "minter ground_identity",
    )
    exact_model = validated_candidate.receipt["exact_model"]
    if (
        ground["status"] != "PASS_BOUND_UPSTREAM_G1_STATIC_GROUND_RECEIPT"
        or ground["physics_rerun_by_this_tool"] is not False
        or ground["upstream_exact_mujoco_backend"] is not True
        or ground["upstream_gates"] != validated_candidate.receipt["gates"]
        or ground["mjcf_sha256"] != exact_model["mjcf_sha256"]
        or ground["compiled_model_sha256"] != exact_model["compiled_model_sha256"]
        or ground["path_model_binding_sha256"]
        != exact_model["path_model_binding_sha256"]
        or ground["ground_model_binding_sha256"]
        != exact_model["ground_model_binding_sha256"]
        or ground["claim_scope"]
        != "content-bound upstream identity and static-ground receipt only"
    ):
        raise MotionRecipeError("minter ground identity does not close")

    face = _exact_keys(
        report["face_identity"],
        _MINTER_FACE_KEYS,
        "minter face_identity",
    )
    if face != {
        "status": "NOT_PROVEN_BY_GROUNDED_READY_RECEIPT",
        "face_neutrality_proven": False,
        "external_face_identity_report_required": True,
        "claim_scope": (
            "right-arm overlay bytes are identified; face FK/neutrality is not"
        ),
    }:
        raise MotionRecipeError(
            "minter report may not claim its own face-neutrality proof"
        )

    compatibility = _exact_keys(
        report["recipe_compatibility"],
        _MINTER_COMPATIBILITY_KEYS,
        "minter recipe_compatibility",
    )
    if compatibility != {
        "strict_nine_key_ready_schema": True,
        "legacy_donor_frame_exact_contract": False,
        "required_recipe_provenance_mode": _GROUNDED_READY_PROVENANCE_MODE,
        "identity_report_must_be_content_bound": True,
    }:
        raise MotionRecipeError("minter recipe compatibility contract changed")
    output = _exact_keys(report["output"], _MINTER_OUTPUT_KEYS, "minter output")
    if (
        output["ready_filename"] != ready.path.name
        or output["ready_npz_sha256"] != ready.sha256
        or output["identity_report_filename"] != "IDENTITY_REPORT.json"
        or output["completion_semantics"]
        != "exclusive_directory_and_identity_report_written_last"
    ):
        raise MotionRecipeError(
            f"minter report output does not bind canonical ready {ready_repo_path}"
        )
    _require_false_authorization(
        report["authorization"], "minter identity report authorization"
    )
    _require_nonempty_string_list(
        report["non_claims"], "minter identity report non_claims"
    )


def _validate_face_neutrality_report(
    report: Mapping[str, Any],
    *,
    repo_root: Path,
    ready: ReadyState,
    grounded_exact_model: Mapping[str, Any],
) -> None:
    _exact_keys(report, _FACE_REPORT_KEYS, "face-neutrality report")
    if (
        report["schema_version"] != 1
        or report["report_type"] != "canonical-ready-face-neutrality-v1"
        or report["artifact_class"] != "independent_exact_fk_face_neutrality_evidence"
        or report["verdict"] != "PASS_FACE_NEUTRAL_READY"
    ):
        raise MotionRecipeError("face-neutrality report schema/verdict changed")

    producer = _exact_keys(
        report["producer"], _FACE_PRODUCER_KEYS, "face-neutrality producer"
    )
    producer_path, producer_sha = _check_bound_file(
        repo_root,
        producer["tool_path"],
        producer["tool_sha256"],
        "face-neutrality producer tool",
    )
    producer_root = (
        repo_root / "hope_training" / "whole_body_tracking" / "scripts"
    ).resolve()
    minter_tool = producer_root / "canonical_ready_sidecar_mint.py"
    try:
        producer_path.relative_to(producer_root)
    except ValueError as exc:
        raise MotionRecipeError(
            "face-neutrality producer must be a content-bound code-root tool"
        ) from exc
    if (
        producer["independent_from_ready_minter"] is not True
        or producer["backend"] != "exact_vendor_mujoco_fk"
        or producer_path == minter_tool.resolve()
        or (minter_tool.is_file() and producer_sha == sha256_file(minter_tool))
    ):
        raise MotionRecipeError(
            "face-neutrality report must be independent exact-vendor-MuJoCo FK"
        )

    ready_identity = _exact_keys(
        report["ready"], _READY_EVIDENCE_KEYS, "face-neutrality ready"
    )
    if ready_identity != {
        "path": ready.path.relative_to(repo_root).as_posix(),
        "sha256": ready.sha256,
        "state_sha256": _ready_state_sha256(ready),
    }:
        raise MotionRecipeError("face-neutrality report binds a different ready state")

    model = _exact_keys(report["model"], _FACE_MODEL_KEYS, "face-neutrality model")
    if (
        model["mjcf_sha256"] != grounded_exact_model["mjcf_sha256"]
        or model["compiled_model_sha256"]
        != grounded_exact_model["compiled_model_sha256"]
        or model["racket_site"] != "right_racket"
        or model["face_normal_convention"]
        != "right_racket_site_local_plus_y_world_signed_face_normal_v1"
    ):
        raise MotionRecipeError(
            "face-neutrality report model/face convention differs from ground identity"
        )

    evaluation = _exact_keys(
        report["evaluation"],
        _FACE_EVALUATION_KEYS,
        "face-neutrality evaluation",
    )
    _check_bound_file(
        repo_root,
        evaluation["target_set_path"],
        evaluation["target_set_sha256"],
        "face-neutrality target set",
    )
    if (
        evaluation["scopes"] != ["upper", "full"]
        or evaluation["phases"] != list(_READY_FACE_PHASES)
        or evaluation["faces"] != ["bh", "fh"]
        or evaluation["all_rows_exact_fk"] is not True
    ):
        raise MotionRecipeError(
            "face-neutrality evaluation must cover exact "
            "upper/full x four phases x bh/fh FK"
        )
    rows = evaluation["rows"]
    expected_rows = [
        (scope, phase) for scope in ("upper", "full") for phase in _READY_FACE_PHASES
    ]
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise MotionRecipeError(
            "face-neutrality report needs all eight ordered scope/phase pair rows"
        )
    asymmetries: list[float] = []
    for index, (expected_scope, expected_phase) in enumerate(expected_rows):
        row = _exact_keys(rows[index], _FACE_ROW_KEYS, f"face-neutrality rows[{index}]")
        if row["scope"] != expected_scope or row["phase"] != expected_phase:
            raise MotionRecipeError("face-neutrality row scope/phase order changed")
        _sha256(
            row["bh_target_sha256"],
            f"face-neutrality {expected_scope}/{expected_phase} bh target",
        )
        _sha256(
            row["fh_target_sha256"],
            f"face-neutrality {expected_scope}/{expected_phase} fh target",
        )
        if row["bh_target_sha256"] == row["fh_target_sha256"]:
            raise MotionRecipeError(
                "face-neutrality BH/FH target identities must remain distinct"
            )
        bh = _finite_number(
            row["bh_distance_rad"],
            f"face-neutrality {expected_scope}/{expected_phase} bh_distance_rad",
            minimum=0.0,
            maximum=math.pi,
        )
        fh = _finite_number(
            row["fh_distance_rad"],
            f"face-neutrality {expected_scope}/{expected_phase} fh_distance_rad",
            minimum=0.0,
            maximum=math.pi,
        )
        asymmetry = _finite_number(
            row["absolute_asymmetry_rad"],
            (
                "face-neutrality "
                f"{expected_scope}/{expected_phase} absolute_asymmetry_rad"
            ),
            minimum=0.0,
            maximum=math.pi,
        )
        if not math.isclose(asymmetry, abs(bh - fh), rel_tol=0.0, abs_tol=1.0e-12):
            raise MotionRecipeError(
                "face-neutrality "
                f"{expected_scope}/{expected_phase} asymmetry is inconsistent"
            )
        asymmetries.append(asymmetry)
    maximum_asymmetry = _finite_number(
        evaluation["maximum_pair_asymmetry_rad"],
        "face-neutrality maximum_pair_asymmetry_rad",
        minimum=0.0,
        maximum=math.pi,
    )
    allowed = _finite_number(
        evaluation["maximum_allowed_pair_asymmetry_rad"],
        "face-neutrality maximum_allowed_pair_asymmetry_rad",
        minimum=0.0,
        maximum=_MAX_READY_FACE_ASYMMETRY_RAD,
    )
    if (
        allowed <= 0.0
        or not math.isclose(
            maximum_asymmetry,
            max(asymmetries),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or maximum_asymmetry > allowed
    ):
        raise MotionRecipeError(
            "face-neutrality asymmetry exceeds or contradicts its strict bound"
        )
    _require_false_authorization(
        report["authorization"], "face-neutrality authorization"
    )
    _require_nonempty_string_list(report["non_claims"], "face-neutrality non_claims")


def _validate_human_adoption_evidence(
    evidence: Mapping[str, Any],
    *,
    repo_root: Path,
    ready: ReadyState,
    candidate_sha: str,
    receipt_file_sha: str,
    receipt_payload_sha: str,
    minter_file_sha: str,
    minter_payload_sha: str,
    face_file_sha: str,
    face_payload_sha: str,
) -> None:
    _exact_keys(evidence, _ADOPTION_EVIDENCE_KEYS, "human adoption evidence")
    if (
        evidence["schema_version"] != 1
        or evidence["evidence_type"] != "canonical-ready-human-adoption-v1"
    ):
        raise MotionRecipeError("human adoption evidence schema/type changed")
    ready_identity = _exact_keys(
        evidence["ready"], _READY_EVIDENCE_KEYS, "human adoption ready"
    )
    if ready_identity != {
        "path": ready.path.relative_to(repo_root).as_posix(),
        "sha256": ready.sha256,
        "state_sha256": _ready_state_sha256(ready),
    }:
        raise MotionRecipeError("human adoption evidence binds a different ready")
    bindings = _exact_keys(
        evidence["evidence_bindings"],
        _ADOPTION_BINDING_KEYS,
        "human adoption evidence_bindings",
    )
    if bindings != {
        "candidate_sha256": candidate_sha,
        "grounded_receipt_sha256": receipt_file_sha,
        "grounded_receipt_payload_sha256": receipt_payload_sha,
        "minter_identity_report_sha256": minter_file_sha,
        "minter_identity_report_payload_sha256": minter_payload_sha,
        "face_neutrality_report_sha256": face_file_sha,
        "face_neutrality_report_payload_sha256": face_payload_sha,
    }:
        raise MotionRecipeError(
            "human adoption evidence does not bind the exact evidence chain"
        )
    decision = _exact_keys(
        evidence["decision"], _ADOPTION_DECISION_KEYS, "human adoption decision"
    )
    decision_maker = _nonempty_string(
        decision["decision_maker"], "human adoption decision_maker"
    )
    lowered = decision_maker.casefold()
    if (
        decision["selected_as_canonical_ready"] is not True
        or decision["decision_scope"]
        != "canonical_ready_identity_for_compiler_candidate_only"
        or decision["decision_maker_kind"] != "human"
        or decision_maker == "UNASSIGNED"
        or any(token in lowered for token in ("codex", "claude", "chatgpt", "agent"))
    ):
        raise MotionRecipeError(
            "canonical ready selection requires one explicit named human decision"
        )
    recorded = _nonempty_string(
        decision["decision_recorded_at_utc"],
        "human adoption decision_recorded_at_utc",
    )
    try:
        from datetime import datetime

        datetime.strptime(recorded, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise MotionRecipeError(
            "human adoption decision_recorded_at_utc must be UTC YYYY-MM-DDTHH:MM:SSZ"
        ) from exc
    _nonempty_string(decision["rationale"], "human adoption rationale")
    _require_false_authorization(
        evidence["authorization"], "human adoption authorization"
    )
    _require_nonempty_string_list(evidence["non_claims"], "human adoption non_claims")


def _load_canonical_ready_contract(
    repo_root: Path,
    raw_contract: Any,
) -> tuple[ReadyState, _ReadyProvenance, Mapping[str, Any]]:
    """Load either the legacy donor-exact or selected grounded-ready contract."""

    if not isinstance(raw_contract, Mapping):
        raise MotionRecipeError("canonical_ready must be an object")
    if frozenset(raw_contract) == _READY_RECIPE_KEYS:
        contract = _exact_keys(raw_contract, _READY_RECIPE_KEYS, "canonical_ready")
        mode = "legacy_donor_frame_exact_v1"
    else:
        contract = _exact_keys(
            raw_contract, _GROUNDED_READY_RECIPE_KEYS, "canonical_ready"
        )
        if contract["provenance_mode"] != _GROUNDED_READY_PROVENANCE_MODE:
            raise MotionRecipeError("canonical ready provenance_mode is not supported")
        mode = _GROUNDED_READY_PROVENANCE_MODE
    if contract["endpoint_velocity_policy"] != _ENDPOINT_VELOCITY_POLICY:
        raise MotionRecipeError("canonical ready endpoint velocity policy changed")
    ready_path, ready_sha = _check_bound_file(
        repo_root,
        contract["path"],
        contract["sha256"],
        "canonical ready",
    )
    ready = _load_ready(ready_path, ready_sha)
    if mode == "legacy_donor_frame_exact_v1":
        return ready, _ReadyProvenance(mode=mode), contract

    candidate_binding = _exact_keys(
        contract["candidate"], _BOUND_FILE_KEYS, "canonical_ready.candidate"
    )
    receipt_binding = _exact_keys(
        contract["grounded_receipt"],
        _BOUND_PAYLOAD_FILE_KEYS,
        "canonical_ready.grounded_receipt",
    )
    candidate_path, candidate_sha = _check_bound_file(
        repo_root,
        candidate_binding["path"],
        candidate_binding["sha256"],
        "canonical ready grounded candidate",
    )
    receipt_path, receipt_file_sha = _check_bound_file(
        repo_root,
        receipt_binding["path"],
        receipt_binding["sha256"],
        "canonical ready grounded receipt",
    )
    receipt_payload_sha = _sha256(
        receipt_binding["payload_sha256"],
        "canonical_ready.grounded_receipt.payload_sha256",
    )
    try:
        from canonical_ready_sidecar_mint import (
            ReadySidecarMintError,
            validate_ready_candidate_bundle,
        )

        validated_candidate = validate_ready_candidate_bundle(
            repo_root=repo_root,
            candidate_path=candidate_path,
            expected_candidate_sha256=candidate_sha,
            receipt_path=receipt_path,
            expected_receipt_sha256=receipt_file_sha,
        )
    except (ReadySidecarMintError, OSError) as exc:
        raise MotionRecipeError(
            f"canonical ready grounded candidate/receipt is invalid: {exc}"
        ) from exc
    if validated_candidate.receipt_payload_sha256 != receipt_payload_sha:
        raise MotionRecipeError(
            "canonical ready grounded receipt payload SHA-256 does not close"
        )
    if (
        ready.source_segment != _GROUNDED_READY_SOURCE_SEGMENT
        or ready.source_frame != 0
        or not np.array_equal(ready.joint_pos, validated_candidate.joint_pos)
        or not np.array_equal(ready.joint_vel, validated_candidate.joint_vel)
        or not np.array_equal(ready.root_pos_w, validated_candidate.root_pos_w)
        or not np.array_equal(ready.root_quat_wxyz, validated_candidate.root_quat_wxyz)
    ):
        raise MotionRecipeError("grounded-neutral ready is not candidate-state exact")
    try:
        with np.load(ready.path, allow_pickle=False) as ready_payload:
            source_npz = _scalar_text(
                ready_payload["source_npz"], "grounded ready source_npz"
            )
            note = _scalar_text(ready_payload["note"], "grounded ready note")
            striking_ids = np.asarray(ready_payload["striking_joint_ids"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, MotionRecipeError):
            raise
        raise MotionRecipeError(
            f"cannot recheck grounded ready metadata: {exc}"
        ) from exc
    if (
        source_npz != candidate_path.relative_to(repo_root).as_posix()
        or "not donor-frame exact" not in note
        or not np.array_equal(striking_ids, validated_candidate.striking_joint_ids)
    ):
        raise MotionRecipeError(
            "grounded-neutral ready metadata is not honest candidate provenance"
        )

    (
        minter_path,
        minter_file_sha,
        minter_payload_sha,
        minter_report,
    ) = _load_bound_payload_json(
        repo_root,
        contract["minter_identity_report"],
        label="canonical ready minter identity report",
        payload_field="report_payload_sha256",
    )
    _validate_minter_identity_report(
        minter_report,
        repo_root=repo_root,
        ready=ready,
        candidate_path=candidate_path,
        candidate_sha=candidate_sha,
        receipt_path=receipt_path,
        receipt_file_sha=receipt_file_sha,
        receipt_payload_sha=receipt_payload_sha,
        validated_candidate=validated_candidate,
    )

    (
        face_path,
        face_file_sha,
        face_payload_sha,
        face_report,
    ) = _load_bound_payload_json(
        repo_root,
        contract["face_neutrality_report"],
        label="canonical ready face-neutrality report",
        payload_field="report_payload_sha256",
    )
    if face_path == minter_path:
        raise MotionRecipeError(
            "face-neutrality report must be independent from the minter report"
        )
    _validate_face_neutrality_report(
        face_report,
        repo_root=repo_root,
        ready=ready,
        grounded_exact_model=validated_candidate.receipt["exact_model"],
    )

    (
        adoption_path,
        adoption_file_sha,
        adoption_payload_sha,
        adoption_evidence,
    ) = _load_bound_payload_json(
        repo_root,
        contract["human_adoption_evidence"],
        label="canonical ready human adoption evidence",
        payload_field="evidence_payload_sha256",
    )
    if adoption_path in {minter_path, face_path, receipt_path}:
        raise MotionRecipeError(
            "human adoption evidence must be a distinct immutable record"
        )
    _validate_human_adoption_evidence(
        adoption_evidence,
        repo_root=repo_root,
        ready=ready,
        candidate_sha=candidate_sha,
        receipt_file_sha=receipt_file_sha,
        receipt_payload_sha=receipt_payload_sha,
        minter_file_sha=minter_file_sha,
        minter_payload_sha=minter_payload_sha,
        face_file_sha=face_file_sha,
        face_payload_sha=face_payload_sha,
    )

    return (
        ready,
        _ReadyProvenance(
            mode=mode,
            candidate_sha256=candidate_sha,
            grounded_receipt_sha256=receipt_file_sha,
            grounded_receipt_payload_sha256=receipt_payload_sha,
            minter_identity_report_sha256=minter_file_sha,
            minter_identity_report_payload_sha256=minter_payload_sha,
            face_neutrality_report_sha256=face_file_sha,
            face_neutrality_report_payload_sha256=face_payload_sha,
            human_adoption_evidence_sha256=adoption_file_sha,
            human_adoption_evidence_payload_sha256=adoption_payload_sha,
            grounded_mjcf_sha256=validated_candidate.receipt["exact_model"][
                "mjcf_sha256"
            ],
            grounded_compiled_model_sha256=validated_candidate.receipt["exact_model"][
                "compiled_model_sha256"
            ],
        ),
        contract,
    )


def _validate_recipe_contract(raw: Mapping[str, Any]) -> None:
    _exact_keys(raw, _TOP_LEVEL_KEYS, "recipe")
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 2:
        raise MotionRecipeError("recipe schema_version must be exactly 2")
    _nonempty_string(raw["library_id"], "library_id")
    if raw["publication_class"] != "compiler_candidate":
        raise MotionRecipeError(
            "this compiler only accepts publication_class='compiler_candidate'"
        )
    if raw["training_authorized"] is not False:
        raise MotionRecipeError("recipe must keep training_authorized=false")
    if raw["hardware_authorized"] is not False:
        raise MotionRecipeError("recipe must keep hardware_authorized=false")
    _nonempty_string(raw["purpose"], "purpose")
    if raw["frame_id"] != "a3_robot_origin_ground_z0":
        raise MotionRecipeError("recipe frame_id is not the canonical motion frame")
    scope = _exact_keys(raw["scope_contract"], _SCOPE_CONTRACT_KEYS, "scope_contract")
    upper = _exact_keys(scope["upper"], _UPPER_SCOPE_KEYS, "scope_contract.upper")
    full = _exact_keys(scope["full"], _FULL_SCOPE_KEYS, "scope_contract.full")
    if upper != {
        "root": "fixed_canonical_ready",
        "lower_and_head": "fixed_canonical_ready",
        "pelvis_relative_rotation": "fold_complete_so3_into_waist_zxy",
        "pelvis_translation": "removed_and_reported",
    }:
        raise MotionRecipeError("upper scope contract changed")
    if full != {
        "root": "one_atomic_se2_frame0_alignment_then_preserve_local_motion",
        "joints": "preserve_full_source_before_ready_connectors",
    }:
        raise MotionRecipeError("full scope contract changed")
    time_law = _exact_keys(raw["time_law"], _TIME_LAW_KEYS, "time_law")
    if float(time_law["fps"]) != 50.0:
        raise MotionRecipeError("canonical output fps must be exactly 50")
    minimum_window = time_law["post_retime_behavior_opportunity_minimum_s"]
    if (
        isinstance(minimum_window, bool)
        or not isinstance(minimum_window, (int, float))
        or not math.isfinite(float(minimum_window))
        or float(minimum_window) <= 0.0
    ):
        raise MotionRecipeError(
            "post-retime behavior opportunity minimum must be finite and positive"
        )
    if time_law.get("legacy_seed_marker_policy") != (
        "search_and_retime_marker_only_never_output_behavior_window"
    ):
        raise MotionRecipeError(
            "legacy stationary-scan seeds may never certify an output window"
        )
    if time_law.get("window_acceleration_allowed_through_end") is not True:
        raise MotionRecipeError("window acceleration-through-end contract changed")
    if time_law.get("kinematic_window_policy") != (
        "nonnegative_scalar_acceleration_through_exact_window_end"
    ):
        raise MotionRecipeError(
            "window-end no-early-scalar-braking policy is mandatory"
        )
    if time_law.get("acceleration_policy") != (
        "grounded_torque_contact_screen_required_before_promotion_beyond_"
        "compiler_candidate"
    ):
        raise MotionRecipeError(
            "grounded torque/contact promotion screen is not mandatory"
        )
    if time_law.get("joint_velocity_limit_fraction") != 1.0:
        raise MotionRecipeError("the compiler must use the exact URDF velocity limit")
    search = _exact_keys(
        raw["entry_exit_search"], _ENTRY_SEARCH_KEYS, "entry_exit_search"
    )
    if search["mode"] != "enumerate_all_then_gate_and_rank":
        raise MotionRecipeError("entry/exit search must enumerate before ranking")
    halo = _integer(
        search["legacy_ge80_halo_source_frames"],
        "entry_exit_search.legacy_ge80_halo_source_frames",
        minimum=1,
    )
    if halo < 1:
        raise MotionRecipeError("legacy ge80 halo must retain at least one frame")
    if search["candidate_eligibility"] != (
        "retain_legacy_ge80_seed_plus_symmetric_halo"
    ):
        raise MotionRecipeError(
            "candidate eligibility must use legacy ge80 plus its halo"
        )
    if search["ranking_preference"] != [
        "opportunity_start",
        "ordinary_nominal_event_if_available",
        "opportunity_end",
    ]:
        raise MotionRecipeError("candidate ranking must be opportunity-start/event/end")
    if search["retained_source_prefix_required"] is not False:
        raise MotionRecipeError("retained source prefix must remain optional")
    if search["retained_source_suffix_required"] is not False:
        raise MotionRecipeError("retained source suffix must remain optional")
    if search.get("historical_adv2c3_role") != "comparator_only_not_default":
        raise MotionRecipeError("adv2c3 may only be a comparator")
    gates = raw["post_build_gates"]
    if not isinstance(gates, list) or tuple(gates) != _POST_BUILD_GATES:
        raise MotionRecipeError("post_build_gates changed or were reordered")


def _validate_output_matrix(raw: Mapping[str, Any]) -> None:
    matrix = raw["required_output_matrix"]
    expected_keys = frozenset({"motion_ids", "scopes", "candidate_count"})
    matrix = _exact_keys(matrix, expected_keys, "required_output_matrix")
    motion_ids = matrix["motion_ids"]
    if not isinstance(motion_ids, list) or any(
        not isinstance(value, str) for value in motion_ids
    ):
        raise MotionRecipeError("required_output_matrix motion_ids must be strings")
    _check_ordered_prefix(tuple(motion_ids), "required_output_matrix motion_ids")
    if matrix["scopes"] != ["upper", "full"]:
        raise MotionRecipeError("required_output_matrix scopes must be upper/full")
    expected_candidates = len(motion_ids) * 2
    if matrix["candidate_count"] != expected_candidates:
        raise MotionRecipeError(
            "required_output_matrix candidate_count must be "
            f"{expected_candidates} (motions x scopes)"
        )


def load_canonical_motion_recipe(
    recipe_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> CanonicalMotionRecipe:
    """Load and close all input-integrity claims in one immutable recipe."""

    path = Path(recipe_path).resolve()
    if not path.is_file():
        raise MotionRecipeError(f"recipe does not exist: {path}")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else path.parents[1].resolve()
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MotionRecipeError("recipe path is outside repo_root") from exc
    try:
        recipe_bytes = path.read_bytes()
    except OSError as exc:
        raise MotionRecipeError(f"cannot read recipe {path}: {exc}") from exc
    raw = _exact_keys(
        _strict_json_bytes(recipe_bytes, "recipe"),
        _TOP_LEVEL_KEYS,
        "recipe",
    )
    _validate_recipe_contract(raw)
    _validate_output_matrix(raw)

    marker_contract = _exact_keys(
        raw["marker_authority"],
        _MARKER_AUTHORITY_KEYS,
        "marker_authority",
    )
    marker_repo_path = _nonempty_string(
        marker_contract["path"], "marker_authority.path"
    )
    # The profile is chosen from the path the CALLER's recipe pins, and that
    # path must be one of the registered authorities.  The authority document
    # never gets to nominate its own validation regime.
    marker_profile = MARKER_AUTHORITY_PROFILE_BY_PATH.get(marker_repo_path)
    if marker_profile is None:
        raise MotionRecipeError(
            "marker_authority.path must be a registered marker authority "
            f"({sorted(MARKER_AUTHORITY_PROFILE_BY_PATH)}); got {marker_repo_path!r}"
        )
    marker_path, marker_sha = _check_bound_file(
        root,
        marker_repo_path,
        marker_contract["sha256"],
        f"marker authority {marker_profile}",
    )
    try:
        marker_semantics = load_canonical_motion_markers(
            marker_path,
            expected_authority_sha256=marker_sha,
            repo_root=root,
            profile=marker_profile,
        )
    except ValueError as exc:
        raise MotionRecipeError(
            f"marker authority {marker_profile} is invalid: {exc}"
        ) from exc

    ready, ready_provenance, ready_contract = _load_canonical_ready_contract(
        root,
        raw["canonical_ready"],
    )

    model = _exact_keys(raw["model_contract"], _MODEL_KEYS, "model_contract")
    model_paths: dict[str, Path] = {}
    model_hashes: dict[str, str] = {}
    for name in ("mjcf", "urdf", "body_order"):
        bound_path, bound_sha = _check_bound_file(
            root, model[f"{name}_path"], model[f"{name}_sha256"], name
        )
        model_paths[name] = bound_path
        model_hashes[name] = bound_sha
    if (
        ready_provenance.mode == _GROUNDED_READY_PROVENANCE_MODE
        and ready_provenance.grounded_mjcf_sha256 != model_hashes["mjcf"]
    ):
        raise MotionRecipeError(
            "grounded-neutral ready evidence binds a different recipe MJCF"
        )

    specs = raw["motion_specs"]
    if not isinstance(specs, list) or len(specs) < len(_REQUIRED_MOTION_IDS):
        raise MotionRecipeError(
            "motion_specs must contain at least the canonical five motions"
        )
    declared_matrix_ids = tuple(raw["required_output_matrix"]["motion_ids"])
    sources: list[MotionSource] = []
    seen: set[str] = set()
    for index, raw_spec in enumerate(specs):
        label = f"motion_specs[{index}]"
        if not isinstance(raw_spec, Mapping):
            raise MotionRecipeError(f"{label} must be an object")
        motion_id = _nonempty_string(raw_spec.get("motion_id"), f"{label}.motion_id")
        expected_keys = (
            _MOTION_BASE_KEYS | _MOTION_SYNTHETIC_EXTRA_KEYS
            if motion_id == "fh_block_syn"
            else _MOTION_BASE_KEYS
        )
        spec = _exact_keys(raw_spec, expected_keys, label)
        if motion_id in seen:
            raise MotionRecipeError(f"duplicate motion_id {motion_id!r}")
        seen.add(motion_id)
        source_path, source_sha = _check_bound_file(
            root, spec["source_path"], spec["source_sha256"], f"{motion_id} source"
        )
        try:
            clip = load_motion(source_path)
        except (OSError, ValueError) as exc:
            raise MotionRecipeError(
                f"{motion_id} source is not exact schema-2: {exc}"
            ) from exc
        _nonempty_string(spec["human_role"], f"{motion_id} human_role")
        authority_row = marker_semantics.row(motion_id)
        source_repo_path = source_path.relative_to(root).as_posix()
        if (
            source_repo_path != authority_row.bound_recipe_source_path
            or source_sha != authority_row.bound_recipe_source_sha256
        ):
            raise MotionRecipeError(
                f"{motion_id} source does not close against marker authority v2"
            )
        # One place asks the authority which frames it asserts, so a new
        # provenance kind cannot slip past the range check by living in a
        # field this loop never learned to read.
        marker_frames = authority_row.authority_frames()
        if not marker_frames:
            raise MotionRecipeError(
                f"{motion_id} marker authority row asserts no source frame at all"
            )
        if any(frame >= clip.n_frames for frame in marker_frames):
            raise MotionRecipeError(
                f"{motion_id} marker authority frame exceeds source frames"
            )
        anchor, anchor_basis = authority_row.contact_anchor()
        if anchor is None:
            raise MotionRecipeError(
                f"{motion_id} marker authority row resolves no contact anchor"
            )
        window, window_basis = authority_row.search_window()
        if window is None:
            raise MotionRecipeError(
                f"{motion_id} marker authority row resolves no search window "
                f"(anchor basis {anchor_basis!r})"
            )
        if window[1] >= clip.n_frames:
            raise MotionRecipeError(f"{motion_id} {window_basis} exceeds source frames")
        if not isinstance(spec["scope_overrides"], Mapping):
            raise MotionRecipeError(f"{motion_id} scope_overrides must be an object")
        if motion_id == "s0_highpress":
            overrides = _exact_keys(
                spec["scope_overrides"], _S0_OVERRIDE_KEYS, "s0 scope_overrides"
            )
            full_override = _exact_keys(
                overrides["full"],
                _S0_FULL_OVERRIDE_KEYS,
                "s0 scope_overrides.full",
            )
            if full_override["grounding_policy"] != (
                "minimum_constant_z_offset_for_1mm_clearance"
            ):
                raise MotionRecipeError("s0 grounding policy changed")
            maximum_grounding = full_override["maximum_grounding_offset_m"]
            if (
                isinstance(maximum_grounding, bool)
                or not isinstance(maximum_grounding, (int, float))
                or not math.isfinite(float(maximum_grounding))
                or float(maximum_grounding) <= 0.0
            ):
                raise MotionRecipeError("s0 maximum grounding offset is invalid")
        elif spec["scope_overrides"]:
            raise MotionRecipeError(f"{motion_id} has an unapproved scope override")

        face: Mapping[str, Any] | None = None
        if motion_id == "fh_block_syn":
            face = _exact_keys(spec["face_manifold"], _FACE_KEYS, "face_manifold")
            if face["mode"] != "signed_raw_plus_y_flip":
                raise MotionRecipeError("synthetic block face mode changed")
            if face["active_joints"] != "right_arm_7":
                raise MotionRecipeError(
                    "synthetic block must solve the right-arm manifold"
                )
            if face["site_position"] != "preserve_per_source_frame":
                raise MotionRecipeError(
                    "synthetic block site-position objective changed"
                )
            if face["orientation"] != "normal_hard_inplane_free":
                raise MotionRecipeError("synthetic block orientation objective changed")
            if face["single_axis_pi_overlay_forbidden"] is not True:
                raise MotionRecipeError("single-axis pi overlay must remain forbidden")
            if authority_row.nominal_event is not None:
                raise MotionRecipeError(
                    "synthetic construction may not acquire a nominal event"
                )
            if authority_row.preferred_seed is not None:
                raise MotionRecipeError(
                    "synthetic construction may not acquire a preferred seed"
                )
            if authority_row.construction_marker is None:
                raise MotionRecipeError(
                    "synthetic face solve requires a construction marker"
                )
        sources.append(
            MotionSource(
                motion_id=motion_id,
                human_role=str(spec["human_role"]),
                path=source_path,
                sha256=source_sha,
                clip=clip,
                face_manifold=face,
                scope_overrides=spec["scope_overrides"],
            )
        )

    spec_ids = tuple(row.motion_id for row in sources)
    _check_ordered_prefix(spec_ids, "motion_specs")
    if spec_ids != declared_matrix_ids:
        raise MotionRecipeError(
            "motion_specs and required_output_matrix.motion_ids disagree: "
            f"{list(spec_ids)} vs {list(declared_matrix_ids)}"
        )

    if ready_provenance.mode == "legacy_donor_frame_exact_v1":
        donor_id = _nonempty_string(
            ready_contract["donor_motion_id"], "ready donor_motion_id"
        )
        donor_frame = _integer(
            ready_contract["donor_source_frame"], "ready donor_source_frame"
        )
        donor_matches = [row for row in sources if row.motion_id == donor_id]
        if len(donor_matches) != 1:
            raise MotionRecipeError("canonical ready donor is not one recipe source")
        donor = donor_matches[0]
        donor_sha = _sha256(
            ready_contract["donor_source_sha256"], "ready donor_source_sha256"
        )
        if donor.sha256 != donor_sha:
            raise MotionRecipeError("canonical ready donor source SHA does not close")
        if donor_frame >= donor.clip.n_frames:
            raise MotionRecipeError("canonical ready donor frame exceeds source")
        if ready.source_segment != donor_id or ready.source_frame != donor_frame:
            raise MotionRecipeError("ready file donor metadata disagrees with recipe")
        if not np.array_equal(ready.joint_pos, donor.clip.joint_pos[donor_frame]):
            raise MotionRecipeError("ready joint pose is not donor frame exact")
        pelvis_index = RUNTIME_BODY_NAMES.index("pelvis_link")
        if not np.array_equal(
            ready.root_pos_w, donor.clip.body_pos_w[donor_frame, pelvis_index]
        ):
            raise MotionRecipeError(
                "ready root position is not donor pelvis frame exact"
            )
        if not np.array_equal(
            ready.root_quat_wxyz, donor.clip.body_quat_w[donor_frame, pelvis_index]
        ):
            raise MotionRecipeError(
                "ready root quaternion is not donor pelvis frame exact"
            )

    synthetic = next(row for row in sources if row.motion_id == "fh_block_syn")
    backhand = next(row for row in sources if row.motion_id == "bh_block")
    if synthetic.sha256 != backhand.sha256 or synthetic.path != backhand.path:
        raise MotionRecipeError(
            "synthetic and backhand block must bind the same source bytes"
        )

    return CanonicalMotionRecipe(
        path=path,
        repo_root=root,
        raw=raw,
        ready=ready,
        sources=tuple(sources),
        marker_semantics=marker_semantics,
        marker_authority_path=marker_path,
        marker_authority_sha256=marker_sha,
        model_paths=model_paths,
        model_hashes=model_hashes,
    )


__all__ = [
    "CanonicalMotionRecipe",
    "MotionRecipeError",
    "MotionSource",
    "ReadyState",
    "load_canonical_motion_recipe",
    "sha256_file",
]
