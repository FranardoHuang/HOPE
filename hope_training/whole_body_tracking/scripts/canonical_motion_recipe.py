#!/usr/bin/env python3
"""Strict loader for canonical motion-library recipe schema 2.

This module deliberately stops before geometry generation.  Its job is to make
the compiler inputs unambiguous:

* the recipe and every nested object use exact keys;
* every path is repository-relative and may not escape the repository;
* model, source-motion, and canonical-ready bytes match their bound SHA-256;
* motion sources are exact schema-2 files;
* the ready file has one exact, zero-velocity state; and
* the ready donor claim is verified against the bound donor source bytes;
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
    MARKER_AUTHORITY_PATH,
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
_REQUIRED_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
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
    if not isinstance(motion_ids, list) or tuple(motion_ids) != _REQUIRED_MOTION_IDS:
        raise MotionRecipeError(
            "required_output_matrix motion_ids must be the canonical ordered five"
        )
    if matrix["scopes"] != ["upper", "full"]:
        raise MotionRecipeError("required_output_matrix scopes must be upper/full")
    if matrix["candidate_count"] != 10:
        raise MotionRecipeError("required_output_matrix candidate_count must be 10")


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
    if marker_repo_path != MARKER_AUTHORITY_PATH:
        raise MotionRecipeError(
            f"marker_authority.path must equal {MARKER_AUTHORITY_PATH!r}"
        )
    marker_path, marker_sha = _check_bound_file(
        root,
        marker_repo_path,
        marker_contract["sha256"],
        "marker authority v2",
    )
    try:
        marker_semantics = load_canonical_motion_markers(
            marker_path,
            expected_authority_sha256=marker_sha,
            repo_root=root,
        )
    except ValueError as exc:
        raise MotionRecipeError(f"marker authority v2 is invalid: {exc}") from exc

    ready_contract = _exact_keys(
        raw["canonical_ready"], _READY_RECIPE_KEYS, "canonical_ready"
    )
    if ready_contract["endpoint_velocity_policy"] != (
        "all_joint_root_body_velocities_exact_zero"
    ):
        raise MotionRecipeError("canonical ready endpoint velocity policy changed")
    ready_path, ready_sha = _check_bound_file(
        root,
        ready_contract["path"],
        ready_contract["sha256"],
        "canonical ready",
    )
    ready = _load_ready(ready_path, ready_sha)

    model = _exact_keys(raw["model_contract"], _MODEL_KEYS, "model_contract")
    model_paths: dict[str, Path] = {}
    model_hashes: dict[str, str] = {}
    for name in ("mjcf", "urdf", "body_order"):
        bound_path, bound_sha = _check_bound_file(
            root, model[f"{name}_path"], model[f"{name}_sha256"], name
        )
        model_paths[name] = bound_path
        model_hashes[name] = bound_sha

    specs = raw["motion_specs"]
    if not isinstance(specs, list) or len(specs) != len(_REQUIRED_MOTION_IDS):
        raise MotionRecipeError("motion_specs must contain exactly five motions")
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
        marker_frames = [
            *authority_row.ge50_seed,
            *authority_row.ge80_seed,
            authority_row.historical_adv2c3_start,
        ]
        if authority_row.nominal_event is not None:
            marker_frames.append(authority_row.nominal_event)
        if authority_row.preferred_seed is not None:
            marker_frames.append(authority_row.preferred_seed)
        if authority_row.construction_marker is not None:
            marker_frames.extend(
                (
                    authority_row.construction_marker.annotation_frame,
                    authority_row.construction_marker.donor_preferred_frame,
                    *authority_row.construction_marker.solve_span,
                )
            )
        if any(frame >= clip.n_frames for frame in marker_frames):
            raise MotionRecipeError(
                f"{motion_id} marker authority frame exceeds source frames"
            )
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

    if tuple(row.motion_id for row in sources) != _REQUIRED_MOTION_IDS:
        raise MotionRecipeError("motion_specs order or membership changed")

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
        raise MotionRecipeError("ready root position is not donor pelvis frame exact")
    if not np.array_equal(
        ready.root_quat_wxyz, donor.clip.body_quat_w[donor_frame, pelvis_index]
    ):
        raise MotionRecipeError("ready root quaternion is not donor pelvis frame exact")

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
