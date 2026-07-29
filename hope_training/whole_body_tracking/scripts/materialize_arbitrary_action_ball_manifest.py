#!/usr/bin/env python3
"""Materialize one exact arbitrary-N compiled bank as ActionBall schema v3.

This is the post-compile identity boundary for a standalone arbitrary-N bank.
It does not choose actions, compile motions, promote motion admission, launch
Isaac, or grant training/deployment/hardware authority.  It only:

1. reopens an exact generic-v2 bank report and schema-v2 scoped registry;
2. follows the arbitrary recipe to its source capsule and the capsule's exact
   ``inputs.action_manifest`` binding;
3. replaces every raw source motion identity with the selected compiled
   scope, preserves source action order and ball-domain semantics, and derives
   a fresh action UID from the compiled bytes;
4. binds one exact N-row schema-v2 stroke prototype plus formal solver/physics
   profile pins; and
5. validates the result through the production ActionBall loader and
   referenced-asset verifier before publishing a no-clobber manifest/receipt
   pair.

The source ActionBall manifest is deliberately a *profile authority*, not a
compiled-motion authority.  Its raw motion path/SHA and timing are checked
against the source capsule, then rejected as final output identities.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPTS_DIR.parents[2]
MDP_DIR_REL = (
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
SCOPES = ("upper", "full")
PROFILE_PINS_SCHEMA_VERSION = 1
PROFILE_PINS_KIND = "whole_body_tracking.action_ball.profile_pins"
FORMAL_SOURCE_AUTHORITY = "external_exact_commit_subset_blob_map_v1"
FORMAL_COMMIT_BINDING = "external_preexec_immutable_launch_capsule_v1"
RECEIPT_TYPE = "arbitrary_action_ball_manifest_materialization_v1"

_SHA256_CHARS = frozenset("0123456789abcdef")
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "verdict",
        "bank_gate_pass",
        "candidate_integrity_pass",
        "grounded_trace_status",
        "publication_class",
        "training_authorized",
        "hardware_authorized",
        "library_id",
        "manifest",
        "bank_dir",
        "bound_inputs",
        "contracts",
        "aggregate",
        "clips",
        "non_claims",
        "selected_registry_binding",
    }
)
_BINDING_KEYS = frozenset({"path", "sha256"})
_PROFILE_PINS_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "source_authority",
        "cfg",
        "geometry",
        "venue_yaml",
        "venue_yaml_sha256",
        "planes",
        "solver_implementation_source_sha256",
        "contact_geometry",
        "physics_profile_sha256",
        "solver_profile_sha256",
        "physics_payload",
        "solver_payload",
    }
)


class MaterializationError(RuntimeError):
    """The final ActionBall identity cannot be closed without weakening it."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise MaterializationError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return value


def _exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterializationError(f"{label} must be one JSON object")
    actual = frozenset(value)
    if actual != expected:
        raise MaterializationError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MaterializationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MaterializationError(
            f"{label} contains forbidden JSON constant {value}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except MaterializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MaterializationError(f"{label} must contain one JSON object")
    return value


def _strict_json_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    expected = _digest(expected_sha256, f"{label} expected SHA-256")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MaterializationError(f"cannot read {label}: {exc}") from exc
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise MaterializationError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return _strict_json_bytes(payload, label), payload


def _root(path: os.PathLike[str] | str) -> Path:
    try:
        result = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError(
            f"repo_root does not resolve: {exc}"
        ) from exc
    if not result.is_dir():
        raise MaterializationError("repo_root must be a directory")
    return result


def _inside_root(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterializationError(
            f"{label} must resolve inside repo_root"
        ) from exc
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise MaterializationError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise MaterializationError(f"{label} must be a regular file")
    return resolved


def _input_path(
    value: os.PathLike[str] | str, root: Path, label: str
) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return _inside_root(candidate, root, label)


def _bound_path(
    binding: Any,
    *,
    root: Path,
    label: str,
    relative_base: Path | None = None,
) -> tuple[Path, str]:
    row = _exact_keys(binding, _BINDING_KEYS, label)
    path_value = row["path"]
    if not isinstance(path_value, str) or not path_value:
        raise MaterializationError(f"{label}.path must be non-empty")
    expected = _digest(row["sha256"], f"{label}.sha256")
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        pure = PurePosixPath(path_value)
        if (
            "\\" in path_value
            or pure.is_absolute()
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise MaterializationError(
                f"{label}.path must be normalized and traversal-free"
            )
        candidate = (relative_base or root).joinpath(*pure.parts)
    path = _inside_root(candidate, root, f"{label}.path")
    actual = _sha256_file(path)
    if actual != expected:
        raise MaterializationError(
            f"{label} bytes drifted: expected {expected}, got {actual}"
        )
    return path, actual


def _repo_relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterializationError(
            f"{label} does not live inside repo_root"
        ) from exc


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and Path(existing_file).resolve() == path.resolve():
            return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializationError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_modules(root: Path):
    mdp = root / MDP_DIR_REL
    if not mdp.is_dir():
        raise MaterializationError(f"missing runtime MDP directory: {mdp}")
    if str(mdp) not in sys.path:
        sys.path.insert(0, str(mdp))
    manifest = _load_module(
        "_materialize_arbitrary_action_ball_manifest_schema",
        mdp / "action_ball_manifest.py",
    )
    return manifest


def _canonical_profile_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaterializationError(
            f"profile payload is not canonical ASCII JSON: {exc}"
        ) from exc
    return _sha256_bytes(payload)


def _validate_profile_pins(
    document: Mapping[str, Any],
) -> tuple[str, str]:
    row = _exact_keys(document, _PROFILE_PINS_KEYS, "profile pins")
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != PROFILE_PINS_SCHEMA_VERSION
        or row["kind"] != PROFILE_PINS_KIND
    ):
        raise MaterializationError(
            "profile pins schema_version/kind is unsupported"
        )
    authority = _exact_keys(
        row["source_authority"],
        frozenset(
            {
                "schema_version",
                "authority",
                "commit_binding",
                "embedded_commit",
                "source_blob_map_sha256",
            }
        ),
        "profile pins source_authority",
    )
    if (
        authority["schema_version"] != 1
        or authority["authority"] != FORMAL_SOURCE_AUTHORITY
        or authority["commit_binding"] != FORMAL_COMMIT_BINDING
        or authority["embedded_commit"] is not False
    ):
        raise MaterializationError(
            "profile pins must carry formal external exact-commit authority"
        )
    _digest(
        authority["source_blob_map_sha256"],
        "profile pins source_blob_map_sha256",
    )
    physics = _digest(
        row["physics_profile_sha256"],
        "physics_profile_sha256",
    )
    solver = _digest(
        row["solver_profile_sha256"],
        "solver_profile_sha256",
    )
    if not isinstance(row["physics_payload"], Mapping):
        raise MaterializationError("physics_payload must be one JSON object")
    if not isinstance(row["solver_payload"], Mapping):
        raise MaterializationError("solver_payload must be one JSON object")
    if _canonical_profile_sha256(row["physics_payload"]) != physics:
        raise MaterializationError(
            "physics_profile_sha256 does not hash physics_payload"
        )
    if _canonical_profile_sha256(row["solver_payload"]) != solver:
        raise MaterializationError(
            "solver_profile_sha256 does not hash solver_payload"
        )
    solver_physics = row["solver_payload"].get("physics_profile_sha256")
    if solver_physics is not None and solver_physics != physics:
        raise MaterializationError(
            "solver_payload physics_profile_sha256 differs from physics pin"
        )
    return solver, physics


def _runtime_style_racket_site_speed(
    npz_path: Path,
    *,
    contact_frame: int,
    fps: float,
    window: int = 2,
) -> float:
    """Match runtime's float32 clamped +/-2-frame official-site FD."""

    import numpy as np

    if not math.isclose(float(fps), 50.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise MaterializationError(
            f"{npz_path}: ActionBall compiled motion fps must be exactly 50"
        )
    try:
        data = np.load(str(npz_path), allow_pickle=False)
    except Exception as exc:
        raise MaterializationError(
            f"cannot load compiled motion {npz_path}: {exc}"
        ) from exc
    try:
        names = tuple(str(value) for value in data["body_names"])
        wrist_index = names.index("right_wrist_yaw_Link")
        positions = np.asarray(
            data["body_pos_w"], dtype=np.float32
        )[:, wrist_index]
        quaternions = np.asarray(
            data["body_quat_w"], dtype=np.float32
        )[:, wrist_index]
    except (KeyError, ValueError, IndexError) as exc:
        raise MaterializationError(
            f"{npz_path}: compiled motion lacks runtime wrist state"
        ) from exc
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or quaternions.shape != (positions.shape[0], 4)
        or not 0 <= contact_frame < positions.shape[0]
    ):
        raise MaterializationError(
            f"{npz_path}: invalid wrist arrays/contact frame"
        )
    offset = np.asarray(
        [0.21021, 0.032078, 0.032036], dtype=np.float32
    )

    def blade(frame: int):
        frame = min(max(frame, 0), positions.shape[0] - 1)
        w, x, y, z = (quaternions[frame, index] for index in range(4))
        rotation = np.asarray(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=np.float32,
        )
        return positions[frame] + rotation @ offset

    difference = blade(contact_frame + window) - blade(
        contact_frame - window
    )
    velocity = difference / np.float32(
        2.0 * float(window) * (1.0 / float(fps))
    )
    speed = float(np.linalg.norm(velocity.astype(np.float32)))
    if not math.isfinite(speed) or speed <= 0.0:
        raise MaterializationError(
            f"{npz_path}: compiled racket-site speed is not positive"
        )
    return speed


def _rebase_ttc(
    *,
    profile: Mapping[str, Any],
    reference_t_hit_s: float,
    reaction_margin_s: float,
    teacher_rate_min: float,
    teacher_rate_max: float,
    label: str,
) -> tuple[dict[str, Any], float]:
    """Preserve asymmetric curriculum widths under compiled contact timing."""

    result = deepcopy(dict(profile))
    lower_initial = float(
        result["time_to_contact_std_lower_initial_s"]
    )
    lower_max = float(result["time_to_contact_std_lower_max_s"])
    upper_initial = float(
        result["time_to_contact_std_upper_initial_s"]
    )
    upper_max = float(result["time_to_contact_std_upper_max_s"])
    if (
        lower_initial < 0.0
        or upper_initial < 0.0
        or lower_max <= 0.0
        or upper_max <= 0.0
        or lower_initial > lower_max
        or upper_initial > upper_max
    ):
        raise MaterializationError(
            f"{label}: source TTC asymmetric widths are invalid"
        )
    total_width = lower_max + upper_max
    denominator = (
        reference_t_hit_s / teacher_rate_max
        + 1.0
        - total_width
        - reaction_margin_s
    )
    if denominator <= 0.0:
        raise MaterializationError(
            f"{label}: compiled t_hit cannot retain the source TTC envelope"
        )
    required_rate_min = reference_t_hit_s / denominator
    adjusted_rate_min = max(
        teacher_rate_min,
        required_rate_min * (1.0 + 1.0e-12),
    )
    if adjusted_rate_min > 1.0 + 1.0e-12:
        raise MaterializationError(
            f"{label}: compiled t_hit requires teacher_rate_min "
            f"{adjusted_rate_min:.9g} > 1"
        )
    adjusted_rate_min = min(adjusted_rate_min, 1.0)
    feasible_min = (
        reference_t_hit_s / adjusted_rate_min + reaction_margin_s
    )
    feasible_max = reference_t_hit_s / teacher_rate_max + 1.0
    center_low = feasible_min + lower_max
    center_high = feasible_max - upper_max
    if center_low > center_high + 1.0e-10:
        raise MaterializationError(
            f"{label}: no TTC center retains both asymmetric sides"
        )
    center = 0.5 * (center_low + center_high)
    result["time_to_contact_center_s"] = center
    result["time_to_contact_min_s"] = center - lower_max
    result["time_to_contact_max_s"] = center + upper_max
    return result, adjusted_rate_min


def _assert_compiled_identity(
    *,
    source_path: Path,
    source_manifest_path: str,
    source_sha256: str,
    compiled_path: Path,
    compiled_manifest_path: str,
    compiled_sha256: str,
    label: str,
) -> None:
    """Reject a final row that retained any raw source path or byte identity."""

    if (
        compiled_path.resolve() == source_path.resolve()
        or compiled_manifest_path == source_manifest_path
        or compiled_sha256 == source_sha256
    ):
        raise MaterializationError(
            f"{label}: final identity still points at raw source motion"
        )


def _publish_pair_no_clobber(
    *,
    manifest_path: Path,
    manifest_bytes: bytes,
    receipt_path: Path,
    receipt_bytes: bytes,
) -> None:
    if manifest_path == receipt_path:
        raise MaterializationError(
            "manifest and receipt output paths must differ"
        )
    for path, label in (
        (manifest_path, "manifest output"),
        (receipt_path, "receipt output"),
    ):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {label}: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    created_manifest = False
    try:
        descriptor = os.open(manifest_path, flags, 0o644)
        created_manifest = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(receipt_path, flags, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(receipt_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created_manifest:
            try:
                manifest_path.unlink()
            except OSError:
                pass
        raise


def _output_path(
    value: os.PathLike[str] | str, root: Path, label: str
) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MaterializationError(
            f"{label} must be inside repo_root"
        ) from exc
    return candidate


def materialize(
    *,
    repo_root: os.PathLike[str] | str,
    bank_report_path: os.PathLike[str] | str,
    expected_bank_report_sha256: str,
    registry_path: os.PathLike[str] | str,
    expected_registry_sha256: str,
    scope: str,
    profile_pins_path: os.PathLike[str] | str,
    expected_profile_pins_sha256: str,
    prototype_path: os.PathLike[str] | str,
    expected_prototype_sha256: str,
    manifest_id: str,
    output_path: os.PathLike[str] | str,
    receipt_output_path: os.PathLike[str] | str,
    holdout_samples_per_action: int = 768,
) -> Mapping[str, Any]:
    """Validate all lineages and publish one final no-clobber artifact pair."""

    if scope not in SCOPES:
        raise MaterializationError(f"scope must be one of {SCOPES}")
    if (
        type(holdout_samples_per_action) is not int
        or holdout_samples_per_action < 768
    ):
        raise MaterializationError(
            "holdout_samples_per_action must be an integer >= 768"
        )
    if not isinstance(manifest_id, str) or not manifest_id:
        raise MaterializationError("manifest_id must be non-empty")

    root = _root(repo_root)
    report_path = _input_path(bank_report_path, root, "bank report")
    report, _report_bytes = _strict_json_file(
        report_path,
        expected_sha256=expected_bank_report_sha256,
        label="generic bank report",
    )
    report = _exact_keys(report, _REPORT_KEYS, "generic bank report")
    if (
        report["schema_version"] != 2
        or report["verdict"] != "PASS"
        or report["bank_gate_pass"] is not True
        or report["candidate_integrity_pass"] is not True
        or report["grounded_trace_status"] != "COMPLETE_PASS"
        or report["publication_class"] != "post_build_diagnostic_only"
        or report["training_authorized"] is not False
        or report["hardware_authorized"] is not False
    ):
        raise MaterializationError(
            "generic bank report is not an exact complete diagnostic PASS"
        )

    build_manifest_path, build_manifest_sha = _bound_path(
        report["manifest"],
        root=root,
        label="generic bank report manifest",
    )
    build_manifest, _ = _strict_json_file(
        build_manifest_path,
        expected_sha256=build_manifest_sha,
        label="arbitrary-N build manifest",
    )
    recipe_path, recipe_sha = _bound_path(
        build_manifest.get("recipe"),
        root=root,
        label="arbitrary-N build recipe",
    )
    bank_dir_value = report["bank_dir"]
    if not isinstance(bank_dir_value, str) or not bank_dir_value:
        raise MaterializationError("generic bank report bank_dir is invalid")
    bank_dir_candidate = Path(bank_dir_value).expanduser()
    if not bank_dir_candidate.is_absolute():
        bank_dir_candidate = root / bank_dir_candidate
    try:
        bank_dir = bank_dir_candidate.resolve(strict=True)
        bank_dir.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterializationError(
            "generic bank directory must resolve inside repo_root"
        ) from exc
    if not bank_dir.is_dir():
        raise MaterializationError("generic bank_dir is not a directory")

    for directory in (SCRIPTS_DIR,):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import canonical_motion_admission as admission
    import canonical_motion_arbitrary_bank as arbitrary
    import canonical_motion_generic_bank_gate as generic
    import canonical_motion_registry as registry_module

    try:
        loaded = arbitrary.load_arbitrary_bank_recipe(
            recipe_path, repo_root=root
        )
        if loaded.sha256 != recipe_sha:
            raise MaterializationError(
                "arbitrary recipe changed after exact binding was opened"
            )
        manifest, matrix = generic._manifest_and_matrix(
            loaded, build_manifest_path, bank_dir
        )
        registry = registry_module.load_canonical_motion_bank_registry(
            registry_path,
            repo_root=root,
            expected_registry_sha256=expected_registry_sha256,
        )
        if (
            registry.schema_version != 2
            or registry.bank_id != loaded.raw["bank_id"]
            or registry.motion_ids != loaded.motion_ids
            or registry.canonical_ready_sha256
            != loaded.canonical_recipe.ready.sha256
        ):
            raise MaterializationError(
                "registry bank/order/ready differs from arbitrary recipe"
            )
        for entry in registry.entries:
            matrix_row, matrix_path = matrix[
                (entry.motion_id, registry.scope)
            ]
            if (
                entry.npz_path.resolve() != matrix_path.resolve()
                or entry.npz_sha256
                != matrix_row["output_npz_sha256"]
            ):
                raise MaterializationError(
                    f"registry row {entry.motion_id!r} differs from the "
                    "selected compiler output"
                )
        promotion_binding = registry_module.bank_promotion_binding(
            registry, authorization_purpose="training"
        )
        selected = {
            "scope": promotion_binding.scope,
            "registry_sha256": promotion_binding.registry_sha256,
            "alignment_sha256": promotion_binding.alignment_sha256,
            "canonical_ready_sha256": (
                promotion_binding.canonical_ready_sha256
            ),
            "canonical_ready_fk_sha256": (
                promotion_binding.canonical_ready_fk_sha256
            ),
            "motion_ids": list(promotion_binding.motion_ids),
            "npz_sha256": list(promotion_binding.npz_sha256),
            "build_manifest_sha256": list(
                promotion_binding.build_manifest_sha256
            ),
        }
    except MaterializationError:
        raise
    except Exception as exc:
        raise MaterializationError(
            f"arbitrary bank/registry validation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if registry.schema_version != 2 or registry.scope != scope:
        raise MaterializationError(
            "registry must be schema v2 and match the requested scope"
        )
    if dict(selected) != report["selected_registry_binding"]:
        raise MaterializationError(
            "bank report selected_registry_binding differs from registry"
        )
    matrix_order = tuple(
        (motion_id, matrix_scope)
        for motion_id in loaded.motion_ids
        for matrix_scope in SCOPES
    )
    all_npz_sha256 = tuple(
        matrix[key][0]["output_npz_sha256"] for key in matrix_order
    )
    try:
        admission._validate_bank_gate_report(
            {
                "path": _repo_relative(
                    report_path, root, "generic bank report"
                ),
                "sha256": expected_bank_report_sha256,
            },
            binding=promotion_binding,
            repo_root=root,
            expected_report_schema_version=2,
            expected_clip_count=2 * len(loaded.motion_ids),
            report_profile="generic_v2",
            expected_manifest_sha256=build_manifest_sha,
            expected_all_npz_sha256=all_npz_sha256,
        )
    except Exception as exc:
        raise MaterializationError(
            f"generic bank report formal validation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    pins_path = _input_path(
        profile_pins_path, root, "profile pins"
    )
    pins, _ = _strict_json_file(
        pins_path,
        expected_sha256=expected_profile_pins_sha256,
        label="profile pins",
    )
    solver_sha, physics_sha = _validate_profile_pins(pins)

    prototype_file = _input_path(
        prototype_path, root, "stroke prototype"
    )
    prototype_sha = _digest(
        expected_prototype_sha256,
        "expected prototype SHA-256",
    )
    if _sha256_file(prototype_file) != prototype_sha:
        raise MaterializationError("stroke prototype bytes drifted")
    manifest_module = _runtime_modules(root)
    prototype_document, _ = _strict_json_file(
        prototype_file,
        expected_sha256=prototype_sha,
        label="stroke prototype",
    )
    scopes = prototype_document.get("scopes")
    if not isinstance(scopes, Mapping):
        raise MaterializationError("stroke prototype scopes is invalid")
    prototype_rows = scopes.get(scope)
    if (
        not isinstance(prototype_rows, list)
        or len(prototype_rows) != len(loaded.motion_ids)
    ):
        raise MaterializationError(
            "selected prototype scope must contain exactly N ordered rows"
        )
    prototype_ids = tuple(
        row.get("motion_id") if isinstance(row, Mapping) else None
        for row in prototype_rows
    )
    prototype_hashes = tuple(
        row.get("npz_sha256") if isinstance(row, Mapping) else None
        for row in prototype_rows
    )
    prototype_indices = tuple(
        row.get("clip_index") if isinstance(row, Mapping) else None
        for row in prototype_rows
    )
    if prototype_ids != loaded.motion_ids:
        raise MaterializationError(
            "selected prototype scope clips differ from source action order"
        )
    if prototype_hashes != tuple(
        entry.npz_sha256 for entry in registry.entries
    ):
        raise MaterializationError(
            "selected prototype scope NPZ SHA order differs from compiled "
            "registry bytes"
        )
    if prototype_indices != tuple(range(len(loaded.motion_ids))):
        raise MaterializationError(
            "selected prototype scope clip_index order differs from actions"
        )

    capsule, _ = _strict_json_file(
        loaded.source_capsule_path,
        expected_sha256=loaded.source_capsule_sha256,
        label="source capsule",
    )
    inputs = capsule.get("inputs")
    if not isinstance(inputs, Mapping):
        raise MaterializationError(
            "source capsule does not bind inputs.action_manifest"
        )
    source_manifest_path, source_manifest_sha = _bound_path(
        inputs.get("action_manifest"),
        root=root,
        relative_base=loaded.source_capsule_path.parent,
        label="source capsule inputs.action_manifest",
    )
    source_document, _ = _strict_json_file(
        source_manifest_path,
        expected_sha256=source_manifest_sha,
        label="source ActionBall manifest",
    )
    if "counter_rally_objective" in source_document:
        raise MaterializationError(
            "arbitrary-N materialization cannot inherit an N=1 objective"
        )
    source_validation = deepcopy(dict(source_document))
    source_curriculum = source_validation.get("curriculum")
    source_holdout = source_validation.get("holdout")
    if not isinstance(source_curriculum, Mapping) or not isinstance(
        source_holdout, Mapping
    ):
        raise MaterializationError(
            "source ActionBall manifest curriculum/holdout is invalid"
        )
    source_holdout["samples_per_action"] = max(
        768,
        int(source_holdout.get("samples_per_action", 0)),
        int(source_curriculum.get("min_proposals", 0)),
        int(source_curriculum.get("min_safe_closed", 0)),
    )
    try:
        source_manifest = manifest_module.ActionBallManifest.from_mapping(
            source_validation
        )
    except Exception as exc:
        raise MaterializationError(
            f"source ActionBall profile manifest is invalid: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if source_manifest.action_order != loaded.motion_ids:
        raise MaterializationError(
            "source ActionBall action order differs from arbitrary recipe"
        )
    if len(registry.entries) != len(loaded.motion_ids):
        raise MaterializationError(
            "selected registry contains a missing or extra action"
        )

    output_actions: list[dict[str, Any]] = []
    receipt_actions: list[dict[str, Any]] = []
    source_actions = tuple(source_manifest.actions)
    for index, (
        motion_id,
        source_timing,
        source_action,
        entry,
        prototype_row,
    ) in enumerate(
        zip(
            loaded.motion_ids,
            loaded.source_timings,
            source_actions,
            registry.entries,
            prototype_rows,
        )
    ):
        label = f"action[{index}] {motion_id}"
        if (
            source_timing.motion_id != motion_id
            or source_action.action_id != motion_id
            or entry.motion_id != motion_id
            or not isinstance(prototype_row, Mapping)
            or prototype_row.get("motion_id") != motion_id
            or prototype_row.get("clip_index") != index
        ):
            raise MaterializationError(
                f"{label}: source/registry/prototype order drifted"
            )
        source_motion_path = _input_path(
            source_action.motion_path, root, f"{label} source motion"
        )
        if (
            source_motion_path != source_timing.source_path.resolve()
            or source_action.motion_sha256 != source_timing.source_sha256
            or source_action.family != source_timing.family
            or not math.isclose(
                source_action.reference_t_hit_s,
                source_timing.t_hit_s,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                source_action.reference_t_cycle_s,
                source_timing.t_cycle_s,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                source_action.strike_phase,
                source_timing.hit_frame / (source_timing.frames - 1),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise MaterializationError(
                f"{label}: source ActionBall identity/timing differs from capsule"
            )
        compiled_path = entry.npz_path.resolve()
        _assert_compiled_identity(
            source_path=source_timing.source_path,
            source_manifest_path=source_action.motion_path,
            source_sha256=source_action.motion_sha256,
            compiled_path=compiled_path,
            compiled_manifest_path=entry.npz_path_text,
            compiled_sha256=entry.npz_sha256,
            label=label,
        )
        if any(
            not math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            for actual, expected in zip(
                source_action.ball_profile.base_spawn_center_w_xy_m,
                source_timing.base_spawn_center_w_xy_m,
            )
        ):
            raise MaterializationError(
                f"{label}: source ball profile base-spawn center differs "
                "from the source capsule placement"
            )
        row_frames = prototype_row.get("frames")
        contact_frame = prototype_row.get("contact_frame")
        contact_window = prototype_row.get("contact_window_frames")
        if (
            type(row_frames) is not int
            or row_frames != entry.frames
            or type(contact_frame) is not int
            or not 0 < contact_frame < entry.frames - 1
            or contact_window
            != list(entry.contact_opportunity_frames)
            or not (
                entry.contact_opportunity_frames[0]
                <= contact_frame
                <= entry.contact_opportunity_frames[1]
            )
        ):
            raise MaterializationError(
                f"{label}: prototype contact frame/window differs from registry"
            )
        compiled_phase = contact_frame / (entry.frames - 1)
        compiled_t_hit = contact_frame / entry.fps
        compiled_t_cycle = (entry.frames - 1) / entry.fps
        if (
            prototype_row.get("scope") != scope
            or prototype_row.get("npz_sha256") != entry.npz_sha256
            or prototype_row.get("family") != entry.family
            or source_action.family != entry.family
            or float(prototype_row.get("face_sign", 0.0))
            != float(entry.mount_normal_sign)
            or int(entry.mount_normal_sign)
            != source_action.mount_normal_sign
            or not math.isclose(
                float(prototype_row.get("strike_phase", -1.0)),
                compiled_phase,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(prototype_row.get("t_prepare_s", -1.0)),
                compiled_t_hit,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        ):
            raise MaterializationError(
                f"{label}: prototype identity/family/face/timing drifted"
            )
        ball_profile, rate_min = _rebase_ttc(
            profile=source_action.ball_profile.to_mapping(),
            reference_t_hit_s=compiled_t_hit,
            reaction_margin_s=source_action.reaction_margin_s,
            teacher_rate_min=source_action.teacher_rate_min,
            teacher_rate_max=source_action.teacher_rate_max,
            label=label,
        )
        site_speed = _runtime_style_racket_site_speed(
            compiled_path,
            contact_frame=contact_frame,
            fps=entry.fps,
        )
        action_uid = manifest_module.derive_action_ball_action_uid(
            motion_id, entry.family, entry.npz_sha256
        )
        output_actions.append(
            {
                "action_id": motion_id,
                "action_uid": action_uid,
                "motion_path": entry.npz_path_text,
                "motion_sha256": entry.npz_sha256,
                "strike_phase": compiled_phase,
                "reference_t_hit_s": compiled_t_hit,
                "reference_t_cycle_s": compiled_t_cycle,
                "reference_racket_site_speed_mps": site_speed,
                "reaction_margin_s": source_action.reaction_margin_s,
                "teacher_rate_min": rate_min,
                "teacher_rate_max": source_action.teacher_rate_max,
                "family": entry.family,
                "mount_normal_sign": int(entry.mount_normal_sign),
                "ball_profile": ball_profile,
            }
        )
        receipt_actions.append(
            {
                "index": index,
                "action_id": motion_id,
                "source_motion_path": source_action.motion_path,
                "source_motion_sha256": source_action.motion_sha256,
                "compiled_motion_path": entry.npz_path_text,
                "compiled_motion_sha256": entry.npz_sha256,
                "action_uid": action_uid,
                "contact_frame": contact_frame,
                "reference_t_hit_s": compiled_t_hit,
                "reference_t_cycle_s": compiled_t_cycle,
                "reference_racket_site_speed_mps": site_speed,
                "teacher_rate_min_source": (
                    source_action.teacher_rate_min
                ),
                "teacher_rate_min_materialized": rate_min,
            }
        )

    required_holdout = max(
        768,
        holdout_samples_per_action,
        source_manifest.holdout.samples_per_action,
        source_manifest.curriculum.min_proposals,
        source_manifest.curriculum.min_safe_closed,
    )
    output_document = {
        "schema_version": 3,
        "manifest_id": manifest_id,
        "mobility_mode": source_manifest.mobility_mode,
        "action_order": list(loaded.motion_ids),
        "prototype": {
            "path": _repo_relative(
                prototype_file, root, "stroke prototype"
            ),
            "sha256": prototype_sha,
            "scope": scope,
        },
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "landing_aim": source_manifest.landing_aim.to_mapping(),
        "actions": output_actions,
        "curriculum": source_manifest.curriculum.to_mapping(),
        "holdout": {
            "seed": source_manifest.holdout.seed,
            "samples_per_action": required_holdout,
            "split_id": source_manifest.holdout.split_id,
        },
        "notes": (
            "Post-compile arbitrary-N materialization; ball profiles and "
            "curriculum originate from the source capsule action manifest, "
            "while every action identity/timing is rebound to the exact "
            f"{scope} compiler output and N-row prototype. Metadata only; "
            "grants no motion admission or training authority."
        ),
    }
    try:
        validated = manifest_module.ActionBallManifest.from_mapping(
            output_document
        )
        manifest_module.verify_action_ball_referenced_assets(
            validated, repo_root=root
        )
    except Exception as exc:
        raise MaterializationError(
            f"final ActionBall manifest validation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    manifest_bytes = (
        json.dumps(
            validated.to_mapping(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_sha = _sha256_bytes(manifest_bytes)
    canonical_sha = manifest_module.canonical_manifest_sha256(validated)

    out = _output_path(output_path, root, "manifest output")
    receipt_out = _output_path(
        receipt_output_path, root, "receipt output"
    )
    producer_path = Path(__file__).resolve()
    producer_relative = _repo_relative(
        producer_path, root, "materializer producer"
    )
    receipt = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "verdict": "PASS_METADATA_MATERIALIZATION_ONLY",
        "manifest_id": manifest_id,
        "scope": scope,
        "action_count": len(loaded.motion_ids),
        "inputs": {
            "bank_report": {
                "path": _repo_relative(
                    report_path, root, "generic bank report"
                ),
                "sha256": expected_bank_report_sha256,
            },
            "registry": {
                "path": _repo_relative(
                    registry.path, root, "generic registry"
                ),
                "sha256": registry.registry_sha256,
                "alignment_sha256": (
                    promotion_binding.alignment_sha256
                ),
            },
            "build_manifest": {
                "path": _repo_relative(
                    build_manifest_path, root, "build manifest"
                ),
                "sha256": build_manifest_sha,
            },
            "arbitrary_recipe": {
                "path": _repo_relative(
                    recipe_path, root, "arbitrary recipe"
                ),
                "sha256": recipe_sha,
            },
            "source_capsule": {
                "path": _repo_relative(
                    loaded.source_capsule_path, root, "source capsule"
                ),
                "sha256": loaded.source_capsule_sha256,
            },
            "source_action_manifest": {
                "path": _repo_relative(
                    source_manifest_path, root, "source ActionBall manifest"
                ),
                "sha256": source_manifest_sha,
            },
            "profile_pins": {
                "path": _repo_relative(
                    pins_path, root, "profile pins"
                ),
                "sha256": expected_profile_pins_sha256,
                "solver_profile_sha256": solver_sha,
                "physics_profile_sha256": physics_sha,
            },
            "prototype": {
                "path": _repo_relative(
                    prototype_file, root, "stroke prototype"
                ),
                "sha256": prototype_sha,
                "scope": scope,
            },
        },
        "producer": {
            "path": producer_relative,
            "sha256": _sha256_file(producer_path),
        },
        "output_manifest": {
            "path": out.relative_to(root).as_posix(),
            "sha256": manifest_sha,
            "canonical_sha256": canonical_sha,
        },
        "action_order": list(loaded.motion_ids),
        "actions": receipt_actions,
        "holdout_samples_per_action": required_holdout,
        "contracts": {
            "source_action_order_preserved": True,
            "compiled_identity_rebound": True,
            "raw_source_identity_rejected": True,
            "selected_scope_exact_n_rows": True,
            "paired_bank_matrix_reopened": True,
            "profile_payloads_rehashed": True,
            "selector_executed": False,
        },
        "authorization": {
            "motion_admission_minted": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "bank gate and metadata materialization do not promote motion admission",
            "no simulator trainer deployment or hardware command was run",
            "the launch boundary must separately bind trusted admission and exact runtime contracts",
        ],
    }
    receipt_bytes = (
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _publish_pair_no_clobber(
        manifest_path=out,
        manifest_bytes=manifest_bytes,
        receipt_path=receipt_out,
        receipt_bytes=receipt_bytes,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--bank-report", required=True)
    parser.add_argument("--expected-bank-report-sha256", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--profile-pins", required=True)
    parser.add_argument("--expected-profile-pins-sha256", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--expected-prototype-sha256", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument(
        "--holdout-samples-per-action",
        type=int,
        default=768,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize(
            repo_root=args.repo_root,
            bank_report_path=args.bank_report,
            expected_bank_report_sha256=(
                args.expected_bank_report_sha256
            ),
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
            scope=args.scope,
            profile_pins_path=args.profile_pins,
            expected_profile_pins_sha256=(
                args.expected_profile_pins_sha256
            ),
            prototype_path=args.prototype,
            expected_prototype_sha256=(
                args.expected_prototype_sha256
            ),
            manifest_id=args.manifest_id,
            output_path=args.out,
            receipt_output_path=args.receipt_out,
            holdout_samples_per_action=(
                args.holdout_samples_per_action
            ),
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
