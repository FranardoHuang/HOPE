#!/usr/bin/env python3
"""Compile an independently ordered arbitrary-N canonical candidate bank.

This adapter intentionally does not alter the historical canonical-five recipe
loader or compiler.  It validates a separate, content-addressed recipe whose
motion order comes from a source-capsule receipt, constructs the exact
``CanonicalMotionRecipe`` value consumed by the existing compiler, and asks the
existing compiler to emit every motion in ``upper`` then ``full`` scope.

The adapter is a compiler-candidate producer only.  A successful build remains
unauthorized until an independent generic bank gate has reopened the complete
N x 2 matrix and its timing, ready/recovery, FK, dynamics, and swept-safety
evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import canonical_motion_compile_cli as compile_cli  # noqa: E402
import canonical_motion_compiler as compiler  # noqa: E402
from canonical_motion_markers import (  # noqa: E402
    MarkerSemantics,
    MarkerSemanticsRow,
)
from canonical_motion_recipe import (  # noqa: E402
    CanonicalMotionRecipe,
    MotionSource,
    ReadyState,
    load_canonical_motion_recipe,
)
from mujoco_motion_player import (  # noqa: E402
    RUNTIME_BODY_NAMES,
    load_motion,
)


RECIPE_TYPE = "canonical_arbitrary_n_recipe_v1"
SOURCE_CAPSULE_INTERFACE = "canonical_arbitrary_n_source_capsule_v1"
PUBLICATION_CLASS = "compiler_candidate"
SCOPES = ("upper", "full")
RESERVED_CANONICAL_IDS = frozenset(
    {"fh_loop", "bh_loop_c", "fh_block_syn", "bh_block", "s0_highpress"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "recipe_type",
        "bank_id",
        "publication_class",
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
        "producer",
        "source_capsule",
        "ordered_motion_ids",
        "shared_ready",
        "marker_policy",
        "compiler_template",
        "compiler_options",
        "required_output_matrix",
        "placement_contract",
        "non_claims",
    }
)
_PATH_HASH_KEYS = frozenset({"path", "sha256"})
_READY_KEYS = frozenset(
    {
        "canonical_ready",
        "source_motion_path",
        "source_motion_sha256",
        "source_frame",
        "hold_tolerances",
        "evidence_status",
    }
)
_READY_TOLERANCE_KEYS = frozenset(
    {
        "joint_position_rad",
        "root_position_m",
        "root_orientation_rad",
        "joint_velocity_rad_s",
        "body_linear_velocity_m_s",
        "body_angular_velocity_rad_s",
    }
)
_READY_TOLERANCE_CAPS = {
    "joint_position_rad": 0.01,
    "root_position_m": 0.005,
    "root_orientation_rad": 0.01,
    "joint_velocity_rad_s": 0.05,
    "body_linear_velocity_m_s": 0.05,
    "body_angular_velocity_rad_s": 0.05,
}
_MARKER_KEYS = frozenset(
    {
        "mode",
        "half_width_frames",
        "minimum_source_preparation_frames",
        "minimum_source_recovery_frames",
        "minimum_compiled_recovery_s",
    }
)
_OPTIONS_KEYS = frozenset(
    {
        "joint_acceleration_receipt",
        "full_root_position_lower",
        "full_root_position_upper",
        "full_root_velocity",
        "full_root_acceleration",
        "samples_per_scaled_unit",
        "min_connector_intervals",
        "min_core_intervals",
        "grid_subdivisions",
        "search_workers",
        "search_parallel_backend",
    }
)
_MATRIX_KEYS = frozenset({"motion_ids", "scopes", "candidate_count"})
_PLACEMENT_KEYS = frozenset(
    {
        "task_frame",
        "source_motion_frame",
        "base_spawn_authority",
        "no_move_goal",
        "move_goal",
        "se2_equivariance",
        "cross_action_station_swap",
    }
)
_PLACEMENT_CONTRACT = {
    "task_frame": "episode_actual_base_yaw_local_v1",
    "source_motion_frame": "action_local_not_absolute_station_world",
    "base_spawn_authority": "source_capsule_per_action_center_xy",
    "no_move_goal": "equals_actual_episode_spawn_xy_yaw",
    "move_goal": "base_local_delta_from_actual_episode_spawn_xy_yaw",
    "se2_equivariance": (
        "common_translation_and_yaw_of_base_ball_contact_and_task_preserves_"
        "base_local_task"
    ),
    "cross_action_station_swap": (
        "reject_unless_base_ball_contact_and_task_share_the_same_se2_transform"
    ),
}
_SOURCE_ACTION_KEYS = frozenset(
    {
        "action_id",
        "family",
        "motion_path",
        "motion_sha256",
        "metadata_path",
        "metadata_sha256",
        "base_spawn_center_w_xy_m",
        "T",
        "fps",
        "hit_frame_50",
        "reference_t_hit_s",
        "reference_t_cycle_s",
    }
)


class ArbitraryBankError(RuntimeError):
    """The arbitrary-N build cannot proceed without weakening a contract."""


@dataclass(frozen=True)
class SourceTiming:
    motion_id: str
    source_path: Path
    source_sha256: str
    frames: int
    fps: float
    hit_frame: int
    t_hit_s: float
    t_cycle_s: float
    window: tuple[int, int]
    family: str
    station_xy_hope_m: tuple[float, float]
    base_spawn_center_w_xy_m: tuple[float, float]
    source_root_start_xy_m: tuple[float, float]
    source_root_travel_min_xy_m: tuple[float, float]
    source_root_travel_max_xy_m: tuple[float, float]


@dataclass(frozen=True)
class LoadedArbitraryRecipe:
    path: Path
    sha256: str
    repo_root: Path
    raw: Mapping[str, Any]
    source_capsule_path: Path
    source_capsule_sha256: str
    source_timings: tuple[SourceTiming, ...]
    canonical_recipe: CanonicalMotionRecipe
    options: compiler.CompilerOptions
    acceleration_receipt_path: Path
    acceleration_receipt_sha256: str
    producer_path: Path
    producer_sha256: str
    ready_hold_summary: Mapping[str, Any]

    @property
    def motion_ids(self) -> tuple[str, ...]:
        return tuple(row.motion_id for row in self.source_timings)


def swept_clearance_recipe_contract(
    loaded: LoadedArbitraryRecipe,
) -> Mapping[str, Any]:
    """Project the strict arbitrary recipe into the clearance producer view.

    The arbitrary recipe deliberately keeps the canonical compiler template
    behind ``compiler_template`` and ``shared_ready`` instead of copying the
    template's model/ready fields into its top level.  Clearance production
    must therefore consume this view only after the full arbitrary loader has
    reopened every bound input.  Absolute paths are intentional: they prevent
    a second consumer from resolving the already-validated template bindings
    against a different repository root.
    """

    if not isinstance(loaded, LoadedArbitraryRecipe):
        raise ArbitraryBankError(
            "swept-clearance contract requires a strictly loaded arbitrary recipe"
        )
    recipe = loaded.canonical_recipe
    if (
        Path(recipe.path).resolve() != loaded.path
        or Path(recipe.repo_root).resolve() != loaded.repo_root
        or _sha256(_read_regular(loaded.path, "arbitrary-N recipe")) != loaded.sha256
    ):
        raise ArbitraryBankError(
            "arbitrary recipe identity/bytes drifted before swept-clearance projection"
        )

    raw_matrix = _exact_keys(
        loaded.raw["required_output_matrix"],
        _MATRIX_KEYS,
        "required_output_matrix",
    )
    canonical_matrix = _exact_keys(
        recipe.raw.get("required_output_matrix"),
        _MATRIX_KEYS,
        "embedded canonical required_output_matrix",
    )
    expected_matrix = {
        "motion_ids": list(loaded.motion_ids),
        "scopes": list(SCOPES),
        "candidate_count": 2 * len(loaded.motion_ids),
    }
    if dict(raw_matrix) != expected_matrix or dict(canonical_matrix) != expected_matrix:
        raise ArbitraryBankError(
            "embedded canonical matrix changed arbitrary-N order or scope identity"
        )
    if tuple(source.motion_id for source in recipe.sources) != loaded.motion_ids:
        raise ArbitraryBankError(
            "embedded canonical source order changed arbitrary-N identity"
        )

    if set(recipe.model_paths) != {"mjcf", "urdf", "body_order"} or set(
        recipe.model_hashes
    ) != {"mjcf", "urdf", "body_order"}:
        raise ArbitraryBankError(
            "embedded canonical model contract is not exact mjcf/urdf/body_order"
        )
    model_contract: dict[str, str] = {}
    for name in ("mjcf", "urdf", "body_order"):
        model_path = Path(recipe.model_paths[name]).resolve()
        try:
            model_path.relative_to(loaded.repo_root)
        except ValueError as exc:
            raise ArbitraryBankError(
                f"embedded canonical {name} leaves repo_root"
            ) from exc
        expected_sha = _digest(
            recipe.model_hashes[name],
            f"embedded canonical {name} SHA-256",
        )
        if _sha256(_read_regular(model_path, f"embedded canonical {name}")) != expected_sha:
            raise ArbitraryBankError(
                f"embedded canonical {name} bytes drifted before swept-clearance projection"
            )
        model_contract[f"{name}_path"] = str(model_path)
        model_contract[f"{name}_sha256"] = expected_sha

    ready_path = Path(recipe.ready.path).resolve()
    try:
        ready_path.relative_to(loaded.repo_root)
    except ValueError as exc:
        raise ArbitraryBankError(
            "embedded canonical ready leaves repo_root"
        ) from exc
    ready_sha = _digest(
        recipe.ready.sha256,
        "embedded canonical ready SHA-256",
    )
    if _sha256(_read_regular(ready_path, "embedded canonical ready")) != ready_sha:
        raise ArbitraryBankError(
            "embedded canonical ready bytes drifted before swept-clearance projection"
        )
    ready_summary = _exact_keys(
        loaded.ready_hold_summary.get("canonical_ready"),
        _PATH_HASH_KEYS | frozenset({"source_segment", "source_frame"}),
        "shared-ready canonical-ready summary",
    )
    summary_path = (loaded.repo_root / str(ready_summary["path"])).resolve()
    if (
        summary_path != ready_path
        or _digest(
            ready_summary["sha256"],
            "shared-ready canonical-ready summary SHA-256",
        )
        != ready_sha
        or ready_summary["source_segment"] != recipe.ready.source_segment
        or ready_summary["source_frame"] != int(recipe.ready.source_frame)
    ):
        raise ArbitraryBankError(
            "shared-ready summary differs from embedded canonical ready identity"
        )

    return MappingProxyType(
        {
            "recipe_type": RECIPE_TYPE,
            "required_output_matrix": copy.deepcopy(expected_matrix),
            "model_contract": model_contract,
            "canonical_ready": {
                "path": str(ready_path),
                "sha256": ready_sha,
            },
        }
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArbitraryBankError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return value


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArbitraryBankError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ArbitraryBankError(
            f"{label} contains non-finite JSON constant {value}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ArbitraryBankError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArbitraryBankError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArbitraryBankError(f"{label} must contain one JSON object")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ArbitraryBankError(
                f"{label} must be a regular non-symlink file: {path}"
            )
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except ArbitraryBankError:
        raise
    except OSError as exc:
        raise ArbitraryBankError(f"cannot read {label} {path}: {exc}") from exc
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if identities[0] != identities[1] or identities[1] != identities[2]:
        raise ArbitraryBankError(f"{label} changed during stable read: {path}")
    return b"".join(chunks)


def _exact_keys(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArbitraryBankError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise ArbitraryBankError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ArbitraryBankError(f"{label} must be an array")
    return value


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise ArbitraryBankError(f"{label} must be one lowercase normalized slug")
    return value


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArbitraryBankError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ArbitraryBankError(f"{label} must be <= {maximum}")
    return value


def _finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArbitraryBankError(f"{label} must be one finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ArbitraryBankError(f"{label} must be one finite number")
    if minimum is not None and result < minimum:
        raise ArbitraryBankError(f"{label} must be >= {minimum}")
    if strictly_positive and result <= 0.0:
        raise ArbitraryBankError(f"{label} must be > 0")
    return result


def _vector(
    value: Any,
    length: int,
    label: str,
    *,
    strictly_positive: bool = False,
) -> np.ndarray:
    values = _sequence(value, label)
    if len(values) != length:
        raise ArbitraryBankError(
            f"{label} must contain exactly {length} entries"
        )
    result = np.asarray(
        [
            _finite(
                item,
                f"{label}[{index}]",
                strictly_positive=strictly_positive,
            )
            for index, item in enumerate(values)
        ],
        dtype=np.float64,
    )
    return result


def world_xy_to_base_yaw_local(
    world_xy: Sequence[float],
    base_xy: Sequence[float],
    base_yaw_rad: float,
) -> np.ndarray:
    """Express one world XY point in the actual episode base-yaw frame."""

    world = np.asarray(world_xy, dtype=np.float64)
    base = np.asarray(base_xy, dtype=np.float64)
    if (
        world.shape != (2,)
        or base.shape != (2,)
        or not np.isfinite(world).all()
        or not np.isfinite(base).all()
    ):
        raise ArbitraryBankError(
            "world_xy and base_xy must be finite two-vectors"
        )
    yaw = _finite(base_yaw_rad, "base_yaw_rad")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation_world_from_base = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    return rotation_world_from_base.T @ (world - base)


def base_yaw_local_xy_to_world(
    local_xy: Sequence[float],
    base_xy: Sequence[float],
    base_yaw_rad: float,
) -> np.ndarray:
    """Map one base-yaw-local XY point into world coordinates."""

    local = np.asarray(local_xy, dtype=np.float64)
    base = np.asarray(base_xy, dtype=np.float64)
    if (
        local.shape != (2,)
        or base.shape != (2,)
        or not np.isfinite(local).all()
        or not np.isfinite(base).all()
    ):
        raise ArbitraryBankError(
            "local_xy and base_xy must be finite two-vectors"
        )
    yaw = _finite(base_yaw_rad, "base_yaw_rad")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation_world_from_base = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    return base + rotation_world_from_base @ local


def _relative_path(value: Any, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise ArbitraryBankError(
            f"{label} must be one normalized repository-relative path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ArbitraryBankError(f"{label} may not contain '.' or '..'")
    return path


def _bound_repo_file(
    repo_root: Path,
    row: Any,
    label: str,
) -> tuple[Path, bytes, str]:
    binding = _exact_keys(row, _PATH_HASH_KEYS, label)
    relative = _relative_path(binding["path"], f"{label}.path")
    path = repo_root.joinpath(*relative.parts)
    payload = _read_regular(path, label)
    actual = _sha256(payload)
    expected = _digest(binding["sha256"], f"{label}.sha256")
    if actual != expected:
        raise ArbitraryBankError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    try:
        resolved_root = repo_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ArbitraryBankError(f"{label} escapes repo root") from exc
    return resolved_path, payload, actual


def _quaternion_distance(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    left /= np.linalg.norm(left, axis=-1, keepdims=True)
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    dot = np.sum(left * right, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def _ready_state(
    repo_root: Path,
    raw: Any,
    *,
    canonical_ready: ReadyState,
) -> tuple[ReadyState, Mapping[str, Any]]:
    ready = _exact_keys(raw, _READY_KEYS, "shared_ready")
    ready_path, _, ready_sha = _bound_repo_file(
        repo_root,
        ready["canonical_ready"],
        "shared_ready canonical-ready sidecar",
    )
    if (
        ready_path != Path(canonical_ready.path).resolve()
        or ready_sha != canonical_ready.sha256
    ):
        raise ArbitraryBankError(
            "shared_ready canonical-ready binding differs from the strict "
            "compiler-template ready sidecar"
        )
    relative = _relative_path(
        ready["source_motion_path"],
        "shared_ready.source_motion_path",
    )
    path = repo_root.joinpath(*relative.parts)
    payload = _read_regular(path, "shared-ready source motion")
    actual_sha = _sha256(payload)
    expected_sha = _digest(
        ready["source_motion_sha256"],
        "shared_ready.source_motion_sha256",
    )
    if actual_sha != expected_sha:
        raise ArbitraryBankError(
            "shared-ready source motion bytes drifted from recipe"
        )
    try:
        clip = load_motion(path)
    except Exception as exc:
        raise ArbitraryBankError(
            f"cannot load shared-ready source motion: {exc}"
        ) from exc
    frame = _integer(
        ready["source_frame"],
        "shared_ready.source_frame",
        maximum=clip.n_frames - 1,
    )
    tolerances_raw = _exact_keys(
        ready["hold_tolerances"],
        _READY_TOLERANCE_KEYS,
        "shared_ready.hold_tolerances",
    )
    tolerances: dict[str, float] = {}
    for key, cap in _READY_TOLERANCE_CAPS.items():
        value = _finite(
            tolerances_raw[key],
            f"shared_ready.hold_tolerances.{key}",
            strictly_positive=True,
        )
        if value > cap:
            raise ArbitraryBankError(
                f"shared_ready tolerance {key}={value} exceeds hard cap {cap}"
            )
        tolerances[key] = value
    if ready["evidence_status"] != "SOURCE_HOLD_ONLY_NOT_GROUNDED_CERTIFICATE":
        raise ArbitraryBankError(
            "shared_ready.evidence_status may not claim grounded admission"
        )
    try:
        pelvis_index = RUNTIME_BODY_NAMES.index("pelvis_link")
    except ValueError as exc:
        raise ArbitraryBankError(
            "runtime body contract has no pelvis_link"
        ) from exc
    joint_reference = np.asarray(clip.joint_pos[frame], dtype=np.float64)
    root_reference = np.asarray(
        clip.body_pos_w[frame, pelvis_index], dtype=np.float64
    )
    quat_reference = np.asarray(
        clip.body_quat_w[frame, pelvis_index], dtype=np.float64
    )
    metrics = {
        "joint_position_rad": float(
            np.max(np.abs(np.asarray(clip.joint_pos) - joint_reference))
        ),
        "root_position_m": float(
            np.max(
                np.linalg.norm(
                    np.asarray(clip.body_pos_w[:, pelvis_index])
                    - root_reference,
                    axis=-1,
                )
            )
        ),
        "root_orientation_rad": float(
            np.max(
                _quaternion_distance(
                    np.asarray(clip.body_quat_w[:, pelvis_index]),
                    quat_reference,
                )
            )
        ),
        "joint_velocity_rad_s": float(
            np.max(np.abs(np.asarray(clip.joint_vel)))
        ),
        "body_linear_velocity_m_s": float(
            np.max(
                np.linalg.norm(
                    np.asarray(clip.body_lin_vel_w),
                    axis=-1,
                )
            )
        ),
        "body_angular_velocity_rad_s": float(
            np.max(
                np.linalg.norm(
                    np.asarray(clip.body_ang_vel_w),
                    axis=-1,
                )
            )
        ),
    }
    failures = {
        key: {"observed": metrics[key], "limit": tolerances[key]}
        for key in metrics
        if metrics[key] > tolerances[key]
    }
    if failures:
        raise ArbitraryBankError(
            f"shared-ready source is not a bounded hold: {failures}"
        )
    summary = MappingProxyType(
        {
            "canonical_ready": {
                "path": ready_path.relative_to(repo_root).as_posix(),
                "sha256": ready_sha,
                "source_segment": canonical_ready.source_segment,
                "source_frame": int(canonical_ready.source_frame),
            },
            "source_motion_path": relative.as_posix(),
            "source_motion_sha256": actual_sha,
            "source_frame": frame,
            "source_frames": clip.n_frames,
            "source_fps": float(clip.fps),
            "observed_hold_metrics": metrics,
            "hold_tolerances": tolerances,
            "evidence_status": ready["evidence_status"],
            "grounded_certificate_present": False,
        }
    )
    return canonical_ready, summary


def _compiler_options(
    repo_root: Path,
    raw: Any,
) -> tuple[compiler.CompilerOptions, Path, str]:
    values = _exact_keys(raw, _OPTIONS_KEYS, "compiler_options")
    acceleration_path, _, acceleration_sha = _bound_repo_file(
        repo_root,
        values["joint_acceleration_receipt"],
        "joint acceleration receipt",
    )
    try:
        acceleration = compile_cli._load_acceleration_receipt(
            acceleration_path,
            acceleration_sha,
        )
    except Exception as exc:
        raise ArbitraryBankError(
            f"joint acceleration receipt is invalid: {exc}"
        ) from exc
    lower = _vector(
        values["full_root_position_lower"],
        6,
        "compiler_options.full_root_position_lower",
    )
    upper = _vector(
        values["full_root_position_upper"],
        6,
        "compiler_options.full_root_position_upper",
    )
    if np.any(lower >= upper):
        raise ArbitraryBankError(
            "every full-root lower bound must be below its upper bound"
        )
    velocity = _vector(
        values["full_root_velocity"],
        6,
        "compiler_options.full_root_velocity",
        strictly_positive=True,
    )
    root_acceleration = _vector(
        values["full_root_acceleration"],
        6,
        "compiler_options.full_root_acceleration",
        strictly_positive=True,
    )
    samples = _finite(
        values["samples_per_scaled_unit"],
        "compiler_options.samples_per_scaled_unit",
        strictly_positive=True,
    )
    connector = _integer(
        values["min_connector_intervals"],
        "compiler_options.min_connector_intervals",
        minimum=5,
    )
    core = _integer(
        values["min_core_intervals"],
        "compiler_options.min_core_intervals",
        minimum=5,
    )
    grid = _integer(
        values["grid_subdivisions"],
        "compiler_options.grid_subdivisions",
        minimum=2,
    )
    workers = _integer(
        values["search_workers"],
        "compiler_options.search_workers",
        minimum=1,
        maximum=64,
    )
    backend = values["search_parallel_backend"]
    if backend not in ("thread", "process"):
        raise ArbitraryBankError(
            "compiler_options.search_parallel_backend must be thread or process"
        )
    options = compiler.CompilerOptions(
        joint_acceleration_limits_rad_s2=(
            acceleration.acceleration_rad_s2
        ),
        full_root_limits=compiler.FullRootPathLimits(
            position_lower=lower,
            position_upper=upper,
            velocity=velocity,
            acceleration=root_acceleration,
        ),
        # The arbitrary-N path forbids the historical ``s0_highpress`` ID, so
        # no source-specific grounding transform is ever selected.  Keep the
        # effective option as the explicit numeric identity transform instead
        # of ``None`` so an independent verifier can close the complete
        # compiler-options receipt without interpreting a nullable exception.
        s0_full_grounding_offset_m=0.0,
        samples_per_scaled_unit=samples,
        min_connector_intervals=connector,
        min_core_intervals=core,
        grid_subdivisions=grid,
        search_workers=workers,
        search_parallel_backend=backend,
    )
    for name in (
        "probe_entry_band",
        "probe_exit_band",
        "probe_exact_pointwise_caps",
        "probe_source_smoothing_tolerance_rad",
        "synthetic_face_solve_span_extension",
    ):
        default = compiler.CompilerOptions.__dataclass_fields__[name].default
        if getattr(options, name) != default:
            raise ArbitraryBankError(
                f"formal arbitrary-N compile enabled probe option {name}"
            )
    return options, acceleration_path, acceleration_sha


def _source_rows(
    repo_root: Path,
    capsule_path: Path,
    capsule_sha: str,
    capsule: Mapping[str, Any],
    motion_ids: tuple[str, ...],
    marker_policy: Mapping[str, Any],
) -> tuple[
    tuple[MotionSource, ...],
    tuple[MarkerSemanticsRow, ...],
    tuple[SourceTiming, ...],
]:
    if capsule.get("schema_version") != 1:
        raise ArbitraryBankError("source capsule schema_version must be 1")
    if capsule.get("consumer_interface") != SOURCE_CAPSULE_INTERFACE:
        raise ArbitraryBankError(
            f"source capsule must expose {SOURCE_CAPSULE_INTERFACE!r}"
        )
    if capsule.get("verdict") != "PASS_SOURCE_INVENTORY_ONLY":
        raise ArbitraryBankError(
            "source capsule verdict must be PASS_SOURCE_INVENTORY_ONLY"
        )
    authorization = capsule.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        key.endswith("_authorized") and value is not False
        for key, value in authorization.items()
    ):
        raise ArbitraryBankError(
            "source capsule may not carry any authorization"
        )
    actions = _sequence(capsule.get("actions"), "source capsule actions")
    if len(actions) != len(motion_ids):
        raise ArbitraryBankError(
            "source capsule action count differs from ordered_motion_ids"
        )
    half_width = _integer(
        marker_policy["half_width_frames"],
        "marker_policy.half_width_frames",
        minimum=1,
        maximum=20,
    )
    minimum_preparation = _integer(
        marker_policy["minimum_source_preparation_frames"],
        "marker_policy.minimum_source_preparation_frames",
        minimum=1,
    )
    minimum_recovery = _integer(
        marker_policy["minimum_source_recovery_frames"],
        "marker_policy.minimum_source_recovery_frames",
        minimum=1,
    )
    sources: list[MotionSource] = []
    markers: list[MarkerSemanticsRow] = []
    timings: list[SourceTiming] = []
    capsule_root = capsule_path.parent
    for index, (raw_action, motion_id) in enumerate(zip(actions, motion_ids)):
        if not isinstance(raw_action, Mapping):
            raise ArbitraryBankError(
                f"source capsule actions[{index}] must be an object"
            )
        action = {
            key: raw_action.get(key)
            for key in _SOURCE_ACTION_KEYS
        }
        if frozenset(action) != _SOURCE_ACTION_KEYS or any(
            value is None for value in action.values()
        ):
            raise ArbitraryBankError(
                f"source capsule actions[{index}] lacks compiler source fields"
            )
        if action["action_id"] != motion_id:
            raise ArbitraryBankError(
                f"source capsule action {index} changed ordered identity"
            )
        relative = _relative_path(
            action["motion_path"],
            f"source capsule actions[{index}].motion_path",
        )
        source_path = capsule_root.joinpath(*relative.parts)
        payload = _read_regular(source_path, f"{motion_id} source motion")
        source_sha = _sha256(payload)
        if source_sha != _digest(
            action["motion_sha256"],
            f"source capsule actions[{index}].motion_sha256",
        ):
            raise ArbitraryBankError(
                f"{motion_id} source motion bytes drifted from capsule"
            )
        try:
            source_path.resolve(strict=True).relative_to(
                repo_root.resolve(strict=True)
            )
        except (OSError, ValueError) as exc:
            raise ArbitraryBankError(
                f"{motion_id} source motion escapes repo root"
            ) from exc
        try:
            clip = load_motion(source_path)
        except Exception as exc:
            raise ArbitraryBankError(
                f"{motion_id} source is not exact schema-2: {exc}"
            ) from exc
        family = action["family"]
        if family not in ("forehand", "backhand"):
            raise ArbitraryBankError(
                f"{motion_id} family must be forehand or backhand"
            )
        metadata_relative = _relative_path(
            action["metadata_path"],
            f"source capsule actions[{index}].metadata_path",
        )
        metadata_path = capsule_root.joinpath(*metadata_relative.parts)
        metadata_payload = _read_regular(
            metadata_path, f"{motion_id} source metadata"
        )
        metadata_sha = _sha256(metadata_payload)
        if metadata_sha != _digest(
            action["metadata_sha256"],
            f"source capsule actions[{index}].metadata_sha256",
        ):
            raise ArbitraryBankError(
                f"{motion_id} metadata bytes drifted from capsule"
            )
        metadata = _strict_json_bytes(
            metadata_payload, f"{motion_id} source metadata"
        )
        station = _vector(
            metadata.get("station_xy_hope_m"),
            2,
            f"{motion_id} metadata station_xy_hope_m",
        )
        base_spawn = _vector(
            action["base_spawn_center_w_xy_m"],
            2,
            f"{motion_id} base_spawn_center_w_xy_m",
        )
        expected_base_spawn = station + np.asarray(
            [0.5, 0.7625], dtype=np.float64
        )
        if not np.allclose(
            base_spawn,
            expected_base_spawn,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ArbitraryBankError(
                f"{motion_id} station/base_spawn mapping changed"
            )
        frames = _integer(
            action["T"],
            f"source capsule actions[{index}].T",
            minimum=2,
        )
        fps = _finite(
            action["fps"],
            f"source capsule actions[{index}].fps",
            strictly_positive=True,
        )
        if frames != clip.n_frames or not math.isclose(
            fps, float(clip.fps), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ArbitraryBankError(
                f"{motion_id} source frames/fps disagree with capsule"
            )
        hit = _integer(
            action["hit_frame_50"],
            f"source capsule actions[{index}].hit_frame_50",
            minimum=1,
            maximum=frames - 2,
        )
        t_hit = _finite(
            action["reference_t_hit_s"],
            f"source capsule actions[{index}].reference_t_hit_s",
            minimum=0.0,
        )
        t_cycle = _finite(
            action["reference_t_cycle_s"],
            f"source capsule actions[{index}].reference_t_cycle_s",
            strictly_positive=True,
        )
        if (
            not math.isclose(t_hit, hit / fps, rel_tol=0.0, abs_tol=1e-10)
            or not math.isclose(
                t_cycle,
                (frames - 1) / fps,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            or not t_cycle > t_hit
        ):
            raise ArbitraryBankError(
                f"{motion_id} source t_hit/t_cycle do not bind hit/T/fps"
            )
        if hit < minimum_preparation or frames - 1 - hit < minimum_recovery:
            raise ArbitraryBankError(
                f"{motion_id} lacks required source preparation/recovery frames"
            )
        window = (
            max(0, hit - half_width),
            min(frames - 1, hit + half_width),
        )
        if not window[0] <= hit <= window[1]:
            raise ArbitraryBankError(
                f"{motion_id} marker window lost the hit frame"
            )
        sources.append(
            MotionSource(
                motion_id=motion_id,
                human_role=f"source_capsule_action_{motion_id}",
                path=source_path.resolve(),
                sha256=source_sha,
                clip=clip,
                face_manifold=None,
                scope_overrides=MappingProxyType({}),
            )
        )
        markers.append(
            MarkerSemanticsRow(
                motion_id=motion_id,
                nominal_event=hit,
                ge50_seed=window,
                ge80_seed=window,
                preferred_seed=None,
                construction_marker=None,
                historical_adv2c3_start=window[0],
                bound_recipe_source_path=relative.as_posix(),
                bound_recipe_source_sha256=source_sha,
                source_scan_remote_path=(
                    f"source-capsule:{capsule_path.name}#actions/{index}"
                ),
                source_scan_sha256=capsule_sha,
                frame_identity=None,
                post_retime_behavior_gate_status=(
                    "PENDING_POST_RETIME_PHYSICAL_RETURN_RESCAN"
                ),
            )
        )
        timings.append(
            SourceTiming(
                motion_id=motion_id,
                source_path=source_path.resolve(),
                source_sha256=source_sha,
                frames=frames,
                fps=fps,
                hit_frame=hit,
                t_hit_s=t_hit,
                t_cycle_s=t_cycle,
                window=window,
                family=family,
                station_xy_hope_m=tuple(float(value) for value in station),
                base_spawn_center_w_xy_m=tuple(
                    float(value) for value in base_spawn
                ),
                source_root_start_xy_m=tuple(
                    float(value)
                    for value in np.asarray(
                        clip.body_pos_w[0, RUNTIME_BODY_NAMES.index("pelvis_link"), :2],
                        dtype=np.float64,
                    )
                ),
                source_root_travel_min_xy_m=tuple(
                    float(value)
                    for value in np.min(
                        np.asarray(
                            clip.body_pos_w[
                                :,
                                RUNTIME_BODY_NAMES.index("pelvis_link"),
                                :2,
                            ],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            clip.body_pos_w[
                                0,
                                RUNTIME_BODY_NAMES.index("pelvis_link"),
                                :2,
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                ),
                source_root_travel_max_xy_m=tuple(
                    float(value)
                    for value in np.max(
                        np.asarray(
                            clip.body_pos_w[
                                :,
                                RUNTIME_BODY_NAMES.index("pelvis_link"),
                                :2,
                            ],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            clip.body_pos_w[
                                0,
                                RUNTIME_BODY_NAMES.index("pelvis_link"),
                                :2,
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                ),
            )
        )
    return tuple(sources), tuple(markers), tuple(timings)


def load_arbitrary_bank_recipe(
    recipe_path: os.PathLike[str] | str,
    *,
    repo_root: os.PathLike[str] | str,
) -> LoadedArbitraryRecipe:
    """Load one strict arbitrary-N recipe and every content-bound input."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ArbitraryBankError("repo_root must be one real directory")
    path = Path(recipe_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArbitraryBankError("recipe must live inside repo_root") from exc
    recipe_bytes = _read_regular(path, "arbitrary-N recipe")
    recipe_sha = _sha256(recipe_bytes)
    raw = _exact_keys(
        _strict_json_bytes(recipe_bytes, "arbitrary-N recipe"),
        _TOP_KEYS,
        "arbitrary-N recipe",
    )
    if raw["schema_version"] != 1 or raw["recipe_type"] != RECIPE_TYPE:
        raise ArbitraryBankError("arbitrary-N recipe schema/type is unsupported")
    bank_id = _slug(raw["bank_id"], "bank_id")
    if raw["publication_class"] != PUBLICATION_CLASS:
        raise ArbitraryBankError(
            "arbitrary-N recipe must remain compiler_candidate"
        )
    for key in (
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        if raw[key] is not False:
            raise ArbitraryBankError(f"{key} must remain false")
    producer_path, _, producer_sha = _bound_repo_file(
        root, raw["producer"], "arbitrary-N producer"
    )
    if producer_path != Path(__file__).resolve():
        raise ArbitraryBankError(
            "recipe producer must bind canonical_motion_arbitrary_bank.py"
        )
    capsule_path, capsule_bytes, capsule_sha = _bound_repo_file(
        root, raw["source_capsule"], "source capsule receipt"
    )
    capsule = _strict_json_bytes(capsule_bytes, "source capsule receipt")
    template_path, _, template_sha = _bound_repo_file(
        root, raw["compiler_template"], "compiler template recipe"
    )
    try:
        template = load_canonical_motion_recipe(
            template_path, repo_root=root
        )
    except Exception as exc:
        raise ArbitraryBankError(
            f"compiler template recipe is invalid: {exc}"
        ) from exc
    if _sha256(_read_regular(template_path, "compiler template recipe")) != (
        template_sha
    ):
        raise ArbitraryBankError("compiler template changed during load")
    ordered_values = _sequence(
        raw["ordered_motion_ids"], "ordered_motion_ids"
    )
    if not ordered_values:
        raise ArbitraryBankError("ordered_motion_ids may not be empty")
    motion_ids = tuple(
        _slug(value, f"ordered_motion_ids[{index}]")
        for index, value in enumerate(ordered_values)
    )
    if len(set(motion_ids)) != len(motion_ids):
        raise ArbitraryBankError("ordered_motion_ids contains duplicates")
    overlap = sorted(set(motion_ids).intersection(RESERVED_CANONICAL_IDS))
    if overlap:
        raise ArbitraryBankError(
            "standalone arbitrary-N bank may not reuse reserved canonical-five "
            f"ids: {overlap}"
        )
    matrix = _exact_keys(
        raw["required_output_matrix"],
        _MATRIX_KEYS,
        "required_output_matrix",
    )
    if (
        matrix["motion_ids"] != list(motion_ids)
        or matrix["scopes"] != list(SCOPES)
        or matrix["candidate_count"] != 2 * len(motion_ids)
    ):
        raise ArbitraryBankError(
            "required_output_matrix must be exact ordered N x upper/full"
        )
    placement = _exact_keys(
        raw["placement_contract"],
        _PLACEMENT_KEYS,
        "placement_contract",
    )
    if dict(placement) != _PLACEMENT_CONTRACT:
        raise ArbitraryBankError(
            "placement_contract must preserve episode-base-local task "
            "semantics and reject cross-action station swaps"
        )
    marker_policy = _exact_keys(
        raw["marker_policy"], _MARKER_KEYS, "marker_policy"
    )
    if marker_policy["mode"] != "source_hit_centered_marker_only_v1":
        raise ArbitraryBankError("marker_policy.mode is unsupported")
    minimum_compiled_recovery = _finite(
        marker_policy["minimum_compiled_recovery_s"],
        "marker_policy.minimum_compiled_recovery_s",
        strictly_positive=True,
    )
    if minimum_compiled_recovery > 2.0:
        raise ArbitraryBankError(
            "minimum_compiled_recovery_s may not exceed 2 seconds"
        )
    sources, marker_rows, timings = _source_rows(
        root,
        capsule_path,
        capsule_sha,
        capsule,
        motion_ids,
        marker_policy,
    )
    ready, ready_summary = _ready_state(
        root,
        raw["shared_ready"],
        canonical_ready=template.ready,
    )
    options, acceleration_path, acceleration_sha = _compiler_options(
        root, raw["compiler_options"]
    )
    template_raw = copy.deepcopy(dict(template.raw))
    template_raw["library_id"] = bank_id
    template_raw["required_output_matrix"] = {
        "motion_ids": list(motion_ids),
        "scopes": list(SCOPES),
        "candidate_count": 2 * len(motion_ids),
    }
    template_raw["motion_specs"] = [
        {
            "motion_id": timing.motion_id,
            "source_path": timing.source_path.relative_to(root).as_posix(),
            "source_sha256": timing.source_sha256,
            "source_hit_frame": timing.hit_frame,
            "source_t_hit_s": timing.t_hit_s,
            "source_t_cycle_s": timing.t_cycle_s,
            "source_marker_window": list(timing.window),
            "family": timing.family,
            "placement": {
                "station_xy_hope_m": list(timing.station_xy_hope_m),
                "base_spawn_center_w_xy_m": list(
                    timing.base_spawn_center_w_xy_m
                ),
                "source_root_start_xy_m": list(
                    timing.source_root_start_xy_m
                ),
                "source_root_travel_min_xy_m": list(
                    timing.source_root_travel_min_xy_m
                ),
                "source_root_travel_max_xy_m": list(
                    timing.source_root_travel_max_xy_m
                ),
                "task_frame": _PLACEMENT_CONTRACT["task_frame"],
                "no_move_goal": _PLACEMENT_CONTRACT["no_move_goal"],
            },
        }
        for timing in timings
    ]
    template_raw["post_build_gates"] = [
        "strict_schema2_and_shared_ready_digest",
        "source_and_compiled_t_hit_t_cycle",
        "minimum_compiled_recovery_and_shared_ready_return",
        "exact_vendor_mujoco_fk_playback",
        "joint_position_velocity_and_plant_specific_torque_screen",
        "self_collision_body_racket_ground_table_net_scan",
        "continuous_swept_clearance_complete_cycle",
        "post_retime_physical_return_per_scope",
        "registry_consumer_export_deploy_contract",
    ]
    marker_semantics = MarkerSemantics(
        path=path,
        repo_root=root,
        sha256=recipe_sha,
        authority_id="canonical_arbitrary_n_marker_semantics_v1",
        review_status="SOURCE_HIT_MARKER_ONLY_NOT_BEHAVIOR_ADMISSION",
        legacy_authority_sha256=recipe_sha,
        rows=marker_rows,
    )
    canonical = CanonicalMotionRecipe(
        path=path,
        repo_root=root,
        raw=MappingProxyType(template_raw),
        ready=ready,
        sources=sources,
        marker_semantics=marker_semantics,
        marker_authority_path=path,
        marker_authority_sha256=recipe_sha,
        model_paths=template.model_paths,
        model_hashes=template.model_hashes,
    )
    non_claims = tuple(
        _sequence(raw["non_claims"], "non_claims")
    )
    if (
        len(non_claims) != len(set(non_claims))
        or any(not isinstance(value, str) or not value for value in non_claims)
    ):
        raise ArbitraryBankError("non_claims must be unique non-empty strings")
    for required in (
        "grounded_ready_certificate",
        "dynamics_or_balance",
        "table_or_collision_safety",
        "physical_ball_return",
        "training_authorization",
        "hardware_authorization",
    ):
        if required not in non_claims:
            raise ArbitraryBankError(f"non_claims lost {required!r}")
    return LoadedArbitraryRecipe(
        path=path,
        sha256=recipe_sha,
        repo_root=root,
        raw=raw,
        source_capsule_path=capsule_path,
        source_capsule_sha256=capsule_sha,
        source_timings=timings,
        canonical_recipe=canonical,
        options=options,
        acceleration_receipt_path=acceleration_path,
        acceleration_receipt_sha256=acceleration_sha,
        producer_path=producer_path,
        producer_sha256=producer_sha,
        ready_hold_summary=ready_summary,
    )


def validate_arbitrary_build_manifest(
    manifest: Mapping[str, Any],
    loaded: LoadedArbitraryRecipe,
) -> None:
    """Fail closed unless the compiler returned the exact ordered N x 2 bank."""

    if (
        manifest.get("schema_version") != 1
        or manifest.get("library_id") != loaded.raw["bank_id"]
        or manifest.get("publication_class") != PUBLICATION_CLASS
        or manifest.get("build_verdict")
        != "PASS_COMPILER_CANDIDATE_ONLY"
        or manifest.get("training_authorized") is not False
        or manifest.get("hardware_authorized") is not False
    ):
        raise ArbitraryBankError(
            "compiler returned an invalid candidate authorization boundary"
        )
    expected = tuple(
        (motion_id, scope)
        for motion_id in loaded.motion_ids
        for scope in SCOPES
    )
    matrix = manifest.get("output_matrix")
    if not isinstance(matrix, Mapping) or (
        matrix.get("motion_ids") != list(loaded.motion_ids)
        or matrix.get("scopes") != list(SCOPES)
        or matrix.get("candidate_count") != len(expected)
    ):
        raise ArbitraryBankError(
            "compiler manifest output_matrix changed arbitrary-N identity"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected):
        raise ArbitraryBankError(
            "compiler manifest omitted an arbitrary-N output"
        )
    observed = tuple(
        (row.get("motion_id"), row.get("scope"))
        for row in outputs
        if isinstance(row, Mapping)
    )
    if observed != expected:
        raise ArbitraryBankError(
            "compiler manifest output order changed arbitrary-N identity"
        )
    minimum_recovery = float(
        loaded.raw["marker_policy"]["minimum_compiled_recovery_s"]
    )
    source_by_id = {
        row.motion_id: row for row in loaded.source_timings
    }
    for index, row in enumerate(outputs):
        motion_id, scope = expected[index]
        filename = f"{motion_id}_{scope}_canonical_v2.npz"
        if (
            row.get("filename") != filename
            or not isinstance(row.get("output_npz_sha256"), str)
            or _SHA256.fullmatch(row["output_npz_sha256"]) is None
        ):
            raise ArbitraryBankError(
                f"compiler output {motion_id}/{scope} lost filename/SHA"
            )
        compiled_t_hit = _finite(
            row.get("source_anchor_time_s"),
            f"outputs[{index}].source_anchor_time_s",
            minimum=0.0,
        )
        compiled_t_cycle = _finite(
            row.get("duration_s"),
            f"outputs[{index}].duration_s",
            strictly_positive=True,
        )
        if (
            not 0.0 < compiled_t_hit < compiled_t_cycle
            or compiled_t_cycle - compiled_t_hit
            < minimum_recovery - 1e-12
        ):
            raise ArbitraryBankError(
                f"compiler output {motion_id}/{scope} lacks required recovery"
            )
        source = source_by_id[motion_id]
        source_entry = _integer(
            row.get("entry_frame"),
            f"outputs[{index}].entry_frame",
            minimum=0,
        )
        source_exit = _integer(
            row.get("exit_frame"),
            f"outputs[{index}].exit_frame",
            minimum=0,
        )
        if source_entry > source_exit or source_exit >= source.frames:
            raise ArbitraryBankError(
                f"compiler output {motion_id}/{scope} source frame range is invalid"
            )
        search = row.get("search")
        if not isinstance(search, Mapping):
            raise ArbitraryBankError(
                f"compiler output {motion_id}/{scope} lacks search receipt"
            )
        opportunity = search.get("contact_opportunity")
        if (
            not isinstance(opportunity, Mapping)
            or opportunity.get("marker_only") is not True
            or opportunity.get(
                "acceleration_allowed_through_window_end"
            )
            is not True
        ):
            raise ArbitraryBankError(
                f"compiler output {motion_id}/{scope} lost marker contract"
            )
        if not (
            0 < source.hit_frame < source.frames - 1
            and math.isclose(
                source.t_hit_s,
                source.hit_frame / source.fps,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            and math.isclose(
                source.t_cycle_s,
                (source.frames - 1) / source.fps,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise ArbitraryBankError(
                f"source timing authority drifted for {motion_id}"
            )


def compile_arbitrary_bank(
    loaded: LoadedArbitraryRecipe,
    *,
    output_directory: os.PathLike[str] | str,
    backend: Any | None = None,
) -> Path:
    """Compile and atomically publish the exact arbitrary-N paired bank."""

    if not isinstance(loaded, LoadedArbitraryRecipe):
        raise ArbitraryBankError(
            "loaded recipe must come from load_arbitrary_bank_recipe"
        )
    output = Path(output_directory).expanduser()
    if not output.is_absolute():
        output = loaded.repo_root / output
    output = Path(os.path.abspath(os.fspath(output)))
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output}"
        )
    library = compiler.compile_loaded_canonical_motion_library(
        loaded.canonical_recipe,
        options=loaded.options,
        backend=backend,
    )
    validate_arbitrary_build_manifest(library.manifest, loaded)
    return compiler.write_compiled_canonical_motion_library(
        library, output
    )


def dry_run_receipt(loaded: LoadedArbitraryRecipe) -> Mapping[str, Any]:
    """Return a source/compiler-input receipt without claiming compiled output."""

    return {
        "schema_version": 1,
        "receipt_type": "canonical_arbitrary_n_dry_run_v1",
        "verdict": "PASS_SOURCE_AND_COMPILER_INPUTS_ONLY",
        "bank_id": loaded.raw["bank_id"],
        "recipe": {
            "path": loaded.path.relative_to(loaded.repo_root).as_posix(),
            "sha256": loaded.sha256,
        },
        "source_capsule": {
            "path": loaded.source_capsule_path.relative_to(
                loaded.repo_root
            ).as_posix(),
            "sha256": loaded.source_capsule_sha256,
        },
        "producer": {
            "path": loaded.producer_path.relative_to(
                loaded.repo_root
            ).as_posix(),
            "sha256": loaded.producer_sha256,
        },
        "ordered_motion_ids": list(loaded.motion_ids),
        "required_output_matrix": {
            "motion_ids": list(loaded.motion_ids),
            "scopes": list(SCOPES),
            "candidate_count": 2 * len(loaded.motion_ids),
        },
        "shared_ready": dict(loaded.ready_hold_summary),
        "source_timing": [
            {
                "motion_id": row.motion_id,
                "source_sha256": row.source_sha256,
                "frames": row.frames,
                "fps": row.fps,
                "hit_frame": row.hit_frame,
                "t_hit_s": row.t_hit_s,
                "t_cycle_s": row.t_cycle_s,
                "marker_window": list(row.window),
                "family": row.family,
                "station_xy_hope_m": list(row.station_xy_hope_m),
                "base_spawn_center_w_xy_m": list(
                    row.base_spawn_center_w_xy_m
                ),
                "source_root_start_xy_m": list(
                    row.source_root_start_xy_m
                ),
                "source_root_travel_min_xy_m": list(
                    row.source_root_travel_min_xy_m
                ),
                "source_root_travel_max_xy_m": list(
                    row.source_root_travel_max_xy_m
                ),
            }
            for row in loaded.source_timings
        ],
        "placement_contract": dict(_PLACEMENT_CONTRACT),
        "joint_acceleration_receipt": {
            "path": loaded.acceleration_receipt_path.relative_to(
                loaded.repo_root
            ).as_posix(),
            "sha256": loaded.acceleration_receipt_sha256,
        },
        "authorization": {
            "compiler_outputs_present": False,
            "bank_gate_pass": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "missing_independent_evidence": [
            "compiled N x 2 output bytes",
            "grounded shared-ready certificate",
            "compiled t_hit/t_cycle and recovery gate",
            "MuJoCo FK and plant-specific dynamics",
            "continuous swept table/net/ground/self-collision safety",
            "post-retime physical ball return",
            "registry/alignment/evidence/adoption admission",
            "Isaac filtered-contact smoke",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate all source/compiler inputs without creating output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        loaded = load_arbitrary_bank_recipe(
            args.recipe,
            repo_root=args.repo_root,
        )
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = loaded.repo_root / output
        if os.path.lexists(output):
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output}"
            )
        if args.dry_run:
            receipt = dry_run_receipt(loaded)
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            return 0
        result = compile_arbitrary_bank(
            loaded,
            output_directory=output,
        )
    except (ArbitraryBankError, FileExistsError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": "PASS_COMPILER_CANDIDATE_ONLY",
                "output": str(result),
                "training_authorized": False,
                "deployment_authorized": False,
                "hardware_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
