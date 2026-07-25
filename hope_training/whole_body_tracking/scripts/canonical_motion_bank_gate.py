#!/usr/bin/env python3
"""Independent post-build verifier for the canonical 5 x 2 motion bank.

The compiler is intentionally not trusted to certify its own output.  This
module reopens ``BUILD_MANIFEST.json`` and every published NPZ, recomputes all
available content hashes, rebinds the exact recipe/model/body-order contracts,
checks the shared canonical-ready endpoints and contact-opportunity markers,
and then invokes two independent consumers:

* :mod:`mujoco_motion_player` for strict schema-2 loading and exact 32-body FK;
* :mod:`canonical_mujoco_dynamics_gate` for plant-specific limits, geometry,
  root residuals, and inverse-dynamics effort diagnostics.

The contact opportunity remains a marker interval.  It is not a pose, speed,
or acceleration freeze, and acceleration may continue through its end.  A
grounded floating-base ``mj_inverse`` result remains
``INCOMPLETE_FAIL_CLOSED`` because schema-2 does not contain the compiler's
exact qacc/geometric-tangent midpoint collocation trace.  This verifier never
substitutes finite-difference qacc for that missing evidence, nor upgrades a
compiler candidate to training or hardware authorization.

The CLI writes its report last, atomically, and without clobbering an existing
path.  It never edits motion assets or steps a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import canonical_mujoco_dynamics_gate as dynamics_gate  # noqa: E402
import mujoco_motion_player as motion_player  # noqa: E402
from canonical_motion_recipe import (  # noqa: E402
    CanonicalMotionRecipe,
    load_canonical_motion_recipe,
)
from motion_kinematics_contract import BODY_LIN_VEL_POINT  # noqa: E402


MANIFEST_NAME = "BUILD_MANIFEST.json"
MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
SCOPES = ("upper", "full")
EXPECTED_MATRIX = tuple((motion_id, scope) for motion_id in MOTION_IDS for scope in SCOPES)
EXPECTED_FILENAMES = tuple(
    f"{motion_id}_{scope}_canonical_v2.npz" for motion_id, scope in EXPECTED_MATRIX
)

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "library_id",
        "publication_class",
        "build_verdict",
        "training_authorized",
        "hardware_authorized",
        "recipe",
        "compiler",
        "geometry_tool",
        "weighted_arc_tool",
        "compiler_options",
        "ready",
        "output_matrix",
        "search_contract",
        "contact_opportunity_contract",
        "time_law_claim",
        "outputs",
        "post_build_gates",
        "non_claims",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "motion_id",
        "scope",
        "filename",
        "output_npz_sha256",
        "entry_frame",
        "exit_frame",
        "duration_s",
        "contact_window_start_s",
        "contact_window_end_s",
        "source_anchor_time_s",
        "scaled_l2_total_variation",
        "search",
        "scope_preprocessing",
        "face_manifold",
        "geometry",
        "retiming",
        "schema2_manifest",
        "schema2_report",
    }
)
_MARKERS = ("window_start", "source_anchor", "window_end")
_MARKER_KEYS = frozenset(
    {
        "source_index",
        "time_s",
        "output_fractional_frame",
        "output_frame",
        "path_position_at_frame",
    }
)
_DIGEST_LENGTH = 64
_FLOAT_TOL = 1.0e-10
_READY_ROOT_TOL = 2.0e-6
_SOURCE_FRAME_MAP_ENCODING = "float64_le_canonical_qnan_v1"
_CANONICAL_QNAN_BITS = np.uint64(0x7FF8000000000000)
_PUBLISHED_INPUT_ARRAY_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_ROOT_COORDINATE_NAMES = (
    "root_x_w",
    "root_y_w",
    "root_z_w",
    "root_rotvec_x_ready",
    "root_rotvec_y_ready",
    "root_rotvec_z_ready",
)
_ROOT_COORDINATE_UNITS = ("m", "m", "m", "rad", "rad", "rad")


class CanonicalMotionBankGateError(ValueError):
    """The bank cannot be accepted without weakening a post-build contract."""


@dataclass(frozen=True)
class _BoundFiles:
    manifest: Path
    bank_dir: Path
    recipe: Path
    compiler: Path
    geometry_tool: Path
    ready: Path
    mjcf: Path
    urdf: Path
    body_order: Path
    hashes: Mapping[str, str]
    compiled_signature: str


@dataclass(frozen=True)
class _ValidatedCompilerOptions:
    sha256: str
    samples_per_scaled_unit: float
    min_connector_intervals: int
    min_core_intervals: int


RecipeLoader = Callable[[Path], CanonicalMotionRecipe]
PlantLoader = Callable[[Path, Path, str, str], Any]
PlayerRunner = Callable[[motion_player.MotionClip], Mapping[str, Any]]
DynamicsRunner = Callable[[Path, Any], Mapping[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CanonicalMotionBankGateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CanonicalMotionBankGateError(f"{label} must be a lowercase SHA-256 string")
    text = value.lower()
    if (
        value != text
        or len(text) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise CanonicalMotionBankGateError(
            f"{label} must be exactly 64 lowercase SHA-256 hex digits"
        )
    return text


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CanonicalMotionBankGateError(
            f"{label} must be one regular, non-symlink file: {path}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalMotionBankGateError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CanonicalMotionBankGateError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalMotionBankGateError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CanonicalMotionBankGateError(f"{label} must contain one JSON object")
    return value


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalMotionBankGateError(f"{label} must be an object")
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise CanonicalMotionBankGateError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalMotionBankGateError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CanonicalMotionBankGateError(f"{label} must be an array")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CanonicalMotionBankGateError(f"{label} must be a JSON boolean")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CanonicalMotionBankGateError(f"{label} must be an integer >= {minimum}")
    return value


def _frame_indices(
    value: Any,
    label: str,
    *,
    frame_count: int,
    unique_sorted: bool,
) -> list[int]:
    rows = _sequence(value, label)
    indices: list[int] = []
    for index, raw in enumerate(rows):
        frame = _integer(raw, f"{label}[{index}]")
        if frame >= frame_count:
            raise CanonicalMotionBankGateError(
                f"{label}[{index}] leaves the {frame_count}-frame motion"
            )
        indices.append(frame)
    if unique_sorted and indices != sorted(set(indices)):
        raise CanonicalMotionBankGateError(f"{label} must contain sorted unique frame indices")
    return indices


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise CanonicalMotionBankGateError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalMotionBankGateError(f"{label} must be finite numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise CanonicalMotionBankGateError(
            f"{label} must be finite" + (f" and >= {minimum}" if minimum is not None else "")
        )
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalMotionBankGateError(f"{label} must be a non-empty string")
    return value


def _resolve_bound_path(value: Any, base: Path, label: str) -> Path:
    text = _nonempty_string(value, label)
    raw = Path(text).expanduser()
    lexical = raw if raw.is_absolute() else base / raw
    lexical = Path(os.path.abspath(os.fspath(lexical)))
    if not lexical.is_file() or lexical.is_symlink():
        raise CanonicalMotionBankGateError(
            f"{label} must resolve to a regular, non-symlink file: {lexical}"
        )
    return lexical.resolve()


def _same_path(actual: Path, expected: Path, label: str) -> None:
    if actual.is_symlink() or expected.is_symlink():
        raise CanonicalMotionBankGateError(f"{label} may not be a symlink")
    if actual.resolve() != expected.resolve():
        raise CanonicalMotionBankGateError(
            f"{label} path changed: expected {expected.resolve()}, got {actual.resolve()}"
        )


def _same_hash(path: Path, expected: Any, label: str) -> str:
    bound = _digest(expected, f"{label} expected hash")
    actual = _sha256_file(path)
    if actual != bound:
        raise CanonicalMotionBankGateError(
            f"{label} SHA-256 mismatch: expected {bound}, got {actual}"
        )
    return actual


def _numeric_vector(
    value: Any,
    length: int,
    label: str,
    *,
    strictly_positive: bool = False,
) -> np.ndarray:
    rows = _sequence(value, label)
    if len(rows) != length or any(isinstance(item, bool) for item in rows):
        raise CanonicalMotionBankGateError(
            f"{label} must contain exactly {length} real numbers"
        )
    try:
        result = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalMotionBankGateError(
            f"{label} must contain exactly {length} real numbers"
        ) from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise CanonicalMotionBankGateError(
            f"{label} must contain exactly {length} finite real numbers"
        )
    if strictly_positive and np.any(result <= 0.0):
        raise CanonicalMotionBankGateError(f"{label} must be strictly positive")
    return np.ascontiguousarray(result)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_array_sha256(value: np.ndarray, claimed: Any, label: str) -> str:
    expected = _digest(claimed, f"{label} sha256")
    actual = _array_sha256(value)
    if actual != expected:
        raise CanonicalMotionBankGateError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _validate_compiler_options(
    raw: Any,
    recipe: CanonicalMotionRecipe,
) -> _ValidatedCompilerOptions:
    """Strictly close the complete effective non-recipe compiler contract."""

    options = _exact_keys(
        raw,
        frozenset(
            {
                "joint_acceleration_limits_rad_s2",
                "full_root_limits",
                "s0_full_grounding_offset_m",
                "geometry_and_grid",
                "face_manifold",
                "effective_path_coordinate_contracts",
                "compiler_options_sha256",
            }
        ),
        "compiler_options",
    )
    claimed_options_sha = _digest(
        options["compiler_options_sha256"], "compiler_options sha256"
    )
    without_digest = dict(options)
    del without_digest["compiler_options_sha256"]
    try:
        canonical = json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalMotionBankGateError(
            f"compiler_options is not canonical finite JSON: {exc}"
        ) from exc
    actual_options_sha = hashlib.sha256(canonical).hexdigest()
    if claimed_options_sha != actual_options_sha:
        raise CanonicalMotionBankGateError(
            "compiler_options_sha256 does not bind the exact option receipt"
        )

    joint_row = _exact_keys(
        options["joint_acceleration_limits_rad_s2"],
        frozenset({"joint_order", "values", "sha256"}),
        "compiler_options joint acceleration",
    )
    if tuple(_sequence(joint_row["joint_order"], "joint acceleration joint_order")) != (
        motion_player.RUNTIME_JOINT_NAMES
    ):
        raise CanonicalMotionBankGateError(
            "compiler joint-acceleration order changed"
        )
    joint_acceleration = _numeric_vector(
        joint_row["values"],
        31,
        "compiler joint acceleration values",
        strictly_positive=True,
    )
    _validate_array_sha256(
        joint_acceleration, joint_row["sha256"], "compiler joint acceleration"
    )

    root_row = _exact_keys(
        options["full_root_limits"],
        frozenset(
            {
                "coordinate_order",
                "coordinate_units",
                "position_lower",
                "position_upper",
                "velocity",
                "acceleration",
                "sha256",
            }
        ),
        "compiler_options full_root_limits",
    )
    if (
        tuple(_sequence(root_row["coordinate_order"], "root coordinate_order"))
        != _ROOT_COORDINATE_NAMES
        or tuple(_sequence(root_row["coordinate_units"], "root coordinate_units"))
        != _ROOT_COORDINATE_UNITS
    ):
        raise CanonicalMotionBankGateError("full-root coordinate contract changed")
    root_lower = _numeric_vector(root_row["position_lower"], 6, "root position_lower")
    root_upper = _numeric_vector(root_row["position_upper"], 6, "root position_upper")
    root_velocity = _numeric_vector(
        root_row["velocity"], 6, "root velocity", strictly_positive=True
    )
    root_acceleration = _numeric_vector(
        root_row["acceleration"], 6, "root acceleration", strictly_positive=True
    )
    if np.any(root_lower >= root_upper):
        raise CanonicalMotionBankGateError(
            "full-root lower limits must be strictly below upper limits"
        )
    root_digest = hashlib.sha256(
        b"".join(
            bytes.fromhex(_array_sha256(value))
            for value in (
                root_lower,
                root_upper,
                root_velocity,
                root_acceleration,
            )
        )
    ).hexdigest()
    if _digest(root_row["sha256"], "full-root sha256") != root_digest:
        raise CanonicalMotionBankGateError("full-root SHA-256 receipt is false")

    grounding = options["s0_full_grounding_offset_m"]
    if isinstance(grounding, bool) or not isinstance(grounding, (int, float)):
        raise CanonicalMotionBankGateError(
            "s0_full_grounding_offset_m must be one finite number"
        )
    grounding_value = float(grounding)
    if not math.isfinite(grounding_value) or grounding_value < 0.0:
        raise CanonicalMotionBankGateError(
            "s0_full_grounding_offset_m must be finite and non-negative"
        )
    s0_specs = [
        spec
        for spec in _sequence(recipe.raw.get("motion_specs"), "recipe motion_specs")
        if isinstance(spec, Mapping) and spec.get("motion_id") == "s0_highpress"
    ]
    if len(s0_specs) != 1:
        raise CanonicalMotionBankGateError(
            "recipe must contain exactly one s0_highpress motion spec"
        )
    try:
        maximum_grounding = float(
            s0_specs[0]["scope_overrides"]["full"]["maximum_grounding_offset_m"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalMotionBankGateError(
            "recipe s0 grounding ceiling is missing"
        ) from exc
    if (
        not math.isfinite(maximum_grounding)
        or grounding_value > maximum_grounding + _FLOAT_TOL
    ):
        raise CanonicalMotionBankGateError(
            "s0 grounding offset exceeds the content-addressed recipe ceiling"
        )

    grid = _exact_keys(
        options["geometry_and_grid"],
        frozenset(
            {
                "samples_per_scaled_unit",
                "min_connector_intervals",
                "min_core_intervals",
                "grid_subdivisions",
                "search_workers",
                "search_parallel_backend",
                "probe_entry_band",
                "probe_exit_band",
                "probe_band_is_exhaustive_enumeration",
                "probe_exact_pointwise_caps",
            }
        ),
        "compiler_options geometry_and_grid",
    )
    # Banded or exact-pointwise builds are timing probes only; the
    # independent verifier accepts exclusively exhaustive-enumeration banks
    # solved under the interval-certified ladder.
    if (
        grid["probe_entry_band"] is not None
        or grid["probe_exit_band"] is not None
        or grid["probe_band_is_exhaustive_enumeration"] is not True
        or grid["probe_exact_pointwise_caps"] is not False
    ):
        raise CanonicalMotionBankGateError(
            "banded probe enumeration is not verifiable; rebuild exhaustively"
        )
    samples = _finite(
        grid["samples_per_scaled_unit"],
        "samples_per_scaled_unit",
        minimum=0.0,
    )
    if samples <= 0.0:
        raise CanonicalMotionBankGateError(
            "samples_per_scaled_unit must be strictly positive"
        )
    min_connector = _integer(
        grid["min_connector_intervals"], "min_connector_intervals", minimum=5
    )
    min_core = _integer(
        grid["min_core_intervals"], "min_core_intervals", minimum=5
    )
    _integer(grid["grid_subdivisions"], "grid_subdivisions", minimum=2)
    workers = _integer(grid["search_workers"], "search_workers", minimum=1)
    if workers > 64:
        raise CanonicalMotionBankGateError("search_workers cannot exceed 64")
    if grid["search_parallel_backend"] not in {"thread", "process"}:
        raise CanonicalMotionBankGateError(
            "search_parallel_backend must be thread or process"
        )

    face = _exact_keys(
        options["face_manifold"],
        frozenset(
            {
                "effective_config",
                "active_candidate_seed_count",
                "active_candidate_seeds_sha256",
            }
        ),
        "compiler_options face_manifold",
    )
    if _integer(
        face["active_candidate_seed_count"],
        "active_candidate_seed_count",
    ) != 0:
        raise CanonicalMotionBankGateError(
            "formal bank gate accepts only the zero-extra-seed face solve"
        )
    empty_seed_digest = hashlib.sha256(
        b"canonical-face-active-seeds-v1\0" + b"0"
    ).hexdigest()
    if (
        _digest(
            face["active_candidate_seeds_sha256"],
            "active_candidate_seeds_sha256",
        )
        != empty_seed_digest
    ):
        raise CanonicalMotionBankGateError(
            "zero-seed face receipt has the wrong SHA-256"
        )
    config = _mapping(face["effective_config"], "face effective_config")
    expected_face_keys = frozenset(
        {
            "mode",
            "position_tolerance_m",
            "normal_tolerance_rad",
            "orientation_tolerance_rad",
            "velocity_tolerance_mps",
            "joint_limit_tolerance_rad",
            "max_step_rad",
            "first_frame_max_delta_from_ready_rad",
            "finite_difference_step_rad",
            "position_residual_scale_m",
            "damping",
            "max_iterations",
            "random_restarts",
            "random_seed",
            "max_solver_step_rad",
            "candidate_dedup_tolerance_rad",
            "candidate_pool_limit",
            "continuation_passes",
            "connector_dynamics_samples",
            "velocity_limit_fraction",
        }
    )
    _exact_keys(config, expected_face_keys, "face effective_config")
    if config["mode"] != "normal":
        raise CanonicalMotionBankGateError("formal face mode must be normal")
    positive_face = (
        "position_tolerance_m",
        "normal_tolerance_rad",
        "orientation_tolerance_rad",
        "velocity_tolerance_mps",
        "finite_difference_step_rad",
        "position_residual_scale_m",
        "damping",
        "max_step_rad",
        "max_solver_step_rad",
        "candidate_dedup_tolerance_rad",
        "velocity_limit_fraction",
    )
    for key in positive_face:
        value = _finite(config[key], f"face {key}", minimum=0.0)
        if value <= 0.0:
            raise CanonicalMotionBankGateError(f"face {key} must be positive")
    if (
        float(config["velocity_limit_fraction"])
        != float(recipe.raw["time_law"]["joint_velocity_limit_fraction"])
    ):
        raise CanonicalMotionBankGateError(
            "face velocity fraction disagrees with the recipe"
        )
    first_delta = config["first_frame_max_delta_from_ready_rad"]
    if first_delta is not None and _finite(
        first_delta, "face first_frame_max_delta_from_ready_rad", minimum=0.0
    ) <= 0.0:
        raise CanonicalMotionBankGateError(
            "face first-frame ready delta must be positive or null"
        )
    for key, minimum in (
        ("max_iterations", 1),
        ("random_restarts", 0),
        ("candidate_pool_limit", 1),
        ("continuation_passes", 0),
        ("connector_dynamics_samples", 2),
    ):
        _integer(config[key], f"face {key}", minimum=minimum)
    if isinstance(config["random_seed"], bool) or not isinstance(
        config["random_seed"], int
    ):
        raise CanonicalMotionBankGateError("face random_seed must be an integer")
    _finite(
        config["joint_limit_tolerance_rad"],
        "face joint_limit_tolerance_rad",
        minimum=0.0,
    )

    contracts = _exact_keys(
        options["effective_path_coordinate_contracts"],
        frozenset(SCOPES),
        "effective_path_coordinate_contracts",
    )
    checked_contracts: dict[str, Mapping[str, Any]] = {}
    checked_arrays: dict[str, tuple[np.ndarray, ...]] = {}
    contract_keys = frozenset(
        {
            "dimension",
            "coordinate_order",
            "coordinate_units",
            "layout",
            "position_lower",
            "position_upper",
            "velocity",
            "acceleration",
            "coordinate_scale",
            "sha256",
        }
    )
    for scope, dimension in (("upper", 31), ("full", 37)):
        contract = _exact_keys(
            contracts[scope], contract_keys, f"{scope} path contract"
        )
        if _integer(contract["dimension"], f"{scope} dimension", minimum=1) != dimension:
            raise CanonicalMotionBankGateError(
                f"{scope} path contract dimension changed"
            )
        expected_order = (
            motion_player.RUNTIME_JOINT_NAMES
            if scope == "upper"
            else motion_player.RUNTIME_JOINT_NAMES + _ROOT_COORDINATE_NAMES
        )
        expected_units = (
            ("rad",) * 31
            if scope == "upper"
            else ("rad",) * 31 + _ROOT_COORDINATE_UNITS
        )
        expected_layout = (
            "joint_31"
            if scope == "upper"
            else (
                "joint_31_then_root_position_xyz_then_"
                "ready_relative_root_rotation_vector_xyz"
            )
        )
        if (
            tuple(_sequence(contract["coordinate_order"], f"{scope} coordinate_order"))
            != expected_order
            or tuple(
                _sequence(contract["coordinate_units"], f"{scope} coordinate_units")
            )
            != expected_units
            or contract["layout"] != expected_layout
        ):
            raise CanonicalMotionBankGateError(
                f"{scope} path coordinate order/units/layout changed"
            )
        lower = _numeric_vector(
            contract["position_lower"], dimension, f"{scope} position_lower"
        )
        upper = _numeric_vector(
            contract["position_upper"], dimension, f"{scope} position_upper"
        )
        velocity = _numeric_vector(
            contract["velocity"],
            dimension,
            f"{scope} velocity",
            strictly_positive=True,
        )
        acceleration = _numeric_vector(
            contract["acceleration"],
            dimension,
            f"{scope} acceleration",
            strictly_positive=True,
        )
        scale = _numeric_vector(
            contract["coordinate_scale"],
            dimension,
            f"{scope} coordinate_scale",
            strictly_positive=True,
        )
        if np.any(lower >= upper) or not np.array_equal(scale, 1.0 / velocity):
            raise CanonicalMotionBankGateError(
                f"{scope} bounds/coordinate scale receipt is inconsistent"
            )
        contract_digest = hashlib.sha256(
            b"".join(
                bytes.fromhex(_array_sha256(value))
                for value in (lower, upper, velocity, acceleration, scale)
            )
        ).hexdigest()
        if _digest(contract["sha256"], f"{scope} contract sha256") != contract_digest:
            raise CanonicalMotionBankGateError(
                f"{scope} path contract SHA-256 receipt is false"
            )
        checked_contracts[scope] = contract
        checked_arrays[scope] = (lower, upper, velocity, acceleration, scale)

    upper_arrays = checked_arrays["upper"]
    full_arrays = checked_arrays["full"]
    for index in range(5):
        if not np.array_equal(full_arrays[index][:31], upper_arrays[index]):
            raise CanonicalMotionBankGateError(
                "full/upper joint path contracts disagree"
            )
    if (
        not np.array_equal(upper_arrays[3], joint_acceleration)
        or not np.array_equal(full_arrays[0][31:], root_lower)
        or not np.array_equal(full_arrays[1][31:], root_upper)
        or not np.array_equal(full_arrays[2][31:], root_velocity)
        or not np.array_equal(full_arrays[3][31:], root_acceleration)
    ):
        raise CanonicalMotionBankGateError(
            "effective path contracts disagree with compiler option arrays"
        )
    return _ValidatedCompilerOptions(
        sha256=actual_options_sha,
        samples_per_scaled_unit=samples,
        min_connector_intervals=min_connector,
        min_core_intervals=min_core,
    )


def _validate_top_contract(manifest: Mapping[str, Any]) -> None:
    _exact_keys(manifest, _MANIFEST_KEYS, "build manifest")
    if manifest["schema_version"] != 1:
        raise CanonicalMotionBankGateError("build manifest schema_version must be 1")
    _nonempty_string(manifest["library_id"], "library_id")
    if manifest["publication_class"] != "compiler_candidate":
        raise CanonicalMotionBankGateError("publication_class must remain compiler_candidate")
    if manifest["build_verdict"] != "PASS_COMPILER_CANDIDATE_ONLY":
        raise CanonicalMotionBankGateError(
            "build_verdict must remain PASS_COMPILER_CANDIDATE_ONLY"
        )
    if (
        _bool(manifest["training_authorized"], "training_authorized") is not False
        or _bool(manifest["hardware_authorized"], "hardware_authorized") is not False
    ):
        raise CanonicalMotionBankGateError(
            "compiler manifest may not authorize training or hardware"
        )

    matrix = _mapping(manifest["output_matrix"], "output_matrix")
    if frozenset(matrix) != frozenset({"motion_ids", "scopes", "candidate_count"}):
        raise CanonicalMotionBankGateError("output_matrix keys changed")
    if tuple(_sequence(matrix["motion_ids"], "output_matrix.motion_ids")) != MOTION_IDS:
        raise CanonicalMotionBankGateError("output_matrix must contain the exact five motions")
    if tuple(_sequence(matrix["scopes"], "output_matrix.scopes")) != SCOPES:
        raise CanonicalMotionBankGateError("output_matrix scopes must be upper then full")
    if matrix["candidate_count"] != len(EXPECTED_MATRIX):
        raise CanonicalMotionBankGateError("output_matrix candidate_count must be 10")

    search = _mapping(manifest["search_contract"], "search_contract")
    if search.get("entry_exit") != "enumerate_all_then_gate_and_rank":
        raise CanonicalMotionBankGateError("entry/exit search is not exhaustive")
    if search.get("adv2c3") != "comparator_only_not_default":
        raise CanonicalMotionBankGateError("adv2c3 was promoted above comparator-only")
    if "no_scalar_braking_through_window_end" not in str(search.get("ranking", "")):
        raise CanonicalMotionBankGateError(
            "search ranking lost the no-scalar-braking-through-window-end criterion"
        )

    opportunity = _mapping(
        manifest["contact_opportunity_contract"], "contact_opportunity_contract"
    )
    expected_opportunity = {
        "marker_only": True,
        "pose_speed_acceleration_freeze": False,
        "acceleration_allowed_through_window_end": True,
        "nonnegative_scalar_acceleration_through_window_end": True,
    }
    if dict(opportunity) != expected_opportunity:
        raise CanonicalMotionBankGateError(
            "contact opportunity must be marker-only, unfrozen, and allow "
            "acceleration through window end"
        )
    if manifest["time_law_claim"] != (
        "weighted_arc_scalar_coordinate_only; "
        "kinematic_velocity_acceleration_and_no_early_scalar_braking_"
        "warm_start_only"
    ):
        raise CanonicalMotionBankGateError("time-law claim was strengthened")

    gates = _sequence(manifest["post_build_gates"], "post_build_gates")
    if not gates:
        raise CanonicalMotionBankGateError("post_build_gates may not be empty")
    names: list[str] = []
    for index, row_raw in enumerate(gates):
        row = _mapping(row_raw, f"post_build_gates[{index}]")
        if frozenset(row) != frozenset({"name", "status"}):
            raise CanonicalMotionBankGateError(f"post_build_gates[{index}] keys changed")
        names.append(_nonempty_string(row["name"], f"post_build_gates[{index}].name"))
        if row["status"] != "pending":
            raise CanonicalMotionBankGateError(
                "compiler may not pre-pass its independent post-build gates"
            )
    if len(names) != len(set(names)):
        raise CanonicalMotionBankGateError("post_build_gates contain duplicate names")

    non_claims = set(_sequence(manifest["non_claims"], "non_claims"))
    for required in (
        "inverse_dynamics_feasibility",
        "balance",
        "training_authorization",
        "hardware_authorization",
    ):
        if required not in non_claims:
            raise CanonicalMotionBankGateError(f"manifest non_claims lost {required!r}")


def _load_and_bind_files(
    manifest_path: Path,
    bank_dir: Path,
    manifest: Mapping[str, Any],
    *,
    mjcf_path: Path,
    urdf_path: Path,
    body_order_path: Path,
    expected_compiled_signature: str,
    recipe_loader: RecipeLoader,
) -> tuple[_BoundFiles, CanonicalMotionRecipe]:
    recipe_row = _exact_keys(manifest["recipe"], frozenset({"path", "sha256"}), "manifest.recipe")
    compiler_row = _exact_keys(
        manifest["compiler"], frozenset({"path", "sha256"}), "manifest.compiler"
    )
    geometry_tool_row = _exact_keys(
        manifest["geometry_tool"],
        frozenset({"path", "sha256"}),
        "manifest.geometry_tool",
    )
    weighted_arc_tool_row = _exact_keys(
        manifest["weighted_arc_tool"],
        frozenset({"path", "sha256"}),
        "manifest.weighted_arc_tool",
    )
    ready_row = _exact_keys(
        manifest["ready"],
        frozenset(
            {
                "path",
                "sha256",
                "direct_endpoint_for_every_motion",
                "old_source_frame_zero_bridge_inserted",
            }
        ),
        "manifest.ready",
    )
    if (
        ready_row["direct_endpoint_for_every_motion"] is not True
        or ready_row["old_source_frame_zero_bridge_inserted"] is not False
    ):
        raise CanonicalMotionBankGateError(
            "every motion must connect directly to canonical ready without an old-frame-0 bridge"
        )

    recipe_path = _resolve_bound_path(
        recipe_row["path"], manifest_path.parent, "manifest recipe path"
    )
    compiler_path = _resolve_bound_path(
        compiler_row["path"], manifest_path.parent, "manifest compiler path"
    )
    geometry_tool_path = _resolve_bound_path(
        geometry_tool_row["path"],
        manifest_path.parent,
        "manifest geometry tool path",
    )
    ready_path = _resolve_bound_path(
        ready_row["path"], manifest_path.parent, "manifest ready path"
    )
    recipe_hash = _same_hash(recipe_path, recipe_row["sha256"], "recipe")
    compiler_hash = _same_hash(compiler_path, compiler_row["sha256"], "compiler")
    geometry_tool_hash = _same_hash(
        geometry_tool_path, geometry_tool_row["sha256"], "geometry tool"
    )
    weighted_arc_tool_path = _resolve_bound_path(
        weighted_arc_tool_row["path"],
        manifest_path.parent,
        "manifest weighted arc tool path",
    )
    _same_hash(
        weighted_arc_tool_path,
        weighted_arc_tool_row["sha256"],
        "weighted arc tool",
    )
    expected_compiler_path = (_HERE / "canonical_motion_compiler.py").resolve()
    expected_geometry_path = (_HERE / "canonical_motion_geometry.py").resolve()
    expected_weighted_arc_path = (
        _HERE / "canonical_weighted_arc_path.py"
    ).resolve()
    _same_path(compiler_path, expected_compiler_path, "formal compiler tool")
    _same_path(
        geometry_tool_path, expected_geometry_path, "formal geometry tool"
    )
    _same_path(
        weighted_arc_tool_path,
        expected_weighted_arc_path,
        "formal weighted arc tool",
    )
    ready_hash = _same_hash(ready_path, ready_row["sha256"], "canonical ready")

    try:
        recipe = recipe_loader(recipe_path)
    except Exception as exc:
        raise CanonicalMotionBankGateError(
            f"strict canonical recipe reload failed: {type(exc).__name__}: {exc}"
        ) from exc
    _same_path(Path(recipe.path), recipe_path, "recipe loader")
    if recipe.raw.get("library_id") != manifest["library_id"]:
        raise CanonicalMotionBankGateError("recipe/manifest library_id mismatch")
    if (
        recipe.raw.get("publication_class") != "compiler_candidate"
        or recipe.raw.get("training_authorized") is not False
        or recipe.raw.get("hardware_authorized") is not False
    ):
        raise CanonicalMotionBankGateError("recipe candidate authorization boundary changed")
    _same_path(Path(recipe.ready.path), ready_path, "canonical ready")
    if recipe.ready.sha256 != ready_hash:
        raise CanonicalMotionBankGateError("recipe/manifest canonical-ready hash mismatch")

    explicit = {
        "mjcf": _resolve_bound_path(os.fspath(mjcf_path), Path.cwd(), "explicit MJCF"),
        "urdf": _resolve_bound_path(os.fspath(urdf_path), Path.cwd(), "explicit URDF"),
        "body_order": _resolve_bound_path(
            os.fspath(body_order_path), Path.cwd(), "explicit body order"
        ),
    }
    hashes: dict[str, str] = {
        "recipe": recipe_hash,
        "compiler": compiler_hash,
        "geometry_tool": geometry_tool_hash,
        "ready": ready_hash,
    }
    for key, path in explicit.items():
        if not path.is_file() or path.is_symlink():
            raise CanonicalMotionBankGateError(
                f"{key} must be a regular, non-symlink file: {path}"
            )
        _same_path(path, Path(recipe.model_paths[key]), f"recipe {key}")
        hashes[key] = _same_hash(path, recipe.model_hashes[key], key)

    try:
        body_names = tuple(
            line.strip()
            for line in explicit["body_order"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as exc:
        raise CanonicalMotionBankGateError(f"cannot read body-order contract: {exc}") from exc
    if body_names != motion_player.RUNTIME_BODY_NAMES:
        raise CanonicalMotionBankGateError(
            "body-order bytes do not decode to the exact 32-name runtime order"
        )

    signature = _digest(expected_compiled_signature, "expected compiled-model signature")
    return (
        _BoundFiles(
            manifest=manifest_path,
            bank_dir=bank_dir,
            recipe=recipe_path,
            compiler=compiler_path,
            geometry_tool=geometry_tool_path,
            ready=ready_path,
            mjcf=explicit["mjcf"],
            urdf=explicit["urdf"],
            body_order=explicit["body_order"],
            hashes=hashes,
            compiled_signature=signature,
        ),
        recipe,
    )


def _validate_bank_file_set(
    bank_dir: Path, outputs: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], tuple[Mapping[str, Any], Path]]:
    if not bank_dir.is_dir() or bank_dir.is_symlink():
        raise CanonicalMotionBankGateError(f"bank directory must be a real directory: {bank_dir}")
    actual_paths = list(bank_dir.glob("*.npz"))
    actual_names = sorted(path.name for path in actual_paths)
    if actual_names != sorted(EXPECTED_FILENAMES):
        missing = sorted(set(EXPECTED_FILENAMES) - set(actual_names))
        extra = sorted(set(actual_names) - set(EXPECTED_FILENAMES))
        raise CanonicalMotionBankGateError(
            f"bank NPZ set is not exactly 5x2; missing={missing}, extra={extra}"
        )
    if any(not path.is_file() or path.is_symlink() for path in actual_paths):
        raise CanonicalMotionBankGateError("bank NPZ files must be regular non-symlinks")
    expected_manifest_sidecars = sorted(
        f"{filename}.manifest.json" for filename in EXPECTED_FILENAMES
    )
    expected_report_sidecars = sorted(f"{filename}.report.json" for filename in EXPECTED_FILENAMES)
    actual_manifest_sidecars = sorted(path.name for path in bank_dir.glob("*.npz.manifest.json"))
    actual_report_sidecars = sorted(path.name for path in bank_dir.glob("*.npz.report.json"))
    if actual_manifest_sidecars != expected_manifest_sidecars:
        raise CanonicalMotionBankGateError(
            "schema2 manifest sidecar set is not exactly 5x2; "
            f"missing={sorted(set(expected_manifest_sidecars) - set(actual_manifest_sidecars))}, "
            f"extra={sorted(set(actual_manifest_sidecars) - set(expected_manifest_sidecars))}"
        )
    if actual_report_sidecars != expected_report_sidecars:
        raise CanonicalMotionBankGateError(
            "schema2 report sidecar set is not exactly 5x2; "
            f"missing={sorted(set(expected_report_sidecars) - set(actual_report_sidecars))}, "
            f"extra={sorted(set(actual_report_sidecars) - set(expected_report_sidecars))}"
        )

    if len(outputs) != len(EXPECTED_MATRIX):
        raise CanonicalMotionBankGateError("manifest outputs must contain exactly 10 rows")
    result: dict[tuple[str, str], tuple[Mapping[str, Any], Path]] = {}
    observed_order: list[tuple[str, str]] = []
    for index, raw in enumerate(outputs):
        row = _exact_keys(raw, _OUTPUT_KEYS, f"outputs[{index}]")
        key = (
            _nonempty_string(row["motion_id"], f"outputs[{index}].motion_id"),
            _nonempty_string(row["scope"], f"outputs[{index}].scope"),
        )
        observed_order.append(key)
        if key in result:
            raise CanonicalMotionBankGateError(f"duplicate manifest output {key}")
        filename = _nonempty_string(row["filename"], f"outputs[{index}].filename")
        expected_filename = f"{key[0]}_{key[1]}_canonical_v2.npz"
        if filename != expected_filename or Path(filename).name != filename:
            raise CanonicalMotionBankGateError(
                f"output {key} filename must be {expected_filename!r}"
            )
        result[key] = (row, bank_dir / filename)
    if tuple(observed_order) != EXPECTED_MATRIX:
        raise CanonicalMotionBankGateError(
            f"manifest output order/matrix changed: {observed_order}"
        )
    return result


def _recipe_source(recipe: CanonicalMotionRecipe, motion_id: str) -> Any:
    sources = getattr(recipe, "sources", ())
    matches = [
        source
        for source in sources
        if getattr(source, "motion_id", None) == motion_id
    ]
    if len(matches) != 1:
        raise CanonicalMotionBankGateError(
            f"recipe must expose exactly one bound source for {motion_id!r}"
        )
    source = matches[0]
    source_path = Path(getattr(source, "path", ""))
    source_sha = _digest(
        getattr(source, "sha256", None), f"{motion_id} recipe source sha256"
    )
    if not source_path.is_file() or source_path.is_symlink():
        raise CanonicalMotionBankGateError(
            f"{motion_id} recipe source must be a regular non-symlink file"
        )
    if _sha256_file(source_path) != source_sha:
        raise CanonicalMotionBankGateError(
            f"{motion_id} recipe source bytes changed after recipe load"
        )
    return source


def _recompute_candidate_input_sha256(
    row: Mapping[str, Any],
    npz_path: Path,
    recipe: CanonicalMotionRecipe,
) -> str:
    """Recompute the compiler-input digest from recipe identity and NPZ bytes."""

    motion_id = _nonempty_string(row["motion_id"], "output motion_id")
    scope = _nonempty_string(row["scope"], "output scope")
    source = _recipe_source(recipe, motion_id)
    header = {
        "digest_contract": "canonical_published_numeric_input_v1",
        "library_id": recipe.raw["library_id"],
        "recipe_sha256": _sha256_file(Path(recipe.path)),
        "ready_sha256": recipe.ready.sha256,
        "source_sha256": source.sha256,
        "motion_id": motion_id,
        "scope": scope,
        "entry_frame": _integer(row["entry_frame"], "output entry_frame"),
        "exit_frame": _integer(row["exit_frame"], "output exit_frame"),
        "window_policy": recipe.raw["time_law"]["legacy_seed_marker_policy"],
        "adv2c3_role": (
            recipe.raw["entry_exit_search"]["historical_adv2c3_role"]
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            for key in _PUBLISHED_INPUT_ARRAY_KEYS:
                if key not in payload.files:
                    raise CanonicalMotionBankGateError(
                        f"{npz_path.name} is missing compiler-input array {key!r}"
                    )
                raw = np.asarray(payload[key])
                if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                    raise CanonicalMotionBankGateError(
                        f"{npz_path.name} compiler-input array {key!r} is not real numeric"
                    )
                array = np.ascontiguousarray(raw, dtype="<f8")
                if not np.all(np.isfinite(array)):
                    raise CanonicalMotionBankGateError(
                        f"{npz_path.name} compiler-input array {key!r} is non-finite"
                    )
                digest.update(key.encode("ascii") + b"\0")
                digest.update(
                    json.dumps(
                        list(array.shape),
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("ascii")
                )
                digest.update(b"\0")
                digest.update(array.tobytes(order="C"))
    except CanonicalMotionBankGateError:
        raise
    except (OSError, ValueError) as exc:
        raise CanonicalMotionBankGateError(
            f"cannot recompute compiler input digest for {npz_path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _validate_schema_receipts(
    row: Mapping[str, Any],
    *,
    npz_path: Path,
    output_sha: str,
    files: _BoundFiles,
    recipe: CanonicalMotionRecipe,
) -> Mapping[str, Any]:
    nested = _mapping(row["schema2_manifest"], "schema2_manifest")
    if (
        nested.get("publication_class") != "compiler_candidate"
        or nested.get("build_verdict") != "PASS_COMPILER_CANDIDATE_ONLY"
        or nested.get("training_authorized") is not False
    ):
        raise CanonicalMotionBankGateError(
            "schema-2 receipt strengthened the compiler-candidate boundary"
        )
    hashes = _mapping(nested.get("hashes"), "schema2_manifest.hashes")
    expected_hashes = {
        "input_sha256": _recompute_candidate_input_sha256(row, npz_path, recipe),
        "ready_sha256": files.hashes["ready"],
        "mjcf_sha256": files.hashes["mjcf"],
        "body_order_sha256": files.hashes["body_order"],
        "output_npz_sha256": output_sha,
    }
    for key, expected in expected_hashes.items():
        if _digest(hashes.get(key), f"schema2_manifest.hashes.{key}") != expected:
            raise CanonicalMotionBankGateError(
                f"schema-2 receipt {key} does not bind the verified bytes"
            )
    _digest(hashes.get("tool_sha256"), "schema2_manifest.hashes.tool_sha256")

    runtime = _mapping(nested.get("runtime_contract"), "schema2 runtime_contract")
    if (
        runtime.get("joint_count") != 31
        or tuple(runtime.get("joint_names", ())) != motion_player.RUNTIME_JOINT_NAMES
        or runtime.get("body_count") != 32
        or tuple(runtime.get("body_names", ())) != motion_player.RUNTIME_BODY_NAMES
        or runtime.get("schema2_field_count") not in (11, 14)
    ):
        raise CanonicalMotionBankGateError("schema-2 runtime order/schema receipt changed")
    kinematics = _mapping(nested.get("kinematics"), "schema2 kinematics")
    if (
        kinematics.get("body_lin_vel_point") != BODY_LIN_VEL_POINT
        or kinematics.get("body_velocity_method") != "mujoco_mj_jacBodyCom_times_qvel"
        or kinematics.get("pose_finite_difference_used") is not False
        or kinematics.get("static_ready_endpoints_required") is not True
    ):
        raise CanonicalMotionBankGateError("schema-2 kinematics receipt changed")

    nested_files = _mapping(nested.get("files"), "schema2 files")
    _same_path(Path(nested_files.get("mjcf_path", "")), files.mjcf, "schema2 MJCF")
    _same_path(
        Path(nested_files.get("body_order_path", "")),
        files.body_order,
        "schema2 body order",
    )
    tool_path = Path(_nonempty_string(nested_files.get("tool_path"), "schema2 tool path"))
    if not tool_path.is_file() or tool_path.is_symlink():
        raise CanonicalMotionBankGateError("schema2 tool path is not a regular file")
    expected_tool_path = _HERE / "canonical_schema2_builder.py"
    _same_path(tool_path, expected_tool_path, "schema2 builder tool")
    if _sha256_file(expected_tool_path) != hashes["tool_sha256"]:
        raise CanonicalMotionBankGateError("schema2 tool bytes changed after compilation")

    schema_report = _mapping(row["schema2_report"], "schema2_report")
    if (
        schema_report.get("publication_class") != "compiler_candidate"
        or schema_report.get("training_authorized") is not False
        or schema_report.get("status") != "PASS"
    ):
        raise CanonicalMotionBankGateError("schema2 report candidate boundary changed")
    report_hashes = _mapping(schema_report.get("hashes"), "schema2_report.hashes")
    if report_hashes != hashes:
        raise CanonicalMotionBankGateError("schema2 manifest/report hash receipts disagree")
    checks = _mapping(schema_report.get("checks"), "schema2_report.checks")
    if (
        checks.get("same_ready_pose_first_last") is not True
        or checks.get("six_velocity_channels_zero_first_last") is not True
    ):
        raise CanonicalMotionBankGateError("schema2 report did not claim static ready endpoints")
    manifest_sidecar_path = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
    report_sidecar_path = npz_path.with_suffix(npz_path.suffix + ".report.json")
    manifest_sidecar = _strict_json(
        manifest_sidecar_path, f"{npz_path.name} schema2 manifest sidecar"
    )
    report_sidecar = _strict_json(report_sidecar_path, f"{npz_path.name} schema2 report sidecar")
    if manifest_sidecar != nested:
        raise CanonicalMotionBankGateError(
            f"{npz_path.name} manifest sidecar disagrees with BUILD_MANIFEST"
        )
    if report_sidecar != schema_report:
        raise CanonicalMotionBankGateError(
            f"{npz_path.name} report sidecar disagrees with BUILD_MANIFEST"
        )
    return {
        "input_sha256": _digest(hashes["input_sha256"], "schema2 input digest"),
        "builder_tool_sha256": _digest(hashes["tool_sha256"], "schema2 builder tool digest"),
        "manifest_sidecar": {
            "path": str(manifest_sidecar_path),
            "sha256": _sha256_file(manifest_sidecar_path),
        },
        "report_sidecar": {
            "path": str(report_sidecar_path),
            "sha256": _sha256_file(report_sidecar_path),
        },
    }


def _orientation_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    first = np.asarray(lhs, dtype=np.float64)
    second = np.asarray(rhs, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-15)
    second /= max(float(np.linalg.norm(second)), 1.0e-15)
    return float(2.0 * math.acos(float(np.clip(abs(first @ second), 0.0, 1.0))))


def _endpoint_receipt(
    clip: motion_player.MotionClip,
    recipe: CanonicalMotionRecipe,
    canonical_body_pos: np.ndarray | None,
    canonical_body_quat: np.ndarray | None,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    if clip.n_frames < 3:
        raise CanonicalMotionBankGateError("dynamics screen requires at least three frames")
    if clip.body_lin_vel_point != BODY_LIN_VEL_POINT:
        raise CanonicalMotionBankGateError(
            "canonical bank requires body COM velocities, not legacy origin velocities"
        )

    endpoint_zero = {
        "joint_start": np.count_nonzero(clip.joint_vel[0]) == 0,
        "joint_end": np.count_nonzero(clip.joint_vel[-1]) == 0,
        "body_linear_start": np.count_nonzero(clip.body_lin_vel_w[0]) == 0,
        "body_linear_end": np.count_nonzero(clip.body_lin_vel_w[-1]) == 0,
        "body_angular_start": np.count_nonzero(clip.body_ang_vel_w[0]) == 0,
        "body_angular_end": np.count_nonzero(clip.body_ang_vel_w[-1]) == 0,
    }
    if not all(endpoint_zero.values()):
        bad = sorted(key for key, value in endpoint_zero.items() if not value)
        raise CanonicalMotionBankGateError(
            f"all six endpoint velocity classes must be exact zero; failed={bad}"
        )

    if not np.array_equal(clip.joint_pos[0], clip.joint_pos[-1]):
        raise CanonicalMotionBankGateError("motion does not return to its joint ready endpoint")
    if not np.array_equal(clip.body_pos_w[0], clip.body_pos_w[-1]):
        raise CanonicalMotionBankGateError("motion does not return to its 32-body ready positions")
    if not np.array_equal(clip.body_quat_w[0], clip.body_quat_w[-1]):
        raise CanonicalMotionBankGateError(
            "motion does not return to its signed 32-body ready quaternions"
        )

    ready_joint = np.asarray(recipe.ready.joint_pos, dtype=np.float64)
    if not np.array_equal(clip.joint_pos[0], ready_joint):
        raise CanonicalMotionBankGateError("motion joint endpoint is not canonical ready")
    root_position_error = float(np.max(np.abs(clip.body_pos_w[0, 0] - recipe.ready.root_pos_w)))
    root_orientation_error = _orientation_error(
        clip.body_quat_w[0, 0], recipe.ready.root_quat_wxyz
    )
    if root_position_error > _READY_ROOT_TOL or root_orientation_error > _READY_ROOT_TOL:
        raise CanonicalMotionBankGateError(
            "motion root endpoint is not canonical ready: "
            f"position={root_position_error:.6g}, orientation={root_orientation_error:.6g}"
        )

    if canonical_body_pos is not None and not np.array_equal(
        clip.body_pos_w[0], canonical_body_pos
    ):
        raise CanonicalMotionBankGateError("motion bank does not share one body ready pose")
    if canonical_body_quat is not None and not np.array_equal(
        clip.body_quat_w[0], canonical_body_quat
    ):
        raise CanonicalMotionBankGateError(
            "motion bank does not share one signed body ready orientation"
        )
    body_pos = np.array(clip.body_pos_w[0], copy=True)
    body_quat = np.array(clip.body_quat_w[0], copy=True)
    return (
        {
            "shared_joint_ready_exact": True,
            "shared_root_ready_tolerance_m_rad": _READY_ROOT_TOL,
            "root_position_max_abs_error_m": root_position_error,
            "root_orientation_error_rad": root_orientation_error,
            "shared_32_body_ready_exact": True,
            "six_velocity_classes_exact_zero": endpoint_zero,
        },
        body_pos,
        body_quat,
    )


def _canonical_source_frame_map(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype="<f8")).copy()
    if result.ndim != 1 or np.any(np.isinf(result)):
        raise CanonicalMotionBankGateError(
            "source-frame map must be one-dimensional finite-or-NaN float64"
        )
    result.view("<u8")[np.isnan(result)] = _CANONICAL_QNAN_BITS
    return result


def _validate_geometry_source_frame_map(
    geometry: Mapping[str, Any],
    *,
    entry_frame: int,
    exit_frame: int,
    compiler_options: _ValidatedCompilerOptions,
) -> tuple[np.ndarray, list[int]]:
    """Independently rebuild the dense map from the segment interval ledger."""

    receipt = _exact_keys(
        geometry.get("source_frame_map_receipt"),
        frozenset(
            {
                "schema_version",
                "encoding",
                "length",
                "sha256",
                "finite_count",
                "nan_count",
                "finite_value_encoding",
                "segment_intervals",
                "source_waypoint_path_indices",
            }
        ),
        "geometry source_frame_map_receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["encoding"] != _SOURCE_FRAME_MAP_ENCODING
        or receipt["finite_value_encoding"]
        != "piecewise_linear_source_frame_by_segment_v1"
    ):
        raise CanonicalMotionBankGateError(
            "geometry source-frame map encoding contract changed"
        )
    segments = _sequence(
        receipt["segment_intervals"], "source-frame segment_intervals"
    )
    expected_segment_count = exit_frame - entry_frame + 2
    if len(segments) != expected_segment_count:
        raise CanonicalMotionBankGateError(
            "source-frame segment ledger does not cover entry/core/exit"
        )

    arrays: list[np.ndarray] = []
    waypoint_indices = [0]
    running_row = 0
    for index, raw in enumerate(segments):
        segment = _exact_keys(
            raw,
            frozenset(
                {
                    "kind",
                    "sample_intervals",
                    "source_frame_start",
                    "source_frame_end",
                }
            ),
            f"source-frame segment[{index}]",
        )
        intervals = _integer(
            segment["sample_intervals"],
            f"source-frame segment[{index}].sample_intervals",
            minimum=1,
        )
        if index == 0:
            if (
                segment["kind"] != "pre_connector"
                or segment["source_frame_start"] is not None
                or _finite(
                    segment["source_frame_end"],
                    "pre connector source_frame_end",
                    minimum=0.0,
                )
                != float(entry_frame)
                or intervals < compiler_options.min_connector_intervals
            ):
                raise CanonicalMotionBankGateError(
                    "pre-connector source-frame ledger is invalid"
                )
            part = np.full(intervals + 1, np.nan, dtype=np.float64)
            part[-1] = float(entry_frame)
            arrays.append(part)
            running_row = intervals
            waypoint_indices[0] = running_row
            continue
        if index == len(segments) - 1:
            if (
                segment["kind"] != "post_connector"
                or _finite(
                    segment["source_frame_start"],
                    "post connector source_frame_start",
                    minimum=0.0,
                )
                != float(exit_frame)
                or segment["source_frame_end"] is not None
                or intervals < compiler_options.min_connector_intervals
            ):
                raise CanonicalMotionBankGateError(
                    "post-connector source-frame ledger is invalid"
                )
            part = np.full(intervals + 1, np.nan, dtype=np.float64)
            part[0] = float(exit_frame)
            arrays.append(part[1:])
            continue

        source_frame = entry_frame + index - 1
        if (
            segment["kind"] != "source_core"
            or _finite(
                segment["source_frame_start"],
                f"core segment[{index}] source_frame_start",
                minimum=0.0,
            )
            != float(source_frame)
            or _finite(
                segment["source_frame_end"],
                f"core segment[{index}] source_frame_end",
                minimum=0.0,
            )
            != float(source_frame + 1)
            or intervals < compiler_options.min_core_intervals
        ):
            raise CanonicalMotionBankGateError(
                f"source-core segment ledger is invalid at frame {source_frame}"
            )
        part = np.linspace(
            float(source_frame),
            float(source_frame + 1),
            intervals + 1,
            dtype=np.float64,
        )
        arrays.append(part[1:])
        running_row += intervals
        waypoint_indices.append(running_row)

    reconstructed = _canonical_source_frame_map(np.concatenate(arrays))
    length = _integer(receipt["length"], "source-frame map length", minimum=1)
    finite_count = _integer(
        receipt["finite_count"], "source-frame map finite_count"
    )
    nan_count = _integer(receipt["nan_count"], "source-frame map nan_count")
    if (
        length != len(reconstructed)
        or finite_count != int(np.count_nonzero(np.isfinite(reconstructed)))
        or nan_count != int(np.count_nonzero(np.isnan(reconstructed)))
        or finite_count + nan_count != length
    ):
        raise CanonicalMotionBankGateError(
            "source-frame map length/count receipt is false"
        )
    actual_sha = hashlib.sha256(reconstructed.tobytes(order="C")).hexdigest()
    if _digest(receipt["sha256"], "source-frame map sha256") != actual_sha:
        raise CanonicalMotionBankGateError(
            "source-frame map SHA-256 does not match independent reconstruction"
        )
    declared_waypoints = [
        _integer(value, f"source_waypoint_path_indices[{index}]")
        for index, value in enumerate(
            _sequence(
                receipt["source_waypoint_path_indices"],
                "source_waypoint_path_indices",
            )
        )
    ]
    if declared_waypoints != waypoint_indices:
        raise CanonicalMotionBankGateError(
            "source waypoint dense rows disagree with independent reconstruction"
        )
    return reconstructed, waypoint_indices


def _validate_marker_contract(
    row: Mapping[str, Any],
    clip: motion_player.MotionClip,
    recipe: CanonicalMotionRecipe,
    compiler_options: _ValidatedCompilerOptions,
) -> Mapping[str, Any]:
    entry_frame = _integer(row["entry_frame"], "output entry_frame")
    exit_frame = _integer(row["exit_frame"], "output exit_frame")
    if exit_frame < entry_frame:
        raise CanonicalMotionBankGateError("output exit_frame precedes entry_frame")
    duration = (clip.n_frames - 1) / clip.fps
    declared_duration = _finite(row["duration_s"], "output duration_s", minimum=0.0)
    if not math.isclose(duration, declared_duration, rel_tol=0.0, abs_tol=_FLOAT_TOL):
        raise CanonicalMotionBankGateError(
            f"NPZ/manifest duration mismatch: {duration} vs {declared_duration}"
        )

    retiming = _mapping(row["retiming"], "output retiming")
    if (
        retiming.get("constraint_model") != "kinematic_velocity_and_acceleration_only"
        or retiming.get("marker_policy")
        != "selected_marker_nonnegative_scalar_acceleration_no_pose_lock"
        or retiming.get("marker_output_frame_policy")
        != "nearest_sample_observation_only_not_interval_gate"
        or retiming.get("marker_interval_discrete_policy")
        != "inclusive_samples_ceil_start_floor_end"
    ):
        raise CanonicalMotionBankGateError(
            "retiming must preserve marker-only, no-window-lock semantics"
        )
    acceleration_policy = _mapping(
        retiming.get("nonnegative_acceleration_until_marker"),
        "retiming nonnegative_acceleration_until_marker",
    )
    if (
        acceleration_policy.get("enabled") is not True
        or acceleration_policy.get("marker") != "window_end"
        or acceleration_policy.get("grid_node_is_exact_marker") is not True
    ):
        raise CanonicalMotionBankGateError(
            "retiming did not enforce nonnegative scalar acceleration through "
            "the exact window-end marker"
        )
    for key in (
        "prefix_scalar_acceleration_min_continuous",
        "prefix_scalar_acceleration_min_50hz",
    ):
        if (
            _finite(
                acceleration_policy.get(key),
                f"retiming nonnegative acceleration {key}",
            )
            < -1.0e-9
        ):
            raise CanonicalMotionBankGateError(
                f"retiming reports negative scalar acceleration in {key}"
            )
    if (
        _finite(retiming.get("fps"), "retiming fps", minimum=0.0) != clip.fps
        or _integer(retiming.get("output_frames"), "retiming output_frames", minimum=1)
        != clip.n_frames
        or not math.isclose(
            _finite(retiming.get("duration_s"), "retiming duration_s", minimum=0.0),
            duration,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
    ):
        raise CanonicalMotionBankGateError("retiming dimensions do not match NPZ")
    input_samples = _integer(retiming.get("input_samples"), "retiming input_samples", minimum=3)

    geometry = _mapping(row["geometry"], "output geometry")
    geometry_selection = _mapping(geometry.get("selection"), "geometry selection")
    geometry_sampling = _mapping(geometry.get("sampling"), "geometry sampling")
    geometry_waypoints = _mapping(geometry.get("source_waypoints"), "geometry source_waypoints")
    geometry_parameterization = _mapping(
        geometry.get("parameterization"), "geometry parameterization"
    )
    if (
        geometry.get("c2_continuous") is not True
        or geometry_selection.get("entry_frame") != entry_frame
        or geometry_selection.get("exit_frame") != exit_frame
        or geometry_selection.get("old_source_frame_zero_retained") is not (entry_frame == 0)
        or _integer(
            geometry_sampling.get("path_points"),
            "geometry sampling path_points",
            minimum=3,
        )
        != input_samples
        or _integer(
            geometry_sampling.get("min_core_intervals"),
            "geometry sampling min_core_intervals",
            minimum=5,
        )
        != compiler_options.min_core_intervals
        or _finite(
            geometry_parameterization.get(
                "samples_per_scaled_coordinate_unit"
            ),
            "geometry samples_per_scaled_coordinate_unit",
            minimum=0.0,
        )
        != compiler_options.samples_per_scaled_unit
        or _integer(
            geometry_waypoints.get("count"),
            "geometry source waypoint count",
            minimum=1,
        )
        != exit_frame - entry_frame + 1
        or _finite(
            geometry_waypoints.get("mapped_frame_min"),
            "geometry mapped_frame_min",
            minimum=0.0,
        )
        != float(entry_frame)
        or _finite(
            geometry_waypoints.get("mapped_frame_max"),
            "geometry mapped_frame_max",
            minimum=0.0,
        )
        != float(exit_frame)
    ):
        raise CanonicalMotionBankGateError(
            "geometry selection/source-waypoint receipt disagrees with output entry/exit"
        )
    source_frame_map, source_waypoint_rows = _validate_geometry_source_frame_map(
        geometry,
        entry_frame=entry_frame,
        exit_frame=exit_frame,
        compiler_options=compiler_options,
    )
    if len(source_frame_map) != input_samples:
        raise CanonicalMotionBankGateError(
            "independently rebuilt source-frame map length disagrees with retimer input"
        )

    scope_receipt = _mapping(row["scope_preprocessing"], "scope_preprocessing")
    pure_math = _mapping(
        scope_receipt.get("pure_math_only"),
        "scope preprocessing pure_math_only",
    )
    if (
        scope_receipt.get("scope") != row["scope"]
        or scope_receipt.get("joint_count") != 31
        or tuple(scope_receipt.get("joint_names", ())) != motion_player.RUNTIME_JOINT_NAMES
        or pure_math.get("time_law_changed") is not False
    ):
        raise CanonicalMotionBankGateError(
            "scope preprocessing receipt disagrees with runtime scope/order"
        )

    markers = _mapping(retiming.get("markers"), "retiming markers")
    if frozenset(markers) != frozenset(_MARKERS):
        raise CanonicalMotionBankGateError(f"retiming markers must be exactly {_MARKERS}")
    marker_rows: list[Mapping[str, Any]] = []
    for name in _MARKERS:
        marker = _exact_keys(markers[name], _MARKER_KEYS, f"marker {name}")
        source_index = _finite(marker["source_index"], f"{name}.source_index", minimum=0.0)
        time_s = _finite(marker["time_s"], f"{name}.time_s", minimum=0.0)
        fractional = _finite(
            marker["output_fractional_frame"],
            f"{name}.output_fractional_frame",
            minimum=0.0,
        )
        output_frame = _integer(marker["output_frame"], f"{name}.output_frame")
        path_position = _finite(
            marker["path_position_at_frame"],
            f"{name}.path_position_at_frame",
            minimum=0.0,
        )
        if time_s > duration + _FLOAT_TOL or output_frame >= clip.n_frames:
            raise CanonicalMotionBankGateError(f"marker {name} leaves the output clip")
        if source_index > input_samples - 1 + _FLOAT_TOL:
            raise CanonicalMotionBankGateError(
                f"marker {name} source index leaves the selected geometry"
            )
        if path_position > input_samples - 1 + _FLOAT_TOL:
            raise CanonicalMotionBankGateError(
                f"marker {name} observed path position leaves the selected geometry"
            )
        if not math.isclose(fractional, time_s * clip.fps, rel_tol=0.0, abs_tol=_FLOAT_TOL):
            raise CanonicalMotionBankGateError(f"marker {name} fractional frame/time mismatch")
        expected_frame = int(np.clip(np.rint(fractional), 0, clip.n_frames - 1))
        if output_frame != expected_frame:
            raise CanonicalMotionBankGateError(
                f"marker {name} does not use nearest-sample observation"
            )
        marker_rows.append(
            {
                "name": name,
                "source_index": source_index,
                "time_s": time_s,
                "output_fractional_frame": fractional,
                "output_frame": output_frame,
                "path_position_at_frame": path_position,
            }
        )

    source_indices = [row_["source_index"] for row_ in marker_rows]
    times = [row_["time_s"] for row_ in marker_rows]
    fractional_frames = [row_["output_fractional_frame"] for row_ in marker_rows]
    output_frames = [row_["output_frame"] for row_ in marker_rows]
    path_positions = [row_["path_position_at_frame"] for row_ in marker_rows]
    # The reviewed anchor is not aliased into the legacy seed window, so only
    # window_start -> window_end must be strictly ordered; the anchor may sit
    # on either side of that span while staying inside the retained core.
    if not (
        all(
            math.isclose(value, round(value), rel_tol=0.0, abs_tol=_FLOAT_TOL)
            for value in source_indices
        )
        and source_indices[0] <= source_indices[2]
        and times[0] <= times[2]
        and fractional_frames[0] <= fractional_frames[2]
        and output_frames[0] <= output_frames[2]
        and path_positions[0] <= path_positions[2]
        and times[0] < times[2]
    ):
        raise CanonicalMotionBankGateError(
            "window start/end indices and times must be monotonic"
        )
    declared_times = (
        _finite(row["contact_window_start_s"], "contact_window_start_s", minimum=0.0),
        _finite(row["source_anchor_time_s"], "source_anchor_time_s", minimum=0.0),
        _finite(row["contact_window_end_s"], "contact_window_end_s", minimum=0.0),
    )
    if any(
        not math.isclose(actual, declared, rel_tol=0.0, abs_tol=_FLOAT_TOL)
        for actual, declared in zip(times, declared_times)
    ):
        raise CanonicalMotionBankGateError(
            "top-level output window times disagree with retiming markers"
        )

    search = _mapping(row["search"], "output search")
    selected = _mapping(search.get("selected"), "search selected")
    if (
        selected.get("entry_frame") != entry_frame
        or selected.get("exit_frame") != exit_frame
        or selected.get("old_frame_zero_forced") is not False
        or selected.get("direct_path") != "canonical_ready_to_selected_core_to_canonical_ready"
        or not math.isclose(
            _finite(selected.get("duration_s"), "search selected duration_s"),
            declared_duration,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
    ):
        raise CanonicalMotionBankGateError(
            "search selected receipt disagrees with output entry/exit/duration"
        )
    opportunity = _mapping(search.get("contact_opportunity"), "search contact_opportunity")
    expected_flags = {
        "marker_only": True,
        "pose_locked": False,
        "velocity_locked": False,
        "acceleration_locked": False,
        "acceleration_allowed_through_window_end": True,
    }
    for key, expected in expected_flags.items():
        if opportunity.get(key) is not expected:
            raise CanonicalMotionBankGateError(f"search contact opportunity weakened {key}")
    if opportunity.get("nonnegative_scalar_acceleration_through_window_end") is not True:
        raise CanonicalMotionBankGateError(
            "search lost nonnegative scalar acceleration through window end"
        )
    no_brake = _mapping(
        retiming.get("scalar_no_early_brake_proxy"),
        "retiming scalar_no_early_brake_proxy",
    )
    if (
        no_brake.get("name") != "scalar_no_early_brake_proxy_v2"
        or no_brake.get("selection_criterion") is not True
        or no_brake.get("proxy_only") is not True
        or no_brake.get("negative_segment_count_before_window_end") != 0
        or no_brake.get("no_negative_scalar_acceleration_before_window_end") is not True
        or not math.isclose(
            _finite(
                no_brake.get("window_start_fractional_frame"),
                "scalar no-brake window_start_fractional_frame",
            ),
            fractional_frames[0],
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
        or not math.isclose(
            _finite(
                no_brake.get("window_end_fractional_frame"),
                "scalar no-brake window_end_fractional_frame",
            ),
            fractional_frames[2],
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
    ):
        raise CanonicalMotionBankGateError(
            "scalar no-early-brake receipt is missing or contradictory"
        )
    if not math.isclose(
        _finite(
            acceleration_policy.get("marker_source_index"),
            "nonnegative acceleration marker_source_index",
        ),
        source_indices[2],
        rel_tol=0.0,
        abs_tol=_FLOAT_TOL,
    ):
        raise CanonicalMotionBankGateError(
            "nonnegative-acceleration policy is not bound to the window-end geometry marker"
        )
    search_source = _sequence(
        opportunity.get("source_span_inclusive"),
        "search contact_opportunity.source_span_inclusive",
    )
    motion_specs = _sequence(recipe.raw.get("motion_specs"), "recipe motion_specs")
    matching_specs = [
        _mapping(spec, "recipe motion spec")
        for spec in motion_specs
        if isinstance(spec, Mapping) and spec.get("motion_id") == row["motion_id"]
    ]
    if len(matching_specs) != 1:
        raise CanonicalMotionBankGateError(
            f"recipe must contain exactly one motion spec for {row['motion_id']!r}"
        )
    try:
        authority_row = recipe.marker_semantics.row(str(row["motion_id"]))
    except Exception as exc:
        raise CanonicalMotionBankGateError(
            f"marker authority row for {row['motion_id']!r} is unavailable: {exc}"
        ) from exc
    recipe_span = tuple(int(value) for value in authority_row.ge80_seed)
    if (
        len(search_source) != 2
        or len(recipe_span) != 2
        or tuple(map(float, search_source)) != tuple(map(float, recipe_span))
    ):
        raise CanonicalMotionBankGateError(
            "search source span disagrees with the marker authority ge80 seed"
        )
    window_start_source = _integer(recipe_span[0], "authority ge80_seed[0]")
    window_end_source = _integer(recipe_span[1], "authority ge80_seed[1]")
    if authority_row.nominal_event is not None:
        source_anchor = _integer(
            authority_row.nominal_event, "authority nominal_event"
        )
    elif authority_row.construction_marker is not None:
        source_anchor = _integer(
            authority_row.construction_marker.annotation_frame,
            "authority construction annotation_frame",
        )
    else:
        raise CanonicalMotionBankGateError(
            f"marker authority row for {row['motion_id']!r} has neither a "
            "nominal event nor a construction annotation anchor"
        )
    if _finite(
        opportunity.get("source_anchor_frame"),
        "search source_anchor_frame",
        minimum=0.0,
    ) != float(source_anchor):
        raise CanonicalMotionBankGateError(
            "search source anchor disagrees with the content-addressed recipe"
        )
    # The reviewed anchor may sit before or after the legacy ge80 seed span
    # (event-to-window aliasing is forbidden), but the retained core must
    # contain the complete seed span and the anchor.
    if not (
        entry_frame <= window_start_source <= window_end_source <= exit_frame
        and entry_frame <= source_anchor <= exit_frame
    ):
        raise CanonicalMotionBankGateError(
            "selected entry/exit does not contain the complete recipe contact opportunity"
        )
    if (
        geometry_selection.get("window_start") != window_start_source
        or geometry_selection.get("window_end") != window_end_source
    ):
        raise CanonicalMotionBankGateError(
            "geometry window selection disagrees with the content-addressed recipe"
        )
    marker_source_frames = {
        "window_start": window_start_source,
        "source_anchor": source_anchor,
        "window_end": window_end_source,
    }
    binding = _mapping(
        geometry.get("recipe_marker_binding"), "geometry recipe_marker_binding"
    )
    if frozenset(binding) != frozenset(_MARKERS):
        raise CanonicalMotionBankGateError(
            "geometry recipe marker binding must contain exactly three markers"
        )
    for marker_index, name in enumerate(_MARKERS):
        frame = marker_source_frames[name]
        expected_dense_row = source_waypoint_rows[frame - entry_frame]
        bound = _exact_keys(
            binding[name],
            frozenset({"source_frame", "dense_row"}),
            f"geometry marker binding {name}",
        )
        if (
            _integer(bound["source_frame"], f"{name} bound source_frame")
            != frame
            or _integer(bound["dense_row"], f"{name} bound dense_row")
            != expected_dense_row
            or not math.isclose(
                source_indices[marker_index],
                float(expected_dense_row),
                rel_tol=0.0,
                abs_tol=_FLOAT_TOL,
            )
            or source_frame_map[expected_dense_row] != float(frame)
        ):
            raise CanonicalMotionBankGateError(
                f"marker {name} does not map recipe source frame {frame} "
                "to the independently rebuilt dense geometry row"
            )
    return {
        "policy": "marker_only_no_pose_speed_or_acceleration_lock",
        "acceleration_allowed_through_window_end": True,
        "indices_and_times_monotonic": True,
        "markers": marker_rows,
    }


def _default_plant_loader(
    mjcf: Path,
    urdf: Path,
    mjcf_sha256: str,
    compiled_signature: str,
) -> Any:
    return dynamics_gate.load_plant(
        mjcf,
        urdf_path=urdf,
        expected_mjcf_sha256=mjcf_sha256,
        expected_compiled_signature=compiled_signature,
    )


def _validate_plant(plant: Any, files: _BoundFiles) -> Mapping[str, Any]:
    expected = {
        "mjcf_sha256": files.hashes["mjcf"],
        "urdf_sha256": files.hashes["urdf"],
        "compiled_signature_sha256": files.compiled_signature,
    }
    for attribute, value in expected.items():
        if getattr(plant, attribute, None) != value:
            raise CanonicalMotionBankGateError(
                f"plant loader did not bind {attribute}: "
                f"expected {value}, got {getattr(plant, attribute, None)}"
            )
    if getattr(plant, "identity_bound", None) is not True:
        raise CanonicalMotionBankGateError("plant identity is not fail-closed bound")
    if tuple(getattr(plant, "runtime_body_names", ())) != motion_player.RUNTIME_BODY_NAMES:
        raise CanonicalMotionBankGateError("plant runtime body order changed")
    return {
        **expected,
        "identity_bound": True,
        "runtime_body_order": list(motion_player.RUNTIME_BODY_NAMES),
    }


def _default_player_runner(mjcf: Path) -> PlayerRunner:
    mujoco = motion_player.load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    binding = motion_player.bind_model(mujoco, model)

    def run(clip: motion_player.MotionClip) -> Mapping[str, Any]:
        return motion_player.smoke_check(clip, mujoco, model, data, binding)

    return run


def _default_dynamics_runner(path: Path, plant: Any) -> Mapping[str, Any]:
    # Keep one authoritative schema-2 reconstruction contract.  In particular,
    # the dynamics loader verifies all 32 body channels in both directions:
    # MuJoCo world-space forward velocities and the full-body Jacobian inverse
    # back to the stored root/joint qvel.  Reimplementing only half of that
    # contract here can silently drift when the dynamics gate is strengthened.
    trajectory = dynamics_gate.trajectory_from_schema2_npz(path, plant)
    return dynamics_gate.analyze_trajectory(plant, trajectory)


def _player_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("verdict") not in ("PASS", "FAIL"):
        raise CanonicalMotionBankGateError("player report has an unknown verdict")
    gates = _mapping(report.get("gates"), "player gates")
    position = _mapping(gates.get("position"), "player position gate")
    orientation = _mapping(gates.get("orientation"), "player orientation gate")
    passed = (
        report.get("verdict") == "PASS"
        and position.get("pass") is True
        and orientation.get("pass") is True
    )
    return {
        "pass": bool(passed),
        "verdict": report["verdict"],
        "position": dict(position),
        "orientation": dict(orientation),
        "evidence_boundary": dict(
            _mapping(report.get("evidence_boundary", {}), "player evidence boundary")
        ),
    }


def _dynamics_summary(
    report: Mapping[str, Any],
    files: _BoundFiles,
    clip: motion_player.MotionClip,
) -> Mapping[str, Any]:
    motion_path = clip.path
    if report.get("schema_version") != 1:
        raise CanonicalMotionBankGateError("dynamics report schema_version must be 1")
    if report.get("plant_specific_screen") is not True:
        raise CanonicalMotionBankGateError("dynamics report is not plant-specific")
    if report.get("verdict") not in ("PASS", "FAIL", "INCOMPLETE_FAIL_CLOSED"):
        raise CanonicalMotionBankGateError("dynamics report has an unknown verdict")
    if not isinstance(report.get("screen_pass"), bool):
        raise CanonicalMotionBankGateError("dynamics screen_pass must be boolean")
    if (report["verdict"] == "PASS") is not report["screen_pass"]:
        raise CanonicalMotionBankGateError(
            "dynamics verdict/screen_pass relationship is inconsistent"
        )
    source = _resolve_bound_path(
        report.get("source"), motion_path.parent, "dynamics report source"
    )
    _same_path(source, motion_path, "dynamics report source")

    plant = _mapping(report.get("plant"), "dynamics report plant")
    expected_plant = {
        "mjcf_sha256": files.hashes["mjcf"],
        "urdf_sha256": files.hashes["urdf"],
        "compiled_model_signature_sha256": files.compiled_signature,
    }
    for key, expected in expected_plant.items():
        if plant.get(key) != expected:
            raise CanonicalMotionBankGateError(
                f"dynamics report plant {key} is not the bound model"
            )
    if plant.get("identity_bound") is not True:
        raise CanonicalMotionBankGateError("dynamics report lost plant identity binding")
    if tuple(plant.get("runtime_joint_names", ())) != motion_player.RUNTIME_JOINT_NAMES:
        raise CanonicalMotionBankGateError("dynamics report runtime joint order changed")

    input_row = _mapping(report.get("input"), "dynamics input")
    expected_duration = (clip.n_frames - 1) / clip.fps
    if (
        _integer(input_row.get("frames"), "dynamics input frames", minimum=3) != clip.n_frames
        or not math.isclose(
            _finite(input_row.get("fps"), "dynamics input fps", minimum=0.0),
            clip.fps,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
        or not math.isclose(
            _finite(
                input_row.get("duration_s"),
                "dynamics input duration_s",
                minimum=0.0,
            ),
            expected_duration,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
    ):
        raise CanonicalMotionBankGateError(
            "dynamics input dimensions do not bind the verified NPZ"
        )
    body_velocity = _exact_keys(
        input_row.get("schema2_body_validation"),
        frozenset(
            {
                "body_count",
                "position_peak_error_m",
                "orientation_peak_error_rad",
                "linear_velocity_peak_error_m_s",
                "angular_velocity_peak_error_rad_s",
                "linear_velocity_peak_abs_error_m_s",
                "angular_velocity_peak_abs_error_rad_s",
                "joint_velocity_peak_abs_mismatch_rad_s",
                "root_linear_velocity_peak_mismatch_m_s",
                "root_angular_velocity_peak_mismatch_rad_s",
                "inverse_body_velocity_peak_abs_residual",
                "velocity_jacobian_min_rank",
                "velocity_jacobian_max_condition",
                "absolute_tolerance",
            }
        ),
        "dynamics schema2_body_validation",
    )
    body_limits = {
        "position_peak_error_m": dynamics_gate.SCHEMA_FK_POSITION_TOL_M,
        "orientation_peak_error_rad": dynamics_gate.SCHEMA_FK_ORIENTATION_TOL_RAD,
        "linear_velocity_peak_error_m_s": (
            dynamics_gate.SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S
        ),
        "angular_velocity_peak_error_rad_s": (
            dynamics_gate.SCHEMA_BODY_ANGULAR_VELOCITY_TOL_RAD_S
        ),
        "linear_velocity_peak_abs_error_m_s": (
            dynamics_gate.SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S
        ),
        "angular_velocity_peak_abs_error_rad_s": (
            dynamics_gate.SCHEMA_BODY_ANGULAR_VELOCITY_TOL_RAD_S
        ),
        "joint_velocity_peak_abs_mismatch_rad_s": (
            dynamics_gate.SCHEMA_JOINT_VELOCITY_TOL_RAD_S
        ),
        "root_linear_velocity_peak_mismatch_m_s": (
            dynamics_gate.SCHEMA_ROOT_LINEAR_VELOCITY_TOL_M_S
        ),
        "root_angular_velocity_peak_mismatch_rad_s": (
            dynamics_gate.SCHEMA_ROOT_ANGULAR_VELOCITY_TOL_RAD_S
        ),
        "inverse_body_velocity_peak_abs_residual": max(
            dynamics_gate.SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S,
            dynamics_gate.SCHEMA_BODY_ANGULAR_VELOCITY_TOL_RAD_S,
        ),
    }
    body_receipt_pass = all(
        _finite(body_velocity.get(key), f"schema2 body receipt {key}", minimum=0.0)
        <= limit
        for key, limit in body_limits.items()
    )
    if (
        _finite(body_velocity.get("body_count"), "schema2 body count") != 32.0
        or _finite(
            body_velocity.get("absolute_tolerance"),
            "schema2 body velocity tolerance",
            minimum=0.0,
        )
        != dynamics_gate.SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S
        or _finite(
            body_velocity.get("velocity_jacobian_min_rank"),
            "schema2 body velocity Jacobian rank",
            minimum=0.0,
        )
        != float(6 + len(motion_player.RUNTIME_JOINT_NAMES))
        or _finite(
            body_velocity.get("velocity_jacobian_max_condition"),
            "schema2 body velocity Jacobian condition",
            minimum=0.0,
        )
        > dynamics_gate.SCHEMA_VELOCITY_JACOBIAN_CONDITION_MAX
        or not body_receipt_pass
    ):
        raise CanonicalMotionBankGateError(
            "dynamics report lacks a passing full-body FK/velocity receipt"
        )
    consistency = _mapping(
        input_row.get("velocity_position_consistency"),
        "dynamics velocity_position_consistency",
    )
    checks = _mapping(report.get("checks"), "dynamics checks")
    limits = _mapping(checks.get("joint_limits"), "dynamics joint_limits")
    geometry = _mapping(checks.get("geometry"), "dynamics geometry")
    inverse = _mapping(checks.get("inverse_dynamics"), "dynamics inverse_dynamics")
    torque = _mapping(inverse.get("torque_interpretation"), "dynamics torque_interpretation")
    root_and_com = _mapping(checks.get("root_and_com"), "dynamics root_and_com")
    thresholds = dynamics_gate.GateThresholds()
    consistency_values = {
        "joint_velocity_peak_abs_mismatch_rad_s": (
            dynamics_gate.SCHEMA_JOINT_VELOCITY_TOL_RAD_S
        ),
        "root_linear_velocity_peak_mismatch_m_s": (
            dynamics_gate.SCHEMA_ROOT_LINEAR_VELOCITY_TOL_M_S
        ),
        "root_angular_velocity_peak_mismatch_rad_s": (
            dynamics_gate.SCHEMA_ROOT_ANGULAR_VELOCITY_TOL_RAD_S
        ),
    }
    consistency_computed = all(
        _finite(consistency.get(key), f"dynamics consistency {key}", minimum=0.0) <= limit
        for key, limit in consistency_values.items()
    )
    if (
        consistency.get("method")
        != (
            "schema2_all_32_body_com_velocity_jacobian_inverse_and_"
            "mujoco_objectVelocity_world_forward"
        )
        or consistency.get("pass") is not consistency_computed
    ):
        raise CanonicalMotionBankGateError(
            "dynamics analytic velocity consistency receipt is invalid"
        )

    position_limits = _mapping(limits.get("position"), "joint position limits")
    velocity_limits = _mapping(limits.get("velocity"), "joint velocity limits")
    position_violation_count = _integer(
        position_limits.get("violation_samples"),
        "joint position violation_samples",
    )
    velocity_violation_count = _integer(
        velocity_limits.get("violation_samples"),
        "joint velocity violation_samples",
    )
    if (
        position_violation_count > clip.n_frames * 31
        or velocity_violation_count > clip.n_frames * 31
    ):
        raise CanonicalMotionBankGateError(
            "joint-limit violation count exceeds the motion sample matrix"
        )
    position_violation_frames = _frame_indices(
        position_limits.get("violation_frames"),
        "joint position violation_frames",
        frame_count=clip.n_frames,
        unique_sorted=True,
    )
    velocity_violation_frames = _frame_indices(
        velocity_limits.get("violation_frames"),
        "joint velocity violation_frames",
        frame_count=clip.n_frames,
        unique_sorted=True,
    )
    for label, receipt, count, frames in (
        (
            "position",
            position_limits,
            position_violation_count,
            position_violation_frames,
        ),
        (
            "velocity",
            velocity_limits,
            velocity_violation_count,
            velocity_violation_frames,
        ),
    ):
        if (count == 0) is not (len(frames) == 0) or len(frames) > count:
            raise CanonicalMotionBankGateError(
                f"joint {label} violation frames contradict sample count"
            )
        _frame_indices(
            [receipt.get("peak_frame")],
            f"joint {label} peak_frame",
            frame_count=clip.n_frames,
            unique_sorted=True,
        )
        peak_joint = receipt.get("peak_joint")
        if peak_joint not in motion_player.RUNTIME_JOINT_NAMES:
            raise CanonicalMotionBankGateError(
                f"joint {label} peak_joint leaves the runtime order"
            )
    position_pass = position_violation_count == 0
    velocity_pass = velocity_violation_count == 0
    limits_computed = position_pass and velocity_pass
    if (
        position_limits.get("pass") is not position_pass
        or velocity_limits.get("pass") is not velocity_pass
        or limits.get("pass") is not limits_computed
    ):
        raise CanonicalMotionBankGateError("joint-limit pass fields contradict violation counts")

    geometry_pairs = (
        (
            "self_collision_violation_count",
            "self_collision_violations",
            thresholds.self_penetration_tolerance_m,
        ),
        (
            "foot_floor_penetration_violation_count",
            "foot_floor_penetration_violations",
            thresholds.foot_floor_penetration_tolerance_m,
        ),
        (
            "nonfoot_floor_penetration_violation_count",
            "nonfoot_floor_penetration_violations",
            thresholds.nonfoot_floor_penetration_tolerance_m,
        ),
        (
            "other_world_penetration_violation_count",
            "other_world_penetration_violations",
            thresholds.nonfoot_floor_penetration_tolerance_m,
        ),
    )
    geometry_counts: list[int] = []
    geometry_violation_rows: dict[str, list[Mapping[str, Any]]] = {}
    for count_key, rows_key, tolerance in geometry_pairs:
        count = _integer(geometry.get(count_key), f"geometry {count_key}")
        rows = _sequence(geometry.get(rows_key), f"geometry {rows_key}")
        if count != len(rows):
            raise CanonicalMotionBankGateError(f"geometry {count_key} contradicts {rows_key}")
        checked_rows: list[Mapping[str, Any]] = []
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"geometry {rows_key}[{index}]")
            _frame_indices(
                [row.get("frame")],
                f"geometry {rows_key}[{index}].frame",
                frame_count=clip.n_frames,
                unique_sorted=True,
            )
            if (
                _finite(
                    row.get("depth_m"),
                    f"geometry {rows_key}[{index}].depth_m",
                    minimum=0.0,
                )
                <= tolerance
            ):
                raise CanonicalMotionBankGateError(
                    f"geometry {rows_key} contains a non-violating contact"
                )
            checked_rows.append(row)
        geometry_violation_rows[rows_key] = checked_rows
        geometry_counts.append(count)
    geometry_computed = not any(geometry_counts)
    if geometry.get("pass") is not geometry_computed:
        raise CanonicalMotionBankGateError(
            "geometry pass contradicts collision/ground violation evidence"
        )
    all_contacts = _mapping(geometry.get("all_contacts"), "geometry.all_contacts")
    contact_groups: dict[str, list[Mapping[str, Any]]] = {}
    contact_frame_evidence: set[int] = set()
    for group in ("self_contacts", "floor_contacts", "other_world_contacts"):
        rows = _sequence(all_contacts.get(group), f"geometry.all_contacts.{group}")
        checked_rows = []
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"geometry.all_contacts.{group}[{index}]")
            frame = _frame_indices(
                [row.get("frame")],
                f"geometry.all_contacts.{group}[{index}].frame",
                frame_count=clip.n_frames,
                unique_sorted=True,
            )[0]
            _finite(
                row.get("depth_m"),
                f"geometry.all_contacts.{group}[{index}].depth_m",
                minimum=0.0,
            )
            if group == "floor_contacts" and not isinstance(row.get("robot_body_is_foot"), bool):
                raise CanonicalMotionBankGateError("floor contact is missing robot_body_is_foot")
            contact_frame_evidence.add(frame)
            checked_rows.append(row)
        contact_groups[group] = checked_rows
    violation_memberships = (
        ("self_collision_violations", "self_contacts", None),
        (
            "foot_floor_penetration_violations",
            "floor_contacts",
            True,
        ),
        (
            "nonfoot_floor_penetration_violations",
            "floor_contacts",
            False,
        ),
        (
            "other_world_penetration_violations",
            "other_world_contacts",
            None,
        ),
    )
    for violation_key, contact_key, foot_value in violation_memberships:
        for row in geometry_violation_rows[violation_key]:
            if foot_value is not None and row.get("robot_body_is_foot") is not foot_value:
                raise CanonicalMotionBankGateError(
                    f"geometry {violation_key} has the wrong foot classification"
                )
            if row not in contact_groups[contact_key]:
                raise CanonicalMotionBankGateError(
                    f"geometry {violation_key} is absent from all_contacts"
                )
    if not isinstance(torque.get("valid"), bool):
        raise CanonicalMotionBankGateError("torque_interpretation.valid must be boolean")
    contact_frames = _frame_indices(
        torque.get("contact_frames"),
        "torque_interpretation.contact_frames",
        frame_count=clip.n_frames,
        unique_sorted=True,
    )
    if contact_frames != sorted(contact_frame_evidence):
        raise CanonicalMotionBankGateError(
            "fake green dynamics report: contact_frames contradict all_contacts"
        )
    reasons = _sequence(torque.get("reasons"), "torque_interpretation.reasons")
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        raise CanonicalMotionBankGateError(
            "torque_interpretation.reasons must be non-empty strings"
        )
    label = torque.get("label")
    root_force_peak = _finite(
        torque.get("root_force_peak_n"),
        "torque root_force_peak_n",
        minimum=0.0,
    )
    root_torque_peak = _finite(
        torque.get("root_torque_peak_nm"),
        "torque root_torque_peak_nm",
        minimum=0.0,
    )
    _frame_indices(
        [torque.get("root_force_peak_frame")],
        "torque root_force_peak_frame",
        frame_count=clip.n_frames,
        unique_sorted=True,
    )
    _frame_indices(
        [torque.get("root_torque_peak_frame")],
        "torque root_torque_peak_frame",
        frame_count=clip.n_frames,
        unique_sorted=True,
    )
    joint_effort_proxy_peak = _finite(
        inverse.get("joint_effort_proxy_peak_utilization"),
        "joint effort proxy peak utilization",
        minimum=0.0,
    )
    actuator_force_proxy_peak = _finite(
        inverse.get("actuator_force_proxy_peak_utilization"),
        "actuator force proxy peak utilization",
        minimum=0.0,
    )
    for prefix in ("joint_effort_proxy", "actuator_force_proxy"):
        _frame_indices(
            [inverse.get(f"{prefix}_peak_frame")],
            f"{prefix} peak_frame",
            frame_count=clip.n_frames,
            unique_sorted=True,
        )
        if inverse.get(f"{prefix}_peak_joint") not in (motion_player.RUNTIME_JOINT_NAMES):
            raise CanonicalMotionBankGateError(f"{prefix} peak_joint leaves the runtime order")
    effort_expected = bool(
        torque["valid"]
        and joint_effort_proxy_peak <= 1.0 + thresholds.velocity_tolerance_fraction
        and actuator_force_proxy_peak <= 1.0 + thresholds.velocity_tolerance_fraction
    )
    if inverse.get("effort_limit_pass_only_if_interpretation_valid") is not effort_expected:
        raise CanonicalMotionBankGateError(
            "fake green dynamics report: inverse-dynamics effort pass "
            "contradicts torque validity/utilization"
        )
    if torque["valid"] is False:
        if (
            report["screen_pass"] is not False
            or report["verdict"] == "PASS"
            or inverse.get("effort_limit_pass_only_if_interpretation_valid") is not False
            or label != "uncertified_contact_disabled_joint_effort_proxy"
            or not reasons
        ):
            raise CanonicalMotionBankGateError(
                "fake green dynamics report: invalid grounded/root-residual torque "
                "was upgraded to a pass"
            )
    else:
        contact_lists = tuple(
            contact_groups[key]
            for key in ("self_contacts", "floor_contacts", "other_world_contacts")
        )
        if (
            report["verdict"] == "INCOMPLETE_FAIL_CLOSED"
            or label != "certified_contact_free_direct_motor_torque"
            or reasons
            or contact_frames
            or any(contact_lists)
            or plant.get("actuation_mapping_verified") is not True
            or root_force_peak > thresholds.root_force_zero_tolerance_n
            or root_torque_peak > thresholds.root_torque_zero_tolerance_nm
        ):
            raise CanonicalMotionBankGateError(
                "fake green dynamics report: torque marked valid without a "
                "contact-free, direct-motor certificate"
            )

    root_required = (
        "root_height_min_m",
        "root_height_max_m",
        "root_tilt_peak_rad",
        "root_xy_displacement_peak_m",
        "com_height_min_m",
        "com_height_max_m",
    )
    root_values = {
        key: _finite(
            root_and_com.get(key),
            f"dynamics root_and_com.{key}",
            minimum=(
                0.0 if key in ("root_tilt_peak_rad", "root_xy_displacement_peak_m") else None
            ),
        )
        for key in root_required
    }
    if (
        root_values["root_height_min_m"] > root_values["root_height_max_m"]
        or root_values["com_height_min_m"] > root_values["com_height_max_m"]
    ):
        raise CanonicalMotionBankGateError("dynamics root/CoM extrema are internally inconsistent")

    non_torque_pass = bool(consistency_computed and limits_computed and geometry_computed)
    expected_verdict = (
        "PASS"
        if non_torque_pass and effort_expected
        else "INCOMPLETE_FAIL_CLOSED"
        if non_torque_pass and torque["valid"] is False
        else "FAIL"
    )
    if report["verdict"] != expected_verdict:
        raise CanonicalMotionBankGateError(
            "dynamics verdict contradicts limits/geometry/root/effort evidence"
        )
    return {
        "verdict": report["verdict"],
        "screen_pass": report["screen_pass"],
        "non_torque_screens_pass": non_torque_pass,
        "velocity_position_consistency": dict(consistency),
        "instantaneous_body_velocity": dict(body_velocity),
        "joint_limits": dict(limits),
        "geometry": dict(geometry),
        "inverse_dynamics": {
            "torque_interpretation": dict(torque),
            "joint_effort_proxy_peak_utilization": joint_effort_proxy_peak,
            "actuator_force_proxy_peak_utilization": actuator_force_proxy_peak,
            "effort_limit_pass_only_if_interpretation_valid": inverse.get(
                "effort_limit_pass_only_if_interpretation_valid"
            ),
        },
        "root_and_com": {**dict(root_and_com), **root_values},
    }


def verify_canonical_motion_bank(
    manifest_path: os.PathLike[str] | str,
    bank_dir: os.PathLike[str] | str,
    *,
    mjcf_path: os.PathLike[str] | str,
    urdf_path: os.PathLike[str] | str,
    body_order_path: os.PathLike[str] | str,
    expected_compiled_signature: str,
    recipe_loader: RecipeLoader = load_canonical_motion_recipe,
    plant_loader: PlantLoader = _default_plant_loader,
    player_runner: PlayerRunner | None = None,
    dynamics_runner: DynamicsRunner = _default_dynamics_runner,
) -> Mapping[str, Any]:
    """Verify one immutable compiler bank and return a JSON-safe aggregate report.

    Dependency injection is deliberately limited to model/consumer execution so
    unit tests can exercise the bank logic without MuJoCo.  Every injected
    result still has to satisfy the same identity and fail-closed report schema.
    """

    manifest_file = Path(os.path.abspath(os.fspath(Path(manifest_path).expanduser())))
    bank = Path(os.path.abspath(os.fspath(Path(bank_dir).expanduser())))
    manifest = _strict_json(manifest_file, "build manifest")
    _validate_top_contract(manifest)
    files, recipe = _load_and_bind_files(
        manifest_file,
        bank,
        manifest,
        mjcf_path=Path(mjcf_path),
        urdf_path=Path(urdf_path),
        body_order_path=Path(body_order_path),
        expected_compiled_signature=expected_compiled_signature,
        recipe_loader=recipe_loader,
    )
    compiler_options = _validate_compiler_options(
        manifest["compiler_options"], recipe
    )
    output_rows = _sequence(manifest["outputs"], "outputs")
    matrix = _validate_bank_file_set(bank, output_rows)

    try:
        plant = plant_loader(
            files.mjcf,
            files.urdf,
            files.hashes["mjcf"],
            files.compiled_signature,
        )
    except Exception as exc:
        raise CanonicalMotionBankGateError(
            f"plant load failed: {type(exc).__name__}: {exc}"
        ) from exc
    plant_receipt = _validate_plant(plant, files)
    if player_runner is None:
        try:
            player_runner = _default_player_runner(files.mjcf)
        except Exception as exc:
            raise CanonicalMotionBankGateError(
                f"FK player setup failed: {type(exc).__name__}: {exc}"
            ) from exc

    canonical_body_pos: np.ndarray | None = None
    canonical_body_quat: np.ndarray | None = None
    clip_reports: list[Mapping[str, Any]] = []
    for motion_id, scope in EXPECTED_MATRIX:
        row, path = matrix[(motion_id, scope)]
        expected_sha = _digest(row["output_npz_sha256"], f"{motion_id}/{scope} output_npz_sha256")
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise CanonicalMotionBankGateError(
                f"{motion_id}/{scope} NPZ SHA mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        schema_receipt = _validate_schema_receipts(
            row,
            npz_path=path,
            output_sha=actual_sha,
            files=files,
            recipe=recipe,
        )
        try:
            clip = motion_player.load_motion(path)
        except Exception as exc:
            raise CanonicalMotionBankGateError(
                f"{motion_id}/{scope} strict schema-2 load failed: " f"{type(exc).__name__}: {exc}"
            ) from exc
        endpoint, body_pos, body_quat = _endpoint_receipt(
            clip, recipe, canonical_body_pos, canonical_body_quat
        )
        if canonical_body_pos is None:
            canonical_body_pos = body_pos
            canonical_body_quat = body_quat
        marker = _validate_marker_contract(
            row, clip, recipe, compiler_options
        )

        try:
            player_raw = player_runner(clip)
        except Exception as exc:
            raise CanonicalMotionBankGateError(
                f"{motion_id}/{scope} FK player failed: {type(exc).__name__}: {exc}"
            ) from exc
        player = _player_summary(_mapping(player_raw, "player report"))
        try:
            dynamics_raw = dynamics_runner(path, plant)
        except Exception as exc:
            raise CanonicalMotionBankGateError(
                f"{motion_id}/{scope} dynamics screen failed: " f"{type(exc).__name__}: {exc}"
            ) from exc
        dynamics = _dynamics_summary(_mapping(dynamics_raw, "dynamics report"), files, clip)
        clip_reports.append(
            {
                "motion_id": motion_id,
                "scope": scope,
                "filename": path.name,
                "sha256": actual_sha,
                "frames": clip.n_frames,
                "fps": clip.fps,
                "duration_s": (clip.n_frames - 1) / clip.fps,
                "schema2_receipts": schema_receipt,
                "strict_schema2_and_ready": endpoint,
                "contact_opportunity": marker,
                "mujoco_fk": player,
                "plant_specific_dynamics": dynamics,
            }
        )

    fk_pass_count = sum(row["mujoco_fk"]["pass"] is True for row in clip_reports)
    non_torque_pass_count = sum(
        row["plant_specific_dynamics"]["non_torque_screens_pass"] is True for row in clip_reports
    )
    complete_count = sum(
        row["plant_specific_dynamics"]["screen_pass"] is True for row in clip_reports
    )
    incomplete_count = sum(
        row["plant_specific_dynamics"]["verdict"] == "INCOMPLETE_FAIL_CLOSED"
        for row in clip_reports
    )
    failed_count = len(clip_reports) - complete_count - incomplete_count
    torque_valid_count = sum(
        row["plant_specific_dynamics"]["inverse_dynamics"]["torque_interpretation"]["valid"]
        is True
        for row in clip_reports
    )
    joint_effort_proxy_peak = max(
        row["plant_specific_dynamics"]["inverse_dynamics"]["joint_effort_proxy_peak_utilization"]
        for row in clip_reports
    )
    actuator_force_proxy_peak = max(
        row["plant_specific_dynamics"]["inverse_dynamics"]["actuator_force_proxy_peak_utilization"]
        for row in clip_reports
    )
    velocity_consistency_pass_count = sum(
        row["plant_specific_dynamics"]["velocity_position_consistency"].get("pass") is True
        for row in clip_reports
    )
    joint_limit_pass_count = sum(
        row["plant_specific_dynamics"]["joint_limits"].get("pass") is True for row in clip_reports
    )
    geometry_pass_count = sum(
        row["plant_specific_dynamics"]["geometry"].get("pass") is True for row in clip_reports
    )

    def total_geometry_count(key: str) -> int:
        total = 0
        for row in clip_reports:
            raw = row["plant_specific_dynamics"]["geometry"].get(key, 0)
            total += _integer(raw, f"dynamics geometry {key}")
        return total

    contact_frame_count = sum(
        len(
            _sequence(
                row["plant_specific_dynamics"]["inverse_dynamics"]["torque_interpretation"].get(
                    "contact_frames", ()
                ),
                "torque contact_frames",
            )
        )
        for row in clip_reports
    )
    clips_with_contact_count = sum(
        bool(
            row["plant_specific_dynamics"]["inverse_dynamics"]["torque_interpretation"].get(
                "contact_frames", ()
            )
        )
        for row in clip_reports
    )

    def extrema(key: str, reducer: Callable[[Sequence[float]], float]) -> float | None:
        values = [
            _finite(
                row["plant_specific_dynamics"]["root_and_com"][key],
                f"root_and_com.{key}",
            )
            for row in clip_reports
            if key in row["plant_specific_dynamics"]["root_and_com"]
        ]
        return reducer(values) if values else None

    structural_pass = bool(
        len(clip_reports) == 10
        and fk_pass_count == 10
        and non_torque_pass_count == 10
        and failed_count == 0
    )
    # Schema-2 does not publish exact qacc/geometric-tangent midpoint
    # collocation.  Reconstructing qacc with finite differences would be a
    # diagnostic, not proof of the compiler's actual continuous time law.
    # Therefore no bank can pass the grounded dynamics sub-gate until a
    # content-addressed exact collocation trace is present and independently
    # screened.  Candidate integrity remains separately reportable.
    grounded_trace_status = "MISSING_INCOMPLETE_FAIL_CLOSED"
    complete_pass = False
    if structural_pass:
        verdict = "INCOMPLETE_FAIL_CLOSED"
    else:
        verdict = "FAIL"

    return {
        "schema_version": 1,
        "verdict": verdict,
        "bank_gate_pass": complete_pass,
        "candidate_integrity_pass": structural_pass,
        "grounded_trace_status": grounded_trace_status,
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "hardware_authorized": False,
        "library_id": manifest["library_id"],
        "manifest": {
            "path": str(files.manifest),
            "sha256": _sha256_file(files.manifest),
        },
        "bank_dir": str(files.bank_dir),
        "bound_inputs": {
            "recipe": {"path": str(files.recipe), "sha256": files.hashes["recipe"]},
            "compiler": {
                "path": str(files.compiler),
                "sha256": files.hashes["compiler"],
            },
            "geometry_tool": {
                "path": str(files.geometry_tool),
                "sha256": files.hashes["geometry_tool"],
            },
            "compiler_options_sha256": compiler_options.sha256,
            "ready": {"path": str(files.ready), "sha256": files.hashes["ready"]},
            "mjcf": {"path": str(files.mjcf), "sha256": files.hashes["mjcf"]},
            "urdf": {"path": str(files.urdf), "sha256": files.hashes["urdf"]},
            "body_order": {
                "path": str(files.body_order),
                "sha256": files.hashes["body_order"],
            },
            "plant": plant_receipt,
            "verifier_tools": {
                "bank_gate": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
                "mujoco_motion_player": {
                    "path": str(Path(motion_player.__file__).resolve()),
                    "sha256": _sha256_file(Path(motion_player.__file__).resolve()),
                },
                "canonical_mujoco_dynamics_gate": {
                    "path": str(Path(dynamics_gate.__file__).resolve()),
                    "sha256": _sha256_file(Path(dynamics_gate.__file__).resolve()),
                    "report_schema_version": 1,
                },
            },
        },
        "contracts": {
            "matrix": {
                "motion_ids": list(MOTION_IDS),
                "scopes": list(SCOPES),
                "count": 10,
            },
            "shared_ready": True,
            "six_endpoint_velocity_classes_exact_zero": True,
            "contact_opportunity_is_marker_only": True,
            "acceleration_allowed_through_window_end": True,
            "nonnegative_scalar_acceleration_through_window_end": True,
            "adv2c3_role": "comparator_only_not_default",
            "grounded_inverse_dynamics": (
                "incomplete_fail_closed_until_content_addressed_actual_time_law_"
                "collocation_trace_and_grounded_double_support_lp_pass"
            ),
            "grounded_trace_status": grounded_trace_status,
        },
        "aggregate": {
            "clip_count": len(clip_reports),
            "fk_pass_count": fk_pass_count,
            "velocity_consistency_pass_count": velocity_consistency_pass_count,
            "joint_limit_pass_count": joint_limit_pass_count,
            "geometry_pass_count": geometry_pass_count,
            "non_torque_dynamics_pass_count": non_torque_pass_count,
            "complete_dynamics_pass_count": complete_count,
            "incomplete_fail_closed_count": incomplete_count,
            "failed_count": failed_count,
            "torque_interpretation_valid_count": torque_valid_count,
            "clips_with_contact_count": clips_with_contact_count,
            "contact_frame_count": contact_frame_count,
            "self_collision_violation_count": total_geometry_count(
                "self_collision_violation_count"
            ),
            "foot_floor_penetration_violation_count": total_geometry_count(
                "foot_floor_penetration_violation_count"
            ),
            "nonfoot_floor_penetration_violation_count": total_geometry_count(
                "nonfoot_floor_penetration_violation_count"
            ),
            "other_world_penetration_violation_count": total_geometry_count(
                "other_world_penetration_violation_count"
            ),
            "joint_effort_proxy_peak_utilization": joint_effort_proxy_peak,
            "actuator_force_proxy_peak_utilization": actuator_force_proxy_peak,
            "root_height_min_m": extrema("root_height_min_m", min),
            "root_height_max_m": extrema("root_height_max_m", max),
            "root_tilt_peak_rad": extrema("root_tilt_peak_rad", max),
            "root_xy_displacement_peak_m": extrema("root_xy_displacement_peak_m", max),
            "com_height_min_m": extrema("com_height_min_m", min),
            "com_height_max_m": extrema("com_height_max_m", max),
        },
        "clips": clip_reports,
        "non_claims": [
            "candidate integrity is not training authorization",
            "candidate integrity is not hardware or deployment authorization",
            "FK playback is not dynamics, balance, control, or learnability",
            "grounded contact-disabled inverse dynamics is not actuator torque",
            "schema2 finite-difference qacc is not an exact collocation trace",
            "contact opportunity markers do not freeze pose, speed, or acceleration",
            "stationary motion-bank verification does not certify locomotion coverage",
        ],
    }


def write_bank_report_no_clobber(
    report: Mapping[str, Any], output_path: os.PathLike[str] | str
) -> Path:
    """Atomically create one JSON report after all verification work has ended."""

    # Do not resolve the leaf: resolving a broken symlink erases the occupied
    # directory entry and would redirect a supposedly no-clobber publication.
    destination = Path(os.path.abspath(os.fspath(Path(output_path).expanduser())))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing report {destination}")
    try:
        payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalMotionBankGateError(f"bank report is not strict JSON: {exc}") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    # The hard-link publication is already atomic.  Some
                    # filesystems reject directory fsync even though the link
                    # itself succeeded, so do not turn a published report into
                    # a misleading retryable failure.
                    pass
            finally:
                os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing report {destination}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _failure_report(exc: Exception) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "verdict": "FAIL",
        "bank_gate_pass": False,
        "candidate_integrity_pass": False,
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "hardware_authorized": False,
        "error": f"{type(exc).__name__}: {exc}",
        "non_claims": [
            "input/model/report failure is fail-closed",
            "no motion asset, training process, simulator step, or hardware was modified",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="compiler BUILD_MANIFEST.json")
    parser.add_argument("--bank-dir", required=True, help="directory containing exactly 10 NPZs")
    parser.add_argument("--mjcf", required=True, help="exact recipe-bound vendor MJCF")
    parser.add_argument("--urdf", required=True, help="exact recipe-bound vendor URDF")
    parser.add_argument(
        "--body-order", required=True, help="exact recipe-bound 32-body order file"
    )
    parser.add_argument(
        "--expected-compiled-signature",
        required=True,
        help="expected canonical_mujoco_dynamics_gate compiled-model SHA-256",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="new report path; existing files are never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_canonical_motion_bank(
            args.manifest,
            args.bank_dir,
            mjcf_path=args.mjcf,
            urdf_path=args.urdf,
            body_order_path=args.body_order,
            expected_compiled_signature=args.expected_compiled_signature,
        )
    except Exception as exc:
        report = _failure_report(exc)
    try:
        write_bank_report_no_clobber(report, args.out)
    except Exception as exc:
        print(f"report write failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0 if report.get("bank_gate_pass") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
