#!/usr/bin/env python3
"""Fail-closed compiler for the canonical five-motion, two-scope library.

The compiler deliberately keeps geometry, timing, and publication as separate
contracts:

1. load the content-addressed recipe;
2. project the complete source into ``upper`` or ``full`` body scope;
3. solve the coordinated right-arm face manifold for the synthetic forehand
   block (never a one-axis pi overlay);
4. enumerate legal retained source cores;
5. build ``canonical ready -> selected core -> canonical ready`` directly;
6. use the kinematic path retimer with the contact opportunity as markers only;
7. decode the free root and rebuild an exact schema-2 candidate in MuJoCo; and
8. publish all ten in-memory results with a candidate-only, no-clobber manifest.

``adv2c3`` is recorded only as a historical comparator.  It never filters,
seeds, or wins a tie in the search.  The contact-opportunity markers do not
freeze poses, velocities, or accelerations; acceleration may continue through
the end marker.

The current retimer is a kinematic warm start, not a torque, balance, contact,
learnability, training, deployment, or hardware certificate.  In particular,
URDF/MJCF do not define a complete acceleration contract.  Callers must supply
explicit joint acceleration limits and full-root coordinate limits.  The
manifest keeps every downstream gate pending.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import ctypes
import errno
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from canonical_body_scope import BodyScopeResult, preprocess_body_scope
from canonical_face_manifold import (
    FaceManifoldConfig,
    FaceManifoldResult,
    MujocoRightRacketBackend,
    RightRacketBackend,
    solve_face_flipped_window,
)
from canonical_motion_geometry import (
    CanonicalMotionGeometry,
    EntryExitCandidate,
    build_canonical_geometry,
    enumerate_entry_exit_candidates,
)
from canonical_motion_recipe import (
    CanonicalMotionRecipe,
    MotionSource,
    load_canonical_motion_recipe,
    sha256_file,
)
from canonical_path_topp import (
    RetimeError,
    RetimeResult,
    ScalarPathCollocationTrace,
    control_tick_at_or_after,
    retime_path,
)
from canonical_weighted_arc_path import (
    WeightedArcPath,
    WeightedArcPathError,
    build_weighted_arc_path,
)
from canonical_root_pose_codec import (
    decode_root_pose,
    encode_root_pose,
    root_coordinate_velocity_to_world_twist,
)
from canonical_schema2_builder import (
    Schema2Candidate,
    build_schema2_candidate,
    write_schema2_candidate,
)
from mujoco_motion_player import RUNTIME_BODY_NAMES, RUNTIME_JOINT_NAMES


PUBLICATION_CLASS = "compiler_candidate"
BUILD_MANIFEST_NAME = "BUILD_MANIFEST.json"
SCOPES = ("upper", "full")
ROOT_COORDINATE_NAMES = (
    "root_x_w",
    "root_y_w",
    "root_z_w",
    "root_rotvec_x_ready",
    "root_rotvec_y_ready",
    "root_rotvec_z_ready",
)
ROOT_COORDINATE_UNITS = ("m", "m", "m", "rad", "rad", "rad")
_PUBLISHED_INPUT_ARRAY_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_POSITION_TOLERANCE = 1.0e-7
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


class CanonicalMotionCompilerError(RuntimeError):
    """The complete library cannot be compiled without weakening a contract."""


@dataclass(frozen=True)
class FullRootPathLimits:
    """Limits for ``[world xyz, ready-relative rotation-vector xyz]``.

    These are path-coordinate limits.  Angular velocity is converted to exact
    world angular velocity before schema-2 publication, but the kinematic warm
    start constrains the rotation-vector rate itself.  A downstream dynamics
    gate must check the decoded free-root motion.
    """

    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass(frozen=True)
class CompilerOptions:
    """Explicit non-recipe inputs needed by the kinematic warm-start compiler."""

    joint_acceleration_limits_rad_s2: np.ndarray
    full_root_limits: FullRootPathLimits
    s0_full_grounding_offset_m: float | None
    samples_per_scaled_unit: float = 24.0
    min_connector_intervals: int = 8
    min_core_intervals: int = 5
    grid_subdivisions: int = 12
    search_workers: int = 1
    search_parallel_backend: str = "thread"
    face_config: FaceManifoldConfig | None = None
    face_active_candidate_seeds: tuple[np.ndarray, ...] = ()
    # Probe-only banding: when set, only the `probe_entry_band` entries
    # closest to window_start-halo and the `probe_exit_band` exits closest to
    # window_end+halo stay eligible for exact retiming.  The strike-first
    # winner lives in that corner (a longer retained prefix can only delay
    # window arrival; a later exit only lengthens recovery), so a band probe
    # reports the same minimal times at a fraction of the exhaustive cost.
    # None (default) keeps the contract's exhaustive enumeration; every use
    # is recorded in the options receipt and is probe-grade, not formal.
    probe_entry_band: int | None = None
    probe_exit_band: int | None = None
    # Probe-only: solve the time law with exact pointwise caps plus the
    # control-rate a-posteriori guard instead of the interval-certified
    # ladder.  Probe-grade only; the independent verifier rejects banks
    # built this way (like banded enumeration).
    probe_exact_pointwise_caps: bool = False
    # Probe-only source smoothing: when set, each motion's scoped source
    # coordinate array is smoothed once (per motion+scope) with iterative
    # binomial [1,4,6,4,1]/16 passes before entry/exit enumeration and
    # geometry construction, stopping before any coordinate's max-abs
    # deviation from the raw source exceeds this tolerance.  It suppresses the
    # phantom curvature spike a noisy quintic connector inherits at junctions
    # (the spike collapses the local speed cap, which the no-early-brake law
    # then propagates backward over the whole approach).  Frame indices are
    # untouched, so markers stay valid; endpoints stay exactly raw.  For root
    # position columns in full scope the tolerance is read in that
    # coordinate's own unit (metres for root xyz, radians for joints and the
    # ready-relative root rotation vector).  None (default) is the exact
    # current behaviour byte-for-byte.  Probe-grade only; the independent
    # verifier rejects banks built this way (like banded enumeration).
    probe_source_smoothing_tolerance_rad: float | None = None


@dataclass(frozen=True)
class _PathContract:
    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    coordinate_scale: np.ndarray
    coordinate_semantics: tuple[str, ...]
    coordinate_units: tuple[str, ...]


@dataclass(frozen=True)
class _ExactSearchContext:
    """Pickle-safe inputs shared once by exact process-search workers."""

    source_coordinates: np.ndarray
    ready_coordinates: np.ndarray
    window_start: int
    window_end: int
    source_anchor: int
    window_halo: int
    samples_per_scaled_unit: float
    min_connector_intervals: int
    min_core_intervals: int
    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    coordinate_scale: np.ndarray
    coordinate_semantics: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    fps: float
    minimum_window_s: float
    grid_subdivisions: int
    exact_pointwise_caps: bool = False


@dataclass(frozen=True)
class _SearchWinner:
    candidate: EntryExitCandidate
    geometry: CanonicalMotionGeometry
    retimed: RetimeResult
    duration_s: float
    total_variation_scaled_l2: float
    score: tuple[int, int, int, int, int, float, int, int]


@dataclass(frozen=True)
class _FormalPathJet:
    """Canonical C2 segment knots, excluding visualization-only dense rows."""

    dense_row_indices: np.ndarray
    q_path: np.ndarray
    path_parameter: np.ndarray
    first_derivative: np.ndarray
    second_derivative: np.ndarray


@dataclass(frozen=True)
class _WeightedFormalPath:
    """Formal C2 knots reparameterized by digest-bound weighted arc length."""

    formal: _FormalPathJet
    arc_path: WeightedArcPath
    q_path: np.ndarray
    l_knots: np.ndarray
    first_derivative: np.ndarray
    second_derivative: np.ndarray


@dataclass(frozen=True)
class CompiledMotion:
    """One selected upper/full compiler candidate and its receipts."""

    motion_id: str
    scope: str
    filename: str
    schema2: Schema2Candidate
    entry_frame: int
    exit_frame: int
    duration_s: float
    contact_window_start_s: float
    contact_window_end_s: float
    source_anchor_time_s: float
    total_variation_scaled_l2: float
    search_report: Mapping[str, Any]
    scope_report: Mapping[str, Any]
    face_report: Mapping[str, Any] | None
    geometry_report: Mapping[str, Any]
    retime_report: Mapping[str, Any]
    collocation_trace: ScalarPathCollocationTrace


@dataclass(frozen=True)
class CompiledLibrary:
    """All ten in-memory candidates plus one JSON-safe build manifest."""

    recipe: CanonicalMotionRecipe
    motions: tuple[CompiledMotion, ...]
    manifest: Mapping[str, Any]


def _finite_vector(
    value: Any,
    length: int,
    label: str,
    *,
    strictly_positive: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise CanonicalMotionCompilerError(f"{label} must be real-valued")
    try:
        out = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalMotionCompilerError(
            f"{label} must be a real numeric vector"
        ) from exc
    if out.shape != (length,) or not np.isfinite(out).all():
        raise CanonicalMotionCompilerError(
            f"{label} must be finite shape ({length},), got {out.shape}"
        )
    if strictly_positive and np.any(out <= 0.0):
        raise CanonicalMotionCompilerError(
            f"{label} must contain strictly positive values"
        )
    return np.ascontiguousarray(out)


def _validate_options(options: CompilerOptions) -> tuple[np.ndarray, FullRootPathLimits]:
    if not isinstance(options, CompilerOptions):
        raise CanonicalMotionCompilerError("options must be CompilerOptions")
    joint_acceleration = _finite_vector(
        options.joint_acceleration_limits_rad_s2,
        31,
        "joint_acceleration_limits_rad_s2",
        strictly_positive=True,
    )
    root = options.full_root_limits
    if not isinstance(root, FullRootPathLimits):
        raise CanonicalMotionCompilerError(
            "full_root_limits must be FullRootPathLimits"
        )
    root_lower = _finite_vector(root.position_lower, 6, "root position lower")
    root_upper = _finite_vector(root.position_upper, 6, "root position upper")
    if np.any(root_lower >= root_upper):
        raise CanonicalMotionCompilerError(
            "every full-root position lower limit must be below its upper limit"
        )
    root_velocity = _finite_vector(
        root.velocity, 6, "root velocity", strictly_positive=True
    )
    root_acceleration = _finite_vector(
        root.acceleration, 6, "root acceleration", strictly_positive=True
    )
    samples = float(options.samples_per_scaled_unit)
    if not math.isfinite(samples) or samples <= 0.0:
        raise CanonicalMotionCompilerError(
            "samples_per_scaled_unit must be finite and positive"
        )
    smoothing_tolerance = options.probe_source_smoothing_tolerance_rad
    if smoothing_tolerance is not None:
        if isinstance(smoothing_tolerance, bool) or not isinstance(
            smoothing_tolerance, (int, float)
        ):
            raise CanonicalMotionCompilerError(
                "probe_source_smoothing_tolerance_rad must be a positive "
                "number or None"
            )
        if (
            not math.isfinite(float(smoothing_tolerance))
            or float(smoothing_tolerance) <= 0.0
        ):
            raise CanonicalMotionCompilerError(
                "probe_source_smoothing_tolerance_rad must be finite and "
                "strictly positive"
            )
    for name in (
        "min_connector_intervals",
        "min_core_intervals",
        "grid_subdivisions",
        "search_workers",
    ):
        value = getattr(options, name)
        minimum = 1 if name == "search_workers" else 2
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise CanonicalMotionCompilerError(
                f"{name} must be an integer >= {minimum}"
            )
    if options.search_workers > 64:
        raise CanonicalMotionCompilerError("search_workers cannot exceed 64")
    if options.search_parallel_backend not in {"thread", "process"}:
        raise CanonicalMotionCompilerError(
            "search_parallel_backend must be 'thread' or 'process'"
        )
    if options.min_connector_intervals < 5 or options.min_core_intervals < 5:
        raise CanonicalMotionCompilerError(
            "C2 quintic geometry requires at least five intervals per segment"
        )
    for index, seed in enumerate(options.face_active_candidate_seeds):
        _finite_vector(seed, 7, f"face_active_candidate_seeds[{index}]")
    normalized_root = FullRootPathLimits(
        position_lower=root_lower,
        position_upper=root_upper,
        velocity=root_velocity,
        acceleration=root_acceleration,
    )
    return joint_acceleration, normalized_root


def _effective_face_config(
    options: CompilerOptions, velocity_fraction: float
) -> FaceManifoldConfig:
    config = options.face_config or FaceManifoldConfig(
        mode="normal", velocity_limit_fraction=velocity_fraction
    )
    if config.mode != "normal":
        raise CanonicalMotionCompilerError(
            "synthetic recipe requires normal-hard/in-plane-free face mode"
        )
    if not np.isclose(
        config.velocity_limit_fraction,
        velocity_fraction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise CanonicalMotionCompilerError(
            "face and path velocity-limit fractions must be identical"
        )
    return config


def _validate_backend(
    backend: RightRacketBackend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(str(name) for name in backend.joint_names)
    if names != tuple(RUNTIME_JOINT_NAMES):
        raise CanonicalMotionCompilerError(
            "plant backend joint order must equal the exact 31-joint runtime order"
        )
    lower = _finite_vector(backend.position_lower, 31, "plant position lower")
    upper = _finite_vector(backend.position_upper, 31, "plant position upper")
    velocity = _finite_vector(
        backend.velocity_limit,
        31,
        "plant velocity limit",
        strictly_positive=True,
    )
    if np.any(lower >= upper):
        raise CanonicalMotionCompilerError(
            "every plant position lower limit must be below its upper limit"
        )
    return lower, upper, velocity


def _path_contract(
    *,
    scope: str,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity: np.ndarray,
    joint_acceleration: np.ndarray,
    root_limits: FullRootPathLimits,
    velocity_fraction: float,
) -> _PathContract:
    if not math.isfinite(velocity_fraction) or not 0.0 < velocity_fraction <= 1.0:
        raise CanonicalMotionCompilerError(
            "recipe joint_velocity_limit_fraction must lie in (0,1]"
        )
    joint_velocity_used = joint_velocity * velocity_fraction
    if scope == "upper":
        lower = joint_lower
        upper = joint_upper
        velocity = joint_velocity_used
        acceleration = joint_acceleration
        semantics = tuple(RUNTIME_JOINT_NAMES)
        units = ("rad",) * 31
    elif scope == "full":
        lower = np.concatenate((joint_lower, root_limits.position_lower))
        upper = np.concatenate((joint_upper, root_limits.position_upper))
        velocity = np.concatenate((joint_velocity_used, root_limits.velocity))
        acceleration = np.concatenate(
            (joint_acceleration, root_limits.acceleration)
        )
        semantics = tuple(RUNTIME_JOINT_NAMES) + ROOT_COORDINATE_NAMES
        units = ("rad",) * 31 + ROOT_COORDINATE_UNITS
    else:
        raise CanonicalMotionCompilerError(f"unknown body scope {scope!r}")
    # Normalize mixed joint/root distances by the exact path-coordinate speed
    # limits.  This is deterministic and avoids adding meters to radians.
    scale = 1.0 / velocity
    return _PathContract(
        position_lower=np.ascontiguousarray(lower),
        position_upper=np.ascontiguousarray(upper),
        velocity=np.ascontiguousarray(velocity),
        acceleration=np.ascontiguousarray(acceleration),
        coordinate_scale=np.ascontiguousarray(scale),
        coordinate_semantics=semantics,
        coordinate_units=units,
    )


def _source_root(source: MotionSource) -> tuple[np.ndarray, np.ndarray]:
    try:
        pelvis_index = tuple(RUNTIME_BODY_NAMES).index("pelvis_link")
    except ValueError as exc:  # pragma: no cover - import-time invariant
        raise CanonicalMotionCompilerError(
            "runtime body order is missing pelvis_link"
        ) from exc
    pos = np.asarray(source.clip.body_pos_w[:, pelvis_index], dtype=np.float64)
    quat = np.asarray(source.clip.body_quat_w[:, pelvis_index], dtype=np.float64)
    expected = source.clip.n_frames
    if pos.shape != (expected, 3) or quat.shape != (expected, 4):
        raise CanonicalMotionCompilerError(
            f"{source.motion_id} pelvis root arrays have invalid shapes"
        )
    return pos, quat


def _grounding_offset(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    scope: str,
    options: CompilerOptions,
) -> tuple[float | None, float]:
    del recipe
    if scope != "full" or source.motion_id != "s0_highpress":
        return None, 0.0
    raw = source.scope_overrides
    try:
        maximum = float(raw["full"]["maximum_grounding_offset_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalMotionCompilerError(
            "s0 full recipe is missing its bounded grounding contract"
        ) from exc
    offset = options.s0_full_grounding_offset_m
    if offset is None or not math.isfinite(float(offset)):
        raise CanonicalMotionCompilerError(
            "s0 full requires an explicitly measured grounding offset"
        )
    value = float(offset)
    if value < 0.0 or value > maximum + 1.0e-12:
        raise CanonicalMotionCompilerError(
            f"s0 full grounding offset {value:.9g} m is outside [0,{maximum:.9g}]"
        )
    return value, maximum


def _preprocess_scope(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    scope: str,
    options: CompilerOptions,
) -> BodyScopeResult:
    root_pos, root_quat = _source_root(source)
    common = {
        "source_joint_pos": source.clip.joint_pos,
        "source_root_pos_w": root_pos,
        "source_root_quat_w": root_quat,
        "joint_names": RUNTIME_JOINT_NAMES,
        "canonical_ready_root_pos_w": recipe.ready.root_pos_w,
        "canonical_ready_root_quat_w": recipe.ready.root_quat_wxyz,
    }
    if scope == "upper":
        return preprocess_body_scope(
            "upper",
            **common,
            canonical_ready_joint_pos=recipe.ready.joint_pos,
        )
    grounding, maximum = _grounding_offset(recipe, source, scope, options)
    kwargs: dict[str, Any] = dict(common)
    if grounding is not None:
        kwargs.update(
            grounding_z_offset_m=grounding,
            max_grounding_correction_m=maximum,
        )
    return preprocess_body_scope("full", **kwargs)


def _solve_synthetic_face(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    scoped: BodyScopeResult,
    *,
    backend: RightRacketBackend,
    options: CompilerOptions,
    velocity_fraction: float,
) -> tuple[BodyScopeResult, Mapping[str, Any] | None]:
    if source.motion_id != "fh_block_syn":
        return scoped, None
    face = source.face_manifold
    if face is None:
        raise CanonicalMotionCompilerError(
            "fh_block_syn is missing its face-manifold contract"
        )
    construction = recipe.marker_semantics.row(source.motion_id).construction_marker
    if construction is None:
        raise CanonicalMotionCompilerError(
            "fh_block_syn is missing its marker-authority construction marker"
        )
    solve_start, solve_end = (int(value) for value in construction.solve_span)
    if not 0 <= solve_start <= solve_end < len(scoped.joint_pos):
        raise CanonicalMotionCompilerError(
            "synthetic face solve span is outside the scoped source"
        )
    if face["orientation"] != "normal_hard_inplane_free":
        raise CanonicalMotionCompilerError(
            "compiler supports only the recipe's normal-hard/in-plane-free solve"
        )
    config = _effective_face_config(options, velocity_fraction)
    solver_anchor = int(construction.annotation_frame)
    try:
        result: FaceManifoldResult = solve_face_flipped_window(
            scoped.joint_pos[solve_start : solve_end + 1],
            scoped.root_pos_w[solve_start : solve_end + 1],
            scoped.root_quat_w[solve_start : solve_end + 1],
            recipe.ready.joint_pos,
            fps=float(source.clip.fps),
            backend=backend,
            config=config,
            frame_indices=tuple(range(solve_start, solve_end + 1)),
            anchor_index=solver_anchor - solve_start,
            active_candidate_seeds=options.face_active_candidate_seeds,
        )
    except Exception as exc:
        raise CanonicalMotionCompilerError(
            f"{source.motion_id} {scoped.report.get('scope')} face solve failed: {exc}"
        ) from exc
    output_q = np.asarray(scoped.joint_pos, dtype=np.float64).copy()
    output_q[solve_start : solve_end + 1] = result.joint_pos
    return (
        BodyScopeResult(
            joint_pos=output_q,
            root_pos_w=scoped.root_pos_w,
            root_quat_w=scoped.root_quat_w,
            report=scoped.report,
        ),
        MappingProxyType(result.summary()),
    )


def _coordinates(
    recipe: CanonicalMotionRecipe,
    scoped: BodyScopeResult,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any] | None]:
    if scope == "upper":
        return (
            np.asarray(scoped.joint_pos, dtype=np.float64),
            np.asarray(recipe.ready.joint_pos, dtype=np.float64),
            None,
        )
    root_encoding = encode_root_pose(
        scoped.root_pos_w,
        scoped.root_quat_w,
        canonical_ready_root_quat_wxyz=recipe.ready.root_quat_wxyz,
    )
    source_coordinates = np.concatenate(
        (np.asarray(scoped.joint_pos, dtype=np.float64), root_encoding.coordinates),
        axis=1,
    )
    ready_coordinates = np.concatenate(
        (
            np.asarray(recipe.ready.joint_pos, dtype=np.float64),
            np.asarray(recipe.ready.root_pos_w, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
    )
    return source_coordinates, ready_coordinates, root_encoding.report


_PROBE_SOURCE_SMOOTHING_MAX_PASSES = 256
_PROBE_SOURCE_SMOOTHING_ALGORITHM = (
    "iterative_binomial_1_4_6_4_1_over_16_edge_replicated_endpoints_pinned"
)


def _binomial_smooth_once(values: np.ndarray) -> np.ndarray:
    """One [1,4,6,4,1]/16 pass along frames with edge replication.

    Endpoints are not preserved by the pass itself; the caller re-pins them.
    """

    padded = np.pad(values, ((2, 2), (0, 0)), mode="edge")
    return (
        padded[:-4]
        + 4.0 * padded[1:-3]
        + 6.0 * padded[2:-2]
        + 4.0 * padded[3:-1]
        + padded[4:]
    ) / 16.0


def _smooth_source_coordinates(
    raw: np.ndarray, tolerance: float
) -> tuple[np.ndarray, int, float]:
    """Iterative binomial smoothing capped by a per-coordinate deviation bound.

    Returns ``(smoothed, passes_applied, max_abs_deviation_reached)``.  The
    first and last frame stay exactly equal to ``raw``; the result is the last
    iterate whose every-coordinate max-abs deviation from ``raw`` is within
    ``tolerance``.  Binomial smoothing moves monotonically away from the raw
    path, so the stop is the first pass that would exceed the bound.
    """

    raw = np.ascontiguousarray(np.asarray(raw, dtype=np.float64))
    current = raw.copy()
    passes = 0
    max_deviation = 0.0
    if raw.shape[0] <= 2:
        # No interior frame to smooth; pinned endpoints already equal raw.
        return current, passes, max_deviation
    for _ in range(_PROBE_SOURCE_SMOOTHING_MAX_PASSES):
        candidate = _binomial_smooth_once(current)
        candidate[0] = raw[0]
        candidate[-1] = raw[-1]
        deviation = float(np.max(np.abs(candidate - raw)))
        if deviation > tolerance:
            break
        converged = np.array_equal(candidate, current)
        current = candidate
        passes += 1
        max_deviation = deviation
        if converged:
            break
    return current, passes, max_deviation


def _apply_probe_source_smoothing(
    source_coordinates: np.ndarray,
    options: CompilerOptions,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Optionally smooth one motion+scope source array; else return it unchanged."""

    tolerance = options.probe_source_smoothing_tolerance_rad
    if tolerance is None:
        # Exact current behaviour: the same array object flows downstream, so
        # geometry inputs stay byte-for-byte identical to the unsmoothed path.
        return source_coordinates, MappingProxyType(
            {
                "active": False,
                "tolerance_rad": None,
                "passes_applied": 0,
                "max_abs_deviation_reached": 0.0,
                "algorithm": _PROBE_SOURCE_SMOOTHING_ALGORITHM,
                "probe_grade_not_verifiable": False,
            }
        )
    smoothed, passes, max_deviation = _smooth_source_coordinates(
        source_coordinates, float(tolerance)
    )
    return smoothed, MappingProxyType(
        {
            "active": True,
            "tolerance_rad": float(tolerance),
            "passes_applied": int(passes),
            "max_abs_deviation_reached": float(max_deviation),
            "endpoints_pinned_exact": True,
            "hard_pass_cap": _PROBE_SOURCE_SMOOTHING_MAX_PASSES,
            "algorithm": _PROBE_SOURCE_SMOOTHING_ALGORITHM,
            # One scalar tolerance, read in each coordinate's own unit: metres
            # for root xyz, radians for joints and the root rotation vector.
            "tolerance_units": (
                "per_coordinate_own_unit_m_for_root_xyz_rad_for_joints_and_"
                "root_rotvec"
            ),
            "probe_grade_not_verifiable": True,
        }
    )


def _authority_markers(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
) -> tuple[tuple[int, int], int, str]:
    """Resolve the retiming window seed and ranking anchor from the v2 authority.

    The legacy >=80% behavior seed is a search-and-retime marker interval only,
    never an output behavior window.  The ranking anchor is the reviewed nominal
    air-swing event; the synthetic forehand block has no reviewed event and must
    use its construction annotation frame, which is lineage-only.
    """

    row = recipe.marker_semantics.row(source.motion_id)
    window = (int(row.ge80_seed[0]), int(row.ge80_seed[1]))
    if row.nominal_event is not None:
        anchor = int(row.nominal_event)
        anchor_semantics = "marker_authority_v2_nominal_event"
    elif row.construction_marker is not None:
        anchor = int(row.construction_marker.annotation_frame)
        anchor_semantics = (
            "marker_authority_v2_construction_annotation_lineage_only"
        )
    else:
        raise CanonicalMotionCompilerError(
            f"{source.motion_id} marker authority row has neither a nominal "
            "event nor a construction annotation anchor"
        )
    # The v2 authority forbids aliasing the reviewed event into the legacy
    # seed window; the anchor may legally sit before or after the ge80 span.
    return window, anchor, anchor_semantics


def _entry_exit_candidates(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    source_coordinates: np.ndarray,
    ready_coordinates: np.ndarray,
    contract: _PathContract,
) -> tuple[list[EntryExitCandidate], list[EntryExitCandidate], int]:
    search = recipe.raw["entry_exit_search"]
    halo = int(search["legacy_ge80_halo_source_frames"])
    window, anchor, _ = _authority_markers(recipe, source)
    enumerated = enumerate_entry_exit_candidates(
        source_coordinates,
        ready_coordinates,
        window[0],
        window[1],
        window_halo=halo,
        coordinate_scale=contract.coordinate_scale,
        coordinate_semantics=contract.coordinate_semantics,
        coordinate_units=contract.coordinate_units,
    )
    # The retained core must keep the reviewed ranking anchor so every
    # published candidate carries all three retiming markers.
    eligible = [
        candidate
        for candidate in enumerated
        if candidate.entry_frame <= anchor <= candidate.exit_frame
    ]
    row = recipe.marker_semantics.row(source.motion_id)
    if row.construction_marker is not None:
        # Frames outside the face-solved construction span still carry the
        # donor's unflipped face and may not enter a synthetic candidate.
        solve_start, solve_end = row.construction_marker.solve_span
        eligible = [
            candidate
            for candidate in eligible
            if (
                solve_start <= candidate.entry_frame
                and candidate.exit_frame <= solve_end
            )
        ]
    if not eligible:
        raise CanonicalMotionCompilerError(
            f"{source.motion_id} has no legal entry/exit candidate"
        )
    return enumerated, eligible, halo


def _apply_probe_band(
    eligible: list[EntryExitCandidate],
    options: CompilerOptions,
) -> list[EntryExitCandidate]:
    """Keep only the near-window candidate corner for a banded probe."""

    entry_band = options.probe_entry_band
    exit_band = options.probe_exit_band
    if entry_band is None and exit_band is None:
        return eligible
    for name, value in (("probe_entry_band", entry_band), ("probe_exit_band", exit_band)):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise CanonicalMotionCompilerError(
                f"{name} must be a positive integer or None"
            )
    banded = eligible
    if entry_band is not None:
        latest_entries = sorted({row.entry_frame for row in banded})[-entry_band:]
        banded = [row for row in banded if row.entry_frame in latest_entries]
    if exit_band is not None:
        earliest_exits = sorted({row.exit_frame for row in banded})[:exit_band]
        banded = [row for row in banded if row.exit_frame in earliest_exits]
    if not banded:
        raise CanonicalMotionCompilerError(
            "probe band left no eligible entry/exit candidate"
        )
    return banded


def _coordinate_total_variation(
    source_coordinates: np.ndarray,
    ready_coordinates: np.ndarray,
    entry: int,
    exit_: int,
) -> np.ndarray:
    core = source_coordinates[entry : exit_ + 1]
    return (
        np.abs(core[0] - ready_coordinates)
        + np.sum(np.abs(np.diff(core, axis=0)), axis=0)
        + np.abs(ready_coordinates - core[-1])
    )


def _duration_lower_bound(
    source_coordinates: np.ndarray,
    ready_coordinates: np.ndarray,
    candidate: EntryExitCandidate,
    contract: _PathContract,
    marker_minimum_s: float,
) -> float:
    variation = _coordinate_total_variation(
        source_coordinates,
        ready_coordinates,
        candidate.entry_frame,
        candidate.exit_frame,
    )
    velocity_bound = float(np.max(variation / contract.velocity))
    retained = source_coordinates[
        candidate.entry_frame : candidate.exit_frame + 1
    ]
    required_waypoints = np.vstack(
        (ready_coordinates[None, :], retained, ready_coordinates[None, :])
    )
    # Each consecutive waypoint pair occupies a disjoint time interval.
    # Its duration is at least the largest component displacement divided by
    # that component's speed cap, so the sum is also an admissible bound.
    sequential_velocity_bound = float(
        np.sum(
            np.max(
                np.abs(np.diff(required_waypoints, axis=0))
                / contract.velocity[None, :],
                axis=1,
            )
        )
    )
    acceleration_bound = float(
        np.max(2.0 * np.sqrt(variation / contract.acceleration))
    )
    return max(
        float(marker_minimum_s),
        velocity_bound,
        sequential_velocity_bound,
        acceleration_bound,
    )


def _marker_path_index(
    geometry: CanonicalMotionGeometry, source_frame: int
) -> int:
    rows = np.flatnonzero(
        np.isfinite(geometry.source_frame_map)
        & np.isclose(
            geometry.source_frame_map,
            float(source_frame),
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    if rows.shape != (1,):
        raise CanonicalMotionCompilerError(
            f"source frame {source_frame} maps to {len(rows)} geometry rows"
        )
    return int(rows[0])


def _formal_path_jet(geometry: CanonicalMotionGeometry) -> _FormalPathJet:
    """Select pre/core/post quintic endpoints, not dense display samples."""

    source_knots = np.asarray(
        geometry.source_waypoint_path_indices, dtype=np.int64
    )
    if (
        source_knots.ndim != 1
        or len(source_knots) < 2
        or np.any(np.diff(source_knots) <= 0)
    ):
        raise CanonicalMotionCompilerError(
            "canonical geometry omitted ordered source-waypoint path knots"
        )
    expected_indices = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            source_knots,
            np.asarray([len(geometry.q_path) - 1], dtype=np.int64),
        )
    )
    dense_indices = np.asarray(
        geometry.canonical_knot_path_indices, dtype=np.int64
    )
    if (
        not np.array_equal(dense_indices, expected_indices)
        or dense_indices.ndim != 1
        or np.any(np.diff(dense_indices) <= 0)
        or dense_indices[1] != source_knots[0]
        or dense_indices[-2] != source_knots[-1]
    ):
        raise CanonicalMotionCompilerError(
            "formal pre/core/post path-jet knot identity is invalid"
        )
    values = (
        np.asarray(geometry.q_path[dense_indices], dtype=np.float64),
        np.asarray(
            geometry.path_parameter[dense_indices], dtype=np.float64
        ),
        np.asarray(geometry.dq_ds[dense_indices], dtype=np.float64),
        np.asarray(geometry.d2q_ds2[dense_indices], dtype=np.float64),
    )
    if (
        values[1][0] != 0.0
        or np.any(np.diff(values[1]) <= 0.0)
        or any(not np.all(np.isfinite(value)) for value in values)
    ):
        raise CanonicalMotionCompilerError(
            "formal canonical path 2-jet is non-finite or has an invalid "
            "geometry parameter"
        )
    return _FormalPathJet(
        dense_row_indices=dense_indices,
        q_path=values[0],
        path_parameter=values[1],
        first_derivative=values[2],
        second_derivative=values[3],
    )


def _formal_marker_index(
    geometry: CanonicalMotionGeometry,
    formal: _FormalPathJet,
    source_frame: int,
) -> int:
    """Map one exact source waypoint to its formal-knot row."""

    dense_index = _marker_path_index(geometry, source_frame)
    matches = np.flatnonzero(formal.dense_row_indices == dense_index)
    if matches.shape != (1,):
        raise CanonicalMotionCompilerError(
            f"source frame {source_frame} is not one formal path-jet knot"
        )
    return int(matches[0])


def _weighted_arc_formal_path(
    geometry: CanonicalMotionGeometry,
    coordinate_scale: np.ndarray,
) -> _WeightedFormalPath:
    """Build the exact weighted-arc coordinate from formal geometry knots."""

    formal = _formal_path_jet(geometry)
    scale = np.asarray(coordinate_scale, dtype=np.float64)
    if (
        scale.shape != (formal.q_path.shape[1],)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise CanonicalMotionCompilerError(
            "weighted arc coordinate_scale must be finite, positive, and "
            "match the formal path dimension"
        )
    try:
        arc_path = build_weighted_arc_path(
            s_knots=formal.path_parameter,
            q=formal.q_path,
            q_s=formal.first_derivative,
            q_ss=formal.second_derivative,
            coordinate_scale=scale,
        )
        q_path, q_l, q_ll = arc_path.evaluate_l(arc_path.l_knots)
    except WeightedArcPathError as exc:
        raise CanonicalMotionCompilerError(
            f"formal weighted-arc construction failed: {exc}"
        ) from exc
    if (
        not arc_path.verify_content_digest()
        or not np.array_equal(arc_path.coordinate_scale, scale)
        or not np.array_equal(q_path, formal.q_path)
        or not np.array_equal(
            arc_path.s_from_l(arc_path.l_knots),
            formal.path_parameter,
        )
    ):
        raise CanonicalMotionCompilerError(
            "formal weighted-arc content, scale, or knot identity changed"
        )
    return _WeightedFormalPath(
        formal=formal,
        arc_path=arc_path,
        q_path=np.asarray(q_path, dtype=np.float64),
        l_knots=np.asarray(arc_path.l_knots, dtype=np.float64),
        first_derivative=np.asarray(q_l, dtype=np.float64),
        second_derivative=np.asarray(q_ll, dtype=np.float64),
    )


def _weighted_arc_marker_indices_and_lengths(
    geometry: CanonicalMotionGeometry,
    weighted: _WeightedFormalPath,
    marker_source_frames: Mapping[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    """Bind each recipe marker to one exact formal knot and exact l value."""

    indices: dict[str, float] = {}
    lengths: dict[str, float] = {}
    for name, source_frame in marker_source_frames.items():
        index = _formal_marker_index(
            geometry, weighted.formal, int(source_frame)
        )
        indices[name] = float(index)
        lengths[name] = float(weighted.l_knots[index])
    if not all(
        weighted.l_knots[int(indices[name])] == lengths[name]
        for name in indices
    ):
        raise CanonicalMotionCompilerError(
            "weighted-arc marker did not map to an exact formal knot"
        )
    return indices, lengths


def _assert_weighted_arc_retime_binding(
    result: RetimeResult,
    weighted: _WeightedFormalPath,
    marker_lengths: Mapping[str, float],
) -> None:
    """Fail closed unless the retimer consumed this exact weighted arc."""

    progress = result.report.get("path_progress", {})
    evaluator = result.report.get("path_evaluator", {})
    receipt = result.report.get("weighted_arc_length", {})
    if (
        progress.get("contract") != "weighted_arc_length_v1"
        or progress.get("explicit") is not True
        or evaluator.get("kind")
        != "weighted_arc_path_evaluate_l_exact_v1"
        or evaluator.get("derivative_inputs_consumed") is not True
        or receipt.get("enabled") is not True
        or receipt.get("contract") != "weighted_arc_length_v1"
        or receipt.get("evaluator_mode") != "direct_exact"
        or receipt.get("exact_direct_evaluator") is not True
        or receipt.get("exact_evaluator_api")
        != "WeightedArcPath.evaluate_l"
        or receipt.get("content_sha256")
        != weighted.arc_path.content_sha256
        or receipt.get("digest_verified") is not True
        or receipt.get("regularity_certified") is not True
    ):
        raise CanonicalMotionCompilerError(
            "canonical retimer did not certify the exact weighted-arc "
            "endpoint-2-jet parameterization"
        )
    expected_scale_digest = hashlib.sha256(
        np.ascontiguousarray(
            weighted.arc_path.coordinate_scale, dtype="<f8"
        ).tobytes(order="C")
    ).hexdigest()
    if receipt.get(
        "coordinate_scale_sha256_float64_le"
    ) != expected_scale_digest:
        raise CanonicalMotionCompilerError(
            "canonical retimer did not bind the weighted-arc coordinate scale"
        )
    report_markers = result.report.get("markers", {})
    for name, expected_length in marker_lengths.items():
        row = report_markers.get(name)
        if (
            not isinstance(row, Mapping)
            or row.get("path_progress") != expected_length
        ):
            raise CanonicalMotionCompilerError(
                f"retimer marker {name!r} is not bound to its exact formal "
                "weighted-arc knot"
            )
    trace = result.collocation_trace
    if (
        trace is None
        or trace.path_progress_contract != "weighted_arc_length_v1"
        or trace.weighted_arc_length_receipt is None
        or trace.weighted_arc_length_receipt.get("receipt_sha256")
        != receipt.get("receipt_sha256")
    ):
        raise CanonicalMotionCompilerError(
            "accepted retime omitted the immutable weighted-arc trace binding"
        )


def _scaled_total_variation(
    q_path: np.ndarray, coordinate_scale: np.ndarray
) -> float:
    return float(
        np.sum(
            np.linalg.norm(
                np.diff(q_path, axis=0) * coordinate_scale[None, :],
                axis=1,
            )
        )
    )


def _control_tick_at_or_after(time_s: float, fps: float) -> int:
    """Delegate ranking quantization to the retimer's shared contract."""

    try:
        return control_tick_at_or_after(time_s, fps)
    except ValueError as exc:
        raise CanonicalMotionCompilerError(str(exc)) from exc


_PLATEAU_ACCELERATION_TOLERANCE = 1.0e-9

# Franco 2026-07-24 ruling: 0.5 s is a REFERENCE line, not a hard gate.  The
# compiler reports each candidate's minimal ready-to-anchor time and whether
# it exceeds the reference; it never fails a build on it.  The number is a
# kinematic warm-start value, not a behavior, torque, or return claim.
_STRIKE_TIME_REFERENCE_S = 0.5


def _candidate_ranking_score(
    *,
    retimed: RetimeResult,
    duration_s: float,
    total_variation_scaled_l2: float,
    entry_frame: int,
    exit_frame: int,
    fps: float,
) -> tuple[int, int, int, int, int, float, int, int]:
    """Quantized hit-priority score after all hard gates have passed."""

    try:
        window_start_time = retimed.markers["window_start"].time_s
        anchor_time = retimed.markers["source_anchor"].time_s
        window_end_time = retimed.markers["window_end"].time_s
    except KeyError as exc:
        raise CanonicalMotionCompilerError(
            f"retimer omitted ranking marker {exc.args[0]!r}"
        ) from exc
    window_start_tick = _control_tick_at_or_after(window_start_time, fps)
    anchor_tick = _control_tick_at_or_after(anchor_time, fps)
    window_end_tick = _control_tick_at_or_after(window_end_time, fps)
    cycle_tick = _control_tick_at_or_after(duration_s, fps)
    # The v2 marker authority forbids aliasing the reviewed event into the
    # legacy seed window, so the anchor tick may legally precede window_start
    # or follow window_end while staying inside the retained cycle.
    if not (
        0 <= window_start_tick <= window_end_tick <= cycle_tick
        and 0 <= anchor_tick <= cycle_tick
    ):
        raise CanonicalMotionCompilerError(
            "retimed marker ticks are not ordered inside the cycle"
        )
    # A planner may commit at the early edge, anchor, or late edge.  The early
    # edge is the worst recovery case, so max(cycle-hit) reduces exactly to
    # cycle-window_start without a floating-point comparison.
    worst_recovery_tick = max(
        cycle_tick - window_start_tick,
        cycle_tick - anchor_tick,
        cycle_tick - window_end_tick,
    )
    # Zero-acceleration plateaus before window end never violate sddot>=0 but
    # waste preparation time, so they rank ahead of recovery/cycle length.
    acceleration = np.asarray(retimed.path_acceleration, dtype=np.float64)
    if acceleration.ndim != 1 or not np.isfinite(acceleration).all():
        raise CanonicalMotionCompilerError(
            "retimer path_acceleration must be a finite one-dimensional array"
        )
    segment_start_s = np.arange(acceleration.shape[0], dtype=np.float64) / fps
    prewindow_plateau_ticks = int(
        np.sum(
            (segment_start_s < window_end_time)
            & (np.abs(acceleration) <= _PLATEAU_ACCELERATION_TOLERANCE)
        )
    )
    return (
        window_start_tick,
        anchor_tick,
        prewindow_plateau_ticks,
        worst_recovery_tick,
        cycle_tick,
        float(total_variation_scaled_l2),
        int(entry_frame),
        int(exit_frame),
    )


def _build_and_retime_exact(
    context: _ExactSearchContext,
    entry_frame: int,
    exit_frame: int,
) -> tuple[CanonicalMotionGeometry, RetimeResult, float, float]:
    """Build and exactly retime one retained core from pickle-safe inputs."""

    geometry = build_canonical_geometry(
        context.source_coordinates,
        context.ready_coordinates,
        entry_frame,
        exit_frame,
        context.window_start,
        context.window_end,
        window_halo=context.window_halo,
        samples_per_rad=context.samples_per_scaled_unit,
        min_connector_intervals=context.min_connector_intervals,
        min_core_intervals=context.min_core_intervals,
        coordinate_scale=context.coordinate_scale,
        coordinate_semantics=context.coordinate_semantics,
        coordinate_units=context.coordinate_units,
    )
    weighted = _weighted_arc_formal_path(
        geometry, context.coordinate_scale
    )
    marker_indices, marker_lengths = (
        _weighted_arc_marker_indices_and_lengths(
            geometry,
            weighted,
            {
                "window_start": context.window_start,
                "source_anchor": context.source_anchor,
                "window_end": context.window_end,
            },
        )
    )
    retimed = retime_path(
        weighted.q_path,
        context.velocity,
        context.acceleration,
        position_lower_limits=context.position_lower,
        position_upper_limits=context.position_upper,
        path_progress=weighted.l_knots,
        path_first_derivative=weighted.first_derivative,
        path_second_derivative=weighted.second_derivative,
        weighted_arc_path=weighted.arc_path,
        fps=context.fps,
        markers=marker_indices,
        marker_min_duration_s={
            ("window_start", "window_end"): context.minimum_window_s
        },
        nonnegative_acceleration_until_marker="window_end",
        nonnegative_acceleration_from_marker=(
            "window_start" if context.exact_pointwise_caps else None
        ),
        grid_subdivisions=context.grid_subdivisions,
        exact_pointwise_caps=context.exact_pointwise_caps,
        position_tolerance=_POSITION_TOLERANCE,
    )
    _assert_weighted_arc_retime_binding(
        retimed, weighted, marker_lengths
    )
    duration = float(retimed.report["duration_s"])
    total_variation = _scaled_total_variation(
        geometry.q_path, context.coordinate_scale
    )
    return geometry, retimed, duration, total_variation


def _retime_uniform_prefix_comparator(
    context: _ExactSearchContext,
    geometry: CanonicalMotionGeometry,
) -> RetimeResult:
    """Retime the selected geometry under the strict scalar-prefix comparator."""

    if context.exact_pointwise_caps:
        # Diagnostic-only receipt: under the exact-pointwise probe this
        # comparator would run the interval-certified ladder for hours in the
        # main process while certifying nothing the probe consumes.  Fail
        # closed into the existing infeasible-comparator receipt path.
        raise RetimeError(
            "SKIPPED_EXACT_POINTWISE_PROBE_MODE: uniform-prefix comparator "
            "is not evaluated under exact-pointwise probe caps"
        )
    weighted = _weighted_arc_formal_path(
        geometry, context.coordinate_scale
    )
    marker_indices, marker_lengths = (
        _weighted_arc_marker_indices_and_lengths(
            geometry,
            weighted,
            {
                "window_start": context.window_start,
                "source_anchor": context.source_anchor,
                "window_end": context.window_end,
            },
        )
    )
    result = retime_path(
        weighted.q_path,
        context.velocity,
        context.acceleration,
        position_lower_limits=context.position_lower,
        position_upper_limits=context.position_upper,
        path_progress=weighted.l_knots,
        path_first_derivative=weighted.first_derivative,
        path_second_derivative=weighted.second_derivative,
        weighted_arc_path=weighted.arc_path,
        fps=context.fps,
        markers=marker_indices,
        marker_min_duration_s={
            ("window_start", "window_end"): context.minimum_window_s
        },
        uniform_scalar_path_acceleration_until_marker="window_end",
        grid_subdivisions=context.grid_subdivisions,
        position_tolerance=_POSITION_TOLERANCE,
    )
    _assert_weighted_arc_retime_binding(
        result, weighted, marker_lengths
    )
    policy = result.report.get(
        "uniform_scalar_path_acceleration_prefix_comparator", {}
    )
    if (
        policy.get("enabled") is not True
        or policy.get("strictly_positive_on_every_guard_cell") is not True
        or int(policy.get("cruise_cells_through_guard", -1)) != 0
    ):
        raise CanonicalMotionCompilerError(
            "uniform scalar-prefix comparator lacks its strict prefix receipt"
        )
    return result


def _uniform_prefix_comparator_receipt(
    no_brake: RetimeResult,
    uniform: RetimeResult,
    *,
    fps: float,
) -> Mapping[str, Any]:
    """Compare two certified time laws on exactly the same geometry."""

    no_brake_arc = no_brake.report.get("weighted_arc_length", {})
    uniform_arc = uniform.report.get("weighted_arc_length", {})
    if (
        no_brake_arc.get("enabled") is not True
        or uniform_arc.get("enabled") is not True
        or no_brake_arc.get("content_sha256")
        != uniform_arc.get("content_sha256")
        or no_brake_arc.get("receipt_sha256")
        != uniform_arc.get("receipt_sha256")
        or no_brake_arc.get("coordinate_scale_sha256_float64_le")
        != uniform_arc.get("coordinate_scale_sha256_float64_le")
    ):
        raise CanonicalMotionCompilerError(
            "time-law comparators are not bound to the same weighted arc"
        )
    marker_rows = {}
    for name in ("window_start", "source_anchor", "window_end"):
        no_brake_time = float(no_brake.markers[name].time_s)
        uniform_time = float(uniform.markers[name].time_s)
        no_brake_tick = _control_tick_at_or_after(no_brake_time, fps)
        uniform_tick = _control_tick_at_or_after(uniform_time, fps)
        if (
            uniform_tick < no_brake_tick
            or uniform_time < no_brake_time - 1.0e-10
        ):
            raise CanonicalMotionCompilerError(
                "uniform scalar-prefix comparator arrived earlier than the "
                f"no-brake candidate at {name}; solver/grid dominance gap"
            )
        marker_rows[name] = {
            "no_brake_time_s": no_brake_time,
            "uniform_time_s": uniform_time,
            "no_brake_control_tick": no_brake_tick,
            "uniform_control_tick": uniform_tick,
            "uniform_minus_no_brake_s": uniform_time - no_brake_time,
        }
    no_brake_cycle = float(no_brake.report["duration_s"])
    uniform_cycle = float(uniform.report["duration_s"])
    no_brake_cycle_tick = _control_tick_at_or_after(no_brake_cycle, fps)
    uniform_cycle_tick = _control_tick_at_or_after(uniform_cycle, fps)
    if (
        uniform_cycle_tick < no_brake_cycle_tick
        or uniform_cycle < no_brake_cycle - 1.0e-10
    ):
        raise CanonicalMotionCompilerError(
            "uniform scalar-prefix comparator cycle is earlier than the "
            "no-brake candidate; solver/grid dominance gap"
        )
    uniform_policy = uniform.report[
        "uniform_scalar_path_acceleration_prefix_comparator"
    ]
    return MappingProxyType(
        {
            "role": (
                "uniform_scalar_path_acceleration_prefix_comparator_only_"
                "not_an_adopted_time_law"
            ),
            "status": "FEASIBLE_STRICT_UNIFORM",
            "same_geometry_and_limits": True,
            "weighted_arc_length": {
                "contract": "weighted_arc_length_v1",
                "content_sha256": no_brake_arc["content_sha256"],
                "receipt_sha256": no_brake_arc["receipt_sha256"],
                "coordinate_scale_sha256_float64_le": no_brake_arc[
                    "coordinate_scale_sha256_float64_le"
                ],
            },
            "same_50hz_guard_and_grid_refinement_contract": True,
            "dominance_gate": (
                "uniform_must_not_arrive_earlier_than_no_brake_at_any_marker_"
                "or_cycle"
            ),
            "dominance_status": "PASS_NO_BRAKE_WEAKLY_DOMINATES",
            "markers": marker_rows,
            "cycle": {
                "no_brake_duration_s": no_brake_cycle,
                "uniform_duration_s": uniform_cycle,
                "no_brake_control_ticks": no_brake_cycle_tick,
                "uniform_control_ticks": uniform_cycle_tick,
                "uniform_minus_no_brake_s": (
                    uniform_cycle - no_brake_cycle
                ),
            },
            "uniform_prefix": {
                "constant_scalar_path_acceleration": uniform_policy[
                    "constant_scalar_path_acceleration"
                ],
                "units": uniform_policy[
                    "constant_scalar_path_acceleration_units"
                ],
                "guard_path_progress": uniform_policy[
                    "control_guard_path_progress"
                ],
                "guard_boundary_frame": uniform_policy[
                    "control_guard_boundary_frame"
                ],
                "control_guard_iteration": uniform_policy[
                    "control_guard_iteration"
                ],
                "cruise_cells_through_guard": uniform_policy[
                    "cruise_cells_through_guard"
                ],
                "suffix_recovery_contract": uniform_policy[
                    "suffix_recovery_contract"
                ],
                "suffix_speed_squared_scale": uniform_policy[
                    "suffix_speed_squared_scale_from_ordinary_profile"
                ],
            },
            "no_brake_grid_refinement": no_brake.report.get(
                "grid_refinement"
            ),
            "uniform_grid_refinement": uniform.report.get(
                "grid_refinement"
            ),
            "non_claims": [
                "uniform_joint_acceleration",
                "uniform_actuator_torque",
                "continuous_time_optimality",
                "adopted_training_time_law",
                "grounded_dynamics_or_learnability",
            ],
        }
    )


_PROCESS_EXACT_SEARCH_CONTEXT: _ExactSearchContext | None = None


def _initialize_process_exact_search(context: _ExactSearchContext) -> None:
    """Install one immutable search context in a spawned worker process."""

    global _PROCESS_EXACT_SEARCH_CONTEXT
    _PROCESS_EXACT_SEARCH_CONTEXT = context


def _measure_process_exact_search(
    candidate_pair: tuple[int, int],
) -> tuple[
    int,
    int,
    float | None,
    float | None,
    tuple[int, int, int, int, int, float, int, int] | None,
    str | None,
]:
    """Process-pool worker returning scalar ranking evidence only."""

    context = _PROCESS_EXACT_SEARCH_CONTEXT
    if context is None:
        return (
            int(candidate_pair[0]),
            int(candidate_pair[1]),
            None,
            None,
            None,
            "CanonicalMotionCompilerError: process search context is missing",
        )
    entry_frame, exit_frame = map(int, candidate_pair)
    try:
        _, retimed, duration, total_variation = _build_and_retime_exact(
            context, entry_frame, exit_frame
        )
        score = _candidate_ranking_score(
            retimed=retimed,
            duration_s=duration,
            total_variation_scaled_l2=total_variation,
            entry_frame=entry_frame,
            exit_frame=exit_frame,
            fps=context.fps,
        )
    except Exception as exc:
        return (
            entry_frame,
            exit_frame,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )
    return entry_frame, exit_frame, duration, total_variation, score, None


def _search_core(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    source_coordinates: np.ndarray,
    ready_coordinates: np.ndarray,
    contract: _PathContract,
    *,
    options: CompilerOptions,
) -> tuple[_SearchWinner, Mapping[str, Any]]:
    enumerated, eligible, halo = _entry_exit_candidates(
        recipe, source, source_coordinates, ready_coordinates, contract
    )
    eligible = _apply_probe_band(eligible, options)
    minimum_window_s = float(
        recipe.raw["time_law"]["post_retime_behavior_opportunity_minimum_s"]
    )
    lower_bounds = {
        (row.entry_frame, row.exit_frame): _duration_lower_bound(
            source_coordinates,
            ready_coordinates,
            row,
            contract,
            minimum_window_s,
        )
        for row in eligible
    }
    ordered = sorted(
        eligible,
        key=lambda row: (
            lower_bounds[(row.entry_frame, row.exit_frame)],
            row.core_arc_length_l2_scaled,
            row.entry_frame,
            row.exit_frame,
        ),
    )
    enumeration_payload = [
        {
            "entry_frame": row.entry_frame,
            "exit_frame": row.exit_frame,
            "duration_lower_bound_s": lower_bounds.get(
                (row.entry_frame, row.exit_frame)
            ),
            "eligible": row in eligible,
        }
        for row in enumerated
    ]
    enumeration_digest = hashlib.sha256(
        json.dumps(
            enumeration_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    authority_window, authority_anchor, authority_anchor_semantics = (
        _authority_markers(recipe, source)
    )
    exact_context = _ExactSearchContext(
        source_coordinates=np.asarray(source_coordinates, dtype=np.float64),
        ready_coordinates=np.asarray(ready_coordinates, dtype=np.float64),
        window_start=int(authority_window[0]),
        window_end=int(authority_window[1]),
        source_anchor=int(authority_anchor),
        window_halo=int(halo),
        samples_per_scaled_unit=float(options.samples_per_scaled_unit),
        min_connector_intervals=int(options.min_connector_intervals),
        min_core_intervals=int(options.min_core_intervals),
        position_lower=np.asarray(contract.position_lower, dtype=np.float64),
        position_upper=np.asarray(contract.position_upper, dtype=np.float64),
        velocity=np.asarray(contract.velocity, dtype=np.float64),
        acceleration=np.asarray(contract.acceleration, dtype=np.float64),
        coordinate_scale=np.asarray(
            contract.coordinate_scale, dtype=np.float64
        ),
        coordinate_semantics=tuple(contract.coordinate_semantics),
        coordinate_units=tuple(contract.coordinate_units),
        fps=float(recipe.raw["time_law"]["fps"]),
        minimum_window_s=minimum_window_s,
        grid_subdivisions=int(options.grid_subdivisions),
        exact_pointwise_caps=bool(options.probe_exact_pointwise_caps),
    )

    def build_and_retime(
        row: EntryExitCandidate,
    ) -> tuple[CanonicalMotionGeometry, RetimeResult, float, float]:
        return _build_and_retime_exact(
            exact_context, row.entry_frame, row.exit_frame
        )

    best: _SearchWinner | None = None
    evaluated = 0
    pruned = 0
    failures: Counter[str] = Counter()
    process_batch_size = 0
    process_batch_count = 0
    if options.search_workers > 1:
        def measure(row: EntryExitCandidate):
            try:
                _, retimed, duration, total_variation = build_and_retime(row)
                score = _candidate_ranking_score(
                    retimed=retimed,
                    duration_s=duration,
                    total_variation_scaled_l2=total_variation,
                    entry_frame=row.entry_frame,
                    exit_frame=row.exit_frame,
                    fps=exact_context.fps,
                )
                return row, duration, total_variation, score, None
            except Exception as exc:
                return (
                    row,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )

        measured: list[
            tuple[
                EntryExitCandidate,
                float,
                float,
                tuple[int, int, int, int, int, float, int, int],
            ]
        ] = []
        if options.search_parallel_backend == "thread":
            with ThreadPoolExecutor(max_workers=options.search_workers) as executor:
                for row, duration, total_variation, score, error in executor.map(
                    measure, ordered
                ):
                    if error is not None:
                        failures[error] += 1
                        continue
                    assert duration is not None and total_variation is not None
                    assert score is not None
                    evaluated += 1
                    measured.append((row, duration, total_variation, score))
        else:
            row_by_pair = {
                (row.entry_frame, row.exit_frame): row for row in ordered
            }
            process_batch_size = max(1, options.search_workers)
            cursor = 0
            with ProcessPoolExecutor(
                max_workers=options.search_workers,
                initializer=_initialize_process_exact_search,
                initargs=(exact_context,),
            ) as executor:
                while cursor < len(ordered):
                    batch_end = min(
                        len(ordered), cursor + process_batch_size
                    )
                    pairs = [
                        (row.entry_frame, row.exit_frame)
                        for row in ordered[cursor:batch_end]
                    ]
                    process_batch_count += 1
                    for (
                        entry_frame,
                        exit_frame,
                        duration,
                        total_variation,
                        score,
                        error,
                    ) in executor.map(_measure_process_exact_search, pairs):
                        if error is not None:
                            failures[error] += 1
                            continue
                        assert duration is not None
                        assert total_variation is not None
                        assert score is not None
                        evaluated += 1
                        measured.append(
                            (
                                row_by_pair[(entry_frame, exit_frame)],
                                duration,
                                total_variation,
                                score,
                            )
                        )
                    cursor = batch_end
        if measured:
            selected_row, _, _, measured_score = min(
                measured, key=lambda item: item[3]
            )
            geometry, retimed, duration, total_variation = build_and_retime(
                selected_row
            )
            score = _candidate_ranking_score(
                retimed=retimed,
                duration_s=duration,
                total_variation_scaled_l2=total_variation,
                entry_frame=selected_row.entry_frame,
                exit_frame=selected_row.exit_frame,
                fps=exact_context.fps,
            )
            if score != measured_score:
                raise CanonicalMotionCompilerError(
                    "exact candidate ranking was not deterministic on "
                    f"rebuild: worker={measured_score!r} rebuild={score!r}"
                )
            best = _SearchWinner(
                candidate=selected_row,
                geometry=geometry,
                retimed=retimed,
                duration_s=duration,
                total_variation_scaled_l2=total_variation,
                score=score,
            )
    else:
        for row in ordered:
            try:
                geometry, retimed, duration, total_variation = (
                    build_and_retime(row)
                )
            except Exception as exc:
                failures[f"{type(exc).__name__}: {exc}"] += 1
                continue
            evaluated += 1
            score = _candidate_ranking_score(
                retimed=retimed,
                duration_s=duration,
                total_variation_scaled_l2=total_variation,
                entry_frame=row.entry_frame,
                exit_frame=row.exit_frame,
                fps=exact_context.fps,
            )
            winner = _SearchWinner(
                candidate=row,
                geometry=geometry,
                retimed=retimed,
                duration_s=duration,
                total_variation_scaled_l2=total_variation,
                score=score,
            )
            if best is None or score < best.score:
                best = winner
    if best is None:
        summary = "; ".join(
            f"{count}x {message}" for message, count in failures.most_common(5)
        )
        raise CanonicalMotionCompilerError(
            f"{source.motion_id} has no kinematically feasible retained core"
            + (f": {summary}" if summary else "")
        )

    comparator_entry = int(
        recipe.marker_semantics.row(source.motion_id).historical_adv2c3_start
    )
    comparator = {
        "role": recipe.raw["entry_exit_search"]["historical_adv2c3_role"],
        "entry_formula": (
            "recorded_by_marker_authority_v2_historical_adv2c3_comparator"
        ),
        "entry_frame": comparator_entry,
        "exit_frame": source.clip.n_frames - 1,
        "forced_or_seeded": False,
        "coincides_with_selected_entry": (
            comparator_entry == best.candidate.entry_frame
        ),
    }
    try:
        uniform_prefix = _retime_uniform_prefix_comparator(
            exact_context, best.geometry
        )
    except RetimeError as exc:
        uniform_prefix_receipt: Mapping[str, Any] = MappingProxyType(
            {
                "role": (
                    "uniform_scalar_path_acceleration_prefix_comparator_only_"
                    "not_an_adopted_time_law"
                ),
                "status": "INFEASIBLE_STRICT_UNIFORM",
                "reason": f"{type(exc).__name__}: {exc}",
                "same_geometry_and_limits": True,
                "no_brake_candidate_remains_valid": True,
                "dominance_status": "NOT_EVALUATED_COMPARATOR_INFEASIBLE",
                "non_claims": [
                    "strict_uniform_is_required_for_the_main_candidate",
                    "uniform_joint_acceleration",
                    "uniform_actuator_torque",
                    "continuous_time_optimality",
                ],
            }
        )
    else:
        uniform_prefix_receipt = _uniform_prefix_comparator_receipt(
            best.retimed,
            uniform_prefix,
            fps=exact_context.fps,
        )
    search_report = {
        "algorithm": (
            (
                "enumerate_all_then_process_parallel_exact_retime_then_"
                "control_tick_lexicographic_rank"
            )
            if (
                options.search_workers > 1
                and options.search_parallel_backend == "process"
            )
            else "enumerate_all_then_parallel_exact_retime_then_control_tick_rank"
            if options.search_workers > 1
            else
            "enumerate_all_then_exact_retime_then_control_tick_rank"
        ),
        "search_workers": int(options.search_workers),
        "search_parallel_backend": options.search_parallel_backend,
        "process_batch_size": int(process_batch_size),
        "process_batch_count": int(process_batch_count),
        "ranking": [
            "kinematic_feasibility",
            "nonnegative_scalar_acceleration_through_window_end",
            "minimum_window_start_control_tick",
            "minimum_source_anchor_control_tick",
            "minimum_prewindow_zero_acceleration_plateau_ticks",
            "minimum_worst_recovery_control_ticks",
            "minimum_cycle_control_ticks",
            "minimum_scaled_l2_total_variation",
            "entry_frame",
            "exit_frame",
        ],
        "control_tick_quantization": (
            "nearest integer inside a 4096*float64-epsilon product-error "
            "bound, otherwise ceil(nextafter(time_s*fps,-infinity)); exact "
            "tick ties advance to the next ranking criterion"
        ),
        "worst_recovery_definition": (
            "max(cycle_tick-hit_tick) over window_start, source_anchor, "
            "window_end; equivalently cycle_tick-window_start_tick"
        ),
        "marker_authority": (
            "legacy_ranking_anchor_only; source_anchor is not asserted to be "
            "nominal_event, preferred_contact, or behavior authority"
        ),
        "time_law_parameterization": (
            "weighted_arc_length_v1_digest_bound_to_coordinate_scale"
        ),
        "duration_lower_bound": (
            "max(contact_minimum, per_coordinate_total_variation/velocity, "
            "sum_per_waypoint_edge(max_component_displacement/velocity), "
            "2*sqrt(per_coordinate_total_variation/acceleration)); "
            "diagnostic ordering only, not an admissible prefix-tick bound"
        ),
        "pruning": (
            "disabled_until_an_admissible_prefix_tick_lower_bound_exists"
        ),
        "enumerated_count": len(enumerated),
        "eligible_count": len(eligible),
        "fully_retimed_feasible_count": evaluated,
        "admissibly_pruned_count": pruned,
        "retime_failure_count": int(sum(failures.values())),
        "retime_failure_histogram": dict(failures.most_common(20)),
        "enumeration_sha256": enumeration_digest,
        "selected": {
            "entry_frame": best.candidate.entry_frame,
            "exit_frame": best.candidate.exit_frame,
            "duration_s": best.duration_s,
            "window_start_control_tick": best.score[0],
            "source_anchor_control_tick": best.score[1],
            "prewindow_zero_acceleration_plateau_ticks": best.score[2],
            "worst_recovery_control_ticks": best.score[3],
            "cycle_control_ticks": best.score[4],
            "scaled_l2_total_variation": best.total_variation_scaled_l2,
            "duration_lower_bound_s": lower_bounds[
                (best.candidate.entry_frame, best.candidate.exit_frame)
            ],
            "includes_source_frame_zero": (
                best.candidate.includes_source_frame_zero
            ),
            "old_frame_zero_forced": False,
            "direct_path": "canonical_ready_to_selected_core_to_canonical_ready",
        },
        "adv2c3_comparator": comparator,
        "time_law_comparators": {
            "window_push_no_brake": {
                "role": "selected_fixed_grid_candidate_for_search_ranking",
                "adopted_for_training": False,
                "weighted_arc_length": dict(
                    best.retimed.report["weighted_arc_length"]
                ),
                "markers": {
                    name: {
                        "time_s": float(mapping.time_s),
                        "control_tick": _control_tick_at_or_after(
                            mapping.time_s, exact_context.fps
                        ),
                    }
                    for name, mapping in best.retimed.markers.items()
                },
                "cycle_duration_s": best.duration_s,
                "cycle_control_ticks": _control_tick_at_or_after(
                    best.duration_s, exact_context.fps
                ),
            },
            "uniform_scalar_path_acceleration_prefix": (
                uniform_prefix_receipt
            ),
        },
        "contact_opportunity": {
            "source_span_inclusive": list(authority_window),
            "source_anchor_frame": authority_anchor,
            "source_anchor_semantics": authority_anchor_semantics,
            "marker_only": True,
            "pose_locked": False,
            "velocity_locked": False,
            "acceleration_locked": False,
            "acceleration_allowed_through_window_end": True,
            "nonnegative_scalar_acceleration_through_window_end": True,
        },
    }
    return best, MappingProxyType(search_report)


def _scalar_no_early_brake_diagnostic(
    retimed: RetimeResult,
    *,
    fps: float,
    negative_tolerance: float = 1.0e-9,
) -> Mapping[str, Any]:
    """Verify the enforced scalar no-braking constraint through window end.

    The constraint participates in every candidate retime, but remains only a
    scalar-path proxy.  It is not a racket-speed, joint-acceleration, actuator
    torque, contact, or useful-return certificate.
    """

    acceleration = np.asarray(retimed.path_acceleration, dtype=np.float64)
    if acceleration.ndim != 1 or not np.isfinite(acceleration).all():
        raise CanonicalMotionCompilerError(
            "retimer path_acceleration must be a finite one-dimensional array"
        )
    if not math.isfinite(negative_tolerance) or negative_tolerance <= 0.0:
        raise CanonicalMotionCompilerError(
            "negative scalar-acceleration tolerance must be positive"
        )
    start_fractional = float(
        retimed.markers["window_start"].output_fractional_frame
    )
    end_fractional = float(
        retimed.markers["window_end"].output_fractional_frame
    )
    policy = retimed.report.get("nonnegative_acceleration_until_marker")
    if (
        not isinstance(policy, Mapping)
        or policy.get("enabled") is not True
        or policy.get("marker") != "window_end"
    ):
        raise CanonicalMotionCompilerError(
            "retimer did not bind the required window_end no-braking policy"
        )
    continuous_min_raw = policy.get(
        "prefix_scalar_acceleration_min_continuous"
    )
    if continuous_min_raw is None:
        raise CanonicalMotionCompilerError(
            "window_end no-braking policy has no continuous prefix evidence"
        )
    continuous_min = float(continuous_min_raw)
    if (
        not math.isfinite(continuous_min)
        or continuous_min < -float(negative_tolerance)
    ):
        raise CanonicalMotionCompilerError(
            "retimer violated the continuous window_end no-braking constraint"
        )
    output_min_raw = policy.get("prefix_scalar_acceleration_min_50hz")
    if output_min_raw is None:
        raise CanonicalMotionCompilerError(
            "window_end no-braking policy has no 50 Hz prefix evidence"
        )
    output_min = float(output_min_raw)
    if (
        not math.isfinite(output_min)
        or output_min < -float(negative_tolerance)
    ):
        raise CanonicalMotionCompilerError(
            "retimer violated the 50 Hz window_end no-braking constraint"
        )
    if (
        policy.get("control_guard_enabled") is not True
        or policy.get("output_interval_policy")
        not in (
            "every_interval_starting_before_exact_marker_must_be_"
            "nonnegative_including_the_straddling_interval",
            "every_interval_starting_at_or_after_from_marker_and_"
            "before_exact_marker_must_be_nonnegative_including_the_"
            "until_straddling_interval",
        )
    ):
        raise CanonicalMotionCompilerError(
            "retimer did not bind the explicit 50 Hz straddling-interval "
            "no-braking guard"
        )

    segment_start = np.arange(len(acceleration), dtype=np.float64)
    starts_before_end = segment_start < end_fractional - 1.0e-12
    # When the retimer scoped the no-brake law to [from_marker, window_end]
    # (Franco 2026-07-25 ruling: the approach may decelerate into the
    # backswing reversal), the recheck covers the same scoped range instead
    # of the whole prefix.
    scoped_from = policy.get("from_marker")
    if scoped_from is not None:
        from_fractional = float(
            retimed.markers[str(scoped_from)].output_fractional_frame
        )
        starts_before_end = starts_before_end & (
            segment_start >= from_fractional - 1.0e-12
        )
    overlaps_window = (segment_start < end_fractional - 1.0e-12) & (
        segment_start + 1.0 > start_fractional + 1.0e-12
    )
    negative = acceleration < -float(negative_tolerance)
    negative_before = np.flatnonzero(starts_before_end & negative)
    negative_window = np.flatnonzero(overlaps_window & negative)
    if len(negative_before):
        raise CanonicalMotionCompilerError(
            "retimed output contains negative scalar acceleration before or "
            "across the exact window_end control interval"
        )
    first = int(negative_before[0]) if len(negative_before) else None
    return MappingProxyType(
        {
            "name": "scalar_no_early_brake_proxy_v2",
            "observed_profile": (
                "fixed_grid_kinematic_candidate_with_nonnegative_scalar_"
                "acceleration_through_the_snapped_50hz_guard"
            ),
            "selection_criterion": True,
            "continuous_prefix_minimum_path_units_s2": continuous_min,
            "output_50hz_prefix_minimum_path_units_s2": output_min,
            "segment_acceleration_semantics": (
                "average scalar path acceleration between output samples"
            ),
            "window_start_fractional_frame": start_fractional,
            "window_end_fractional_frame": end_fractional,
            "negative_tolerance_path_units_s2": float(negative_tolerance),
            "negative_segment_count_before_window_end": int(
                len(negative_before)
            ),
            "negative_segment_count_overlapping_window": int(
                len(negative_window)
            ),
            "no_negative_scalar_acceleration_before_window_end": (
                len(negative_before) == 0
            ),
            "no_negative_scalar_acceleration_inside_window": (
                len(negative_window) == 0
            ),
            "first_negative_segment_before_window_end": first,
            "first_negative_time_s": (
                None if first is None else float(first) / float(fps)
            ),
            "minimum_scalar_path_acceleration_before_window_end": (
                None
                if not np.any(starts_before_end)
                else float(np.min(acceleration[starts_before_end]))
            ),
            "proxy_only": True,
            "does_not_establish": [
                "useful_racket_speed",
                "racket_speed_monotonicity",
                "joint_acceleration_sign",
                "uniform_joint_acceleration",
                "uniform_actuator_torque",
                "contact_or_return_quality",
            ],
        }
    )


def _decode_timed_candidate(
    recipe: CanonicalMotionRecipe,
    scope: str,
    retimed: RetimeResult,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if scope == "upper":
        joint_pos = np.asarray(retimed.q, dtype=np.float64)
        joint_vel = np.asarray(retimed.qdot, dtype=np.float64)
        frames = len(joint_pos)
        root_pos = np.broadcast_to(recipe.ready.root_pos_w, (frames, 3)).copy()
        ready_quat = np.asarray(recipe.ready.root_quat_wxyz, dtype=np.float64)
        ready_quat = ready_quat / np.linalg.norm(ready_quat)
        root_quat = np.broadcast_to(ready_quat, (frames, 4)).copy()
        root_lin = np.zeros((frames, 3), dtype=np.float64)
        root_ang = np.zeros((frames, 3), dtype=np.float64)
    elif scope == "full":
        joint_pos = np.asarray(retimed.q[:, :31], dtype=np.float64)
        joint_vel = np.asarray(retimed.qdot[:, :31], dtype=np.float64)
        root_coordinates = np.asarray(retimed.q[:, 31:], dtype=np.float64)
        root_coordinate_velocity = np.asarray(
            retimed.qdot[:, 31:], dtype=np.float64
        )
        root_pos, root_quat = decode_root_pose(
            root_coordinates,
            canonical_ready_root_quat_wxyz=recipe.ready.root_quat_wxyz,
        )
        root_lin, root_ang = root_coordinate_velocity_to_world_twist(
            root_coordinates,
            root_coordinate_velocity,
            canonical_ready_root_quat_wxyz=recipe.ready.root_quat_wxyz,
        )
    else:  # pragma: no cover - guarded above
        raise CanonicalMotionCompilerError(f"unknown scope {scope!r}")

    if not np.array_equal(joint_pos[0], recipe.ready.joint_pos) or not np.array_equal(
        joint_pos[-1], recipe.ready.joint_pos
    ):
        raise CanonicalMotionCompilerError(
            "retimed candidate did not retain exact shared-ready joint endpoints"
        )
    for label, value in (
        ("joint velocity", joint_vel),
        ("root linear velocity", root_lin),
        ("root angular velocity", root_ang),
    ):
        if np.count_nonzero(value[[0, -1]]) != 0:
            raise CanonicalMotionCompilerError(
                f"{label} endpoints are not exactly zero"
            )
    if not np.array_equal(root_pos[0], root_pos[-1]) or not np.array_equal(
        root_quat[0], root_quat[-1]
    ):
        raise CanonicalMotionCompilerError(
            "retimed candidate did not return to one shared root ready pose"
        )
    return joint_pos, joint_vel, root_pos, root_quat, root_lin, root_ang


def _stable_candidate_input_sha256(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    scope: str,
    winner: _SearchWinner,
    published_arrays: Mapping[str, np.ndarray],
) -> str:
    """Digest recipe identity plus the exact numeric payload published to NPZ.

    This is deliberately recomputable by an independent post-build verifier.
    The schema-2 output hash already binds the complete archive; this second
    digest gives the verifier a stable compiler-input contract without asking
    it to trust unpublished float64 retimer intermediates.
    """

    header = {
        "digest_contract": "canonical_published_numeric_input_v1",
        "library_id": recipe.raw["library_id"],
        "recipe_sha256": sha256_file(recipe.path),
        "ready_sha256": recipe.ready.sha256,
        "source_sha256": source.sha256,
        "motion_id": source.motion_id,
        "scope": scope,
        "entry_frame": winner.candidate.entry_frame,
        "exit_frame": winner.candidate.exit_frame,
        "window_policy": (
            recipe.raw["time_law"]["legacy_seed_marker_policy"]
        ),
        "adv2c3_role": (
            recipe.raw["entry_exit_search"]["historical_adv2c3_role"]
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    for key in _PUBLISHED_INPUT_ARRAY_KEYS:
        if key not in published_arrays:
            raise CanonicalMotionCompilerError(
                f"schema-2 candidate is missing digest input array {key!r}"
            )
        raw = np.asarray(published_arrays[key])
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise CanonicalMotionCompilerError(
                f"schema-2 digest input {key!r} must be real numeric"
            )
        array = np.ascontiguousarray(raw, dtype="<f8")
        if not np.all(np.isfinite(array)):
            raise CanonicalMotionCompilerError(
                f"schema-2 digest input {key!r} became non-finite"
            )
        digest.update(key.encode("ascii") + b"\0")
        digest.update(
            json.dumps(
                list(array.shape), separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _rebind_schema2_input_sha256(
    candidate: Schema2Candidate, input_sha256: str
) -> Schema2Candidate:
    """Replace only the schema receipts after deriving a published-array digest."""

    manifest = _json_safe(candidate.manifest)
    report = _json_safe(candidate.report)
    for payload, label in ((manifest, "manifest"), (report, "report")):
        hashes = payload.get("hashes")
        if not isinstance(hashes, dict) or "input_sha256" not in hashes:
            raise CanonicalMotionCompilerError(
                f"schema-2 {label} omitted hashes.input_sha256"
            )
        hashes["input_sha256"] = input_sha256
    return Schema2Candidate(
        arrays=candidate.arrays,
        npz_bytes=candidate.npz_bytes,
        manifest=MappingProxyType(manifest),
        report=MappingProxyType(report),
    )


def _compile_motion_scope(
    recipe: CanonicalMotionRecipe,
    source: MotionSource,
    scope: str,
    *,
    backend: RightRacketBackend,
    options: CompilerOptions,
    contract: _PathContract,
) -> CompiledMotion:
    scoped = _preprocess_scope(recipe, source, scope, options)
    scoped, face_report = _solve_synthetic_face(
        recipe,
        source,
        scoped,
        backend=backend,
        options=options,
        velocity_fraction=float(
            recipe.raw["time_law"]["joint_velocity_limit_fraction"]
        ),
    )
    source_coordinates, ready_coordinates, root_report = _coordinates(
        recipe, scoped, scope
    )
    # Optional probe-only source smoothing runs once here, after body-scope
    # preprocessing (and the synthetic face solve) and before any entry/exit
    # enumeration or geometry construction, so frame indices stay valid.
    source_coordinates, smoothing_report = _apply_probe_source_smoothing(
        source_coordinates, options
    )
    if np.any(
        source_coordinates
        < contract.position_lower[None, :] - _POSITION_TOLERANCE
    ) or np.any(
        source_coordinates
        > contract.position_upper[None, :] + _POSITION_TOLERANCE
    ):
        raise CanonicalMotionCompilerError(
            f"{source.motion_id} {scope} scoped source leaves declared path bounds"
        )
    winner, search_report = _search_core(
        recipe,
        source,
        source_coordinates,
        ready_coordinates,
        contract,
        options=options,
    )
    if winner.retimed.collocation_trace is None:
        raise CanonicalMotionCompilerError(
            "canonical explicit-progress retimer omitted its accepted "
            "in-memory collocation trace"
        )
    decoded = _decode_timed_candidate(recipe, scope, winner.retimed)
    schema2 = build_schema2_candidate(
        joint_pos=decoded[0],
        joint_vel=decoded[1],
        root_pos_w=decoded[2],
        root_quat_wxyz=decoded[3],
        root_lin_vel_w=decoded[4],
        root_ang_vel_w=decoded[5],
        fps=float(recipe.raw["time_law"]["fps"]),
        mjcf_path=recipe.model_paths["mjcf"],
        # The deterministic NPZ payload does not contain this receipt.  Build
        # once, derive the independently reproducible digest from its exact
        # published arrays, then rebind only the two JSON receipts below.
        input_sha256="0" * 64,
        ready_sha256=recipe.ready.sha256,
        body_order_path=recipe.model_paths["body_order"],
    )
    input_digest = _stable_candidate_input_sha256(
        recipe, source, scope, winner, schema2.arrays
    )
    schema2 = _rebind_schema2_input_sha256(schema2, input_digest)
    if (
        schema2.manifest.get("publication_class") != PUBLICATION_CLASS
        or schema2.manifest.get("training_authorized") is not False
    ):
        raise CanonicalMotionCompilerError(
            "schema-2 builder weakened the compiler-candidate publication class"
        )
    window_start = winner.retimed.markers["window_start"].time_s
    window_end = winner.retimed.markers["window_end"].time_s
    anchor = winner.retimed.markers["source_anchor"].time_s
    authority_window, authority_anchor, _ = _authority_markers(recipe, source)
    scope_report = dict(scoped.report)
    if root_report is not None:
        scope_report["root_coordinate_codec"] = dict(root_report)
    # Per motion+scope provenance for the probe smoothing knob (value plus the
    # passes and max deviation actually reached); the knob's global value lives
    # in the compiler-options receipt alongside probe_entry_band et al.
    scope_report["probe_source_smoothing"] = dict(smoothing_report)
    retime_report = dict(winner.retimed.report)
    retime_report["scalar_no_early_brake_proxy"] = dict(
        _scalar_no_early_brake_diagnostic(
            winner.retimed,
            fps=float(recipe.raw["time_law"]["fps"]),
        )
    )
    return CompiledMotion(
        motion_id=source.motion_id,
        scope=scope,
        filename=f"{source.motion_id}_{scope}_canonical_v2.npz",
        schema2=schema2,
        entry_frame=winner.candidate.entry_frame,
        exit_frame=winner.candidate.exit_frame,
        duration_s=winner.duration_s,
        contact_window_start_s=float(window_start),
        contact_window_end_s=float(window_end),
        source_anchor_time_s=float(anchor),
        total_variation_scaled_l2=winner.total_variation_scaled_l2,
        search_report=search_report,
        scope_report=MappingProxyType(scope_report),
        face_report=face_report,
        geometry_report=MappingProxyType(
            {
                **dict(winner.geometry.continuity_report),
                "recipe_marker_binding": {
                    name: {
                        "source_frame": int(source_frame),
                        "dense_row": int(
                            _marker_path_index(winner.geometry, source_frame)
                        ),
                    }
                    for name, source_frame in (
                        ("window_start", authority_window[0]),
                        ("source_anchor", authority_anchor),
                        ("window_end", authority_window[1]),
                    )
                },
            }
        ),
        retime_report=MappingProxyType(retime_report),
        collocation_trace=winner.retimed.collocation_trace,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seed_sha256(seeds: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(b"canonical-face-active-seeds-v1\0")
    digest.update(str(len(seeds)).encode("ascii"))
    for seed in seeds:
        value = np.ascontiguousarray(np.asarray(seed, dtype="<f8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _compiler_options_receipt(
    options: CompilerOptions,
    *,
    joint_acceleration: np.ndarray,
    root_limits: FullRootPathLimits,
    face_config: FaceManifoldConfig,
    contracts: Mapping[str, _PathContract],
) -> Mapping[str, Any]:
    """Bind every effective non-recipe compiler option and coordinate contract."""

    seed_values = tuple(
        _finite_vector(seed, 7, f"face_active_candidate_seeds[{index}]")
        for index, seed in enumerate(options.face_active_candidate_seeds)
    )
    payload: dict[str, Any] = {
        "joint_acceleration_limits_rad_s2": {
            "joint_order": list(RUNTIME_JOINT_NAMES),
            "values": joint_acceleration.tolist(),
            "sha256": _array_sha256(joint_acceleration),
        },
        "full_root_limits": {
            "coordinate_order": list(ROOT_COORDINATE_NAMES),
            "coordinate_units": list(ROOT_COORDINATE_UNITS),
            "position_lower": root_limits.position_lower.tolist(),
            "position_upper": root_limits.position_upper.tolist(),
            "velocity": root_limits.velocity.tolist(),
            "acceleration": root_limits.acceleration.tolist(),
            "sha256": hashlib.sha256(
                b"".join(
                    bytes.fromhex(_array_sha256(value))
                    for value in (
                        root_limits.position_lower,
                        root_limits.position_upper,
                        root_limits.velocity,
                        root_limits.acceleration,
                    )
                )
            ).hexdigest(),
        },
        "s0_full_grounding_offset_m": (
            None
            if options.s0_full_grounding_offset_m is None
            else float(options.s0_full_grounding_offset_m)
        ),
        "geometry_and_grid": {
            "samples_per_scaled_unit": float(options.samples_per_scaled_unit),
            "min_connector_intervals": int(options.min_connector_intervals),
            "min_core_intervals": int(options.min_core_intervals),
            "grid_subdivisions": int(options.grid_subdivisions),
            "search_workers": int(options.search_workers),
            "search_parallel_backend": options.search_parallel_backend,
            "probe_entry_band": (
                None
                if options.probe_entry_band is None
                else int(options.probe_entry_band)
            ),
            "probe_exit_band": (
                None
                if options.probe_exit_band is None
                else int(options.probe_exit_band)
            ),
            "probe_band_is_exhaustive_enumeration": (
                options.probe_entry_band is None
                and options.probe_exit_band is None
            ),
            "probe_exact_pointwise_caps": bool(
                options.probe_exact_pointwise_caps
            ),
            "probe_source_smoothing_tolerance_rad": (
                None
                if options.probe_source_smoothing_tolerance_rad is None
                else float(options.probe_source_smoothing_tolerance_rad)
            ),
            "probe_source_smoothing_is_identity": (
                options.probe_source_smoothing_tolerance_rad is None
            ),
        },
        "face_manifold": {
            "effective_config": _json_safe(asdict(face_config)),
            "active_candidate_seed_count": len(seed_values),
            "active_candidate_seeds_sha256": _seed_sha256(seed_values),
        },
        "effective_path_coordinate_contracts": {
            scope: {
                "dimension": int(len(contract.velocity)),
                "coordinate_order": list(contract.coordinate_semantics),
                "coordinate_units": list(contract.coordinate_units),
                "layout": (
                    "joint_31_then_root_position_xyz_then_"
                    "ready_relative_root_rotation_vector_xyz"
                    if scope == "full"
                    else "joint_31"
                ),
                "position_lower": contract.position_lower.tolist(),
                "position_upper": contract.position_upper.tolist(),
                "velocity": contract.velocity.tolist(),
                "acceleration": contract.acceleration.tolist(),
                "coordinate_scale": contract.coordinate_scale.tolist(),
                "sha256": hashlib.sha256(
                    b"".join(
                        bytes.fromhex(_array_sha256(value))
                        for value in (
                            contract.position_lower,
                            contract.position_upper,
                            contract.velocity,
                            contract.acceleration,
                            contract.coordinate_scale,
                        )
                    )
                ).hexdigest(),
            }
            for scope, contract in contracts.items()
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["compiler_options_sha256"] = hashlib.sha256(canonical).hexdigest()
    return MappingProxyType(payload)


def _library_manifest(
    recipe: CanonicalMotionRecipe,
    motions: Sequence[CompiledMotion],
    *,
    options_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    tool_path = Path(__file__).resolve()
    geometry_tool_path = Path(build_canonical_geometry.__code__.co_filename).resolve()
    weighted_arc_tool_path = Path(
        build_weighted_arc_path.__code__.co_filename
    ).resolve()
    outputs = []
    for motion in motions:
        outputs.append(
            {
                "motion_id": motion.motion_id,
                "scope": motion.scope,
                "filename": motion.filename,
                "output_npz_sha256": motion.schema2.output_sha256,
                "entry_frame": motion.entry_frame,
                "exit_frame": motion.exit_frame,
                "duration_s": motion.duration_s,
                "contact_window_start_s": motion.contact_window_start_s,
                "contact_window_end_s": motion.contact_window_end_s,
                "source_anchor_time_s": motion.source_anchor_time_s,
                "scaled_l2_total_variation": (
                    motion.total_variation_scaled_l2
                ),
                "search": _json_safe(motion.search_report),
                "scope_preprocessing": _json_safe(motion.scope_report),
                "face_manifold": _json_safe(motion.face_report),
                "geometry": _json_safe(motion.geometry_report),
                "retiming": _json_safe(motion.retime_report),
                "schema2_manifest": _json_safe(motion.schema2.manifest),
                "schema2_report": _json_safe(motion.schema2.report),
            }
        )
    manifest = {
        "schema_version": 1,
        "library_id": recipe.raw["library_id"],
        "publication_class": PUBLICATION_CLASS,
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "training_authorized": False,
        "hardware_authorized": False,
        "recipe": {
            "path": str(recipe.path),
            "sha256": sha256_file(recipe.path),
        },
        "compiler": {
            "path": str(tool_path),
            "sha256": sha256_file(tool_path),
        },
        "geometry_tool": {
            "path": str(geometry_tool_path),
            "sha256": sha256_file(geometry_tool_path),
        },
        "weighted_arc_tool": {
            "path": str(weighted_arc_tool_path),
            "sha256": sha256_file(weighted_arc_tool_path),
        },
        "compiler_options": _json_safe(options_receipt),
        "ready": {
            "path": str(recipe.ready.path),
            "sha256": recipe.ready.sha256,
            "direct_endpoint_for_every_motion": True,
            "old_source_frame_zero_bridge_inserted": False,
        },
        "output_matrix": {
            "motion_ids": [source.motion_id for source in recipe.sources],
            "scopes": list(SCOPES),
            "candidate_count": len(outputs),
        },
        "search_contract": {
            "entry_exit": "enumerate_all_then_gate_and_rank",
            "strike_time_reference_s": _STRIKE_TIME_REFERENCE_S,
            "strike_time_reference_is_hard_gate": False,
            "ranking": (
                "feasibility_and_no_scalar_braking_through_window_end_then_"
                "window_start_tick_then_anchor_tick_then_prewindow_plateau_ticks_"
                "then_worst_recovery_tick_"
                "then_cycle_tick_then_scaled_path_total_variation"
            ),
            "adv2c3": "comparator_only_not_default",
            "marker_authority": (
                "legacy_ranking_anchor_only_not_nominal_event_or_preferred"
            ),
            "time_law_parameterization": (
                "weighted_arc_length_v1_digest_bound_to_coordinate_scale"
            ),
        },
        "contact_opportunity_contract": {
            "marker_only": True,
            "pose_speed_acceleration_freeze": False,
            "acceleration_allowed_through_window_end": True,
            "nonnegative_scalar_acceleration_through_window_end": True,
        },
        "time_law_claim": (
            "weighted_arc_scalar_coordinate_only; kinematic_velocity_"
            "acceleration_and_no_early_scalar_braking_warm_start_only"
        ),
        "outputs": outputs,
        "post_build_gates": [
            {"name": name, "status": "pending"}
            for name in recipe.raw["post_build_gates"]
        ],
        "non_claims": [
            "uniform_actuator_torque",
            "inverse_dynamics_feasibility",
            "balance",
            "collision_clearance",
            "table_tennis_contact_or_return",
            "nominal_event_or_preferred_contact_authority",
            "learnability",
            "training_authorization",
            "deployment",
            "hardware_authorization",
        ],
    }
    return MappingProxyType(_json_safe(manifest))


def compile_loaded_canonical_motion_library(
    recipe: CanonicalMotionRecipe,
    *,
    options: CompilerOptions,
    backend: RightRacketBackend | None = None,
) -> CompiledLibrary:
    """Compile an already integrity-checked recipe into ten in-memory candidates."""

    if not isinstance(recipe, CanonicalMotionRecipe):
        raise CanonicalMotionCompilerError(
            "recipe must come from load_canonical_motion_recipe"
        )
    if recipe.raw["publication_class"] != PUBLICATION_CLASS:
        raise CanonicalMotionCompilerError(
            "recipe publication_class must remain compiler_candidate"
        )
    if (
        recipe.raw["training_authorized"] is not False
        or recipe.raw["hardware_authorized"] is not False
    ):
        raise CanonicalMotionCompilerError(
            "candidate compiler refuses an authorized recipe"
        )
    if recipe.raw["entry_exit_search"]["historical_adv2c3_role"] != (
        "comparator_only_not_default"
    ):
        raise CanonicalMotionCompilerError(
            "adv2c3 may only be a comparator"
        )
    time_law = recipe.raw["time_law"]
    if (
        time_law["legacy_seed_marker_policy"]
        != "search_and_retime_marker_only_never_output_behavior_window"
        or time_law["window_acceleration_allowed_through_end"] is not True
    ):
        raise CanonicalMotionCompilerError(
            "contact opportunity must remain marker-only with acceleration allowed"
        )
    joint_acceleration, root_limits = _validate_options(options)
    exact_backend = backend or MujocoRightRacketBackend(
        recipe.model_paths["mjcf"],
        RUNTIME_JOINT_NAMES,
        urdf_path=recipe.model_paths["urdf"],
    )
    joint_lower, joint_upper, joint_velocity = _validate_backend(exact_backend)
    velocity_fraction = float(time_law["joint_velocity_limit_fraction"])
    contracts = {
        scope: _path_contract(
            scope=scope,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            joint_velocity=joint_velocity,
            joint_acceleration=joint_acceleration,
            root_limits=root_limits,
            velocity_fraction=velocity_fraction,
        )
        for scope in SCOPES
    }
    face_config = _effective_face_config(options, velocity_fraction)
    options_receipt = _compiler_options_receipt(
        options,
        joint_acceleration=joint_acceleration,
        root_limits=root_limits,
        face_config=face_config,
        contracts=contracts,
    )
    compiled: list[CompiledMotion] = []
    for source in recipe.sources:
        for scope in SCOPES:
            compiled.append(
                _compile_motion_scope(
                    recipe,
                    source,
                    scope,
                    backend=exact_backend,
                    options=options,
                    contract=contracts[scope],
                )
            )
    expected = [
        (source.motion_id, scope)
        for source in recipe.sources
        for scope in SCOPES
    ]
    actual = [(row.motion_id, row.scope) for row in compiled]
    if actual != expected or len(compiled) != len(expected):
        raise CanonicalMotionCompilerError(
            f"required recipe output matrix changed: {actual}"
        )
    manifest = _library_manifest(
        recipe,
        compiled,
        options_receipt=options_receipt,
    )
    return CompiledLibrary(
        recipe=recipe,
        motions=tuple(compiled),
        manifest=manifest,
    )


def compile_canonical_motion_library(
    recipe_path: str | Path,
    *,
    options: CompilerOptions,
    repo_root: str | Path | None = None,
    backend: RightRacketBackend | None = None,
) -> CompiledLibrary:
    """Load the strict JSON recipe, then compile all five motions in both scopes."""

    recipe = load_canonical_motion_recipe(recipe_path, repo_root=repo_root)
    return compile_loaded_canonical_motion_library(
        recipe, options=options, backend=backend
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                errno.ENOSYS,
                "atomic renameat2(RENAME_NOREPLACE) is unavailable",
                str(destination),
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(
                errno.ENOSYS,
                "atomic renamex_np(RENAME_EXCL) is unavailable",
                str(destination),
            )
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            source_bytes,
            destination_bytes,
            _RENAME_EXCL,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            str(destination),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"refusing to overwrite concurrently created output: {destination}",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def write_compiled_canonical_motion_library(
    library: CompiledLibrary,
    output_directory: str | Path,
) -> Path:
    """Publish all ten candidates atomically to a previously absent directory.

    The complete library is first written into a private sibling staging
    directory.  The final directory must not exist (including as a broken
    symlink), and every schema-2 writer also uses exclusive no-clobber paths.
    On failure, only the compiler-created staging directory is removed.
    """

    if not isinstance(library, CompiledLibrary):
        raise CanonicalMotionCompilerError("library must be CompiledLibrary")
    declared = library.manifest.get("output_matrix", {})
    declared_count = declared.get("candidate_count")
    if (
        not isinstance(declared_count, int)
        or declared_count < 1
        or len(library.motions) != declared_count
    ):
        raise CanonicalMotionCompilerError("refusing to publish an incomplete library")
    output = Path(os.path.abspath(os.fspath(Path(output_directory).expanduser())))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=str(output.parent),
        )
    )
    try:
        for motion in library.motions:
            write_schema2_candidate(
                motion.schema2,
                staging / motion.filename,
            )
        manifest_path = staging / BUILD_MANIFEST_NAME
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(library.manifest))
        _rename_directory_noreplace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


__all__ = [
    "BUILD_MANIFEST_NAME",
    "CanonicalMotionCompilerError",
    "CompiledLibrary",
    "CompiledMotion",
    "CompilerOptions",
    "FullRootPathLimits",
    "compile_canonical_motion_library",
    "compile_loaded_canonical_motion_library",
    "write_compiled_canonical_motion_library",
]
