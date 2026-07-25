#!/usr/bin/env python3
"""Canonical-ready closed-loop motion geometry.

This module solves one deliberately narrow problem: turn a recorded coordinate
path (joints, or joints plus a continuous root representation) into a
*geometric* closed loop

    canonical ready -> retained source core -> canonical ready

without first travelling through the recording's old frame 0.  It does not
assign timestamps, velocities in seconds, torques, or feasibility.  A later
time-law/dynamics stage must do that work.

The generated curve is C2 and *regular* in a path parameter ``s``:

* a quintic connector leaves canonical ready with a deterministic non-zero
  chord tangent and zero path curvature, then joins the selected source entry
  with the source tangent and curvature;
* every retained source waypoint is joined by a quintic Hermite segment using a
  shared derivative estimate, so the recorded waypoints remain exact while
  their in-between sampling is no longer treated as immutable time bytes;
* a symmetric connector leaves the selected source exit and returns to
  canonical ready with a deterministic non-zero chord tangent and zero path
  curvature.

``dq/ds != 0`` is a geometric regularity requirement.  It is intentionally not
a claim that the robot has non-zero physical velocity while waiting at ready:
the later time law may set ``ds/dt = 0`` at either endpoint.  Putting physical
holds into the geometry as ``dq/ds = 0`` would instead create a singular scalar
retiming problem.

Sampling density is controlled by ``coordinate_scale * native displacement``.
The legacy ``samples_per_rad`` name remains exact for the default all-joint-rad
contract and means samples per scaled coordinate unit for mixed joint/root
paths.  It is only geometric resolution.  Neither the number of returned rows
nor the path parameter is a duration, and coordinates are never rescaled in the
returned path.

Pure NumPy.  No FK, SO(3) unwrap, time law, torque model, simulator, or file I/O
lives here.  Root rotation coordinates must already be continuous (for example,
an unwrapped rotation vector).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import numpy as np


_LABEL_READY_START = "canonical_ready_start"
_LABEL_PRE = "ready_to_source_core"
_LABEL_CORE = "source_core"
_LABEL_WINDOW = "contact_opportunity"
_LABEL_POST = "source_core_to_ready"
_LABEL_READY_END = "canonical_ready_end"
_MIN_QUINTIC_INTERVALS = 5
_MIN_CONNECTOR_PARAMETER_SPAN = 1e-6
_REGULARITY_EPS_FACTOR = 4096.0
_REGULARITY_MAX_SUBDIVISION_DEPTH = 32
_CANONICAL_QNAN_BITS = np.uint64(0x7FF8000000000000)
SOURCE_FRAME_MAP_ENCODING = "float64_le_canonical_qnan_v1"


@dataclass(frozen=True)
class CanonicalMotionGeometry:
    """Dense, C2 coordinate-space geometry and its provenance markers.

    ``dq_ds`` and ``d2q_ds2`` are derivatives with respect to
    ``path_parameter``.  They are not physical velocities or accelerations
    (neither rad/s nor m/s).
    """

    q_path: np.ndarray
    path_parameter: np.ndarray
    dq_ds: np.ndarray
    d2q_ds2: np.ndarray
    source_frame_map: np.ndarray
    segment_labels: np.ndarray
    window_mask: np.ndarray
    window_path_start: int
    window_path_end: int
    source_waypoint_path_indices: np.ndarray
    entry_frame: int
    exit_frame: int
    window_start: int
    window_end: int
    continuity_report: dict[str, Any]
    # Default preserves construction compatibility for external test doubles
    # created before schema v2.  Real builders always populate this field.
    canonical_knot_path_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


@dataclass(frozen=True)
class EntryExitCandidate:
    """One unranked retained-core candidate.

    These are measurements, not a hidden optimizer.  A later FK/dynamics stage
    decides which candidate is executable.  Generic distance fields use
    ``coordinate_scale * native_delta``.  Legacy ``*_rad`` fields are populated
    only when every declared coordinate unit is ``rad``; otherwise they are
    ``None`` rather than mislabelling meters as radians.
    """

    entry_frame: int
    exit_frame: int
    core_frame_count: int
    entry_to_window_frames: int
    window_to_exit_frames: int
    skipped_prefix_frames: int
    skipped_suffix_frames: int
    includes_source_frame_zero: bool
    ready_to_entry_max_abs_scaled: float
    exit_to_ready_max_abs_scaled: float
    ready_to_entry_l2_scaled: float
    exit_to_ready_l2_scaled: float
    core_arc_length_l2_scaled: float
    coordinate_units_mixed: bool
    ready_to_entry_max_abs_rad: float | None
    exit_to_ready_max_abs_rad: float | None
    ready_to_entry_l2_rad: float | None
    exit_to_ready_l2_rad: float | None
    core_arc_length_l2_rad: float | None


@dataclass(frozen=True)
class _CoordinateContract:
    scale: np.ndarray
    semantics: tuple[str, ...]
    units: tuple[str, ...]
    all_rad: bool
    mixed_units: bool


@dataclass(frozen=True)
class _Segment:
    q0: np.ndarray
    v0: np.ndarray
    a0: np.ndarray
    q1: np.ndarray
    v1: np.ndarray
    a1: np.ndarray
    s0: float
    span: float
    sample_intervals: int
    label: str
    source_frame_start: float | None
    source_frame_end: float | None


def _finite_float_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values, got {raw.dtype}")
    out = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or infinity")
    return out


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be > 0 and finite, got {value!r}")
    return result


def _validate_source_ready(
    source_q: Any,
    ready_q: Any,
) -> tuple[np.ndarray, np.ndarray]:
    source = _finite_float_array(source_q, name="source_q", ndim=2)
    ready = _finite_float_array(ready_q, name="ready_q", ndim=1)
    frames, joints = source.shape
    if frames < 3:
        raise ValueError(
            "source_q needs at least 3 frames for a fail-closed C2 derivative estimate"
        )
    if ready.shape != (joints,):
        raise ValueError(
            f"ready_q must have shape ({joints},), got {ready.shape}"
        )
    return source, ready


def _string_vector(
    value: Any,
    *,
    name: str,
    size: int,
    default: str,
) -> tuple[str, ...]:
    if value is None:
        value = default
    if isinstance(value, str):
        values = (value,) * size
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a string or a length-{size} sequence of strings"
            ) from exc
        if len(values) != size:
            raise ValueError(
                f"{name} must have length {size}, got {len(values)}"
            )
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} entries must be non-empty strings")
    return tuple(item.strip() for item in values)


def _coordinate_contract(
    dimension: int,
    *,
    coordinate_scale: Any,
    coordinate_semantics: Any,
    coordinate_units: Any,
) -> _CoordinateContract:
    if coordinate_scale is None:
        scale = np.ones(dimension, dtype=np.float64)
    else:
        scale = _finite_float_array(
            coordinate_scale, name="coordinate_scale", ndim=1
        )
        if scale.shape != (dimension,):
            raise ValueError(
                f"coordinate_scale must have shape ({dimension},), "
                f"got {scale.shape}"
            )
        if np.any(scale <= 0.0):
            raise ValueError("coordinate_scale entries must all be > 0")
        scale = np.array(scale, copy=True)
    semantics = _string_vector(
        coordinate_semantics,
        name="coordinate_semantics",
        size=dimension,
        default="joint_rad",
    )
    units = _string_vector(
        coordinate_units,
        name="coordinate_units",
        size=dimension,
        default="rad",
    )
    unique_units = set(units)
    return _CoordinateContract(
        scale=scale,
        semantics=semantics,
        units=units,
        all_rad=unique_units == {"rad"},
        mixed_units=len(unique_units) > 1,
    )


def scaled_arc_length_progress(
    q_path: Any,
    coordinate_scale: Any,
) -> np.ndarray:
    """Return cumulative content-bound progress in scaled coordinate space.

    The returned coordinate starts at exact zero and advances by

    ``||coordinate_scale * (q[i + 1] - q[i])||_2``.

    It is deliberately independent of dense-row indices.  Every consecutive
    displacement must be non-zero so callers cannot silently hand a
    non-regular path to the scalar retimer.
    """

    path = _finite_float_array(q_path, name="q_path", ndim=2)
    if path.shape[0] < 2:
        raise ValueError("q_path needs at least two rows")
    scale = _finite_float_array(
        coordinate_scale, name="coordinate_scale", ndim=1
    )
    if scale.shape != (path.shape[1],):
        raise ValueError(
            "coordinate_scale must have shape "
            f"({path.shape[1]},), got {scale.shape}"
        )
    if np.any(scale <= 0.0):
        raise ValueError("coordinate_scale entries must all be > 0")
    increments = np.linalg.norm(
        np.diff(path, axis=0) * scale[None, :],
        axis=1,
    )
    path_scale = max(1.0, float(np.max(np.abs(path * scale[None, :]))))
    if np.any(increments <= _MIN_CONNECTOR_PARAMETER_SPAN * path_scale * 1.0e-4):
        raise ValueError(
            "q_path contains a duplicate or numerically degenerate "
            "consecutive row in scaled coordinate space"
        )
    progress = np.empty(path.shape[0], dtype=np.float64)
    progress[0] = 0.0
    progress[1:] = np.cumsum(increments, dtype=np.float64)
    if not np.all(np.isfinite(progress)) or np.any(np.diff(progress) <= 0.0):
        raise ValueError("scaled arc-length accumulation is not strictly increasing")
    return progress


def _validate_window(
    frame_count: int,
    window_start: Any,
    window_end: Any,
) -> tuple[int, int]:
    w0 = _integer(window_start, name="window_start", minimum=0)
    w1 = _integer(window_end, name="window_end", minimum=0)
    if w0 > w1:
        raise ValueError(
            f"window_start must be <= window_end, got [{w0}, {w1}]"
        )
    if w1 >= frame_count:
        raise ValueError(
            f"contact opportunity [{w0}, {w1}] exceeds source frame "
            f"range [0, {frame_count - 1}]"
        )
    return w0, w1


def _validate_core(
    frame_count: int,
    entry_frame: Any,
    exit_frame: Any,
    window_start: int,
    window_end: int,
    window_halo: Any,
) -> tuple[int, int, int]:
    entry = _integer(entry_frame, name="entry_frame", minimum=0)
    exit_ = _integer(exit_frame, name="exit_frame", minimum=0)
    halo = _integer(window_halo, name="window_halo", minimum=0)
    if entry >= frame_count or exit_ >= frame_count:
        raise ValueError(
            f"entry/exit [{entry}, {exit_}] exceed source frame range "
            f"[0, {frame_count - 1}]"
        )
    if entry >= exit_:
        raise ValueError(
            f"entry_frame must be strictly before exit_frame, got {entry}, {exit_}"
        )
    latest_entry = window_start - halo
    earliest_exit = window_end + halo
    if latest_entry < 0 or earliest_exit >= frame_count:
        raise ValueError(
            f"window_halo={halo} does not fit around contact opportunity "
            f"[{window_start}, {window_end}] in {frame_count} frames"
        )
    if entry > latest_entry:
        raise ValueError(
            f"entry_frame={entry} must be <= window_start-window_halo="
            f"{latest_entry}"
        )
    if exit_ < earliest_exit:
        raise ValueError(
            f"exit_frame={exit_} must be >= window_end+window_halo="
            f"{earliest_exit}"
        )
    return entry, exit_, halo


def _source_derivatives(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative estimates in source-frame path coordinates, never seconds."""
    if len(source) == 2:
        # A two-waypoint retained core has only one defensible tangent: its
        # secant.  Crucially, do not reach back into a prefix/suffix that the
        # caller explicitly discarded.
        secant = source[1] - source[0]
        first = np.stack((secant, secant))
        second = np.zeros_like(first)
        return first, second
    first = np.gradient(source, axis=0, edge_order=2)
    second = np.gradient(first, axis=0, edge_order=2)
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)


def _quintic_coefficients(
    segment: _Segment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalized-u polynomial coefficients for one Hermite segment."""
    p0, p1 = segment.q0, segment.q1
    d0, d1 = segment.v0 * segment.span, segment.v1 * segment.span
    dd0 = segment.a0 * segment.span**2
    dd1 = segment.a1 * segment.span**2
    c0 = p0
    c1 = d0
    c2 = 0.5 * dd0
    delta = p1 - p0
    c3 = 10.0 * delta - 6.0 * d0 - 4.0 * d1 - 1.5 * dd0 + 0.5 * dd1
    c4 = (
        -15.0 * delta
        + 8.0 * d0
        + 7.0 * d1
        + 1.5 * dd0
        - dd1
    )
    c5 = 6.0 * delta - 3.0 * d0 - 3.0 * d1 - 0.5 * dd0 + 0.5 * dd1
    coefficients = (c0, c1, c2, c3, c4, c5)
    if not all(np.all(np.isfinite(value)) for value in coefficients):
        raise ValueError("quintic coefficients became non-finite")
    return coefficients


def _power_to_bernstein(
    power_coefficients: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Convert one scalar power polynomial to same-degree Bernstein form."""

    from math import comb

    power = np.asarray(power_coefficients, dtype=np.float64)
    if power.shape != (degree + 1,):
        raise ValueError(
            f"power polynomial must have {degree + 1} coefficients, "
            f"got {power.shape}"
        )
    if not np.all(np.isfinite(power)):
        raise ValueError("power polynomial contains NaN or infinity")
    bernstein = np.empty(degree + 1, dtype=np.float64)
    for index in range(degree + 1):
        bernstein[index] = sum(
            comb(index, order) / comb(degree, order) * power[order]
            for order in range(index + 1)
        )
    if not np.all(np.isfinite(bernstein)):
        raise ValueError("power-to-Bernstein conversion became non-finite")
    return bernstein


def _split_bernstein_half(
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split scalar Bernstein controls at exact local parameter ``u=1/2``."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("Bernstein controls must be a non-empty vector")
    levels = [values.copy()]
    for _ in range(1, len(values)):
        levels.append(0.5 * (levels[-1][:-1] + levels[-1][1:]))
    left = np.asarray([level[0] for level in levels], dtype=np.float64)
    right = np.asarray(
        [level[-1] for level in reversed(levels)], dtype=np.float64
    )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Bernstein subdivision became non-finite")
    return left, right


def _segment_regularity_certificate(
    segment: _Segment,
    coordinate_scale: np.ndarray,
    *,
    segment_index: int,
) -> dict[str, Any]:
    """Prove ``||W dq/ds|| > 0`` over one full quintic segment.

    The cheap proof projects the degree-4 derivative onto the segment chord.
    Positive Bernstein controls prove strict forward progress, but failure of
    that test is not failure of the path: a physically sensible connector may
    initially continue a follow-through before curving back toward ready.

    The complete fallback forms the degree-8 polynomial
    ``||W dq/du||^2`` and recursively subdivides its Bernstein controls.
    A leaf is certified only when *every* control is above the reported
    round-off margin.  The convex-hull property then supplies an analytic lower
    bound for every point of the interval; no dense samples are used as proof.
    """

    scale = np.asarray(coordinate_scale, dtype=np.float64)
    coefficients = _quintic_coefficients(segment)
    derivative_power = np.stack(
        [
            (order + 1) * coefficients[order + 1]
            for order in range(5)
        ],
        axis=0,
    )
    weighted_derivative_power = derivative_power * scale[None, :]
    if not np.all(np.isfinite(weighted_derivative_power)):
        raise ValueError(
            f"segment {segment_index} weighted derivative became non-finite"
        )

    # This is a declared numerical-resolution contract, not a motion tuning
    # knob.  It scales with the represented coordinates so a displacement below
    # their floating-point resolution cannot receive a false regularity proof.
    weighted_positions = np.stack((segment.q0, segment.q1)) * scale[None, :]
    position_scale = max(1.0, float(np.max(np.abs(weighted_positions))))
    roundoff_factor = _REGULARITY_EPS_FACTOR * max(1, int(scale.size))
    speed_margin_u = (
        roundoff_factor
        * np.finfo(np.float64).eps
        * position_scale
    )

    weighted_chord = (segment.q1 - segment.q0) * scale
    chord_norm = float(np.linalg.norm(weighted_chord))
    projection_controls: np.ndarray | None = None
    projection_lower_bound: float | None = None
    if chord_norm > speed_margin_u:
        chord_direction = weighted_chord / chord_norm
        projection_power = weighted_derivative_power @ chord_direction
        projection_controls = _power_to_bernstein(
            projection_power, degree=4
        )
        projection_lower_bound = float(np.min(projection_controls))
        projection_roundoff_margin = (
            roundoff_factor
            * np.finfo(np.float64).eps
            * max(
                float(np.max(np.abs(projection_power))),
                float(np.max(np.abs(projection_controls))),
                speed_margin_u,
            )
        )
        projection_required_control_min = (
            projection_roundoff_margin + speed_margin_u
        )
        if projection_lower_bound > projection_required_control_min:
            conservative_projection_lower_bound = (
                projection_lower_bound - projection_roundoff_margin
            )
            return {
                "segment_index": segment_index,
                "label": segment.label,
                "path_parameter_start": segment.s0,
                "path_parameter_span": segment.span,
                "source_frame_start": segment.source_frame_start,
                "source_frame_end": segment.source_frame_end,
                "weighted_chord_norm": chord_norm,
                "certification_method": (
                    "degree4_chord_projection_bernstein_controls_v1"
                ),
                "chord_projection_control_min_per_local_u": (
                    projection_lower_bound
                ),
                "chord_projection_controls_per_local_u": (
                    projection_controls.tolist()
                ),
                "chord_projection_roundoff_margin_per_local_u": (
                    projection_roundoff_margin
                ),
                "weighted_speed_lower_bound_per_path_unit": (
                    conservative_projection_lower_bound / segment.span
                ),
                "required_weighted_speed_margin_per_path_unit": (
                    speed_margin_u / segment.span
                ),
                "roundoff_factor": roundoff_factor,
                "subdivision_max_depth": 0,
                "subdivision_leaf_count": 1,
            }

    norm_squared_power = np.zeros(9, dtype=np.float64)
    for coordinate in range(weighted_derivative_power.shape[1]):
        norm_squared_power += np.convolve(
            weighted_derivative_power[:, coordinate],
            weighted_derivative_power[:, coordinate],
        )
    if not np.all(np.isfinite(norm_squared_power)):
        raise ValueError(
            f"segment {segment_index} weighted speed polynomial became non-finite"
        )
    norm_squared_controls = _power_to_bernstein(
        norm_squared_power, degree=8
    )
    polynomial_scale = max(
        float(np.max(np.abs(norm_squared_power))),
        float(np.max(np.abs(norm_squared_controls))),
        speed_margin_u**2,
    )
    squared_roundoff_margin = (
        roundoff_factor
        * np.finfo(np.float64).eps
        * polynomial_scale
    )
    required_squared_control_min = (
        speed_margin_u**2 + squared_roundoff_margin
    )

    pending: list[tuple[np.ndarray, int]] = [
        (norm_squared_controls, 0)
    ]
    certified_lower_bounds: list[float] = []
    maximum_depth_used = 0
    leaf_count = 0
    while pending:
        controls, depth = pending.pop()
        maximum_depth_used = max(maximum_depth_used, depth)
        lower_bound = float(np.min(controls))
        if lower_bound > required_squared_control_min:
            certified_lower_bounds.append(
                lower_bound - squared_roundoff_margin
            )
            leaf_count += 1
            continue
        if depth >= _REGULARITY_MAX_SUBDIVISION_DEPTH:
            projection_text = (
                "unavailable"
                if projection_lower_bound is None
                else f"{projection_lower_bound:.17g}"
            )
            raise ValueError(
                "path regularity could not be certified for quintic segment "
                f"{segment_index} ({segment.label}); degree-4 chord projection "
                f"lower bound={projection_text}, degree-8 weighted-speed-squared "
                f"Bernstein lower bound={lower_bound:.17g}, required>"
                f"{required_squared_control_min:.17g} after {depth} "
                "subdivisions. "
                "Refusing a possible cusp or numerically unresolved path."
            )
        left, right = _split_bernstein_half(controls)
        # Reverse push order so traversal and receipt are deterministic.
        pending.append((right, depth + 1))
        pending.append((left, depth + 1))

    squared_lower_bound = min(certified_lower_bounds)
    speed_lower_bound_u = float(np.sqrt(squared_lower_bound))
    if not np.isfinite(speed_lower_bound_u) or not (
        speed_lower_bound_u > speed_margin_u
    ):
        raise ValueError(
            f"segment {segment_index} regularity lower bound is unresolved"
        )
    return {
        "segment_index": segment_index,
        "label": segment.label,
        "path_parameter_start": segment.s0,
        "path_parameter_span": segment.span,
        "source_frame_start": segment.source_frame_start,
        "source_frame_end": segment.source_frame_end,
        "weighted_chord_norm": chord_norm,
        "certification_method": (
            "degree8_weighted_speed_squared_bernstein_subdivision_v1"
        ),
        "chord_projection_control_min_per_local_u": (
            projection_lower_bound
        ),
        "chord_projection_controls_per_local_u": (
            None
            if projection_controls is None
            else projection_controls.tolist()
        ),
        "weighted_speed_squared_control_min_per_local_u2": (
            squared_lower_bound
        ),
        "weighted_speed_squared_required_margin_per_local_u2": (
            required_squared_control_min
        ),
        "weighted_speed_lower_bound_per_path_unit": (
            speed_lower_bound_u / segment.span
        ),
        "required_weighted_speed_margin_per_path_unit": (
            speed_margin_u / segment.span
        ),
        "roundoff_factor": roundoff_factor,
        "subdivision_max_depth": maximum_depth_used,
        "subdivision_leaf_count": leaf_count,
    }


def _path_regularity_report(
    segments: list[_Segment],
    coordinate_scale: np.ndarray,
) -> dict[str, Any]:
    """Return a fail-closed analytic regularity certificate for all segments."""

    certificates = [
        _segment_regularity_certificate(
            segment,
            coordinate_scale,
            segment_index=index,
        )
        for index, segment in enumerate(segments)
    ]
    lower_bound = min(
        certificate["weighted_speed_lower_bound_per_path_unit"]
        for certificate in certificates
    )
    required_margin = max(
        certificate["required_weighted_speed_margin_per_path_unit"]
        for certificate in certificates
    )
    minimum_margin_ratio = min(
        certificate["weighted_speed_lower_bound_per_path_unit"]
        / certificate["required_weighted_speed_margin_per_path_unit"]
        for certificate in certificates
    )
    return {
        "schema_version": 1,
        "regular": True,
        "proof_domain": "every_original_quintic_segment_full_interval",
        "dense_samples_used_as_proof": False,
        "metric": "l2_norm(coordinate_scale * dq_ds)",
        "minimum_weighted_speed_lower_bound_per_path_unit": lower_bound,
        "maximum_required_weighted_speed_margin_per_path_unit": (
            required_margin
        ),
        "minimum_certified_speed_to_required_margin_ratio": (
            minimum_margin_ratio
        ),
        "roundoff_margin_formula": (
            "4096 * coordinate_dimension * float64_epsilon * "
            "represented_coordinate_scale; "
            "squared-norm proof additionally covers polynomial conversion "
            "roundoff"
        ),
        "maximum_bernstein_subdivision_depth": (
            _REGULARITY_MAX_SUBDIVISION_DEPTH
        ),
        "segments": certificates,
    }


def _polyval_ascending(coefficients: list[float], x: np.ndarray) -> np.ndarray:
    value = np.zeros_like(x, dtype=np.float64)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _segment_total_variation(
    segment: _Segment,
    coordinate_scale: np.ndarray,
) -> float:
    """Maximum scaled-coordinate analytic total variation over u in [0, 1].

    Endpoint chord alone is insufficient: equal endpoints can hide a sizeable
    Hermite lobe.  A quintic's turning points are the real roots of its quartic
    derivative, so evaluating endpoints plus those roots gives exact
    one-dimensional total variation (up to root-solver roundoff).
    """
    coefficients = _quintic_coefficients(segment)
    max_variation = 0.0
    for joint in range(len(segment.q0)):
        derivative_descending = np.asarray(
            [
                5.0 * coefficients[5][joint],
                4.0 * coefficients[4][joint],
                3.0 * coefficients[3][joint],
                2.0 * coefficients[2][joint],
                coefficients[1][joint],
            ],
            dtype=np.float64,
        )
        nonzero = np.flatnonzero(np.abs(derivative_descending) > 1e-14)
        roots_in_unit: list[float] = []
        if len(nonzero):
            roots = np.roots(derivative_descending[nonzero[0] :])
            for root in roots:
                if abs(float(root.imag)) <= 1e-9 * (1.0 + abs(float(root.real))):
                    real = float(root.real)
                    if 0.0 < real < 1.0:
                        roots_in_unit.append(real)
        knots = np.asarray([0.0, *sorted(set(roots_in_unit)), 1.0])
        values = _polyval_ascending(
            [float(coefficient[joint]) for coefficient in coefficients], knots
        )
        variation = float(np.sum(np.abs(np.diff(values))))
        if not np.isfinite(variation):
            raise ValueError("quintic analytic excursion became non-finite")
        max_variation = max(
            max_variation, variation * float(coordinate_scale[joint])
        )
    return max_variation


def _with_sampling(
    segment: _Segment,
    *,
    samples_per_rad: float,
    minimum: int,
    coordinate_scale: np.ndarray,
) -> _Segment:
    variation = _segment_total_variation(segment, coordinate_scale)
    intervals = max(minimum, int(np.ceil(variation * samples_per_rad)))
    return replace(segment, sample_intervals=intervals)


def _quintic_eval(
    segment: _Segment,
    local_parameter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate position and first/second derivatives in global path ``s``."""
    u = np.asarray(local_parameter, dtype=np.float64) / segment.span
    if np.any(u < -1e-12) or np.any(u > 1.0 + 1e-12):
        raise ValueError("quintic evaluation left its segment domain")
    u = np.clip(u, 0.0, 1.0)

    p0, p1 = segment.q0, segment.q1
    d0, d1 = segment.v0 * segment.span, segment.v1 * segment.span
    dd0 = segment.a0 * segment.span**2
    dd1 = segment.a1 * segment.span**2
    c0, c1, c2, c3, c4, c5 = _quintic_coefficients(segment)

    column_u = u[:, None]
    q = (
        c0
        + c1 * column_u
        + c2 * column_u**2
        + c3 * column_u**3
        + c4 * column_u**4
        + c5 * column_u**5
    )
    dq_du = (
        c1
        + 2.0 * c2 * column_u
        + 3.0 * c3 * column_u**2
        + 4.0 * c4 * column_u**3
        + 5.0 * c5 * column_u**4
    )
    d2q_du2 = (
        2.0 * c2
        + 6.0 * c3 * column_u
        + 12.0 * c4 * column_u**2
        + 20.0 * c5 * column_u**3
    )
    # Polynomial cancellation at u=0/1 can leave float dust (for example
    # -1.38e-17 instead of an exact recorded zero).  Endpoint rows are provenance
    # anchors, so put the declared boundary states back exactly.
    at_start = u == 0.0
    at_end = u == 1.0
    q[at_start], dq_du[at_start], d2q_du2[at_start] = p0, d0, dd0
    q[at_end], dq_du[at_end], d2q_du2[at_end] = p1, d1, dd1
    return q, dq_du / segment.span, d2q_du2 / segment.span**2


def _segment_endpoint_state(
    segment: _Segment,
    *,
    at_end: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.asarray([segment.span if at_end else 0.0])
    q, dq, ddq = _quintic_eval(segment, local)
    return q[0], dq[0], ddq[0]


def _max_abs_native(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def _max_abs_scaled(
    value: np.ndarray,
    coordinate_scale: np.ndarray,
) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(array * coordinate_scale)))


def _l2_scaled(
    value: np.ndarray,
    coordinate_scale: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(np.asarray(value, dtype=np.float64) * coordinate_scale)
    )


def _continuity_report(
    segments: list[_Segment],
    *,
    ready: np.ndarray,
    source: np.ndarray,
    entry: int,
    exit_: int,
    source_frame_map: np.ndarray,
    q_path: np.ndarray,
    source_waypoint_path_indices: np.ndarray,
    canonical_knot_path_indices: np.ndarray,
    samples_per_rad: float,
    connector_parameter_span: float,
    coordinate_contract: _CoordinateContract,
    path_regularity: dict[str, Any],
) -> dict[str, Any]:
    scale = coordinate_contract.scale
    join_position_scaled: list[float] = []
    join_first_scaled: list[float] = []
    join_second_scaled: list[float] = []
    join_position_native: list[float] = []
    join_first_native: list[float] = []
    join_second_native: list[float] = []
    for left, right in zip(segments[:-1], segments[1:]):
        lq, ldq, lddq = _segment_endpoint_state(left, at_end=True)
        rq, rdq, rddq = _segment_endpoint_state(right, at_end=False)
        join_position_scaled.append(_max_abs_scaled(lq - rq, scale))
        join_first_scaled.append(_max_abs_scaled(ldq - rdq, scale))
        join_second_scaled.append(_max_abs_scaled(lddq - rddq, scale))
        join_position_native.append(_max_abs_native(lq - rq))
        join_first_native.append(_max_abs_native(ldq - rdq))
        join_second_native.append(_max_abs_native(lddq - rddq))

    start_q, start_dq, start_ddq = _segment_endpoint_state(
        segments[0], at_end=False
    )
    end_q, end_dq, end_ddq = _segment_endpoint_state(segments[-1], at_end=True)
    expected_source = source[entry : exit_ + 1]
    actual_source = q_path[source_waypoint_path_indices]
    waypoint_delta = actual_source - expected_source
    waypoint_error_scaled = _max_abs_scaled(waypoint_delta, scale)
    waypoint_error_native = _max_abs_native(waypoint_delta)

    tolerance = 5e-10
    maxima = {
        "position_max_abs_scaled": max(join_position_scaled, default=0.0),
        "first_derivative_max_abs_scaled_per_path_unit": max(
            join_first_scaled, default=0.0
        ),
        "second_derivative_max_abs_scaled_per_path_unit2": max(
            join_second_scaled, default=0.0
        ),
    }
    endpoints = {
        "start_position_max_abs_scaled": _max_abs_scaled(
            start_q - ready, scale
        ),
        "end_position_max_abs_scaled": _max_abs_scaled(end_q - ready, scale),
        "start_first_derivative_max_abs_scaled_per_path_unit": _max_abs_scaled(
            start_dq, scale
        ),
        "end_first_derivative_max_abs_scaled_per_path_unit": _max_abs_scaled(
            end_dq, scale
        ),
        "start_second_derivative_max_abs_scaled_per_path_unit2": _max_abs_scaled(
            start_ddq, scale
        ),
        "end_second_derivative_max_abs_scaled_per_path_unit2": _max_abs_scaled(
            end_ddq, scale
        ),
        "ready_path_tangent_contract": (
            "native_chord_divided_by_connector_parameter_span"
        ),
        "path_derivatives_are_physical_state": False,
        "physical_endpoint_velocity_claimed": False,
        "physical_endpoint_acceleration_claimed": False,
        "physical_endpoint_state_assigned_by": "later_time_law",
    }
    if coordinate_contract.all_rad:
        maxima.update(
            {
                "position_max_abs_rad": max(
                    join_position_native, default=0.0
                ),
                "first_derivative_max_abs_rad_per_path_unit": max(
                    join_first_native, default=0.0
                ),
                "second_derivative_max_abs_rad_per_path_unit2": max(
                    join_second_native, default=0.0
                ),
            }
        )
        endpoints.update(
            {
                "start_position_max_abs_rad": _max_abs_native(start_q - ready),
                "end_position_max_abs_rad": _max_abs_native(end_q - ready),
                "start_first_derivative_max_abs_rad_per_path_unit": (
                    _max_abs_native(start_dq)
                ),
                "end_first_derivative_max_abs_rad_per_path_unit": (
                    _max_abs_native(end_dq)
                ),
                "start_second_derivative_max_abs_rad_per_path_unit2": (
                    _max_abs_native(start_ddq)
                ),
                "end_second_derivative_max_abs_rad_per_path_unit2": (
                    _max_abs_native(end_ddq)
                ),
            }
        )
    # Endpoint path-derivative magnitudes are deliberate declarations, not
    # continuity residuals.  Only mismatches enter this C2 residual set.
    continuity_residuals = [
        maxima["position_max_abs_scaled"],
        maxima["first_derivative_max_abs_scaled_per_path_unit"],
        maxima["second_derivative_max_abs_scaled_per_path_unit2"],
        endpoints["start_position_max_abs_scaled"],
        endpoints["end_position_max_abs_scaled"],
        waypoint_error_scaled,
    ]
    endpoint_measurements = [
        endpoints["start_first_derivative_max_abs_scaled_per_path_unit"],
        endpoints["end_first_derivative_max_abs_scaled_per_path_unit"],
        endpoints["start_second_derivative_max_abs_scaled_per_path_unit2"],
        endpoints["end_second_derivative_max_abs_scaled_per_path_unit2"],
    ]
    if not np.all(
        np.isfinite(
            np.asarray(
                [*continuity_residuals, *endpoint_measurements],
                dtype=np.float64,
            )
        )
    ):
        raise ValueError("continuity residuals became non-finite")
    parameterization = {
        "name": "dimensionless_geometric_path_parameter",
        "is_time_parameter": False,
        "sample_count_has_time_meaning": False,
        "source_frame_interval_span": 1.0,
        "connector_parameter_span": connector_parameter_span,
        "samples_per_scaled_coordinate_unit": samples_per_rad,
        "connector_rows_are_source_time_bytes": False,
        "source_waypoints_are_exact": True,
        "path_regularity_required": True,
        "physical_endpoint_hold_encoded_here": False,
    }
    if coordinate_contract.all_rad:
        parameterization["samples_per_rad"] = samples_per_rad
    source_waypoints = {
        "count": int(len(source_waypoint_path_indices)),
        "max_abs_error_scaled": waypoint_error_scaled,
        "mapped_frame_min": float(
            np.min(source_frame_map[np.isfinite(source_frame_map)])
        ),
        "mapped_frame_max": float(
            np.max(source_frame_map[np.isfinite(source_frame_map)])
        ),
    }
    if coordinate_contract.all_rad:
        source_waypoints["max_abs_error_rad"] = waypoint_error_native
    canonical_map = np.ascontiguousarray(
        np.asarray(source_frame_map, dtype="<f8")
    ).copy()
    canonical_map.view("<u8")[np.isnan(canonical_map)] = _CANONICAL_QNAN_BITS
    if np.any(np.isinf(canonical_map)):
        raise ValueError("source_frame_map may contain only finite values or NaN")
    source_frame_map_receipt = {
        "schema_version": 1,
        "encoding": SOURCE_FRAME_MAP_ENCODING,
        "length": int(len(canonical_map)),
        "sha256": hashlib.sha256(canonical_map.tobytes(order="C")).hexdigest(),
        "finite_count": int(np.count_nonzero(np.isfinite(canonical_map))),
        "nan_count": int(np.count_nonzero(np.isnan(canonical_map))),
        "finite_value_encoding": (
            "piecewise_linear_source_frame_by_segment_v1"
        ),
        "segment_intervals": [
            {
                "kind": (
                    "pre_connector"
                    if index == 0
                    else (
                        "post_connector"
                        if index == len(segments) - 1
                        else "source_core"
                    )
                ),
                "sample_intervals": int(segment.sample_intervals),
                "source_frame_start": segment.source_frame_start,
                "source_frame_end": segment.source_frame_end,
            }
            for index, segment in enumerate(segments)
        ],
        "source_waypoint_path_indices": (
            np.asarray(source_waypoint_path_indices, dtype=np.int64).tolist()
        ),
        "canonical_knot_path_indices": (
            np.asarray(canonical_knot_path_indices, dtype=np.int64).tolist()
        ),
    }
    return {
        "coordinate_contract": {
            "dimension": int(len(scale)),
            "scale": scale.tolist(),
            "semantics": list(coordinate_contract.semantics),
            "units": list(coordinate_contract.units),
            "unique_units": sorted(set(coordinate_contract.units)),
            "mixed_units": coordinate_contract.mixed_units,
            "physical_coordinates_modified": False,
            "distance_formula": "native_coordinate_delta * coordinate_scale",
            "legacy_rad_metrics_published": coordinate_contract.all_rad,
        },
        "parameterization": parameterization,
        "joins": {
            "count": len(segments) - 1,
            **maxima,
        },
        "endpoints": endpoints,
        "path_regularity": path_regularity,
        "canonical_knots": {
            "composition": (
                "canonical_ready_start + retained_source_waypoints + "
                "canonical_ready_end"
            ),
            "count": int(len(canonical_knot_path_indices)),
            "path_indices": canonical_knot_path_indices.tolist(),
            "q_and_path_jets_independent_of_visualization_sampling_density": (
                True
            ),
        },
        "limit_claims": {
            "exact_joint_extrema_computed": False,
            "external_position_limits_checked": False,
            "position_limit_compliance_claimed": False,
            "assigned_to": "downstream_compiler_with_explicit_limits",
        },
        "source_waypoints": source_waypoints,
        "source_frame_map_receipt": source_frame_map_receipt,
        "scaled_coordinate_tolerance": tolerance,
        **({"tolerance": tolerance} if coordinate_contract.all_rad else {}),
        "c2_continuous": bool(
            max(continuity_residuals, default=0.0) <= tolerance
        ),
    }


def build_canonical_geometry(
    source_q: Any,
    ready_q: Any,
    entry_frame: Any,
    exit_frame: Any,
    window_start: Any,
    window_end: Any,
    *,
    window_halo: int = 2,
    samples_per_rad: float = 24.0,
    min_connector_intervals: int = 8,
    min_core_intervals: int = _MIN_QUINTIC_INTERVALS,
    connector_parameter_span: float = 1.0,
    coordinate_scale: Any = None,
    coordinate_semantics: Any = "joint_rad",
    coordinate_units: Any = "rad",
) -> CanonicalMotionGeometry:
    """Build a C2 ready->source-core->ready geometric loop.

    Contact opportunity bounds and source frame bounds are inclusive.  The
    retained core must include ``window_halo`` source frames on both sides of
    that opportunity so downstream derivative/FK gates have real neighbours.

    ``samples_per_rad`` and the two minimum interval counts affect only geometric
    sampling resolution.  They do not specify frames per second or motion time.
    The legacy ``samples_per_rad`` name means samples per *scaled coordinate
    unit* when mixed coordinates are supplied.  ``coordinate_scale`` affects
    sampling/report metrics only; returned physical coordinates are untouched.
    """

    source, ready = _validate_source_ready(source_q, ready_q)
    coordinate_contract = _coordinate_contract(
        source.shape[1],
        coordinate_scale=coordinate_scale,
        coordinate_semantics=coordinate_semantics,
        coordinate_units=coordinate_units,
    )
    w0, w1 = _validate_window(len(source), window_start, window_end)
    entry, exit_, halo = _validate_core(
        len(source), entry_frame, exit_frame, w0, w1, window_halo
    )
    density = _positive_float(samples_per_rad, name="samples_per_rad")
    min_connector = _integer(
        min_connector_intervals,
        name="min_connector_intervals",
        minimum=_MIN_QUINTIC_INTERVALS,
    )
    min_core = _integer(
        min_core_intervals,
        name="min_core_intervals",
        minimum=_MIN_QUINTIC_INTERVALS,
    )
    connector_span = _positive_float(
        connector_parameter_span, name="connector_parameter_span"
    )
    if connector_span < _MIN_CONNECTOR_PARAMETER_SPAN:
        raise ValueError(
            "connector_parameter_span must be >= "
            f"{_MIN_CONNECTOR_PARAMETER_SPAN:g}; smaller values are numerically "
            "ill-conditioned and have no time-law meaning"
        )

    retained_source = source[entry : exit_ + 1]
    source_first, source_second = _source_derivatives(retained_source)
    zeros = np.zeros(source.shape[1], dtype=np.float64)
    # The weighted chord is ``W (q1-q0)``.  Mapping its native-coordinate
    # tangent back through positive diagonal W gives the deterministic
    # ``(q1-q0)/span`` declaration below.  The explicit span division is
    # essential: these are dq/ds values, not normalized-u derivatives.
    ready_start_tangent = (source[entry] - ready) / connector_span
    ready_end_tangent = (ready - source[exit_]) / connector_span

    pre_segment = _with_sampling(
        _Segment(
            q0=ready,
            v0=ready_start_tangent,
            a0=zeros,
            q1=source[entry],
            v1=source_first[0],
            a1=source_second[0],
            s0=0.0,
            span=connector_span,
            sample_intervals=0,
            label=_LABEL_PRE,
            source_frame_start=None,
            source_frame_end=float(entry),
        ),
        samples_per_rad=density,
        minimum=min_connector,
        coordinate_scale=coordinate_contract.scale,
    )
    segments: list[_Segment] = [pre_segment]
    core_parameter_start = connector_span
    for frame in range(entry, exit_):
        local_frame = frame - entry
        segments.append(
            _with_sampling(
                _Segment(
                    q0=source[frame],
                    v0=source_first[local_frame],
                    a0=source_second[local_frame],
                    q1=source[frame + 1],
                    v1=source_first[local_frame + 1],
                    a1=source_second[local_frame + 1],
                    s0=core_parameter_start + local_frame,
                    span=1.0,
                    sample_intervals=0,
                    label=_LABEL_CORE,
                    source_frame_start=float(frame),
                    source_frame_end=float(frame + 1),
                ),
                samples_per_rad=density,
                minimum=min_core,
                coordinate_scale=coordinate_contract.scale,
            )
        )
    post_parameter_start = core_parameter_start + (exit_ - entry)
    post_segment = _with_sampling(
        _Segment(
            q0=source[exit_],
            v0=source_first[-1],
            a0=source_second[-1],
            q1=ready,
            v1=ready_end_tangent,
            a1=zeros,
            s0=post_parameter_start,
            span=connector_span,
            sample_intervals=0,
            label=_LABEL_POST,
            source_frame_start=float(exit_),
            source_frame_end=None,
        ),
        samples_per_rad=density,
        minimum=min_connector,
        coordinate_scale=coordinate_contract.scale,
    )
    segments.append(post_segment)
    path_regularity = _path_regularity_report(
        segments,
        coordinate_contract.scale,
    )
    pre_intervals = pre_segment.sample_intervals
    post_intervals = post_segment.sample_intervals

    q_parts: list[np.ndarray] = []
    dq_parts: list[np.ndarray] = []
    ddq_parts: list[np.ndarray] = []
    parameter_parts: list[np.ndarray] = []
    map_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for segment_index, segment in enumerate(segments):
        local = np.linspace(
            0.0,
            segment.span,
            segment.sample_intervals + 1,
            dtype=np.float64,
        )
        q, dq, ddq = _quintic_eval(segment, local)
        path_parameter = segment.s0 + local
        if (
            segment.source_frame_start is not None
            and segment.source_frame_end is not None
        ):
            source_map = np.linspace(
                segment.source_frame_start,
                segment.source_frame_end,
                segment.sample_intervals + 1,
                dtype=np.float64,
            )
        else:
            source_map = np.full(segment.sample_intervals + 1, np.nan)
            if segment.source_frame_end is not None:
                source_map[-1] = segment.source_frame_end
            if segment.source_frame_start is not None:
                source_map[0] = segment.source_frame_start
        labels = np.full(
            segment.sample_intervals + 1, segment.label, dtype="<U24"
        )
        if segment_index:
            q, dq, ddq = q[1:], dq[1:], ddq[1:]
            path_parameter, source_map, labels = (
                path_parameter[1:],
                source_map[1:],
                labels[1:],
            )
        q_parts.append(q)
        dq_parts.append(dq)
        ddq_parts.append(ddq)
        parameter_parts.append(path_parameter)
        map_parts.append(source_map)
        label_parts.append(labels)

    q_path = np.concatenate(q_parts, axis=0)
    dq_ds = np.concatenate(dq_parts, axis=0)
    d2q_ds2 = np.concatenate(ddq_parts, axis=0)
    path_parameter = np.concatenate(parameter_parts)
    source_frame_map = np.concatenate(map_parts)
    segment_labels = np.concatenate(label_parts)
    for name, value in (
        ("q_path", q_path),
        ("dq_ds", dq_ds),
        ("d2q_ds2", d2q_ds2),
        ("path_parameter", path_parameter),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} became non-finite")
    if not np.all(np.diff(path_parameter) > 0.0):
        raise ValueError("path_parameter must remain finite and strictly increasing")
    duplicate_steps = np.all(q_path[1:] == q_path[:-1], axis=1)
    if np.any(duplicate_steps):
        index = int(np.flatnonzero(duplicate_steps)[0])
        raise ValueError(
            "zero-length adjacent geometry at path rows "
            f"{index}/{index + 1}; stationary holds belong in the later time law"
        )

    finite_map = np.isfinite(source_frame_map)
    window_mask = (
        finite_map
        & (source_frame_map >= float(w0) - 1e-12)
        & (source_frame_map <= float(w1) + 1e-12)
    )
    if not np.any(window_mask):
        raise RuntimeError("internal error: contact opportunity disappeared")
    window_indices = np.flatnonzero(window_mask)
    window_path_start = int(window_indices[0])
    window_path_end = int(window_indices[-1])

    segment_labels[finite_map] = _LABEL_CORE
    segment_labels[window_mask] = _LABEL_WINDOW
    segment_labels[0] = _LABEL_READY_START
    segment_labels[-1] = _LABEL_READY_END

    waypoint_indices: list[int] = []
    for frame in range(entry, exit_ + 1):
        exact = np.flatnonzero(
            finite_map
            & np.isclose(
                source_frame_map, float(frame), atol=1e-12, rtol=0.0
            )
        )
        if len(exact) != 1:
            raise RuntimeError(
                f"internal error: source frame {frame} maps to {len(exact)} rows"
            )
        waypoint_indices.append(int(exact[0]))
    source_waypoint_path_indices = np.asarray(waypoint_indices, dtype=np.int64)
    canonical_knot_path_indices = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            source_waypoint_path_indices,
            np.asarray([len(q_path) - 1], dtype=np.int64),
        )
    )
    if len(np.unique(canonical_knot_path_indices)) != len(
        canonical_knot_path_indices
    ):
        raise RuntimeError("internal error: canonical knot indices are not unique")
    if np.any(np.diff(canonical_knot_path_indices) <= 0):
        raise RuntimeError(
            "internal error: canonical knot indices are not strictly increasing"
        )

    report = _continuity_report(
        segments,
        ready=ready,
        source=source,
        entry=entry,
        exit_=exit_,
        source_frame_map=source_frame_map,
        q_path=q_path,
        source_waypoint_path_indices=source_waypoint_path_indices,
        canonical_knot_path_indices=canonical_knot_path_indices,
        samples_per_rad=density,
        connector_parameter_span=connector_span,
        coordinate_contract=coordinate_contract,
        path_regularity=path_regularity,
    )
    report["selection"] = {
        "entry_frame": entry,
        "exit_frame": exit_,
        "window_start": w0,
        "window_end": w1,
        "window_halo": halo,
        "old_source_frame_zero_retained": entry == 0,
    }
    report["sampling"] = {
        "path_points": int(len(q_path)),
        "pre_connector_intervals": pre_intervals,
        "post_connector_intervals": post_intervals,
        "min_core_intervals": min_core,
    }

    if not report["c2_continuous"]:
        raise RuntimeError(
            "internal C2 construction failed; refusing to return a broken path"
        )
    return CanonicalMotionGeometry(
        q_path=q_path,
        path_parameter=path_parameter,
        dq_ds=dq_ds,
        d2q_ds2=d2q_ds2,
        source_frame_map=source_frame_map,
        segment_labels=segment_labels,
        window_mask=window_mask,
        window_path_start=window_path_start,
        window_path_end=window_path_end,
        source_waypoint_path_indices=source_waypoint_path_indices,
        canonical_knot_path_indices=canonical_knot_path_indices,
        entry_frame=entry,
        exit_frame=exit_,
        window_start=w0,
        window_end=w1,
        continuity_report=report,
    )


def iter_entry_exit_candidates(
    source_q: Any,
    ready_q: Any,
    window_start: Any,
    window_end: Any,
    *,
    window_halo: int = 2,
    coordinate_scale: Any = None,
    coordinate_semantics: Any = "joint_rad",
    coordinate_units: Any = "rad",
) -> Iterator[EntryExitCandidate]:
    """Yield every entry/exit pair that retains the requested window halo.

    No 2/3 cut, ranking, or acceptance decision is hidden here:

    ``entry = 0 .. window_start-window_halo``
    ``exit  = window_end+window_halo .. T-1``
    """

    source, ready = _validate_source_ready(source_q, ready_q)
    coordinate_contract = _coordinate_contract(
        source.shape[1],
        coordinate_scale=coordinate_scale,
        coordinate_semantics=coordinate_semantics,
        coordinate_units=coordinate_units,
    )
    w0, w1 = _validate_window(len(source), window_start, window_end)
    halo = _integer(window_halo, name="window_halo", minimum=0)
    latest_entry = w0 - halo
    earliest_exit = w1 + halo
    if latest_entry < 0 or earliest_exit >= len(source):
        raise ValueError(
            f"window_halo={halo} leaves no legal entry/exit candidates around "
            f"[{w0}, {w1}] in {len(source)} frames"
        )

    scaled_step_l2 = np.linalg.norm(
        np.diff(source, axis=0) * coordinate_contract.scale, axis=1
    )
    cumulative_scaled_l2 = np.concatenate(
        ([0.0], np.cumsum(scaled_step_l2))
    )
    native_step_l2 = np.linalg.norm(np.diff(source, axis=0), axis=1)
    cumulative_native_l2 = np.concatenate(([0.0], np.cumsum(native_step_l2)))

    for entry in range(latest_entry + 1):
        pre_delta = source[entry] - ready
        for exit_ in range(max(earliest_exit, entry + 1), len(source)):
            post_delta = ready - source[exit_]
            legacy_rad = coordinate_contract.all_rad
            yield EntryExitCandidate(
                entry_frame=entry,
                exit_frame=exit_,
                core_frame_count=exit_ - entry + 1,
                entry_to_window_frames=w0 - entry,
                window_to_exit_frames=exit_ - w1,
                skipped_prefix_frames=entry,
                skipped_suffix_frames=len(source) - 1 - exit_,
                includes_source_frame_zero=entry == 0,
                ready_to_entry_max_abs_scaled=_max_abs_scaled(
                    pre_delta, coordinate_contract.scale
                ),
                exit_to_ready_max_abs_scaled=_max_abs_scaled(
                    post_delta, coordinate_contract.scale
                ),
                ready_to_entry_l2_scaled=_l2_scaled(
                    pre_delta, coordinate_contract.scale
                ),
                exit_to_ready_l2_scaled=_l2_scaled(
                    post_delta, coordinate_contract.scale
                ),
                core_arc_length_l2_scaled=float(
                    cumulative_scaled_l2[exit_]
                    - cumulative_scaled_l2[entry]
                ),
                coordinate_units_mixed=coordinate_contract.mixed_units,
                ready_to_entry_max_abs_rad=(
                    _max_abs_native(pre_delta) if legacy_rad else None
                ),
                exit_to_ready_max_abs_rad=(
                    _max_abs_native(post_delta) if legacy_rad else None
                ),
                ready_to_entry_l2_rad=(
                    float(np.linalg.norm(pre_delta)) if legacy_rad else None
                ),
                exit_to_ready_l2_rad=(
                    float(np.linalg.norm(post_delta)) if legacy_rad else None
                ),
                core_arc_length_l2_rad=(
                    float(
                        cumulative_native_l2[exit_]
                        - cumulative_native_l2[entry]
                    )
                    if legacy_rad
                    else None
                ),
            )


def enumerate_entry_exit_candidates(
    source_q: Any,
    ready_q: Any,
    window_start: Any,
    window_end: Any,
    **kwargs: Any,
) -> list[EntryExitCandidate]:
    """Materialize :func:`iter_entry_exit_candidates` as a deterministic list."""

    return list(
        iter_entry_exit_candidates(
            source_q,
            ready_q,
            window_start,
            window_end,
            **kwargs,
        )
    )


# The longer alias makes call sites self-documenting while retaining the compact
# public name used in the motion compiler design notes.
build_canonical_ready_path = build_canonical_geometry


__all__ = [
    "CanonicalMotionGeometry",
    "EntryExitCandidate",
    "SOURCE_FRAME_MAP_ENCODING",
    "build_canonical_geometry",
    "build_canonical_ready_path",
    "enumerate_entry_exit_candidates",
    "iter_entry_exit_candidates",
    "scaled_arc_length_progress",
]
