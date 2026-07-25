#!/usr/bin/env python3
"""Pure-NumPy scalar-path retiming for canonical motion paths.

This module deliberately retimes *geometry*, not source frame bytes.  A smooth
joint-space path ``q(s)`` is kept as the geometric prior while a forward/backward
speed pass finds a rest-to-rest time law.  Markers (including strike-window
markers) are observations on that time law; they do not pin samples or forbid
acceleration inside a window.

The implementation is intentionally smaller and more conservative than a full
robot-dynamics TOPP solver:

* generic position-only samples use a shape-preserving C1 PCHIP-Hermite path;
* canonical callers may instead provide a complete endpoint 2-jet, reconstructed
  as a knot-verified C2 quintic Hermite path;
* velocity and joint-acceleration limits are projected onto a scalar path;
* a forward/backward bang-bang envelope is solved on a dense path grid;
* the result is sampled on a uniform output grid (50 Hz by default);
* position bounds are checked at every exact cubic extremum;
* velocity and acceleration are checked at every exact polynomial extremum of
  every scalar cell, in addition to output and finite-difference checks; and
* the entire time law is slowed iteratively if validation exposes a violation.

No torque, collision, balance, contact, or MuJoCo constraint is implied.  Those
remain downstream gates.  This module fails closed on malformed/non-regular
paths and on any limit validation that does not converge.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from canonical_weighted_arc_path import (
    ALGORITHM_ID as WEIGHTED_ARC_ALGORITHM_ID,
    WeightedArcPath,
    WeightedArcPathError,
    _power_intervals_to_bernstein as _wa_power_intervals_to_bernstein,
    _weighted_speed_squared_power_intervals as _wa_weighted_speed_squared_power_intervals,
)


class RetimeError(ValueError):
    """The supplied path is invalid or could not be retimed safely."""


class _ControlGuardConvergenceError(RetimeError):
    """One fixed grid could not construct its finite 50 Hz guard."""


@dataclass(frozen=True)
class MarkerMapping:
    """Where one source-path marker lands on the uniform output timeline."""

    source_index: float
    time_s: float
    output_fractional_frame: float
    output_frame: int
    path_position_at_frame: float


@dataclass(frozen=True)
class ScalarPathCollocationTrace:
    """Immutable exact state retained from one accepted scalar-path solve.

    Node/midpoint arrays are evaluated directly on the accepted collocation
    grid.  Legacy PCHIP geometry is C1 rather than C2, while an explicit
    endpoint-2-jet path is required to be C2 at every internal knot.  Both
    ``q_ss_node_left[i] = lim(s->s_i-) q_ss`` and
    ``q_ss_node_right[i] = lim(s->s_i+) q_ss`` are retained in either case so
    the evidence format never hides a one-sided mismatch.  Scalar cell ``i``
    must use ``q_ss_node_right[i]`` at its start and
    ``q_ss_node_left[i + 1]`` at its end.  Tick arrays come from the same
    analytic scalar sampler used to build the returned motion; they are not
    reconstructed from schema-2 bytes or finite differences.
    ``tick_cell_side`` records which one-sided scalar acceleration is
    represented when a control tick lands exactly on a knot.
    """

    s_node: np.ndarray
    s_mid: np.ndarray
    q_node: np.ndarray
    q_s_node: np.ndarray
    q_ss_node_left: np.ndarray
    q_ss_node_right: np.ndarray
    q_mid: np.ndarray
    q_s_mid: np.ndarray
    q_ss_mid: np.ndarray
    x_node: np.ndarray
    u_cell: np.ndarray
    time_node_s: np.ndarray
    time_mid_s: np.ndarray
    tick_time_s: np.ndarray
    tick_s: np.ndarray
    tick_x: np.ndarray
    tick_q: np.ndarray
    tick_q_s: np.ndarray
    tick_q_ss: np.ndarray
    tick_qdot: np.ndarray
    tick_qdd: np.ndarray
    tick_scalar_acceleration: np.ndarray
    tick_cell_index: np.ndarray
    tick_cell_side: np.ndarray
    path_progress_contract: str
    path_evaluator_kind: str
    path_evaluator_sha256_float64_le: str
    geometry_continuity_contract: str
    node_second_derivative_contract: str
    boundary_second_derivative_contract: str
    tick_second_derivative_contract: str
    grid_subdivisions: int
    time_scale: float
    weighted_arc_length_receipt: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        """Own and freeze every array so callers cannot mutate solver state."""

        integer_fields = {"tick_cell_index"}
        string_fields = {"tick_cell_side"}
        for field_name in (
            "s_node",
            "s_mid",
            "q_node",
            "q_s_node",
            "q_ss_node_left",
            "q_ss_node_right",
            "q_mid",
            "q_s_mid",
            "q_ss_mid",
            "x_node",
            "u_cell",
            "time_node_s",
            "time_mid_s",
            "tick_time_s",
            "tick_s",
            "tick_x",
            "tick_q",
            "tick_q_s",
            "tick_q_ss",
            "tick_qdot",
            "tick_qdd",
            "tick_scalar_acceleration",
            "tick_cell_index",
            "tick_cell_side",
        ):
            value = getattr(self, field_name)
            if field_name in integer_fields:
                owned = np.array(value, dtype=np.int64, order="C", copy=True)
            elif field_name in string_fields:
                owned = np.array(value, dtype="U32", order="C", copy=True)
            else:
                owned = np.array(value, dtype=np.float64, order="C", copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, field_name, owned)
        receipt = self.weighted_arc_length_receipt
        if receipt is not None:
            if not isinstance(receipt, Mapping):
                raise ValueError(
                    "weighted_arc_length_receipt must be a mapping or None"
                )
            object.__setattr__(
                self,
                "weighted_arc_length_receipt",
                MappingProxyType(dict(receipt)),
            )


@dataclass(frozen=True)
class RetimeResult:
    """A uniform-rate, rest-to-rest traversal of the input geometric path.

    ``path_speed`` and ``path_acceleration`` are per-output-segment average
    values and therefore both have shape ``(len(q) - 1,)``.
    """

    q: np.ndarray
    qdot: np.ndarray
    path_position: np.ndarray
    path_speed: np.ndarray
    path_acceleration: np.ndarray
    markers: Dict[str, MarkerMapping]
    report: dict
    collocation_trace: Optional[ScalarPathCollocationTrace] = None


@dataclass(frozen=True)
class _ContinuousCellPeaks:
    """Time-scale-independent exact peaks for every scalar cell and joint."""

    velocity: np.ndarray
    acceleration: np.ndarray


_EPS = 1e-12
_REGULARITY_EPS = 1e-10
_CONTROL_GUARD_ITERATION_LIMIT_OVERRIDE: Optional[int] = None
_EXPLICIT_GRID_REFINEMENT_LEVELS = 6
_EXPLICIT_GRID_STABLE_LEVELS_REQUIRED = 3
_EXPLICIT_GRID_MIN_LEVELS_BEFORE_ACCEPT = 4
_EXPLICIT_GRID_TIME_TOLERANCE_TICKS = 0.1
# Probe-grade exact-pointwise-cap base-grid stability probe (informational).
_EXACT_POINTWISE_DURATION_TOLERANCE_TICKS = 0.5


def control_tick_at_or_after(time_s: float, fps: float) -> int:
    """First observable control tick with a float-error-boundary snap."""

    if not np.isfinite(fps) or float(fps) <= 0.0:
        raise RetimeError("fps must be finite and positive")
    scaled = float(time_s) * float(fps)
    if not np.isfinite(scaled) or scaled < 0.0:
        raise RetimeError(
            "time_s and fps must produce a finite non-negative control tick"
        )
    nearest = float(np.rint(scaled))
    boundary_tolerance = (
        4096.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(scaled))
    )
    if abs(scaled - nearest) <= boundary_tolerance:
        return int(nearest)
    return int(np.ceil(float(np.nextafter(scaled, -np.inf))))


def _as_limit_vector(name: str, value: np.ndarray, joints: int) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise RetimeError(f"{name} must be real-valued")
    try:
        out = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetimeError(f"{name} must be a real-valued numeric vector") from exc
    if out.shape != (joints,):
        raise RetimeError(f"{name} must have shape ({joints},), got {out.shape}")
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise RetimeError(f"{name} must contain finite, strictly positive values")
    return out


def _validate_position_limits(
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the mandatory, paired joint-position contract."""

    if (lower is None) != (upper is None):
        raise RetimeError(
            "position_lower_limits and position_upper_limits must be provided together"
        )
    if lower is None or upper is None:
        raise RetimeError(
            "position_lower_limits and position_upper_limits must be provided together"
        )
    lower_raw = np.asarray(lower)
    upper_raw = np.asarray(upper)
    if np.iscomplexobj(lower_raw) or np.iscomplexobj(upper_raw):
        raise RetimeError("position limits must be real-valued")
    try:
        lower_out = lower_raw.astype(np.float64, copy=False)
        upper_out = upper_raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetimeError("position limits must be real-valued numeric vectors") from exc
    expected = (joints,)
    if lower_out.shape != expected:
        raise RetimeError(
            f"position_lower_limits must have shape {expected}, got {lower_out.shape}"
        )
    if upper_out.shape != expected:
        raise RetimeError(
            f"position_upper_limits must have shape {expected}, got {upper_out.shape}"
        )
    if not np.all(np.isfinite(lower_out)) or not np.all(np.isfinite(upper_out)):
        raise RetimeError("position limits must contain only finite values")
    if np.any(lower_out >= upper_out):
        raise RetimeError(
            "every position lower limit must be strictly lower than its upper limit"
        )
    return lower_out, upper_out


def _validate_markers(
    markers: Optional[Mapping[str, float]], path_end: float
) -> Dict[str, float]:
    checked: Dict[str, float] = {}
    if markers is None:
        return checked
    for raw_name, raw_position in markers.items():
        name = str(raw_name)
        if not name:
            raise RetimeError("marker names must be non-empty")
        position = float(raw_position)
        if not np.isfinite(position) or position < 0.0 or position > path_end:
            raise RetimeError(
                f"marker {name!r}={position!r} is outside source-index range "
                f"[0, {path_end}]"
            )
        checked[name] = position
    return checked


def _validate_marker_min_durations(
    requested: Optional[Mapping[Tuple[str, str], float]],
    markers: Mapping[str, float],
) -> Dict[Tuple[str, str], float]:
    """Validate minimum traversal times between named path markers."""

    checked: Dict[Tuple[str, str], float] = {}
    if requested is None:
        return checked
    for raw_pair, raw_duration in requested.items():
        if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
            raise RetimeError(
                "marker_min_duration_s keys must be (start_marker, end_marker) tuples"
            )
        start_name, end_name = str(raw_pair[0]), str(raw_pair[1])
        if start_name not in markers or end_name not in markers:
            raise RetimeError(
                "marker_min_duration_s references an unknown marker: "
                f"{(start_name, end_name)!r}"
            )
        if markers[end_name] <= markers[start_name]:
            raise RetimeError(
                "marker_min_duration_s requires end marker after start marker: "
                f"{(start_name, end_name)!r}"
            )
        duration = float(raw_duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise RetimeError("marker minimum durations must be finite and positive")
        checked[(start_name, end_name)] = duration
    return checked


def _inclusive_discrete_marker_interval(
    start_time_base: float,
    end_time_base: float,
    time_scale: float,
    fps: float,
) -> dict:
    """Full output samples contained in a continuous marker interval.

    A marker at a fractional frame does not make the nearest output sample part
    of the interval.  The first consumable sample is ``ceil(start)`` and the
    last is ``floor(end)``.  Their difference is the number of complete 50 Hz
    control intervals; an inclusive pair of adjacent samples therefore gives
    one control interval.
    """

    start_fractional = start_time_base * time_scale * fps
    end_fractional = end_time_base * time_scale * fps
    frame_tolerance = 1e-10
    start_frame = int(np.ceil(start_fractional - frame_tolerance))
    end_frame = int(np.floor(end_fractional + frame_tolerance))
    sample_count = max(0, end_frame - start_frame + 1)
    control_intervals = max(0, end_frame - start_frame)
    return {
        "start_fractional_frame": float(start_fractional),
        "end_fractional_frame": float(end_fractional),
        "discrete_start_frame": start_frame,
        "discrete_end_frame": end_frame,
        "discrete_sample_count": sample_count,
        "discrete_control_intervals": control_intervals,
        "discrete_duration_s": control_intervals / float(fps),
    }


def _discrete_marker_duration_status(
    marker_time_base: Mapping[str, float],
    marker_min_durations: Mapping[Tuple[str, str], float],
    time_scale: float,
    fps: float,
) -> tuple[bool, Dict[Tuple[str, str], dict]]:
    status: Dict[Tuple[str, str], dict] = {}
    all_pass = True
    for pair, minimum_duration in marker_min_durations.items():
        start_name, end_name = pair
        interval = _inclusive_discrete_marker_interval(
            marker_time_base[start_name],
            marker_time_base[end_name],
            time_scale,
            fps,
        )
        required_intervals = int(
            np.ceil(minimum_duration * float(fps) - 1e-12)
        )
        passed = interval["discrete_control_intervals"] >= required_intervals
        interval["required_control_intervals"] = required_intervals
        interval["passed"] = bool(passed)
        status[pair] = interval
        all_pass = all_pass and passed
    return all_pass, status


def _next_discrete_safe_time_scale(
    marker_time_base: Mapping[str, float],
    marker_min_durations: Mapping[Tuple[str, str], float],
    base_duration: float,
    fps: float,
    current_output_intervals: int,
) -> float:
    """Find the next quantised whole-clip duration passing all marker gates."""

    lower_intervals = current_output_intervals + 1
    guaranteed_intervals = lower_intervals
    for pair, minimum_duration in marker_min_durations.items():
        start_name, end_name = pair
        interval_fraction = (
            marker_time_base[end_name] - marker_time_base[start_name]
        ) / base_duration
        if not np.isfinite(interval_fraction) or interval_fraction <= 0.0:
            raise RetimeError(f"marker interval {pair!r} has invalid base timing")
        required = int(np.ceil(minimum_duration * float(fps) - 1e-12))
        # Any interval of width R+2 frames contains at least R complete
        # ceil(start)->floor(end) intervals regardless of fractional alignment.
        guaranteed_intervals = max(
            guaranteed_intervals,
            int(np.ceil((required + 2.0 + 1e-10) / interval_fraction)),
        )
        lower_intervals = max(
            lower_intervals,
            int(np.ceil(required / interval_fraction - 1e-12)),
        )

    search_count = guaranteed_intervals - lower_intervals + 1
    if search_count > 1_000_000:
        raise RetimeError(
            "discrete marker-duration search is pathologically large; "
            "marker interval is too narrow for the requested output duration"
        )
    for output_intervals in range(lower_intervals, guaranteed_intervals + 1):
        candidate_scale = output_intervals / (base_duration * float(fps))
        passed, _ = _discrete_marker_duration_status(
            marker_time_base,
            marker_min_durations,
            candidate_scale,
            fps,
        )
        if passed:
            return candidate_scale
    raise RetimeError("could not construct a discrete-safe marker timeline")


def _path_tangents(
    q_path: np.ndarray,
    path_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Shape-preserving PCHIP tangents in the declared path coordinate."""

    nodes = (
        np.arange(len(q_path), dtype=np.float64)
        if path_nodes is None
        else np.asarray(path_nodes, dtype=np.float64)
    )
    if (
        nodes.shape != (len(q_path),)
        or not np.isfinite(nodes).all()
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise RetimeError(
            "path_nodes must be finite, one-dimensional, and strictly increasing"
        )
    widths = np.diff(nodes)
    chord = np.diff(q_path, axis=0) / widths[:, None]
    tangents = np.zeros_like(q_path, dtype=np.float64)
    if len(q_path) > 2:
        left = chord[:-1]
        right = chord[1:]
        same_sign = (left * right) > 0.0
        left_width = widths[:-1, None]
        right_width = widths[1:, None]
        weight_left = 2.0 * right_width + left_width
        weight_right = right_width + 2.0 * left_width
        denominator = np.zeros_like(left)
        np.divide(
            weight_left,
            left,
            out=denominator,
            where=np.abs(left) > _EPS,
        )
        right_term = np.zeros_like(right)
        np.divide(
            weight_right,
            right,
            out=right_term,
            where=np.abs(right) > _EPS,
        )
        denominator += right_term
        safe = same_sign & (np.abs(denominator) > _EPS)
        interior = np.zeros_like(left)
        numerator = weight_left + weight_right
        np.divide(
            np.broadcast_to(numerator, denominator.shape),
            denominator,
            out=interior,
            where=safe,
        )
        tangents[1:-1] = interior

    def endpoint_tangent(
        first: np.ndarray,
        second: np.ndarray,
        first_width: float,
        second_width: float,
    ) -> np.ndarray:
        candidate = (
            (2.0 * first_width + second_width) * first
            - first_width * second
        ) / (first_width + second_width)
        candidate = np.where(candidate * first <= 0.0, 0.0, candidate)
        opposite = first * second <= 0.0
        too_large = np.abs(candidate) > 3.0 * np.abs(first)
        return np.where(opposite & too_large, 3.0 * first, candidate)

    tangents[0] = endpoint_tangent(
        chord[0], chord[1], float(widths[0]), float(widths[1])
    )
    tangents[-1] = endpoint_tangent(
        chord[-1], chord[-2], float(widths[-1]), float(widths[-2])
    )
    return tangents


def _hermite_coefficients(
    q_path: np.ndarray,
    tangents: np.ndarray,
    path_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return local-path-coordinate cubic coefficients per segment/joint."""

    nodes = (
        np.arange(len(q_path), dtype=np.float64)
        if path_nodes is None
        else np.asarray(path_nodes, dtype=np.float64)
    )
    widths = np.diff(nodes)[:, None]
    q0 = q_path[:-1]
    delta = q_path[1:] - q0
    m0 = tangents[:-1]
    m1 = tangents[1:]
    return np.stack(
        (
            q0,
            m0,
            3.0 * delta / np.square(widths)
            - (2.0 * m0 + m1) / widths,
            -2.0 * delta / np.power(widths, 3)
            + (m0 + m1) / np.square(widths),
        ),
        axis=-1,
    )


def _quintic_hermite_coefficients(
    q_path: np.ndarray,
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
    path_nodes: np.ndarray,
) -> np.ndarray:
    """Local-coordinate quintics fixed by endpoint position and 2-jets."""

    nodes = np.asarray(path_nodes, dtype=np.float64)
    widths = np.diff(nodes)[:, None]
    q0 = q_path[:-1]
    delta = q_path[1:] - q0
    d0 = first_derivative[:-1] * widths
    d1 = first_derivative[1:] * widths
    dd0 = second_derivative[:-1] * np.square(widths)
    dd1 = second_derivative[1:] * np.square(widths)
    normalized = np.stack(
        (
            q0,
            d0,
            0.5 * dd0,
            10.0 * delta
            - 6.0 * d0
            - 4.0 * d1
            - 1.5 * dd0
            + 0.5 * dd1,
            -15.0 * delta
            + 8.0 * d0
            + 7.0 * d1
            + 1.5 * dd0
            - dd1,
            6.0 * delta
            - 3.0 * d0
            - 3.0 * d1
            - 0.5 * dd0
            + 0.5 * dd1,
        ),
        axis=-1,
    )
    powers = np.arange(6, dtype=np.float64)
    coefficients = normalized / np.power(
        widths[..., None], powers[None, None, :]
    )
    if not np.all(np.isfinite(coefficients)):
        raise RetimeError("quintic path-jet coefficients became non-finite")
    return coefficients


def _eval_polynomial_path(
    coefficients: np.ndarray,
    path_nodes: np.ndarray,
    path_position: np.ndarray,
    *,
    side: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a piecewise local-coordinate polynomial and its first 2 jets."""

    nodes = np.asarray(path_nodes, dtype=np.float64)
    s = np.asarray(path_position, dtype=np.float64)
    if side not in {"left", "right"}:
        raise RetimeError("path evaluator side must be 'left' or 'right'")
    if s.ndim != 1:
        raise RetimeError("path_position must be one-dimensional")
    if np.any(s < nodes[0] - _EPS) or np.any(s > nodes[-1] + _EPS):
        raise RetimeError("path evaluation requested outside its declared range")
    s = np.clip(s, nodes[0], nodes[-1])
    segment = np.searchsorted(nodes, s, side=side) - 1
    segment = np.clip(segment, 0, len(nodes) - 2)
    local = s - nodes[segment]
    selected = coefficients[segment]
    degree = selected.shape[-1] - 1

    q = selected[..., -1].copy()
    for coefficient_index in range(degree - 1, -1, -1):
        q = q * local[:, None] + selected[..., coefficient_index]

    if degree >= 1:
        q_s = degree * selected[..., degree]
        for coefficient_index in range(degree - 1, 0, -1):
            q_s = (
                q_s * local[:, None]
                + coefficient_index * selected[..., coefficient_index]
            )
    else:  # pragma: no cover - supported evaluators are cubic/quintic
        q_s = np.zeros_like(q)

    if degree >= 2:
        q_ss = degree * (degree - 1) * selected[..., degree]
        for coefficient_index in range(degree - 1, 1, -1):
            q_ss = (
                q_ss * local[:, None]
                + coefficient_index
                * (coefficient_index - 1)
                * selected[..., coefficient_index]
            )
    else:  # pragma: no cover - supported evaluators are cubic/quintic
        q_ss = np.zeros_like(q)
    return q, q_s, q_ss


def _path_evaluator_digest(
    *,
    q_path: np.ndarray,
    path_nodes: np.ndarray,
    first_derivative: Optional[np.ndarray],
    second_derivative: Optional[np.ndarray],
    evaluator_kind: str,
    parameterization_sha256: Optional[str] = None,
) -> str:
    """Content-bind the exact evaluator inputs, including shapes and kind."""

    digest = hashlib.sha256()
    digest.update(evaluator_kind.encode("ascii"))
    digest.update(b"\0parameterization_sha256\0")
    digest.update(
        b"NONE"
        if parameterization_sha256 is None
        else parameterization_sha256.encode("ascii")
    )
    for name, value in (
        ("q_path", q_path),
        ("path_nodes", path_nodes),
        ("first_derivative", first_derivative),
        ("second_derivative", second_derivative),
    ):
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"NONE")
            continue
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _weighted_arc_length_receipt(
    path: WeightedArcPath,
    *,
    evaluator_mode: str = "direct_exact",
) -> Mapping[str, object]:
    """Content-bind the certified weighted-arc coordinate used by one solve."""

    if not isinstance(path, WeightedArcPath):
        raise RetimeError(
            "weighted_arc_path must be a WeightedArcPath instance"
        )
    if evaluator_mode not in {
        "direct_exact",
        "endpoint_arc_jet_quintic_approximation",
    }:
        raise RetimeError(
            "weighted arc evaluator mode must be 'direct_exact' or "
            "'endpoint_arc_jet_quintic_approximation'"
        )
    try:
        digest_verified = path.verify_content_digest()
    except (WeightedArcPathError, ValueError, TypeError) as exc:
        raise RetimeError(
            "weighted arc path digest verification failed"
        ) from exc
    if not digest_verified:
        raise RetimeError(
            "weighted arc path content digest does not match its bound inputs"
        )
    if (
        len(path.segment_audits) != len(path.segment_lengths)
        or len(path.segment_lengths) != len(path.l_knots) - 1
    ):
        raise RetimeError("weighted arc path has inconsistent segment receipts")
    certified_minimum = min(
        audit.certified_min_weighted_speed_per_s
        for audit in path.segment_audits
    )
    observed_minimum = min(
        audit.observed_min_weighted_speed_per_s
        for audit in path.segment_audits
    )
    if (
        not np.isfinite(certified_minimum)
        or certified_minimum <= path.regularity_margin
        or not np.isfinite(observed_minimum)
        or observed_minimum <= path.regularity_margin
    ):
        raise RetimeError(
            "weighted arc path lacks a valid positive regularity receipt"
        )
    coordinate_scale_sha256 = hashlib.sha256(
        np.ascontiguousarray(path.coordinate_scale, dtype="<f8").tobytes(
            order="C"
        )
    ).hexdigest()
    l_knots_sha256 = hashlib.sha256(
        np.ascontiguousarray(path.l_knots, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    payload = {
        "enabled": True,
        "contract": (
            "weighted_arc_length_v1"
            if evaluator_mode == "direct_exact"
            else "endpoint_arc_jet_quintic_approximation_v1"
        ),
        "evaluator_mode": evaluator_mode,
        "exact_direct_evaluator": evaluator_mode == "direct_exact",
        "exact_evaluator_api": (
            "WeightedArcPath.evaluate_l"
            if evaluator_mode == "direct_exact"
            else None
        ),
        "algorithm_id": WEIGHTED_ARC_ALGORITHM_ID,
        "content_sha256": path.content_sha256,
        "coordinate_scale_sha256_float64_le": coordinate_scale_sha256,
        "coordinate_dimension": int(path.dimension),
        "formal_knot_count": int(len(path.l_knots)),
        "l_knots_sha256_float64_le": l_knots_sha256,
        "total_length": float(path.total_length),
        "arc_absolute_tolerance": float(path.arc_absolute_tolerance),
        "arc_relative_tolerance": float(path.arc_relative_tolerance),
        "quadrature_max_depth": int(path.quadrature_max_depth),
        "quadrature_error_estimate_sum": float(
            sum(
                audit.quadrature_error_estimate
                for audit in path.segment_audits
            )
        ),
        "regularity_margin": float(path.regularity_margin),
        "regularity_max_depth": int(path.regularity_max_depth),
        "certified_min_weighted_speed_per_s": float(certified_minimum),
        "observed_min_weighted_speed_per_s": float(observed_minimum),
        "inverse_absolute_tolerance": float(
            path.inverse_absolute_tolerance
        ),
        "inverse_relative_tolerance": float(
            path.inverse_relative_tolerance
        ),
        "inverse_parameter_tolerance": float(
            path.inverse_parameter_tolerance
        ),
        "inverse_max_iterations": int(path.inverse_max_iterations),
        "digest_verified": True,
        "regularity_certified": True,
        "formal_knots_exact": True,
        "uniform_scalar_acceleration_semantics": (
            "constant_weighted_arc_coordinate_acceleration_only"
            if evaluator_mode == "direct_exact"
            else (
                "constant_endpoint_arc_jet_quintic_approximation_"
                "coordinate_acceleration_only"
            )
        ),
        "non_claims": (
            "uniform_joint_acceleration",
            "uniform_actuator_torque",
            "contact_or_return_quality",
            "balance_or_executability",
            *(
                ()
                if evaluator_mode == "direct_exact"
                else ("exact_weighted_arc_length_parameterization",)
            ),
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(
        b"canonical-weighted-arc-retime-receipt-v1\0" + encoded
    ).hexdigest()
    return MappingProxyType(payload)


def _real_stationary_points(
    coefficients: np.ndarray, lower: float, upper: float
) -> list[float]:
    """Real roots of a polynomial derivative in a closed interval."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    last = len(coefficients) - 1
    while last > 0 and abs(coefficients[last]) <= 32.0 * np.finfo(float).eps * scale:
        last -= 1
    if last <= 1:
        return []
    derivative = np.arange(1, last + 1, dtype=np.float64) * coefficients[1 : last + 1]
    if last == 2:
        # A quadratic polynomial has one linear-derivative root.  np.roots
        # constructs and solves a companion eigenproblem even for this case;
        # the direct expression is the same root without that setup cost.
        roots = np.asarray(
            [-derivative[0] / derivative[1]], dtype=np.complex128
        )
    elif last == 3:
        # Preserve the same real/near-real filtering below while avoiding a
        # 2x2 companion eigensolve for cubic path-position extrema.
        a = float(derivative[2])
        b = float(derivative[1])
        c = float(derivative[0])
        discriminant = b * b - 4.0 * a * c
        if np.isfinite(discriminant):
            if discriminant >= 0.0:
                square_root = float(np.sqrt(discriminant))
                q = -0.5 * (b + np.copysign(square_root, b))
                if q == 0.0:
                    repeated = -b / (2.0 * a)
                    roots = np.asarray(
                        [repeated, repeated], dtype=np.complex128
                    )
                else:
                    roots = np.asarray(
                        [q / a, c / q], dtype=np.complex128
                    )
            else:
                real = -b / (2.0 * a)
                imaginary = float(np.sqrt(-discriminant)) / (2.0 * abs(a))
                roots = np.asarray(
                    [real + 1j * imaginary, real - 1j * imaginary],
                    dtype=np.complex128,
                )
        else:
            # Extreme finite coefficients can overflow the discriminant even
            # though NumPy's normalized companion solve remains well-defined.
            roots = np.roots(derivative[::-1])
    else:
        roots = np.roots(derivative[::-1])
    if np.any(~np.isfinite(roots)):
        raise RetimeError("polynomial-extremum validation produced non-finite roots")
    tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
    points = []
    for root in roots:
        if abs(float(root.imag)) <= tolerance:
            real = float(root.real)
            if lower - tolerance <= real <= upper + tolerance:
                points.append(float(np.clip(real, lower, upper)))
    return points


def _polynomial_range(
    coefficients: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    points = [lower, upper]
    points.extend(_real_stationary_points(coefficients, lower, upper))
    values = np.polynomial.polynomial.polyval(points, coefficients)
    if np.any(~np.isfinite(values)):
        raise RetimeError("polynomial-extremum validation produced non-finite values")
    return float(np.min(values)), float(np.max(values))


def _batched_polynomial_ranges(
    coefficients: np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ranges for a batch of ascending-order real polynomials.

    The last axis stores coefficients.  Leading axes are independent
    polynomials, and ``lower``/``upper`` broadcast over those axes.  Degree
    trimming and near-real/in-interval root tolerances intentionally match
    :func:`_real_stationary_points`.

    Companion matrices for equal-degree derivatives are solved in one NumPy
    batch.  This changes only scheduling: each matrix is byte-for-byte the one
    used by ``np.roots`` after coefficient reversal.
    """

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise RetimeError(
            "batched polynomial coefficients must have at least two dimensions"
        )
    leading_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    if np.any(~np.isfinite(flat)):
        raise RetimeError(
            "polynomial-extremum validation produced non-finite values"
        )
    try:
        lower_flat = np.broadcast_to(
            np.asarray(lower, dtype=np.float64), leading_shape
        ).reshape(-1)
        upper_flat = np.broadcast_to(
            np.asarray(upper, dtype=np.float64), leading_shape
        ).reshape(-1)
    except ValueError as exc:
        raise RetimeError(
            "polynomial range bounds do not broadcast over the batch"
        ) from exc
    if (
        np.any(~np.isfinite(lower_flat))
        or np.any(~np.isfinite(upper_flat))
        or np.any(upper_flat < lower_flat)
    ):
        raise RetimeError("polynomial range bounds must be finite and ordered")

    scale = np.maximum(1.0, np.max(np.abs(flat), axis=1))
    trim_tolerance = 32.0 * np.finfo(float).eps * scale
    degree = np.zeros(len(flat), dtype=np.int64)
    for coefficient_index in range(1, flat.shape[1]):
        degree = np.where(
            np.abs(flat[:, coefficient_index]) > trim_tolerance,
            coefficient_index,
            degree,
        )

    def evaluate(rows: np.ndarray, points: np.ndarray) -> np.ndarray:
        result = flat[rows, -1].copy()
        for coefficient_index in range(flat.shape[1] - 2, -1, -1):
            result = (
                result * points + flat[rows, coefficient_index]
            )
        return result

    all_rows = np.arange(len(flat), dtype=np.int64)
    low_values = evaluate(all_rows, lower_flat)
    high_values = evaluate(all_rows, upper_flat)
    minimum = np.minimum(low_values, high_values)
    maximum = np.maximum(low_values, high_values)

    for polynomial_degree in range(2, flat.shape[1]):
        rows = np.flatnonzero(degree == polynomial_degree)
        if len(rows) == 0:
            continue
        # Degree-8 derivative companions arise for quintic qdot**2.  Bound
        # peak memory without changing any individual polynomial or root solve.
        root_batch_size = 8192
        for batch_start in range(0, len(rows), root_batch_size):
            batch_rows = rows[batch_start : batch_start + root_batch_size]
            derivative_ascending = (
                flat[batch_rows, 1 : polynomial_degree + 1]
                * np.arange(1, polynomial_degree + 1, dtype=np.float64)[
                    None, :
                ]
            )
            derivative_order = polynomial_degree - 1
            if derivative_order == 1:
                roots = (
                    -derivative_ascending[:, :1]
                    / derivative_ascending[:, 1:2]
                ).astype(np.complex128)
            else:
                descending = derivative_ascending[:, ::-1]
                companion = np.zeros(
                    (
                        len(batch_rows),
                        derivative_order,
                        derivative_order,
                    ),
                    dtype=np.float64,
                )
                companion[:, 0, :] = (
                    -descending[:, 1:] / descending[:, :1]
                )
                index = np.arange(1, derivative_order)
                companion[:, index, index - 1] = 1.0
                roots = np.linalg.eigvals(companion)
            if np.any(~np.isfinite(roots)):
                raise RetimeError(
                    "polynomial-extremum validation produced non-finite roots"
                )
            root_tolerance = 1e-10 * np.maximum.reduce(
                (
                    np.ones(len(batch_rows), dtype=np.float64),
                    np.abs(lower_flat[batch_rows]),
                    np.abs(upper_flat[batch_rows]),
                )
            )
            for root_index in range(roots.shape[1]):
                root = roots[:, root_index]
                real = np.asarray(root.real, dtype=np.float64)
                valid = (
                    (np.abs(root.imag) <= root_tolerance)
                    & (real >= lower_flat[batch_rows] - root_tolerance)
                    & (real <= upper_flat[batch_rows] + root_tolerance)
                )
                clipped = np.clip(
                    real,
                    lower_flat[batch_rows],
                    upper_flat[batch_rows],
                )
                evaluated = evaluate(batch_rows, clipped)
                minimum[batch_rows] = np.where(
                    valid,
                    np.minimum(minimum[batch_rows], evaluated),
                    minimum[batch_rows],
                )
                maximum[batch_rows] = np.where(
                    valid,
                    np.maximum(maximum[batch_rows], evaluated),
                    maximum[batch_rows],
                )
    if np.any(~np.isfinite(minimum)) or np.any(~np.isfinite(maximum)):
        raise RetimeError(
            "polynomial-extremum validation produced non-finite values"
        )
    return minimum.reshape(leading_shape), maximum.reshape(leading_shape)


def _continuous_position_range(
    hermite_coefficients: np.ndarray,
    segment_widths: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial-stationary-point min/max over all path segments."""

    upper = (
        1.0
        if segment_widths is None
        else np.asarray(segment_widths, dtype=np.float64)[:, None]
    )
    low, high = _batched_polynomial_ranges(
        hermite_coefficients, 0.0, upper
    )
    return np.min(low, axis=0), np.max(high, axis=0)


def _eval_path(
    q_path: np.ndarray,
    tangents: np.ndarray,
    path_position: np.ndarray,
    path_nodes: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate cubic-Hermite ``q``, ``dq/ds``, and ``d2q/ds2``."""

    nodes = (
        np.arange(len(q_path), dtype=np.float64)
        if path_nodes is None
        else np.asarray(path_nodes, dtype=np.float64)
    )
    s = np.asarray(path_position, dtype=np.float64)
    path_start = float(nodes[0])
    path_end = float(nodes[-1])
    if np.any(s < path_start - _EPS) or np.any(s > path_end + _EPS):
        raise RetimeError("path evaluation requested outside its declared range")
    s = np.clip(s, path_start, path_end)
    segment = np.searchsorted(nodes, s, side="right") - 1
    segment = np.clip(segment, 0, len(q_path) - 2)
    width = nodes[segment + 1] - nodes[segment]
    u = (s - nodes[segment]) / width
    u_col = u[:, None]
    width_col = width[:, None]

    q0 = q_path[segment]
    q1 = q_path[segment + 1]
    m0 = tangents[segment] * width_col
    m1 = tangents[segment + 1] * width_col

    u2 = u_col * u_col
    u3 = u2 * u_col
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u_col
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    q = h00 * q0 + h10 * m0 + h01 * q1 + h11 * m1

    dh00 = 6.0 * u2 - 6.0 * u_col
    dh10 = 3.0 * u2 - 4.0 * u_col + 1.0
    dh01 = -6.0 * u2 + 6.0 * u_col
    dh11 = 3.0 * u2 - 2.0 * u_col
    q_s = (
        dh00 * q0 + dh10 * m0 + dh01 * q1 + dh11 * m1
    ) / width_col

    d2h00 = 12.0 * u_col - 6.0
    d2h10 = 6.0 * u_col - 4.0
    d2h01 = -12.0 * u_col + 6.0
    d2h11 = 6.0 * u_col - 2.0
    q_ss = (
        d2h00 * q0 + d2h10 * m0 + d2h01 * q1 + d2h11 * m1
    ) / np.square(width_col)
    return q, q_s, q_ss


def _eval_path_node_second_derivative_sides(
    hermite_coefficients: np.ndarray,
    path_nodes: np.ndarray,
    path_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact lower-s and higher-s ``q_ss`` limits at path nodes.

    A C1 PCHIP may have a second-derivative jump at an input knot; a verified C2
    endpoint-2-jet path must produce equal values.  At the path boundaries,
    where only one physical side exists, the unavailable side is duplicated
    from the available side to keep a finite rectangular trace.
    """

    nodes = np.asarray(path_nodes, dtype=np.float64)
    positions = np.asarray(path_position, dtype=np.float64)
    if positions.ndim != 1:
        raise RetimeError("path_position must be one-dimensional")

    from_left = _eval_polynomial_path(
        hermite_coefficients, nodes, positions, side="left"
    )[2]
    from_right = _eval_polynomial_path(
        hermite_coefficients, nodes, positions, side="right"
    )[2]
    from_left[0] = from_right[0]
    from_right[-1] = from_left[-1]
    return from_left, from_right


def _scalar_caps(
    q_s: np.ndarray,
    q_ss: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
) -> np.ndarray:
    """Maximum scalar speed squared from joint velocity and curvature limits."""

    speed_sq_cap = np.full(len(q_s), np.inf, dtype=np.float64)
    for joint in range(q_s.shape[1]):
        slope = np.abs(q_s[:, joint])
        curved = np.abs(q_ss[:, joint])
        active_slope = slope > _REGULARITY_EPS
        active_curve = curved > _REGULARITY_EPS
        speed_sq_cap[active_slope] = np.minimum(
            speed_sq_cap[active_slope],
            (velocity_limits[joint] / slope[active_slope]) ** 2,
        )
        speed_sq_cap[active_curve] = np.minimum(
            speed_sq_cap[active_curve],
            acceleration_limits[joint] / curved[active_curve],
        )
    if np.any(~np.isfinite(speed_sq_cap)) or np.any(speed_sq_cap <= 0.0):
        raise RetimeError("path has an unbounded or non-positive scalar speed cap")
    return speed_sq_cap


def _tangential_acceleration_cap(
    q_s: np.ndarray,
    q_ss: np.ndarray,
    speed_sq: float,
    acceleration_limits: np.ndarray,
) -> float:
    """Conservative symmetric bound on ``|sddot|`` at one path location.

    Triangle inequality is used here: ``|q_ss| sdot^2`` spends curvature budget,
    and the remainder is available to ``|q_s| |sddot|``.  This gives up
    cancellation between the two terms, but every accepted profile is directly
    checkable.
    """

    remaining = acceleration_limits - np.abs(q_ss) * float(speed_sq)
    if np.any(remaining < -1e-10):
        return -1.0
    active = np.abs(q_s) > _REGULARITY_EPS
    if not np.any(active):
        return np.inf
    return float(np.min(np.maximum(remaining[active], 0.0) / np.abs(q_s[active])))


def _max_reachable_neighbor(
    fixed_speed_sq: float,
    neighbor_cap: float,
    segment_cap: float,
    ds: float,
    q_s_mid: np.ndarray,
    q_ss_mid: np.ndarray,
    acceleration_limits: np.ndarray,
) -> float:
    """Largest neighboring ``sdot^2`` reachable across one path segment."""

    upper = min(float(neighbor_cap), max(0.0, 2.0 * segment_cap - fixed_speed_sq))
    if upper <= fixed_speed_sq:
        # The opposite sweep will reduce ``fixed_speed_sq`` if this deceleration
        # is itself too sharp.  Returning the cap preserves monotone tightening.
        return max(0.0, upper)

    def feasible(candidate: float) -> bool:
        middle = 0.5 * (fixed_speed_sq + candidate)
        accel_cap = _tangential_acceleration_cap(
            q_s_mid, q_ss_mid, middle, acceleration_limits
        )
        required = (candidate - fixed_speed_sq) / (2.0 * ds)
        return accel_cap >= 0.0 and required <= accel_cap * (1.0 + 1e-12)

    if feasible(upper):
        return upper

    # ``feasible(candidate)`` is a conjunction of affine inequalities.  For
    # every active joint, expanding its tangential-acceleration constraint
    #
    #   (candidate-fixed)/(2*ds)
    #       <= (1+1e-12) * (a-|qss|*(fixed+candidate)/2) / |qs|
    #
    # gives one closed-form upper bound on ``candidate``.  Inactive joints
    # retain the original curvature-budget tolerance.  Taking the minimum is
    # therefore exactly the same monotone feasible frontier that the previous
    # 60-step bisection approached.
    slope = np.abs(q_s_mid)
    curvature = np.abs(q_ss_mid)
    multiplier = 1.0 + 1e-12
    candidate = upper
    curved = curvature > 0.0
    if np.any(curved):
        curvature_bound = (
            2.0
            * (acceleration_limits[curved] + 1e-10)
            / curvature[curved]
            - fixed_speed_sq
        )
        candidate = min(candidate, float(np.min(curvature_bound)))
    active = slope > _REGULARITY_EPS
    if np.any(active):
        active_slope = slope[active]
        active_curvature = curvature[active]
        denominator = (
            active_slope + ds * multiplier * active_curvature
        )
        numerator = (
            2.0 * ds * multiplier * acceleration_limits[active]
            + fixed_speed_sq
            * (active_slope - ds * multiplier * active_curvature)
        )
        candidate = min(candidate, float(np.min(numerator / denominator)))
    candidate = max(fixed_speed_sq, min(upper, candidate))

    # Closed-form arithmetic can land one ulp above the same floating-point
    # inequality, or a few ulps below it after cancellation.  First step toward
    # the known-feasible endpoint, then advance to the greatest adjacent
    # feasible float.  This also reproduces the converged bisection endpoint.
    for _ in range(4):
        if feasible(candidate):
            for _ in range(8):
                next_candidate = float(np.nextafter(candidate, upper))
                if (
                    next_candidate == candidate
                    or next_candidate > upper
                    or not feasible(next_candidate)
                ):
                    break
                candidate = next_candidate
            return candidate
        next_candidate = float(np.nextafter(candidate, fixed_speed_sq))
        if next_candidate == candidate:
            break
        candidate = next_candidate

    low = fixed_speed_sq
    high = candidate
    for _ in range(60):
        middle = 0.5 * (low + high)
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def _forward_backward_profile(
    path_grid: np.ndarray,
    q_s_mid: np.ndarray,
    q_ss_mid: np.ndarray,
    node_caps: np.ndarray,
    segment_caps: np.ndarray,
    acceleration_limits: np.ndarray,
    max_sweeps: int,
) -> tuple[np.ndarray, int]:
    """Solve a rest-to-rest scalar-speed envelope by monotone sweeps."""

    speed_sq = node_caps.copy()
    speed_sq[0] = 0.0
    speed_sq[-1] = 0.0
    sweeps_used = 0
    for sweep in range(max_sweeps):
        before = speed_sq.copy()
        for i in range(len(path_grid) - 1):
            ds = float(path_grid[i + 1] - path_grid[i])
            reachable = _max_reachable_neighbor(
                speed_sq[i],
                node_caps[i + 1],
                segment_caps[i],
                ds,
                q_s_mid[i],
                q_ss_mid[i],
                acceleration_limits,
            )
            speed_sq[i + 1] = min(speed_sq[i + 1], reachable)
        speed_sq[-1] = 0.0
        for i in range(len(path_grid) - 2, -1, -1):
            ds = float(path_grid[i + 1] - path_grid[i])
            reachable = _max_reachable_neighbor(
                speed_sq[i + 1],
                node_caps[i],
                segment_caps[i],
                ds,
                q_s_mid[i],
                q_ss_mid[i],
                acceleration_limits,
            )
            speed_sq[i] = min(speed_sq[i], reachable)
        speed_sq[0] = 0.0
        sweeps_used = sweep + 1
        if np.max(np.abs(before - speed_sq)) <= 1e-11 * max(
            1.0, float(np.max(node_caps))
        ):
            break
    else:
        raise RetimeError("forward/backward scalar-path pass did not converge")

    if np.any(~np.isfinite(speed_sq)) or np.any(speed_sq < -1e-12):
        raise RetimeError("scalar-path solver produced an invalid speed profile")
    return np.maximum(speed_sq, 0.0), sweeps_used


def _insert_exact_grid_node(
    path_grid: np.ndarray,
    position: float,
) -> tuple[np.ndarray, int, bool]:
    """Return a grid containing ``position`` as an exact node.

    A marker inside a scalar cell cannot safely be rounded down: that would
    leave part of the containing cell outside a marker-bound acceleration
    policy.  A numerically coincident node is replaced by the exact marker
    value; otherwise the cell is explicitly split.
    """

    grid = np.asarray(path_grid, dtype=np.float64)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise RetimeError("path_grid must be a finite strictly increasing vector")
    target = float(position)
    if not np.isfinite(target) or target < grid[0] or target > grid[-1]:
        raise RetimeError("grid marker lies outside the scalar path")

    index = int(np.searchsorted(grid, target, side="left"))
    scale = max(1.0, abs(target), abs(float(grid[-1])))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    candidates = []
    if index < len(grid):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    for candidate in candidates:
        if abs(float(grid[candidate]) - target) <= tolerance:
            exact = grid.copy()
            exact[candidate] = target
            if np.any(np.diff(exact) <= 0.0):
                raise RetimeError("exact marker insertion collapsed a scalar cell")
            return exact, candidate, False

    inserted = np.insert(grid, index, target)
    if np.any(np.diff(inserted) <= 0.0):
        raise RetimeError("exact marker insertion produced a non-increasing grid")
    return inserted, index, True


def _greatest_nondecreasing_minorant_until(
    speed_sq: np.ndarray,
    marker_index: int,
) -> np.ndarray:
    """Project a speed-squared profile to no scalar braking before a node.

    For ``i <= marker_index`` the result is the suffix minimum

    ``result[i] = min(speed_sq[i:marker_index + 1])``.

    This is the pointwise greatest nondecreasing sequence dominated by the
    ordinary profile on that prefix.  The marker node and every later node are
    unchanged.
    """

    profile = np.asarray(speed_sq, dtype=np.float64)
    if profile.ndim != 1 or len(profile) < 2 or not np.all(np.isfinite(profile)):
        raise RetimeError("speed_sq must be a finite one-dimensional profile")
    if np.any(profile < 0.0):
        raise RetimeError("speed_sq must be non-negative")
    if not isinstance(marker_index, (int, np.integer)):
        raise RetimeError("marker_index must be an integer")
    marker_index = int(marker_index)
    if marker_index < 0 or marker_index >= len(profile):
        raise RetimeError("marker_index lies outside speed_sq")

    projected = profile.copy()
    prefix_reversed = projected[: marker_index + 1][::-1]
    projected[: marker_index + 1] = np.minimum.accumulate(prefix_reversed)[::-1]
    return projected


def _reject_zero_speed_bottleneck(
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    *,
    policy_name: str,
) -> None:
    """Reject positive-distance cells whose two endpoint speeds are zero."""

    grid = np.asarray(path_grid, dtype=np.float64)
    profile = np.asarray(speed_sq, dtype=np.float64)
    if profile.shape != grid.shape:
        raise RetimeError("path_grid and speed_sq shapes disagree")
    trapped = (
        (np.diff(grid) > 0.0)
        & (profile[:-1] <= _EPS)
        & (profile[1:] <= _EPS)
    )
    if np.any(trapped):
        cell = int(np.flatnonzero(trapped)[0])
        raise RetimeError(
            f"{policy_name} creates an internal zero-speed bottleneck in "
            f"scalar cell {cell}; finite-time traversal is impossible"
        )


def _profile_time_knots(
    path_grid: np.ndarray, speed_sq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(speed_sq)
    ds = np.diff(path_grid)
    denom = speed[:-1] + speed[1:]
    if np.any(denom <= _EPS):
        raise RetimeError("positive path distance is trapped between zero-speed nodes")
    dt = 2.0 * ds / denom
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise RetimeError("scalar-path solver produced an invalid timeline")
    time = np.concatenate(([0.0], np.cumsum(dt)))
    segment_accel = np.diff(speed_sq) / (2.0 * ds)
    return time, segment_accel


def _sample_scalar_profile(
    output_time: np.ndarray,
    time_scale: float,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    time_knots: np.ndarray,
    segment_accel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``s``, ``sdot``, and ``sddot`` from piecewise bang-bang arcs."""

    base_time = np.clip(output_time / time_scale, 0.0, time_knots[-1])
    segment = np.searchsorted(time_knots, base_time, side="right") - 1
    segment = np.clip(segment, 0, len(path_grid) - 2)
    local_t = base_time - time_knots[segment]
    start_speed = np.sqrt(speed_sq[segment])
    accel = segment_accel[segment]
    local_s = start_speed * local_t + 0.5 * accel * local_t * local_t
    path_position = path_grid[segment] + local_s
    base_speed = np.maximum(start_speed + accel * local_t, 0.0)
    path_speed = base_speed / time_scale
    path_accel = accel / (time_scale * time_scale)

    path_position[0] = path_grid[0]
    path_position[-1] = path_grid[-1]
    path_speed[0] = 0.0
    path_speed[-1] = 0.0
    return path_position, path_speed, path_accel


def _time_at_path_position(
    position: float,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    time_knots: np.ndarray,
    segment_accel: np.ndarray,
) -> float:
    if position <= path_grid[0]:
        return 0.0
    if position >= path_grid[-1]:
        return float(time_knots[-1])
    i = int(np.searchsorted(path_grid, position, side="right") - 1)
    distance = float(position - path_grid[i])
    start_speed = float(np.sqrt(speed_sq[i]))
    accel = float(segment_accel[i])
    end_speed = float(np.sqrt(max(speed_sq[i] + 2.0 * accel * distance, 0.0)))
    denom = start_speed + end_speed
    if denom <= _EPS:
        raise RetimeError("marker lies in a zero-speed path interval")
    return float(time_knots[i] + 2.0 * distance / denom)


def _build_collocation_trace(
    *,
    path_grid: np.ndarray,
    path_mid: np.ndarray,
    q_node: np.ndarray,
    q_s_node: np.ndarray,
    q_ss_node_left: np.ndarray,
    q_ss_node_right: np.ndarray,
    q_mid: np.ndarray,
    q_s_mid: np.ndarray,
    q_ss_mid: np.ndarray,
    speed_sq: np.ndarray,
    segment_accel: np.ndarray,
    time_knots: np.ndarray,
    time_scale: float,
    output_time: np.ndarray,
    s_out: np.ndarray,
    sdot_out: np.ndarray,
    sddot_out: np.ndarray,
    q_out: np.ndarray,
    q_s_out: np.ndarray,
    q_ss_out: np.ndarray,
    qdot_out: np.ndarray,
    qddot_out: np.ndarray,
    path_progress_contract: str,
    path_evaluator_kind: str,
    path_evaluator_sha256: str,
    geometry_continuity_contract: str,
    grid_subdivisions: int,
    weighted_arc_length_receipt: Optional[Mapping[str, object]],
) -> ScalarPathCollocationTrace:
    """Freeze the exact collocation and tick state of the accepted solve."""

    time_mid = np.asarray(
        [
            _time_at_path_position(
                float(position),
                path_grid,
                speed_sq,
                time_knots,
                segment_accel,
            )
            for position in path_mid
        ],
        dtype=np.float64,
    )
    tick_base_time = np.clip(
        output_time / time_scale, 0.0, time_knots[-1]
    )
    tick_cell_index = np.searchsorted(
        time_knots, tick_base_time, side="right"
    ).astype(np.int64) - 1
    tick_cell_index = np.clip(
        tick_cell_index, 0, len(path_grid) - 2
    )
    tick_cell_side = np.full(
        len(s_out), "cell_interior", dtype="U32"
    )
    knot_candidate = np.searchsorted(
        time_knots, tick_base_time, side="left"
    )
    valid_candidate = knot_candidate < len(time_knots)
    knot_distance = np.full(len(tick_base_time), np.inf, dtype=np.float64)
    knot_distance[valid_candidate] = np.abs(
        time_knots[knot_candidate[valid_candidate]]
        - tick_base_time[valid_candidate]
    )
    knot_tolerance = (
        128.0
        * np.finfo(np.float64).eps
        * np.maximum(1.0, np.abs(tick_base_time))
    )
    at_knot = valid_candidate & (knot_distance <= knot_tolerance)
    tick_cell_side[at_knot] = "right_cell_at_knot"
    at_end = at_knot & (knot_candidate == len(time_knots) - 1)
    tick_cell_side[at_end] = "left_cell_at_path_end"

    return ScalarPathCollocationTrace(
        s_node=path_grid,
        s_mid=path_mid,
        q_node=q_node,
        q_s_node=q_s_node,
        q_ss_node_left=q_ss_node_left,
        q_ss_node_right=q_ss_node_right,
        q_mid=q_mid,
        q_s_mid=q_s_mid,
        q_ss_mid=q_ss_mid,
        x_node=speed_sq / (time_scale * time_scale),
        u_cell=segment_accel / (time_scale * time_scale),
        time_node_s=time_knots * time_scale,
        time_mid_s=time_mid * time_scale,
        tick_time_s=output_time,
        tick_s=s_out,
        tick_x=np.square(sdot_out),
        tick_q=q_out,
        tick_q_s=q_s_out,
        tick_q_ss=q_ss_out,
        tick_qdot=qdot_out,
        tick_qdd=qddot_out,
        tick_scalar_acceleration=sddot_out,
        tick_cell_index=tick_cell_index,
        tick_cell_side=tick_cell_side,
        path_progress_contract=path_progress_contract,
        path_evaluator_kind=path_evaluator_kind,
        path_evaluator_sha256_float64_le=path_evaluator_sha256,
        geometry_continuity_contract=geometry_continuity_contract,
        node_second_derivative_contract=(
            "q_ss_node_left_is_lower_s_limit;_q_ss_node_right_is_higher_s_"
            "limit;_cell_i_uses_right_i_at_start_and_left_i_plus_1_at_end"
        ),
        boundary_second_derivative_contract=(
            "start_left_duplicates_start_right;_end_right_duplicates_end_left"
        ),
        tick_second_derivative_contract=(
            "tick_q_ss_uses_the_cell_selected_by_tick_cell_index_and_"
            "tick_cell_side"
        ),
        grid_subdivisions=int(grid_subdivisions),
        time_scale=float(time_scale),
        weighted_arc_length_receipt=weighted_arc_length_receipt,
    )


def _shift_cubic(coefficients: np.ndarray, offset: float) -> np.ndarray:
    """Translate ``p(u)`` to ascending coefficients of ``p(offset + x)``."""

    c0, c1, c2, c3 = coefficients
    return np.array(
        [
            c0 + c1 * offset + c2 * offset**2 + c3 * offset**3,
            c1 + 2.0 * c2 * offset + 3.0 * c3 * offset**2,
            c2 + 3.0 * c3 * offset,
            c3,
        ],
        dtype=np.float64,
    )


def _continuous_cell_peaks(
    hermite_coefficients: np.ndarray,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    segment_accel: np.ndarray,
    path_nodes: Optional[np.ndarray] = None,
) -> _ContinuousCellPeaks:
    """Prepare exact unscaled peaks for constant-acceleration scalar cells.

    Within one scalar cell, ``sdot**2`` is linear in path position.  For a
    degree-``d`` path, squared joint velocity is degree ``2d-1`` and joint
    acceleration is degree ``d-1``.  Evaluating their polynomial stationary
    points closes the gap left by checking only a cell midpoint.  The supported
    evaluators are cubic PCHIP and endpoint-2-jet quintic Hermite.

    These extrema depend on the fixed path and base scalar envelope, but not on
    a later global time scale.  Preparing them once lets validation iterations
    preserve the exact same gates without repeating polynomial construction and
    root solves.
    """

    coefficients = np.asarray(hermite_coefficients, dtype=np.float64)
    grid = np.asarray(path_grid, dtype=np.float64)
    speed = np.asarray(speed_sq, dtype=np.float64)
    scalar_acceleration = np.asarray(segment_accel, dtype=np.float64)
    if (
        coefficients.ndim != 3
        or coefficients.shape[2] not in (4, 6)
        or grid.ndim != 1
        or speed.shape != grid.shape
        or scalar_acceleration.shape != (len(grid) - 1,)
    ):
        raise RetimeError("continuous-cell validation received inconsistent shapes")
    if (
        np.any(~np.isfinite(coefficients))
        or np.any(~np.isfinite(grid))
        or np.any(~np.isfinite(speed))
        or np.any(~np.isfinite(scalar_acceleration))
        or np.any(np.diff(grid) <= 0.0)
        or np.any(speed < 0.0)
    ):
        raise RetimeError("continuous-cell validation received invalid values")

    path_segments = hermite_coefficients.shape[0]
    nodes = (
        np.arange(path_segments + 1, dtype=np.float64)
        if path_nodes is None
        else np.asarray(path_nodes, dtype=np.float64)
    )
    if (
        nodes.shape != (path_segments + 1,)
        or not np.isfinite(nodes).all()
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise RetimeError(
            "continuous-cell path nodes must be finite and strictly increasing"
        )
    start = grid[:-1]
    end = grid[1:]
    width = end - start
    midpoint = 0.5 * (start + end)
    source_segment = np.searchsorted(nodes, midpoint, side="right") - 1
    source_segment = np.clip(source_segment, 0, path_segments - 1)
    source_start = nodes[source_segment]
    source_end = nodes[source_segment + 1]
    if np.any(start < source_start - 1e-10) or np.any(
        end > source_end + 1e-10
    ):
        raise RetimeError(
            "a scalar validation cell crosses a Hermite segment boundary"
        )
    local_start = start - source_start
    source = coefficients[source_segment]
    degree = source.shape[-1] - 1
    x = local_start[:, None]
    shifted = np.zeros_like(source)
    for output_degree in range(degree + 1):
        for input_degree in range(output_degree, degree + 1):
            shifted[..., output_degree] += (
                source[..., input_degree]
                * math.comb(input_degree, output_degree)
                * np.power(x, input_degree - output_degree)
            )
    slope = shifted[..., 1:] * np.arange(
        1, degree + 1, dtype=np.float64
    )
    curvature = shifted[..., 2:] * (
        np.arange(2, degree + 1, dtype=np.float64)
        * np.arange(1, degree, dtype=np.float64)
    )

    slope_squared = np.zeros(
        (*slope.shape[:-1], 2 * slope.shape[-1] - 1),
        dtype=np.float64,
    )
    for left_degree in range(slope.shape[-1]):
        for right_degree in range(slope.shape[-1]):
            slope_squared[..., left_degree + right_degree] += (
                slope[..., left_degree] * slope[..., right_degree]
            )
    scalar_speed_0 = speed[:-1, None]
    scalar_speed_1 = (2.0 * scalar_acceleration)[:, None]
    velocity_coefficients = np.zeros(
        (*slope_squared.shape[:-1], slope_squared.shape[-1] + 1),
        dtype=np.float64,
    )
    velocity_coefficients[..., :-1] += (
        slope_squared * scalar_speed_0[..., None]
    )
    velocity_coefficients[..., 1:] += (
        slope_squared * scalar_speed_1[..., None]
    )

    acceleration_coefficients = np.zeros(
        (*slope.shape[:-1], slope.shape[-1]), dtype=np.float64
    )
    acceleration_coefficients[..., : curvature.shape[-1]] += (
        curvature * scalar_speed_0[..., None]
    )
    acceleration_coefficients[..., 1 : curvature.shape[-1] + 1] += (
        curvature * scalar_speed_1[..., None]
    )
    acceleration_coefficients += (
        scalar_acceleration[:, None, None] * slope
    )
    upper = width[:, None]
    velocity_low, velocity_high = _batched_polynomial_ranges(
        velocity_coefficients, 0.0, upper
    )
    numerical_scale = np.maximum(1.0, np.abs(velocity_high))
    if np.any(velocity_low < -1e-10 * numerical_scale):
        raise RetimeError(
            "continuous-cell validation found negative squared velocity"
        )
    acceleration_low, acceleration_high = _batched_polynomial_ranges(
        acceleration_coefficients, 0.0, upper
    )
    return _ContinuousCellPeaks(
        velocity=np.sqrt(np.maximum(velocity_high, 0.0)),
        acceleration=np.maximum(
            np.abs(acceleration_low), np.abs(acceleration_high)
        ),
    )


def _weighted_arc_position_range(
    path: WeightedArcPath,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact coordinate ranges of the digest-bound source quintics.

    Reparameterizing by a strictly increasing weighted arc coordinate does not
    change the geometric image.  The stationary-point solve therefore operates
    on the exact normalized-s quintics stored by ``WeightedArcPath`` rather
    than manufacturing a second polynomial in ``l``.
    """

    if not path.verify_content_digest():
        raise RetimeError(
            "weighted arc path digest changed before position validation"
        )
    low, high = _batched_polynomial_ranges(
        path._coefficients,
        0.0,
        1.0,
    )
    return np.min(low, axis=0), np.max(high, axis=0)


# Fixed per-formal-segment subdivision count for the grid-independent candidate
# envelope.  It is deliberately independent of the solver's ``grid_subdivisions``
# so the seed profile is byte-identical across fixed-grid refinement levels while
# still pairing local curvature with a *local* certified weighted speed instead of
# the whole-segment minimum.
_ENVELOPE_SEGMENT_SUBDIVISIONS = 16


def _interval_multiply_vector(
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise outward-rounded interval product for array operands."""

    products = (
        left_low * right_low,
        left_low * right_high,
        left_high * right_low,
        left_high * right_high,
    )
    low = np.minimum(np.minimum(products[0], products[1]), np.minimum(products[2], products[3]))
    high = np.maximum(np.maximum(products[0], products[1]), np.maximum(products[2], products[3]))
    return np.nextafter(low, -np.inf), np.nextafter(high, np.inf)


def _interval_add_vector(
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise outward-rounded interval sum for array operands."""

    return (
        np.nextafter(left_low + right_low, -np.inf),
        np.nextafter(left_high + right_high, np.inf),
    )


def _decasteljau_interval_split_vector(
    low: np.ndarray,
    high: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised outward-rounded de Casteljau interval split at per-row ``t``.

    ``low``/``high`` are ``(rows, controls)`` Bernstein interval controls on
    ``[0, 1]``.  Each row is subdivided at its own exact parameter ``t[row]``
    (with ``1 - t`` carried as an outward interval), returning the left ``[0,t]``
    and right ``[t,1]`` control intervals.  This generalises the exact-midpoint
    split used by the weighted-arc regularity certificate to arbitrary ``t`` and
    therefore preserves the same convex-hull enclosure guarantee.
    """

    level_low = np.asarray(low, dtype=np.float64).copy()
    level_high = np.asarray(high, dtype=np.float64).copy()
    rows, count = level_low.shape
    t_col = np.asarray(t, dtype=np.float64)[:, None]
    omt_low = np.nextafter(1.0 - t_col, -np.inf)
    omt_high = np.nextafter(1.0 - t_col, np.inf)
    left_low = np.empty((rows, count), dtype=np.float64)
    left_high = np.empty((rows, count), dtype=np.float64)
    right_low = np.empty((rows, count), dtype=np.float64)
    right_high = np.empty((rows, count), dtype=np.float64)
    left_low[:, 0] = level_low[:, 0]
    left_high[:, 0] = level_high[:, 0]
    right_low[:, -1] = level_low[:, -1]
    right_high[:, -1] = level_high[:, -1]
    for depth in range(1, count):
        a_low, a_high = _interval_multiply_vector(
            omt_low, omt_high, level_low[:, :-1], level_high[:, :-1]
        )
        b_low, b_high = _interval_multiply_vector(
            t_col, t_col, level_low[:, 1:], level_high[:, 1:]
        )
        level_low, level_high = _interval_add_vector(a_low, a_high, b_low, b_high)
        left_low[:, depth] = level_low[:, 0]
        left_high[:, depth] = level_high[:, 0]
        right_low[:, -depth - 1] = level_low[:, -1]
        right_high[:, -depth - 1] = level_high[:, -1]
    return left_low, left_high, right_low, right_high


def _weighted_arc_subcell_speed_lower_bounds(
    path: WeightedArcPath,
    source_segment: np.ndarray,
    lower_u: np.ndarray,
    upper_u: np.ndarray,
) -> np.ndarray:
    """Certified per-cell weighted-speed lower bounds via Bernstein subdivision.

    For each cell ``i`` (bound to formal segment ``source_segment[i]`` over the
    local ``[lower_u[i], upper_u[i]]``) the exact digest-bound
    ``||W dq/ds||**2`` power polynomial is taken to outward-rounded Bernstein
    interval form on ``[0, 1]`` (reusing the weighted-arc regularity machinery)
    and restricted to the cell's own sub-interval by de Casteljau subdivision.
    The minimum lower control is a certified lower bound on ``||W dq/ds||**2``
    over that sub-interval (Bernstein convex-hull property); its square root is
    rounded down.  The result is finally lifted to be no smaller than the
    segment-global certified minimum, so it can never fall below the build-time
    regularity certificate.
    """

    seg = np.asarray(source_segment, dtype=np.int64)
    lo = np.clip(np.asarray(lower_u, dtype=np.float64), 0.0, 1.0)
    hi = np.clip(np.asarray(upper_u, dtype=np.float64), 0.0, 1.0)
    s_widths = np.diff(path.s_knots)
    control_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index in np.unique(seg):
        seg_index = int(index)
        power_low, power_high = _wa_weighted_speed_squared_power_intervals(
            path._coefficients[seg_index],
            segment_width=float(s_widths[seg_index]),
            coordinate_scale=path.coordinate_scale,
        )
        control_cache[seg_index] = _wa_power_intervals_to_bernstein(
            power_low, power_high
        )
    controls = next(iter(control_cache.values()))
    control_count = len(controls[0])
    rows = len(seg)
    bernstein_low = np.empty((rows, control_count), dtype=np.float64)
    bernstein_high = np.empty((rows, control_count), dtype=np.float64)
    for i in range(rows):
        low_i, high_i = control_cache[int(seg[i])]
        bernstein_low[i] = low_i
        bernstein_high[i] = high_i
    valid = hi > lo
    b_split = np.where(valid, hi, 1.0)
    left_low, left_high, _, _ = _decasteljau_interval_split_vector(
        bernstein_low, bernstein_high, b_split
    )
    inner_t = np.where(
        valid & (b_split > 0.0),
        np.clip(lo / np.where(b_split > 0.0, b_split, 1.0), 0.0, 1.0),
        0.0,
    )
    _, _, right_low, _ = _decasteljau_interval_split_vector(
        left_low, left_high, inner_t
    )
    speed_squared_lb = np.min(right_low, axis=1)
    # Degenerate/inverted cells fall back to the whole-segment control hull.
    speed_squared_lb = np.where(
        valid, speed_squared_lb, np.min(bernstein_low, axis=1)
    )
    cell_speed = np.nextafter(
        np.sqrt(np.maximum(speed_squared_lb, 0.0)), -np.inf
    )
    segment_minimum = np.asarray(
        [
            path.segment_audits[int(index)].certified_min_weighted_speed_per_s
            for index in seg
        ],
        dtype=np.float64,
    )
    # The per-cell bound may only tighten (raise) the certified speed; it can
    # never fall below the build-time segment certificate, which is already
    # strictly above the regularity margin.
    return np.maximum(cell_speed, segment_minimum)


def _weighted_arc_conservative_cell_derivatives(
    path: WeightedArcPath,
    source_segment: np.ndarray,
    lower_u: np.ndarray,
    upper_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative per-cell ``|q_l|``/``|q_ll|`` bounds for the exact arc curve.

    ``q_s``/``q_ss`` are bounded over each cell's *own* local ``u`` sub-interval
    and divided by a *certified per-cell* weighted-speed lower bound rather than
    the segment-global minimum (which pairs a cell's peak derivative with the
    slowest point of an entire formal segment and thereby inflates the bound by
    orders of magnitude under ``coordinate_scale = 1 / v``).  The exact
    reparameterization invariant ``||W q_l|| = 1`` — equivalently
    ``|q_l_j| <= 1 / W_j`` — is applied as a hard clamp so the tangent bound can
    never exceed the physical velocity budget.  Returns
    ``(q_l_abs, q_ll_abs, certified_cell_speed)``; every returned bound still
    dominates the exact ``evaluate_l`` values point-for-point.
    """

    seg = np.asarray(source_segment, dtype=np.int64)
    lo = np.asarray(lower_u, dtype=np.float64)
    hi = np.asarray(upper_u, dtype=np.float64)
    coefficients = path._coefficients[seg]
    source_s_width = np.diff(path.s_knots)[seg]
    first_coefficients = (
        coefficients[..., 1:]
        * np.arange(1, 6, dtype=np.float64)[None, None, :]
        / source_s_width[:, None, None]
    )
    second_coefficients = (
        coefficients[..., 2:]
        * (
            np.arange(2, 6, dtype=np.float64)
            * np.arange(1, 5, dtype=np.float64)
        )[None, None, :]
        / np.square(source_s_width[:, None, None])
    )
    first_low, first_high = _batched_polynomial_ranges(
        first_coefficients, lo[:, None], hi[:, None]
    )
    second_low, second_high = _batched_polynomial_ranges(
        second_coefficients, lo[:, None], hi[:, None]
    )
    roundoff = 4096.0 * np.finfo(np.float64).eps
    q_s_abs = (
        np.maximum(np.abs(first_low), np.abs(first_high)) * (1.0 + roundoff)
        + roundoff
    )
    q_ss_abs = (
        np.maximum(np.abs(second_low), np.abs(second_high)) * (1.0 + roundoff)
        + roundoff
    )
    certified_speed = _weighted_arc_subcell_speed_lower_bounds(
        path, seg, lo, hi
    )
    if np.any(certified_speed <= path.regularity_margin):
        raise RetimeError(
            "weighted arc per-cell weighted-speed bound lost its regularity "
            "certificate"
        )
    q_l_abs = q_s_abs / certified_speed[:, None]
    # Exact invariant ||W q_l|| = 1  =>  |q_l_j| <= 1 / W_j = v_j.  Clamping here
    # is what stops the velocity cap (v_j / |q_l_j|)**2 from collapsing below 1.
    inverse_scale = (1.0 + roundoff) / path.coordinate_scale[None, :]
    q_l_abs = np.minimum(q_l_abs, inverse_scale)
    weighted_q_ss_norm = np.linalg.norm(
        q_ss_abs * path.coordinate_scale[None, :], axis=1
    )
    # |q_ll_j| <= |q_ss_j|/speed**2 + |q_l_j|*|speed_s|/speed**2 with
    # |speed_s| <= ||W q_ss||.  Using the clamped q_l_abs keeps this an upper
    # bound while tightening the curvature term.
    q_ll_abs = (
        q_ss_abs / np.square(certified_speed[:, None])
        + q_l_abs
        * weighted_q_ss_norm[:, None]
        / np.square(certified_speed[:, None])
    )
    return q_l_abs, q_ll_abs, certified_speed


def _weighted_arc_formal_segment_solver_envelope(
    path: WeightedArcPath,
    path_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-independent q_l/q_ll envelopes for the scalar candidate solver.

    Every scalar cell inside one formal segment receives the same conservative
    bound.  This intentionally gives up some local tightness so fixed-grid
    refinement cannot manufacture a faster envelope merely by shrinking the
    cell used for derivative estimation.  A segment whose nonlinear
    coefficients are only floating-point construction residue is normalized
    to its affine chord.  That normalization only seeds the candidate time
    law: the separate continuous-cell gate still bounds and checks the exact
    digest-bound evaluator before any result can be admitted.
    """

    grid = np.asarray(path_grid, dtype=np.float64)
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    source_segment = (
        np.searchsorted(path.l_knots, midpoint, side="right") - 1
    )
    source_segment = np.clip(
        source_segment, 0, len(path.segment_lengths) - 1
    ).astype(np.int64)
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(path.total_length))
    )
    if np.any(grid[:-1] < path.l_knots[source_segment] - tolerance) or np.any(
        grid[1:] > path.l_knots[source_segment + 1] + tolerance
    ):
        raise RetimeError(
            "a scalar envelope cell crosses an exact weighted-arc segment"
        )
    # Grid-independent, non-catastrophic bound: subdivide each formal segment
    # into a *fixed* number of local u-subintervals (independent of the solver
    # grid), bound |q_l|/|q_ll| on each sub-interval against its own certified
    # local weighted speed, and take the worst sub-interval as the whole-segment
    # bound.  Every scalar cell inside one formal segment still receives the same
    # bound (the documented grid-independent seed), but the segment's peak
    # curvature is no longer cross-paired with the slowest point of the entire
    # segment -- which was inflating the seed hundreds-fold under W = 1 / v.
    coefficients = path._coefficients[source_segment]
    sub_edges = np.linspace(
        0.0, 1.0, _ENVELOPE_SEGMENT_SUBDIVISIONS + 1, dtype=np.float64
    )
    sub_lower = sub_edges[:-1]
    sub_upper = sub_edges[1:]
    dimension = int(path.dimension)
    segment_q_l: Dict[int, np.ndarray] = {}
    segment_q_ll: Dict[int, np.ndarray] = {}
    for index in np.unique(source_segment):
        seg_index = int(index)
        seg_ids = np.full(len(sub_lower), seg_index, dtype=np.int64)
        sub_q_l, sub_q_ll, _sub_speed = (
            _weighted_arc_conservative_cell_derivatives(
                path, seg_ids, sub_lower, sub_upper
            )
        )
        segment_q_l[seg_index] = np.max(sub_q_l, axis=0)
        segment_q_ll[seg_index] = np.max(sub_q_ll, axis=0)
    q_l_abs = np.empty((len(source_segment), dimension), dtype=np.float64)
    q_ll_abs = np.empty((len(source_segment), dimension), dtype=np.float64)
    for cell in range(len(source_segment)):
        seg_index = int(source_segment[cell])
        q_l_abs[cell] = segment_q_l[seg_index]
        q_ll_abs[cell] = segment_q_ll[seg_index]
    # Quintic endpoint-jet assembly can leave sub-ulp cubic/quartic
    # coefficients on a mathematically affine segment.  Feeding those
    # representation-dependent crumbs into the scalar solver moves a
    # velocity-cap transition away from the same physical chord when the
    # identical line is stored at a different knot density.  Canonicalize
    # only that roundoff-sized case.  This is deliberately *not* used by
    # _weighted_arc_continuous_cell_peaks, which validates the exact stored
    # curve and therefore remains the hard safety gate.
    coefficient_scale = np.maximum(
        1.0,
        np.max(np.abs(coefficients), axis=(1, 2)),
    )
    nonlinear_magnitude = np.max(
        np.abs(coefficients[..., 2:]), axis=(1, 2)
    )
    affine_roundoff = (
        nonlinear_magnitude
        <= 4096.0 * np.finfo(np.float64).eps * coefficient_scale
    )
    if np.any(affine_roundoff):
        segment_index = source_segment[affine_roundoff]
        chord = np.abs(
            path.q_knots[segment_index + 1] - path.q_knots[segment_index]
        )
        chord_q_l = (
            chord
            / path.segment_lengths[segment_index, None]
        )
        q_l_abs[affine_roundoff] = chord_q_l
        q_ll_abs[affine_roundoff] = 0.0
    return q_l_abs, q_ll_abs


def _weighted_arc_continuous_cell_peaks(
    path: WeightedArcPath,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    segment_accel: np.ndarray,
) -> _ContinuousCellPeaks:
    """Conservative continuous-cell bounds for the exact arc evaluator.

    The retimed state itself is always evaluated by
    :meth:`WeightedArcPath.evaluate_l`.  For a whole scalar cell, a finite set
    of point evaluations cannot certify a maximum, so this function also
    derives conservative component bounds from the *same digest-bound source
    quintic* and its Bernstein-certified positive weighted speed.  No
    endpoint-jet polynomial in ``l`` is constructed.
    """

    if not isinstance(path, WeightedArcPath) or not path.verify_content_digest():
        raise RetimeError(
            "weighted arc continuous validation requires one intact "
            "digest-bound path"
        )
    grid = np.asarray(path_grid, dtype=np.float64)
    scalar_speed_sq = np.asarray(speed_sq, dtype=np.float64)
    scalar_accel = np.asarray(segment_accel, dtype=np.float64)
    if (
        grid.ndim != 1
        or scalar_speed_sq.shape != grid.shape
        or scalar_accel.shape != (len(grid) - 1,)
        or np.any(~np.isfinite(grid))
        or np.any(~np.isfinite(scalar_speed_sq))
        or np.any(~np.isfinite(scalar_accel))
        or np.any(np.diff(grid) <= 0.0)
        or np.any(scalar_speed_sq < 0.0)
    ):
        raise RetimeError(
            "weighted arc continuous validation received invalid scalar cells"
        )

    start = grid[:-1]
    end = grid[1:]
    midpoint = 0.5 * (start + end)
    source_segment = (
        np.searchsorted(path.l_knots, midpoint, side="right") - 1
    )
    source_segment = np.clip(
        source_segment, 0, len(path.segment_lengths) - 1
    ).astype(np.int64)
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(path.total_length))
    )
    if np.any(start < path.l_knots[source_segment] - tolerance) or np.any(
        end > path.l_knots[source_segment + 1] + tolerance
    ):
        raise RetimeError(
            "a scalar validation cell crosses an exact weighted-arc segment"
        )

    try:
        edge_s_lower, edge_s_upper = path.s_interval_from_l(
            np.concatenate((start, end))
        )
    except WeightedArcPathError as exc:
        raise RetimeError(
            "weighted arc cell-boundary inverse interval failed"
        ) from exc
    edge_s_lower = np.asarray(edge_s_lower, dtype=np.float64)
    edge_s_upper = np.asarray(edge_s_upper, dtype=np.float64)
    start_s = edge_s_lower[: len(start)]
    end_s = edge_s_upper[len(start) :]
    source_s_start = path.s_knots[source_segment]
    source_s_width = np.diff(path.s_knots)[source_segment]
    lower_u = np.clip(
        (start_s - source_s_start) / source_s_width, 0.0, 1.0
    )
    upper_u = np.clip(
        (end_s - source_s_start) / source_s_width, 0.0, 1.0
    )
    if np.any(upper_u <= lower_u):
        raise RetimeError(
            "weighted arc cell collapsed while mapping length to source "
            "parameter"
        )

    # Numerator (q_s/q_ss) AND denominator (weighted-speed lower bound) are now
    # both taken over each cell's own local u sub-interval.  Dividing the cell's
    # peak derivative by the *segment-global* minimum weighted speed previously
    # inflated |q_l|/|q_ll| by orders of magnitude (worse under W = 1 / v), which
    # collapsed the velocity/acceleration caps and stalled fixed-grid
    # convergence.  The per-cell certified speed is fail-closed and never falls
    # below the build-time segment certificate.
    q_l_abs, q_ll_abs, _cell_certified_speed = (
        _weighted_arc_conservative_cell_derivatives(
            path, source_segment, lower_u, upper_u
        )
    )

    max_x = np.maximum(
        scalar_speed_sq[:-1], scalar_speed_sq[1:]
    )[:, None]
    velocity_bound = q_l_abs * np.sqrt(max_x)
    acceleration_bound = (
        q_ll_abs * max_x + q_l_abs * np.abs(scalar_accel)[:, None]
    )

    # Direct point evaluations are retained as a contract check on the bound
    # algebra and ensure the exact evaluator is exercised by this gate too.
    fractions = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    probe_l = (
        start[:, None]
        + (end - start)[:, None] * fractions[None, :]
    )
    flat_probe = probe_l.reshape(-1)
    try:
        _, probe_q_l, probe_q_ll = path.evaluate_l(flat_probe)
    except WeightedArcPathError as exc:
        raise RetimeError(
            "exact weighted arc evaluator failed during continuous validation"
        ) from exc
    probe_q_l = probe_q_l.reshape(len(start), len(fractions), path.dimension)
    probe_q_ll = probe_q_ll.reshape(
        len(start), len(fractions), path.dimension
    )
    probe_x = (
        scalar_speed_sq[:-1, None]
        + 2.0
        * scalar_accel[:, None]
        * (probe_l - start[:, None])
    )
    if np.any(probe_x < -1.0e-10):
        raise RetimeError(
            "weighted arc continuous probe found negative scalar speed squared"
        )
    probe_x = np.maximum(probe_x, 0.0)
    probe_velocity = np.max(
        np.abs(probe_q_l) * np.sqrt(probe_x)[..., None], axis=1
    )
    probe_acceleration = np.max(
        np.abs(
            probe_q_ll * probe_x[..., None]
            + probe_q_l * scalar_accel[:, None, None]
        ),
        axis=1,
    )
    bound_tolerance = 1.0e-10
    if np.any(
        probe_velocity > velocity_bound * (1.0 + bound_tolerance) + 1.0e-12
    ) or np.any(
        probe_acceleration
        > acceleration_bound * (1.0 + bound_tolerance) + 1.0e-12
    ):
        raise RetimeError(
            "weighted arc conservative cell bound does not dominate direct "
            "evaluate_l probes"
        )
    return _ContinuousCellPeaks(
        velocity=np.maximum(velocity_bound, probe_velocity),
        acceleration=np.maximum(acceleration_bound, probe_acceleration),
    )


def _continuous_cell_ratios_from_peaks(
    peaks: _ContinuousCellPeaks,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    time_scale: float,
) -> tuple[float, float]:
    """Apply one global time scale without repeating exact extrema work."""

    if not np.isfinite(time_scale) or time_scale <= 0.0:
        raise RetimeError("continuous-cell validation received an invalid time scale")
    velocity_peak = np.asarray(peaks.velocity, dtype=np.float64)
    acceleration_peak = np.asarray(peaks.acceleration, dtype=np.float64)
    if (
        velocity_peak.ndim != 2
        or acceleration_peak.shape != velocity_peak.shape
        or velocity_peak.shape[1] != len(velocity_limits)
        or acceleration_peak.shape[1] != len(acceleration_limits)
    ):
        raise RetimeError("continuous-cell peak and limit shapes disagree")

    # Keep the original cell-major/joint-minor reduction and division order.
    # This makes cached validation numerically identical to recomputing the
    # time-scale-only arithmetic in every validation iteration.
    velocity_ratio = 0.0
    acceleration_ratio = 0.0
    acceleration_time_scale = time_scale * time_scale
    for cell in range(velocity_peak.shape[0]):
        for joint in range(velocity_peak.shape[1]):
            joint_velocity_peak = (
                float(velocity_peak[cell, joint]) / time_scale
            )
            velocity_ratio = max(
                velocity_ratio,
                joint_velocity_peak / float(velocity_limits[joint]),
            )
            joint_acceleration_peak = (
                float(acceleration_peak[cell, joint])
                / acceleration_time_scale
            )
            acceleration_ratio = max(
                acceleration_ratio,
                joint_acceleration_peak / float(acceleration_limits[joint]),
            )
    return float(velocity_ratio), float(acceleration_ratio)


def _continuous_cell_ratios(
    hermite_coefficients: np.ndarray,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    segment_accel: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    time_scale: float,
    path_nodes: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Exact continuous-cell ratios for callers that do not retain a cache."""

    peaks = _continuous_cell_peaks(
        hermite_coefficients,
        path_grid,
        speed_sq,
        segment_accel,
        path_nodes=path_nodes,
    )
    return _continuous_cell_ratios_from_peaks(
        peaks, velocity_limits, acceleration_limits, time_scale
    )


def _exact_pointwise_aposteriori_guard(
    *,
    evaluate_path,
    time_scale: float,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    time_knots: np.ndarray,
    segment_accel: np.ndarray,
    duration: float,
    fps: float,
    velocity_cap: np.ndarray,
    acceleration_cap: np.ndarray,
    guard_rate_multiple: int,
    guard_probe_margin: float,
    velocity_tolerance: float,
) -> dict:
    """Verify the ACHIEVED time law against v/a by dense time-domain sampling.

    The exact-pointwise-cap solve enforces caps at collocation nodes and
    midpoints; between them a real-mocap quintic can carry a razor-thin
    sub-control-rate curvature spike (e.g. a Hermite overshoot at a connector
    knot).  This guard samples the achieved trajectory at ``guard_rate_multiple``
    times the output rate and checks the joint velocity/acceleration the 50 Hz
    controller actually realises.  Velocity is a hard limit (it never collapses);
    acceleration may exceed the limit only up to ``(1 + guard_probe_margin)`` at
    the oversample rate, tolerating sub-control-rate geometric curvature the
    output stream does not resolve.  Fail closed on any broader excursion.  This
    is a probe-grade control-rate feasibility receipt, never a torque, contact,
    or balance certificate.
    """

    intervals = max(
        2, int(np.ceil(float(duration) * float(fps) * guard_rate_multiple - 1e-12))
    )
    sample_time = np.arange(intervals + 1, dtype=np.float64) * (
        float(duration) / intervals
    )
    s_g, sdot_g, sddot_g = _sample_scalar_profile(
        sample_time, time_scale, path_grid, speed_sq, time_knots, segment_accel
    )
    _, q_s_g, q_ss_g = evaluate_path(s_g)
    qdot_g = q_s_g * sdot_g[:, None]
    qddot_g = q_ss_g * (sdot_g * sdot_g)[:, None] + q_s_g * sddot_g[:, None]
    velocity_ratio = float(np.max(np.abs(qdot_g) / velocity_cap[None, :]))
    acceleration_ratio = float(
        np.max(np.abs(qddot_g) / acceleration_cap[None, :])
    )
    velocity_limit_ratio = 1.0 + max(float(velocity_tolerance), 1e-9)
    acceleration_limit_ratio = 1.0 + float(guard_probe_margin)
    velocity_ok = velocity_ratio <= velocity_limit_ratio
    acceleration_ok = acceleration_ratio <= acceleration_limit_ratio
    passed = bool(velocity_ok and acceleration_ok)
    receipt = {
        "kind": "exact_pointwise_time_domain_aposteriori_guard_v1",
        "guard_rate_multiple": int(guard_rate_multiple),
        "guard_sample_count": int(len(sample_time)),
        "output_rate_hz": float(fps),
        "max_velocity_ratio": velocity_ratio,
        "max_acceleration_ratio": acceleration_ratio,
        "velocity_limit_ratio": velocity_limit_ratio,
        "acceleration_limit_ratio": acceleration_limit_ratio,
        "velocity_ok": bool(velocity_ok),
        "acceleration_ok": bool(acceleration_ok),
        "passed": passed,
        "semantics": (
            "control-rate feasibility on the achieved trajectory; acceleration "
            "may exceed the limit up to (1+probe_margin) at the oversample rate "
            "to tolerate sub-control-rate quintic curvature; not a torque, "
            "contact, or balance certificate"
        ),
    }
    if not passed:
        raise RetimeError(
            "exact-pointwise a-posteriori guard failed: max qdot/v="
            f"{velocity_ratio:.4f} (limit {velocity_limit_ratio:.4f}), "
            f"max qddot/a={acceleration_ratio:.4f} "
            f"(limit {acceleration_limit_ratio:.4f}) at {guard_rate_multiple}x "
            f"the {fps:g} Hz control rate"
        )
    return receipt


def _retime_path_impl(
    q_path: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    position_lower_limits: np.ndarray,
    position_upper_limits: np.ndarray,
    path_progress: Optional[np.ndarray] = None,
    path_first_derivative: Optional[np.ndarray] = None,
    path_second_derivative: Optional[np.ndarray] = None,
    weighted_arc_path: Optional[WeightedArcPath] = None,
    weighted_arc_evaluator_mode: str = "direct_exact",
    fps: float = 50.0,
    markers: Optional[Mapping[str, float]] = None,
    marker_min_duration_s: Optional[Mapping[Tuple[str, str], float]] = None,
    nonnegative_acceleration_until_marker: Optional[str] = None,
    uniform_scalar_path_acceleration_until_marker: Optional[str] = None,
    grid_subdivisions: int = 12,
    max_sweeps: int = 100,
    max_validation_iterations: int = 12,
    validation_tolerance: float = 1e-6,
    position_tolerance: float = 1e-7,
    exact_pointwise_caps: bool = False,
    guard_rate_multiple: int = 4,
    guard_probe_margin: float = 1.0,
    _control_guard_path_position: Optional[float] = None,
    _control_guard_iteration: int = 0,
    _include_collocation_trace: bool = False,
) -> RetimeResult:
    """Retime a smooth discrete joint path into a uniform-rate rest-to-rest clip.

    Marker values are fractional source sample indices in ``[0, N-1]``.  For
    example, ``{"window_start": 34, "contact": 44, "window_end": 48}`` merely
    records where those geometric events land.  No marker locks source bytes or
    changes the acceleration profile.

    ``path_progress`` optionally declares a strictly increasing scalar
    coordinate at every input sample.  Marker sample indices are mapped
    piecewise-linearly into that coordinate.  The retimer does not infer that
    an arbitrary caller coordinate is continuous arc length.  ``None``
    preserves the legacy dense-sample-index parameterization.

    ``path_first_derivative`` and ``path_second_derivative`` optionally provide
    the complete endpoint 2-jet with respect to ``path_progress``.  When both
    are present, each adjacent pair is reconstructed as a quintic Hermite
    segment and one-sided knot jets are verified C2.  Canonical compilation
    uses this backend on formal geometry knots.  A nonregular endpoint fails
    closed: physical rest belongs in scalar speed, not a zero path tangent.

    ``weighted_arc_path`` binds the explicit progress and endpoint 2-jet to a
    digest-verified ``l(s)=integral ||W dq/ds|| ds`` construction.  Its exact
    formal ``l_knots``, ``q_l``, and ``q_ll`` must match the supplied arrays.
    The default ``weighted_arc_evaluator_mode="direct_exact"`` evaluates every
    node, midpoint, output tick, and kinematic constraint from that same
    object's ``evaluate_l`` method.  The only other mode,
    ``"endpoint_arc_jet_quintic_approximation"``, is an explicit warm-start
    comparator; it is not an exact weighted-arc parameterization and cannot
    carry the ``weighted_arc_length_v1`` contract.  Constant scalar
    acceleration in the direct coordinate is constant weighted-arc
    acceleration only, never constant joint acceleration or actuator torque.

    ``position_lower_limits`` and ``position_upper_limits`` are mandatory,
    finite vectors.  Source samples and the exact extrema of every interpolated
    cubic segment must remain inside them.  ``position_tolerance`` is a small
    absolute allowance for declared source quantisation (for example a
    float32 sample of an exact URDF limit); it is not a dynamics margin.

    ``marker_min_duration_s`` optionally gives lower bounds such as
    ``{("window_start", "window_end"): 0.16}``.  They are enforced by slowing
    the scalar time law globally, which only lowers path speed/acceleration and
    preserves the bang-bang profile's sign inside the interval.  It never makes
    a marker interval constant-speed or byte-identical to the source.  The
    bound must pass twice: first on continuous marker times, then on complete
    output-grid control intervals from ``ceil(start_frame)`` through
    ``floor(end_frame)``.  Thus a 0.021 s request at 50 Hz requires two complete
    control intervals (three inclusive samples), not one nearest-rounded tick.

    ``nonnegative_acceleration_until_marker`` optionally names one marker whose
    exact path position is inserted into the scalar grid.  After the ordinary
    rest-to-rest profile is solved, its prefix through that node is replaced by
    the greatest nondecreasing minorant (the suffix minimum of speed squared).
    This guarantees piecewise scalar ``sddot >= 0`` through cells ending at the
    marker under this conservative discrete kinematic model.  It does *not*
    guarantee strictly positive or uniform acceleration, increasing racket
    speed, uniform actuator torque, or torque/contact feasibility.

    ``uniform_scalar_path_acceleration_until_marker`` is an explicit-progress
    comparator only.  It constructs a strictly positive constant scalar
    path-parameter acceleration through the snapped 50 Hz marker guard, then
    follows a scaled rest-reachable suffix.  Unless the caller supplied an
    independently certified arc-length coordinate, this is not uniform
    weighted-arc-length acceleration.  It is not selected by the canonical
    compiler and makes no joint-acceleration, torque, contact, or
    time-optimality claim.

    ``exact_pointwise_caps`` selects a probe-grade solve for the digest-bound
    weighted-arc path.  Velocity/curvature caps are taken EXACTLY at collocation
    nodes and midpoints via ``WeightedArcPath.evaluate_l`` (the same evaluator the
    interval gate trusts as its probe) instead of interval sup-bounds over each
    cell.  Real 50 Hz mocap, quintic-interpolated, carries razor-thin
    sub-control-rate curvature spikes (e.g. a Hermite overshoot at a connector
    knot) whose analytic ``q_ll`` sup collapses the interval curvature cap by
    hundreds-fold even though the 50 Hz controller never traverses the spike.
    The pointwise caps ignore those sub-collocation singularities; correctness is
    then re-verified a-posteriori by ``guard_rate_multiple`` x output-rate
    sampling of the ACHIEVED trajectory (velocity is a hard limit;
    acceleration may exceed the limit up to ``1 + guard_probe_margin`` at the
    oversample rate).  The guard fails closed on any broader excursion.  This is
    a control-rate feasibility receipt, never a torque/contact/balance
    certificate, and the interval-certified ladder remains available with
    ``exact_pointwise_caps=False``.
    """

    q_source = np.asarray(q_path)
    if q_source.ndim != 2 or q_source.shape[0] < 3 or q_source.shape[1] < 1:
        raise RetimeError("q_path must have shape (samples>=3, joints>=1)")
    if not np.issubdtype(q_source.dtype, np.number):
        raise RetimeError("q_path must be numeric")
    if np.iscomplexobj(q_source):
        raise RetimeError("q_path must be real-valued")
    try:
        q_source = q_source.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetimeError("q_path must be a real-valued numeric array") from exc
    if not np.all(np.isfinite(q_source)):
        raise RetimeError("q_path must contain only finite values")
    if not np.isfinite(fps) or fps <= 0.0:
        raise RetimeError("fps must be finite and positive")
    if not isinstance(grid_subdivisions, int) or grid_subdivisions < 2:
        raise RetimeError("grid_subdivisions must be an integer >= 2")
    if not isinstance(max_sweeps, int) or max_sweeps < 1:
        raise RetimeError("max_sweeps must be a positive integer")
    if not isinstance(max_validation_iterations, int) or max_validation_iterations < 1:
        raise RetimeError("max_validation_iterations must be a positive integer")
    if not np.isfinite(validation_tolerance) or validation_tolerance < 0.0:
        raise RetimeError("validation_tolerance must be finite and non-negative")
    if not np.isfinite(position_tolerance) or position_tolerance < 0.0:
        raise RetimeError("position_tolerance must be finite and non-negative")
    if weighted_arc_evaluator_mode not in {
        "direct_exact",
        "endpoint_arc_jet_quintic_approximation",
    }:
        raise RetimeError(
            "weighted_arc_evaluator_mode must be 'direct_exact' or "
            "'endpoint_arc_jet_quintic_approximation'"
        )
    if (
        weighted_arc_path is None
        and weighted_arc_evaluator_mode != "direct_exact"
    ):
        raise RetimeError(
            "endpoint arc-jet approximation mode requires weighted_arc_path"
        )
    if exact_pointwise_caps and not (
        weighted_arc_path is not None
        and weighted_arc_evaluator_mode == "direct_exact"
    ):
        raise RetimeError(
            "exact_pointwise_caps requires a digest-bound weighted-arc path in "
            "direct_exact mode (it evaluates caps and its a-posteriori guard "
            "through WeightedArcPath.evaluate_l)"
        )
    if not isinstance(guard_rate_multiple, (int, np.integer)) or int(
        guard_rate_multiple
    ) < 1:
        raise RetimeError("guard_rate_multiple must be an integer >= 1")
    guard_rate_multiple = int(guard_rate_multiple)
    if not np.isfinite(guard_probe_margin) or guard_probe_margin < 0.0:
        raise RetimeError("guard_probe_margin must be finite and non-negative")

    joints = q_source.shape[1]
    velocity_cap = _as_limit_vector("velocity_limits", velocity_limits, joints)
    acceleration_cap = _as_limit_vector(
        "acceleration_limits", acceleration_limits, joints
    )
    position_lower, position_upper = _validate_position_limits(
        position_lower_limits, position_upper_limits, joints
    )
    numerical_position_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(position_lower))),
            float(np.max(np.abs(position_upper))),
        )
    )
    input_position_tolerance = max(
        float(position_tolerance), float(numerical_position_tolerance)
    )
    if np.any(q_source < position_lower[None, :] - input_position_tolerance) or np.any(
        q_source > position_upper[None, :] + input_position_tolerance
    ):
        raise RetimeError("q_path contains a source sample outside position limits")
    sample_index_end = float(len(q_source) - 1)
    marker_sample_positions = _validate_markers(markers, sample_index_end)
    path_jet_requested = (
        path_first_derivative is not None
        or path_second_derivative is not None
    )
    if (path_first_derivative is None) != (path_second_derivative is None):
        raise RetimeError(
            "path_first_derivative and path_second_derivative must be "
            "provided together"
        )
    if weighted_arc_path is not None and not path_jet_requested:
        raise RetimeError(
            "weighted_arc_path requires an explicit path endpoint 2-jet"
        )
    if path_progress is None:
        if path_jet_requested:
            raise RetimeError(
                "an explicit path 2-jet requires explicit path_progress"
            )
        path_nodes = np.arange(len(q_source), dtype=np.float64)
        path_progress_contract = "legacy_dense_sample_index_v1"
    else:
        raw_progress = np.asarray(path_progress)
        if np.iscomplexobj(raw_progress):
            raise RetimeError("path_progress must be real-valued")
        try:
            path_nodes = raw_progress.astype(np.float64, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RetimeError(
                "path_progress must be a real numeric vector"
            ) from exc
        if (
            path_nodes.shape != (len(q_source),)
            or not np.isfinite(path_nodes).all()
            or path_nodes[0] != 0.0
            or np.any(np.diff(path_nodes) <= 0.0)
        ):
            raise RetimeError(
                "path_progress must be finite shape (samples,), start at "
                "exact zero, and be strictly increasing"
            )
        path_nodes = np.ascontiguousarray(path_nodes)
        path_progress_contract = (
            (
                "weighted_arc_length_v1"
                if weighted_arc_evaluator_mode == "direct_exact"
                else "endpoint_arc_jet_quintic_approximation_v1"
            )
            if weighted_arc_path is not None
            else (
                "explicit_geometry_parameter_2jet_v1"
                if path_jet_requested
                else "explicit_caller_progress_v2"
            )
        )
    path_first: Optional[np.ndarray] = None
    path_second: Optional[np.ndarray] = None
    weighted_arc_receipt: Optional[Mapping[str, object]] = None
    if path_jet_requested:
        jets = []
        for name, value in (
            ("path_first_derivative", path_first_derivative),
            ("path_second_derivative", path_second_derivative),
        ):
            raw = np.asarray(value)
            if np.iscomplexobj(raw):
                raise RetimeError(f"{name} must be real-valued")
            try:
                jet = raw.astype(np.float64, copy=False)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RetimeError(
                    f"{name} must be a real numeric array"
                ) from exc
            if jet.shape != q_source.shape or not np.all(np.isfinite(jet)):
                raise RetimeError(
                    f"{name} must be finite with shape {q_source.shape}"
                )
            jets.append(np.ascontiguousarray(jet))
        path_first, path_second = jets
        endpoint_tangent_norm = np.linalg.norm(
            path_first[[0, -1]], axis=1
        )
        if np.any(
            endpoint_tangent_norm
            <= _REGULARITY_EPS * max(1.0, float(np.max(np.abs(q_source))))
        ):
            raise RetimeError(
                "explicit path 2-jet has a nonregular endpoint tangent; "
                "physical rest must come from scalar speed, not zero geometric "
                "tangent"
            )
        if np.any(
            np.linalg.norm(path_first, axis=1)
            <= _REGULARITY_EPS * max(1.0, float(np.max(np.abs(q_source))))
        ):
            raise RetimeError(
                "explicit path 2-jet is nonregular at an internal knot"
            )
    if weighted_arc_path is not None:
        weighted_arc_receipt = _weighted_arc_length_receipt(
            weighted_arc_path,
            evaluator_mode=weighted_arc_evaluator_mode,
        )
        try:
            arc_q, arc_q_l, arc_q_ll = weighted_arc_path.evaluate_l(
                weighted_arc_path.l_knots
            )
        except (WeightedArcPathError, ValueError, TypeError) as exc:
            raise RetimeError(
                "weighted arc endpoint 2-jet evaluation failed"
            ) from exc
        exact_pairs = (
            ("path_progress", path_nodes, weighted_arc_path.l_knots),
            ("q_path", q_source, arc_q),
            ("path_first_derivative", path_first, arc_q_l),
            ("path_second_derivative", path_second, arc_q_ll),
        )
        for name, actual, expected in exact_pairs:
            if actual is None or not np.array_equal(actual, expected):
                raise RetimeError(
                    f"{name} does not exactly match the digest-bound "
                    "weighted arc endpoint 2-jet"
                )

    def marker_progress(sample_index: float) -> float:
        left = min(int(np.floor(sample_index)), len(path_nodes) - 1)
        right = min(left + 1, len(path_nodes) - 1)
        fraction = float(sample_index - left)
        return float(
            (1.0 - fraction) * path_nodes[left]
            + fraction * path_nodes[right]
        )

    marker_positions = {
        name: marker_progress(value)
        for name, value in marker_sample_positions.items()
    }
    path_end = float(path_nodes[-1])
    marker_min_durations = _validate_marker_min_durations(
        marker_min_duration_s, marker_positions
    )
    if (
        nonnegative_acceleration_until_marker is not None
        and uniform_scalar_path_acceleration_until_marker is not None
    ):
        raise RetimeError(
            "nonnegative and uniform scalar-prefix policies are mutually exclusive"
        )
    acceleration_marker_name: Optional[str] = None
    acceleration_policy_kind: Optional[str] = None
    if nonnegative_acceleration_until_marker is not None:
        if (
            not isinstance(nonnegative_acceleration_until_marker, str)
            or not nonnegative_acceleration_until_marker
        ):
            raise RetimeError(
                "nonnegative_acceleration_until_marker must be a non-empty "
                "marker name or None"
            )
        if nonnegative_acceleration_until_marker not in marker_positions:
            raise RetimeError(
                "nonnegative_acceleration_until_marker references an unknown "
                f"marker: {nonnegative_acceleration_until_marker!r}"
            )
        acceleration_marker_name = nonnegative_acceleration_until_marker
        acceleration_policy_kind = "nonnegative_scalar_prefix"
    if uniform_scalar_path_acceleration_until_marker is not None:
        if (
            not isinstance(
                uniform_scalar_path_acceleration_until_marker, str
            )
            or not uniform_scalar_path_acceleration_until_marker
        ):
            raise RetimeError(
                "uniform_scalar_path_acceleration_until_marker must be a "
                "non-empty marker name or None"
            )
        if path_progress is None:
            raise RetimeError(
                "uniform scalar-path acceleration requires explicit "
                "content-bound path_progress"
            )
        if (
            uniform_scalar_path_acceleration_until_marker
            not in marker_positions
        ):
            raise RetimeError(
                "uniform_scalar_path_acceleration_until_marker references "
                "an unknown marker: "
                f"{uniform_scalar_path_acceleration_until_marker!r}"
            )
        acceleration_marker_name = (
            uniform_scalar_path_acceleration_until_marker
        )
        acceleration_policy_kind = "uniform_scalar_prefix_comparator"
    if (
        not isinstance(_control_guard_iteration, (int, np.integer))
        or int(_control_guard_iteration) < 0
    ):
        raise RetimeError("internal control-guard iteration must be non-negative")
    _control_guard_iteration = int(_control_guard_iteration)
    if _control_guard_path_position is not None:
        if path_progress is None or acceleration_marker_name is None:
            raise RetimeError(
                "internal control guard requires explicit path progress and "
                "a scalar-prefix acceleration marker"
            )
        _control_guard_path_position = float(_control_guard_path_position)
        if (
            not np.isfinite(_control_guard_path_position)
            or _control_guard_path_position
            < marker_positions[acceleration_marker_name]
            or _control_guard_path_position > path_end
        ):
            raise RetimeError(
                "internal control guard lies outside marker-to-path-end range"
            )

    chord_norm = np.linalg.norm(np.diff(q_source, axis=0), axis=1)
    path_scale = max(1.0, float(np.max(np.abs(q_source))))
    if np.any(chord_norm <= _REGULARITY_EPS * path_scale):
        raise RetimeError(
            "q_path contains a duplicate/degenerate consecutive sample; "
            "remove stationary rows before geometric retiming"
        )

    path_evaluator_continuity_residual = {
        "max_q_jump": 0.0,
        "max_q_s_jump": 0.0,
        "max_q_ss_jump": None,
        "max_declared_2jet_residual": None,
    }
    weighted_arc_exact = (
        weighted_arc_path is not None
        and weighted_arc_evaluator_mode == "direct_exact"
    )
    weighted_arc_approximation = (
        weighted_arc_path is not None
        and weighted_arc_evaluator_mode
        == "endpoint_arc_jet_quintic_approximation"
    )
    hermite_coefficients: Optional[np.ndarray]
    if weighted_arc_exact:
        tangents = None
        hermite_coefficients = None

        def evaluate_path(
            positions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            try:
                return weighted_arc_path.evaluate_l(positions)
            except WeightedArcPathError as exc:
                raise RetimeError(
                    "digest-bound WeightedArcPath.evaluate_l failed"
                ) from exc

        direct_q, direct_q_l, direct_q_ll = evaluate_path(path_nodes)
        direct_residual = max(
            float(np.max(np.abs(direct_q - q_source))),
            float(np.max(np.abs(direct_q_l - path_first))),
            float(np.max(np.abs(direct_q_ll - path_second))),
        )
        weighted_speed_residual = float(
            np.max(
                np.abs(
                    np.linalg.norm(
                        direct_q_l
                        * weighted_arc_path.coordinate_scale[None, :],
                        axis=1,
                    )
                    - 1.0
                )
            )
        )
        direct_scale = max(
            1.0,
            float(np.max(np.abs(q_source))),
            float(np.max(np.abs(path_first))),
            float(np.max(np.abs(path_second))),
        )
        direct_tolerance = 2.0e-10 * direct_scale
        if (
            direct_residual > direct_tolerance
            or weighted_speed_residual > 2.0e-10
        ):
            raise RetimeError(
                "exact weighted-arc evaluator failed its knot or unit-speed "
                "identity"
            )
        path_evaluator_kind = "weighted_arc_path_evaluate_l_exact_v1"
        geometry_continuity_contract = (
            "C2_exact_weighted_arc_reparameterization_digest_bound"
        )
        path_evaluator_continuity_residual = {
            "max_q_jump": 0.0,
            "max_q_s_jump": 0.0,
            "max_q_ss_jump": 0.0,
            "max_declared_2jet_residual": direct_residual,
            "max_weighted_unit_speed_residual_at_knots": (
                weighted_speed_residual
            ),
            "verification_tolerance": direct_tolerance,
        }
    elif path_first is not None and path_second is not None:
        tangents = None
        hermite_coefficients = _quintic_hermite_coefficients(
            q_source,
            path_first,
            path_second,
            path_nodes,
        )

        def evaluate_path(
            positions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _eval_polynomial_path(
                hermite_coefficients, path_nodes, positions, side="right"
            )

        left_jet = _eval_polynomial_path(
            hermite_coefficients, path_nodes, path_nodes, side="left"
        )
        right_jet = _eval_polynomial_path(
            hermite_coefficients, path_nodes, path_nodes, side="right"
        )
        jump = [
            float(np.max(np.abs(left - right)))
            for left, right in zip(left_jet, right_jet)
        ]
        declared_residual = max(
            float(np.max(np.abs(right_jet[0] - q_source))),
            float(np.max(np.abs(right_jet[1] - path_first))),
            float(np.max(np.abs(right_jet[2] - path_second))),
            float(np.max(np.abs(left_jet[0] - q_source))),
            float(np.max(np.abs(left_jet[1] - path_first))),
            float(np.max(np.abs(left_jet[2] - path_second))),
        )
        jet_scale = max(
            1.0,
            float(np.max(np.abs(q_source))),
            float(np.max(np.abs(path_first))),
            float(np.max(np.abs(path_second))),
        )
        jet_tolerance = 2.0e-9 * jet_scale
        if max(*jump, declared_residual) > jet_tolerance:
            raise RetimeError(
                "quintic endpoint-2-jet reconstruction failed its C2 knot "
                "identity"
            )
        path_evaluator_kind = (
            "endpoint_arc_jet_quintic_approximation_v1"
            if weighted_arc_approximation
            else "quintic_hermite_endpoint_2jet_v1"
        )
        geometry_continuity_contract = (
            "C2_endpoint_2jet_declared_and_one_sided_knot_verified"
        )
        path_evaluator_continuity_residual = {
            "max_q_jump": jump[0],
            "max_q_s_jump": jump[1],
            "max_q_ss_jump": jump[2],
            "max_declared_2jet_residual": declared_residual,
            "verification_tolerance": jet_tolerance,
        }
    else:
        tangents = (
            _path_tangents(q_source)
            if path_progress is None
            else _path_tangents(q_source, path_nodes)
        )
        hermite_coefficients = _hermite_coefficients(
            q_source, tangents, path_nodes
        )

        def evaluate_path(
            positions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _eval_path(
                q_source, tangents, positions, path_nodes
            )

        path_evaluator_kind = "pchip_cubic_position_only_v1"
        geometry_continuity_contract = (
            "C1_PCHIP_q_and_q_s_continuous_q_ss_may_jump_at_input_knots"
        )
    path_evaluator_sha256 = _path_evaluator_digest(
        q_path=q_source,
        path_nodes=path_nodes,
        first_derivative=path_first,
        second_derivative=path_second,
        evaluator_kind=path_evaluator_kind,
        parameterization_sha256=(
            weighted_arc_path.content_sha256
            if weighted_arc_path is not None
            else None
        ),
    )
    if weighted_arc_exact:
        continuous_position_min, continuous_position_max = (
            _weighted_arc_position_range(weighted_arc_path)
        )
    else:
        assert hermite_coefficients is not None
        continuous_position_min, continuous_position_max = (
            _continuous_position_range(
                hermite_coefficients, np.diff(path_nodes)
            )
        )
    if np.any(
        continuous_position_min < position_lower - input_position_tolerance
    ) or np.any(
        continuous_position_max > position_upper + input_position_tolerance
    ):
        raise RetimeError(
            "interpolated q_path leaves position limits at a continuous cubic extremum"
        )
    grid_parts = [
        np.linspace(
            path_nodes[index],
            path_nodes[index + 1],
            grid_subdivisions + 1,
            dtype=np.float64,
        )[:-1]
        for index in range(len(path_nodes) - 1)
    ]
    path_grid = np.concatenate(
        (*grid_parts, np.asarray([path_nodes[-1]], dtype=np.float64))
    )
    base_path_grid = path_grid.copy()
    control_guard_iteration_limit = int(len(base_path_grid))
    if _CONTROL_GUARD_ITERATION_LIMIT_OVERRIDE is not None:
        control_guard_iteration_limit = int(
            _CONTROL_GUARD_ITERATION_LIMIT_OVERRIDE
        )
        if control_guard_iteration_limit < 0:
            raise RetimeError(
                "control-guard iteration-limit override must be non-negative"
            )
    acceleration_marker_index: Optional[int] = None
    acceleration_marker_inserted = False
    acceleration_guard_index: Optional[int] = None
    acceleration_guard_inserted = False
    acceleration_guard_position: Optional[float] = None
    if acceleration_marker_name is not None:
        path_grid, acceleration_marker_index, acceleration_marker_inserted = (
            _insert_exact_grid_node(
                path_grid,
                marker_positions[acceleration_marker_name],
            )
        )
        acceleration_guard_position = (
            marker_positions[acceleration_marker_name]
            if _control_guard_path_position is None
            else _control_guard_path_position
        )
        (
            path_grid,
            acceleration_guard_index,
            acceleration_guard_inserted,
        ) = _insert_exact_grid_node(path_grid, acceleration_guard_position)
    path_mid = 0.5 * (path_grid[:-1] + path_grid[1:])
    q_grid_node, q_s_node, q_ss_node_selected = evaluate_path(path_grid)
    if weighted_arc_exact:
        q_ss_node_left = np.array(q_ss_node_selected, copy=True)
        q_ss_node_right = np.array(q_ss_node_selected, copy=True)
        weighted_speed_node_residual = float(
            np.max(
                np.abs(
                    np.linalg.norm(
                        q_s_node
                        * weighted_arc_path.coordinate_scale[None, :],
                        axis=1,
                    )
                    - 1.0
                )
            )
        )
        if weighted_speed_node_residual > 2.0e-10:
            raise RetimeError(
                "direct weighted-arc node evaluation lost unit-speed identity"
            )
    else:
        assert hermite_coefficients is not None
        q_ss_node_left, q_ss_node_right = (
            _eval_path_node_second_derivative_sides(
                hermite_coefficients, path_nodes, path_grid
            )
        )
    if not np.allclose(
        q_ss_node_selected,
        q_ss_node_right,
        rtol=5e-11,
        atol=5e-11,
    ):
        raise RetimeError(
            "one-sided node q_ss convention disagrees with path evaluator"
        )
    q_grid_mid, q_s_mid, q_ss_mid = evaluate_path(path_mid)
    if np.any(np.linalg.norm(q_s_mid, axis=1) <= _REGULARITY_EPS * path_scale):
        raise RetimeError("q_path is not a regular curve (zero path tangent)")

    node_caps = _scalar_caps(
        q_s_node, q_ss_node_right, velocity_cap, acceleration_cap
    )
    if weighted_arc_exact and not exact_pointwise_caps:
        # Interval-certified sup-bound envelope (safety-grade ladder).
        envelope_q_s, envelope_q_ss = (
            _weighted_arc_formal_segment_solver_envelope(
                weighted_arc_path, path_grid
            )
        )
        segment_caps = _scalar_caps(
            envelope_q_s,
            envelope_q_ss,
            velocity_cap,
            acceleration_cap,
        )
    else:
        # Exact pointwise caps at collocation nodes/midpoints from the
        # digest-bound evaluator (probe-grade path, and the legacy non-weighted
        # branch).  Sub-collocation curvature spikes -- e.g. quintic-Hermite
        # overshoot at a connector knot from real mocap -- are NOT sup-bounded
        # here; the a-posteriori guard below verifies the achieved trajectory.
        envelope_q_s, envelope_q_ss = q_s_mid, q_ss_mid
        segment_caps = _scalar_caps(
            q_s_mid, q_ss_mid, velocity_cap, acceleration_cap
        )
    # A node must satisfy the two adjacent segment caps as well.
    node_caps[:-1] = np.minimum(node_caps[:-1], segment_caps)
    node_caps[1:] = np.minimum(node_caps[1:], segment_caps)
    speed_sq, sweeps_used = _forward_backward_profile(
        path_grid,
        envelope_q_s,
        envelope_q_ss,
        node_caps,
        segment_caps,
        acceleration_cap,
        max_sweeps,
    )
    ordinary_speed_sq = speed_sq.copy()
    uniform_prefix_acceleration_unscaled: Optional[float] = None
    uniform_suffix_speed_sq_scale: Optional[float] = None
    if acceleration_guard_index is not None:
        if acceleration_policy_kind == "nonnegative_scalar_prefix":
            speed_sq = _greatest_nondecreasing_minorant_until(
                ordinary_speed_sq,
                acceleration_guard_index,
            )
            policy_name = "nonnegative-acceleration projection"
        elif acceleration_policy_kind == "uniform_scalar_prefix_comparator":
            prefix_progress = path_grid[: acceleration_guard_index + 1]
            positive = prefix_progress > _EPS
            if not np.any(positive):
                raise RetimeError(
                    "uniform scalar-prefix guard has zero path length"
                )
            acceleration_candidates = (
                ordinary_speed_sq[: acceleration_guard_index + 1][positive]
                / (2.0 * prefix_progress[positive])
            )
            uniform_prefix_acceleration_unscaled = float(
                np.min(acceleration_candidates)
            )
            if (
                not np.isfinite(uniform_prefix_acceleration_unscaled)
                or uniform_prefix_acceleration_unscaled <= 0.0
            ):
                raise RetimeError(
                    "uniform scalar-prefix acceleration is not positive"
                )
            speed_sq = ordinary_speed_sq.copy()
            speed_sq[: acceleration_guard_index + 1] = (
                2.0
                * uniform_prefix_acceleration_unscaled
                * prefix_progress
            )
            ordinary_guard_speed_sq = float(
                ordinary_speed_sq[acceleration_guard_index]
            )
            if ordinary_guard_speed_sq <= 0.0:
                raise RetimeError(
                    "uniform scalar-prefix guard has no recoverable speed"
                )
            uniform_suffix_speed_sq_scale = float(
                speed_sq[acceleration_guard_index]
                / ordinary_guard_speed_sq
            )
            if not 0.0 < uniform_suffix_speed_sq_scale <= 1.0 + 1e-12:
                raise RetimeError(
                    "uniform scalar-prefix suffix scale is outside (0,1]"
                )
            uniform_suffix_speed_sq_scale = min(
                1.0, uniform_suffix_speed_sq_scale
            )
            speed_sq[acceleration_guard_index:] = (
                ordinary_speed_sq[acceleration_guard_index:]
                * uniform_suffix_speed_sq_scale
            )
            speed_sq[acceleration_guard_index] = (
                2.0
                * uniform_prefix_acceleration_unscaled
                * path_grid[acceleration_guard_index]
            )
            policy_name = "uniform scalar-prefix comparator"
        else:  # pragma: no cover - validation above closes the state space
            raise RetimeError("unknown scalar-prefix acceleration policy")
        _reject_zero_speed_bottleneck(
            path_grid,
            speed_sq,
            policy_name=policy_name,
        )
    time_knots, segment_accel = _profile_time_knots(path_grid, speed_sq)
    if acceleration_guard_index is not None:
        prefix_accel = segment_accel[:acceleration_guard_index]
        if np.any(prefix_accel < -1e-12):
            raise RetimeError(
                "scalar-prefix policy left a negative scalar "
                "acceleration before its marker"
            )
        if acceleration_policy_kind == "uniform_scalar_prefix_comparator":
            uniform_error = float(
                np.max(
                    np.abs(
                        prefix_accel
                        - float(uniform_prefix_acceleration_unscaled)
                    )
                )
            )
            if uniform_error > 1e-10 * max(
                1.0, float(uniform_prefix_acceleration_unscaled)
            ):
                raise RetimeError(
                    "uniform scalar-prefix profile is not constant on every "
                    "guard cell"
                )
    base_duration = float(time_knots[-1])
    marker_time_base = {
        name: _time_at_path_position(
            position, path_grid, speed_sq, time_knots, segment_accel
        )
        for name, position in marker_positions.items()
    }

    time_scale = 1.0
    marker_duration_base: Dict[Tuple[str, str], float] = {}
    for pair, minimum_duration in marker_min_durations.items():
        start_name, end_name = pair
        start_time = marker_time_base[start_name]
        end_time = marker_time_base[end_name]
        base_interval = float(end_time - start_time)
        if base_interval <= _EPS:
            raise RetimeError(f"marker interval {pair!r} has zero traversal time")
        marker_duration_base[pair] = base_interval
        time_scale = max(time_scale, minimum_duration / base_interval)

    validation_iterations = 0
    accepted = None
    prepared_continuous_cell_peaks: _ContinuousCellPeaks | None = None
    for iteration in range(max_validation_iterations):
        validation_iterations = iteration + 1
        required_duration_before_output_quantization = (
            base_duration * time_scale
        )
        intervals_out = max(2, int(np.ceil(base_duration * time_scale * fps - 1e-12)))
        duration = intervals_out / float(fps)
        time_scale = duration / base_duration
        output_time = np.arange(intervals_out + 1, dtype=np.float64) / float(fps)
        s_out, sdot_out, sddot_out = _sample_scalar_profile(
            output_time,
            time_scale,
            path_grid,
            speed_sq,
            time_knots,
            segment_accel,
        )
        q_out, q_s_out, q_ss_out = evaluate_path(s_out)
        qdot_out = q_s_out * sdot_out[:, None]
        qddot_out = (
            q_ss_out * (sdot_out * sdot_out)[:, None]
            + q_s_out * sddot_out[:, None]
        )
        q_out[0] = q_source[0]
        q_out[-1] = q_source[-1]
        qdot_out[0] = 0.0
        qdot_out[-1] = 0.0

        fd_velocity = np.diff(q_out, axis=0) * float(fps)
        fd_acceleration = np.diff(qdot_out, axis=0) * float(fps)
        analytic_velocity_ratio = float(
            np.max(np.abs(qdot_out) / velocity_cap[None, :])
        )
        analytic_acceleration_ratio = float(
            np.max(np.abs(qddot_out) / acceleration_cap[None, :])
        )
        fd_velocity_ratio = float(
            np.max(np.abs(fd_velocity) / velocity_cap[None, :])
        )
        fd_acceleration_ratio = float(
            np.max(np.abs(fd_acceleration) / acceleration_cap[None, :])
        )
        if exact_pointwise_caps:
            # Probe-grade path: the interval sup-bound gate is intentionally not
            # used to drive slowdown -- it bounds sub-collocation curvature
            # singularities the 50 Hz controller never experiences.  Feasibility
            # is enforced at the control rate by the analytic/finite-difference
            # output-tick ratios above and, after acceptance, by the exact
            # a-posteriori guard sampling the achieved trajectory.
            continuous_velocity_ratio = 0.0
            continuous_acceleration_ratio = 0.0
        else:
            if prepared_continuous_cell_peaks is None:
                if weighted_arc_exact:
                    prepared_continuous_cell_peaks = (
                        _weighted_arc_continuous_cell_peaks(
                            weighted_arc_path,
                            path_grid,
                            speed_sq,
                            segment_accel,
                        )
                    )
                else:
                    assert hermite_coefficients is not None
                    prepared_continuous_cell_peaks = _continuous_cell_peaks(
                        hermite_coefficients,
                        path_grid,
                        speed_sq,
                        segment_accel,
                        path_nodes=path_nodes,
                    )
            continuous_velocity_ratio, continuous_acceleration_ratio = (
                _continuous_cell_ratios_from_peaks(
                    prepared_continuous_cell_peaks,
                    velocity_cap,
                    acceleration_cap,
                    time_scale,
                )
            )
        worst = max(
            analytic_velocity_ratio,
            analytic_acceleration_ratio,
            fd_velocity_ratio,
            fd_acceleration_ratio,
            continuous_velocity_ratio,
            continuous_acceleration_ratio,
        )
        discrete_markers_pass, discrete_marker_status = (
            _discrete_marker_duration_status(
                marker_time_base,
                marker_min_durations,
                time_scale,
                fps,
            )
        )
        accepted = (
            output_time,
            s_out,
            sdot_out,
            sddot_out,
            q_out,
            q_s_out,
            q_ss_out,
            qdot_out,
            qddot_out,
            analytic_velocity_ratio,
            analytic_acceleration_ratio,
            fd_velocity_ratio,
            fd_acceleration_ratio,
            continuous_velocity_ratio,
            continuous_acceleration_ratio,
            discrete_marker_status,
            required_duration_before_output_quantization,
        )
        if worst <= 1.0 + validation_tolerance and discrete_markers_pass:
            break
        if worst > 1.0 + validation_tolerance:
            required_slowdown = max(
                analytic_velocity_ratio,
                fd_velocity_ratio,
                continuous_velocity_ratio,
                np.sqrt(max(analytic_acceleration_ratio, 0.0)),
                np.sqrt(max(fd_acceleration_ratio, 0.0)),
                np.sqrt(max(continuous_acceleration_ratio, 0.0)),
            )
            if not np.isfinite(required_slowdown) or required_slowdown <= 1.0:
                raise RetimeError(
                    "finite-difference validation produced no safe slowdown"
                )
            time_scale *= required_slowdown * 1.001
        else:
            time_scale = _next_discrete_safe_time_scale(
                marker_time_base,
                marker_min_durations,
                base_duration,
                fps,
                intervals_out,
            )
    else:
        raise RetimeError(
            "velocity/acceleration/discrete-marker validation did not converge"
        )

    assert accepted is not None
    (
        output_time,
        s_out,
        sdot_out,
        sddot_out,
        q_out,
        q_s_out,
        q_ss_out,
        qdot_out,
        qddot_out,
        analytic_velocity_ratio,
        analytic_acceleration_ratio,
        fd_velocity_ratio,
        fd_acceleration_ratio,
        continuous_velocity_ratio,
        continuous_acceleration_ratio,
        discrete_marker_status,
        required_duration_before_output_quantization,
    ) = accepted

    dt = 1.0 / float(fps)
    segment_path_speed = np.diff(s_out) / dt
    segment_path_acceleration = np.diff(sdot_out) / dt
    output_acceleration_policy = None
    if acceleration_marker_name is not None:
        acceleration_marker_time = (
            marker_time_base[acceleration_marker_name] * time_scale
        )
        explicit_control_guard = path_progress is not None
        if explicit_control_guard:
            guard_boundary_frame = control_tick_at_or_after(
                acceleration_marker_time, fps
            )
            if (
                guard_boundary_frame < 0
                or guard_boundary_frame >= len(s_out)
            ):
                raise RetimeError(
                    "50 Hz control-guard boundary lies outside the output cycle"
                )
            required_guard_position = float(s_out[guard_boundary_frame])
            current_guard_position = float(acceleration_guard_position)
            guard_tolerance = (
                128.0
                * np.finfo(np.float64).eps
                * max(1.0, abs(path_end))
            )
            if required_guard_position > current_guard_position + guard_tolerance:
                if _control_guard_iteration >= control_guard_iteration_limit:
                    raise _ControlGuardConvergenceError(
                        "50 Hz no-brake control guard did not converge"
                    )
                snap_index = int(
                    np.searchsorted(
                        base_path_grid,
                        required_guard_position,
                        side="left",
                    )
                )
                if (
                    snap_index < len(base_path_grid)
                    and base_path_grid[snap_index]
                    <= current_guard_position + guard_tolerance
                ):
                    snap_index += 1
                if snap_index >= len(base_path_grid):
                    raise _ControlGuardConvergenceError(
                        "50 Hz no-brake control guard cannot advance "
                        "before the path end"
                    )
                snapped_guard_position = float(base_path_grid[snap_index])
                return _retime_path_impl(
                    q_source,
                    velocity_cap,
                    acceleration_cap,
                    position_lower_limits=position_lower,
                    position_upper_limits=position_upper,
                    path_progress=path_nodes,
                    path_first_derivative=path_first,
                    path_second_derivative=path_second,
                    weighted_arc_path=weighted_arc_path,
                    weighted_arc_evaluator_mode=weighted_arc_evaluator_mode,
                    fps=fps,
                    markers=marker_sample_positions,
                    marker_min_duration_s=marker_min_durations,
                    nonnegative_acceleration_until_marker=(
                        acceleration_marker_name
                        if acceleration_policy_kind
                        == "nonnegative_scalar_prefix"
                        else None
                    ),
                    uniform_scalar_path_acceleration_until_marker=(
                        acceleration_marker_name
                        if acceleration_policy_kind
                        == "uniform_scalar_prefix_comparator"
                        else None
                    ),
                    grid_subdivisions=grid_subdivisions,
                    max_sweeps=max_sweeps,
                    max_validation_iterations=max_validation_iterations,
                    validation_tolerance=validation_tolerance,
                    position_tolerance=position_tolerance,
                    exact_pointwise_caps=exact_pointwise_caps,
                    guard_rate_multiple=guard_rate_multiple,
                    guard_probe_margin=guard_probe_margin,
                    _control_guard_path_position=snapped_guard_position,
                    _control_guard_iteration=_control_guard_iteration + 1,
                    _include_collocation_trace=_include_collocation_trace,
                )
            output_intervals_checked = (
                output_time[:-1] < acceleration_marker_time - 1e-12
            )
            output_interval_policy = (
                "every_interval_starting_before_exact_marker_must_be_"
                "nonnegative_including_the_straddling_interval"
            )
        else:
            # Preserve the pre-existing dense-sample-index API exactly.
            output_intervals_checked = (
                output_time[1:] <= acceleration_marker_time + 1e-12
            )
            guard_boundary_frame = None
            required_guard_position = float(
                marker_positions[acceleration_marker_name]
            )
            current_guard_position = required_guard_position
            output_interval_policy = (
                "legacy_only_intervals_ending_at_or_before_exact_marker"
            )
        output_acceleration_min = (
            float(
                np.min(
                    segment_path_acceleration[
                        output_intervals_checked
                    ]
                )
            )
            if np.any(output_intervals_checked)
            else None
        )
        if (
            output_acceleration_min is not None
            and output_acceleration_min < -1e-9
        ):
            raise RetimeError(
                "50 Hz validation found negative scalar acceleration before "
                "or in a control interval overlapping the selected marker"
            )
        prefix_accel = segment_accel[: int(acceleration_guard_index)]
        output_acceleration_policy = {
            "enabled": True,
            "policy_kind": acceleration_policy_kind,
            "marker": acceleration_marker_name,
            "marker_source_index": float(
                marker_sample_positions[acceleration_marker_name]
            ),
            "marker_path_progress": float(
                marker_positions[acceleration_marker_name]
            ),
            "marker_grid_index": int(acceleration_marker_index),
            "marker_was_inserted_into_grid": bool(acceleration_marker_inserted),
            "grid_node_is_exact_marker": bool(
                path_grid[int(acceleration_marker_index)]
                == marker_positions[acceleration_marker_name]
            ),
            "control_guard_enabled": bool(explicit_control_guard),
            "control_guard_iteration": int(_control_guard_iteration),
            "control_guard_max_iterations": int(
                control_guard_iteration_limit
            ),
            "control_guard_grid_contract": (
                "required_right_boundary_progress_snapped_up_to_the_current_"
                "immutable_content_grid;_each_iteration_advances_at_least_one_"
                "grid_node_and_fails_closed_at_the_grid_node_count"
            ),
            "control_guard_boundary_frame": guard_boundary_frame,
            "control_guard_path_progress": current_guard_position,
            "required_control_boundary_path_progress": (
                required_guard_position
            ),
            "guard_grid_index": int(acceleration_guard_index),
            "guard_was_inserted_into_grid": bool(
                acceleration_guard_inserted
            ),
            "grid_node_is_exact_guard": bool(
                path_grid[int(acceleration_guard_index)]
                == current_guard_position
            ),
            "ordinary_profile_changed": bool(
                not np.array_equal(speed_sq, ordinary_speed_sq)
            ),
            "prefix_scalar_acceleration_min_continuous": (
                float(np.min(prefix_accel)) / (time_scale * time_scale)
                if len(prefix_accel)
                else None
            ),
            "prefix_scalar_acceleration_min_50hz": output_acceleration_min,
            "output_interval_policy": output_interval_policy,
            "zero_acceleration_cells_through_guard": int(
                np.sum(np.abs(prefix_accel) <= 1e-12)
            ),
            "zero_acceleration_cells_before_marker": int(
                np.sum(np.abs(prefix_accel) <= 1e-12)
            ),
        }
        if acceleration_policy_kind == "uniform_scalar_prefix_comparator":
            physical_uniform_acceleration = float(
                uniform_prefix_acceleration_unscaled
            ) / (time_scale * time_scale)
            output_acceleration_policy.update(
                {
                    "constant_scalar_path_acceleration_unscaled": float(
                        uniform_prefix_acceleration_unscaled
                    ),
                    "constant_scalar_path_acceleration": (
                        physical_uniform_acceleration
                    ),
                    "constant_scalar_path_acceleration_units": (
                        "weighted_arc_length_per_second_squared"
                        if weighted_arc_exact
                        else (
                            "endpoint_arc_jet_quintic_approximation_"
                            "coordinate_per_second_squared"
                            if weighted_arc_approximation
                            else (
                            "geometry_parameter_per_second_squared"
                            if path_first is not None
                            else "caller_progress_units_per_second_squared"
                            )
                        )
                    ),
                    "prefix_scalar_acceleration_max_abs_error_continuous": (
                        float(
                            np.max(
                                np.abs(
                                    prefix_accel
                                    / (time_scale * time_scale)
                                    - physical_uniform_acceleration
                                )
                            )
                        )
                        if len(prefix_accel)
                        else None
                    ),
                    "strictly_positive_on_every_guard_cell": bool(
                        len(prefix_accel)
                        and np.all(prefix_accel > 0.0)
                    ),
                    "cruise_cells_through_guard": int(
                        np.sum(np.abs(prefix_accel) <= 1e-12)
                    ),
                    "suffix_speed_squared_scale_from_ordinary_profile": (
                        float(uniform_suffix_speed_sq_scale)
                    ),
                    "suffix_recovery_contract": (
                        "ordinary_rest_to_rest_backward_reachable_suffix_"
                        "scaled_down_in_speed_squared_from_the_guard"
                    ),
                    "guarantee": (
                        "constant strictly positive scalar weighted-arc "
                        if weighted_arc_exact
                        else (
                            "constant strictly positive scalar endpoint-arc-"
                            "jet approximation "
                            if weighted_arc_approximation
                            else (
                                "constant strictly positive scalar "
                                "path-parameter "
                            )
                        )
                    )
                    + (
                        "acceleration on every cell through the snapped 50 Hz "
                        "guard, followed by a validated rest recovery"
                    ),
                    "non_guarantees": [
                        "does not guarantee uniform joint acceleration",
                        "does not guarantee uniform actuator torque",
                        "does not guarantee increasing task-space racket speed",
                        "does not provide torque, contact, balance, or learnability",
                    ]
                    + (
                        []
                        if weighted_arc_exact
                        else [
                            (
                                "endpoint-arc-jet approximation acceleration "
                                "is not exact weighted-arc-length acceleration"
                                if weighted_arc_approximation
                                else (
                                    "geometry-parameter acceleration is not "
                                    "continuous weighted-arc-length acceleration"
                                )
                            )
                        ]
                    ),
                }
            )
        else:
            output_acceleration_policy.update(
                {
                    "guarantee": (
                        "piecewise scalar sddot >= 0 through every cell ending "
                        "at the snapped 50 Hz guard under the conservative "
                        "fixed-grid kinematic model"
                    ),
                    "non_guarantees": [
                        "does not guarantee strictly positive scalar acceleration",
                        "does not guarantee uniform scalar or joint acceleration",
                        "does not guarantee increasing task-space racket speed",
                        "does not guarantee uniform actuator torque",
                        "does not provide torque, contact, balance, or recovery feasibility",
                    ],
                }
            )
    else:
        output_acceleration_policy = {
            "enabled": False,
            "marker": None,
        }
    marker_map: Dict[str, MarkerMapping] = {}
    for name, position in marker_positions.items():
        marker_time = marker_time_base[name] * time_scale
        fractional_frame = marker_time * float(fps)
        output_frame = int(np.clip(np.rint(fractional_frame), 0, len(q_out) - 1))
        marker_map[name] = MarkerMapping(
            source_index=marker_sample_positions[name],
            time_s=float(marker_time),
            output_fractional_frame=float(fractional_frame),
            output_frame=output_frame,
            path_position_at_frame=float(s_out[output_frame]),
        )
    marker_duration_report = {}
    for pair, minimum_duration in marker_min_durations.items():
        start_name, end_name = pair
        actual_duration = marker_map[end_name].time_s - marker_map[start_name].time_s
        if actual_duration + 1e-12 < minimum_duration:
            raise RetimeError(
                f"marker interval {pair!r} missed its minimum duration after sampling"
            )
        marker_duration_report[f"{start_name}->{end_name}"] = {
            "minimum_s": minimum_duration,
            "base_s": marker_duration_base[pair],
            "actual_s": actual_duration,
            **discrete_marker_status[pair],
        }

    exact_pointwise_guard_receipt: Optional[dict] = None
    if exact_pointwise_caps:
        exact_pointwise_guard_receipt = _exact_pointwise_aposteriori_guard(
            evaluate_path=evaluate_path,
            time_scale=time_scale,
            path_grid=path_grid,
            speed_sq=speed_sq,
            time_knots=time_knots,
            segment_accel=segment_accel,
            duration=float(output_time[-1]),
            fps=fps,
            velocity_cap=velocity_cap,
            acceleration_cap=acceleration_cap,
            guard_rate_multiple=guard_rate_multiple,
            guard_probe_margin=guard_probe_margin,
            velocity_tolerance=validation_tolerance,
        )

    report = {
        "algorithm": "shape_preserving_pchip_forward_backward_scalar_path",
        "constraint_model": "kinematic_velocity_and_acceleration_only",
        "torque_semantics": (
            "joint kinematic acceleration is not uniform actuator torque; "
            "inverse-dynamics torque remains a downstream gate"
        ),
        "marker_policy": (
            "observe_only_no_window_lock"
            if acceleration_marker_name is None
            else (
                "selected_marker_nonnegative_scalar_acceleration_no_pose_lock"
                if acceleration_policy_kind == "nonnegative_scalar_prefix"
                else "uniform_scalar_path_acceleration_prefix_comparator"
            )
        ),
        "marker_output_frame_policy": (
            "nearest_sample_observation_only_not_interval_gate"
        ),
        "marker_interval_discrete_policy": (
            "inclusive_samples_ceil_start_floor_end"
        ),
        "fps": float(fps),
        "input_samples": int(len(q_source)),
        "joints": int(joints),
        "output_frames": int(len(q_out)),
        "duration_s": float(output_time[-1]),
        "base_continuous_duration_s": base_duration,
        "required_continuous_duration_before_output_quantization_s": float(
            required_duration_before_output_quantization
        ),
        "time_scale": float(time_scale),
        "grid_subdivisions": int(grid_subdivisions),
        "exact_pointwise_caps": {
            "enabled": bool(exact_pointwise_caps),
            "solver": (
                "exact_node_and_midpoint_caps_via_evaluate_l"
                if exact_pointwise_caps
                else "interval_certified_sup_bound_ladder"
            ),
            "aposteriori_guard": exact_pointwise_guard_receipt,
        },
        "path_progress": {
            "contract": path_progress_contract,
            "explicit": path_progress is not None,
            "units": (
                "weighted_arc_length"
                if weighted_arc_exact
                else (
                    "endpoint_arc_jet_quintic_approximation_coordinate"
                    if weighted_arc_approximation
                    else (
                    "geometry_declared_dimensionless_parameter"
                    if path_first is not None
                    else "caller_declared_progress_not_certified_as_"
                    "continuous_arc_length"
                    )
                )
                if path_progress is not None
                else "dense_sample_index"
            ),
            "start": float(path_nodes[0]),
            "end": float(path_nodes[-1]),
            "node_count": int(len(path_nodes)),
            "sha256_float64_le": hashlib.sha256(
                np.ascontiguousarray(path_nodes, dtype="<f8").tobytes(
                    order="C"
                )
            ).hexdigest(),
        },
        "path_evaluator": {
            "kind": path_evaluator_kind,
            "sha256_float64_le": path_evaluator_sha256,
            "derivative_inputs_consumed": path_first is not None,
            "continuity_contract": geometry_continuity_contract,
            "continuity_residual": path_evaluator_continuity_residual,
            "parameterization_claim": (
                "weighted_arc_length_v1_digest_bound_direct_evaluate_l"
                if weighted_arc_exact
                else (
                    "endpoint_arc_jet_quintic_approximation_not_exact_"
                    "weighted_arc_length"
                    if weighted_arc_approximation
                    else (
                    "geometry_parameter_not_continuous_weighted_arc_length"
                    if path_first is not None
                    else "caller_progress_only_not_independently_certified"
                    )
                )
            ),
        },
        "weighted_arc_length": (
            dict(weighted_arc_receipt)
            if weighted_arc_receipt is not None
            else {"enabled": False, "contract": None}
        ),
        "forward_backward_sweeps": int(sweeps_used),
        "validation_iterations": int(validation_iterations),
        "start_speed": float(np.max(np.abs(qdot_out[0]))),
        "end_speed": float(np.max(np.abs(qdot_out[-1]))),
        "continuous_position_min": continuous_position_min.tolist(),
        "continuous_position_max": continuous_position_max.tolist(),
        "position_lower_limits": position_lower.tolist(),
        "position_upper_limits": position_upper.tolist(),
        "position_tolerance": float(input_position_tolerance),
        "max_ratio": {
            "analytic_velocity": analytic_velocity_ratio,
            "analytic_acceleration": analytic_acceleration_ratio,
            "finite_difference_velocity": fd_velocity_ratio,
            "finite_difference_acceleration": fd_acceleration_ratio,
            "continuous_cell_velocity": continuous_velocity_ratio,
            "continuous_cell_acceleration": continuous_acceleration_ratio,
        },
        "markers": {
            name: {
                "source_index": mapping.source_index,
                "time_s": mapping.time_s,
                "output_fractional_frame": mapping.output_fractional_frame,
                "output_frame": mapping.output_frame,
                "path_position_at_frame": mapping.path_position_at_frame,
                "path_progress": float(marker_positions[name]),
            }
            for name, mapping in marker_map.items()
        },
        "marker_min_duration_s": marker_duration_report,
        "nonnegative_acceleration_until_marker": (
            output_acceleration_policy
            if acceleration_policy_kind == "nonnegative_scalar_prefix"
            else {"enabled": False, "marker": None}
        ),
        "uniform_scalar_path_acceleration_prefix_comparator": (
            output_acceleration_policy
            if acceleration_policy_kind
            == "uniform_scalar_prefix_comparator"
            else {"enabled": False, "marker": None}
        ),
        "collocation_trace": {
            "enabled": bool(_include_collocation_trace),
            "storage": "in_memory_only_not_schema2_or_bank_artifact",
            "path_evaluator_kind": path_evaluator_kind,
            "path_evaluator_sha256_float64_le": path_evaluator_sha256,
            "weighted_arc_length_receipt_sha256": (
                None
                if weighted_arc_receipt is None
                else weighted_arc_receipt["receipt_sha256"]
            ),
            "geometry_continuity_contract": geometry_continuity_contract,
            "node_second_derivative_contract": (
                (
                    "exact_weighted_arc_q_ll_is_unique_and_left_equals_right"
                    if weighted_arc_exact
                    else (
                        "q_ss_node_left_is_lower_s_limit;_q_ss_node_right_is_"
                        "higher_s_limit;_cell_i_uses_right_i_at_start_and_"
                        "left_i_plus_1_at_end"
                    )
                )
            ),
            "c2_one_sided_equality_required": path_first is not None,
        },
        "limitations": [
            "geometric path is fixed; this module does not solve path shape",
            "kinematic joint acceleration is not actuator torque or uniform torque",
            "nonnegative scalar acceleration is not strict or uniform acceleration",
            "uniform scalar-path acceleration is not uniform joint acceleration",
            *(
                []
                if weighted_arc_exact
                else [
                    (
                        "endpoint-arc-jet approximation is a warm-start "
                        "comparator, not exact weighted arc length"
                        if weighted_arc_approximation
                        else (
                            "geometry-parameter acceleration is not continuous "
                            "weighted-arc-length acceleration"
                        )
                    )
                ]
            ),
            "nonnegative scalar acceleration does not imply increasing racket speed",
            "inverse dynamics remains mandatory for any torque-feasibility claim",
            "collision, balance, contact, and MuJoCo replay require downstream gates",
        ],
    }
    collocation_trace = (
        _build_collocation_trace(
            path_grid=path_grid,
            path_mid=path_mid,
            q_node=q_grid_node,
            q_s_node=q_s_node,
            q_ss_node_left=q_ss_node_left,
            q_ss_node_right=q_ss_node_right,
            q_mid=q_grid_mid,
            q_s_mid=q_s_mid,
            q_ss_mid=q_ss_mid,
            speed_sq=speed_sq,
            segment_accel=segment_accel,
            time_knots=time_knots,
            time_scale=time_scale,
            output_time=output_time,
            s_out=s_out,
            sdot_out=sdot_out,
            sddot_out=sddot_out,
            q_out=q_out,
            q_s_out=q_s_out,
            q_ss_out=q_ss_out,
            qdot_out=qdot_out,
            qddot_out=qddot_out,
            path_progress_contract=path_progress_contract,
            path_evaluator_kind=path_evaluator_kind,
            path_evaluator_sha256=path_evaluator_sha256,
            geometry_continuity_contract=geometry_continuity_contract,
            grid_subdivisions=grid_subdivisions,
            weighted_arc_length_receipt=weighted_arc_receipt,
        )
        if _include_collocation_trace
        else None
    )
    return RetimeResult(
        q=q_out,
        qdot=qdot_out,
        path_position=s_out,
        path_speed=segment_path_speed,
        path_acceleration=segment_path_acceleration,
        markers=marker_map,
        report=report,
        collocation_trace=collocation_trace,
    )


def _control_tick_signature(
    result: RetimeResult,
    *,
    fps: float,
) -> tuple[tuple[tuple[str, int], ...], int]:
    return (
        tuple(
            (name, control_tick_at_or_after(mapping.time_s, fps))
            for name, mapping in sorted(result.markers.items())
        ),
        control_tick_at_or_after(float(result.report["duration_s"]), fps),
    )


def _retime_exact_pointwise(
    q_path: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    grid_subdivisions: int,
    fps: float,
    common: dict,
) -> RetimeResult:
    """Probe-grade acceptance for the exact-pointwise-cap solve.

    The candidate is the solve on the caller's ``grid_subdivisions`` (the base
    grid).  It already fails closed if its exact a-posteriori guard finds a
    control-rate velocity/acceleration violation, which is the acceptance gate.

    Unlike the interval-certified ladder, the exact pointwise caps are evaluated
    at the collocation nodes; near a *sub-collocation* curvature spike (a
    razor-thin quintic-Hermite overshoot from real 50 Hz mocap) whether a node
    lands on the spike is grid-placement dependent, so refining the solve grid is
    ill-posed -- the cycle duration oscillates instead of converging.  The base
    grid is therefore the contract; a single ``2x`` solve is run only as an
    informational stability probe and never gates acceptance.  For grid-stable
    convergence the spike must be removed upstream (source smoothing), reported
    separately.
    """

    tolerance_s = float(_EXACT_POINTWISE_DURATION_TOLERANCE_TICKS) / float(fps)
    base = _retime_path_impl(
        q_path,
        velocity_limits,
        acceleration_limits,
        grid_subdivisions=int(grid_subdivisions),
        _include_collocation_trace=True,
        **common,
    )
    base_duration = float(base.report["duration_s"])
    refined_duration: Optional[float] = None
    refined_error: Optional[str] = None
    try:
        refined = _retime_path_impl(
            q_path,
            velocity_limits,
            acceleration_limits,
            grid_subdivisions=int(grid_subdivisions) * 2,
            _include_collocation_trace=False,
            **common,
        )
        refined_duration = float(refined.report["duration_s"])
    except RetimeError as exc:
        refined_error = str(exc)
    converged = (
        None
        if refined_duration is None
        else bool(abs(refined_duration - base_duration) <= tolerance_s)
    )
    base.report["exact_pointwise_caps"]["grid_stability_probe"] = {
        "criterion": (
            "abs(cycle-duration change) <= "
            f"{_EXACT_POINTWISE_DURATION_TOLERANCE_TICKS} control tick vs a 2x "
            "grid (informational only; exact pointwise caps are grid-placement "
            "sensitive near sub-collocation curvature spikes and this probe does "
            "not gate acceptance)"
        ),
        "base_grid_subdivisions": int(grid_subdivisions),
        "base_duration_s": base_duration,
        "refined_grid_subdivisions": int(grid_subdivisions) * 2,
        "refined_duration_s": refined_duration,
        "refined_error": refined_error,
        "stable": converged,
    }
    return base


def retime_path(
    q_path: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    position_lower_limits: np.ndarray,
    position_upper_limits: np.ndarray,
    path_progress: Optional[np.ndarray] = None,
    path_first_derivative: Optional[np.ndarray] = None,
    path_second_derivative: Optional[np.ndarray] = None,
    weighted_arc_path: Optional[WeightedArcPath] = None,
    weighted_arc_evaluator_mode: str = "direct_exact",
    fps: float = 50.0,
    markers: Optional[Mapping[str, float]] = None,
    marker_min_duration_s: Optional[Mapping[Tuple[str, str], float]] = None,
    nonnegative_acceleration_until_marker: Optional[str] = None,
    uniform_scalar_path_acceleration_until_marker: Optional[str] = None,
    grid_subdivisions: int = 12,
    max_sweeps: int = 100,
    max_validation_iterations: int = 12,
    validation_tolerance: float = 1e-6,
    position_tolerance: float = 1e-7,
    exact_pointwise_caps: bool = False,
    guard_rate_multiple: int = 4,
    guard_probe_margin: float = 1.0,
) -> RetimeResult:
    """Public retimer with a canonical-only fixed-grid convergence gate.

    Legacy callers that omit ``path_progress`` execute one unchanged dense-row
    solve and return no collocation trace.  Explicit-progress callers execute
    all six preregistered refinements from ``N`` through ``32N`` and only return
    after the configured consecutive exact controller-tick signatures and
    their continuous times converge.  The accepted finest solve retains an
    immutable in-memory collocation trace.  This is a convergence certificate
    for a fixed-grid candidate, not a claim of a continuous time optimum.
    """

    common = dict(
        position_lower_limits=position_lower_limits,
        position_upper_limits=position_upper_limits,
        path_progress=path_progress,
        path_first_derivative=path_first_derivative,
        path_second_derivative=path_second_derivative,
        weighted_arc_path=weighted_arc_path,
        weighted_arc_evaluator_mode=weighted_arc_evaluator_mode,
        fps=fps,
        markers=markers,
        marker_min_duration_s=marker_min_duration_s,
        nonnegative_acceleration_until_marker=(
            nonnegative_acceleration_until_marker
        ),
        uniform_scalar_path_acceleration_until_marker=(
            uniform_scalar_path_acceleration_until_marker
        ),
        max_sweeps=max_sweeps,
        max_validation_iterations=max_validation_iterations,
        validation_tolerance=validation_tolerance,
        position_tolerance=position_tolerance,
        exact_pointwise_caps=exact_pointwise_caps,
        guard_rate_multiple=guard_rate_multiple,
        guard_probe_margin=guard_probe_margin,
    )
    if path_progress is None:
        return _retime_path_impl(
            q_path,
            velocity_limits,
            acceleration_limits,
            grid_subdivisions=grid_subdivisions,
            **common,
        )

    if exact_pointwise_caps:
        return _retime_exact_pointwise(
            q_path,
            velocity_limits,
            acceleration_limits,
            grid_subdivisions=grid_subdivisions,
            fps=fps,
            common=common,
        )

    levels = int(_EXPLICIT_GRID_REFINEMENT_LEVELS)
    stable_required = int(_EXPLICIT_GRID_STABLE_LEVELS_REQUIRED)
    minimum_levels_before_accept = int(
        _EXPLICIT_GRID_MIN_LEVELS_BEFORE_ACCEPT
    )
    if (
        levels < 3
        or stable_required < 3
        or minimum_levels_before_accept < 3
        or stable_required > levels
        or minimum_levels_before_accept > levels
    ):
        raise RetimeError(
            "explicit grid-refinement contract requires at least three levels "
            "and at least three consecutive stable signatures"
        )
    receipts = []
    recent_successes = []
    time_tolerance_s = (
        float(_EXPLICIT_GRID_TIME_TOLERANCE_TICKS) / float(fps)
    )
    if not np.isfinite(time_tolerance_s) or time_tolerance_s <= 0.0:
        raise RetimeError(
            "explicit grid-refinement time tolerance must be positive"
        )
    final_candidate = None
    for level in range(levels):
        final_candidate = None
        subdivisions = int(grid_subdivisions) * (2**level)
        try:
            result = _retime_path_impl(
                q_path,
                velocity_limits,
                acceleration_limits,
                grid_subdivisions=subdivisions,
                _include_collocation_trace=(level == levels - 1),
                **common,
            )
        except _ControlGuardConvergenceError as exc:
            receipts.append(
                {
                    "level": level,
                    "grid_subdivisions": subdivisions,
                    "status": "INCONCLUSIVE_CONTROL_GUARD",
                    "error": str(exc),
                }
            )
            recent_successes = []
            continue
        signature = _control_tick_signature(result, fps=fps)
        receipt = {
            "level": level,
            "grid_subdivisions": subdivisions,
            "status": "SOLVED",
            "marker_times_s": {
                name: float(mapping.time_s)
                for name, mapping in sorted(result.markers.items())
            },
            "cycle_duration_s": float(result.report["duration_s"]),
            "base_continuous_duration_s": float(
                result.report["base_continuous_duration_s"]
            ),
            "required_continuous_duration_before_output_quantization_s": (
                float(
                    result.report[
                        "required_continuous_duration_before_output_"
                        "quantization_s"
                    ]
                )
            ),
            "marker_control_ticks": {
                name: tick for name, tick in signature[0]
            },
            "cycle_control_ticks": signature[1],
        }
        receipts.append(receipt)
        if (
            recent_successes
            and recent_successes[-1]["signature"] == signature
        ):
            recent_successes.append(
                {"signature": signature, "receipt": receipt}
            )
        else:
            recent_successes = [
                {"signature": signature, "receipt": receipt}
            ]
        if len(recent_successes) > stable_required:
            recent_successes.pop(0)
        signature_stable = (
            len(recent_successes) == stable_required
        )
        time_span_s = math.inf
        marker_time_spans = {}
        marker_tick_boundary_margins = {}
        base_duration_span_s = math.inf
        required_duration_span_s = math.inf
        required_duration_tick_boundary_margin_s = 0.0
        boundary_safe = False
        if signature_stable:
            recent_receipts = [
                item["receipt"] for item in recent_successes
            ]
            time_keys = tuple(recent_receipts[-1]["marker_times_s"])
            marker_time_spans = {
                name: (
                    max(
                        row["marker_times_s"][name]
                        for row in recent_receipts
                    )
                    - min(
                        row["marker_times_s"][name]
                        for row in recent_receipts
                    )
                )
                for name in time_keys
            }
            base_duration_span_s = (
                max(
                    row["base_continuous_duration_s"]
                    for row in recent_receipts
                )
                - min(
                    row["base_continuous_duration_s"]
                    for row in recent_receipts
                )
            )
            required_duration_span_s = (
                max(
                    row[
                        "required_continuous_duration_before_"
                        "output_quantization_s"
                    ]
                    for row in recent_receipts
                )
                - min(
                    row[
                        "required_continuous_duration_before_"
                        "output_quantization_s"
                    ]
                    for row in recent_receipts
                )
            )
            time_span_s = max(
                (*marker_time_spans.values(), required_duration_span_s),
                default=0.0,
            )
            final_marker_times = recent_receipts[-1]["marker_times_s"]
            marker_tick_boundary_margins = {
                name: abs(
                    time_s * float(fps)
                    - float(np.rint(time_s * float(fps)))
                )
                / float(fps)
                for name, time_s in final_marker_times.items()
            }
            required_duration = recent_receipts[-1][
                "required_continuous_duration_before_output_quantization_s"
            ]
            required_duration_tick_boundary_margin_s = abs(
                required_duration * float(fps)
                - float(np.rint(required_duration * float(fps)))
            ) / float(fps)
            numerical_time_tolerance = (
                128.0
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    max(final_marker_times.values(), default=0.0),
                    required_duration,
                )
            )
            boundary_safe = all(
                span <= numerical_time_tolerance
                or marker_tick_boundary_margins[name]
                > span + numerical_time_tolerance
                for name, span in marker_time_spans.items()
            ) and (
                required_duration_span_s <= numerical_time_tolerance
                or required_duration_tick_boundary_margin_s
                > required_duration_span_s + numerical_time_tolerance
            )
        if (
            level + 1 >= minimum_levels_before_accept
            and signature_stable
            and time_span_s <= time_tolerance_s
            and boundary_safe
        ):
            final_candidate = (
                result,
                signature,
                subdivisions,
                time_span_s,
                marker_time_spans,
                base_duration_span_s,
                required_duration_span_s,
                marker_tick_boundary_margins,
                required_duration_tick_boundary_margin_s,
            )
    if final_candidate is not None:
        (
            result,
            signature,
            subdivisions,
            time_span_s,
            marker_time_spans,
            base_duration_span_s,
            required_duration_span_s,
            marker_tick_boundary_margins,
            required_duration_tick_boundary_margin_s,
        ) = final_candidate
        report = dict(result.report)
        report["grid_refinement"] = {
            "enabled": True,
            "contract": (
                "explicit_progress_fixed_grid_empirical_convergence_v1"
            ),
            "minimum_levels": 3,
            "minimum_levels_before_accept": minimum_levels_before_accept,
            "all_preregistered_levels_evaluated": True,
            "stable_levels_required": stable_required,
            "continuous_time_tolerance_s": time_tolerance_s,
            "continuous_time_tolerance_control_ticks": float(
                _EXPLICIT_GRID_TIME_TOLERANCE_TICKS
            ),
            "accepted_max_marker_or_required_duration_time_span_s": float(
                time_span_s
            ),
            "accepted_marker_time_spans_s": marker_time_spans,
            "accepted_base_continuous_duration_span_s": float(
                base_duration_span_s
            ),
            "accepted_required_continuous_duration_span_s": float(
                required_duration_span_s
            ),
            "accepted_marker_tick_boundary_margins_s": (
                marker_tick_boundary_margins
            ),
            "accepted_required_duration_tick_boundary_margin_s": float(
                required_duration_tick_boundary_margin_s
            ),
            "tick_boundary_margin_exceeds_observed_time_error": True,
            "levels_evaluated": len(receipts),
            "accepted_grid_subdivisions": subdivisions,
            "tick_signature": {
                "markers": dict(signature[0]),
                "cycle": signature[1],
            },
            "receipts": receipts,
            "optimality_claim": (
                "fixed_grid_candidate_only_not_continuous_time_optimal"
            ),
        }
        return replace(result, report=report)
    inconclusive_count = sum(
        row.get("status") == "INCONCLUSIVE_CONTROL_GUARD"
        for row in receipts
    )
    detail = (
        "; no-brake control guard did not converge at "
        f"{inconclusive_count}/{len(receipts)} refinement levels"
        if inconclusive_count
        else ""
    )
    try:
        import json as _dbg_json

        _dbg = _dbg_json.dumps(receipts, default=str)[:4000]
    except Exception:  # pragma: no cover - diagnostics only
        _dbg = "<receipts not serializable>"
    raise RetimeError(
        "explicit path retiming control ticks did not converge across "
        f"grid refinements{detail}; DEBUG_RECEIPTS={_dbg}"
    )


__all__ = [
    "MarkerMapping",
    "RetimeError",
    "RetimeResult",
    "ScalarPathCollocationTrace",
    "control_tick_at_or_after",
    "retime_path",
]
