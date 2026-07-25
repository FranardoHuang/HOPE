#!/usr/bin/env python3
"""Sampling-independent weighted arc length for canonical C2 path knots.

This module gives one geometric path a canonical scalar coordinate

    l(s) = integral ||W dq/ds|| ds

where ``W`` is the caller-supplied positive ``coordinate_scale`` vector.
Adjacent knot 2-jets are joined by the unique quintic Hermite polynomial.  The
integral is evaluated directly on those polynomials with adaptive dual-order
Gauss-Legendre quadrature; visualization or output sampling density is not an
input.

The result is geometry only.  ``q_l`` and ``q_ll`` are derivatives with
respect to weighted arc length, not physical velocity/acceleration.  This
module does not assign time, optimize a time law, evaluate torque, or make
claims about collision, balance, contact, or executability.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


ALGORITHM_ID = (
    "canonical_weighted_arc_path:"
    "endpoint_2jet_quintic:"
    "adaptive_gauss_legendre_8_16:"
    "bernstein_regularity_v1"
)
_LOW_ORDER = 8
_HIGH_ORDER = 16
_LOW_NODES, _LOW_WEIGHTS = np.polynomial.legendre.leggauss(_LOW_ORDER)
_HIGH_NODES, _HIGH_WEIGHTS = np.polynomial.legendre.leggauss(_HIGH_ORDER)
_FLOAT_EPS = np.finfo(np.float64).eps


class WeightedArcPathError(ValueError):
    """The knot path is malformed, non-regular, or cannot be certified."""


@dataclass(frozen=True)
class SegmentArcAudit:
    """Numerical evidence for one accepted quintic segment.

    ``quadrature_error_estimate`` is the accumulated absolute difference
    between the order-8 and order-16 rules over accepted leaves.  It is an
    auditable numerical estimate, not a formal interval-arithmetic bound.
    Regularity is separately fail-closed by a Bernstein lower-bound
    certificate.
    """

    segment_index: int
    s_start: float
    s_end: float
    length: float
    quadrature_error_estimate: float
    quadrature_leaf_count: int
    quadrature_evaluation_count: int
    quadrature_max_depth_used: int
    observed_min_weighted_speed_per_s: float
    certified_min_weighted_speed_per_s: float
    regularity_leaf_count: int
    regularity_max_depth_used: int


@dataclass(frozen=True)
class _RegularityAudit:
    observed_min_speed: float
    certified_min_speed: float
    leaf_count: int
    max_depth_used: int


@dataclass(frozen=True)
class _QuadratureAudit:
    value: float
    error_estimate: float
    leaf_count: int
    evaluation_count: int
    max_depth_used: int


def _real_f64_array(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise WeightedArcPathError(
            f"{name} must be {ndim}-D, got shape {raw.shape}"
        )
    if raw.size == 0:
        raise WeightedArcPathError(f"{name} must be non-empty")
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise WeightedArcPathError(f"{name} must contain real numeric values")
    out = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(out)):
        raise WeightedArcPathError(f"{name} contains NaN or infinity")
    return out


def _positive_float(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise WeightedArcPathError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WeightedArcPathError(
            f"{name} must be a finite real number"
        ) from exc
    if not np.isfinite(result) or (
        result < 0.0 if allow_zero else result <= 0.0
    ):
        relation = ">=" if allow_zero else ">"
        raise WeightedArcPathError(
            f"{name} must be finite and {relation} 0, got {value!r}"
        )
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise WeightedArcPathError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise WeightedArcPathError(f"{name} must be > 0")
    return result


def _quintic_coefficients_normalized(
    s_knots: np.ndarray,
    q: np.ndarray,
    q_s: np.ndarray,
    q_ss: np.ndarray,
) -> np.ndarray:
    """Ascending-power quintic coefficients in local ``u in [0, 1]``."""

    widths = np.diff(s_knots)[:, None]
    delta = q[1:] - q[:-1]
    d0 = q_s[:-1] * widths
    d1 = q_s[1:] * widths
    dd0 = q_ss[:-1] * np.square(widths)
    dd1 = q_ss[1:] * np.square(widths)
    coefficients = np.stack(
        (
            q[:-1],
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
    if not np.all(np.isfinite(coefficients)):
        raise WeightedArcPathError(
            "quintic coefficients became non-finite"
        )
    return np.ascontiguousarray(coefficients, dtype=np.float64)


def _evaluate_normalized(
    selected_coefficients: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``q``, ``dq/du``, and ``d2q/du2`` for selected segments."""

    coefficients = np.asarray(selected_coefficients, dtype=np.float64)
    local = np.asarray(u, dtype=np.float64)
    if coefficients.ndim != 3 or coefficients.shape[-1] != 6:
        raise WeightedArcPathError("invalid internal quintic coefficient shape")
    if local.shape != (coefficients.shape[0],):
        raise WeightedArcPathError("invalid internal normalized-coordinate shape")

    q = coefficients[..., 5].copy()
    for power in range(4, -1, -1):
        q = q * local[:, None] + coefficients[..., power]

    q_u = 5.0 * coefficients[..., 5]
    for power in range(4, 0, -1):
        q_u = q_u * local[:, None] + power * coefficients[..., power]

    q_uu = 20.0 * coefficients[..., 5]
    for power in range(4, 1, -1):
        q_uu = (
            q_uu * local[:, None]
            + power * (power - 1) * coefficients[..., power]
        )
    return q, q_u, q_uu


def _derivative_coefficients(coefficients: np.ndarray) -> np.ndarray:
    powers = np.arange(1, 6, dtype=np.float64)
    return coefficients[:, 1:] * powers[None, :]


def _weighted_speed_squared_power_coefficients(
    coefficients: np.ndarray,
    *,
    segment_width: float,
    coordinate_scale: np.ndarray,
) -> np.ndarray:
    """Power coefficients of ``||W dq/ds||^2`` on local ``u``."""

    q_u_coefficients = _derivative_coefficients(coefficients)
    weighted_q_s = (
        coordinate_scale[:, None] * q_u_coefficients / segment_width
    )
    speed_squared = np.zeros(9, dtype=np.float64)
    for coordinate in weighted_q_s:
        product = np.polynomial.polynomial.polymul(coordinate, coordinate)
        speed_squared[: len(product)] += product
    if not np.all(np.isfinite(speed_squared)):
        raise WeightedArcPathError(
            "weighted-speed regularity polynomial became non-finite"
        )
    return speed_squared


def _round_down(value: float) -> float:
    return float(np.nextafter(float(value), -np.inf))


def _round_up(value: float) -> float:
    return float(np.nextafter(float(value), np.inf))


def _interval_add(
    left_low: float,
    left_high: float,
    right_low: float,
    right_high: float,
) -> tuple[float, float]:
    return (
        _round_down(left_low + right_low),
        _round_up(left_high + right_high),
    )


def _interval_multiply(
    left_low: float,
    left_high: float,
    right_low: float,
    right_high: float,
) -> tuple[float, float]:
    products = (
        left_low * right_low,
        left_low * right_high,
        left_high * right_low,
        left_high * right_high,
    )
    return _round_down(min(products)), _round_up(max(products))


def _weighted_speed_squared_power_intervals(
    coefficients: np.ndarray,
    *,
    segment_width: float,
    coordinate_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward-rounded intervals for ``||W dq/ds||^2`` coefficients."""

    low = np.zeros(9, dtype=np.float64)
    high = np.zeros(9, dtype=np.float64)
    for coordinate_index in range(coefficients.shape[0]):
        derivative_low = np.empty(5, dtype=np.float64)
        derivative_high = np.empty(5, dtype=np.float64)
        scale = float(coordinate_scale[coordinate_index])
        reciprocal = 1.0 / segment_width
        reciprocal_low = _round_down(reciprocal)
        reciprocal_high = _round_up(reciprocal)
        for power in range(5):
            coefficient = float(coefficients[coordinate_index, power + 1])
            derivative_factor = float(power + 1)
            term_low, term_high = _interval_multiply(
                coefficient,
                coefficient,
                derivative_factor,
                derivative_factor,
            )
            term_low, term_high = _interval_multiply(
                term_low, term_high, scale, scale
            )
            derivative_low[power], derivative_high[power] = (
                _interval_multiply(
                    term_low,
                    term_high,
                    reciprocal_low,
                    reciprocal_high,
                )
            )
        for left_power in range(5):
            for right_power in range(5):
                product_low, product_high = _interval_multiply(
                    float(derivative_low[left_power]),
                    float(derivative_high[left_power]),
                    float(derivative_low[right_power]),
                    float(derivative_high[right_power]),
                )
                destination = left_power + right_power
                low[destination], high[destination] = _interval_add(
                    float(low[destination]),
                    float(high[destination]),
                    product_low,
                    product_high,
                )
    if (
        not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(low > high)
    ):
        raise WeightedArcPathError(
            "weighted-speed interval construction became non-finite"
        )
    return low, high


def _power_intervals_to_bernstein(
    power_low: np.ndarray,
    power_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward-rounded power-to-Bernstein conversion on [0,1]."""

    low_values = np.asarray(power_low, dtype=np.float64)
    high_values = np.asarray(power_high, dtype=np.float64)
    if low_values.shape != high_values.shape or np.any(low_values > high_values):
        raise WeightedArcPathError("invalid internal polynomial intervals")
    degree = len(low_values) - 1
    out_low = np.empty(degree + 1, dtype=np.float64)
    out_high = np.empty(degree + 1, dtype=np.float64)
    # b_k = sum_{j<=k} C(k,j) / C(n,j) * a_j.
    # Python's exact integer combinations keep this deterministic and avoid a
    # dependency on scipy.special.
    import math

    for k in range(degree + 1):
        total_low = 0.0
        total_high = 0.0
        for j in range(k + 1):
            ratio = math.comb(k, j) / math.comb(degree, j)
            term_low, term_high = _interval_multiply(
                float(low_values[j]),
                float(high_values[j]),
                _round_down(ratio),
                _round_up(ratio),
            )
            total_low, total_high = _interval_add(
                total_low,
                total_high,
                term_low,
                term_high,
            )
        out_low[k] = total_low
        out_high[k] = total_high
    return out_low, out_high


def _split_bernstein_intervals_half(
    coefficient_low: np.ndarray,
    coefficient_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Outward-rounded de Casteljau interval split at one half."""

    level_low = np.asarray(coefficient_low, dtype=np.float64).copy()
    level_high = np.asarray(coefficient_high, dtype=np.float64).copy()
    count = len(level_low)
    left_low = np.empty(count, dtype=np.float64)
    left_high = np.empty(count, dtype=np.float64)
    right_low = np.empty(count, dtype=np.float64)
    right_high = np.empty(count, dtype=np.float64)
    left_low[0] = level_low[0]
    left_high[0] = level_high[0]
    right_low[-1] = level_low[-1]
    right_high[-1] = level_high[-1]
    for depth in range(1, count):
        next_low = np.empty(len(level_low) - 1, dtype=np.float64)
        next_high = np.empty(len(level_high) - 1, dtype=np.float64)
        for index in range(len(next_low)):
            sum_low, sum_high = _interval_add(
                float(level_low[index]),
                float(level_high[index]),
                float(level_low[index + 1]),
                float(level_high[index + 1]),
            )
            next_low[index], next_high[index] = _interval_multiply(
                sum_low, sum_high, 0.5, 0.5
            )
        level_low = next_low
        level_high = next_high
        left_low[depth] = level_low[0]
        left_high[depth] = level_high[0]
        right_low[-depth - 1] = level_low[-1]
        right_high[-depth - 1] = level_high[-1]
    return left_low, left_high, right_low, right_high


def _certify_segment_regularity(
    coefficients: np.ndarray,
    *,
    segment_width: float,
    coordinate_scale: np.ndarray,
    regularity_margin: float,
    max_depth: int,
) -> _RegularityAudit:
    """Fail closed unless weighted speed exceeds a positive margin everywhere."""

    speed_squared = _weighted_speed_squared_power_coefficients(
        coefficients,
        segment_width=segment_width,
        coordinate_scale=coordinate_scale,
    )
    speed_squared_low, speed_squared_high = (
        _weighted_speed_squared_power_intervals(
            coefficients,
            segment_width=segment_width,
            coordinate_scale=coordinate_scale,
        )
    )
    margin_squared = regularity_margin * regularity_margin
    margin_squared_low = _round_down(margin_squared)
    margin_squared_high = _round_up(margin_squared)

    derivative = (
        np.arange(1, len(speed_squared), dtype=np.float64)
        * speed_squared[1:]
    )
    trim_scale = max(float(np.max(np.abs(derivative))), np.finfo(float).tiny)
    last = len(derivative) - 1
    while (
        last > 0
        and abs(float(derivative[last]))
        <= 64.0 * _FLOAT_EPS * trim_scale
    ):
        last -= 1
    candidates = [0.0, 1.0]
    if last > 0:
        roots = np.roots(derivative[: last + 1][::-1])
        if not np.all(np.isfinite(roots)):
            raise WeightedArcPathError(
                "regularity stationary-point solve became non-finite"
            )
        root_tolerance = 2e-9
        for root in roots:
            if (
                abs(float(root.imag)) <= root_tolerance
                and -root_tolerance <= float(root.real) <= 1.0 + root_tolerance
            ):
                candidates.append(float(np.clip(root.real, 0.0, 1.0)))
    candidate_values = np.polynomial.polynomial.polyval(
        np.asarray(candidates, dtype=np.float64), speed_squared
    )
    if not np.all(np.isfinite(candidate_values)):
        raise WeightedArcPathError(
            "regularity stationary-point evaluation became non-finite"
        )
    observed_min_squared = float(np.min(candidate_values))
    if observed_min_squared <= margin_squared_high:
        raise WeightedArcPathError(
            "weighted path is non-regular or lacks the required strict "
            "speed margin: observed min "
            f"{np.sqrt(max(observed_min_squared, 0.0)):.17g}, "
            f"required > {regularity_margin:.17g}"
        )

    positive_low = speed_squared_low.copy()
    positive_high = speed_squared_high.copy()
    positive_low[0] = _round_down(
        float(positive_low[0]) - margin_squared_high
    )
    positive_high[0] = _round_up(
        float(positive_high[0]) - margin_squared_low
    )
    root_low, root_high = _power_intervals_to_bernstein(
        positive_low, positive_high
    )
    stack: list[tuple[np.ndarray, np.ndarray, float, float, int]] = [
        (root_low, root_high, 0.0, 1.0, 0)
    ]
    certified_lower_squared = np.inf
    leaf_count = 0
    max_depth_used = 0
    while stack:
        bernstein_low, bernstein_high, lower, upper, depth = stack.pop()
        minimum_lower = float(np.min(bernstein_low))
        maximum_upper = float(np.max(bernstein_high))
        if minimum_lower > 0.0:
            lower_squared = _round_down(
                margin_squared_low + minimum_lower
            )
            certified_lower_squared = min(
                certified_lower_squared, lower_squared
            )
            leaf_count += 1
            max_depth_used = max(max_depth_used, depth)
            continue
        if maximum_upper <= 0.0:
            raise WeightedArcPathError(
                "Bernstein certificate found a non-positive weighted-speed "
                "margin interval"
            )
        midpoint = 0.5 * (lower + upper)
        midpoint_value = float(
            np.polynomial.polynomial.polyval(
                midpoint,
                speed_squared,
            )
        )
        if (
            not np.isfinite(midpoint_value)
            or midpoint_value <= margin_squared_high
        ):
            raise WeightedArcPathError(
                "weighted path is non-regular inside a quintic segment"
            )
        if depth >= max_depth:
            raise WeightedArcPathError(
                "weighted-speed positivity could not be certified before "
                f"regularity max depth {max_depth}"
            )
        left_low, left_high, right_low, right_high = (
            _split_bernstein_intervals_half(
                bernstein_low, bernstein_high
            )
        )
        stack.append(
            (right_low, right_high, midpoint, upper, depth + 1)
        )
        stack.append(
            (left_low, left_high, lower, midpoint, depth + 1)
        )

    if not np.isfinite(certified_lower_squared):
        raise WeightedArcPathError("regularity certificate produced no leaves")
    return _RegularityAudit(
        observed_min_speed=float(np.sqrt(observed_min_squared)),
        certified_min_speed=_round_down(
            float(np.sqrt(certified_lower_squared))
        ),
        leaf_count=leaf_count,
        max_depth_used=max_depth_used,
    )


def _weighted_u_speed(
    coefficients: np.ndarray,
    coordinate_scale: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    selected = np.broadcast_to(
        coefficients[None, :, :],
        (len(u), coefficients.shape[0], coefficients.shape[1]),
    )
    _, q_u, _ = _evaluate_normalized(selected, u)
    speed = np.linalg.norm(q_u * coordinate_scale[None, :], axis=1)
    if not np.all(np.isfinite(speed)) or np.any(speed <= 0.0):
        raise WeightedArcPathError(
            "quadrature encountered non-positive weighted path speed"
        )
    return speed


def _gauss_legendre_interval(
    coefficients: np.ndarray,
    coordinate_scale: np.ndarray,
    lower: float,
    upper: float,
    *,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    half = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)
    u = center + half * nodes
    values = _weighted_u_speed(coefficients, coordinate_scale, u)
    result = half * float(np.dot(weights, values))
    if not np.isfinite(result) or result <= 0.0:
        raise WeightedArcPathError(
            "Gauss-Legendre quadrature produced a non-positive length"
        )
    return result


def _adaptive_segment_integral(
    coefficients: np.ndarray,
    coordinate_scale: np.ndarray,
    lower: float,
    upper: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_depth: int,
) -> _QuadratureAudit:
    if lower == upper:
        return _QuadratureAudit(0.0, 0.0, 0, 0, 0)
    if not (0.0 <= lower < upper <= 1.0):
        raise WeightedArcPathError("invalid internal quadrature interval")

    stack: list[tuple[float, float, float, int]] = [
        (lower, upper, absolute_tolerance, 0)
    ]
    value = 0.0
    error = 0.0
    leaves = 0
    evaluations = 0
    max_depth_used = 0
    while stack:
        left, right, local_absolute_tolerance, depth = stack.pop()
        low = _gauss_legendre_interval(
            coefficients,
            coordinate_scale,
            left,
            right,
            nodes=_LOW_NODES,
            weights=_LOW_WEIGHTS,
        )
        high = _gauss_legendre_interval(
            coefficients,
            coordinate_scale,
            left,
            right,
            nodes=_HIGH_NODES,
            weights=_HIGH_WEIGHTS,
        )
        evaluations += _LOW_ORDER + _HIGH_ORDER
        estimate = abs(high - low)
        allowed = local_absolute_tolerance + relative_tolerance * abs(high)
        if estimate <= allowed:
            value += high
            error += estimate
            leaves += 1
            max_depth_used = max(max_depth_used, depth)
            continue
        if depth >= max_depth:
            raise WeightedArcPathError(
                "weighted arc-length quadrature did not converge before "
                f"max depth {max_depth}: estimate={estimate:.17g}, "
                f"allowed={allowed:.17g}"
            )
        midpoint = 0.5 * (left + right)
        child_absolute_tolerance = 0.5 * local_absolute_tolerance
        stack.append(
            (
                midpoint,
                right,
                child_absolute_tolerance,
                depth + 1,
            )
        )
        stack.append(
            (
                left,
                midpoint,
                child_absolute_tolerance,
                depth + 1,
            )
        )
    if not np.isfinite(value) or value <= 0.0:
        raise WeightedArcPathError(
            "adaptive quadrature produced a non-positive segment length"
        )
    return _QuadratureAudit(
        value=float(value),
        error_estimate=float(error),
        leaf_count=leaves,
        evaluation_count=evaluations,
        max_depth_used=max_depth_used,
    )


def _digest_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))


def _content_digest(
    *,
    s_knots: np.ndarray,
    q: np.ndarray,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    coordinate_scale: np.ndarray,
    quintic_coefficients: np.ndarray,
    segment_lengths: np.ndarray,
    l_knots: np.ndarray,
    segment_error_estimates: np.ndarray,
    segment_audit_starts: np.ndarray,
    segment_audit_ends: np.ndarray,
    segment_audit_lengths: np.ndarray,
    segment_observed_min_speeds: np.ndarray,
    segment_certified_min_speeds: np.ndarray,
    segment_audit_indices: np.ndarray,
    segment_quadrature_leaf_counts: np.ndarray,
    segment_quadrature_evaluation_counts: np.ndarray,
    segment_quadrature_depths: np.ndarray,
    segment_regularity_leaf_counts: np.ndarray,
    segment_regularity_depths: np.ndarray,
    arc_absolute_tolerance: float,
    arc_relative_tolerance: float,
    regularity_margin: float,
    quadrature_max_depth: int,
    regularity_max_depth: int,
    inverse_absolute_tolerance: float,
    inverse_relative_tolerance: float,
    inverse_parameter_tolerance: float,
    inverse_max_iterations: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(ALGORITHM_ID.encode("ascii"))
    digest.update(b"\0")
    for name, array in (
        ("gauss_low_nodes", _LOW_NODES),
        ("gauss_low_weights", _LOW_WEIGHTS),
        ("gauss_high_nodes", _HIGH_NODES),
        ("gauss_high_weights", _HIGH_WEIGHTS),
        ("s_knots", s_knots),
        ("q", q),
        ("q_s", q_s),
        ("q_ss", q_ss),
        ("coordinate_scale", coordinate_scale),
        ("quintic_coefficients", quintic_coefficients),
        ("segment_lengths", segment_lengths),
        ("l_knots", l_knots),
        ("segment_error_estimates", segment_error_estimates),
        ("segment_audit_starts", segment_audit_starts),
        ("segment_audit_ends", segment_audit_ends),
        ("segment_audit_lengths", segment_audit_lengths),
        ("segment_observed_min_speeds", segment_observed_min_speeds),
        ("segment_certified_min_speeds", segment_certified_min_speeds),
    ):
        _digest_array(digest, name, array)
    for name, value in (
        ("arc_absolute_tolerance", arc_absolute_tolerance),
        ("arc_relative_tolerance", arc_relative_tolerance),
        ("regularity_margin", regularity_margin),
        ("inverse_absolute_tolerance", inverse_absolute_tolerance),
        ("inverse_relative_tolerance", inverse_relative_tolerance),
        ("inverse_parameter_tolerance", inverse_parameter_tolerance),
    ):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray([value], dtype="<f8").tobytes(order="C"))
    for name, value in (
        ("quadrature_max_depth", quadrature_max_depth),
        ("regularity_max_depth", regularity_max_depth),
        ("inverse_max_iterations", inverse_max_iterations),
    ):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray([value], dtype="<i8").tobytes(order="C"))
    for name, array in (
        ("segment_audit_indices", segment_audit_indices),
        ("segment_quadrature_leaf_counts", segment_quadrature_leaf_counts),
        (
            "segment_quadrature_evaluation_counts",
            segment_quadrature_evaluation_counts,
        ),
        ("segment_quadrature_depths", segment_quadrature_depths),
        ("segment_regularity_leaf_counts", segment_regularity_leaf_counts),
        ("segment_regularity_depths", segment_regularity_depths),
    ):
        values = np.ascontiguousarray(array, dtype="<i8")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes(order="C"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class WeightedArcPath:
    """An immutable quintic path parameterized by weighted arc length."""

    s_knots: np.ndarray
    q_knots: np.ndarray
    q_s_knots: np.ndarray
    q_ss_knots: np.ndarray
    coordinate_scale: np.ndarray
    segment_lengths: np.ndarray
    l_knots: np.ndarray
    segment_audits: tuple[SegmentArcAudit, ...]
    arc_absolute_tolerance: float
    arc_relative_tolerance: float
    regularity_margin: float
    quadrature_max_depth: int
    regularity_max_depth: int
    inverse_absolute_tolerance: float
    inverse_relative_tolerance: float
    inverse_parameter_tolerance: float
    inverse_max_iterations: int
    content_sha256: str
    _coefficients: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "s_knots",
            "q_knots",
            "q_s_knots",
            "q_ss_knots",
            "coordinate_scale",
            "segment_lengths",
            "l_knots",
            "_coefficients",
        ):
            value = np.array(
                getattr(self, name), dtype=np.float64, order="C", copy=True
            )
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "segment_audits", tuple(self.segment_audits))

    @property
    def dimension(self) -> int:
        return int(self.q_knots.shape[1])

    @property
    def total_length(self) -> float:
        return float(self.l_knots[-1])

    def _digest_inputs(self) -> dict[str, Any]:
        return {
            "s_knots": self.s_knots,
            "q": self.q_knots,
            "q_s": self.q_s_knots,
            "q_ss": self.q_ss_knots,
            "coordinate_scale": self.coordinate_scale,
            "quintic_coefficients": self._coefficients,
            "segment_lengths": self.segment_lengths,
            "l_knots": self.l_knots,
            "segment_error_estimates": np.asarray(
                [a.quadrature_error_estimate for a in self.segment_audits],
                dtype=np.float64,
            ),
            "segment_audit_starts": np.asarray(
                [a.s_start for a in self.segment_audits],
                dtype=np.float64,
            ),
            "segment_audit_ends": np.asarray(
                [a.s_end for a in self.segment_audits],
                dtype=np.float64,
            ),
            "segment_audit_lengths": np.asarray(
                [a.length for a in self.segment_audits],
                dtype=np.float64,
            ),
            "segment_observed_min_speeds": np.asarray(
                [a.observed_min_weighted_speed_per_s for a in self.segment_audits],
                dtype=np.float64,
            ),
            "segment_certified_min_speeds": np.asarray(
                [a.certified_min_weighted_speed_per_s for a in self.segment_audits],
                dtype=np.float64,
            ),
            "segment_audit_indices": np.asarray(
                [a.segment_index for a in self.segment_audits],
                dtype=np.int64,
            ),
            "segment_quadrature_leaf_counts": np.asarray(
                [a.quadrature_leaf_count for a in self.segment_audits],
                dtype=np.int64,
            ),
            "segment_quadrature_evaluation_counts": np.asarray(
                [a.quadrature_evaluation_count for a in self.segment_audits],
                dtype=np.int64,
            ),
            "segment_quadrature_depths": np.asarray(
                [a.quadrature_max_depth_used for a in self.segment_audits],
                dtype=np.int64,
            ),
            "segment_regularity_leaf_counts": np.asarray(
                [a.regularity_leaf_count for a in self.segment_audits],
                dtype=np.int64,
            ),
            "segment_regularity_depths": np.asarray(
                [a.regularity_max_depth_used for a in self.segment_audits],
                dtype=np.int64,
            ),
            "arc_absolute_tolerance": self.arc_absolute_tolerance,
            "arc_relative_tolerance": self.arc_relative_tolerance,
            "regularity_margin": self.regularity_margin,
            "quadrature_max_depth": self.quadrature_max_depth,
            "regularity_max_depth": self.regularity_max_depth,
            "inverse_absolute_tolerance": self.inverse_absolute_tolerance,
            "inverse_relative_tolerance": self.inverse_relative_tolerance,
            "inverse_parameter_tolerance": self.inverse_parameter_tolerance,
            "inverse_max_iterations": self.inverse_max_iterations,
        }

    def recompute_content_sha256(self) -> str:
        """Recompute the content binding from current immutable state."""

        return _content_digest(**self._digest_inputs())

    def verify_content_digest(self) -> bool:
        """Whether the stored digest still matches every bound input/result."""

        return self.content_sha256 == self.recompute_content_sha256()

    def _flatten_domain(
        self,
        value: Any,
        *,
        name: str,
        lower: float,
        upper: float,
    ) -> tuple[np.ndarray, tuple[int, ...], bool]:
        raw = np.asarray(value)
        if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
            raise WeightedArcPathError(f"{name} must be real numeric")
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise WeightedArcPathError(f"{name} contains NaN or infinity")
        tolerance = (
            128.0
            * _FLOAT_EPS
            * max(1.0, abs(lower), abs(upper))
        )
        if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
            raise WeightedArcPathError(
                f"{name} requested outside [{lower:.17g}, {upper:.17g}]"
            )
        clipped = np.clip(values, lower, upper)
        return clipped.reshape(-1), values.shape, values.ndim == 0

    def evaluate_s(
        self, s: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the exact quintic ``q``, ``q_s``, and ``q_ss``."""

        flat, shape, scalar = self._flatten_domain(
            s,
            name="s",
            lower=float(self.s_knots[0]),
            upper=float(self.s_knots[-1]),
        )
        segment = np.searchsorted(self.s_knots, flat, side="right") - 1
        segment = np.clip(segment, 0, len(self.s_knots) - 2)
        widths = np.diff(self.s_knots)[segment]
        u = (flat - self.s_knots[segment]) / widths
        q, q_u, q_uu = _evaluate_normalized(
            self._coefficients[segment], u
        )
        q_s = q_u / widths[:, None]
        q_ss = q_uu / np.square(widths[:, None])

        # Exact caller knots are part of the content contract.  Return their
        # stored 2-jets bit-for-bit rather than a rounded polynomial endpoint.
        knot_candidate = np.searchsorted(self.s_knots, flat, side="left")
        at_knot = (
            (knot_candidate < len(self.s_knots))
            & (
                self.s_knots[
                    np.minimum(knot_candidate, len(self.s_knots) - 1)
                ]
                == flat
            )
        )
        if np.any(at_knot):
            indices = knot_candidate[at_knot]
            q[at_knot] = self.q_knots[indices]
            q_s[at_knot] = self.q_s_knots[indices]
            q_ss[at_knot] = self.q_ss_knots[indices]

        output_shape = shape + (self.dimension,)
        q = q.reshape(output_shape)
        q_s = q_s.reshape(output_shape)
        q_ss = q_ss.reshape(output_shape)
        if scalar:
            return q.reshape(self.dimension), q_s.reshape(
                self.dimension
            ), q_ss.reshape(self.dimension)
        return q, q_s, q_ss

    def _partial_segment_length(
        self,
        segment: int,
        u: float,
        *,
        tighter_absolute_tolerance: float | None = None,
    ) -> float:
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return float(self.segment_lengths[segment])
        absolute_tolerance = self.arc_absolute_tolerance
        if tighter_absolute_tolerance is not None:
            absolute_tolerance = min(
                absolute_tolerance, tighter_absolute_tolerance
            )
        audit = _adaptive_segment_integral(
            self._coefficients[segment],
            self.coordinate_scale,
            0.0,
            float(u),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=self.arc_relative_tolerance,
            max_depth=self.quadrature_max_depth,
        )
        return audit.value

    def arc_length_at_s(self, s: Any) -> np.ndarray | float:
        """Map arbitrary path parameter values to cumulative weighted length."""

        flat, shape, scalar = self._flatten_domain(
            s,
            name="s",
            lower=float(self.s_knots[0]),
            upper=float(self.s_knots[-1]),
        )
        result = np.empty_like(flat)
        for index, value in enumerate(flat):
            knot = int(np.searchsorted(self.s_knots, value, side="left"))
            if knot < len(self.s_knots) and self.s_knots[knot] == value:
                result[index] = self.l_knots[knot]
                continue
            segment = int(
                np.clip(
                    np.searchsorted(self.s_knots, value, side="right") - 1,
                    0,
                    len(self.s_knots) - 2,
                )
            )
            width = self.s_knots[segment + 1] - self.s_knots[segment]
            u = (value - self.s_knots[segment]) / width
            result[index] = self.l_knots[segment] + self._partial_segment_length(
                segment, float(u)
            )
        if scalar:
            return float(result[0])
        return result.reshape(shape)

    def _u_from_local_length(self, segment: int, target: float) -> float:
        segment_length = float(self.segment_lengths[segment])
        if target <= 0.0:
            return 0.0
        if target >= segment_length:
            return 1.0
        lower = 0.0
        upper = 1.0
        u = float(np.clip(target / segment_length, 0.0, 1.0))
        length_tolerance = max(
            self.inverse_absolute_tolerance,
            self.inverse_relative_tolerance * segment_length,
            8.0
            * float(
                self.segment_audits[segment].quadrature_error_estimate
            ),
        )
        partial_absolute_tolerance = max(
            np.finfo(float).tiny, 0.05 * length_tolerance
        )
        for _ in range(self.inverse_max_iterations):
            partial = self._partial_segment_length(
                segment,
                u,
                tighter_absolute_tolerance=partial_absolute_tolerance,
            )
            residual = partial - target
            if abs(residual) <= length_tolerance:
                return float(u)
            if residual < 0.0:
                lower = u
            else:
                upper = u
            if upper - lower <= self.inverse_parameter_tolerance:
                return float(0.5 * (lower + upper))

            speed_u = float(
                _weighted_u_speed(
                    self._coefficients[segment],
                    self.coordinate_scale,
                    np.asarray([u], dtype=np.float64),
                )[0]
            )
            newton = u - residual / speed_u
            if (
                not np.isfinite(newton)
                or newton <= lower
                or newton >= upper
            ):
                u = 0.5 * (lower + upper)
            else:
                u = float(newton)
        raise WeightedArcPathError(
            "bracketed arc-length inversion did not converge within "
            f"{self.inverse_max_iterations} iterations"
        )

    def s_from_l(self, length: Any) -> np.ndarray | float:
        """Invert cumulative length with bracketed Newton/bisection."""

        flat, shape, scalar = self._flatten_domain(
            length,
            name="length",
            lower=0.0,
            upper=self.total_length,
        )
        result = np.empty_like(flat)
        for index, value in enumerate(flat):
            knot = int(np.searchsorted(self.l_knots, value, side="left"))
            if knot < len(self.l_knots) and self.l_knots[knot] == value:
                result[index] = self.s_knots[knot]
                continue
            segment = int(
                np.clip(
                    np.searchsorted(self.l_knots, value, side="right") - 1,
                    0,
                    len(self.segment_lengths) - 1,
                )
            )
            target = float(value - self.l_knots[segment])
            u = self._u_from_local_length(segment, target)
            result[index] = self.s_knots[segment] + u * (
                self.s_knots[segment + 1] - self.s_knots[segment]
            )
        if scalar:
            return float(result[0])
        return result.reshape(shape)

    def s_interval_from_l(
        self, length: Any
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Outward interval containing the true inverse ``s(l)``.

        ``s_from_l`` is a point estimate and may stop once its *length*
        residual meets the configured tolerance.  A caller performing an
        interval extremum check must not treat that point as an exact cell
        boundary.  This method converts the complete length uncertainty into
        source-parameter uncertainty using the segment's certified positive
        ``dl/ds`` lower bound, adds the possible bracket-width termination,
        and rounds both sides outward.
        """

        flat, shape, scalar = self._flatten_domain(
            length,
            name="length",
            lower=0.0,
            upper=self.total_length,
        )
        point = np.asarray(self.s_from_l(flat), dtype=np.float64).reshape(-1)
        lower = np.empty_like(point)
        upper = np.empty_like(point)
        for index, value in enumerate(flat):
            knot = int(np.searchsorted(self.l_knots, value, side="left"))
            if knot < len(self.l_knots) and self.l_knots[knot] == value:
                lower[index] = self.s_knots[knot]
                upper[index] = self.s_knots[knot]
                continue
            segment = int(
                np.clip(
                    np.searchsorted(self.l_knots, value, side="right") - 1,
                    0,
                    len(self.segment_lengths) - 1,
                )
            )
            segment_length = float(self.segment_lengths[segment])
            audit = self.segment_audits[segment]
            minimum_speed = float(
                audit.certified_min_weighted_speed_per_s
            )
            if minimum_speed <= self.regularity_margin:
                raise WeightedArcPathError(
                    "inverse interval lost its certified positive dl/ds bound"
                )
            length_tolerance = max(
                self.inverse_absolute_tolerance,
                self.inverse_relative_tolerance * segment_length,
                8.0 * float(audit.quadrature_error_estimate),
            )
            segment_width = float(
                self.s_knots[segment + 1] - self.s_knots[segment]
            )
            # Two length-tolerance units cover the accepted residual plus the
            # tighter partial-integral numerical error used by the point
            # inverse.  The second term covers termination by u-bracket width.
            padding = (
                2.0 * length_tolerance / minimum_speed
                + self.inverse_parameter_tolerance * segment_width
            )
            roundoff = (
                512.0
                * _FLOAT_EPS
                * max(
                    1.0,
                    abs(float(point[index])),
                    abs(float(self.s_knots[segment])),
                    abs(float(self.s_knots[segment + 1])),
                )
            )
            lower[index] = max(
                float(self.s_knots[segment]),
                float(
                    np.nextafter(
                        point[index] - padding - roundoff, -np.inf
                    )
                ),
            )
            upper[index] = min(
                float(self.s_knots[segment + 1]),
                float(
                    np.nextafter(
                        point[index] + padding + roundoff, np.inf
                    )
                ),
            )
            if lower[index] > upper[index]:
                raise WeightedArcPathError(
                    "inverse interval became empty after outward padding"
                )
        if scalar:
            return float(lower[0]), float(upper[0])
        return lower.reshape(shape), upper.reshape(shape)

    def evaluate_l(
        self, length: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``q``, ``dq/dl``, and ``d2q/dl2`` at weighted length."""

        s = self.s_from_l(length)
        q, q_s, q_ss = self.evaluate_s(s)
        weighted_q_s = q_s * self.coordinate_scale
        speed = np.linalg.norm(weighted_q_s, axis=-1)
        if np.any(speed <= self.regularity_margin):
            raise WeightedArcPathError(
                "stored path no longer satisfies its regularity margin"
            )
        speed_s = np.sum(
            weighted_q_s * (q_ss * self.coordinate_scale), axis=-1
        ) / speed
        q_l = q_s / np.expand_dims(speed, axis=-1)
        q_ll = (
            q_ss / np.expand_dims(np.square(speed), axis=-1)
            - q_s
            * np.expand_dims(
                speed_s / np.power(speed, 3), axis=-1
            )
        )
        return q, q_l, q_ll


def build_weighted_arc_path(
    *,
    s_knots: Any,
    q: Any,
    q_s: Any,
    q_ss: Any,
    coordinate_scale: Any,
    arc_absolute_tolerance: float = 1e-12,
    arc_relative_tolerance: float = 1e-11,
    regularity_margin: float = 1e-10,
    quadrature_max_depth: int = 20,
    regularity_max_depth: int = 24,
    inverse_absolute_tolerance: float = 2e-12,
    inverse_relative_tolerance: float = 2e-11,
    inverse_parameter_tolerance: float = 2e-13,
    inverse_max_iterations: int = 64,
) -> WeightedArcPath:
    """Build and certify a weighted-arc-length path evaluator.

    The caller must provide a complete C2 knot 2-jet.  Each segment is exactly
    the endpoint-constrained quintic; no dense path samples are accepted or
    generated as part of the geometry contract.
    """

    s_values = _real_f64_array(s_knots, name="s_knots", ndim=1)
    q_values = _real_f64_array(q, name="q", ndim=2)
    q_s_values = _real_f64_array(q_s, name="q_s", ndim=2)
    q_ss_values = _real_f64_array(q_ss, name="q_ss", ndim=2)
    scale = _real_f64_array(
        coordinate_scale, name="coordinate_scale", ndim=1
    )
    if len(s_values) < 2:
        raise WeightedArcPathError("at least two path knots are required")
    if np.any(np.diff(s_values) <= 0.0):
        raise WeightedArcPathError("s_knots must be strictly increasing")
    if q_values.shape[0] != len(s_values) or q_values.shape[1] < 1:
        raise WeightedArcPathError(
            "q must have shape (len(s_knots), positive_dimension)"
        )
    if q_s_values.shape != q_values.shape or q_ss_values.shape != q_values.shape:
        raise WeightedArcPathError(
            "q_s and q_ss must have exactly the same shape as q"
        )
    if scale.shape != (q_values.shape[1],):
        raise WeightedArcPathError(
            f"coordinate_scale must have shape ({q_values.shape[1]},)"
        )
    if np.any(scale <= 0.0):
        raise WeightedArcPathError(
            "coordinate_scale must contain strictly positive values"
        )

    arc_absolute_tolerance = _positive_float(
        arc_absolute_tolerance,
        name="arc_absolute_tolerance",
    )
    arc_relative_tolerance = _positive_float(
        arc_relative_tolerance,
        name="arc_relative_tolerance",
    )
    regularity_margin = _positive_float(
        regularity_margin,
        name="regularity_margin",
    )
    inverse_absolute_tolerance = _positive_float(
        inverse_absolute_tolerance,
        name="inverse_absolute_tolerance",
    )
    inverse_relative_tolerance = _positive_float(
        inverse_relative_tolerance,
        name="inverse_relative_tolerance",
    )
    inverse_parameter_tolerance = _positive_float(
        inverse_parameter_tolerance,
        name="inverse_parameter_tolerance",
    )
    quadrature_max_depth = _positive_integer(
        quadrature_max_depth, name="quadrature_max_depth"
    )
    regularity_max_depth = _positive_integer(
        regularity_max_depth, name="regularity_max_depth"
    )
    inverse_max_iterations = _positive_integer(
        inverse_max_iterations, name="inverse_max_iterations"
    )

    coefficients = _quintic_coefficients_normalized(
        s_values, q_values, q_s_values, q_ss_values
    )
    segment_lengths = np.empty(len(s_values) - 1, dtype=np.float64)
    audits: list[SegmentArcAudit] = []
    for segment in range(len(segment_lengths)):
        width = float(s_values[segment + 1] - s_values[segment])
        regularity = _certify_segment_regularity(
            coefficients[segment],
            segment_width=width,
            coordinate_scale=scale,
            regularity_margin=regularity_margin,
            max_depth=regularity_max_depth,
        )
        quadrature = _adaptive_segment_integral(
            coefficients[segment],
            scale,
            0.0,
            1.0,
            absolute_tolerance=arc_absolute_tolerance,
            relative_tolerance=arc_relative_tolerance,
            max_depth=quadrature_max_depth,
        )
        segment_lengths[segment] = quadrature.value
        audits.append(
            SegmentArcAudit(
                segment_index=segment,
                s_start=float(s_values[segment]),
                s_end=float(s_values[segment + 1]),
                length=quadrature.value,
                quadrature_error_estimate=quadrature.error_estimate,
                quadrature_leaf_count=quadrature.leaf_count,
                quadrature_evaluation_count=quadrature.evaluation_count,
                quadrature_max_depth_used=quadrature.max_depth_used,
                observed_min_weighted_speed_per_s=(
                    regularity.observed_min_speed
                ),
                certified_min_weighted_speed_per_s=(
                    regularity.certified_min_speed
                ),
                regularity_leaf_count=regularity.leaf_count,
                regularity_max_depth_used=regularity.max_depth_used,
            )
        )
    l_knots = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.cumsum(segment_lengths, dtype=np.float64),
        )
    )
    if (
        not np.all(np.isfinite(l_knots))
        or np.any(np.diff(l_knots) <= 0.0)
    ):
        raise WeightedArcPathError(
            "cumulative weighted arc length is not finite and increasing"
        )

    digest_inputs = {
        "s_knots": s_values,
        "q": q_values,
        "q_s": q_s_values,
        "q_ss": q_ss_values,
        "coordinate_scale": scale,
        "quintic_coefficients": coefficients,
        "segment_lengths": segment_lengths,
        "l_knots": l_knots,
        "segment_error_estimates": np.asarray(
            [a.quadrature_error_estimate for a in audits], dtype=np.float64
        ),
        "segment_audit_starts": np.asarray(
            [a.s_start for a in audits], dtype=np.float64
        ),
        "segment_audit_ends": np.asarray(
            [a.s_end for a in audits], dtype=np.float64
        ),
        "segment_audit_lengths": np.asarray(
            [a.length for a in audits], dtype=np.float64
        ),
        "segment_observed_min_speeds": np.asarray(
            [a.observed_min_weighted_speed_per_s for a in audits],
            dtype=np.float64,
        ),
        "segment_certified_min_speeds": np.asarray(
            [a.certified_min_weighted_speed_per_s for a in audits],
            dtype=np.float64,
        ),
        "segment_audit_indices": np.asarray(
            [a.segment_index for a in audits], dtype=np.int64
        ),
        "segment_quadrature_leaf_counts": np.asarray(
            [a.quadrature_leaf_count for a in audits], dtype=np.int64
        ),
        "segment_quadrature_evaluation_counts": np.asarray(
            [a.quadrature_evaluation_count for a in audits], dtype=np.int64
        ),
        "segment_quadrature_depths": np.asarray(
            [a.quadrature_max_depth_used for a in audits], dtype=np.int64
        ),
        "segment_regularity_leaf_counts": np.asarray(
            [a.regularity_leaf_count for a in audits], dtype=np.int64
        ),
        "segment_regularity_depths": np.asarray(
            [a.regularity_max_depth_used for a in audits], dtype=np.int64
        ),
        "arc_absolute_tolerance": arc_absolute_tolerance,
        "arc_relative_tolerance": arc_relative_tolerance,
        "regularity_margin": regularity_margin,
        "quadrature_max_depth": quadrature_max_depth,
        "regularity_max_depth": regularity_max_depth,
        "inverse_absolute_tolerance": inverse_absolute_tolerance,
        "inverse_relative_tolerance": inverse_relative_tolerance,
        "inverse_parameter_tolerance": inverse_parameter_tolerance,
        "inverse_max_iterations": inverse_max_iterations,
    }
    content_sha256 = _content_digest(**digest_inputs)
    return WeightedArcPath(
        s_knots=s_values,
        q_knots=q_values,
        q_s_knots=q_s_values,
        q_ss_knots=q_ss_values,
        coordinate_scale=scale,
        segment_lengths=segment_lengths,
        l_knots=l_knots,
        segment_audits=tuple(audits),
        arc_absolute_tolerance=arc_absolute_tolerance,
        arc_relative_tolerance=arc_relative_tolerance,
        regularity_margin=regularity_margin,
        quadrature_max_depth=quadrature_max_depth,
        regularity_max_depth=regularity_max_depth,
        inverse_absolute_tolerance=inverse_absolute_tolerance,
        inverse_relative_tolerance=inverse_relative_tolerance,
        inverse_parameter_tolerance=inverse_parameter_tolerance,
        inverse_max_iterations=inverse_max_iterations,
        content_sha256=content_sha256,
        _coefficients=coefficients,
    )


__all__ = [
    "ALGORITHM_ID",
    "SegmentArcAudit",
    "WeightedArcPath",
    "WeightedArcPathError",
    "build_weighted_arc_path",
]
