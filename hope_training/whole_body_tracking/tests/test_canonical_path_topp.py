"""Pure-CPU tests for the canonical geometric-path TOPP helper."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "canonical_path_topp", _SCRIPTS / "canonical_path_topp.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["canonical_path_topp"] = _MOD
_SPEC.loader.exec_module(_MOD)

retime_path = _MOD.retime_path
RetimeError = _MOD.RetimeError
from canonical_weighted_arc_path import build_weighted_arc_path  # noqa: E402
from canonical_motion_geometry import build_canonical_geometry  # noqa: E402


def _retime(
    path: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    **kwargs,
):
    """Call the strict public API with deliberately broad test joint limits."""

    joints = path.shape[1]
    return retime_path(
        path,
        velocity_limits,
        acceleration_limits,
        position_lower_limits=np.full(joints, -10.0),
        position_upper_limits=np.full(joints, 10.0),
        **kwargs,
    )


def _linear_path(samples: int = 21) -> np.ndarray:
    x = np.linspace(0.0, 1.0, samples)
    return np.column_stack((x, -0.4 * x, 0.2 * x))


def _arc_progress(path: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray([0.0]),
            np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)),
        )
    )


def _legacy_polynomial_range(
    coefficients: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    """Reference the pre-optimization scalar np.roots implementation."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    last = len(coefficients) - 1
    while (
        last > 0
        and abs(coefficients[last])
        <= 32.0 * np.finfo(float).eps * scale
    ):
        last -= 1
    points = [lower, upper]
    if last > 1:
        derivative = (
            np.arange(1, last + 1, dtype=np.float64)
            * coefficients[1 : last + 1]
        )
        roots = np.roots(derivative[::-1])
        tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
        for root in roots:
            if abs(float(root.imag)) <= tolerance:
                real = float(root.real)
                if lower - tolerance <= real <= upper + tolerance:
                    points.append(float(np.clip(real, lower, upper)))
    values = np.polynomial.polynomial.polyval(points, coefficients)
    return float(np.min(values)), float(np.max(values))


def _legacy_continuous_cell_ratios(
    hermite_coefficients: np.ndarray,
    path_grid: np.ndarray,
    speed_sq: np.ndarray,
    segment_accel: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    time_scale: float,
) -> tuple[float, float]:
    """Reference the original cell-major exact-extremum loop."""

    velocity_ratio = 0.0
    acceleration_ratio = 0.0
    path_segments = hermite_coefficients.shape[0]
    for cell in range(len(path_grid) - 1):
        start = float(path_grid[cell])
        end = float(path_grid[cell + 1])
        width = end - start
        source_segment = min(
            int(np.floor(0.5 * (start + end))), path_segments - 1
        )
        local_start = start - float(source_segment)
        scalar_accel = float(segment_accel[cell])
        scalar_speed_sq = np.array(
            [float(speed_sq[cell]), 2.0 * scalar_accel]
        )
        for joint in range(hermite_coefficients.shape[1]):
            shifted = _MOD._shift_cubic(
                hermite_coefficients[source_segment, joint], local_start
            )
            slope = np.array(
                [shifted[1], 2.0 * shifted[2], 3.0 * shifted[3]]
            )
            curvature = np.array(
                [2.0 * shifted[2], 6.0 * shifted[3]]
            )
            velocity_sq_poly = np.polynomial.polynomial.polymul(
                np.polynomial.polynomial.polymul(slope, slope),
                scalar_speed_sq,
            )
            velocity_low, velocity_high = _legacy_polynomial_range(
                velocity_sq_poly, 0.0, width
            )
            assert velocity_low >= (
                -1e-10 * max(1.0, abs(velocity_high))
            )
            joint_velocity_peak = (
                np.sqrt(max(velocity_high, 0.0)) / time_scale
            )
            velocity_ratio = max(
                velocity_ratio,
                joint_velocity_peak / velocity_limits[joint],
            )
            acceleration_poly = np.polynomial.polynomial.polyadd(
                np.polynomial.polynomial.polymul(
                    curvature, scalar_speed_sq
                ),
                scalar_accel * slope,
            )
            acceleration_low, acceleration_high = (
                _legacy_polynomial_range(
                    acceleration_poly, 0.0, width
                )
            )
            joint_acceleration_peak = max(
                abs(acceleration_low), abs(acceleration_high)
            ) / (time_scale * time_scale)
            acceleration_ratio = max(
                acceleration_ratio,
                joint_acceleration_peak / acceleration_limits[joint],
            )
    return velocity_ratio, acceleration_ratio


def _legacy_max_reachable_neighbor(
    fixed_speed_sq: float,
    neighbor_cap: float,
    segment_cap: float,
    ds: float,
    q_s_mid: np.ndarray,
    q_ss_mid: np.ndarray,
    acceleration_limits: np.ndarray,
) -> float:
    """Reference the previous 60-step monotone bisection."""

    upper = min(
        float(neighbor_cap),
        max(0.0, 2.0 * segment_cap - fixed_speed_sq),
    )
    if upper <= fixed_speed_sq:
        return max(0.0, upper)

    def feasible(candidate: float) -> bool:
        middle = 0.5 * (fixed_speed_sq + candidate)
        accel_cap = _MOD._tangential_acceleration_cap(
            q_s_mid, q_ss_mid, middle, acceleration_limits
        )
        required = (candidate - fixed_speed_sq) / (2.0 * ds)
        return (
            accel_cap >= 0.0
            and required <= accel_cap * (1.0 + 1e-12)
        )

    if feasible(upper):
        return upper
    low = fixed_speed_sq
    high = upper
    for _ in range(60):
        middle = 0.5 * (low + high)
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def test_rest_to_rest_and_uniform_50hz_output():
    result = _retime(
        _linear_path(),
        velocity_limits=np.array([1.2, 0.8, 0.8]),
        acceleration_limits=np.array([2.0, 1.5, 1.5]),
    )
    assert np.array_equal(result.q[0], _linear_path()[0])
    assert np.array_equal(result.q[-1], _linear_path()[-1])
    assert np.array_equal(result.qdot[0], np.zeros(3))
    assert np.array_equal(result.qdot[-1], np.zeros(3))
    assert result.report["fps"] == 50.0
    assert result.report["duration_s"] * 50.0 == pytest.approx(len(result.q) - 1)
    assert result.path_speed.shape == (len(result.q) - 1,)
    assert result.path_acceleration.shape == (len(result.q) - 1,)
    assert result.collocation_trace is None


def test_joint_velocity_and_acceleration_caps_are_fail_closed_verified():
    velocity = np.array([0.55, 0.25, 0.20])
    acceleration = np.array([0.75, 0.35, 0.30])
    result = _retime(
        _linear_path(31),
        velocity_limits=velocity,
        acceleration_limits=acceleration,
    )
    finite_difference_velocity = np.abs(np.diff(result.q, axis=0)) * 50.0
    finite_difference_acceleration = np.abs(np.diff(result.qdot, axis=0)) * 50.0
    assert np.all(finite_difference_velocity <= velocity[None, :] * (1.0 + 2e-6))
    assert np.all(
        finite_difference_acceleration <= acceleration[None, :] * (1.0 + 2e-6)
    )
    assert max(result.report["max_ratio"].values()) <= 1.0 + 1e-6


def test_strike_window_markers_do_not_lock_or_stop_acceleration():
    result = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.array([1.0, 1.0, 1.0]),
        markers={"window_start": 2.0, "contact": 3.0, "window_end": 4.0},
    )
    segment_midpoint = 0.5 * (
        result.path_position[:-1] + result.path_position[1:]
    )
    inside = (segment_midpoint >= 2.0) & (segment_midpoint <= 4.0)
    assert inside.any()
    assert np.all(result.path_acceleration[inside] > 0.0)
    assert result.path_speed[inside][-1] > result.path_speed[inside][0]
    assert result.report["marker_policy"] == "observe_only_no_window_lock"


def test_suffix_minimum_is_the_greatest_nondecreasing_minorant():
    ordinary = np.array([0.0, 9.0, 4.0, 7.0, 3.0, 8.0, 1.0])
    projected = _MOD._greatest_nondecreasing_minorant_until(ordinary, 5)
    assert np.array_equal(projected, np.array([0.0, 3.0, 3.0, 3.0, 3.0, 8.0, 1.0]))
    assert np.all(np.diff(projected[:6]) >= 0.0)
    assert projected[5] == ordinary[5]
    assert np.array_equal(projected[5:], ordinary[5:])

    # Exhaust the small integer lattice: every nondecreasing minorant is
    # pointwise dominated by the suffix-minimum projection.
    for values in __import__("itertools").product(range(9), repeat=4):
        candidate = np.array((0.0, *values, 8.0))
        if np.all(np.diff(candidate) >= 0.0) and np.all(
            candidate <= ordinary[:6]
        ):
            assert np.all(candidate <= projected[:6])


def test_fractional_marker_legacy_policy_preserves_exact_marker_semantics():
    marker = 16.37  # deliberately not on the 1/12-source-frame scalar grid
    kwargs = dict(
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        markers={"window_end": marker},
    )
    baseline = _retime(_linear_path(), **kwargs)
    constrained = _retime(
        _linear_path(),
        **kwargs,
        nonnegative_acceleration_until_marker="window_end",
    )
    interval_end_time = np.arange(1, len(baseline.q)) / 50.0
    baseline_before_marker = (
        interval_end_time <= baseline.markers["window_end"].time_s + 1e-12
    )
    assert np.min(baseline.path_acceleration[baseline_before_marker]) < 0.0

    constrained_interval_end_time = np.arange(1, len(constrained.q)) / 50.0
    constrained_before_marker = (
        constrained_interval_end_time
        <= constrained.markers["window_end"].time_s + 1e-12
    )
    assert constrained_before_marker.any()
    assert np.min(
        constrained.path_acceleration[constrained_before_marker]
    ) >= -1e-9
    policy = constrained.report["nonnegative_acceleration_until_marker"]
    assert policy["marker_was_inserted_into_grid"] is True
    assert policy["grid_node_is_exact_marker"] is True
    assert policy["control_guard_enabled"] is False
    assert policy["output_interval_policy"].startswith("legacy_")


def test_explicit_progress_extends_no_brake_guard_through_80_percent_tick():
    path = _linear_path()
    result = _retime(
        path,
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        path_progress=_arc_progress(path),
        markers={"window_end": 0.8 * (len(path) - 1)},
        nonnegative_acceleration_until_marker="window_end",
    )
    marker_time = result.markers["window_end"].time_s
    starts = np.arange(len(result.path_acceleration)) / 50.0
    overlapping_prefix = starts < marker_time - 1e-12
    assert overlapping_prefix.any()
    assert np.min(result.path_acceleration[overlapping_prefix]) >= -1e-9

    policy = result.report["nonnegative_acceleration_until_marker"]
    assert policy["control_guard_enabled"] is True
    assert policy["control_guard_iteration"] >= 1
    assert policy["control_guard_path_progress"] > policy["marker_path_progress"]
    assert (
        policy["control_guard_path_progress"]
        >= policy["required_control_boundary_path_progress"] - 1e-12
    )
    assert policy["grid_node_is_exact_guard"] is True
    assert "straddling_interval" in policy["output_interval_policy"]


def test_explicit_progress_resolved_line_control_ticks_are_density_invariant():
    receipts = []
    for samples in (21, 41):
        path = _linear_path(samples)
        result = _retime(
            path,
            velocity_limits=np.full(3, 20.0),
            acceleration_limits=np.ones(3),
            path_progress=_arc_progress(path),
            markers={"window_end": 0.8 * (samples - 1)},
            nonnegative_acceleration_until_marker="window_end",
        )
        receipts.append(
            (
                int(
                    np.ceil(
                        np.nextafter(
                            result.markers["window_end"].time_s * 50.0,
                            -np.inf,
                        )
                    )
                ),
                len(result.q) - 1,
                result.report["path_progress"]["contract"],
            )
        )
    assert len(set(receipts)) == 1
    assert receipts[0][2] == "explicit_caller_progress_v2"


def test_explicit_progress_fixed_curved_input_runs_every_refinement_level():
    samples = 33
    parameter = np.linspace(0.0, 1.0, samples)
    path = np.column_stack(
        (
            np.sin(0.5 * np.pi * parameter),
            0.5 * (1.0 - np.cos(0.5 * np.pi * parameter)),
        )
    )
    result = _retime(
        path,
        velocity_limits=np.ones(2),
        acceleration_limits=np.full(2, 2.0),
        path_progress=_arc_progress(path),
        markers={
            "window_start": 0.4 * (samples - 1),
            "source_anchor": 0.6 * (samples - 1),
            "window_end": 0.8 * (samples - 1),
        },
        nonnegative_acceleration_until_marker="window_end",
    )
    refinement = result.report["grid_refinement"]
    assert refinement["enabled"] is True
    assert refinement["all_preregistered_levels_evaluated"] is True
    assert refinement["levels_evaluated"] == _MOD._EXPLICIT_GRID_REFINEMENT_LEVELS
    solved = [
        row for row in refinement["receipts"] if row["status"] == "SOLVED"
    ]
    assert len(solved) >= 3
    assert len(
        {
            (
                tuple(sorted(row["marker_control_ticks"].items())),
                row["cycle_control_ticks"],
            )
            for row in solved[-3:]
        }
    ) == 1
    assert (
        refinement["accepted_max_marker_or_required_duration_time_span_s"]
        <= refinement["continuous_time_tolerance_s"]
    )
    trace = result.collocation_trace
    assert trace is not None
    assert trace.path_progress_contract == "explicit_caller_progress_v2"
    assert trace.path_evaluator_kind == "pchip_cubic_position_only_v1"
    assert len(trace.path_evaluator_sha256_float64_le) == 64
    assert trace.grid_subdivisions == refinement["accepted_grid_subdivisions"]
    assert trace.s_mid.shape == (len(trace.s_node) - 1,)
    assert trace.q_node.shape == (
        len(trace.s_node),
        path.shape[1],
    )
    assert trace.q_mid.shape == (
        len(trace.s_mid),
        path.shape[1],
    )
    assert trace.q_ss_node_left.shape == trace.q_node.shape
    assert trace.q_ss_node_right.shape == trace.q_node.shape
    assert trace.x_node.shape == trace.s_node.shape
    assert trace.u_cell.shape == trace.s_mid.shape
    np.testing.assert_allclose(
        np.diff(trace.x_node) / (2.0 * np.diff(trace.s_node)),
        trace.u_cell,
        rtol=5e-12,
        atol=1e-12,
    )
    assert np.all(trace.time_node_s[:-1] < trace.time_mid_s)
    assert np.all(trace.time_mid_s < trace.time_node_s[1:])
    assert trace.time_node_s[-1] == pytest.approx(
        result.report["duration_s"], abs=2e-12
    )
    np.testing.assert_array_equal(trace.tick_time_s, np.arange(len(result.q)) / 50.0)
    np.testing.assert_array_equal(trace.tick_s, result.path_position)
    np.testing.assert_array_equal(trace.tick_q, result.q)
    np.testing.assert_array_equal(trace.tick_qdot, result.qdot)
    np.testing.assert_allclose(
        trace.tick_qdot,
        trace.tick_q_s * np.sqrt(trace.tick_x)[:, None],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        trace.tick_qdd,
        trace.tick_q_ss * trace.tick_x[:, None]
        + trace.tick_q_s
        * trace.tick_scalar_acceleration[:, None],
        rtol=2e-13,
        atol=2e-13,
    )
    source_progress = _arc_progress(path)
    source_knot = 16
    trace_knot = np.flatnonzero(
        trace.s_node == source_progress[source_knot]
    )
    assert trace_knot.shape == (1,)
    trace_knot = int(trace_knot[0])
    cubic = _MOD._hermite_coefficients(
        path, _MOD._path_tangents(path, source_progress), source_progress
    )
    left_width = source_progress[source_knot] - source_progress[source_knot - 1]
    expected_left = (
        2.0 * cubic[source_knot - 1, :, 2]
        + 6.0 * cubic[source_knot - 1, :, 3] * left_width
    )
    expected_right = 2.0 * cubic[source_knot, :, 2]
    np.testing.assert_allclose(
        trace.q_ss_node_left[trace_knot], expected_left, atol=2e-12
    )
    np.testing.assert_allclose(
        trace.q_ss_node_right[trace_knot], expected_right, atol=2e-12
    )
    assert np.max(np.abs(expected_left - expected_right)) > 1e-5
    assert "cell_i_uses_right_i_at_start" in (
        trace.node_second_derivative_contract
    )
    assert trace.tick_cell_side[0] == "right_cell_at_knot"
    assert trace.tick_cell_side[-1] == "left_cell_at_path_end"
    assert set(np.unique(trace.tick_cell_side)) <= {
        "cell_interior",
        "right_cell_at_knot",
        "left_cell_at_path_end",
    }
    for field_name in trace.__dataclass_fields__:
        value = getattr(trace, field_name)
        if isinstance(value, np.ndarray):
            assert value.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        trace.tick_q[0, 0] = 123.0


def test_endpoint_2jet_quintic_reproduces_known_polynomial_and_is_c2():
    nodes = np.asarray([0.0, 0.19, 0.53, 0.78, 1.0])

    def exact(s):
        s = np.asarray(s, dtype=np.float64)
        q0 = 0.3 + 0.8 * s - 0.2 * s**2 + 0.1 * s**3 - 0.05 * s**4 + 0.02 * s**5
        q1 = -0.1 + 0.4 * s + 0.3 * s**2 - 0.12 * s**3 + 0.04 * s**5
        q = np.column_stack((q0, q1))
        d0 = 0.8 - 0.4 * s + 0.3 * s**2 - 0.2 * s**3 + 0.1 * s**4
        d1 = 0.4 + 0.6 * s - 0.36 * s**2 + 0.2 * s**4
        d = np.column_stack((d0, d1))
        dd0 = -0.4 + 0.6 * s - 0.6 * s**2 + 0.4 * s**3
        dd1 = 0.6 - 0.72 * s + 0.8 * s**3
        dd = np.column_stack((dd0, dd1))
        return q, d, dd

    q_node, d_node, dd_node = exact(nodes)
    coefficients = _MOD._quintic_hermite_coefficients(
        q_node, d_node, dd_node, nodes
    )
    samples = np.linspace(0.0, 1.0, 257)
    actual = _MOD._eval_polynomial_path(
        coefficients, nodes, samples
    )
    expected = exact(samples)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=2e-11, atol=2e-11
        )
    left = _MOD._eval_polynomial_path(
        coefficients, nodes, nodes, side="left"
    )
    right = _MOD._eval_polynomial_path(
        coefficients, nodes, nodes, side="right"
    )
    for left_value, right_value, declared in zip(
        left, right, (q_node, d_node, dd_node)
    ):
        np.testing.assert_allclose(left_value, right_value, atol=2e-11)
        np.testing.assert_allclose(right_value, declared, atol=2e-11)


def test_endpoint_2jet_linear_retime_is_sampling_density_invariant():
    results = []
    for count in (5, 9):
        progress = np.linspace(0.0, 1.0, count)
        path = np.column_stack((progress, 0.2 * progress))
        first = np.tile(np.asarray([1.0, 0.2]), (count, 1))
        second = np.zeros_like(first)
        result = _retime(
            path,
            velocity_limits=np.asarray([1.0, 1.0]),
            acceleration_limits=np.asarray([2.0, 2.0]),
            path_progress=progress,
            path_first_derivative=first,
            path_second_derivative=second,
            markers={"window_end": 0.75 * (count - 1)},
            nonnegative_acceleration_until_marker="window_end",
        )
        trace = result.collocation_trace
        assert trace is not None
        assert trace.path_progress_contract == (
            "explicit_geometry_parameter_2jet_v1"
        )
        assert trace.path_evaluator_kind == (
            "quintic_hermite_endpoint_2jet_v1"
        )
        np.testing.assert_allclose(
            trace.q_ss_node_left,
            trace.q_ss_node_right,
            atol=2e-11,
        )
        assert result.report["path_evaluator"]["continuity_contract"].startswith(
            "C2_"
        )
        results.append(result)
    assert results[0].markers["window_end"].time_s == pytest.approx(
        results[1].markers["window_end"].time_s, abs=2e-12
    )
    assert results[0].report["duration_s"] == pytest.approx(
        results[1].report["duration_s"], abs=2e-12
    )


def _weighted_line(count: int, parameter_scale: float):
    physical = np.linspace(0.0, 1.0, count)
    parameter = parameter_scale * physical
    direction = np.asarray([1.0, 0.2])
    q = physical[:, None] * direction[None, :]
    q_s = np.broadcast_to(
        direction / parameter_scale, q.shape
    ).copy()
    q_ss = np.zeros_like(q)
    arc = build_weighted_arc_path(
        s_knots=parameter,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        coordinate_scale=np.asarray([2.0, 0.5]),
    )
    q_l, first, second = arc.evaluate_l(arc.l_knots)
    return arc, q_l, first, second


def _real_connector_weighted_arc():
    """One actual ready->source->ready C2 connector, not a mock curve."""

    angle = np.linspace(0.0, np.pi, 9)
    source = np.column_stack((np.cos(angle), np.sin(angle)))
    ready = np.asarray([0.0, -1.0])
    geometry = build_canonical_geometry(
        source,
        ready,
        entry_frame=0,
        exit_frame=6,
        window_start=4,
        window_end=5,
        window_halo=1,
        samples_per_rad=6.0,
        min_connector_intervals=5,
        min_core_intervals=5,
        coordinate_scale=np.ones(2),
        coordinate_semantics=("joint_0", "joint_1"),
        coordinate_units=("rad", "rad"),
    )
    formal = geometry.canonical_knot_path_indices
    arc = build_weighted_arc_path(
        s_knots=geometry.path_parameter[formal],
        q=geometry.q_path[formal],
        q_s=geometry.dq_ds[formal],
        q_ss=geometry.d2q_ds2[formal],
        coordinate_scale=np.ones(2),
    )
    q, q_l, q_ll = arc.evaluate_l(arc.l_knots)
    formal_source = geometry.source_frame_map[formal]

    def marker_index(source_frame):
        matches = np.flatnonzero(formal_source == float(source_frame))
        assert matches.shape == (1,)
        return float(matches[0])

    markers = {
        "window_start": marker_index(4),
        "source_anchor": marker_index(4),
        "window_end": marker_index(5),
    }
    return arc, q, q_l, q_ll, markers


def test_real_connector_uses_exact_arc_at_nodes_midpoints_ticks_and_markers():
    arc, q, first, second, markers = _real_connector_weighted_arc()

    # The rejected endpoint-jet-in-l construction is observably a different
    # curve on the real return connector, even though it matches every knot
    # 2-jet.  It may remain an explicit comparator, never the exact contract.
    approximation_coefficients = _MOD._quintic_hermite_coefficients(
        q, first, second, arc.l_knots
    )
    probe = np.linspace(arc.l_knots[-2], arc.l_knots[-1], 257)
    exact_q, _, _ = arc.evaluate_l(probe)
    approximation_q, approximation_q_l, _ = _MOD._eval_polynomial_path(
        approximation_coefficients,
        arc.l_knots,
        probe,
        side="right",
    )
    weighted_position_error = np.max(
        np.linalg.norm(approximation_q - exact_q, axis=1)
    )
    weighted_speed_error = np.max(
        np.abs(np.linalg.norm(approximation_q_l, axis=1) - 1.0)
    )
    assert weighted_position_error > 0.05
    assert weighted_speed_error > 0.05

    exact = _MOD._retime_path_impl(
        q,
        np.full(2, 10.0),
        np.full(2, 10_000.0),
        position_lower_limits=np.full(2, -2.0),
        position_upper_limits=np.full(2, 2.0),
        path_progress=arc.l_knots,
        path_first_derivative=first,
        path_second_derivative=second,
        weighted_arc_path=arc,
        fps=50.0,
        markers=markers,
        nonnegative_acceleration_until_marker="window_end",
        grid_subdivisions=16,
        _include_collocation_trace=True,
    )
    trace = exact.collocation_trace
    assert trace is not None
    for length, actual in (
        (trace.s_node, (trace.q_node, trace.q_s_node, trace.q_ss_node_right)),
        (trace.s_mid, (trace.q_mid, trace.q_s_mid, trace.q_ss_mid)),
        (trace.tick_s, (trace.tick_q, trace.tick_q_s, trace.tick_q_ss)),
    ):
        expected = arc.evaluate_l(length)
        for observed, wanted in zip(actual, expected):
            np.testing.assert_array_equal(observed, wanted)
        np.testing.assert_allclose(
            np.linalg.norm(actual[1], axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-10,
        )
    assert trace.path_progress_contract == "weighted_arc_length_v1"
    assert trace.path_evaluator_kind == (
        "weighted_arc_path_evaluate_l_exact_v1"
    )
    receipt = exact.report["weighted_arc_length"]
    assert receipt["contract"] == "weighted_arc_length_v1"
    assert receipt["evaluator_mode"] == "direct_exact"
    assert receipt["exact_evaluator_api"] == "WeightedArcPath.evaluate_l"
    for name, formal_index in markers.items():
        assert exact.report["markers"][name]["path_progress"] == (
            arc.l_knots[int(formal_index)]
        )

    no_brake = exact.report["nonnegative_acceleration_until_marker"]
    assert no_brake["enabled"] is True
    assert no_brake["marker"] == "window_end"
    assert no_brake["control_guard_path_progress"] >= (
        arc.l_knots[int(markers["window_end"])]
    )
    assert no_brake["prefix_scalar_acceleration_min_continuous"] >= 0.0
    assert no_brake["prefix_scalar_acceleration_min_50hz"] >= -1.0e-9
    assert "straddling_interval" in no_brake["output_interval_policy"]

    approximation = _MOD._retime_path_impl(
        q,
        np.full(2, 10.0),
        np.full(2, 10_000.0),
        position_lower_limits=np.full(2, -2.0),
        position_upper_limits=np.full(2, 2.0),
        path_progress=arc.l_knots,
        path_first_derivative=first,
        path_second_derivative=second,
        weighted_arc_path=arc,
        weighted_arc_evaluator_mode=(
            "endpoint_arc_jet_quintic_approximation"
        ),
        fps=50.0,
        grid_subdivisions=4,
    )
    approximation_receipt = approximation.report["weighted_arc_length"]
    assert approximation.report["path_progress"]["contract"] == (
        "endpoint_arc_jet_quintic_approximation_v1"
    )
    assert approximation.report["path_evaluator"]["kind"] == (
        "endpoint_arc_jet_quintic_approximation_v1"
    )
    assert approximation_receipt["contract"] == (
        "endpoint_arc_jet_quintic_approximation_v1"
    )
    assert approximation_receipt["exact_direct_evaluator"] is False
    assert "exact_weighted_arc_length_parameterization" in (
        approximation_receipt["non_claims"]
    )
    assert any(
        "warm-start comparator" in row
        for row in approximation.report["limitations"]
    )


def test_weighted_arc_retime_is_s_reparameterization_and_density_invariant():
    results = []
    for count, parameter_scale in ((5, 1.0), (9, 7.25)):
        arc, q, first, second = _weighted_line(
            count, parameter_scale
        )
        marker_index = int(0.75 * (count - 1))
        result = _retime(
            q,
            velocity_limits=np.asarray([1.0, 1.0]),
            acceleration_limits=np.asarray([2.0, 2.0]),
            path_progress=arc.l_knots,
            path_first_derivative=first,
            path_second_derivative=second,
            weighted_arc_path=arc,
            markers={"window_end": float(marker_index)},
            nonnegative_acceleration_until_marker="window_end",
        )
        trace = result.collocation_trace
        assert trace is not None
        assert trace.path_progress_contract == "weighted_arc_length_v1"
        assert trace.path_evaluator_kind == (
            "weighted_arc_path_evaluate_l_exact_v1"
        )
        receipt = result.report["weighted_arc_length"]
        assert receipt["enabled"] is True
        assert receipt["content_sha256"] == arc.content_sha256
        assert receipt["digest_verified"] is True
        assert receipt["regularity_certified"] is True
        assert (
            result.report["markers"]["window_end"]["path_progress"]
            == arc.l_knots[marker_index]
        )
        assert (
            trace.weighted_arc_length_receipt["receipt_sha256"]
            == receipt["receipt_sha256"]
        )
        results.append(result)
    np.testing.assert_allclose(
        results[0].q, results[1].q, rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        results[0].qdot, results[1].qdot, rtol=0.0, atol=2e-12
    )
    assert results[0].markers["window_end"].time_s == pytest.approx(
        results[1].markers["window_end"].time_s, abs=2e-12
    )
    assert results[0].report["duration_s"] == pytest.approx(
        results[1].report["duration_s"], abs=2e-12
    )


def test_weighted_arc_binding_tamper_and_endpoint_mismatch_fail_closed():
    arc, q, first, second = _weighted_line(5, 1.0)
    common = {
        "path_progress": arc.l_knots,
        "path_first_derivative": first,
        "path_second_derivative": second,
        "weighted_arc_path": arc,
    }
    with pytest.raises(RetimeError, match="does not exactly match"):
        _retime(
            q,
            np.ones(2),
            np.ones(2),
            **{
                **common,
                "path_first_derivative": np.nextafter(first, np.inf),
            },
        )

    tampered = arc.segment_lengths.copy()
    tampered[0] = np.nextafter(tampered[0], np.inf)
    tampered.setflags(write=False)
    object.__setattr__(arc, "segment_lengths", tampered)
    with pytest.raises(RetimeError, match="content digest"):
        _retime(q, np.ones(2), np.ones(2), **common)


def test_uniform_weighted_arc_prefix_has_only_scalar_arc_semantics():
    arc, q, first, second = _weighted_line(5, 1.0)
    result = _retime(
        q,
        velocity_limits=np.asarray([1.0, 1.0]),
        acceleration_limits=np.asarray([2.0, 2.0]),
        path_progress=arc.l_knots,
        path_first_derivative=first,
        path_second_derivative=second,
        weighted_arc_path=arc,
        markers={"window_end": 3.0},
        uniform_scalar_path_acceleration_until_marker="window_end",
    )
    policy = result.report[
        "uniform_scalar_path_acceleration_prefix_comparator"
    ]
    assert policy["constant_scalar_path_acceleration_units"] == (
        "weighted_arc_length_per_second_squared"
    )
    assert "weighted-arc acceleration" in policy["guarantee"]
    assert "does not guarantee uniform actuator torque" in (
        policy["non_guarantees"]
    )


def test_endpoint_2jet_nonregular_endpoint_fails_closed():
    progress = np.asarray([0.0, 0.5, 1.0])
    path = np.power(progress, 3)[:, None]
    first = (3.0 * np.square(progress))[:, None]
    second = (6.0 * progress)[:, None]
    with pytest.raises(RetimeError, match="nonregular endpoint tangent"):
        _retime(
            path,
            velocity_limits=np.ones(1),
            acceleration_limits=np.ones(1),
            path_progress=progress,
            path_first_derivative=first,
            path_second_derivative=second,
        )


def test_explicit_progress_unstable_curved_grid_fails_closed(monkeypatch):
    samples = 17
    parameter = np.linspace(0.0, 1.0, samples)
    path = np.column_stack(
        (
            np.sin(0.5 * np.pi * parameter),
            0.5 * (1.0 - np.cos(0.5 * np.pi * parameter)),
        )
    )
    monkeypatch.setattr(_MOD, "_EXPLICIT_GRID_REFINEMENT_LEVELS", 3)
    monkeypatch.setattr(_MOD, "_EXPLICIT_GRID_STABLE_LEVELS_REQUIRED", 3)
    monkeypatch.setattr(_MOD, "_EXPLICIT_GRID_MIN_LEVELS_BEFORE_ACCEPT", 3)
    with pytest.raises(RetimeError, match="did not converge across grid"):
        _retime(
            path,
            velocity_limits=np.ones(2),
            acceleration_limits=np.full(2, 2.0),
            path_progress=_arc_progress(path),
            markers={
                "window_start": 0.4 * (samples - 1),
                "source_anchor": 0.6 * (samples - 1),
                "window_end": 0.8 * (samples - 1),
            },
            nonnegative_acceleration_until_marker="window_end",
            grid_subdivisions=2,
        )


def test_explicit_progress_rejects_tick_stable_but_time_drifting_grids(
    monkeypatch,
):
    calls = []

    def drifting_impl(q_path, *args, grid_subdivisions, **kwargs):
        del args, kwargs
        calls.append(grid_subdivisions)
        level = len(calls) - 1
        marker_time = 0.101 + 0.003 * level
        cycle_time = 0.301 + 0.003 * level
        mapping = _MOD.MarkerMapping(
            source_index=1.0,
            time_s=marker_time,
            output_fractional_frame=marker_time * 50.0,
            output_frame=6,
            path_position_at_frame=1.0,
        )
        path = np.asarray(q_path, dtype=np.float64)
        return _MOD.RetimeResult(
            q=path,
            qdot=np.zeros_like(path),
            path_position=np.linspace(0.0, 1.0, len(path)),
            path_speed=np.zeros(len(path) - 1),
            path_acceleration=np.zeros(len(path) - 1),
            markers={"window_end": mapping},
            report={
                "duration_s": cycle_time,
                "base_continuous_duration_s": cycle_time - 0.001,
                "required_continuous_duration_before_output_quantization_s": (
                    cycle_time - 0.0005
                ),
            },
        )

    monkeypatch.setattr(_MOD, "_retime_path_impl", drifting_impl)
    monkeypatch.setattr(_MOD, "_EXPLICIT_GRID_REFINEMENT_LEVELS", 3)
    monkeypatch.setattr(_MOD, "_EXPLICIT_GRID_STABLE_LEVELS_REQUIRED", 3)
    monkeypatch.setattr(_MOD, "_EXPLICIT_GRID_MIN_LEVELS_BEFORE_ACCEPT", 3)
    path = _linear_path(3)
    with pytest.raises(RetimeError, match="did not converge across grid"):
        _retime(
            path,
            velocity_limits=np.ones(3),
            acceleration_limits=np.ones(3),
            path_progress=_arc_progress(path),
            markers={"window_end": 1.0},
        )
    assert calls == [12, 24, 48]


def test_explicit_progress_marker_exactly_on_tick_needs_no_guard_extension():
    path = _linear_path()
    progress = _arc_progress(path)
    exact_frame = 30
    # For this straight scaled-arc path, the conservative scalar acceleration
    # is exactly progress[-1] and t=0.6 s maps to source index 3.6.
    marker_source_index = 3.6
    result = _retime(
        path,
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        path_progress=progress,
        markers={"window_end": marker_source_index},
        nonnegative_acceleration_until_marker="window_end",
    )
    assert result.markers["window_end"].time_s * 50.0 == pytest.approx(
        exact_frame, abs=2e-12
    )
    policy = result.report["nonnegative_acceleration_until_marker"]
    assert policy["control_guard_iteration"] == 0
    assert policy["control_guard_boundary_frame"] == exact_frame
    assert policy["control_guard_path_progress"] == pytest.approx(
        policy["required_control_boundary_path_progress"], abs=1e-12
    )


def test_explicit_progress_control_guard_iteration_limit_fails_closed(monkeypatch):
    path = _linear_path()
    monkeypatch.setattr(
        _MOD, "_CONTROL_GUARD_ITERATION_LIMIT_OVERRIDE", 0
    )
    with pytest.raises(RetimeError, match="control guard did not converge"):
        _retime(
            path,
            velocity_limits=np.full(3, 20.0),
            acceleration_limits=np.ones(3),
            path_progress=_arc_progress(path),
            markers={"window_end": 0.8 * (len(path) - 1)},
            nonnegative_acceleration_until_marker="window_end",
        )


def test_grid_aligned_marker_is_not_rounded_or_reinserted():
    result = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        markers={"window_end": 4.0},
        nonnegative_acceleration_until_marker="window_end",
    )
    policy = result.report["nonnegative_acceleration_until_marker"]
    assert policy["marker_was_inserted_into_grid"] is False
    assert policy["grid_node_is_exact_marker"] is True
    assert policy["marker_source_index"] == 4.0


def test_nonnegative_acceleration_policy_default_none_is_bytewise_noop():
    kwargs = dict(
        velocity_limits=np.array([1.2, 0.8, 0.8]),
        acceleration_limits=np.array([2.0, 1.5, 1.5]),
        markers={"window_end": 12.25},
    )
    implicit = _retime(_linear_path(), **kwargs)
    explicit = _retime(
        _linear_path(),
        **kwargs,
        nonnegative_acceleration_until_marker=None,
    )
    assert np.array_equal(implicit.q, explicit.q)
    assert np.array_equal(implicit.qdot, explicit.qdot)
    assert np.array_equal(implicit.path_position, explicit.path_position)
    assert np.array_equal(implicit.path_speed, explicit.path_speed)
    assert np.array_equal(implicit.path_acceleration, explicit.path_acceleration)
    assert implicit.report == explicit.report
    assert implicit.report["nonnegative_acceleration_until_marker"] == {
        "enabled": False,
        "marker": None,
    }


def test_nonnegative_policy_and_discrete_marker_duration_are_both_revalidated():
    markers = {"window_start": 6.8, "window_end": 8.37}
    baseline = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        markers=markers,
        nonnegative_acceleration_until_marker="window_end",
    )
    baseline_width = (
        baseline.markers["window_end"].time_s
        - baseline.markers["window_start"].time_s
    )
    requested = baseline_width * 1.4
    result = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        markers=markers,
        marker_min_duration_s={("window_start", "window_end"): requested},
        nonnegative_acceleration_until_marker="window_end",
    )
    interval = result.report["marker_min_duration_s"]["window_start->window_end"]
    assert interval["actual_s"] >= requested
    assert interval["discrete_duration_s"] >= requested
    assert (
        result.report["nonnegative_acceleration_until_marker"][
            "prefix_scalar_acceleration_min_50hz"
        ]
        >= -1e-9
    )
    assert max(result.report["max_ratio"].values()) <= 1.0 + 1e-6


def _stroke_path(
    samples: int = 41,
    amplitude: float = 1.0,
    sharpness: float = 0.015,
    reversal: float = 0.62,
) -> np.ndarray:
    """A stroke-shaped 2-joint path.

    Joint 0 advances monotonically; joint 1 traces a smooth, sharp near-zero
    backswing reversal at ``reversal`` (a high-curvature notch).  A rest-to-rest
    profile therefore swings out, decelerates hard into the reversal, then
    accelerates back out through the later strike window.  The high curvature at
    the notch is the sole scalar-speed bottleneck.
    """

    s = np.linspace(0.0, 1.0, samples)
    return np.column_stack(
        (s, amplitude * np.sqrt((s - reversal) ** 2 + sharpness ** 2))
    )


def _stroke_no_brake(nonnegative_acceleration_from_marker):
    """Retime the stroke path with a high acceleration and binding velocity cap.

    The reversal sits just before ``window_start``; the strike window is on the
    accelerating recovery.  A generous acceleration limit keeps the reversal
    notch narrow while the velocity limit binds on the straight approach, so the
    from-path-start floor and the scoped floor differ dramatically.
    """

    path = _stroke_path()
    samples = len(path)
    markers = {
        "window_start": 0.64 * (samples - 1),
        "source_anchor": 0.70 * (samples - 1),
        "window_end": 0.76 * (samples - 1),
    }
    return _MOD._retime_path_impl(
        path,
        np.full(2, 5.0),
        np.full(2, 60.0),
        position_lower_limits=np.full(2, -10.0),
        position_upper_limits=np.full(2, 10.0),
        path_progress=_arc_progress(path),
        markers=markers,
        nonnegative_acceleration_until_marker="window_end",
        nonnegative_acceleration_from_marker=(
            nonnegative_acceleration_from_marker
        ),
        grid_subdivisions=12,
    )


def test_from_scoped_no_brake_lifts_backswing_and_still_guards_window():
    from_start = _stroke_no_brake(None)
    scoped = _stroke_no_brake("window_start")

    # (a) Scoping the no-brake floor to [window_start, window_end] frees the
    # pre-window approach, so window_start is reached several-fold sooner than
    # when the floor spans the backswing reversal from the path start.
    from_start_arrival = from_start.markers["window_start"].time_s
    scoped_arrival = scoped.markers["window_start"].time_s
    assert scoped_arrival > 0.0
    assert from_start_arrival > 3.0 * scoped_arrival

    dt = 1.0 / 50.0
    interval_start = np.arange(len(scoped.path_acceleration)) * dt
    interval_end = interval_start + dt
    ws_time = scoped.markers["window_start"].time_s
    we_time = scoped.markers["window_end"].time_s

    # (b) Inside [window_start, window_end] the scoped profile still has
    # nonnegative discrete acceleration, including the interval straddling the
    # exact window_end marker.
    in_window = (interval_start >= ws_time - 1e-12) & (
        interval_start < we_time - 1e-12
    )
    straddles_window_end = (interval_start < we_time - 1e-12) & (
        interval_end > we_time + 1e-12
    )
    assert in_window.any()
    assert straddles_window_end.any()
    # The straddling interval is inside the checked set.
    assert np.all(in_window[straddles_window_end])
    assert np.min(scoped.path_acceleration[in_window]) >= -1e-9

    # (c) The freed segment before window_start demonstrably decelerates
    # (the backswing reversal), proving the constraint really lifted there.
    free = interval_start < ws_time - 1e-12
    assert free.any()
    assert np.min(scoped.path_acceleration[free]) < 0.0
    # Contrast: the from-path-start policy admits no braking anywhere before its
    # marker, so that same reversal is floored out.
    fs_start = np.arange(len(from_start.path_acceleration)) * dt
    fs_before_end = fs_start < from_start.markers["window_end"].time_s - 1e-12
    assert np.min(from_start.path_acceleration[fs_before_end]) >= -1e-9

    # The receipt records the scoped range and the exact from-marker grid node.
    policy = scoped.report["nonnegative_acceleration_until_marker"]
    assert policy["from_marker"] == "window_start"
    assert policy["nonnegative_range"] == "from_marker_to_marker"
    assert policy["from_marker_grid_index"] == policy["prefix_start_grid_index"]
    assert policy["grid_node_is_exact_from_marker"] is True
    assert "until_straddling_interval" in policy["output_interval_policy"]
    assert policy["prefix_scalar_acceleration_min_50hz"] >= -1e-9

    # Default (from=None) keeps the original from-path-start receipt semantics.
    baseline_policy = from_start.report[
        "nonnegative_acceleration_until_marker"
    ]
    assert baseline_policy["from_marker"] is None
    assert baseline_policy["nonnegative_range"] == "path_start_to_marker"
    assert baseline_policy["prefix_start_grid_index"] == 0


def test_from_scoped_no_brake_invalid_from_marker_fails_closed():
    path = _stroke_path()
    samples = len(path)
    markers = {
        "window_start": 0.64 * (samples - 1),
        "window_end": 0.76 * (samples - 1),
    }
    base = dict(
        position_lower_limits=np.full(2, -10.0),
        position_upper_limits=np.full(2, 10.0),
        path_progress=_arc_progress(path),
        markers=markers,
        grid_subdivisions=12,
    )

    def solve(until, from_marker):
        return _MOD._retime_path_impl(
            path,
            np.full(2, 5.0),
            np.full(2, 60.0),
            nonnegative_acceleration_until_marker=until,
            nonnegative_acceleration_from_marker=from_marker,
            **base,
        )

    # Unknown from-marker.
    with pytest.raises(RetimeError):
        solve("window_end", "does_not_exist")
    # from-marker at or after the until-marker.
    with pytest.raises(RetimeError):
        solve("window_start", "window_end")
    with pytest.raises(RetimeError):
        solve("window_end", "window_end")
    # from-marker requested without an until-marker.
    with pytest.raises(RetimeError):
        solve(None, "window_start")

    # Sanity: the valid scoped combination still solves and binds the range.
    ok = solve("window_end", "window_start")
    assert (
        ok.report["nonnegative_acceleration_until_marker"]["from_marker"]
        == "window_start"
    )


def test_uniform_scalar_prefix_matches_1d_recovery_bound_and_is_dominated():
    path = np.linspace(0.0, 1.0, 5)[:, None]
    common = dict(
        velocity_limits=np.asarray([10.0]),
        acceleration_limits=np.asarray([2.0]),
        path_progress=path[:, 0],
        markers={
            "window_start": 2.0,
            "source_anchor": 2.5,
            "window_end": 3.0,
        },
    )
    no_brake = _retime(
        path,
        **common,
        nonnegative_acceleration_until_marker="window_end",
    )
    uniform = _retime(
        path,
        **common,
        uniform_scalar_path_acceleration_until_marker="window_end",
    )
    policy = uniform.report[
        "uniform_scalar_path_acceleration_prefix_comparator"
    ]
    assert policy["enabled"] is True
    assert policy["constant_scalar_path_acceleration"] == pytest.approx(
        2.0 / 3.0, abs=2e-12
    )
    assert uniform.markers["window_end"].time_s == pytest.approx(
        1.5, abs=2e-12
    )
    assert uniform.report["duration_s"] == pytest.approx(2.0, abs=2e-12)
    assert policy["strictly_positive_on_every_guard_cell"] is True
    assert policy["cruise_cells_through_guard"] == 0
    assert policy[
        "prefix_scalar_acceleration_max_abs_error_continuous"
    ] < 1e-10
    assert policy["suffix_recovery_contract"].startswith(
        "ordinary_rest_to_rest_backward_reachable"
    )
    for marker in ("window_start", "source_anchor", "window_end"):
        assert (
            no_brake.markers[marker].time_s
            <= uniform.markers[marker].time_s + 1e-12
        )
    assert no_brake.report["duration_s"] <= uniform.report["duration_s"]


def test_uniform_scalar_prefix_has_no_velocity_cap_cruise_segment():
    path = np.linspace(0.0, 1.0, 5)[:, None]
    result = _retime(
        path,
        velocity_limits=np.asarray([0.8]),
        acceleration_limits=np.asarray([2.0]),
        path_progress=path[:, 0],
        markers={"window_end": 3.0},
        uniform_scalar_path_acceleration_until_marker="window_end",
    )
    policy = result.report[
        "uniform_scalar_path_acceleration_prefix_comparator"
    ]
    assert policy["strictly_positive_on_every_guard_cell"] is True
    assert policy["cruise_cells_through_guard"] == 0
    assert policy["constant_scalar_path_acceleration"] > 0.0
    assert policy[
        "constant_scalar_path_acceleration_unscaled"
    ] == pytest.approx(
        0.8**2 / (2.0 * policy["control_guard_path_progress"]),
        rel=2e-12,
    )
    assert max(result.report["max_ratio"].values()) <= 1.0 + 1e-6


def test_uniform_scalar_prefix_extends_through_straddling_50hz_guard():
    path = np.linspace(0.0, 1.0, 5)[:, None]
    result = _retime(
        path,
        velocity_limits=np.asarray([10.0]),
        acceleration_limits=np.asarray([2.0]),
        path_progress=path[:, 0],
        markers={"window_end": 2.7},
        uniform_scalar_path_acceleration_until_marker="window_end",
    )
    policy = result.report[
        "uniform_scalar_path_acceleration_prefix_comparator"
    ]
    assert policy["control_guard_iteration"] >= 1
    assert (
        policy["control_guard_path_progress"]
        > policy["marker_path_progress"]
    )
    assert policy["prefix_scalar_acceleration_min_50hz"] > 0.0
    assert policy["cruise_cells_through_guard"] == 0
    checked = (
        np.arange(len(result.path_acceleration)) / 50.0
        < result.markers["window_end"].time_s - 1e-12
    )
    np.testing.assert_allclose(
        result.path_acceleration[checked],
        policy["constant_scalar_path_acceleration"],
        rtol=2e-11,
        atol=2e-11,
    )


def test_uniform_scalar_prefix_reports_global_time_scaled_acceleration():
    path = np.linspace(0.0, 1.0, 5)[:, None]
    result = _retime(
        path,
        velocity_limits=np.asarray([10.0]),
        acceleration_limits=np.asarray([2.0]),
        path_progress=path[:, 0],
        markers={
            "window_start": 2.0,
            "source_anchor": 2.5,
            "window_end": 3.0,
        },
        marker_min_duration_s={("window_start", "window_end"): 0.6},
        uniform_scalar_path_acceleration_until_marker="window_end",
    )
    policy = result.report[
        "uniform_scalar_path_acceleration_prefix_comparator"
    ]
    assert result.report["time_scale"] > 1.0
    assert policy["constant_scalar_path_acceleration"] == pytest.approx(
        policy["constant_scalar_path_acceleration_unscaled"]
        / result.report["time_scale"] ** 2,
        rel=1e-12,
    )


def test_uniform_scalar_prefix_requires_explicit_progress_and_is_mutually_exclusive():
    path = np.linspace(0.0, 1.0, 5)[:, None]
    with pytest.raises(RetimeError, match="requires explicit"):
        _retime(
            path,
            velocity_limits=np.asarray([10.0]),
            acceleration_limits=np.asarray([2.0]),
            markers={"window_end": 3.0},
            uniform_scalar_path_acceleration_until_marker="window_end",
        )
    with pytest.raises(RetimeError, match="mutually exclusive"):
        _retime(
            path,
            velocity_limits=np.asarray([10.0]),
            acceleration_limits=np.asarray([2.0]),
            path_progress=path[:, 0],
            markers={"window_end": 3.0},
            nonnegative_acceleration_until_marker="window_end",
            uniform_scalar_path_acceleration_until_marker="window_end",
        )


def test_nonnegative_policy_rejects_internal_zero_speed_bottleneck(monkeypatch):
    def profile_with_internal_stop(path_grid, *args, **kwargs):
        profile = np.ones(len(path_grid), dtype=np.float64)
        profile[0] = 0.0
        profile[-1] = 0.0
        profile[len(path_grid) // 2] = 0.0
        return profile, 1

    monkeypatch.setattr(
        _MOD,
        "_forward_backward_profile",
        profile_with_internal_stop,
    )
    with pytest.raises(RetimeError, match="zero-speed bottleneck"):
        _retime(
            _linear_path(),
            velocity_limits=np.full(3, 20.0),
            acceleration_limits=np.ones(3),
            markers={"window_end": 16.37},
            nonnegative_acceleration_until_marker="window_end",
        )


def test_nonnegative_policy_marker_edges_and_invalid_names():
    start = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.ones(3),
        markers={"start": 0.0},
        nonnegative_acceleration_until_marker="start",
    )
    policy = start.report["nonnegative_acceleration_until_marker"]
    assert policy["marker_grid_index"] == 0
    assert policy["ordinary_profile_changed"] is False
    assert policy["prefix_scalar_acceleration_min_continuous"] is None

    with pytest.raises(RetimeError, match="zero-speed bottleneck"):
        _retime(
            _linear_path(),
            velocity_limits=np.full(3, 20.0),
            acceleration_limits=np.ones(3),
            markers={"finish": 20.0},
            nonnegative_acceleration_until_marker="finish",
        )
    with pytest.raises(RetimeError, match="unknown marker"):
        _retime(
            _linear_path(),
            velocity_limits=np.full(3, 20.0),
            acceleration_limits=np.ones(3),
            markers={"window_end": 4.0},
            nonnegative_acceleration_until_marker="missing",
        )
    with pytest.raises(RetimeError, match="non-empty marker name"):
        _retime(
            _linear_path(),
            velocity_limits=np.full(3, 20.0),
            acceleration_limits=np.ones(3),
            markers={"window_end": 4.0},
            nonnegative_acceleration_until_marker="",
        )


def test_minimum_window_duration_slows_speed_without_freezing_acceleration():
    marker_positions = {"window_start": 2.0, "contact": 3.0, "window_end": 4.0}
    baseline = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.array([1.0, 1.0, 1.0]),
        markers=marker_positions,
    )
    baseline_width = (
        baseline.markers["window_end"].time_s
        - baseline.markers["window_start"].time_s
    )
    minimum_width = baseline_width * 1.35
    result = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.array([1.0, 1.0, 1.0]),
        markers=marker_positions,
        marker_min_duration_s={
            ("window_start", "window_end"): minimum_width,
        },
    )
    actual_width = (
        result.markers["window_end"].time_s
        - result.markers["window_start"].time_s
    )
    segment_midpoint = 0.5 * (
        result.path_position[:-1] + result.path_position[1:]
    )
    inside = (segment_midpoint >= 2.0) & (segment_midpoint <= 4.0)
    assert actual_width >= minimum_width
    assert np.all(result.path_acceleration[inside] > 0.0)
    assert result.path_speed[inside][-1] > result.path_speed[inside][0]
    assert result.report["time_scale"] > baseline.report["time_scale"]


def test_minimum_window_duration_is_also_consumable_on_50hz_grid():
    # 0.021 s is continuously longer than one 50 Hz tick, but one 0.02 s
    # control interval is still too short.  Nearest-frame rounding can report a
    # false pass; only full samples inside [ceil(start), floor(end)] count.
    minimum_width = 0.021
    result = _retime(
        _linear_path(),
        velocity_limits=np.full(3, 20.0),
        acceleration_limits=np.full(3, 20.0),
        markers={"window_start": 2.0, "window_end": 2.1},
        marker_min_duration_s={
            ("window_start", "window_end"): minimum_width,
        },
    )
    interval = result.report["marker_min_duration_s"]["window_start->window_end"]
    start_fractional = result.markers["window_start"].output_fractional_frame
    end_fractional = result.markers["window_end"].output_fractional_frame
    assert interval["actual_s"] >= minimum_width
    assert interval["discrete_duration_s"] >= minimum_width
    assert interval["discrete_control_intervals"] >= 2
    assert interval["discrete_start_frame"] == int(
        np.ceil(start_fractional - 1e-10)
    )
    assert interval["discrete_end_frame"] == int(
        np.floor(end_fractional + 1e-10)
    )
    assert (
        interval["discrete_end_frame"] - interval["discrete_start_frame"]
        == interval["discrete_control_intervals"]
    )
    assert (
        result.report["marker_interval_discrete_policy"]
        == "inclusive_samples_ceil_start_floor_end"
    )


def test_marker_mapping_is_ordered_and_endpoint_exact():
    result = _retime(
        _linear_path(),
        velocity_limits=np.ones(3),
        acceleration_limits=np.ones(3),
        markers={"ready": 0.0, "contact": 8.5, "finish": 20.0},
    )
    ready = result.markers["ready"]
    contact = result.markers["contact"]
    finish = result.markers["finish"]
    assert ready.time_s == 0.0
    assert ready.output_frame == 0
    assert finish.output_frame == len(result.q) - 1
    assert ready.time_s < contact.time_s < finish.time_s
    assert ready.source_index < contact.source_index < finish.source_index
    assert contact.path_position_at_frame == pytest.approx(8.5, abs=0.25)


def test_curved_path_respects_caps_after_uniform_grid_sampling():
    u = np.linspace(0.0, 1.0, 41)
    q = np.column_stack(
        (
            0.8 * u,
            0.25 * np.sin(np.pi * u),
            0.15 * np.sin(2.0 * np.pi * u),
        )
    )
    result = _retime(
        q,
        velocity_limits=np.array([0.8, 0.7, 0.7]),
        acceleration_limits=np.array([1.2, 0.8, 0.8]),
        grid_subdivisions=16,
    )
    assert np.all(np.diff(result.path_position) >= -1e-12)
    assert result.path_position[0] == 0.0
    assert result.path_position[-1] == len(q) - 1
    assert max(result.report["max_ratio"].values()) <= 1.0 + 1e-6


@pytest.mark.parametrize(
    ("path", "velocity", "acceleration", "message"),
    [
        (
            np.zeros((5, 2)),
            np.ones(2),
            np.ones(2),
            "duplicate/degenerate",
        ),
        (
            _linear_path(),
            np.array([1.0, 0.0, 1.0]),
            np.ones(3),
            "strictly positive",
        ),
        (
            _linear_path(),
            np.ones(3),
            np.array([1.0, np.nan, 1.0]),
            "strictly positive",
        ),
    ],
)
def test_degenerate_or_infeasible_inputs_fail_closed(
    path: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    message: str,
):
    with pytest.raises(RetimeError, match=message):
        _retime(path, velocity, acceleration)


def test_invalid_marker_and_too_short_path_fail_closed():
    with pytest.raises(RetimeError, match="outside"):
        _retime(
            _linear_path(),
            np.ones(3),
            np.ones(3),
            markers={"contact": 21.0},
        )
    with pytest.raises(RetimeError, match="samples>=3"):
        _retime(np.zeros((2, 3)), np.ones(3), np.ones(3))
    with pytest.raises(RetimeError, match="unknown marker"):
        _retime(
            _linear_path(),
            np.ones(3),
            np.ones(3),
            markers={"window_start": 2.0},
            marker_min_duration_s={
                ("window_start", "window_end"): 0.1,
            },
        )


def test_position_limits_are_required_as_a_complete_strict_contract():
    path = np.array([[0.0], [0.5], [1.0]])
    with pytest.raises(TypeError, match="position_lower_limits"):
        retime_path(path, np.ones(1), np.ones(1))
    with pytest.raises(RetimeError, match="provided together"):
        retime_path(
            path,
            np.ones(1),
            np.ones(1),
            position_lower_limits=np.array([-0.1]),
            position_upper_limits=None,
        )
    with pytest.raises(RetimeError, match="lower than"):
        retime_path(
            path,
            np.ones(1),
            np.ones(1),
            position_lower_limits=np.array([1.0]),
            position_upper_limits=np.array([1.0]),
        )
    with pytest.raises(RetimeError, match="outside position limits"):
        retime_path(
            path,
            np.ones(1),
            np.ones(1),
            position_lower_limits=np.array([-0.1]),
            position_upper_limits=np.array([0.9]),
        )
    with pytest.raises(RetimeError, match="position_tolerance"):
        retime_path(
            path,
            np.ones(1),
            np.ones(1),
            position_lower_limits=np.array([-0.1]),
            position_upper_limits=np.array([1.0]),
            position_tolerance=-1.0,
        )


def test_position_tolerance_accepts_float32_limit_quantisation_only():
    # Model a float32-authored source whose value sits only one declared
    # quantisation allowance above the exact URDF bound.
    exact_upper = 0.1 - 4e-8
    quantised_upper = float(np.float32(0.1))
    assert 0.0 < quantised_upper - exact_upper < 1e-7
    accepted = retime_path(
        np.array([[0.0], [quantised_upper], [0.0]], dtype=np.float64),
        np.ones(1),
        np.ones(1),
        position_lower_limits=np.array([-exact_upper]),
        position_upper_limits=np.array([exact_upper]),
        position_tolerance=1e-7,
    )
    assert accepted.report["position_tolerance"] == pytest.approx(1e-7)

    with pytest.raises(RetimeError, match="outside position limits"):
        retime_path(
            np.array([[0.0], [exact_upper + 2e-7], [0.0]], dtype=np.float64),
            np.ones(1),
            np.ones(1),
            position_lower_limits=np.array([-exact_upper]),
            position_upper_limits=np.array([exact_upper]),
            position_tolerance=1e-7,
        )


def test_shape_preserving_path_cannot_overshoot_position_limits():
    # A centred-difference Hermite tangent overshoots the last segment here:
    # its incoming slope is ~50x the final chord.  The path contract must check
    # continuous cubic extrema, not merely source/output samples.
    path = np.array([[0.0], [1.0], [1.01]])
    result = retime_path(
        path,
        np.array([3.0]),
        np.array([4.0]),
        position_lower_limits=np.array([0.0]),
        position_upper_limits=np.array([1.01]),
    )
    assert result.report["continuous_position_min"][0] >= -1e-12
    assert result.report["continuous_position_max"][0] <= 1.01 + 1e-12
    assert np.all(result.q >= -1e-12)
    assert np.all(result.q <= 1.01 + 1e-12)


def test_continuous_position_extrema_fail_closed_if_tangents_regress(monkeypatch):
    path = np.array([[0.0], [1.0], [1.01]])
    old_centered_tangents = np.array([[1.0], [0.505], [0.01]])
    monkeypatch.setattr(
        _MOD,
        "_path_tangents",
        lambda ignored_path: old_centered_tangents.copy(),
    )
    with pytest.raises(RetimeError, match="continuous cubic extremum"):
        retime_path(
            path,
            np.array([3.0]),
            np.array([4.0]),
            position_lower_limits=np.array([0.0]),
            position_upper_limits=np.array([1.01]),
        )


def test_continuous_cell_acceleration_extremum_closes_midpoint_leak():
    # Regression: midpoint-only checking accepted this path at a time scale
    # whose true within-cell peak was 1.020828... times the acceleration cap.
    result = retime_path(
        np.array([[0.0], [2.0], [3.0]]),
        np.array([1.0]),
        np.array([1.0]),
        position_lower_limits=np.array([0.0]),
        position_upper_limits=np.array([3.0]),
        grid_subdivisions=12,
    )
    ratios = result.report["max_ratio"]
    assert ratios["continuous_cell_acceleration"] <= 1.0 + 1e-6
    assert ratios["continuous_cell_velocity"] <= 1.0 + 1e-6


def test_batched_polynomial_ranges_match_legacy_random_and_degenerate_cases():
    rng = np.random.default_rng(20260724)
    coefficients = rng.normal(size=(96, 3, 6))
    coefficients *= 10.0 ** rng.uniform(
        -7.0, 7.0, size=(96, 3, 1)
    )
    # Exercise exact trimming, just-inside/outside thresholds, missing terms,
    # repeated roots, and the degree-2/3 analytic paths.
    epsilon = np.finfo(float).eps
    coefficients[:8] = 0.0
    coefficients[0, 0, :3] = [1.0, -2.0, 32.0 * epsilon]
    coefficients[1, 0, :3] = [
        1.0,
        -2.0,
        np.nextafter(32.0 * epsilon, np.inf),
    ]
    coefficients[2, 0, :4] = [1.0, -3.0, 3.0, -1.0]
    coefficients[3, 0, :4] = [0.0, 1.0, -2.0, 1.0]
    coefficients[4, 0, :] = [1.0, 0.0, -3.0, 0.0, 2.0, 0.0]
    coefficients[5, 0, :] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    lower = rng.uniform(-1.5, -0.05, size=(96, 3))
    upper = rng.uniform(0.05, 1.5, size=(96, 3))

    actual_low, actual_high = _MOD._batched_polynomial_ranges(
        coefficients, lower, upper
    )
    expected_low = np.empty_like(actual_low)
    expected_high = np.empty_like(actual_high)
    for index in np.ndindex(expected_low.shape):
        expected_low[index], expected_high[index] = (
            _legacy_polynomial_range(
                coefficients[index], lower[index], upper[index]
            )
        )
    np.testing.assert_allclose(
        actual_low, expected_low, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        actual_high, expected_high, rtol=2e-11, atol=2e-11
    )


def test_scalar_analytic_low_degree_roots_match_legacy_random_and_near_real():
    rng = np.random.default_rng(710223)
    cases = [
        rng.normal(size=3)
        for _ in range(128)
    ] + [
        rng.normal(size=4)
        for _ in range(128)
    ]
    # A cubic whose derivative has an imaginary part below the established
    # near-real tolerance exercises that behavior in the analytic branch.
    cases.extend(
        (
            np.array([0.3, (5e-11) ** 2, 0.0, 1.0 / 3.0]),
            np.array([0.3, -(5e-11) ** 2, 0.0, 1.0 / 3.0]),
            np.array([1.0, -2.0, 32.0 * np.finfo(float).eps, 0.0]),
        )
    )
    for coefficients in cases:
        expected = _legacy_polynomial_range(coefficients, -1.25, 0.75)
        actual = _MOD._polynomial_range(coefficients, -1.25, 0.75)
        np.testing.assert_allclose(
            actual, expected, rtol=2e-12, atol=2e-12
        )


def test_cached_continuous_cell_envelope_matches_legacy_random_differential():
    rng = np.random.default_rng(731994)
    q_path = np.cumsum(rng.normal(scale=0.08, size=(7, 5)), axis=0)
    tangents = _MOD._path_tangents(q_path)
    coefficients = _MOD._hermite_coefficients(q_path, tangents)
    subdivisions = 4
    path_grid = np.linspace(
        0.0,
        float(len(q_path) - 1),
        (len(q_path) - 1) * subdivisions + 1,
    )
    raw_speed = rng.uniform(0.2, 2.0, size=len(path_grid))
    speed_sq = raw_speed * raw_speed
    ds = np.diff(path_grid)
    segment_accel = np.diff(speed_sq) / (2.0 * ds)
    velocity_limits = rng.uniform(0.5, 2.0, size=q_path.shape[1])
    acceleration_limits = rng.uniform(1.0, 5.0, size=q_path.shape[1])
    peaks = _MOD._continuous_cell_peaks(
        coefficients, path_grid, speed_sq, segment_accel
    )

    for time_scale in (1.0, 1.137, 3.75):
        expected = _legacy_continuous_cell_ratios(
            coefficients,
            path_grid,
            speed_sq,
            segment_accel,
            velocity_limits,
            acceleration_limits,
            time_scale,
        )
        actual = _MOD._continuous_cell_ratios_from_peaks(
            peaks,
            velocity_limits,
            acceleration_limits,
            time_scale,
        )
        np.testing.assert_allclose(
            actual, expected, rtol=2e-11, atol=2e-11
        )


def test_retime_prepares_continuous_cell_extrema_once_across_slowdowns(
    monkeypatch,
):
    calls = 0
    original = _MOD._continuous_cell_peaks

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_MOD, "_continuous_cell_peaks", counted)
    result = retime_path(
        np.array([[0.0], [2.0], [3.0]]),
        np.array([1.0]),
        np.array([1.0]),
        position_lower_limits=np.array([0.0]),
        position_upper_limits=np.array([3.0]),
        grid_subdivisions=12,
    )
    assert result.report["validation_iterations"] > 1
    assert calls == 1


def test_closed_form_reachability_matches_legacy_bisection_random_differential():
    rng = np.random.default_rng(882701)
    for case in range(400):
        joints = 7
        fixed_speed_sq = float(rng.uniform(0.0, 4.0))
        neighbor_cap = float(rng.uniform(0.0, 10.0))
        segment_cap = float(rng.uniform(0.0, 10.0))
        ds = float(10.0 ** rng.uniform(-3.0, 0.5))
        q_s_mid = rng.normal(size=joints)
        if case % 7 == 0:
            q_s_mid[::2] = 0.5 * _MOD._REGULARITY_EPS
        q_ss_mid = rng.normal(size=joints)
        acceleration_limits = (
            np.abs(q_ss_mid) * fixed_speed_sq
            + rng.uniform(0.05, 8.0, size=joints)
        )
        arguments = (
            fixed_speed_sq,
            neighbor_cap,
            segment_cap,
            ds,
            q_s_mid,
            q_ss_mid,
            acceleration_limits,
        )
        expected = _legacy_max_reachable_neighbor(*arguments)
        actual = _MOD._max_reachable_neighbor(*arguments)
        assert actual == pytest.approx(
            expected,
            rel=2e-13,
            abs=2e-13 * max(1.0, abs(expected)),
        )


@pytest.mark.parametrize(
    ("path", "velocity", "acceleration", "lower", "upper"),
    [
        (
            np.array([[0.0j], [0.5 + 0.1j], [1.0j]]),
            np.ones(1),
            np.ones(1),
            np.array([-2.0]),
            np.array([2.0]),
        ),
        (
            np.array([[0.0], [0.5], [1.0]]),
            np.array([1.0 + 0.1j]),
            np.ones(1),
            np.array([-2.0]),
            np.array([2.0]),
        ),
        (
            np.array([[0.0], [0.5], [1.0]]),
            np.ones(1),
            np.array([1.0 - 0.1j]),
            np.array([-2.0]),
            np.array([2.0]),
        ),
        (
            np.array([[0.0], [0.5], [1.0]]),
            np.ones(1),
            np.ones(1),
            np.array([-2.0 + 0.1j]),
            np.array([2.0]),
        ),
        (
            np.array([[0.0], [0.5], [1.0]]),
            np.ones(1),
            np.ones(1),
            np.array([-2.0]),
            np.array([2.0 - 0.1j]),
        ),
    ],
)
def test_complex_paths_and_limits_fail_closed_before_float_conversion(
    path, velocity, acceleration, lower, upper
):
    with pytest.raises(RetimeError, match="real-valued"):
        retime_path(
            path,
            velocity,
            acceleration,
            position_lower_limits=lower,
            position_upper_limits=upper,
        )


def _velocity_dip_weighted_arc():
    """A ready->core->ready loop whose dominant joint decelerates mid-motion.

    Under ``coordinate_scale = 1 / v`` the per-formal-segment *minimum* weighted
    speed falls far below the local weighted speed wherever the fast joint slows
    (a velocity dip / joint hand-off).  That is exactly the condition that made
    the pre-fix segment-global-min-speed denominator inflate ``|q_l|``/``|q_ll|``
    by orders of magnitude.  ``v`` is deliberately heterogeneous.
    """

    velocity = np.asarray([15.7, 5.0])
    scale = 1.0 / velocity
    samples = np.linspace(0.0, 1.0, 15)
    source = np.column_stack(
        (
            1.2 * np.sin(0.9 * np.pi * samples),
            0.8 * (1.0 - np.cos(np.pi * samples)),
        )
    )
    ready = source[0].copy()
    geometry = build_canonical_geometry(
        source,
        ready,
        entry_frame=2,
        exit_frame=12,
        window_start=5,
        window_end=9,
        window_halo=2,
        samples_per_rad=24.0,
        coordinate_scale=scale,
        coordinate_semantics=("j0", "j1"),
        coordinate_units=("rad", "rad"),
    )
    formal = geometry.canonical_knot_path_indices
    arc = build_weighted_arc_path(
        s_knots=geometry.path_parameter[formal],
        q=geometry.q_path[formal],
        q_s=geometry.dq_ds[formal],
        q_ss=geometry.d2q_ds2[formal],
        coordinate_scale=scale,
    )
    q, q_l, q_ll = arc.evaluate_l(arc.l_knots)
    formal_source = geometry.source_frame_map[formal]

    def marker_index(source_frame):
        matches = np.flatnonzero(formal_source == float(source_frame))
        assert matches.shape == (1,)
        return float(matches[0])

    markers = {
        "window_start": marker_index(5),
        "source_anchor": marker_index(7),
        "window_end": marker_index(9),
    }
    return velocity, scale, arc, q, q_l, q_ll, markers


def test_weighted_arc_envelope_respects_velocity_invariant_under_scale_1_over_v():
    """The candidate envelope's ``|q_l|`` may never exceed ``v_j = 1 / W_j``.

    Pre-fix (segment-max derivative / segment-global-min weighted speed) this
    returned ~182 rad/s for a ``v_j = 5`` rad/s joint, collapsing the velocity
    cap ``(v_j / |q_l_j|)**2`` far below one.
    """

    velocity, _scale, arc, _q, _q_l, _q_ll, _markers = _velocity_dip_weighted_arc()
    nodes = np.asarray(arc.l_knots)
    subdivisions = 12
    parts = [
        np.linspace(nodes[i], nodes[i + 1], subdivisions + 1)[:-1]
        for i in range(len(nodes) - 1)
    ]
    path_grid = np.concatenate((*parts, np.asarray([nodes[-1]])))
    q_l_abs, q_ll_abs = _MOD._weighted_arc_formal_segment_solver_envelope(
        arc, path_grid
    )
    assert np.all(np.isfinite(q_l_abs)) and np.all(np.isfinite(q_ll_abs))
    worst_ratio = float(np.max(q_l_abs / velocity[None, :]))
    assert worst_ratio <= 1.0 + 1e-6, worst_ratio


def test_weighted_arc_conservative_cell_bounds_respect_velocity_invariant():
    """The shared per-cell bound helper (used by the candidate envelope AND the
    hard continuous-cell gate) must respect ``|q_l_j| <= v_j = 1 / W_j`` on every
    formal segment.  Pre-fix, dividing a segment's peak derivative by its global
    minimum weighted speed returned ~182 rad/s for a ``v_j = 5`` rad/s joint.
    """

    velocity, _scale, arc, _q, _q_l, _q_ll, _markers = _velocity_dip_weighted_arc()
    segments = np.arange(len(arc.segment_lengths), dtype=np.int64)
    cell_q_l, cell_q_ll, cell_speed = (
        _MOD._weighted_arc_conservative_cell_derivatives(
            arc,
            segments,
            np.zeros(len(segments)),
            np.ones(len(segments)),
        )
    )
    assert np.all(cell_speed > arc.regularity_margin)
    assert np.all(np.isfinite(cell_q_ll))
    assert np.all(cell_q_l <= velocity[None, :] * (1.0 + 1e-6)), float(
        np.max(cell_q_l / velocity[None, :])
    )


def test_weighted_arc_velocity_dip_retime_is_not_inflated_under_scale_1_over_v():
    """A single fixed-grid solve places the strike anchor at a physical time.

    Pre-fix, the segment-global-min-speed denominator collapsed both the velocity
    cap ``(v/|q_l|)**2`` and the curvature cap ``a/|q_ll|``, so the anchor landed
    tens of seconds out (the reported ~71 s anchor / ~96 s cycle) and the public
    grid-refinement loop rejected every level.  With the certified per-cell speed
    the base scalar profile is no longer catastrophically inflated (the candidate
    envelope alone dropped from ~13 s to ~1.4 s on this geometry) and the strike
    anchor lands well under a second.  A single fixed grid is used so the check is
    independent of the six-level tick-convergence gate.
    """

    velocity, _scale, arc, q, q_l, q_ll, markers = _velocity_dip_weighted_arc()

    result = _MOD._retime_path_impl(
        q,
        velocity,
        np.asarray([1000.0, 1000.0]),
        position_lower_limits=np.full(2, -5.0),
        position_upper_limits=np.full(2, 5.0),
        path_progress=arc.l_knots,
        path_first_derivative=q_l,
        path_second_derivative=q_ll,
        weighted_arc_path=arc,
        fps=50.0,
        markers=markers,
        marker_min_duration_s={("window_start", "window_end"): 0.12},
        nonnegative_acceleration_until_marker="window_end",
        grid_subdivisions=24,
    )
    anchor_time = float(result.markers["source_anchor"].time_s)
    assert anchor_time < 1.0, anchor_time
    window_duration = float(
        result.markers["window_end"].time_s
        - result.markers["window_start"].time_s
    )
    assert 0.0 < window_duration < 1.0, window_duration


def test_exact_pointwise_caps_control_rate_feasible_with_guard_receipt():
    """Probe-grade exact-pointwise solve returns a guard-verified receipt.

    The exact-pointwise mode evaluates velocity/curvature caps at collocation
    nodes/midpoints and verifies the achieved trajectory a-posteriori at the
    control rate.  The public entry point returns the base-grid candidate with
    both the guard receipt and the informational grid-stability probe.
    """

    velocity, _scale, arc, q, q_l, q_ll, markers = _velocity_dip_weighted_arc()
    result = retime_path(
        q,
        velocity,
        np.asarray([1000.0, 1000.0]),
        position_lower_limits=np.full(2, -5.0),
        position_upper_limits=np.full(2, 5.0),
        path_progress=arc.l_knots,
        path_first_derivative=q_l,
        path_second_derivative=q_ll,
        weighted_arc_path=arc,
        fps=50.0,
        markers=markers,
        marker_min_duration_s={("window_start", "window_end"): 0.08},
        nonnegative_acceleration_until_marker="window_end",
        grid_subdivisions=16,
        exact_pointwise_caps=True,
    )
    report = result.report["exact_pointwise_caps"]
    assert report["enabled"] is True
    assert report["solver"] == "exact_node_and_midpoint_caps_via_evaluate_l"
    guard = report["aposteriori_guard"]
    assert guard is not None
    assert guard["passed"] is True
    assert guard["max_velocity_ratio"] <= guard["velocity_limit_ratio"]
    assert guard["max_acceleration_ratio"] <= guard["acceleration_limit_ratio"]
    assert "grid_stability_probe" in report
    total_l = float(arc.l_knots[-1])
    assert 0.0 < result.report["duration_s"] < 3.0 * total_l


def test_exact_pointwise_caps_never_slower_than_interval_ladder():
    """On one fixed grid the exact pointwise caps dominate the interval sup
    bounds, so the exact solve is never slower than the interval-certified
    ladder (equal caps at worst; strictly looser only where the sup over a cell
    exceeds the pointwise value).  Both are solved on the same grid so the
    comparison is deterministic.
    """

    velocity, _scale, arc, q, q_l, q_ll, _markers = _velocity_dip_weighted_arc()
    common = dict(
        position_lower_limits=np.full(2, -5.0),
        position_upper_limits=np.full(2, 5.0),
        path_progress=arc.l_knots,
        path_first_derivative=q_l,
        path_second_derivative=q_ll,
        weighted_arc_path=arc,
        fps=50.0,
        grid_subdivisions=16,
    )
    exact = _MOD._retime_path_impl(
        q,
        velocity,
        np.asarray([1000.0, 1000.0]),
        exact_pointwise_caps=True,
        **common,
    )
    interval = _MOD._retime_path_impl(
        q,
        velocity,
        np.asarray([1000.0, 1000.0]),
        exact_pointwise_caps=False,
        **common,
    )
    assert exact.report["exact_pointwise_caps"]["aposteriori_guard"]["passed"]
    assert (
        exact.report["duration_s"]
        <= interval.report["duration_s"] + 1e-9
    )


def test_exact_pointwise_aposteriori_guard_fails_closed_on_velocity_excess():
    """The a-posteriori guard fails closed when the achieved trajectory exceeds
    a joint velocity limit at the sampled control rate (velocity is a hard bound
    with no probe margin).  A straight weighted-arc segment traversed at a known
    scalar speed realises a known qdot; a deliberately undersized velocity cap
    must raise.
    """

    length = 1.2
    s_knots = np.asarray([0.0, 0.5, 1.0])
    q = np.asarray([[0.0], [0.5 * length], [length]])
    q_s = np.asarray([[length], [length], [length]])
    q_ss = np.zeros_like(q)
    arc = build_weighted_arc_path(
        s_knots=s_knots,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        coordinate_scale=np.asarray([1.0 / 8.0]),
    )
    path_grid = np.asarray(arc.l_knots)
    speed_sq = np.asarray([0.0, 0.5, 0.0])
    time_knots, segment_accel = _MOD._profile_time_knots(path_grid, speed_sq)

    def evaluate_path(length_values):
        return arc.evaluate_l(length_values)

    with pytest.raises(RetimeError, match="a-posteriori guard failed"):
        _MOD._exact_pointwise_aposteriori_guard(
            evaluate_path=evaluate_path,
            time_scale=1.0,
            path_grid=path_grid,
            speed_sq=speed_sq,
            time_knots=time_knots,
            segment_accel=segment_accel,
            duration=float(time_knots[-1]),
            fps=50.0,
            velocity_cap=np.asarray([0.001]),
            acceleration_cap=np.asarray([1.0e9]),
            guard_rate_multiple=4,
            guard_probe_margin=1.0,
            velocity_tolerance=1e-6,
        )
