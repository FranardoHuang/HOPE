"""Pure-NumPy tests for canonical-ready closed-loop motion geometry.

Run:
    python3 -m pytest \
      hope_training/whole_body_tracking/tests/test_canonical_motion_geometry.py -q
"""
from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "canonical_motion_geometry", _SCRIPTS / "canonical_motion_geometry.py"
)
cmg = importlib.util.module_from_spec(_SPEC)
sys.modules["canonical_motion_geometry"] = cmg
_SPEC.loader.exec_module(cmg)


def _source(frames: int = 13, joints: int = 3) -> np.ndarray:
    f = np.arange(frames, dtype=np.float64)
    q = np.empty((frames, joints), dtype=np.float64)
    q[:, 0] = 0.025 * f**2 - 0.1 * f
    q[:, 1] = 0.4 * np.sin(0.25 * f)
    q[:, 2] = -0.2 + 0.03 * f + 0.002 * f**2
    # Make old frame 0 a unique, obviously wrong waypoint for the non-serial test.
    q[0] = np.array([20.0, -19.0, 18.0])
    return q


READY = np.array([-0.7, 0.2, 0.5], dtype=np.float64)
ENTRY = 3
EXIT = 10
W0 = 5
W1 = 7
HALO = 2


def _build(**overrides):
    kwargs = dict(
        source_q=_source(),
        ready_q=READY,
        entry_frame=ENTRY,
        exit_frame=EXIT,
        window_start=W0,
        window_end=W1,
        window_halo=HALO,
        samples_per_rad=18.0,
    )
    kwargs.update(overrides)
    return cmg.build_canonical_geometry(**kwargs)


def test_closed_loop_has_regular_ready_tangents_and_c2_joins():
    result = _build()
    source = _source()
    assert np.array_equal(result.q_path[0], READY)
    assert np.allclose(result.q_path[-1], READY, atol=1e-12)
    assert np.all(np.diff(result.path_parameter) > 0.0)
    np.testing.assert_allclose(
        result.dq_ds[0], source[ENTRY] - READY, atol=1e-12
    )
    np.testing.assert_allclose(
        result.dq_ds[-1], READY - source[EXIT], atol=1e-12
    )
    assert np.linalg.norm(result.dq_ds[0]) > 0.0
    assert np.linalg.norm(result.dq_ds[-1]) > 0.0
    assert np.allclose(result.d2q_ds2[[0, -1]], 0.0, atol=1e-12)

    report = result.continuity_report
    assert report["c2_continuous"] is True
    assert report["path_regularity"]["regular"] is True
    assert report["path_regularity"]["dense_samples_used_as_proof"] is False
    assert (
        report["path_regularity"][
            "minimum_certified_speed_to_required_margin_ratio"
        ]
        > 1.0
    )
    assert report["joins"]["position_max_abs_rad"] < 1e-10
    assert (
        report["joins"]["first_derivative_max_abs_rad_per_path_unit"] < 1e-10
    )
    assert (
        report["joins"]["second_derivative_max_abs_rad_per_path_unit2"] < 1e-10
    )
    assert (
        report["endpoints"]["start_first_derivative_max_abs_rad_per_path_unit"]
        > 0.0
    )
    assert (
        report["endpoints"]["end_second_derivative_max_abs_rad_per_path_unit2"]
        < 1e-12
    )
    assert report["endpoints"]["path_derivatives_are_physical_state"] is False
    assert report["endpoints"]["physical_endpoint_velocity_claimed"] is False
    assert report["endpoints"]["physical_endpoint_acceleration_claimed"] is False


def test_ready_tangent_scales_with_connector_parameter_span():
    source = _source()
    result = _build(connector_parameter_span=2.5)
    np.testing.assert_allclose(
        result.dq_ds[0], (source[ENTRY] - READY) / 2.5, atol=1e-12
    )
    np.testing.assert_allclose(
        result.dq_ds[-1], (READY - source[EXIT]) / 2.5, atol=1e-12
    )


def test_source_waypoints_and_contact_opportunity_are_exactly_mapped():
    source = _source()
    result = _build(source_q=source)
    expected_frames = np.arange(ENTRY, EXIT + 1)
    expected_first, expected_second = cmg._source_derivatives(
        source[ENTRY : EXIT + 1]
    )
    assert np.array_equal(
        result.q_path[result.source_waypoint_path_indices],
        source[ENTRY : EXIT + 1],
    )
    np.testing.assert_array_equal(
        result.dq_ds[result.source_waypoint_path_indices],
        expected_first,
    )
    np.testing.assert_array_equal(
        result.d2q_ds2[result.source_waypoint_path_indices],
        expected_second,
    )
    assert np.allclose(
        result.source_frame_map[result.source_waypoint_path_indices],
        expected_frames,
        atol=0.0,
    )

    assert result.source_frame_map[result.window_path_start] == W0
    assert result.source_frame_map[result.window_path_end] == W1
    assert np.all(
        (result.source_frame_map[result.window_mask] >= W0)
        & (result.source_frame_map[result.window_mask] <= W1)
    )
    assert result.segment_labels[result.window_path_start] == "contact_opportunity"
    assert result.segment_labels[result.window_path_end] == "contact_opportunity"

    receipt = result.continuity_report["source_frame_map_receipt"]
    canonical = np.ascontiguousarray(result.source_frame_map, dtype="<f8").copy()
    canonical.view("<u8")[np.isnan(canonical)] = np.uint64(
        0x7FF8000000000000
    )
    assert receipt["encoding"] == cmg.SOURCE_FRAME_MAP_ENCODING
    assert receipt["length"] == len(canonical)
    assert receipt["sha256"] == hashlib.sha256(
        canonical.tobytes(order="C")
    ).hexdigest()
    assert receipt["source_waypoint_path_indices"] == (
        result.source_waypoint_path_indices.tolist()
    )


def test_ready_connector_skips_old_frame_zero_instead_of_serializing_it():
    source = _source()
    result = _build(source_q=source)
    finite_map = result.source_frame_map[np.isfinite(result.source_frame_map)]
    assert finite_map.min() == ENTRY
    assert not np.any(np.isclose(finite_map, 0.0))
    assert not np.any(np.all(result.q_path == source[0], axis=1))
    assert result.continuity_report["selection"]["old_source_frame_zero_retained"] is False
    assert result.segment_labels[0] == "canonical_ready_start"
    assert result.segment_labels[-1] == "canonical_ready_end"


def test_sampling_density_is_geometry_resolution_not_duration():
    sparse = _build(samples_per_rad=4.0, min_connector_intervals=5)
    dense = _build(samples_per_rad=40.0, min_connector_intervals=5)
    assert len(dense.q_path) > len(sparse.q_path)
    assert np.array_equal(
        sparse.q_path[sparse.source_waypoint_path_indices],
        dense.q_path[dense.source_waypoint_path_indices],
    )
    assert np.array_equal(
        sparse.canonical_knot_path_indices[1:-1],
        sparse.source_waypoint_path_indices,
    )
    assert np.array_equal(
        dense.canonical_knot_path_indices[1:-1],
        dense.source_waypoint_path_indices,
    )
    assert sparse.canonical_knot_path_indices[0] == 0
    assert dense.canonical_knot_path_indices[0] == 0
    assert sparse.canonical_knot_path_indices[-1] == len(sparse.q_path) - 1
    assert dense.canonical_knot_path_indices[-1] == len(dense.q_path) - 1
    for field in ("q_path", "dq_ds", "d2q_ds2"):
        sparse_values = getattr(sparse, field)[sparse.canonical_knot_path_indices]
        dense_values = getattr(dense, field)[dense.canonical_knot_path_indices]
        np.testing.assert_array_equal(sparse_values, dense_values)
    for result in (sparse, dense):
        semantics = result.continuity_report["parameterization"]
        assert semantics["is_time_parameter"] is False
        assert semantics["sample_count_has_time_meaning"] is False
        assert semantics["connector_rows_are_source_time_bytes"] is False
        knots = result.continuity_report["canonical_knots"]
        assert knots["q_and_path_jets_independent_of_visualization_sampling_density"]
        assert knots["path_indices"] == result.canonical_knot_path_indices.tolist()


def test_scaled_arc_length_progress_is_content_bound_not_row_index():
    sparse = np.asarray([[0.0, 0.0], [0.6, 0.8], [1.2, 1.6]])
    dense = np.asarray(
        [[0.0, 0.0], [0.3, 0.4], [0.6, 0.8], [0.9, 1.2], [1.2, 1.6]]
    )
    scale = np.asarray([2.0, 0.5])
    sparse_progress = cmg.scaled_arc_length_progress(sparse, scale)
    dense_progress = cmg.scaled_arc_length_progress(dense, scale)
    assert sparse_progress[0] == 0.0
    assert dense_progress[0] == 0.0
    assert np.all(np.diff(sparse_progress) > 0.0)
    assert np.all(np.diff(dense_progress) > 0.0)
    assert sparse_progress[-1] == pytest.approx(dense_progress[-1])
    assert dense_progress[2] == pytest.approx(sparse_progress[1])
    assert not np.array_equal(
        dense_progress, np.arange(len(dense_progress), dtype=np.float64)
    )


def test_scaled_arc_length_progress_rejects_degenerate_or_bad_scale():
    with pytest.raises(ValueError, match="duplicate or numerically degenerate"):
        cmg.scaled_arc_length_progress(
            np.asarray([[0.0], [0.0], [1.0]]), np.ones(1)
        )
    with pytest.raises(ValueError, match="must all be > 0"):
        cmg.scaled_arc_length_progress(
            np.asarray([[0.0], [1.0], [2.0]]), np.zeros(1)
        )


def test_mixed_joint_root_units_affect_only_metrics_and_sampling():
    source = _source()
    semantics = ["root_x", "root_y", "joint_0"]
    units = ["m", "m", "rad"]
    scale = np.array([20.0, 20.0, 1.0])
    native = _build(source_q=source, samples_per_rad=8.0)
    mixed = _build(
        source_q=source,
        samples_per_rad=8.0,
        coordinate_scale=scale,
        coordinate_semantics=semantics,
        coordinate_units=units,
    )
    assert len(mixed.q_path) > len(native.q_path)
    assert np.array_equal(mixed.q_path[0], READY)
    assert np.array_equal(mixed.q_path[-1], READY)
    assert np.array_equal(
        mixed.q_path[mixed.source_waypoint_path_indices],
        source[ENTRY : EXIT + 1],
    )

    report = mixed.continuity_report
    contract = report["coordinate_contract"]
    assert contract["scale"] == scale.tolist()
    assert contract["semantics"] == semantics
    assert contract["units"] == units
    assert contract["mixed_units"] is True
    assert contract["physical_coordinates_modified"] is False
    assert contract["legacy_rad_metrics_published"] is False
    assert "position_max_abs_scaled" in report["joins"]
    assert "position_max_abs_rad" not in report["joins"]
    assert "start_position_max_abs_rad" not in report["endpoints"]
    assert "max_abs_error_rad" not in report["source_waypoints"]
    assert "samples_per_rad" not in report["parameterization"]

    candidates = cmg.enumerate_entry_exit_candidates(
        source,
        READY,
        window_start=W0,
        window_end=W1,
        window_halo=HALO,
        coordinate_scale=scale,
        coordinate_semantics=semantics,
        coordinate_units=units,
    )
    candidate = next(c for c in candidates if c.entry_frame == ENTRY)
    pre_delta = source[ENTRY] - READY
    assert candidate.ready_to_entry_max_abs_scaled == pytest.approx(
        np.max(np.abs(pre_delta * scale))
    )
    assert candidate.ready_to_entry_l2_scaled == pytest.approx(
        np.linalg.norm(pre_delta * scale)
    )
    assert candidate.coordinate_units_mixed is True
    assert candidate.ready_to_entry_max_abs_rad is None
    assert candidate.ready_to_entry_l2_rad is None
    assert candidate.core_arc_length_l2_rad is None

    explicit_default = _build(
        source_q=source,
        coordinate_scale=np.ones(3),
        coordinate_semantics="joint_rad",
        coordinate_units="rad",
    )
    implicit_default = _build(source_q=source)
    assert np.array_equal(explicit_default.q_path, implicit_default.q_path)
    assert (
        explicit_default.continuity_report["coordinate_contract"][
            "legacy_rad_metrics_published"
        ]
        is True
    )


def test_entry_exit_candidates_enumerate_every_legal_frame_pair():
    source = _source(frames=9)
    candidates = cmg.enumerate_entry_exit_candidates(
        source,
        READY,
        window_start=2,
        window_end=5,
        window_halo=1,
    )
    expected_pairs = {
        (entry, exit_) for entry in range(0, 2) for exit_ in range(6, 9)
    }
    assert {(c.entry_frame, c.exit_frame) for c in candidates} == expected_pairs
    assert len(candidates) == 6
    assert [c.entry_frame for c in candidates] == [0, 0, 0, 1, 1, 1]
    assert all(c.entry_to_window_frames >= 1 for c in candidates)
    assert all(c.window_to_exit_frames >= 1 for c in candidates)
    assert any(c.includes_source_frame_zero for c in candidates)
    assert any(not c.includes_source_frame_zero for c in candidates)
    assert all(c.ready_to_entry_max_abs_rad is not None for c in candidates)
    assert all(
        c.ready_to_entry_max_abs_scaled == c.ready_to_entry_max_abs_rad
        for c in candidates
    )

    # A one-frame opportunity with halo=0 still emits only buildable
    # entry<exit pairs; it never invents a zero-length retained core.
    single_frame = cmg.enumerate_entry_exit_candidates(
        source,
        READY,
        window_start=4,
        window_end=4,
        window_halo=0,
    )
    assert len(single_frame) == 24
    assert all(c.entry_frame < c.exit_frame for c in single_frame)


def test_discarded_prefix_cannot_pollute_retained_core_derivatives():
    source_a = np.asarray(
        [[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    source_b = source_a.copy()
    source_b[0] = np.asarray([1000.0, -1000.0])
    kwargs = dict(
        ready_q=np.array([0.0, -1.0]),
        entry_frame=1,
        exit_frame=3,
        window_start=1,
        window_end=3,
        window_halo=0,
    )
    result_a = cmg.build_canonical_geometry(source_a, **kwargs)
    result_b = cmg.build_canonical_geometry(source_b, **kwargs)
    assert np.array_equal(result_a.path_parameter, result_b.path_parameter)
    np.testing.assert_array_equal(
        result_a.source_frame_map, result_b.source_frame_map
    )
    assert np.array_equal(result_a.q_path, result_b.q_path)
    assert np.array_equal(result_a.dq_ds, result_b.dq_ds)
    assert np.array_equal(result_a.d2q_ds2, result_b.d2q_ds2)


def test_reverse_chord_projection_is_accepted_when_speed_norm_is_regular():
    segment = cmg._Segment(
        q0=np.array([0.0, 0.0]),
        v0=np.array([1.0, 0.0]),
        a0=np.zeros(2),
        q1=np.array([1.0, 0.0]),
        # The segment arrives while moving partly away from its chord target.
        # That is analogous to follow-through before a return-to-ready curve.
        v1=np.array([-0.2, 0.5]),
        a1=np.zeros(2),
        s0=0.0,
        span=1.0,
        sample_intervals=5,
        label="reverse_entry_tangent_fixture",
        source_frame_start=None,
        source_frame_end=None,
    )
    certificate = cmg._segment_regularity_certificate(
        segment, np.ones(2), segment_index=0
    )
    assert (
        certificate["certification_method"]
        == "degree8_weighted_speed_squared_bernstein_subdivision_v1"
    )
    assert certificate["chord_projection_control_min_per_local_u"] < 0.0
    assert (
        certificate["weighted_speed_lower_bound_per_path_unit"]
        > certificate["required_weighted_speed_margin_per_path_unit"]
    )
    assert certificate["subdivision_max_depth"] > 0


def test_nearly_zero_ready_chord_fails_closed_as_numerically_nonregular():
    segment = cmg._Segment(
        q0=np.zeros(2),
        v0=np.array([1e-14, 0.0]),
        a0=np.zeros(2),
        q1=np.array([1e-14, 0.0]),
        v1=np.array([1e-14, 0.0]),
        a1=np.zeros(2),
        s0=0.0,
        span=1.0,
        sample_intervals=5,
        label="near_zero_chord_fixture",
        source_frame_start=None,
        source_frame_end=None,
    )
    with pytest.raises(ValueError, match="regularity could not be certified"):
        cmg._segment_regularity_certificate(
            segment, np.ones(2), segment_index=0
        )


def test_internal_cusp_fails_closed_without_dense_sampling():
    segment = cmg._Segment(
        q0=np.zeros(2),
        v0=np.array([1.0, 0.0]),
        a0=np.zeros(2),
        q1=np.zeros(2),
        v1=np.array([1.0, 0.0]),
        a1=np.zeros(2),
        s0=0.0,
        span=1.0,
        sample_intervals=5,
        label="internal_cusp_fixture",
        source_frame_start=None,
        source_frame_end=None,
    )
    with pytest.raises(
        ValueError, match="Refusing a possible cusp or numerically unresolved"
    ):
        cmg._segment_regularity_certificate(
            segment, np.ones(2), segment_index=0
        )


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"source_q": np.zeros((2, 3))}, ValueError, "at least 3 frames"),
        ({"ready_q": np.zeros(4)}, ValueError, "ready_q must have shape"),
        ({"entry_frame": 4}, ValueError, "window_start-window_halo"),
        ({"exit_frame": 8}, ValueError, "window_end\\+window_halo"),
        ({"entry_frame": 10, "exit_frame": 10}, ValueError, "strictly before"),
        ({"window_start": 8, "window_end": 7}, ValueError, "must be <="),
        ({"entry_frame": 3.0}, TypeError, "must be an integer"),
        ({"samples_per_rad": 0}, ValueError, "must be > 0"),
        ({"connector_parameter_span": np.inf}, ValueError, "finite"),
        ({"connector_parameter_span": 1e-200}, ValueError, "ill-conditioned"),
        ({"min_core_intervals": 4}, ValueError, "must be >= 5"),
        ({"coordinate_scale": np.ones(2)}, ValueError, "shape \\(3,\\)"),
        (
            {"coordinate_scale": np.array([1.0, 0.0, 1.0])},
            ValueError,
            "must all be > 0",
        ),
        (
            {"coordinate_semantics": ["root_x", "root_y"]},
            ValueError,
            "must have length 3",
        ),
        (
            {"coordinate_units": ["m", "", "rad"]},
            ValueError,
            "non-empty strings",
        ),
    ],
)
def test_bad_inputs_fail_closed(overrides, error, match):
    with pytest.raises(error, match=match):
        _build(**overrides)


def test_nonfinite_input_and_impossible_candidate_halo_fail_closed():
    source = _source()
    source[4, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        _build(source_q=source)

    with pytest.raises(ValueError, match="leaves no legal"):
        cmg.enumerate_entry_exit_candidates(
            _source(frames=9),
            READY,
            window_start=1,
            window_end=7,
            window_halo=2,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="non-finite"):
            cmg.build_canonical_geometry(
                np.asarray([1e308, -1e308, 1e308])[:, None],
                ready_q=np.array([0.0]),
                entry_frame=0,
                exit_frame=2,
                window_start=0,
                window_end=2,
                window_halo=0,
            )

    with pytest.raises(ValueError, match="path regularity could not be certified"):
        cmg.build_canonical_geometry(
            np.zeros((3, 1)),
            ready_q=np.array([-1.0]),
            entry_frame=0,
            exit_frame=2,
            window_start=0,
            window_end=2,
            window_halo=0,
        )
