"""Tests for the sampling-independent weighted arc-length evaluator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _REPO
    / "hope_training"
    / "whole_body_tracking"
    / "scripts"
    / "canonical_weighted_arc_path.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "canonical_weighted_arc_path", _MODULE_PATH
)
M = importlib.util.module_from_spec(_SPEC)
sys.modules["canonical_weighted_arc_path"] = M
_SPEC.loader.exec_module(M)


def _nonlinear_knots(parameter_scale: float = 1.0):
    """Exact 2-jets of one regular nonlinear curve under s re-scaling."""

    base = np.asarray([0.0, 0.3, 0.8, 1.4, 2.0])
    t = base
    q = np.column_stack(
        (
            t + 0.08 * np.sin(1.3 * t),
            0.25 * t**2 + 0.12 * np.sin(0.7 * t),
            -0.2 * t + 0.04 * t**3,
        )
    )
    q_t = np.column_stack(
        (
            1.0 + 0.104 * np.cos(1.3 * t),
            0.5 * t + 0.084 * np.cos(0.7 * t),
            -0.2 + 0.12 * t**2,
        )
    )
    q_tt = np.column_stack(
        (
            -0.1352 * np.sin(1.3 * t),
            0.5 - 0.0588 * np.sin(0.7 * t),
            0.24 * t,
        )
    )
    s = parameter_scale * base
    return (
        s,
        q,
        q_t / parameter_scale,
        q_tt / parameter_scale**2,
    )


def _build_nonlinear(parameter_scale: float = 1.0, **kwargs):
    s, q, q_s, q_ss = _nonlinear_knots(parameter_scale)
    return M.build_weighted_arc_path(
        s_knots=s,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        coordinate_scale=np.asarray([1.0, 1.7, 0.6]),
        **kwargs,
    )


def test_straight_line_has_closed_form_length_and_arc_jets():
    s = np.asarray([-2.0, 0.25, 3.0])
    direction = np.asarray([1.0, -0.4, 0.25])
    origin = np.asarray([0.3, -0.2, 0.7])
    q = origin + s[:, None] * direction
    q_s = np.broadcast_to(direction, q.shape).copy()
    q_ss = np.zeros_like(q)
    scale = np.asarray([2.0, 0.5, 1.25])
    path = M.build_weighted_arc_path(
        s_knots=s,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        coordinate_scale=scale,
    )
    speed = np.linalg.norm(scale * direction)
    np.testing.assert_allclose(
        path.segment_lengths, np.diff(s) * speed, rtol=2e-14, atol=2e-14
    )
    assert path.total_length == pytest.approx(
        (s[-1] - s[0]) * speed, rel=2e-14, abs=2e-14
    )

    lengths = np.linspace(0.0, path.total_length, 17)
    expected_s = s[0] + lengths / speed
    np.testing.assert_allclose(
        path.s_from_l(lengths), expected_s, rtol=1e-12, atol=2e-12
    )
    q_l, expected_q_l = None, direction / speed
    values, first, second = path.evaluate_l(lengths)
    np.testing.assert_allclose(
        values, origin + expected_s[:, None] * direction, atol=2e-12
    )
    np.testing.assert_allclose(
        first,
        np.broadcast_to(expected_q_l, first.shape),
        atol=2e-12,
    )
    np.testing.assert_allclose(second, 0.0, atol=2e-12)


def test_exact_quintic_two_jet_reconstruction_on_nonuniform_knots():
    knots = np.asarray([-0.4, 0.2, 1.3])

    def polynomial(s):
        return np.column_stack(
            (
                0.2 - 0.7 * s + 0.3 * s**2 + 0.1 * s**3
                - 0.04 * s**4 + 0.02 * s**5,
                -0.5 + 0.4 * s - 0.2 * s**2 + 0.03 * s**5,
            )
        )

    def first(s):
        return np.column_stack(
            (
                -0.7 + 0.6 * s + 0.3 * s**2
                - 0.16 * s**3 + 0.1 * s**4,
                0.4 - 0.4 * s + 0.15 * s**4,
            )
        )

    def second(s):
        return np.column_stack(
            (
                0.6 + 0.6 * s - 0.48 * s**2 + 0.4 * s**3,
                -0.4 + 0.6 * s**3,
            )
        )

    path = M.build_weighted_arc_path(
        s_knots=knots,
        q=polynomial(knots),
        q_s=first(knots),
        q_ss=second(knots),
        coordinate_scale=np.asarray([1.0, 0.8]),
    )
    probe = np.linspace(knots[0], knots[-1], 151)
    q, q_s, q_ss = path.evaluate_s(probe)
    np.testing.assert_allclose(q, polynomial(probe), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(q_s, first(probe), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(q_ss, second(probe), rtol=2e-11, atol=2e-11)


def test_nonlinear_lengths_match_independent_high_resolution_reference():
    path = _build_nonlinear(
        arc_absolute_tolerance=2e-13,
        arc_relative_tolerance=2e-13,
    )
    # Independent composite trapezoid reference evaluates the exact quintic
    # q_s directly on a deliberately dense parameter grid.
    for segment in range(len(path.segment_lengths)):
        s_grid = np.linspace(
            path.s_knots[segment],
            path.s_knots[segment + 1],
            300_001,
        )
        _, q_s, _ = path.evaluate_s(s_grid)
        speed = np.linalg.norm(
            q_s * path.coordinate_scale[None, :], axis=1
        )
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        reference = trapezoid(speed, s_grid)
        assert path.segment_lengths[segment] == pytest.approx(
            reference, rel=3e-10, abs=3e-11
        )
        audit = path.segment_audits[segment]
        assert audit.quadrature_evaluation_count >= 24
        assert audit.quadrature_error_estimate >= 0.0
        assert (
            audit.certified_min_weighted_speed_per_s
            > path.regularity_margin
        )


def test_parameter_rescaling_preserves_arc_geometry_not_content_identity():
    original = _build_nonlinear(1.0)
    rescaled = _build_nonlinear(7.25)
    np.testing.assert_allclose(
        rescaled.segment_lengths,
        original.segment_lengths,
        rtol=2e-12,
        atol=2e-12,
    )
    assert rescaled.total_length == pytest.approx(
        original.total_length, rel=2e-12, abs=2e-12
    )
    l = np.linspace(0.0, original.total_length, 101)
    oq, oq_l, oq_ll = original.evaluate_l(l)
    rq, rq_l, rq_ll = rescaled.evaluate_l(l)
    np.testing.assert_allclose(rq, oq, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(rq_l, oq_l, rtol=5e-9, atol=5e-10)
    np.testing.assert_allclose(rq_ll, oq_ll, rtol=2e-8, atol=2e-9)
    assert original.content_sha256 != rescaled.content_sha256


def test_length_parameter_round_trip_and_knot_exactness():
    path = _build_nonlinear()
    rng = np.random.default_rng(20260724)
    s = np.concatenate(
        (
            path.s_knots,
            rng.uniform(path.s_knots[0], path.s_knots[-1], 80),
        )
    )
    length = path.arc_length_at_s(s)
    recovered = path.s_from_l(length)
    np.testing.assert_allclose(recovered, s, rtol=0.0, atol=3e-10)
    assert np.array_equal(path.s_from_l(path.l_knots), path.s_knots)
    q, q_s, q_ss = path.evaluate_s(path.s_knots)
    assert np.array_equal(q, path.q_knots)
    assert np.array_equal(q_s, path.q_s_knots)
    assert np.array_equal(q_ss, path.q_ss_knots)


def test_inverse_interval_contains_true_boundary_when_point_inverse_stops_inward(
    monkeypatch,
):
    """A tolerance-valid inward point must not shrink an extremum interval."""

    path = _build_nonlinear(
        inverse_absolute_tolerance=2.0e-4,
        inverse_relative_tolerance=1.0e-12,
        inverse_parameter_tolerance=1.0e-12,
    )
    true_s = 0.27
    target_length = path.arc_length_at_s(true_s)
    original_inverse = M.WeightedArcPath._u_from_local_length

    def tolerance_valid_inward_inverse(self, segment, target):
        exactish_u = original_inverse(self, segment, target)
        # First-segment width is 0.3, so this is a 3e-5 inward shift in s.
        # Its weighted-length residual is comfortably below the configured
        # 2e-4 stopping tolerance and is therefore a legal point estimate.
        return max(0.0, exactish_u - 1.0e-4)

    monkeypatch.setattr(
        M.WeightedArcPath,
        "_u_from_local_length",
        tolerance_valid_inward_inverse,
    )
    point = path.s_from_l(target_length)
    lower, upper = path.s_interval_from_l(target_length)
    assert point < true_s
    assert lower <= true_s <= upper

    # This coordinate's q_s is increasing here.  Treating the inward point as
    # the exact upper cell boundary would miss the true boundary extremum;
    # the outward interval includes and therefore dominates it.
    _, point_q_s, _ = path.evaluate_s(point)
    _, true_q_s, _ = path.evaluate_s(true_s)
    _, upper_q_s, _ = path.evaluate_s(upper)
    assert point_q_s[1] < true_q_s[1]
    assert upper_q_s[1] >= true_q_s[1]

    knot_lower, knot_upper = path.s_interval_from_l(path.l_knots)
    assert np.array_equal(knot_lower, path.s_knots)
    assert np.array_equal(knot_upper, path.s_knots)


@pytest.mark.parametrize(
    "q_s",
    [
        np.asarray([[0.0], [0.0]]),
        # q(u)=u^2 has a zero-speed endpoint despite nonzero displacement.
        np.asarray([[0.0], [2.0]]),
        # Hermite data q(0)=0,q(1)=0 and opposite endpoint slopes forces an
        # interior turnaround/cusp.
        np.asarray([[1.0], [-1.0]]),
    ],
)
def test_zero_speed_endpoint_or_interior_cusp_is_rejected(q_s):
    q = (
        np.asarray([[0.0], [1.0]])
        if q_s[1, 0] >= 0.0
        else np.asarray([[0.0], [0.0]])
    )
    with pytest.raises(M.WeightedArcPathError, match="regular|margin|speed"):
        M.build_weighted_arc_path(
            s_knots=np.asarray([0.0, 1.0]),
            q=q,
            q_s=q_s,
            q_ss=np.zeros_like(q),
            coordinate_scale=np.asarray([1.0]),
        )


def test_near_zero_speed_below_explicit_margin_is_rejected():
    q = np.asarray([[0.0], [1e-12]])
    with pytest.raises(M.WeightedArcPathError, match="margin"):
        M.build_weighted_arc_path(
            s_knots=np.asarray([0.0, 1.0]),
            q=q,
            q_s=np.asarray([[1e-12], [1e-12]]),
            q_ss=np.zeros_like(q),
            coordinate_scale=np.asarray([1.0]),
            regularity_margin=1e-10,
        )


def test_digest_is_deterministic_binds_results_and_owned_inputs():
    s, q, q_s, q_ss = _nonlinear_knots()
    scale = np.asarray([1.0, 1.7, 0.6])
    first = M.build_weighted_arc_path(
        s_knots=s,
        q=q,
        q_s=q_s,
        q_ss=q_ss,
        coordinate_scale=scale,
    )
    second = M.build_weighted_arc_path(
        s_knots=s.copy(),
        q=q.copy(),
        q_s=q_s.copy(),
        q_ss=q_ss.copy(),
        coordinate_scale=scale.copy(),
    )
    assert first.content_sha256 == second.content_sha256
    assert first.verify_content_digest()
    assert len(first.content_sha256) == 64

    # Caller mutation cannot alter the evaluator because inputs are owned.
    q[0, 0] += 99.0
    scale[0] += 10.0
    assert first.verify_content_digest()
    assert first.content_sha256 == second.content_sha256
    with pytest.raises(ValueError):
        first.segment_lengths[0] += 1.0

    changed = _build_nonlinear()
    changed_scale = M.build_weighted_arc_path(
        s_knots=changed.s_knots,
        q=changed.q_knots,
        q_s=changed.q_s_knots,
        q_ss=changed.q_ss_knots,
        coordinate_scale=np.asarray([1.0, 1.7001, 0.6]),
    )
    assert changed_scale.content_sha256 != changed.content_sha256

    # A forced in-memory alteration is detected on recomputation as well.
    tampered = changed.segment_lengths.copy()
    tampered[0] = np.nextafter(tampered[0], np.inf)
    tampered.setflags(write=False)
    object.__setattr__(changed, "segment_lengths", tampered)
    assert not changed.verify_content_digest()

    audit_tampered = _build_nonlinear()
    object.__setattr__(
        audit_tampered.segment_audits[0],
        "length",
        np.nextafter(audit_tampered.segment_audits[0].length, np.inf),
    )
    assert not audit_tampered.verify_content_digest()


def test_visualization_sampling_density_is_not_an_api_or_digest_input():
    path = _build_nonlinear()
    sparse = path.evaluate_l(np.linspace(0.0, path.total_length, 7))[0]
    dense = path.evaluate_l(np.linspace(0.0, path.total_length, 7001))[0]
    assert sparse.shape == (7, path.dimension)
    assert dense.shape == (7001, path.dimension)
    assert path.verify_content_digest()
    assert "sampling" not in M.build_weighted_arc_path.__annotations__


def test_chain_rule_arc_jets_match_finite_difference_away_from_knots():
    path = _build_nonlinear(
        arc_absolute_tolerance=1e-13,
        arc_relative_tolerance=1e-13,
    )
    l = np.asarray(
        [
            0.17 * path.total_length,
            0.41 * path.total_length,
            0.73 * path.total_length,
        ]
    )
    step = 2e-5 * path.total_length
    q, q_l, q_ll = path.evaluate_l(l)
    q_minus = path.evaluate_l(l - step)[0]
    q_plus = path.evaluate_l(l + step)[0]
    fd_first = (q_plus - q_minus) / (2.0 * step)
    fd_second = (q_plus - 2.0 * q + q_minus) / (step * step)
    np.testing.assert_allclose(q_l, fd_first, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(q_ll, fd_second, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "override",
    [
        {"s_knots": [0.0, 0.0]},
        {"coordinate_scale": [1.0, 0.0, 1.0]},
        {"q_s": [[1.0, 0.0, 0.0]]},
        {"q_ss": [[0.0, 0.0, np.nan], [0.0, 0.0, 0.0]]},
    ],
)
def test_malformed_inputs_fail_closed(override):
    values = {
        "s_knots": [0.0, 1.0],
        "q": [[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]],
        "q_s": [[1.0, 0.2, -0.1], [1.0, 0.2, -0.1]],
        "q_ss": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "coordinate_scale": [1.0, 1.0, 1.0],
    }
    values.update(override)
    with pytest.raises(M.WeightedArcPathError):
        M.build_weighted_arc_path(**values)
