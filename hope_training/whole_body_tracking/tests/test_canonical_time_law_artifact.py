"""Pure-array and file-integrity tests for canonical_time_law_artifact."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "canonical_time_law_artifact.py"
)
_SPEC = importlib.util.spec_from_file_location("canonical_time_law_artifact", _SCRIPT)
artifact_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = artifact_module
_SPEC.loader.exec_module(artifact_module)


def _qpos(path_s: np.ndarray) -> np.ndarray:
    value = np.asarray(path_s, dtype=np.float64)
    return np.stack((value, value * value), axis=-1)


def _q_s(path_s: np.ndarray) -> np.ndarray:
    value = np.asarray(path_s, dtype=np.float64)
    return np.stack((np.ones_like(value), 2.0 * value), axis=-1)


def _q_ss(path_s: np.ndarray) -> np.ndarray:
    value = np.asarray(path_s, dtype=np.float64)
    return np.stack((np.zeros_like(value), 2.0 * np.ones_like(value)), axis=-1)


def _trace() -> artifact_module.TimeLawTrace:
    # Four equal path cells.  x=[0,1,1,1,0] gives
    # u=[+2,0,0,-2] and an exactly 1.5 s rest-to-rest traversal.
    s_node = np.linspace(0.0, 1.0, 5)
    s_mid = 0.5 * (s_node[:-1] + s_node[1:])
    x_node = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    x_mid = 0.5 * (x_node[:-1] + x_node[1:])
    ds = np.diff(s_node)
    u_cell = np.diff(x_node) / (2.0 * ds)
    speed = np.sqrt(x_node)
    cell_dt = 2.0 * ds / (speed[:-1] + speed[1:])
    time_node = np.concatenate(([0.0], np.cumsum(cell_dt)))
    time_mid = time_node[:-1] + ds / (speed[:-1] + np.sqrt(x_mid))

    qpos_node = _qpos(s_node)
    qpos_mid = _qpos(s_mid)
    q_s_node = _q_s(s_node)
    q_s_mid = _q_s(s_mid)
    q_ss_node_left = _q_ss(s_node)
    q_ss_node_right = _q_ss(s_node)
    q_ss_mid = _q_ss(s_mid)

    collocation_qpos = np.stack((qpos_node[:-1], qpos_mid, qpos_node[1:]), axis=1)
    collocation_q_s = np.stack((q_s_node[:-1], q_s_mid, q_s_node[1:]), axis=1)
    collocation_q_ss = np.stack(
        (q_ss_node_right[:-1], q_ss_mid, q_ss_node_left[1:]),
        axis=1,
    )
    collocation_x = np.stack((x_node[:-1], x_mid, x_node[1:]), axis=1)
    collocation_qvel = collocation_q_s * np.sqrt(collocation_x)[:, :, None]
    collocation_qacc = (
        collocation_q_s * u_cell[:, None, None]
        + collocation_q_ss * collocation_x[:, :, None]
    )

    tick_time = np.arange(76, dtype=np.float64) / 50.0
    tick_s = np.empty(len(tick_time), np.float64)
    tick_x = np.empty(len(tick_time), np.float64)
    tick_u = np.empty(len(tick_time), np.float64)
    for index, time_s in enumerate(tick_time):
        if time_s == time_node[-1]:
            cell = len(u_cell) - 1
            tick_s[index] = s_node[-1]
            tick_x[index] = x_node[-1]
        else:
            cell = int(np.searchsorted(time_node, time_s, side="right") - 1)
            delta_t = time_s - time_node[cell]
            tick_s[index] = (
                s_node[cell]
                + speed[cell] * delta_t
                + 0.5 * u_cell[cell] * delta_t * delta_t
            )
            tick_x[index] = x_node[cell] + 2.0 * u_cell[cell] * (
                tick_s[index] - s_node[cell]
            )
        tick_u[index] = u_cell[cell]
    tick_qpos = _qpos(tick_s)
    tick_q_s = _q_s(tick_s)
    tick_q_ss = _q_ss(tick_s)
    tick_qvel = tick_q_s * np.sqrt(tick_x)[:, None]
    tick_qacc = tick_q_s * tick_u[:, None] + tick_q_ss * tick_x[:, None]

    return artifact_module.TimeLawTrace(
        s_node=s_node,
        s_mid=s_mid,
        qpos_node=qpos_node,
        q_s_node=q_s_node,
        q_ss_node_left=q_ss_node_left,
        q_ss_node_right=q_ss_node_right,
        qpos_mid=qpos_mid,
        q_s_mid=q_s_mid,
        q_ss_mid=q_ss_mid,
        x_node=x_node,
        x_mid=x_mid,
        u_cell=u_cell,
        time_node_s=time_node,
        time_mid_s=time_mid,
        collocation_qpos=collocation_qpos,
        collocation_q_s=collocation_q_s,
        collocation_qvel=collocation_qvel,
        collocation_qacc=collocation_qacc,
        tick_s=tick_s,
        tick_qpos=tick_qpos,
        tick_q_s=tick_q_s,
        tick_q_ss=tick_q_ss,
        tick_qvel=tick_qvel,
        tick_qacc=tick_qacc,
    )


def _bindings(
    trace: artifact_module.TimeLawTrace | None = None,
    *,
    derivative_method: str = "analytic",
    weighted_arc_length: artifact_module.WeightedArcLengthBinding | None = None,
) -> artifact_module.ArtifactBindings:
    evaluated_trace = _trace() if trace is None else trace
    evaluated_arrays_sha256 = artifact_module.path_evaluation_array_sha256(
        evaluated_trace
    )
    producer_receipt_sha256 = artifact_module.path_evaluation_receipt_sha256(
        source_sha256="2" * 64,
        evaluator_id="test_exact_path",
        evaluator_version="1.0",
        derivative_method=derivative_method,
        evaluator_contract_sha256="c" * 64,
        evaluator_implementation_sha256="d" * 64,
        evaluated_arrays_sha256=evaluated_arrays_sha256,
    )
    return artifact_module.ArtifactBindings(
        recipe_sha256="1" * 64,
        source_sha256="2" * 64,
        ready_sha256="3" * 64,
        mjcf_sha256="4" * 64,
        urdf_sha256="5" * 64,
        model_binding_sha256="6" * 64,
        actuator_contract_sha256="7" * 64,
        tools_sha256={
            "torque_solver": "9" * 64,
            "path_builder": "8" * 64,
        },
        solver=artifact_module.SolverBinding(
            solver_id="scipy.optimize.linprog:highs",
            solver_version="test-1.0",
            solver_contract_sha256="a" * 64,
            solver_implementation_sha256="b" * 64,
        ),
        path_evaluator=artifact_module.PathEvaluatorBinding(
            evaluator_id="test_exact_path",
            evaluator_version="1.0",
            derivative_method=derivative_method,
            evaluator_contract_sha256="c" * 64,
            evaluator_implementation_sha256="d" * 64,
            evaluated_arrays_sha256=evaluated_arrays_sha256,
            producer_receipt_sha256=producer_receipt_sha256,
        ),
        weighted_arc_length=weighted_arc_length,
    )


def _weighted_arc_binding(
    trace: artifact_module.TimeLawTrace | None = None,
    *,
    evaluated_arrays_sha256: str | None = None,
) -> artifact_module.WeightedArcLengthBinding:
    evaluated_trace = _trace() if trace is None else trace
    provisional = artifact_module.WeightedArcLengthBinding(
        algorithm_id="canonical_weighted_arc_path:test-v1",
        content_sha256="e" * 64,
        retimer_receipt_sha256="f" * 64,
        coordinate_scale_sha256_float64_le="0" * 64,
        l_knots_sha256_float64_le="1" * 64,
        total_length=float(evaluated_trace.s_node[-1]),
        formal_knot_count=len(evaluated_trace.s_node),
        arc_absolute_tolerance=1.0e-12,
        arc_relative_tolerance=1.0e-11,
        quadrature_max_depth=20,
        quadrature_error_estimate_sum=2.0e-13,
        regularity_margin=1.0e-10,
        regularity_max_depth=24,
        certified_min_weighted_speed_per_s=0.9,
        observed_min_weighted_speed_per_s=1.0,
        inverse_absolute_tolerance=2.0e-12,
        inverse_relative_tolerance=2.0e-11,
        inverse_parameter_tolerance=2.0e-13,
        inverse_max_iterations=64,
        evaluated_arrays_sha256=(
            artifact_module.path_evaluation_array_sha256(evaluated_trace)
            if evaluated_arrays_sha256 is None
            else evaluated_arrays_sha256
        ),
        producer_receipt_sha256="2" * 64,
    )
    receipt = artifact_module.weighted_arc_length_receipt_sha256(
        source_sha256="2" * 64,
        binding=provisional,
    )
    return replace(provisional, producer_receipt_sha256=receipt)


def _build(
    *,
    trace: artifact_module.TimeLawTrace | None = None,
    marker_path_s: dict[str, float] | None = None,
    bindings: artifact_module.ArtifactBindings | None = None,
    motion_id: str = "fh_loop",
) -> artifact_module.TimeLawArtifact:
    evaluated_trace = _trace() if trace is None else trace
    return artifact_module.build_time_law_artifact(
        motion_id=motion_id,
        scope="full",
        trace=evaluated_trace,
        marker_path_s=(
            {
                "window_start": 0.25,
                "source_anchor": 0.5,
                "window_end": 0.75,
            }
            if marker_path_s is None
            else marker_path_s
        ),
        bindings=(_bindings(evaluated_trace) if bindings is None else bindings),
    )


def _plain_manifest(
    artifact: artifact_module.TimeLawArtifact,
) -> dict[str, object]:
    return json.loads(artifact_module._json_bytes(artifact.manifest).decode("utf-8"))


def _rewrite_hashes(
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
    npz_bytes: bytes,
) -> None:
    receipts = artifact_module._array_receipts(arrays)
    hashes = manifest["hashes"]
    assert isinstance(hashes, dict)
    hashes["artifact_npz_sha256"] = hashlib.sha256(npz_bytes).hexdigest()
    hashes["array_set_sha256"] = artifact_module._array_set_sha256(receipts)
    hashes["arrays"] = receipts


def _write_malicious_pair(
    tmp_path: Path,
    artifact: artifact_module.TimeLawArtifact,
    *,
    mutate_arrays=None,
    mutate_manifest=None,
) -> tuple[Path, Path]:
    arrays = {key: np.array(value, copy=True) for key, value in artifact.arrays.items()}
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    npz_bytes = artifact_module._deterministic_npz(arrays)
    manifest = _plain_manifest(artifact)
    _rewrite_hashes(manifest, arrays, npz_bytes)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    output = tmp_path / "malicious.npz"
    sidecar = output.with_suffix(".npz.manifest.json")
    output.write_bytes(npz_bytes)
    sidecar.write_bytes(artifact_module._json_bytes(manifest))
    return output, sidecar


def test_build_is_deterministic_little_endian_and_candidate_only():
    first = _build()
    second = _build()
    same_npz_different_manifest = _build(motion_id="bh_loop")

    assert first.npz_bytes == second.npz_bytes
    assert first.output_sha256 == second.output_sha256
    assert first.output_sha256 == first.bundle_sha256
    assert first.output_sha256 != first.npz_sha256
    assert first.manifest_sha256 == hashlib.sha256(first.manifest_bytes).hexdigest()
    assert same_npz_different_manifest.npz_sha256 == first.npz_sha256
    assert same_npz_different_manifest.manifest_sha256 != first.manifest_sha256
    assert same_npz_different_manifest.bundle_sha256 != first.bundle_sha256
    assert first.manifest == second.manifest
    assert tuple(first.arrays) == artifact_module.ARRAY_KEYS
    assert all(
        value.dtype.str in {"<f8", "<i8", "|i1"} for value in first.arrays.values()
    )
    assert all(value.flags.writeable is False for value in first.arrays.values())

    manifest = first.manifest
    assert manifest["training_authorized"] is False
    assert manifest["deployment_authorized"] is False
    assert manifest["hardware_authorized"] is False
    assert manifest["claims"] == {
        "sampled_left_midpoint_right_only": True,
        "continuous_cell_certificate": False,
        "global_optimum_certificate": False,
        "producer_path_evaluation_receipt_bound": True,
        "path_derivative_consistency_independently_verified": False,
        "exact_path_evaluation_independently_recomputed": False,
        "finite_difference_generation_forbidden_by_contract": True,
    }
    assert (
        manifest["semantics"]["publication_protocol"]
        == "npz_first_manifest_receipt_last_orphan_npz_is_incomplete"
    )
    assert (
        manifest["semantics"]["path_geometry_evidence"]
        == "producer_receipt_bound_not_independently_recomputed"
    )
    assert list(manifest["bindings"]["tools_sha256"]) == [
        "path_builder",
        "torque_solver",
    ]
    assert tuple(manifest["window"]["positive_overlap_cell_indices"]) == (0, 1, 2)
    assert manifest["markers"]["window_start"]["at_or_after_tick"] == 25
    assert manifest["markers"]["source_anchor"]["at_or_after_tick"] == 38
    assert manifest["markers"]["window_end"]["at_or_after_tick"] == 50
    assert first.arrays["tick_cell_index"][50] == 3
    assert first.arrays["tick_cell_side"][50] == artifact_module.CELL_SIDE_RIGHT


def test_manifest_is_deeply_immutable_after_build():
    artifact = _build()
    with pytest.raises(TypeError):
        artifact.manifest["bindings"]["source_sha256"] = "f" * 64
    with pytest.raises(TypeError):
        artifact.manifest["claims"][
            "path_derivative_consistency_independently_verified"
        ] = True
    with pytest.raises(AttributeError):
        artifact.manifest["window"]["positive_overlap_cell_indices"].append(3)
    artifact_module.validate_time_law_artifact(artifact)


def test_weighted_arc_binding_roundtrips_and_binds_receipt_fields(
    tmp_path: Path,
):
    trace = _trace()
    weighted = _weighted_arc_binding(trace)
    artifact = _build(
        trace=trace,
        bindings=_bindings(trace, weighted_arc_length=weighted),
    )
    stored = artifact.manifest["bindings"]["weighted_arc_length"]
    assert stored["contract"] == "weighted_arc_length_v1"
    assert stored["content_sha256"] == weighted.content_sha256
    assert (
        stored["retimer_receipt_sha256"]
        == weighted.retimer_receipt_sha256
    )
    assert (
        stored["coordinate_scale_sha256_float64_le"]
        == weighted.coordinate_scale_sha256_float64_le
    )
    assert stored["regularity_margin"] == weighted.regularity_margin
    assert (
        stored["producer_receipt_sha256"]
        == weighted.producer_receipt_sha256
    )

    output, _ = artifact_module.write_time_law_artifact(
        artifact, tmp_path / "weighted_time_law.npz"
    )
    reopened = artifact_module.read_time_law_artifact(output)
    assert (
        reopened.manifest["bindings"]["weighted_arc_length"]
        == stored
    )

    digest_changed = replace(
        weighted,
        content_sha256="d" * 64,
        producer_receipt_sha256="2" * 64,
    )
    tolerance_changed = replace(
        weighted,
        arc_absolute_tolerance=2.0e-12,
        producer_receipt_sha256="2" * 64,
    )
    regularity_changed = replace(
        weighted,
        certified_min_weighted_speed_per_s=0.8,
        producer_receipt_sha256="2" * 64,
    )
    receipts = {
        artifact_module.weighted_arc_length_receipt_sha256(
            source_sha256="2" * 64,
            binding=candidate,
        )
        for candidate in (
            weighted,
            digest_changed,
            tolerance_changed,
            regularity_changed,
        )
    }
    assert len(receipts) == 4


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("content_sha256", "d" * 64),
        ("arc_absolute_tolerance", 2.0e-12),
        ("regularity_margin", 2.0e-10),
        ("retimer_receipt_sha256", "a" * 64),
    ],
)
def test_reader_rejects_weighted_arc_receipt_tamper(
    tmp_path: Path,
    field: str,
    replacement: object,
):
    trace = _trace()
    weighted = _weighted_arc_binding(trace)
    artifact = _build(
        trace=trace,
        bindings=_bindings(trace, weighted_arc_length=weighted),
    )

    def mutate(manifest):
        manifest["bindings"]["weighted_arc_length"][field] = replacement

    output, manifest = _write_malicious_pair(
        tmp_path,
        artifact,
        mutate_manifest=mutate,
    )
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="weighted_arc_length producer receipt",
    ):
        artifact_module.read_time_law_artifact(
            output, manifest_path=manifest
        )


def test_weighted_arc_binding_fails_closed_on_array_or_regularity_mismatch():
    trace = _trace()
    wrong_array_binding = _weighted_arc_binding(
        trace,
        evaluated_arrays_sha256="a" * 64,
    )
    wrong_array_bindings = _bindings(
        trace, weighted_arc_length=wrong_array_binding
    )
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="weighted arc evidence does not bind",
    ):
        _build(trace=trace, bindings=wrong_array_bindings)

    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="regularity minima must exceed",
    ):
        replace(
            _weighted_arc_binding(trace),
            certified_min_weighted_speed_per_s=5.0e-11,
        )


def test_coordinated_fake_path_derivatives_fail_without_new_producer_evidence():
    trace = _trace()
    original_evidence = _bindings(trace)
    q_s_node = trace.q_s_node + 0.25
    q_s_mid = trace.q_s_mid + 0.25
    tick_q_s = trace.tick_q_s + 0.25
    collocation_q_s = np.stack((q_s_node[:-1], q_s_mid, q_s_node[1:]), axis=1)
    collocation_x = np.stack((trace.x_node[:-1], trace.x_mid, trace.x_node[1:]), axis=1)
    collocation_q_ss = np.stack(
        (
            trace.q_ss_node_right[:-1],
            trace.q_ss_mid,
            trace.q_ss_node_left[1:],
        ),
        axis=1,
    )
    collocation_qvel = collocation_q_s * np.sqrt(collocation_x)[:, :, None]
    collocation_qacc = (
        collocation_q_s * trace.u_cell[:, None, None]
        + collocation_q_ss * collocation_x[:, :, None]
    )
    derived = artifact_module._derived_tick_arrays(
        s_node=trace.s_node,
        x_node=trace.x_node,
        u_cell=trace.u_cell,
        time_node_s=trace.time_node_s,
        tick_count=len(trace.tick_s),
        fps_hz=artifact_module.FPS_HZ,
    )
    tick_u = trace.u_cell[derived["tick_cell_index"]]
    tick_qvel = tick_q_s * np.sqrt(derived["tick_x"])[:, None]
    tick_qacc = (
        tick_q_s * tick_u[:, None] + trace.tick_q_ss * derived["tick_x"][:, None]
    )
    forged = replace(
        trace,
        q_s_node=q_s_node,
        q_s_mid=q_s_mid,
        collocation_q_s=collocation_q_s,
        collocation_qvel=collocation_qvel,
        collocation_qacc=collocation_qacc,
        tick_q_s=tick_q_s,
        tick_qvel=tick_qvel,
        tick_qacc=tick_qacc,
    )
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="producer path-evaluation evidence",
    ):
        _build(trace=forged, bindings=original_evidence)


def test_path_evaluator_contract_rejects_finite_difference_method():
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="derivative_method",
    ):
        _bindings(derivative_method="finite_difference")


def test_exact_node_side_and_ceil_do_not_snap_adjacent_floats():
    cell, side, _, _ = artifact_module._cell_for_time(
        0.5,
        np.array([0.0, np.nextafter(0.5, np.inf), 1.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, -2.0]),
    )
    assert (cell, side) == (0, artifact_module.CELL_SIDE_INTERIOR)

    cell, side, _, _ = artifact_module._cell_for_time(
        0.5,
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, -2.0]),
    )
    assert (cell, side) == (1, artifact_module.CELL_SIDE_RIGHT)
    cell, side, _, _ = artifact_module._cell_for_time(
        np.nextafter(0.5, np.inf),
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, -2.0]),
    )
    assert (cell, side) == (1, artifact_module.CELL_SIDE_INTERIOR)
    assert artifact_module._tick_at_or_after(1.0, 1.0) == 1
    assert artifact_module._tick_at_or_after(np.nextafter(1.0, np.inf), 1.0) == 2

    trace = _trace()
    nominal_node_time = trace.time_node_s.copy()
    nominal_node_time[1] = np.nextafter(nominal_node_time[1], np.inf)
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="cell-side semantics would be ambiguous",
    ):
        _build(
            trace=replace(trace, time_node_s=nominal_node_time),
            bindings=_bindings(trace),
        )


def test_internal_node_acceleration_preserves_both_cell_sides():
    artifact = _build()
    left_cell_right_qacc = artifact.arrays["collocation_qacc"][0, 2]
    right_cell_left_qacc = artifact.arrays["collocation_qacc"][1, 0]
    assert not np.array_equal(left_cell_right_qacc, right_cell_left_qacc)
    assert (
        artifact.manifest["semantics"]["node_acceleration"]
        == "cell_sided; q_ss_node_left is the lower-s limit and "
        "q_ss_node_right is the higher-s limit; cell i uses right[i] at "
        "its start and left[i+1] at its end"
    )


def test_q_ss_jump_uses_explicit_cell_endpoint_and_tick_sides():
    trace = _trace()
    q_ss_left = np.array(trace.q_ss_node_left, copy=True)
    q_ss_right = np.array(trace.q_ss_node_right, copy=True)
    # Node 3 lands exactly at t=1.0 (tick 50).  Give its two geometric
    # one-sided limits visibly different values.
    q_ss_left[3, 1] = 5.0
    q_ss_right[3, 1] = 7.0
    collocation_q_ss = np.stack(
        (q_ss_right[:-1], trace.q_ss_mid, q_ss_left[1:]), axis=1
    )
    collocation_x = np.stack(
        (trace.x_node[:-1], trace.x_mid, trace.x_node[1:]), axis=1
    )
    collocation_qacc = (
        trace.collocation_q_s * trace.u_cell[:, None, None]
        + collocation_q_ss * collocation_x[:, :, None]
    )
    derived = artifact_module._derived_tick_arrays(
        s_node=trace.s_node,
        x_node=trace.x_node,
        u_cell=trace.u_cell,
        time_node_s=trace.time_node_s,
        tick_count=len(trace.tick_s),
        fps_hz=artifact_module.FPS_HZ,
    )
    tick_q_ss = np.array(trace.tick_q_ss, copy=True)
    tick_q_ss[50] = q_ss_right[3]
    tick_u = trace.u_cell[derived["tick_cell_index"]]
    tick_qacc = (
        trace.tick_q_s * tick_u[:, None]
        + tick_q_ss * derived["tick_x"][:, None]
    )
    sided = replace(
        trace,
        q_ss_node_left=q_ss_left,
        q_ss_node_right=q_ss_right,
        collocation_qacc=collocation_qacc,
        tick_q_ss=tick_q_ss,
        tick_qacc=tick_qacc,
    )
    artifact = _build(trace=sided)
    np.testing.assert_array_equal(
        artifact.arrays["q_ss_node_left"][3], q_ss_left[3]
    )
    np.testing.assert_array_equal(
        artifact.arrays["q_ss_node_right"][3], q_ss_right[3]
    )

    wrong_collocation_q_ss = np.stack(
        (q_ss_right[:-1], trace.q_ss_mid, q_ss_right[1:]), axis=1
    )
    wrong_collocation_qacc = (
        trace.collocation_q_s * trace.u_cell[:, None, None]
        + wrong_collocation_q_ss * collocation_x[:, :, None]
    )
    with pytest.raises(
        artifact_module.TimeLawArtifactError, match="collocation_qacc"
    ):
        _build(
            trace=replace(
                sided, collocation_qacc=wrong_collocation_qacc
            )
        )

    wrong_tick_q_ss = np.array(tick_q_ss, copy=True)
    wrong_tick_q_ss[50] = q_ss_left[3]
    wrong_tick_qacc = (
        trace.tick_q_s * tick_u[:, None]
        + wrong_tick_q_ss * derived["tick_x"][:, None]
    )
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="exact-node one-sided q_ss",
    ):
        _build(
            trace=replace(
                sided,
                tick_q_ss=wrong_tick_q_ss,
                tick_qacc=wrong_tick_qacc,
            )
        )


def test_write_read_roundtrip_and_no_clobber(tmp_path: Path):
    artifact = _build()
    output = tmp_path / "time_law.npz"
    paths = artifact_module.write_time_law_artifact(artifact, output)
    assert paths == (output, output.with_suffix(".npz.manifest.json"))

    reopened = artifact_module.read_time_law_artifact(output)
    assert reopened.npz_bytes == artifact.npz_bytes
    assert reopened.manifest == artifact.manifest
    for key in artifact_module.ARRAY_KEYS:
        np.testing.assert_array_equal(reopened.arrays[key], artifact.arrays[key])

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        artifact_module.write_time_law_artifact(artifact, output)


def test_manifest_collision_leaves_no_partial_npz(tmp_path: Path):
    artifact = _build()
    output = tmp_path / "time_law.npz"
    manifest = output.with_suffix(".npz.manifest.json")
    manifest.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        artifact_module.write_time_law_artifact(artifact, output)
    assert not output.exists()
    assert manifest.read_text(encoding="utf-8") == "occupied\n"


def test_broken_symlink_is_occupied_for_no_clobber(tmp_path: Path):
    artifact = _build()
    output = tmp_path / "time_law.npz"
    output.symlink_to(tmp_path / "missing-target")
    assert output.exists() is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        artifact_module.write_time_law_artifact(artifact, output)
    assert output.is_symlink()


def test_reader_rejects_artifact_leaf_symlink(tmp_path: Path):
    artifact = _build()
    real = tmp_path / "real.npz"
    _, manifest = artifact_module.write_time_law_artifact(artifact, real)
    alias = tmp_path / "alias.npz"
    alias.symlink_to(real)
    with pytest.raises(
        artifact_module.TimeLawArtifactError, match="regular non-symlink"
    ):
        artifact_module.read_time_law_artifact(alias, manifest_path=manifest)


@pytest.mark.parametrize(
    ("field", "index", "delta", "message"),
    [
        ("x_mid", (0,), 0.05, "x_mid"),
        ("u_cell", (0,), 0.05, "u_cell"),
        ("time_node_s", (2,), 0.01, "time_node_s"),
        ("collocation_qpos", (0, 1, 0), 0.01, "midpoint qpos"),
        ("collocation_qvel", (0, 1, 0), 0.01, "collocation_qvel"),
        ("collocation_qacc", (0, 1, 0), 0.01, "collocation_qacc"),
        ("tick_s", (10,), 0.01, "tick_s"),
        ("tick_qacc", (10, 0), 0.01, "tick_qacc"),
    ],
)
def test_builder_rejects_critical_array_tamper(
    field: str,
    index: tuple[int, ...],
    delta: float,
    message: str,
):
    trace = _trace()
    changed = np.array(getattr(trace, field), copy=True)
    changed[index] += delta
    with pytest.raises(artifact_module.TimeLawArtifactError, match=message):
        _build(trace=replace(trace, **{field: changed}))


def test_builder_rejects_nonzero_endpoint_speed():
    trace = _trace()
    x_node = trace.x_node.copy()
    x_node[0] = 0.01
    with pytest.raises(artifact_module.TimeLawArtifactError, match="start/end x_node"):
        _build(trace=replace(trace, x_node=x_node))


@pytest.mark.parametrize("source_anchor", [0.125, 0.5, 0.875])
def test_builder_preserves_source_anchor_before_inside_or_after_window(
    source_anchor,
):
    artifact = _build(
        marker_path_s={
            "window_start": 0.25,
            "source_anchor": source_anchor,
            "window_end": 0.75,
        }
    )
    assert artifact.manifest["markers"]["source_anchor"]["path_s"] == (
        source_anchor
    )
    assert artifact.manifest["window"]["window_start_s"] == 0.25
    assert artifact.manifest["window"]["window_end_s"] == 0.75
    assert (
        artifact.manifest["semantics"]["source_anchor_role"]
        == "lineage_and_ranking_reference_may_precede_or_follow_the_"
        "protected_window_not_contact_truth"
    )


@pytest.mark.parametrize("source_anchor", [-0.01, 1.01])
def test_builder_rejects_source_anchor_outside_solved_path(source_anchor):
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="leave the solved path",
    ):
        _build(
            marker_path_s={
                "window_start": 0.25,
                "source_anchor": source_anchor,
                "window_end": 0.75,
            }
        )


def test_builder_rejects_reversed_window_and_early_brake_overlap():
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="window_start <= window_end",
    ):
        _build(
            marker_path_s={
                "window_start": 0.8,
                "source_anchor": 0.5,
                "window_end": 0.75,
            }
        )

    # Extending the window into cell 3 exposes u=-2 before window_end.
    with pytest.raises(artifact_module.TimeLawArtifactError, match="non-negative"):
        _build(
            marker_path_s={
                "window_start": 0.25,
                "source_anchor": 0.5,
                "window_end": 0.9,
            }
        )


def test_builder_rejects_50hz_finite_difference_velocity_fallback():
    trace = _trace()
    finite_difference = np.gradient(
        trace.tick_qpos, 1.0 / artifact_module.FPS_HZ, axis=0, edge_order=2
    )
    assert not np.allclose(finite_difference, trace.tick_qvel)
    with pytest.raises(artifact_module.TimeLawArtifactError, match="tick_qvel"):
        _build(trace=replace(trace, tick_qvel=finite_difference))


def test_bindings_are_strict_and_content_addressed():
    with pytest.raises(
        artifact_module.TimeLawArtifactError, match="lowercase hexadecimal"
    ):
        artifact_module.SolverBinding(
            solver_id="solver",
            solver_version="1",
            solver_contract_sha256="A" * 64,
            solver_implementation_sha256="b" * 64,
        )
    with pytest.raises(artifact_module.TimeLawArtifactError, match="non-empty mapping"):
        artifact_module.ArtifactBindings(
            recipe_sha256="1" * 64,
            source_sha256="2" * 64,
            ready_sha256="3" * 64,
            mjcf_sha256="4" * 64,
            urdf_sha256="5" * 64,
            model_binding_sha256="6" * 64,
            actuator_contract_sha256="7" * 64,
            tools_sha256={},
            solver=artifact_module.SolverBinding(
                solver_id="solver",
                solver_version="1",
                solver_contract_sha256="a" * 64,
                solver_implementation_sha256="b" * 64,
            ),
            path_evaluator=_bindings().path_evaluator,
        )
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="producer_receipt_sha256",
    ):
        replace(_bindings(), source_sha256="f" * 64)


def test_reader_rejects_stale_hash_after_npz_byte_tamper(tmp_path: Path):
    artifact = _build()
    output, _manifest = artifact_module.write_time_law_artifact(
        artifact, tmp_path / "time_law.npz"
    )
    payload = bytearray(output.read_bytes())
    payload[-1] ^= 0x01
    output.write_bytes(payload)
    with pytest.raises(artifact_module.TimeLawArtifactError):
        artifact_module.read_time_law_artifact(output)


def test_reader_rejects_algebra_tamper_even_after_all_hashes_are_recomputed(
    tmp_path: Path,
):
    artifact = _build()

    def mutate(arrays):
        arrays["collocation_qacc"][1, 1, 0] += 0.25

    output, manifest = _write_malicious_pair(tmp_path, artifact, mutate_arrays=mutate)
    with pytest.raises(artifact_module.TimeLawArtifactError, match="collocation_qacc"):
        artifact_module.read_time_law_artifact(output, manifest_path=manifest)


def test_reader_rejects_tick_side_tamper_after_rehash(tmp_path: Path):
    artifact = _build()

    def mutate(arrays):
        arrays["tick_cell_side"][1] = artifact_module.CELL_SIDE_LEFT

    output, manifest = _write_malicious_pair(tmp_path, artifact, mutate_arrays=mutate)
    with pytest.raises(artifact_module.TimeLawArtifactError, match="tick_cell_side"):
        artifact_module.read_time_law_artifact(output, manifest_path=manifest)


def test_reader_rejects_authorization_escalation(tmp_path: Path):
    artifact = _build()

    def mutate(manifest):
        manifest["training_authorized"] = True

    output, manifest = _write_malicious_pair(tmp_path, artifact, mutate_manifest=mutate)
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="training_authorized must be exactly false",
    ):
        artifact_module.read_time_law_artifact(output, manifest_path=manifest)


def test_reader_rejects_bool_masquerading_as_schema_integer(tmp_path: Path):
    artifact = _build()

    def mutate(manifest):
        manifest["schema_version"] = True

    output, manifest = _write_malicious_pair(tmp_path, artifact, mutate_manifest=mutate)
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="schema_version",
    ):
        artifact_module.read_time_law_artifact(output, manifest_path=manifest)


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    [
        ("schema_version", 1),
        ("artifact_type", "canonical_time_law_collocation_v1"),
    ],
)
def test_reader_rejects_legacy_v1_identity(
    tmp_path: Path, field: str, legacy_value: object
):
    artifact = _build()

    def mutate(manifest):
        manifest[field] = legacy_value

    output, manifest = _write_malicious_pair(
        tmp_path,
        artifact,
        mutate_manifest=mutate,
    )
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="identity/version/publication contract changed",
    ):
        artifact_module.read_time_law_artifact(
            output, manifest_path=manifest
        )


def test_reader_rejects_noncanonical_json_bytes(tmp_path: Path):
    artifact = _build()
    output, manifest = artifact_module.write_time_law_artifact(
        artifact, tmp_path / "time_law.npz"
    )
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(artifact_module.TimeLawArtifactError, match="canonical"):
        artifact_module.read_time_law_artifact(output)


def test_reader_rejects_compressed_npz_even_with_recomputed_hash(
    tmp_path: Path,
):
    artifact = _build()
    stream = io.BytesIO()
    np.savez_compressed(stream, **dict(artifact.arrays))
    compressed = stream.getvalue()
    manifest = _plain_manifest(artifact)
    _rewrite_hashes(
        manifest,
        {key: np.array(value) for key, value in artifact.arrays.items()},
        compressed,
    )
    output = tmp_path / "compressed.npz"
    sidecar = output.with_suffix(".npz.manifest.json")
    output.write_bytes(compressed)
    sidecar.write_bytes(artifact_module._json_bytes(manifest))
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="member order|uncompressed|stored|deterministic",
    ):
        artifact_module.read_time_law_artifact(output)


def test_reader_rejects_missing_explicit_qacc_without_fallback(
    tmp_path: Path,
):
    artifact = _build()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in artifact_module.ARRAY_KEYS:
            if key == "collocation_qacc":
                continue
            array_stream = io.BytesIO()
            np.lib.format.write_array(
                array_stream, artifact.arrays[key], allow_pickle=False
            )
            archive.writestr(f"{key}.npy", array_stream.getvalue())
    output = tmp_path / "missing_qacc.npz"
    sidecar = output.with_suffix(".npz.manifest.json")
    output.write_bytes(stream.getvalue())
    sidecar.write_bytes(artifact_module._json_bytes(artifact.manifest))
    with pytest.raises(artifact_module.TimeLawArtifactError, match="keyset"):
        artifact_module.read_time_law_artifact(output)


def test_writer_rejects_symlink_parent_and_cross_directory_pair(tmp_path: Path):
    artifact = _build()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="parent directory.*non-symlink",
    ):
        artifact_module.write_time_law_artifact(artifact, alias_parent / "time_law.npz")
    assert not (real_parent / "time_law.npz").exists()
    artifact_module.write_time_law_artifact(artifact, real_parent / "time_law.npz")
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="parent directory.*non-symlink",
    ):
        artifact_module.read_time_law_artifact(alias_parent / "time_law.npz")

    other_parent = tmp_path / "other"
    other_parent.mkdir()
    with pytest.raises(
        artifact_module.TimeLawArtifactError,
        match="same parent directory",
    ):
        artifact_module.write_time_law_artifact(
            artifact,
            real_parent / "time_law.npz",
            manifest_path=other_parent / "time_law.manifest.json",
        )


def test_manifest_is_receipt_last_and_second_write_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = _build()
    output = tmp_path / "time_law.npz"
    manifest = output.with_suffix(".npz.manifest.json")
    original = artifact_module._exclusive_write
    calls = 0

    def fail_manifest(path, payload, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest receipt failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(artifact_module, "_exclusive_write", fail_manifest)
    with pytest.raises(OSError, match="receipt failure"):
        artifact_module.write_time_law_artifact(artifact, output)
    assert not output.exists()
    assert not manifest.exists()

    # A crash can leave the payload alone; without the receipt it is invalid.
    output.write_bytes(artifact.npz_bytes)
    with pytest.raises(artifact_module.TimeLawArtifactError):
        artifact_module.read_time_law_artifact(output)
