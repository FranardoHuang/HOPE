#!/usr/bin/env python3
"""Content-addressed sampled-collocation time-law artifacts.

This module is deliberately independent of the canonical compiler, schema-2,
MuJoCo, and the grounded torque solver.  A producer must supply the exact
floating-point arrays that it actually solved and checked.  This layer only:

* validates the scalar time law and left/midpoint/right algebra;
* binds explicit parent/model/tool/solver identities;
* records an exact 50 Hz sampling with an unambiguous cell-side convention;
* serializes deterministic little-endian NPZ bytes and a strict JSON manifest;
* reads and writes the pair without following leaf symlinks or clobbering files.

It never reconstructs velocity or acceleration from schema-2 samples.  The
producer must bind q/q_s/q_ss to a non-finite-difference evaluator receipt.
This validator checks that binding and the ``q_s/x/u/q_ss`` time-law equations;
it deliberately does not claim to differentiate qpos independently.

The artifact is sampled evidence only.  Building or successfully reopening one
does not establish a continuous-cell proof, a globally optimal time law, or
training, deployment, or hardware authorization.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_TYPE = "canonical_time_law_collocation_v2"
PUBLICATION_CLASS = "time_law_collocation_candidate"
BUILD_VERDICT = "PASS_SAMPLED_COLLOCATION_CANDIDATE_ONLY"
FPS_HZ = 50.0
COLLOCATION_ORDER = ("left", "midpoint", "right")
SCOPES = ("upper", "full")
PATH_DERIVATIVE_METHODS = (
    "analytic",
    "automatic_differentiation",
    "exact_spline_derivative",
    "symbolic",
)

CELL_SIDE_LEFT = -1
CELL_SIDE_INTERIOR = 0
CELL_SIDE_RIGHT = 1

MARKER_NAMES = ("window_start", "source_anchor", "window_end")

_FLOAT_DTYPE = np.dtype("<f8")
_INDEX_DTYPE = np.dtype("<i8")
_SIDE_DTYPE = np.dtype("i1")
_ABS_TOL = 2.0e-12
_REL_TOL = 2.0e-11
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_CALLER_FLOAT_KEYS = (
    "s_node",
    "s_mid",
    "qpos_node",
    "q_s_node",
    "q_ss_node_left",
    "q_ss_node_right",
    "qpos_mid",
    "q_s_mid",
    "q_ss_mid",
    "x_node",
    "x_mid",
    "u_cell",
    "time_node_s",
    "time_mid_s",
    "collocation_qpos",
    "collocation_q_s",
    "collocation_qvel",
    "collocation_qacc",
    "tick_s",
    "tick_qpos",
    "tick_q_s",
    "tick_q_ss",
    "tick_qvel",
    "tick_qacc",
)
_DERIVED_KEYS = (
    "tick_index",
    "tick_time_s",
    "tick_x",
    "tick_cell_index",
    "tick_cell_side",
)
ARRAY_KEYS = (
    "s_node",
    "s_mid",
    "qpos_node",
    "q_s_node",
    "q_ss_node_left",
    "q_ss_node_right",
    "qpos_mid",
    "q_s_mid",
    "q_ss_mid",
    "x_node",
    "x_mid",
    "u_cell",
    "time_node_s",
    "time_mid_s",
    "collocation_qpos",
    "collocation_q_s",
    "collocation_qvel",
    "collocation_qacc",
    "tick_index",
    "tick_time_s",
    "tick_s",
    "tick_x",
    "tick_cell_index",
    "tick_cell_side",
    "tick_qpos",
    "tick_q_s",
    "tick_q_ss",
    "tick_qvel",
    "tick_qacc",
)
PATH_EVALUATION_ARRAY_KEYS = (
    "qpos_node",
    "q_s_node",
    "q_ss_node_left",
    "q_ss_node_right",
    "qpos_mid",
    "q_s_mid",
    "q_ss_mid",
    "collocation_qpos",
    "collocation_q_s",
    "tick_qpos",
    "tick_q_s",
    "tick_q_ss",
)
_ARRAY_DTYPES = {
    **{key: _FLOAT_DTYPE for key in _CALLER_FLOAT_KEYS},
    "tick_index": _INDEX_DTYPE,
    "tick_time_s": _FLOAT_DTYPE,
    "tick_x": _FLOAT_DTYPE,
    "tick_cell_index": _INDEX_DTYPE,
    "tick_cell_side": _SIDE_DTYPE,
}

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "publication_class",
        "build_verdict",
        "motion_id",
        "scope",
        "fps_hz",
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
        "dimensions",
        "hashes",
        "bindings",
        "markers",
        "window",
        "semantics",
        "claims",
    }
)
_BINDING_KEYS = frozenset(
    {
        "recipe_sha256",
        "source_sha256",
        "ready_sha256",
        "mjcf_sha256",
        "urdf_sha256",
        "model_binding_sha256",
        "actuator_contract_sha256",
        "tools_sha256",
        "solver",
        "path_evaluator",
        "weighted_arc_length",
    }
)
_SOLVER_KEYS = frozenset(
    {
        "solver_id",
        "solver_version",
        "solver_contract_sha256",
        "solver_implementation_sha256",
    }
)
_PATH_EVALUATOR_KEYS = frozenset(
    {
        "evaluator_id",
        "evaluator_version",
        "derivative_method",
        "evaluator_contract_sha256",
        "evaluator_implementation_sha256",
        "evaluated_arrays_sha256",
        "producer_receipt_sha256",
    }
)
_WEIGHTED_ARC_KEYS = frozenset(
    {
        "contract",
        "algorithm_id",
        "content_sha256",
        "retimer_receipt_sha256",
        "coordinate_scale_sha256_float64_le",
        "l_knots_sha256_float64_le",
        "total_length",
        "formal_knot_count",
        "arc_absolute_tolerance",
        "arc_relative_tolerance",
        "quadrature_max_depth",
        "quadrature_error_estimate_sum",
        "regularity_margin",
        "regularity_max_depth",
        "certified_min_weighted_speed_per_s",
        "observed_min_weighted_speed_per_s",
        "inverse_absolute_tolerance",
        "inverse_relative_tolerance",
        "inverse_parameter_tolerance",
        "inverse_max_iterations",
        "evaluated_arrays_sha256",
        "producer_receipt_sha256",
    }
)
_MARKER_ROW_KEYS = frozenset(
    {
        "path_s",
        "path_s_hex",
        "time_s",
        "time_s_hex",
        "fractional_tick",
        "at_or_after_tick",
        "nearest_observation_tick",
    }
)

_SEMANTICS = {
    "state": "x=path_speed_squared",
    "control": "u=piecewise_constant_scalar_path_acceleration",
    "qvel_formula": "qvel=q_s*sqrt(x)",
    "qacc_formula": "qacc=q_s*u+q_ss*x",
    "collocation_order": list(COLLOCATION_ORDER),
    "node_acceleration": (
        "cell_sided; q_ss_node_left is the lower-s limit and "
        "q_ss_node_right is the higher-s limit; cell i uses right[i] at "
        "its start and left[i+1] at its end"
    ),
    "tick_cell_side_codes": {
        str(CELL_SIDE_LEFT): "left_cell_at_final_node",
        str(CELL_SIDE_INTERIOR): "strict_cell_interior",
        str(CELL_SIDE_RIGHT): "right_cell_at_exact_node",
    },
    "tick_state_source": "caller_supplied_path_evaluator_output",
    "time_law_input_contract": (
        "explicit_solver_arrays_only_schema2_finite_difference_forbidden"
    ),
    "path_geometry_evidence": ("producer_receipt_bound_not_independently_recomputed"),
    "source_anchor_role": (
        "lineage_and_ranking_reference_may_precede_or_follow_the_protected_"
        "window_not_contact_truth"
    ),
    "protected_window_role": (
        "window_start_to_window_end_only_controls_contact_observation_and_"
        "no_early_brake"
    ),
    "publication_protocol": (
        "npz_first_manifest_receipt_last_orphan_npz_is_incomplete"
    ),
    "bundle_identity": (
        "sha256_domain_separated_npz_bytes_plus_canonical_manifest_bytes"
    ),
}
_CLAIMS = {
    "sampled_left_midpoint_right_only": True,
    "continuous_cell_certificate": False,
    "global_optimum_certificate": False,
    "producer_path_evaluation_receipt_bound": True,
    "path_derivative_consistency_independently_verified": False,
    "exact_path_evaluation_independently_recomputed": False,
    "finite_difference_generation_forbidden_by_contract": True,
}


class TimeLawArtifactError(ValueError):
    """The artifact cannot be accepted without weakening its contract."""


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TimeLawArtifactError(f"{label} must be a non-empty trimmed string")
    if any(ord(char) < 32 for char in value):
        raise TimeLawArtifactError(f"{label} contains a control character")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _strict_string(value, label)
    if _IDENTIFIER.fullmatch(text) is None:
        raise TimeLawArtifactError(
            f"{label} must contain only letters, digits, '.', '_' or '-'"
        )
    return text


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise TimeLawArtifactError(
            f"{label} must be exactly 64 lowercase hexadecimal SHA-256 digits"
        )
    return value


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TimeLawArtifactError(f"{label} must be a JSON object")
    actual = frozenset(str(key) for key in value)
    if actual != expected or any(not isinstance(key, str) for key in value):
        raise TimeLawArtifactError(
            f"{label} keys changed: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimeLawArtifactError(f"{label} must be one finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TimeLawArtifactError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimeLawArtifactError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise TimeLawArtifactError(f"{label} must be >= {minimum}")
    return int(value)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TimeLawArtifactError("manifest JSON object keys must be strings")
            frozen[key] = _deep_freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TimeLawArtifactError(
        f"manifest contains non-JSON value of type {type(value).__name__}"
    )


def _deep_thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        thawed: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TimeLawArtifactError("manifest JSON object keys must be strings")
            thawed[key] = _deep_thaw_json(item)
        return thawed
    if isinstance(value, (list, tuple)):
        return [_deep_thaw_json(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TimeLawArtifactError(
        f"manifest contains non-JSON value of type {type(value).__name__}"
    )


def _json_same(actual: Any, expected: Any) -> bool:
    """JSON equality that never treats bool/int or int/float as interchangeable."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _json_same(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_same(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


@dataclass(frozen=True)
class SolverBinding:
    """Content identity of the exact solver contract and implementation."""

    solver_id: str
    solver_version: str
    solver_contract_sha256: str
    solver_implementation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "solver_id", _strict_string(self.solver_id, "solver_id")
        )
        object.__setattr__(
            self,
            "solver_version",
            _strict_string(self.solver_version, "solver_version"),
        )
        object.__setattr__(
            self,
            "solver_contract_sha256",
            _digest(self.solver_contract_sha256, "solver_contract_sha256"),
        )
        object.__setattr__(
            self,
            "solver_implementation_sha256",
            _digest(
                self.solver_implementation_sha256,
                "solver_implementation_sha256",
            ),
        )


@dataclass(frozen=True)
class PathEvaluatorBinding:
    """Producer evidence for q(s), q_s(s), and q_ss(s).

    The artifact binds this receipt and the evaluated arrays, but deliberately
    does not claim to rerun the evaluator.  The derivative method is restricted
    to methods that do not manufacture production derivatives by finite
    differencing sampled qpos.
    """

    evaluator_id: str
    evaluator_version: str
    derivative_method: str
    evaluator_contract_sha256: str
    evaluator_implementation_sha256: str
    evaluated_arrays_sha256: str
    producer_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluator_id", _strict_string(self.evaluator_id, "evaluator_id")
        )
        object.__setattr__(
            self,
            "evaluator_version",
            _strict_string(self.evaluator_version, "evaluator_version"),
        )
        method = _strict_string(self.derivative_method, "derivative_method")
        if method not in PATH_DERIVATIVE_METHODS:
            raise TimeLawArtifactError(
                "derivative_method must be one of "
                f"{PATH_DERIVATIVE_METHODS}; finite differences are forbidden"
            )
        object.__setattr__(self, "derivative_method", method)
        for field_name in (
            "evaluator_contract_sha256",
            "evaluator_implementation_sha256",
            "evaluated_arrays_sha256",
            "producer_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class WeightedArcLengthBinding:
    """Digest-bound weighted-arc coordinate used by the sampled time law."""

    algorithm_id: str
    content_sha256: str
    retimer_receipt_sha256: str
    coordinate_scale_sha256_float64_le: str
    l_knots_sha256_float64_le: str
    total_length: float
    formal_knot_count: int
    arc_absolute_tolerance: float
    arc_relative_tolerance: float
    quadrature_max_depth: int
    quadrature_error_estimate_sum: float
    regularity_margin: float
    regularity_max_depth: int
    certified_min_weighted_speed_per_s: float
    observed_min_weighted_speed_per_s: float
    inverse_absolute_tolerance: float
    inverse_relative_tolerance: float
    inverse_parameter_tolerance: float
    inverse_max_iterations: int
    evaluated_arrays_sha256: str
    producer_receipt_sha256: str
    contract: str = "weighted_arc_length_v1"

    def __post_init__(self) -> None:
        if self.contract != "weighted_arc_length_v1":
            raise TimeLawArtifactError(
                "weighted arc contract must be weighted_arc_length_v1"
            )
        object.__setattr__(
            self,
            "algorithm_id",
            _strict_string(self.algorithm_id, "weighted arc algorithm_id"),
        )
        for field_name in (
            "content_sha256",
            "retimer_receipt_sha256",
            "coordinate_scale_sha256_float64_le",
            "l_knots_sha256_float64_le",
            "evaluated_arrays_sha256",
            "producer_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(
                    getattr(self, field_name),
                    f"weighted arc {field_name}",
                ),
            )
        for field_name in (
            "total_length",
            "arc_absolute_tolerance",
            "arc_relative_tolerance",
            "regularity_margin",
            "certified_min_weighted_speed_per_s",
            "observed_min_weighted_speed_per_s",
            "inverse_absolute_tolerance",
            "inverse_relative_tolerance",
            "inverse_parameter_tolerance",
        ):
            value = _finite_float(
                getattr(self, field_name), f"weighted arc {field_name}"
            )
            if value <= 0.0:
                raise TimeLawArtifactError(
                    f"weighted arc {field_name} must be positive"
                )
            object.__setattr__(self, field_name, value)
        error = _finite_float(
            self.quadrature_error_estimate_sum,
            "weighted arc quadrature_error_estimate_sum",
        )
        if error < 0.0:
            raise TimeLawArtifactError(
                "weighted arc quadrature_error_estimate_sum must be non-negative"
            )
        object.__setattr__(
            self, "quadrature_error_estimate_sum", error
        )
        for field_name, minimum in (
            ("formal_knot_count", 2),
            ("quadrature_max_depth", 1),
            ("regularity_max_depth", 1),
            ("inverse_max_iterations", 1),
        ):
            object.__setattr__(
                self,
                field_name,
                _integer(
                    getattr(self, field_name),
                    f"weighted arc {field_name}",
                    minimum=minimum,
                ),
            )
        if (
            self.certified_min_weighted_speed_per_s
            <= self.regularity_margin
            or self.observed_min_weighted_speed_per_s
            <= self.regularity_margin
        ):
            raise TimeLawArtifactError(
                "weighted arc regularity minima must exceed the margin"
            )

    def receipt_payload(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in _WEIGHTED_ARC_KEYS
            if field_name != "producer_receipt_sha256"
        }


def weighted_arc_length_receipt_sha256(
    *,
    source_sha256: str,
    binding: WeightedArcLengthBinding,
) -> str:
    """Bind one weighted arc, its tolerances, regularity, and arrays."""

    if not isinstance(binding, WeightedArcLengthBinding):
        raise TimeLawArtifactError(
            "binding must be a WeightedArcLengthBinding"
        )
    payload = {
        "source_sha256": _digest(
            source_sha256, "weighted arc receipt source_sha256"
        ),
        **binding.receipt_payload(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(
        b"canonical-weighted-arc-artifact-receipt-v1\0" + encoded
    ).hexdigest()


@dataclass(frozen=True)
class ArtifactBindings:
    """All parent, plant, actuator, tool, and solver content identities."""

    recipe_sha256: str
    source_sha256: str
    ready_sha256: str
    mjcf_sha256: str
    urdf_sha256: str
    model_binding_sha256: str
    actuator_contract_sha256: str
    tools_sha256: Mapping[str, str]
    solver: SolverBinding
    path_evaluator: PathEvaluatorBinding
    weighted_arc_length: WeightedArcLengthBinding | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "recipe_sha256",
            "source_sha256",
            "ready_sha256",
            "mjcf_sha256",
            "urdf_sha256",
            "model_binding_sha256",
            "actuator_contract_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        if not isinstance(self.solver, SolverBinding):
            raise TimeLawArtifactError("solver must be a SolverBinding")
        if not isinstance(self.path_evaluator, PathEvaluatorBinding):
            raise TimeLawArtifactError("path_evaluator must be a PathEvaluatorBinding")
        expected_receipt = path_evaluation_receipt_sha256(
            source_sha256=self.source_sha256,
            evaluator_id=self.path_evaluator.evaluator_id,
            evaluator_version=self.path_evaluator.evaluator_version,
            derivative_method=self.path_evaluator.derivative_method,
            evaluator_contract_sha256=(self.path_evaluator.evaluator_contract_sha256),
            evaluator_implementation_sha256=(
                self.path_evaluator.evaluator_implementation_sha256
            ),
            evaluated_arrays_sha256=(self.path_evaluator.evaluated_arrays_sha256),
        )
        if self.path_evaluator.producer_receipt_sha256 != expected_receipt:
            raise TimeLawArtifactError(
                "producer_receipt_sha256 does not bind source, evaluator "
                "contract/implementation, derivative method, and evaluated arrays"
            )
        if self.weighted_arc_length is not None:
            if not isinstance(
                self.weighted_arc_length, WeightedArcLengthBinding
            ):
                raise TimeLawArtifactError(
                    "weighted_arc_length must be a "
                    "WeightedArcLengthBinding or None"
                )
            expected_arc_receipt = weighted_arc_length_receipt_sha256(
                source_sha256=self.source_sha256,
                binding=self.weighted_arc_length,
            )
            if (
                self.weighted_arc_length.producer_receipt_sha256
                != expected_arc_receipt
            ):
                raise TimeLawArtifactError(
                    "weighted arc producer receipt does not bind source, "
                    "arc digest/tolerances/regularity, and evaluated arrays"
                )
        if not isinstance(self.tools_sha256, Mapping) or not self.tools_sha256:
            raise TimeLawArtifactError("tools_sha256 must be a non-empty mapping")
        tools: dict[str, str] = {}
        for raw_name, raw_digest in self.tools_sha256.items():
            name = _identifier(raw_name, "tool name")
            if name in tools:
                raise TimeLawArtifactError(f"duplicate tool binding {name!r}")
            tools[name] = _digest(raw_digest, f"tools_sha256[{name!r}]")
        object.__setattr__(
            self, "tools_sha256", MappingProxyType(dict(sorted(tools.items())))
        )


@dataclass(frozen=True)
class TimeLawTrace:
    """Exact caller-supplied path, time law, collocation, and 50 Hz samples."""

    s_node: np.ndarray
    s_mid: np.ndarray
    qpos_node: np.ndarray
    q_s_node: np.ndarray
    q_ss_node_left: np.ndarray
    q_ss_node_right: np.ndarray
    qpos_mid: np.ndarray
    q_s_mid: np.ndarray
    q_ss_mid: np.ndarray
    x_node: np.ndarray
    x_mid: np.ndarray
    u_cell: np.ndarray
    time_node_s: np.ndarray
    time_mid_s: np.ndarray
    collocation_qpos: np.ndarray
    collocation_q_s: np.ndarray
    collocation_qvel: np.ndarray
    collocation_qacc: np.ndarray
    tick_s: np.ndarray
    tick_qpos: np.ndarray
    tick_q_s: np.ndarray
    tick_q_ss: np.ndarray
    tick_qvel: np.ndarray
    tick_qacc: np.ndarray


@dataclass(frozen=True)
class TimeLawArtifact:
    """Validated in-memory artifact, deterministic bytes, and strict manifest."""

    arrays: Mapping[str, np.ndarray]
    npz_bytes: bytes
    manifest: Mapping[str, Any]
    manifest_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.npz_bytes, bytes):
            raise TimeLawArtifactError("npz_bytes must be immutable bytes")
        if not isinstance(self.manifest_bytes, bytes):
            raise TimeLawArtifactError("manifest_bytes must be immutable bytes")
        object.__setattr__(self, "arrays", _freeze_arrays(self.arrays))
        object.__setattr__(
            self,
            "manifest",
            _deep_freeze_json(_deep_thaw_json(self.manifest)),
        )

    @property
    def output_sha256(self) -> str:
        """Bundle identity; unlike the legacy NPZ hash this binds the manifest."""
        return self.bundle_sha256

    @property
    def npz_sha256(self) -> str:
        return str(self.manifest["hashes"]["artifact_npz_sha256"])

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    @property
    def bundle_sha256(self) -> str:
        return _bundle_sha256(self.npz_bytes, self.manifest_bytes)


def _float_array(value: Any, label: str, *, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TimeLawArtifactError(f"{label} must be a real numeric array")
    try:
        array = np.ascontiguousarray(raw, dtype=_FLOAT_DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimeLawArtifactError(f"{label} cannot be converted to float64") from exc
    if array.ndim != ndim:
        raise TimeLawArtifactError(
            f"{label} must have ndim={ndim}, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise TimeLawArtifactError(f"{label} contains NaN/Inf")
    return array


def _allclose(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.allclose(
        actual,
        expected,
        rtol=_REL_TOL,
        atol=_ABS_TOL,
        equal_nan=False,
    ):
        difference = float(np.max(np.abs(actual - expected)))
        raise TimeLawArtifactError(
            f"{label} violates the exact artifact algebra "
            f"(max abs residual {difference:.9g})"
        )


def _array_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(actual, expected):
        raise TimeLawArtifactError(f"{label} must match exactly")


def _cell_for_time(
    time_s: float,
    time_node_s: np.ndarray,
    s_node: np.ndarray,
    x_node: np.ndarray,
    u_cell: np.ndarray,
) -> tuple[int, int, float, float]:
    cells = len(u_cell)
    if time_s == float(time_node_s[-1]):
        return (
            cells - 1,
            CELL_SIDE_LEFT,
            float(s_node[-1]),
            float(x_node[-1]),
        )

    insertion = int(np.searchsorted(time_node_s, time_s, side="right"))
    cell = insertion - 1
    if cell < 0 or cell >= cells:
        raise TimeLawArtifactError("50 Hz tick time lies outside the node timeline")
    if time_s == float(time_node_s[cell]):
        return (
            cell,
            CELL_SIDE_RIGHT,
            float(s_node[cell]),
            float(x_node[cell]),
        )

    delta_t = float(time_s - time_node_s[cell])
    speed = math.sqrt(float(x_node[cell]))
    expected_s = (
        float(s_node[cell])
        + speed * delta_t
        + 0.5 * float(u_cell[cell]) * delta_t * delta_t
    )
    expected_x = float(x_node[cell]) + 2.0 * float(u_cell[cell]) * (
        expected_s - float(s_node[cell])
    )
    if expected_x < -_ABS_TOL:
        raise TimeLawArtifactError(
            "50 Hz inversion produced negative path speed squared"
        )
    return cell, CELL_SIDE_INTERIOR, expected_s, max(0.0, expected_x)


def _derived_tick_arrays(
    *,
    s_node: np.ndarray,
    x_node: np.ndarray,
    u_cell: np.ndarray,
    time_node_s: np.ndarray,
    tick_count: int,
    fps_hz: float,
) -> dict[str, np.ndarray]:
    duration = float(time_node_s[-1])
    intervals_float = duration * fps_hz
    intervals = int(round(intervals_float))
    if intervals_float != float(intervals):
        raise TimeLawArtifactError(
            "final duration must land on an exact 50 Hz tick before publication"
        )
    if tick_count != intervals + 1:
        raise TimeLawArtifactError(
            f"tick arrays need {intervals + 1} rows for duration {duration:.9g}s, "
            f"got {tick_count}"
        )

    tick_index = np.arange(tick_count, dtype=_INDEX_DTYPE)
    tick_time = np.ascontiguousarray(tick_index.astype(_FLOAT_DTYPE) / fps_hz)
    if float(tick_time[-1]) != duration:
        raise TimeLawArtifactError(
            "final duration must equal the final 50 Hz tick exactly"
        )
    tick_s = np.empty(tick_count, dtype=_FLOAT_DTYPE)
    tick_x = np.empty(tick_count, dtype=_FLOAT_DTYPE)
    tick_cell = np.empty(tick_count, dtype=_INDEX_DTYPE)
    tick_side = np.empty(tick_count, dtype=_SIDE_DTYPE)
    for index, time_s in enumerate(tick_time):
        cell, side, path_s, speed_sq = _cell_for_time(
            float(time_s), time_node_s, s_node, x_node, u_cell
        )
        tick_cell[index] = cell
        tick_side[index] = side
        tick_s[index] = path_s
        tick_x[index] = speed_sq
    return {
        "tick_index": tick_index,
        "tick_time_s": tick_time,
        "tick_x": tick_x,
        "tick_cell_index": tick_cell,
        "tick_cell_side": tick_side,
        "_expected_tick_s": tick_s,
    }


def _reject_nominal_tick_side_ambiguity(time_node_s: np.ndarray, fps_hz: float) -> None:
    nearest_tick = np.rint(time_node_s * fps_hz)
    exact_tick_time = nearest_tick / fps_hz
    nominally_equal = np.isclose(
        time_node_s,
        exact_tick_time,
        rtol=_REL_TOL,
        atol=_ABS_TOL,
    )
    not_exact = time_node_s != exact_tick_time
    ambiguous = np.flatnonzero(nominally_equal & not_exact)
    if len(ambiguous):
        raise TimeLawArtifactError(
            "time_node_s lies nominally on a 50 Hz tick but is not exactly "
            f"equal; cell-side semantics would be ambiguous at nodes "
            f"{ambiguous.tolist()}"
        )


def _validate_scalar_time_law_inputs(
    values: Mapping[str, np.ndarray], fps_hz: float
) -> None:
    """Fail before tick derivation when the caller's scalar time law is invalid."""
    s_node = values["s_node"]
    nodes = len(s_node)
    if nodes < 3:
        raise TimeLawArtifactError("s_node must contain at least three nodes")
    cells = nodes - 1
    expected_shapes = {
        "s_mid": (cells,),
        "x_node": (nodes,),
        "x_mid": (cells,),
        "u_cell": (cells,),
        "time_node_s": (nodes,),
        "time_mid_s": (cells,),
    }
    for key, expected in expected_shapes.items():
        if values[key].shape != expected:
            raise TimeLawArtifactError(
                f"{key} must have shape {expected}, got {values[key].shape}"
            )
    if np.any(np.diff(s_node) <= 0.0):
        raise TimeLawArtifactError("s_node must be strictly increasing")

    s_mid_expected = 0.5 * (s_node[:-1] + s_node[1:])
    _allclose(values["s_mid"], s_mid_expected, "s_mid")

    x_node = values["x_node"]
    if np.any(x_node < 0.0):
        raise TimeLawArtifactError("x_node must be non-negative")
    if float(x_node[0]) != 0.0 or float(x_node[-1]) != 0.0:
        raise TimeLawArtifactError("start/end x_node must be exactly zero")
    x_mid_expected = 0.5 * (x_node[:-1] + x_node[1:])
    _allclose(values["x_mid"], x_mid_expected, "x_mid")

    ds = np.diff(s_node)
    u_expected = np.diff(x_node) / (2.0 * ds)
    _allclose(values["u_cell"], u_expected, "u_cell=delta_x/(2*delta_s)")

    time_node = values["time_node_s"]
    if float(time_node[0]) != 0.0 or np.any(np.diff(time_node) <= 0.0):
        raise TimeLawArtifactError(
            "time_node_s must start at exact zero and be strictly increasing"
        )
    speed_node = np.sqrt(x_node)
    denominator = speed_node[:-1] + speed_node[1:]
    if np.any(denominator <= 0.0):
        raise TimeLawArtifactError(
            "positive path width is trapped between zero-speed nodes"
        )
    dt_expected = 2.0 * ds / denominator
    time_expected = np.concatenate(
        (np.zeros(1, dtype=_FLOAT_DTYPE), np.cumsum(dt_expected))
    )
    _allclose(time_node, time_expected, "time_node_s")
    _reject_nominal_tick_side_ambiguity(time_node, fps_hz)

    speed_mid = np.sqrt(x_mid_expected)
    mid_denominator = speed_node[:-1] + speed_mid
    if np.any(mid_denominator <= 0.0):
        raise TimeLawArtifactError("midpoint timeline has a zero denominator")
    time_mid_expected = time_node[:-1] + ds / mid_denominator
    _allclose(values["time_mid_s"], time_mid_expected, "time_mid_s")
    if np.any(values["time_mid_s"] <= time_node[:-1]) or np.any(
        values["time_mid_s"] >= time_node[1:]
    ):
        raise TimeLawArtifactError("time_mid_s must lie strictly inside each cell")


def _materialize_trace(trace: TimeLawTrace, fps_hz: float) -> dict[str, np.ndarray]:
    if not isinstance(trace, TimeLawTrace):
        raise TimeLawArtifactError("trace must be a TimeLawTrace")
    values = {
        "s_node": _float_array(trace.s_node, "s_node", ndim=1),
        "s_mid": _float_array(trace.s_mid, "s_mid", ndim=1),
        "qpos_node": _float_array(trace.qpos_node, "qpos_node", ndim=2),
        "q_s_node": _float_array(trace.q_s_node, "q_s_node", ndim=2),
        "q_ss_node_left": _float_array(
            trace.q_ss_node_left, "q_ss_node_left", ndim=2
        ),
        "q_ss_node_right": _float_array(
            trace.q_ss_node_right, "q_ss_node_right", ndim=2
        ),
        "qpos_mid": _float_array(trace.qpos_mid, "qpos_mid", ndim=2),
        "q_s_mid": _float_array(trace.q_s_mid, "q_s_mid", ndim=2),
        "q_ss_mid": _float_array(trace.q_ss_mid, "q_ss_mid", ndim=2),
        "x_node": _float_array(trace.x_node, "x_node", ndim=1),
        "x_mid": _float_array(trace.x_mid, "x_mid", ndim=1),
        "u_cell": _float_array(trace.u_cell, "u_cell", ndim=1),
        "time_node_s": _float_array(trace.time_node_s, "time_node_s", ndim=1),
        "time_mid_s": _float_array(trace.time_mid_s, "time_mid_s", ndim=1),
        "collocation_qpos": _float_array(
            trace.collocation_qpos, "collocation_qpos", ndim=3
        ),
        "collocation_q_s": _float_array(
            trace.collocation_q_s, "collocation_q_s", ndim=3
        ),
        "collocation_qvel": _float_array(
            trace.collocation_qvel, "collocation_qvel", ndim=3
        ),
        "collocation_qacc": _float_array(
            trace.collocation_qacc, "collocation_qacc", ndim=3
        ),
        "tick_s": _float_array(trace.tick_s, "tick_s", ndim=1),
        "tick_qpos": _float_array(trace.tick_qpos, "tick_qpos", ndim=2),
        "tick_q_s": _float_array(trace.tick_q_s, "tick_q_s", ndim=2),
        "tick_q_ss": _float_array(trace.tick_q_ss, "tick_q_ss", ndim=2),
        "tick_qvel": _float_array(trace.tick_qvel, "tick_qvel", ndim=2),
        "tick_qacc": _float_array(trace.tick_qacc, "tick_qacc", ndim=2),
    }
    _validate_scalar_time_law_inputs(values, fps_hz)
    derived = _derived_tick_arrays(
        s_node=values["s_node"],
        x_node=values["x_node"],
        u_cell=values["u_cell"],
        time_node_s=values["time_node_s"],
        tick_count=len(values["tick_s"]),
        fps_hz=fps_hz,
    )
    expected_tick_s = derived.pop("_expected_tick_s")
    _array_equal(values["tick_s"], expected_tick_s, "tick_s")
    values.update(derived)
    return {key: values[key] for key in ARRAY_KEYS}


def _time_at_path_position(
    path_s: float,
    *,
    s_node: np.ndarray,
    x_node: np.ndarray,
    u_cell: np.ndarray,
    time_node_s: np.ndarray,
) -> float:
    if path_s < float(s_node[0]) or path_s > float(s_node[-1]):
        raise TimeLawArtifactError("marker path position leaves the solved path")
    node_matches = np.flatnonzero(s_node == path_s)
    if len(node_matches):
        return float(time_node_s[int(node_matches[0])])
    cell = int(np.searchsorted(s_node, path_s, side="right") - 1)
    if cell < 0 or cell >= len(u_cell):
        raise TimeLawArtifactError("marker path position has no containing cell")
    delta_s = float(path_s - s_node[cell])
    speed_sq = float(x_node[cell]) + 2.0 * float(u_cell[cell]) * delta_s
    if speed_sq < -_ABS_TOL:
        raise TimeLawArtifactError("marker interpolation produced negative x")
    speed = math.sqrt(max(0.0, speed_sq))
    denominator = math.sqrt(float(x_node[cell])) + speed
    if denominator <= 0.0:
        raise TimeLawArtifactError("marker lies in a zero-speed positive-width cell")
    return float(time_node_s[cell]) + 2.0 * delta_s / denominator


def _tick_at_or_after(time_s: float, fps_hz: float) -> int:
    scaled = float(time_s) * float(fps_hz)
    return int(math.ceil(scaled))


def _marker_and_window_receipts(
    marker_path_s: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    fps_hz: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(marker_path_s, Mapping):
        raise TimeLawArtifactError("marker_path_s must be a mapping")
    if frozenset(marker_path_s) != frozenset(MARKER_NAMES):
        raise TimeLawArtifactError(
            f"marker_path_s keys must be exactly {list(MARKER_NAMES)}"
        )
    path_positions = {
        name: _finite_float(marker_path_s[name], f"marker_path_s[{name!r}]")
        for name in MARKER_NAMES
    }
    path_start = float(arrays["s_node"][0])
    path_end = float(arrays["s_node"][-1])
    outside = [
        name
        for name, path_s in path_positions.items()
        if path_s < path_start or path_s > path_end
    ]
    if outside:
        raise TimeLawArtifactError(
            "marker path positions leave the solved path: "
            f"{sorted(outside)}"
        )
    window_start_s = path_positions["window_start"]
    window_end_s = path_positions["window_end"]
    if window_start_s > window_end_s:
        raise TimeLawArtifactError(
            "protected window endpoints must satisfy "
            "window_start <= window_end"
        )

    tick_count = len(arrays["tick_index"])
    markers: dict[str, Any] = {}
    for name in MARKER_NAMES:
        path_s = path_positions[name]
        time_s = _time_at_path_position(
            path_s,
            s_node=arrays["s_node"],
            x_node=arrays["x_node"],
            u_cell=arrays["u_cell"],
            time_node_s=arrays["time_node_s"],
        )
        fractional = time_s * fps_hz
        at_or_after = _tick_at_or_after(time_s, fps_hz)
        nearest = int(np.clip(np.rint(fractional), 0, tick_count - 1))
        if at_or_after < 0 or at_or_after >= tick_count:
            raise TimeLawArtifactError(f"marker {name} leaves the 50 Hz timeline")
        markers[name] = {
            "path_s": path_s,
            "path_s_hex": float(path_s).hex(),
            "time_s": time_s,
            "time_s_hex": float(time_s).hex(),
            "fractional_tick": fractional,
            "at_or_after_tick": at_or_after,
            "nearest_observation_tick": nearest,
        }

    times = [
        markers[name]["time_s"]
        for name in ("window_start", "window_end")
    ]
    after_ticks = [
        markers[name]["at_or_after_tick"]
        for name in ("window_start", "window_end")
    ]
    nearest_ticks = [
        markers[name]["nearest_observation_tick"]
        for name in ("window_start", "window_end")
    ]
    if not (
        times[0] <= times[1]
        and after_ticks[0] <= after_ticks[1]
        and nearest_ticks[0] <= nearest_ticks[1]
    ):
        raise TimeLawArtifactError(
            "protected window time/tick mappings are not monotonic"
        )

    # A cell is constrained iff it owns any positive-width left-time interval
    # before exact window_end.  A cell beginning exactly at window_end may
    # brake; an exact-node tick belongs to that right-side cell.
    overlap = np.flatnonzero(arrays["s_node"][:-1] < window_end_s)
    if len(overlap) == 0:
        raise TimeLawArtifactError("no cell overlaps the no-early-brake interval")
    negative = overlap[arrays["u_cell"][overlap] < 0.0]
    if len(negative):
        raise TimeLawArtifactError(
            "u_cell must remain non-negative for every positive-width cell "
            f"overlapping path start through window_end; bad cells={negative.tolist()}"
        )

    start_fractional = float(markers["window_start"]["fractional_tick"])
    end_fractional = float(markers["window_end"]["fractional_tick"])
    inclusive_start = int(math.ceil(start_fractional))
    inclusive_end = int(math.floor(end_fractional))
    if inclusive_start > inclusive_end:
        raise TimeLawArtifactError(
            "contact window contains no inclusive 50 Hz observation tick"
        )
    if inclusive_start < 0 or inclusive_end >= tick_count:
        raise TimeLawArtifactError("contact window tick interval leaves the artifact")
    window = {
        "window_start_s": window_start_s,
        "window_end_s": window_end_s,
        "no_early_brake_from_path_start_through_window_end": True,
        "positive_overlap_cell_indices": overlap.astype(int).tolist(),
        "inclusive_tick_start": inclusive_start,
        "inclusive_tick_end": inclusive_end,
        "inclusive_tick_policy": "ceil_start_floor_end",
    }
    return markers, window


def _validate_array_contract(arrays: Mapping[str, np.ndarray]) -> None:
    if not isinstance(arrays, Mapping):
        raise TimeLawArtifactError("arrays must be a mapping")
    if tuple(arrays) != ARRAY_KEYS:
        raise TimeLawArtifactError(
            f"artifact array order/keyset changed: {tuple(arrays)}"
        )
    for key in ARRAY_KEYS:
        value = np.asarray(arrays[key])
        expected_dtype = _ARRAY_DTYPES[key]
        if value.dtype != expected_dtype:
            raise TimeLawArtifactError(
                f"{key} dtype must be {expected_dtype.str}, got {value.dtype.str}"
            )
        if not value.flags.c_contiguous:
            raise TimeLawArtifactError(f"{key} must be C-contiguous")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise TimeLawArtifactError(f"{key} contains NaN/Inf")


def _validate_materialized_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    marker_path_s: Mapping[str, Any],
    fps_hz: float,
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    _validate_array_contract(arrays)
    s_node = arrays["s_node"]
    nodes = len(s_node)
    if s_node.ndim != 1 or nodes < 3:
        raise TimeLawArtifactError("s_node must contain at least three nodes")
    cells = nodes - 1
    if np.any(np.diff(s_node) <= 0.0):
        raise TimeLawArtifactError("s_node must be strictly increasing")

    qpos_node = arrays["qpos_node"]
    q_s_node = arrays["q_s_node"]
    if qpos_node.ndim != 2 or qpos_node.shape[0] != nodes or qpos_node.shape[1] < 1:
        raise TimeLawArtifactError("qpos_node must have shape (N,nq>=1)")
    if q_s_node.ndim != 2 or q_s_node.shape[0] != nodes or q_s_node.shape[1] < 1:
        raise TimeLawArtifactError("q_s_node must have shape (N,nv>=1)")
    nq = qpos_node.shape[1]
    nv = q_s_node.shape[1]
    ticks = len(arrays["tick_index"])
    if ticks < 2:
        raise TimeLawArtifactError("artifact must contain at least two 50 Hz ticks")

    expected_shapes = {
        "s_node": (nodes,),
        "s_mid": (cells,),
        "qpos_node": (nodes, nq),
        "q_s_node": (nodes, nv),
        "q_ss_node_left": (nodes, nv),
        "q_ss_node_right": (nodes, nv),
        "qpos_mid": (cells, nq),
        "q_s_mid": (cells, nv),
        "q_ss_mid": (cells, nv),
        "x_node": (nodes,),
        "x_mid": (cells,),
        "u_cell": (cells,),
        "time_node_s": (nodes,),
        "time_mid_s": (cells,),
        "collocation_qpos": (cells, 3, nq),
        "collocation_q_s": (cells, 3, nv),
        "collocation_qvel": (cells, 3, nv),
        "collocation_qacc": (cells, 3, nv),
        "tick_index": (ticks,),
        "tick_time_s": (ticks,),
        "tick_s": (ticks,),
        "tick_x": (ticks,),
        "tick_cell_index": (ticks,),
        "tick_cell_side": (ticks,),
        "tick_qpos": (ticks, nq),
        "tick_q_s": (ticks, nv),
        "tick_q_ss": (ticks, nv),
        "tick_qvel": (ticks, nv),
        "tick_qacc": (ticks, nv),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise TimeLawArtifactError(
                f"{key} must have shape {expected}, got {arrays[key].shape}"
            )

    s_mid_expected = 0.5 * (s_node[:-1] + s_node[1:])
    _allclose(arrays["s_mid"], s_mid_expected, "s_mid")
    _array_equal(arrays["collocation_qpos"][:, 0], qpos_node[:-1], "left qpos")
    _array_equal(arrays["collocation_qpos"][:, 1], arrays["qpos_mid"], "midpoint qpos")
    _array_equal(arrays["collocation_qpos"][:, 2], qpos_node[1:], "right qpos")
    _array_equal(arrays["collocation_q_s"][:, 0], q_s_node[:-1], "left q_s")
    _array_equal(arrays["collocation_q_s"][:, 1], arrays["q_s_mid"], "midpoint q_s")
    _array_equal(arrays["collocation_q_s"][:, 2], q_s_node[1:], "right q_s")

    x_node = arrays["x_node"]
    if np.any(x_node < 0.0):
        raise TimeLawArtifactError("x_node must be non-negative")
    if float(x_node[0]) != 0.0 or float(x_node[-1]) != 0.0:
        raise TimeLawArtifactError("start/end x_node must be exactly zero")
    x_mid_expected = 0.5 * (x_node[:-1] + x_node[1:])
    _allclose(arrays["x_mid"], x_mid_expected, "x_mid")
    ds = np.diff(s_node)
    u_expected = np.diff(x_node) / (2.0 * ds)
    _allclose(arrays["u_cell"], u_expected, "u_cell=delta_x/(2*delta_s)")

    time_node = arrays["time_node_s"]
    if float(time_node[0]) != 0.0 or np.any(np.diff(time_node) <= 0.0):
        raise TimeLawArtifactError(
            "time_node_s must start at exact zero and be strictly increasing"
        )
    speed_node = np.sqrt(x_node)
    denominator = speed_node[:-1] + speed_node[1:]
    if np.any(denominator <= 0.0):
        raise TimeLawArtifactError(
            "positive path width is trapped between zero-speed nodes"
        )
    dt_expected = 2.0 * ds / denominator
    time_expected = np.concatenate(
        (np.zeros(1, dtype=_FLOAT_DTYPE), np.cumsum(dt_expected))
    )
    _allclose(time_node, time_expected, "time_node_s")
    _reject_nominal_tick_side_ambiguity(time_node, fps_hz)

    speed_mid = np.sqrt(x_mid_expected)
    mid_denominator = speed_node[:-1] + speed_mid
    if np.any(mid_denominator <= 0.0):
        raise TimeLawArtifactError("midpoint timeline has a zero denominator")
    time_mid_expected = time_node[:-1] + ds / mid_denominator
    _allclose(arrays["time_mid_s"], time_mid_expected, "time_mid_s")
    if np.any(arrays["time_mid_s"] <= time_node[:-1]) or np.any(
        arrays["time_mid_s"] >= time_node[1:]
    ):
        raise TimeLawArtifactError("time_mid_s must lie strictly inside each cell")

    collocation_x = np.stack((x_node[:-1], x_mid_expected, x_node[1:]), axis=1)
    q_ss_lmr = np.stack(
        (
            arrays["q_ss_node_right"][:-1],
            arrays["q_ss_mid"],
            arrays["q_ss_node_left"][1:],
        ),
        axis=1,
    )
    qvel_expected = arrays["collocation_q_s"] * np.sqrt(collocation_x)[:, :, None]
    qacc_expected = (
        arrays["collocation_q_s"] * arrays["u_cell"][:, None, None]
        + q_ss_lmr * collocation_x[:, :, None]
    )
    _allclose(arrays["collocation_qvel"], qvel_expected, "collocation_qvel")
    _allclose(arrays["collocation_qacc"], qacc_expected, "collocation_qacc")
    if (
        np.count_nonzero(arrays["collocation_qvel"][0, 0]) != 0
        or np.count_nonzero(arrays["collocation_qvel"][-1, 2]) != 0
    ):
        raise TimeLawArtifactError(
            "left start and right end collocation qvel must be exactly zero"
        )

    derived = _derived_tick_arrays(
        s_node=s_node,
        x_node=x_node,
        u_cell=arrays["u_cell"],
        time_node_s=time_node,
        tick_count=ticks,
        fps_hz=fps_hz,
    )
    expected_tick_s = derived.pop("_expected_tick_s")
    for key, expected in derived.items():
        if arrays[key].dtype.kind in "iu":
            _array_equal(arrays[key], expected, key)
        else:
            _allclose(arrays[key], expected, key)
    _array_equal(arrays["tick_s"], expected_tick_s, "tick_s")
    if np.any(np.diff(arrays["tick_s"]) < -_ABS_TOL):
        raise TimeLawArtifactError("tick_s must be non-decreasing")
    if not math.isclose(
        float(arrays["tick_s"][0]),
        float(s_node[0]),
        rel_tol=_REL_TOL,
        abs_tol=_ABS_TOL,
    ) or not math.isclose(
        float(arrays["tick_s"][-1]),
        float(s_node[-1]),
        rel_tol=_REL_TOL,
        abs_tol=_ABS_TOL,
    ):
        raise TimeLawArtifactError("50 Hz ticks must cover both path endpoints")

    tick_u = arrays["u_cell"][arrays["tick_cell_index"]]
    tick_qvel_expected = arrays["tick_q_s"] * np.sqrt(arrays["tick_x"])[:, None]
    tick_qacc_expected = (
        arrays["tick_q_s"] * tick_u[:, None]
        + arrays["tick_q_ss"] * arrays["tick_x"][:, None]
    )
    _allclose(arrays["tick_qvel"], tick_qvel_expected, "tick_qvel")
    _allclose(arrays["tick_qacc"], tick_qacc_expected, "tick_qacc")
    if np.count_nonzero(arrays["tick_qvel"][[0, -1]]) != 0:
        raise TimeLawArtifactError("50 Hz endpoint qvel must be exactly zero")

    _allclose(arrays["tick_qpos"][0], qpos_node[0], "tick start qpos")
    _allclose(arrays["tick_qpos"][-1], qpos_node[-1], "tick end qpos")
    _allclose(arrays["tick_q_s"][0], q_s_node[0], "tick start q_s")
    _allclose(arrays["tick_q_s"][-1], q_s_node[-1], "tick end q_s")
    _allclose(
        arrays["tick_q_ss"][0],
        arrays["q_ss_node_right"][0],
        "tick start right-limit q_ss",
    )
    _allclose(
        arrays["tick_q_ss"][-1],
        arrays["q_ss_node_left"][-1],
        "tick end left-limit q_ss",
    )

    for tick in range(ticks):
        cell = int(arrays["tick_cell_index"][tick])
        side = int(arrays["tick_cell_side"][tick])
        if side == CELL_SIDE_RIGHT:
            node = cell
        elif side == CELL_SIDE_LEFT:
            node = cell + 1
        else:
            node = -1
        if node >= 0:
            _allclose(
                arrays["tick_qpos"][tick],
                qpos_node[node],
                f"tick {tick} exact-node qpos",
            )
            _allclose(
                arrays["tick_q_s"][tick],
                q_s_node[node],
                f"tick {tick} exact-node q_s",
            )
            _allclose(
                arrays["tick_q_ss"][tick],
                (
                    arrays["q_ss_node_right"][node]
                    if side == CELL_SIDE_RIGHT
                    else arrays["q_ss_node_left"][node]
                ),
                f"tick {tick} exact-node one-sided q_ss",
            )
        midpoint_matches = np.flatnonzero(arrays["s_mid"] == arrays["tick_s"][tick])
        if len(midpoint_matches):
            midpoint = int(midpoint_matches[0])
            _allclose(
                arrays["tick_qpos"][tick],
                arrays["qpos_mid"][midpoint],
                f"tick {tick} midpoint qpos",
            )
            _allclose(
                arrays["tick_q_s"][tick],
                arrays["q_s_mid"][midpoint],
                f"tick {tick} midpoint q_s",
            )
            _allclose(
                arrays["tick_q_ss"][tick],
                arrays["q_ss_mid"][midpoint],
                f"tick {tick} midpoint q_ss",
            )

    markers, window = _marker_and_window_receipts(marker_path_s, arrays, fps_hz)
    dimensions = {
        "nodes": nodes,
        "cells": cells,
        "nq": nq,
        "nv": nv,
        "ticks": ticks,
    }
    return dimensions, markers, window


def _array_content_sha256(key: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"canonical-time-law-array-v1\0")
    digest.update(key.encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _path_evaluation_array_sha256_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"canonical-path-evaluation-arrays-v1\0")
    for key in PATH_EVALUATION_ARRAY_KEYS:
        digest.update(key.encode("ascii") + b"\0")
        digest.update(_array_content_sha256(key, arrays[key]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def path_evaluation_array_sha256(trace: TimeLawTrace) -> str:
    """Digest the exact producer q/q_s/q_ss evidence expected by the builder.

    This helper only canonicalizes and hashes caller-supplied evaluator output.
    It neither estimates derivatives nor asserts that the evaluator is correct.
    """

    if not isinstance(trace, TimeLawTrace):
        raise TimeLawArtifactError("trace must be a TimeLawTrace")
    arrays: dict[str, np.ndarray] = {}
    for key in PATH_EVALUATION_ARRAY_KEYS:
        value = getattr(trace, key)
        ndim = (
            3
            if key.startswith("collocation_")
            else (1 if key in ("s_node", "s_mid") else 2)
        )
        arrays[key] = _float_array(value, key, ndim=ndim)
    return _path_evaluation_array_sha256_from_arrays(arrays)


def path_evaluation_receipt_sha256(
    *,
    source_sha256: str,
    evaluator_id: str,
    evaluator_version: str,
    derivative_method: str,
    evaluator_contract_sha256: str,
    evaluator_implementation_sha256: str,
    evaluated_arrays_sha256: str,
) -> str:
    """Canonical producer-receipt digest bound to source, evaluator, and arrays."""

    checked_method = _strict_string(derivative_method, "receipt derivative_method")
    if checked_method not in PATH_DERIVATIVE_METHODS:
        raise TimeLawArtifactError(
            "receipt derivative_method must be one of " f"{PATH_DERIVATIVE_METHODS}"
        )
    payload = {
        "source_sha256": _digest(source_sha256, "receipt source_sha256"),
        "evaluator_id": _strict_string(evaluator_id, "receipt evaluator_id"),
        "evaluator_version": _strict_string(
            evaluator_version, "receipt evaluator_version"
        ),
        "derivative_method": checked_method,
        "evaluator_contract_sha256": _digest(
            evaluator_contract_sha256,
            "receipt evaluator_contract_sha256",
        ),
        "evaluator_implementation_sha256": _digest(
            evaluator_implementation_sha256,
            "receipt evaluator_implementation_sha256",
        ),
        "evaluated_arrays_sha256": _digest(
            evaluated_arrays_sha256,
            "receipt evaluated_arrays_sha256",
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(
        b"canonical-path-evaluation-producer-receipt-v1\0" + encoded
    ).hexdigest()


def _array_receipts(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {
            "dtype": arrays[key].dtype.str,
            "shape": list(arrays[key].shape),
            "sha256": _array_content_sha256(key, arrays[key]),
        }
        for key in ARRAY_KEYS
    }


def _array_set_sha256(receipts: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        receipts, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"canonical-time-law-array-set-v1\0" + canonical).hexdigest()


def _bundle_sha256(npz_bytes: bytes, manifest_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"canonical-time-law-bundle-v1\0")
    digest.update(len(npz_bytes).to_bytes(8, "big"))
    digest.update(npz_bytes)
    digest.update(len(manifest_bytes).to_bytes(8, "big"))
    digest.update(manifest_bytes)
    return digest.hexdigest()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in ARRAY_KEYS:
            array_stream = io.BytesIO()
            np.lib.format.write_array(array_stream, arrays[key], allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_stream.getvalue())
    return stream.getvalue()


def _bindings_payload(bindings: ArtifactBindings) -> dict[str, Any]:
    if not isinstance(bindings, ArtifactBindings):
        raise TimeLawArtifactError("bindings must be ArtifactBindings")
    return {
        "recipe_sha256": bindings.recipe_sha256,
        "source_sha256": bindings.source_sha256,
        "ready_sha256": bindings.ready_sha256,
        "mjcf_sha256": bindings.mjcf_sha256,
        "urdf_sha256": bindings.urdf_sha256,
        "model_binding_sha256": bindings.model_binding_sha256,
        "actuator_contract_sha256": bindings.actuator_contract_sha256,
        "tools_sha256": dict(bindings.tools_sha256),
        "solver": {
            "solver_id": bindings.solver.solver_id,
            "solver_version": bindings.solver.solver_version,
            "solver_contract_sha256": bindings.solver.solver_contract_sha256,
            "solver_implementation_sha256": (
                bindings.solver.solver_implementation_sha256
            ),
        },
        "path_evaluator": {
            "evaluator_id": bindings.path_evaluator.evaluator_id,
            "evaluator_version": bindings.path_evaluator.evaluator_version,
            "derivative_method": bindings.path_evaluator.derivative_method,
            "evaluator_contract_sha256": (
                bindings.path_evaluator.evaluator_contract_sha256
            ),
            "evaluator_implementation_sha256": (
                bindings.path_evaluator.evaluator_implementation_sha256
            ),
            "evaluated_arrays_sha256": (
                bindings.path_evaluator.evaluated_arrays_sha256
            ),
            "producer_receipt_sha256": (
                bindings.path_evaluator.producer_receipt_sha256
            ),
        },
        "weighted_arc_length": (
            None
            if bindings.weighted_arc_length is None
            else {
                **bindings.weighted_arc_length.receipt_payload(),
                "producer_receipt_sha256": (
                    bindings.weighted_arc_length.producer_receipt_sha256
                ),
            }
        ),
    }


def _validate_bindings_payload(raw: Any) -> dict[str, Any]:
    value = _exact_keys(raw, _BINDING_KEYS, "bindings")
    payload = {
        key: _digest(value[key], f"bindings.{key}")
        for key in (
            "recipe_sha256",
            "source_sha256",
            "ready_sha256",
            "mjcf_sha256",
            "urdf_sha256",
            "model_binding_sha256",
            "actuator_contract_sha256",
        )
    }
    tools_raw = value["tools_sha256"]
    if not isinstance(tools_raw, Mapping) or not tools_raw:
        raise TimeLawArtifactError("bindings.tools_sha256 must be non-empty")
    tools: dict[str, str] = {}
    for raw_name, raw_digest in tools_raw.items():
        name = _identifier(raw_name, "bindings tool name")
        tools[name] = _digest(raw_digest, f"bindings.tools_sha256[{name!r}]")
    payload["tools_sha256"] = dict(sorted(tools.items()))

    solver_raw = _exact_keys(value["solver"], _SOLVER_KEYS, "bindings.solver")
    payload["solver"] = {
        "solver_id": _strict_string(
            solver_raw["solver_id"], "bindings.solver.solver_id"
        ),
        "solver_version": _strict_string(
            solver_raw["solver_version"], "bindings.solver.solver_version"
        ),
        "solver_contract_sha256": _digest(
            solver_raw["solver_contract_sha256"],
            "bindings.solver.solver_contract_sha256",
        ),
        "solver_implementation_sha256": _digest(
            solver_raw["solver_implementation_sha256"],
            "bindings.solver.solver_implementation_sha256",
        ),
    }
    evaluator_raw = _exact_keys(
        value["path_evaluator"],
        _PATH_EVALUATOR_KEYS,
        "bindings.path_evaluator",
    )
    method = _strict_string(
        evaluator_raw["derivative_method"],
        "bindings.path_evaluator.derivative_method",
    )
    if method not in PATH_DERIVATIVE_METHODS:
        raise TimeLawArtifactError(
            "bindings.path_evaluator.derivative_method must be one of "
            f"{PATH_DERIVATIVE_METHODS}"
        )
    payload["path_evaluator"] = {
        "evaluator_id": _strict_string(
            evaluator_raw["evaluator_id"],
            "bindings.path_evaluator.evaluator_id",
        ),
        "evaluator_version": _strict_string(
            evaluator_raw["evaluator_version"],
            "bindings.path_evaluator.evaluator_version",
        ),
        "derivative_method": method,
        "evaluator_contract_sha256": _digest(
            evaluator_raw["evaluator_contract_sha256"],
            "bindings.path_evaluator.evaluator_contract_sha256",
        ),
        "evaluator_implementation_sha256": _digest(
            evaluator_raw["evaluator_implementation_sha256"],
            "bindings.path_evaluator.evaluator_implementation_sha256",
        ),
        "evaluated_arrays_sha256": _digest(
            evaluator_raw["evaluated_arrays_sha256"],
            "bindings.path_evaluator.evaluated_arrays_sha256",
        ),
        "producer_receipt_sha256": _digest(
            evaluator_raw["producer_receipt_sha256"],
            "bindings.path_evaluator.producer_receipt_sha256",
        ),
    }
    evaluator = payload["path_evaluator"]
    expected_receipt = path_evaluation_receipt_sha256(
        source_sha256=payload["source_sha256"],
        evaluator_id=evaluator["evaluator_id"],
        evaluator_version=evaluator["evaluator_version"],
        derivative_method=evaluator["derivative_method"],
        evaluator_contract_sha256=evaluator["evaluator_contract_sha256"],
        evaluator_implementation_sha256=(evaluator["evaluator_implementation_sha256"]),
        evaluated_arrays_sha256=evaluator["evaluated_arrays_sha256"],
    )
    if evaluator["producer_receipt_sha256"] != expected_receipt:
        raise TimeLawArtifactError(
            "bindings.path_evaluator producer receipt does not bind source, "
            "evaluator, and evaluated arrays"
        )
    weighted_raw = value["weighted_arc_length"]
    if weighted_raw is None:
        payload["weighted_arc_length"] = None
    else:
        checked_weighted_raw = _exact_keys(
            weighted_raw,
            _WEIGHTED_ARC_KEYS,
            "bindings.weighted_arc_length",
        )
        try:
            weighted = WeightedArcLengthBinding(**checked_weighted_raw)
        except TypeError as exc:
            raise TimeLawArtifactError(
                "bindings.weighted_arc_length fields are malformed"
            ) from exc
        expected_weighted_receipt = weighted_arc_length_receipt_sha256(
            source_sha256=payload["source_sha256"],
            binding=weighted,
        )
        if (
            weighted.producer_receipt_sha256
            != expected_weighted_receipt
        ):
            raise TimeLawArtifactError(
                "bindings.weighted_arc_length producer receipt does not bind "
                "source, digest/tolerances/regularity, and evaluated arrays"
            )
        payload["weighted_arc_length"] = {
            **weighted.receipt_payload(),
            "producer_receipt_sha256": (
                weighted.producer_receipt_sha256
            ),
        }
    return payload


def _manifest(
    *,
    motion_id: str,
    scope: str,
    dimensions: Mapping[str, int],
    arrays: Mapping[str, np.ndarray],
    npz_bytes: bytes,
    bindings: Mapping[str, Any],
    markers: Mapping[str, Any],
    window: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = _array_receipts(arrays)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "publication_class": PUBLICATION_CLASS,
        "build_verdict": BUILD_VERDICT,
        "motion_id": motion_id,
        "scope": scope,
        "fps_hz": FPS_HZ,
        "training_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
        "dimensions": dict(dimensions),
        "hashes": {
            "artifact_npz_sha256": hashlib.sha256(npz_bytes).hexdigest(),
            "array_set_sha256": _array_set_sha256(receipts),
            "arrays": receipts,
        },
        "bindings": dict(bindings),
        "markers": dict(markers),
        "window": dict(window),
        "semantics": dict(_SEMANTICS),
        "claims": dict(_CLAIMS),
    }


def _marker_path_s_from_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    markers = manifest.get("markers")
    if not isinstance(markers, Mapping) or frozenset(markers) != frozenset(
        MARKER_NAMES
    ):
        raise TimeLawArtifactError(
            f"manifest markers must be exactly {list(MARKER_NAMES)}"
        )
    result: dict[str, float] = {}
    for name in MARKER_NAMES:
        row = _exact_keys(markers[name], _MARKER_ROW_KEYS, f"markers.{name}")
        result[name] = _finite_float(row["path_s"], f"markers.{name}.path_s")
    return result


def _validate_manifest(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    npz_bytes: bytes,
) -> None:
    value = _exact_keys(manifest, _TOP_LEVEL_KEYS, "manifest")
    motion_id = _identifier(value["motion_id"], "manifest.motion_id")
    scope = _strict_string(value["scope"], "manifest.scope")
    if scope not in SCOPES:
        raise TimeLawArtifactError(f"manifest.scope must be one of {SCOPES}")
    schema_version = _integer(
        value["schema_version"], "manifest.schema_version", minimum=1
    )
    fps_hz = _finite_float(value["fps_hz"], "manifest.fps_hz")
    if (
        schema_version != ARTIFACT_SCHEMA_VERSION
        or value["artifact_type"] != ARTIFACT_TYPE
        or value["publication_class"] != PUBLICATION_CLASS
        or value["build_verdict"] != BUILD_VERDICT
        or fps_hz != FPS_HZ
    ):
        raise TimeLawArtifactError(
            "manifest identity/version/publication contract changed"
        )
    for key in (
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        if value[key] is not False:
            raise TimeLawArtifactError(f"manifest {key} must be exactly false")

    bindings = _validate_bindings_payload(value["bindings"])
    marker_path_s = _marker_path_s_from_manifest(value)
    dimensions, markers, window = _validate_materialized_arrays(
        arrays, marker_path_s=marker_path_s, fps_hz=FPS_HZ
    )
    evaluated_arrays_sha256 = _path_evaluation_array_sha256_from_arrays(arrays)
    if bindings["path_evaluator"]["evaluated_arrays_sha256"] != evaluated_arrays_sha256:
        raise TimeLawArtifactError(
            "producer path-evaluation evidence does not bind the stored "
            "qpos/q_s/q_ss arrays"
        )
    weighted = bindings["weighted_arc_length"]
    if (
        weighted is not None
        and weighted["evaluated_arrays_sha256"]
        != evaluated_arrays_sha256
    ):
        raise TimeLawArtifactError(
            "weighted arc evidence does not bind the stored qpos/q_s/q_ss "
            "arrays"
        )
    expected = _manifest(
        motion_id=motion_id,
        scope=scope,
        dimensions=dimensions,
        arrays=arrays,
        npz_bytes=npz_bytes,
        bindings=bindings,
        markers=markers,
        window=window,
    )
    if not _json_same(dict(value), expected):
        raise TimeLawArtifactError(
            "manifest does not exactly match the validated arrays and bindings"
        )


def _freeze_arrays(
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        value = np.array(arrays[key], copy=True, order="C")
        value.setflags(write=False)
        frozen[key] = value
    return MappingProxyType(frozen)


def build_time_law_artifact(
    *,
    motion_id: str,
    scope: str,
    trace: TimeLawTrace,
    marker_path_s: Mapping[str, Any],
    bindings: ArtifactBindings,
    fps_hz: float = FPS_HZ,
) -> TimeLawArtifact:
    """Validate and build one deterministic sampled-collocation candidate.

    ``trace`` must contain the exact path evaluation and time-law arrays from the
    producer.  This function has no schema-2 input and performs no finite
    differences.
    """

    checked_motion_id = _identifier(motion_id, "motion_id")
    checked_scope = _strict_string(scope, "scope")
    if checked_scope not in SCOPES:
        raise TimeLawArtifactError(f"scope must be one of {SCOPES}")
    checked_fps = _finite_float(fps_hz, "fps_hz")
    if checked_fps != FPS_HZ:
        raise TimeLawArtifactError(f"fps_hz must be exactly {FPS_HZ:g}")
    arrays = _materialize_trace(trace, checked_fps)
    dimensions, markers, window = _validate_materialized_arrays(
        arrays, marker_path_s=marker_path_s, fps_hz=checked_fps
    )
    binding_payload = _bindings_payload(bindings)
    evaluated_arrays_sha256 = _path_evaluation_array_sha256_from_arrays(arrays)
    if (
        binding_payload["path_evaluator"]["evaluated_arrays_sha256"]
        != evaluated_arrays_sha256
    ):
        raise TimeLawArtifactError(
            "producer path-evaluation evidence does not bind the supplied "
            "qpos/q_s/q_ss arrays"
        )
    weighted = binding_payload["weighted_arc_length"]
    if (
        weighted is not None
        and weighted["evaluated_arrays_sha256"]
        != evaluated_arrays_sha256
    ):
        raise TimeLawArtifactError(
            "weighted arc evidence does not bind the supplied "
            "qpos/q_s/q_ss arrays"
        )
    npz_bytes = _deterministic_npz(arrays)
    manifest = _manifest(
        motion_id=checked_motion_id,
        scope=checked_scope,
        dimensions=dimensions,
        arrays=arrays,
        npz_bytes=npz_bytes,
        bindings=binding_payload,
        markers=markers,
        window=window,
    )
    _validate_manifest(manifest, arrays, npz_bytes)
    manifest_bytes = _json_bytes(manifest)
    return TimeLawArtifact(
        arrays=arrays,
        npz_bytes=npz_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
    )


def validate_time_law_artifact(artifact: TimeLawArtifact) -> None:
    """Revalidate every byte, digest, field, and algebraic relation in memory."""

    if not isinstance(artifact, TimeLawArtifact):
        raise TimeLawArtifactError("artifact must be a TimeLawArtifact")
    plain_manifest = _deep_thaw_json(artifact.manifest)
    if _json_bytes(plain_manifest) != artifact.manifest_bytes:
        raise TimeLawArtifactError(
            "manifest_bytes do not match the sealed canonical manifest"
        )
    _validate_manifest(plain_manifest, artifact.arrays, artifact.npz_bytes)
    deterministic = _deterministic_npz(artifact.arrays)
    if deterministic != artifact.npz_bytes:
        raise TimeLawArtifactError(
            "in-memory NPZ bytes are not the deterministic encoding of arrays"
        )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    plain = _deep_thaw_json(value)
    if not isinstance(plain, dict):
        raise TimeLawArtifactError("manifest top level must be a JSON object")
    return (json.dumps(plain, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TimeLawArtifactError(f"{label} is not UTF-8") from exc

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TimeLawArtifactError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise TimeLawArtifactError(f"{label} contains forbidden JSON constant {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except TimeLawArtifactError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TimeLawArtifactError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TimeLawArtifactError(f"{label} top level must be a JSON object")
    if _json_bytes(value) != payload:
        raise TimeLawArtifactError(
            f"{label} bytes are not the canonical sorted/indented JSON encoding"
        )
    return value


def _absolute_leaf(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_directory_nofollow(path: Path, label: str) -> tuple[int, tuple[int, int]]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TimeLawArtifactError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise TimeLawArtifactError(
            f"{label} must be a directory and immediate parent must be non-symlink"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TimeLawArtifactError(f"cannot open {label} {path}: {exc}") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
    ):
        os.close(descriptor)
        raise TimeLawArtifactError(f"{label} changed identity while opening")
    return descriptor, (opened.st_dev, opened.st_ino)


def _assert_directory_identity(
    descriptor: int, identity: tuple[int, int], label: str
) -> None:
    current = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise TimeLawArtifactError(f"{label} changed identity during operation")


def _assert_directory_path_identity(
    path: Path, identity: tuple[int, int], label: str
) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise TimeLawArtifactError(f"cannot restat {label} {path}: {exc}") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise TimeLawArtifactError(f"{label} path changed identity during operation")


def _read_regular_nofollow(
    path: Path,
    label: str,
    *,
    directory_fd: int | None = None,
) -> bytes:
    leaf: str | Path = path.name if directory_fd is not None else path
    try:
        if directory_fd is None:
            before = path.lstat()
        else:
            before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise TimeLawArtifactError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TimeLawArtifactError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        if directory_fd is None:
            descriptor = os.open(leaf, flags)
        else:
            descriptor = os.open(leaf, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise TimeLawArtifactError(f"cannot open {label} {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise TimeLawArtifactError(f"{label} changed identity while opening")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise TimeLawArtifactError(f"{label} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise TimeLawArtifactError(f"{label} size changed while being read")
        try:
            if directory_fd is None:
                named = path.lstat()
            else:
                named = os.stat(
                    leaf,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
        except OSError as exc:
            raise TimeLawArtifactError(
                f"{label} leaf changed while being read: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or named.st_dev != opened.st_dev
            or named.st_ino != opened.st_ino
        ):
            raise TimeLawArtifactError(
                f"{label} leaf changed identity while being read"
            )
        return payload
    finally:
        os.close(descriptor)


def _load_npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            expected_names = [f"{key}.npy" for key in ARRAY_KEYS]
            actual_names = [info.filename for info in infos]
            if actual_names != expected_names or len(set(actual_names)) != len(
                actual_names
            ):
                raise TimeLawArtifactError(
                    "NPZ member order/keyset is not the exact artifact contract"
                )
            for info in infos:
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits & 0x1
                ):
                    raise TimeLawArtifactError(
                        "NPZ members must be unencrypted stored regular entries"
                    )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != ARRAY_KEYS:
                raise TimeLawArtifactError("NPZ array key order changed")
            arrays = {
                key: np.ascontiguousarray(np.asarray(archive[key]))
                for key in ARRAY_KEYS
            }
    except TimeLawArtifactError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise TimeLawArtifactError(f"cannot decode artifact NPZ: {exc}") from exc
    _validate_array_contract(arrays)
    if _deterministic_npz(arrays) != payload:
        raise TimeLawArtifactError("NPZ bytes are not the deterministic encoding")
    return arrays


def read_time_law_artifact(
    artifact_path: os.PathLike[str] | str,
    *,
    manifest_path: os.PathLike[str] | str | None = None,
) -> TimeLawArtifact:
    """Strictly reopen and independently validate one artifact pair."""

    artifact_file = _absolute_leaf(artifact_path)
    manifest_file = (
        _absolute_leaf(manifest_path)
        if manifest_path is not None
        else artifact_file.with_suffix(artifact_file.suffix + ".manifest.json")
    )
    if artifact_file == manifest_file:
        raise TimeLawArtifactError("artifact and manifest paths must be distinct")
    if artifact_file.parent != manifest_file.parent:
        raise TimeLawArtifactError(
            "artifact and manifest must use the same parent directory"
        )
    directory_fd, directory_identity = _open_directory_nofollow(
        artifact_file.parent, "artifact pair parent directory"
    )
    try:
        npz_bytes = _read_regular_nofollow(
            artifact_file,
            "artifact NPZ",
            directory_fd=directory_fd,
        )
        manifest_bytes = _read_regular_nofollow(
            manifest_file,
            "artifact manifest",
            directory_fd=directory_fd,
        )
        _assert_directory_identity(
            directory_fd,
            directory_identity,
            "artifact pair parent directory",
        )
        _assert_directory_path_identity(
            artifact_file.parent,
            directory_identity,
            "artifact pair parent directory",
        )
    finally:
        os.close(directory_fd)
    manifest = _strict_json_bytes(manifest_bytes, "artifact manifest")
    arrays = _load_npz_bytes(npz_bytes)
    _validate_manifest(manifest, arrays, npz_bytes)
    return TimeLawArtifact(
        arrays=arrays,
        npz_bytes=npz_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
    )


def _unlink_if_same(
    path: Path,
    identity: tuple[int, int],
    *,
    directory_fd: int | None = None,
) -> None:
    leaf: str | Path = path.name if directory_fd is not None else path
    try:
        if directory_fd is None:
            current = path.lstat()
        else:
            current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        try:
            if directory_fd is None:
                path.unlink()
            else:
                os.unlink(leaf, dir_fd=directory_fd)
        except OSError:
            pass


def _exclusive_write(
    path: Path,
    payload: bytes,
    *,
    directory_fd: int | None = None,
) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    leaf: str | Path = path.name if directory_fd is not None else path
    if directory_fd is None:
        descriptor = os.open(leaf, flags, 0o600)
    else:
        descriptor = os.open(leaf, flags, 0o600, dir_fd=directory_fd)
    identity: tuple[int, int] | None = None
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("exclusive artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        if identity is not None:
            _unlink_if_same(path, identity, directory_fd=directory_fd)
        raise
    os.close(descriptor)
    assert identity is not None
    return identity


def _entry_exists(path: Path, *, directory_fd: int) -> bool:
    try:
        os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TimeLawArtifactError(f"cannot inspect output path {path}: {exc}") from exc
    return True


def write_time_law_artifact(
    artifact: TimeLawArtifact,
    artifact_path: os.PathLike[str] | str,
    *,
    manifest_path: os.PathLike[str] | str | None = None,
) -> tuple[Path, Path]:
    """Publish NPZ first and its validating manifest receipt last.

    Two filesystem entries cannot be committed atomically.  A crash may leave
    an orphan NPZ, which is explicitly incomplete and rejected by the reader.
    In-process failures roll back only the exact inodes created by this call.
    """

    validate_time_law_artifact(artifact)
    output = _absolute_leaf(artifact_path)
    if output.suffix != ".npz":
        raise TimeLawArtifactError("artifact output path must end in .npz")
    manifest = (
        _absolute_leaf(manifest_path)
        if manifest_path is not None
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    if output == manifest:
        raise TimeLawArtifactError("artifact and manifest paths must be distinct")
    if output.parent != manifest.parent:
        raise TimeLawArtifactError(
            "artifact and manifest must use the same parent directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    directory_fd, directory_identity = _open_directory_nofollow(
        output.parent, "artifact pair parent directory"
    )
    try:
        occupied = [
            str(path)
            for path in (output, manifest)
            if _entry_exists(path, directory_fd=directory_fd)
        ]
        if occupied:
            raise FileExistsError(
                "refusing to overwrite existing path(s): " + ", ".join(occupied)
            )
        created: list[tuple[Path, tuple[int, int]]] = []
        try:
            created.append(
                (
                    output,
                    _exclusive_write(
                        output,
                        artifact.npz_bytes,
                        directory_fd=directory_fd,
                    ),
                )
            )
            created.append(
                (
                    manifest,
                    _exclusive_write(
                        manifest,
                        artifact.manifest_bytes,
                        directory_fd=directory_fd,
                    ),
                )
            )
            os.fsync(directory_fd)
            written_npz = _read_regular_nofollow(
                output, "artifact NPZ", directory_fd=directory_fd
            )
            written_manifest = _read_regular_nofollow(
                manifest,
                "artifact manifest",
                directory_fd=directory_fd,
            )
            _assert_directory_identity(
                directory_fd,
                directory_identity,
                "artifact pair parent directory",
            )
            _assert_directory_path_identity(
                output.parent,
                directory_identity,
                "artifact pair parent directory",
            )
            if (
                written_npz != artifact.npz_bytes
                or written_manifest != artifact.manifest_bytes
            ):
                raise TimeLawArtifactError(
                    "written artifact pair differs from the in-memory candidate"
                )
            reopened_manifest = _strict_json_bytes(
                written_manifest, "artifact manifest"
            )
            reopened_arrays = _load_npz_bytes(written_npz)
            _validate_manifest(reopened_manifest, reopened_arrays, written_npz)
        except BaseException:
            for path, identity in reversed(created):
                _unlink_if_same(path, identity, directory_fd=directory_fd)
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            raise
    finally:
        os.close(directory_fd)
    return output, manifest


__all__ = [
    "ARRAY_KEYS",
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_TYPE",
    "ArtifactBindings",
    "BUILD_VERDICT",
    "CELL_SIDE_INTERIOR",
    "CELL_SIDE_LEFT",
    "CELL_SIDE_RIGHT",
    "COLLOCATION_ORDER",
    "FPS_HZ",
    "PATH_DERIVATIVE_METHODS",
    "PATH_EVALUATION_ARRAY_KEYS",
    "PathEvaluatorBinding",
    "PUBLICATION_CLASS",
    "SolverBinding",
    "TimeLawArtifact",
    "TimeLawArtifactError",
    "TimeLawTrace",
    "WeightedArcLengthBinding",
    "build_time_law_artifact",
    "path_evaluation_array_sha256",
    "path_evaluation_receipt_sha256",
    "read_time_law_artifact",
    "validate_time_law_artifact",
    "weighted_arc_length_receipt_sha256",
    "write_time_law_artifact",
]
