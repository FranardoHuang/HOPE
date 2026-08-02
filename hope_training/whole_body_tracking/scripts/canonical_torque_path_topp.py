#!/usr/bin/env python3
"""Fail-closed torque-aware scalar retiming for a MuJoCo tangent path.

This module is the dynamics stage *after* geometric path construction and the
kinematic :mod:`canonical_path_topp` warm start.  Its state and control are

``x = sdot**2`` and ``u = sddot``.

For a path expressed in MuJoCo tangent coordinates,

``qvel = q_s * sqrt(x)``
``qacc = q_s * u + q_ss * x``.

At one path location the smooth inverse dynamics are identified as

``tau(u, x) = A*u + c0 + c_sqrt*sqrt(x) + c_x*x``.

The square-root term is retained deliberately: native viscous damping is not
affine in ``x``.  Identification is checked against independent probes and
fails closed if the supplied inverse dynamics do not have this form.

The fixed-base path remains supported.  A floating root with no support is
rejected.  A grounded A3 path is accepted only through the explicit
:class:`MujocoGroundContactLPSolver`: at every sampled path state it solves the
full floating-base equation for actuator generalized forces and double-support
foot forces.  The solver uses an exact bound MuJoCo model, active floor/foot
geometry, point Jacobians, unilateral normal forces, a conservative
linear-friction pyramid, and the convex hull of each collision sole as the CoP
support domain.  Missing contact/model/solver evidence fails closed.

MuJoCo ``frictionloss`` is never counted as free assistance.  The smooth
inverse-dynamics evaluator must have it disabled, while the complete absolute
frictionloss value is deducted from both sides of every generalized-force
range.  This is conservative for either impending direction.

The returned time law is a three-point-per-cell collocation candidate (left,
midpoint, right).  Even in grounded mode it is only a reproducible *sampled
path* certificate, not continuous-contact TOPP-RA, closed-loop balance, or a
hardware certificate.  Those limitations are machine-readable in ``report``.
No training, GPU, viewer, or robot command is used here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


class TorqueRetimeError(ValueError):
    """The torque retiming problem is malformed or infeasible."""


class IncompleteTorqueCertificate(TorqueRetimeError):
    """A required physical model is absent; no feasibility claim was made."""

    def __init__(
        self,
        reason: str,
        *,
        missing: Sequence[str],
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.report = {
            "status": "INCOMPLETE_FAIL_CLOSED",
            "reason": str(reason),
            "missing": [str(item) for item in missing],
        }
        if details is not None:
            self.report["details"] = dict(details)


InverseDynamics = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

GROUND_LP_OBJECTIVE_FEASIBILITY = "feasibility"
GROUND_LP_OBJECTIVE_HOLD_MINIMAX = (
    "hold_minimax_normalized_available_torque"
)


@dataclass(frozen=True)
class DirectActuatorContract:
    """Direct-drive limits in actuator-force and generalized-force units.

    ``dof_to_actuator[i]`` names the unique actuator driving tangent DoF ``i``.
    ``actuator_gear[a]`` maps actuator force to generalized force.  Control
    bounds are actuator-force bounds because this contract accepts only a
    stateless, fixed, unit-gain motor.  Infinite source bounds are permitted,
    but their intersection must give a finite effective range for every DoF.

    ``joint_force_*`` are MuJoCo joint ``actfrcrange`` limits in generalized
    force units.  ``frictionloss`` is also generalized force (N or N*m).
    """

    dof_to_actuator: np.ndarray
    actuator_gear: np.ndarray
    actuator_force_lower: np.ndarray
    actuator_force_upper: np.ndarray
    actuator_control_lower: np.ndarray
    actuator_control_upper: np.ndarray
    joint_force_lower: np.ndarray
    joint_force_upper: np.ndarray
    frictionloss: np.ndarray
    free_dof_count: int = 0
    support_mode: str = "fixed_base"
    contact_mode: Optional[str] = None
    fixed_lp_solver: Optional[str] = None
    model_binding: Optional[str] = None


@dataclass(frozen=True)
class TangentPathGrid:
    """MuJoCo qpos path plus exact tangent derivatives at nodes and midpoints.

    Quaternion qpos columns are configurations, not linearly interpolated
    coordinates.  The caller must evaluate ``q_s`` and ``q_ss`` in MuJoCo's
    ``nv``-dimensional tangent space.  Midpoint arrays correspond exactly to
    ``0.5 * (path_position[i] + path_position[i+1])``.
    """

    path_position: np.ndarray
    qpos: np.ndarray
    q_s: np.ndarray
    q_ss: np.ndarray
    midpoint_qpos: np.ndarray
    midpoint_q_s: np.ndarray
    midpoint_q_ss: np.ndarray


@dataclass(frozen=True)
class DynamicsSlice:
    """Identified smooth inverse dynamics at one path location."""

    A: np.ndarray
    c0: np.ndarray
    c_sqrt: np.ndarray
    c_x: np.ndarray
    max_probe_residual: float

    def generalized_force(self, speed_sq: float, path_acceleration: float) -> np.ndarray:
        x = float(speed_sq)
        u = float(path_acceleration)
        if not np.isfinite(x) or x < 0.0 or not np.isfinite(u):
            raise TorqueRetimeError("x must be non-negative and u must be finite")
        return self.A * u + self.c0 + self.c_sqrt * np.sqrt(x) + self.c_x * x


@dataclass(frozen=True)
class TorqueRetimeResult:
    """Scalar time law satisfying the fixed-base collocation constraints."""

    path_position: np.ndarray
    speed_squared: np.ndarray
    path_speed: np.ndarray
    path_acceleration: np.ndarray
    time_s: np.ndarray
    report: dict


@dataclass(frozen=True)
class GroundContactLPSolution:
    """One exact sampled-state actuator/contact-force LP result."""

    feasible: bool
    actuator_generalized_force: np.ndarray
    point_force_floor: np.ndarray
    equality_residual: float
    root_residual: float
    report: dict


@dataclass(frozen=True)
class GroundContactConfig:
    """Strict A3 double-support contact and CPU LP contract.

    ``expected_model_binding`` is mandatory.  It binds all dynamics,
    actuator, collision, friction, and mesh arrays hashed by
    :func:`_mujoco_model_binding`; merely passing any loadable MJCF is not
    enough.
    """

    expected_model_binding: str
    model_source_path: str
    expected_source_sha256: str
    floor_geom_name: str = "floor"
    foot_body_names: Tuple[str, str] = (
        "left_ankle_roll_Link",
        "right_ankle_roll_Link",
    )
    solver: str = "scipy.optimize.linprog:highs"
    friction_safety_fraction: float = 0.8
    contact_gap_tolerance_m: float = 2.0e-3
    penetration_tolerance_m: float = 2.0e-3
    support_surface_band_m: float = 2.0e-3
    sole_hull_containment_tolerance_m: float = 2.0e-4
    max_support_points_per_foot: int = 32
    minimum_normal_force_per_foot_n: float = 1.0
    minimum_normal_force_per_contact_n: float = 0.0
    position_tolerance_rad: float = 1.0e-8
    velocity_tolerance: float = 1.0e-8
    contact_path_tangent_tolerance_m: float = 2.0e-6
    contact_velocity_tolerance_m_s: float = 2.0e-5
    contact_acceleration_tolerance_m_s2: float = 2.0e-4
    path_tangent_fd_absolute_tolerance: float = 1.0e-3
    path_tangent_fd_relative_tolerance: float = 0.08
    path_curvature_fd_absolute_tolerance: float = 1.0e-2
    path_curvature_fd_relative_tolerance: float = 0.15
    equality_residual_tolerance: float = 2.0e-7

    def __post_init__(self) -> None:
        binding = str(self.expected_model_binding)
        if len(binding) != 64 or any(c not in "0123456789abcdef" for c in binding):
            raise TorqueRetimeError(
                "expected_model_binding must be 64 lowercase hexadecimal digits"
            )
        source_hash = str(self.expected_source_sha256)
        if len(source_hash) != 64 or any(
            c not in "0123456789abcdef" for c in source_hash
        ):
            raise TorqueRetimeError(
                "expected_source_sha256 must be 64 lowercase hexadecimal digits"
            )
        if not str(self.model_source_path):
            raise TorqueRetimeError("model_source_path must be explicit")
        if self.solver != "scipy.optimize.linprog:highs":
            raise TorqueRetimeError(
                "ground contact solver must be scipy.optimize.linprog:highs"
            )
        if len(self.foot_body_names) != 2 or len(set(self.foot_body_names)) != 2:
            raise TorqueRetimeError("exactly two distinct foot body names are required")
        for name, value in (
            ("friction_safety_fraction", self.friction_safety_fraction),
            ("contact_gap_tolerance_m", self.contact_gap_tolerance_m),
            ("penetration_tolerance_m", self.penetration_tolerance_m),
            ("support_surface_band_m", self.support_surface_band_m),
            (
                "sole_hull_containment_tolerance_m",
                self.sole_hull_containment_tolerance_m,
            ),
            ("minimum_normal_force_per_foot_n", self.minimum_normal_force_per_foot_n),
            (
                "minimum_normal_force_per_contact_n",
                self.minimum_normal_force_per_contact_n,
            ),
            ("position_tolerance_rad", self.position_tolerance_rad),
            ("velocity_tolerance", self.velocity_tolerance),
            (
                "contact_path_tangent_tolerance_m",
                self.contact_path_tangent_tolerance_m,
            ),
            (
                "contact_velocity_tolerance_m_s",
                self.contact_velocity_tolerance_m_s,
            ),
            (
                "contact_acceleration_tolerance_m_s2",
                self.contact_acceleration_tolerance_m_s2,
            ),
            (
                "path_tangent_fd_absolute_tolerance",
                self.path_tangent_fd_absolute_tolerance,
            ),
            (
                "path_tangent_fd_relative_tolerance",
                self.path_tangent_fd_relative_tolerance,
            ),
            (
                "path_curvature_fd_absolute_tolerance",
                self.path_curvature_fd_absolute_tolerance,
            ),
            (
                "path_curvature_fd_relative_tolerance",
                self.path_curvature_fd_relative_tolerance,
            ),
            ("equality_residual_tolerance", self.equality_residual_tolerance),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise TorqueRetimeError(f"{name} must be finite and non-negative")
        if not 0.0 < float(self.friction_safety_fraction) <= 1.0:
            raise TorqueRetimeError("friction_safety_fraction must lie in (0, 1]")
        if float(self.minimum_normal_force_per_foot_n) <= 0.0:
            raise TorqueRetimeError(
                "minimum_normal_force_per_foot_n must be strictly positive"
            )
        if (
            float(self.path_tangent_fd_relative_tolerance) > 1.0
            or float(self.path_curvature_fd_relative_tolerance) > 1.0
        ):
            raise TorqueRetimeError(
                "path finite-difference relative tolerances must not exceed 1"
            )
        if (
            not isinstance(self.max_support_points_per_foot, int)
            or self.max_support_points_per_foot < 3
        ):
            raise TorqueRetimeError(
                "max_support_points_per_foot must be an integer >= 3"
            )


@dataclass(frozen=True)
class _GroundGeometry:
    generalized_force_map: np.ndarray
    point_world: np.ndarray
    point_foot_index: np.ndarray
    floor_basis_world: np.ndarray
    friction_coefficient: float
    report: dict


GroundedStateFeasibility = Callable[
    [float, float, bool], GroundContactLPSolution
]


_EPS = 1.0e-12


def _real_array(name: str, value: Any, *, ndim: Optional[int] = None) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise TorqueRetimeError(f"{name} must be a real numeric array")
    out = raw.astype(np.float64, copy=False)
    if ndim is not None and out.ndim != ndim:
        raise TorqueRetimeError(f"{name} must have ndim={ndim}, got {out.ndim}")
    if not np.all(np.isfinite(out)):
        raise TorqueRetimeError(f"{name} must contain only finite values")
    return out


def _bounded_array(name: str, value: Any, size: int) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise TorqueRetimeError(f"{name} must be a real numeric vector")
    out = raw.astype(np.float64, copy=False)
    if out.shape != (size,):
        raise TorqueRetimeError(f"{name} must have shape ({size},), got {out.shape}")
    if np.any(np.isnan(out)):
        raise TorqueRetimeError(f"{name} must not contain NaN")
    return out


def _validate_path(path: TangentPathGrid) -> tuple[TangentPathGrid, int, int]:
    s = _real_array("path_position", path.path_position, ndim=1)
    if len(s) < 3 or np.any(np.diff(s) <= 0.0):
        raise TorqueRetimeError(
            "path_position must contain at least three strictly increasing values"
        )
    qpos = _real_array("qpos", path.qpos, ndim=2)
    q_s = _real_array("q_s", path.q_s, ndim=2)
    q_ss = _real_array("q_ss", path.q_ss, ndim=2)
    mid_qpos = _real_array("midpoint_qpos", path.midpoint_qpos, ndim=2)
    mid_q_s = _real_array("midpoint_q_s", path.midpoint_q_s, ndim=2)
    mid_q_ss = _real_array("midpoint_q_ss", path.midpoint_q_ss, ndim=2)
    nodes = len(s)
    if qpos.shape[0] != nodes:
        raise TorqueRetimeError("qpos row count must equal path_position length")
    if q_s.shape[0] != nodes or q_ss.shape != q_s.shape:
        raise TorqueRetimeError("q_s/q_ss must have one equal-width row per path node")
    if mid_qpos.shape != (nodes - 1, qpos.shape[1]):
        raise TorqueRetimeError(
            "midpoint_qpos must have shape (nodes-1, nq)"
        )
    if mid_q_s.shape != (nodes - 1, q_s.shape[1]) or mid_q_ss.shape != mid_q_s.shape:
        raise TorqueRetimeError(
            "midpoint_q_s/midpoint_q_ss must have shape (nodes-1, nv)"
        )
    tangent_norm = np.linalg.norm(mid_q_s, axis=1)
    scale = max(1.0, float(np.max(np.abs(mid_q_s))))
    if np.any(tangent_norm <= 1.0e-11 * scale):
        raise TorqueRetimeError("path contains a zero tangent at a cell midpoint")
    return (
        TangentPathGrid(s, qpos, q_s, q_ss, mid_qpos, mid_q_s, mid_q_ss),
        int(qpos.shape[1]),
        int(q_s.shape[1]),
    )


def resolve_generalized_force_limits(
    contract: DirectActuatorContract,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Intersect actuator, control, gear, joint, and dry-friction limits.

    A negative gear reverses interval endpoints.  Full ``frictionloss`` is then
    removed from *both* sides; this treats dry friction as a worst-case load,
    never as free assistance.
    """

    mapping_raw = np.asarray(contract.dof_to_actuator)
    if mapping_raw.ndim != 1 or not np.issubdtype(mapping_raw.dtype, np.integer):
        raise TorqueRetimeError("dof_to_actuator must be a one-dimensional integer array")
    mapping = mapping_raw.astype(np.int64, copy=False)
    nv = len(mapping)
    gear = _real_array("actuator_gear", contract.actuator_gear, ndim=1)
    nu = len(gear)
    if nu < 1 or np.any(np.abs(gear) <= _EPS):
        raise TorqueRetimeError("every actuator gear must be finite and non-zero")
    if np.any(mapping < 0) or np.any(mapping >= nu):
        raise TorqueRetimeError(
            "fixed-base direct-drive mode requires every DoF to map to one actuator"
        )
    if len(np.unique(mapping)) != nv:
        raise TorqueRetimeError("dof_to_actuator must be one-to-one")

    force_lo = _bounded_array(
        "actuator_force_lower", contract.actuator_force_lower, nu
    )
    force_hi = _bounded_array(
        "actuator_force_upper", contract.actuator_force_upper, nu
    )
    ctrl_lo = _bounded_array(
        "actuator_control_lower", contract.actuator_control_lower, nu
    )
    ctrl_hi = _bounded_array(
        "actuator_control_upper", contract.actuator_control_upper, nu
    )
    if np.any(force_lo >= force_hi) or np.any(ctrl_lo >= ctrl_hi):
        raise TorqueRetimeError("actuator force/control lower bounds must be below upper")
    joint_lo = _bounded_array("joint_force_lower", contract.joint_force_lower, nv)
    joint_hi = _bounded_array("joint_force_upper", contract.joint_force_upper, nv)
    friction = _real_array("frictionloss", contract.frictionloss, ndim=1)
    if friction.shape != (nv,) or np.any(friction < 0.0):
        raise TorqueRetimeError(
            f"frictionloss must have shape ({nv},) and be non-negative"
        )
    if np.any(joint_lo >= joint_hi):
        raise TorqueRetimeError("joint force lower bounds must be below upper bounds")

    effective_actuator_lo = np.maximum(force_lo, ctrl_lo)
    effective_actuator_hi = np.minimum(force_hi, ctrl_hi)
    if np.any(effective_actuator_lo >= effective_actuator_hi):
        raise TorqueRetimeError("actuator force/control intervals have empty intersection")

    generalized_lo = np.empty(nv, dtype=np.float64)
    generalized_hi = np.empty(nv, dtype=np.float64)
    for dof, actuator in enumerate(mapping):
        products = gear[actuator] * np.array(
            [effective_actuator_lo[actuator], effective_actuator_hi[actuator]]
        )
        transmitted_lo = float(np.min(products))
        transmitted_hi = float(np.max(products))
        generalized_lo[dof] = max(transmitted_lo, joint_lo[dof]) + friction[dof]
        generalized_hi[dof] = min(transmitted_hi, joint_hi[dof]) - friction[dof]
    if (
        np.any(~np.isfinite(generalized_lo))
        or np.any(~np.isfinite(generalized_hi))
        or np.any(generalized_lo >= generalized_hi)
    ):
        raise TorqueRetimeError(
            "effective generalized-force interval is empty/unbounded after "
            "actuator, joint, and conservative frictionloss intersection"
        )
    return generalized_lo, generalized_hi, {
        "dof_to_actuator": mapping.tolist(),
        "actuator_gear": gear.tolist(),
        "actuator_force_after_control": np.column_stack(
            (effective_actuator_lo, effective_actuator_hi)
        ).tolist(),
        "joint_force_before_friction": np.column_stack((joint_lo, joint_hi)).tolist(),
        "frictionloss_full_margin": friction.tolist(),
        "generalized_force_after_friction": np.column_stack(
            (generalized_lo, generalized_hi)
        ).tolist(),
    }


def _resolve_grounded_actuator_limits(
    contract: DirectActuatorContract,
    nv: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Resolve only mapped post-root rows for a grounded floating plant."""

    free = int(contract.free_dof_count)
    if free != 6:
        raise IncompleteTorqueCertificate(
            "grounded contact LP requires the A3 six-DoF free root",
            missing=["free_dof_count=6"],
            details={"free_dof_count": free},
        )
    mapping = np.asarray(contract.dof_to_actuator)
    if mapping.shape != (nv,) or not np.issubdtype(mapping.dtype, np.integer):
        raise TorqueRetimeError(
            f"grounded dof_to_actuator must be an integer vector of shape ({nv},)"
        )
    if np.any(mapping[:free] != -1):
        raise TorqueRetimeError("free-root rows must use the unmapped sentinel -1")
    if np.any(mapping[free:] < 0):
        raise TorqueRetimeError("every post-root DoF must have one direct actuator")
    friction = _real_array("frictionloss", contract.frictionloss, ndim=1)
    if friction.shape != (nv,):
        raise TorqueRetimeError(f"frictionloss must have shape ({nv},)")
    if np.any(np.abs(friction[:free]) > _EPS):
        raise TorqueRetimeError("free-root rows cannot reserve joint frictionloss")

    sliced = DirectActuatorContract(
        dof_to_actuator=mapping[free:].astype(np.int64, copy=True),
        actuator_gear=np.asarray(contract.actuator_gear).copy(),
        actuator_force_lower=np.asarray(contract.actuator_force_lower).copy(),
        actuator_force_upper=np.asarray(contract.actuator_force_upper).copy(),
        actuator_control_lower=np.asarray(contract.actuator_control_lower).copy(),
        actuator_control_upper=np.asarray(contract.actuator_control_upper).copy(),
        joint_force_lower=np.asarray(contract.joint_force_lower)[free:].copy(),
        joint_force_upper=np.asarray(contract.joint_force_upper)[free:].copy(),
        frictionloss=friction[free:].copy(),
    )
    lower, upper, report = resolve_generalized_force_limits(sliced)
    dof_indices = np.arange(free, nv, dtype=np.int64)
    report = dict(report)
    report["free_root_unactuated_dof_indices"] = list(range(free))
    report["actuated_dof_indices"] = dof_indices.tolist()
    return lower, upper, dof_indices, report


def identify_dynamics_slice(
    qpos: np.ndarray,
    q_s: np.ndarray,
    q_ss: np.ndarray,
    inverse_dynamics: InverseDynamics,
    *,
    relative_tolerance: float = 2.0e-9,
    absolute_tolerance: float = 2.0e-10,
) -> DynamicsSlice:
    """Identify and independently verify ``A*u+c0+c√*sqrt(x)+cx*x``."""

    q = _real_array("qpos", qpos, ndim=1)
    qs = _real_array("q_s", q_s, ndim=1)
    qss = _real_array("q_ss", q_ss, ndim=1)
    if qss.shape != qs.shape:
        raise TorqueRetimeError("q_s and q_ss must have equal shape")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise TorqueRetimeError("relative_tolerance must be finite and non-negative")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise TorqueRetimeError("absolute_tolerance must be finite and non-negative")

    nv = len(qs)

    def evaluate(x: float, u: float) -> np.ndarray:
        force = _real_array(
            "inverse_dynamics output",
            inverse_dynamics(
                q.copy(),
                qs * np.sqrt(float(x)),
                qs * float(u) + qss * float(x),
            ),
            ndim=1,
        )
        if force.shape != (nv,):
            raise TorqueRetimeError(
                f"inverse_dynamics output must have shape ({nv},), got {force.shape}"
            )
        return force

    c0 = evaluate(0.0, 0.0)
    A = evaluate(0.0, 1.0) - c0
    at_one = evaluate(1.0, 0.0) - c0
    at_four = evaluate(4.0, 0.0) - c0
    c_x = 0.5 * (at_four - 2.0 * at_one)
    c_sqrt = at_one - c_x
    result = DynamicsSlice(A=A, c0=c0, c_sqrt=c_sqrt, c_x=c_x, max_probe_residual=0.0)

    probes = ((0.0, -1.0), (0.25, 0.7), (2.25, -0.5), (3.1, 1.3))
    worst = 0.0
    for x, u in probes:
        actual = evaluate(x, u)
        predicted = result.generalized_force(x, u)
        residual = float(np.max(np.abs(actual - predicted)))
        scale = max(
            1.0,
            float(np.max(np.abs(actual))),
            float(np.max(np.abs(predicted))),
        )
        allowed = absolute_tolerance + relative_tolerance * scale
        if residual > allowed:
            raise TorqueRetimeError(
                "inverse dynamics is not affine in path acceleration with the "
                "supported c0+c_sqrt*sqrt(x)+c_x*x speed dependence; "
                f"probe (x={x}, u={u}) residual={residual:.9g}, allowed={allowed:.9g}. "
                "Disable frictionloss inside inverse dynamics and reserve it as a limit margin."
            )
        worst = max(worst, residual)
    return DynamicsSlice(
        A=A,
        c0=c0,
        c_sqrt=c_sqrt,
        c_x=c_x,
        max_probe_residual=worst,
    )


def _path_acceleration_interval(
    dynamics: DynamicsSlice,
    speed_sq: float,
    effort_lower: np.ndarray,
    effort_upper: np.ndarray,
    *,
    tolerance: float,
) -> tuple[float, float]:
    """Intersect all per-DoF inequalities into one scalar-u interval."""

    x = float(speed_sq)
    if not np.isfinite(x) or x < 0.0:
        raise TorqueRetimeError("speed squared must be finite and non-negative")
    bias = dynamics.generalized_force(x, 0.0)
    lower = -np.inf
    upper = np.inf
    for dof, coefficient in enumerate(dynamics.A):
        lo = float(effort_lower[dof])
        hi = float(effort_upper[dof])
        b = float(bias[dof])
        if abs(float(coefficient)) <= tolerance:
            if b < lo - tolerance or b > hi + tolerance:
                return np.inf, -np.inf
            continue
        endpoints = np.array([(lo - b) / coefficient, (hi - b) / coefficient])
        lower = max(lower, float(np.min(endpoints)))
        upper = min(upper, float(np.max(endpoints)))
    return float(lower), float(upper)


def _transition_feasible(
    cell: int,
    left_x: float,
    right_x: float,
    path_position: np.ndarray,
    node_dynamics: Sequence[DynamicsSlice],
    midpoint_dynamics: Sequence[DynamicsSlice],
    effort_lower: np.ndarray,
    effort_upper: np.ndarray,
    *,
    node_grounded: Optional[Sequence[GroundedStateFeasibility]],
    midpoint_grounded: Optional[Sequence[GroundedStateFeasibility]],
    no_brake_interval: Optional[tuple[float, float]],
    tolerance: float,
) -> bool:
    ds = float(path_position[cell + 1] - path_position[cell])
    u = (float(right_x) - float(left_x)) / (2.0 * ds)
    midpoint_x = 0.5 * (float(left_x) + float(right_x))
    if no_brake_interval is not None:
        lo, hi = no_brake_interval
        cell_lo = float(path_position[cell])
        cell_hi = float(path_position[cell + 1])
        overlaps = max(cell_lo, lo) < min(cell_hi, hi)
        if overlaps:
            # This is a directional policy, so a feasibility tolerance must
            # not turn a tiny negative acceleration into a legal "no brake"
            # step.  False rejection at roundoff is safer than a fake green.
            if u < 0.0:
                return False
    grounded_callbacks: tuple[
        Optional[GroundedStateFeasibility],
        Optional[GroundedStateFeasibility],
        Optional[GroundedStateFeasibility],
    ] = (
        (
            None
            if node_grounded is None
            else node_grounded[cell]
        ),
        (
            None
            if midpoint_grounded is None
            else midpoint_grounded[cell]
        ),
        (
            None
            if node_grounded is None
            else node_grounded[cell + 1]
        ),
    )
    for dynamics, x, grounded in (
        (node_dynamics[cell], left_x, grounded_callbacks[0]),
        (midpoint_dynamics[cell], midpoint_x, grounded_callbacks[1]),
        (node_dynamics[cell + 1], right_x, grounded_callbacks[2]),
    ):
        if grounded is not None:
            if not grounded(float(x), float(u), False).feasible:
                return False
        else:
            u_lo, u_hi = _path_acceleration_interval(
                dynamics,
                x,
                effort_lower,
                effort_upper,
                tolerance=tolerance,
            )
            if u < u_lo - tolerance or u > u_hi + tolerance:
                return False
    return True


def _largest_feasible_neighbor(
    *,
    cell: int,
    fixed_x: float,
    candidate_upper: float,
    forward: bool,
    path_position: np.ndarray,
    node_dynamics: Sequence[DynamicsSlice],
    midpoint_dynamics: Sequence[DynamicsSlice],
    effort_lower: np.ndarray,
    effort_upper: np.ndarray,
    node_grounded: Optional[Sequence[GroundedStateFeasibility]],
    midpoint_grounded: Optional[Sequence[GroundedStateFeasibility]],
    no_brake_interval: Optional[tuple[float, float]],
    tolerance: float,
    search_samples: int,
) -> Optional[float]:
    """Return a verified high feasible neighbor; false negatives remain closed."""

    upper = max(0.0, float(candidate_upper))
    ds = float(path_position[cell + 1] - path_position[cell])
    if node_grounded is None:
        endpoint_dynamics = (
            node_dynamics[cell] if forward else node_dynamics[cell + 1]
        )
        endpoint_u_lo, endpoint_u_hi = _path_acceleration_interval(
            endpoint_dynamics,
            fixed_x,
            effort_lower,
            effort_upper,
            tolerance=tolerance,
        )
        if endpoint_u_lo > endpoint_u_hi:
            return None
        if forward:
            bracket_lo = fixed_x + 2.0 * ds * endpoint_u_lo
            bracket_hi = fixed_x + 2.0 * ds * endpoint_u_hi
        else:
            # u=(right_x-left_x)/(2ds), with right_x=fixed_x.
            bracket_lo = fixed_x - 2.0 * ds * endpoint_u_hi
            bracket_hi = fixed_x - 2.0 * ds * endpoint_u_lo
        lower = max(0.0, float(bracket_lo))
        upper = min(upper, float(bracket_hi))
        if not np.isfinite(lower):
            return None
        if not np.isfinite(upper):
            upper = max(0.0, float(candidate_upper))
        if lower > upper + tolerance:
            return None
    else:
        # Contact feasibility is a coupled LP rather than independent
        # per-row u intervals.  Search the entire finite velocity-derived
        # neighbor range.  This can reject a narrow feasible island (closed
        # false negative), but never fabricates a grounded pass.
        lower = 0.0

    def feasible(candidate: float) -> bool:
        left, right = (
            (fixed_x, candidate) if forward else (candidate, fixed_x)
        )
        return _transition_feasible(
            cell,
            left,
            right,
            path_position,
            node_dynamics,
            midpoint_dynamics,
            effort_lower,
            effort_upper,
            node_grounded=node_grounded,
            midpoint_grounded=midpoint_grounded,
            no_brake_interval=no_brake_interval,
            tolerance=tolerance,
        )

    candidates = np.linspace(upper, lower, search_samples, dtype=np.float64)
    found_index: Optional[int] = None
    for index, candidate in enumerate(candidates):
        if feasible(float(candidate)):
            found_index = index
            break
    if found_index is None:
        return None
    accepted = float(candidates[found_index])
    if found_index == 0:
        return accepted
    rejected = float(candidates[found_index - 1])
    # Refine only between a directly observed feasible/rejected pair.  The
    # final returned value is always rechecked, so a non-monotone feasible set
    # can cause conservative slowdown/failure but never a false pass.
    low = accepted
    high = rejected
    for _ in range(48):
        middle = 0.5 * (low + high)
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low if feasible(low) else accepted


def _solve_speed_profile(
    path_position: np.ndarray,
    speed_sq_cap: np.ndarray,
    node_dynamics: Sequence[DynamicsSlice],
    midpoint_dynamics: Sequence[DynamicsSlice],
    effort_lower: np.ndarray,
    effort_upper: np.ndarray,
    *,
    node_grounded: Optional[Sequence[GroundedStateFeasibility]],
    midpoint_grounded: Optional[Sequence[GroundedStateFeasibility]],
    start_speed: float,
    end_speed: float,
    no_brake_interval: Optional[tuple[float, float]],
    tolerance: float,
    search_samples: int,
    max_sweeps: int,
) -> tuple[np.ndarray, int]:
    """Monotone forward/backward envelope with verified cell transitions."""

    profile = speed_sq_cap.copy()
    profile[0] = start_speed * start_speed
    profile[-1] = end_speed * end_speed
    if profile[0] > speed_sq_cap[0] + tolerance or profile[-1] > speed_sq_cap[-1] + tolerance:
        raise TorqueRetimeError("start/end path speed exceeds the velocity-derived cap")

    for sweep in range(max_sweeps):
        before = profile.copy()
        profile[0] = start_speed * start_speed
        # The terminal node is a fixed boundary, not an upper cap that a
        # forward reachability pass can satisfy in one last cell.  The backward
        # pass owns that last transition.  Symmetrically, the forward pass owns
        # the first transition from the fixed start boundary.
        for cell in range(len(profile) - 2):
            candidate = _largest_feasible_neighbor(
                cell=cell,
                fixed_x=float(profile[cell]),
                candidate_upper=float(min(profile[cell + 1], speed_sq_cap[cell + 1])),
                forward=True,
                path_position=path_position,
                node_dynamics=node_dynamics,
                midpoint_dynamics=midpoint_dynamics,
                effort_lower=effort_lower,
                effort_upper=effort_upper,
                node_grounded=node_grounded,
                midpoint_grounded=midpoint_grounded,
                no_brake_interval=no_brake_interval,
                tolerance=tolerance,
                search_samples=search_samples,
            )
            if candidate is None:
                if node_grounded is not None:
                    raise IncompleteTorqueCertificate(
                        f"no sampled grounded contact-LP-feasible forward "
                        f"transition in cell {cell}",
                        missing=[
                            "sampled actuator/contact-force-feasible speed transition"
                        ],
                    )
                raise TorqueRetimeError(
                    f"no torque-feasible forward transition in cell {cell}; "
                    "the fixed path or current boundary speed is infeasible"
                )
            profile[cell + 1] = min(profile[cell + 1], candidate)

        profile[-1] = end_speed * end_speed
        for cell in range(len(profile) - 2, 0, -1):
            candidate = _largest_feasible_neighbor(
                cell=cell,
                fixed_x=float(profile[cell + 1]),
                candidate_upper=float(min(profile[cell], speed_sq_cap[cell])),
                forward=False,
                path_position=path_position,
                node_dynamics=node_dynamics,
                midpoint_dynamics=midpoint_dynamics,
                effort_lower=effort_lower,
                effort_upper=effort_upper,
                node_grounded=node_grounded,
                midpoint_grounded=midpoint_grounded,
                no_brake_interval=no_brake_interval,
                tolerance=tolerance,
                search_samples=search_samples,
            )
            if candidate is None:
                reason = (
                    "no-early-brake window leaves insufficient followthrough to stop"
                    if no_brake_interval is not None
                    else "no torque-feasible backward transition"
                )
                if node_grounded is not None:
                    raise IncompleteTorqueCertificate(
                        f"{reason} in grounded cell {cell}",
                        missing=[
                            "sampled actuator/contact-force-feasible recovery transition"
                        ],
                    )
                raise TorqueRetimeError(f"{reason} in cell {cell}")
            profile[cell] = min(profile[cell], candidate)
        profile[0] = start_speed * start_speed

        if np.max(np.abs(profile - before)) <= tolerance * max(
            1.0, float(np.max(speed_sq_cap))
        ):
            break
    else:
        if node_grounded is not None:
            raise IncompleteTorqueCertificate(
                "grounded contact-LP forward/backward envelope did not converge",
                missing=["converged sampled grounded speed profile"],
            )
        raise TorqueRetimeError("torque forward/backward envelope did not converge")

    for cell in range(len(profile) - 1):
        if not _transition_feasible(
            cell,
            float(profile[cell]),
            float(profile[cell + 1]),
            path_position,
            node_dynamics,
            midpoint_dynamics,
            effort_lower,
            effort_upper,
            node_grounded=node_grounded,
            midpoint_grounded=midpoint_grounded,
            no_brake_interval=no_brake_interval,
            tolerance=tolerance,
        ):
            if node_grounded is not None:
                raise IncompleteTorqueCertificate(
                    f"grounded final transition validation rejected cell {cell}",
                    missing=[
                        "sampled actuator/contact-force-feasible final transition"
                    ],
                )
            raise TorqueRetimeError(
                f"internal final validation rejected torque transition in cell {cell}"
            )
    return np.maximum(profile, 0.0), sweep + 1


def retime_torque_path(
    path: TangentPathGrid,
    inverse_dynamics: InverseDynamics,
    actuator_contract: DirectActuatorContract,
    velocity_limits: np.ndarray,
    *,
    grounded_contact_solver: Optional["MujocoGroundContactLPSolver"] = None,
    start_speed: float = 0.0,
    end_speed: float = 0.0,
    window: Optional[tuple[float, float]] = None,
    no_early_brake: bool = False,
    dynamics_relative_tolerance: float = 2.0e-9,
    dynamics_absolute_tolerance: float = 2.0e-10,
    feasibility_tolerance: float = 1.0e-9,
    search_samples: int = 129,
    max_sweeps: int = 64,
) -> TorqueRetimeResult:
    """Retime one fixed-base or explicit double-support tangent path.

    ``no_early_brake=True`` enforces ``u >= 0`` for every cell with any
    positive-width overlap between path start and the end of ``window``.  It is
    a hard comparison
    policy, not a claim that every
    actuator torque is constant or that every joint acceleration is positive.
    If recovery geometry is too short, the call fails and asks the caller to
    extend the followthrough instead of silently braking inside the window.
    """

    checked_path, _nq, nv = _validate_path(path)
    if not isinstance(actuator_contract.free_dof_count, (int, np.integer)):
        raise TorqueRetimeError("free_dof_count must be an integer")
    if actuator_contract.free_dof_count < 0 or actuator_contract.free_dof_count > nv:
        raise TorqueRetimeError("free_dof_count is outside [0, nv]")
    support_mode = str(actuator_contract.support_mode)
    contract_binding = actuator_contract.model_binding
    inverse_binding = getattr(inverse_dynamics, "model_binding", None)
    if contract_binding is not None and inverse_binding is not None:
        if str(contract_binding) != str(inverse_binding):
            raise TorqueRetimeError(
                "inverse dynamics and actuator contract have different model bindings"
            )
        plant_binding_status = "BOUND_MATCH"
    else:
        plant_binding_status = "UNBOUND_CALLER_SUPPLIED_CANDIDATE_ONLY"
    grounded_mode = bool(actuator_contract.free_dof_count)
    if actuator_contract.free_dof_count:
        if support_mode in {"none", "floating_no_contact"}:
            raise IncompleteTorqueCertificate(
                "floating-root inverse dynamics cannot be balanced by joint actuators "
                "without external support",
                missing=["ground contact mode", "contact force variables"],
            )
        if support_mode != "ground":
            raise IncompleteTorqueCertificate(
                f"unsupported floating-root support_mode={support_mode!r}",
                missing=["support_mode='ground'"],
            )
        if (
            actuator_contract.fixed_lp_solver
            != "scipy.optimize.linprog:highs"
        ):
            raise IncompleteTorqueCertificate(
                "grounded retiming requires the audited fixed CPU LP solver",
                missing=[
                    "fixed CPU LP solver",
                    "scipy.optimize.linprog:highs",
                ],
                details={
                    "fixed_lp_solver": actuator_contract.fixed_lp_solver
                },
            )
        if actuator_contract.contact_mode != "double_support_floor":
            raise IncompleteTorqueCertificate(
                "grounded retiming requires the explicit A3 double-support mode",
                missing=["contact_mode='double_support_floor'"],
            )
        if grounded_contact_solver is None:
            raise IncompleteTorqueCertificate(
                "grounded floating-root torque retiming has no contact solver",
                missing=[
                    "MujocoGroundContactLPSolver",
                    "friction-pyramid/contact-kinematics constraints",
                ],
            )
        solver_binding = getattr(
            grounded_contact_solver, "model_binding", None
        )
        if (
            contract_binding is None
            or inverse_binding is None
            or solver_binding is None
            or len({str(contract_binding), str(inverse_binding), str(solver_binding)})
            != 1
        ):
            raise IncompleteTorqueCertificate(
                "grounded path requires one explicit model binding shared by "
                "contract, inverse dynamics, and contact LP",
                missing=["exact shared MuJoCo model/hash"],
                details={
                    "contract_binding": contract_binding,
                    "inverse_binding": inverse_binding,
                    "solver_binding": solver_binding,
                },
            )
        plant_binding_status = "BOUND_MATCH_GROUNDED_TRIPLE"
    elif support_mode != "fixed_base":
        raise IncompleteTorqueCertificate(
            f"unsupported support_mode={support_mode!r}",
            missing=["fixed-base declaration or grounded contact LP"],
        )
    elif grounded_contact_solver is not None:
        raise TorqueRetimeError(
            "grounded_contact_solver must not be supplied for fixed_base mode"
        )

    if grounded_mode:
        (
            effort_lower,
            effort_upper,
            actuated_dof_indices,
            force_report,
        ) = _resolve_grounded_actuator_limits(actuator_contract, nv)
    else:
        effort_lower, effort_upper, force_report = resolve_generalized_force_limits(
            actuator_contract
        )
        if len(effort_lower) != nv:
            raise TorqueRetimeError(
                f"actuator contract has nv={len(effort_lower)}, tangent path has nv={nv}"
            )
        actuated_dof_indices = np.arange(nv, dtype=np.int64)
    velocity = _real_array("velocity_limits", velocity_limits, ndim=1)
    if velocity.shape != (nv,) or np.any(velocity <= 0.0):
        raise TorqueRetimeError(
            f"velocity_limits must have shape ({nv},) and be strictly positive"
        )
    for name, value in (("start_speed", start_speed), ("end_speed", end_speed)):
        if not np.isfinite(value) or value < 0.0:
            raise TorqueRetimeError(f"{name} must be finite and non-negative")
    if not isinstance(search_samples, int) or search_samples < 17:
        raise TorqueRetimeError("search_samples must be an integer >= 17")
    if not isinstance(max_sweeps, int) or max_sweeps < 1:
        raise TorqueRetimeError("max_sweeps must be a positive integer")
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance < 0.0:
        raise TorqueRetimeError(
            "feasibility_tolerance must be finite and non-negative"
        )
    grounded_path_derivative_report: Optional[dict] = None
    if grounded_mode:
        assert grounded_contact_solver is not None
        validator = getattr(grounded_contact_solver, "validate_path_grid", None)
        if not callable(validator):
            raise IncompleteTorqueCertificate(
                "ground contact solver has no independent qpos/q_s/q_ss validator",
                missing=["MuJoCo differentiatePos tangent-path validation"],
            )
        grounded_path_derivative_report = dict(validator(checked_path))

    no_brake_interval: Optional[tuple[float, float]] = None
    if window is not None:
        if not isinstance(window, tuple) or len(window) != 2:
            raise TorqueRetimeError("window must be a (start_s, end_s) tuple")
        window_start, window_end = map(float, window)
        if (
            not np.isfinite(window_start)
            or not np.isfinite(window_end)
            or window_start < checked_path.path_position[0]
            or window_end > checked_path.path_position[-1]
            or window_start >= window_end
        ):
            raise TorqueRetimeError("window is invalid or outside the path")
        if no_early_brake:
            no_brake_interval = (
                float(checked_path.path_position[0]),
                window_end,
            )
    elif no_early_brake:
        raise TorqueRetimeError("no_early_brake=True requires an explicit window")

    node_dynamics = [
        identify_dynamics_slice(
            checked_path.qpos[i],
            checked_path.q_s[i],
            checked_path.q_ss[i],
            inverse_dynamics,
            relative_tolerance=dynamics_relative_tolerance,
            absolute_tolerance=dynamics_absolute_tolerance,
        )
        for i in range(len(checked_path.path_position))
    ]
    midpoint_dynamics = [
        identify_dynamics_slice(
            checked_path.midpoint_qpos[i],
            checked_path.midpoint_q_s[i],
            checked_path.midpoint_q_ss[i],
            inverse_dynamics,
            relative_tolerance=dynamics_relative_tolerance,
            absolute_tolerance=dynamics_absolute_tolerance,
        )
        for i in range(len(checked_path.path_position) - 1)
    ]

    node_grounded: Optional[list[GroundedStateFeasibility]] = None
    midpoint_grounded: Optional[list[GroundedStateFeasibility]] = None
    if grounded_mode:
        assert grounded_contact_solver is not None

        def build_grounded_callback(
            qpos: np.ndarray,
            q_s: np.ndarray,
            q_ss: np.ndarray,
            dynamics: DynamicsSlice,
        ) -> GroundedStateFeasibility:
            qpos_bound = qpos.copy()
            q_s_bound = q_s.copy()
            q_ss_bound = q_ss.copy()

            def evaluate(
                speed_sq: float,
                path_acceleration: float,
                final_validation: bool,
            ) -> GroundContactLPSolution:
                x = float(speed_sq)
                u = float(path_acceleration)
                fitted = dynamics.generalized_force(x, u)
                return grounded_contact_solver.solve(
                    qpos_bound,
                    q_s_bound * np.sqrt(x),
                    q_s_bound * u + q_ss_bound * x,
                    actuated_dof_indices,
                    effort_lower,
                    effort_upper,
                    velocity,
                    expected_generalized_force=fitted,
                    path_tangent=q_s_bound,
                )

            return evaluate

        node_grounded = [
            build_grounded_callback(
                checked_path.qpos[i],
                checked_path.q_s[i],
                checked_path.q_ss[i],
                node_dynamics[i],
            )
            for i in range(len(checked_path.path_position))
        ]
        midpoint_grounded = [
            build_grounded_callback(
                checked_path.midpoint_qpos[i],
                checked_path.midpoint_q_s[i],
                checked_path.midpoint_q_ss[i],
                midpoint_dynamics[i],
            )
            for i in range(len(checked_path.path_position) - 1)
        ]

    speed_sq_cap = np.full(len(checked_path.path_position), np.inf, dtype=np.float64)
    for dof in range(nv):
        slope = np.abs(checked_path.q_s[:, dof])
        active = slope > 1.0e-12
        speed_sq_cap[active] = np.minimum(
            speed_sq_cap[active], (velocity[dof] / slope[active]) ** 2
        )
    midpoint_cap = np.full(len(checked_path.path_position) - 1, np.inf)
    for dof in range(nv):
        slope = np.abs(checked_path.midpoint_q_s[:, dof])
        active = slope > 1.0e-12
        midpoint_cap[active] = np.minimum(
            midpoint_cap[active], (velocity[dof] / slope[active]) ** 2
        )
    speed_sq_cap[:-1] = np.minimum(speed_sq_cap[:-1], midpoint_cap)
    speed_sq_cap[1:] = np.minimum(speed_sq_cap[1:], midpoint_cap)
    if np.any(~np.isfinite(speed_sq_cap)) or np.any(speed_sq_cap <= 0.0):
        raise TorqueRetimeError(
            "velocity limits do not produce finite positive scalar speed caps"
        )

    speed_sq, sweeps = _solve_speed_profile(
        checked_path.path_position,
        speed_sq_cap,
        node_dynamics,
        midpoint_dynamics,
        effort_lower,
        effort_upper,
        node_grounded=node_grounded,
        midpoint_grounded=midpoint_grounded,
        start_speed=float(start_speed),
        end_speed=float(end_speed),
        no_brake_interval=no_brake_interval,
        tolerance=float(feasibility_tolerance),
        search_samples=search_samples,
        max_sweeps=max_sweeps,
    )
    speed = np.sqrt(speed_sq)
    ds = np.diff(checked_path.path_position)
    denominator = speed[:-1] + speed[1:]
    if np.any(denominator <= _EPS):
        if no_brake_interval is not None:
            raise TorqueRetimeError(
                "no-early-brake window leaves insufficient followthrough; "
                "positive path distance is trapped between zero-speed nodes"
            )
        raise TorqueRetimeError("positive path distance is trapped between zero-speed nodes")
    dt = 2.0 * ds / denominator
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise TorqueRetimeError("retimer produced an invalid timeline")
    time_s = np.concatenate(([0.0], np.cumsum(dt)))
    path_acceleration = np.diff(speed_sq) / (2.0 * ds)

    worst_ratio = 0.0
    minimum_margin = np.inf
    max_fitted_to_actual_force_residual = 0.0
    max_ground_equality_residual = 0.0
    max_ground_root_residual = 0.0
    grounded_lp_sample_count = 0
    minimum_normal_force_per_foot = np.full(2, np.inf, np.float64)
    grounded_solver_summary: Optional[dict] = None
    grounded_kinematic_maxima = {
        "max_abs_Jp_qs_m_per_path_unit": 0.0,
        "max_abs_contact_velocity_m_s": 0.0,
        "max_abs_contact_acceleration_m_s2": 0.0,
    }
    grounded_min_friction = np.inf
    grounded_min_support_vertices = np.full(2, np.inf, np.float64)
    grounded_max_support_gap = 0.0
    grounded_max_penetration = 0.0
    grounded_geometry_digests: set[str] = set()
    max_applied_force_validation_tolerance = 0.0
    force_validation_tolerance = max(
        4.0 * float(feasibility_tolerance),
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(effort_lower))),
            float(np.max(np.abs(effort_upper))),
        ),
    )
    for cell, u in enumerate(path_acceleration):
        for dynamics, x, qpos, q_s, q_ss in (
            (
                node_dynamics[cell],
                speed_sq[cell],
                checked_path.qpos[cell],
                checked_path.q_s[cell],
                checked_path.q_ss[cell],
            ),
            (
                midpoint_dynamics[cell],
                0.5 * (speed_sq[cell] + speed_sq[cell + 1]),
                checked_path.midpoint_qpos[cell],
                checked_path.midpoint_q_s[cell],
                checked_path.midpoint_q_ss[cell],
            ),
            (
                node_dynamics[cell + 1],
                speed_sq[cell + 1],
                checked_path.qpos[cell + 1],
                checked_path.q_s[cell + 1],
                checked_path.q_ss[cell + 1],
            ),
        ):
            fitted_force = dynamics.generalized_force(float(x), float(u))
            force = _real_array(
                "inverse_dynamics final validation output",
                inverse_dynamics(
                    qpos.copy(),
                    q_s * np.sqrt(float(x)),
                    q_s * float(u) + q_ss * float(x),
                ),
                ndim=1,
            )
            if force.shape != (nv,):
                raise TorqueRetimeError(
                    "inverse_dynamics final validation output has wrong shape"
                )
            max_fitted_to_actual_force_residual = max(
                max_fitted_to_actual_force_residual,
                float(np.max(np.abs(force - fitted_force))),
            )
            if grounded_mode:
                assert grounded_contact_solver is not None
                solution = grounded_contact_solver.solve(
                    qpos,
                    q_s * np.sqrt(float(x)),
                    q_s * float(u) + q_ss * float(x),
                    actuated_dof_indices,
                    effort_lower,
                    effort_upper,
                    velocity,
                    expected_generalized_force=force,
                    path_tangent=q_s,
                )
                if not solution.feasible:
                    raise IncompleteTorqueCertificate(
                        "achieved sampled state has no feasible grounded "
                        "actuator/contact-force distribution",
                        missing=[
                            "LP-feasible actuator/contact force distribution "
                            "at every sampled state"
                        ],
                        details=solution.report,
                    )
                achieved_effort = solution.actuator_generalized_force
                achieved_force_tolerance = max(
                    force_validation_tolerance,
                    float(
                        solution.report.get(
                            "allowed_residual",
                            force_validation_tolerance,
                        )
                    ),
                )
                max_ground_equality_residual = max(
                    max_ground_equality_residual,
                    float(solution.equality_residual),
                )
                max_ground_root_residual = max(
                    max_ground_root_residual,
                    float(solution.root_residual),
                )
                grounded_lp_sample_count += 1
                normals = np.asarray(
                    solution.report["normal_force_per_foot_n"], np.float64
                )
                minimum_normal_force_per_foot = np.minimum(
                    minimum_normal_force_per_foot, normals
                )
                contact_kinematics = solution.report[
                    "contact_kinematics"
                ]
                for key in grounded_kinematic_maxima:
                    grounded_kinematic_maxima[key] = max(
                        grounded_kinematic_maxima[key],
                        float(contact_kinematics[key]),
                    )
                contact_geometry = solution.report["contact_geometry"]
                grounded_min_friction = min(
                    grounded_min_friction,
                    float(contact_geometry["friction_used"]),
                )
                for foot_row in contact_geometry["feet"]:
                    foot_index = int(foot_row["foot_index"])
                    grounded_min_support_vertices[foot_index] = min(
                        grounded_min_support_vertices[foot_index],
                        float(foot_row["support_vertices"]),
                    )
                    grounded_max_support_gap = max(
                        grounded_max_support_gap,
                        max(
                            abs(float(value))
                            for value in foot_row[
                                "support_vertex_floor_distance_m"
                            ]
                        ),
                    )
                    grounded_max_penetration = max(
                        grounded_max_penetration,
                        max(
                            0.0,
                            -float(
                                foot_row["minimum_floor_distance_m"]
                            ),
                        ),
                    )
                grounded_geometry_digests.add(
                    hashlib.sha256(
                        json.dumps(
                            contact_geometry,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                )
                if grounded_solver_summary is None:
                    grounded_solver_summary = {
                        key: solution.report[key]
                        for key in (
                            "solver",
                            "solver_library",
                            "scipy_version",
                            "friction_pyramid",
                            "friction_coefficient",
                            "minimum_normal_force_per_foot_n",
                            "model_binding",
                            "inverse_dynamics",
                            "contact_geometry",
                            "contact_kinematics",
                        )
                    }
            else:
                achieved_effort = force
                achieved_force_tolerance = force_validation_tolerance
            max_applied_force_validation_tolerance = max(
                max_applied_force_validation_tolerance,
                achieved_force_tolerance,
            )
            below = achieved_effort - effort_lower
            above = effort_upper - achieved_effort
            if np.any(below < -achieved_force_tolerance) or np.any(
                above < -achieved_force_tolerance
            ):
                if grounded_mode:
                    raise IncompleteTorqueCertificate(
                        "final grounded actuator-force validation failed",
                        missing=["actuator effort within bound limits"],
                    )
                raise TorqueRetimeError("final actuator-force validation failed")
            minimum_margin = min(
                minimum_margin, float(np.min(below)), float(np.min(above))
            )
            center = 0.5 * (effort_lower + effort_upper)
            half_width = 0.5 * (effort_upper - effort_lower)
            worst_ratio = max(
                worst_ratio,
                float(np.max(np.abs(achieved_effort - center) / half_width)),
            )

    window_accelerations: list[float] = []
    if window is not None:
        start, end = window
        cell_midpoint = 0.5 * (
            checked_path.path_position[:-1] + checked_path.path_position[1:]
        )
        inside = (cell_midpoint >= start) & (cell_midpoint <= end)
        window_accelerations = path_acceleration[inside].tolist()

    report = {
        "status": (
            "CANDIDATE_GROUNDED_DOUBLE_SUPPORT_SAMPLED_LP"
            if grounded_mode
            else "CANDIDATE_FIXED_BASE_THREE_POINT_COLLOCATION"
        ),
        "algorithm": (
            "scalar_x_u_forward_backward_grounded_contact_lp"
            if grounded_mode
            else "scalar_x_u_forward_backward_direct_drive"
        ),
        "state": "x=path_speed_squared",
        "control": "u=scalar_path_acceleration",
        "dynamics_equation": (
            "tau=A*u+c0+c_sqrt*sqrt(x)+c_x*x; "
            "qvel=q_s*sqrt(x); qacc=q_s*u+q_ss*x"
        ),
        "inverse_dynamics_semantics": str(
            getattr(
                inverse_dynamics,
                "semantics",
                "caller_supplied_must_be_unconstrained_and_frictionloss_disabled",
            )
        ),
        "plant_binding_status": plant_binding_status,
        "model_binding": (
            str(contract_binding)
            if plant_binding_status.startswith("BOUND_MATCH")
            else None
        ),
        "uniform_actuator_torque_enforced": False,
        "uniform_torque_explanation": (
            "u is one scalar path acceleration; actuator torque is inverse dynamics "
            "and generally differs by joint and varies with q, x, gravity, damping, "
            "curvature, gear, and friction reserve"
        ),
        "frictionloss_policy": (
            "disabled_in_inverse_dynamics_and_full_absolute_value_deducted_from_both_limits"
        ),
        "support_mode": support_mode,
        "contact_certificate": (
            "SAMPLED_LEFT_MIDPOINT_RIGHT_EACH_CELL_ONLY"
            if grounded_mode
            else False
        ),
        "balance_certificate": (
            "SAMPLED_FLOATING_BASE_EQUILIBRIUM_ONLY"
            if grounded_mode
            else False
        ),
        "continuous_cell_certificate": False,
        "path_derivative_provenance": (
            (
                "independently_checked_with_MuJoCo_differentiatePos_at_"
                "interleaved_nodes_and_midpoints"
                if grounded_mode
                else "caller_supplied_q_s_q_ss_not_independently_differentiated"
            )
        ),
        "path_derivative_validation": grounded_path_derivative_report,
        "collocation": "left_midpoint_right_each_cell",
        "nodes": int(len(checked_path.path_position)),
        "cells": int(len(path_acceleration)),
        "duration_s": float(time_s[-1]),
        "start_speed": float(speed[0]),
        "end_speed": float(speed[-1]),
        "forward_backward_sweeps": int(sweeps),
        "search_samples_per_neighbor": int(search_samples),
        "max_identification_probe_residual": float(
            max(
                item.max_probe_residual
                for item in [*node_dynamics, *midpoint_dynamics]
            )
        ),
        "max_fitted_to_actual_force_residual": float(
            max_fitted_to_actual_force_residual
        ),
        "max_effort_interval_ratio": float(worst_ratio),
        "minimum_effort_margin": float(minimum_margin),
        "force_validation_tolerance": float(
            max_applied_force_validation_tolerance
        ),
        "force_contract": force_report,
        "grounded_contact_lp": (
            None
            if not grounded_mode
            else {
                "sampled_lp_solves_in_final_validation": int(
                    grounded_lp_sample_count
                ),
                "max_equality_residual": float(
                    max_ground_equality_residual
                ),
                "max_unactuated_root_residual": float(
                    max_ground_root_residual
                ),
                "minimum_normal_force_per_foot_n": (
                    minimum_normal_force_per_foot.tolist()
                ),
                "contact_kinematic_maxima": grounded_kinematic_maxima,
                "minimum_friction_coefficient": float(
                    grounded_min_friction
                ),
                "minimum_support_vertices_per_foot": (
                    grounded_min_support_vertices.astype(int).tolist()
                ),
                "maximum_support_vertex_abs_floor_gap_m": float(
                    grounded_max_support_gap
                ),
                "maximum_foot_penetration_m": float(
                    grounded_max_penetration
                ),
                "unique_contact_geometry_count": len(
                    grounded_geometry_digests
                ),
                "unique_contact_geometry_sha256": sorted(
                    grounded_geometry_digests
                ),
                "solver_and_geometry": grounded_solver_summary,
                "claim_boundary": (
                    "sampled path feasibility; not continuous global TOPP-RA"
                ),
            }
        ),
        "window": None if window is None else [float(window[0]), float(window[1])],
        "window_policy": (
            "u_nonnegative_from_path_start_through_window_end"
            if no_early_brake
            else "ordinary_torque_feasible_candidate_braking_allowed"
        ),
        "window_path_acceleration": window_accelerations,
        "limitations": [
            "fixed geometric path; no path-shape optimization",
            "three-point cell collocation is not a continuous-cell proof",
            *(
                []
                if grounded_mode
                else [
                    "caller-supplied q_s/q_ss are not independently verified against qpos"
                ]
            ),
            "finite neighbor search is feasible-candidate construction, not a "
            "global minimum-time proof",
            "velocity and effort inputs are not independently bound to an URDF SHA",
            "scalar path acceleration is not uniform joint acceleration or "
            "uniform actuator torque",
            *(
                [
                    "q_s/q_ss are finite-difference checked only at interleaved "
                    "nodes/midpoints, not proven continuously",
                    "ground contact, friction pyramid, CoP support, and "
                    "floating-base balance are checked only at sampled "
                    "left/midpoint/right states",
                    "no continuous-contact TOPP-RA, contact-mode transition, "
                    "closed-loop balance, collision-free interval, or hardware certificate",
                ]
                if grounded_mode
                else [
                    "fixed-base direct-drive only",
                    "no ground-contact, friction-cone, CoP, balance, collision, "
                    "or hardware certificate",
                ]
            ),
        ],
    }
    return TorqueRetimeResult(
        path_position=checked_path.path_position.copy(),
        speed_squared=speed_sq,
        path_speed=speed,
        path_acceleration=path_acceleration,
        time_s=time_s,
        report=report,
    )


def _mujoco_model_binding(model: Any) -> str:
    """Digest dynamics and direct-actuator arrays shared by wrapper/contract."""

    digest = hashlib.sha256()
    for scalar_name in (
        "nq",
        "nv",
        "nu",
        "njnt",
        "nbody",
        "neq",
        "ntendon",
        "nflex",
        "nplugin",
        "npair",
        "nexclude",
    ):
        if not hasattr(model, scalar_name):
            continue
        digest.update(scalar_name.encode("ascii"))
        digest.update(np.asarray([int(getattr(model, scalar_name))], np.int64).tobytes())
    for array_name in (
        "body_parentid",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",
        "body_gravcomp",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_bodyid",
        "jnt_pos",
        "jnt_axis",
        "jnt_stiffness",
        "jnt_actgravcomp",
        "qpos_spring",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcelimited",
        "actuator_forcerange",
        "actuator_ctrllimited",
        "actuator_ctrlrange",
        "jnt_actfrclimited",
        "jnt_actfrcrange",
        "jnt_limited",
        "jnt_range",
        "geom_bodyid",
        "geom_type",
        "geom_dataid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_priority",
        "geom_friction",
        "geom_margin",
        "geom_gap",
        "geom_solmix",
        "geom_solref",
        "geom_solimp",
        "geom_fluid",
        "mesh_vertadr",
        "mesh_vertnum",
        "mesh_vert",
        "mesh_faceadr",
        "mesh_facenum",
        "mesh_face",
        "mesh_scale",
        "mesh_pos",
        "mesh_quat",
        "mesh_graphadr",
        "mesh_graph",
        "pair_dim",
        "pair_geom1",
        "pair_geom2",
        "pair_signature",
        "pair_solref",
        "pair_solreffriction",
        "pair_solimp",
        "pair_margin",
        "pair_gap",
        "pair_friction",
        "exclude_signature",
        "site_bodyid",
        "site_pos",
        "site_quat",
        "name_bodyadr",
        "name_jntadr",
        "name_geomadr",
        "name_siteadr",
        "name_actuatoradr",
        "names",
    ):
        if not hasattr(model, array_name):
            continue
        value = np.ascontiguousarray(np.asarray(getattr(model, array_name)))
        digest.update(array_name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    for option_name in ("gravity", "wind"):
        if hasattr(model.opt, option_name):
            value = np.ascontiguousarray(
                np.asarray(getattr(model.opt, option_name), np.float64)
            )
            digest.update(option_name.encode("ascii"))
            digest.update(value.tobytes())
    for option_name in ("density", "viscosity", "magnetic"):
        if hasattr(model.opt, option_name):
            value = np.ascontiguousarray(
                np.asarray(getattr(model.opt, option_name), np.float64)
            )
            digest.update(option_name.encode("ascii"))
            digest.update(value.tobytes())
    for option_name in (
        "timestep",
        "apirate",
        "impratio",
        "tolerance",
        "ls_tolerance",
        "noslip_tolerance",
        "ccd_tolerance",
    ):
        if hasattr(model.opt, option_name):
            digest.update(option_name.encode("ascii"))
            digest.update(
                np.asarray(
                    [float(getattr(model.opt, option_name))], np.float64
                ).tobytes()
            )
    for option_name in (
        "integrator",
        "cone",
        "jacobian",
        "solver",
        "iterations",
        "ls_iterations",
        "noslip_iterations",
        "ccd_iterations",
        "disableflags",
        "enableflags",
    ):
        if hasattr(model.opt, option_name):
            digest.update(option_name.encode("ascii"))
            digest.update(
                np.asarray(
                    [int(getattr(model.opt, option_name))], np.int64
                ).tobytes()
            )
    return digest.hexdigest()


def validate_mujoco_tangent_path_grid(
    path: TangentPathGrid,
    model: Any,
    *,
    tangent_absolute_tolerance: float,
    tangent_relative_tolerance: float,
    curvature_absolute_tolerance: float,
    curvature_relative_tolerance: float,
) -> dict:
    """Independently finite-difference ``qpos`` with MuJoCo tangent APIs.

    Nodes and supplied cell midpoints are interleaved.  At every interior
    sample, backward and forward configuration differences are both expressed
    in the *current* MuJoCo tangent frame, then combined into second-order
    nonuniform-grid estimates of ``q_s`` and ``q_ss``.  This matters for a free
    joint: quaternion columns are never treated as linear coordinates.

    The test is a sampled finite-difference consistency gate, not a symbolic
    proof between samples.  Endpoints are excluded because a one-sided
    derivative would mix first-order truncation error into the physical gate.
    """

    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover - pod-dependent
        raise IncompleteTorqueCertificate(
            "mujoco is unavailable for tangent-path validation",
            missing=["mujoco.mj_differentiatePos", "mujoco.mj_integratePos"],
        ) from exc
    checked, nq, nv = _validate_path(path)
    if int(model.nq) != nq or int(model.nv) != nv:
        raise TorqueRetimeError(
            "tangent-path grid dimensions do not match the bound MuJoCo model"
        )
    tolerances = {
        "tangent_absolute_tolerance": float(tangent_absolute_tolerance),
        "tangent_relative_tolerance": float(tangent_relative_tolerance),
        "curvature_absolute_tolerance": float(curvature_absolute_tolerance),
        "curvature_relative_tolerance": float(curvature_relative_tolerance),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances.values()):
        raise TorqueRetimeError(
            "tangent-path finite-difference tolerances must be finite and non-negative"
        )

    samples = 2 * len(checked.path_position) - 1
    combined_s = np.empty(samples, np.float64)
    combined_qpos = np.empty((samples, nq), np.float64)
    combined_q_s = np.empty((samples, nv), np.float64)
    combined_q_ss = np.empty((samples, nv), np.float64)
    combined_s[0::2] = checked.path_position
    combined_s[1::2] = 0.5 * (
        checked.path_position[:-1] + checked.path_position[1:]
    )
    combined_qpos[0::2] = checked.qpos
    combined_qpos[1::2] = checked.midpoint_qpos
    combined_q_s[0::2] = checked.q_s
    combined_q_s[1::2] = checked.midpoint_q_s
    combined_q_ss[0::2] = checked.q_ss
    combined_q_ss[1::2] = checked.midpoint_q_ss

    max_tangent_residual = 0.0
    max_curvature_residual = 0.0
    max_roundtrip_residual = 0.0
    worst_tangent: Optional[dict] = None
    worst_curvature: Optional[dict] = None
    failures: list[dict] = []
    for sample in range(1, samples - 1):
        h_backward = float(combined_s[sample] - combined_s[sample - 1])
        h_forward = float(combined_s[sample + 1] - combined_s[sample])
        current = combined_qpos[sample]
        backward_raw = np.empty(nv, np.float64)
        forward = np.empty(nv, np.float64)
        mujoco.mj_differentiatePos(
            model,
            backward_raw,
            h_backward,
            current,
            combined_qpos[sample - 1],
        )
        backward = -backward_raw
        mujoco.mj_differentiatePos(
            model,
            forward,
            h_forward,
            current,
            combined_qpos[sample + 1],
        )
        numeric_tangent = (
            h_forward * backward + h_backward * forward
        ) / (h_backward + h_forward)
        numeric_curvature = (
            2.0 * (forward - backward) / (h_backward + h_forward)
        )

        for direction, step, target in (
            (forward, h_forward, combined_qpos[sample + 1]),
            (-backward, h_backward, combined_qpos[sample - 1]),
        ):
            reconstructed = current.copy()
            mujoco.mj_integratePos(model, reconstructed, direction, step)
            roundtrip = np.empty(nv, np.float64)
            mujoco.mj_differentiatePos(
                model, roundtrip, 1.0, reconstructed, target
            )
            max_roundtrip_residual = max(
                max_roundtrip_residual,
                float(np.max(np.abs(roundtrip))),
            )

        tangent_error = np.abs(combined_q_s[sample] - numeric_tangent)
        tangent_allowed = float(tangent_absolute_tolerance) + float(
            tangent_relative_tolerance
        ) * np.maximum(
            np.abs(combined_q_s[sample]), np.abs(numeric_tangent)
        )
        curvature_error = np.abs(
            combined_q_ss[sample] - numeric_curvature
        )
        curvature_allowed = float(curvature_absolute_tolerance) + float(
            curvature_relative_tolerance
        ) * np.maximum(
            np.abs(combined_q_ss[sample]), np.abs(numeric_curvature)
        )
        tangent_dof = int(np.argmax(tangent_error))
        curvature_dof = int(np.argmax(curvature_error))
        tangent_peak = float(tangent_error[tangent_dof])
        curvature_peak = float(curvature_error[curvature_dof])
        if tangent_peak >= max_tangent_residual:
            max_tangent_residual = tangent_peak
            worst_tangent = {
                "combined_sample": sample,
                "path_position": float(combined_s[sample]),
                "dof": tangent_dof,
                "provided": float(combined_q_s[sample, tangent_dof]),
                "finite_difference": float(numeric_tangent[tangent_dof]),
                "residual": tangent_peak,
                "allowed": float(tangent_allowed[tangent_dof]),
            }
        if curvature_peak >= max_curvature_residual:
            max_curvature_residual = curvature_peak
            worst_curvature = {
                "combined_sample": sample,
                "path_position": float(combined_s[sample]),
                "dof": curvature_dof,
                "provided": float(combined_q_ss[sample, curvature_dof]),
                "finite_difference": float(numeric_curvature[curvature_dof]),
                "residual": curvature_peak,
                "allowed": float(curvature_allowed[curvature_dof]),
            }
        bad_tangent = np.flatnonzero(tangent_error > tangent_allowed)
        bad_curvature = np.flatnonzero(curvature_error > curvature_allowed)
        if len(bad_tangent) or len(bad_curvature):
            failures.append(
                {
                    "combined_sample": sample,
                    "path_position": float(combined_s[sample]),
                    "tangent_bad_dofs": bad_tangent.tolist(),
                    "curvature_bad_dofs": bad_curvature.tolist(),
                }
            )
    roundtrip_allowed = 2.0e-9
    report = {
        "status": (
            "PASS_SAMPLED_MUJOCO_FINITE_DIFFERENCE"
            if not failures and max_roundtrip_residual <= roundtrip_allowed
            else "INCOMPLETE_FAIL_CLOSED"
        ),
        "method": (
            "interleaved_nodes_midpoints_current_frame_"
            "mj_differentiatePos_nonuniform_second_order"
        ),
        "samples_total": int(samples),
        "interior_samples_checked": int(samples - 2),
        "endpoints_checked": False,
        "free_joint_semantics": (
            "MuJoCo tangent nv via mj_differentiatePos/mj_integratePos; "
            "qpos quaternion never differenced linearly"
        ),
        "max_tangent_residual": max_tangent_residual,
        "max_curvature_residual": max_curvature_residual,
        "max_integrate_differentiate_roundtrip_residual": (
            max_roundtrip_residual
        ),
        "roundtrip_tolerance": roundtrip_allowed,
        "worst_tangent": worst_tangent,
        "worst_curvature": worst_curvature,
        "tolerances": tolerances,
        "failures": failures[:32],
        "claim_boundary": (
            "sampled finite-difference consistency; no continuous derivative proof"
        ),
    }
    if failures or max_roundtrip_residual > roundtrip_allowed:
        raise IncompleteTorqueCertificate(
            "q_s/q_ss are inconsistent with qpos under MuJoCo tangent semantics",
            missing=["MuJoCo-finite-difference-consistent tangent path"],
            details=report,
        )
    return report


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise convex hull without optional dependencies."""

    xy = _real_array("support polygon points", points, ndim=2)
    if xy.shape[1] != 2 or len(xy) < 3:
        raise IncompleteTorqueCertificate(
            "foot collision mesh does not expose a two-dimensional sole polygon",
            missing=["at least three non-collinear sole vertices"],
        )
    unique = np.unique(xy, axis=0)
    if len(unique) < 3:
        raise IncompleteTorqueCertificate(
            "foot sole vertices collapse to fewer than three points",
            missing=["non-degenerate foot support polygon"],
        )

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1.0e-14:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in unique[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1.0e-14:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], np.float64)
    if len(hull) < 3:
        raise IncompleteTorqueCertificate(
            "foot sole polygon is collinear",
            missing=["non-degenerate foot support polygon"],
        )
    twice_area = abs(
        float(
            np.dot(hull[:, 0], np.roll(hull[:, 1], -1))
            - np.dot(hull[:, 1], np.roll(hull[:, 0], -1))
        )
    )
    if twice_area <= 1.0e-8:
        raise IncompleteTorqueCertificate(
            "foot sole support polygon area is numerically zero",
            missing=["finite-area foot support polygon"],
            details={"twice_area_m2": twice_area},
        )
    return hull


def _conservative_polygon_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    """Select existing hull vertices; the resulting support set only shrinks."""

    if len(points) <= maximum:
        return points.copy()
    indices = np.floor(
        np.arange(maximum, dtype=np.float64) * len(points) / maximum
    ).astype(np.int64)
    return points[indices].copy()


def _points_inside_convex_polygon(
    polygon: np.ndarray,
    points: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    """Closed-set containment for a counter-clockwise convex polygon."""

    for point in points:
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge_length = float(np.linalg.norm(end - start))
            if edge_length <= 1.0e-12:
                return False
            cross = float(
                (end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0])
            )
            signed_distance = cross / edge_length
            if signed_distance < -float(tolerance):
                return False
    return True


def _solve_ground_contact_force_lp(
    generalized_force: np.ndarray,
    generalized_force_map: np.ndarray,
    point_foot_index: np.ndarray,
    actuated_dof_indices: np.ndarray,
    effort_lower: np.ndarray,
    effort_upper: np.ndarray,
    *,
    friction_coefficient: float,
    minimum_normal_force_per_foot_n: float,
    residual_tolerance: float,
    solver_name: str,
    lp_objective: str = GROUND_LP_OBJECTIVE_FEASIBILITY,
    minimum_normal_force_per_contact_n: float = 0.0,
) -> GroundContactLPSolution:
    """Solve ``M qdd+h = S' tau + J' f`` at one sampled state.

    The default ``feasibility`` objective deliberately retains the historical
    zero-cost LP exactly.  ``hold_minimax_normalized_available_torque`` is an
    explicit opt-in for selecting a low-utilization static-hold decomposition
    from the same physical feasible set.
    """

    if lp_objective not in (
        GROUND_LP_OBJECTIVE_FEASIBILITY,
        GROUND_LP_OBJECTIVE_HOLD_MINIMAX,
    ):
        raise TorqueRetimeError(
            "ground LP objective must be 'feasibility' or "
            "'hold_minimax_normalized_available_torque'"
        )
    if solver_name != "scipy.optimize.linprog:highs":
        raise IncompleteTorqueCertificate(
            "the requested fixed CPU LP solver is not the audited solver",
            missing=["scipy.optimize.linprog:highs"],
            details={"requested_solver": solver_name},
        )
    try:
        import scipy  # type: ignore
        from scipy.optimize import linprog  # type: ignore
    except ImportError as exc:
        raise IncompleteTorqueCertificate(
            "SciPy HiGHS LP solver is unavailable",
            missing=["scipy", "scipy.optimize.linprog(method='highs')"],
        ) from exc

    demand = _real_array("generalized_force", generalized_force, ndim=1)
    force_map = _real_array(
        "ground generalized_force_map", generalized_force_map, ndim=2
    )
    nv = len(demand)
    if force_map.shape[0] != nv or force_map.shape[1] % 3:
        raise TorqueRetimeError(
            "ground generalized_force_map must have shape (nv, 3*points)"
        )
    points = force_map.shape[1] // 3
    foot_index = np.asarray(point_foot_index)
    if (
        foot_index.shape != (points,)
        or not np.issubdtype(foot_index.dtype, np.integer)
        or set(foot_index.astype(int).tolist()) != {0, 1}
    ):
        raise TorqueRetimeError(
            "point_foot_index must assign at least one support point to each foot"
        )
    actuated = np.asarray(actuated_dof_indices)
    if (
        actuated.ndim != 1
        or not np.issubdtype(actuated.dtype, np.integer)
        or len(np.unique(actuated)) != len(actuated)
        or np.any(actuated < 0)
        or np.any(actuated >= nv)
    ):
        raise TorqueRetimeError("actuated_dof_indices is not a valid unique index set")
    tau_lo = _real_array("effort_lower", effort_lower, ndim=1)
    tau_hi = _real_array("effort_upper", effort_upper, ndim=1)
    if tau_lo.shape != (len(actuated),) or tau_hi.shape != tau_lo.shape:
        raise TorqueRetimeError("ground effort limits disagree with actuated DoFs")
    if np.any(tau_lo >= tau_hi):
        raise TorqueRetimeError("ground effort intervals are empty")
    mu = float(friction_coefficient)
    if not math.isfinite(mu) or mu <= 0.0:
        raise TorqueRetimeError("friction_coefficient must be finite and positive")
    minimum_contact_normal = float(minimum_normal_force_per_contact_n)
    if not math.isfinite(minimum_contact_normal) or minimum_contact_normal < 0.0:
        raise TorqueRetimeError(
            "minimum_normal_force_per_contact_n must be finite and non-negative"
        )

    na = len(actuated)
    hold_minimax = lp_objective == GROUND_LP_OBJECTIVE_HOLD_MINIMAX
    variables = na + 3 * points + int(hold_minimax)
    minimax_index = variables - 1 if hold_minimax else None
    equality = np.zeros((nv, variables), np.float64)
    equality[actuated, np.arange(na)] = 1.0
    equality[:, na : na + 3 * points] = force_map

    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    # |ft1| + |ft2| <= mu*fn is a diamond strictly inside the circular
    # Coulomb cone.  It is conservative in every tangential direction.
    for point in range(points):
        base = na + 3 * point
        for sign_1, sign_2 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            row = np.zeros(variables, np.float64)
            row[base] = sign_1
            row[base + 1] = sign_2
            row[base + 2] = -mu
            inequalities.append(row)
            inequality_rhs.append(0.0)
    for foot in (0, 1):
        row = np.zeros(variables, np.float64)
        for point in np.flatnonzero(foot_index == foot):
            row[na + 3 * int(point) + 2] = -1.0
        inequalities.append(row)
        inequality_rhs.append(-float(minimum_normal_force_per_foot_n))

    if hold_minimax:
        assert minimax_index is not None
        if np.any(tau_lo >= 0.0) or np.any(tau_hi <= 0.0):
            raise TorqueRetimeError(
                "hold minimax effort intervals must strictly contain zero"
            )
        # Physical ready corresponds to zero hold torque.  Normalize each sign
        # by its own executable headroom because runtime-qdes projection can
        # produce asymmetric lower/upper torque availability.
        for actuator in range(na):
            positive = np.zeros(variables, np.float64)
            positive[actuator] = 1.0
            positive[minimax_index] = -float(tau_hi[actuator])
            inequalities.append(positive)
            inequality_rhs.append(0.0)

            negative = np.zeros(variables, np.float64)
            negative[actuator] = -1.0
            negative[minimax_index] = float(tau_lo[actuator])
            inequalities.append(negative)
            inequality_rhs.append(0.0)

    bounds: list[tuple[Optional[float], Optional[float]]] = [
        (float(lo), float(hi)) for lo, hi in zip(tau_lo, tau_hi)
    ]
    bounds.extend(
        [
            (None, None),
            (None, None),
            (minimum_contact_normal, None),
        ]
        * points
    )
    if hold_minimax:
        bounds.append((0.0, 1.0))
    objective = np.zeros(variables, np.float64)
    if hold_minimax:
        assert minimax_index is not None
        objective[minimax_index] = 1.0
    result = linprog(
        objective,
        A_ub=np.asarray(inequalities, np.float64),
        b_ub=np.asarray(inequality_rhs, np.float64),
        A_eq=equality,
        b_eq=demand,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    base_report = {
        "solver": solver_name,
        "solver_library": "scipy",
        "scipy_version": str(scipy.__version__),
        "highs_status": int(result.status),
        "highs_message": str(result.message),
        "highs_iterations": int(getattr(result, "nit", 0)),
        "variables": int(variables),
        "equalities": int(nv),
        "inequalities": int(len(inequalities)),
        "friction_pyramid": "abs(ft1)+abs(ft2)<=mu*fn",
        "friction_coefficient": mu,
        "minimum_normal_force_per_foot_n": float(
            minimum_normal_force_per_foot_n
        ),
        "minimum_normal_force_per_contact_n": minimum_contact_normal,
    }
    if hold_minimax:
        base_report.update(
            {
                "objective_mode": lp_objective,
                "lp_objective": (
                    "MINIMIZE_MAX_NORMALIZED_AVAILABLE_HOLD_TORQUE"
                ),
                "lp_objective_expression": (
                    "min rho subject to tau<=rho*tau_hi and "
                    "-tau<=rho*(-tau_lo)"
                ),
                "objective_effort_lower": tau_lo.tolist(),
                "objective_effort_upper": tau_hi.tolist(),
                "negative_available_hold_torque": (-tau_lo).tolist(),
                "positive_available_hold_torque": tau_hi.tolist(),
            }
        )
    if int(result.status) == 2:
        return GroundContactLPSolution(
            feasible=False,
            actuator_generalized_force=np.empty(0, np.float64),
            point_force_floor=np.empty((0, 3), np.float64),
            equality_residual=np.inf,
            root_residual=np.inf,
            report={**base_report, "status": "INFEASIBLE"},
        )
    if not bool(result.success) or int(result.status) != 0 or result.x is None:
        raise IncompleteTorqueCertificate(
            "ground contact LP did not converge to an audited optimum",
            missing=["converged scipy HiGHS feasibility solve"],
            details=base_report,
        )

    vector = _real_array("linprog solution", result.x, ndim=1)
    if vector.shape != (variables,):
        raise IncompleteTorqueCertificate(
            "ground contact LP returned a malformed solution",
            missing=["complete LP primal vector"],
            details=base_report,
        )
    equality_error = equality @ vector - demand
    scale = max(1.0, float(np.max(np.abs(demand))))
    allowed = float(residual_tolerance) * scale
    residual = float(np.max(np.abs(equality_error)))
    free_rows = np.setdiff1d(np.arange(nv), actuated, assume_unique=True)
    root_residual = (
        0.0
        if len(free_rows) == 0
        else float(np.max(np.abs(equality_error[free_rows])))
    )
    inequality_error = (
        np.asarray(inequalities, np.float64) @ vector
        - np.asarray(inequality_rhs, np.float64)
    )
    max_inequality_violation = max(0.0, float(np.max(inequality_error)))
    tau = vector[:na]
    forces = vector[na : na + 3 * points].reshape(points, 3)
    bound_violation = max(
        0.0,
        float(np.max(tau_lo - tau)),
        float(np.max(tau - tau_hi)),
        float(np.max(-forces[:, 2])),
    )
    if (
        residual > allowed
        or root_residual > allowed
        or max_inequality_violation > allowed
        or bound_violation > allowed
    ):
        raise IncompleteTorqueCertificate(
            "ground contact LP primal residual exceeds the fail-closed tolerance",
            missing=["numerically balanced floating-root/contact solution"],
            details={
                **base_report,
                "equality_residual": residual,
                "root_residual": root_residual,
                "inequality_violation": max_inequality_violation,
                "bound_violation": bound_violation,
                "allowed_residual": allowed,
            },
        )
    half_width = 0.5 * (tau_hi - tau_lo)
    center = 0.5 * (tau_hi + tau_lo)
    effort_ratio = float(np.max(np.abs(tau - center) / half_width))
    objective_report: dict = {}
    if hold_minimax:
        assert minimax_index is not None
        minimax_value = float(vector[minimax_index])
        normalized_hold_torque = float(
            np.max(
                np.maximum(
                    np.maximum(tau, 0.0) / tau_hi,
                    np.maximum(-tau, 0.0) / (-tau_lo),
                )
            )
        )
        normalized_allowed = max(
            float(residual_tolerance),
            allowed
            / float(np.min(np.minimum(tau_hi, -tau_lo))),
        )
        if (
            not math.isfinite(minimax_value)
            or minimax_value < -normalized_allowed
            or minimax_value > 1.0 + normalized_allowed
            or normalized_hold_torque > minimax_value + normalized_allowed
        ):
            raise IncompleteTorqueCertificate(
                "ground contact minimax LP returned an invalid objective witness",
                missing=["valid normalized available-hold-torque minimax witness"],
                details={
                    **base_report,
                    "minimax_objective_value": minimax_value,
                    "max_effort_interval_ratio": effort_ratio,
                    "max_normalized_available_hold_torque": (
                        normalized_hold_torque
                    ),
                    "allowed_residual": allowed,
                    "normalized_allowed_residual": normalized_allowed,
                },
            )
        objective_report = {
            "minimax_objective_value": minimax_value,
            "minimax_objective_solver_fun": float(result.fun),
            "max_normalized_available_hold_torque": (
                normalized_hold_torque
            ),
            "optimum_max_normalized_available_hold_torque": minimax_value,
            "minimax_normalized_allowed_residual": normalized_allowed,
        }
    return GroundContactLPSolution(
        feasible=True,
        actuator_generalized_force=tau.copy(),
        point_force_floor=forces.copy(),
        equality_residual=residual,
        root_residual=root_residual,
        report={
            **base_report,
            "status": "FEASIBLE_SAMPLED_STATE",
            "allowed_residual": allowed,
            "equality_residual": residual,
            "root_residual": root_residual,
            "max_inequality_violation": max_inequality_violation,
            "bound_violation": bound_violation,
            "max_effort_interval_ratio": effort_ratio,
            **objective_report,
        },
    )


class MujocoGroundContactLPSolver:
    """Exact-model A3 double-support inverse-dynamics feasibility solver.

    The class is intentionally stateful and not thread-safe: each instance owns
    isolated MuJoCo ``MjData`` objects and a small qpos geometry cache.  Create
    one instance per retiming worker/process.
    """

    def __init__(self, model: Any, config: GroundContactConfig) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError as exc:  # pragma: no cover - pod-dependent
            raise IncompleteTorqueCertificate(
                "mujoco is unavailable for grounded contact feasibility",
                missing=["mujoco"],
            ) from exc
        self.config = config
        self.model_binding = _mujoco_model_binding(model)
        if self.model_binding != config.expected_model_binding:
            raise IncompleteTorqueCertificate(
                "ground solver model binding does not match the explicitly expected model",
                missing=["exact expected MuJoCo model/hash"],
                details={
                    "expected_model_binding": config.expected_model_binding,
                    "actual_model_binding": self.model_binding,
                },
            )
        source_path = Path(config.model_source_path).expanduser().resolve()
        if not source_path.is_file():
            raise IncompleteTorqueCertificate(
                "ground solver model source path does not exist",
                missing=["pinned MJCF source file"],
                details={"model_source_path": str(source_path)},
            )
        source_digest = hashlib.sha256()
        with source_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                source_digest.update(chunk)
        self.model_source_sha256 = source_digest.hexdigest()
        self.model_source_path = str(source_path)
        if self.model_source_sha256 != config.expected_source_sha256:
            raise IncompleteTorqueCertificate(
                "ground solver MJCF source SHA does not match the pinned receipt",
                missing=["exact expected MJCF source SHA-256"],
                details={
                    "model_source_path": self.model_source_path,
                    "expected_source_sha256": config.expected_source_sha256,
                    "actual_source_sha256": self.model_source_sha256,
                },
            )
        try:
            contact_model = copy.copy(model)
            inverse_model = copy.copy(model)
        except Exception as exc:  # pragma: no cover - binding-specific
            raise IncompleteTorqueCertificate(
                "could not isolate MuJoCo models for contact and inverse dynamics",
                missing=["independent MuJoCo model copies"],
            ) from exc
        if contact_model is model or inverse_model is model or contact_model is inverse_model:
            raise IncompleteTorqueCertificate(
                "MuJoCo model copies unexpectedly alias",
                missing=["independent contact/inverse model state"],
            )
        if _mujoco_model_binding(contact_model) != self.model_binding:
            raise IncompleteTorqueCertificate(
                "copied contact model changed the bound plant",
                missing=["bit-equivalent MuJoCo model copy"],
            )
        if np.shares_memory(
            np.asarray(inverse_model.dof_frictionloss),
            np.asarray(contact_model.dof_frictionloss),
        ):
            raise IncompleteTorqueCertificate(
                "inverse/contact model copies share frictionloss storage",
                missing=["isolated MuJoCo inverse-dynamics model"],
            )
        if (
            int(contact_model.nq) != 38
            or int(contact_model.nv) != 37
            or int(contact_model.nu) != 31
        ):
            raise IncompleteTorqueCertificate(
                "grounded solver is restricted to the exact 38/37/31 A3 plant",
                missing=["A3 nq=38 nv=37 nu=31 model"],
                details={
                    "nq": int(contact_model.nq),
                    "nv": int(contact_model.nv),
                    "nu": int(contact_model.nu),
                },
            )
        unsupported = {
            name: int(getattr(contact_model, name, 0))
            for name in ("neq", "ntendon", "nflex", "nplugin", "npair")
            if int(getattr(contact_model, name, 0)) != 0
        }
        if unsupported:
            raise IncompleteTorqueCertificate(
                "grounded solver does not model equality/tendon/flex/plugin forces",
                missing=["plain A3 rigid-body contact plant"],
                details={"unsupported_counts": unsupported},
            )
        free_ids = np.flatnonzero(
            np.asarray(contact_model.jnt_type)
            == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if free_ids.tolist() != [0]:
            raise IncompleteTorqueCertificate(
                "A3 model must have exactly free joint 0",
                missing=["six-DoF root at joint 0"],
                details={"free_joint_ids": free_ids.tolist()},
            )
        if (
            int(contact_model.jnt_qposadr[0]) != 0
            or int(contact_model.jnt_dofadr[0]) != 0
        ):
            raise IncompleteTorqueCertificate(
                "A3 free root must occupy qpos[0:7] and tangent rows [0:6]",
                missing=["canonical MuJoCo free-root addressing"],
            )

        floor_geom = int(
            mujoco.mj_name2id(
                contact_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                config.floor_geom_name,
            )
        )
        if floor_geom < 0:
            raise IncompleteTorqueCertificate(
                "named floor geom is absent",
                missing=[f"floor geom {config.floor_geom_name!r}"],
            )
        if (
            int(contact_model.geom_bodyid[floor_geom]) != 0
            or int(contact_model.geom_type[floor_geom])
            != int(mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            raise IncompleteTorqueCertificate(
                "floor geom must be a world-attached MuJoCo plane",
                missing=["world plane floor geometry"],
            )
        foot_bodies: list[int] = []
        foot_geoms: list[tuple[int, ...]] = []
        for body_name in config.foot_body_names:
            body = int(
                mujoco.mj_name2id(
                    contact_model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
            )
            if body < 0:
                raise IncompleteTorqueCertificate(
                    "named foot body is absent",
                    missing=[f"foot body {body_name!r}"],
                )
            geoms = tuple(
                geom
                for geom in range(int(contact_model.ngeom))
                if int(contact_model.geom_bodyid[geom]) == body
                and int(contact_model.geom_contype[geom]) != 0
                and int(contact_model.geom_conaffinity[geom]) != 0
            )
            if not geoms:
                raise IncompleteTorqueCertificate(
                    "foot body has no enabled collision geometry",
                    missing=[f"enabled collision geom on {body_name!r}"],
                )
            if any(
                int(contact_model.geom_type[geom])
                != int(mujoco.mjtGeom.mjGEOM_MESH)
                for geom in geoms
            ):
                raise IncompleteTorqueCertificate(
                    "A3 support-polygon extraction currently requires mesh foot collision geoms",
                    missing=["mesh foot collision geometry"],
                    details={"foot_body": body_name, "geom_ids": list(geoms)},
                )
            foot_bodies.append(body)
            foot_geoms.append(geoms)

        try:
            import scipy  # type: ignore
            from scipy.optimize import linprog as _linprog  # noqa: F401
        except ImportError as exc:
            raise IncompleteTorqueCertificate(
                "SciPy HiGHS is unavailable",
                missing=["scipy.optimize.linprog(method='highs')"],
            ) from exc
        self._mujoco = mujoco
        self._contact_model = contact_model
        self._inverse_model = inverse_model
        self._contact_data = mujoco.MjData(contact_model)
        self._inverse_data = mujoco.MjData(inverse_model)
        inverse_model.opt.disableflags = int(inverse_model.opt.disableflags) | int(
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        )
        inverse_model.dof_frictionloss[:] = 0.0
        if (
            _mujoco_model_binding(contact_model) != self.model_binding
            or _mujoco_model_binding(model) != self.model_binding
        ):
            raise IncompleteTorqueCertificate(
                "isolating inverse dynamics mutated the contact or source model",
                missing=["independent MuJoCo model storage"],
            )
        self._floor_geom = floor_geom
        self._foot_bodies = (foot_bodies[0], foot_bodies[1])
        self._foot_geoms = (foot_geoms[0], foot_geoms[1])
        self._all_robot_bodies = set(
            range(1, int(contact_model.nbody))
        )
        self._geometry_cache: Dict[bytes, _GroundGeometry] = {}
        self._geometry_cache_order: list[bytes] = []
        self._lp_cache: Dict[bytes, GroundContactLPSolution] = {}
        self._lp_cache_order: list[bytes] = []
        self.scipy_version = str(scipy.__version__)
        model_contract = direct_actuator_contract_from_mujoco(
            contact_model,
            support_mode="ground",
            contact_mode="double_support_floor",
            fixed_lp_solver=config.solver,
        )
        (
            model_effort_lower,
            model_effort_upper,
            model_actuated_dofs,
            _model_force_report,
        ) = _resolve_grounded_actuator_limits(
            model_contract, int(contact_model.nv)
        )
        self._actuated_dof_indices = model_actuated_dofs
        self._effort_lower = model_effort_lower
        self._effort_upper = model_effort_upper
        self.semantics = (
            "exact_bound_mujoco_constraint_disabled_smooth_net_demand_plus_"
            "double_support_point_force_lp"
        )

    def validate_path_grid(self, path: TangentPathGrid) -> dict:
        """Bind supplied q_s/q_ss to qpos with MuJoCo finite differences."""

        return validate_mujoco_tangent_path_grid(
            path,
            self._contact_model,
            tangent_absolute_tolerance=float(
                self.config.path_tangent_fd_absolute_tolerance
            ),
            tangent_relative_tolerance=float(
                self.config.path_tangent_fd_relative_tolerance
            ),
            curvature_absolute_tolerance=float(
                self.config.path_curvature_fd_absolute_tolerance
            ),
            curvature_relative_tolerance=float(
                self.config.path_curvature_fd_relative_tolerance
            ),
        )

    def _validate_qpos_limits(self, qpos: np.ndarray) -> None:
        model = self._contact_model
        mujoco = self._mujoco
        q = _real_array("ground qpos", qpos, ndim=1)
        if q.shape != (int(model.nq),):
            raise TorqueRetimeError("ground qpos shape does not match model.nq")
        quat_norm = float(np.linalg.norm(q[3:7]))
        if abs(quat_norm - 1.0) > 1.0e-8:
            raise IncompleteTorqueCertificate(
                "floating-root quaternion is not unit length",
                missing=["normalized MuJoCo free-joint quaternion"],
                details={"quaternion_norm": quat_norm},
            )
        tolerance = float(self.config.position_tolerance_rad)
        violations: list[dict] = []
        for joint in range(int(model.njnt)):
            if not bool(model.jnt_limited[joint]):
                continue
            kind = int(model.jnt_type[joint])
            if kind not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise IncompleteTorqueCertificate(
                    "limited ball/free joints are outside the audited A3 path",
                    missing=["hinge/slide-only post-root joints"],
                    details={"joint_id": joint, "joint_type": kind},
                )
            address = int(model.jnt_qposadr[joint])
            value = float(q[address])
            lower, upper = map(float, model.jnt_range[joint])
            if value < lower - tolerance or value > upper + tolerance:
                violations.append(
                    {
                        "joint_id": joint,
                        "qpos_address": address,
                        "value": value,
                        "range": [lower, upper],
                    }
                )
        if violations:
            raise IncompleteTorqueCertificate(
                "path configuration violates MuJoCo joint position limits",
                missing=["position-limit-feasible path"],
                details={"violations": violations[:16]},
            )

    def _mesh_vertices_world(self, geom: int) -> np.ndarray:
        model = self._contact_model
        data = self._contact_data
        mesh = int(model.geom_dataid[geom])
        if mesh < 0:
            raise IncompleteTorqueCertificate(
                "foot mesh geom has no compiled mesh data",
                missing=["compiled foot mesh vertices"],
                details={"geom_id": geom},
            )
        address = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        if count < 3:
            raise IncompleteTorqueCertificate(
                "foot mesh has too few vertices",
                missing=["compiled foot mesh vertices"],
                details={"geom_id": geom, "vertex_count": count},
            )
        local = np.asarray(
            model.mesh_vert[address : address + count], np.float64
        )
        rotation = np.asarray(data.geom_xmat[geom], np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom], np.float64)
        return local @ rotation.T + position

    def _prepare_geometry(self, qpos: np.ndarray) -> _GroundGeometry:
        q = _real_array("ground qpos", qpos, ndim=1)
        self._validate_qpos_limits(q)
        key = np.ascontiguousarray(q).tobytes()
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached

        model = self._contact_model
        data = self._contact_data
        mujoco = self._mujoco
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        floor_basis = np.asarray(
            data.geom_xmat[self._floor_geom], np.float64
        ).reshape(3, 3)
        floor_origin = np.asarray(
            data.geom_xpos[self._floor_geom], np.float64
        )
        normal = floor_basis[:, 2]
        gravity = np.asarray(model.opt.gravity, np.float64)
        gravity_norm = float(np.linalg.norm(gravity))
        if (
            gravity_norm <= _EPS
            or float(np.dot(normal, -gravity / gravity_norm)) < 0.999
        ):
            raise IncompleteTorqueCertificate(
                "floor normal is not aligned against model gravity",
                missing=["horizontal gravity-opposing support plane"],
                details={
                    "floor_normal_world": normal.tolist(),
                    "gravity_world": gravity.tolist(),
                },
            )

        active_geoms: list[set[int]] = [set(), set()]
        contact_rows: list[dict] = []
        active_contact_friction: list[float] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            distance = float(contact.dist)
            if distance > float(self.config.contact_gap_tolerance_m):
                continue
            geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
            pair = {geom_1, geom_2}
            matched_foot: Optional[int] = None
            if self._floor_geom in pair:
                robot_geom = geom_2 if geom_1 == self._floor_geom else geom_1
                for foot in (0, 1):
                    if robot_geom in self._foot_geoms[foot]:
                        matched_foot = foot
                        active_geoms[foot].add(robot_geom)
                        break
            if matched_foot is None:
                raise IncompleteTorqueCertificate(
                    "an active contact is not an audited foot-floor contact",
                    missing=["exclusive double-foot floor contact mode"],
                    details={
                        "contact_index": index,
                        "geom_pair": [geom_1, geom_2],
                        "distance_m": distance,
                    },
                )
            contact_dim = int(contact.dim)
            if contact_dim < 3:
                raise IncompleteTorqueCertificate(
                    "active foot-floor contact has no audited tangential friction dimensions",
                    missing=["contact.dim >= 3"],
                    details={
                        "contact_index": index,
                        "contact_dim": contact_dim,
                    },
                )
            tangential_friction = np.asarray(
                contact.friction[:2], np.float64
            )
            combined_friction = float(np.min(tangential_friction))
            if not math.isfinite(combined_friction) or combined_friction <= 0.0:
                raise IncompleteTorqueCertificate(
                    "active foot-floor contact has no positive combined sliding friction",
                    missing=["positive MuJoCo contact sliding friction"],
                    details={
                        "contact_index": index,
                        "combined_tangential_friction": (
                            tangential_friction.tolist()
                        ),
                    },
                )
            active_contact_friction.append(combined_friction)
            if distance < -float(self.config.penetration_tolerance_m):
                raise IncompleteTorqueCertificate(
                    "MuJoCo reports excessive foot-floor penetration",
                    missing=["non-penetrating double support"],
                    details={
                        "contact_index": index,
                        "foot_index": matched_foot,
                        "distance_m": distance,
                    },
                )
            contact_rows.append(
                {
                    "contact_index": index,
                    "foot_index": matched_foot,
                    "geom_pair": [geom_1, geom_2],
                    "distance_m": distance,
                    "position_world_m": np.asarray(
                        contact.pos, np.float64
                    ).tolist(),
                    "contact_dim": contact_dim,
                    "combined_tangential_friction": (
                        tangential_friction.tolist()
                    ),
                    "combined_sliding_friction": combined_friction,
                }
            )
        for foot in (0, 1):
            if not active_geoms[foot]:
                raise IncompleteTorqueCertificate(
                    "both named feet must have active MuJoCo floor contact",
                    missing=[f"active floor contact for {self.config.foot_body_names[foot]}"],
                    details={"active_contacts": contact_rows},
                )

        point_world: list[np.ndarray] = []
        point_foot: list[int] = []
        foot_reports: list[dict] = []
        friction_values = [
            float(model.geom_friction[self._floor_geom, 0]),
            *active_contact_friction,
        ]
        for foot in (0, 1):
            vertices = np.concatenate(
                [
                    self._mesh_vertices_world(geom)
                    for geom in sorted(active_geoms[foot])
                ],
                axis=0,
            )
            signed_distance = (vertices - floor_origin) @ normal
            minimum = float(np.min(signed_distance))
            if minimum < -float(self.config.penetration_tolerance_m):
                raise IncompleteTorqueCertificate(
                    "foot collision mesh penetrates the floor beyond tolerance",
                    missing=["non-penetrating foot collision geometry"],
                    details={
                        "foot": self.config.foot_body_names[foot],
                        "minimum_signed_distance_m": minimum,
                    },
                )
            if minimum > float(self.config.contact_gap_tolerance_m):
                raise IncompleteTorqueCertificate(
                    "active contact disagrees with foot collision-mesh floor gap",
                    missing=["geometrically grounded foot sole"],
                    details={
                        "foot": self.config.foot_body_names[foot],
                        "minimum_signed_distance_m": minimum,
                    },
                )
            sole_mask = (
                signed_distance
                <= minimum + float(self.config.support_surface_band_m)
            ) & (
                signed_distance
                <= float(self.config.contact_gap_tolerance_m)
            )
            sole = vertices[sole_mask]
            tangent_xy = np.column_stack(
                (
                    (sole - floor_origin) @ floor_basis[:, 0],
                    (sole - floor_origin) @ floor_basis[:, 1],
                )
            )
            sole_hull_xy = _convex_hull_2d(tangent_xy)
            sole_hull_xy = _conservative_polygon_subsample(
                sole_hull_xy, self.config.max_support_points_per_foot
            )
            active_contact_world = np.asarray(
                [
                    row["position_world_m"]
                    for row in contact_rows
                    if int(row["foot_index"]) == foot
                ],
                np.float64,
            )
            active_contact_xy = np.column_stack(
                (
                    (active_contact_world - floor_origin)
                    @ floor_basis[:, 0],
                    (active_contact_world - floor_origin)
                    @ floor_basis[:, 1],
                )
            )
            if not _points_inside_convex_polygon(
                sole_hull_xy,
                active_contact_xy,
                tolerance=float(
                    self.config.sole_hull_containment_tolerance_m
                ),
            ):
                raise IncompleteTorqueCertificate(
                    "active MuJoCo foot contacts escape the assumed rigid sole patch",
                    missing=["active-contact hull inside sole support hull"],
                    details={
                        "foot": self.config.foot_body_names[foot],
                        "active_contact_floor_xy_m": active_contact_xy.tolist(),
                        "sole_hull_floor_xy_m": sole_hull_xy.tolist(),
                    },
                )
            # The LP force variables live only at MuJoCo's actual discrete
            # contact positions.  Their convex hull is the sampled CoP domain;
            # the wider collision-mesh sole hull is validation evidence only.
            contact_hull_xy = _convex_hull_2d(active_contact_xy)
            selected: list[np.ndarray] = []
            selected_floor_distance: list[float] = []
            for xy in contact_hull_xy:
                nearest = int(
                    np.argmin(np.linalg.norm(active_contact_xy - xy, axis=1))
                )
                selected.append(active_contact_world[nearest].copy())
                selected_floor_distance.append(
                    float(
                        (active_contact_world[nearest] - floor_origin)
                        @ normal
                    )
                )
            if max(selected_floor_distance) > float(
                self.config.contact_gap_tolerance_m
            ) + 1.0e-12:
                raise IncompleteTorqueCertificate(
                    "support polygon contains a point outside the active contact gap",
                    missing=["contact-proximate sole support polygon"],
                    details={
                        "foot": self.config.foot_body_names[foot],
                        "support_floor_distance_m": selected_floor_distance,
                    },
                )
            start = len(point_world)
            point_world.extend(selected)
            point_foot.extend([foot] * len(selected))
            friction_values.extend(
                float(model.geom_friction[geom, 0])
                for geom in active_geoms[foot]
            )
            foot_reports.append(
                {
                    "foot_index": foot,
                    "foot_body": self.config.foot_body_names[foot],
                    "body_id": int(self._foot_bodies[foot]),
                    "active_geom_ids": sorted(active_geoms[foot]),
                    "minimum_floor_distance_m": minimum,
                    "support_vertices": int(len(selected)),
                    "support_polygon_floor_xy_m": contact_hull_xy.tolist(),
                    "collision_sole_hull_floor_xy_m": sole_hull_xy.tolist(),
                    "active_contact_floor_xy_m": active_contact_xy.tolist(),
                    "active_contact_hull_contained": True,
                    "active_contact_hull_containment_tolerance_m": float(
                        self.config.sole_hull_containment_tolerance_m
                    ),
                    "support_vertex_floor_distance_m": selected_floor_distance,
                    "support_point_range": [start, len(point_world)],
                }
            )

        points_array = np.asarray(point_world, np.float64)
        foot_array = np.asarray(point_foot, np.int64)
        generalized_map = np.empty(
            (int(model.nv), 3 * len(points_array)), np.float64
        )
        for point, (world, foot) in enumerate(zip(points_array, foot_array)):
            jacobian_position = np.empty((3, int(model.nv)), np.float64)
            jacobian_rotation = np.empty((3, int(model.nv)), np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                world,
                int(self._foot_bodies[int(foot)]),
            )
            mapped = jacobian_position.T @ floor_basis
            generalized_map[:, 3 * point : 3 * point + 3] = mapped
            # Independent MuJoCo force application cross-check catches a
            # transposed Jacobian, wrong world/local basis, or force sign.
            for axis in range(3):
                applied = np.zeros(int(model.nv), np.float64)
                mujoco.mj_applyFT(
                    model,
                    data,
                    floor_basis[:, axis],
                    np.zeros(3, np.float64),
                    world,
                    int(self._foot_bodies[int(foot)]),
                    applied,
                )
                mismatch = float(np.max(np.abs(applied - mapped[:, axis])))
                if mismatch > 2.0e-10 * max(
                    1.0,
                    float(np.max(np.abs(applied))),
                    float(np.max(np.abs(mapped[:, axis]))),
                ):
                    raise IncompleteTorqueCertificate(
                        "MuJoCo mj_jac and mj_applyFT disagree for a foot force",
                        missing=["verified J_transpose force mapping"],
                        details={
                            "point_index": point,
                            "axis": axis,
                            "max_residual": mismatch,
                        },
                    )
        if not np.all(np.isfinite(generalized_map)):
            raise IncompleteTorqueCertificate(
                "MuJoCo foot point Jacobian contains non-finite values",
                missing=["finite contact Jacobian"],
            )
        friction = (
            min(friction_values) * float(self.config.friction_safety_fraction)
        )
        if not math.isfinite(friction) or friction <= 0.0:
            raise IncompleteTorqueCertificate(
                "bound model has no positive sliding-friction support",
                missing=["positive floor-foot friction coefficient"],
                details={"raw_sliding_friction": friction_values},
            )
        geometry = _GroundGeometry(
            generalized_force_map=generalized_map,
            point_world=points_array,
            point_foot_index=foot_array,
            floor_basis_world=floor_basis.copy(),
            friction_coefficient=friction,
            report={
                "model_binding": self.model_binding,
                "model_source_path": self.model_source_path,
                "model_source_sha256": self.model_source_sha256,
                "floor_geom": self.config.floor_geom_name,
                "floor_geom_id": int(self._floor_geom),
                "floor_basis_world": floor_basis.tolist(),
                "active_contacts": contact_rows,
                "feet": foot_reports,
                "friction_raw_min": min(friction_values),
                "friction_safety_fraction": float(
                    self.config.friction_safety_fraction
                ),
                "friction_used": friction,
                "cop_constraint": (
                    "nonnegative_normal_forces_at_actual_MuJoCo_contact_hull_vertices"
                ),
                "support_patch_semantics": (
                    "exact_discrete_MuJoCo_contact_positions; their convex hull "
                    "is contained in the contact-proximate collision sole hull"
                ),
            },
        )
        self._geometry_cache[key] = geometry
        self._geometry_cache_order.append(key)
        if len(self._geometry_cache_order) > 256:
            expired = self._geometry_cache_order.pop(0)
            self._geometry_cache.pop(expired, None)
        return geometry

    def solve(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        actuated_dof_indices: np.ndarray,
        effort_lower: np.ndarray,
        effort_upper: np.ndarray,
        velocity_limits: np.ndarray,
        *,
        expected_generalized_force: Optional[np.ndarray] = None,
        path_tangent: Optional[np.ndarray] = None,
        lp_objective: str = GROUND_LP_OBJECTIVE_FEASIBILITY,
    ) -> GroundContactLPSolution:
        """Solve and independently validate one grounded sampled state.

        ``lp_objective`` is per-call and cache-bound; retiming callers retain
        the historical feasibility solve unless they explicitly request the
        hold minimax objective.
        """

        q = _real_array("ground qpos", qpos, ndim=1)
        qd = _real_array("ground qvel", qvel, ndim=1)
        qdd = _real_array("ground qacc", qacc, ndim=1)
        if lp_objective not in (
            GROUND_LP_OBJECTIVE_FEASIBILITY,
            GROUND_LP_OBJECTIVE_HOLD_MINIMAX,
        ):
            raise TorqueRetimeError(
                "ground LP objective must be 'feasibility' or "
                "'hold_minimax_normalized_available_torque'"
            )
        model = self._inverse_model
        if q.shape != (int(model.nq),):
            raise TorqueRetimeError("ground qpos shape does not match model.nq")
        if qd.shape != (int(model.nv),) or qdd.shape != qd.shape:
            raise TorqueRetimeError("ground qvel/qacc shape does not match model.nv")
        velocity = _real_array("ground velocity_limits", velocity_limits, ndim=1)
        if velocity.shape != qd.shape or np.any(velocity <= 0.0):
            raise TorqueRetimeError(
                "ground velocity_limits must be positive and cover all tangent DoFs"
            )
        actuated = np.asarray(actuated_dof_indices)
        tau_lower = _real_array("ground effort_lower", effort_lower, ndim=1)
        tau_upper = _real_array("ground effort_upper", effort_upper, ndim=1)
        if not np.array_equal(actuated, self._actuated_dof_indices):
            raise IncompleteTorqueCertificate(
                "caller actuator rows do not match exact A3 post-root DoFs 6..36",
                missing=["exact A3 actuator selection"],
                details={
                    "expected": self._actuated_dof_indices.tolist(),
                    "actual": actuated.tolist(),
                },
            )
        effort_shape_matches = (
            tau_lower.shape == self._effort_lower.shape
            and tau_upper.shape == self._effort_upper.shape
        )
        if not effort_shape_matches:
            raise IncompleteTorqueCertificate(
                "caller effort intervals do not cover the exact A3 actuator rows",
                missing=["one effort interval per exact A3 actuator row"],
            )
        if lp_objective == GROUND_LP_OBJECTIVE_FEASIBILITY:
            if not np.array_equal(
                tau_lower, self._effort_lower
            ) or not np.array_equal(tau_upper, self._effort_upper):
                raise IncompleteTorqueCertificate(
                    "caller effort intervals do not equal the bound MuJoCo "
                    "actuator contract",
                    missing=["exact model-derived actuator effort bounds"],
                )
        elif (
            np.any(tau_lower < self._effort_lower)
            or np.any(tau_upper > self._effort_upper)
            or np.any(tau_lower >= tau_upper)
            or np.any(tau_lower >= 0.0)
            or np.any(tau_upper <= 0.0)
        ):
            raise IncompleteTorqueCertificate(
                "hold minimax effort intervals are not zero-containing, "
                "non-empty subsets of the bound MuJoCo actuator contract",
                missing=[
                    "runtime-qdes-derived effort interval intersected with "
                    "exact model effort bounds"
                ],
                details={
                    "bound_effort_lower": self._effort_lower.tolist(),
                    "bound_effort_upper": self._effort_upper.tolist(),
                    "caller_effort_lower": tau_lower.tolist(),
                    "caller_effort_upper": tau_upper.tolist(),
                },
            )
        velocity_excess = np.abs(qd) - velocity
        if np.any(velocity_excess > float(self.config.velocity_tolerance)):
            worst = int(np.argmax(velocity_excess))
            raise IncompleteTorqueCertificate(
                "sampled grounded state violates a velocity limit",
                missing=["velocity-limit-feasible grounded path state"],
                details={
                    "dof": worst,
                    "absolute_velocity": float(abs(qd[worst])),
                    "limit": float(velocity[worst]),
                },
            )
        geometry = self._prepare_geometry(q)
        if path_tangent is None:
            raise IncompleteTorqueCertificate(
                "grounded contact solve is missing the geometric path tangent",
                missing=["q_s for no-slip contact-kinematics validation"],
            )
        tangent = _real_array("ground path_tangent", path_tangent, ndim=1)
        if tangent.shape != qd.shape:
            raise TorqueRetimeError("ground path_tangent shape does not match model.nv")
        contact_data = self._contact_data
        contact_data.qpos[:] = q
        contact_data.qvel[:] = qd
        self._mujoco.mj_forward(self._contact_model, contact_data)
        if not hasattr(self._mujoco, "mj_jacDot"):
            raise IncompleteTorqueCertificate(
                "MuJoCo build has no mj_jacDot contact-acceleration API",
                missing=["mujoco.mj_jacDot"],
            )
        max_tangent_velocity = 0.0
        max_contact_velocity = 0.0
        max_contact_acceleration = 0.0
        for world, foot in zip(
            geometry.point_world, geometry.point_foot_index
        ):
            jacobian_position = np.empty((3, int(model.nv)), np.float64)
            jacobian_rotation = np.empty((3, int(model.nv)), np.float64)
            jacobian_dot_position = np.empty((3, int(model.nv)), np.float64)
            jacobian_dot_rotation = np.empty((3, int(model.nv)), np.float64)
            body = int(self._foot_bodies[int(foot)])
            self._mujoco.mj_jac(
                self._contact_model,
                contact_data,
                jacobian_position,
                jacobian_rotation,
                world,
                body,
            )
            self._mujoco.mj_jacDot(
                self._contact_model,
                contact_data,
                jacobian_dot_position,
                jacobian_dot_rotation,
                world,
                body,
            )
            tangent_velocity = jacobian_position @ tangent
            point_velocity = jacobian_position @ qd
            point_acceleration = (
                jacobian_position @ qdd
                + jacobian_dot_position @ qd
            )
            max_tangent_velocity = max(
                max_tangent_velocity,
                float(np.max(np.abs(tangent_velocity))),
            )
            max_contact_velocity = max(
                max_contact_velocity,
                float(np.max(np.abs(point_velocity))),
            )
            max_contact_acceleration = max(
                max_contact_acceleration,
                float(np.max(np.abs(point_acceleration))),
            )
        contact_kinematics_report = {
            "max_abs_Jp_qs_m_per_path_unit": max_tangent_velocity,
            "max_abs_contact_velocity_m_s": max_contact_velocity,
            "max_abs_contact_acceleration_m_s2": max_contact_acceleration,
            "path_tangent_tolerance": float(
                self.config.contact_path_tangent_tolerance_m
            ),
            "contact_velocity_tolerance_m_s": float(
                self.config.contact_velocity_tolerance_m_s
            ),
            "contact_acceleration_tolerance_m_s2": float(
                self.config.contact_acceleration_tolerance_m_s2
            ),
            "acceleration_formula": "Jp*qacc+mj_jacDot_p*qvel",
        }
        if (
            max_tangent_velocity
            > float(self.config.contact_path_tangent_tolerance_m)
            or max_contact_velocity
            > float(self.config.contact_velocity_tolerance_m_s)
            or max_contact_acceleration
            > float(self.config.contact_acceleration_tolerance_m_s2)
        ):
            return GroundContactLPSolution(
                feasible=False,
                actuator_generalized_force=np.empty(0, np.float64),
                point_force_floor=np.empty((0, 3), np.float64),
                equality_residual=np.inf,
                root_residual=np.inf,
                report={
                    "status": "CONTACT_KINEMATICS_INFEASIBLE",
                    "reason": (
                        "prescribed path slips, lifts, or accelerates through "
                        "the fixed floor support"
                    ),
                    "model_binding": self.model_binding,
                    "contact_kinematics": contact_kinematics_report,
                    "contact_geometry": geometry.report,
                },
            )
        data = self._inverse_data
        data.qpos[:] = q
        data.qvel[:] = qd
        data.qacc[:] = qdd
        self._mujoco.mj_inverse(model, data)
        demand = np.asarray(data.qfrc_inverse, np.float64).copy()
        if not np.all(np.isfinite(demand)):
            raise IncompleteTorqueCertificate(
                "MuJoCo inverse dynamics returned non-finite generalized force",
                missing=["finite M(q)qacc+bias"],
            )
        inverse_match_residual = 0.0
        if expected_generalized_force is not None:
            expected = _real_array(
                "expected_generalized_force", expected_generalized_force, ndim=1
            )
            if expected.shape != demand.shape:
                raise TorqueRetimeError(
                    "expected_generalized_force shape does not match model.nv"
                )
            inverse_match_residual = float(np.max(np.abs(expected - demand)))
            allowed = float(self.config.equality_residual_tolerance) * max(
                1.0, float(np.max(np.abs(demand)))
            )
            if inverse_match_residual > allowed:
                raise IncompleteTorqueCertificate(
                    "ground LP and path inverse dynamics disagree on the bound plant",
                    missing=["identical exact-model inverse dynamics semantics"],
                    details={
                        "inverse_match_residual": inverse_match_residual,
                        "allowed_residual": allowed,
                    },
                )
        cache_digest = hashlib.sha256()
        for label, array in (
            ("qpos", q),
            ("qvel", qd),
            ("qacc", qdd),
            ("path_tangent", tangent),
            ("demand", demand),
            ("actuated_dof_indices", actuated_dof_indices),
            ("effort_lower", effort_lower),
            ("effort_upper", effort_upper),
        ):
            value = np.ascontiguousarray(np.asarray(array))
            cache_digest.update(label.encode("ascii"))
            cache_digest.update(str(value.dtype).encode("ascii"))
            cache_digest.update(np.asarray(value.shape, np.int64).tobytes())
            cache_digest.update(value.tobytes())
        if lp_objective != GROUND_LP_OBJECTIVE_FEASIBILITY:
            cache_digest.update(b"lp_objective")
            cache_digest.update(str(lp_objective).encode("ascii"))
        cache_key = cache_digest.digest()
        solution = self._lp_cache.get(cache_key)
        lp_cache_hit = solution is not None
        if solution is None:
            solution = _solve_ground_contact_force_lp(
                demand,
                geometry.generalized_force_map,
                geometry.point_foot_index,
                actuated_dof_indices,
                effort_lower,
                effort_upper,
                friction_coefficient=geometry.friction_coefficient,
                minimum_normal_force_per_foot_n=float(
                    self.config.minimum_normal_force_per_foot_n
                ),
                residual_tolerance=float(
                    self.config.equality_residual_tolerance
                ),
                solver_name=self.config.solver,
                lp_objective=lp_objective,
                minimum_normal_force_per_contact_n=float(
                    self.config.minimum_normal_force_per_contact_n
                ),
            )
            # Cache only a completed physical feasible/infeasible solve.
            # Solver-unavailable/numerical exceptions never reach this line.
            self._lp_cache[cache_key] = solution
            self._lp_cache_order.append(cache_key)
            if len(self._lp_cache_order) > 32768:
                expired = self._lp_cache_order.pop(0)
                self._lp_cache.pop(expired, None)
        if not solution.feasible:
            return GroundContactLPSolution(
                feasible=False,
                actuator_generalized_force=solution.actuator_generalized_force,
                point_force_floor=solution.point_force_floor,
                equality_residual=solution.equality_residual,
                root_residual=solution.root_residual,
                report={
                    **solution.report,
                    "model_binding": self.model_binding,
                    "exact_state_lp_cache_hit": lp_cache_hit,
                    "inverse_match_residual": inverse_match_residual,
                    "contact_geometry": geometry.report,
                    "contact_kinematics": contact_kinematics_report,
                },
            )

        normal = solution.point_force_floor[:, 2]
        minimum_contact_normal = float(
            self.config.minimum_normal_force_per_contact_n
        )
        if np.any(normal < minimum_contact_normal - 1.0e-7):
            raise IncompleteTorqueCertificate(
                "LP solution violates its per-contact normal-force reserve",
                missing=["per-contact normal-force reserve"],
                details={
                    "required_n": minimum_contact_normal,
                    "observed_n": normal.tolist(),
                },
            )
        per_foot_normal: list[float] = []
        per_foot_cop_world: list[list[float]] = []
        per_foot_cop_interior_margin: list[float] = []
        for foot in (0, 1):
            mask = geometry.point_foot_index == foot
            total = float(np.sum(normal[mask]))
            if total < float(self.config.minimum_normal_force_per_foot_n) - 1.0e-7:
                raise IncompleteTorqueCertificate(
                    "LP solution does not load both feet",
                    missing=["positive normal support on both feet"],
                    details={"foot_index": foot, "normal_force_n": total},
                )
            cop = np.sum(
                geometry.point_world[mask] * normal[mask, None], axis=0
            ) / total
            per_foot_normal.append(total)
            per_foot_cop_world.append(cop.tolist())
            polygon = np.asarray(
                geometry.report["feet"][foot]["support_polygon_floor_xy_m"],
                np.float64,
            )
            edge_distances = []
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge = end - start
                edge_length = float(np.linalg.norm(edge))
                edge_distances.append(
                    float(
                        edge[0] * (cop[1] - start[1])
                        - edge[1] * (cop[0] - start[0])
                    )
                    / edge_length
                )
            per_foot_cop_interior_margin.append(min(edge_distances))
        return GroundContactLPSolution(
            feasible=True,
            actuator_generalized_force=solution.actuator_generalized_force,
            point_force_floor=solution.point_force_floor,
            equality_residual=solution.equality_residual,
            root_residual=solution.root_residual,
            report={
                **solution.report,
                "model_binding": self.model_binding,
                "exact_state_lp_cache_hit": lp_cache_hit,
                "inverse_dynamics": (
                    "MuJoCo mj_inverse with constraints and frictionloss disabled; "
                    "smooth net demand includes inertia/bias minus remaining "
                    "passive damping/spring forces"
                ),
                "inverse_match_residual": inverse_match_residual,
                "normal_force_per_foot_n": per_foot_normal,
                "normal_force_per_contact_n": normal.tolist(),
                "cop_interior_margin_per_foot_m": (
                    per_foot_cop_interior_margin
                ),
                "cop_world_per_foot_m": per_foot_cop_world,
                "contact_geometry": geometry.report,
                "contact_kinematics": contact_kinematics_report,
            },
        )


class MujocoSmoothInverseDynamics:
    """Callable inverse dynamics on an isolated MuJoCo model copy.

    The copy has ``dof_frictionloss`` set to zero.  Original values must be
    carried in :class:`DirectActuatorContract` and are reserved as full margins.
    """

    def __init__(self, model: Any) -> None:
        self.model_binding = _mujoco_model_binding(model)
        try:
            import mujoco  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on pod environment
            raise TorqueRetimeError("mujoco is not installed") from exc
        try:
            copied = copy.copy(model)
        except Exception as exc:  # pragma: no cover - binding-specific
            raise TorqueRetimeError("could not isolate a copy of the MuJoCo model") from exc
        if copied is model:
            raise TorqueRetimeError("MuJoCo model copy unexpectedly aliases the source object")
        if _mujoco_model_binding(copied) != self.model_binding:
            raise TorqueRetimeError(
                "MuJoCo inverse-dynamics copy changed the bound source model"
            )
        if np.shares_memory(
            np.asarray(copied.dof_frictionloss), np.asarray(model.dof_frictionloss)
        ):
            raise TorqueRetimeError(
                "MuJoCo model copy shares frictionloss storage with the source model"
            )
        # The actuator-only fixed-base equation must be the unconstrained
        # smooth equation M*qacc+bias-passive=tau.  Leaving contacts, joint
        # limits, or equality constraints enabled can inject qfrc_constraint
        # into mj_inverse and make A/b look like actuator demand.
        copied.opt.disableflags = int(copied.opt.disableflags) | int(
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        )
        copied.dof_frictionloss[:] = 0.0
        if _mujoco_model_binding(model) != self.model_binding:
            raise TorqueRetimeError(
                "isolating inverse dynamics mutated the source MuJoCo model"
            )
        self._mujoco = mujoco
        self._model = copied
        self._data = mujoco.MjData(copied)
        self.semantics = (
            "isolated_mujoco_model_constraints_disabled_frictionloss_zero"
        )

    def __call__(
        self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray
    ) -> np.ndarray:
        if np.asarray(qpos).shape != (int(self._model.nq),):
            raise TorqueRetimeError("qpos shape does not match MuJoCo model.nq")
        if np.asarray(qvel).shape != (int(self._model.nv),):
            raise TorqueRetimeError("qvel shape does not match MuJoCo model.nv")
        if np.asarray(qacc).shape != (int(self._model.nv),):
            raise TorqueRetimeError("qacc shape does not match MuJoCo model.nv")
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.qacc[:] = qacc
        self._mujoco.mj_inverse(self._model, self._data)
        result = np.asarray(self._data.qfrc_inverse, np.float64).copy()
        if not np.all(np.isfinite(result)):
            raise TorqueRetimeError("MuJoCo inverse dynamics returned non-finite force")
        return result


def direct_actuator_contract_from_mujoco(
    model: Any,
    *,
    support_mode: str,
    contact_mode: Optional[str] = None,
    fixed_lp_solver: Optional[str] = None,
) -> DirectActuatorContract:
    """Extract a strict one-motor-per-DoF contract from a MuJoCo model.

    Non-unit gain, actuator dynamics/bias, non-joint transmissions, tendons,
    coupled motors, and passive/unmapped fixed-base DoFs are rejected.  A free
    joint is recorded so :func:`retime_torque_path` can issue the explicit
    support/contact ``INCOMPLETE_FAIL_CLOSED`` result.
    """

    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on pod environment
        raise TorqueRetimeError("mujoco is not installed") from exc

    nv = int(model.nv)
    nu = int(model.nu)
    unsupported_counts = {
        name: int(getattr(model, name, 0))
        for name in ("neq", "ntendon", "nflex", "nplugin")
        if int(getattr(model, name, 0)) != 0
    }
    if unsupported_counts:
        raise TorqueRetimeError(
            "fixed-base direct-drive core rejects equality, tendon, flex, and "
            f"plugin dynamics: {unsupported_counts}"
        )
    if hasattr(model, "jnt_actgravcomp") and np.any(
        np.asarray(model.jnt_actgravcomp) != 0
    ):
        raise TorqueRetimeError("joint actuator gravity compensation is unsupported")
    mapping = np.full(nv, -1, dtype=np.int64)
    gear = np.empty(nu, dtype=np.float64)
    force_lo = np.full(nu, -np.inf, dtype=np.float64)
    force_hi = np.full(nu, np.inf, dtype=np.float64)
    ctrl_lo = np.full(nu, -np.inf, dtype=np.float64)
    ctrl_hi = np.full(nu, np.inf, dtype=np.float64)

    for actuator in range(nu):
        if int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise TorqueRetimeError("only direct joint transmissions are supported")
        joint = int(model.actuator_trnid[actuator, 0])
        dof = int(model.jnt_dofadr[joint])
        joint_type = int(model.jnt_type[joint])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise TorqueRetimeError("each actuator must drive one hinge/slide DoF")
        if mapping[dof] >= 0:
            raise TorqueRetimeError("multiple actuators drive the same DoF")
        if int(model.actuator_dyntype[actuator]) != int(mujoco.mjtDyn.mjDYN_NONE):
            raise TorqueRetimeError("actuator dynamics are unsupported")
        if int(model.actuator_gaintype[actuator]) != int(mujoco.mjtGain.mjGAIN_FIXED):
            raise TorqueRetimeError("actuator gain must be fixed")
        if int(model.actuator_biastype[actuator]) != int(mujoco.mjtBias.mjBIAS_NONE):
            raise TorqueRetimeError("actuator bias must be absent")
        gain = np.asarray(model.actuator_gainprm[actuator], np.float64)
        bias = np.asarray(model.actuator_biasprm[actuator], np.float64)
        if (
            not np.isclose(gain[0], 1.0, atol=1.0e-12, rtol=0.0)
            or np.max(np.abs(gain[1:])) > 1.0e-12
            or np.max(np.abs(bias)) > 1.0e-12
        ):
            raise TorqueRetimeError("actuator must have stateless unit gain and zero bias")
        gear_row = np.asarray(model.actuator_gear[actuator], np.float64)
        if abs(gear_row[0]) <= _EPS or np.max(np.abs(gear_row[1:])) > 1.0e-12:
            raise TorqueRetimeError("actuator gear must be one non-zero scalar")
        gear[actuator] = gear_row[0]
        mapping[dof] = actuator
        if bool(model.actuator_forcelimited[actuator]):
            force_lo[actuator], force_hi[actuator] = model.actuator_forcerange[actuator]
        if bool(model.actuator_ctrllimited[actuator]):
            ctrl_lo[actuator], ctrl_hi[actuator] = model.actuator_ctrlrange[actuator]

    joint_lo = np.full(nv, -np.inf, dtype=np.float64)
    joint_hi = np.full(nv, np.inf, dtype=np.float64)
    free_dof_count = 0
    for joint in range(int(model.njnt)):
        joint_type = int(model.jnt_type[joint])
        dof = int(model.jnt_dofadr[joint])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            free_dof_count += 6
            continue
        width = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        if bool(model.jnt_actfrclimited[joint]):
            lo, hi = model.jnt_actfrcrange[joint]
            joint_lo[dof : dof + width] = lo
            joint_hi[dof : dof + width] = hi

    # Free root rows are intentionally left unmapped; every remaining DoF must
    # be direct-drive mapped.  The retimer rejects the free-root case before
    # resolving fixed-base force limits.
    if np.any(mapping[free_dof_count:] < 0):
        raise TorqueRetimeError("a non-root DoF has no unique direct actuator")
    return DirectActuatorContract(
        dof_to_actuator=mapping,
        actuator_gear=gear,
        actuator_force_lower=force_lo,
        actuator_force_upper=force_hi,
        actuator_control_lower=ctrl_lo,
        actuator_control_upper=ctrl_hi,
        joint_force_lower=joint_lo,
        joint_force_upper=joint_hi,
        frictionloss=np.asarray(model.dof_frictionloss, np.float64).copy(),
        free_dof_count=free_dof_count,
        support_mode=str(support_mode),
        contact_mode=contact_mode,
        fixed_lp_solver=fixed_lp_solver,
        model_binding=_mujoco_model_binding(model),
    )


__all__ = [
    "DirectActuatorContract",
    "DynamicsSlice",
    "GroundContactConfig",
    "GroundContactLPSolution",
    "IncompleteTorqueCertificate",
    "MujocoGroundContactLPSolver",
    "MujocoSmoothInverseDynamics",
    "TangentPathGrid",
    "TorqueRetimeError",
    "TorqueRetimeResult",
    "direct_actuator_contract_from_mujoco",
    "identify_dynamics_slice",
    "resolve_generalized_force_limits",
    "retime_torque_path",
    "validate_mujoco_tangent_path_grid",
]
