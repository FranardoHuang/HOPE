#!/usr/bin/env python3
"""Build exact-model, stationary double-support ready *candidates*.

This module closes one deliberately narrow gap in the canonical motion
pipeline: the shared ready pose must be a real, static A3 pose with both feet
on the floor.  It does not select a final ready, retime a strike, authorize
training, or alter the canonical body-scope compiler.

Two candidates are kept distinct:

``G1``
    Keep the donor root and every non-leg joint fixed.  Preserve each donor
    foot's floor-plane position and yaw, make both collision soles flat on the
    floor, and solve only the twelve leg joints with damped least-squares
    continuation.

``G2``
    Take the root and twelve leg joints from one explicitly supplied vendor
    keyframe, then copy the donor (or caller supplied) non-leg posture onto
    that lower body and re-audit the complete state.

Both paths are model-bound and fail closed.  The caller supplies all expected
source/compiled-model digests; a backend must expose the same immutable
identity.  Static geometry, double support, support margin, joint limits,
collision, and the 12-by-12 leg-to-foot Jacobian are checked for every
candidate.  A sampled static contact-force LP is also run when the exact
MuJoCo/SciPy backend supports it; absence is recorded as
``INCOMPLETE_FAIL_CLOSED`` rather than silently promoted.

The result and receipt are immutable diagnostics.  Publication uses an
exclusive directory and writes the receipt last.  Every receipt contains
``training_authorized=false`` and ``hardware_authorized=false``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from audit_motion_npz import ISAAC_JOINT_NAMES


RUNTIME_JOINT_NAMES = tuple(str(name) for name in ISAAC_JOINT_NAMES)
LEFT_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)
RIGHT_LEG_JOINT_NAMES = (
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
LEG_JOINT_NAMES = LEFT_LEG_JOINT_NAMES + RIGHT_LEG_JOINT_NAMES
UPPER_JOINT_NAMES = tuple(
    name for name in RUNTIME_JOINT_NAMES if name not in LEG_JOINT_NAMES
)
FOOT_BODY_NAMES = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
)
FOOT_COLLISION_GEOM_NAMES = (
    "left_ankle_roll_collision",
    "right_ankle_roll_collision",
)
FLOOR_GEOM_NAME = "floor"
ROOT_BODY_NAME = "pelvis_link"
ARTIFACT_FILENAME = "grounded_ready_candidate_v1.npz"
RECEIPT_FILENAME = "RECEIPT.json"

if len(RUNTIME_JOINT_NAMES) != 31 or len(set(RUNTIME_JOINT_NAMES)) != 31:
    raise RuntimeError("canonical runtime joint contract must contain 31 names")
if len(LEG_JOINT_NAMES) != 12 or len(set(LEG_JOINT_NAMES)) != 12:
    raise RuntimeError("grounded-ready leg contract must contain 12 names")


class GroundedReadyError(RuntimeError):
    """A candidate cannot satisfy a fail-closed grounded-ready contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "GROUNDED_READY_ERROR",
        report: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.report = report


@dataclass(frozen=True)
class ExactModelIdentity:
    """All externally supplied pins needed by the exact A3 backend."""

    mjcf_path: str | Path
    mjcf_sha256: str
    compiled_model_sha256: str
    path_model_binding_sha256: str
    ground_model_binding_sha256: str
    xml_model_name: str = "A3T2.5_pingpong_0519"

    def __post_init__(self) -> None:
        for label in (
            "mjcf_sha256",
            "compiled_model_sha256",
            "path_model_binding_sha256",
            "ground_model_binding_sha256",
        ):
            _require_sha256(str(getattr(self, label)), label)
        if (
            not isinstance(self.xml_model_name, str)
            or not self.xml_model_name
            or self.xml_model_name.strip() != self.xml_model_name
        ):
            raise ValueError("xml_model_name must be a non-empty trimmed string")


@dataclass(frozen=True)
class ReadyState:
    """One complete, zero-velocity ready configuration."""

    joint_pos: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_pos",
            _readonly_vector(self.joint_pos, 31, "state joint_pos"),
        )
        object.__setattr__(
            self,
            "root_pos_w",
            _readonly_vector(self.root_pos_w, 3, "state root_pos_w"),
        )
        quat = _readonly_vector(self.root_quat_wxyz, 4, "state root_quat_wxyz")
        norm = float(np.linalg.norm(quat))
        if abs(norm - 1.0) > 1.0e-8:
            raise GroundedReadyError(
                f"state root quaternion must be unit length, got {norm:.12g}",
                code="INVALID_ROOT_QUATERNION",
            )
        quat = np.ascontiguousarray(quat / norm, dtype=np.float64)
        quat.setflags(write=False)
        object.__setattr__(self, "root_quat_wxyz", quat)


@dataclass(frozen=True)
class FootPose:
    """World pose of one audited foot body."""

    position_w: np.ndarray
    rotation_w: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_w",
            _readonly_vector(self.position_w, 3, "foot position"),
        )
        rotation = np.array(self.rotation_w, dtype=np.float64, copy=True, order="C")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise GroundedReadyError(
                "foot rotation must be one finite 3x3 matrix",
                code="INVALID_FOOT_ROTATION",
            )
        if (
            np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1.0e-8
            or np.linalg.det(rotation) < 1.0 - 1.0e-8
        ):
            raise GroundedReadyError(
                "foot rotation must be a proper orthonormal matrix",
                code="INVALID_FOOT_ROTATION",
            )
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation_w", rotation)


@dataclass(frozen=True)
class StaticScene:
    """Backend-independent exact-model static scene evidence."""

    foot_contact_count: tuple[int, int]
    contact_points_w: np.ndarray
    contact_foot_indices: np.ndarray
    floor_origin_w: np.ndarray
    floor_basis_w: np.ndarray
    sole_minimum_distance_m: np.ndarray
    com_w: np.ndarray
    maximum_foot_penetration_m: float
    unsupported_contacts: tuple[Mapping[str, Any], ...] = ()
    self_collision_pairs: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.foot_contact_count) != 2 or any(
            int(value) < 0 for value in self.foot_contact_count
        ):
            raise GroundedReadyError(
                "foot_contact_count must contain two non-negative integers",
                code="INVALID_STATIC_SCENE",
            )
        points = _readonly_matrix(self.contact_points_w, 3, "scene contact_points_w")
        feet = np.asarray(self.contact_foot_indices)
        if (
            feet.ndim != 1
            or len(feet) != len(points)
            or not np.issubdtype(feet.dtype, np.integer)
            or np.any((feet < 0) | (feet > 1))
        ):
            raise GroundedReadyError(
                "scene contact_foot_indices must map every point to foot 0/1",
                code="INVALID_STATIC_SCENE",
            )
        feet = np.array(feet, dtype=np.int64, copy=True, order="C")
        feet.setflags(write=False)
        basis = np.array(self.floor_basis_w, dtype=np.float64, copy=True, order="C")
        if (
            basis.shape != (3, 3)
            or not np.isfinite(basis).all()
            or np.max(np.abs(basis.T @ basis - np.eye(3))) > 1.0e-8
            or np.linalg.det(basis) < 1.0 - 1.0e-8
        ):
            raise GroundedReadyError(
                "scene floor_basis_w must be a proper orthonormal matrix",
                code="INVALID_STATIC_SCENE",
            )
        penetration = float(self.maximum_foot_penetration_m)
        if not math.isfinite(penetration) or penetration < 0.0:
            raise GroundedReadyError(
                "maximum_foot_penetration_m must be finite and non-negative",
                code="INVALID_STATIC_SCENE",
            )
        basis.setflags(write=False)
        object.__setattr__(self, "contact_points_w", points)
        object.__setattr__(self, "contact_foot_indices", feet)
        object.__setattr__(
            self,
            "floor_origin_w",
            _readonly_vector(self.floor_origin_w, 3, "scene floor_origin_w"),
        )
        object.__setattr__(self, "floor_basis_w", basis)
        object.__setattr__(
            self,
            "sole_minimum_distance_m",
            _readonly_vector(
                self.sole_minimum_distance_m,
                2,
                "scene sole_minimum_distance_m",
            ),
        )
        object.__setattr__(
            self, "com_w", _readonly_vector(self.com_w, 3, "scene com_w")
        )
        object.__setattr__(self, "maximum_foot_penetration_m", penetration)


@dataclass(frozen=True)
class GroundedReadyConfig:
    """Numerical contract for G1/G2 solving and static auditing."""

    continuation_steps: int = 12
    maximum_iterations_per_step: int = 80
    finite_difference_step_rad: float = 2.0e-6
    damping: float = 1.0e-7
    maximum_joint_step_rad: float = 0.12
    translation_weight_per_m: float = 1.0
    rotation_weight_per_rad: float = 0.25
    target_contact_preload_m: float = 5.0e-4
    foot_position_tolerance_m: float = 2.0e-5
    foot_rotation_tolerance_rad: float = 2.0e-4
    floor_gap_tolerance_m: float = 2.0e-3
    penetration_tolerance_m: float = 2.0e-3
    support_hull_tolerance_m: float = 2.0e-4
    minimum_support_margin_m: float = 0.0
    joint_limit_tolerance_rad: float = 1.0e-10
    jacobian_rank_tolerance: float = 1.0e-7
    maximum_jacobian_condition: float = 1.0e4

    def __post_init__(self) -> None:
        if self.continuation_steps < 1:
            raise ValueError("continuation_steps must be positive")
        if self.maximum_iterations_per_step < 1:
            raise ValueError("maximum_iterations_per_step must be positive")
        for name in (
            "finite_difference_step_rad",
            "damping",
            "maximum_joint_step_rad",
            "translation_weight_per_m",
            "rotation_weight_per_rad",
            "foot_position_tolerance_m",
            "foot_rotation_tolerance_rad",
            "floor_gap_tolerance_m",
            "penetration_tolerance_m",
            "support_hull_tolerance_m",
            "jacobian_rank_tolerance",
            "maximum_jacobian_condition",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(float(self.target_contact_preload_m))
            or float(self.target_contact_preload_m) < 0.0
            or float(self.target_contact_preload_m)
            > float(self.penetration_tolerance_m)
        ):
            raise ValueError(
                "target_contact_preload_m must lie between zero and the "
                "penetration tolerance"
            )
        if (
            not math.isfinite(float(self.minimum_support_margin_m))
            or float(self.minimum_support_margin_m) < 0.0
        ):
            raise ValueError("minimum_support_margin_m must be finite and non-negative")
        if (
            not math.isfinite(float(self.joint_limit_tolerance_rad))
            or float(self.joint_limit_tolerance_rad) < 0.0
        ):
            raise ValueError(
                "joint_limit_tolerance_rad must be finite and non-negative"
            )


@dataclass(frozen=True)
class _GroundedReadyAuditDraft:
    receipt_payload: Mapping[str, Any]
    receipt_sha256: str
    geometry_passed: bool
    ground_dynamics_passed: bool | None


@dataclass(frozen=True)
class GroundedReadyResult:
    """Immutable diagnostic candidate plus its content-sealed receipt."""

    candidate_id: str
    state: ReadyState
    target_foot_poses: tuple[FootPose, FootPose]
    geometry_passed: bool
    ground_dynamics_passed: bool | None
    receipt: Mapping[str, Any]
    receipt_sha256: str
    training_authorized: bool = False
    hardware_authorized: bool = False
    verification_backend: InitVar[Any] = None
    expected_model_identity: InitVar[Any] = None
    verification_config: InitVar[Any] = None
    construction_source: InitVar[Any] = None

    def __post_init__(
        self,
        verification_backend: Any,
        expected_model_identity: Any,
        verification_config: Any,
        construction_source: Any,
    ) -> None:
        if self.training_authorized or self.hardware_authorized:
            raise GroundedReadyError(
                "grounded-ready diagnostics cannot authorize training/hardware",
                code="AUTHORIZATION_FORBIDDEN",
            )
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise GroundedReadyError(
                "candidate_id must be one non-empty string",
                code="INVALID_RESULT",
            )
        targets = tuple(self.target_foot_poses)
        if len(targets) != 2 or any(not isinstance(row, FootPose) for row in targets):
            raise GroundedReadyError(
                "result must retain exactly two immutable FootPose targets",
                code="INVALID_RESULT",
            )
        payload = _jsonable(self.receipt)
        exact_backend = _validate_result_receipt_payload(self, targets, payload)
        if exact_backend:
            _revalidate_exact_result_receipt(
                self,
                targets,
                payload,
                backend=verification_backend,
                expected_model_identity=expected_model_identity,
                config=verification_config,
                construction_source=construction_source,
            )
        object.__setattr__(self, "target_foot_poses", targets)
        object.__setattr__(self, "receipt", _deep_freeze(payload))


@dataclass(frozen=True)
class PublishedGroundedReady:
    """Exclusive diagnostic bundle paths and hashes."""

    directory: Path
    candidate_npz: Path
    receipt_json: Path
    candidate_npz_sha256: str
    receipt_json_sha256: str


class GroundedReadyBackend(Protocol):
    """Dependency-injected exact-model kinematics/static-audit backend."""

    joint_names: Sequence[str]
    position_lower: np.ndarray
    position_upper: np.ndarray
    exact_model_identity: ExactModelIdentity

    def foot_poses(self, state: ReadyState) -> tuple[FootPose, FootPose]:
        ...

    def flat_foot_targets(
        self,
        state: ReadyState,
        *,
        contact_preload_m: float,
    ) -> tuple[FootPose, FootPose]:
        ...

    def static_scene(
        self,
        state: ReadyState,
        *,
        contact_gap_tolerance_m: float,
        penetration_tolerance_m: float,
    ) -> StaticScene:
        ...

    def static_ground_dynamics(
        self,
        state: ReadyState,
        *,
        contact_gap_tolerance_m: float,
        penetration_tolerance_m: float,
    ) -> Mapping[str, Any]:
        ...

    def vendor_key_state(self, key_index: int = 0) -> ReadyState:
        ...


def solve_g1_donor_root(
    donor_state: ReadyState,
    *,
    backend: GroundedReadyBackend,
    expected_model_identity: ExactModelIdentity,
    upper_candidate_joint_pos: np.ndarray | None = None,
    upper_joint_names: Sequence[str] | None = None,
    config: GroundedReadyConfig | None = None,
) -> GroundedReadyResult:
    """Solve G1 with donor root and non-leg joints held exactly fixed."""

    cfg = GroundedReadyConfig() if config is None else config
    _verify_backend_contract(backend, expected_model_identity)
    seed, overlay_receipt = _overlay_state(
        donor_state,
        upper_candidate_joint_pos,
        upper_joint_names,
        backend=backend,
    )
    source_poses = backend.foot_poses(seed)
    targets = backend.flat_foot_targets(
        seed, contact_preload_m=cfg.target_contact_preload_m
    )
    q, trace = _solve_leg_continuation(
        seed,
        source_poses=source_poses,
        target_poses=targets,
        backend=backend,
        config=cfg,
    )
    solved = ReadyState(q, seed.root_pos_w, seed.root_quat_wxyz)
    nonleg = _joint_indices(backend.joint_names, UPPER_JOINT_NAMES)
    if not np.array_equal(solved.joint_pos[nonleg], seed.joint_pos[nonleg]):
        raise GroundedReadyError(
            "G1 changed a non-leg joint",
            code="G1_NONLEG_MUTATION",
        )
    if not np.array_equal(
        solved.root_pos_w, donor_state.root_pos_w
    ) or not np.array_equal(solved.root_quat_wxyz, donor_state.root_quat_wxyz):
        raise GroundedReadyError(
            "G1 changed the donor root",
            code="G1_ROOT_MUTATION",
        )
    source = {
        "mode": "G1_donor_root_flat_feet_leg12_continuation",
        "donor_state_sha256": state_digest(donor_state),
        "seed_state_sha256": state_digest(seed),
        "upper_overlay": overlay_receipt,
        "root_bitwise_preserved": True,
        "nonleg_joint_values_bitwise_preserved": True,
        "target_semantics": (
            "preserve_each_donor_foot_floor_xy_and_yaw;"
            "flat_collision_sole_with_configured_bounded_contact_preload"
        ),
        "target_contact_preload_m": cfg.target_contact_preload_m,
        "solver_trace": trace,
    }
    return _audit_and_build_result(
        "G1",
        solved,
        targets,
        source=source,
        backend=backend,
        expected_model_identity=expected_model_identity,
        config=cfg,
    )


def build_g2_vendor_key_candidate(
    donor_state: ReadyState,
    *,
    backend: GroundedReadyBackend,
    expected_model_identity: ExactModelIdentity,
    vendor_key_index: int = 0,
    upper_candidate_joint_pos: np.ndarray | None = None,
    upper_joint_names: Sequence[str] | None = None,
    config: GroundedReadyConfig | None = None,
) -> GroundedReadyResult:
    """Build G2 from vendor root/lower plus donor or supplied upper posture."""

    cfg = GroundedReadyConfig() if config is None else config
    _verify_backend_contract(backend, expected_model_identity)
    try:
        vendor_key_state = backend.vendor_key_state(int(vendor_key_index))
    except Exception as exc:
        raise GroundedReadyError(
            f"cannot resolve exact backend vendor key {vendor_key_index}: {exc}",
            code="VENDOR_KEY_MISSING",
        ) from exc
    upper_seed, overlay_receipt = _overlay_state(
        donor_state,
        upper_candidate_joint_pos,
        upper_joint_names,
        backend=backend,
    )
    leg = _joint_indices(backend.joint_names, LEG_JOINT_NAMES)
    nonleg = _joint_indices(backend.joint_names, UPPER_JOINT_NAMES)
    q = np.asarray(vendor_key_state.joint_pos, dtype=np.float64).copy()
    q[nonleg] = upper_seed.joint_pos[nonleg]
    candidate = ReadyState(
        q,
        vendor_key_state.root_pos_w,
        vendor_key_state.root_quat_wxyz,
    )
    if not np.array_equal(candidate.joint_pos[leg], vendor_key_state.joint_pos[leg]):
        raise GroundedReadyError(
            "G2 failed to preserve vendor lower body",
            code="G2_LOWER_MUTATION",
        )
    targets = backend.foot_poses(vendor_key_state)
    source = {
        "mode": "G2_vendor_key_root_lower_plus_upper_overlay",
        "donor_state_sha256": state_digest(donor_state),
        "vendor_key_state_sha256": state_digest(vendor_key_state),
        "vendor_key_index": int(vendor_key_index),
        "upper_seed_state_sha256": state_digest(upper_seed),
        "upper_overlay": overlay_receipt,
        "root_bitwise_equal_vendor_key": True,
        "leg_joint_values_bitwise_equal_vendor_key": True,
        "nonleg_joint_values_bitwise_equal_upper_seed": bool(
            np.array_equal(candidate.joint_pos[nonleg], upper_seed.joint_pos[nonleg])
        ),
        "target_semantics": "exact_vendor_key_foot_body_poses",
    }
    return _audit_and_build_result(
        "G2",
        candidate,
        targets,
        source=source,
        backend=backend,
        expected_model_identity=expected_model_identity,
        config=cfg,
    )


def revalidate_upper_overlay(
    base: GroundedReadyResult,
    upper_candidate_joint_pos: np.ndarray,
    *,
    backend: GroundedReadyBackend,
    expected_model_identity: ExactModelIdentity,
    upper_joint_names: Sequence[str] | None = None,
    config: GroundedReadyConfig | None = None,
) -> GroundedReadyResult:
    """Overlay non-leg/right-arm joints and rerun every static ground gate."""

    cfg = GroundedReadyConfig() if config is None else config
    _verify_backend_contract(backend, expected_model_identity)
    state, overlay_receipt = _overlay_state(
        base.state,
        upper_candidate_joint_pos,
        upper_joint_names,
        backend=backend,
    )
    source = {
        "mode": "revalidated_upper_overlay",
        "base_candidate_id": base.candidate_id,
        "base_receipt_sha256": base.receipt_sha256,
        "base_state_sha256": state_digest(base.state),
        "upper_overlay": overlay_receipt,
        "lower_and_root_bitwise_preserved": True,
        "all_static_gates_rerun": True,
    }
    return _audit_and_build_result(
        f"{base.candidate_id}+overlay",
        state,
        base.target_foot_poses,
        source=source,
        backend=backend,
        expected_model_identity=expected_model_identity,
        config=cfg,
    )


def _rebuild_exact_publication_candidate(
    proposed: GroundedReadyResult,
    *,
    expected_model_identity: Any,
    config: Any,
    construction_mode: Any,
    donor_state: Any,
    upper_candidate_joint_pos: Any,
    upper_joint_names: Any,
    vendor_key_index: Any,
) -> GroundedReadyResult:
    """Rebuild G1/G2 from trusted construction inputs, never receipt source."""

    if (
        not isinstance(expected_model_identity, ExactModelIdentity)
        or not isinstance(config, GroundedReadyConfig)
        or not isinstance(donor_state, ReadyState)
        or construction_mode not in ("G1", "G2")
    ):
        raise GroundedReadyError(
            "exact publication requires pinned identity/config, explicit G1/G2 "
            "mode, and the donor ReadyState",
            code="EXACT_PUBLICATION_CONSTRUCTION_REQUIRED",
        )
    try:
        fresh_backend = MujocoGroundedReadyBackend.load(expected_model_identity)
    except Exception as exc:
        raise GroundedReadyError(
            "fresh exact-MuJoCo publication backend reload failed: "
            f"{type(exc).__name__}: {exc}",
            code="EXACT_PUBLICATION_FRESH_BACKEND_REQUIRED",
        ) from exc

    if construction_mode == "G1":
        if vendor_key_index is not None:
            raise GroundedReadyError(
                "G1 publication may not receive a vendor key index",
                code="EXACT_PUBLICATION_INPUT_MISMATCH",
            )
        rebuilt = solve_g1_donor_root(
            donor_state,
            backend=fresh_backend,
            expected_model_identity=expected_model_identity,
            upper_candidate_joint_pos=upper_candidate_joint_pos,
            upper_joint_names=upper_joint_names,
            config=config,
        )
    else:
        if (
            isinstance(vendor_key_index, bool)
            or not isinstance(vendor_key_index, int)
            or vendor_key_index < 0
        ):
            raise GroundedReadyError(
                "G2 publication requires one explicit non-negative vendor key index",
                code="EXACT_PUBLICATION_CONSTRUCTION_REQUIRED",
            )
        rebuilt = build_g2_vendor_key_candidate(
            donor_state,
            backend=fresh_backend,
            expected_model_identity=expected_model_identity,
            vendor_key_index=vendor_key_index,
            upper_candidate_joint_pos=upper_candidate_joint_pos,
            upper_joint_names=upper_joint_names,
            config=config,
        )

    if not _same_grounded_ready_result(rebuilt, proposed):
        raise GroundedReadyError(
            "construction-bound exact rebuild differs from the proposed result",
            code="EXACT_PUBLICATION_CONSTRUCTION_MISMATCH",
            report={
                "construction_mode": construction_mode,
                "proposed_candidate_id": proposed.candidate_id,
                "rebuilt_candidate_id": rebuilt.candidate_id,
                "proposed_state_sha256": state_digest(proposed.state),
                "rebuilt_state_sha256": state_digest(rebuilt.state),
                "proposed_receipt_sha256": proposed.receipt_sha256,
                "rebuilt_receipt_sha256": rebuilt.receipt_sha256,
            },
        )
    return rebuilt


def _same_grounded_ready_result(
    left: GroundedReadyResult,
    right: GroundedReadyResult,
) -> bool:
    if (
        left.candidate_id != right.candidate_id
        or left.geometry_passed is not right.geometry_passed
        or left.ground_dynamics_passed is not right.ground_dynamics_passed
        or left.training_authorized is not right.training_authorized
        or left.hardware_authorized is not right.hardware_authorized
        or left.receipt_sha256 != right.receipt_sha256
        or not np.array_equal(left.state.joint_pos, right.state.joint_pos)
        or not np.array_equal(left.state.root_pos_w, right.state.root_pos_w)
        or not np.array_equal(
            left.state.root_quat_wxyz,
            right.state.root_quat_wxyz,
        )
        or len(left.target_foot_poses) != len(right.target_foot_poses)
    ):
        return False
    for left_pose, right_pose in zip(
        left.target_foot_poses,
        right.target_foot_poses,
    ):
        if not np.array_equal(
            left_pose.position_w,
            right_pose.position_w,
        ) or not np.array_equal(
            left_pose.rotation_w,
            right_pose.rotation_w,
        ):
            return False
    return _canonical_json_bytes(_jsonable(left.receipt)) == _canonical_json_bytes(
        _jsonable(right.receipt)
    )


def publish_grounded_ready_candidate(
    result: GroundedReadyResult,
    output_directory: str | Path,
    *,
    expected_model_identity: ExactModelIdentity | None = None,
    verification_config: GroundedReadyConfig | None = None,
    construction_mode: str | None = None,
    construction_donor_state: ReadyState | None = None,
    construction_upper_candidate_joint_pos: np.ndarray | None = None,
    construction_upper_joint_names: Sequence[str] | None = None,
    construction_vendor_key_index: int | None = None,
) -> PublishedGroundedReady:
    """Construction-rebuild and publish one diagnostic without overwriting."""

    if not result.geometry_passed:
        raise GroundedReadyError(
            "refusing to publish a candidate that failed static geometry",
            code="PUBLICATION_GEOMETRY_FAILED",
        )
    proposed_payload = _jsonable(result.receipt)
    exact_claim = bool(
        isinstance(proposed_payload.get("exact_model"), Mapping)
        and proposed_payload["exact_model"].get("exact_mujoco_backend") is True
    )
    if exact_claim:
        audited = _rebuild_exact_publication_candidate(
            result,
            expected_model_identity=expected_model_identity,
            config=verification_config,
            construction_mode=construction_mode,
            donor_state=construction_donor_state,
            upper_candidate_joint_pos=construction_upper_candidate_joint_pos,
            upper_joint_names=construction_upper_joint_names,
            vendor_key_index=construction_vendor_key_index,
        )
    else:
        # Protocol-test artifacts remain explicitly labelled test-only.
        audited = GroundedReadyResult(
            candidate_id=result.candidate_id,
            state=result.state,
            target_foot_poses=result.target_foot_poses,
            geometry_passed=result.geometry_passed,
            ground_dynamics_passed=result.ground_dynamics_passed,
            receipt=result.receipt,
            receipt_sha256=result.receipt_sha256,
            training_authorized=result.training_authorized,
            hardware_authorized=result.hardware_authorized,
        )
    output = Path(output_directory).expanduser().absolute()
    parent_input = output.parent
    try:
        parent = parent_input.resolve(strict=True)
    except OSError as exc:
        raise GroundedReadyError(
            f"cannot resolve output parent: {exc}",
            code="INVALID_OUTPUT_PARENT",
        ) from exc
    if parent_input != parent or not parent.is_dir():
        raise GroundedReadyError(
            f"output parent may not contain a symlink component: {parent_input}",
            code="INVALID_OUTPUT_PARENT",
        )
    name = output.name
    if not name or name in (".", ".."):
        raise GroundedReadyError(
            "output directory needs one concrete final component",
            code="INVALID_OUTPUT_PARENT",
        )
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    output_fd = -1
    created = False
    try:
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite grounded-ready bundle: {output}"
            ) from None
        output_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        output_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(name, output_flags, dir_fd=parent_fd)

        candidate_buffer = io.BytesIO()
        np.savez(
            candidate_buffer,
            joint_pos=np.asarray(audited.state.joint_pos, np.float64),
            joint_vel=np.zeros(31, np.float64),
            root_pos_w=np.asarray(audited.state.root_pos_w, np.float64),
            root_quat_w=np.asarray(audited.state.root_quat_wxyz, np.float64),
            root_lin_vel_w=np.zeros(3, np.float64),
            root_ang_vel_w=np.zeros(3, np.float64),
            candidate_id=np.asarray(audited.candidate_id),
            receipt_sha256=np.asarray(audited.receipt_sha256),
            training_authorized=np.asarray(False, np.bool_),
            hardware_authorized=np.asarray(False, np.bool_),
        )
        candidate_payload = candidate_buffer.getvalue()
        candidate_sha = hashlib.sha256(candidate_payload).hexdigest()
        _exclusive_write_at(output_fd, ARTIFACT_FILENAME, candidate_payload)

        receipt = _jsonable(audited.receipt)
        receipt["publication"] = {
            "candidate_filename": ARTIFACT_FILENAME,
            "candidate_npz_sha256": candidate_sha,
            "receipt_filename": RECEIPT_FILENAME,
            "completion_semantics": ("exclusive_directory_and_receipt_written_last"),
        }
        canonical_without_seal = _canonical_json_bytes(receipt)
        receipt["publication_payload_sha256"] = hashlib.sha256(
            canonical_without_seal
        ).hexdigest()
        payload = _canonical_json_bytes(receipt)
        _exclusive_write_at(output_fd, RECEIPT_FILENAME, payload)
        os.fsync(output_fd)
        os.fsync(parent_fd)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode) or not os.path.samestat(
            entry, os.fstat(output_fd)
        ):
            raise GroundedReadyError(
                "published directory identity changed during write",
                code="OUTPUT_DIRECTORY_RACE",
            )
        published_output = parent / name
        return PublishedGroundedReady(
            directory=published_output,
            candidate_npz=published_output / ARTIFACT_FILENAME,
            receipt_json=published_output / RECEIPT_FILENAME,
            candidate_npz_sha256=candidate_sha,
            receipt_json_sha256=hashlib.sha256(payload).hexdigest(),
        )
    except Exception:
        if output_fd >= 0:
            for filename in (RECEIPT_FILENAME, ARTIFACT_FILENAME):
                try:
                    os.unlink(filename, dir_fd=output_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
        if created:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)


def _exclusive_write_at(directory_fd: int, filename: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, mode=0o644, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("exclusive artifact write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def state_digest(state: ReadyState) -> str:
    """Return a deterministic content digest for one complete ready state."""

    digest = hashlib.sha256()
    _hash_array(digest, "joint_pos", state.joint_pos)
    _hash_array(digest, "root_pos_w", state.root_pos_w)
    _hash_array(digest, "root_quat_wxyz", state.root_quat_wxyz)
    return digest.hexdigest()


def _solve_leg_continuation(
    seed: ReadyState,
    *,
    source_poses: tuple[FootPose, FootPose],
    target_poses: tuple[FootPose, FootPose],
    backend: GroundedReadyBackend,
    config: GroundedReadyConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    leg = _joint_indices(backend.joint_names, LEG_JOINT_NAMES)
    lower = _finite_backend_vector(
        backend.position_lower, len(backend.joint_names), "position_lower"
    )
    upper = _finite_backend_vector(
        backend.position_upper, len(backend.joint_names), "position_upper"
    )
    if np.any(lower > upper):
        raise GroundedReadyError(
            "backend joint position ranges are inverted",
            code="INVALID_MODEL_LIMITS",
        )
    q = np.asarray(seed.joint_pos, np.float64).copy()
    fixed = np.setdiff1d(np.arange(len(q), dtype=np.int64), leg)
    fixed_bytes = q[fixed].copy()
    trace: list[dict[str, Any]] = []
    weights = np.asarray(
        [config.translation_weight_per_m] * 3 + [config.rotation_weight_per_rad] * 3,
        np.float64,
    )
    weights = np.tile(weights, 2)

    for stage in range(1, config.continuation_steps + 1):
        alpha = float(stage) / float(config.continuation_steps)
        stage_targets = tuple(
            FootPose(
                (1.0 - alpha) * source.position_w + alpha * target.position_w,
                _interpolate_rotation(source.rotation_w, target.rotation_w, alpha),
            )
            for source, target in zip(source_poses, target_poses)
        )
        damping = float(config.damping)
        iterations = 0
        accepted = 0
        for iteration in range(config.maximum_iterations_per_step):
            iterations = iteration + 1
            state = ReadyState(q, seed.root_pos_w, seed.root_quat_wxyz)
            residual = _foot_pose_residual(backend.foot_poses(state), stage_targets)
            position_peak, rotation_peak = _residual_peaks(residual)
            if (
                position_peak <= config.foot_position_tolerance_m
                and rotation_peak <= config.foot_rotation_tolerance_rad
            ):
                break
            jacobian = _finite_difference_leg_jacobian(
                state,
                stage_targets,
                leg,
                backend=backend,
                step=config.finite_difference_step_rad,
            )
            weighted_jacobian = weights[:, None] * jacobian
            weighted_residual = weights * residual
            lhs = weighted_jacobian.T @ weighted_jacobian
            lhs += damping * np.eye(len(leg), dtype=np.float64)
            rhs = -(weighted_jacobian.T @ weighted_residual)
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            peak = float(np.max(np.abs(delta)))
            if peak > config.maximum_joint_step_rad:
                delta *= config.maximum_joint_step_rad / peak
            objective = float(np.linalg.norm(weighted_residual))
            found = False
            scale = 1.0
            for _ in range(12):
                trial = q.copy()
                trial[leg] = np.clip(
                    q[leg] + scale * delta,
                    lower[leg],
                    upper[leg],
                )
                trial_state = ReadyState(trial, seed.root_pos_w, seed.root_quat_wxyz)
                trial_residual = _foot_pose_residual(
                    backend.foot_poses(trial_state), stage_targets
                )
                trial_objective = float(np.linalg.norm(weights * trial_residual))
                if trial_objective < objective - 1.0e-14:
                    q = trial
                    accepted += 1
                    damping = max(config.damping, damping * 0.3)
                    found = True
                    break
                scale *= 0.5
            if not found:
                damping *= 10.0
                if damping > 1.0e8:
                    break
        final_state = ReadyState(q, seed.root_pos_w, seed.root_quat_wxyz)
        final_residual = _foot_pose_residual(
            backend.foot_poses(final_state), stage_targets
        )
        position_peak, rotation_peak = _residual_peaks(final_residual)
        jacobian = _finite_difference_leg_jacobian(
            final_state,
            stage_targets,
            leg,
            backend=backend,
            step=config.finite_difference_step_rad,
        )
        rank, condition, singular_values = _jacobian_metrics(
            jacobian, config.jacobian_rank_tolerance
        )
        trace.append(
            {
                "stage": stage,
                "alpha": alpha,
                "iterations": iterations,
                "accepted_steps": accepted,
                "maximum_foot_position_residual_m": position_peak,
                "maximum_foot_rotation_residual_rad": rotation_peak,
                "jacobian_rank": rank,
                "jacobian_condition": condition,
                "jacobian_singular_values": singular_values.tolist(),
            }
        )
        if rank < 12 or condition > config.maximum_jacobian_condition:
            raise GroundedReadyError(
                "G1 leg-to-foot Jacobian is singular or ill-conditioned",
                code="FOOT_MANIFOLD_SINGULAR",
                report={"solver_trace": trace},
            )
        if (
            position_peak > config.foot_position_tolerance_m
            or rotation_peak > config.foot_rotation_tolerance_rad
        ):
            raise GroundedReadyError(
                "G1 leg continuation did not reach the requested foot poses",
                code="READY_GROUNDING_INFEASIBLE",
                report={"solver_trace": trace},
            )

    if not np.array_equal(q[fixed], fixed_bytes):
        raise GroundedReadyError(
            "leg continuation changed a fixed coordinate",
            code="G1_FIXED_COORDINATE_MUTATION",
        )
    return q, trace


def _audit_and_build_result(
    candidate_id: str,
    state: ReadyState,
    targets: tuple[FootPose, FootPose],
    *,
    source: Mapping[str, Any],
    backend: GroundedReadyBackend,
    expected_model_identity: ExactModelIdentity,
    config: GroundedReadyConfig,
    _return_draft: bool = False,
) -> GroundedReadyResult | _GroundedReadyAuditDraft:
    model_receipt = _verify_backend_contract(backend, expected_model_identity)
    lower = _finite_backend_vector(
        backend.position_lower, len(backend.joint_names), "position_lower"
    )
    upper = _finite_backend_vector(
        backend.position_upper, len(backend.joint_names), "position_upper"
    )
    below = lower - state.joint_pos
    above = state.joint_pos - upper
    limit_excess = np.maximum(below, above)
    worst_limit_excess = float(max(0.0, np.max(limit_excess)))
    limits_pass = bool(worst_limit_excess <= config.joint_limit_tolerance_rad)

    poses = backend.foot_poses(state)
    residual = _foot_pose_residual(poses, targets)
    position_peak, rotation_peak = _residual_peaks(residual)
    foot_pose_pass = bool(
        position_peak <= config.foot_position_tolerance_m
        and rotation_peak <= config.foot_rotation_tolerance_rad
    )
    leg = _joint_indices(backend.joint_names, LEG_JOINT_NAMES)
    jacobian = _finite_difference_leg_jacobian(
        state,
        targets,
        leg,
        backend=backend,
        step=config.finite_difference_step_rad,
    )
    rank, condition, singular_values = _jacobian_metrics(
        jacobian, config.jacobian_rank_tolerance
    )
    jacobian_pass = bool(rank == 12 and condition <= config.maximum_jacobian_condition)

    scene = backend.static_scene(
        state,
        contact_gap_tolerance_m=config.floor_gap_tolerance_m,
        penetration_tolerance_m=config.penetration_tolerance_m,
    )
    double_support_pass = bool(
        scene.foot_contact_count[0] > 0 and scene.foot_contact_count[1] > 0
    )
    sole_gap_pass = bool(
        np.max(np.abs(scene.sole_minimum_distance_m)) <= config.floor_gap_tolerance_m
    )
    collision_pass = bool(
        not scene.unsupported_contacts
        and not scene.self_collision_pairs
        and scene.maximum_foot_penetration_m <= config.penetration_tolerance_m
    )
    hull, com_floor_xy, support_margin = _support_margin(scene)
    support_pass = bool(
        support_margin is not None
        and support_margin
        >= config.minimum_support_margin_m - config.support_hull_tolerance_m
    )
    geometry_pass = bool(
        limits_pass
        and foot_pose_pass
        and jacobian_pass
        and double_support_pass
        and sole_gap_pass
        and collision_pass
        and support_pass
    )

    try:
        dynamics = _strict_json_mapping(
            backend.static_ground_dynamics(
                state,
                contact_gap_tolerance_m=config.floor_gap_tolerance_m,
                penetration_tolerance_m=config.penetration_tolerance_m,
            ),
            "static_ground_dynamics",
        )
    except Exception as exc:
        dynamics = {
            "status": "INCOMPLETE_FAIL_CLOSED",
            "feasible": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    feasible_raw = dynamics.get("feasible")
    ground_dynamics_passed: bool | None
    if feasible_raw is True:
        ground_dynamics_passed = True
    elif feasible_raw is False:
        ground_dynamics_passed = False
    else:
        ground_dynamics_passed = None

    if not geometry_pass or ground_dynamics_passed is False:
        verdict = "FAIL_STATIC_GROUNDED_READY"
    elif ground_dynamics_passed is None:
        verdict = "INCOMPLETE_FAIL_CLOSED"
    elif not bool(model_receipt["exact_mujoco_backend"]):
        verdict = "PASS_TEST_BACKEND_ONLY"
    else:
        verdict = "PASS_STATIC_GROUNDED_READY_CANDIDATE"
    target_digest = _foot_target_digest(targets)
    receipt_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_class": "diagnostic_stationary_grounded_ready_candidate",
        "trust_scope": {
            "value_class": (
                "UNTRUSTED_DIAGNOSTIC_UNTIL_CONSTRUCTION_BOUND_PUBLICATION"
            ),
            "receipt_sha_semantics": "CONTENT_INTEGRITY_NOT_AUTHENTICATION",
            "trusted_compute_base": "LOADED_CODE_AND_INTERPRETER",
        },
        "candidate_id": str(candidate_id),
        "verdict": verdict,
        "candidate": {
            "state_sha256": state_digest(state),
            "joint_pos_sha256": _array_sha256(state.joint_pos),
            "joint_pos": state.joint_pos.tolist(),
            "joint_vel": np.zeros(31, np.float64).tolist(),
            "root_pos_w": state.root_pos_w.tolist(),
            "root_quat_wxyz": state.root_quat_wxyz.tolist(),
            "root_lin_vel_w": np.zeros(3, np.float64).tolist(),
            "root_ang_vel_w": np.zeros(3, np.float64).tolist(),
            "zero_velocity_emitted": True,
        },
        "source": _jsonable(source),
        "exact_model": model_receipt,
        "foot_targets": {
            "sha256": target_digest,
            "foot_body_names": list(FOOT_BODY_NAMES),
            "rows": [
                {
                    "position_w": pose.position_w.tolist(),
                    "rotation_w": pose.rotation_w.tolist(),
                }
                for pose in targets
            ],
        },
        "static_geometry": {
            "joint_limits": {
                "passed": limits_pass,
                "maximum_excess_rad": worst_limit_excess,
            },
            "foot_pose": {
                "passed": foot_pose_pass,
                "maximum_position_residual_m": position_peak,
                "maximum_rotation_residual_rad": rotation_peak,
            },
            "leg_to_foot_jacobian": {
                "passed": jacobian_pass,
                "shape": list(jacobian.shape),
                "rank": rank,
                "condition": condition if math.isfinite(condition) else None,
                "singular_values": singular_values.tolist(),
                "rank_tolerance": config.jacobian_rank_tolerance,
            },
            "double_support": {
                "passed": double_support_pass,
                "foot_contact_count": list(scene.foot_contact_count),
                "contact_point_count": int(len(scene.contact_points_w)),
                "contact_foot_indices": (scene.contact_foot_indices.tolist()),
            },
            "sole_floor": {
                "passed": sole_gap_pass,
                "minimum_distance_m": (scene.sole_minimum_distance_m.tolist()),
                "maximum_foot_penetration_m": (scene.maximum_foot_penetration_m),
            },
            "collision": {
                "passed": collision_pass,
                "unsupported_contacts": _jsonable(scene.unsupported_contacts),
                "self_collision_pairs": _jsonable(scene.self_collision_pairs),
            },
            "support": {
                "passed": support_pass,
                "contact_hull_floor_xy_m": hull.tolist(),
                "com_projection_floor_xy_m": (
                    None if com_floor_xy is None else com_floor_xy.tolist()
                ),
                "margin_m": support_margin,
                "required_margin_m": config.minimum_support_margin_m,
            },
            "backend_details": _jsonable(scene.details or {}),
            "passed": geometry_pass,
        },
        "static_ground_dynamics": dynamics,
        "gates": {
            "exact_model_identity": (
                "PASS" if model_receipt["exact_mujoco_backend"] else "TEST_BACKEND_ONLY"
            ),
            "joint_limits": "PASS" if limits_pass else "FAIL_CLOSED",
            "foot_pose": "PASS" if foot_pose_pass else "FAIL_CLOSED",
            "leg_to_foot_jacobian": ("PASS" if jacobian_pass else "FAIL_CLOSED"),
            "double_support": ("PASS" if double_support_pass else "FAIL_CLOSED"),
            "sole_floor": "PASS" if sole_gap_pass else "FAIL_CLOSED",
            "collision": "PASS" if collision_pass else "FAIL_CLOSED",
            "support_margin": "PASS" if support_pass else "FAIL_CLOSED",
            "static_ground_dynamics": (
                "PASS"
                if ground_dynamics_passed is True
                else (
                    "FAIL_CLOSED"
                    if ground_dynamics_passed is False
                    else "INCOMPLETE_FAIL_CLOSED"
                )
            ),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "selection": {
            "selected_as_canonical_ready": False,
            "automatic_G1_or_G2_adoption": False,
            "requires_outer_comparison_across_all_five_motions": True,
        },
        "non_claims": [
            "not a final G1/G2 selection",
            "not a strike-path or C2 connector certificate",
            "not a closed-loop balance or behavior certificate",
            "not training, deployment, or hardware authorization",
        ],
        "config": {
            name: _jsonable(getattr(config, name))
            for name in config.__dataclass_fields__
        },
    }
    canonical = _canonical_json_bytes(receipt_payload)
    receipt_sha = hashlib.sha256(canonical).hexdigest()
    receipt_payload["receipt_payload_sha256"] = receipt_sha
    frozen = _deep_freeze(receipt_payload)
    if _return_draft:
        return _GroundedReadyAuditDraft(
            receipt_payload=frozen,
            receipt_sha256=receipt_sha,
            geometry_passed=geometry_pass,
            ground_dynamics_passed=ground_dynamics_passed,
        )
    exact_backend = bool(model_receipt["exact_mujoco_backend"])
    return GroundedReadyResult(
        candidate_id=str(candidate_id),
        state=state,
        target_foot_poses=targets,
        geometry_passed=geometry_pass,
        ground_dynamics_passed=ground_dynamics_passed,
        receipt=frozen,
        receipt_sha256=receipt_sha,
        verification_backend=backend if exact_backend else None,
        expected_model_identity=(expected_model_identity if exact_backend else None),
        verification_config=config if exact_backend else None,
        construction_source=source if exact_backend else None,
    )


def _revalidate_exact_result_receipt(
    result: GroundedReadyResult,
    targets: tuple[FootPose, FootPose],
    payload: Mapping[str, Any],
    *,
    backend: Any,
    expected_model_identity: Any,
    config: Any,
    construction_source: Any,
) -> None:
    """Rerun an exact claim against a live, pinned backend before accepting it."""

    if (
        type(backend) is not MujocoGroundedReadyBackend
        or not isinstance(expected_model_identity, ExactModelIdentity)
        or not isinstance(config, GroundedReadyConfig)
    ):
        raise GroundedReadyError(
            "an exact-MuJoCo receipt requires a live exact backend, "
            "pinned model identity, and audit config",
            code="EXACT_RESULT_REAUDIT_REQUIRED",
        )
    if not isinstance(construction_source, Mapping):
        raise GroundedReadyError(
            "an exact-MuJoCo diagnostic requires its construction source input",
            code="EXACT_RESULT_SOURCE_REQUIRED",
        )
    trusted_source = _jsonable(construction_source)
    embedded_source = payload.get("source")
    if not isinstance(embedded_source, Mapping) or trusted_source != _jsonable(
        embedded_source
    ):
        raise GroundedReadyError(
            "exact-MuJoCo receipt source differs from the construction input",
            code="EXACT_RESULT_SOURCE_MISMATCH",
        )
    try:
        fresh_backend = MujocoGroundedReadyBackend.load(expected_model_identity)
    except Exception as exc:
        raise GroundedReadyError(
            "fresh exact-MuJoCo backend reload failed before receipt acceptance: "
            f"{type(exc).__name__}: {exc}",
            code="EXACT_RESULT_FRESH_BACKEND_REQUIRED",
        ) from exc
    draft = _audit_and_build_result(
        result.candidate_id,
        result.state,
        targets,
        source=trusted_source,
        backend=fresh_backend,
        expected_model_identity=expected_model_identity,
        config=config,
        _return_draft=True,
    )
    if not isinstance(draft, _GroundedReadyAuditDraft):
        raise AssertionError("exact re-audit did not return an audit draft")
    rebuilt_payload = _jsonable(draft.receipt_payload)
    if (
        draft.receipt_sha256 != result.receipt_sha256
        or rebuilt_payload != _jsonable(payload)
        or draft.geometry_passed is not result.geometry_passed
        or draft.ground_dynamics_passed is not result.ground_dynamics_passed
    ):
        raise GroundedReadyError(
            "exact-MuJoCo receipt does not match a fresh live re-audit",
            code="EXACT_RESULT_REAUDIT_MISMATCH",
        )


def _overlay_state(
    base: ReadyState,
    candidate_joint_pos: np.ndarray | None,
    names: Sequence[str] | None,
    *,
    backend: GroundedReadyBackend,
) -> tuple[ReadyState, dict[str, Any]]:
    if candidate_joint_pos is None:
        if names is not None:
            raise GroundedReadyError(
                "upper_joint_names requires upper_candidate_joint_pos",
                code="INVALID_UPPER_OVERLAY",
            )
        return base, {
            "applied": False,
            "joint_names": [],
            "input_joint_pos_sha256": None,
        }
    candidate = _readonly_vector(
        candidate_joint_pos, len(backend.joint_names), "upper candidate joint_pos"
    )
    selected_names = (
        UPPER_JOINT_NAMES if names is None else tuple(str(name) for name in names)
    )
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise GroundedReadyError(
            "upper overlay joint names must be unique and non-empty",
            code="INVALID_UPPER_OVERLAY",
        )
    unknown = sorted(set(selected_names) - set(backend.joint_names))
    leg_overlap = sorted(set(selected_names) & set(LEG_JOINT_NAMES))
    if unknown or leg_overlap:
        raise GroundedReadyError(
            f"upper overlay has unknown={unknown} or leg={leg_overlap} names",
            code="INVALID_UPPER_OVERLAY",
        )
    indices = _joint_indices(backend.joint_names, selected_names)
    leg = _joint_indices(backend.joint_names, LEG_JOINT_NAMES)
    q = np.asarray(base.joint_pos, dtype=np.float64).copy()
    q[indices] = candidate[indices]
    if not np.array_equal(q[leg], base.joint_pos[leg]):
        raise GroundedReadyError(
            "upper overlay changed a leg coordinate",
            code="UPPER_OVERLAY_LOWER_MUTATION",
        )
    return (
        ReadyState(q, base.root_pos_w, base.root_quat_wxyz),
        {
            "applied": True,
            "joint_names": list(selected_names),
            "joint_indices": indices.tolist(),
            "input_joint_pos_sha256": _array_sha256(candidate),
            "copied_values_sha256": _array_sha256(candidate[indices]),
            "root_preserved": True,
            "lower_preserved": True,
        },
    )


def _verify_backend_contract(
    backend: GroundedReadyBackend,
    expected: ExactModelIdentity,
) -> dict[str, Any]:
    names = tuple(str(name) for name in backend.joint_names)
    if names != RUNTIME_JOINT_NAMES:
        raise GroundedReadyError(
            "backend joint order differs from canonical runtime order",
            code="JOINT_ORDER_MISMATCH",
        )
    actual = backend.exact_model_identity
    if not isinstance(actual, ExactModelIdentity):
        raise GroundedReadyError(
            "backend does not expose an ExactModelIdentity",
            code="MODEL_IDENTITY_MISSING",
        )
    expected_input = Path(expected.mjcf_path).expanduser().absolute()
    actual_input = Path(actual.mjcf_path).expanduser().absolute()
    try:
        expected_path = expected_input.resolve(strict=True)
        actual_path = actual_input.resolve(strict=True)
    except OSError as exc:
        raise GroundedReadyError(
            f"cannot resolve exact MJCF: {exc}",
            code="MODEL_SOURCE_INVALID",
        ) from exc
    if (
        expected_input != expected_path
        or actual_input != actual_path
        or expected_input.is_symlink()
        or actual_input.is_symlink()
    ):
        raise GroundedReadyError(
            "exact MJCF path may not contain a symlink component",
            code="MODEL_SOURCE_INVALID",
        )
    if expected_path != actual_path:
        raise GroundedReadyError(
            f"backend MJCF path mismatch: {actual_path} != {expected_path}",
            code="MODEL_IDENTITY_MISMATCH",
        )
    if not expected_path.is_file():
        raise GroundedReadyError(
            f"exact MJCF must be a regular non-symlink file: {expected_path}",
            code="MODEL_SOURCE_INVALID",
        )
    live_sha = _sha256_file(expected_path)
    scalar_fields = (
        "mjcf_sha256",
        "compiled_model_sha256",
        "path_model_binding_sha256",
        "ground_model_binding_sha256",
        "xml_model_name",
    )
    mismatches = {
        name: {
            "expected": str(getattr(expected, name)),
            "actual": str(getattr(actual, name)),
        }
        for name in scalar_fields
        if str(getattr(expected, name)) != str(getattr(actual, name))
    }
    if live_sha != expected.mjcf_sha256:
        mismatches["live_mjcf_sha256"] = {
            "expected": expected.mjcf_sha256,
            "actual": live_sha,
        }
    if mismatches:
        raise GroundedReadyError(
            f"exact model identity mismatch: {mismatches}",
            code="MODEL_IDENTITY_MISMATCH",
            report={"mismatches": mismatches},
        )
    exact_backend = type(backend) is MujocoGroundedReadyBackend
    if exact_backend:
        backend.assert_current_model_identity(expected)
    lower = _finite_backend_vector(backend.position_lower, len(names), "position_lower")
    upper = _finite_backend_vector(backend.position_upper, len(names), "position_upper")
    if np.any(lower > upper):
        raise GroundedReadyError(
            "backend joint position limits are inverted",
            code="INVALID_MODEL_LIMITS",
        )
    return {
        "status": (
            "PASS_EXACT_MUJOCO" if exact_backend else "PASS_PROTOCOL_TEST_BACKEND_ONLY"
        ),
        "exact_mujoco_backend": exact_backend,
        "backend_type": (f"{type(backend).__module__}.{type(backend).__qualname__}"),
        "mjcf_path": str(expected_path),
        "mjcf_sha256": expected.mjcf_sha256,
        "compiled_model_sha256": expected.compiled_model_sha256,
        "path_model_binding_sha256": (expected.path_model_binding_sha256),
        "ground_model_binding_sha256": (expected.ground_model_binding_sha256),
        "xml_model_name": expected.xml_model_name,
        "joint_order": list(names),
        "joint_position_lower_sha256": _array_sha256(lower),
        "joint_position_upper_sha256": _array_sha256(upper),
    }


def _finite_difference_leg_jacobian(
    state: ReadyState,
    targets: tuple[FootPose, FootPose],
    leg_indices: np.ndarray,
    *,
    backend: GroundedReadyBackend,
    step: float,
) -> np.ndarray:
    jacobian = np.empty((12, len(leg_indices)), dtype=np.float64)
    lower = _finite_backend_vector(
        backend.position_lower, len(backend.joint_names), "position_lower"
    )
    upper = _finite_backend_vector(
        backend.position_upper, len(backend.joint_names), "position_upper"
    )
    base_residual = _foot_pose_residual(backend.foot_poses(state), targets)
    for column, joint in enumerate(leg_indices):
        index = int(joint)
        value = float(state.joint_pos[index])
        can_plus = value + step <= float(upper[index]) + 1.0e-15
        can_minus = value - step >= float(lower[index]) - 1.0e-15
        plus_residual: np.ndarray | None = None
        minus_residual: np.ndarray | None = None
        if can_plus:
            plus = np.asarray(state.joint_pos, np.float64).copy()
            plus[index] += step
            plus_residual = _foot_pose_residual(
                backend.foot_poses(
                    ReadyState(plus, state.root_pos_w, state.root_quat_wxyz)
                ),
                targets,
            )
        if can_minus:
            minus = np.asarray(state.joint_pos, np.float64).copy()
            minus[index] -= step
            minus_residual = _foot_pose_residual(
                backend.foot_poses(
                    ReadyState(minus, state.root_pos_w, state.root_quat_wxyz)
                ),
                targets,
            )
        if plus_residual is not None and minus_residual is not None:
            jacobian[:, column] = (plus_residual - minus_residual) / (2.0 * step)
        elif plus_residual is not None:
            jacobian[:, column] = (plus_residual - base_residual) / step
        elif minus_residual is not None:
            jacobian[:, column] = (base_residual - minus_residual) / step
        else:
            jacobian[:, column] = 0.0
    if not np.isfinite(jacobian).all():
        raise GroundedReadyError(
            "leg-to-foot finite-difference Jacobian contains NaN/Inf",
            code="FOOT_JACOBIAN_INVALID",
        )
    return jacobian


def _foot_pose_residual(
    current: Sequence[FootPose],
    targets: Sequence[FootPose],
) -> np.ndarray:
    if len(current) != 2 or len(targets) != 2:
        raise GroundedReadyError(
            "exactly two foot poses are required",
            code="INVALID_FOOT_POSE_COUNT",
        )
    rows: list[np.ndarray] = []
    for actual, target in zip(current, targets):
        rows.append(actual.position_w - target.position_w)
        rows.append(_so3_log(target.rotation_w.T @ actual.rotation_w))
    return np.concatenate(rows)


def _residual_peaks(residual: np.ndarray) -> tuple[float, float]:
    row = np.asarray(residual, np.float64).reshape(2, 6)
    return (
        float(np.max(np.linalg.norm(row[:, :3], axis=1))),
        float(np.max(np.linalg.norm(row[:, 3:], axis=1))),
    )


def _jacobian_metrics(
    jacobian: np.ndarray, rank_tolerance: float
) -> tuple[int, float, np.ndarray]:
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.sum(singular > rank_tolerance))
    condition = (
        math.inf
        if len(singular) == 0 or singular[-1] <= rank_tolerance
        else float(singular[0] / singular[-1])
    )
    return rank, condition, singular


def _support_margin(
    scene: StaticScene,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    if len(scene.contact_points_w) < 3:
        return np.empty((0, 2), np.float64), None, None
    relative = scene.contact_points_w - scene.floor_origin_w[None, :]
    projected = np.column_stack(
        (
            relative @ scene.floor_basis_w[:, 0],
            relative @ scene.floor_basis_w[:, 1],
        )
    )
    hull = _convex_hull(projected)
    if len(hull) < 3:
        return hull, None, None
    com_relative = scene.com_w - scene.floor_origin_w
    com = np.asarray(
        [
            float(com_relative @ scene.floor_basis_w[:, 0]),
            float(com_relative @ scene.floor_basis_w[:, 1]),
        ],
        np.float64,
    )
    margins = []
    for index, point in enumerate(hull):
        following = hull[(index + 1) % len(hull)]
        edge = following - point
        length = float(np.linalg.norm(edge))
        if length <= 1.0e-12:
            continue
        delta = com - point
        signed_twice_area = float(edge[0] * delta[1] - edge[1] * delta[0])
        margins.append(signed_twice_area / length)
    return hull, com, None if not margins else float(min(margins))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    rows = sorted(
        {(float(row[0]), float(row[1])) for row in np.asarray(points, np.float64)}
    )
    if len(rows) < 3:
        return np.asarray(rows, np.float64).reshape(-1, 2)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in rows:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(rows):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def _interpolate_rotation(
    first: np.ndarray, second: np.ndarray, alpha: float
) -> np.ndarray:
    return first @ _so3_exp(alpha * _so3_log(first.T @ second))


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        np.float64,
    )
    if angle < 1.0e-9:
        return 0.5 * vee
    if math.pi - angle < 1.0e-6:
        symmetric = 0.5 * (matrix + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        pivot = int(np.argmax(axis))
        if axis[pivot] <= 1.0e-10:
            raise GroundedReadyError(
                "cannot resolve near-pi SO(3) logarithm",
                code="ROTATION_LOG_SINGULAR",
            )
        for index in range(3):
            if index != pivot:
                axis[index] = symmetric[pivot, index] / axis[pivot]
        axis /= np.linalg.norm(axis)
        skew_hint = vee
        if float(axis @ skew_hint) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * math.sin(angle))) * vee


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, np.float64)
    angle = float(np.linalg.norm(value))
    if angle < 1.0e-12:
        skew = _skew(value)
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    axis = value / angle
    skew = _skew(axis)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = map(float, value)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        np.float64,
    )


def _foot_target_digest(targets: Sequence[FootPose]) -> str:
    digest = hashlib.sha256()
    for index, target in enumerate(targets):
        _hash_array(digest, f"foot{index}_position", target.position_w)
        _hash_array(digest, f"foot{index}_rotation", target.rotation_w)
    return digest.hexdigest()


def _joint_indices(joint_names: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    names = tuple(str(name) for name in joint_names)
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise GroundedReadyError(
            f"backend is missing joints {missing}",
            code="JOINT_ORDER_MISMATCH",
        )
    return np.asarray([lookup[name] for name in selected], np.int64)


def _readonly_vector(value: Any, length: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.shape != (length,)
        or raw.dtype.kind not in "iuf"
        or raw.dtype.kind == "b"
        or np.iscomplexobj(raw)
    ):
        raise GroundedReadyError(
            f"{label} must be one real vector of shape ({length},)",
            code="INVALID_ARRAY",
        )
    out = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(out).all():
        raise GroundedReadyError(f"{label} contains NaN/Inf", code="INVALID_ARRAY")
    out.setflags(write=False)
    return out


def _readonly_matrix(value: Any, width: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[1] != width
        or raw.dtype.kind not in "iuf"
        or raw.dtype.kind == "b"
        or np.iscomplexobj(raw)
    ):
        raise GroundedReadyError(
            f"{label} must be one real matrix with width {width}",
            code="INVALID_ARRAY",
        )
    out = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(out).all():
        raise GroundedReadyError(f"{label} contains NaN/Inf", code="INVALID_ARRAY")
    out.setflags(write=False)
    return out


def _finite_backend_vector(value: Any, length: int, label: str) -> np.ndarray:
    return np.asarray(_readonly_vector(value, length, label), np.float64)


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            f"{label} must contain 64 lowercase hexadecimal SHA-256 digits"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "array", value)
    return digest.hexdigest()


def _hash_array(digest: Any, label: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _strict_json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GroundedReadyError(
            f"{label} must return a mapping",
            code="INVALID_BACKEND_REPORT",
        )
    payload = _jsonable(value)
    _canonical_json_bytes(payload)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GroundedReadyError("receipt contains NaN/Inf", code="INVALID_RECEIPT")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise GroundedReadyError(
        f"receipt contains unsupported type {type(value).__name__}",
        code="INVALID_RECEIPT",
    )


def _validate_result_receipt_payload(
    result: GroundedReadyResult,
    targets: tuple[FootPose, FootPose],
    payload: Mapping[str, Any],
) -> bool:
    """Recompute every receipt field that is derivable from retained state."""

    def malformed(reason: str) -> GroundedReadyError:
        return GroundedReadyError(
            f"result receipt is inconsistent: {reason}",
            code="INVALID_RESULT_RECEIPT",
        )

    def exact_array(value: Any, expected: np.ndarray, label: str) -> None:
        try:
            actual = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise malformed(f"{label} is not numeric") from exc
        if (
            actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.array_equal(actual, expected)
        ):
            raise malformed(f"{label} differs from retained state")

    if type(result.geometry_passed) is not bool or (
        result.ground_dynamics_passed is not None
        and type(result.ground_dynamics_passed) is not bool
    ):
        raise malformed("result gate fields must be bool or None")
    try:
        embedded_seal = str(payload["receipt_payload_sha256"])
        embedded_candidate_id = str(payload["candidate_id"])
        candidate = payload["candidate"]
        foot_targets = payload["foot_targets"]
        static = payload["static_geometry"]
        dynamics = payload["static_ground_dynamics"]
        exact_model = payload["exact_model"]
        gates = payload["gates"]
        authorization = payload["authorization"]
        verdict = str(payload["verdict"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise malformed(f"missing required section: {exc}") from exc
    for label, section in (
        ("candidate", candidate),
        ("foot_targets", foot_targets),
        ("static_geometry", static),
        ("static_ground_dynamics", dynamics),
        ("exact_model", exact_model),
        ("gates", gates),
        ("authorization", authorization),
    ):
        if not isinstance(section, Mapping):
            raise malformed(f"{label} must be a mapping")

    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    recomputed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if embedded_seal != result.receipt_sha256 or recomputed != result.receipt_sha256:
        raise malformed("payload SHA-256 seal does not match")
    if embedded_candidate_id != result.candidate_id:
        raise malformed("candidate_id differs from result")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_class")
        != "diagnostic_stationary_grounded_ready_candidate"
    ):
        raise malformed("schema/artifact class differs from contract")
    if payload.get("trust_scope") != {
        "value_class": ("UNTRUSTED_DIAGNOSTIC_UNTIL_CONSTRUCTION_BOUND_PUBLICATION"),
        "receipt_sha_semantics": "CONTENT_INTEGRITY_NOT_AUTHENTICATION",
        "trusted_compute_base": "LOADED_CODE_AND_INTERPRETER",
    }:
        raise malformed("trust scope differs from the diagnostic contract")

    state = result.state
    if candidate.get("state_sha256") != state_digest(state):
        raise malformed("candidate state digest differs from retained state")
    if candidate.get("joint_pos_sha256") != _array_sha256(state.joint_pos):
        raise malformed("candidate joint digest differs from retained state")
    exact_array(candidate.get("joint_pos"), state.joint_pos, "joint_pos")
    exact_array(
        candidate.get("joint_vel"),
        np.zeros(31, np.float64),
        "joint_vel",
    )
    exact_array(candidate.get("root_pos_w"), state.root_pos_w, "root_pos_w")
    exact_array(
        candidate.get("root_quat_wxyz"),
        state.root_quat_wxyz,
        "root_quat_wxyz",
    )
    exact_array(
        candidate.get("root_lin_vel_w"),
        np.zeros(3, np.float64),
        "root_lin_vel_w",
    )
    exact_array(
        candidate.get("root_ang_vel_w"),
        np.zeros(3, np.float64),
        "root_ang_vel_w",
    )
    if candidate.get("zero_velocity_emitted") is not True:
        raise malformed("zero_velocity_emitted must be true")

    if foot_targets.get("sha256") != _foot_target_digest(targets):
        raise malformed("foot target digest differs from retained targets")
    if tuple(foot_targets.get("foot_body_names", ())) != FOOT_BODY_NAMES:
        raise malformed("foot target body order differs from contract")
    target_rows = foot_targets.get("rows")
    if not isinstance(target_rows, list) or len(target_rows) != 2:
        raise malformed("foot target rows must contain exactly two entries")
    for index, (row, expected) in enumerate(zip(target_rows, targets)):
        if not isinstance(row, Mapping):
            raise malformed(f"foot target row {index} is not a mapping")
        exact_array(
            row.get("position_w"),
            expected.position_w,
            f"foot target {index} position",
        )
        exact_array(
            row.get("rotation_w"),
            expected.rotation_w,
            f"foot target {index} rotation",
        )

    exact_backend = exact_model.get("exact_mujoco_backend")
    if type(exact_backend) is not bool:
        raise malformed("exact_mujoco_backend must be boolean")
    expected_model_status = (
        "PASS_EXACT_MUJOCO" if exact_backend else "PASS_PROTOCOL_TEST_BACKEND_ONLY"
    )
    if exact_model.get("status") != expected_model_status:
        raise malformed("exact model status contradicts backend class")
    if tuple(exact_model.get("joint_order", ())) != RUNTIME_JOINT_NAMES:
        raise malformed("exact model joint order differs from runtime order")
    for label in (
        "mjcf_sha256",
        "compiled_model_sha256",
        "path_model_binding_sha256",
        "ground_model_binding_sha256",
        "joint_position_lower_sha256",
        "joint_position_upper_sha256",
    ):
        try:
            _require_sha256(str(exact_model[label]), f"exact_model.{label}")
        except (KeyError, ValueError) as exc:
            raise malformed(f"invalid exact model digest {label}") from exc

    static_keys = (
        "joint_limits",
        "foot_pose",
        "leg_to_foot_jacobian",
        "double_support",
        "sole_floor",
        "collision",
        "support",
    )
    static_pass: dict[str, bool] = {}
    for label in static_keys:
        section = static.get(label)
        if not isinstance(section, Mapping) or type(section.get("passed")) is not bool:
            raise malformed(f"static gate {label} lacks a boolean result")
        static_pass[label] = bool(section["passed"])
    derived_geometry = all(static_pass.values())
    if (
        static.get("passed") is not derived_geometry
        or result.geometry_passed is not derived_geometry
    ):
        raise malformed("aggregate static geometry result is inconsistent")

    embedded_feasible = dynamics.get("feasible")
    if embedded_feasible not in (True, False, None) or (
        embedded_feasible is not None and type(embedded_feasible) is not bool
    ):
        raise malformed("dynamics feasible must be true, false, or null")
    if embedded_feasible is not result.ground_dynamics_passed:
        raise malformed("ground-dynamics feasibility differs from result")
    expected_dynamics_status = (
        "PASS_STATIC_GROUND_CONTACT_LP"
        if embedded_feasible is True
        else (
            "FAIL_STATIC_GROUND_CONTACT_LP"
            if embedded_feasible is False
            else "INCOMPLETE_FAIL_CLOSED"
        )
    )
    if dynamics.get("status") != expected_dynamics_status:
        raise malformed("ground-dynamics status contradicts feasibility")

    expected_gates = {
        "exact_model_identity": ("PASS" if exact_backend else "TEST_BACKEND_ONLY"),
        "joint_limits": ("PASS" if static_pass["joint_limits"] else "FAIL_CLOSED"),
        "foot_pose": ("PASS" if static_pass["foot_pose"] else "FAIL_CLOSED"),
        "leg_to_foot_jacobian": (
            "PASS" if static_pass["leg_to_foot_jacobian"] else "FAIL_CLOSED"
        ),
        "double_support": ("PASS" if static_pass["double_support"] else "FAIL_CLOSED"),
        "sole_floor": ("PASS" if static_pass["sole_floor"] else "FAIL_CLOSED"),
        "collision": ("PASS" if static_pass["collision"] else "FAIL_CLOSED"),
        "support_margin": ("PASS" if static_pass["support"] else "FAIL_CLOSED"),
        "static_ground_dynamics": (
            "PASS"
            if embedded_feasible is True
            else (
                "FAIL_CLOSED"
                if embedded_feasible is False
                else "INCOMPLETE_FAIL_CLOSED"
            )
        ),
    }
    if dict(gates) != expected_gates:
        raise malformed("gates mapping is missing or contradicts evidence")
    expected_verdict = (
        "FAIL_STATIC_GROUNDED_READY"
        if not derived_geometry or embedded_feasible is False
        else (
            "INCOMPLETE_FAIL_CLOSED"
            if embedded_feasible is None
            else (
                "PASS_STATIC_GROUNDED_READY_CANDIDATE"
                if exact_backend
                else "PASS_TEST_BACKEND_ONLY"
            )
        )
    )
    if verdict != expected_verdict:
        raise malformed("verdict contradicts exact/static gate evidence")

    if (
        authorization.get("training_authorized") is not False
        or authorization.get("deployment_authorized") is not False
        or authorization.get("hardware_authorized") is not False
    ):
        raise GroundedReadyError(
            "result receipt must deny training/deployment/hardware",
            code="AUTHORIZATION_FORBIDDEN",
        )
    selection = payload.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selected_as_canonical_ready") is not False
        or selection.get("automatic_G1_or_G2_adoption") is not False
        or selection.get("requires_outer_comparison_across_all_five_motions")
        is not True
    ):
        raise malformed("selection section may not auto-adopt a candidate")
    return bool(exact_backend)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


class MujocoGroundedReadyBackend:
    """Exact A3 implementation; imports MuJoCo/SciPy only when requested."""

    joint_names = RUNTIME_JOINT_NAMES

    @classmethod
    def load(
        cls, expected_identity: ExactModelIdentity
    ) -> "MujocoGroundedReadyBackend":
        try:
            import mujoco
            import canonical_mujoco_path_adapter as path_adapter
            import canonical_torque_path_topp as torque_topp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GroundedReadyError(
                f"exact MuJoCo backend is unavailable: {exc}",
                code="MUJOCO_BACKEND_UNAVAILABLE",
            ) from exc
        try:
            model, binding = path_adapter.load_exact_mujoco_model(
                mjcf_path=expected_identity.mjcf_path,
                expected_mjcf_sha256=expected_identity.mjcf_sha256,
                expected_compiled_model_sha256=(
                    expected_identity.compiled_model_sha256
                ),
                expected_xml_model_name=expected_identity.xml_model_name,
                mujoco_module=mujoco,
            )
            ground_binding = torque_topp._mujoco_model_binding(model)
        except Exception as exc:
            raise GroundedReadyError(
                f"cannot load/bind exact A3 MuJoCo model: {exc}",
                code="MODEL_IDENTITY_MISMATCH",
            ) from exc
        actual = ExactModelIdentity(
            mjcf_path=binding.mjcf_path,
            mjcf_sha256=binding.mjcf_sha256,
            compiled_model_sha256=binding.compiled_model_sha256,
            path_model_binding_sha256=binding.model_binding_sha256,
            ground_model_binding_sha256=ground_binding,
            xml_model_name=binding.xml_model_name,
        )
        backend = cls(
            mujoco_module=mujoco,
            model=model,
            binding=binding,
            exact_model_identity=actual,
        )
        _verify_backend_contract(backend, expected_identity)
        return backend

    def __init__(
        self,
        *,
        mujoco_module: Any,
        model: Any,
        binding: Any,
        exact_model_identity: ExactModelIdentity,
    ) -> None:
        self._mujoco = mujoco_module
        self.model = model
        self._binding = binding
        self.exact_model_identity = exact_model_identity
        self._data = mujoco_module.MjData(model)
        self.position_lower = np.asarray(
            model.jnt_range[np.asarray(binding.joint_ids, np.int64), 0],
            np.float64,
        )
        self.position_upper = np.asarray(
            model.jnt_range[np.asarray(binding.joint_ids, np.int64), 1],
            np.float64,
        )
        self._floor_geom = self._name_id(
            mujoco_module.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME
        )
        self._root_body = self._name_id(mujoco_module.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME)
        self._foot_bodies = tuple(
            self._name_id(mujoco_module.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODY_NAMES
        )
        self._foot_collision_geoms = tuple(
            self._name_id(mujoco_module.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_COLLISION_GEOM_NAMES
        )
        self._foot_geom_sets = tuple(
            frozenset(
                int(index)
                for index in range(int(model.ngeom))
                if int(model.geom_bodyid[index]) == body
                and int(model.geom_contype[index]) != 0
            )
            for body in self._foot_bodies
        )
        if any(not rows for rows in self._foot_geom_sets):
            raise GroundedReadyError(
                "exact A3 foot body has no collision geometry",
                code="MODEL_FOOT_GEOMETRY_MISSING",
            )

    def assert_current_model_identity(self, expected: ExactModelIdentity) -> None:
        """Rebind both live and freshly compiled models before each audit."""

        try:
            import canonical_mujoco_path_adapter as path_adapter
            import canonical_torque_path_topp as torque_topp

            current = path_adapter.bind_exact_mujoco_model(
                self._mujoco,
                self.model,
                mjcf_path=expected.mjcf_path,
                expected_mjcf_sha256=expected.mjcf_sha256,
                expected_compiled_model_sha256=(expected.compiled_model_sha256),
                expected_xml_model_name=expected.xml_model_name,
            )
            fresh_model, fresh = path_adapter.load_exact_mujoco_model(
                mjcf_path=expected.mjcf_path,
                expected_mjcf_sha256=expected.mjcf_sha256,
                expected_compiled_model_sha256=(expected.compiled_model_sha256),
                expected_xml_model_name=expected.xml_model_name,
                mujoco_module=self._mujoco,
            )
            current_ground = torque_topp._mujoco_model_binding(self.model)
            fresh_ground = torque_topp._mujoco_model_binding(fresh_model)
        except Exception as exc:
            raise GroundedReadyError(
                f"exact model changed after backend construction: {exc}",
                code="MODEL_IDENTITY_MISMATCH",
            ) from exc
        rows = {
            "current_path": current.model_binding_sha256,
            "fresh_path": fresh.model_binding_sha256,
            "current_ground": current_ground,
            "fresh_ground": fresh_ground,
        }
        expected_rows = {
            "current_path": expected.path_model_binding_sha256,
            "fresh_path": expected.path_model_binding_sha256,
            "current_ground": expected.ground_model_binding_sha256,
            "fresh_ground": expected.ground_model_binding_sha256,
        }
        if rows != expected_rows:
            raise GroundedReadyError(
                f"live/fresh exact model identity mismatch: {rows}",
                code="MODEL_IDENTITY_MISMATCH",
                report={"expected": expected_rows, "actual": rows},
            )

    def _name_id(self, object_type: Any, name: str) -> int:
        value = int(self._mujoco.mj_name2id(self.model, object_type, str(name)))
        if value < 0:
            raise GroundedReadyError(
                f"exact model is missing {name!r}",
                code="MODEL_NAME_MISSING",
            )
        return value

    def _install(self, state: ReadyState) -> None:
        data = self._data
        data.qpos[:] = np.asarray(self.model.qpos0, np.float64)
        data.qpos[0:3] = state.root_pos_w
        data.qpos[3:7] = state.root_quat_wxyz
        data.qpos[np.asarray(self._binding.joint_qpos_adrs, np.int64)] = state.joint_pos
        data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, data)

    def _qpos(self, state: ReadyState) -> np.ndarray:
        qpos = np.asarray(self.model.qpos0, np.float64).copy()
        qpos[0:3] = state.root_pos_w
        qpos[3:7] = state.root_quat_wxyz
        qpos[np.asarray(self._binding.joint_qpos_adrs, np.int64)] = state.joint_pos
        return qpos

    def foot_poses(self, state: ReadyState) -> tuple[FootPose, FootPose]:
        self._install(state)
        return tuple(
            FootPose(
                np.asarray(self._data.xpos[body], np.float64),
                np.asarray(self._data.xmat[body], np.float64).reshape(3, 3),
            )
            for body in self._foot_bodies
        )  # type: ignore[return-value]

    def flat_foot_targets(
        self,
        state: ReadyState,
        *,
        contact_preload_m: float,
    ) -> tuple[FootPose, FootPose]:
        self._install(state)
        floor_origin = np.asarray(self._data.geom_xpos[self._floor_geom], np.float64)
        floor_basis = np.asarray(
            self._data.geom_xmat[self._floor_geom], np.float64
        ).reshape(3, 3)
        rows: list[FootPose] = []
        for body, geom in zip(self._foot_bodies, self._foot_collision_geoms):
            body_pos = np.asarray(self._data.xpos[body], np.float64)
            body_rotation = np.asarray(self._data.xmat[body], np.float64).reshape(3, 3)
            body_in_floor = floor_basis.T @ body_rotation
            yaw = math.atan2(
                float(body_in_floor[1, 0]),
                float(body_in_floor[0, 0]),
            )
            target_rotation = floor_basis @ _rotation_z(yaw)
            body_floor = floor_basis.T @ (body_pos - floor_origin)
            local_vertices = self._collision_vertices_in_body(
                geom, body, body_pos, body_rotation
            )
            rotated = local_vertices @ target_rotation.T
            target_floor = body_floor.copy()
            # ``rotated`` above contains body-local vertices rotated about the
            # world origin, so only its normal projection is relevant.
            target_floor[2] = -float(np.min(rotated @ floor_basis[:, 2])) - float(
                contact_preload_m
            )
            target_position = floor_origin + floor_basis @ target_floor
            rows.append(FootPose(target_position, target_rotation))
        return tuple(rows)  # type: ignore[return-value]

    def _collision_vertices_in_body(
        self,
        geom: int,
        body: int,
        body_pos: np.ndarray,
        body_rotation: np.ndarray,
    ) -> np.ndarray:
        model = self.model
        mesh = int(model.geom_dataid[geom])
        if mesh < 0:
            raise GroundedReadyError(
                "foot collision geom is not a compiled mesh",
                code="MODEL_FOOT_GEOMETRY_MISSING",
            )
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local_geom = np.asarray(model.mesh_vert[start : start + count], np.float64)
        geom_rotation = np.asarray(self._data.geom_xmat[geom], np.float64).reshape(3, 3)
        geom_position = np.asarray(self._data.geom_xpos[geom], np.float64)
        world = local_geom @ geom_rotation.T + geom_position
        del body
        return (world - body_pos[None, :]) @ body_rotation

    def static_scene(
        self,
        state: ReadyState,
        *,
        contact_gap_tolerance_m: float,
        penetration_tolerance_m: float,
    ) -> StaticScene:
        self._install(state)
        model = self.model
        data = self._data
        floor_origin = np.asarray(data.geom_xpos[self._floor_geom], np.float64)
        floor_basis = np.asarray(data.geom_xmat[self._floor_geom], np.float64).reshape(
            3, 3
        )
        counts = [0, 0]
        points: list[np.ndarray] = []
        foot_indices: list[int] = []
        unsupported: list[dict[str, Any]] = []
        self_collisions: list[dict[str, Any]] = []
        maximum_penetration = 0.0
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            distance = float(contact.dist)
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair = {geom1, geom2}
            matched: int | None = None
            if self._floor_geom in pair:
                robot_geom = geom2 if geom1 == self._floor_geom else geom1
                for foot in (0, 1):
                    if robot_geom in self._foot_geom_sets[foot]:
                        matched = foot
                        break
            if matched is not None and distance <= float(contact_gap_tolerance_m):
                counts[matched] += 1
                points.append(np.asarray(contact.pos, np.float64).copy())
                foot_indices.append(matched)
                maximum_penetration = max(maximum_penetration, max(0.0, -distance))
            elif distance <= 0.0:
                row = {
                    "contact_index": index,
                    "geom_ids": [geom1, geom2],
                    "distance_m": distance,
                }
                if self._floor_geom in pair:
                    unsupported.append(row)
                else:
                    self_collisions.append(row)
        sole = []
        for body, geom in zip(self._foot_bodies, self._foot_collision_geoms):
            body_pos = np.asarray(data.xpos[body], np.float64)
            body_rotation = np.asarray(data.xmat[body], np.float64).reshape(3, 3)
            local = self._collision_vertices_in_body(
                geom, body, body_pos, body_rotation
            )
            world = local @ body_rotation.T + body_pos
            sole.append(
                float(np.min((world - floor_origin[None, :]) @ floor_basis[:, 2]))
            )
        return StaticScene(
            foot_contact_count=(counts[0], counts[1]),
            contact_points_w=np.asarray(points, np.float64).reshape(-1, 3),
            contact_foot_indices=np.asarray(foot_indices, np.int64),
            floor_origin_w=floor_origin,
            floor_basis_w=floor_basis,
            sole_minimum_distance_m=np.asarray(sole, np.float64),
            com_w=np.asarray(data.subtree_com[self._root_body], np.float64),
            maximum_foot_penetration_m=maximum_penetration,
            unsupported_contacts=tuple(unsupported),
            self_collision_pairs=tuple(self_collisions),
            details={
                "active_contact_count": int(data.ncon),
                "foot_collision_geom_names": list(FOOT_COLLISION_GEOM_NAMES),
            },
        )

    def static_ground_dynamics(
        self,
        state: ReadyState,
        *,
        contact_gap_tolerance_m: float,
        penetration_tolerance_m: float,
    ) -> Mapping[str, Any]:
        try:
            import scipy.optimize  # noqa: F401
            import canonical_torque_path_topp as torque_topp
        except ImportError as exc:
            return {
                "status": "INCOMPLETE_FAIL_CLOSED",
                "feasible": None,
                "reason": f"missing CPU dependency: {exc}",
            }
        try:
            config = torque_topp.GroundContactConfig(
                expected_model_binding=(
                    self.exact_model_identity.ground_model_binding_sha256
                ),
                model_source_path=str(
                    Path(self.exact_model_identity.mjcf_path).expanduser().resolve()
                ),
                expected_source_sha256=(self.exact_model_identity.mjcf_sha256),
                contact_gap_tolerance_m=float(contact_gap_tolerance_m),
                penetration_tolerance_m=float(penetration_tolerance_m),
            )
            solver = torque_topp.MujocoGroundContactLPSolver(self.model, config)
            contract = torque_topp.direct_actuator_contract_from_mujoco(
                self.model,
                support_mode="ground",
                contact_mode="double_support_floor",
                fixed_lp_solver="scipy.optimize.linprog:highs",
            )
            (
                lower,
                upper,
                actuated,
                limit_report,
            ) = torque_topp._resolve_grounded_actuator_limits(
                contract, int(self.model.nv)
            )
            solution = solver.solve(
                self._qpos(state),
                np.zeros(int(self.model.nv), np.float64),
                np.zeros(int(self.model.nv), np.float64),
                actuated,
                lower,
                upper,
                np.full(int(self.model.nv), 1.0e6, np.float64),
                path_tangent=np.zeros(int(self.model.nv), np.float64),
            )
            return {
                "status": (
                    "PASS_STATIC_GROUND_CONTACT_LP"
                    if solution.feasible
                    else "FAIL_STATIC_GROUND_CONTACT_LP"
                ),
                "feasible": bool(solution.feasible),
                "equality_residual": float(solution.equality_residual),
                "root_residual": float(solution.root_residual),
                "actuator_limit_contract": limit_report,
                "solver_report": solution.report,
            }
        except Exception as exc:
            return {
                "status": "INCOMPLETE_FAIL_CLOSED",
                "feasible": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def vendor_key_state(self, key_index: int = 0) -> ReadyState:
        """Extract one exact compiled-model keyframe as a zero-speed state."""

        index = int(key_index)
        if index < 0 or index >= int(self.model.nkey):
            raise GroundedReadyError(
                f"vendor key index {index} is out of range",
                code="VENDOR_KEY_MISSING",
            )
        key_velocity = np.asarray(self.model.key_qvel[index], np.float64)
        if (
            key_velocity.shape != (int(self.model.nv),)
            or not np.isfinite(key_velocity).all()
            or np.max(np.abs(key_velocity)) > 1.0e-12
        ):
            raise GroundedReadyError(
                "vendor keyframe does not encode an exact zero-speed state",
                code="VENDOR_KEY_NOT_STATIC",
            )
        qpos = np.asarray(self.model.key_qpos[index], np.float64)
        joints = qpos[np.asarray(self._binding.joint_qpos_adrs, np.int64)]
        return ReadyState(joints, qpos[0:3], qpos[3:7])


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        np.float64,
    )
