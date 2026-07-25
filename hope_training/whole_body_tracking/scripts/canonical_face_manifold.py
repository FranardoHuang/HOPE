#!/usr/bin/env python3
"""Solve a signed racket-face manifold on the exact vendor MuJoCo model.

The public API operates entirely in memory.  For each source backhand frame it:

1. evaluates the exact ``right_racket`` MuJoCo site;
2. keeps that site's world position fixed;
3. flips signed raw-A (the site's local ``+Y`` axis);
4. optionally fixes the complete target orientation
   ``R_target = R_source @ diag(1, -1, -1)``;
5. changes only the named seven-joint right striking chain;
6. preserves diverse projected solutions at every frame and backtracks over
   the complete window instead of committing to a numerical anchor; and
7. ranks full-window branches first by an exact-plant, URDF-bound optimistic
   ready-connector time lower bound, then by window smoothness.

Holding the site position at every source sample preserves the source
finite-difference site velocity.  This module verifies that invariant after
solving the complete window.  It does not write NPZ/FK data, publish assets,
retime a trajectory, or claim collision, dynamics, balance, training, deploy,
or hardware authorization.

``mujoco`` is imported only when :class:`MujocoRightRacketBackend` is created.
Pure helpers and an injected kinematics backend therefore remain testable on a
host without MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np


RIGHT_STRIKE_CHAIN = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CANONICAL_RACKET_SITE = "right_racket"
CANONICAL_RACKET_SITE_POS_WRIST_M = np.asarray(
    [0.210210, 0.032078, 0.032036], dtype=np.float64
)
FULL_FACE_FLIP_LOCAL = np.diag([1.0, -1.0, -1.0])
Mode = Literal["normal", "full-so3"]


class FaceManifoldError(RuntimeError):
    """The request or a computed frame violates the fail-closed contract."""


class RightRacketBackend(Protocol):
    """Minimal kinematics surface consumed by the hierarchical solver."""

    joint_names: tuple[str, ...]
    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity_limit: np.ndarray
    effort_limit: np.ndarray
    root_body_name: str

    def site_pose(
        self,
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return world site position and world-from-site rotation."""

    def diagonal_dynamics(
        self,
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return joint-space ``Mjj`` and ``abs(bias(q, qdot=0))``."""


@dataclass(frozen=True)
class FaceManifoldConfig:
    """Numerical and fail-closed thresholds for one complete window."""

    mode: Mode = "normal"
    position_tolerance_m: float = 2.0e-6
    normal_tolerance_rad: float = 2.0e-5
    orientation_tolerance_rad: float = 2.0e-5
    velocity_tolerance_mps: float = 5.0e-4
    joint_limit_tolerance_rad: float = 1.0e-10
    max_step_rad: float = math.radians(35.0)
    first_frame_max_delta_from_ready_rad: float | None = None
    finite_difference_step_rad: float = 1.0e-5
    position_residual_scale_m: float = 0.1
    damping: float = 1.0e-5
    max_iterations: int = 240
    random_restarts: int = 12
    random_seed: int = 20260724
    max_solver_step_rad: float = 0.18
    candidate_dedup_tolerance_rad: float = 2.0e-4
    candidate_pool_limit: int = 64
    continuation_passes: int = 1
    connector_dynamics_samples: int = 41
    velocity_limit_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("normal", "full-so3"):
            raise ValueError(f"unsupported face-manifold mode {self.mode!r}")
        positive = (
            "position_tolerance_m",
            "normal_tolerance_rad",
            "orientation_tolerance_rad",
            "velocity_tolerance_mps",
            "finite_difference_step_rad",
            "position_residual_scale_m",
            "damping",
            "max_step_rad",
            "max_solver_step_rad",
            "candidate_dedup_tolerance_rad",
            "velocity_limit_fraction",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.first_frame_max_delta_from_ready_rad is not None
            and (
                not np.isfinite(self.first_frame_max_delta_from_ready_rad)
                or self.first_frame_max_delta_from_ready_rad <= 0.0
            )
        ):
            raise ValueError(
                "first_frame_max_delta_from_ready_rad must be positive or None"
            )
        if self.max_iterations <= 0 or self.random_restarts < 0:
            raise ValueError("iteration/restart counts must be non-negative")
        if self.candidate_pool_limit <= 0 or self.continuation_passes < 0:
            raise ValueError("candidate pool/pass counts must be non-negative")
        if self.connector_dynamics_samples < 2:
            raise ValueError("connector_dynamics_samples must be at least two")
        if self.velocity_limit_fraction > 1.0:
            raise ValueError("velocity_limit_fraction cannot exceed the URDF limit")


@dataclass(frozen=True)
class FrameDiagnostics:
    """Checked residuals for one source/output frame pair."""

    frame_index: int
    position_residual_m: float
    normal_residual_rad: float
    orientation_residual_rad: float | None
    max_joint_limit_violation_rad: float
    max_step_from_previous_rad: float | None
    ready_max_abs_delta_rad: float
    previous_max_abs_delta_rad: float | None
    ready_connector_time_lower_bound_s: float
    ready_connector_limiting_joint: str
    iterations: int
    restart_index: int


@dataclass(frozen=True)
class ConnectorDiagnostics:
    """Optimistic right-arm rest-to-rest lower bound for one boundary edge."""

    start: str
    end: str
    time_lower_bound_s: float
    limiting_joint: str
    per_joint_time_lower_bound_s: tuple[float, ...]
    velocity_limit_rad_s: tuple[float, ...]
    acceleration_lower_envelope_rad_s2: tuple[float, ...]
    method: str = "diagonal_timing_envelope_v1"


@dataclass(frozen=True)
class _FrameCandidate:
    active_joint_pos: np.ndarray
    diagnostics: FrameDiagnostics
    direct_ready_connector: ConnectorDiagnostics


@dataclass(frozen=True)
class FaceManifoldResult:
    """In-memory output.  No method in this type publishes an artifact."""

    joint_pos: np.ndarray
    source_site_pos_w: np.ndarray
    output_site_pos_w: np.ndarray
    target_raw_plus_y_w: np.ndarray
    output_raw_plus_y_w: np.ndarray
    target_rotation_w: np.ndarray | None
    output_rotation_w: np.ndarray
    source_segment_velocity_w: np.ndarray
    output_segment_velocity_w: np.ndarray
    frame_diagnostics: tuple[FrameDiagnostics, ...]
    max_velocity_residual_mps: float
    mode: Mode
    active_joint_names: tuple[str, ...]
    annotation_frame_index: int | None
    entry_connector: ConnectorDiagnostics
    exit_connector: ConnectorDiagnostics
    maximum_direct_ready_time_lower_bound_s: float
    maximum_window_velocity_utilization: float
    candidate_counts_per_frame: tuple[int, ...]
    solver_strategy: str = "anchor_independent_full_window_candidate_graph"

    def summary(self) -> dict:
        """Return a JSON-safe numerical receipt without trajectory arrays."""

        return {
            "mode": self.mode,
            "frames": int(len(self.joint_pos)),
            "active_joint_names": list(self.active_joint_names),
            "solver_strategy": self.solver_strategy,
            "annotation_frame_index": self.annotation_frame_index,
            "candidate_counts_per_frame": list(self.candidate_counts_per_frame),
            "max_position_residual_m": max(
                row.position_residual_m for row in self.frame_diagnostics
            ),
            "max_normal_residual_rad": max(
                row.normal_residual_rad for row in self.frame_diagnostics
            ),
            "max_orientation_residual_rad": (
                max(
                    row.orientation_residual_rad
                    for row in self.frame_diagnostics
                    if row.orientation_residual_rad is not None
                )
                if self.mode == "full-so3"
                else None
            ),
            "max_velocity_residual_mps": self.max_velocity_residual_mps,
            "maximum_direct_ready_time_lower_bound_s": (
                self.maximum_direct_ready_time_lower_bound_s
            ),
            "maximum_window_velocity_utilization": (
                self.maximum_window_velocity_utilization
            ),
            "window_edges": {
                "entry": asdict(self.entry_connector),
                "solved_window": {
                    "start_frame_index": self.frame_diagnostics[0].frame_index,
                    "end_frame_index": self.frame_diagnostics[-1].frame_index,
                    "inclusive_frame_count": len(self.frame_diagnostics),
                    "internal_segment_count": max(
                        0, len(self.frame_diagnostics) - 1
                    ),
                    "maximum_velocity_utilization": (
                        self.maximum_window_velocity_utilization
                    ),
                },
                "exit": asdict(self.exit_connector),
            },
            "max_step_rad": max(
                (
                    row.max_step_from_previous_rad
                    for row in self.frame_diagnostics
                    if row.max_step_from_previous_rad is not None
                ),
                default=0.0,
            ),
            "frames_diagnostics": [asdict(row) for row in self.frame_diagnostics],
            "non_claims": [
                "no_NPZ_or_FK_publication",
                "no_retiming",
                "connector_time_is_optimistic_diagonal_lower_bound_only",
                "no_collision_or_table_net_certificate",
                "no_inverse_dynamics_or_balance_certificate",
                "no_training_deploy_or_hardware_authorization",
            ],
        }


@dataclass(frozen=True)
class SiteNormalCandidate:
    """One checked right-arm solution for an arbitrary site/normal target.

    This is deliberately only an in-memory inverse-kinematics candidate.  It
    does not publish a ready state or inherit any collision, balance, torque,
    training, deployment, or hardware capability.
    """

    joint_pos: np.ndarray
    site_pos_w: np.ndarray
    raw_plus_y_w: np.ndarray
    diagnostics: FrameDiagnostics
    original_ready_connector: ConnectorDiagnostics


class MujocoRightRacketBackend:
    """Exact vendor-MuJoCo backend with a delayed ``mujoco`` import."""

    def __init__(
        self,
        mjcf_path: str | Path,
        joint_names: Sequence[str],
        *,
        urdf_path: str | Path,
        site_name: str = CANONICAL_RACKET_SITE,
        validate_canonical_site_offset: bool = True,
    ) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on host package
            raise FaceManifoldError(
                "production face-manifold solving requires the mujoco package"
            ) from exc

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path)))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(str(name) for name in joint_names)
        if len(self.joint_names) != len(set(self.joint_names)):
            raise FaceManifoldError("joint_names must be unique")

        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if site_id < 0:
            raise FaceManifoldError(f"MJCF is missing site {site_name!r}")
        self._site_id = int(site_id)
        if validate_canonical_site_offset and site_name == CANONICAL_RACKET_SITE:
            actual = np.asarray(self.model.site_pos[self._site_id], dtype=np.float64)
            if not np.allclose(
                actual, CANONICAL_RACKET_SITE_POS_WRIST_M, rtol=0.0, atol=1.0e-6
            ):
                raise FaceManifoldError(
                    "right_racket site offset changed: "
                    f"{actual.tolist()} != {CANONICAL_RACKET_SITE_POS_WRIST_M.tolist()}"
                )

        joint_ids: list[int] = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise FaceManifoldError(f"MJCF is missing joint {name!r}")
            joint_ids.append(int(joint_id))
        self._qpos_addresses = np.asarray(
            [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids],
            dtype=np.int64,
        )
        self._dof_addresses = np.asarray(
            [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids],
            dtype=np.int64,
        )

        lower = np.full(len(joint_ids), -np.inf, dtype=np.float64)
        upper = np.full(len(joint_ids), np.inf, dtype=np.float64)
        for column, joint_id in enumerate(joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                lower[column], upper[column] = self.model.jnt_range[joint_id]
        self.position_lower = lower
        self.position_upper = upper
        self.velocity_limit, self.effort_limit = _read_urdf_motion_limits(
            Path(urdf_path), self.joint_names
        )
        for column, joint_id in enumerate(joint_ids):
            if not bool(self.model.jnt_actfrclimited[joint_id]):
                raise FaceManifoldError(
                    f"MJCF joint {self.joint_names[column]!r} lacks an "
                    "actuator-force limit"
                )
            force_lower, force_upper = self.model.jnt_actfrcrange[joint_id]
            mjcf_effort = min(abs(float(force_lower)), abs(float(force_upper)))
            if not np.isclose(
                mjcf_effort,
                self.effort_limit[column],
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise FaceManifoldError(
                    f"URDF/MJCF effort mismatch for {self.joint_names[column]!r}: "
                    f"{self.effort_limit[column]:.9g} != {mjcf_effort:.9g} Nm"
                )

        free_joints = [
            joint_id
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joints) != 1:
            raise FaceManifoldError(
                f"MJCF must contain exactly one free root, got {len(free_joints)}"
            )
        free_joint = free_joints[0]
        self._root_qpos_address = int(self.model.jnt_qposadr[free_joint])
        root_body = int(self.model.jnt_bodyid[free_joint])
        self.root_body_name = str(
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, root_body
            )
        )

    def site_pose(
        self,
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = _finite_vector(joint_pos, len(self.joint_names), "joint_pos")
        root_pos = _finite_vector(root_pos_w, 3, "root_pos_w")
        root_quat = _finite_vector(root_quat_w, 4, "root_quat_w")
        quat_norm = float(np.linalg.norm(root_quat))
        if quat_norm <= 1.0e-12:
            raise FaceManifoldError("root_quat_w is degenerate")

        self._set_state(q, root_pos, root_quat / quat_norm)
        position = np.asarray(
            self.data.site_xpos[self._site_id], dtype=np.float64
        ).copy()
        rotation = np.asarray(
            self.data.site_xmat[self._site_id], dtype=np.float64
        ).reshape(3, 3).copy()
        return position, rotation

    def diagonal_dynamics(
        self,
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact-plant diagonal inertia and zero-speed bias magnitude."""

        q = _finite_vector(joint_pos, len(self.joint_names), "joint_pos")
        root_pos = _finite_vector(root_pos_w, 3, "root_pos_w")
        root_quat = _finite_vector(root_quat_w, 4, "root_quat_w")
        quat_norm = float(np.linalg.norm(root_quat))
        if quat_norm <= 1.0e-12:
            raise FaceManifoldError("root_quat_w is degenerate")
        self._set_state(q, root_pos, root_quat / quat_norm)
        mass = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        try:
            self._mujoco.mj_fullM(self.model, self.data, mass)
        except TypeError:  # pragma: no cover - compatibility with old bindings
            self._mujoco.mj_fullM(self.model, mass, self.data.qM)
        diagonal = np.diag(mass)[self._dof_addresses].copy()
        bias = np.abs(
            np.asarray(self.data.qfrc_bias, dtype=np.float64)[
                self._dof_addresses
            ]
        ).copy()
        if (
            not np.isfinite(diagonal).all()
            or not np.isfinite(bias).all()
            or np.any(diagonal <= 0.0)
        ):
            raise FaceManifoldError("MuJoCo produced an invalid diagonal envelope")
        return diagonal, bias

    def _set_state(
        self,
        joint_pos: np.ndarray,
        root_pos_w: np.ndarray,
        normalized_root_quat_w: np.ndarray,
    ) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        root = self._root_qpos_address
        self.data.qpos[root : root + 3] = root_pos_w
        self.data.qpos[root + 3 : root + 7] = normalized_root_quat_w
        self.data.qpos[self._qpos_addresses] = joint_pos
        self._mujoco.mj_forward(self.model, self.data)


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return the principal SO(3) logarithm as a three-vector."""

    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise FaceManifoldError("rotation must be finite shape (3,3)")
    cosine = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * vee
    if math.pi - angle < 1.0e-5:
        values, vectors = np.linalg.eigh(0.5 * (R + np.eye(3)))
        axis = vectors[:, int(np.argmax(values))]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axis = -axis
        return angle * axis
    return angle * vee / (2.0 * math.sin(angle))


def exact_face_flip_target(
    source_rotation_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed raw ``+Y`` and optional full-SO(3) face-flip targets."""

    source = np.asarray(source_rotation_w, dtype=np.float64)
    if source.shape != (3, 3) or not np.isfinite(source).all():
        raise FaceManifoldError("source_rotation_w must be finite shape (3,3)")
    target_rotation = source @ FULL_FACE_FLIP_LOCAL
    return target_rotation[:, 1].copy(), target_rotation


def rotation_with_raw_plus_y(
    target_raw_plus_y_w: np.ndarray,
    reference_rotation_w: np.ndarray,
) -> np.ndarray:
    """Complete a target ``+Y`` normal with the closest stable in-plane axis.

    Normal-only inverse kinematics still needs one full-SO(3) bootstrap to
    escape antipodal stationary points.  The returned rotation preserves the
    reference ``+X`` direction after projection into the target-normal plane.
    If that direction is degenerate, the reference ``+Z`` direction and then a
    deterministic world basis are tried.  The public normal-only solve remains
    free to choose its final in-plane orientation.
    """

    target_y = _unit_vector(
        target_raw_plus_y_w, "target_raw_plus_y_w"
    )
    reference = np.asarray(reference_rotation_w, dtype=np.float64)
    if reference.shape != (3, 3) or not np.isfinite(reference).all():
        raise FaceManifoldError(
            "reference_rotation_w must be finite shape (3,3)"
        )
    candidates = [
        reference[:, 0],
        reference[:, 2],
        *np.eye(3, dtype=np.float64),
    ]
    target_x = None
    for candidate in candidates:
        projected = candidate - target_y * float(candidate @ target_y)
        norm = float(np.linalg.norm(projected))
        if norm > 1.0e-10:
            target_x = projected / norm
            break
    if target_x is None:  # pragma: no cover - world basis closes R3
        raise FaceManifoldError(
            "cannot construct an in-plane axis for target_raw_plus_y_w"
        )
    target_z = np.cross(target_x, target_y)
    target_z /= np.linalg.norm(target_z)
    rotation = np.column_stack((target_x, target_y, target_z))
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
        or float(np.linalg.det(rotation)) <= 0.0
    ):
        raise FaceManifoldError(
            "constructed target rotation is not right-handed orthonormal"
        )
    return rotation


def pose_residual(
    position_w: np.ndarray,
    rotation_w: np.ndarray,
    *,
    target_position_w: np.ndarray,
    target_raw_plus_y_w: np.ndarray,
    target_rotation_w: np.ndarray | None,
    position_scale_m: float,
) -> np.ndarray:
    """Build the primary IK residual; useful independently of MuJoCo."""

    position = _finite_vector(position_w, 3, "position_w")
    target_position = _finite_vector(
        target_position_w, 3, "target_position_w"
    )
    rotation = np.asarray(rotation_w, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise FaceManifoldError("rotation_w must be finite shape (3,3)")
    target_normal = _unit_vector(
        target_raw_plus_y_w, "target_raw_plus_y_w"
    )
    if not np.isfinite(position_scale_m) or position_scale_m <= 0.0:
        raise FaceManifoldError("position_scale_m must be positive")
    position_residual = (position - target_position) / position_scale_m
    if target_rotation_w is None:
        return np.concatenate((position_residual, rotation[:, 1] - target_normal))
    target_rotation = np.asarray(target_rotation_w, dtype=np.float64)
    if target_rotation.shape != (3, 3) or not np.isfinite(target_rotation).all():
        raise FaceManifoldError("target_rotation_w must be finite shape (3,3)")
    return np.concatenate(
        (position_residual, rotation_log(target_rotation.T @ rotation))
    )


def solve_face_flipped_window(
    source_joint_pos: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    canonical_ready_joint_pos: np.ndarray,
    *,
    fps: float,
    backend: RightRacketBackend,
    config: FaceManifoldConfig | None = None,
    frame_indices: Sequence[int] | None = None,
    anchor_index: int | None = None,
    active_candidate_seeds: Sequence[np.ndarray] | None = None,
) -> FaceManifoldResult:
    """Solve a complete BH window and return only checked in-memory arrays.

    ``source_joint_pos`` and root arrays must already be sliced to the intended
    inclusive source window.  ``frame_indices`` is diagnostic provenance only.
    ``anchor_index`` is retained solely as a contact/behaviour annotation; the
    numerical solver never uses it to seed, order, or rank candidates.
    ``active_candidate_seeds`` may add motion-specific seven-joint seeds, but
    each seed is independently projected at every frame and is never accepted
    as an answer without the same residual/limit checks as every other seed.
    """

    cfg = config or FaceManifoldConfig()
    source_q = np.asarray(source_joint_pos, dtype=np.float64)
    root_pos = np.asarray(root_pos_w, dtype=np.float64)
    root_quat = np.asarray(root_quat_w, dtype=np.float64)
    ready = np.asarray(canonical_ready_joint_pos, dtype=np.float64)
    joint_names = tuple(backend.joint_names)
    T = len(source_q)
    J = len(joint_names)
    if source_q.shape != (T, J) or T == 0:
        raise FaceManifoldError(
            f"source_joint_pos must have nonempty shape (T,{J})"
        )
    if root_pos.shape != (T, 3) or root_quat.shape != (T, 4):
        raise FaceManifoldError("root pose arrays must have shapes (T,3)/(T,4)")
    if ready.shape != (J,):
        raise FaceManifoldError(f"canonical ready must have shape ({J},)")
    if not all(
        np.isfinite(value).all()
        for value in (source_q, root_pos, root_quat, ready)
    ):
        raise FaceManifoldError("input arrays must be finite")
    if not np.isfinite(fps) or fps <= 0.0:
        raise FaceManifoldError("fps must be finite and positive")
    if frame_indices is None:
        frames = tuple(range(T))
    else:
        frames = tuple(int(value) for value in frame_indices)
        if len(frames) != T:
            raise FaceManifoldError("frame_indices length must equal T")
        if any(right != left + 1 for left, right in zip(frames[:-1], frames[1:])):
            raise FaceManifoldError("frame_indices must be consecutive")
    requested_anchor = None if anchor_index is None else int(anchor_index)
    if requested_anchor is not None and not 0 <= requested_anchor < T:
        raise FaceManifoldError(
            f"anchor_index {requested_anchor} is outside [0,{T})"
        )

    if any(name not in joint_names for name in RIGHT_STRIKE_CHAIN):
        missing = [name for name in RIGHT_STRIKE_CHAIN if name not in joint_names]
        raise FaceManifoldError(f"backend is missing striking joints {missing}")
    active = np.asarray(
        [joint_names.index(name) for name in RIGHT_STRIKE_CHAIN], dtype=np.int64
    )
    lower = np.asarray(backend.position_lower, dtype=np.float64)
    upper = np.asarray(backend.position_upper, dtype=np.float64)
    velocity_limit = np.asarray(backend.velocity_limit, dtype=np.float64)
    effort_limit = np.asarray(backend.effort_limit, dtype=np.float64)
    if lower.shape != (J,) or upper.shape != (J,):
        raise FaceManifoldError("backend joint-limit arrays have wrong shape")
    if (
        velocity_limit.shape != (J,)
        or effort_limit.shape != (J,)
        or not np.isfinite(velocity_limit[active]).all()
        or not np.isfinite(effort_limit[active]).all()
        or np.any(velocity_limit[active] <= 0.0)
        or np.any(effort_limit[active] <= 0.0)
    ):
        raise FaceManifoldError(
            "all seven striking joints need finite positive URDF "
            "velocity/effort limits"
        )
    if (
        not np.isfinite(lower[active]).all()
        or not np.isfinite(upper[active]).all()
        or np.any(lower[active] >= upper[active])
    ):
        raise FaceManifoldError("all seven striking joints need finite hard limits")
    _assert_within_limits(
        source_q[:, active],
        lower[active],
        upper[active],
        cfg.joint_limit_tolerance_rad,
        "source striking chain",
    )
    _assert_within_limits(
        ready[active],
        lower[active],
        upper[active],
        cfg.joint_limit_tolerance_rad,
        "canonical ready striking chain",
    )
    extra_active_seeds: tuple[np.ndarray, ...] = ()
    if active_candidate_seeds is not None:
        parsed_seeds: list[np.ndarray] = []
        for seed_index, raw_seed in enumerate(active_candidate_seeds):
            seed = _finite_vector(
                raw_seed, len(active), f"active_candidate_seeds[{seed_index}]"
            )
            _assert_within_limits(
                seed,
                lower[active],
                upper[active],
                cfg.joint_limit_tolerance_rad,
                f"active candidate seed {seed_index}",
            )
            parsed_seeds.append(seed.copy())
        extra_active_seeds = tuple(parsed_seeds)

    source_positions = np.empty((T, 3), dtype=np.float64)
    source_rotations = np.empty((T, 3, 3), dtype=np.float64)
    target_normals = np.empty((T, 3), dtype=np.float64)
    target_rotations = (
        np.empty((T, 3, 3), dtype=np.float64)
        if cfg.mode == "full-so3"
        else None
    )
    for row in range(T):
        position, rotation = backend.site_pose(
            source_q[row], root_pos[row], root_quat[row]
        )
        _validate_pose(position, rotation, f"source frame {frames[row]}")
        source_positions[row] = position
        source_rotations[row] = rotation
        target_normal, target_rotation = exact_face_flip_target(rotation)
        target_normals[row] = target_normal
        if target_rotations is not None:
            target_rotations[row] = target_rotation

    candidate_pools: list[list[_FrameCandidate]] = []
    for row in range(T):
        pool = _build_frame_candidate_pool(
            source_q[row],
            root_pos[row],
            root_quat[row],
            ready,
            target_position=source_positions[row],
            target_normal=target_normals[row],
            target_rotation=(
                target_rotations[row] if target_rotations is not None else None
            ),
            bootstrap_target_rotation=source_rotations[row]
            @ FULL_FACE_FLIP_LOCAL,
            backend=backend,
            active=active,
            lower=lower[active],
            upper=upper[active],
            velocity_limit=velocity_limit[active],
            effort_limit=effort_limit[active],
            config=cfg,
            frame_index=frames[row],
            extra_seeds=extra_active_seeds,
        )
        candidate_pools.append(pool)

    # Propagate *all* neighbouring branches in both directions.  This expands
    # the candidate graph but never greedily commits to an anchor or previous
    # frame; final selection happens only after the complete graph exists.
    for _ in range(cfg.continuation_passes):
        for rows, neighbour_offset in (
            (range(1, T), -1),
            (range(T - 2, -1, -1), 1),
        ):
            for row in rows:
                neighbour = row + neighbour_offset
                propagated = tuple(
                    candidate.active_joint_pos
                    for candidate in candidate_pools[neighbour]
                )
                candidate_pools[row] = _build_frame_candidate_pool(
                    source_q[row],
                    root_pos[row],
                    root_quat[row],
                    ready,
                    target_position=source_positions[row],
                    target_normal=target_normals[row],
                    target_rotation=(
                        target_rotations[row]
                        if target_rotations is not None
                        else None
                    ),
                    bootstrap_target_rotation=source_rotations[row]
                    @ FULL_FACE_FLIP_LOCAL,
                    backend=backend,
                    active=active,
                    lower=lower[active],
                    upper=upper[active],
                    velocity_limit=velocity_limit[active],
                    effort_limit=effort_limit[active],
                    config=cfg,
                    frame_index=frames[row],
                    extra_seeds=propagated,
                    existing=candidate_pools[row],
                    base_seeds=False,
                )

    if cfg.first_frame_max_delta_from_ready_rad is not None:
        candidate_pools[0] = [
            candidate
            for candidate in candidate_pools[0]
            if candidate.diagnostics.ready_max_abs_delta_rad
            <= cfg.first_frame_max_delta_from_ready_rad
            + cfg.joint_limit_tolerance_rad
        ]
        if not candidate_pools[0]:
            raise FaceManifoldError(
                "no first-frame candidate satisfies "
                "first_frame_max_delta_from_ready_rad"
            )

    selected_candidates, maximum_window_velocity_utilization = (
        _select_full_window_path(
            candidate_pools,
            velocity_limit=velocity_limit[active]
            * cfg.velocity_limit_fraction,
            fps=float(fps),
            max_step_rad=cfg.max_step_rad,
            tolerance=cfg.joint_limit_tolerance_rad,
            frame_indices=frames,
        )
    )
    output_q = source_q.copy()
    for row, candidate in enumerate(selected_candidates):
        output_q[row, active] = candidate.active_joint_pos
    nonactive = np.setdiff1d(np.arange(J), active)
    if not np.array_equal(output_q[:, nonactive], source_q[:, nonactive]):
        raise FaceManifoldError("solver changed a non-striking joint")
    solved_diagnostics = tuple(
        candidate.diagnostics for candidate in selected_candidates
    )
    entry_connector = replace(
        selected_candidates[0].direct_ready_connector,
        start="canonical_ready",
        end=f"solved_frame_{frames[0]}",
    )
    exit_connector = replace(
        selected_candidates[-1].direct_ready_connector,
        start=f"solved_frame_{frames[-1]}",
        end="canonical_ready",
    )
    maximum_direct_ready_time = max(
        candidate.direct_ready_connector.time_lower_bound_s
        for candidate in selected_candidates
    )

    output_positions = np.empty_like(source_positions)
    output_rotations = np.empty_like(source_rotations)
    output_normals = np.empty_like(target_normals)
    checked_diagnostics: list[FrameDiagnostics] = []
    for row in range(T):
        position, rotation = backend.site_pose(
            output_q[row], root_pos[row], root_quat[row]
        )
        _validate_pose(position, rotation, f"output frame {frames[row]}")
        output_positions[row] = position
        output_rotations[row] = rotation
        output_normals[row] = rotation[:, 1]
        checked = _checked_frame_diagnostics(
            output_q[row],
            output_q[row - 1] if row else None,
            position,
            rotation,
            target_position=source_positions[row],
            target_normal=target_normals[row],
            target_rotation=(
                target_rotations[row] if target_rotations is not None else None
            ),
            ready=ready,
            active=active,
            lower=lower[active],
            upper=upper[active],
            config=cfg,
            frame_index=frames[row],
            iterations=solved_diagnostics[row].iterations,
            restart_index=solved_diagnostics[row].restart_index,
            ready_connector=selected_candidates[row].direct_ready_connector,
        )
        _enforce_frame(checked, cfg)
        checked_diagnostics.append(checked)

    source_segment_velocity = np.diff(source_positions, axis=0) * float(fps)
    output_segment_velocity = np.diff(output_positions, axis=0) * float(fps)
    velocity_residual = (
        float(
            np.max(
                np.linalg.norm(
                    output_segment_velocity - source_segment_velocity, axis=1
                )
            )
        )
        if T > 1
        else 0.0
    )
    if velocity_residual > cfg.velocity_tolerance_mps:
        raise FaceManifoldError(
            "per-frame point preservation did not preserve segment velocity: "
            f"{velocity_residual:.9g} > {cfg.velocity_tolerance_mps:.9g} m/s"
        )
    if not np.isfinite(output_q).all():
        raise FaceManifoldError("solver produced non-finite joint positions")

    return FaceManifoldResult(
        joint_pos=np.ascontiguousarray(output_q),
        source_site_pos_w=source_positions,
        output_site_pos_w=output_positions,
        target_raw_plus_y_w=target_normals,
        output_raw_plus_y_w=output_normals,
        target_rotation_w=target_rotations,
        output_rotation_w=output_rotations,
        source_segment_velocity_w=source_segment_velocity,
        output_segment_velocity_w=output_segment_velocity,
        frame_diagnostics=tuple(checked_diagnostics),
        max_velocity_residual_mps=velocity_residual,
        mode=cfg.mode,
        active_joint_names=RIGHT_STRIKE_CHAIN,
        annotation_frame_index=(
            None if requested_anchor is None else frames[requested_anchor]
        ),
        entry_connector=entry_connector,
        exit_connector=exit_connector,
        maximum_direct_ready_time_lower_bound_s=maximum_direct_ready_time,
        maximum_window_velocity_utilization=(
            maximum_window_velocity_utilization
        ),
        candidate_counts_per_frame=tuple(
            len(pool) for pool in candidate_pools
        ),
    )


def _build_frame_candidate_pool(
    source_q: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    ready: np.ndarray,
    *,
    target_position: np.ndarray,
    target_normal: np.ndarray,
    target_rotation: np.ndarray | None,
    bootstrap_target_rotation: np.ndarray,
    backend: RightRacketBackend,
    active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limit: np.ndarray,
    effort_limit: np.ndarray,
    config: FaceManifoldConfig,
    frame_index: int,
    extra_seeds: Sequence[np.ndarray] = (),
    existing: Sequence[_FrameCandidate] = (),
    base_seeds: bool = True,
) -> list[_FrameCandidate]:
    """Project diverse seeds and retain a checked, time-ranked candidate pool."""

    ready_active = ready[active]
    source_active = source_q[active]
    seeds: list[np.ndarray] = [np.asarray(seed, dtype=np.float64).copy() for seed in extra_seeds]
    if base_seeds:
        seeds.extend((source_active.copy(), ready_active.copy()))
        wrist_roll = RIGHT_STRIKE_CHAIN.index("right_wrist_roll_joint")
        for offset in (-math.pi, math.pi):
            legacy = source_active.copy()
            legacy[wrist_roll] += offset
            if np.all(legacy >= lower) and np.all(legacy <= upper):
                seeds.append(legacy)
        rng = np.random.default_rng(config.random_seed + int(frame_index))
        span = upper - lower
        for _ in range(config.random_restarts):
            seeds.append(lower + rng.random(len(active)) * span)
    if not seeds:
        return list(existing)

    if config.mode == "normal":
        # ``n - n_target`` has a true antipodal stationary point at the source
        # backhand normal: J.T @ residual == 0.  Escape deterministically by
        # first solving the exact face-flip orientation, then release in-plane
        # orientation for the larger normal-only manifold.
        bootstrap_config = replace(config, mode="full-so3")
        bootstrap_seeds: list[np.ndarray] = []
        for seed in tuple(seeds):
            bootstrap, _ = _iterate_hierarchical(
                np.clip(seed, lower, upper),
                source_q,
                root_pos,
                root_quat,
                target_position=target_position,
                target_normal=target_normal,
                target_rotation=bootstrap_target_rotation,
                backend=backend,
                active=active,
                lower=lower,
                upper=upper,
                config=bootstrap_config,
            )
            full_q = source_q.copy()
            full_q[active] = bootstrap
            position, rotation = backend.site_pose(full_q, root_pos, root_quat)
            if (
                np.linalg.norm(position - target_position)
                <= config.position_tolerance_m
                and np.linalg.norm(
                    rotation_log(bootstrap_target_rotation.T @ rotation)
                )
                <= config.orientation_tolerance_rad
            ):
                bootstrap_seeds.append(bootstrap)
        seeds = bootstrap_seeds + seeds

    candidates = list(existing)
    failure_rows: list[tuple[float, float, float | None]] = []
    for restart_index, seed in enumerate(seeds):
        solved, iterations = _iterate_hierarchical(
            np.clip(seed, lower, upper),
            source_q,
            root_pos,
            root_quat,
            target_position=target_position,
            target_normal=target_normal,
            target_rotation=target_rotation,
            backend=backend,
            active=active,
            lower=lower,
            upper=upper,
            config=config,
        )
        full_q = source_q.copy()
        full_q[active] = solved
        position, rotation = backend.site_pose(full_q, root_pos, root_quat)
        row = _checked_frame_diagnostics(
            full_q,
            None,
            position,
            rotation,
            target_position=target_position,
            target_normal=target_normal,
            target_rotation=target_rotation,
            ready=ready,
            active=active,
            lower=lower,
            upper=upper,
            config=config,
            frame_index=frame_index,
            iterations=iterations,
            restart_index=restart_index,
        )
        failure_rows.append(
            (
                row.position_residual_m,
                row.normal_residual_rad,
                row.orientation_residual_rad,
            )
        )
        if not _frame_feasible(row, config):
            continue
        connector = _connector_time_lower_bound(
            source_q,
            solved,
            ready,
            root_pos,
            root_quat,
            backend=backend,
            active=active,
            velocity_limit=velocity_limit,
            effort_limit=effort_limit,
            config=config,
            start_label="canonical_ready",
            end_label=f"frame_{frame_index}",
        )
        candidates.append(
            _FrameCandidate(
                active_joint_pos=solved.copy(),
                diagnostics=replace(
                    row,
                    ready_connector_time_lower_bound_s=(
                        connector.time_lower_bound_s
                    ),
                    ready_connector_limiting_joint=connector.limiting_joint,
                ),
                direct_ready_connector=connector,
            )
        )

    candidates = _deduplicate_candidates(candidates, config)
    if not candidates:
        if not failure_rows:
            raise FaceManifoldError(
                f"frame {frame_index} had no numerical candidate seeds"
            )
        position, normal, orientation = min(
            failure_rows,
            key=lambda values: (
                values[0] / config.position_tolerance_m
                + values[1] / config.normal_tolerance_rad
                + (
                    0.0
                    if values[2] is None
                    else values[2] / config.orientation_tolerance_rad
                )
            ),
        )
        raise FaceManifoldError(
            f"frame {frame_index} produced no feasible manifold candidate: "
            f"best position={position:.9g} m normal={normal:.9g} rad "
            f"orientation={orientation!r}"
        )
    return candidates


def _deduplicate_candidates(
    candidates: Sequence[_FrameCandidate],
    config: FaceManifoldConfig,
) -> list[_FrameCandidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.direct_ready_connector.time_lower_bound_s,
            candidate.diagnostics.ready_max_abs_delta_rad,
            candidate.diagnostics.restart_index,
        ),
    )
    unique: list[_FrameCandidate] = []
    for candidate in ordered:
        if any(
            float(
                np.max(
                    np.abs(
                        candidate.active_joint_pos - kept.active_joint_pos
                    )
                )
            )
            <= config.candidate_dedup_tolerance_rad
            for kept in unique
        ):
            continue
        unique.append(candidate)
    if len(unique) <= config.candidate_pool_limit:
        return unique

    # Keep both time-optimal and geometrically diverse branches.  Truncating
    # solely by per-frame connector time can discard the only branch that
    # connects through the next frame, defeating whole-window backtracking.
    fastest_count = max(1, config.candidate_pool_limit // 2)
    retained = unique[:fastest_count]
    remaining = unique[fastest_count:]
    while len(retained) < config.candidate_pool_limit and remaining:
        distances = [
            min(
                float(
                    np.linalg.norm(
                        candidate.active_joint_pos
                        - kept.active_joint_pos
                    )
                )
                for kept in retained
            )
            for candidate in remaining
        ]
        selected = int(np.argmax(distances))
        retained.append(remaining.pop(selected))
    retained.sort(
        key=lambda candidate: (
            candidate.direct_ready_connector.time_lower_bound_s,
            candidate.diagnostics.ready_max_abs_delta_rad,
            candidate.diagnostics.restart_index,
        )
    )
    return retained


def _connector_time_lower_bound(
    source_q: np.ndarray,
    value_active: np.ndarray,
    ready: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    *,
    backend: RightRacketBackend,
    active: np.ndarray,
    velocity_limit: np.ndarray,
    effort_limit: np.ndarray,
    config: FaceManifoldConfig,
    start_label: str,
    end_label: str,
) -> ConnectorDiagnostics:
    """Evaluate the named diagonal timing envelope along a straight connector.

    The returned time is an optimistic necessary lower bound, not a retimed
    connector or a dynamics certificate.  It uses the exact URDF velocity and
    effort limits, and the exact MuJoCo ``Mjj`` and zero-speed bias at every
    interpolation sample:

    ``a_i = min_s (effort_i - abs(bias_i(q(s)))) / M_ii(q(s))``.
    """

    start_active = ready[active]
    end_active = _finite_vector(value_active, len(active), "value_active")
    acceleration = np.full(len(active), np.inf, dtype=np.float64)
    for fraction in np.linspace(
        0.0, 1.0, config.connector_dynamics_samples
    ):
        q = source_q.copy()
        q[active] = (1.0 - fraction) * start_active + fraction * end_active
        diagonal, bias = backend.diagonal_dynamics(q, root_pos, root_quat)
        diagonal = np.asarray(diagonal, dtype=np.float64)
        bias = np.asarray(bias, dtype=np.float64)
        if diagonal.shape != source_q.shape or bias.shape != source_q.shape:
            raise FaceManifoldError(
                "backend diagonal dynamics arrays must match joint_pos"
            )
        available = effort_limit - bias[active]
        if np.any(available <= 0.0):
            joint = RIGHT_STRIKE_CHAIN[int(np.argmin(available))]
            raise FaceManifoldError(
                f"connector has no positive static effort margin at {joint!r}"
            )
        acceleration = np.minimum(acceleration, available / diagonal[active])
    velocity = velocity_limit * config.velocity_limit_fraction
    per_joint = _minimum_rest_to_rest_times(
        np.abs(end_active - start_active), velocity, acceleration
    )
    limiting = int(np.argmax(per_joint))
    return ConnectorDiagnostics(
        start=start_label,
        end=end_label,
        time_lower_bound_s=float(per_joint[limiting]),
        limiting_joint=RIGHT_STRIKE_CHAIN[limiting],
        per_joint_time_lower_bound_s=tuple(float(value) for value in per_joint),
        velocity_limit_rad_s=tuple(float(value) for value in velocity),
        acceleration_lower_envelope_rad_s2=tuple(
            float(value) for value in acceleration
        ),
    )


def _minimum_rest_to_rest_times(
    displacement_rad: np.ndarray,
    velocity_limit_rad_s: np.ndarray,
    acceleration_limit_rad_s2: np.ndarray,
) -> np.ndarray:
    """Per-axis triangular/trapezoidal rest-to-rest minimum-time lower bound."""

    displacement = np.asarray(displacement_rad, dtype=np.float64)
    velocity = np.asarray(velocity_limit_rad_s, dtype=np.float64)
    acceleration = np.asarray(acceleration_limit_rad_s2, dtype=np.float64)
    if (
        displacement.shape != velocity.shape
        or velocity.shape != acceleration.shape
        or not np.isfinite(displacement).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(acceleration).all()
        or np.any(displacement < 0.0)
        or np.any(velocity <= 0.0)
        or np.any(acceleration <= 0.0)
    ):
        raise FaceManifoldError("invalid rest-to-rest envelope inputs")
    triangular_distance = np.square(velocity) / acceleration
    return np.where(
        displacement <= triangular_distance,
        2.0 * np.sqrt(displacement / acceleration),
        displacement / velocity + velocity / acceleration,
    )


def _select_full_window_path(
    candidate_pools: Sequence[Sequence[_FrameCandidate]],
    *,
    velocity_limit: np.ndarray,
    fps: float,
    max_step_rad: float,
    tolerance: float,
    frame_indices: Sequence[int],
) -> tuple[tuple[_FrameCandidate, ...], float]:
    """Backtrack over the complete candidate graph before selecting a branch."""

    if not candidate_pools or any(not pool for pool in candidate_pools):
        raise FaceManifoldError("every window frame needs at least one candidate")
    best: tuple[
        tuple[float, float, float, float],
        tuple[_FrameCandidate, ...],
        float,
    ] | None = None
    first_pool = candidate_pools[0]
    for start_index, start in enumerate(first_pool):
        states: dict[
            int, tuple[float, float, tuple[int, ...], float]
        ] = {
            start_index: (
                start.direct_ready_connector.time_lower_bound_s,
                0.0,
                (start_index,),
                0.0,
            )
        }
        for row in range(1, len(candidate_pools)):
            next_states: dict[
                int, tuple[float, float, tuple[int, ...], float]
            ] = {}
            for current_index, current in enumerate(candidate_pools[row]):
                for previous_index, state in states.items():
                    previous = candidate_pools[row - 1][previous_index]
                    delta = np.abs(
                        current.active_joint_pos
                        - previous.active_joint_pos
                    )
                    if float(np.max(delta)) > max_step_rad + tolerance:
                        continue
                    utilization = delta * float(fps) / velocity_limit
                    peak_utilization = float(np.max(utilization))
                    if peak_utilization > 1.0 + tolerance:
                        continue
                    proposal = (
                        max(
                            state[0],
                            current.direct_ready_connector.time_lower_bound_s,
                        ),
                        state[1] + float(np.sum(np.square(utilization))),
                        state[2] + (current_index,),
                        max(state[3], peak_utilization),
                    )
                    incumbent = next_states.get(current_index)
                    if incumbent is None or proposal[:2] < incumbent[:2]:
                        next_states[current_index] = proposal
            states = next_states
            if not states:
                break
        for end_index, state in states.items():
            if len(state[2]) != len(candidate_pools):
                continue
            end = candidate_pools[-1][end_index]
            entry = start.direct_ready_connector.time_lower_bound_s
            exit_time = end.direct_ready_connector.time_lower_bound_s
            score = (
                state[0],
                max(entry, exit_time),
                entry + exit_time,
                state[1],
            )
            path = tuple(
                candidate_pools[row][index]
                for row, index in enumerate(state[2])
            )
            if best is None or score < best[0]:
                best = (score, path, state[3])
    if best is None:
        edge_rows: list[dict[str, float | int]] = []
        reachable = set(range(len(candidate_pools[0])))
        for row in range(1, len(candidate_pools)):
            feasible_edges = 0
            reachable_next: set[int] = set()
            minimum_step = math.inf
            minimum_velocity_utilization = math.inf
            for previous_index, previous in enumerate(candidate_pools[row - 1]):
                for current_index, current in enumerate(candidate_pools[row]):
                    delta = np.abs(
                        current.active_joint_pos - previous.active_joint_pos
                    )
                    step = float(np.max(delta))
                    utilization = float(
                        np.max(delta * float(fps) / velocity_limit)
                    )
                    minimum_step = min(minimum_step, step)
                    minimum_velocity_utilization = min(
                        minimum_velocity_utilization, utilization
                    )
                    if (
                        step <= max_step_rad + tolerance
                        and utilization <= 1.0 + tolerance
                    ):
                        feasible_edges += 1
                        if previous_index in reachable:
                            reachable_next.add(current_index)
            edge_rows.append(
                {
                    "left_frame": int(frame_indices[row - 1]),
                    "right_frame": int(frame_indices[row]),
                    "minimum_max_step_rad": minimum_step,
                    "minimum_peak_velocity_utilization": (
                        minimum_velocity_utilization
                    ),
                    "feasible_edge_count": feasible_edges,
                    "reachable_candidate_count": len(reachable_next),
                }
            )
            reachable = reachable_next
        raise FaceManifoldError(
            "no continuous full-window candidate path satisfies both the "
            "per-frame max-step gate and exact URDF velocity limits; "
            f"frames={list(frame_indices)} pools="
            f"{[len(pool) for pool in candidate_pools]} edges={edge_rows}"
        )
    return best[1], best[2]


def _iterate_hierarchical(
    initial: np.ndarray,
    source_q: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    *,
    target_position: np.ndarray,
    target_normal: np.ndarray,
    target_rotation: np.ndarray | None,
    backend: RightRacketBackend,
    active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: FaceManifoldConfig,
) -> tuple[np.ndarray, int]:
    x = np.asarray(initial, dtype=np.float64).copy()

    def residual(value: np.ndarray) -> np.ndarray:
        q = source_q.copy()
        q[active] = value
        position, rotation = backend.site_pose(q, root_pos, root_quat)
        return pose_residual(
            position,
            rotation,
            target_position_w=target_position,
            target_raw_plus_y_w=target_normal,
            target_rotation_w=target_rotation,
            position_scale_m=config.position_residual_scale_m,
        )

    for iteration in range(1, config.max_iterations + 1):
        current = residual(x)
        jacobian = _finite_difference_jacobian(
            residual, x, config.finite_difference_step_rad
        )
        pinv = _damped_pseudoinverse(jacobian, config.damping)
        primary = -pinv @ current
        primary_feasible = _raw_residual_feasible(current, config)
        if primary_feasible:
            # Preserve the projected manifold point as a distinct graph node.
            # Connector time and whole-window smoothness are ranked globally;
            # a local range-normalized nullspace descent would collapse these
            # alternatives and recreate the wide-range wrist-roll bias.
            return x, iteration
        step = primary
        largest = float(np.max(np.abs(step)))
        if largest > config.max_solver_step_rad:
            step *= config.max_solver_step_rad / largest
        # Far from the manifold (notably near a 180-degree face change), the
        # Gauss-Newton model can require a temporary increase in the combined
        # position+normal norm.  Prefer a residual-decreasing trust step for
        # continuation seeds, but retain the bounded full step as the
        # antipodal escape when no decreasing scale exists.
        current_norm = float(np.linalg.norm(current))
        candidate = np.clip(x + step, lower, upper)
        for exponent in range(12):
            scaled = np.clip(x + (0.5**exponent) * step, lower, upper)
            if float(np.linalg.norm(residual(scaled))) < current_norm:
                candidate = scaled
                break
        if np.max(np.abs(candidate - x)) < 1.0e-12:
            break
        x = candidate
    return x, iteration


def _checked_frame_diagnostics(
    q: np.ndarray,
    previous_q: np.ndarray | None,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    target_position: np.ndarray,
    target_normal: np.ndarray,
    target_rotation: np.ndarray | None,
    ready: np.ndarray,
    active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: FaceManifoldConfig,
    frame_index: int,
    iterations: int,
    restart_index: int,
    previous_active_override: np.ndarray | None = None,
    ready_connector: ConnectorDiagnostics | None = None,
) -> FrameDiagnostics:
    position_residual = float(np.linalg.norm(position - target_position))
    normal = _unit_vector(rotation[:, 1], "output raw +Y")
    target = _unit_vector(target_normal, "target raw +Y")
    normal_residual = float(
        np.arccos(np.clip(float(normal @ target), -1.0, 1.0))
    )
    orientation_residual = (
        None
        if target_rotation is None
        else float(np.linalg.norm(rotation_log(target_rotation.T @ rotation)))
    )
    active_q = q[active]
    violation = np.maximum(lower - active_q, active_q - upper)
    maximum_violation = max(0.0, float(np.max(violation)))
    previous_active = (
        previous_active_override
        if previous_active_override is not None
        else (None if previous_q is None else previous_q[active])
    )
    return FrameDiagnostics(
        frame_index=int(frame_index),
        position_residual_m=position_residual,
        normal_residual_rad=normal_residual,
        orientation_residual_rad=orientation_residual,
        max_joint_limit_violation_rad=maximum_violation,
        max_step_from_previous_rad=(
            None
            if previous_active is None
            else float(np.max(np.abs(active_q - previous_active)))
        ),
        ready_max_abs_delta_rad=float(
            np.max(np.abs(active_q - ready[active]))
        ),
        previous_max_abs_delta_rad=(
            None
            if previous_active is None
            else float(np.max(np.abs(active_q - previous_active)))
        ),
        ready_connector_time_lower_bound_s=(
            0.0
            if ready_connector is None
            else ready_connector.time_lower_bound_s
        ),
        ready_connector_limiting_joint=(
            "pending"
            if ready_connector is None
            else ready_connector.limiting_joint
        ),
        iterations=int(iterations),
        restart_index=int(restart_index),
    )


def _enforce_frame(row: FrameDiagnostics, config: FaceManifoldConfig) -> None:
    if not _frame_feasible(row, config):
        raise FaceManifoldError(
            f"frame {row.frame_index} failed residual/limit verification: "
            f"{asdict(row)}"
        )
    if (
        row.max_step_from_previous_rad is not None
        and row.max_step_from_previous_rad
        > config.max_step_rad + config.joint_limit_tolerance_rad
    ):
        raise FaceManifoldError(
            f"frame {row.frame_index} continuity step "
            f"{row.max_step_from_previous_rad:.9g} exceeds "
            f"{config.max_step_rad:.9g} rad"
        )


def _frame_feasible(
    row: FrameDiagnostics, config: FaceManifoldConfig
) -> bool:
    return bool(
        row.position_residual_m <= config.position_tolerance_m
        and row.normal_residual_rad <= config.normal_tolerance_rad
        and (
            row.orientation_residual_rad is None
            or row.orientation_residual_rad <= config.orientation_tolerance_rad
        )
        and row.max_joint_limit_violation_rad
        <= config.joint_limit_tolerance_rad
    )


def _raw_residual_feasible(
    residual: np.ndarray, config: FaceManifoldConfig
) -> bool:
    return bool(float(np.linalg.norm(residual)) <= _raw_residual_threshold(config))


def _raw_residual_threshold(config: FaceManifoldConfig) -> float:
    angular = (
        config.normal_tolerance_rad
        if config.mode == "normal"
        else config.orientation_tolerance_rad
    )
    return math.sqrt(
        (config.position_tolerance_m / config.position_residual_scale_m) ** 2
        + angular**2
    )


def _finite_difference_jacobian(function, value, epsilon):
    baseline = np.asarray(function(value), dtype=np.float64)
    jacobian = np.empty((len(baseline), len(value)), dtype=np.float64)
    for column in range(len(value)):
        plus = value.copy()
        minus = value.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        jacobian[:, column] = (function(plus) - function(minus)) / (2.0 * epsilon)
    if not np.isfinite(jacobian).all():
        raise FaceManifoldError("finite-difference Jacobian is non-finite")
    return jacobian


def _damped_pseudoinverse(matrix: np.ndarray, damping: float) -> np.ndarray:
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    inverse = singular / (np.square(singular) + damping * damping)
    return (right_t.T * inverse) @ left.T


def _read_urdf_motion_limits(
    path: Path, joint_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Read exact named velocity/effort limits from the bound vendor URDF."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise FaceManifoldError(f"cannot parse URDF motion limits: {path}") from exc
    limits: dict[str, tuple[float, float]] = {}
    requested = set(joint_names)
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in requested:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise FaceManifoldError(f"URDF joint {name!r} lacks a limit element")
        try:
            velocity = float(limit.attrib["velocity"])
            effort = float(limit.attrib["effort"])
        except (KeyError, ValueError) as exc:
            raise FaceManifoldError(
                f"URDF joint {name!r} has malformed velocity/effort"
            ) from exc
        if (
            not np.isfinite(velocity)
            or not np.isfinite(effort)
            or velocity <= 0.0
            or effort <= 0.0
        ):
            raise FaceManifoldError(
                f"URDF joint {name!r} needs positive velocity/effort"
            )
        limits[name] = (velocity, effort)
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise FaceManifoldError(f"URDF misses named joints: {missing}")
    return (
        np.asarray([limits[name][0] for name in joint_names], dtype=np.float64),
        np.asarray([limits[name][1] for name in joint_names], dtype=np.float64),
    )


def _validate_pose(position, rotation, label):
    position = np.asarray(position)
    rotation = np.asarray(rotation)
    if (
        position.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
    ):
        raise FaceManifoldError(f"{label} produced invalid site pose")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2.0e-6):
        raise FaceManifoldError(f"{label} site rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999998:
        raise FaceManifoldError(f"{label} site rotation is not right-handed")


def _finite_vector(value, length, label):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise FaceManifoldError(f"{label} must be finite shape ({length},)")
    return vector


def _unit_vector(value, label):
    vector = _finite_vector(value, 3, label)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise FaceManifoldError(f"{label} is degenerate")
    return vector / norm


def _assert_within_limits(values, lower, upper, tolerance, label):
    array = np.asarray(values, dtype=np.float64)
    violation = np.maximum(lower - array, array - upper)
    if float(np.max(violation)) > tolerance:
        index = np.unravel_index(int(np.argmax(violation)), violation.shape)
        raise FaceManifoldError(
            f"{label} exceeds a hard position limit at {index} by "
            f"{float(violation[index]):.9g} rad"
        )


def _parse_inclusive_span(text: str) -> tuple[int, int]:
    try:
        left, right = text.split(":", 1)
        start, end = int(left), int(right)
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("window must be START:END") from exc
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("window must satisfy 0 <= START <= END")
    return start, end


def _parse_active_seed_degrees(text: str) -> np.ndarray:
    try:
        values = np.asarray(
            [float(value) for value in text.split(",")], dtype=np.float64
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "active seed must be seven comma-separated degree values"
        ) from exc
    if values.shape != (len(RIGHT_STRIKE_CHAIN),) or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(
            "active seed must be seven finite comma-separated degree values"
        )
    return np.radians(values)


def _read_joint_order(path: Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not names or len(names) != len(set(names)):
        raise FaceManifoldError("joint-order file must contain unique names")
    return names


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read an exact BH schema-2 motion and print an in-memory "
            "right-racket face-manifold solve receipt; writes no artifact."
        )
    )
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--canonical-ready", required=True)
    parser.add_argument(
        "--joint-order",
        help=(
            "Runtime/schema-2 joint-order text file. Defaults to the repository "
            "contract when this script is executed from its checked-in path."
        ),
    )
    parser.add_argument("--window", type=_parse_inclusive_span, required=True)
    parser.add_argument("--mode", choices=("normal", "full-so3"), default="normal")
    parser.add_argument("--contact-frame", type=int)
    parser.add_argument("--max-step-deg", type=float, default=35.0)
    parser.add_argument("--random-restarts", type=int, default=12)
    parser.add_argument(
        "--active-seed-deg",
        action="append",
        type=_parse_active_seed_degrees,
        default=[],
        help=(
            "Optional seven-value shP,shR,shY,elbow,wrR,wrP,wrY seed in "
            "degrees. Repeatable; each value is reprojected at every frame."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    try:
        if args.joint_order:
            joint_order_path = Path(args.joint_order)
        else:
            try:
                repo_root = Path(__file__).resolve().parents[3]
            except IndexError as exc:
                raise FaceManifoldError(
                    "--joint-order is required outside the repository layout"
                ) from exc
            joint_order_path = (
                repo_root / "configs/a3_runtime_articulation_joint_order.txt"
            )
        joint_names = _read_joint_order(joint_order_path)
        backend = MujocoRightRacketBackend(
            args.mjcf, joint_names, urdf_path=args.urdf
        )
        with np.load(args.source, allow_pickle=False) as source:
            required = {
                "fps",
                "joint_pos",
                "body_pos_w",
                "body_quat_w",
                "body_names",
            }
            if not required.issubset(source.files):
                raise FaceManifoldError(
                    f"source NPZ is missing {sorted(required - set(source.files))}"
                )
            start, end = args.window
            q = np.asarray(source["joint_pos"], dtype=np.float64)
            if q.ndim != 2 or q.shape[1] != len(joint_names) or end >= len(q):
                raise FaceManifoldError("source/window shape is inconsistent")
            body_names = tuple(str(value) for value in source["body_names"].tolist())
            if backend.root_body_name not in body_names:
                raise FaceManifoldError(
                    f"source body order lacks root {backend.root_body_name!r}"
                )
            root_column = body_names.index(backend.root_body_name)
            source_q = q[start : end + 1]
            root_pos = np.asarray(
                source["body_pos_w"][start : end + 1, root_column],
                dtype=np.float64,
            )
            root_quat = np.asarray(
                source["body_quat_w"][start : end + 1, root_column],
                dtype=np.float64,
            )
            fps = float(np.asarray(source["fps"]).reshape(-1)[0])
        with np.load(args.canonical_ready, allow_pickle=False) as ready:
            if "joint_pos" not in ready.files:
                raise FaceManifoldError("canonical-ready NPZ lacks joint_pos")
            ready_q = np.asarray(ready["joint_pos"], dtype=np.float64)
        config = FaceManifoldConfig(
            mode=args.mode,
            max_step_rad=math.radians(args.max_step_deg),
            random_restarts=args.random_restarts,
        )
        frames = tuple(range(start, end + 1))
        result = solve_face_flipped_window(
            source_q,
            root_pos,
            root_quat,
            ready_q,
            fps=fps,
            backend=backend,
            config=config,
            frame_indices=frames,
            anchor_index=(
                args.contact_frame - start
                if args.contact_frame is not None
                else None
            ),
            active_candidate_seeds=args.active_seed_deg,
        )
        receipt = result.summary()
        receipt.update(
            {
                "source": str(Path(args.source)),
                "mjcf": str(Path(args.mjcf)),
                "urdf": str(Path(args.urdf)),
                "window_source_inclusive": [start, end],
                "writes_artifacts": False,
            }
        )
        if args.contact_frame is not None:
            if not start <= args.contact_frame <= end:
                raise FaceManifoldError("contact frame is outside --window")
            row = args.contact_frame - start
            active_indices = [joint_names.index(name) for name in RIGHT_STRIKE_CHAIN]
            receipt["contact"] = {
                "frame_source": args.contact_frame,
                "joint_pos_rad": {
                    name: float(result.joint_pos[row, column])
                    for name, column in zip(RIGHT_STRIKE_CHAIN, active_indices)
                },
                "joint_pos_deg": {
                    name: float(np.degrees(result.joint_pos[row, column]))
                    for name, column in zip(RIGHT_STRIKE_CHAIN, active_indices)
                },
                "site_pos_w_m": result.output_site_pos_w[row].tolist(),
                "raw_plus_y_w": result.output_raw_plus_y_w[row].tolist(),
            }
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (FaceManifoldError, OSError, ValueError) as exc:
        print(f"[canonical-face-manifold] FAIL {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
