#!/usr/bin/env python3
"""Build a fail-closed, face-neutral canonical-ready *challenger*.

This module is deliberately a library, not a CLI.  It solves only the seven
right striking-arm joints while keeping the donor ready root, legs, head,
waist, left arm, and every other joint bit-for-bit unchanged.

The two striking-face normals can be antipodal.  In that case a spherical
geodesic midpoint is not unique: it is the complete great circle orthogonal to
the face axis.  A finite angular optimizer locus is solved exactly when the
input axes have rank two or three.  The rank-one continuous locus is sampled
only for diagnostic IK tie-breaking and is never described as exhaustive.
The resulting branches are ranked against all ``upper/full x
opportunity-start/construction-donor-preferred/nominal-event/opportunity-end
x BH/FH`` pose rows.  The synthetic forehand has no reviewed behavioral
preferred frame: the donor's f42 construction seed and the f44 nominal
air-swing event are intentionally separate, lineage-only events.

This is intentionally a restricted challenger generator: it keeps the donor
ready racket-site position fixed, optimizes angular symmetry first, and searches
only the seven right-arm joints.  It does not jointly optimize ready position,
whole-body posture, or the final 5 x 2 timed paths.

The result remains diagnostic.  Raw caller-supplied contact arrays are
content-sealed but cannot be proven to come from the named source frame; those
calls remain publication fail-closed.  The companion
``canonical_neutral_ready_cli`` adapter may attach a sealed reconstruction proof
from the content-pinned recipe.  Even a successfully published bundle contains
``training_authorized=false`` and ``hardware_authorized=false``.  Publication
additionally requires:

* donor-ready, exact MJCF, and exact URDF bytes match explicit SHA-256 pins;
* the compiled MuJoCo model signature matches an explicit pin;
* backend position/velocity/effort arrays match an explicit digest and the
  velocity/effort arrays equal the pinned URDF values;
* exact site/normal IK and joint limits pass;
* all non-right-arm joints are byte-identical to the donor; and
* an exact MuJoCo collision, double-foot-ground, penetration, and support-hull
  screen passes.

The publisher creates a new directory with exclusive semantics and writes the
receipt last.  It never overwrites the donor ready or an existing candidate.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import ctypes
import errno
import sys
import stat
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

import canonical_face_manifold as face


SCOPES = ("upper", "full")
PHASES = (
    "opportunity_start",
    "construction_donor_preferred",
    "nominal_event",
    "opportunity_end",
)
FACES = ("bh", "fh")
NORMAL_CONVENTION = (
    "right_racket_site_local_plus_y_world_signed_face_normal_v1"
)
CANDIDATE_FILENAME = "canonical_neutral_ready_candidate_v1.npz"
RECEIPT_FILENAME = "RECEIPT.json"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
CANONICAL_READY_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "source_segment",
        "source_npz",
        "source_frame",
        "striking_joint_ids",
        "note",
    }
)
CONTACT_SOURCE_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "builder_contract",
        "file_bindings",
        "contact_matrix_sha256",
        "contact_row_count",
        "phase_source_frames",
        "scope_contract",
        "face_contract",
        "model_binding",
        "rows",
    }
)


class NeutralReadyError(RuntimeError):
    """The input or candidate violates a fail-closed ready-state contract."""


@dataclass(frozen=True)
class ReadySourceBinding:
    """Content pin for the immutable donor-ready NPZ."""

    path: str | Path
    expected_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.expected_sha256, "ready expected_sha256")


@dataclass(frozen=True)
class ExactModelBinding:
    """Content pins for MJCF, compiled MuJoCo arrays, and URDF limits."""

    mjcf_path: str | Path
    expected_mjcf_sha256: str
    expected_compiled_model_sha256: str
    urdf_path: str | Path
    expected_urdf_sha256: str
    expected_backend_limits_sha256: str
    expected_backend_model_contract_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.expected_mjcf_sha256, "model expected_mjcf_sha256"
        )
        _require_sha256(
            self.expected_compiled_model_sha256,
            "model expected_compiled_model_sha256",
        )
        _require_sha256(
            self.expected_urdf_sha256, "model expected_urdf_sha256"
        )
        _require_sha256(
            self.expected_backend_limits_sha256,
            "model expected_backend_limits_sha256",
        )
        _require_sha256(
            self.expected_backend_model_contract_sha256,
            "model expected_backend_model_contract_sha256",
        )


@dataclass(frozen=True)
class ContactPose:
    """One exact scoped contact target used to judge ready neutrality.

    ``site_rotation_w[:, 1]`` and ``signed_face_normal_w`` must agree.  No
    automatic ``n/-n`` orientation is allowed.
    """

    scope: Literal["upper", "full"]
    phase: Literal[
        "opportunity_start",
        "construction_donor_preferred",
        "nominal_event",
        "opportunity_end",
    ]
    face_name: Literal["bh", "fh"]
    joint_pos: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    site_pos_w: np.ndarray
    site_rotation_w: np.ndarray
    signed_face_normal_w: np.ndarray
    source_motion_path: str | Path
    source_motion_sha256: str
    source_frame_index: int
    pose_content_sha256: str
    pair_contract_sha256: str

    @property
    def label(self) -> str:
        return f"{self.scope}:{self.phase}:{self.face_name}"


@dataclass(frozen=True)
class ContactSourceProof:
    """Sealed receipt from the source-frame reconstruction adapter."""

    receipt: Mapping[str, Any]
    payload_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.payload_sha256, "contact source proof payload_sha256"
        )


@dataclass(frozen=True)
class NeutralReadyConfig:
    """Numerical limits and deterministic search breadth."""

    site_tolerance_m: float = 2.0e-6
    normal_tolerance_rad: float = 2.0e-5
    input_site_tolerance_m: float = 2.0e-6
    input_rotation_tolerance_rad: float = 2.0e-5
    paired_face_antipodal_tolerance_rad: float = 2.0e-5
    paired_site_tolerance_m: float = 2.0e-6
    maximum_neutrality_asymmetry_rad: float = math.radians(5.0)
    global_minimax_bound_gap_tolerance_rad: float = 1.0e-3
    joint_limit_tolerance_rad: float = 1.0e-10
    antipodal_dot_tolerance: float = 1.0e-8
    antipodal_circle_samples: int = 36
    maximum_target_normals_for_ik: int = 512
    random_restarts: int = 8
    random_seed: int = 20260724
    max_iterations: int = 240
    finite_difference_step_rad: float = 1.0e-5
    damping: float = 1.0e-5
    max_solver_step_rad: float = 0.18
    candidate_dedup_tolerance_rad: float = 2.0e-4
    connector_dynamics_samples: int = 21
    collision_penetration_tolerance_m: float = 2.0e-3
    contact_gap_tolerance_m: float = 2.0e-3
    support_hull_tolerance_m: float = 2.0e-4
    ranking_angle_bucket_rad: float = 1.0e-6
    ranking_time_bucket_s: float = 1.0e-5
    ranking_residual_bucket: float = 1.0e-9
    ranking_support_bucket_m: float = 1.0e-6
    floor_geom_name: str = "floor"
    foot_body_names: tuple[str, str] = (
        "left_ankle_roll_Link",
        "right_ankle_roll_Link",
    )

    def __post_init__(self) -> None:
        for name in (
            "site_tolerance_m",
            "normal_tolerance_rad",
            "input_site_tolerance_m",
            "input_rotation_tolerance_rad",
            "paired_face_antipodal_tolerance_rad",
            "paired_site_tolerance_m",
            "maximum_neutrality_asymmetry_rad",
            "global_minimax_bound_gap_tolerance_rad",
            "antipodal_dot_tolerance",
            "finite_difference_step_rad",
            "damping",
            "max_solver_step_rad",
            "candidate_dedup_tolerance_rad",
            "collision_penetration_tolerance_m",
            "contact_gap_tolerance_m",
            "support_hull_tolerance_m",
            "ranking_angle_bucket_rad",
            "ranking_time_bucket_s",
            "ranking_residual_bucket",
            "ranking_support_bucket_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.joint_limit_tolerance_rad)
            or self.joint_limit_tolerance_rad < 0.0
        ):
            raise ValueError(
                "joint_limit_tolerance_rad must be finite and non-negative"
            )
        if self.antipodal_dot_tolerance >= 1.0:
            raise ValueError("antipodal_dot_tolerance must be below one")
        if (
            self.paired_face_antipodal_tolerance_rad >= math.pi / 2.0
            or self.maximum_neutrality_asymmetry_rad >= math.pi
            or self.global_minimax_bound_gap_tolerance_rad >= math.pi
        ):
            raise ValueError(
                "paired-face/neutrality angular tolerances are too large"
            )
        if (
            self.antipodal_circle_samples < 8
            or self.antipodal_circle_samples % 2
        ):
            raise ValueError(
                "antipodal_circle_samples must be even and at least eight"
            )
        if self.maximum_target_normals_for_ik < 1:
            raise ValueError("maximum_target_normals_for_ik must be positive")
        if self.random_restarts < 0 or self.max_iterations < 1:
            raise ValueError("restart/iteration counts are invalid")
        if self.connector_dynamics_samples < 2:
            raise ValueError("connector_dynamics_samples must be at least two")
        if (
            len(self.foot_body_names) != 2
            or len(set(self.foot_body_names)) != 2
        ):
            raise ValueError("exactly two distinct foot bodies are required")


@dataclass(frozen=True)
class MidpointTarget:
    """One physical normal on a pair's spherical midpoint locus."""

    normal_w: np.ndarray
    source_pair: str
    antipodal: bool
    theta_rad: float | None
    basis_u_w: np.ndarray | None
    basis_v_w: np.ndarray | None
    pair_dot: float


@dataclass(frozen=True)
class CandidateMetrics:
    """Minimax quantities used to select one IK branch."""

    maximum_pair_angular_asymmetry_rad: float
    maximum_contact_angle_rad: float
    worst_right_arm_diagonal_time_proxy_s: float
    worst_connector_label: str
    donor_ready_to_candidate_diagonal_time_proxy_s: float
    site_residual_m: float
    target_normal_residual_rad: float
    minimum_support_margin_m: float | None
    safety_status: str
    right_arm_changed_joint_count: int
    candidate_sha256: str


@dataclass(frozen=True)
class SafetyCheck:
    """Exact static MuJoCo collision/ground result for one candidate."""

    status: str
    exact_model: bool
    passed: bool
    collision_passed: bool
    double_foot_ground_passed: bool
    support_hull_passed: bool
    maximum_allowed_foot_penetration_m: float | None
    unsupported_contact_count: int | None
    active_contact_count: int | None
    support_margin_m: float | None
    details: Mapping[str, Any]


@dataclass(frozen=True)
class GlobalAngularSolution:
    """Exact antipodal-axis angular minimax locus or sampled rank-one locus."""

    targets: tuple[MidpointTarget, ...]
    axis_rank: int
    method: str
    maximum_absolute_axis_projection: float
    angular_objective_lower_bound_rad: float
    angular_objective_upper_bound_rad: float
    angular_minimax_certified: bool
    finite_optimizer_locus: bool
    sampled_continuous_locus: bool
    report: Mapping[str, Any]


@dataclass(frozen=True)
class NeutralReadyPoolEntry:
    """One feasible IK seed retained for downstream 5 x 2 compiler search."""

    candidate: face.SiteNormalCandidate
    target: MidpointTarget
    metrics: CandidateMetrics
    safety: SafetyCheck
    connectors: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class NeutralReadyResult:
    """In-memory diagnostic candidate and strict receipt."""

    candidate: face.SiteNormalCandidate
    target: MidpointTarget
    metrics: CandidateMetrics
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    receipt: Mapping[str, Any]
    source_ready_path: Path
    publication_allowed: bool
    backend: face.RightRacketBackend
    model_binding: ExactModelBinding | None
    payload_seal_sha256: str
    candidate_pool: tuple[NeutralReadyPoolEntry, ...]
    contacts: tuple[ContactPose, ...]
    contact_source_proof: ContactSourceProof | None


@dataclass(frozen=True)
class PublishedNeutralReady:
    """Paths and hashes for one newly created diagnostic bundle."""

    directory: Path
    candidate_npz: Path
    receipt_json: Path
    candidate_npz_sha256: str
    receipt_json_sha256: str


def spherical_geodesic_midpoint_targets(
    normal_a_w: np.ndarray,
    normal_b_w: np.ndarray,
    *,
    reference_rotation_w: np.ndarray,
    source_pair: str,
    antipodal_dot_tolerance: float = 1.0e-8,
    antipodal_circle_samples: int = 36,
) -> tuple[MidpointTarget, ...]:
    """Return the unique midpoint or a complete sampled antipodal great circle.

    For antipodal normals, ``normalize(a+b)`` is undefined.  The returned
    target set instead spans ``cos(theta)*u + sin(theta)*v`` for the complete
    orthogonal great circle.  The basis is derived from the ready orientation,
    canonicalized in sign, and recorded with every target.
    """

    a = _unit(normal_a_w, "normal_a_w")
    b = _unit(normal_b_w, "normal_b_w")
    reference = _rotation(reference_rotation_w, "reference_rotation_w")
    tolerance = float(antipodal_dot_tolerance)
    samples = int(antipodal_circle_samples)
    if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
        raise NeutralReadyError(
            "antipodal_dot_tolerance must lie strictly between zero and one"
        )
    if samples < 8 or samples % 2:
        raise NeutralReadyError(
            "antipodal_circle_samples must be even and at least eight"
        )
    pair_dot = float(np.clip(a @ b, -1.0, 1.0))
    if pair_dot > -1.0 + tolerance:
        total = a + b
        if float(np.linalg.norm(total)) <= 1.0e-12:
            raise NeutralReadyError(
                "non-antipodal midpoint unexpectedly has a zero vector"
            )
        return (
            MidpointTarget(
                normal_w=_unit(total, "geodesic midpoint"),
                source_pair=str(source_pair),
                antipodal=False,
                theta_rad=None,
                basis_u_w=None,
                basis_v_w=None,
                pair_dot=pair_dot,
            ),
        )

    # The difference is a numerically stable representative of the antipodal
    # face axis.  Midpoints are every unit vector orthogonal to this axis.
    axis = _unit(a - b, "antipodal face axis")
    basis_u = _projected_basis(axis, reference)
    basis_v = _unit(np.cross(axis, basis_u), "antipodal basis_v")
    basis_u, basis_v = _canonicalize_plane_basis(basis_u, basis_v)
    targets: list[MidpointTarget] = []
    for index in range(samples):
        theta = 2.0 * math.pi * float(index) / float(samples)
        normal = _unit(
            math.cos(theta) * basis_u + math.sin(theta) * basis_v,
            "antipodal midpoint",
        )
        targets.append(
            MidpointTarget(
                normal_w=normal,
                source_pair=str(source_pair),
                antipodal=True,
                theta_rad=theta,
                basis_u_w=basis_u.copy(),
                basis_v_w=basis_v.copy(),
                pair_dot=pair_dot,
            )
        )
    return tuple(targets)


def select_minimax_metrics(
    rows: Sequence[tuple[CandidateMetrics, int]],
    *,
    angle_bucket_rad: float = 1.0e-6,
    time_bucket_s: float = 1.0e-5,
    residual_bucket: float = 1.0e-9,
    support_bucket_m: float = 1.0e-6,
) -> int:
    """Select a branch by hard safety and worst-case rather than averages.

    The integer paired with each metric row is an opaque stable index.  This
    helper is public so the minimax/tie-break contract can be unit tested
    independently of MuJoCo.
    """

    if not rows:
        raise NeutralReadyError("cannot select from an empty candidate set")
    for name, value in (
        ("angle_bucket_rad", angle_bucket_rad),
        ("time_bucket_s", time_bucket_s),
        ("residual_bucket", residual_bucket),
        ("support_bucket_m", support_bucket_m),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise NeutralReadyError(f"{name} must be finite and positive")

    def key(row: tuple[CandidateMetrics, int]) -> tuple[Any, ...]:
        metrics, stable_index = row
        safety_rank = {
            "PASS_EXACT_MUJOCO_STATIC_READY": 0,
            "NOT_EVALUATED_FAIL_CLOSED": 1,
        }.get(metrics.safety_status, 2)
        support = metrics.minimum_support_margin_m
        return (
            safety_rank,
            _tolerance_bucket(
                metrics.maximum_pair_angular_asymmetry_rad,
                angle_bucket_rad,
            ),
            _tolerance_bucket(
                metrics.worst_right_arm_diagonal_time_proxy_s,
                time_bucket_s,
            ),
            _tolerance_bucket(
                metrics.maximum_contact_angle_rad,
                angle_bucket_rad,
            ),
            _tolerance_bucket(
                metrics.site_residual_m, residual_bucket
            ),
            _tolerance_bucket(
                metrics.target_normal_residual_rad, residual_bucket
            ),
            (
                math.inf
                if support is None
                else -_signed_tolerance_bucket(support, support_bucket_m)
            ),
            _tolerance_bucket(
                metrics.donor_ready_to_candidate_diagonal_time_proxy_s,
                time_bucket_s,
            ),
            metrics.candidate_sha256,
            int(stable_index),
        )

    return int(min(rows, key=key)[1])


def solve_neutral_ready_candidate(
    donor_ready_joint_pos: np.ndarray,
    donor_ready_root_pos_w: np.ndarray,
    donor_ready_root_quat_w: np.ndarray,
    contacts: Sequence[ContactPose],
    *,
    backend: face.RightRacketBackend,
    ready_binding: ReadySourceBinding,
    model_binding: ExactModelBinding | None = None,
    contact_source_proof: ContactSourceProof | None = None,
    config: NeutralReadyConfig | None = None,
) -> NeutralReadyResult:
    """Solve and audit one face-neutral donor-ready candidate.

    Without ``model_binding`` the function can still return an in-memory IK
    diagnostic, but its safety status is ``NOT_EVALUATED_FAIL_CLOSED`` and the
    publisher will refuse it.
    """

    cfg = NeutralReadyConfig() if config is None else config
    ready = _finite_vector(
        donor_ready_joint_pos, len(backend.joint_names), "donor ready joint_pos"
    )
    root_pos = _finite_vector(
        donor_ready_root_pos_w, 3, "donor ready root_pos_w"
    )
    root_quat = _ready_quaternion(
        donor_ready_root_quat_w, "donor ready root_quat_w"
    )
    source_path, source_sha = _verify_ready_source(
        ready_binding, ready, root_pos, root_quat
    )
    active = _active_indices(backend.joint_names)
    lower = _finite_vector(
        backend.position_lower, len(ready), "backend position_lower"
    )
    upper = _finite_vector(
        backend.position_upper, len(ready), "backend position_upper"
    )
    if np.any(lower > upper):
        raise NeutralReadyError("backend joint position bounds are inverted")
    _enforce_joint_limits(ready, lower, upper, cfg, "donor ready")

    ordered_contacts = _validate_contacts(
        contacts, backend=backend, config=cfg, joint_count=len(ready)
    )
    for contact in ordered_contacts:
        _enforce_joint_limits(
            np.asarray(contact.joint_pos, np.float64),
            lower,
            upper,
            cfg,
            contact.label,
        )

    donor_site, donor_rotation = backend.site_pose(ready, root_pos, root_quat)
    donor_site = _finite_vector(donor_site, 3, "donor ready site position")
    donor_rotation = _rotation(donor_rotation, "donor ready site rotation")
    exact_model = _verify_exact_model(backend, model_binding)
    source_proof_receipt = _verify_contact_source_proof(
        contact_source_proof,
        ordered_contacts,
        backend=backend,
        model_binding=model_binding,
    )

    (
        pair_locus_seeds,
        pair_receipts,
        face_axes,
        paired_normals,
    ) = _all_midpoint_targets(
        ordered_contacts,
        reference_rotation=donor_rotation,
        config=cfg,
    )
    global_angular = global_antipodal_angular_minimax_targets(
        face_axes,
        reference_rotation_w=donor_rotation,
        config=cfg,
        paired_normals_w=paired_normals,
    )
    search_targets = global_angular.targets
    if not search_targets:
        raise NeutralReadyError("global angular minimax produced no IK target")
    if len(search_targets) > cfg.maximum_target_normals_for_ik:
        raise NeutralReadyError(
            "global angular optimizer locus exceeds "
            "maximum_target_normals_for_ik; refusing a truncated search"
        )

    candidate_rows: list[
        tuple[
            face.SiteNormalCandidate,
            MidpointTarget,
            CandidateMetrics,
            SafetyCheck,
            list[dict[str, Any]],
        ]
    ] = []
    solved_target_count = 0
    for target_index, target in enumerate(search_targets):
        solved = _solve_target_multistart(
            ready,
            root_pos,
            root_quat,
            donor_site,
            donor_rotation,
            target,
            ordered_contacts,
            backend=backend,
            active=active,
            lower=lower,
            upper=upper,
            config=cfg,
            target_index=target_index,
        )
        if solved:
            solved_target_count += 1
        for candidate in solved:
            safety = _evaluate_safety(
                candidate.joint_pos,
                root_pos,
                root_quat,
                backend=backend,
                exact_model=exact_model,
                config=cfg,
            )
            metrics, connector_rows = _candidate_metrics(
                candidate,
                target,
                ready,
                ordered_contacts,
                backend=backend,
                active=active,
                config=cfg,
                safety=safety,
            )
            candidate_rows.append(
                (candidate, target, metrics, safety, connector_rows)
            )
    if not candidate_rows:
        raise NeutralReadyError(
            "no midpoint target produced an exact site/normal IK candidate"
        )

    selected_index = select_minimax_metrics(
        [(row[2], index) for index, row in enumerate(candidate_rows)],
        angle_bucket_rad=cfg.ranking_angle_bucket_rad,
        time_bucket_s=cfg.ranking_time_bucket_s,
        residual_bucket=cfg.ranking_residual_bucket,
        support_bucket_m=cfg.ranking_support_bucket_m,
    )
    candidate, target, metrics, safety, connector_rows = candidate_rows[
        selected_index
    ]
    nonactive = np.setdiff1d(np.arange(len(ready)), active)
    if not np.array_equal(candidate.joint_pos[nonactive], ready[nonactive]):
        raise NeutralReadyError("selected candidate changed a fixed joint")

    # Only exact static-safety PASS rows may leave the solver as inputs to the
    # outer 5 x 2 compiler.  Failed IK branches remain counted in the receipt
    # as diagnostics, but must never be serialized into the reusable pool.
    safe_candidate_rows = [row for row in candidate_rows if row[3].passed]
    candidate_pool = tuple(
        NeutralReadyPoolEntry(
            candidate=row[0],
            target=row[1],
            metrics=row[2],
            safety=row[3],
            connectors=tuple(row[4]),
        )
        for row in safe_candidate_rows
    )
    _reverify_contact_source_files(ordered_contacts)
    if contact_source_proof is not None:
        _reverify_contact_source_proof_files(source_proof_receipt)
    upstream_source_pose_reconstruction_verified = (
        source_proof_receipt is not None
    )
    all_global_targets_exact_ik = solved_target_count == len(search_targets)
    finite_global_locus = global_angular.finite_optimizer_locus
    global_bound_gap_pass = bool(
        global_angular.angular_objective_upper_bound_rad
        - global_angular.angular_objective_lower_bound_rad
        <= cfg.global_minimax_bound_gap_tolerance_rad
    )
    neutrality_pass = bool(
        metrics.maximum_pair_angular_asymmetry_rad
        <= cfg.maximum_neutrality_asymmetry_rad
    )
    all_exact_gates_pass = bool(
        exact_model["status"] == "PASS"
        and safety.passed
        and global_angular.angular_minimax_certified
        and finite_global_locus
        and global_bound_gap_pass
        and all_global_targets_exact_ik
        and neutrality_pass
        and upstream_source_pose_reconstruction_verified
        and metrics.site_residual_m <= cfg.site_tolerance_m
        and metrics.target_normal_residual_rad <= cfg.normal_tolerance_rad
    )
    receipt = _build_receipt(
        source_path=source_path,
        source_sha=source_sha,
        model_receipt=exact_model,
        ready=ready,
        root_pos=root_pos,
        root_quat=root_quat,
        ordered_contacts=ordered_contacts,
        joint_names=backend.joint_names,
        active=active,
        candidate=candidate,
        target=target,
        metrics=metrics,
        safety=safety,
        connector_rows=connector_rows,
        candidate_pool=candidate_pool,
        pair_receipts=pair_receipts,
        pair_locus_seed_count=len(pair_locus_seeds),
        global_angular=global_angular,
        solved_count=len(candidate_rows),
        rejected_safety_count=len(candidate_rows) - len(safe_candidate_rows),
        solved_target_count=solved_target_count,
        all_global_targets_exact_ik=all_global_targets_exact_ik,
        global_bound_gap_pass=global_bound_gap_pass,
        neutrality_pass=neutrality_pass,
        upstream_source_pose_reconstruction_verified=(
            upstream_source_pose_reconstruction_verified
        ),
        source_proof_receipt=source_proof_receipt,
        config=cfg,
        publication_allowed=all_exact_gates_pass,
    )
    _assert_strict_json(receipt)
    payload_seal = _result_payload_seal(
        candidate=candidate,
        target=target,
        metrics=metrics,
        root_pos=root_pos,
        root_quat=root_quat,
        receipt=receipt,
        source_ready_path=source_path,
        publication_allowed=all_exact_gates_pass,
        candidate_pool=candidate_pool,
        contacts=ordered_contacts,
        contact_source_proof=contact_source_proof,
    )
    return NeutralReadyResult(
        candidate=candidate,
        target=target,
        metrics=metrics,
        root_pos_w=root_pos.copy(),
        root_quat_w=root_quat.copy(),
        receipt=receipt,
        source_ready_path=source_path,
        publication_allowed=all_exact_gates_pass,
        backend=backend,
        model_binding=model_binding,
        payload_seal_sha256=payload_seal,
        candidate_pool=candidate_pool,
        contacts=ordered_contacts,
        contact_source_proof=contact_source_proof,
    )


def _publication_config_from_receipt(
    receipt: Mapping[str, Any],
) -> NeutralReadyConfig:
    """Recover only the code-approved publication policy from a receipt.

    Search breadth may vary, but callers cannot relax any geometric, safety,
    ranking, or model tolerance and then self-issue a publishable receipt.
    """

    try:
        raw = receipt["config"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise NeutralReadyError(
            f"candidate receipt lacks publication config: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise NeutralReadyError("candidate publication config must be a map")
    expected_keys = frozenset(asdict(NeutralReadyConfig()))
    if frozenset(raw) != expected_keys:
        raise NeutralReadyError(
            "candidate publication config exact field set changed"
        )
    detached = dict(raw)
    foot_names = detached.get("foot_body_names")
    if not isinstance(foot_names, (list, tuple)):
        raise NeutralReadyError(
            "candidate publication foot_body_names must be a sequence"
        )
    detached["foot_body_names"] = tuple(str(value) for value in foot_names)
    for name in ("random_restarts", "antipodal_circle_samples"):
        if isinstance(detached[name], bool) or not isinstance(
            detached[name], int
        ):
            raise NeutralReadyError(
                f"candidate publication {name} must be an integer"
            )
    try:
        config = NeutralReadyConfig(**detached)
    except (TypeError, ValueError) as exc:
        raise NeutralReadyError(
            f"candidate publication config is invalid: {exc}"
        ) from exc
    if config.random_restarts > 64:
        raise NeutralReadyError(
            "publication random_restarts exceeds the reviewed bound of 64"
        )
    if config.antipodal_circle_samples > 512:
        raise NeutralReadyError(
            "publication antipodal_circle_samples exceeds 512"
        )
    approved = NeutralReadyConfig(
        random_restarts=config.random_restarts,
        antipodal_circle_samples=config.antipodal_circle_samples,
    )
    if asdict(config) != asdict(approved):
        raise NeutralReadyError(
            "publication may vary search breadth only; geometric, safety, "
            "ranking, and model tolerances must equal code defaults"
        )
    return config


def _load_ready_replay_arrays(
    path: Path,
    expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the three replay inputs from the exact bytes that were hashed."""

    payload_bytes, actual_sha = _read_real_file_snapshot(
        path, "publication donor ready"
    )
    if actual_sha != expected_sha256:
        raise NeutralReadyError(
            "publication donor ready no longer matches its content pin"
        )
    try:
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as payload:
            return (
                np.asarray(payload["joint_pos"], dtype=np.float64).copy(),
                np.asarray(payload["root_pos_w"], dtype=np.float64).copy(),
                np.asarray(payload["root_quat_w"], dtype=np.float64).copy(),
            )
    except (KeyError, OSError, ValueError) as exc:
        raise NeutralReadyError(
            f"cannot reconstruct publication donor ready: {exc}"
        ) from exc


def _reaudit_result_for_publication(
    result: NeutralReadyResult,
) -> NeutralReadyResult:
    """Re-run the complete deterministic solve instead of trusting a seal.

    A SHA-256 stored next to caller-owned data is an integrity checksum, not an
    authenticity boundary: a caller can modify both.  Publication therefore
    reloads the donor, replays the trusted 16-row source adapter, redoes exact
    model/FK/IK/static-safety checks, rebuilds every reusable pool row, and
    requires byte-identical receipt semantics.
    """

    if not result.publication_allowed:
        raise NeutralReadyError(
            "candidate is not publication-eligible; exact model/safety gates "
            "remain fail-closed"
        )
    _require_publishable_receipt(result.receipt)
    if result.model_binding is None or result.contact_source_proof is None:
        raise NeutralReadyError(
            "publication requires retained exact model and contact-source proof"
        )
    config = _publication_config_from_receipt(result.receipt)
    actual_seal = _result_payload_seal(
        candidate=result.candidate,
        target=result.target,
        metrics=result.metrics,
        root_pos=result.root_pos_w,
        root_quat=result.root_quat_w,
        receipt=result.receipt,
        source_ready_path=result.source_ready_path,
        publication_allowed=result.publication_allowed,
        candidate_pool=result.candidate_pool,
        contacts=result.contacts,
        contact_source_proof=result.contact_source_proof,
    )
    if actual_seal != result.payload_seal_sha256:
        raise NeutralReadyError("candidate result payload changed after solve")
    try:
        source_receipt = result.receipt["source_ready"]
        receipt_path = Path(str(source_receipt["path"])).absolute()
        source_sha = _require_sha256(
            str(source_receipt["sha256"]),
            "publication source-ready sha256",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NeutralReadyError(
            f"publication source-ready receipt is invalid: {exc}"
        ) from exc
    source_path = Path(result.source_ready_path).expanduser().absolute()
    if source_path != receipt_path:
        raise NeutralReadyError(
            "result source-ready path disagrees with solved receipt"
        )
    ready, root_pos, root_quat = _load_ready_replay_arrays(
        source_path, source_sha
    )
    replayed = solve_neutral_ready_candidate(
        ready,
        root_pos,
        root_quat,
        result.contacts,
        backend=result.backend,
        ready_binding=ReadySourceBinding(source_path, source_sha),
        model_binding=result.model_binding,
        contact_source_proof=result.contact_source_proof,
        config=config,
    )
    if not replayed.publication_allowed:
        raise NeutralReadyError(
            "fresh publication replay no longer closes every exact gate"
        )
    if (
        replayed.payload_seal_sha256 != result.payload_seal_sha256
        or _canonical_json_bytes(replayed.receipt)
        != _canonical_json_bytes(result.receipt)
    ):
        raise NeutralReadyError(
            "fresh publication replay does not reproduce candidate/pool/receipt"
        )
    return replayed


def publish_neutral_ready_candidate(
    result: NeutralReadyResult,
    output_directory: str | Path,
) -> PublishedNeutralReady:
    """Publish only a fresh, independently replayed diagnostic bundle."""

    audited = _reaudit_result_for_publication(result)
    output = Path(output_directory).expanduser().absolute()
    source = audited.source_ready_path.absolute()
    if output == source:
        raise NeutralReadyError("candidate output may not overwrite donor ready")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise NeutralReadyError(
            f"candidate parent must be a real existing directory: {parent}"
        )
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite candidate bundle: {output}")

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=str(parent))
    )
    stage_identity = stage.stat()
    stage_fd = -1
    published = False
    try:
        stage_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        stage_flags |= getattr(os, "O_DIRECTORY", 0)
        stage_flags |= getattr(os, "O_NOFOLLOW", 0)
        stage_fd = os.open(stage, stage_flags)
        if not os.path.samestat(os.fstat(stage_fd), stage_identity):
            raise NeutralReadyError(
                "staging directory identity changed immediately after creation"
            )
        candidate_stage = stage / CANDIDATE_FILENAME
        _write_candidate_npz(candidate_stage, audited)
        candidate_sha = _sha256_file(candidate_stage)
        publication = {
            "candidate_filename": CANDIDATE_FILENAME,
            "candidate_npz_sha256": candidate_sha,
            "receipt_filename": RECEIPT_FILENAME,
            "completion_semantics": (
                "receipt_written_last_inside_exclusively_created_directory"
            ),
        }
        # Detach the published receipt from the independently replayed result.
        receipt = json.loads(
            _canonical_json_bytes(audited.receipt).decode("utf-8")
        )
        receipt["publication"] = publication
        canonical = _canonical_json_bytes(receipt)
        receipt["receipt_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        receipt_bytes = _canonical_json_bytes(receipt)
        _assert_strict_json(receipt)
        receipt_stage = stage / RECEIPT_FILENAME
        _exclusive_write(receipt_stage, receipt_bytes)
        _fsync_directory(stage)

        # Recheck mutable inputs and staged semantic bytes after all long
        # verification/writes, immediately before atomic no-replace publish.
        final_audited = _reaudit_result_for_publication(result)
        if (
            final_audited.payload_seal_sha256
            != audited.payload_seal_sha256
        ):
            raise NeutralReadyError(
                "independent replay changed while staging publication"
            )
        _validate_staged_candidate_npz(
            candidate_stage,
            result=final_audited,
            expected_sha256=candidate_sha,
        )
        expected_receipt = json.loads(
            _canonical_json_bytes(final_audited.receipt).decode("utf-8")
        )
        expected_receipt["publication"] = publication
        expected_canonical = _canonical_json_bytes(expected_receipt)
        expected_receipt["receipt_payload_sha256"] = hashlib.sha256(
            expected_canonical
        ).hexdigest()
        expected_receipt_bytes = _canonical_json_bytes(expected_receipt)
        if expected_receipt_bytes != receipt_bytes:
            raise NeutralReadyError(
                "staged receipt is a mixed or stale result snapshot"
            )
        if _sha256_file(receipt_stage) != hashlib.sha256(
            receipt_bytes
        ).hexdigest():
            raise NeutralReadyError(
                "staged receipt bytes changed before publication"
            )

        _rename_directory_noreplace(stage, output)
        published = True
        _fsync_directory(parent)
        receipt_sha = _sha256_file(output / RECEIPT_FILENAME)
        return PublishedNeutralReady(
            directory=output,
            candidate_npz=output / CANDIDATE_FILENAME,
            receipt_json=output / RECEIPT_FILENAME,
            candidate_npz_sha256=candidate_sha,
            receipt_json_sha256=receipt_sha,
        )
    finally:
        if not published:
            _cleanup_owned_stage(stage, stage_identity, stage_fd)
        if stage_fd >= 0:
            os.close(stage_fd)


def _all_midpoint_targets(
    contacts: Sequence[ContactPose],
    *,
    reference_rotation: np.ndarray,
    config: NeutralReadyConfig,
) -> tuple[
    tuple[MidpointTarget, ...],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    lookup = {(row.scope, row.phase, row.face_name): row for row in contacts}
    targets: list[MidpointTarget] = []
    pair_receipts: list[dict[str, Any]] = []
    axes: list[np.ndarray] = []
    paired_normals: list[np.ndarray] = []
    for scope in SCOPES:
        for phase in PHASES:
            bh = lookup[(scope, phase, "bh")]
            fh = lookup[(scope, phase, "fh")]
            pair_label = f"{scope}:{phase}:bh_fh"
            separation = _angle(
                bh.signed_face_normal_w, fh.signed_face_normal_w
            )
            antipodal_error = abs(math.pi - separation)
            site_error = float(
                np.linalg.norm(bh.site_pos_w - fh.site_pos_w)
            )
            if (
                antipodal_error
                > config.paired_face_antipodal_tolerance_rad
            ):
                raise NeutralReadyError(
                    f"{pair_label} is not a signed antipodal BH/FH pair: "
                    f"pi-angle={antipodal_error:.9g} rad"
                )
            if site_error > config.paired_site_tolerance_m:
                raise NeutralReadyError(
                    f"{pair_label} BH/FH racket sites differ by "
                    f"{site_error:.9g} m"
                )
            expected_pair_contract = contact_pair_contract_digest(bh, fh)
            if (
                bh.source_frame_index != fh.source_frame_index
                or bh.pair_contract_sha256 != expected_pair_contract
                or fh.pair_contract_sha256 != expected_pair_contract
            ):
                raise NeutralReadyError(
                    f"{pair_label} lacks one shared frame/pair content contract"
                )
            axis = _unit(
                bh.signed_face_normal_w - fh.signed_face_normal_w,
                f"{pair_label} signed face axis",
            )
            axes.append(axis)
            paired_normals.append(
                np.stack(
                    (
                        bh.signed_face_normal_w,
                        fh.signed_face_normal_w,
                    )
                )
            )
            rows = spherical_geodesic_midpoint_targets(
                bh.signed_face_normal_w,
                fh.signed_face_normal_w,
                reference_rotation_w=reference_rotation,
                source_pair=pair_label,
                antipodal_dot_tolerance=config.antipodal_dot_tolerance,
                antipodal_circle_samples=config.antipodal_circle_samples,
            )
            targets.extend(rows)
            pair_receipts.append(
                {
                    "pair": pair_label,
                    "dot": float(rows[0].pair_dot),
                    "separation_rad": separation,
                    "antipodal_error_rad": antipodal_error,
                    "paired_site_residual_m": site_error,
                    "source_frame_index": bh.source_frame_index,
                    "pair_contract_sha256": bh.pair_contract_sha256,
                    "bh_pose_content_sha256": (
                        bh.pose_content_sha256
                    ),
                    "fh_pose_content_sha256": (
                        fh.pose_content_sha256
                    ),
                    "face_axis_w": axis.tolist(),
                    "antipodal_degeneracy": bool(rows[0].antipodal),
                    "sampled_pair_locus_seed_count": len(rows),
                    "basis_u_w": (
                        None
                        if rows[0].basis_u_w is None
                        else rows[0].basis_u_w.tolist()
                    ),
                    "basis_v_w": (
                        None
                        if rows[0].basis_v_w is None
                        else rows[0].basis_v_w.tolist()
                    ),
                    "sampled_theta_grid_rad": [
                        float(row.theta_rad)
                        for row in rows
                        if row.theta_rad is not None
                    ],
                }
            )
    return (
        _deduplicate_targets(targets),
        pair_receipts,
        np.asarray(axes, dtype=np.float64),
        np.asarray(paired_normals, dtype=np.float64),
    )


def global_antipodal_angular_minimax_targets(
    face_axes_w: np.ndarray,
    *,
    reference_rotation_w: np.ndarray,
    config: NeutralReadyConfig,
    paired_normals_w: np.ndarray | None = None,
) -> GlobalAngularSolution:
    """Globally minimize worst BH/FH angular asymmetry for antipodal pairs.

    For an antipodal pair with unit axis ``a``, angular asymmetry is
    ``2*asin(abs(a dot n))``.  The global problem therefore minimizes
    ``max_i abs(a_i dot n)`` over unit normals ``n``.

    If the axes span R3, let ``P={x: |A x|<=1}``.  The optimum equals
    ``1/max_{x in P} ||x||``; the maximum of the convex norm over this bounded
    polytope occurs at a vertex.  Enumerating every independent three-plane
    intersection and sign pattern is consequently an exact finite proof in
    three dimensions.  Rank-two axes have the two nullspace directions as the
    exact finite locus.  Rank-one axes have a continuous great-circle optimum;
    that locus is explicitly labelled sampled, never exhaustive.
    """

    axes = np.asarray(face_axes_w, dtype=np.float64)
    if (
        axes.ndim != 2
        or axes.shape[1] != 3
        or len(axes) < 1
        or not np.isfinite(axes).all()
    ):
        raise NeutralReadyError("face_axes_w must be a finite (N,3) array")
    axes = np.asarray(
        [_unit(row, "global face axis") for row in axes],
        dtype=np.float64,
    )
    if paired_normals_w is None:
        paired = np.stack((axes, -axes), axis=1)
    else:
        paired = np.asarray(paired_normals_w, dtype=np.float64)
        if (
            paired.shape != (len(axes), 2, 3)
            or not np.isfinite(paired).all()
        ):
            raise NeutralReadyError(
                "paired_normals_w must have finite shape (N,2,3)"
            )
        paired = np.asarray(
            [
                [_unit(row[0], "paired BH"), _unit(row[1], "paired FH")]
                for row in paired
            ],
            dtype=np.float64,
        )
    pair_errors = np.asarray(
        [
            abs(math.pi - _angle(row[0], row[1]))
            for row in paired
        ],
        dtype=np.float64,
    )
    maximum_pair_error = float(np.max(pair_errors))
    reference = _rotation(reference_rotation_w, "reference_rotation_w")
    singular = np.linalg.svd(axes, compute_uv=False)
    rank_tolerance = max(
        max(axes.shape) * np.finfo(float).eps * singular[0] * 64.0,
        math.sqrt(float(len(axes)))
        * max(maximum_pair_error, config.normal_tolerance_rad)
        * 0.5,
    )
    rank = int(np.linalg.matrix_rank(axes, tol=rank_tolerance))
    targets: tuple[MidpointTarget, ...]
    method: str
    finite_locus: bool
    sampled_locus: bool
    vertex_count = 0
    feasible_vertex_count = 0

    if rank == 3:
        vertices: list[np.ndarray] = []
        feasibility_tolerance = 2.0e-9
        for indices in combinations(range(len(axes)), 3):
            matrix = axes[np.asarray(indices, dtype=np.int64)]
            matrix_singular = np.linalg.svd(matrix, compute_uv=False)
            if float(matrix_singular[-1]) <= rank_tolerance:
                continue
            for signs in product((-1.0, 1.0), repeat=3):
                vertex_count += 1
                value = np.linalg.solve(
                    matrix, np.asarray(signs, dtype=np.float64)
                )
                if float(np.max(np.abs(axes @ value))) <= (
                    1.0 + feasibility_tolerance
                ):
                    feasible_vertex_count += 1
                    vertices.append(value)
        if not vertices:
            raise NeutralReadyError(
                "rank-three antipodal minimax polytope has no verified vertex"
            )
        norms = np.asarray([np.linalg.norm(row) for row in vertices])
        maximum_norm = float(np.max(norms))
        if not np.isfinite(maximum_norm) or maximum_norm <= 0.0:
            raise NeutralReadyError("antipodal minimax vertex norm is invalid")
        norm_tolerance = max(1.0e-10, maximum_norm * 2.0e-9)
        directions = [
            _unit(row, "global minimax direction")
            for row, norm in zip(vertices, norms)
            if maximum_norm - float(norm) <= norm_tolerance
        ]
        unique: list[np.ndarray] = []
        for direction in sorted(directions, key=_vector_tie_key):
            if any(_angle(direction, kept) <= 1.0e-9 for kept in unique):
                continue
            unique.append(direction)
        targets = tuple(
            MidpointTarget(
                normal_w=direction,
                source_pair="global_antipodal_axis_minimax",
                antipodal=True,
                theta_rad=None,
                basis_u_w=None,
                basis_v_w=None,
                pair_dot=-1.0,
            )
            for direction in unique
        )
        optimum = 1.0 / maximum_norm
        method = "exact_rank3_symmetric_polytope_vertex_enumeration_v1"
        finite_locus = True
        sampled_locus = False
    elif rank == 2:
        _, _, vh = np.linalg.svd(axes, full_matrices=True)
        direction = _unit(vh[-1], "rank-two axis nullspace")
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction = -direction
        targets = tuple(
            MidpointTarget(
                normal_w=sign * direction,
                source_pair="global_antipodal_axis_nullspace",
                antipodal=True,
                theta_rad=None,
                basis_u_w=None,
                basis_v_w=None,
                pair_dot=-1.0,
            )
            for sign in (1.0, -1.0)
        )
        optimum = 0.0
        method = "exact_rank2_nullspace_locus_v1"
        finite_locus = True
        sampled_locus = False
    elif rank == 1:
        axis = _unit(axes[0], "rank-one face axis")
        targets = spherical_geodesic_midpoint_targets(
            axis,
            -axis,
            reference_rotation_w=reference,
            source_pair="global_rank1_continuous_minimax_locus_sample",
            antipodal_dot_tolerance=config.antipodal_dot_tolerance,
            antipodal_circle_samples=config.antipodal_circle_samples,
        )
        optimum = 0.0
        method = "exact_rank1_angular_locus_sampled_for_kinematic_tie_break_v1"
        finite_locus = False
        sampled_locus = True
    else:  # pragma: no cover - nonzero unit axes guarantee rank >= 1
        raise NeutralReadyError("face-axis matrix unexpectedly has rank zero")

    verified = max(
        float(np.max(np.abs(axes @ target.normal_w)))
        for target in targets
    )
    if rank == 3 and verified > optimum + 5.0e-8:
        raise NeutralReadyError(
            "global antipodal minimax target does not attain its lower bound"
        )
    ideal_lower = 2.0 * math.asin(
        float(np.clip(optimum, 0.0, 1.0))
    )
    actual_objectives = [
        max(
            abs(
                _angle(target.normal_w, row[0])
                - _angle(target.normal_w, row[1])
            )
            for row in paired
        )
        for target in targets
    ]
    lower = max(0.0, ideal_lower - maximum_pair_error)
    upper = float(min(actual_objectives))
    return GlobalAngularSolution(
        targets=targets,
        axis_rank=rank,
        method=method,
        maximum_absolute_axis_projection=verified,
        angular_objective_lower_bound_rad=lower,
        angular_objective_upper_bound_rad=upper,
        angular_minimax_certified=True,
        finite_optimizer_locus=finite_locus,
        sampled_continuous_locus=sampled_locus,
        report={
            "face_axis_count": int(len(axes)),
            "face_axes_sha256": _array_sha256(axes),
            "singular_values": singular.tolist(),
            "rank_tolerance": rank_tolerance,
            "paired_antipodal_error_rad": pair_errors.tolist(),
            "maximum_paired_antipodal_error_rad": maximum_pair_error,
            "ideal_axis_objective_lower_bound_rad": ideal_lower,
            "actual_objective_per_target_rad": actual_objectives,
            "certified_objective_gap_rad": upper - lower,
            "polytope_intersection_count": vertex_count,
            "polytope_feasible_vertex_count": feasible_vertex_count,
            "optimizer_target_count": len(targets),
            "proof": (
                "min_unit_max_abs_A_n_equals_inverse_max_norm_over_"
                "symmetric_polytope_abs_A_x_le_one"
            ),
            "continuous_locus_time_tie_break_is_sampled": sampled_locus,
        },
    )


def _deduplicate_targets(
    targets: Iterable[MidpointTarget],
    *,
    tolerance_rad: float = 1.0e-10,
) -> tuple[MidpointTarget, ...]:
    kept: list[MidpointTarget] = []
    for target in targets:
        if any(
            _angle(target.normal_w, row.normal_w) <= tolerance_rad
            for row in kept
        ):
            continue
        kept.append(target)
    return tuple(kept)


def _shortlist_target_normals(
    targets: Sequence[MidpointTarget],
    contacts: Sequence[ContactPose],
    config: NeutralReadyConfig,
) -> tuple[MidpointTarget, ...]:
    lookup = {(row.scope, row.phase, row.face_name): row for row in contacts}

    def score(target: MidpointTarget) -> tuple[Any, ...]:
        asymmetry = 0.0
        maximum = 0.0
        for scope in SCOPES:
            for phase in PHASES:
                bh = lookup[(scope, phase, "bh")]
                fh = lookup[(scope, phase, "fh")]
                bh_angle = _angle(target.normal_w, bh.signed_face_normal_w)
                fh_angle = _angle(target.normal_w, fh.signed_face_normal_w)
                asymmetry = max(asymmetry, abs(bh_angle - fh_angle))
                maximum = max(maximum, bh_angle, fh_angle)
        return (
            asymmetry,
            maximum,
            _vector_tie_key(target.normal_w),
            target.source_pair,
            -1.0 if target.theta_rad is None else target.theta_rad,
        )

    ordered = sorted(targets, key=score)
    return tuple(ordered[: config.maximum_target_normals_for_ik])


def _solve_target_multistart(
    ready: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    donor_site: np.ndarray,
    donor_rotation: np.ndarray,
    target: MidpointTarget,
    contacts: Sequence[ContactPose],
    *,
    backend: face.RightRacketBackend,
    active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: NeutralReadyConfig,
    target_index: int,
) -> tuple[face.SiteNormalCandidate, ...]:
    active_lower = lower[active]
    active_upper = upper[active]
    seeds: list[np.ndarray] = [ready[active].copy()]
    seeds.extend(np.asarray(row.joint_pos)[active].copy() for row in contacts)
    seeds.extend(
        0.5 * (ready[active] + np.asarray(row.joint_pos)[active])
        for row in contacts
    )
    rng = np.random.default_rng(config.random_seed + 7919 * target_index)
    for _ in range(config.random_restarts):
        seeds.append(
            active_lower + rng.random(len(active)) * (active_upper - active_lower)
        )
    unique_seeds: list[np.ndarray] = []
    for seed in seeds:
        if any(
            float(np.max(np.abs(seed - kept)))
            <= config.candidate_dedup_tolerance_rad
            for kept in unique_seeds
        ):
            continue
        unique_seeds.append(np.asarray(seed, dtype=np.float64).copy())
    seeds = unique_seeds

    base_config = face.FaceManifoldConfig(
        mode="normal",
        position_tolerance_m=config.site_tolerance_m,
        normal_tolerance_rad=config.normal_tolerance_rad,
        orientation_tolerance_rad=config.normal_tolerance_rad,
        joint_limit_tolerance_rad=config.joint_limit_tolerance_rad,
        finite_difference_step_rad=config.finite_difference_step_rad,
        damping=config.damping,
        max_iterations=config.max_iterations,
        random_restarts=0,
        max_solver_step_rad=config.max_solver_step_rad,
        connector_dynamics_samples=config.connector_dynamics_samples,
    )
    bootstrap_rotation = face.rotation_with_raw_plus_y(
        target.normal_w, donor_rotation
    )
    projected_seeds: list[np.ndarray] = []
    # Full-SO(3) projection is only a bootstrap; normal-only IK is the actual
    # candidate manifold and can choose its in-plane rotation freely.
    bootstrap_config = replace(base_config, mode="full-so3")
    for seed in seeds:
        solved, _ = face._iterate_hierarchical(
            np.clip(seed, active_lower, active_upper),
            ready,
            root_pos,
            root_quat,
            target_position=donor_site,
            target_normal=target.normal_w,
            target_rotation=bootstrap_rotation,
            backend=backend,
            active=active,
            lower=active_lower,
            upper=active_upper,
            config=bootstrap_config,
        )
        projected_seeds.extend((seed, solved))

    candidates: list[face.SiteNormalCandidate] = []
    for restart_index, seed in enumerate(projected_seeds):
        solved, iterations = face._iterate_hierarchical(
            np.clip(seed, active_lower, active_upper),
            ready,
            root_pos,
            root_quat,
            target_position=donor_site,
            target_normal=target.normal_w,
            target_rotation=None,
            backend=backend,
            active=active,
            lower=active_lower,
            upper=active_upper,
            config=base_config,
        )
        q = ready.copy()
        q[active] = solved
        position, rotation = backend.site_pose(q, root_pos, root_quat)
        position = _finite_vector(position, 3, "solved site position")
        rotation = _rotation(rotation, "solved site rotation")
        site_residual = float(np.linalg.norm(position - donor_site))
        normal_residual = _angle(rotation[:, 1], target.normal_w)
        violation = max(
            0.0,
            float(np.max(active_lower - solved)),
            float(np.max(solved - active_upper)),
        )
        if (
            site_residual > config.site_tolerance_m
            or normal_residual > config.normal_tolerance_rad
            or violation > config.joint_limit_tolerance_rad
        ):
            continue
        if not np.array_equal(
            q[np.setdiff1d(np.arange(len(q)), active)],
            ready[np.setdiff1d(np.arange(len(q)), active)],
        ):
            raise NeutralReadyError("IK changed a fixed joint")
        if any(
            float(
                np.max(
                    np.abs(
                        row.joint_pos[active]
                        - q[active]
                    )
                )
            )
            <= config.candidate_dedup_tolerance_rad
            for row in candidates
        ):
            continue
        original_connector = _right_arm_diagonal_time_proxy(
            source_q=ready,
            value_active=q[active],
            ready=ready,
            root_pos=root_pos,
            root_quat=root_quat,
            backend=backend,
            active=active,
            velocity_limit=np.asarray(backend.velocity_limit)[active],
            effort_limit=np.asarray(backend.effort_limit)[active],
            config=base_config,
            start_label="donor_ready_v1",
            end_label="neutral_ready_candidate",
        )
        diagnostics = face.FrameDiagnostics(
            frame_index=-1,
            position_residual_m=site_residual,
            normal_residual_rad=normal_residual,
            orientation_residual_rad=None,
            max_joint_limit_violation_rad=violation,
            max_step_from_previous_rad=None,
            ready_max_abs_delta_rad=float(
                np.max(np.abs(q[active] - ready[active]))
            ),
            previous_max_abs_delta_rad=None,
            ready_connector_time_lower_bound_s=(
                original_connector.time_lower_bound_s
            ),
            ready_connector_limiting_joint=(
                original_connector.limiting_joint
            ),
            iterations=int(iterations),
            restart_index=int(restart_index),
        )
        candidates.append(
            face.SiteNormalCandidate(
                joint_pos=q,
                site_pos_w=position,
                raw_plus_y_w=rotation[:, 1].copy(),
                diagnostics=diagnostics,
                original_ready_connector=original_connector,
            )
        )
    return tuple(candidates)


def _right_arm_diagonal_time_proxy(
    source_q: np.ndarray,
    value_active: np.ndarray,
    ready: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    *,
    backend: face.RightRacketBackend,
    active: np.ndarray,
    velocity_limit: np.ndarray,
    effort_limit: np.ndarray,
    config: face.FaceManifoldConfig,
    start_label: str,
    end_label: str,
) -> face.ConnectorDiagnostics:
    """Return the local seven-axis tie-break proxy with symmetric effort.

    A rest-to-rest move needs actuator authority in both acceleration
    directions.  At every straight-line sample the usable symmetric magnitude
    is therefore ``effort_limit - abs(zero_speed_bias)``.  Using
    ``effort_limit - bias`` would incorrectly reward negative bias and change
    candidate ranking under a mere sign reversal.
    """

    source = _finite_vector(source_q, len(backend.joint_names), "source_q")
    start = _finite_vector(ready, len(source), "ready")[active]
    end = _finite_vector(value_active, len(active), "value_active")
    velocity = _finite_vector(
        velocity_limit, len(active), "velocity_limit"
    )
    effort = _finite_vector(effort_limit, len(active), "effort_limit")
    if np.any(velocity <= 0.0) or np.any(effort <= 0.0):
        raise NeutralReadyError(
            "diagonal timing proxy requires positive velocity/effort limits"
        )
    acceleration = np.full(len(active), np.inf, dtype=np.float64)
    for fraction in np.linspace(
        0.0, 1.0, config.connector_dynamics_samples
    ):
        q = source.copy()
        q[active] = (1.0 - fraction) * start + fraction * end
        diagonal, bias = backend.diagonal_dynamics(q, root_pos, root_quat)
        diagonal = _finite_vector(
            diagonal, len(source), "diagonal dynamics mass"
        )
        bias = _finite_vector(bias, len(source), "zero-speed dynamics bias")
        if np.any(diagonal[active] <= 0.0):
            joint = face.RIGHT_STRIKE_CHAIN[
                int(np.argmin(diagonal[active]))
            ]
            raise NeutralReadyError(
                f"diagonal timing proxy has non-positive inertia at {joint!r}"
            )
        available = effort - np.abs(bias[active])
        if np.any(available <= 0.0):
            joint = face.RIGHT_STRIKE_CHAIN[int(np.argmin(available))]
            raise NeutralReadyError(
                "diagonal timing proxy has no bidirectional static effort "
                f"margin at {joint!r}"
            )
        acceleration = np.minimum(
            acceleration, available / diagonal[active]
        )
    velocity = velocity * config.velocity_limit_fraction
    displacement = np.abs(end - start)
    triangular_distance = np.square(velocity) / acceleration
    per_joint = np.where(
        displacement <= triangular_distance,
        2.0 * np.sqrt(displacement / acceleration),
        displacement / velocity + velocity / acceleration,
    )
    limiting = int(np.argmax(per_joint))
    return face.ConnectorDiagnostics(
        start=start_label,
        end=end_label,
        time_lower_bound_s=float(per_joint[limiting]),
        limiting_joint=face.RIGHT_STRIKE_CHAIN[limiting],
        per_joint_time_lower_bound_s=tuple(
            float(value) for value in per_joint
        ),
        velocity_limit_rad_s=tuple(float(value) for value in velocity),
        acceleration_lower_envelope_rad_s2=tuple(
            float(value) for value in acceleration
        ),
        method="neutral_abs_bias_diagonal_timing_proxy_v1",
    )


def _candidate_metrics(
    candidate: face.SiteNormalCandidate,
    target: MidpointTarget,
    ready: np.ndarray,
    contacts: Sequence[ContactPose],
    *,
    backend: face.RightRacketBackend,
    active: np.ndarray,
    config: NeutralReadyConfig,
    safety: SafetyCheck,
) -> tuple[CandidateMetrics, list[dict[str, Any]]]:
    lookup = {(row.scope, row.phase, row.face_name): row for row in contacts}
    maximum_asymmetry = 0.0
    maximum_angle = 0.0
    for scope in SCOPES:
        for phase in PHASES:
            bh = lookup[(scope, phase, "bh")]
            fh = lookup[(scope, phase, "fh")]
            bh_angle = _angle(candidate.raw_plus_y_w, bh.signed_face_normal_w)
            fh_angle = _angle(candidate.raw_plus_y_w, fh.signed_face_normal_w)
            maximum_asymmetry = max(
                maximum_asymmetry, abs(bh_angle - fh_angle)
            )
            maximum_angle = max(maximum_angle, bh_angle, fh_angle)

    cfm_config = face.FaceManifoldConfig(
        mode="normal",
        position_tolerance_m=config.site_tolerance_m,
        normal_tolerance_rad=config.normal_tolerance_rad,
        orientation_tolerance_rad=config.normal_tolerance_rad,
        joint_limit_tolerance_rad=config.joint_limit_tolerance_rad,
        random_restarts=0,
        connector_dynamics_samples=config.connector_dynamics_samples,
    )
    connector_rows: list[dict[str, Any]] = []
    for contact in contacts:
        connector = _right_arm_diagonal_time_proxy(
            source_q=np.asarray(contact.joint_pos, np.float64),
            value_active=np.asarray(contact.joint_pos, np.float64)[active],
            ready=candidate.joint_pos,
            root_pos=np.asarray(contact.root_pos_w, np.float64),
            root_quat=np.asarray(contact.root_quat_w, np.float64),
            backend=backend,
            active=active,
            velocity_limit=np.asarray(backend.velocity_limit)[active],
            effort_limit=np.asarray(backend.effort_limit)[active],
            config=cfm_config,
            start_label="neutral_ready_candidate",
            end_label=contact.label,
        )
        connector_rows.append(
            {
                "contact": contact.label,
                "diagonal_time_proxy_s": connector.time_lower_bound_s,
                "limiting_joint": connector.limiting_joint,
                "per_joint_diagonal_time_proxy_s": list(
                    connector.per_joint_time_lower_bound_s
                ),
                "timing_proxy_method": connector.method,
                "interpretation": (
                    "right_arm_diagonal_timing_proxy_not_a_certified_"
                    "minimum_time_lower_bound"
                ),
            }
        )
    worst = max(
        connector_rows,
        key=lambda row: (
            float(row["diagonal_time_proxy_s"]),
            str(row["contact"]),
        ),
    )
    changed = int(
        np.count_nonzero(
            np.abs(candidate.joint_pos[active] - ready[active]) > 1.0e-8
        )
    )
    digest = _array_sha256(candidate.joint_pos)
    return (
        CandidateMetrics(
            maximum_pair_angular_asymmetry_rad=maximum_asymmetry,
            maximum_contact_angle_rad=maximum_angle,
            worst_right_arm_diagonal_time_proxy_s=float(
                worst["diagonal_time_proxy_s"]
            ),
            worst_connector_label=str(worst["contact"]),
            donor_ready_to_candidate_diagonal_time_proxy_s=(
                candidate.original_ready_connector.time_lower_bound_s
            ),
            site_residual_m=candidate.diagnostics.position_residual_m,
            target_normal_residual_rad=candidate.diagnostics.normal_residual_rad,
            minimum_support_margin_m=safety.support_margin_m,
            safety_status=safety.status,
            right_arm_changed_joint_count=changed,
            candidate_sha256=digest,
        ),
        connector_rows,
    )


def _validate_contacts(
    contacts: Sequence[ContactPose],
    *,
    backend: face.RightRacketBackend,
    config: NeutralReadyConfig,
    joint_count: int,
) -> tuple[ContactPose, ...]:
    expected = {
        (scope, phase, face_name)
        for scope in SCOPES
        for phase in PHASES
        for face_name in FACES
    }
    actual: dict[tuple[str, str, str], ContactPose] = {}
    for row in contacts:
        key = (row.scope, row.phase, row.face_name)
        if key not in expected:
            raise NeutralReadyError(f"unexpected contact identity {key!r}")
        if key in actual:
            raise NeutralReadyError(f"duplicate contact identity {key!r}")
        q = _finite_vector(row.joint_pos, joint_count, f"{row.label} joint_pos")
        root_pos = _finite_vector(
            row.root_pos_w, 3, f"{row.label} root_pos_w"
        )
        root_quat = _ready_quaternion(
            row.root_quat_w, f"{row.label} root_quat_w"
        )
        supplied_pos = _finite_vector(
            row.site_pos_w, 3, f"{row.label} site_pos_w"
        )
        supplied_rotation = _rotation(
            row.site_rotation_w, f"{row.label} site_rotation_w"
        )
        # The pose digest seals the caller's exact float64 bytes. Validating
        # by normalizing here would silently replace those bytes and make an
        # otherwise valid unit vector fail its own content digest whenever a
        # second floating-point normalization rounds differently. Require a
        # unit vector, but preserve the sealed representation.
        supplied_normal = _unit_preserve(
            row.signed_face_normal_w,
            f"{row.label} signed_face_normal_w",
        )
        if _angle(supplied_rotation[:, 1], supplied_normal) > 1.0e-10:
            raise NeutralReadyError(
                f"{row.label} signed normal disagrees with site local +Y"
            )
        if (
            isinstance(row.source_frame_index, bool)
            or not isinstance(row.source_frame_index, (int, np.integer))
            or int(row.source_frame_index) < 0
        ):
            raise NeutralReadyError(
                f"{row.label} source_frame_index must be non-negative integer"
            )
        try:
            _require_sha256(
                row.source_motion_sha256,
                f"{row.label} source_motion_sha256",
            )
            _require_sha256(
                row.pose_content_sha256,
                f"{row.label} pose_content_sha256",
            )
            _require_sha256(
                row.pair_contract_sha256,
                f"{row.label} pair_contract_sha256",
            )
        except ValueError as exc:
            raise NeutralReadyError(str(exc)) from exc
        source_motion_path = Path(
            row.source_motion_path
        ).expanduser().absolute()
        if (
            not source_motion_path.is_file()
            or source_motion_path.is_symlink()
            or _sha256_file(source_motion_path)
            != row.source_motion_sha256
        ):
            raise NeutralReadyError(
                f"{row.label} source motion file/hash binding failed"
            )
        exact_pos, exact_rotation = backend.site_pose(q, root_pos, root_quat)
        exact_pos = _finite_vector(
            exact_pos, 3, f"{row.label} exact site position"
        )
        exact_rotation = _rotation(
            exact_rotation, f"{row.label} exact site rotation"
        )
        position_error = float(np.linalg.norm(exact_pos - supplied_pos))
        rotation_error = float(
            np.linalg.norm(face.rotation_log(supplied_rotation.T @ exact_rotation))
        )
        if (
            position_error > config.input_site_tolerance_m
            or rotation_error > config.input_rotation_tolerance_rad
        ):
            raise NeutralReadyError(
                f"{row.label} supplied pose is not exact backend FK: "
                f"position={position_error:.9g} m rotation={rotation_error:.9g} rad"
            )
        checked = ContactPose(
            scope=row.scope,
            phase=row.phase,
            face_name=row.face_name,
            joint_pos=q,
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            site_pos_w=supplied_pos,
            site_rotation_w=supplied_rotation,
            signed_face_normal_w=supplied_normal,
            source_motion_path=source_motion_path,
            source_motion_sha256=str(row.source_motion_sha256),
            source_frame_index=int(row.source_frame_index),
            pose_content_sha256=str(row.pose_content_sha256),
            pair_contract_sha256=str(row.pair_contract_sha256),
        )
        if contact_pose_digest(checked) != checked.pose_content_sha256:
            raise NeutralReadyError(
                f"{row.label} pose content digest is not content-derived"
            )
        actual[key] = checked
    missing = expected - set(actual)
    if missing:
        raise NeutralReadyError(
            f"contact matrix is incomplete; missing={sorted(missing)}"
        )
    return tuple(
        actual[(scope, phase, face_name)]
        for scope in SCOPES
        for phase in PHASES
        for face_name in FACES
    )


def contact_pose_digest(contact: ContactPose) -> str:
    """Canonical digest of caller-supplied pose content and source labels.

    This does not reconstruct the pose from the named source frame and must
    never be presented as an upstream lineage proof.
    """

    payload = {
        "schema_version": 1,
        "scope": str(contact.scope),
        "phase": str(contact.phase),
        "face_name": str(contact.face_name),
        "source_motion_sha256": str(contact.source_motion_sha256),
        "source_frame_index": int(contact.source_frame_index),
        "normal_convention": NORMAL_CONVENTION,
        "array_sha256": {
            name: _array_sha256(np.asarray(getattr(contact, name)))
            for name in (
                "joint_pos",
                "root_pos_w",
                "root_quat_w",
                "site_pos_w",
                "site_rotation_w",
                "signed_face_normal_w",
            )
        },
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def contact_pair_contract_digest(
    bh: ContactPose, fh: ContactPose
) -> str:
    """Bind both exact poses to one same-site signed-antipodal pair contract."""

    if bh.scope != fh.scope or bh.phase != fh.phase:
        raise NeutralReadyError("contact pair scope/phase mismatch")
    if bh.face_name != "bh" or fh.face_name != "fh":
        raise NeutralReadyError("contact pair must be ordered BH then FH")
    payload = {
        "schema_version": 1,
        "scope": bh.scope,
        "phase": bh.phase,
        "source_frame_index": [
            int(bh.source_frame_index),
            int(fh.source_frame_index),
        ],
        "source_motion_sha256": [
            bh.source_motion_sha256,
            fh.source_motion_sha256,
        ],
        "pose_sha256": [
            contact_pose_digest(bh),
            contact_pose_digest(fh),
        ],
        "transform_contract": (
            "same_site_signed_antipodal_right_racket_plus_y_pair_v1"
        ),
        "normal_convention": NORMAL_CONVENTION,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _verify_ready_source(
    binding: ReadySourceBinding,
    ready: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> tuple[Path, str]:
    path = Path(binding.path).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise NeutralReadyError(
            f"donor ready must be a real regular NPZ file: {path}"
        )
    ready_bytes, actual_sha = _read_real_file_snapshot(
        path, "donor ready"
    )
    if actual_sha != binding.expected_sha256:
        raise NeutralReadyError(
            "donor ready SHA-256 mismatch: "
            f"{actual_sha} != {binding.expected_sha256}"
        )
    try:
        import canonical_mujoco_dynamics_gate as dynamics_gate

        runtime_joint_names = tuple(dynamics_gate.RUNTIME_JOINT_NAMES)
        with np.load(io.BytesIO(ready_bytes), allow_pickle=False) as payload:
            keys = frozenset(payload.files)
            if keys != CANONICAL_READY_KEYS:
                raise NeutralReadyError(
                    "donor ready NPZ exact field set changed: "
                    f"got={sorted(keys)} expected={sorted(CANONICAL_READY_KEYS)}"
                )
            file_joint = np.asarray(payload["joint_pos"])
            file_joint_vel = np.asarray(payload["joint_vel"])
            file_root_pos = np.asarray(payload["root_pos_w"])
            file_root_quat = np.asarray(payload["root_quat_w"])
            for key, value, shape in (
                ("joint_pos", file_joint, (len(runtime_joint_names),)),
                ("joint_vel", file_joint_vel, (len(runtime_joint_names),)),
                ("root_pos_w", file_root_pos, (3,)),
                ("root_quat_w", file_root_quat, (4,)),
            ):
                if value.dtype != np.dtype(np.float64) or value.shape != shape:
                    raise NeutralReadyError(
                        f"donor ready {key} must be exact float64 {shape}, "
                        f"got {value.dtype}/{value.shape}"
                    )
                if not np.isfinite(value).all():
                    raise NeutralReadyError(
                        f"donor ready {key} contains NaN/Inf"
                    )
            if np.count_nonzero(file_joint_vel) != 0:
                raise NeutralReadyError(
                    "donor ready joint_vel must be exact zero"
                )
            for key in ("source_segment", "source_npz", "note"):
                value = np.asarray(payload[key])
                if (
                    value.shape != ()
                    or value.dtype.kind not in ("U", "S")
                    or not str(value.item()).strip()
                ):
                    raise NeutralReadyError(
                        f"donor ready {key} must be one non-empty text scalar"
                    )
            source_frame = np.asarray(payload["source_frame"])
            if (
                source_frame.shape != ()
                or not np.issubdtype(source_frame.dtype, np.integer)
                or int(source_frame) < 0
            ):
                raise NeutralReadyError(
                    "donor ready source_frame must be a non-negative integer"
                )
            striking_ids = np.asarray(payload["striking_joint_ids"])
            expected_striking_ids = np.asarray(
                [
                    runtime_joint_names.index(name)
                    for name in face.RIGHT_STRIKE_CHAIN
                ],
                dtype=striking_ids.dtype,
            )
            if (
                striking_ids.shape != (len(face.RIGHT_STRIKE_CHAIN),)
                or not np.issubdtype(striking_ids.dtype, np.integer)
                or not np.array_equal(striking_ids, expected_striking_ids)
            ):
                raise NeutralReadyError(
                    "donor ready striking_joint_ids do not exactly identify "
                    "the canonical seven-joint right striking chain"
                )
    except (OSError, ValueError) as exc:
        raise NeutralReadyError(f"cannot load donor ready NPZ: {exc}") from exc
    if (
        not np.array_equal(file_joint, ready)
        or not np.array_equal(file_root_pos, root_pos)
        or not np.array_equal(file_root_quat, root_quat)
    ):
        raise NeutralReadyError(
            "donor ready arrays do not match the content-pinned NPZ"
        )
    return path, actual_sha


def mjcf_include_closure_digest(
    mjcf_path: str | Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Hash the real-file XML ``<include>`` closure rooted at one MJCF.

    The compiled-model contract binds this digest in addition to the top-level
    XML hash.  Mesh/hfield/contact semantics are separately bound from compiled
    model arrays; this closure closes the otherwise unbound external XML byte
    graph.
    """

    root = Path(mjcf_path).expanduser().absolute()
    visiting: set[Path] = set()
    visited: dict[Path, dict[str, Any]] = {}

    def require_real_file(path: Path, label: str) -> Path:
        absolute = path.expanduser().absolute()
        try:
            resolved = absolute.resolve(strict=True)
        except OSError as exc:
            raise NeutralReadyError(f"cannot resolve {label}: {exc}") from exc
        if (
            absolute != resolved
            or absolute.is_symlink()
            or not absolute.is_file()
        ):
            raise NeutralReadyError(
                f"{label} must be a real regular file with no symlink "
                f"component: {absolute}"
            )
        return absolute

    root = require_real_file(root, "MJCF root")
    root_parent = root.parent

    def label_for(path: Path) -> str:
        try:
            return path.relative_to(root_parent).as_posix()
        except ValueError:
            return str(path)

    def walk(path: Path) -> None:
        checked = require_real_file(path, "MJCF include")
        if checked in visiting:
            raise NeutralReadyError(
                f"MJCF include cycle detected at {checked}"
            )
        if checked in visited:
            return
        visiting.add(checked)
        try:
            payload = checked.read_bytes()
            xml_root = ET.fromstring(payload)
        except (OSError, ET.ParseError) as exc:
            raise NeutralReadyError(
                f"cannot parse MJCF include closure member {checked}: {exc}"
            ) from exc
        children: list[Path] = []
        for element in xml_root.iter():
            if str(element.tag).rsplit("}", 1)[-1] != "include":
                continue
            raw = element.attrib.get("file")
            if raw is None or not str(raw).strip():
                raise NeutralReadyError(
                    f"MJCF include lacks a non-empty file attribute: {checked}"
                )
            child = Path(
                os.path.normpath(str(checked.parent / str(raw)))
            ).absolute()
            children.append(require_real_file(child, "MJCF include"))
        row = {
            "path": label_for(checked),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "includes": sorted(label_for(child) for child in children),
        }
        visited[checked] = row
        for child in sorted(children, key=lambda value: str(value)):
            walk(child)
        visiting.remove(checked)

    walk(root)
    rows = sorted(visited.values(), key=lambda row: str(row["path"]))
    closure = {
        "schema_version": 1,
        "root": label_for(root),
        "members": rows,
    }
    return hashlib.sha256(_canonical_json_bytes(closure)).hexdigest(), rows


def _verify_exact_model(
    backend: face.RightRacketBackend,
    binding: ExactModelBinding | None,
) -> dict[str, Any]:
    if binding is None:
        return {
            "status": "MISSING_FAIL_CLOSED",
            "mjcf_path": None,
            "mjcf_sha256": None,
            "mjcf_dependency_closure_sha256": None,
            "mjcf_dependency_closure": None,
            "compiled_model_sha256": None,
            "compiled_mjb_sha256": None,
            "urdf_path": None,
            "urdf_sha256": None,
            "backend_limits_sha256": None,
            "backend_limit_array_sha256": None,
            "backend_model_contract_sha256": None,
            "backend_model_contract": None,
            "toolchain": None,
        }
    path = Path(binding.mjcf_path).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise NeutralReadyError(f"exact MJCF must be a regular file: {path}")
    source_sha = _sha256_file(path)
    if source_sha != binding.expected_mjcf_sha256:
        raise NeutralReadyError(
            f"MJCF SHA-256 mismatch: {source_sha} != "
            f"{binding.expected_mjcf_sha256}"
        )
    dependency_closure_sha, dependency_closure_rows = (
        mjcf_include_closure_digest(path)
    )
    urdf_path = Path(binding.urdf_path).expanduser().absolute()
    if not urdf_path.is_file() or urdf_path.is_symlink():
        raise NeutralReadyError(
            f"exact URDF must be a regular file: {urdf_path}"
        )
    urdf_sha = _sha256_file(urdf_path)
    if urdf_sha != binding.expected_urdf_sha256:
        raise NeutralReadyError(
            f"URDF SHA-256 mismatch: {urdf_sha} != "
            f"{binding.expected_urdf_sha256}"
        )
    if type(backend) is not face.MujocoRightRacketBackend:
        raise NeutralReadyError(
            "exact publication requires the canonical "
            "MujocoRightRacketBackend implementation, not a Protocol shim "
            "or override"
        )
    try:
        urdf_velocity, urdf_effort = face._read_urdf_motion_limits(
            urdf_path, backend.joint_names
        )
    except Exception as exc:
        raise NeutralReadyError(
            f"cannot parse exact URDF motion limits: {exc}"
        ) from exc
    backend_velocity = np.asarray(backend.velocity_limit, dtype=np.float64)
    backend_effort = np.asarray(backend.effort_limit, dtype=np.float64)
    if (
        backend_velocity.shape != urdf_velocity.shape
        or backend_effort.shape != urdf_effort.shape
        or not np.array_equal(backend_velocity, urdf_velocity)
        or not np.array_equal(backend_effort, urdf_effort)
    ):
        raise NeutralReadyError(
            "backend velocity/effort limits do not exactly equal the pinned URDF"
        )
    limits_sha, limit_array_sha = backend_limits_digest(backend)
    if limits_sha != binding.expected_backend_limits_sha256:
        raise NeutralReadyError(
            "backend position/velocity/effort limit digest mismatch: "
            f"{limits_sha} != {binding.expected_backend_limits_sha256}"
        )
    model = getattr(backend, "model", None)
    if model is None:
        raise NeutralReadyError(
            "an exact model binding requires a MuJoCo backend with compiled model"
        )
    try:
        import canonical_mujoco_dynamics_gate as dynamics_gate

        _require_exact_runtime_joint_order(
            backend.joint_names,
            expected=dynamics_gate.RUNTIME_JOINT_NAMES,
        )
        compiled_sha = dynamics_gate.compiled_model_signature(model)
        mujoco_module = getattr(backend, "_mujoco")
        fresh_model = mujoco_module.MjModel.from_xml_path(str(path))
        backend_mjb_sha = _compiled_mjb_sha256(model, mujoco_module)
        fresh_mjb_sha = _compiled_mjb_sha256(fresh_model, mujoco_module)
        if backend_mjb_sha != fresh_mjb_sha:
            raise NeutralReadyError(
                "backend compiled model is not the model freshly compiled "
                "from the pinned MJCF dependency graph"
            )
        closure_sha_after, closure_rows_after = (
            mjcf_include_closure_digest(path)
        )
        if (
            closure_sha_after != dependency_closure_sha
            or closure_rows_after != dependency_closure_rows
        ):
            raise NeutralReadyError(
                "MJCF include closure changed while compiling the exact model"
            )
    except NeutralReadyError:
        raise
    except Exception as exc:
        raise NeutralReadyError(
            f"cannot compute exact compiled-model signature: {exc}"
        ) from exc
    if compiled_sha != binding.expected_compiled_model_sha256:
        raise NeutralReadyError(
            "compiled MuJoCo model SHA-256 mismatch: "
            f"{compiled_sha} != {binding.expected_compiled_model_sha256}"
        )
    backend_model_sha, backend_model_contract = (
        exact_backend_model_contract_digest(
            backend,
            compiled_model_sha256=compiled_sha,
            compiled_mjb_sha256=backend_mjb_sha,
            backend_limits_sha256=limits_sha,
            mjcf_dependency_closure_sha256=dependency_closure_sha,
        )
    )
    toolchain = {
        "canonical_face_manifold_path": str(
            Path(face.__file__).resolve()
        ),
        "canonical_face_manifold_sha256": _sha256_file(
            Path(face.__file__).resolve()
        ),
        "compiled_signature_helper_path": str(
            Path(dynamics_gate.__file__).resolve()
        ),
        "compiled_signature_helper_sha256": _sha256_file(
            Path(dynamics_gate.__file__).resolve()
        ),
        "mujoco_version": str(
            getattr(mujoco_module, "__version__", "unknown")
        ),
        "numpy_version": str(np.__version__),
    }
    if (
        backend_model_sha
        != binding.expected_backend_model_contract_sha256
    ):
        raise NeutralReadyError(
            "backend exact model/site/address contract digest mismatch: "
            f"{backend_model_sha} != "
            f"{binding.expected_backend_model_contract_sha256}"
        )
    return {
        "status": "PASS",
        "mjcf_path": str(path),
        "mjcf_sha256": source_sha,
        "mjcf_dependency_closure_sha256": dependency_closure_sha,
        "mjcf_dependency_closure": dependency_closure_rows,
        "compiled_model_sha256": compiled_sha,
        "compiled_mjb_sha256": backend_mjb_sha,
        "urdf_path": str(urdf_path),
        "urdf_sha256": urdf_sha,
        "backend_limits_sha256": limits_sha,
        "backend_limit_array_sha256": limit_array_sha,
        "backend_model_contract_sha256": backend_model_sha,
        "backend_model_contract": backend_model_contract,
        "toolchain": toolchain,
    }


def _evaluate_safety(
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    *,
    backend: face.RightRacketBackend,
    exact_model: Mapping[str, Any],
    config: NeutralReadyConfig,
) -> SafetyCheck:
    if exact_model["status"] != "PASS":
        return SafetyCheck(
            status="NOT_EVALUATED_FAIL_CLOSED",
            exact_model=False,
            passed=False,
            collision_passed=False,
            double_foot_ground_passed=False,
            support_hull_passed=False,
            maximum_allowed_foot_penetration_m=None,
            unsupported_contact_count=None,
            active_contact_count=None,
            support_margin_m=None,
            details={"reason": "exact source and compiled model are not bound"},
        )
    model = getattr(backend, "model", None)
    data = getattr(backend, "data", None)
    mujoco = getattr(backend, "_mujoco", None)
    if model is None or data is None or mujoco is None:
        raise NeutralReadyError(
            "exact model passed but backend lacks MuJoCo model/data/module"
        )
    # The canonical backend contract says site_pose installs the exact state
    # and calls mj_forward.  Verify that side effect independently before
    # trusting data.contact/subtree_com.
    returned_site_pos, returned_site_rotation = backend.site_pose(
        joint_pos, root_pos, root_quat
    )
    qpos_addresses = np.asarray(
        getattr(backend, "_qpos_addresses"), dtype=np.int64
    )
    root_qpos_address = int(getattr(backend, "_root_qpos_address"))
    installed_joint = np.asarray(data.qpos, dtype=np.float64)[qpos_addresses]
    installed_root_pos = np.asarray(data.qpos, dtype=np.float64)[
        root_qpos_address : root_qpos_address + 3
    ]
    installed_root_quat = np.asarray(data.qpos, dtype=np.float64)[
        root_qpos_address + 3 : root_qpos_address + 7
    ]
    site_id = int(getattr(backend, "_site_id"))
    data_site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    data_site_rotation = np.asarray(
        data.site_xmat[site_id], dtype=np.float64
    ).reshape(3, 3)
    if (
        not np.array_equal(installed_joint, np.asarray(joint_pos))
        or not np.array_equal(installed_root_pos, np.asarray(root_pos))
        or not np.allclose(
            installed_root_quat,
            np.asarray(root_quat)
            / float(np.linalg.norm(np.asarray(root_quat))),
            rtol=0.0,
            atol=1.0e-14,
        )
        or not np.array_equal(
            np.asarray(returned_site_pos), data_site_pos
        )
        or not np.array_equal(
            np.asarray(returned_site_rotation), data_site_rotation
        )
        or np.any(np.asarray(data.qvel) != 0.0)
    ):
        raise NeutralReadyError(
            "canonical MuJoCo backend did not install/recompute the exact "
            "candidate state before the safety screen"
        )
    floor_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, config.floor_geom_name
        )
    )
    if floor_id < 0:
        raise NeutralReadyError(
            f"exact model lacks floor geom {config.floor_geom_name!r}"
        )
    foot_ids: list[int] = []
    for name in config.foot_body_names:
        body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        )
        if body_id < 0:
            raise NeutralReadyError(f"exact model lacks foot body {name!r}")
        foot_ids.append(body_id)
    root_body_name = str(backend.root_body_name)
    root_body_id = int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
        )
    )
    if root_body_id < 0:
        raise NeutralReadyError(
            f"exact model lacks root body {root_body_name!r}"
        )

    contact_points: list[np.ndarray] = []
    contacted_feet: set[int] = set()
    unsupported: list[dict[str, Any]] = []
    maximum_penetration = 0.0
    active_count = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        distance = float(contact.dist)
        if distance > config.contact_gap_tolerance_m:
            continue
        active_count += 1
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        pair = (geom1, geom2)
        floor_pair = floor_id in pair
        robot_body = body2 if geom1 == floor_id else body1
        foot_index = (
            foot_ids.index(robot_body) if robot_body in foot_ids else None
        )
        if floor_pair and foot_index is not None:
            contacted_feet.add(foot_index)
            maximum_penetration = max(maximum_penetration, max(0.0, -distance))
            contact_points.append(
                np.asarray(contact.pos, dtype=np.float64).copy()
            )
            continue
        if distance <= 0.0:
            unsupported.append(
                {
                    "contact_index": index,
                    "distance_m": distance,
                    "geom_ids": [geom1, geom2],
                    "body_ids": [body1, body2],
                }
            )
    collision_pass = bool(
        not unsupported
        and maximum_penetration
        <= config.collision_penetration_tolerance_m
    )
    double_foot_pass = contacted_feet == {0, 1}
    support_margin: float | None = None
    support_pass = False
    support_details: dict[str, Any] = {}
    if double_foot_pass and len(contact_points) >= 3:
        floor_rotation = np.asarray(
            data.geom_xmat[floor_id], dtype=np.float64
        ).reshape(3, 3)
        floor_origin = np.asarray(
            data.geom_xpos[floor_id], dtype=np.float64
        )
        basis_x = floor_rotation[:, 0]
        basis_y = floor_rotation[:, 1]
        projected = np.asarray(
            [
                [
                    float((point - floor_origin) @ basis_x),
                    float((point - floor_origin) @ basis_y),
                ]
                for point in contact_points
            ],
            dtype=np.float64,
        )
        hull = _convex_hull(projected)
        com = np.asarray(
            data.subtree_com[root_body_id], dtype=np.float64
        )
        com_2d = np.asarray(
            [
                float((com - floor_origin) @ basis_x),
                float((com - floor_origin) @ basis_y),
            ]
        )
        if len(hull) >= 3:
            support_margin = _convex_polygon_margin(hull, com_2d)
            support_pass = bool(
                support_margin >= -config.support_hull_tolerance_m
            )
        support_details = {
            "contact_hull_xy_m": hull.tolist(),
            "com_projection_xy_m": com_2d.tolist(),
        }
    passed = collision_pass and double_foot_pass and support_pass
    return SafetyCheck(
        status=(
            "PASS_EXACT_MUJOCO_STATIC_READY"
            if passed
            else "FAIL_EXACT_MUJOCO_STATIC_READY"
        ),
        exact_model=True,
        passed=passed,
        collision_passed=collision_pass,
        double_foot_ground_passed=double_foot_pass,
        support_hull_passed=support_pass,
        maximum_allowed_foot_penetration_m=maximum_penetration,
        unsupported_contact_count=len(unsupported),
        active_contact_count=active_count,
        support_margin_m=support_margin,
        details={
            "contacted_foot_indices": sorted(contacted_feet),
            "unsupported_contacts": unsupported,
            **support_details,
        },
    )


def _candidate_diagnostics_receipt(
    diagnostics: face.FrameDiagnostics,
) -> dict[str, Any]:
    """Relabel the upstream diagonal envelope without inheriting its claim."""

    row = asdict(diagnostics)
    raw = row.pop("ready_connector_time_lower_bound_s")
    row["donor_ready_to_candidate_diagonal_time_proxy_s"] = raw
    row["diagonal_time_interpretation"] = (
        "tie_break_proxy_not_a_certified_minimum_time_lower_bound"
    )
    return row


def _build_receipt(
    *,
    source_path: Path,
    source_sha: str,
    model_receipt: Mapping[str, Any],
    ready: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    ordered_contacts: Sequence[ContactPose],
    joint_names: Sequence[str],
    active: np.ndarray,
    candidate: face.SiteNormalCandidate,
    target: MidpointTarget,
    metrics: CandidateMetrics,
    safety: SafetyCheck,
    connector_rows: Sequence[Mapping[str, Any]],
    candidate_pool: Sequence[NeutralReadyPoolEntry],
    pair_receipts: Sequence[Mapping[str, Any]],
    pair_locus_seed_count: int,
    global_angular: GlobalAngularSolution,
    solved_count: int,
    rejected_safety_count: int,
    solved_target_count: int,
    all_global_targets_exact_ik: bool,
    global_bound_gap_pass: bool,
    neutrality_pass: bool,
    upstream_source_pose_reconstruction_verified: bool,
    source_proof_receipt: Mapping[str, Any] | None,
    config: NeutralReadyConfig,
    publication_allowed: bool,
) -> dict[str, Any]:
    fixed = np.setdiff1d(np.arange(len(ready)), active)
    candidate_joint_names = tuple(str(name) for name in joint_names)
    if len(candidate_joint_names) != len(ready):
        raise NeutralReadyError("receipt joint order length changed")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_class": "diagnostic_face_neutral_ready_candidate",
        "candidate_scope": (
            "right_arm_posture_challenger_only_not_shared_or_full_ready"
        ),
        "verdict": (
            "PASS_DIAGNOSTIC_PUBLICATION_GATE"
            if publication_allowed
            else "INCOMPLETE_FAIL_CLOSED"
        ),
        "source_ready": {
            "path": str(source_path),
            "sha256": source_sha,
            "joint_pos_sha256": _array_sha256(ready),
            "root_pos_w_sha256": _array_sha256(root_pos),
            "root_quat_w_sha256": _array_sha256(root_quat),
            "joint_velocity_zero_verified_from_source": True,
            "root_velocity_fields_present_in_source": False,
            "complete_rest_state_proven": False,
            "immutable_donor_baseline": True,
            "overwritten": False,
        },
        "model": dict(model_receipt),
        "normal_contract": {
            "convention": NORMAL_CONVENTION,
            "signed": True,
            "automatic_n_minus_n_flip": False,
            "site_axis": "local_plus_y",
        },
        "joint_contract": {
            "all_joint_names": list(candidate_joint_names),
            "active_joint_names": list(face.RIGHT_STRIKE_CHAIN),
            "active_joint_indices": active.tolist(),
            "fixed_joint_names": [
                candidate_joint_names[int(index)] for index in fixed
            ],
            "fixed_joint_indices": fixed.tolist(),
            "fixed_joint_values_bitwise_equal": bool(
                np.array_equal(candidate.joint_pos[fixed], ready[fixed])
            ),
            "single_joint_face_assumption": False,
        },
        "contact_matrix": {
            "required_shape": (
                "upper_full_x_opportunity_start_construction_donor_"
                "preferred_nominal_event_opportunity_end_x_bh_fh"
            ),
            "row_count": len(ordered_contacts),
            "input_sha256": _contact_digest(ordered_contacts),
            "rows": [
                {
                    "label": row.label,
                    "joint_pos_sha256": _array_sha256(row.joint_pos),
                    "root_pos_w_sha256": _array_sha256(row.root_pos_w),
                    "root_quat_w_sha256": _array_sha256(row.root_quat_w),
                    "source_motion_path": str(row.source_motion_path),
                    "source_motion_sha256": row.source_motion_sha256,
                    "source_frame_index": row.source_frame_index,
                    "pose_content_sha256": (
                        row.pose_content_sha256
                    ),
                    "pair_contract_sha256": row.pair_contract_sha256,
                    "site_pos_w": row.site_pos_w.tolist(),
                    "signed_face_normal_w": (
                        row.signed_face_normal_w.tolist()
                    ),
                }
                for row in ordered_contacts
            ],
            "caller_arrays_are_content_sealed": True,
            "source_frame_arrays_independently_reconstructed": bool(
                upstream_source_pose_reconstruction_verified
            ),
            "source_reconstruction_proof": (
                None
                if source_proof_receipt is None
                else dict(source_proof_receipt)
            ),
        },
        "neutral_ready_search": {
            "method": (
                "content_bound_antipodal_pairs_plus_global_axis_"
                "angular_minimax_then_right_arm_ik_v1"
            ),
            "antipodal_rule": (
                "never_normalize_n_bh_plus_n_fh_when_dot_near_minus_one"
            ),
            "pairs": list(pair_receipts),
            "pair_locus_samples_are_seeds_not_exhaustive": True,
            "pair_locus_seed_count": int(pair_locus_seed_count),
            "global_angular_solution": {
                "axis_rank": global_angular.axis_rank,
                "method": global_angular.method,
                "maximum_absolute_axis_projection": (
                    global_angular.maximum_absolute_axis_projection
                ),
                "objective_lower_bound_rad": (
                    global_angular.angular_objective_lower_bound_rad
                ),
                "objective_upper_bound_rad": (
                    global_angular.angular_objective_upper_bound_rad
                ),
                "certified_bound_gap_rad": (
                    global_angular.angular_objective_upper_bound_rad
                    - global_angular.angular_objective_lower_bound_rad
                ),
                "angular_minimax_certified": (
                    global_angular.angular_minimax_certified
                ),
                "finite_optimizer_locus": (
                    global_angular.finite_optimizer_locus
                ),
                "sampled_continuous_locus": (
                    global_angular.sampled_continuous_locus
                ),
                "optimizer_target_count": len(global_angular.targets),
                "report": dict(global_angular.report),
            },
            "all_global_targets_received_exact_ik": bool(
                all_global_targets_exact_ik
            ),
            "solved_global_target_count": int(solved_target_count),
            "solved_ik_branch_count": int(solved_count),
            "static_safety_rejected_branch_count": int(
                rejected_safety_count
            ),
            "selected": {
                "normal_w": target.normal_w.tolist(),
                "source_pair": target.source_pair,
                "antipodal": target.antipodal,
                "theta_rad": target.theta_rad,
                "basis_u_w": (
                    None
                    if target.basis_u_w is None
                    else target.basis_u_w.tolist()
                ),
                "basis_v_w": (
                    None
                    if target.basis_v_w is None
                    else target.basis_v_w.tolist()
                ),
                "pair_dot": target.pair_dot,
            },
            "selection_order": [
                "exact_static_safety_pass",
                "bucketed_worst_bh_fh_angular_asymmetry",
                "bucketed_worst_right_arm_diagonal_time_proxy",
                "bucketed_maximum_contact_angle",
                "bucketed_exact_IK_residual",
                "maximize_support_hull_margin",
                "stable_candidate_sha256",
            ],
        },
        "candidate": {
            "joint_pos_sha256": _array_sha256(candidate.joint_pos),
            "joint_pos": candidate.joint_pos.tolist(),
            "joint_vel": np.zeros_like(candidate.joint_pos).tolist(),
            "root_pos_w": root_pos.tolist(),
            "root_quat_w": root_quat.tolist(),
            "root_lin_vel_w": np.zeros(3, dtype=np.float64).tolist(),
            "root_ang_vel_w": np.zeros(3, dtype=np.float64).tolist(),
            "velocity_contract": (
                "candidate_emits_explicit_zero_joint_and_root_velocities;"
                "donor_source_only_proves_zero_joint_vel"
            ),
            "site_pos_w": candidate.site_pos_w.tolist(),
            "signed_face_normal_w": candidate.raw_plus_y_w.tolist(),
            "diagnostics": _candidate_diagnostics_receipt(
                candidate.diagnostics
            ),
            "metrics": asdict(metrics),
            "connectors": list(connector_rows),
        },
        "candidate_pool": {
            "retained_for_outer_5x2_compiler_search": True,
            "eligibility_contract": (
                "every_serialized_row_passes_exact_mujoco_static_safety"
            ),
            "diagnostic_rejected_rows_are_not_serialized": True,
            "count": len(candidate_pool),
            "rows": [
                {
                    "joint_pos_sha256": _array_sha256(
                        row.candidate.joint_pos
                    ),
                    "joint_pos": row.candidate.joint_pos.tolist(),
                    "target_normal_w": row.target.normal_w.tolist(),
                    "target_source": row.target.source_pair,
                    "metrics": asdict(row.metrics),
                    "safety": asdict(row.safety),
                    "outer_compiler_eligible": True,
                    "connectors": list(row.connectors),
                }
                for row in candidate_pool
            ],
        },
        "gates": {
            "source_ready_hash": "PASS",
            "input_contact_exact_fk": "PASS",
            "exact_site_normal_ik": "PASS",
            "joint_limits": "PASS",
            "fixed_joints": "PASS",
            "paired_face_and_site_content_contract": "PASS",
            "upstream_source_pose_reconstruction": (
                "PASS"
                if upstream_source_pose_reconstruction_verified
                else "INCOMPLETE_FAIL_CLOSED"
            ),
            "exact_model": model_receipt["status"],
            "global_angular_minimax_bound": (
                "PASS"
                if global_bound_gap_pass
                else "INCOMPLETE_FAIL_CLOSED"
            ),
            "finite_global_optimizer_locus": (
                "PASS"
                if global_angular.finite_optimizer_locus
                else "SAMPLED_CONTINUOUS_LOCUS_FAIL_CLOSED"
            ),
            "all_global_targets_exact_ik": (
                "PASS"
                if all_global_targets_exact_ik
                else "INCOMPLETE_FAIL_CLOSED"
            ),
            "neutrality_threshold": (
                "PASS" if neutrality_pass else "FAIL_CLOSED"
            ),
            "static_safety": asdict(safety),
            "grounded_ready_required_and_unresolved": bool(
                not safety.double_foot_ground_passed
            ),
            "publication_allowed": bool(publication_allowed),
        },
        "authorization": {
            "training_authorized": False,
            "deploy_authorized": False,
            "hardware_authorized": False,
        },
        "config": asdict(config),
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "non_claims": [
            (
                "connector times are seven-right-arm diagonal dynamics proxies "
                "only, not certified minimum-time lower bounds; coupled "
                "dynamics, root, waist, legs, and contact-window velocity are "
                "not timed"
            ),
            "no trajectory retiming or contact-window behavior certificate",
            "no grounded torque or closed-loop balance certificate",
            (
                "selected ready is a challenger only; final selection requires "
                "compiling all 5x2 paths against donor ready and comparing "
                "worst opportunity-start, construction-donor-preferred, "
                "nominal-event, opportunity-end, and cycle ticks plus "
                "grounded gates"
            ),
            (
                "search fixes the donor racket-site position and solves angular "
                "neutrality before timing; it omits nearby site positions and "
                "angular epsilon-sublevel postures that may yield better paths"
            ),
            (
                "contact source reconstruction remains absent"
                if not upstream_source_pose_reconstruction_verified
                else "source reconstruction proves input lineage only, not "
                "the final 5x2 path or behavior"
            ),
            (
                "donor ready asset has no root linear/angular velocity fields; "
                "the candidate emits zeros, but complete rest-state acceptance "
                "remains a compiler/consumer gate"
            ),
            "no training, deployment, or hardware authorization",
            (
                "a right-arm-only challenger cannot repair donor root/leg "
                "grounding; current donor grounding must be closed by the "
                "outer full-ready compiler"
            ),
        ],
    }
    return receipt


def _write_candidate_npz(path: Path, result: NeutralReadyResult) -> None:
    try:
        with path.open("xb") as stream:
            np.savez(
                stream,
                joint_pos=np.asarray(
                    result.candidate.joint_pos, dtype=np.float64
                ),
                joint_vel=np.zeros_like(
                    result.candidate.joint_pos, dtype=np.float64
                ),
                root_pos_w=np.asarray(result.root_pos_w, dtype=np.float64),
                root_quat_w=np.asarray(result.root_quat_w, dtype=np.float64),
                root_lin_vel_w=np.zeros(3, dtype=np.float64),
                root_ang_vel_w=np.zeros(3, dtype=np.float64),
                active_joint_names=np.asarray(
                    face.RIGHT_STRIKE_CHAIN, dtype="<U32"
                ),
                source_ready_sha256=np.asarray(
                    result.receipt["source_ready"]["sha256"]
                ),
                mjcf_sha256=np.asarray(
                    result.receipt["model"]["mjcf_sha256"] or ""
                ),
                compiled_model_sha256=np.asarray(
                    result.receipt["model"]["compiled_model_sha256"] or ""
                ),
                compiled_mjb_sha256=np.asarray(
                    result.receipt["model"]["compiled_mjb_sha256"] or ""
                ),
                urdf_sha256=np.asarray(
                    result.receipt["model"]["urdf_sha256"] or ""
                ),
                backend_limits_sha256=np.asarray(
                    result.receipt["model"]["backend_limits_sha256"] or ""
                ),
                backend_model_contract_sha256=np.asarray(
                    result.receipt["model"][
                        "backend_model_contract_sha256"
                    ]
                    or ""
                ),
                candidate_pool_joint_pos=np.stack(
                    [
                        np.asarray(
                            row.candidate.joint_pos, dtype=np.float64
                        )
                        for row in result.candidate_pool
                    ]
                ),
                candidate_pool_target_normal_w=np.stack(
                    [
                        np.asarray(row.target.normal_w, dtype=np.float64)
                        for row in result.candidate_pool
                    ]
                ),
                candidate_pool_joint_pos_sha256=np.asarray(
                    [
                        _array_sha256(row.candidate.joint_pos)
                        for row in result.candidate_pool
                    ],
                    dtype="<U64",
                ),
                candidate_pool_static_safety_passed=np.asarray(
                    [row.safety.passed for row in result.candidate_pool],
                    dtype=np.bool_,
                ),
                candidate_pool_support_margin_m=np.asarray(
                    [
                        row.safety.support_margin_m
                        for row in result.candidate_pool
                    ],
                    dtype=np.float64,
                ),
                candidate_pool_safety_sha256=np.asarray(
                    [
                        hashlib.sha256(
                            _canonical_json_bytes(asdict(row.safety))
                        ).hexdigest()
                        for row in result.candidate_pool
                    ],
                    dtype="<U64",
                ),
                normal_convention=np.asarray(NORMAL_CONVENTION),
                training_authorized=np.asarray(False),
                deploy_authorized=np.asarray(False),
                hardware_authorized=np.asarray(False),
            )
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as exc:
        raise NeutralReadyError(f"cannot write candidate NPZ: {exc}") from exc


def _validate_staged_candidate_npz(
    path: Path,
    *,
    result: NeutralReadyResult,
    expected_sha256: str,
) -> None:
    """Reopen staged bytes and compare every publication-critical field."""

    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != expected_sha256
    ):
        raise NeutralReadyError("staged candidate NPZ file/hash changed")
    expected_keys = {
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "active_joint_names",
        "source_ready_sha256",
        "mjcf_sha256",
        "compiled_model_sha256",
        "compiled_mjb_sha256",
        "urdf_sha256",
        "backend_limits_sha256",
        "backend_model_contract_sha256",
        "candidate_pool_joint_pos",
        "candidate_pool_target_normal_w",
        "candidate_pool_joint_pos_sha256",
        "candidate_pool_static_safety_passed",
        "candidate_pool_support_margin_m",
        "candidate_pool_safety_sha256",
        "normal_convention",
        "training_authorized",
        "deploy_authorized",
        "hardware_authorized",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise NeutralReadyError(
                    "staged candidate NPZ exact field set changed"
                )
            for key, expected in (
                ("joint_pos", result.candidate.joint_pos),
                ("joint_vel", np.zeros_like(result.candidate.joint_pos)),
                ("root_pos_w", result.root_pos_w),
                ("root_quat_w", result.root_quat_w),
                ("root_lin_vel_w", np.zeros(3, dtype=np.float64)),
                ("root_ang_vel_w", np.zeros(3, dtype=np.float64)),
            ):
                actual = np.asarray(payload[key])
                if (
                    actual.dtype != np.dtype(np.float64)
                    or not np.array_equal(actual, np.asarray(expected))
                ):
                    raise NeutralReadyError(
                        f"staged candidate {key} changed"
                    )
            for key in (
                "training_authorized",
                "deploy_authorized",
                "hardware_authorized",
            ):
                value = np.asarray(payload[key])
                if value.shape != () or bool(value.item()) is not False:
                    raise NeutralReadyError(
                        f"staged candidate {key} must be scalar false"
                    )
            if str(np.asarray(payload["normal_convention"]).item()) != (
                NORMAL_CONVENTION
            ):
                raise NeutralReadyError(
                    "staged candidate normal convention changed"
                )
            active_names = tuple(
                str(value)
                for value in np.asarray(payload["active_joint_names"]).tolist()
            )
            if active_names != tuple(face.RIGHT_STRIKE_CHAIN):
                raise NeutralReadyError(
                    "staged candidate active joint order changed"
                )
            scalar_bindings = {
                "source_ready_sha256": result.receipt["source_ready"]["sha256"],
                "mjcf_sha256": result.receipt["model"]["mjcf_sha256"],
                "compiled_model_sha256": result.receipt["model"][
                    "compiled_model_sha256"
                ],
                "compiled_mjb_sha256": result.receipt["model"][
                    "compiled_mjb_sha256"
                ],
                "urdf_sha256": result.receipt["model"]["urdf_sha256"],
                "backend_limits_sha256": result.receipt["model"][
                    "backend_limits_sha256"
                ],
                "backend_model_contract_sha256": result.receipt["model"][
                    "backend_model_contract_sha256"
                ],
            }
            for key, expected in scalar_bindings.items():
                actual = np.asarray(payload[key])
                if (
                    actual.shape != ()
                    or str(actual.item()) != str(expected or "")
                ):
                    raise NeutralReadyError(
                        f"staged candidate {key} changed"
                    )
            pool = np.stack(
                [
                    np.asarray(row.candidate.joint_pos, dtype=np.float64)
                    for row in result.candidate_pool
                ]
            )
            if not np.array_equal(
                np.asarray(payload["candidate_pool_joint_pos"]), pool
            ):
                raise NeutralReadyError(
                    "staged candidate pool joint arrays changed"
                )
            target_pool = np.stack(
                [
                    np.asarray(row.target.normal_w, dtype=np.float64)
                    for row in result.candidate_pool
                ]
            )
            if not np.array_equal(
                np.asarray(payload["candidate_pool_target_normal_w"]),
                target_pool,
            ):
                raise NeutralReadyError(
                    "staged candidate pool target normals changed"
                )
            pool_hashes = np.asarray(
                [
                    _array_sha256(row.candidate.joint_pos)
                    for row in result.candidate_pool
                ],
                dtype="<U64",
            )
            if not np.array_equal(
                np.asarray(payload["candidate_pool_joint_pos_sha256"]),
                pool_hashes,
            ):
                raise NeutralReadyError(
                    "staged candidate pool hashes changed"
                )
            safety_passed = np.asarray(
                [row.safety.passed for row in result.candidate_pool],
                dtype=np.bool_,
            )
            if (
                not bool(np.all(safety_passed))
                or not np.array_equal(
                    np.asarray(
                        payload["candidate_pool_static_safety_passed"]
                    ),
                    safety_passed,
                )
            ):
                raise NeutralReadyError(
                    "staged candidate pool contains an unsafe row"
                )
            support_margins = np.asarray(
                [
                    row.safety.support_margin_m
                    for row in result.candidate_pool
                ],
                dtype=np.float64,
            )
            if not np.array_equal(
                np.asarray(payload["candidate_pool_support_margin_m"]),
                support_margins,
            ):
                raise NeutralReadyError(
                    "staged candidate pool support margins changed"
                )
            safety_hashes = np.asarray(
                [
                    hashlib.sha256(
                        _canonical_json_bytes(asdict(row.safety))
                    ).hexdigest()
                    for row in result.candidate_pool
                ],
                dtype="<U64",
            )
            if not np.array_equal(
                np.asarray(payload["candidate_pool_safety_sha256"]),
                safety_hashes,
            ):
                raise NeutralReadyError(
                    "staged candidate pool safety receipts changed"
                )
    except (OSError, ValueError) as exc:
        if isinstance(exc, NeutralReadyError):
            raise
        raise NeutralReadyError(
            f"cannot validate staged candidate NPZ: {exc}"
        ) from exc


def _cleanup_owned_stage(
    stage: Path,
    identity: os.stat_result,
    directory_fd: int,
) -> None:
    """Unlink known files through the exact directory descriptor we created."""

    if directory_fd < 0:
        return
    try:
        if not os.path.samestat(os.fstat(directory_fd), identity):
            return
    except OSError:
        return
    for filename in (CANDIDATE_FILENAME, RECEIPT_FILENAME):
        try:
            row = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError:
            return
        if not stat.S_ISREG(row.st_mode):
            return
        try:
            os.unlink(filename, dir_fd=directory_fd)
        except OSError:
            return
    try:
        current = os.stat(stage, follow_symlinks=False)
        if not os.path.samestat(current, identity):
            return
        os.rmdir(stage)
    except OSError:
        pass


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a complete directory without replacing any target."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        primitive = getattr(libc, "renameat2", None)
        if primitive is None:
            raise OSError(
                errno.ENOSYS,
                "atomic renameat2(RENAME_NOREPLACE) is unavailable",
                str(destination),
            )
        primitive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        primitive = getattr(libc, "renamex_np", None)
        if primitive is None:
            raise OSError(
                errno.ENOSYS,
                "atomic renamex_np(RENAME_EXCL) is unavailable",
                str(destination),
            )
        primitive.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            source_bytes,
            destination_bytes,
            _RENAME_EXCL,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            str(destination),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"refusing to overwrite concurrently created output: {destination}",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _result_payload_seal(
    *,
    candidate: face.SiteNormalCandidate,
    target: MidpointTarget,
    metrics: CandidateMetrics,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    receipt: Mapping[str, Any],
    source_ready_path: Path,
    publication_allowed: bool,
    candidate_pool: Sequence[NeutralReadyPoolEntry],
    contacts: Sequence[ContactPose],
    contact_source_proof: ContactSourceProof | None,
) -> str:
    """Seal every mutable value used by the publisher."""

    payload = {
        "schema_version": 1,
        "candidate_arrays": {
            "joint_pos": _array_sha256(candidate.joint_pos),
            "site_pos_w": _array_sha256(candidate.site_pos_w),
            "raw_plus_y_w": _array_sha256(candidate.raw_plus_y_w),
        },
        "candidate_diagnostics": asdict(candidate.diagnostics),
        "candidate_original_ready_connector": asdict(
            candidate.original_ready_connector
        ),
        "target": {
            "normal_w_sha256": _array_sha256(target.normal_w),
            "source_pair": target.source_pair,
            "antipodal": target.antipodal,
            "theta_rad": target.theta_rad,
            "basis_u_w_sha256": (
                None
                if target.basis_u_w is None
                else _array_sha256(target.basis_u_w)
            ),
            "basis_v_w_sha256": (
                None
                if target.basis_v_w is None
                else _array_sha256(target.basis_v_w)
            ),
            "pair_dot": target.pair_dot,
        },
        "metrics": asdict(metrics),
        "root_pos_w_sha256": _array_sha256(root_pos),
        "root_quat_w_sha256": _array_sha256(root_quat),
        "receipt_sha256": hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest(),
        "source_ready_path": str(Path(source_ready_path).absolute()),
        "publication_allowed": bool(publication_allowed),
        "contact_matrix_sha256": _contact_digest(contacts),
        "contact_source_proof": (
            None
            if contact_source_proof is None
            else {
                "payload_sha256": contact_source_proof.payload_sha256,
                "receipt_sha256": hashlib.sha256(
                    _canonical_json_bytes(contact_source_proof.receipt)
                ).hexdigest(),
            }
        ),
        "candidate_pool": [
            {
                "joint_pos_sha256": _array_sha256(row.candidate.joint_pos),
                "site_pos_w_sha256": _array_sha256(
                    row.candidate.site_pos_w
                ),
                "normal_sha256": _array_sha256(
                    row.candidate.raw_plus_y_w
                ),
                "target_normal_sha256": _array_sha256(
                    row.target.normal_w
                ),
                "metrics": asdict(row.metrics),
                "safety": asdict(row.safety),
                "connectors": list(row.connectors),
            }
            for row in candidate_pool
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_publishable_receipt(receipt: Mapping[str, Any]) -> None:
    """Recheck all in-memory hard gates before any output directory exists."""

    try:
        gates = receipt["gates"]
        model = receipt["model"]
        safety = gates["static_safety"]
        authorization = receipt["authorization"]
        pool = receipt["candidate_pool"]
        pool_rows = pool["rows"]
        complete = bool(
            receipt["verdict"] == "PASS_DIAGNOSTIC_PUBLICATION_GATE"
            and gates["publication_allowed"] is True
            and gates["source_ready_hash"] == "PASS"
            and gates["input_contact_exact_fk"] == "PASS"
            and gates["exact_site_normal_ik"] == "PASS"
            and gates["joint_limits"] == "PASS"
            and gates["fixed_joints"] == "PASS"
            and gates["paired_face_and_site_content_contract"] == "PASS"
            and gates["upstream_source_pose_reconstruction"] == "PASS"
            and gates["exact_model"] == "PASS"
            and gates["global_angular_minimax_bound"] == "PASS"
            and gates["finite_global_optimizer_locus"] == "PASS"
            and gates["all_global_targets_exact_ik"] == "PASS"
            and gates["neutrality_threshold"] == "PASS"
            and model["status"] == "PASS"
            and all(
                isinstance(model[key], str) and bool(model[key])
                for key in (
                    "mjcf_path",
                    "mjcf_sha256",
                    "mjcf_dependency_closure_sha256",
                    "compiled_model_sha256",
                    "compiled_mjb_sha256",
                    "urdf_path",
                    "urdf_sha256",
                    "backend_limits_sha256",
                    "backend_model_contract_sha256",
                )
            )
            and safety["exact_model"] is True
            and safety["passed"] is True
            and safety["collision_passed"] is True
            and safety["double_foot_ground_passed"] is True
            and safety["support_hull_passed"] is True
            and gates["grounded_ready_required_and_unresolved"] is False
            and receipt["joint_contract"][
                "fixed_joint_values_bitwise_equal"
            ]
            is True
            and pool["retained_for_outer_5x2_compiler_search"] is True
            and pool["eligibility_contract"]
            == "every_serialized_row_passes_exact_mujoco_static_safety"
            and pool["diagnostic_rejected_rows_are_not_serialized"] is True
            and isinstance(pool["count"], int)
            and not isinstance(pool["count"], bool)
            and pool["count"] > 0
            and isinstance(pool_rows, list)
            and len(pool_rows) == pool["count"]
            and all(
                row["outer_compiler_eligible"] is True
                and row["safety"]["status"]
                == "PASS_EXACT_MUJOCO_STATIC_READY"
                and row["safety"]["exact_model"] is True
                and row["safety"]["passed"] is True
                and row["safety"]["collision_passed"] is True
                and row["safety"]["double_foot_ground_passed"] is True
                and row["safety"]["support_hull_passed"] is True
                for row in pool_rows
            )
            and authorization
            == {
                "training_authorized": False,
                "deploy_authorized": False,
                "hardware_authorized": False,
            }
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise NeutralReadyError(
            f"candidate publication receipt is incomplete: {exc}"
        ) from exc
    if not complete:
        raise NeutralReadyError(
            "candidate publication receipt does not independently close all "
            "exact model, URDF, IK, collision, ground, and authorization gates"
        )


def contact_matrix_digest(contacts: Sequence[ContactPose]) -> str:
    """Return the canonical digest used by adapter and solver receipts."""

    return _contact_digest(contacts)


def array_content_sha256(value: np.ndarray) -> str:
    """Public canonical array digest for source-reconstruction receipts."""

    return _array_sha256(value)


def _contact_digest(contacts: Sequence[ContactPose]) -> str:
    payload = {
        "schema_version": 2,
        "rows": [
            {
                "label": row.label,
                "source_motion_path": str(
                    Path(row.source_motion_path).expanduser().absolute()
                ),
                "source_motion_sha256": row.source_motion_sha256,
                "source_frame_index": int(row.source_frame_index),
                "pose_content_sha256": row.pose_content_sha256,
                "pair_contract_sha256": row.pair_contract_sha256,
                "recomputed_pose_content_sha256": contact_pose_digest(row),
            }
            for row in contacts
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def seal_contact_source_proof(
    receipt: Mapping[str, Any],
    contacts: Sequence[ContactPose],
) -> ContactSourceProof:
    """Seal one adapter receipt against the exact 16 reconstructed contacts."""

    detached = json.loads(_canonical_json_bytes(receipt).decode("utf-8"))
    if frozenset(detached) != CONTACT_SOURCE_PROOF_KEYS:
        raise NeutralReadyError(
            "contact source proof keys changed: "
            f"got={sorted(detached)} expected={sorted(CONTACT_SOURCE_PROOF_KEYS)}"
        )
    if detached.get("contact_row_count") != len(contacts):
        raise NeutralReadyError("contact source proof row count changed")
    digest = _contact_digest(contacts)
    if detached.get("contact_matrix_sha256") != digest:
        raise NeutralReadyError(
            "contact source proof does not bind the supplied contact matrix"
        )
    payload_sha = hashlib.sha256(
        _canonical_json_bytes(detached)
    ).hexdigest()
    return ContactSourceProof(
        receipt=detached,
        payload_sha256=payload_sha,
    )


def _verify_contact_source_proof(
    proof: ContactSourceProof | None,
    contacts: Sequence[ContactPose],
    *,
    backend: face.RightRacketBackend,
    model_binding: ExactModelBinding | None,
) -> Mapping[str, Any] | None:
    if proof is None:
        return None
    if not isinstance(proof, ContactSourceProof):
        raise NeutralReadyError(
            "contact_source_proof must be ContactSourceProof or None"
        )
    detached = json.loads(
        _canonical_json_bytes(proof.receipt).decode("utf-8")
    )
    if frozenset(detached) != CONTACT_SOURCE_PROOF_KEYS:
        raise NeutralReadyError("contact source proof exact key set changed")
    actual_payload_sha = hashlib.sha256(
        _canonical_json_bytes(detached)
    ).hexdigest()
    if actual_payload_sha != proof.payload_sha256:
        raise NeutralReadyError("contact source proof payload seal changed")
    if (
        detached.get("schema_version") != 1
        or detached.get("builder_contract")
        != "canonical_block_lineage_pose_reconstruction_v2"
        or detached.get("contact_row_count")
        != len(SCOPES) * len(PHASES) * len(FACES)
        or detached.get("contact_matrix_sha256") != _contact_digest(contacts)
    ):
        raise NeutralReadyError(
            "contact source proof contract or matrix binding changed"
        )
    if model_binding is None:
        raise NeutralReadyError(
            "verified contact reconstruction requires an exact model binding"
        )
    model_row = detached.get("model_binding")
    expected_model_row = {
        "mjcf_path": str(
            Path(model_binding.mjcf_path).expanduser().absolute()
        ),
        "mjcf_sha256": model_binding.expected_mjcf_sha256,
        "compiled_model_sha256": (
            model_binding.expected_compiled_model_sha256
        ),
        "urdf_path": str(
            Path(model_binding.urdf_path).expanduser().absolute()
        ),
        "urdf_sha256": model_binding.expected_urdf_sha256,
        "backend_limits_sha256": (
            model_binding.expected_backend_limits_sha256
        ),
        "backend_model_contract_sha256": (
            model_binding.expected_backend_model_contract_sha256
        ),
        "racket_site_name": face.CANONICAL_RACKET_SITE,
        "normal_convention": NORMAL_CONVENTION,
    }
    if model_row != expected_model_row:
        raise NeutralReadyError(
            "contact reconstruction exact model/site binding changed"
        )
    file_rows = detached.get("file_bindings")
    if not isinstance(file_rows, list):
        raise NeutralReadyError(
            "contact source proof file_bindings must be a list"
        )
    required_roles = {
        "adapter_tool",
        "recipe",
        "body_scope_tool",
        "face_tool",
        "motion_source",
        "canonical_ready",
        "mjcf_root",
        "urdf",
        "phase_authority",
        "motion_player_tool",
        "motion_npz_decoder_tool",
        "motion_kinematics_contract_tool",
        "racket_geometry_contract_tool",
        "body_order",
    }
    roles: set[str] = set()
    expected_by_path: dict[Path, str] = {}
    for row in file_rows:
        if not isinstance(row, Mapping) or frozenset(row) != {
            "role",
            "path",
            "sha256",
        }:
            raise NeutralReadyError(
                "contact source proof file row schema changed"
            )
        role = str(row["role"])
        if not role or role in roles:
            raise NeutralReadyError(
                f"contact source proof duplicate/empty file role {role!r}"
            )
        roles.add(role)
        path = Path(str(row["path"])).expanduser().absolute()
        expected = _require_sha256(
            str(row["sha256"]),
            f"contact source proof {role} sha256",
        )
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise NeutralReadyError(
                f"contact source proof has conflicting pins for {path}"
            )
    if not required_roles.issubset(roles):
        raise NeutralReadyError(
            "contact source proof lacks required file roles: "
            f"{sorted(required_roles - roles)}"
        )
    for path, expected in expected_by_path.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256_file(path) != expected
        ):
            raise NeutralReadyError(
                f"contact source proof file/hash binding failed: {path}"
            )

    phase_frames = detached.get("phase_source_frames")
    if (
        not isinstance(phase_frames, Mapping)
        or frozenset(phase_frames) != frozenset(PHASES)
        or any(
            isinstance(phase_frames[name], bool)
            or not isinstance(phase_frames[name], int)
            or phase_frames[name] < 0
            for name in PHASES
        )
        or not (
            phase_frames["opportunity_start"]
            < phase_frames["construction_donor_preferred"]
            < phase_frames["nominal_event"]
            < phase_frames["opportunity_end"]
        )
    ):
        raise NeutralReadyError(
            "contact source proof phase frame contract changed"
        )
    face_contract = detached.get("face_contract")
    if (
        not isinstance(face_contract, Mapping)
        or face_contract.get("synthetic_behavior_preferred_frame") is not None
        or face_contract.get("construction_donor_preferred_frame")
        != phase_frames["construction_donor_preferred"]
        or face_contract.get("solver_anchor_annotation_frame")
        != phase_frames["nominal_event"]
        or face_contract.get(
            "post_retime_forehand_behavior_rescan_required"
        )
        is not True
        or face_contract.get("contact_truth_observed") is not False
    ):
        raise NeutralReadyError(
            "contact source proof invented synthetic behavior/contact truth"
        )
    expected_rows = {
        row.label: {
            "label": row.label,
            "source_frame_index": int(row.source_frame_index),
            "pose_content_sha256": row.pose_content_sha256,
            "pair_contract_sha256": row.pair_contract_sha256,
            "joint_pos_sha256": _array_sha256(row.joint_pos),
            "root_pos_w_sha256": _array_sha256(row.root_pos_w),
            "root_quat_w_sha256": _array_sha256(row.root_quat_w),
            "site_pos_w_sha256": _array_sha256(row.site_pos_w),
            "site_rotation_w_sha256": _array_sha256(row.site_rotation_w),
            "signed_face_normal_w_sha256": _array_sha256(
                row.signed_face_normal_w
            ),
        }
        for row in contacts
    }
    proof_rows = detached.get("rows")
    if (
        not isinstance(proof_rows, list)
        or len(proof_rows) != len(SCOPES) * len(PHASES) * len(FACES)
    ):
        raise NeutralReadyError(
            "contact source proof must contain exactly 16 row receipts"
        )
    actual_rows: dict[str, Any] = {}
    for row in proof_rows:
        if not isinstance(row, Mapping) or "label" not in row:
            raise NeutralReadyError("contact source proof row is malformed")
        label = str(row["label"])
        if label in actual_rows:
            raise NeutralReadyError(
                f"contact source proof duplicates row {label!r}"
            )
        actual_rows[label] = dict(row)
    if actual_rows != expected_rows:
        raise NeutralReadyError(
            "contact source proof row arrays/frames changed"
        )
    for row in contacts:
        if row.source_frame_index != int(phase_frames[row.phase]):
            raise NeutralReadyError(
                f"{row.label} frame disagrees with source proof phase map"
            )
    try:
        import canonical_neutral_ready_cli as trusted_adapter

        trusted_adapter.verify_real_contact_source_reconstruction(
            proof,
            tuple(contacts),
            backend=backend,
            model_binding=model_binding,
        )
    except NeutralReadyError:
        raise
    except Exception as exc:
        raise NeutralReadyError(
            "trusted source-frame reconstruction did not reproduce the "
            f"16 contact rows: {exc}"
        ) from exc
    return {
        "payload_sha256": proof.payload_sha256,
        "receipt": detached,
    }


def _reverify_contact_source_files(
    contacts: Sequence[ContactPose],
) -> None:
    """Rehash every unique contact source and reject conflicting bindings."""

    expected_by_path: dict[Path, str] = {}
    for row in contacts:
        path = Path(row.source_motion_path).expanduser().absolute()
        expected = str(row.source_motion_sha256)
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise NeutralReadyError(
                f"contact source has conflicting SHA-256 pins: {path}"
            )
    for path, expected in expected_by_path.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256_file(path) != expected
        ):
            raise NeutralReadyError(
                f"contact source changed during solve: {path}"
            )


def _reverify_contact_source_proof_files(
    wrapped_proof: Mapping[str, Any] | None,
) -> None:
    """Rehash every trusted-adapter proof input after the long IK search."""

    if wrapped_proof is None:
        raise NeutralReadyError(
            "verified contact source proof unexpectedly disappeared"
        )
    try:
        receipt = wrapped_proof["receipt"]
        expected_payload = str(wrapped_proof["payload_sha256"])
        file_rows = receipt["file_bindings"]
    except (KeyError, TypeError) as exc:
        raise NeutralReadyError(
            f"contact source proof receipt is incomplete: {exc}"
        ) from exc
    if hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest() != (
        expected_payload
    ):
        raise NeutralReadyError(
            "contact source proof payload changed during solve"
        )
    expected_by_path: dict[Path, str] = {}
    for row in file_rows:
        try:
            path = Path(str(row["path"])).expanduser().absolute()
            expected = _require_sha256(
                str(row["sha256"]), "contact proof file sha256"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NeutralReadyError(
                f"contact proof file binding is invalid: {exc}"
            ) from exc
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise NeutralReadyError(
                f"contact proof has conflicting file pins: {path}"
            )
    for path, expected in expected_by_path.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256_file(path) != expected
        ):
            raise NeutralReadyError(
                f"contact proof input changed during solve: {path}"
            )


def _reverify_receipt_contact_source_files(
    receipt: Mapping[str, Any],
) -> None:
    """Rehash source pins retained in a solved receipt immediately pre-write."""

    try:
        rows = receipt["contact_matrix"]["rows"]
    except (KeyError, TypeError) as exc:
        raise NeutralReadyError(
            f"contact source receipt is incomplete: {exc}"
        ) from exc
    if (
        not isinstance(rows, list)
        or len(rows) != len(SCOPES) * len(PHASES) * len(FACES)
    ):
        raise NeutralReadyError(
            "contact source receipt must contain the exact 16-row matrix"
        )
    expected_by_path: dict[Path, str] = {}
    for row in rows:
        try:
            path = Path(str(row["source_motion_path"])).expanduser().absolute()
            expected = _require_sha256(
                str(row["source_motion_sha256"]),
                "receipt contact source_motion_sha256",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NeutralReadyError(
                f"contact source receipt row is invalid: {exc}"
            ) from exc
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise NeutralReadyError(
                f"receipt contact source has conflicting pins: {path}"
            )
    for path, expected in expected_by_path.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256_file(path) != expected
        ):
            raise NeutralReadyError(
                f"contact source changed after candidate solve: {path}"
            )


def backend_limits_digest(
    backend: face.RightRacketBackend,
) -> tuple[str, dict[str, str]]:
    """Bind the exact joint order and all motion-limit arrays used by IK/LB."""

    joint_names = tuple(str(name) for name in backend.joint_names)
    arrays = {
        "position_lower": np.asarray(backend.position_lower, dtype=np.float64),
        "position_upper": np.asarray(backend.position_upper, dtype=np.float64),
        "velocity_limit": np.asarray(backend.velocity_limit, dtype=np.float64),
        "effort_limit": np.asarray(backend.effort_limit, dtype=np.float64),
    }
    for label, value in arrays.items():
        if value.shape != (len(joint_names),):
            raise NeutralReadyError(
                f"backend {label} must match the exact joint order"
            )
    array_hashes = {
        label: _array_sha256(value) for label, value in arrays.items()
    }
    payload = {
        "schema_version": 1,
        "joint_names": list(joint_names),
        "array_sha256": array_hashes,
    }
    combined = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return combined, array_hashes


def _compiled_mjb_sha256(model: Any, mujoco_module: Any) -> str:
    """Serialize the entire compiled model and hash the resulting MJB bytes."""

    to_bytes = getattr(model, "to_bytes", None)
    if callable(to_bytes):
        payload = to_bytes()
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise NeutralReadyError("MjModel.to_bytes returned non-bytes")
        return hashlib.sha256(bytes(payload)).hexdigest()

    descriptor, raw_path = tempfile.mkstemp(prefix="neutral-ready-", suffix=".mjb")
    os.close(descriptor)
    path = Path(raw_path)
    try:
        saved = False
        for arguments in (
            (model, str(path), None, 0),
            (model, str(path)),
        ):
            try:
                mujoco_module.mj_saveModel(*arguments)
                saved = True
                break
            except TypeError:
                continue
        if not saved or not path.is_file() or path.stat().st_size <= 0:
            raise NeutralReadyError(
                "MuJoCo binding cannot serialize the compiled model to MJB"
            )
        return _sha256_file(path)
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def exact_backend_model_contract_digest(
    backend: face.RightRacketBackend,
    *,
    compiled_model_sha256: str | None = None,
    compiled_mjb_sha256: str | None = None,
    backend_limits_sha256: str | None = None,
    mjcf_dependency_closure_sha256: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Bind compiled dynamics/collision arrays plus the racket-site contract."""

    model = getattr(backend, "model", None)
    mujoco = getattr(backend, "_mujoco", None)
    if model is None or mujoco is None:
        raise NeutralReadyError(
            "exact backend model contract requires MuJoCo model/module"
        )
    if compiled_model_sha256 is None:
        try:
            from canonical_mujoco_dynamics_gate import compiled_model_signature

            compiled_model_sha256 = compiled_model_signature(model)
        except Exception as exc:
            raise NeutralReadyError(
                f"cannot compute compiled-model signature: {exc}"
            ) from exc
    _require_sha256(
        compiled_model_sha256, "compiled_model_sha256"
    )
    if compiled_mjb_sha256 is None:
        compiled_mjb_sha256 = _compiled_mjb_sha256(model, mujoco)
    _require_sha256(compiled_mjb_sha256, "compiled_mjb_sha256")
    if backend_limits_sha256 is None:
        backend_limits_sha256, _ = backend_limits_digest(backend)
    _require_sha256(
        backend_limits_sha256, "backend_limits_sha256"
    )
    if mjcf_dependency_closure_sha256 is None:
        # This helper is also useful during binding construction.  Omitting the
        # closure remains representable, but _verify_exact_model always supplies
        # and therefore requires the real closure-bound form.
        closure_pin: str | None = None
    else:
        closure_pin = _require_sha256(
            mjcf_dependency_closure_sha256,
            "mjcf_dependency_closure_sha256",
        )
    site_id = getattr(backend, "_site_id", None)
    qpos_addresses = getattr(backend, "_qpos_addresses", None)
    dof_addresses = getattr(backend, "_dof_addresses", None)
    root_qpos_address = getattr(backend, "_root_qpos_address", None)
    if (
        not isinstance(site_id, (int, np.integer))
        or qpos_addresses is None
        or dof_addresses is None
        or not isinstance(root_qpos_address, (int, np.integer))
    ):
        raise NeutralReadyError(
            "exact backend lacks site/joint/root address bindings"
        )
    site_id = int(site_id)
    site_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_SITE, site_id
    )
    if site_name != face.CANONICAL_RACKET_SITE:
        raise NeutralReadyError(
            "backend site binding is not the canonical right_racket site"
        )
    site_body_id = int(model.site_bodyid[site_id])
    site_body_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, site_body_id
    )
    arrays = {
        "site_pos": np.asarray(model.site_pos[site_id], dtype=np.float64),
        "site_quat": np.asarray(model.site_quat[site_id], dtype=np.float64),
        "site_size": np.asarray(model.site_size[site_id], dtype=np.float64),
        "joint_qpos_addresses": np.asarray(qpos_addresses, dtype=np.int64),
        "joint_dof_addresses": np.asarray(dof_addresses, dtype=np.int64),
    }
    safety_array_names = (
        # Contact filtering and contact-parameter semantics omitted by the
        # legacy compiled signature.
        "geom_bodyid",
        "geom_type",
        "geom_dataid",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_priority",
        "geom_solmix",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "geom_friction",
        "geom_size",
        "geom_pos",
        "geom_quat",
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
        # Geometry loaded from external assets.
        "mesh_vert",
        "mesh_normal",
        "mesh_texcoord",
        "mesh_face",
        "mesh_graph",
        "mesh_scale",
        "mesh_pos",
        "mesh_quat",
        "hfield_size",
        "hfield_nrow",
        "hfield_ncol",
        "hfield_adr",
        "hfield_data",
        # Static kinematics and CoM/support-hull semantics.
        "body_parentid",
        "body_weldid",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_pos",
        "jnt_axis",
        "jnt_limited",
        "jnt_range",
        "qpos0",
    )
    safety_array_hashes: dict[str, str] = {}
    for name in safety_array_names:
        if hasattr(model, name):
            safety_array_hashes[name] = _array_sha256(
                np.asarray(getattr(model, name))
            )
    required_contact_arrays = (
        "geom_condim",
        "geom_priority",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "pair_signature",
        "exclude_signature",
    )
    missing_contact_arrays = [
        name for name in required_contact_arrays
        if name not in safety_array_hashes
    ]
    if missing_contact_arrays:
        raise NeutralReadyError(
            "compiled model lacks required static-contact arrays: "
            f"{missing_contact_arrays}"
        )
    option_fields: dict[str, int | float] = {}
    model_option = getattr(model, "opt", None)
    if model_option is None:
        raise NeutralReadyError("exact compiled model lacks mjOption")
    for name in (
        "disableflags",
        "enableflags",
        "cone",
        "jacobian",
        "integrator",
        "solver",
        "iterations",
        "ls_iterations",
        "timestep",
        "tolerance",
        "ls_tolerance",
        "impratio",
    ):
        if not hasattr(model_option, name):
            continue
        raw = getattr(model_option, name)
        if isinstance(raw, (int, np.integer)):
            option_fields[name] = int(raw)
        elif isinstance(raw, (float, np.floating)):
            value = float(raw)
            if not np.isfinite(value):
                raise NeutralReadyError(
                    f"compiled model option {name} is not finite"
                )
            option_fields[name] = value
        else:
            try:
                option_fields[name] = int(raw)
            except (TypeError, ValueError) as exc:
                raise NeutralReadyError(
                    f"cannot canonicalize compiled model option {name}"
                ) from exc
    if "disableflags" not in option_fields:
        raise NeutralReadyError(
            "compiled model contract lacks opt.disableflags"
        )
    optional_site_scalars: dict[str, int] = {}
    for name in ("site_type", "site_sameframe", "site_group"):
        if hasattr(model, name):
            optional_site_scalars[name] = int(getattr(model, name)[site_id])
    contract = {
        "schema_version": 1,
        "compiled_model_sha256": compiled_model_sha256,
        "compiled_mjb_sha256": compiled_mjb_sha256,
        "mjcf_dependency_closure_sha256": closure_pin,
        "backend_limits_sha256": backend_limits_sha256,
        "joint_names": [str(name) for name in backend.joint_names],
        "joint_qpos_addresses": arrays[
            "joint_qpos_addresses"
        ].tolist(),
        "joint_dof_addresses": arrays[
            "joint_dof_addresses"
        ].tolist(),
        "root_qpos_address": int(root_qpos_address),
        "root_body_name": str(backend.root_body_name),
        "site": {
            "id": site_id,
            "name": str(site_name),
            "body_id": site_body_id,
            "body_name": str(site_body_name),
            "array_sha256": {
                name: _array_sha256(value)
                for name, value in arrays.items()
                if name.startswith("site_")
            },
            **optional_site_scalars,
        },
        "address_array_sha256": {
            name: _array_sha256(value)
            for name, value in arrays.items()
            if name.endswith("_addresses")
        },
        "static_safety_model": {
            "array_sha256": safety_array_hashes,
            "option_fields": option_fields,
            "required_contact_arrays_present": True,
        },
    }
    digest = hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()
    return digest, contract


def _active_indices(joint_names: Sequence[str]) -> np.ndarray:
    names = tuple(str(name) for name in joint_names)
    if len(names) != len(set(names)):
        raise NeutralReadyError("backend joint names are not unique")
    missing = [name for name in face.RIGHT_STRIKE_CHAIN if name not in names]
    if missing:
        raise NeutralReadyError(
            f"backend lacks right striking-chain joints: {missing}"
        )
    return np.asarray(
        [names.index(name) for name in face.RIGHT_STRIKE_CHAIN], dtype=np.int64
    )


def _require_exact_runtime_joint_order(
    joint_names: Sequence[str],
    *,
    expected: Sequence[str] | None = None,
) -> None:
    if expected is None:
        import canonical_mujoco_dynamics_gate as dynamics_gate

        expected = dynamics_gate.RUNTIME_JOINT_NAMES
    actual_names = tuple(str(name) for name in joint_names)
    expected_names = tuple(str(name) for name in expected)
    if actual_names != expected_names or len(expected_names) != 31:
        raise NeutralReadyError(
            "exact model path requires the canonical 31-joint runtime order"
        )


def _projected_basis(axis: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    for candidate in (
        rotation[:, 1],
        rotation[:, 0],
        rotation[:, 2],
        *np.eye(3, dtype=np.float64),
    ):
        projected = candidate - axis * float(candidate @ axis)
        if float(np.linalg.norm(projected)) > 1.0e-10:
            return _unit(projected, "antipodal basis_u")
    raise NeutralReadyError("cannot construct antipodal midpoint basis")


def _canonicalize_plane_basis(
    basis_u: np.ndarray, basis_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u = _unit(basis_u, "basis_u")
    v = _unit(basis_v - u * float(basis_v @ u), "basis_v")
    pivot = int(np.argmax(np.abs(u)))
    if u[pivot] < 0.0:
        u = -u
        v = -v
    return u, v


def _convex_hull(points: np.ndarray) -> np.ndarray:
    rows = np.asarray(points, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(rows).all():
        raise NeutralReadyError("support points must be a finite (N,2) array")
    unique = sorted(set((float(x), float(y)) for x, y in rows))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64).reshape(-1, 2)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (
            a[1] - o[1]
        ) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _convex_polygon_margin(hull: np.ndarray, point: np.ndarray) -> float:
    polygon = np.asarray(hull, dtype=np.float64)
    value = _finite_vector(point, 2, "support point")
    if len(polygon) < 3:
        raise NeutralReadyError("support hull needs at least three vertices")
    margins: list[float] = []
    for index in range(len(polygon)):
        left = polygon[index]
        right = polygon[(index + 1) % len(polygon)]
        edge = right - left
        length = float(np.linalg.norm(edge))
        if length <= 1.0e-12:
            raise NeutralReadyError("support hull contains a zero edge")
        margins.append(
            float(
                (edge[0] * (value[1] - left[1])
                 - edge[1] * (value[0] - left[0]))
                / length
            )
        )
    return float(min(margins))


def _enforce_joint_limits(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: NeutralReadyConfig,
    label: str,
) -> None:
    q = np.asarray(values, dtype=np.float64)
    violation = max(
        0.0,
        float(np.max(lower - q)),
        float(np.max(q - upper)),
    )
    if violation > config.joint_limit_tolerance_rad:
        raise NeutralReadyError(
            f"{label} exceeds a joint limit by {violation:.9g} rad"
        )


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    a = _unit(left, "left normal")
    b = _unit(right, "right normal")
    return float(
        math.atan2(
            float(np.linalg.norm(np.cross(a, b))),
            float(np.clip(a @ b, -1.0, 1.0)),
        )
    )


def _tolerance_bucket(value: float, width: float) -> int:
    number = float(value)
    bucket = float(width)
    if not np.isfinite(number) or number < 0.0:
        raise NeutralReadyError("ranking value must be finite and non-negative")
    return int(math.floor(number / bucket + 0.5))


def _signed_tolerance_bucket(value: float, width: float) -> int:
    number = float(value)
    bucket = float(width)
    if not np.isfinite(number):
        raise NeutralReadyError("ranking value must be finite")
    return int(math.floor(number / bucket + 0.5))


def _vector_tie_key(value: np.ndarray) -> tuple[float, float, float]:
    vector = _unit(value, "tie-break normal")
    return tuple(float(row) for row in np.round(vector, decimals=14))


def _rotation(value: np.ndarray, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise NeutralReadyError(f"{label} must be finite shape (3,3)")
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
        or float(np.linalg.det(rotation)) <= 0.0
    ):
        raise NeutralReadyError(f"{label} is not a right-handed rotation")
    return rotation.copy()


def _finite_vector(value: np.ndarray, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise NeutralReadyError(
            f"{label} must be a finite vector of shape ({length},)"
        )
    return vector.copy()


def _unit(value: np.ndarray, label: str) -> np.ndarray:
    vector = _finite_vector(value, 3, label)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise NeutralReadyError(f"{label} is degenerate")
    return vector / norm


def _unit_preserve(
    value: np.ndarray,
    label: str,
    *,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Validate a sealed unit vector without changing its float64 bytes."""

    vector = _finite_vector(value, 3, label)
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > tolerance:
        raise NeutralReadyError(
            f"{label} must already be unit length; norm={norm:.17g}"
        )
    return vector


def _unit_quaternion(value: np.ndarray, label: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, label)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise NeutralReadyError(f"{label} must already be unit length")
    return quaternion / norm


def _ready_quaternion(value: np.ndarray, label: str) -> np.ndarray:
    """Validate but preserve the content-pinned donor quaternion bytes."""

    quaternion = _finite_vector(value, 4, label)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise NeutralReadyError(f"{label} must be unit within 1e-6")
    return quaternion


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise NeutralReadyError("object arrays cannot enter a content receipt")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().newbyteorder("<")
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_real_file_snapshot(
    path: Path,
    label: str,
) -> tuple[bytes, str]:
    """Read/hash one no-symlink regular-file descriptor and detect mutation."""

    target = Path(path).expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise NeutralReadyError(f"cannot open {label} snapshot: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise NeutralReadyError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(after, name)
            for name in identity_fields
        ):
            raise NeutralReadyError(f"{label} changed while being read")
        try:
            current = target.stat(follow_symlinks=False)
        except TypeError:  # pragma: no cover - old pathlib compatibility
            current = os.stat(target, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
        ):
            raise NeutralReadyError(
                f"{label} pathname changed while snapshotting"
            )
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise NeutralReadyError(
                f"{label} snapshot length disagrees with fstat"
            )
        return payload, hashlib.sha256(payload).hexdigest()
    except OSError as exc:
        raise NeutralReadyError(f"cannot read {label} snapshot: {exc}") from exc
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return _read_real_file_snapshot(path, str(path))[1]


def _require_sha256(value: str, label: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return text


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NeutralReadyError(f"receipt is not strict JSON: {exc}") from exc


def _assert_strict_json(value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    decoded = json.loads(
        payload.decode("utf-8"),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {constant}")
        ),
    )
    if not isinstance(decoded, dict):
        raise NeutralReadyError("receipt must encode one JSON object")


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
