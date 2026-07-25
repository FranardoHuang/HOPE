"""Pure-NumPy tests for the diagnostic face-neutral ready solver."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_face_manifold as face  # noqa: E402
import canonical_mujoco_dynamics_gate as dynamics_gate  # noqa: E402
import canonical_neutral_ready as neutral  # noqa: E402


def rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


class CoupledReadyBackend:
    """Two active joints must cooperate to preserve the site while rotating."""

    joint_names = tuple(dynamics_gate.RUNTIME_JOINT_NAMES)
    root_body_name = "pelvis_link"

    def __init__(self) -> None:
        count = len(self.joint_names)
        self.position_lower = np.full(count, -4.0, dtype=np.float64)
        self.position_upper = np.full(count, 4.0, dtype=np.float64)
        self.velocity_limit = np.full(count, 10.0, dtype=np.float64)
        self.effort_limit = np.full(count, 100.0, dtype=np.float64)

    def site_pose(self, joint_pos, root_pos_w, root_quat_w):
        del root_pos_w, root_quat_w
        q = np.asarray(joint_pos, dtype=np.float64)
        shoulder = self.joint_names.index("right_shoulder_pitch_joint")
        wrist = self.joint_names.index("right_wrist_roll_joint")
        # Keeping x=0 requires shoulder=-wrist.  Face angle is their
        # difference, so a face change cannot be a one-joint wrist rewrite.
        position = np.asarray(
            [
                q[shoulder] + q[wrist],
                q[self.joint_names.index("right_shoulder_roll_joint")],
                q[self.joint_names.index("right_shoulder_yaw_joint")],
            ]
        )
        return position, rotation_x(q[shoulder] - q[wrist])

    def diagonal_dynamics(self, joint_pos, root_pos_w, root_quat_w):
        del joint_pos, root_pos_w, root_quat_w
        count = len(self.joint_names)
        return np.ones(count), np.zeros(count)


class SignedBiasBackend(CoupledReadyBackend):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self._bias = float(bias)

    def diagonal_dynamics(self, joint_pos, root_pos_w, root_quat_w):
        del joint_pos, root_pos_w, root_quat_w
        count = len(self.joint_names)
        return np.ones(count), np.full(count, self._bias)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(backend: CoupledReadyBackend) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.zeros(len(backend.joint_names), dtype=np.float64)
    q[backend.joint_names.index("left_shoulder_pitch_joint")] = 0.37
    q[backend.joint_names.index("left_knee_joint")] = -0.29
    shoulder = backend.joint_names.index("right_shoulder_pitch_joint")
    wrist = backend.joint_names.index("right_wrist_roll_joint")
    q[shoulder] = 0.20
    q[wrist] = -0.20
    return q, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


def _contacts(
    backend: CoupledReadyBackend, tmp_path: Path
) -> tuple[neutral.ContactPose, ...]:
    rows: list[neutral.ContactPose] = []
    shoulder = backend.joint_names.index("right_shoulder_pitch_joint")
    wrist = backend.joint_names.index("right_wrist_roll_joint")
    root_pos = np.zeros(3)
    root_quat = np.asarray([1.0, 0.0, 0.0, 0.0])
    source_path = tmp_path / "contact_source.npz"
    if not source_path.exists():
        np.savez(source_path, fixture=np.asarray([20260724], dtype=np.int64))
    source_sha = _sha256(source_path)
    for scope_index, scope in enumerate(neutral.SCOPES):
        for phase_index, phase in enumerate(neutral.PHASES):
            # A common-mode shift changes the contact site but not the face.
            common = 0.01 * (scope_index + phase_index - 1)
            pair: list[neutral.ContactPose] = []
            for face_name, sign in (("bh", 1.0), ("fh", -1.0)):
                q = np.zeros(len(backend.joint_names), dtype=np.float64)
                q[
                    backend.joint_names.index("left_shoulder_pitch_joint")
                ] = 0.37
                q[backend.joint_names.index("left_knee_joint")] = -0.29
                q[shoulder] = sign * math.pi / 4.0 + common
                q[wrist] = -sign * math.pi / 4.0 + common
                position, rotation = backend.site_pose(q, root_pos, root_quat)
                pair.append(
                    neutral.ContactPose(
                        scope=scope,
                        phase=phase,
                        face_name=face_name,
                        joint_pos=q,
                        root_pos_w=root_pos,
                        root_quat_w=root_quat,
                        site_pos_w=position,
                        site_rotation_w=rotation,
                        signed_face_normal_w=rotation[:, 1],
                        source_motion_path=source_path,
                        source_motion_sha256=source_sha,
                        source_frame_index=scope_index * 3 + phase_index,
                        pose_content_sha256="0" * 64,
                        pair_contract_sha256="0" * 64,
                    )
                )
            pair = [
                replace(
                    row,
                    pose_content_sha256=neutral.contact_pose_digest(row),
                )
                for row in pair
            ]
            pair_digest = neutral.contact_pair_contract_digest(pair[0], pair[1])
            rows.extend(
                replace(row, pair_contract_sha256=pair_digest) for row in pair
            )
    return tuple(rows)


def _source_ready(
    tmp_path: Path,
    q: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> neutral.ReadySourceBinding:
    path = tmp_path / "canonical_ready_v1.npz"
    np.savez(
        path,
        joint_pos=q,
        joint_vel=np.zeros_like(q),
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        source_segment=np.asarray("bh_loop_c"),
        source_npz=np.asarray("fixture_source.npz"),
        source_frame=np.asarray(0, dtype=np.int64),
        striking_joint_ids=np.asarray(
            [
                backend_index
                for backend_index, name in enumerate(
                    dynamics_gate.RUNTIME_JOINT_NAMES
                )
                if name in face.RIGHT_STRIKE_CHAIN
            ],
            dtype=np.int64,
        ),
        note=np.asarray("unit-test canonical ready"),
    )
    return neutral.ReadySourceBinding(path=path, expected_sha256=_sha256(path))


def _solve(tmp_path: Path) -> neutral.NeutralReadyResult:
    backend = CoupledReadyBackend()
    ready, root_pos, root_quat = _ready(backend)
    return neutral.solve_neutral_ready_candidate(
        ready,
        root_pos,
        root_quat,
        _contacts(backend, tmp_path),
        backend=backend,
        ready_binding=_source_ready(tmp_path, ready, root_pos, root_quat),
        config=neutral.NeutralReadyConfig(
            antipodal_circle_samples=8,
            maximum_target_normals_for_ik=8,
            random_restarts=0,
            max_iterations=100,
            connector_dynamics_samples=2,
            site_tolerance_m=1.0e-8,
            normal_tolerance_rad=1.0e-7,
            input_site_tolerance_m=1.0e-10,
            input_rotation_tolerance_rad=1.0e-10,
        ),
    )


def test_strict_antipodal_midpoint_is_complete_symmetric_and_finite():
    normal_a = np.asarray([1.0, 0.0, 0.0])
    normal_b = -normal_a
    rows = neutral.spherical_geodesic_midpoint_targets(
        normal_a,
        normal_b,
        reference_rotation_w=np.eye(3),
        source_pair="upper:nominal_event:bh_fh",
        antipodal_circle_samples=12,
    )
    assert len(rows) == 12
    assert all(row.antipodal for row in rows)
    assert all(np.isfinite(row.normal_w).all() for row in rows)
    assert all(np.linalg.norm(row.normal_w) == pytest.approx(1.0) for row in rows)
    assert max(abs(float(row.normal_w @ normal_a)) for row in rows) < 1.0e-14
    for row in rows:
        assert neutral._angle(row.normal_w, normal_a) == pytest.approx(
            math.pi / 2.0
        )
        assert neutral._angle(row.normal_w, normal_b) == pytest.approx(
            math.pi / 2.0
        )


def test_rank1_circle_grid_is_sampled_and_phase_dependent():
    normal_a = np.asarray([1.0, 0.0, 0.0])
    normal_b = -normal_a
    first = neutral.spherical_geodesic_midpoint_targets(
        normal_a,
        normal_b,
        reference_rotation_w=np.eye(3),
        source_pair="p",
        antipodal_circle_samples=8,
    )
    second = neutral.spherical_geodesic_midpoint_targets(
        normal_a,
        normal_b,
        reference_rotation_w=rotation_x(math.radians(10.0)),
        source_pair="p",
        antipodal_circle_samples=8,
    )

    def physical_set(rows):
        return {
            tuple(float(value) for value in np.round(row.normal_w, 12))
            for row in rows
        }

    assert physical_set(first) != physical_set(second)
    solution = neutral.global_antipodal_angular_minimax_targets(
        np.asarray([normal_a]),
        reference_rotation_w=np.eye(3),
        config=neutral.NeutralReadyConfig(antipodal_circle_samples=8),
    )
    assert solution.axis_rank == 1
    assert solution.sampled_continuous_locus
    assert not solution.finite_optimizer_locus


def test_global_rank3_xyz_minimax_beats_every_pair_locus():
    solution = neutral.global_antipodal_angular_minimax_targets(
        np.eye(3, dtype=np.float64),
        reference_rotation_w=np.eye(3),
        config=neutral.NeutralReadyConfig(),
    )
    expected = 2.0 * math.asin(1.0 / math.sqrt(3.0))
    assert solution.axis_rank == 3
    assert solution.finite_optimizer_locus
    assert solution.angular_minimax_certified
    assert solution.angular_objective_lower_bound_rad == pytest.approx(
        expected, abs=1.0e-10
    )
    assert solution.angular_objective_upper_bound_rad == pytest.approx(
        expected, abs=1.0e-10
    )
    assert math.degrees(expected) == pytest.approx(70.528779, abs=1.0e-6)
    assert len(solution.targets) == 8
    for target in solution.targets:
        np.testing.assert_allclose(
            np.abs(target.normal_w),
            np.full(3, 1.0 / math.sqrt(3.0)),
            atol=1.0e-10,
        )
    # Any point on one pair's midpoint great circle has one zero projection
    # and a worst angular asymmetry of at least 90 degrees for x/y/z axes.
    assert expected < math.pi / 2.0


def test_global_rank2_minimax_is_exact_plus_minus_nullspace():
    axes = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    solution = neutral.global_antipodal_angular_minimax_targets(
        axes,
        reference_rotation_w=np.eye(3),
        config=neutral.NeutralReadyConfig(),
    )
    assert solution.axis_rank == 2
    assert solution.finite_optimizer_locus
    assert not solution.sampled_continuous_locus
    assert solution.angular_objective_lower_bound_rad == pytest.approx(0.0)
    assert solution.angular_objective_upper_bound_rad == pytest.approx(0.0)
    assert {
        tuple(np.round(row.normal_w, 12)) for row in solution.targets
    } == {(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)}


def test_near_antipodal_bound_is_conservative_not_false_equality():
    epsilon = 1.0e-4
    bh = np.asarray([1.0, 0.0, 0.0])
    fh = np.asarray(
        [-math.cos(epsilon), math.sin(epsilon), 0.0],
        dtype=np.float64,
    )
    axis = (bh - fh) / np.linalg.norm(bh - fh)
    solution = neutral.global_antipodal_angular_minimax_targets(
        np.asarray(
            [axis, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        paired_normals_w=np.asarray(
            [
                [bh, fh],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ],
            dtype=np.float64,
        ),
        reference_rotation_w=np.eye(3),
        config=neutral.NeutralReadyConfig(
            paired_face_antipodal_tolerance_rad=2.0e-4
        ),
    )
    assert solution.angular_objective_lower_bound_rad <= (
        solution.angular_objective_upper_bound_rad
    )
    assert (
        solution.angular_objective_upper_bound_rad
        - solution.angular_objective_lower_bound_rad
    ) > 0.0
    assert float(
        solution.angular_objective_upper_bound_rad
        - solution.angular_objective_lower_bound_rad
    ) <= 2.0 * epsilon + 1.0e-8


def test_nearly_coplanar_axis_is_not_promoted_to_fake_rank3():
    axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0e-8],
        ],
        dtype=np.float64,
    )
    solution = neutral.global_antipodal_angular_minimax_targets(
        axes,
        reference_rotation_w=np.eye(3),
        config=neutral.NeutralReadyConfig(normal_tolerance_rad=1.0e-5),
    )
    assert solution.axis_rank == 2


def test_plus_minus_90_face_construction_is_neutral_not_single_joint(tmp_path):
    backend = CoupledReadyBackend()
    donor, _, _ = _ready(backend)
    result = _solve(tmp_path)
    shoulder = backend.joint_names.index("right_shoulder_pitch_joint")
    wrist = backend.joint_names.index("right_wrist_roll_joint")

    np.testing.assert_allclose(result.candidate.raw_plus_y_w, [0.0, 1.0, 0.0], atol=1e-7)
    assert result.metrics.maximum_pair_angular_asymmetry_rad < 1.0e-7
    assert result.metrics.maximum_contact_angle_rad == pytest.approx(
        math.pi / 2.0, abs=1.0e-7
    )
    assert abs(result.candidate.joint_pos[shoulder] - donor[shoulder]) > 0.1
    assert abs(result.candidate.joint_pos[wrist] - donor[wrist]) > 0.1
    assert result.metrics.right_arm_changed_joint_count >= 2
    assert (
        result.receipt["joint_contract"]["single_joint_face_assumption"]
        is False
    )


def test_every_fixed_joint_is_bitwise_unchanged(tmp_path):
    backend = CoupledReadyBackend()
    donor, _, _ = _ready(backend)
    result = _solve(tmp_path)
    active = np.asarray(
        [backend.joint_names.index(name) for name in face.RIGHT_STRIKE_CHAIN]
    )
    fixed = np.setdiff1d(np.arange(len(donor)), active)
    np.testing.assert_array_equal(
        result.candidate.joint_pos[fixed], donor[fixed]
    )
    assert result.receipt["joint_contract"][
        "fixed_joint_values_bitwise_equal"
    ]
    assert result.receipt["joint_contract"]["active_joint_names"] == list(
        face.RIGHT_STRIKE_CHAIN
    )


def _metrics(*, worst: float, status: str, digest: str) -> neutral.CandidateMetrics:
    return neutral.CandidateMetrics(
        maximum_pair_angular_asymmetry_rad=0.0,
        maximum_contact_angle_rad=math.pi / 2.0,
        worst_right_arm_diagonal_time_proxy_s=worst,
        worst_connector_label="upper:opportunity_start:bh",
        donor_ready_to_candidate_diagonal_time_proxy_s=0.1,
        site_residual_m=0.0,
        target_normal_residual_rad=0.0,
        minimum_support_margin_m=0.01,
        safety_status=status,
        right_arm_changed_joint_count=2,
        candidate_sha256=digest * 64,
    )


def test_multibranch_selection_uses_worst_case_and_hard_safety():
    # Branch 1 could have a better average but has a worse maximum.  Only the
    # explicit maximum enters the selector.
    selected = neutral.select_minimax_metrics(
        [
            (_metrics(worst=0.9, status="NOT_EVALUATED_FAIL_CLOSED", digest="a"), 0),
            (_metrics(worst=0.6, status="NOT_EVALUATED_FAIL_CLOSED", digest="b"), 1),
        ]
    )
    assert selected == 1
    # A proven-safe branch outranks an unverified branch, even when the latter
    # has an optimistic connector lower bound.
    selected = neutral.select_minimax_metrics(
        [
            (_metrics(worst=0.7, status="PASS_EXACT_MUJOCO_STATIC_READY", digest="c"), 2),
            (_metrics(worst=0.1, status="NOT_EVALUATED_FAIL_CLOSED", digest="d"), 3),
        ]
    )
    assert selected == 2


def test_ranking_buckets_ignore_subtolerance_angle_and_support_noise():
    first = replace(
        _metrics(
            worst=0.40,
            status="NOT_EVALUATED_FAIL_CLOSED",
            digest="e",
        ),
        maximum_pair_angular_asymmetry_rad=2.1e-7,
        minimum_support_margin_m=0.010000000000001,
    )
    second = replace(
        _metrics(
            worst=0.20,
            status="NOT_EVALUATED_FAIL_CLOSED",
            digest="f",
        ),
        maximum_pair_angular_asymmetry_rad=2.9e-7,
        minimum_support_margin_m=0.010000000000002,
    )
    assert neutral.select_minimax_metrics(
        [(first, 0), (second, 1)],
        angle_bucket_rad=1.0e-6,
        support_bucket_m=1.0e-6,
    ) == 1


def test_same_face_pair_is_rejected_even_when_fk_and_digests_match(tmp_path):
    backend = CoupledReadyBackend()
    contacts = list(_contacts(backend, tmp_path))
    bh = contacts[0]
    old_fh = contacts[1]
    copied_fh = replace(
        old_fh,
        joint_pos=bh.joint_pos.copy(),
        root_pos_w=bh.root_pos_w.copy(),
        root_quat_w=bh.root_quat_w.copy(),
        site_pos_w=bh.site_pos_w.copy(),
        site_rotation_w=bh.site_rotation_w.copy(),
        signed_face_normal_w=bh.signed_face_normal_w.copy(),
        pose_content_sha256="0" * 64,
        pair_contract_sha256="0" * 64,
    )
    copied_fh = replace(
        copied_fh,
        pose_content_sha256=neutral.contact_pose_digest(copied_fh),
    )
    pair_digest = neutral.contact_pair_contract_digest(bh, copied_fh)
    contacts[0] = replace(bh, pair_contract_sha256=pair_digest)
    contacts[1] = replace(copied_fh, pair_contract_sha256=pair_digest)
    ready, root_pos, root_quat = _ready(backend)
    with pytest.raises(neutral.NeutralReadyError, match="not a signed antipodal"):
        neutral.solve_neutral_ready_candidate(
            ready,
            root_pos,
            root_quat,
            contacts,
            backend=backend,
            ready_binding=_source_ready(
                tmp_path, ready, root_pos, root_quat
            ),
            config=neutral.NeutralReadyConfig(
                antipodal_circle_samples=8,
                random_restarts=0,
            ),
        )


def test_pair_site_and_content_contract_tampering_fail_closed(tmp_path):
    backend = CoupledReadyBackend()
    contacts = list(_contacts(backend, tmp_path))
    with pytest.raises(neutral.NeutralReadyError, match="content contract"):
        tampered = list(contacts)
        tampered[0] = replace(
            tampered[0], pair_contract_sha256="f" * 64
        )
        neutral._all_midpoint_targets(
            tampered,
            reference_rotation=np.eye(3),
            config=neutral.NeutralReadyConfig(),
        )

    shifted = list(contacts)
    bh, fh = shifted[0], shifted[1]
    shifted_fh = replace(
        fh,
        site_pos_w=fh.site_pos_w + np.asarray([1.0e-3, 0.0, 0.0]),
        pose_content_sha256="0" * 64,
        pair_contract_sha256="0" * 64,
    )
    shifted_fh = replace(
        shifted_fh,
        pose_content_sha256=neutral.contact_pose_digest(shifted_fh),
    )
    shifted_pair = neutral.contact_pair_contract_digest(bh, shifted_fh)
    shifted[0] = replace(bh, pair_contract_sha256=shifted_pair)
    shifted[1] = replace(
        shifted_fh, pair_contract_sha256=shifted_pair
    )
    with pytest.raises(neutral.NeutralReadyError, match="racket sites differ"):
        neutral._all_midpoint_targets(
            shifted,
            reference_rotation=np.eye(3),
            config=neutral.NeutralReadyConfig(),
        )


def test_contact_source_mutation_and_ready_schema_drift_fail_closed(tmp_path):
    backend = CoupledReadyBackend()
    ready, root_pos, root_quat = _ready(backend)
    contacts = _contacts(backend, tmp_path)
    Path(contacts[0].source_motion_path).write_bytes(b"mutated")
    with pytest.raises(neutral.NeutralReadyError, match="source motion file/hash"):
        neutral.solve_neutral_ready_candidate(
            ready,
            root_pos,
            root_quat,
            contacts,
            backend=backend,
            ready_binding=_source_ready(
                tmp_path, ready, root_pos, root_quat
            ),
            config=neutral.NeutralReadyConfig(
                antipodal_circle_samples=8,
                random_restarts=0,
            ),
        )

    bad_ready = tmp_path / "bad_ready.npz"
    np.savez(
        bad_ready,
        joint_pos=ready.astype(np.float32),
        root_pos_w=root_pos,
        root_quat_w=root_quat,
    )
    with pytest.raises(neutral.NeutralReadyError, match="field set changed"):
        neutral._verify_ready_source(
            neutral.ReadySourceBinding(bad_ready, _sha256(bad_ready)),
            ready,
            root_pos,
            root_quat,
        )


def test_exact_runtime_order_rejects_a_31_joint_permutation():
    names = list(dynamics_gate.RUNTIME_JOINT_NAMES)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(
        neutral.NeutralReadyError, match="canonical 31-joint runtime order"
    ):
        neutral._require_exact_runtime_joint_order(names)


def test_receipt_binds_hashes_and_never_claims_training(tmp_path):
    result = _solve(tmp_path)
    receipt = result.receipt
    assert receipt["source_ready"]["sha256"] == _sha256(
        result.source_ready_path
    )
    assert receipt["source_ready"]["immutable_donor_baseline"]
    assert receipt["source_ready"]["overwritten"] is False
    assert receipt["model"]["status"] == "MISSING_FAIL_CLOSED"
    assert (
        receipt["normal_contract"]["convention"]
        == neutral.NORMAL_CONVENTION
    )
    assert receipt["contact_matrix"]["row_count"] == 16
    assert len(receipt["contact_matrix"]["input_sha256"]) == 64
    assert receipt["candidate"]["joint_pos_sha256"] == neutral._array_sha256(
        result.candidate.joint_pos
    )
    assert receipt["tool"]["sha256"] == _sha256(
        Path(neutral.__file__).resolve()
    )
    assert receipt["authorization"] == {
        "training_authorized": False,
        "deploy_authorized": False,
        "hardware_authorized": False,
    }
    assert receipt["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert not result.publication_allowed
    # Strict JSON: no NaN/Infinity and no object-only Python values.
    json.dumps(receipt, allow_nan=False)


def test_payload_seal_detects_root_and_nested_receipt_mutation(tmp_path):
    result = _solve(tmp_path)
    original = result.payload_seal_sha256
    result.root_pos_w[0] += 1.0e-3
    changed_root = neutral._result_payload_seal(
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
    assert changed_root != original
    result.root_pos_w[0] -= 1.0e-3
    result.receipt["authorization"]["training_authorized"] = True
    changed_receipt = neutral._result_payload_seal(
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
    assert changed_receipt != original


def test_exact_model_binding_rejects_urdf_hash_mismatch(tmp_path):
    backend = CoupledReadyBackend()
    mjcf = tmp_path / "model.xml"
    urdf = tmp_path / "robot.urdf"
    mjcf.write_text("<mujoco/>", encoding="utf-8")
    urdf.write_text("<robot name='wrong-bytes'/>", encoding="utf-8")
    limits_sha, _ = neutral.backend_limits_digest(backend)
    binding = neutral.ExactModelBinding(
        mjcf_path=mjcf,
        expected_mjcf_sha256=_sha256(mjcf),
        expected_compiled_model_sha256="1" * 64,
        urdf_path=urdf,
        expected_urdf_sha256="0" * 64,
        expected_backend_limits_sha256=limits_sha,
        expected_backend_model_contract_sha256="2" * 64,
    )
    with pytest.raises(neutral.NeutralReadyError, match="URDF SHA-256 mismatch"):
        neutral._verify_exact_model(backend, binding)


def test_failed_or_unverified_result_cannot_publish_or_overwrite(tmp_path):
    result = _solve(tmp_path)
    output = tmp_path / "neutral_candidate"
    with pytest.raises(neutral.NeutralReadyError, match="not publication"):
        neutral.publish_neutral_ready_candidate(result, output)
    assert not output.exists()
    np.testing.assert_array_equal(
        np.load(result.source_ready_path, allow_pickle=False)["joint_pos"],
        _ready(CoupledReadyBackend())[0],
    )

    # Bad exact-FK input fails before any publication path is touched.
    backend = CoupledReadyBackend()
    ready, root_pos, root_quat = _ready(backend)
    bad_contacts = list(_contacts(backend, tmp_path))
    bad = bad_contacts[0]
    bad_contacts[0] = neutral.ContactPose(
        scope=bad.scope,
        phase=bad.phase,
        face_name=bad.face_name,
        joint_pos=bad.joint_pos,
        root_pos_w=bad.root_pos_w,
        root_quat_w=bad.root_quat_w,
        site_pos_w=bad.site_pos_w + np.asarray([0.1, 0.0, 0.0]),
        site_rotation_w=bad.site_rotation_w,
        signed_face_normal_w=bad.signed_face_normal_w,
        source_motion_path=bad.source_motion_path,
        source_motion_sha256=bad.source_motion_sha256,
        source_frame_index=bad.source_frame_index,
        pose_content_sha256=bad.pose_content_sha256,
        pair_contract_sha256=bad.pair_contract_sha256,
    )
    never_created = tmp_path / "bad_candidate"
    with pytest.raises(neutral.NeutralReadyError, match="not exact backend FK"):
        neutral.solve_neutral_ready_candidate(
            ready,
            root_pos,
            root_quat,
            bad_contacts,
            backend=backend,
            ready_binding=neutral.ReadySourceBinding(
                result.source_ready_path, _sha256(result.source_ready_path)
            ),
            config=neutral.NeutralReadyConfig(
                antipodal_circle_samples=8,
                random_restarts=0,
            ),
        )
    assert not never_created.exists()


def test_sealed_unit_vector_validation_preserves_exact_bytes():
    raw = np.asarray(
        [0.123456789, 0.987654321, 0.333333333],
        dtype=np.float64,
    )
    sealed = raw / np.linalg.norm(raw)
    assert not np.array_equal(
        sealed,
        neutral._unit(sealed, "second normalization"),
    )
    validated = neutral._unit_preserve(sealed, "sealed normal")
    np.testing.assert_array_equal(validated, sealed)

    with pytest.raises(neutral.NeutralReadyError, match="already be unit"):
        neutral._unit_preserve(1.01 * sealed, "scaled normal")


def test_diagonal_proxy_is_invariant_to_bias_sign():
    ready, root_pos, root_quat = _ready(CoupledReadyBackend())
    active = neutral._active_indices(dynamics_gate.RUNTIME_JOINT_NAMES)
    end = ready[active].copy()
    end[0] += 0.5
    rows = []
    for bias in (10.0, -10.0):
        backend = SignedBiasBackend(bias)
        rows.append(
            neutral._right_arm_diagonal_time_proxy(
                source_q=ready,
                value_active=end,
                ready=ready,
                root_pos=root_pos,
                root_quat=root_quat,
                backend=backend,
                active=active,
                velocity_limit=backend.velocity_limit[active],
                effort_limit=backend.effort_limit[active],
                config=face.FaceManifoldConfig(
                    connector_dynamics_samples=2
                ),
                start_label="ready",
                end_label="target",
            )
        )
    assert rows[0].time_lower_bound_s == pytest.approx(
        rows[1].time_lower_bound_s
    )
    assert rows[0].acceleration_lower_envelope_rad_s2 == pytest.approx(
        (90.0,) * len(active)
    )


def test_atomic_directory_publish_never_clobbers_existing_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("new", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    (target / "payload").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        neutral._rename_directory_noreplace(source, target)
    assert (target / "payload").read_text(encoding="utf-8") == "old"
    assert (source / "payload").read_text(encoding="utf-8") == "new"

    fresh = tmp_path / "fresh"
    neutral._rename_directory_noreplace(source, fresh)
    assert not source.exists()
    assert (fresh / "payload").read_text(encoding="utf-8") == "new"


def test_stage_cleanup_is_dirfd_bound_and_refuses_unknown_files(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / neutral.CANDIDATE_FILENAME).write_bytes(b"candidate")
    (stage / neutral.RECEIPT_FILENAME).write_bytes(b"receipt")
    identity = stage.stat()
    descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        neutral._cleanup_owned_stage(stage, identity, descriptor)
    finally:
        os.close(descriptor)
    assert not stage.exists()

    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (guarded / neutral.CANDIDATE_FILENAME).write_bytes(b"candidate")
    (guarded / "unexpected").write_bytes(b"keep")
    identity = guarded.stat()
    descriptor = os.open(
        guarded, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        neutral._cleanup_owned_stage(guarded, identity, descriptor)
    finally:
        os.close(descriptor)
    assert guarded.is_dir()
    assert (guarded / "unexpected").read_bytes() == b"keep"


def test_nonexact_result_exports_no_outer_compiler_pool(tmp_path):
    result = _solve(tmp_path)
    assert result.candidate_pool == ()
    assert result.receipt["candidate_pool"]["count"] == 0
    assert (
        result.receipt["candidate_pool"]["eligibility_contract"]
        == "every_serialized_row_passes_exact_mujoco_static_safety"
    )
    assert (
        result.receipt["gates"]["grounded_ready_required_and_unresolved"]
        is True
    )


def test_publication_policy_rejects_relaxed_geometry_tolerance(tmp_path):
    result = _solve(tmp_path)
    receipt = copy.deepcopy(result.receipt)
    receipt["config"]["site_tolerance_m"] *= 100.0
    with pytest.raises(
        neutral.NeutralReadyError,
        match="may vary search breadth only",
    ):
        neutral._publication_config_from_receipt(receipt)


def test_reaudit_rejects_caller_resealed_forged_candidate(
    tmp_path, monkeypatch
):
    pristine = _solve(tmp_path)
    forged = copy.deepcopy(pristine)
    forged.candidate.joint_pos[0] += 0.01
    source_sha = _sha256(forged.source_ready_path)
    fake_binding = neutral.ExactModelBinding(
        mjcf_path=forged.source_ready_path,
        expected_mjcf_sha256=source_sha,
        expected_compiled_model_sha256="1" * 64,
        urdf_path=forged.source_ready_path,
        expected_urdf_sha256=source_sha,
        expected_backend_limits_sha256="2" * 64,
        expected_backend_model_contract_sha256="3" * 64,
    )
    empty_receipt: dict[str, object] = {}
    fake_proof = neutral.ContactSourceProof(
        receipt=empty_receipt,
        payload_sha256=hashlib.sha256(
            neutral._canonical_json_bytes(empty_receipt)
        ).hexdigest(),
    )
    forged = replace(
        forged,
        publication_allowed=True,
        model_binding=fake_binding,
        contact_source_proof=fake_proof,
    )
    forged = replace(
        forged,
        payload_seal_sha256=neutral._result_payload_seal(
            candidate=forged.candidate,
            target=forged.target,
            metrics=forged.metrics,
            root_pos=forged.root_pos_w,
            root_quat=forged.root_quat_w,
            receipt=forged.receipt,
            source_ready_path=forged.source_ready_path,
            publication_allowed=True,
            candidate_pool=forged.candidate_pool,
            contacts=forged.contacts,
            contact_source_proof=fake_proof,
        ),
    )
    replay = replace(pristine, publication_allowed=True)
    monkeypatch.setattr(neutral, "_require_publishable_receipt", lambda _: None)
    monkeypatch.setattr(
        neutral,
        "_publication_config_from_receipt",
        lambda _: neutral.NeutralReadyConfig(),
    )
    monkeypatch.setattr(
        neutral, "solve_neutral_ready_candidate", lambda *args, **kwargs: replay
    )
    with pytest.raises(
        neutral.NeutralReadyError,
        match="does not reproduce candidate",
    ):
        neutral._reaudit_result_for_publication(forged)
