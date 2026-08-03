"""Strict fake-backend tests for stationary grounded-ready candidates."""

from __future__ import annotations

import hashlib
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

import canonical_grounded_ready as grounded  # noqa: E402


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


class StrictFakeBackend:
    joint_names = grounded.RUNTIME_JOINT_NAMES

    def __init__(
        self,
        identity: grounded.ExactModelIdentity,
        *,
        dynamics: bool | None = True,
        singular: bool = False,
    ) -> None:
        self.exact_model_identity = identity
        self.position_lower = np.full(31, -2.0, np.float64)
        self.position_upper = np.full(31, 2.0, np.float64)
        self._dynamics = dynamics
        self._singular = singular
        self._leg = (
            grounded._joint_indices(self.joint_names, grounded.LEFT_LEG_JOINT_NAMES),
            grounded._joint_indices(self.joint_names, grounded.RIGHT_LEG_JOINT_NAMES),
        )

    def foot_poses(
        self, state: grounded.ReadyState
    ) -> tuple[grounded.FootPose, grounded.FootPose]:
        rows = []
        for foot, indices in enumerate(self._leg):
            values = state.joint_pos[indices]
            position = np.asarray(
                [
                    values[0],
                    (0.13 if foot == 0 else -0.13) + values[1],
                    values[2],
                ],
                np.float64,
            )
            rotation_vector = values[3:6].copy()
            if self._singular:
                rotation_vector[2] = 0.0
            rows.append(
                grounded.FootPose(
                    position,
                    grounded._so3_exp(rotation_vector),
                )
            )
        return tuple(rows)

    def flat_foot_targets(
        self,
        state: grounded.ReadyState,
        *,
        contact_preload_m: float,
    ) -> tuple[grounded.FootPose, grounded.FootPose]:
        targets = []
        for pose in self.foot_poses(state):
            yaw = math.atan2(
                float(pose.rotation_w[1, 0]),
                float(pose.rotation_w[0, 0]),
            )
            position = pose.position_w.copy()
            position[2] = -float(contact_preload_m)
            targets.append(grounded.FootPose(position, _rotation_z(yaw)))
        return tuple(targets)

    def static_scene(
        self,
        state: grounded.ReadyState,
        *,
        contact_gap_tolerance_m: float,
        penetration_tolerance_m: float,
    ) -> grounded.StaticScene:
        del contact_gap_tolerance_m, penetration_tolerance_m
        poses = self.foot_poses(state)
        counts = []
        points = []
        foot_indices = []
        for foot, pose in enumerate(poses):
            flat = abs(float(pose.rotation_w[2, 2]) - 1.0) < 2.0e-3
            grounded_foot = abs(float(pose.position_w[2])) < 2.0e-3
            count = 4 if flat and grounded_foot else 0
            counts.append(count)
            if count:
                for dx, dy in (
                    (-0.09, -0.045),
                    (-0.09, 0.045),
                    (0.09, -0.045),
                    (0.09, 0.045),
                ):
                    points.append(
                        [pose.position_w[0] + dx, pose.position_w[1] + dy, 0.0]
                    )
                    foot_indices.append(foot)
        shoulder = self.joint_names.index("right_shoulder_roll_joint")
        collision = (
            (
                {
                    "geom_ids": [7, 11],
                    "distance_m": -0.001,
                    "reason": "fixture upper collision",
                },
            )
            if state.joint_pos[shoulder] > 0.8
            else ()
        )
        return grounded.StaticScene(
            foot_contact_count=(counts[0], counts[1]),
            contact_points_w=np.asarray(points, np.float64).reshape(-1, 3),
            contact_foot_indices=np.asarray(foot_indices, np.int64),
            floor_origin_w=np.zeros(3),
            floor_basis_w=np.eye(3),
            sole_minimum_distance_m=np.asarray(
                [pose.position_w[2] for pose in poses], np.float64
            ),
            com_w=np.asarray([0.0, 0.0, 0.9]),
            maximum_foot_penetration_m=max(
                0.0,
                -min(float(pose.position_w[2]) for pose in poses),
            ),
            self_collision_pairs=collision,
            details={"backend": "strict_fake"},
        )

    def static_ground_dynamics(
        self,
        state: grounded.ReadyState,
        *,
        contact_gap_tolerance_m: float,
        penetration_tolerance_m: float,
    ) -> dict:
        del state, contact_gap_tolerance_m, penetration_tolerance_m
        if self._dynamics is None:
            return {
                "status": "INCOMPLETE_FAIL_CLOSED",
                "feasible": None,
                "reason": "fixture has no LP",
            }
        return {
            "status": (
                "PASS_STATIC_GROUND_CONTACT_LP"
                if self._dynamics
                else "FAIL_STATIC_GROUND_CONTACT_LP"
            ),
            "feasible": self._dynamics,
            "root_residual": 0.0,
        }

    def vendor_key_state(self, key_index: int = 0) -> grounded.ReadyState:
        if int(key_index) != 0:
            raise IndexError(key_index)
        return _vendor_key(self)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def identity(tmp_path: Path) -> grounded.ExactModelIdentity:
    model = tmp_path / "fixture.xml"
    model.write_text("<mujoco model='fixture'/>", encoding="utf-8")
    return grounded.ExactModelIdentity(
        mjcf_path=model,
        mjcf_sha256=_sha(model),
        compiled_model_sha256="1" * 64,
        path_model_binding_sha256="2" * 64,
        ground_model_binding_sha256="3" * 64,
        xml_model_name="fixture",
    )


def _donor(backend: StrictFakeBackend) -> grounded.ReadyState:
    q = np.zeros(31, np.float64)
    for foot, indices in enumerate(backend._leg):
        q[indices[0]] = 0.01 * foot
        q[indices[1]] = -0.005 * foot
        q[indices[2]] = 0.012 + 0.004 * foot
        q[indices[3:6]] = [0.08, -0.055, 0.12 if foot == 0 else -0.09]
    q[backend.joint_names.index("waist_yaw_joint")] = 0.17
    q[backend.joint_names.index("right_shoulder_pitch_joint")] = -0.31
    return grounded.ReadyState(
        q,
        np.asarray([0.0, 0.0, 0.92]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )


def _vendor_key(backend: StrictFakeBackend) -> grounded.ReadyState:
    q = np.full(31, 0.03, np.float64)
    for indices in backend._leg:
        q[indices] = 0.0
    return grounded.ReadyState(
        q,
        np.asarray([0.0, 0.0, 1.07]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )


def _config() -> grounded.GroundedReadyConfig:
    return grounded.GroundedReadyConfig(
        continuation_steps=5,
        maximum_iterations_per_step=35,
        finite_difference_step_rad=1.0e-6,
        foot_position_tolerance_m=2.0e-7,
        foot_rotation_tolerance_rad=2.0e-7,
        jacobian_rank_tolerance=1.0e-6,
    )


def test_support_margin_uses_numpy2_compatible_scalar_cross(
    monkeypatch: pytest.MonkeyPatch,
):
    scene = grounded.StaticScene(
        foot_contact_count=(2, 2),
        contact_points_w=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            np.float64,
        ),
        contact_foot_indices=np.asarray([0, 0, 1, 1], np.int64),
        floor_origin_w=np.zeros(3, np.float64),
        floor_basis_w=np.eye(3, dtype=np.float64),
        sole_minimum_distance_m=np.zeros(2, np.float64),
        com_w=np.asarray([0.0, 0.0, 0.9], np.float64),
        maximum_foot_penetration_m=0.0,
    )

    def reject_deprecated_vector_cross(*_args, **_kwargs):
        raise AssertionError("2-D np.cross must not be used")

    monkeypatch.setattr(np, "cross", reject_deprecated_vector_cross)
    hull, com_floor_xy, margin = grounded._support_margin(scene)
    assert hull.shape == (4, 2)
    np.testing.assert_array_equal(com_floor_xy, np.zeros(2, np.float64))
    assert margin == pytest.approx(1.0, abs=1.0e-15)


def test_g1_preserves_root_nonlegs_and_solves_flat_double_support(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity)
    donor = _donor(backend)
    source = backend.foot_poses(donor)
    result = grounded.solve_g1_donor_root(
        donor,
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )

    assert result.candidate_id == "G1"
    assert result.geometry_passed
    assert result.ground_dynamics_passed is True
    assert result.receipt["verdict"] == "PASS_TEST_BACKEND_ONLY"
    assert result.receipt["authorization"]["training_authorized"] is False
    assert result.receipt["selection"]["selected_as_canonical_ready"] is False
    np.testing.assert_array_equal(result.state.root_pos_w, donor.root_pos_w)
    np.testing.assert_array_equal(result.state.root_quat_wxyz, donor.root_quat_wxyz)
    nonleg = grounded._joint_indices(backend.joint_names, grounded.UPPER_JOINT_NAMES)
    np.testing.assert_array_equal(
        result.state.joint_pos[nonleg], donor.joint_pos[nonleg]
    )
    target = result.target_foot_poses
    for old, new in zip(source, target):
        np.testing.assert_allclose(new.position_w[:2], old.position_w[:2])
        assert new.position_w[2] == pytest.approx(-_config().target_contact_preload_m)
        assert new.rotation_w[2, 2] == pytest.approx(1.0)
        old_yaw = math.atan2(old.rotation_w[1, 0], old.rotation_w[0, 0])
        new_yaw = math.atan2(new.rotation_w[1, 0], new.rotation_w[0, 0])
        assert new_yaw == pytest.approx(old_yaw)
    assert len(result.receipt["source"]["solver_trace"]) == _config().continuation_steps
    with pytest.raises(ValueError):
        result.state.joint_pos[0] = 99.0
    with pytest.raises(TypeError):
        result.receipt["verdict"] = "mutated"
    json.dumps(grounded._jsonable(result.receipt), allow_nan=False)


def test_g1s_derives_support_edge_shift_and_changes_only_leg12(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity)
    donor = _donor(backend)
    original_static_scene = backend.static_scene

    def support_edge_scene(state, **kwargs):
        scene = original_static_scene(state, **kwargs)
        return replace(
            scene,
            com_w=np.asarray([0.1003, 0.0, 0.9], np.float64),
        )

    backend.static_scene = support_edge_scene
    result = grounded.solve_g1_support_edge_projection(
        donor,
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )

    assert result.candidate_id == "G1S"
    assert result.geometry_passed is True
    assert result.ground_dynamics_passed is True
    support = result.receipt["static_geometry"]["support"]
    assert support["margin_m"] >= 5.0e-4
    source = result.receipt["source"]
    assert source["mode"] == "G1S_donor_root_flat_feet_support_edge_projection"
    assert len(source["projection_attempts"]) == 2
    attempt0, attempt1 = source["projection_attempts"]
    assert attempt0["support_margin_m"] < 0.0
    assert attempt1["support_margin_m"] >= 5.0e-4
    assert attempt1["static_ground_dynamics"]["feasible"] is True
    assert result.receipt["gates"]["static_ground_dynamics"] == "PASS"
    final_shift = np.asarray(
        source["final_common_target_shift_floor_xy_m"], np.float64
    )
    assert 5.0e-4 <= np.linalg.norm(final_shift) < 3.0e-2
    inward, edge_margin = grounded._limiting_support_edge(
        np.asarray(attempt0["support_hull_floor_xy_m"], np.float64),
        np.asarray(attempt0["com_projection_floor_xy_m"], np.float64),
    )
    projection = grounded.SupportEdgeProjectionConfig()
    expected_shift = -(
        projection.required_support_margin_m
        - edge_margin
        + projection.correction_guard_m
    ) * inward
    np.testing.assert_allclose(final_shift, expected_shift, atol=1.0e-15)
    assert source["donor_state_sha256"] == grounded.state_digest(donor)
    assert source["projection_config"]["required_support_margin_m"] == 5.0e-4

    leg = grounded._joint_indices(backend.joint_names, grounded.LEG_JOINT_NAMES)
    nonleg = grounded._joint_indices(backend.joint_names, grounded.UPPER_JOINT_NAMES)
    changed = np.flatnonzero(result.state.joint_pos != donor.joint_pos)
    assert set(changed).issubset(set(leg))
    assert list(source["changed_joint_indices"]) == changed.tolist()
    assert list(source["changed_joint_names"]) == [
        backend.joint_names[int(index)] for index in changed
    ]
    np.testing.assert_array_equal(
        result.state.joint_pos[nonleg], donor.joint_pos[nonleg]
    )
    np.testing.assert_array_equal(result.state.root_pos_w, donor.root_pos_w)
    np.testing.assert_array_equal(
        result.state.root_quat_wxyz, donor.root_quat_wxyz
    )


@pytest.mark.parametrize("dynamics", [False, None])
def test_g1s_fails_closed_when_margin_passes_but_ground_lp_does_not(
    identity: grounded.ExactModelIdentity,
    dynamics: bool | None,
):
    backend = StrictFakeBackend(identity, dynamics=dynamics)
    with pytest.raises(
        grounded.GroundedReadyError,
        match="support margin.*static ground LP",
    ) as caught:
        grounded.solve_g1_support_edge_projection(
            _donor(backend),
            backend=backend,
            expected_model_identity=identity,
            config=_config(),
        )
    assert caught.value.code == "G1S_GROUND_DYNAMICS_FAILED"
    assert caught.value.report["projection_attempts"][-1][
        "static_ground_dynamics"
    ]["feasible"] is dynamics


def test_g2_uses_vendor_root_lower_and_supplied_upper_then_reaudits(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity)
    donor = _donor(backend)
    upper = donor.joint_pos.copy()
    right_wrist = backend.joint_names.index("right_wrist_roll_joint")
    upper[right_wrist] = 0.63
    result = grounded.build_g2_vendor_key_candidate(
        donor,
        backend=backend,
        expected_model_identity=identity,
        upper_candidate_joint_pos=upper,
        upper_joint_names=("right_wrist_roll_joint",),
        config=_config(),
    )
    leg = grounded._joint_indices(backend.joint_names, grounded.LEG_JOINT_NAMES)
    vendor = backend.vendor_key_state(0)
    np.testing.assert_array_equal(result.state.joint_pos[leg], vendor.joint_pos[leg])
    np.testing.assert_array_equal(result.state.root_pos_w, vendor.root_pos_w)
    assert result.state.joint_pos[right_wrist] == pytest.approx(0.63)
    assert result.geometry_passed
    assert result.receipt["source"]["upper_overlay"]["applied"] is True


def test_overlay_revalidation_can_invalidate_grounded_base(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity)
    base = grounded.solve_g1_donor_root(
        _donor(backend),
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )
    upper = base.state.joint_pos.copy()
    shoulder = backend.joint_names.index("right_shoulder_roll_joint")
    upper[shoulder] = 1.0
    result = grounded.revalidate_upper_overlay(
        base,
        upper,
        backend=backend,
        expected_model_identity=identity,
        upper_joint_names=("right_shoulder_roll_joint",),
        config=_config(),
    )
    assert not result.geometry_passed
    assert result.receipt["verdict"] == "FAIL_STATIC_GROUNDED_READY"
    assert result.receipt["gates"]["collision"] == "FAIL_CLOSED"
    leg = grounded._joint_indices(backend.joint_names, grounded.LEG_JOINT_NAMES)
    np.testing.assert_array_equal(
        result.state.joint_pos[leg], base.state.joint_pos[leg]
    )
    np.testing.assert_array_equal(result.state.root_pos_w, base.state.root_pos_w)


def test_model_identity_mismatch_fails_before_solving(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity)
    wrong = replace(identity, ground_model_binding_sha256="f" * 64)
    with pytest.raises(
        grounded.GroundedReadyError, match="identity mismatch"
    ) as caught:
        grounded.solve_g1_donor_root(
            _donor(backend),
            backend=backend,
            expected_model_identity=wrong,
            config=_config(),
        )
    assert caught.value.code == "MODEL_IDENTITY_MISMATCH"


def test_g1_singular_foot_manifold_fails_closed(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity, singular=True)
    with pytest.raises(
        grounded.GroundedReadyError, match="singular|ill-conditioned"
    ) as caught:
        grounded.solve_g1_donor_root(
            _donor(backend),
            backend=backend,
            expected_model_identity=identity,
            config=_config(),
        )
    assert caught.value.code == "FOOT_MANIFOLD_SINGULAR"


def test_missing_ground_lp_is_explicit_incomplete_not_a_pass(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity, dynamics=None)
    result = grounded.solve_g1_donor_root(
        _donor(backend),
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )
    assert result.geometry_passed
    assert result.ground_dynamics_passed is None
    assert result.receipt["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert result.receipt["gates"]["static_ground_dynamics"] == "INCOMPLETE_FAIL_CLOSED"


def test_publication_is_receipt_last_and_never_overwrites(
    identity: grounded.ExactModelIdentity,
    tmp_path: Path,
):
    backend = StrictFakeBackend(identity)
    result = grounded.solve_g1_donor_root(
        _donor(backend),
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )
    output = tmp_path / "published"
    published = grounded.publish_grounded_ready_candidate(result, output)
    assert published.candidate_npz.is_file()
    assert published.receipt_json.is_file()
    assert _sha(published.candidate_npz) == published.candidate_npz_sha256
    assert _sha(published.receipt_json) == published.receipt_json_sha256
    receipt = json.loads(published.receipt_json.read_text(encoding="ascii"))
    assert receipt["publication"]["candidate_npz_sha256"] == _sha(
        published.candidate_npz
    )
    with np.load(published.candidate_npz, allow_pickle=False) as data:
        assert not bool(data["training_authorized"])
        assert str(data["receipt_sha256"]) == result.receipt_sha256
    with pytest.raises(FileExistsError, match="overwrite"):
        grounded.publish_grounded_ready_candidate(result, output)


def test_leg_names_are_forbidden_in_upper_overlay(
    identity: grounded.ExactModelIdentity,
):
    backend = StrictFakeBackend(identity)
    donor = _donor(backend)
    with pytest.raises(grounded.GroundedReadyError, match="leg=") as caught:
        grounded.solve_g1_donor_root(
            donor,
            backend=backend,
            expected_model_identity=identity,
            upper_candidate_joint_pos=donor.joint_pos,
            upper_joint_names=("left_knee_joint",),
            config=_config(),
        )
    assert caught.value.code == "INVALID_UPPER_OVERLAY"


def test_result_rejects_receipt_hash_and_authorization_forgery(
    identity: grounded.ExactModelIdentity,
    tmp_path: Path,
):
    backend = StrictFakeBackend(identity)
    result = grounded.solve_g1_donor_root(
        _donor(backend),
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )
    spoof_exact = object.__new__(grounded.MujocoGroundedReadyBackend)
    spoof_exact.exact_model_identity = identity
    spoof_exact.position_lower = backend.position_lower
    spoof_exact.position_upper = backend.position_upper
    spoof_exact.assert_current_model_identity = lambda _expected: None
    for method_name in (
        "foot_poses",
        "flat_foot_targets",
        "static_scene",
        "static_ground_dynamics",
        "vendor_key_state",
    ):
        setattr(spoof_exact, method_name, getattr(backend, method_name))
    forged = grounded._jsonable(result.receipt)
    forged["authorization"]["training_authorized"] = True
    forged.pop("receipt_payload_sha256")
    forged_sha = hashlib.sha256(grounded._canonical_json_bytes(forged)).hexdigest()
    forged["receipt_payload_sha256"] = forged_sha
    with pytest.raises(grounded.GroundedReadyError, match="deny training") as caught:
        grounded.GroundedReadyResult(
            candidate_id=result.candidate_id,
            state=result.state,
            target_foot_poses=result.target_foot_poses,
            geometry_passed=result.geometry_passed,
            ground_dynamics_passed=result.ground_dynamics_passed,
            receipt=forged,
            receipt_sha256=forged_sha,
        )
    assert caught.value.code == "AUTHORIZATION_FORBIDDEN"

    with pytest.raises(grounded.GroundedReadyError, match="seal") as caught:
        grounded.GroundedReadyResult(
            candidate_id=result.candidate_id,
            state=result.state,
            target_foot_poses=result.target_foot_poses,
            geometry_passed=result.geometry_passed,
            ground_dynamics_passed=result.ground_dynamics_passed,
            receipt=result.receipt,
            receipt_sha256="f" * 64,
        )
    assert caught.value.code == "INVALID_RESULT_RECEIPT"

    def reseal(receipt: dict[str, object]) -> str:
        receipt.pop("receipt_payload_sha256", None)
        digest = hashlib.sha256(grounded._canonical_json_bytes(receipt)).hexdigest()
        receipt["receipt_payload_sha256"] = digest
        return digest

    def rebuild(receipt: dict[str, object], digest: str) -> None:
        grounded.GroundedReadyResult(
            candidate_id=result.candidate_id,
            state=result.state,
            target_foot_poses=result.target_foot_poses,
            geometry_passed=result.geometry_passed,
            ground_dynamics_passed=result.ground_dynamics_passed,
            receipt=receipt,
            receipt_sha256=digest,
        )

    forged = grounded._jsonable(result.receipt)
    forged["gates"] = {}
    with pytest.raises(grounded.GroundedReadyError, match="gates mapping") as caught:
        rebuild(forged, reseal(forged))
    assert caught.value.code == "INVALID_RESULT_RECEIPT"

    forged = grounded._jsonable(result.receipt)
    forged["candidate"]["joint_pos"][0] += 0.125
    with pytest.raises(
        grounded.GroundedReadyError, match="joint_pos differs"
    ) as caught:
        rebuild(forged, reseal(forged))
    assert caught.value.code == "INVALID_RESULT_RECEIPT"

    forged = grounded._jsonable(result.receipt)
    forged["foot_targets"]["rows"][0]["position_w"][1] += 0.125
    with pytest.raises(
        grounded.GroundedReadyError, match="foot target 0 position"
    ) as caught:
        rebuild(forged, reseal(forged))
    assert caught.value.code == "INVALID_RESULT_RECEIPT"

    forged = grounded._jsonable(result.receipt)
    forged["exact_model"]["exact_mujoco_backend"] = True
    forged["exact_model"]["status"] = "PASS_EXACT_MUJOCO"
    forged["gates"]["exact_model_identity"] = "PASS"
    forged["verdict"] = "PASS_STATIC_GROUNDED_READY_CANDIDATE"
    forged_sha = reseal(forged)
    with pytest.raises(
        grounded.GroundedReadyError, match="live exact backend"
    ) as caught:
        rebuild(forged, forged_sha)
    assert caught.value.code == "EXACT_RESULT_REAUDIT_REQUIRED"
    with pytest.raises(
        grounded.GroundedReadyError, match="live exact backend"
    ) as caught:
        grounded.GroundedReadyResult(
            candidate_id=result.candidate_id,
            state=result.state,
            target_foot_poses=result.target_foot_poses,
            geometry_passed=result.geometry_passed,
            ground_dynamics_passed=result.ground_dynamics_passed,
            receipt=forged,
            receipt_sha256=forged_sha,
            verification_backend=backend,
            expected_model_identity=identity,
            verification_config=_config(),
        )
    assert caught.value.code == "EXACT_RESULT_REAUDIT_REQUIRED"
    with pytest.raises(
        grounded.GroundedReadyError, match="fresh exact-MuJoCo backend reload"
    ) as caught:
        grounded.GroundedReadyResult(
            candidate_id=result.candidate_id,
            state=result.state,
            target_foot_poses=result.target_foot_poses,
            geometry_passed=result.geometry_passed,
            ground_dynamics_passed=result.ground_dynamics_passed,
            receipt=forged,
            receipt_sha256=forged_sha,
            verification_backend=spoof_exact,
            expected_model_identity=identity,
            verification_config=_config(),
            construction_source=forged["source"],
        )
    assert caught.value.code == "EXACT_RESULT_FRESH_BACKEND_REQUIRED"
    assert not hasattr(grounded, "_EXACT_RESULT_ATTESTOR")

    # Even bypassing the frozen dataclass cannot cross the publication boundary:
    # exact publication requires a full construction-bound rebuild.
    object.__setattr__(result, "receipt", grounded._deep_freeze(forged))
    object.__setattr__(result, "receipt_sha256", forged_sha)
    with pytest.raises(
        grounded.GroundedReadyError, match="exact publication requires"
    ) as caught:
        grounded.publish_grounded_ready_candidate(
            result,
            tmp_path / "forged_exact",
        )
    assert caught.value.code == "EXACT_PUBLICATION_CONSTRUCTION_REQUIRED"
    assert not (tmp_path / "forged_exact").exists()
    with pytest.raises(
        grounded.GroundedReadyError, match="publication backend reload"
    ) as caught:
        grounded.publish_grounded_ready_candidate(
            result,
            tmp_path / "forged_exact_spoof_backend",
            expected_model_identity=identity,
            verification_config=_config(),
            construction_mode="G1",
            construction_donor_state=_donor(backend),
        )
    assert caught.value.code == "EXACT_PUBLICATION_FRESH_BACKEND_REQUIRED"
    assert not (tmp_path / "forged_exact_spoof_backend").exists()

    with pytest.raises(
        grounded.GroundedReadyError, match="fresh exact-MuJoCo backend reload"
    ) as caught:
        grounded.solve_g1_donor_root(
            _donor(backend),
            backend=spoof_exact,
            expected_model_identity=identity,
            config=_config(),
        )
    assert caught.value.code == "EXACT_RESULT_FRESH_BACKEND_REQUIRED"


def test_publication_rejects_symlinked_parent_component(
    identity: grounded.ExactModelIdentity,
    tmp_path: Path,
):
    backend = StrictFakeBackend(identity)
    result = grounded.solve_g1_donor_root(
        _donor(backend),
        backend=backend,
        expected_model_identity=identity,
        config=_config(),
    )
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        grounded.GroundedReadyError, match="symlink component"
    ) as caught:
        grounded.publish_grounded_ready_candidate(result, linked_parent / "candidate")
    assert caught.value.code == "INVALID_OUTPUT_PARENT"
    assert not (real_parent / "candidate").exists()


def test_exact_a3_grounded_ready_candidates_when_explicitly_configured(
    tmp_path: Path,
):
    names = {
        "model": "GROUNDED_READY_A3_MJCF",
        "ready": "GROUNDED_READY_DONOR_NPZ",
        "compiled": "GROUNDED_READY_COMPILED_SHA256",
        "path": "GROUNDED_READY_PATH_BINDING_SHA256",
        "ground": "GROUNDED_READY_GROUND_BINDING_SHA256",
    }
    values = {key: os.environ.get(env) for key, env in names.items()}
    if any(value is None for value in values.values()):
        pytest.skip("set the five GROUNDED_READY_* exact A3 pins")
    identity = grounded.ExactModelIdentity(
        mjcf_path=str(values["model"]),
        mjcf_sha256=hashlib.sha256(Path(str(values["model"])).read_bytes()).hexdigest(),
        compiled_model_sha256=str(values["compiled"]),
        path_model_binding_sha256=str(values["path"]),
        ground_model_binding_sha256=str(values["ground"]),
    )
    backend = grounded.MujocoGroundedReadyBackend.load(identity)
    with np.load(str(values["ready"]), allow_pickle=False) as data:
        donor = grounded.ReadyState(
            data["joint_pos"], data["root_pos_w"], data["root_quat_w"]
        )
    config = grounded.GroundedReadyConfig()
    g1 = grounded.solve_g1_donor_root(
        donor,
        backend=backend,
        expected_model_identity=identity,
        config=config,
    )
    assert g1.receipt["verdict"] == "PASS_STATIC_GROUNDED_READY_CANDIDATE"
    assert g1.geometry_passed
    assert g1.ground_dynamics_passed is True
    assert all(
        count >= 1
        for count in g1.receipt["static_geometry"]["double_support"][
            "foot_contact_count"
        ]
    )
    assert g1.receipt["static_geometry"]["support"]["margin_m"] > 0.0
    assert g1.receipt["static_geometry"]["leg_to_foot_jacobian"]["rank"] == 12
    published = grounded.publish_grounded_ready_candidate(
        g1,
        tmp_path / "exact_g1",
        expected_model_identity=identity,
        verification_config=config,
        construction_mode="G1",
        construction_donor_state=donor,
    )
    assert published.receipt_json.is_file()

    forged_payload = grounded._jsonable(g1.receipt)
    forged_payload["source"]["mode"] = "forged_source_mode"
    forged_payload.pop("receipt_payload_sha256")
    forged_sha = hashlib.sha256(
        grounded._canonical_json_bytes(forged_payload)
    ).hexdigest()
    forged_payload["receipt_payload_sha256"] = forged_sha
    forged_source_result = grounded.GroundedReadyResult(
        candidate_id=g1.candidate_id,
        state=g1.state,
        target_foot_poses=g1.target_foot_poses,
        geometry_passed=g1.geometry_passed,
        ground_dynamics_passed=g1.ground_dynamics_passed,
        receipt=forged_payload,
        receipt_sha256=forged_sha,
        verification_backend=backend,
        expected_model_identity=identity,
        verification_config=config,
        construction_source=forged_payload["source"],
    )
    with pytest.raises(
        grounded.GroundedReadyError, match="construction-bound exact rebuild"
    ) as caught:
        grounded.publish_grounded_ready_candidate(
            forged_source_result,
            tmp_path / "exact_g1_forged_source",
            expected_model_identity=identity,
            verification_config=config,
            construction_mode="G1",
            construction_donor_state=donor,
        )
    assert caught.value.code == "EXACT_PUBLICATION_CONSTRUCTION_MISMATCH"
    assert not (tmp_path / "exact_g1_forged_source").exists()

    wrong_q = donor.joint_pos.copy()
    wrong_q[grounded.RUNTIME_JOINT_NAMES.index("waist_yaw_joint")] += 0.025
    wrong_donor = grounded.ReadyState(
        wrong_q,
        donor.root_pos_w,
        donor.root_quat_wxyz,
    )
    with pytest.raises(grounded.GroundedReadyError):
        grounded.publish_grounded_ready_candidate(
            g1,
            tmp_path / "exact_g1_wrong_donor",
            expected_model_identity=identity,
            verification_config=config,
            construction_mode="G1",
            construction_donor_state=wrong_donor,
        )
    assert not (tmp_path / "exact_g1_wrong_donor").exists()

    wrong_upper = donor.joint_pos.copy()
    wrong_upper[
        grounded.RUNTIME_JOINT_NAMES.index("right_shoulder_pitch_joint")
    ] += 0.05
    with pytest.raises(grounded.GroundedReadyError):
        grounded.publish_grounded_ready_candidate(
            g1,
            tmp_path / "exact_g1_wrong_overlay",
            expected_model_identity=identity,
            verification_config=config,
            construction_mode="G1",
            construction_donor_state=donor,
            construction_upper_candidate_joint_pos=wrong_upper,
            construction_upper_joint_names=("right_shoulder_pitch_joint",),
        )
    assert not (tmp_path / "exact_g1_wrong_overlay").exists()

    g2 = grounded.build_g2_vendor_key_candidate(
        donor,
        backend=backend,
        expected_model_identity=identity,
        vendor_key_index=0,
        config=config,
    )
    assert not g2.geometry_passed
    assert g2.receipt["gates"]["support_margin"] == "FAIL_CLOSED"
