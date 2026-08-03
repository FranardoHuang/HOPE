"""Pure-CPU gates for the native MuJoCo selected-rubber classifier."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import selected_rubber_classifier as classifier  # noqa: E402
from mujoco_native import n1_ball_core  # noqa: E402


MANIFEST = (
    REPO_ROOT
    / "configs/action_ball_n1_measured_20260803/"
    "fresh_core_seed0_20260803_take061_robust20n_r8_splitready/"
    "take_061_unit04_bh.full.manifest.v3.7d2139028427.json"
)
IMMUTABLE_TAPE = (
    REPO_ROOT
    / "configs/action_ball_n1_measured_20260803/"
    "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready/"
    "immutable_n1_tape.v1.22052606032f.json"
)


def _binding():
    geometry = classifier.verify_urdf_mjcf_geometry()
    scene = {
        "binding_sha256": "a" * 64,
        "canonical_mjcf_sha256": geometry["canonical_mjcf_sha256"],
        "assembled_xml_sha256": "b" * 64,
        "compiled_runtime": {
            "mujoco_version": "unit-test-backend",
            "mesh_source_closure_sha256": {
                "meshes/pingpang_red_Link.STL": geometry[
                    "red_rubber_mesh_sha256"
                ],
                "meshes/pingpang_black_Link.STL": geometry[
                    "black_rubber_mesh_sha256"
                ],
                "meshes/collision_optimized/right_racket_face_collision.STL": (
                    geometry["generic_blade_mesh_sha256"]
                ),
            },
        },
    }
    return classifier.build_classifier_binding(
        scene_binding=scene, mjcf_path=classifier.DEFAULT_MJCF
    )


def _lineage(binding):
    return classifier.bind_action_manifest(
        manifest_path=MANIFEST,
        expected_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        action_uid=5527597793770800,
        motion_sha256=(
            "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
        ),
        mount_normal_sign=1,
        geometry_source_sha256=(
            "2451e2fa1c29036d650d5ff4a1630a0d41c7ccb5730400270a2c69a6905ce29e"
        ),
        physics_sha256=(
            "aa5c9085f9b48ca65b3a0ee2cbb35588a5e85a08e84dc3f2ce552d3ef4af85b7"
        ),
        classifier_binding=binding,
    )


def _classify(local_ball, binding, lineage):
    return classifier.classify_observed_generic_blade_contact(
        ball_center_w_m=local_ball,
        racket_site_position_w_m=(0.0, 0.0, 0.0),
        racket_rotation_w_from_local=np.eye(3),
        action_lineage=lineage,
        classifier_binding=binding,
        policy_tick=4,
        physics_substep=2,
    )


def test_geometry_and_action_lineage_are_exactly_source_bound():
    binding = _binding()
    geometry = binding["geometry"]
    assert geometry["canonical_mjcf_sha256"] == (
        "70c4fd6534f259d12990cef731cfdf8f8557f92fd0ca81cc4fc1c75a39336c0a"
    )
    assert abs(geometry["red_outer_y_from_site_m"]) < 2.0e-9
    assert geometry["black_outer_y_from_site_m"] < 0.0
    lineage = _lineage(binding)
    assert lineage["action_id"] == "take_061_unit04_bh"
    assert lineage["action_uid"] == 5527597793770800
    assert lineage["mount_normal_sign"] == 1
    assert classifier.validate_action_lineage(
        lineage, classifier_binding=binding
    ) == lineage


def test_selected_and_opposite_faces_use_measured_mount_sign():
    binding = _binding()
    lineage = _lineage(binding)
    center_x, center_z = binding["geometry"][
        "face_area_center_xz_from_site_m"
    ]
    red = _classify((center_x, 0.020, center_z), binding, lineage)
    black = _classify((center_x, -0.033208, center_z), binding, lineage)
    assert red["status"] == classifier.STATUS_SELECTED
    assert red["observed_face_sign"] == 1
    assert red["selected_rubber"] is True
    assert black["status"] == classifier.STATUS_OPPOSITE
    assert black["observed_face_sign"] == -1
    assert black["selected_rubber"] is False
    assert classifier.validate_classification_seal(
        red, action_lineage=lineage
    ) == red


def test_edge_rim_and_between_planes_fail_ambiguous():
    binding = _binding()
    lineage = _lineage(binding)
    center_x, center_z = binding["geometry"][
        "face_area_center_xz_from_site_m"
    ]
    safe_radius = binding["geometry"][
        "safe_ball_center_tangential_radius_m"
    ]
    edge = _classify(
        (center_x + safe_radius, 0.020, center_z), binding, lineage
    )
    between = _classify((center_x, -0.006, center_z), binding, lineage)
    assert edge["status"] == classifier.STATUS_EDGE_RIM_AMBIGUOUS
    assert edge["selected_rubber"] is None
    assert between["status"] == classifier.STATUS_BETWEEN_PLANES_AMBIGUOUS
    assert between["selected_rubber"] is None


def test_immutable_question_binds_action_mount_scene_and_backend(tmp_path):
    binding = _binding()
    payload = n1_ball_core.build_question_from_immutable_tape(
        immutable_tape_path=IMMUTABLE_TAPE,
        expected_immutable_tape_sha256=hashlib.sha256(
            IMMUTABLE_TAPE.read_bytes()
        ).hexdigest(),
        target_recipe="outcome_dense_only",
        action_manifest_path=MANIFEST,
        selected_rubber_classifier_binding=binding,
        scene_binding_sha256=binding["scene_binding_sha256"],
        physical_launch_position_w_m=(2.3, 0.0, 1.5),
        physical_launch_velocity_w_mps=(-1.0, 0.0, 0.0),
    )
    lineage = payload["authority"]["selected_rubber_action_lineage"]
    assert lineage["action_id"] == "take_061_unit04_bh"
    assert lineage["mount_normal_sign"] == 1
    assert lineage["scene_binding_sha256"] == "a" * 64
    assert lineage["mujoco_backend_version"] == "unit-test-backend"

    question_path = tmp_path / "question.json"
    question_sha = n1_ball_core.write_question(question_path, payload)
    loaded = n1_ball_core.load_question(
        question_path,
        expected_file_sha256=question_sha,
        scene_binding_sha256="a" * 64,
        selected_rubber_classifier_binding=binding,
    )
    assert loaded.selected_rubber_action_lineage == lineage
