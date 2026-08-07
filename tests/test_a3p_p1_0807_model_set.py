"""Contract checks for the 2026-08-07 A3P-P1 dual-engine model set.

Two producers are under test: the raw-bundle intake and the Isaac+MuJoCo model-set builder.  The
tests that need the private vendor bytes skip on a fresh clone; everything that can be asserted
from tracked receipts and tracked source is asserted unconditionally.
"""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INTAKE = _load("intake_a3p_p1_0807_bundle")
MODELSET = _load("prepare_a3p_p1_0807_model_set")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------
# intake
# ----------------------------------------------------------------------------------------


def test_intake_records_that_the_bundle_is_assembled_not_a_vendor_closure():
    receipt = _json(INTAKE.DEFAULT_RECEIPT)
    assert receipt["manifest_type"] == "a3p_p1_0807_op3_pingpang_raw_intake_v1"
    # The single most misreadable thing about this delivery: it is not one vendor package.
    assert receipt["assembled_not_a_vendor_closure"] is True
    provenance = receipt["mesh_provenance"]
    assert provenance["resolved_count"] == INTAKE.EXPECTED_RESOLVABLE_MESH_COUNT
    assert provenance["distinct_reference_count"] == INTAKE.EXPECTED_MESH_REFERENCE_COUNT
    # Every mesh byte comes from the 0803 delivery; the OmniPicker3 package adds nothing.
    assert provenance["source_package_histogram"] == {
        "A3-P1-32dof-0803-BerkeleyPingpang-90deg": INTAKE.EXPECTED_RESOLVABLE_MESH_COUNT
    }
    assert receipt["omnipicker3_cross_check"]["adds_new_geometry"] is False
    assert receipt["omnipicker3_cross_check"]["all_byte_identical_to_bundled_meshes"] is True
    assert receipt["omnipicker3_cross_check"]["mesh_count"] == 20
    assert list(provenance["predecessor_meshes_unused_by_0807"]) == list(
        INTAKE.EXPECTED_UNUSED_PREDECESSOR_MESHES
    )


def test_intake_pins_what_0807_fixed_and_what_it_did_not():
    receipt = _json(INTAKE.DEFAULT_RECEIPT)
    fixed = receipt["fixed_since_predecessor"]
    assert fixed["right_elbow_joint_origin_x"]["predecessor"] == "0.001 0 -0.1325"
    assert fixed["right_elbow_joint_origin_x"]["delivered"] == "0.01 0 -0.1325"
    assert fixed["illegal_axis_on_fixed_joints"] == {"predecessor_count": 5, "delivered_count": 0}
    assert fixed["ankle_pitch_lateral_asymmetry"]["delivered_y"] == "0"
    structure = receipt["structure"]
    assert structure["fixed_joints_with_illegal_axis"] == []
    # Still broken, and the receipt must keep saying so rather than quietly normalising it away.
    assert structure["duplicate_link_names"] == ["imu_in_pelvis_link"]
    assert structure["nonfinite_rgba_values"] == ["nan nan nan nan"]
    assert structure["mimic_joint_count"] == 0
    assert structure["transmission_count"] == 0
    assert structure["dynamics_element_count"] == 0
    assert structure["movable_joint_count"] == 40
    assert structure["body_movable_joint_count"] == 31
    assert structure["gripper_movable_joint_count"] == 9
    for defect in INTAKE.EXPECTED_STILL_PRESENT_DEFECTS:
        assert defect in receipt["still_present_defects"]


def test_intake_records_the_gripper_confirmation_as_verbal_and_unwritten():
    receipt = _json(INTAKE.DEFAULT_RECEIPT)
    unresolved = receipt["unresolved_gripper_collision_references"]
    assert unresolved["count"] == INTAKE.EXPECTED_UNRESOLVED_GRIPPER_COLLISION_COUNT
    assert unresolved["materialised_in_this_bundle"] is False
    confirmation = unresolved["vendor_confirmation"]
    assert confirmation["written_evidence_on_file"] is False
    assert confirmation["channel"] == "relayed_by_project_owner_from_vendor"
    assert "coupling" in confirmation["still_unconfirmed"]


def test_intake_refuses_to_overwrite_an_existing_bundle():
    if not INTAKE.DEFAULT_BUNDLE_ROOT.is_dir():
        pytest.skip("private raw bundle is intentionally absent from a fresh clone")
    with pytest.raises(INTAKE.IntakeError, match="refusing to overwrite"):
        INTAKE.build(
            INTAKE.DEFAULT_BUNDLE_ROOT,
            INTAKE.DEFAULT_RECEIPT,
            INTAKE.DEFAULT_BUNDLE_ROOT / "urdf" / INTAKE.DELIVERED_URDF_NAME,
            INTAKE.DEFAULT_BUNDLE_ROOT / "vendor_packages" / INTAKE.DELIVERED_OP3_NAME,
            INTAKE.PREDECESSOR_ROOT,
        )


# ----------------------------------------------------------------------------------------
# model set: shared
# ----------------------------------------------------------------------------------------


def test_model_set_refuses_to_write_into_either_in_service_artifact():
    with pytest.raises(MODELSET.ModelSetError, match="current Isaac runtime asset"):
        MODELSET.require_isolated_outputs(MODELSET.ACTIVE_ISAAC_ASSET_ROOT, MODELSET.MJCF_OUTPUT_DIR)
    with pytest.raises(MODELSET.ModelSetError, match="in-service MuJoCo model directory"):
        MODELSET.require_isolated_outputs(MODELSET.ISAAC_ASSET_ROOT, MODELSET.INCUMBENT_MJCF_DIR)
    MODELSET.require_isolated_outputs(MODELSET.ISAAC_ASSET_ROOT, MODELSET.MJCF_OUTPUT_DIR)


def test_model_set_receipt_never_claims_verification_it_did_not_perform():
    receipt = _json(MODELSET.MODEL_SET_RECEIPT)
    auth = receipt["authorization"]
    # This host has neither Isaac Lab nor mujoco. Everything that would need them stays false.
    for key in (
        "isaac_import_verified",
        "mujoco_compile_verified",
        "mujoco_warp_load_verified",
        "mujoco_identity_v3_minted",
        "cross_engine_parity_verified",
        "motion_bank_revalidated",
        "training_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        assert auth[key] is False, key
    assert auth["current_isaac_runtime_pointer_changed"] is False
    assert auth["in_service_mjcf_edited"] is False
    assert receipt["isaac"]["verification_boundary"]["imported_with_isaac_lab"] is False
    assert receipt["mujoco"]["derivation"]["verification_boundary"]["compiled_with_mujoco"] is False
    assert receipt["cross_engine"]["parity_verified"] is False


def test_in_service_mjcf_is_still_the_pinned_bytes():
    assert MODELSET.sha256_path(MODELSET.INCUMBENT_MJCF) == MODELSET.INCUMBENT_MJCF_SHA256


# ----------------------------------------------------------------------------------------
# model set: Isaac
# ----------------------------------------------------------------------------------------


def test_isaac_normalization_preserves_the_31_action_and_32_body_abi():
    receipt = _json(MODELSET.MODEL_SET_RECEIPT)
    normalization = receipt["isaac"]["normalization"]
    assert normalization["movable_joint_count"] == MODELSET.EXPECTED_MOVABLE_JOINTS
    assert normalization["movable_joint_document_order"] == MODELSET.read_order(MODELSET.GMR_JOINT_ORDER)
    assert set(normalization["movable_joint_document_order"]) == set(
        MODELSET.read_order(MODELSET.RUNTIME_JOINT_ORDER)
    )
    assert normalization["link_count"] == 63
    assert normalization["unique_link_total_mass_kg"] == pytest.approx(
        MODELSET.EXPECTED_NORMALIZED_UNIQUE_LINK_MASS_KG, abs=1e-9
    )
    lock = normalization["project_locked_gripper_joints"]
    assert len(lock["names"]) == MODELSET.EXPECTED_GRIPPER_MOVABLE_JOINTS
    assert lock["lock_contract"] == MODELSET.PROJECT_GRIPPER_LOCK_CONTRACT
    # The delivered placeholder limits are archived, not silently discarded.
    for name, original in lock["delivered_originals"].items():
        assert original["type"] in {"revolute", "prismatic"}
        assert "lower" in original["limit"] and "upper" in original["limit"]


def test_gripper_collisions_resolve_to_their_visual_under_a_recorded_authority():
    receipt = _json(MODELSET.MODEL_SET_RECEIPT)
    normalization = receipt["isaac"]["normalization"]
    contract = normalization["gripper_collision_contract"]
    assert contract == MODELSET.GRIPPER_COLLISION_EQUALS_VISUAL_CONTRACT
    assert contract["vendor_written_confirmation_on_file"] is False
    assert contract["supersedes"] == "the 0803 collision-disabled gripper contract"

    # Nothing is written as a second copy: identical geometry is stored once and referenced twice.
    assert normalization["materialised_gripper_collision_meshes"] == []
    rows = normalization["collision_references_resolved_to_visual"]
    gripper = [r for r in rows if r["kind"] == "gripper_collision_authorised_as_visual"]
    delivered = [r for r in rows if r["kind"] == "delivered_collision_equal_to_visual"]
    assert normalization["gripper_collision_reference_count"] == (
        MODELSET.EXPECTED_MATERIALISED_COLLISION_COUNT
    )
    assert len(gripper) == MODELSET.EXPECTED_MATERIALISED_COLLISION_COUNT
    assert delivered, "the delivery's own byte-identical collision copies must also be deduplicated"
    for row in rows:
        assert row["byte_identical"] is True
        # The target must be the POST-alias on-disk name or the receipt cannot be checked.
        assert "-" not in row["points_at"]
        assert row["points_at"] == row["collision_reference"].replace("_collision", "").replace(
            "-", "_"
        )


def test_deduplication_actually_shrinks_what_git_carries():
    receipt = _json(MODELSET.MODEL_SET_RECEIPT)
    closure = receipt["isaac"]["closure"]
    # 124 references collapse to 62 unique meshes + 1 URDF.
    assert closure["file_count"] == 63
    names = [item["path"] for item in closure["files"]]
    assert sum(1 for n in names if n.endswith("_collision.stl")) == 0
    digests = [item["sha256"] for item in closure["files"]]
    assert len(digests) == len(set(digests)), "a deduplicated closure must carry no repeated bytes"


def test_isaac_asset_on_disk_resolves_and_its_collisions_really_equal_their_visuals():
    if not MODELSET.ISAAC_ASSET_ROOT.is_dir():
        pytest.skip("private generated Isaac asset is intentionally absent from a fresh clone")
    root = ET.parse(MODELSET.ISAAC_ASSET_ROOT / "urdf" / "model.urdf").getroot()
    names = [link.get("name") for link in root.findall("link")]
    assert len(names) == len(set(names))
    refs = {Path(mesh.get("filename")).name for mesh in root.iter("mesh")}
    on_disk = {p.name for p in (MODELSET.ISAAC_ASSET_ROOT / "meshes").iterdir() if p.is_file()}
    assert not refs - on_disk
    assert not any("-" in name for name in on_disk)
    assert not [
        element.get("rgba")
        for element in root.findall(".//*[@rgba]")
        if "nan" in (element.get("rgba") or "").lower()
    ]
    for row in _json(MODELSET.MODEL_SET_RECEIPT)["isaac"]["normalization"][
        "materialised_gripper_collision_meshes"
    ]:
        collision = MODELSET.ISAAC_ASSET_ROOT / "meshes" / row["collision_basename"]
        visual = MODELSET.ISAAC_ASSET_ROOT / "meshes" / row["visual_basename"]
        assert MODELSET.sha256_path(collision) == MODELSET.sha256_path(visual)


# ----------------------------------------------------------------------------------------
# model set: MuJoCo
# ----------------------------------------------------------------------------------------


def test_mjcf_derivation_changes_only_geometry_and_leaves_the_tuning_alone():
    derivation = _json(MODELSET.MODEL_SET_RECEIPT)["mujoco"]["derivation"]
    assert derivation["derived_from"]["sha256"] == MODELSET.INCUMBENT_MJCF_SHA256
    assert derivation["derived_from"]["edited_in_place"] is False
    assert derivation["body_count"] == MODELSET.EXPECTED_MJCF_BODY_COUNT
    assert derivation["joint_count"] == MODELSET.EXPECTED_MOVABLE_JOINTS
    assert derivation["actuator_count"] == MODELSET.EXPECTED_MJCF_ACTUATOR_COUNT
    assert derivation["keyframe_qpos_width"] == MODELSET.EXPECTED_MJCF_KEYFRAME_QPOS_WIDTH

    real = [r for r in derivation["body_pos_updates"] if r["classification"] == "geometry_change"]
    rounding = [r for r in derivation["body_pos_updates"] if r["classification"] == "coordinate_rounding"]
    assert {r["body"] for r in real} == {
        "left_ankle_pitch_Link",
        "right_ankle_pitch_Link",
        "right_hip_roll_Link",
        "right_elbow_Link",
    }
    assert len(rounding) == 4
    assert max(r["delta_m"] for r in rounding) < 1.0e-5

    updated = {r["body"] for r in derivation["inertial_updates"]}
    assert updated == {
        "pelvis_link",
        "torso_Link",
        "left_shoulder_roll_Link",
        "right_shoulder_roll_Link",
        "left_elbow_Link",
        "right_elbow_Link",
        "left_wrist_yaw_Link",
    }
    for row in derivation["inertial_updates"]:
        assert row["eigendecomposition_residual"] < 1e-9


def test_mjcf_folds_the_gripper_into_the_wrist_instead_of_adding_bodies():
    derivation = _json(MODELSET.MODEL_SET_RECEIPT)["mujoco"]["derivation"]
    wrist = next(r for r in derivation["inertial_updates"] if r["body"] == "left_wrist_yaw_Link")
    # 0.080677560 wrist + 0.766262094 OP3 subtree; the incumbent still folded the 0.2 kg left hand.
    assert wrist["delivered_mass_kg"] == pytest.approx(0.846939654, abs=1e-9)
    assert wrist["incumbent_mass_kg"] == pytest.approx(0.280678, abs=1e-6)
    assert wrist["merged_link_count"] == 22
    geoms = derivation["gripper_geoms_on_host_body"]
    assert geoms["host_body"] == "left_wrist_yaw_Link"
    assert geoms["count"] == MODELSET.EXPECTED_MATERIALISED_COLLISION_COUNT
    assert all(row["collision_is_visual_hull"] for row in geoms["geoms"])
    assert {row["name"] for row in derivation["retired_left_hand_assets"]} == {
        "left_hand_Link",
        "collision_left_hand_Link",
    }
    # The pelvis IMU fold is a deliberate parity fix and must stay declared, not silent.
    assert "pelvis_imu_fold" in derivation["declared_deviations"]
    pelvis = next(r for r in derivation["inertial_updates"] if r["body"] == "pelvis_link")
    assert pelvis["delta_mass_kg"] == pytest.approx(0.02, abs=1e-9)


def test_mjcf_on_disk_keeps_the_racket_contract_and_closes_its_mesh_set():
    if not MODELSET.MJCF_OUTPUT_DIR.is_dir():
        pytest.skip("private generated MJCF is intentionally absent from a fresh clone")
    root = ET.parse(MODELSET.MJCF_OUTPUT_DIR / MODELSET.MJCF_OUTPUT_NAME).getroot()
    names = {element.get("name") for element in root.iter() if element.get("name")}
    for required in MODELSET.REQUIRED_MJCF_RACKET_NAMES:
        assert required in names
    site = root.find(".//site[@name='right_racket']")
    assert tuple(float(v) for v in site.get("pos").split()) == MODELSET.OFFICIAL_RACKET_SITE_XYZ_M
    assert not [name for name in names if name and "left_hand" in name]
    declared = {mesh.get("file") for mesh in root.findall("asset/mesh")}
    mesh_dir = MODELSET.MJCF_OUTPUT_DIR / "meshes"
    on_disk = {p.relative_to(mesh_dir).as_posix() for p in mesh_dir.rglob("*") if p.is_file()}
    assert declared == on_disk
    assert len(list(root.iter("body"))) == MODELSET.EXPECTED_MJCF_BODY_COUNT
    assert len(root.findall("actuator/motor")) == MODELSET.EXPECTED_MJCF_ACTUATOR_COUNT
    assert len(root.findall(".//freejoint")) == 1
    # Repo-owned tuning must have survived the port verbatim.
    joints = [j for j in root.iter("joint") if j.get("name")]
    assert len(joints) == MODELSET.EXPECTED_MOVABLE_JOINTS
    assert all(j.get("armature") and j.get("damping") and j.get("frictionloss") for j in joints)
    assert any(
        mesh.get("file", "").endswith("right_racket_face_collision.STL")
        for mesh in root.findall("asset/mesh")
    )


def test_model_set_check_reproduces_both_engines():
    for path in (MODELSET.BUNDLE_ROOT, MODELSET.ISAAC_ASSET_ROOT, MODELSET.MJCF_OUTPUT_DIR):
        if not path.is_dir():
            pytest.skip("private A3P-P1 artifacts are intentionally absent from a fresh clone")
    report = MODELSET.check(
        MODELSET.BUNDLE_ROOT,
        MODELSET.ISAAC_ASSET_ROOT,
        MODELSET.MJCF_OUTPUT_DIR,
        MODELSET.MODEL_SET_RECEIPT,
    )
    assert report["status"] == "PASS"
    assert report["isaac_movable_joint_count"] == 31
    assert report["mujoco_body_count"] == 32
    assert report["mujoco_compile_verified"] is False
