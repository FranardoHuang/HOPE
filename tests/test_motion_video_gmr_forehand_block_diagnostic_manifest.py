from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = (
    ROOT / "configs" / "motion_video_gmr_forehand_block_diagnostic_20260711.json"
)
GMR_RESULTS_PATH = ROOT / "configs" / "motion_video_gmr_results_20260711.json"
JOINT_ORDER_PATH = ROOT / "hope_training" / "config" / "joint_order_agibot_a3.yaml"
MJCF_PATH = (
    ROOT
    / "agi"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
    / "a3_pingpong.xml"
)
URDF_PATH = (
    ROOT
    / "agi"
    / "URDF"
    / "A3T2.5-URDF-std-pingpang"
    / "urdf"
    / "URDF-JOINT-LINK.urdf"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_joint_order() -> list[str]:
    return [
        line.strip()[2:]
        for line in JOINT_ORDER_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    ]


def test_forehand_block_diagnostic_schema_and_lineage_are_cross_bound():
    diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    gmr_results = json.loads(GMR_RESULTS_PATH.read_text(encoding="utf-8"))
    accepted = next(
        row for row in gmr_results["results"] if row["asset_id"] == diagnostic["asset_id"]
    )
    lineage = diagnostic["lineage"]
    output = lineage["accepted_gmr_queue_output"]

    assert diagnostic["schema_version"] == 1
    assert diagnostic["status"] == "complete_diagnostic_failure"
    assert diagnostic["verdict"] == "fail_ground_before_schema2"
    assert diagnostic["formal_eligible"] is False
    assert lineage["source_gvhmr"]["sha256"] == accepted["source_sha256"]
    assert output["sha256"] == accepted["output_sha256"]
    assert output["binding_sha256"] == accepted["binding_sha256"]
    assert output["audit_sha256"] == accepted["structural_audit_sha256"]
    assert output["run_log_sha256"] == accepted["run_log_sha256"]
    assert lineage["gmr"]["commit"] == gmr_results["processing_contract"]["gmr_commit"]
    for value in (
        lineage["source_gvhmr"]["sha256"],
        output["sha256"],
        output["binding_sha256"],
        output["audit_sha256"],
        output["run_log_sha256"],
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", value)

    assert lineage["canonical_mjcf"]["sha256"] == _sha256(MJCF_PATH)
    assert lineage["urdf_limit_contract"]["sha256"] == _sha256(URDF_PATH)
    joint_contract = diagnostic["joint_order_contract"]
    assert joint_contract["tracked_sha256"] == _sha256(JOINT_ORDER_PATH)
    assert joint_contract["status"] == "exact_match"
    assert joint_contract["count"] == 31
    assert joint_contract["names"] == _tracked_joint_order()


def test_forehand_block_diagnostic_preserves_measured_failure_and_limits():
    diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    structural = diagnostic["structural_audit"]
    kinematic = diagnostic["kinematic_audit"]
    collision = diagnostic["mujoco_collision_audit"]

    assert structural["frames"] == 65
    assert structural["dof"] == 31
    assert structural["fps"] == 30.0
    assert structural["finite"] is True
    assert structural["root_rotation_max_norm_error"] == 9.992007221626409e-16
    assert structural["warmup"]["rounds"] == 22
    assert structural["warmup"]["final_max_dq_rad"] == 7.81e-05

    position = kinematic["position_limits"]
    velocity = kinematic["velocity_limits"]
    assert position["violation_elements"] == 0
    assert position["minimum_signed_margin"]["exact_recomputed_rad"] == (
        0.040620150344593875
    )
    assert position["minimum_signed_margin"]["approximate_prior_summary_rad"] == 0.0406203
    assert position["minimum_signed_margin"]["frame"] == 26
    assert position["minimum_signed_margin"]["joint"] == "right_ankle_roll_joint"
    assert velocity["violation_elements"] == 0
    assert velocity["maximum"]["exact_rad_per_s"] == 8.451850092996589
    assert velocity["maximum"]["urdf_limit_exact_rad_per_s"] == 15.707963267948966
    assert velocity["maximum"]["exact_limit_ratio"] == 0.5380614882288409
    assert kinematic["acceleration"]["maximum_exact_rad_per_s2"] == 89.77406718270811
    assert kinematic["acceleration"]["limit_contract_available"] is False
    assert kinematic["endpoint"]["strict_static_ready"] is False

    self_collision = collision["self_collision"]
    floor = collision["floor_penetration"]
    assert self_collision["sampled_poses"] == 641
    assert self_collision["robot_self_contact_events"] == 0
    assert self_collision["continuous_collision_proven"] is False
    assert collision["scene_coverage"]["table_present"] is False
    assert collision["scene_coverage"]["net_present"] is False
    assert floor["status"] == "fail"
    assert floor["frames_with_penetration"] == floor["total_frames"] == 65
    assert floor["per_frame_deepest_exact_range_m"] == {
        "minimum": -0.08414975109946404,
        "maximum": -0.07727632254752706,
    }
    assert diagnostic["decision"]["verdict"] == "fail_ground_before_schema2"
    assert diagnostic["decision"]["schema2_conversion_allowed_now"] is False
    assert diagnostic["decision"]["training_asset_allowed_now"] is False
    assert diagnostic["decision"]["real_robot_execution_allowed"] is False
