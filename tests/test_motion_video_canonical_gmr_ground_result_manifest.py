from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "configs" / "motion_video_canonical_gmr_ground_results_20260711.json"
PLAN_PATH = ROOT / "configs" / "motion_video_canonical_gmr_ground_prereg_v2_20260711.json"
SOURCE_PATH = ROOT / "configs" / "motion_video_canonical_gmr_results_20260711.json"
QUEUE_PATH = ROOT / "scripts" / "run_motion_video_canonical_gmr_ground_queue.py"
GROUND_PATH = ROOT / "scripts" / "ground_gmr_pkl.py"
MJCF_PATH = (
    ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_result_binds_prereg_source_tools_mjcf_collision_and_canonical_lineage():
    result = _load(RESULT_PATH)
    plan = _load(PLAN_PATH)
    source = _load(SOURCE_PATH)

    assert result["status"] == "complete_diagnostic_canonical_grounding"
    assert result["body_shape_contract"] == plan["body_shape_contract"]
    assert result["preregistration"]["sha256"] == _sha(PLAN_PATH)
    assert result["preregistration"]["bytes"] == PLAN_PATH.stat().st_size
    assert result["source_lineage"]["canonical_gmr_result"]["sha256"] == _sha(SOURCE_PATH)
    assert result["source_lineage"]["canonical_gmr_result"]["bytes"] == SOURCE_PATH.stat().st_size
    assert result["source_lineage"]["canonical_vector_sha256"] == (
        "a03f1642151453316f0c99f81a743a604e29c656c9fffd4bac89353f7c4d9cc6"
    )
    assert result["source_lineage"]["a3_calibrated"] is False
    assert result["source_lineage"]["measured_height_m"] is None

    processing = result["processing_contract"]
    assert processing["queue_tool"]["sha256"] == _sha(QUEUE_PATH)
    assert processing["queue_tool"]["bytes"] == QUEUE_PATH.stat().st_size
    assert processing["grounding_tool"]["sha256"] == _sha(GROUND_PATH)
    assert processing["grounding_tool"]["bytes"] == GROUND_PATH.stat().st_size
    assert processing["mjcf"]["sha256"] == _sha(MJCF_PATH)
    assert processing["mjcf"]["bytes"] == MJCF_PATH.stat().st_size
    collision = processing["compiled_collision_contract"]
    assert collision == plan["compiled_collision_contract"]
    assert collision["expected_sha256"] == (
        "18e7f6ffbefba9dbd988f7c3cb9fb92b250777862fc25fa3d4a0b2ca0f8386e5"
    )
    assert len(collision["enabled_robot_geom_ids"]) == collision["enabled_robot_geom_count"] == 37
    assert processing["joint_contract"]["count"] == 31

    assert source["body_shape_contract"] == result["body_shape_contract"]
    assert source["formal_eligible"] is False


def test_failed_v1_is_preserved_and_not_mixed_into_accepted_results():
    result = _load(RESULT_PATH)
    failed = result["predecessor_attempt"]
    assert failed["status"] == "failed_pre_output_preserved"
    assert failed["failure_class"] == "launcher_environment_binding_bug"
    assert failed["grounding_contract_changed"] is False
    assert failed["accepted_outputs"] == failed["output_root_file_count"] == 0
    assert failed["preserved_evidence_physical_sha_and_bytes_recomputed"] is True
    assert failed["accepted_as_grounding_evidence"] is False
    assert set(failed["preserved_evidence"]) == {
        "launcher_log",
        "queue_state",
        "failed_binding",
        "failed_child_log",
    }
    for binding in failed["preserved_evidence"].values():
        assert binding["bytes"] > 0 and SHA256.fullmatch(binding["sha256"])


def test_all_ten_results_bind_inputs_outputs_reports_and_exact_root_z_only_invariants():
    result = _load(RESULT_PATH)
    plan = _load(PLAN_PATH)
    rows = result["results"]
    plan_by_id = {row["asset_id"]: row for row in plan["inputs"]}

    assert [row["asset_id"] for row in rows] == plan["processing_order"]
    assert len(rows) == len(plan_by_id) == 10
    for row in rows:
        assert row["status"] == "complete_diagnostic_canonical_grounding"
        assert row["input"] == plan_by_id[row["asset_id"]]["input"]
        assert row["structure"]["frames"] == plan_by_id[row["asset_id"]]["frames"]
        assert row["structure"]["fps"] == 30.0
        assert row["structure"]["finite"] is True
        assert row["structure"]["finite_elements"] > 0
        for field in ("output", "report", "state_binding", "log"):
            binding = row[field]
            assert binding["bytes"] > 0
            assert SHA256.fullmatch(binding["sha256"])
            assert binding["path"].startswith("/workspace/codexschema/motion_video_intake_20260711/")
        invariants = row["invariants"]
        for field in (
            "payload_keys_exact",
            "all_non_root_pos_fields_exact",
            "root_xy_exact",
            "root_rotation_exact",
            "dof_position_exact",
            "root_z_only_one_constant_shift",
        ):
            assert invariants[field] is True
        assert invariants["root_z_relative_trajectory_max_error_m"] == 0.0
        assert invariants["fatal_log_token_scan"] == "pass"
        assert row["grounding"]["shift"]["spread_m"] == 0.0
        assert row["joint_contract"]["max_range_excess_rad"] == 0.0
        after_minimum = row["grounding"]["after"]["minimum_clearance_m"]
        assert 1e-5 - 5e-7 <= after_minimum <= 1e-3 + 5e-7
        assert row["formal_eligible"] is False


def test_aggregate_ranges_and_cpu_only_checkout_gpu_observations_are_consistent():
    result = _load(RESULT_PATH)
    rows = result["results"]
    aggregate = result["aggregate"]
    assert aggregate["before_minimum_clearance_range_m"] == [
        min(row["grounding"]["before"]["minimum_clearance_m"] for row in rows),
        max(row["grounding"]["before"]["minimum_clearance_m"] for row in rows),
    ]
    assert aggregate["applied_constant_root_z_shift_range_m"] == [
        min(row["grounding"]["shift"]["applied_constant_m"] for row in rows),
        max(row["grounding"]["shift"]["applied_constant_m"] for row in rows),
    ]
    assert aggregate["after_minimum_clearance_range_m"] == [
        min(row["grounding"]["after"]["minimum_clearance_m"] for row in rows),
        max(row["grounding"]["after"]["minimum_clearance_m"] for row in rows),
    ]
    audit = result["independent_postrun_audit"]
    assert audit["accepted_results"] == 10 and audit["failed_results"] == 0
    assert audit["output_files_found"] == 20
    assert audit["bindings_found"] == audit["logs_found"] == 10
    assert audit["max_applied_shift_spread_m"] == 0.0
    assert audit["max_root_z_relative_trajectory_error_m"] == 0.0
    assert audit["max_joint_range_excess_rad"] == 0.0

    runtime = result["runtime"]
    assert runtime["cpu_only"] is True and runtime["CUDA_VISIBLE_DEVICES"] == ""
    assert runtime["queue_state"]["status"] == "complete"
    assert all(item["clean_before"] and item["clean_after"] for item in runtime["read_only_checkouts_before_and_after"])
    gpu = runtime["gpu_observation"]
    assert gpu["queue_environment_precluded_cuda"] is True
    assert gpu["queue_or_ground_processes_after"] == 0
    assert gpu["memory_used_unchanged_mib"] is True
    assert [item["memory_used_mib"] for item in gpu["before"]] == [
        item["memory_used_mib"] for item in gpu["after"]
    ]


def test_result_remains_diagnostic_and_does_not_authorize_training_or_robot():
    result = _load(RESULT_PATH)
    assert result["formal_eligible"] is False
    assert result["remaining_gates"] == {
        "inter_frame_continuous_ground_clearance": "not_performed",
        "self_collision": "not_performed",
        "racket_and_handle_to_body_swept_clearance": "not_performed",
        "dynamics_and_balance_feasibility": "not_performed",
        "table_and_net_swept_clearance": "not_performed",
        "returnability": "not_performed",
        "schema2_conversion_and_audit": "not_performed",
        "training": "not_authorized_by_this_result",
        "robot_execution": "prohibited",
    }
    assert len(result["formal_blockers"]) == 4
