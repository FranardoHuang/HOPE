from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
GMR_PATH = ROOT / "configs" / "motion_video_gmr_results_20260711.json"
GROUND_PATH = ROOT / "configs" / "motion_video_gmr_ground_results_20260711.json"
GROUND_TOOL_PATH = ROOT / "scripts" / "ground_gmr_pkl.py"
MJCF_PATH = (
    ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")

SPEC = importlib.util.spec_from_file_location("ground_gmr_manifest_tool", GROUND_TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
GROUND_TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GROUND_TOOL
SPEC.loader.exec_module(GROUND_TOOL)

TOOL_SHA256 = "db5bd1670cd113ce6ada8b53a7d4b0e5e25f0cc51665983c4958146e99ce4591"
MJCF_SHA256 = "2ab1cd31bffaaef979b4d9f35699bf1e6bec3a127be96c9266af131eee3feb97"
COMPILED_COLLISION_SHA256 = "18e7f6ffbefba9dbd988f7c3cb9fb92b250777862fc25fa3d4a0b2ca0f8386e5"


EXPECTED_REMOTE_EVIDENCE = {
    "franco_forehand_block": {
        "output_sha256": "cdbfaeebae26e55c43cf1313ddfd29fbf0a5fea2f8726fb9a42835f2f0bf2abf",
        "output_bytes": 20022,
        "report_sha256": "dade8898ca0040d437fc6ffc7312fe39a965aad163b36ffe78cdd73d82b1431e",
        "report_bytes": 9469,
        "before_min": -0.08414975109946435,
        "before_max": -0.07727632254752678,
        "shift": 0.08415975109946434,
        "applied": 0.08415975109946439,
        "after_min": 9.999999999919795e-06,
        "after_max": 0.006883428551937611,
        "frame": 2,
        "geom_id": 65,
        "geom_name": "left_ankle_roll_collision",
        "body_id": 26,
        "body_name": "left_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "franco_backhand_block": {
        "output_sha256": "b8c68a902d9096bbcc5f98f1d12dead57d26e1c200e83261b6c2f409a85a5d5d",
        "output_bytes": 19414,
        "report_sha256": "4350e5a7766ff999379f596bb4a2f55c048d4c4f2409a7ca547fb1faa3989d40",
        "report_bytes": 9333,
        "before_min": -0.08506299442982071,
        "before_max": -0.07723022168382573,
        "shift": 0.0850729944298207,
        "applied": 0.08507299442982075,
        "after_min": 1.0000000000037756e-05,
        "after_max": 0.007842772745995014,
        "frame": 6,
        "geom_id": 65,
        "geom_name": "left_ankle_roll_collision",
        "body_id": 26,
        "body_name": "left_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "franco_forehand_loop": {
        "output_sha256": "b6ebecda79bcf042542ad955e330a9979640fd29d958df9a9bd47331a3296ecd",
        "output_bytes": 21542,
        "report_sha256": "c7aeb2e0029e4fe0628a2cf071983a2249d4a582be7848e6669c2dcd477c878b",
        "report_bytes": 9779,
        "before_min": -0.0828585746010495,
        "before_max": -0.07585754815796399,
        "shift": 0.0828685746010495,
        "applied": 0.08286857460104946,
        "after_min": 9.999999999850406e-06,
        "after_max": 0.007011026443085475,
        "frame": 64,
        "geom_id": 65,
        "geom_name": "left_ankle_roll_collision",
        "body_id": 26,
        "body_name": "left_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "franco_backhand_loop_a": {
        "output_sha256": "4e48147083b946bc934efea41cf53b27fa7947dd935c2fb1a7c7d1f4d3a0689f",
        "output_bytes": 21238,
        "report_sha256": "d2ccece7608fb2910ffa229e234f3e8819974d81336695f93bcc0b437a023356",
        "report_bytes": 9668,
        "before_min": -0.08715994746268714,
        "before_max": -0.07547803160010809,
        "shift": 0.08716994746268714,
        "applied": 0.08716994746268714,
        "after_min": 9.9999999998851e-06,
        "after_max": 0.011691915862579047,
        "frame": 1,
        "geom_id": 65,
        "geom_name": "left_ankle_roll_collision",
        "body_id": 26,
        "body_name": "left_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "franco_backhand_loop_b": {
        "output_sha256": "3254ccb4fe3ccd24e3417b8e251ee7d1e135d59b63daf6fa44744d3bf85855c8",
        "output_bytes": 27926,
        "report_sha256": "8b0992643db5c9b386e94f8b5babe761318bf26e294ffcd8243d0edb0c3233e9",
        "report_bytes": 11035,
        "before_min": -0.08489999220482566,
        "before_max": -0.07800788994095698,
        "shift": 0.08490999220482566,
        "applied": 0.08490999220482565,
        "after_min": 9.999999999975306e-06,
        "after_max": 0.006902102263868677,
        "frame": 44,
        "geom_id": 65,
        "geom_name": "left_ankle_roll_collision",
        "body_id": 26,
        "body_name": "left_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "franco_backhand_loop_c": {
        "output_sha256": "f08dd09ac85af78c683c10aa361be411baeb40be002a2e3affc95e540d70f193",
        "output_bytes": 30054,
        "report_sha256": "25115863dd9cc338d3844f71855202a0a951e5742a54e22fbe3ee43da5f55b9d",
        "report_bytes": 11467,
        "before_min": -0.08513665808136611,
        "before_max": -0.07547190293087568,
        "shift": 0.08514665808136611,
        "applied": 0.08514665808136612,
        "after_min": 1.0000000000003062e-05,
        "after_max": 0.009674755150490447,
        "frame": 0,
        "geom_id": 65,
        "geom_name": "left_ankle_roll_collision",
        "body_id": 26,
        "body_name": "left_ankle_roll_Link",
        "joint_excess": 1.3877787807814457e-17,
    },
    "v6_forehand_block": {
        "output_sha256": "628d41b2d7ba86f04dcbc2e2a38eaaf7632c2c064ae89f3c636fa15f45955a2b",
        "output_bytes": 10902,
        "report_sha256": "00e6ef92dde01940af4839cd64eb3617ffa9197831406b11974b7181ab1df2a6",
        "report_bytes": 7667,
        "before_min": -0.08132114657492334,
        "before_max": -0.07943275735253116,
        "shift": 0.08133114657492334,
        "applied": 0.08133114657492335,
        "after_min": 1.0000000000003062e-05,
        "after_max": 0.0018983892223921814,
        "frame": 5,
        "geom_id": 78,
        "geom_name": "right_ankle_roll_collision",
        "body_id": 32,
        "body_name": "right_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "v6_backhand_block": {
        "output_sha256": "0dda32e1c7e6ad5400f96914217ee48719a79fd1ca7d9b657ac6e2164befe9f9",
        "output_bytes": 10902,
        "report_sha256": "ca20e8e37b995353a8371bf2621dc7ffa6e9efb569299f13db8bce0c31335ca4",
        "report_bytes": 7661,
        "before_min": -0.081899096802118,
        "before_max": -0.07999874194936452,
        "shift": 0.081909096802118,
        "applied": 0.081909096802118,
        "after_min": 9.999999999996123e-06,
        "after_max": 0.001910354852753482,
        "frame": 16,
        "geom_id": 78,
        "geom_name": "right_ankle_roll_collision",
        "body_id": 32,
        "body_name": "right_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "v7_forehand_block": {
        "output_sha256": "f2d71779a6925bf70921efeef890442e4e0bb51a096191e9aa8729347c37a634",
        "output_bytes": 19718,
        "report_sha256": "0ef5ffab0108d45d0a8d56a9331eb09a857063b0a71b91cfc18b0a21895c3f35",
        "report_bytes": 9415,
        "before_min": -0.08072308877320576,
        "before_max": -0.07733181421975804,
        "shift": 0.08073308877320576,
        "applied": 0.08073308877320573,
        "after_min": 9.999999999968368e-06,
        "after_max": 0.003401274553447693,
        "frame": 62,
        "geom_id": 78,
        "geom_name": "right_ankle_roll_collision",
        "body_id": 32,
        "body_name": "right_ankle_roll_Link",
        "joint_excess": 0.0,
    },
    "v7_backhand_block": {
        "output_sha256": "79651bfa0df7bd40be4bc4a657f8ea6db26cfe61a57139214e0cf7a4d11d9028",
        "output_bytes": 19718,
        "report_sha256": "5ae8cd2d88411c0666a4ead96d530ce4e9cfcbe8fa7736750b7683ad464293fe",
        "report_bytes": 9438,
        "before_min": -0.08254845541401618,
        "before_max": -0.08119781696989911,
        "shift": 0.08255845541401617,
        "applied": 0.08255845541401619,
        "after_min": 1.0000000000010001e-05,
        "after_max": 0.0013606384441170669,
        "frame": 58,
        "geom_id": 78,
        "geom_name": "right_ankle_roll_collision",
        "body_id": 32,
        "body_name": "right_ankle_roll_Link",
        "joint_excess": 0.0,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ground_manifest_binds_source_tool_mjcf_collision_and_joint_contracts():
    manifest = json.loads(GROUND_PATH.read_text(encoding="utf-8"))
    contract = manifest["processing_contract"]

    assert manifest["status"] == "complete_grounding_diagnostic"
    assert manifest["collection_cross_checks"] == {
        "reports_found": 10,
        "outputs_found": 10,
        "report_sha_and_bytes_recomputed": True,
        "output_sha_and_bytes_recomputed": True,
        "all_reports_status_pass": True,
        "all_reports_scope_match": True,
        "all_reports_formal_eligible_false": True,
        "all_report_inputs_match_source_manifest_sha_and_bytes": True,
        "all_report_outputs_match_physical_sha_and_bytes": True,
        "all_reports_bind_same_tool_sha256": True,
        "all_reports_bind_same_mjcf_sha256": True,
        "all_reports_bind_same_compiled_collision_sha256": True,
        "all_reports_bind_same_collision_geom_set": True,
        "all_reports_bind_same_31_joint_order": True,
        "all_reports_frames_fps_finite_match_source_manifest": True,
        "all_reports_preserve_required_invariants": True,
        "all_reports_state_discrete_frame_only_limit": True,
    }
    assert manifest["source_gmr_manifest"]["sha256"] == _sha256(GMR_PATH)
    assert contract["tool"]["sha256"] == TOOL_SHA256 == _sha256(GROUND_TOOL_PATH)
    assert contract["mjcf"]["sha256"] == MJCF_SHA256 == _sha256(MJCF_PATH)
    assert contract["mjcf"]["bytes"] == MJCF_PATH.stat().st_size == 49107
    assert contract["mjcf"]["compiled_kinematic_collision_sha256"] == COMPILED_COLLISION_SHA256
    assert contract["mjcf"]["ground_geom"] == "floor"
    assert contract["mjcf"]["ground_z_m"] == 0.0
    assert contract["joint_contract"]["count"] == 31
    assert contract["joint_contract"]["model_joint_ids"] == list(range(1, 32))
    assert contract["joint_contract"]["names"] == list(GROUND_TOOL.A3_GMR_JOINT_NAMES)
    assert len(set(contract["joint_contract"]["names"])) == 31
    assert contract["joint_contract"]["input_joint_names_present"] is False
    assert contract["joint_contract"]["interpretation"] == "A3_GMR_JOINT_NAMES"
    collision = contract["collision_contract"]
    assert collision["enabled_robot_geom_count"] == len(collision["enabled_robot_geom_ids"]) == 37
    assert collision["visual_only_geoms_excluded"] is True
    assert collision["surface_method"] == "analytic_primitive_support_or_compiled_mesh_vertices"


def test_each_ground_result_cross_binds_upstream_and_exact_remote_evidence():
    upstream = json.loads(GMR_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(GROUND_PATH.read_text(encoding="utf-8"))
    source_rows = {row["asset_id"]: row for row in upstream["results"]}
    rows = manifest["results"]
    target = manifest["processing_contract"]["grounding"]["target_clearance_m"]
    contact_ceiling = manifest["processing_contract"]["grounding"]["max_grounded_clearance_m"]

    assert [row["asset_id"] for row in rows] == [row["asset_id"] for row in upstream["results"]]
    assert set(source_rows) == set(EXPECTED_REMOTE_EVIDENCE)
    assert len(rows) == 10
    for row in rows:
        source = source_rows[row["asset_id"]]
        expected = EXPECTED_REMOTE_EVIDENCE[row["asset_id"]]
        assert row["status"] == "complete_grounding_diagnostic"
        assert row["input"] == {
            "path": source["output_path"],
            "sha256": source["output_sha256"],
            "bytes": source["output_bytes"],
        }
        assert row["output"]["sha256"] == expected["output_sha256"]
        assert row["output"]["bytes"] == expected["output_bytes"]
        assert row["output"]["physical_sha_and_bytes_recomputed"] is True
        assert row["report"]["sha256"] == expected["report_sha256"]
        assert row["report"]["bytes"] == expected["report_bytes"]
        assert SHA256.fullmatch(row["output"]["sha256"])
        assert SHA256.fullmatch(row["report"]["sha256"])
        assert row["output"]["path"].endswith(f"/{row['asset_id']}.grounded.pkl")
        assert row["report"]["path"].endswith(f"/{row['asset_id']}.grounding.json")

        structure = row["structure"]
        frames = source["frames"]
        assert structure["frames"] == frames
        assert structure["fps"] == source["fps"] == 30.0
        assert structure["finite_elements"] == source["finite_elements"] == frames * 38 + 1
        assert structure["root_rotation_max_norm_error"] == source["root_rotation_max_norm_error"]
        assert structure["root_rotation_max_norm_error"] < 1e-6
        assert structure["dtypes"] == {
            "dof_pos": "float64",
            "root_pos": "float64",
            "root_rot": "float64",
        }
        assert structure["shapes"] == {
            "dof_pos": [frames, 31],
            "root_pos": [frames, 3],
            "root_rot": [frames, 4],
        }
        assert row["joint_contract"] == {
            "count": 31,
            "max_range_excess_rad": expected["joint_excess"],
        }
        assert row["joint_contract"]["max_range_excess_rad"] <= 1e-5

        grounding = row["grounding"]
        before, shift, after = grounding["before"], grounding["shift"], grounding["after"]
        assert before == {
            "minimum_clearance_m": expected["before_min"],
            "maximum_of_frame_minima_m": expected["before_max"],
            "minimum_frame": expected["frame"],
            "minimum_geom_id": expected["geom_id"],
            "minimum_geom_name": expected["geom_name"],
            "minimum_body_id": expected["body_id"],
            "minimum_body_name": expected["body_name"],
        }
        assert shift == {
            "requested_m": expected["shift"],
            "applied_min_m": expected["applied"],
            "applied_max_m": expected["applied"],
            "spread_m": 0.0,
        }
        assert after == {
            "minimum_clearance_m": expected["after_min"],
            "maximum_of_frame_minima_m": expected["after_max"],
            "same_minimum_frame_geom_body": True,
        }
        assert math.isclose(before["minimum_clearance_m"] + shift["requested_m"], target, abs_tol=1e-15)
        assert math.isclose(after["minimum_clearance_m"], target, abs_tol=2e-15)
        assert after["minimum_clearance_m"] >= target - 2e-15
        assert after["minimum_clearance_m"] <= contact_ceiling
        assert before["maximum_of_frame_minima_m"] >= before["minimum_clearance_m"]
        assert after["maximum_of_frame_minima_m"] >= after["minimum_clearance_m"]
        assert row["invariants"] == {
            "all_other_payload_fields_shallow_preserved": True,
            "dof_position_exact": True,
            "root_pos_dtype_preserved": True,
            "root_rotation_exact": True,
            "root_xy_exact": True,
            "root_z_relative_trajectory_max_error_m": 0.0,
        }
        assert row["body_shape_contract"] == "diagnostic_video_betas"
        assert row["clearance_sampling"] == "original_discrete_frames_only"
        assert row["formal_eligible"] is False


def test_grounding_completion_does_not_promote_unrun_safety_or_schema_gates():
    manifest = json.loads(GROUND_PATH.read_text(encoding="utf-8"))

    assert manifest["body_shape_contract"] == "diagnostic_video_betas"
    assert manifest["canonical_betas"] == {
        "status": "not_performed",
        "canonical_betas_bound": False,
    }
    assert manifest["formal_eligible"] is False
    assert set(manifest["remaining_gates"].values()) == {"not_performed"}
    assert set(manifest["remaining_gates"]) == {
        "canonical_betas",
        "inter_frame_continuous_ground_clearance",
        "self_collision",
        "dynamics_and_balance_feasibility",
        "table_net_swept_clearance",
        "returnability",
        "schema2_conversion_and_audit",
    }
    blockers = " ".join(manifest["formal_blockers"]).lower()
    for required in (
        "canonical",
        "inter-frame",
        "self-collision",
        "dynamics",
        "table/net",
        "returnability",
        "schema-2",
    ):
        assert required in blockers
    grounding = manifest["processing_contract"]["grounding"]
    assert grounding["clearance_sampling"] == "original_discrete_frames_only"
    assert grounding["continuous_time_clearance_proven"] is False
