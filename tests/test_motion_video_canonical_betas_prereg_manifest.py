from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "motion_video_canonical_betas_prereg_20260711.json"
SOURCE_PATH = ROOT / "configs" / "motion_video_gvhmr_results_20260711.json"
RESULT_PATH = ROOT / "configs" / "motion_video_canonical_betas_result_20260711.json"
TOOL_PATH = ROOT / "scripts" / "materialize_canonical_gvhmr_betas.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prereg_binds_complete_gvhmr_cohort_and_stays_diagnostic():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

    assert plan["status"] == "preregistered_not_executed"
    assert plan["formal_eligible"] is False
    assert plan["a3_calibrated"] is False
    assert plan["measured_height_m"] is None
    assert plan["body_shape_contract"] == (
        "diagnostic_same_performer_coordinatewise_median_betas_v1"
    )
    assert plan["source_results_manifest"]["sha256"] == _sha(SOURCE_PATH)
    assert plan["aggregation"]["same_performer_asserted"] is True
    assert plan["aggregation"]["expected_inputs"] == 10
    assert plan["aggregation"]["method"] == (
        "coordinatewise_median_of_per_video_coordinatewise_medians"
    )
    assert plan["aggregation"]["changed_field_allowlist"] == [
        "smpl_params_global.betas"
    ]
    assert plan["execution_contract"] == {
        "cpu_only": True,
        "CUDA_VISIBLE_DEVICES": "",
        "python_executable": (
            "/workspace/yikang/miniforge3/envs/hope-motion-py310/bin/python3.10"
        ),
        "python_version": "Python 3.10.20",
        "pip_freeze_sha256": (
            "56b0f8af9677b279bbb4925b6f49113f484dcb9ded1ed8d9bc56af71f304c694"
        ),
    }

    fields = ("result_path", "result_sha256", "result_bytes", "frames")
    source_by_id = {row["asset_id"]: row for row in source["results"]}
    plan_by_id = {row["asset_id"]: row for row in plan["inputs"]}
    assert len(source_by_id) == len(plan_by_id) == 10
    assert source_by_id.keys() == plan_by_id.keys()
    for asset_id, row in plan_by_id.items():
        assert {field: row[field] for field in fields} == {
            field: source_by_id[asset_id][field] for field in fields
        }


def test_prereg_cannot_mislabel_heuristic_height_or_unbound_loader():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    loader = plan["gmr_loader_observation"]
    blockers = "\n".join(plan["formal_blockers"])

    assert "betas[0]" in loader["semantics"]
    assert "1.66 + 0.1 * beta[0]" in loader["semantics"]
    assert loader["exact_on_pod_loader_sha256"] is None
    assert loader["current_repo_queue_compatibility"].startswith("incompatible:")
    assert loader["rerun_status"].startswith("blocked_until")
    assert "no measured performer height" in blockers
    assert "exact on-Pod GMR loader source SHA" in blockers
    assert "existing GMR queue hard-codes diagnostic_video_betas" in blockers
    assert plan["output_contract"]["no_clobber"] is True
    assert plan["output_contract"]["completion_manifest_published_last"] is True


def test_materialized_result_cross_binds_plan_tool_and_ten_outputs():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "complete_diagnostic_canonical_betas_materialization"
    assert result["plan"]["sha256"] == _sha(PLAN_PATH)
    assert result["source_results_manifest"]["sha256"] == _sha(SOURCE_PATH)
    assert result["tool"]["sha256"] == _sha(TOOL_PATH)
    assert result["body_shape_contract"] == plan["body_shape_contract"]
    canonical = result["canonical_betas"]
    assert len(canonical["components"]) == 10
    assert canonical["measured_height_m"] is None
    assert canonical["a3_calibrated"] is False
    assert canonical["formal_eligible"] is False
    assert len(result["results"]) == 10
    assert {row["asset_id"] for row in result["results"]} == {
        row["asset_id"] for row in plan["inputs"]
    }
    for row in result["results"]:
        assert len(row["output_sha256"]) == 64
        assert len(row["non_beta_semantic_sha256"]) == 64
        assert row["output_bytes"] > 0
    invariants = result["invariants"]
    assert invariants["only_smpl_params_global_betas_changed"] is True
    assert invariants["all_non_beta_semantic_digests_match_after_save_reload"] is True
    assert invariants["all_outputs_use_the_same_canonical_vector"] is True
    correction = result["loader_observation_correction"]
    assert correction["status"].endswith("revoked_by_exact_loader_source")
    assert correction["zero_padding"] is False
    assert correction["actual_selection"].endswith("numpy()[:10]")
    superseding = ROOT / correction["superseding_preregistration"]["path"]
    assert _sha(superseding) == correction["superseding_preregistration"]["sha256"]
    assert result["next_gate"]["canonical_gmr_rerun_status"] == "complete_diagnostic"
    assert result["next_gate"]["result_manifest"].endswith(
        "motion_video_canonical_gmr_results_20260711.json"
    )
