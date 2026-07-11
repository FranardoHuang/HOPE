from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs" / "motion_video_canonical_gmr_results_20260711.json"
PLAN = ROOT / "configs" / "motion_video_canonical_gmr_prereg_20260711.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTRACT = "diagnostic_same_performer_coordinatewise_median_betas_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_gmr_result_binds_plan_order_and_diagnostic_lineage():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["status"] == "complete_diagnostic_canonical_gmr"
    assert result["body_shape_contract"] == CONTRACT
    assert result["preregistration"]["sha256"] == _sha(PLAN)
    assert result["formal_eligible"] is False
    assert result["a3_calibrated"] is False
    assert result["measured_height_m"] is None
    rows = result["results"]
    assert [row["asset_id"] for row in rows] == plan["processing_order"]
    assert len(rows) == 10
    assert result["invariants"]["accepted_results"] == len(rows)
    assert result["runtime"]["cpu_only"] is True
    assert result["runtime"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["gmr_contract"]["loader_zero_padding"] is False
    assert result["gmr_contract"]["loader_selection"].endswith("[:10]")


def test_canonical_gmr_rows_have_finite_shape_warmup_and_content_bindings():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    seen_outputs = set()
    for row in result["results"]:
        assert row["frames"] > 1
        assert row["finite_elements"] == row["frames"] * (3 + 4 + 31) + 1
        assert 1 <= row["warmup_rounds"] <= 200
        assert 0.0 < row["warmup_max_dq"] < 1e-4
        assert row["output_bytes"] > 0
        assert row["output_path"].endswith(
            ".diagnostic_cohort_median_betas.gmr.pkl"
        )
        assert row["output_path"] not in seen_outputs
        seen_outputs.add(row["output_path"])
        for field in (
            "source_sha256",
            "output_sha256",
            "binding_sha256",
            "run_log_sha256",
            "structural_audit_sha256",
        ):
            assert SHA256.fullmatch(row[field])
    for field in ("queue_state", "launcher_log"):
        binding = result["runtime"][field]
        assert binding["bytes"] > 0
        assert SHA256.fullmatch(binding["sha256"])
