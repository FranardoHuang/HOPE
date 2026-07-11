from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INTAKE_PATH = ROOT / "configs" / "motion_video_intake_20260711.json"
RESULT_PATH = ROOT / "configs" / "motion_video_gvhmr_results_20260711.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_tracked_gvhmr_results_bind_the_exact_intake_and_every_asset():
    intake = json.loads(INTAKE_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    intake_sha = hashlib.sha256(INTAKE_PATH.read_bytes()).hexdigest()
    queue_tool_sha = hashlib.sha256(
        (ROOT / "scripts" / "run_motion_video_gvhmr_queue.py").read_bytes()
    ).hexdigest()
    result_auditor_sha = hashlib.sha256(
        (ROOT / "scripts" / "audit_gvhmr_result.py").read_bytes()
    ).hexdigest()
    assert result["status"] == "complete"
    assert result["processing_contract"]["manifest_sha256"] == intake_sha
    assert result["processing_contract"]["queue_tool_sha256"] == queue_tool_sha
    assert result["processing_contract"]["result_auditor_sha256"] == result_auditor_sha
    assert SHA256.fullmatch(result["queue_state_sha256"])
    assert result["formal_eligible"] is False
    assert result["formal_blockers"]

    assets = {asset["id"]: asset for asset in intake["assets"]}
    rows = result["results"]
    assert [row["asset_id"] for row in rows] == intake["processing_order"]
    assert len(rows) == len(assets) == 10
    for row in rows:
        asset = assets[row["asset_id"]]
        assert row["source_sha256"] == asset["sha256"]
        assert row["frames"] == asset["media"]["frames"]
        assert row["result_bytes"] > 0
        assert row["finite_elements"] > 0
        assert row["structural_status"] == "pass"
        assert SHA256.fullmatch(row["result_sha256"])
        assert SHA256.fullmatch(row["structural_audit_sha256"])
