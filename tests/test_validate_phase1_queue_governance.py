from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_phase1_queue_governance.py"
SPEC = importlib.util.spec_from_file_location("phase1_queue_governance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GOVERNANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOVERNANCE)


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _scaleout_causal() -> dict:
    return _load(
        "configs/phase1_checkpoint_curve_scaleout_causal_pod1_20260711.json"
    )


def test_repository_contract_covers_full_papers_and_fixed_default_order():
    result = GOVERNANCE.validate_repository(ROOT)
    assert result["status"] == "pass"
    assert result["scaleout"]["manifest_jobs"] == 142
    assert result["cadence"]["planned_slots"] == 24
    assert result["cadence"]["manifest_active_tail_jobs"] == 23
    assert result["cadence"]["prior_completed_slots"] == 1
    assert result["q50"]["curve_worker_jobs_allowed"] is False
    assert result["runbook_dual_queues_concurrent"] is True
    assert result["motion_default_order"] == {
        "asset_count": 10,
        "first_asset_id": "franco_forehand_block",
    }
    assert result["real_robot_commands"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("screen_policy", "screen_only"), False, "screen_only"),
        (("screen_policy", "stop_or_promote_allowed"), True, "stop_or_promote"),
        (("screen_policy", "schedule_k"), 100, "q10-only"),
        (("screen_policy", "attempts_per_side"), 50, "attempts_per_side"),
        (("screen_policy", "decision_followup"), "q10 may decide", "q50 decision"),
        (("jobs", 0, "screen_only"), False, "screen_only"),
    ],
)
def test_screen_and_q50_policy_cannot_be_relabelled(path, value, message):
    manifest = _scaleout_causal()
    cursor = manifest
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(GOVERNANCE.ContractError, match=message):
        GOVERNANCE.validate_manifest(
            manifest,
            label="mutated",
            require_readiness_barrier=True,
        )


def test_empty_q50_template_cannot_be_executed_as_a_curve_queue():
    manifest = _scaleout_causal()
    manifest["screen_policy"]["schedule_k"] = 100
    manifest["screen_policy"]["attempts_per_side"] = 50
    manifest["jobs"] = []
    with pytest.raises(GOVERNANCE.ContractError, match="q10-only"):
        GOVERNANCE.validate_manifest(manifest, label="q50-template")


def test_readiness_metadata_and_every_milestone_barrier_are_mandatory():
    manifest = _scaleout_causal()
    del manifest["checkpoint_readiness_barrier"]
    with pytest.raises(GOVERNANCE.ContractError, match="requires readiness"):
        GOVERNANCE.validate_manifest(
            manifest,
            label="missing-readiness",
            require_readiness_barrier=True,
        )

    manifest = _scaleout_causal()
    del manifest["jobs"][1]["barrier_id"]
    with pytest.raises(GOVERNANCE.ContractError, match="only part"):
        GOVERNANCE.validate_manifest(
            manifest,
            label="partial-barrier",
            require_readiness_barrier=True,
        )


def test_reordering_or_reusing_milestone_groups_fails_closed():
    manifest = _scaleout_causal()
    manifest["jobs"][2], manifest["jobs"][4] = (
        manifest["jobs"][4],
        manifest["jobs"][2],
    )
    with pytest.raises(GOVERNANCE.ContractError, match="contiguous and strictly increasing"):
        GOVERNANCE.validate_manifest(
            manifest,
            label="reordered",
            require_readiness_barrier=True,
        )

    manifest = _scaleout_causal()
    for job in manifest["jobs"][2:4]:
        job["barrier_id"] = "causal_18000"
    with pytest.raises(GOVERNANCE.ContractError, match="disagrees with milestone|reused"):
        GOVERNANCE.validate_manifest(
            manifest,
            label="reused",
            require_readiness_barrier=True,
        )


def test_job_checkpoint_identity_and_exactness_escape_are_not_advisory():
    manifest = _scaleout_causal()
    manifest["jobs"][0]["checkpoint"] = manifest["jobs"][0]["checkpoint"].replace(
        "model_18000.pt", "model_18001.pt"
    )
    with pytest.raises(GOVERNANCE.ContractError, match="id/checkpoint"):
        GOVERNANCE.validate_manifest(
            manifest,
            label="wrong-checkpoint",
            require_readiness_barrier=True,
        )

    manifest = _scaleout_causal()
    manifest["jobs"][0]["extra_args"] = ["--schedule-k", "20"]
    with pytest.raises(GOVERNANCE.ContractError, match="judge args"):
        GOVERNANCE.validate_manifest(
            manifest,
            label="missing-inexact-escape",
            require_readiness_barrier=True,
        )


def test_historical_pre_governance_manifest_is_rejected_not_laundered():
    historical = _load("configs/phase1_checkpoint_curve_initial_pod1_20260711.json")
    with pytest.raises(GOVERNANCE.ContractError, match="screen_policy"):
        GOVERNANCE.validate_manifest(historical, label="historical")


def test_cli_manifest_mode_is_read_only_and_reports_groups(tmp_path, capsys):
    manifest_path = tmp_path / "queue.json"
    manifest_path.write_text(json.dumps(_scaleout_causal()), encoding="utf-8")
    assert GOVERNANCE.main(
        [
            "--manifest",
            str(manifest_path),
            "--require-readiness-barrier",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    summary = report["manifests"][str(manifest_path.resolve())]
    assert summary["job_count"] == 8
    assert [row["iteration"] for row in summary["milestone_groups"]] == [
        18000,
        19000,
        20000,
        20998,
    ]
