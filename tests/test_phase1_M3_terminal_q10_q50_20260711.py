from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
Q10_PATH = ROOT / "configs" / "phase1_M3_terminal_q10_pair_20260711.json"
Q50_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_prereg_20260711.json"
Q50_RESULT_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_result_20260711.json"
Q50_EXECUTION_PATH = ROOT / "configs" / "phase1_M3_terminal_q50_execution_20260711.json"
Q50_RUNNER_PATH = ROOT / "scripts" / "run_phase1_paired_bank_q50.py"
OLD_AUDIT_PATH = ROOT / "configs" / "phase1_M3_old_terminal_audit_20260711.json"
S1_AUDIT_PATH = ROOT / "configs" / "phase1_M3_S1_terminal_audit_20260711.json"
SCHEDULE_SHA256 = "7a908142eb8b47d9cbc30bb599c690a5b61e9d1238e9c793bd143e44514ed614"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m3_terminal_q10_is_same_schedule_cross_bound_screen_only_evidence():
    q10 = _load(Q10_PATH)
    old_audit = _load(OLD_AUDIT_PATH)
    s1_audit = _load(S1_AUDIT_PATH)

    assert q10["training_commit"] == "6d93bcb16c422a2f42748c2dc99432559653480b"
    assert q10["eval_commit"] == "46a0ce24524fdb843e55fe82ba4c045f2adc090f"
    assert q10["immutable_schedule"]["sha256"] == SCHEDULE_SHA256
    assert q10["immutable_schedule"]["schedule_k"] == 20
    assert q10["immutable_schedule"]["attempts_per_side"] == 10
    assert q10["immutable_schedule"]["seed"] == 0
    assert q10["immutable_schedule"]["noise_scale"] == 0.0
    assert q10["evaluation_contract_exact"] is False
    assert q10["causal"] is True
    assert q10["screen_only"] is True
    assert q10["stop_or_promote_allowed"] is False

    old = q10["cells"]["M3_old"]
    s1 = q10["cells"]["M3_S1"]
    assert old["checkpoint"]["sha256"] == old_audit["checkpoint"]["sha256"]
    assert old["training_contract"]["sha256"] == old_audit["training_contract"]["sha256"]
    assert s1["checkpoint"]["sha256"] == s1_audit["checkpoint"]["sha256"]
    assert s1["training_contract"]["sha256"] == s1_audit["training_contract"]["sha256"]
    for cell in (old, s1):
        assert cell["checkpoint_iteration"] == 20998
        assert cell["training_contract"]["checkpoint_embedded_sha256"] == (
            cell["training_contract"]["sha256"]
        )
        assert cell["training_contract"]["lineage_exact"] is False
        assert cell["worker_state"]["status"] == "complete"
        assert cell["worker_state"]["returncode"] == 0
        assert cell["evaluation_contract_exact"] is False
        assert cell["causal"] is True
        assert cell["screen_only"] is True
        for artifact in (
            "checkpoint",
            "training_contract",
            "worker_state",
            "worker_log",
            "judge_report",
            "exam_log",
            "exam_summary",
            "exam_attempt_ledger",
        ):
            assert re.fullmatch(r"[0-9a-f]{64}", cell[artifact]["sha256"])

    assert old["returns"] == {
        "forehand": "5/10",
        "backhand": "4/10",
        "aggregate": "9/20",
        "forehand_rate": 0.5,
        "backhand_rate": 0.4,
        "aggregate_rate": 0.45,
    }
    assert s1["returns"] == {
        "forehand": "10/10",
        "backhand": "10/10",
        "aggregate": "20/20",
        "forehand_rate": 1.0,
        "backhand_rate": 1.0,
        "aggregate_rate": 1.0,
    }
    assert q10["paired_delta_s1_minus_old"] == {
        "forehand_rate": 0.5,
        "backhand_rate": 0.6,
        "aggregate_rate": 0.55,
    }
    assert q10["q50_trigger"]["numeric_threshold_abs_rate"] == 0.2
    assert q10["q50_trigger"]["observed_terminal_aggregate_abs_delta"] == 0.55
    assert q10["q50_trigger"]["same_milestone_matched_control_trigger_met"] is True
    assert q10["q50_trigger"]["q50_started_by_this_result"] is False


def test_m3_terminal_q50_is_bound_preregistration_and_not_a_formal_or_deploy_claim():
    q10 = _load(Q10_PATH)
    q50 = _load(Q50_PATH)

    assert q50["source_trigger"]["q10_pair_sha256"] == _sha256(Q10_PATH)
    assert q50["source_trigger"]["trigger_met"] is True
    assert q50["status"] == "preregistered_not_started"
    assert q50["auto_activate"] is False
    assert q50["jobs_started"] == 0
    assert q50["runtime_state"] is None
    paper = q50["paper"]
    assert paper["seed"] == 0
    assert paper["noise_scales"] == [0.0]
    assert paper["schedule_k"] == 100
    assert paper["attempts_per_side"] == 50
    assert paper["same_immutable_schedule_required_for_both_arms"] is True
    assert paper["schedule_materialization"]["status"] == "not_materialized"
    assert paper["schedule_materialization"]["sha256"] is None
    assert paper["allow_inexact_contract_required"] is True
    assert paper["expected_evaluation_contract_exact"] is False

    for name in ("M3_old", "M3_S1"):
        prereg = q50["arms"][name]
        observed = q10["cells"][name]
        assert prereg["checkpoint_iteration"] == 20998
        assert prereg["checkpoint_sha256"] == observed["checkpoint"]["sha256"]
        assert prereg["training_contract_sha256"] == observed["training_contract"]["sha256"]
        assert prereg["checkpoint_embedded_training_contract_sha256"] == (
            prereg["training_contract_sha256"]
        )
        assert prereg["lineage_exact"] is False
        assert prereg["job_status"] == "not_started"
        assert prereg["pid"] is None and prereg["pgid"] is None
        assert prereg["result"] is None

    semantics = q50["diagnostic_semantics"]
    assert semantics["causal"] is True
    assert semantics["evaluation_contract_exact"] is False
    assert semantics["formal_target"] is False
    assert semantics["deploy_gate"] is False
    assert semantics["decision_scope"]
    assert "formal Phase-1 acceptance" in semantics["prohibited_claims"]
    assert "deployment readiness" in semantics["prohibited_claims"]
    assert q50["activation"]["authorized_to_start_by_this_file"] is False
    assert q50["activation"]["started_at_creation"] is False


def test_m3_terminal_q50_result_is_same_paper_complete_and_still_inexact():
    q10 = _load(Q10_PATH)
    prereg = _load(Q50_PATH)
    result = _load(Q50_RESULT_PATH)

    assert result["source_preregistration"]["sha256"] == _sha256(Q50_PATH)
    assert result["source_q10_pair"]["sha256"] == _sha256(Q10_PATH)
    assert result["accepted_execution"]["config"]["sha256"] == _sha256(
        Q50_EXECUTION_PATH
    )
    assert result["accepted_execution"]["runner"]["sha256"] == _sha256(
        Q50_RUNNER_PATH
    )
    paper = result["immutable_schedule"]
    assert paper["schedule_k"] == 100
    assert paper["attempts_per_side"] == 50
    assert paper["seed"] == 0 and paper["noise_scale"] == 0.0
    assert paper["same_artifact_for_both_arms"] is True
    assert result["shared_runtime"]["attempts_uncensored"] is True
    assert result["shared_runtime"]["question_order_exact"] is True

    old = result["arms"]["M3_old"]
    s1 = result["arms"]["M3_S1"]
    for name, cell in (("M3_old", old), ("M3_S1", s1)):
        frozen = prereg["arms"][name]
        assert cell["checkpoint_iteration"] == 20998
        assert cell["checkpoint_sha256"] == frozen["checkpoint_sha256"]
        assert cell["training_contract_sha256"] == frozen["training_contract_sha256"]
        assert cell["lineage_exact"] is False
        assert cell["evaluation_contract_exact"] is False
        for key, value in cell.items():
            if key.endswith("_sha256"):
                assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert old["returns"]["aggregate"] == "42/100"
    assert old["returns"]["forehand"] == "31/50"
    assert old["returns"]["backhand"] == "11/50"
    assert old["physical_falls"] == 9
    assert s1["returns"]["aggregate"] == "100/100"
    assert s1["returns"]["forehand"] == "50/50"
    assert s1["returns"]["backhand"] == "50/50"
    assert s1["physical_falls"] == 0
    assert result["paired_delta_s1_minus_old"]["aggregate_rate"] == 0.58
    assert result["decision"]["mujoco_same_family_terminal_selection"] == "M3_S1"
    assert result["decision"]["requires_isaac_same_paper_companion"] is True
    assert result["decision"]["authorized_formal_promotion"] is False
    assert result["decision"]["authorized_deployment"] is False
    assert result["decision"]["authorized_real_robot"] is False
    failed = result["preserved_validator_failure"]
    assert failed["retry_changed_schedule"] is False
    assert failed["retry_changed_checkpoint_or_recipe"] is False
    assert failed["schedule_file_sha256"] == paper["file_sha256"]
    assert failed["schedule_semantic_sha256"] == paper["semantic_sha256"]
