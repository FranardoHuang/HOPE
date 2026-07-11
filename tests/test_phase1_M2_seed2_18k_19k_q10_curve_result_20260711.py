import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs" / "phase1_M2_seed2_18k_19k_q10_curve_result_20260711.json"


def test_curve_result_is_screen_only_and_same_paper():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    screen = data["screen_contract"]
    assert screen["evaluation_contract_exact"] is False
    assert screen["lineage"] == "causal"
    assert screen["schedule_k"] == 20
    assert screen["attempts_per_side"] == 10
    assert screen["q10_role"] == "screen_only"
    assert screen["stop_authorized"] is False
    assert screen["promote_authorized"] is False
    assert screen["q50_triggered"] is False
    assert len(screen["immutable_schedule_sha256"]) == 64


def test_adjacent_milestones_cross_without_decision():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    old = data["arms"]["old"]["milestones"]
    s1 = data["arms"]["S1"]["milestones"]
    assert old["18000"]["return_rate"] == {
        "forehand": 0.0,
        "backhand": 0.8,
        "aggregate": 0.4,
    }
    assert old["19000"]["return_rate"]["aggregate"] == 0.5
    assert s1["18000"]["return_rate"]["aggregate"] == 0.6
    assert s1["19000"]["return_rate"] == {
        "forehand": 0.0,
        "backhand": 0.8,
        "aggregate": 0.4,
    }
    paired = data["paired_readout"]
    assert paired["old_minus_S1_aggregate_at_18000"] == -0.2
    assert paired["old_minus_S1_aggregate_at_19000"] == 0.1
    assert paired["ranking_crossed_between_adjacent_q10_milestones"] is True


def test_19k_checkpoints_are_finite_and_bound():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    for arm in data["arms"].values():
        row = arm["milestones"]["19000"]
        assert row["returncode"] == 0
        assert row["checkpoint_filename_iteration"] == 19000
        assert row["checkpoint_embedded_iteration"] == 19000
        assert row["checkpoint_all_tensors_finite"] is True
        assert row["checkpoint_contract_sha_matches_adjacent"] is True
        assert row["checkpoint_lineage_exact"] == 0
        for key in (
            "checkpoint_sha256",
            "state_sha256",
            "report_sha256",
            "job_spec_sha256",
            "job_contract_sha256",
        ):
            assert len(row[key]) == 64
