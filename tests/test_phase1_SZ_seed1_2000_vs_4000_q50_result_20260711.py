import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "configs" / "phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_sz_peak_q50_result_is_exact_but_selection_only():
    data = json.loads(RESULT.read_text())

    assert data["status"] == "complete_fresh_exact_checkpoint_selection_screen"
    assert data["training_commit"] == "6d93bcb16c422a2f42748c2dc99432559653480b"
    assert data["eval_commit"] == "46a0ce24524fdb843e55fe82ba4c045f2adc090f"
    assert data["shared_runtime"]["evaluation_contract_exact"] is True
    assert data["shared_runtime"]["fresh_lineage"] is True
    assert data["shared_runtime"]["formal_target"] is True

    trigger = data["source_trigger"]
    assert trigger["model_2000_aggregate_q10"] == 0.9
    assert trigger["model_4000_aggregate_q10"] == 0.5
    assert trigger["q10_used_for_decision"] is False
    prereg = ROOT / trigger["preregistration"]["path"]
    assert _sha(prereg) == trigger["preregistration"]["sha256"]

    execution = data["accepted_execution"]
    for key in ("runner", "shared_runner", "config"):
        artifact = ROOT / execution[key]["path"]
        assert _sha(artifact) == execution[key]["sha256"]
    assert execution["paired_result"]["sha256"] == (
        "b95ba6c4657e4b542ea3c5f08d02941c830ea48d71b56bb1a247ef7a46730478"
    )

    paper = data["immutable_schedule"]
    assert paper["schedule_k"] == 100
    assert paper["attempts_per_side"] == 50
    assert paper["seed"] == 0
    assert paper["noise_scale"] == 0.0
    assert paper["one_question_reset"] is True
    assert paper["same_artifact_for_both_checkpoints"] is True
    assert paper["censored_attempts"] == 0

    model_2000 = data["arms"]["model_2000"]
    model_4000 = data["arms"]["model_4000"]
    assert model_2000["returns"]["aggregate"] == "83/100"
    assert model_2000["returns"]["forehand"] == "33/50"
    assert model_2000["returns"]["backhand"] == "50/50"
    assert model_4000["returns"]["aggregate"] == "50/100"
    assert model_4000["returns"]["forehand"] == "0/50"
    assert model_4000["returns"]["backhand"] == "50/50"
    assert model_2000["physical_falls"] == model_4000["physical_falls"] == 0
    assert model_2000["guard_resets"] == model_4000["guard_resets"] == 100

    selection = data["selection"]
    assert selection["selected_checkpoint"] == "model_2000"
    assert selection["whole_arm_action"] == "continue_unmodified"
    assert selection["whole_arm_stop_allowed"] is False
    assert selection["whole_arm_promote_allowed"] is False
    assert selection["deploy_gate"] is False
    assert selection["real_robot_authorized"] is False
    assert "not evidence of recovery" in data["limitations"]["guard_reset_interpretation"]


def test_all_bound_result_hashes_have_full_sha256_shape():
    data = json.loads(RESULT.read_text())
    hashes = []
    hashes.extend(
        value["sha256"]
        for value in data["accepted_execution"].values()
        if isinstance(value, dict) and "sha256" in value
    )
    hashes.extend(
        [
            data["source_trigger"]["preregistration"]["sha256"],
            data["immutable_schedule"]["file_sha256"],
            data["immutable_schedule"]["semantic_sha256"],
            data["shared_runtime"]["training_contract_sha256"],
            data["shared_runtime"]["mjcf_sha256"],
            data["shared_runtime"]["execution_contract_sha256"],
            data["shared_runtime"]["ready_state_sha256"],
        ]
    )
    for arm in data["arms"].values():
        hashes.extend(value for key, value in arm.items() if key.endswith("_sha256"))
    assert hashes
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in hashes)
