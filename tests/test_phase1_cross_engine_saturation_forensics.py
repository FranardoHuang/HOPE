import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_phase1_cross_engine_forensics.py"
INPUTS = ROOT / "configs" / "phase1_cross_engine_saturation_forensic_inputs_20260711.json"
RESULT = ROOT / "configs" / "phase1_cross_engine_saturation_forensic_result_20260711.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("phase1_cross_engine_forensics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_result_is_bound_to_analyzer_and_input_contract():
    result = json.loads(RESULT.read_text())
    assert result["status"] == "complete_read_only_question_aligned_forensic"
    assert result["analyzer"]["sha256"] == sha256(SCRIPT)
    assert result["input_contract"]["sha256"] == sha256(INPUTS)
    assert sha256(RESULT) == "aff8f4e665d20bb76a56e079735f32b6766388ee05f61c51e93adeb568be45c9"


def test_fresh_exact_split_is_localized_without_threshold_tuning():
    result = json.loads(RESULT.read_text())
    diagnosis = result["diagnosis"]["fresh_SZ"]
    evidence = diagnosis["primary_evidence"]
    assert diagnosis["classification"] == (
        "engine_execution_divergence_at_capture_margin_plus_isaac_virtual_metric_ceiling"
    )
    assert evidence["model_4000_forehand_mujoco_hit_count"] == 0
    assert evidence["model_4000_forehand_isaac_virtual_hit_count"] == 49
    assert evidence["model_4000_forehand_mujoco_mean_position_error_m"] > 0.095
    assert evidence["model_4000_forehand_isaac_mean_position_error_m"] < 0.095
    assert result["limitations"]["no_rescoring_or_threshold_change"] is True


def test_M3_split_exposes_signed_face_blindness_and_termination_union():
    result = json.loads(RESULT.read_text())
    diagnosis = result["diagnosis"]["causal_M3"]
    assert diagnosis["primary_evidence"]["M3_old_backhand_isaac_mean_normal_error_deg"] > 160
    assert diagnosis["primary_evidence"]["M3_S1_backhand_isaac_mean_normal_error_deg"] < 10
    assert diagnosis["primary_evidence"]["M3_old_mujoco_contact_but_net_fail_count"] == 47
    old = result["pairs"]["causal_M3_old_vs_S1"]["arms"]["M3_old"]
    termination = old["base_and_termination"]["termination"]
    assert termination["mujoco_attempt_physical_fall_count"] == 1
    assert termination["mujoco_attempt_guard_reset_count"] == 8
    assert termination["mujoco_summary_fell_union_count"] == 9
    assert termination["summary_fell_is_physical_plus_guard_union"] is True


def test_artifact_sha_mismatch_fails_closed(tmp_path):
    module = load_module()
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n")
    with pytest.raises(module.ForensicError, match="artifact SHA mismatch"):
        module.require_artifact(
            tmp_path,
            {"path": "artifact.json", "sha256": "0" * 64},
        )
