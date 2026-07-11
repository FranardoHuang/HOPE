from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/phase1_recovery_tuple_abc_prereg_20260712.json"
VALIDATOR_PATH = ROOT / "scripts/validate_phase1_recovery_tuple_prereg.py"

SPEC = importlib.util.spec_from_file_location("validate_phase1_recovery_tuple_prereg", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _load() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _validate_mutation(tmp_path: Path, value: dict, name: str = "mutated.json") -> None:
    path = _write(tmp_path / name, value)
    VALIDATOR.validate_prereg(path, VALIDATOR.sha256_file(path))


def test_checked_in_prereg_passes_design_and_deliberately_blocks_launch(capsys):
    digest = VALIDATOR.sha256_file(PREREG)
    assert digest == "39b97915bbb37bcf69e9a8a5eb87cb928bc1e7a6425c1f1555dd61c128b71e1a"
    assert VALIDATOR.main(
        [
            "--prereg",
            str(PREREG),
            "--expected-prereg-sha256",
            digest,
            "--mode",
            "design-check",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"status": "pass_design_only"' in output
    assert '"current_hybrid_formal": false' in output
    assert '"current_179_B_usable": false' in output

    assert VALIDATOR.main(
        [
            "--prereg",
            str(PREREG),
            "--expected-prereg-sha256",
            digest,
            "--mode",
            "launch-check",
        ]
    ) == 1
    blocked = capsys.readouterr().err
    assert "LAUNCH BLOCKED" in blocked
    assert "A_bridge_trajectory_certificate" in blocked
    assert "B_fresh_checkpoint_set" in blocked
    assert "vendor_gate3_continuous_no_reset_judge" in blocked


def test_hash_training_and_gate3_source_bindings_fail_closed(tmp_path: Path):
    with pytest.raises(VALIDATOR.ContractError, match="prereg SHA mismatch"):
        VALIDATOR.validate_prereg(PREREG, "0" * 64)

    value = _load()
    value["audited_training_source"]["git_blobs"]["event_timing"]["sha256"] = "0" * 64
    with pytest.raises(VALIDATOR.ContractError, match="training source binding"):
        _validate_mutation(tmp_path, value, "bad-training.json")

    value = _load()
    value["gate3_readonly_evidence"]["sha256"] = "0" * 64
    with pytest.raises(VALIDATOR.ContractError, match="Gate3 read-only evidence"):
        _validate_mutation(tmp_path, value, "bad-gate3.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["observed_training_semantics"].__setitem__(
                "mixed_generation_tuple_seen_in_training", True
            ),
            "training tuple semantics",
        ),
        (
            lambda value: value["rejected_current_hybrid"].__setitem__("formal_arm_allowed", True),
            "rejected hybrid",
        ),
        (
            lambda value: value["rejected_current_hybrid"].__setitem__(
                "idle_velocity_generation", "zero_velocity"
            ),
            "rejected hybrid",
        ),
        (
            lambda value: value["rejected_current_hybrid"].__setitem__(
                "parameter_tuning_can_make_formal", True
            ),
            "rejected hybrid",
        ),
    ],
)
def test_training_semantics_and_mixed_generation_hybrid_cannot_be_rewritten(
    tmp_path: Path, mutation, message: str
):
    value = _load()
    mutation(value)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        _validate_mutation(tmp_path, value)


def test_ordered_arms_and_current_checkpoint_boundaries_are_explicit(tmp_path: Path):
    value = _load()
    compatibility = value["structural_axis"]["current_179_checkpoint_compatibility"]
    assert compatibility["A"]["usable_now"].startswith("frozen_swing_subpolicy_diagnostic")
    assert compatibility["A"]["formal_ABC_causal_comparison_requires_fresh"] is True
    assert compatibility["B"]["usable_now"] is False
    assert compatibility["B"]["fresh_retrain_required"] is True
    assert compatibility["C"]["usable_now"] == "zero_shot_coherent_tuple_diagnostic_only"
    assert compatibility["C"]["fresh_retrain_required_for_learned_recovery_claim"] is True
    assert compatibility["current_checkpoint_may_be_relabeled_T1_trained"] is False

    value["structural_axis"]["arms"][0], value["structural_axis"]["arms"][1] = (
        value["structural_axis"]["arms"][1],
        value["structural_axis"]["arms"][0],
    )
    with pytest.raises(VALIDATOR.ContractError, match="ordered A/B/C"):
        _validate_mutation(tmp_path, value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["ready_set_contract"].__setitem__("ready_is_exact_motion_frame0", True),
            "never exact frame 0",
        ),
        (
            lambda value: value["ready_set_contract"]["required_conjuncts"].remove(
                "next_family_and_deadline_reachability"
            ),
            "conjuncts changed",
        ),
        (
            lambda value: value["ready_set_contract"].__setitem__(
                "empty_global_intersection_policy", "silently_use_forehand_only"
            ),
            "cannot silently shrink",
        ),
    ],
)
def test_ready_is_a_reachable_safety_set_not_dead_frame0(tmp_path: Path, mutation, message: str):
    value = _load()
    mutation(value)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        _validate_mutation(tmp_path, value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["frozen_structural_comparison"].__setitem__(
                "mid_sequence_reset_teleport_last_action_history_noise_reset", True
            ),
            "not frozen",
        ),
        (
            lambda value: value["frozen_structural_comparison"].__setitem__(
                "reward_source_weights_and_total_budget", "different_per_arm"
            ),
            "not frozen",
        ),
        (
            lambda value: value["conditional_reward_followup"].__setitem__("status", "authorized_now"),
            "prematurely authorized",
        ),
        (
            lambda value: value["conditional_reward_followup"].__setitem__(
                "interaction_design", "one_factor_at_a_time"
            ),
            r"full 2\^3",
        ),
        (
            lambda value: value["conditional_reward_followup"].__setitem__(
                "positive_brake_hold_survival_income_allowed", True
            ),
            "positive hold income",
        ),
    ],
)
def test_structure_precedes_reward_and_reward_budget_cannot_leak(tmp_path: Path, mutation, message: str):
    value = _load()
    mutation(value)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        _validate_mutation(tmp_path, value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["evaluation_contract"].__setitem__(
                "q10_role", "decision_and_early_stop"
            ),
            "evaluation contract",
        ),
        (
            lambda value: value["evaluation_contract"].__setitem__(
                "vendor_gate3_is_final_arbiter", False
            ),
            "evaluation contract",
        ),
        (
            lambda value: value["evaluation_contract"].__setitem__(
                "isaac_vendor_disagreement_policy", "average_scores"
            ),
            "evaluation contract",
        ),
        (
            lambda value: value["evaluation_contract"].__setitem__(
                "mid_sequence_reset_or_teleport", True
            ),
            "evaluation contract",
        ),
    ],
)
def test_vendor_gate3_no_reset_exam_and_q10_governance_are_immutable(
    tmp_path: Path, mutation, message: str
):
    value = _load()
    mutation(value)
    with pytest.raises(VALIDATOR.ContractError, match=message):
        _validate_mutation(tmp_path, value)


def test_blocked_design_cannot_sneak_in_one_unreviewed_binding(tmp_path: Path):
    value = _load()
    value["implementation_bindings"]["A_bridge_source"] = {
        "repo_path": "unreviewed.cpp",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    with pytest.raises(VALIDATOR.ContractError, match="every execution binding null"):
        _validate_mutation(tmp_path, value)
