from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/action_ball_comparison_question.py"
SPEC = importlib.util.spec_from_file_location("comparison_question_under_test", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
TAPE_PATH = (
    ROOT.parents[1]
    / "configs/action_ball_n1_measured_20260803"
    / "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready"
    / "immutable_n1_tape.v1.22052606032f.json"
)


def _receipt(tape=None, *, target_recipe="outcome_dense_only"):
    tape = json.loads(TAPE_PATH.read_text()) if tape is None else tape
    tape_bytes = M._canonical_bytes(tape) + b"\n"
    return M.build_comparison_question_receipt(
        tape,
        tape_bytes=tape_bytes,
        action_id="take_061_unit04_bh",
        teacher_id="Take_061_unit04_BH",
        target_recipe=target_recipe,
    )


def test_real_tape_projects_only_the_common_a_c_question():
    receipt = _receipt()
    assert M.validate_comparison_question_receipt(receipt) == receipt
    assert receipt["identity"]["action_uid"] == 5527597793770800
    assert receipt["incoming_ball"]["velocity_w_mps"] == [
        -2.8339611740381896,
        1.0515251133682382,
        0.23989999999999997,
    ]
    clock = receipt["clock"]
    assert clock["pre_swing_wait_s"] + clock["scaled_t_hit_s"] == pytest.approx(
        clock["time_to_contact_s"], abs=2.0e-6
    )


def test_target_validity_reward_and_actor_tail_are_excluded_treatment_axes():
    arm_a = {
        "comparison_question": _receipt(target_recipe="current_lm"),
        "target_validity": [True, True, True],
        "actor_tail": "desired contact position/velocity/face",
        "reward": "A-only",
    }
    arm_c = {
        "comparison_question": _receipt(target_recipe="outcome_dense_only"),
        "target_validity": [False, False, False],
        "actor_tail": "incoming p/v/spin",
        "reward": "C-only",
    }
    assert M.require_same_comparison_question(
        arm_a["comparison_question"], arm_c["comparison_question"]
    ) == arm_a["comparison_question"]["comparison_question_sha256"]


def test_selected_target_runtime_clock_must_match_the_common_source():
    tape = json.loads(TAPE_PATH.read_text())
    changed = copy.deepcopy(tape)
    changed["targets"]["outcome_dense_only"]["runtime_target"][
        "pre_swing_wait_s"
    ] += 0.01
    changed_unsigned = dict(changed)
    changed_unsigned.pop("canonical_sha256")
    changed["canonical_sha256"] = M._digest(changed_unsigned)
    changed_bytes = M._canonical_bytes(changed) + b"\n"
    with pytest.raises(M.ComparisonQuestionError, match="target runtime clock differs"):
        M.build_comparison_question_receipt(
            changed,
            tape_bytes=changed_bytes,
            action_id="take_061_unit04_bh",
            teacher_id="Take_061_unit04_BH",
            target_recipe="outcome_dense_only",
        )


@pytest.mark.parametrize(
    "path,value",
    (
        (("question", "incoming_velocity_w_mps"), [-9.0, 0.0, 0.0]),
        (("source_task_receipt", "base_goal_w_m"), [0.0, 0.0, 1.0]),
        (("source_task_receipt", "landing_aim_w_xy_m"), [2.0, 0.0]),
        (("source_task_receipt", "pre_swing_wait_s"), 0.0),
        (("source_task_receipt", "action_uid"), 1),
    ),
)
def test_any_common_question_drift_is_rejected_or_changes_identity(path, value):
    tape = json.loads(TAPE_PATH.read_text())
    original = _receipt(tape)
    changed = copy.deepcopy(tape)
    changed[path[0]][path[1]] = value
    try:
        candidate = _receipt(changed)
    except M.ComparisonQuestionError:
        return
    assert candidate["comparison_question_sha256"] != original[
        "comparison_question_sha256"
    ]
    with pytest.raises(M.ComparisonQuestionError, match="not identical"):
        M.require_same_comparison_question(original, candidate)


def test_cannot_reseal_only_the_digest_or_use_one_self_receipt_as_pair_evidence():
    receipt = _receipt()
    forged = copy.deepcopy(receipt)
    forged["clock"]["time_to_contact_tick"] += 1
    with pytest.raises(M.ComparisonQuestionError, match="clock does not close|sha256 differs"):
        M.validate_comparison_question_receipt(forged)
    with pytest.raises(M.ComparisonQuestionError, match="not identical"):
        other = copy.deepcopy(receipt)
        other["identity"]["teacher_id"] = "other"
        payload = dict(other)
        payload.pop("comparison_question_sha256")
        other["comparison_question_sha256"] = M._digest(payload)
        M.require_same_comparison_question(receipt, other)


def test_builder_recomputes_tape_question_source_and_file_hashes():
    tape = json.loads(TAPE_PATH.read_text())
    raw = TAPE_PATH.read_bytes()
    with pytest.raises(M.ComparisonQuestionError, match="canonical supplied tape"):
        M.build_comparison_question_receipt(
            tape,
            tape_bytes=raw + b" ",
            action_id="take_061_unit04_bh",
            teacher_id="Take_061_unit04_BH",
            target_recipe="outcome_dense_only",
        )
    changed = copy.deepcopy(tape)
    changed["question_sha256"] = "0" * 64
    with pytest.raises(M.ComparisonQuestionError, match="question_sha256"):
        _receipt(changed)
    changed = copy.deepcopy(tape)
    changed["source_task_receipt"]["canonical_sha256"] = "0" * 64
    changed_unsigned = dict(changed)
    changed_unsigned.pop("canonical_sha256")
    changed["canonical_sha256"] = M._digest(changed_unsigned)
    with pytest.raises(M.ComparisonQuestionError, match="source task receipt"):
        _receipt(changed)


def test_self_resealed_nested_schema_drift_is_still_rejected():
    receipt = _receipt()
    malformed = copy.deepcopy(receipt)
    malformed["incoming_ball"]["extra"] = [0.0, 0.0, 0.0]
    unsigned = dict(malformed)
    unsigned.pop("comparison_question_sha256")
    malformed["comparison_question_sha256"] = M._digest(unsigned)
    with pytest.raises(M.ComparisonQuestionError, match="incoming-ball schema"):
        M.validate_comparison_question_receipt(malformed)
