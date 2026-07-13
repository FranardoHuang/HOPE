from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT / "configs" / "motion_backhand_loop_bc_se2_materialization_results_20260714.json"
EXAM = ROOT / "configs" / "phase1_signed_face_exam_bank_rebind_results_20260714.json"
BOOT = ROOT / "configs" / "phase1_signed_face_v8_d_boot_failure_20260714.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_values(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_values(child)
    else:
        yield value


def test_motion_pair_is_exactly_materialized_but_not_promoted() -> None:
    result = _load(MOTION)
    assert result["status"] == "complete_exact_pair_materialized_certificate_blocked"
    assert result["human_owner"] == "Franco"
    assert result["consumer"]["sha256"] == _sha256(
        ROOT / result["consumer"]["path"]
    )
    assert result["selection_result"]["sha256"] == _sha256(
        ROOT / result["selection_result"]["path"]
    )

    rows = result["assets"]
    assert [row["asset_id"] for row in rows] == [
        "franco_backhand_loop_b",
        "franco_backhand_loop_c",
    ]
    assert [row["candidate_id"] for row in rows] == [
        "98e7b883b29d302dc7a24fd3c564648c1f929ff2391e24e58558dcba58af3c14",
        "aa0c86fd350987bf30e56aebde9789bf9df430b0ec5c3c15cd235410794af299",
    ]
    expected_receipts = {
        "franco_backhand_loop_b": {
            "source": (27926, "90c23a8826397f13c39e5ca023c145c064dd5adfe49feb19043887897c60c17e"),
            "output": (27927, "278279125528c827e0a980389b040d54d16140620c59c67c878286be9d1c8ad6"),
            "report": (4051, "a238c077524586b2f1181cd24cb84ee29aa985ab274cfb43292f3159c0daadf3"),
        },
        "franco_backhand_loop_c": {
            "source": (30054, "4eb40301a51346fd3ad6cae52b13e93ca91b135f9eb9b38e16f7d89e456e9cb6"),
            "output": (30055, "0dd981a6d29c0c5321c905d1591a59fbb79763de6e43d92d4d76aefdc29ff48b"),
            "report": (4068, "b3b93d2cdb0a288f04aed764e5fdca92182cee625715a953600439088f59ff67"),
        },
    }
    for row in rows:
        expected = expected_receipts[row["asset_id"]]
        assert (row["source_motion"]["bytes"], row["source_motion"]["sha256"]) == expected["source"]
        assert (row["output_motion"]["bytes"], row["output_motion"]["sha256"]) == expected["output"]
        assert (row["report"]["bytes"], row["report"]["sha256"]) == expected["report"]
        assert row["report"]["status"] == "complete_exact_whole_motion_se2_materialization"
        assert row["publication"]["report_written_after_motion_observed"] is True
        assert max(
            row["invariants"]["root_position_inverse_max_abs_error"],
            row["invariants"]["root_quaternion_inverse_max_abs_error"],
            row["invariants"]["root_pairwise_distance_max_abs_error_m"],
        ) < 1e-12
        assert row["invariants"]["no_mirror"] is True
        assert row["invariants"]["no_resample_or_topp"] is True
        assert set(row["authorization"].values()) == {False}

    decision = result["pair_decision"]
    assert decision["materialized_motion_count"] == 2
    assert decision["certificate_count_before"] == decision["certificate_count_after"] == 0
    assert decision["accepted_motion_count"] == 0
    assert decision["only_unlocked_next_step"] == "schema2_materialization_preregistration"


def test_exam_rebind_is_e2_exact_data_only_and_keeps_consumers_blocked() -> None:
    result = _load(EXAM)
    assert result["status"] == "complete_exact_e2_data_gate_schedule_and_judge_blocked"
    assert result["preregistration"]["sha256"] == _sha256(
        ROOT / result["preregistration"]["path"]
    )
    assert result["consumer"]["sha256"] == _sha256(
        ROOT / result["consumer"]["path"]
    )
    assert result["input_exam_bank"]["question_counts"] == {
        "forehand": 183,
        "backhand": 188,
        "total": 371,
    }
    assert (result["input_exam_bank"]["bytes"], result["input_exam_bank"]["sha256"]) == (
        63968,
        "d7db2568beee990ef1d64b2dce9f0ab56ca76377f8993d820b6388292d0f5096",
    )

    execution = result["execution"]
    assert execution["no_write_validation_passed"] is True
    assert execution["run_published"] is True
    assert execution["non_metadata_array_count"] == 24
    assert execution["question_arrays_changed"] is False
    assert execution["legacy_load_used"] is False
    assert set(execution["old_new_all_output_bytes_equal"].values()) == {True}
    assert execution["landing_valid_count"] == {"forehand": 183, "backhand": 188}
    assert execution["net_clear_count"] == {"forehand": 183, "backhand": 188}
    assert result["output"]["report"]["report_written_last"] is True
    assert (result["output"]["bank"]["bytes"], result["output"]["bank"]["sha256"]) == (
        63643,
        "60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca",
    )
    assert (
        result["output"]["report"]["bytes"],
        result["output"]["report"]["sha256"],
        result["output"]["report"]["content_sha256"],
    ) == (
        18795,
        "dd4332edb47f1fb1f4d51ca00ceed612dbcadf9e395eb536c9b73bef9de69ad0",
        "7bdf4d6c2fccaf0b377e6bb76188c0b1d9abd1cd67cd322cbca2b1539a8a19d4",
    )

    decision = result["decision"]
    assert decision["evidence_level"] == "E2"
    assert decision["data_gate_passed"] is True
    assert decision["bank_adopted_as_exact_rebound_exam_input"] is True
    for key in (
        "immutable_schedule_materialized",
        "paper_activation_materialized",
        "judge_authorized",
        "l2_training_authorized",
        "formal_score_authorized",
        "gate3_authorized",
        "hardware_authorized",
    ):
        assert decision[key] is False


def test_v8_d_failure_preserves_serial_truth_and_forbids_blind_retry() -> None:
    result = _load(BOOT)
    assert result["status"] == "terminal_pre_contract_boot_timeout_automatic_retry_forbidden"
    assert result["source"]["clean"] is True
    assert result["source"]["bundle"]["advertised_commit"] == result["source"]["commit"]
    assert result["source"]["bundle"]["sha256"] == (
        "adf1c0be8a1e066f80dc96011c799d6eab99cc5e610a08d9234d6a6af4f1efc3"
    )
    assert (result["control"]["manifest_bytes"], result["control"]["manifest_sha256"]) == (
        21863,
        "f786da9f6f46c2bebba73edbf786a1c2cdf9eb2ecaa49820ffa56b9575ec8029",
    )
    assert (result["control"]["launcher_bytes"], result["control"]["launcher_sha256"]) == (
        158209,
        "58e798fc7122e702195fe9f8c26ee8cb20874f78b8ab714165cb8281dafa6afa",
    )
    assert result["control"]["locked_wrapper_bytes"] == 3170
    assert result["control"]["locked_wrapper_sha256"] == (
        "b250ec6d1cb3700bd45b7ede79e3d124125a0ae586a12dee16510b7cf647fa14"
    )
    serial = result["serial_context"]
    assert serial["cell_order"] == ["A", "B", "C", "D"]
    assert serial["d_launch_ordinal_zero_based"] == 3
    assert serial["v6_artifacts_adopted"] is False
    assert serial["v8_a_b_c_predecessors_reported_terminal"] is True
    assert serial["ledger_directly_archives_full_a_b_c_reaudit"] is False
    predecessor = serial["direct_predecessor_receipt"]
    assert predecessor["cell_id"] == "C"
    assert predecessor["checkpoint_iteration"] == 24
    assert predecessor["terminal_process_exit_observed"] is True

    failed = result["failed_cell"]
    assert failed["cell_id"] == "D"
    assert failed["boot_timeout_s"] == 900
    assert failed["pid"] == failed["pgid"] == 1782834
    assert failed["exit_code"] == 124
    assert failed["signal_scope"] == "exact_pgid"
    assert failed["signal_sent_by_locked_wrapper"] is True
    assert failed["failure_kind"] == "timeout_before_hard_contract"
    assert set(result["negative_evidence"].values()) == {False}
    assert result["artifacts"]["failure"] == {
        "path": "/workspace/codexschema/phase1_signed_face_rescue_20260713/runs/l1/phase1_signed_face_l1_v8_D_fresh_guidance_seed3/terminal_barrier_failure.json",
        "bytes": 4785,
        "sha256": "0e5bb13b072c2c1e029a93e9d2d037adbfc1d74da9e1c9450acea333173f98a9",
    }
    assert result["artifacts"]["launch_state"]["bytes"] == 3783
    assert result["artifacts"]["launch_state"]["sha256"] == (
        "80939e6df8184590df121f5edbf9f2de9188116438b6b328abd63152cdd2c90e"
    )
    assert result["artifacts"]["launch_contract"]["bytes"] == 13169
    assert result["artifacts"]["launch_contract"]["sha256"] == (
        "5649884dea8e1da690b51290a2eda583bca793ba36630883d48a350c2d192de5"
    )
    assert result["artifacts"]["run_log"]["bytes"] == 34731
    assert result["artifacts"]["run_log"]["sha256"] == (
        "5b2c91ac81963c1cef337ecdc3415e06fcfba4c410e21d72e721b8f3a530d43e"
    )

    decision = result["decision"]
    assert decision["automatic_retry_authorized"] is False
    assert decision["next_step"] == "boot_root_cause_before_any_new_versioned_attempt"
    for key in (
        "l1_activation_materialized",
        "l2_training_authorized",
        "judge_authorized",
        "second_seed_authorized",
        "formal_score_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        assert decision[key] is False
    assert result["final_pod1_audit"]["trainer_count"] == 0
    assert result["final_pod1_audit"]["gpu_compute_process_count"] == 0


def test_runtime_ledgers_have_no_ephemeral_local_snapshot_dependency() -> None:
    for path in (MOTION, EXAM, BOOT):
        values = [str(value) for value in _all_values(_load(path))]
        assert all("/private/tmp" not in value for value in values)
        assert all("/tmp/" not in value for value in values)
