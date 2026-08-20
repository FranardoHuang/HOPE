from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
TRANSITIONS = 4096 * 48
UID = 6907688916670928
COMMIT = "a" * 40
NAMESPACE = "mujoco-fullmdp-consumer-test-0001"
IDENTITY = {"source_commit": COMMIT, "run_namespace": NAMESPACE}
OTHER_IDENTITY = {
    "source_commit": "b" * 40,
    "run_namespace": "mujoco-fullmdp-consumer-other-0002",
}
RUNNER_EVENT_KEYS = {
    "reveal_rows", "reveal_due_rows", "reveal_deferred_rows", "launch_rows",
    "flight_terminal_rows", "shot_retired_rows", "selected_reset_rows",
    "completed_action_epoch_rows",
    "racket_contact_eligible_rows", "racket_contact_rows", "selected_contact_rows",
    "opposite_contact_rows", "edge_contact_rows", "between_contact_rows",
    "invalid_contact_rows", "r03_present_rows", "r03_physically_valid_rows",
    "landing_crossing_rows", "r06_present_rows", "r06_eligible_rows",
    "r06_common_rows", "r07_present_rows", "r07_eligible_rows",
    "recovery_success_rows", "recovery_failure_rows", "recovery_timeout_rows",
}
RUNNER_LIFECYCLE_KEYS = {
    "gym_reset_rows", "unknown_terminal_rows", "invalid_done_rows",
    "done_explanation_fault_rows", "time_out_rows", "timeout_fault_rows",
    "selected_reset_fault_rows", "reset_generation_rows",
    "reset_generation_fault_rows", "resolved_table_rows",
    "landing_on_opponent_rows", "landing_opponent_bound_rows",
    "classification_unknown_rows", "event_semantics_fault_rows",
}


def _load():
    path = LANE / "mujoco_full_mdp_longrun_consumer.py"
    name = "mujoco_full_mdp_longrun_consumer_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _base_record(module, index, *, identity=IDENTITY):
    events = {key: 0 for key in RUNNER_EVENT_KEYS}
    terminal = {key: 0 for key in module.TERMINAL_KEYS}
    lifecycle = {key: 0 for key in RUNNER_LIFECYCLE_KEYS}
    return {
        "schema_version": 2,
        "record_type": "mujoco_full_mdp_update_ack",
        "diagnostic_unauthorized": True,
        "update_index": index,
        "run_identity": dict(identity),
        "num_envs": 4096,
        "num_steps_per_env": 48,
        "transitions_delta": TRANSITIONS,
        "transitions_cumulative": TRANSITIONS * (index + 1),
        "environment_steps_delta": 48,
        "environment_steps_cumulative": 48 * (index + 1),
        "storage_finite": {"rewards": True, "returns": True, "advantages": True},
        "extras_counts": events,
        "terminal_bit_counts": terminal,
        "classification_status_counts": {
            "0": TRANSITIONS, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
        },
        "outcome_code_counts": {str(value): 0 for value in range(7)},
        # This is the runner/ledger ACK wire shape for an idle rollout. Phase
        # zero is valid before the portable cadence accepts a reveal.
        "phase_counts": {"0": TRANSITIONS, "2": 0, "5": 0, "6": 0, "8": 0},
        "episodes": {"completed_count": 0, "return_sum": 0.0, "length_sum": 0},
        "rollout_policy_mean_std": 0.02,
        "selected_reset_rows": 0,
        "gym_reset_rows": 0,
        "lifecycle_counts": lifecycle,
        "reward20": {
            "term_sums": [0.0] * 20,
            "actual_reward_sum": 0.0,
            "reward20_finite_rows": TRANSITIONS,
            "reward20_nonfinite_rows": 0,
            "actual_reward_finite_rows": TRANSITIONS,
            "actual_reward_nonfinite_rows": 0,
            "conservation_fault_rows": 0,
        },
        "action_identity": {
            "action_slot": 0,
            "action_uid": UID,
            "mount_normal_sign": 1,
            "family": "forehand",
            "family_source": "runner_pinned_identity",
            "observed_rows": TRANSITIONS,
            "slot0_rows": TRANSITIONS,
            "uid_rows": TRANSITIONS,
            "mount_sign_rows": TRANSITIONS,
            "identity_rows": TRANSITIONS,
            "family_counts": {"forehand": TRANSITIONS, "backhand": 0},
        },
    }


def _add_business_chain(row):
    events = row["extras_counts"]
    events.update({
        "reveal_due_rows": 1,
        "reveal_rows": 1,
        "launch_rows": 1,
        "racket_contact_eligible_rows": 1,
        "racket_contact_rows": 1,
        "selected_contact_rows": 1,
        "r03_present_rows": 1,
        "r03_physically_valid_rows": 1,
        "landing_crossing_rows": 1,
        "flight_terminal_rows": 1,
        "r06_present_rows": 1,
        "r06_eligible_rows": 1,
        "r06_common_rows": 1,
        "r07_present_rows": 1,
        "r07_eligible_rows": 1,
        "recovery_success_rows": 1,
        "shot_retired_rows": 1,
        "completed_action_epoch_rows": 1,
    })
    row["classification_status_counts"].update({"0": TRANSITIONS - 1, "1": 1})
    row["outcome_code_counts"]["3"] = 1
    row["phase_counts"] = {
        "0": TRANSITIONS - 4, "2": 1, "5": 1, "6": 1, "8": 1,
    }
    row["lifecycle_counts"].update({
        "landing_on_opponent_rows": 1,
        "landing_opponent_bound_rows": 1,
    })


def _finalize(row):
    raw = json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    row["prepared_update_sha256"] = hashlib.sha256(raw).hexdigest()
    row["snapshot"] = None
    row["optimizer_metrics"] = {
        "value_function": 0.25, "surrogate": -0.125, "entropy": 1.5,
    }
    row["learning_rate"] = 1.0e-3
    row["timings"] = {
        "collection_seconds": 1.0,
        "learning_seconds": 0.5,
        "pre_ack_iteration_seconds": 1.6,
        "run_elapsed_pre_ack_seconds": 2.0 * (row["update_index"] + 1),
    }
    return row


def _records(module, count, *, identity=IDENTITY, business=False, mutation=None):
    rows = [_base_record(module, index, identity=identity) for index in range(count)]
    if business:
        _add_business_chain(rows[0])
    if mutation is not None:
        mutation(rows)
    return [_finalize(row) for row in rows]


def _payload(module, index, row, *, identity=IDENTITY, toy=False):
    if toy:
        model = {"weight": torch.zeros(1)}
        optimizer = {
            "state": {0: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }},
            "param_groups": [{"lr": 1.0e-3, "params": [0]}],
        }
    else:
        model = {}
        state = {}
        for param_id, (name, shape) in enumerate(module.MODEL_SHAPES):
            # Shared references keep unit-test files small; shape, dtype and
            # finite validation remains the exact production ABI.
            tensor = torch.zeros(shape, dtype=torch.float32)
            model[name] = tensor
            state[param_id] = {
                "step": torch.tensor(float(index + 1)),
                "exp_avg": tensor,
                "exp_avg_sq": tensor,
            }
        optimizer = {
            "state": state,
            "param_groups": [{
                "lr": 1.0e-3,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "params": list(range(len(module.MODEL_SHAPES))),
            }],
        }
    return {
        "model_state_dict": model,
        "optimizer_state_dict": optimizer,
        "iter": index,
        "infos": {
            "diagnostic_unauthorized": True,
            "checkpoint_authority": False,
            "resume_authority": False,
            "update_index": index,
            "completed_updates": index + 1,
            "run_identity": dict(identity),
            "action_ball_full_mdp_ppo_recipe_sha256": (
                module.FULL_MDP_PPO_RECIPE_SHA256
            ),
            "prepared_update_sha256": row["prepared_update_sha256"],
        },
    }


def _write_snapshots(module, root, rows, *, complete, identity=IDENTITY,
                     toy=False, mutate_payload=None):
    receipts = []
    for index in module._snapshot_indices(complete):
        payload = _payload(module, index, rows[index], identity=identity,
                           toy=toy and index == 0)
        if mutate_payload is not None:
            mutate_payload(index, payload)
        path = root / f"model_{index}.pt"
        torch.save(payload, path)
        raw = path.read_bytes()
        receipt = {
            "name": path.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        rows[index]["snapshot"] = receipt
        receipts.append(receipt)
    return receipts


def _write_evidence(path, rows):
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8") + b"\n"
        for row in rows
    )
    path.write_bytes(raw)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_completion(path, module, count, evidence_inventory, receipts,
                      *, identity=IDENTITY, mutation=None):
    record = {
        "schema_version": 2,
        "record_type": "mujoco_full_mdp_completion",
        "diagnostic_unauthorized": True,
        "run_identity": dict(identity),
        "num_envs": 4096,
        "num_steps_per_env": 48,
        "completed_updates": count,
        "environment_steps": 48 * count,
        "transitions": TRANSITIONS * count,
        "evidence_jsonl": evidence_inventory,
        "snapshot_receipts": receipts,
        "final_observation_finite": True,
        "rollout_storage_finite": True,
        "optimizer_state_present": True,
        "optimizer_state_finite": True,
        "checkpoint_authority": False,
        "resume_authority": False,
        "action_contract": dict(module.ACTION_CONTRACT),
        "action_ball_full_mdp_ppo_recipe_sha256": (
            module.FULL_MDP_PPO_RECIPE_SHA256
        ),
    }
    if mutation is not None:
        mutation(record)
    path.write_bytes(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n")


def _artifacts(module, tmp_path, count, *, complete, identity=IDENTITY,
               business=False, row_mutation=None, toy=False,
               snapshot_identity=None, snapshot_mutation=None, seal=True,
               seal_mutation=None):
    evidence = tmp_path / "updates.jsonl"
    snapshots = tmp_path / "snapshots"
    completion = tmp_path / "completion.json"
    snapshots.mkdir()
    rows = _records(
        module, count, identity=identity, business=business, mutation=row_mutation
    )
    receipts = _write_snapshots(
        module, snapshots, rows, complete=complete,
        identity=snapshot_identity or identity, toy=toy,
        mutate_payload=snapshot_mutation,
    )
    evidence_inventory = _write_evidence(evidence, rows)
    if seal:
        _write_completion(
            completion, module, count, evidence_inventory, receipts,
            identity=identity, mutation=seal_mutation,
        )
    return evidence, snapshots, completion, rows


def _consume(module, evidence, snapshots, count, completion=None,
             *, commit=COMMIT, namespace=NAMESPACE):
    return module.consume(
        evidence,
        expected_updates=count,
        expected_source_commit=commit,
        expected_run_namespace=namespace,
        snapshot_dir=snapshots,
        completion_json=completion,
    )


def test_prefix_five_verifies_model_zero_but_stays_advisory(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 5, complete=False
    )
    summary = _consume(module, evidence, snapshots, 5)
    assert summary["evidence_level"] == "advisory_prefix"
    assert summary["engineering_run_complete"] is False
    assert summary["business_chain_complete"] is False
    assert summary["full_a_complete"] is False
    assert summary["snapshot_count"] == 1
    assert summary["snapshot_inventory"][0]["name"] == "model_0.pt"
    assert summary["model_abi_verified"] is True
    assert summary["optimizer_state_verified"] is True
    assert summary["completion_seal_verified"] is False
    assert summary["action_contract"] is None
    assert summary["opportunity_d05"] == {
        "status": "not_produced", "denominator": None,
    }
    assert summary["portable_reveal_opportunity"] == {
        "due_rows": 0, "accepted_rows": 0, "deferred_rows": 0,
        "accept_rate": None, "defer_rate": None,
    }
    assert summary["action_coverage"]["backhand"] == {
        "status": "未测", "observed_rows": 0, "denominator": 0,
    }
    assert summary["rates"] == {
        "selected_contact_per_eligible": None,
        "opponent_landing_per_crossing": None,
        "recovery_success_per_terminal": None,
    }


def test_runner_update_ack_fixture_matches_exact_v3_wire(tmp_path):
    module = _load()
    assert module.EVENT_KEYS == RUNNER_EVENT_KEYS
    assert module.LIFECYCLE_KEYS == RUNNER_LIFECYCLE_KEYS
    assert len(module.EVENT_KEYS) == 26
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["reveal_due_rows"] == 0
    assert summary["engineering_run_complete"] is False


def test_snapshot_schedule_preserves_twenty_six_artifacts_for_v2():
    module = _load()
    indices = module._snapshot_indices(True)
    assert indices == list(range(0, 12_500, 500)) + [12_499]
    assert len(indices) == 26


def test_sealed_all_zero_longrun_is_engineering_not_business_completion(
        monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True
    )
    summary = _consume(module, evidence, snapshots, 2, completion)
    assert summary["engineering_run_complete"] is True
    assert summary["business_chain_complete"] is False
    assert summary["full_a_complete"] is False
    assert summary["completion_seal_verified"] is True
    assert summary["action_contract"] == module.ACTION_CONTRACT
    assert summary["action_contract"]["transfer_authority"] is False
    assert summary["action_contract"]["matched_cross_backend_authority"] is False
    assert "selected_contact_rows" in summary["business_chain_missing"]
    assert all(value == 0 for value in summary["milestones"].values())


def test_sealed_slot0_chain_never_claims_formal_full_a_completion(
        monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True, business=True
    )
    summary = _consume(module, evidence, snapshots, 2, completion)
    assert summary["diagnostic_unauthorized"] is True
    assert summary["engineering_run_complete"] is True
    assert summary["business_chain_complete"] is True
    assert summary["full_a_complete"] is False
    assert summary["business_chain_missing"] == []
    assert summary["milestones"]["selected_contact_rows"] == 1
    assert summary["milestones"]["selected_reset_rows"] == 0
    assert summary["milestones"]["gym_reset_rows"] == 0
    assert summary["portable_reveal_opportunity"] == {
        "due_rows": 1, "accepted_rows": 1, "deferred_rows": 0,
        "accept_rate": 1.0, "defer_rate": 0.0,
    }
    assert summary["rates"] == {
        "selected_contact_per_eligible": 1.0,
        "opponent_landing_per_crossing": 1.0,
        "recovery_success_per_terminal": 1.0,
    }


def test_cross_env_marginals_cannot_complete_one_action_epoch(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)

    def drop_joint_event(rows):
        rows[0]["extras_counts"]["completed_action_epoch_rows"] = 0

    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True, business=True,
        row_mutation=drop_joint_event,
    )
    summary = _consume(module, evidence, snapshots, 2, completion)
    assert summary["engineering_run_complete"] is True
    assert summary["business_chain_complete"] is False
    assert summary["full_a_complete"] is False
    assert summary["business_chain_missing"] == ["completed_action_epoch_rows"]


def test_portable_due_accept_defer_rates_use_due_denominator(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"].update({
            "reveal_due_rows": 2, "reveal_deferred_rows": 1,
        })

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, business=True, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["portable_reveal_opportunity"] == {
        "due_rows": 2, "accepted_rows": 1, "deferred_rows": 1,
        "accept_rate": 0.5, "defer_rate": 0.5,
    }


def test_complete_rejects_missing_final_seal(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True, seal=False
    )
    with pytest.raises(ValueError, match="completion receipt open"):
        _consume(module, evidence, snapshots, 2, completion)


def test_complete_rejects_false_final_gate_and_seal_inventory_drift(
        monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True,
        seal_mutation=lambda row: row.__setitem__("optimizer_state_finite", False),
    )
    with pytest.raises(ValueError, match="completion seal binding"):
        _consume(module, evidence, snapshots, 2, completion)


def test_complete_rejects_action_contract_drift(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True,
        seal_mutation=lambda row: row["action_contract"].__setitem__(
            "raw_action_clip", 4.0
        ),
    )
    with pytest.raises(ValueError, match="completion seal binding"):
        _consume(module, evidence, snapshots, 2, completion)


def test_rejects_cross_run_evidence_identity(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, identity=OTHER_IDENTITY
    )
    with pytest.raises(ValueError, match="run identity at update 0"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_cross_run_snapshot_infos_even_with_matching_receipt(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, snapshot_identity=OTHER_IDENTITY
    )
    with pytest.raises(ValueError, match="snapshot infos binding"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_finite_toy_snapshot_that_is_not_exact_model_abi(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, toy=True
    )
    with pytest.raises(ValueError, match="snapshot model ABI"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_optimizer_state_shape_drift(tmp_path):
    module = _load()

    def mutate(index, payload):
        if index == 0:
            payload["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.zeros(1)

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, snapshot_mutation=mutate
    )
    with pytest.raises(ValueError, match="optimizer state shape"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_snapshot_ack_receipt_not_bound_to_same_file(tmp_path):
    module = _load()
    evidence, snapshots, _completion, rows = _artifacts(
        module, tmp_path, 1, complete=False
    )
    rows[0]["snapshot"]["sha256"] = "0" * 64
    _write_evidence(evidence, rows)
    with pytest.raises(ValueError, match="snapshot ACK receipt binding"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_bad_event_subset_even_when_fault_counter_claims_zero(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"]["r06_common_rows"] = 1

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="event subset conservation"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_legacy_phase_seven_wire_key(tmp_path):
    module = _load()

    def phase_mutation(rows):
        rows[0]["phase_counts"] = {
            "0": TRANSITIONS, "2": 0, "5": 0, "6": 0, "7": 0,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=phase_mutation
    )
    with pytest.raises(ValueError, match="phase_counts keys"):
        _consume(module, evidence, snapshots, 1)


def test_gym_reset_recovery_failure_is_not_a_shot_retirement(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "selected_reset_rows": 1, "recovery_failure_rows": 1,
        })
        row["selected_reset_rows"] = 1
        row["gym_reset_rows"] = 1
        row["terminal_bit_counts"]["base_fell_tilt"] = 1
        row["lifecycle_counts"].update({
            "gym_reset_rows": 1, "reset_generation_rows": 1,
        })
        row["episodes"] = {"completed_count": 1, "return_sum": 0.0,
                           "length_sum": 1}

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["recovery_failure_rows"] == 1
    assert summary["milestones"]["shot_retired_rows"] == 0
    assert "shot_retired_rows" in summary["business_chain_missing"]


def test_rejects_recovery_failure_masquerading_as_shot_retirement(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"].update({
            "shot_retired_rows": 1, "recovery_failure_rows": 1,
        })

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="event subset conservation"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_prepared_hash_drift_and_duplicate_json_key(tmp_path):
    module = _load()
    evidence, snapshots, _completion, rows = _artifacts(
        module, tmp_path, 1, complete=False
    )
    rows[0]["extras_counts"]["reveal_rows"] = 1
    _write_evidence(evidence, rows)
    with pytest.raises(ValueError, match="prepared update hash"):
        _consume(module, evidence, snapshots, 1)

    record = json.dumps(rows[0])
    evidence.write_text(record[:-1] + ',"schema_version":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _consume(module, evidence, snapshots, 1)


def test_artifact_mode_is_exact(tmp_path):
    module = _load()
    evidence = tmp_path / "updates.jsonl"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    with pytest.raises(ValueError, match="artifact mode"):
        module.consume(
            evidence,
            expected_updates=1,
            expected_source_commit=COMMIT,
            expected_run_namespace=NAMESPACE,
            snapshot_dir=None,
        )
    with pytest.raises(ValueError, match="expected update count"):
        module.consume(
            evidence,
            expected_updates=2,
            expected_source_commit=COMMIT,
            expected_run_namespace=NAMESPACE,
            snapshot_dir=snapshots,
        )
