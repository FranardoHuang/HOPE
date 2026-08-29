from __future__ import annotations

import hashlib
import importlib.util
import json
import copy
from pathlib import Path
import re
import sys
import types

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
PPO_RECIPE_PATH = LANE.parent / (
    "source/whole_body_tracking/action_ball_full_mdp_ppo_recipe.py"
)
PLANT_XML = LANE.parents[2] / (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3p_pingpong_0807/a3p_pingpong_0807.xml"
)


def _load_ppo_recipe():
    spec = importlib.util.spec_from_file_location(
        "_mujoco_consumer_test_ppo_recipe", PPO_RECIPE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ACTION_BALL_FULL_MDP_PPO_RECIPE


PPO_RECIPE = _load_ppo_recipe()
NUM_ENVS = PPO_RECIPE.num_envs
STEPS_PER_UPDATE = PPO_RECIPE.num_steps_per_env
TRANSITIONS = NUM_ENVS * STEPS_PER_UPDATE
UID = 2552478955674699
COMMIT = "a" * 40
NAMESPACE = "mujoco-fullmdp-consumer-test-0001"
MUJOCO_WARP_RUNTIME = {
    "schema_version": 1,
    "distribution": "mujoco-warp",
    "fork_id": "hope_mujoco_warp_epa48_v1",
    "version": "3.10.0.3+hope.epa48.1",
    "epa_horizon": 48,
    "types_py_sha256": (
        "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696"
    ),
    "wheel_sha256": (
        "58f47b1c3b4249d82666f25d3a302ff5a215043a3d7a3b9445a5ca7ef15b561a"
    ),
    "build_receipt_sha256": (
        "336f6454296d3c062e26fb0c330d6dbca4b2fd0ad4e50f386f8a647db013e041"
    ),
    "import_scope": "fresh_run_local_site",
}
RSL_RL_RUNTIME = {
    "distribution": "rsl-rl-lib",
    "version": "3.1.2",
    "wheel_sha256": (
        "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
    ),
    "import_scope": "fresh_run_local_site",
}
MJLAB_RUNTIME = {
    "schema_version": 1,
    "distribution": "mjlab",
    "version": "1.5.3",
    "import_scope": "verified_venv_distribution",
    "selected_tree_scope": "mjlab/**/*.py+mjlab/scene/scene.xml",
    "selected_file_count": 193,
    "selected_byte_count": 1_399_177,
    "selected_tree_sha256": (
        "88c9725d0416b4ac3e21f6752ad423c13ea3b8cfb9e23ca664f8aba146cec33d"
    ),
    "mjlab_tasks_entry_point_count": 0,
}


def _runtime_stack():
    return {
        "schema_version": 1,
        "mujoco_warp": dict(MUJOCO_WARP_RUNTIME),
        "rsl_rl": dict(RSL_RL_RUNTIME),
        "mjlab": dict(MJLAB_RUNTIME),
    }
VERIFICATION_RECEIPT_SHA256 = "c" * 64
OWNER_LOCAL_FRAME_SHA256 = "d" * 64


def _plant_contract():
    name = "_consumer_test_mujoco_full_mdp_plant_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = LANE / "mujoco_full_mdp_plant_contract.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLANT_MODEL = _plant_contract().verified_plant_model_identity(
    verification_receipt_sha256=VERIFICATION_RECEIPT_SHA256,
    owner_local_frame_sha256=OWNER_LOCAL_FRAME_SHA256,
    final_augmented_mjb=_plant_contract().expected_plant_model_identity()[
        "runtime_attach"
    ]["final_augmented_mjb"],
)
IDENTITY = {
    "source_commit": COMMIT,
    "run_namespace": NAMESPACE,
    "runtime_stack": _runtime_stack(),
    "plant_model": PLANT_MODEL,
}
OTHER_IDENTITY = {
    "source_commit": "b" * 40,
    "run_namespace": "mujoco-fullmdp-consumer-other-0002",
    "runtime_stack": _runtime_stack(),
    "plant_model": PLANT_MODEL,
}


def _copy_identity(identity):
    return {
        "source_commit": identity["source_commit"],
        "run_namespace": identity["run_namespace"],
        "runtime_stack": {
            "schema_version": identity["runtime_stack"]["schema_version"],
            "mujoco_warp": dict(identity["runtime_stack"]["mujoco_warp"]),
            "rsl_rl": dict(identity["runtime_stack"]["rsl_rl"]),
            "mjlab": dict(identity["runtime_stack"]["mjlab"]),
        },
        "plant_model": _plant_contract().clone_plant_model_identity(
            identity["plant_model"]
        ),
    }


RUNNER_EVENT_KEYS = {
    "scheduled_due_rows", "due_terminal_overlap_rows",
    "reveal_rows", "reveal_due_rows", "reveal_deferred_rows", "launch_rows",
    "missed_launch_rows",
    "flight_terminal_rows", "shot_retired_rows", "selected_reset_rows",
    "completed_action_epoch_rows",
    "racket_contact_rows", "selected_contact_rows",
    "opposite_contact_rows", "edge_contact_rows", "between_contact_rows",
    "invalid_contact_rows", "actual_hard_edge_rows",
    "qdes_guard_intervention_rows", "r03_present_rows",
    "r03_physically_valid_rows",
    "landing_crossing_rows", "r06_present_rows", "r06_eligible_rows",
    "r06_common_rows", "r07_present_rows", "r07_eligible_rows",
    "recovery_success_rows", "recovery_failure_rows", "recovery_timeout_rows",
    "recovery_completion_fault_rows",
}
RUNNER_LIFECYCLE_KEYS = {
    "gym_reset_rows", "unknown_terminal_rows", "invalid_done_rows",
    "done_explanation_fault_rows", "time_out_rows", "timeout_fault_rows",
    "selected_reset_fault_rows", "reset_generation_rows",
    "reset_generation_fault_rows", "resolved_table_rows",
    "landing_on_opponent_rows", "landing_opponent_bound_rows",
    "classification_unknown_rows",
}
RUNNER_FACT_INTEGRITY_KEYS = {
    "fact_integrity_r03_nonfinite_rows",
    "fact_integrity_r06_source_invalid_rows",
    "fact_integrity_r07_sequence_rows",
    "fact_integrity_r07_nonfinite_rows",
    "fact_integrity_unknown_bits_rows",
}


def _load(*, stub_runtime_mjb=True):
    path = LANE / "mujoco_full_mdp_longrun_consumer.py"
    name = "mujoco_full_mdp_longrun_consumer_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    source = _plant_contract().expected_plant_model_identity()["source_plant"]

    def verify(*, mjcf_path, **_kwargs):
        path = Path(mjcf_path)
        if (
            path.name != source["root_filename"]
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != source["root_mjcf_sha256"]
        ):
            raise ValueError("fixture plant root differs")
        class Verified:
            portable_identity_sha256 = source["portable_identity_sha256"]
            verification_receipt_sha256 = VERIFICATION_RECEIPT_SHA256

            def consume_verified_model(self, consumer):
                return consumer(object())

        return Verified()

    module._canonical_mujoco_identity_module = lambda: types.SimpleNamespace(
        verify_exact_mujoco_identity=verify
    )
    module._mujoco_module = lambda: object()
    module._table_termination_module = lambda: types.SimpleNamespace(
        consume_verified_owner_frame_contract=lambda _mujoco, verified: (
            verified.consume_verified_model(
                lambda _model: {
                    "content_sha256": OWNER_LOCAL_FRAME_SHA256,
                }
            )
        )
    )
    runtime_verification = object()
    module._epa48_runtime_module = lambda: types.SimpleNamespace(
        verify_runtime_stack_preimport=lambda: runtime_verification,
        verified_runtime_stack_identity=lambda verification: (
            _runtime_stack()
            if verification is runtime_verification
            else pytest.fail("consumer runtime verification token differs")
        ),
    )
    if stub_runtime_mjb:
        module._verified_runtime_mjb = lambda _evidence: dict(
            _plant_contract().expected_plant_model_identity()["runtime_attach"][
                "final_augmented_mjb"
            ]
        )
    return module


def _base_record(module, index, *, identity=IDENTITY):
    events = {key: 0 for key in RUNNER_EVENT_KEYS}
    terminal = {key: 0 for key in module.TERMINAL_KEYS}
    lifecycle = {key: 0 for key in RUNNER_LIFECYCLE_KEYS}
    return {
        "schema_version": 10,
        "record_type": "mujoco_full_mdp_update_ack",
        "diagnostic_unauthorized": True,
        "update_index": index,
        "run_identity": _copy_identity(identity),
        "num_envs": NUM_ENVS,
        "num_steps_per_env": STEPS_PER_UPDATE,
        "transitions_delta": TRANSITIONS,
        "transitions_cumulative": TRANSITIONS * (index + 1),
        "environment_steps_delta": STEPS_PER_UPDATE,
        "environment_steps_cumulative": STEPS_PER_UPDATE * (index + 1),
        "storage_finite": {
            "observations_policy": True,
            "observations_critic": True,
            "actions": True,
            "values": True,
            "actions_log_prob": True,
            "mu": True,
            "sigma": True,
            "rewards": True,
            "returns": True,
            "advantages": True,
        },
        "storage_domains": {
            "dones_binary": True, "sigma_positive": True,
        },
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
        "fact_integrity_counts": {
            key: 0 for key in RUNNER_FACT_INTEGRITY_KEYS
        },
        "reward_graph": {
            "term_names": list(module.REWARD_TERM_NAMES),
            "term_count": module.REWARD_TERM_COUNT,
            "term_sums": [0.0] * module.REWARD_TERM_COUNT,
            "actual_reward_sum": 0.0,
            "reward_terms_finite_rows": TRANSITIONS,
            "reward_terms_nonfinite_rows": 0,
            "actual_reward_finite_rows": TRANSITIONS,
            "actual_reward_nonfinite_rows": 0,
            "conservation_fault_rows": 0,
            "playback_paddle_prior": {
                "term_names": list(module.PADDLE_PRIOR_TERM_NAMES),
                "row_count": 0,
                "finite_rows": [0] * module.PADDLE_PRIOR_TERM_COUNT,
                "kernel_sum": [0.0] * module.PADDLE_PRIOR_TERM_COUNT,
                "kernel_sumsq": [0.0] * module.PADDLE_PRIOR_TERM_COUNT,
                "domain_violation_rows": [
                    0
                ] * module.PADDLE_PRIOR_TERM_COUNT,
                "error_names": [
                    "position", "velocity", "signed_face", "long_axis"
                ],
                "error_units": ["m", "m_per_s", "rad", "rad"],
                "error_finite_rows": [0] * module.PADDLE_PRIOR_TERM_COUNT,
                "error_sum": [0.0] * module.PADDLE_PRIOR_TERM_COUNT,
                "error_sumsq": [0.0] * module.PADDLE_PRIOR_TERM_COUNT,
            },
        },
        "action_identity": {
            "action_slot": 0,
            "action_uid": UID,
            "mount_normal_sign": 1,
            "family": "backhand",
            "family_source": "runner_pinned_identity",
            "observed_rows": TRANSITIONS,
            "slot0_rows": TRANSITIONS,
            "uid_rows": TRANSITIONS,
            "mount_sign_rows": TRANSITIONS,
            "identity_rows": TRANSITIONS,
            "family_counts": {"forehand": 0, "backhand": TRANSITIONS},
        },
    }


def _add_producer_attested_milestone_coverage(row):
    events = row["extras_counts"]
    events.update({
        "scheduled_due_rows": 1,
        "reveal_due_rows": 1,
        "reveal_rows": 1,
        "launch_rows": 1,
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


def _records(
    module, count, *, identity=IDENTITY, milestone_coverage=False, mutation=None
):
    rows = [_base_record(module, index, identity=identity) for index in range(count)]
    if milestone_coverage:
        _add_producer_attested_milestone_coverage(rows[0])
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
            "run_identity": _copy_identity(identity),
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
        "schema_version": module.COMPLETION_SCHEMA_VERSION,
        "record_type": "mujoco_full_mdp_completion",
        "diagnostic_unauthorized": True,
        "run_identity": _copy_identity(identity),
        "num_envs": NUM_ENVS,
        "num_steps_per_env": STEPS_PER_UPDATE,
        "completed_updates": count,
        "environment_steps": STEPS_PER_UPDATE * count,
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
               milestone_coverage=False, row_mutation=None, toy=False,
               snapshot_identity=None, snapshot_mutation=None, seal=True,
               seal_mutation=None):
    evidence = tmp_path / "updates.jsonl"
    snapshots = tmp_path / "snapshots"
    completion = tmp_path / "completion.json"
    snapshots.mkdir()
    rows = _records(
        module, count, identity=identity,
        milestone_coverage=milestone_coverage, mutation=row_mutation,
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
             *, commit=COMMIT, namespace=NAMESPACE, plant_xml=PLANT_XML):
    return module.consume(
        evidence,
        expected_updates=count,
        expected_source_commit=commit,
        expected_run_namespace=namespace,
        expected_plant_xml=plant_xml,
        snapshot_dir=snapshots,
        completion_json=completion,
    )


def test_prefix_five_verifies_model_zero_but_stays_advisory(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 5, complete=False
    )
    summary = _consume(module, evidence, snapshots, 5)
    assert summary["schema_version"] == 6
    assert summary["run_identity"] == IDENTITY
    assert summary["evidence_level"] == "advisory_prefix"
    assert summary["engineering_run_complete"] is False
    assert summary["producer_attested_milestone_coverage_complete"] is False
    assert summary["same_epoch_chain_replay_status"] == "not_produced"
    assert summary["full_a_complete"] is False
    assert summary["snapshot_count"] == 1
    assert summary["snapshot_inventory"][0]["name"] == "model_0.pt"
    assert summary["model_abi_verified"] is True
    assert summary["optimizer_state_verified"] is True
    assert summary["runtime_mjb_verified"] is True
    assert summary["completion_seal_verified"] is False
    assert summary["action_contract"] is None
    assert summary["opportunity_d05"] == {
        "status": "not_produced", "denominator": None,
    }
    assert summary["portable_reveal_opportunity"] == {
        "scheduled_rows": 0,
        "terminal_overlap_rows": 0,
        "due_rows": 0, "accepted_rows": 0, "deferred_rows": 0,
        "accept_rate": None, "defer_rate": None,
    }
    assert summary["action_coverage"]["forehand"] == {
        "status": "未测", "observed_rows": 0, "denominator": 0,
    }
    assert summary["action_coverage"]["backhand"] == {
        "status": "observed",
        "observed_rows": 5 * TRANSITIONS,
        "denominator": 5 * TRANSITIONS,
    }
    assert summary["hit_opportunity_r03"] == {
        "present_rows": 0,
        "physically_valid_rows": 0,
        "selected_contact_rows": 0,
        "selected_contact_to_physically_valid_ratio": None,
    }
    assert summary["rates"] == {
        "selected_contact_per_launch": None,
        "r03_physically_valid_per_present": None,
        "r06_common_per_eligible": None,
        "opponent_landing_per_crossing": None,
        "recovery_success_per_terminal": None,
    }


def test_runner_update_ack_fixture_matches_exact_evidence_v10_wire(tmp_path):
    module = _load()
    assert module.EVENT_KEYS == RUNNER_EVENT_KEYS
    assert module.LIFECYCLE_KEYS == RUNNER_LIFECYCLE_KEYS
    assert module.FACT_INTEGRITY_KEYS == RUNNER_FACT_INTEGRITY_KEYS
    assert len(module.EVENT_KEYS) == 31
    assert module.REWARD_TERM_COUNT == len(module.REWARD_TERM_NAMES)
    assert module.REWARD_TERM_NAMES == (
        module._reward_contract_module().MANAGER_NAMES
    )
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["reveal_due_rows"] == 0
    assert summary["engineering_run_complete"] is False


@pytest.mark.parametrize(
    "case,error",
    (
        ("legacy_top_key", "top-level keys"),
        ("term_names", "reward graph term contract"),
        ("term_count", "reward graph term contract"),
        ("term_sums", "reward graph term contract"),
        ("extra_key", "reward graph keys"),
        ("legacy_finite_key", "reward graph keys"),
    ),
)
def test_reward_graph_wire_rejects_legacy_or_nonexact_contract(
    case, error, tmp_path
):
    module = _load()

    def mutate(rows):
        row = rows[0]
        reward = row["reward_graph"]
        if case == "legacy_top_key":
            row["reward20"] = row.pop("reward_graph")
        elif case == "term_names":
            reward["term_names"][:2] = reversed(reward["term_names"][:2])
        elif case == "term_count":
            reward["term_count"] += 1
        elif case == "term_sums":
            reward["term_sums"].pop()
        elif case == "extra_key":
            reward["extra"] = 0
        else:
            reward["reward20_finite_rows"] = reward.pop(
                "reward_terms_finite_rows"
            )

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match=error):
        _consume(module, evidence, snapshots, 1)


def test_playback_paddle_prior_wire_accepts_non_gating_finite_moments(
    tmp_path,
):
    module = _load()

    def mutate(rows):
        rows[0]["reward_graph"]["playback_paddle_prior"].update({
            "row_count": 3,
            "finite_rows": [3, 3, 3, 3],
            "kernel_sum": [1.5, 1.5, 1.2, 2.4],
            "kernel_sumsq": [1.875, 1.25, 0.48, 1.92],
            "domain_violation_rows": [2, 0, 0, 0],
            "error_finite_rows": [3, 3, 2, 3],
            "error_sum": [0.6, 1.5, 0.5, 2.4],
            "error_sumsq": [0.14, 0.77, 0.13, 1.94],
        })

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["engineering_run_complete"] is False


@pytest.mark.parametrize(
    "case,error",
    (
        ("missing_key", "playback paddle prior keys"),
        ("term_names", "playback paddle prior term contract"),
        ("row_count", "playback paddle prior row_count"),
        ("finite_rows", "playback paddle prior finite_rows"),
        ("domain_rows", "playback paddle prior domain_violation_rows"),
        ("negative_sumsq", "playback paddle prior moments"),
        ("impossible_moments", "playback paddle prior moments"),
        ("zero_finite_nonzero_moment", "playback paddle prior moments"),
        ("error_names", "playback paddle prior term contract"),
        ("error_finite_rows", "playback paddle prior error_finite_rows"),
        ("negative_error_sumsq", "playback paddle prior error moments"),
        ("impossible_error_moments", "playback paddle prior error moments"),
    ),
)
def test_playback_paddle_prior_wire_rejects_nonexact_or_impossible_moments(
    case, error, tmp_path
):
    module = _load()

    def mutate(rows):
        paddle = rows[0]["reward_graph"]["playback_paddle_prior"]
        if case == "missing_key":
            paddle.pop("domain_violation_rows")
        elif case == "term_names":
            paddle["term_names"][:2] = reversed(paddle["term_names"][:2])
        elif case == "row_count":
            paddle["row_count"] = TRANSITIONS + 1
        elif case == "finite_rows":
            paddle["row_count"] = 1
            paddle["finite_rows"][0] = 2
        elif case == "domain_rows":
            paddle["row_count"] = 1
            paddle["finite_rows"][0] = 1
            paddle["domain_violation_rows"][0] = 2
        elif case == "negative_sumsq":
            paddle["row_count"] = 1
            paddle["finite_rows"][0] = 1
            paddle["kernel_sumsq"][0] = -0.1
        elif case == "impossible_moments":
            paddle["row_count"] = 1
            paddle["finite_rows"][0] = 1
            paddle["kernel_sum"][0] = 1.0
            paddle["kernel_sumsq"][0] = 0.5
        elif case == "zero_finite_nonzero_moment":
            paddle["kernel_sum"][0] = 1.0
        elif case == "error_names":
            paddle["error_names"][:2] = reversed(paddle["error_names"][:2])
        elif case == "error_finite_rows":
            paddle["row_count"] = 1
            paddle["error_finite_rows"][0] = 2
        elif case == "negative_error_sumsq":
            paddle["row_count"] = 1
            paddle["error_finite_rows"][0] = 1
            paddle["error_sumsq"][0] = -0.1
        elif case == "impossible_error_moments":
            paddle["row_count"] = 1
            paddle["error_finite_rows"][0] = 1
            paddle["error_sum"][0] = 1.0
            paddle["error_sumsq"][0] = 0.5

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match=error):
        _consume(module, evidence, snapshots, 1)


def test_runtime_mjb_verification_is_once_per_consume_not_once_per_ack(tmp_path):
    module = _load()
    calls = []
    expected = dict(
        _plant_contract().expected_plant_model_identity()["runtime_attach"][
            "final_augmented_mjb"
        ]
    )

    def verify(evidence):
        calls.append(Path(evidence))
        return dict(expected)

    module._verified_runtime_mjb = verify
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 5, complete=False,
    )
    _consume(module, evidence, snapshots, 5)
    assert calls == [evidence]


def test_runtime_stack_is_independently_verified_before_runtime_mjb_and_ack(
    tmp_path,
):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False,
    )
    trace = []
    verification = object()
    expected_mjb = dict(
        _plant_contract().expected_plant_model_identity()["runtime_attach"][
            "final_augmented_mjb"
        ]
    )
    module._epa48_runtime_module = lambda: types.SimpleNamespace(
        verify_runtime_stack_preimport=lambda: (
            trace.append("runtime_preimport") or verification
        ),
        verified_runtime_stack_identity=lambda actual: (
            trace.append("runtime_identity") or _runtime_stack()
            if actual is verification
            else pytest.fail("runtime verification token differs")
        ),
    )
    module._verified_runtime_mjb = lambda _evidence: (
        trace.append("runtime_mjb") or expected_mjb
    )
    original_read_rows = module._read_rows

    def read_rows(*args, **kwargs):
        trace.append("ack_read")
        return original_read_rows(*args, **kwargs)

    module._read_rows = read_rows
    _consume(module, evidence, snapshots, 1)
    assert trace[:4] == [
        "runtime_preimport", "runtime_identity", "runtime_mjb", "ack_read",
    ]


@pytest.mark.parametrize("message", (
    "MJLab selected code-tree SHA differs",
    "RSL-RL 3 wheel SHA differs",
))
def test_runtime_stack_drift_is_rejected_before_runtime_mjb_or_ack(
    tmp_path, message,
):
    module = _load()
    evidence = tmp_path / "updates.jsonl"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    module._epa48_runtime_module = lambda: types.SimpleNamespace(
        verify_runtime_stack_preimport=lambda: (_ for _ in ()).throw(
            RuntimeError(message)
        )
    )
    module._verified_runtime_mjb = lambda _path: pytest.fail(
        "runtime MJB must not be read after runtime-stack drift"
    )
    module._read_rows = lambda *_args, **_kwargs: pytest.fail(
        "ACK must not be read after runtime-stack drift"
    )
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _consume(module, evidence, snapshots, 1)


@pytest.mark.parametrize(
    "field",
    (
        "observations_policy", "observations_critic", "actions", "values",
        "actions_log_prob", "mu", "sigma", "rewards", "returns",
        "advantages",
    ),
)
def test_consumer_rejects_each_nonfinite_storage_receipt(field, tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["storage_finite"][field] = False

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="nonfinite rollout storage"):
        _consume(module, evidence, snapshots, 1)


def test_consumer_rejects_storage_domains_and_named_fault_receipts(tmp_path):
    module = _load()
    for label, mutation, error in (
        (
            "dones",
            lambda rows: rows[0]["storage_domains"].__setitem__(
                "dones_binary", False
            ),
            "rollout storage domain",
        ),
        (
            "missed",
            lambda rows: rows[0]["extras_counts"].__setitem__(
                "missed_launch_rows", 1
            ),
            "named event fault counter.*missed_launch_rows",
        ),
        (
            "recovery-completion",
            lambda rows: rows[0]["extras_counts"].__setitem__(
                "recovery_completion_fault_rows", 1
            ),
            "named event fault counter.*recovery_completion_fault_rows",
        ),
        (
            "sigma",
            lambda rows: rows[0]["storage_domains"].__setitem__(
                "sigma_positive", False
            ),
            "sigma_positive",
        ),
    ):
        root = tmp_path / label
        root.mkdir()
        evidence, snapshots, _completion, _rows = _artifacts(
            module, root, 1, complete=False, row_mutation=mutation
        )
        with pytest.raises(ValueError, match=error):
            _consume(module, evidence, snapshots, 1)


def test_evidence_v7_and_completion_v4_are_rejected_by_separate_schemas(
        monkeypatch, tmp_path):
    module = _load()

    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    evidence, snapshots, _completion, rows = _artifacts(
        module, evidence_root, 1, complete=False
    )
    rows[0]["schema_version"] = 7
    _write_evidence(evidence, rows)
    with pytest.raises(ValueError, match="fixed fields at update 0"):
        _consume(module, evidence, snapshots, 1)

    monkeypatch.setattr(module, "COMPLETE_UPDATES", 1)
    completion_root = tmp_path / "completion"
    completion_root.mkdir()
    evidence, snapshots, completion, _rows = _artifacts(
        module, completion_root, 1, complete=True,
        seal_mutation=lambda row: row.__setitem__("schema_version", 4),
    )
    with pytest.raises(ValueError, match="completion seal binding"):
        _consume(module, evidence, snapshots, 1, completion)


def test_snapshot_schedule_follows_the_typed_finite_recipe():
    module = _load()
    indices = module._snapshot_indices(True)
    expected = list(
        range(0, module.COMPLETE_UPDATES, module.SAVE_INTERVAL)
    )
    if module.COMPLETE_UPDATES - 1 not in expected:
        expected.append(module.COMPLETE_UPDATES - 1)
    assert indices == expected
    assert indices[-1] == module.COMPLETE_UPDATES - 1


def test_sealed_all_zero_longrun_is_not_milestone_coverage_completion(
        monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True
    )
    summary = _consume(module, evidence, snapshots, 2, completion)
    assert summary["engineering_run_complete"] is True
    assert summary["producer_attested_milestone_coverage_complete"] is False
    assert summary["full_a_complete"] is False
    assert summary["completion_seal_verified"] is True
    assert summary["action_contract"] == module.ACTION_CONTRACT
    assert summary["action_contract"]["transfer_authority"] is False
    assert summary["action_contract"]["matched_cross_backend_authority"] is False
    assert "selected_contact_rows" in summary[
        "producer_attested_milestone_coverage_missing"
    ]
    assert all(value == 0 for value in summary["milestones"].values())


def test_joint_safety_telemetry_reaches_summary_without_a_done_bit(tmp_path):
    module = _load()

    def add_telemetry(rows):
        rows[0]["extras_counts"]["actual_hard_edge_rows"] = 1
        rows[0]["extras_counts"]["qdes_guard_intervention_rows"] = 1

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=add_telemetry
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["actual_hard_edge_rows"] == 1
    assert summary["milestones"]["qdes_guard_intervention_rows"] == 1
    assert summary["terminal_bit_totals"]["joint_qdes_forbidden"] == 0
    assert summary["milestones"]["gym_reset_rows"] == 0


def test_sealed_slot0_milestone_coverage_never_claims_formal_completion(
        monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)
    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True, milestone_coverage=True
    )
    summary = _consume(module, evidence, snapshots, 2, completion)
    assert summary["diagnostic_unauthorized"] is True
    assert summary["engineering_run_complete"] is True
    assert summary["producer_attested_milestone_coverage_complete"] is True
    assert summary["same_epoch_chain_replay_status"] == "not_produced"
    assert summary["full_a_complete"] is False
    assert summary["producer_attested_milestone_coverage_missing"] == []
    assert "business_chain_complete" not in summary
    assert "business_chain_missing" not in summary
    assert summary["milestones"]["selected_contact_rows"] == 1
    assert summary["milestones"]["selected_reset_rows"] == 0
    assert summary["milestones"]["gym_reset_rows"] == 0
    assert summary["portable_reveal_opportunity"] == {
        "scheduled_rows": 1,
        "terminal_overlap_rows": 0,
        "due_rows": 1, "accepted_rows": 1, "deferred_rows": 0,
        "accept_rate": 1.0, "defer_rate": 0.0,
    }
    assert summary["hit_opportunity_r03"] == {
        "present_rows": 1,
        "physically_valid_rows": 1,
        "selected_contact_rows": 1,
        "selected_contact_to_physically_valid_ratio": 1.0,
    }
    assert summary["rates"] == {
        "selected_contact_per_launch": 1.0,
        "r03_physically_valid_per_present": 1.0,
        "r06_common_per_eligible": 1.0,
        "opponent_landing_per_crossing": 1.0,
        "recovery_success_per_terminal": 1.0,
    }


def test_missing_completed_milestone_keeps_aggregate_coverage_incomplete(
    monkeypatch, tmp_path
):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 2)

    def drop_joint_event(rows):
        rows[0]["extras_counts"]["completed_action_epoch_rows"] = 0

    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 2, complete=True, milestone_coverage=True,
        row_mutation=drop_joint_event,
    )
    summary = _consume(module, evidence, snapshots, 2, completion)
    assert summary["engineering_run_complete"] is True
    assert summary["producer_attested_milestone_coverage_complete"] is False
    assert summary["full_a_complete"] is False
    assert summary["producer_attested_milestone_coverage_missing"] == [
        "completed_action_epoch_rows"
    ]


@pytest.mark.parametrize(
    "predecessor",
    ("selected_contact_rows", "r03_physically_valid_rows", "r06_eligible_rows"),
)
def test_consumer_reports_missing_predecessor_without_inferring_a_chain(
    predecessor, tmp_path
):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"][predecessor] = 0
        if predecessor == "selected_contact_rows":
            row["extras_counts"]["racket_contact_rows"] = 0
            row["classification_status_counts"].update({
                "0": TRANSITIONS, "1": 0,
            })
        if predecessor == "r06_eligible_rows":
            row["extras_counts"]["r06_common_rows"] = 0

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, milestone_coverage=True,
        row_mutation=mutate,
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert predecessor in summary[
        "producer_attested_milestone_coverage_missing"
    ]
    assert summary["milestones"]["completed_action_epoch_rows"] == 1


@pytest.mark.parametrize(
    "case",
    (
        "launch_without_reveal",
        "contact_without_launch",
        "r03_without_launch",
        "terminal_without_launch",
        "landing_without_selected",
        "retire_without_launch",
    ),
)
def test_consumer_reports_marginal_rows_without_reconstructing_causal_bounds(
    case, tmp_path
):
    module = _load()

    def mutate(rows):
        row = rows[0]
        event = row["extras_counts"]
        if case == "launch_without_reveal":
            event["launch_rows"] = 1
            row["phase_counts"] = {
                "0": TRANSITIONS - 1, "2": 0, "5": 1, "6": 0, "8": 0,
            }
        elif case == "contact_without_launch":
            event.update({"racket_contact_rows": 1, "selected_contact_rows": 1})
            row["classification_status_counts"].update({
                "0": TRANSITIONS - 1, "1": 1,
            })
            row["phase_counts"] = {
                "0": TRANSITIONS - 1, "2": 0, "5": 1, "6": 0, "8": 0,
            }
        elif case == "r03_without_launch":
            event["r03_present_rows"] = 1
            row["phase_counts"] = {
                "0": TRANSITIONS - 1, "2": 0, "5": 1, "6": 0, "8": 0,
            }
        elif case == "terminal_without_launch":
            event.update({"flight_terminal_rows": 1, "r06_present_rows": 1,
                          "r06_eligible_rows": 1})
            row["outcome_code_counts"]["5"] = 1
            row["phase_counts"] = {
                "0": TRANSITIONS - 1, "2": 0, "5": 0, "6": 1, "8": 0,
            }
        elif case == "landing_without_selected":
            event.update({
                "scheduled_due_rows": 1, "reveal_due_rows": 1,
                "reveal_rows": 1, "launch_rows": 1,
                "flight_terminal_rows": 1, "landing_crossing_rows": 1,
                "r06_present_rows": 1, "r06_eligible_rows": 1,
            })
            row["outcome_code_counts"]["4"] = 1
            row["phase_counts"] = {
                "0": TRANSITIONS - 2, "2": 1, "5": 0, "6": 1, "8": 0,
            }
        else:
            event.update({"shot_retired_rows": 1, "recovery_success_rows": 1})
            row["phase_counts"] = {
                "0": TRANSITIONS - 1, "2": 0, "5": 0, "6": 0, "8": 1,
            }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    observed_key = {
        "launch_without_reveal": "launch_rows",
        "contact_without_launch": "selected_contact_rows",
        "r03_without_launch": "r03_present_rows",
        "terminal_without_launch": "flight_terminal_rows",
        "landing_without_selected": "landing_crossing_rows",
        "retire_without_launch": "shot_retired_rows",
    }[case]
    assert summary["milestones"][observed_key] == 1


def test_r06_common_uses_all_source_valid_outcomes_as_denominator(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"]["r06_common_rows"] = 0

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, milestone_coverage=True,
        row_mutation=mutate,
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["rates"]["r06_common_per_eligible"] == 0.0
    assert summary["rates"]["opponent_landing_per_crossing"] == 1.0


def test_portable_due_accept_defer_rates_use_due_denominator(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"].update({
            "scheduled_due_rows": 2,
            "reveal_due_rows": 2,
            "reveal_deferred_rows": 1,
        })

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, milestone_coverage=True,
        row_mutation=mutate,
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["portable_reveal_opportunity"] == {
        "scheduled_rows": 2,
        "terminal_overlap_rows": 0,
        "due_rows": 2, "accepted_rows": 1, "deferred_rows": 1,
        "accept_rate": 0.5, "defer_rate": 0.5,
    }


@pytest.mark.parametrize(
    "broken_counts",
    (
        {
            "scheduled_due_rows": 1,
            "due_terminal_overlap_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
        },
        {
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 2,
            "reveal_deferred_rows": 0,
        },
    ),
)
def test_consumer_rejects_broken_curriculum_opportunity_partition(
    tmp_path, broken_counts
):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"].update(broken_counts)

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="curriculum opportunity partition"):
        _consume(module, evidence, snapshots, 1)


@pytest.mark.parametrize(
    "broken_counts",
    (
        {"r03_present_rows": 1, "r03_physically_valid_rows": 2},
        {"r06_present_rows": 1, "r06_eligible_rows": 2},
        {
            "r06_present_rows": 2,
            "r06_eligible_rows": 1,
            "r06_common_rows": 2,
        },
    ),
)
def test_consumer_rejects_broken_reward_opportunity_subset(
    tmp_path, broken_counts
):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"].update(broken_counts)

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="reward opportunity subset"):
        _consume(module, evidence, snapshots, 1)


def test_early_selected_contact_without_r03_is_reported_not_rejected(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
            "launch_rows": 1,
            "racket_contact_rows": 1,
            "selected_contact_rows": 1,
        })
        row["classification_status_counts"].update({
            "0": TRANSITIONS - 1,
            "1": 1,
        })
        row["phase_counts"] = {
            "0": TRANSITIONS - 2,
            "2": 1,
            "5": 1,
            "6": 0,
            "8": 0,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["hit_opportunity_r03"] == {
        "present_rows": 0,
        "physically_valid_rows": 0,
        "selected_contact_rows": 1,
        "selected_contact_to_physically_valid_ratio": None,
    }


def test_scheduled_due_terminal_overlap_stays_distinct_from_public_due(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "due_terminal_overlap_rows": 1,
            "selected_reset_rows": 1,
        })
        row["selected_reset_rows"] = 1
        row["gym_reset_rows"] = 1
        row["terminal_bit_counts"]["base_fell_tilt"] = 1
        row["lifecycle_counts"].update({
            "gym_reset_rows": 1,
            "reset_generation_rows": 1,
        })
        row["episodes"] = {
            "completed_count": 1,
            "return_sum": 0.0,
            "length_sum": 1,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["portable_reveal_opportunity"] == {
        "scheduled_rows": 1,
        "terminal_overlap_rows": 1,
        "due_rows": 0,
        "accepted_rows": 0,
        "deferred_rows": 0,
        "accept_rate": None,
        "defer_rate": None,
    }


def test_retirement_and_r07_marginals_fit_final_reveal_phase(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
            "launch_rows": 1,
            "flight_terminal_rows": 1,
            "r06_present_rows": 1,
            "r07_present_rows": 1,
            "r07_eligible_rows": 1,
            "recovery_timeout_rows": 1,
            "shot_retired_rows": 1,
        })
        row["outcome_code_counts"]["1"] = 1
        row["phase_counts"] = {
            "0": TRANSITIONS - 2,
            "2": 1,
            "5": 1,
            "6": 0,
            "8": 0,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["reveal_rows"] == 1
    assert summary["milestones"]["shot_retired_rows"] == 1
    assert summary["milestones"]["r07_present_rows"] == 1


def test_consumer_does_not_rebuild_reveal_r07_phase_implication(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
            "r07_present_rows": 1,
            "r07_eligible_rows": 1,
        })
        row["phase_counts"] = {
            "0": TRANSITIONS - 1,
            "2": 1,
            "5": 0,
            "6": 0,
            "8": 0,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["reveal_rows"] == 1
    assert summary["milestones"]["r07_present_rows"] == 1


def test_immediate_terminal_launch_uses_terminal_phase_capacity(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
            "launch_rows": 1,
            "flight_terminal_rows": 1,
            "r06_present_rows": 1,
            "r06_eligible_rows": 1,
        })
        row["outcome_code_counts"]["5"] = 1
        row["phase_counts"] = {
            "0": TRANSITIONS - 2, "2": 1, "5": 0, "6": 1, "8": 0,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["launch_rows"] == 1
    assert summary["milestones"]["flight_terminal_rows"] == 1


def test_consumer_does_not_rebuild_launch_phase_implication(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
            "launch_rows": 1,
        })
        row["phase_counts"] = {
            "0": TRANSITIONS - 1, "2": 1, "5": 0, "6": 0, "8": 0,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["launch_rows"] == 1


@pytest.mark.parametrize("name", sorted(RUNNER_FACT_INTEGRITY_KEYS))
def test_rejects_each_durable_fact_integrity_counter(name, tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["fact_integrity_counts"][name] = 1

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="fact integrity fault counter"):
        _consume(module, evidence, snapshots, 1)


def test_resolved_table_requires_canonical_robot_hit_table_reason(tmp_path):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"]["selected_reset_rows"] = 1
        row["selected_reset_rows"] = 1
        row["gym_reset_rows"] = 1
        row["terminal_bit_counts"]["robot_hit_table"] = 1
        row["lifecycle_counts"].update({
            "gym_reset_rows": 1,
            "reset_generation_rows": 1,
            "resolved_table_rows": 1,
        })
        row["episodes"] = {
            "completed_count": 1,
            "return_sum": 0.0,
            "length_sum": 1,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["terminal_bit_totals"]["robot_hit_table"] == 1
    assert summary["milestones"]["gym_reset_rows"] == 1
    assert summary["table_terminal"] == {
        "robot_hit_table_rows": 1,
        "resolved_rows": 1,
        "keepout_only_rows": 0,
    }

    keepout_root = tmp_path / "keepout-only"
    keepout_root.mkdir()

    def keepout_only(rows):
        row = rows[0]
        row["extras_counts"]["selected_reset_rows"] = 1
        row["selected_reset_rows"] = 1
        row["gym_reset_rows"] = 1
        row["terminal_bit_counts"]["robot_hit_table"] = 1
        row["lifecycle_counts"].update({
            "gym_reset_rows": 1, "reset_generation_rows": 1,
        })
        row["episodes"] = {
            "completed_count": 1, "return_sum": 0.0, "length_sum": 1,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, keepout_root, 1, complete=False, row_mutation=keepout_only
    )
    keepout_summary = _consume(module, evidence, snapshots, 1)
    assert keepout_summary["table_terminal"] == {
        "robot_hit_table_rows": 1,
        "resolved_rows": 0,
        "keepout_only_rows": 1,
    }

    def omit_reason(rows):
        mutate(rows)
        rows[0]["terminal_bit_counts"]["robot_hit_table"] = 0

    bad_root = tmp_path / "missing-table-reason"
    bad_root.mkdir()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, bad_root, 1, complete=False, row_mutation=omit_reason
    )
    with pytest.raises(ValueError, match="lifecycle cross-check"):
        _consume(module, evidence, snapshots, 1)


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


def test_rejects_mutated_runtime_identity_in_evidence_even_with_new_hash(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["run_identity"]["runtime_stack"]["mujoco_warp"][
            "epa_horizon"
        ] = 24

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="run identity at update 0"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_legacy_mujoco_warp_runtime_identity_wire(tmp_path):
    module = _load()

    def mutate(rows):
        identity = rows[0]["run_identity"]
        stack = identity.pop("runtime_stack")
        identity["mujoco_warp_runtime"] = stack["mujoco_warp"]

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="run identity keys"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_mutated_plant_identity_in_evidence_even_with_new_hash(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["run_identity"]["plant_model"]["runtime_attach"][
            "final_augmented_mjb"
        ]["sha256"] = "0" * 64

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="run identity at update 0"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_mutated_owner_frame_receipt_even_with_new_outer_hash(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["run_identity"]["plant_model"]["runtime_attach"][
            "owner_local_frame_sha256"
        ] = "e" * 64

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    with pytest.raises(ValueError, match="run identity at update 0"):
        _consume(module, evidence, snapshots, 1)


def test_expected_plant_xml_is_canonical_and_sha_pinned(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False
    )
    wrong = tmp_path / "wrong.xml"
    wrong.write_text('<mujoco model="A3T2.5_pingpong_0519"/>\n')
    with pytest.raises(ValueError, match="expected plant exact verification"):
        _consume(module, evidence, snapshots, 1, plant_xml=wrong)


def test_consumer_verifies_base_once_and_consumes_same_model_for_owner(monkeypatch):
    module = _load()
    counts = {"verify": 0, "consume": 0}
    source = _plant_contract().expected_plant_model_identity()["source_plant"]

    class Verified:
        portable_identity_sha256 = source["portable_identity_sha256"]
        verification_receipt_sha256 = VERIFICATION_RECEIPT_SHA256

        def consume_verified_model(self, consumer):
            counts["consume"] += 1
            return consumer(object())

    def verify(**_kwargs):
        counts["verify"] += 1
        return Verified()

    monkeypatch.setattr(
        module,
        "_canonical_mujoco_identity_module",
        lambda: types.SimpleNamespace(verify_exact_mujoco_identity=verify),
    )
    monkeypatch.setattr(module, "_mujoco_module", lambda: object())
    monkeypatch.setattr(
        module,
        "_table_termination_module",
        lambda: types.SimpleNamespace(
            consume_verified_owner_frame_contract=lambda _mujoco, verified: (
                verified.consume_verified_model(
                    lambda _model: {
                        "content_sha256": OWNER_LOCAL_FRAME_SHA256,
                    }
                )
            )
        ),
    )
    assert module._verified_plant_model(
        PLANT_XML,
        _plant_contract().expected_plant_model_identity()["runtime_attach"][
            "final_augmented_mjb"
        ],
    ) == PLANT_MODEL
    assert counts == {"verify": 1, "consume": 1}


def test_consumer_hashes_and_loads_the_run_owned_runtime_mjb_once(
    monkeypatch, tmp_path,
):
    module = _load(stub_runtime_mjb=False)
    payload = b"independently observed augmented MJB"
    runtime_mjb = tmp_path / "runtime.mjb"
    runtime_mjb.write_bytes(payload)
    receipt = {
        "relative_locator": "runtime.mjb",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    contract = module._plant_contract_module()
    expected = contract.expected_plant_model_identity()

    def expected_with_fixture():
        value = copy.deepcopy(expected)
        value["runtime_attach"]["final_augmented_mjb"] = dict(receipt)
        return value

    monkeypatch.setattr(contract, "expected_plant_model_identity", expected_with_fixture)
    loads = []

    class FakeModel:
        @classmethod
        def from_binary_path(cls, path):
            loads.append(path)
            assert Path(path).read_bytes() == payload
            return object()

    monkeypatch.setattr(
        module, "_mujoco_module",
        lambda: types.SimpleNamespace(MjModel=FakeModel),
    )
    assert module._verified_runtime_mjb(tmp_path / "evidence.jsonl") == receipt
    assert loads == [str(runtime_mjb)]


def test_consumer_rejects_runtime_mjb_byte_drift_before_loading(
    monkeypatch, tmp_path,
):
    module = _load(stub_runtime_mjb=False)
    (tmp_path / "runtime.mjb").write_bytes(b"drift")
    loads = []

    class FakeModel:
        @classmethod
        def from_binary_path(cls, path):
            loads.append(path)
            return object()

    monkeypatch.setattr(
        module, "_mujoco_module",
        lambda: types.SimpleNamespace(MjModel=FakeModel),
    )
    with pytest.raises(ValueError, match="runtime MJB receipt"):
        module._verified_runtime_mjb(tmp_path / "evidence.jsonl")
    assert loads == []


def test_consumer_locator_path_is_not_part_of_wire_identity(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False
    )
    relocated = tmp_path / "relocated" / PLANT_XML.name
    relocated.parent.mkdir()
    relocated.write_bytes(PLANT_XML.read_bytes())
    summary = _consume(
        module, evidence, snapshots, 1, plant_xml=relocated
    )
    assert summary["run_identity"] == IDENTITY


def test_rejects_cross_run_snapshot_infos_even_with_matching_receipt(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, snapshot_identity=OTHER_IDENTITY
    )
    with pytest.raises(ValueError, match="snapshot infos binding"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_mutated_runtime_identity_in_snapshot_infos(tmp_path):
    module = _load()

    def mutate(index, payload):
        if index == 0:
            payload["infos"]["run_identity"]["runtime_stack"]["mujoco_warp"][
                "wheel_sha256"
            ] = "0" * 64

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, snapshot_mutation=mutate
    )
    with pytest.raises(ValueError, match="snapshot infos binding"):
        _consume(module, evidence, snapshots, 1)


def test_rejects_mutated_runtime_identity_in_completion(monkeypatch, tmp_path):
    module = _load()
    monkeypatch.setattr(module, "COMPLETE_UPDATES", 1)

    def mutate(record):
        record["run_identity"]["runtime_stack"]["mujoco_warp"][
            "version"
        ] = "3.10.0.3"

    evidence, snapshots, completion, _rows = _artifacts(
        module, tmp_path, 1, complete=True, seal_mutation=mutate
    )
    with pytest.raises(ValueError, match="completion seal binding"):
        _consume(module, evidence, snapshots, 1, completion)


def test_rejects_finite_toy_snapshot_that_is_not_exact_model_abi(tmp_path):
    module = _load()
    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, toy=True
    )
    with pytest.raises(ValueError, match="snapshot model ABI"):
        _consume(module, evidence, snapshots, 1)


@pytest.mark.parametrize(
    ("parameter_name", "v2_shape"),
    (
        ("actor.0.weight", (512, 203)),
        ("critic.0.weight", (512, 219)),
    ),
)
def test_fresh_model_abi_is_semantic_v3_and_rejects_v2_snapshots(
    tmp_path, parameter_name, v2_shape
):
    module = _load()
    contract = module._portable_observation_module()
    shapes = dict(module.MODEL_SHAPES)
    assert contract.ACTOR_WIDTH_V3 == 215
    assert contract.CRITIC_WIDTH_V3 == 231
    assert module.ACTOR_OBSERVATION_WIDTH == contract.ACTOR_WIDTH_V3
    assert module.CRITIC_OBSERVATION_WIDTH == contract.CRITIC_WIDTH_V3
    assert shapes["actor.0.weight"] == (512, contract.ACTOR_WIDTH_V3)
    assert shapes["critic.0.weight"] == (512, contract.CRITIC_WIDTH_V3)

    def mutate(index, payload):
        if index == 0:
            payload["model_state_dict"][parameter_name] = torch.zeros(
                v2_shape, dtype=torch.float32
            )

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, snapshot_mutation=mutate
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
    assert "shot_retired_rows" in summary[
        "producer_attested_milestone_coverage_missing"
    ]


def test_consumer_does_not_infer_missing_fact_integrity_from_event_marginal(
    tmp_path,
):
    module = _load()

    def mutate(rows):
        row = rows[0]
        row["extras_counts"].update({
            "scheduled_due_rows": 1,
            "reveal_due_rows": 1,
            "reveal_rows": 1,
            "launch_rows": 1,
            "racket_contact_rows": 1,
            "invalid_contact_rows": 1,
            "flight_terminal_rows": 1,
            "r06_present_rows": 1,
            "selected_reset_rows": 1,
            "recovery_failure_rows": 1,
        })
        row["selected_reset_rows"] = 1
        row["gym_reset_rows"] = 1
        row["terminal_bit_counts"]["base_fell_tilt"] = 1
        row["classification_status_counts"].update({
            "0": TRANSITIONS - 1, "5": 1,
        })
        row["outcome_code_counts"]["6"] = 1
        row["phase_counts"] = {
            "0": TRANSITIONS - 3, "2": 1, "5": 1, "6": 1, "8": 0,
        }
        row["lifecycle_counts"].update({
            "gym_reset_rows": 1, "reset_generation_rows": 1,
        })
        row["episodes"] = {
            "completed_count": 1, "return_sum": 0.0, "length_sum": 1,
        }

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["invalid_contact_rows"] == 1


def test_consumer_does_not_rebuild_recovery_retirement_implication(tmp_path):
    module = _load()

    def mutate(rows):
        rows[0]["extras_counts"].update({
            "shot_retired_rows": 1, "recovery_failure_rows": 1,
        })

    evidence, snapshots, _completion, _rows = _artifacts(
        module, tmp_path, 1, complete=False, row_mutation=mutate
    )
    summary = _consume(module, evidence, snapshots, 1)
    assert summary["milestones"]["recovery_failure_rows"] == 1
    assert summary["milestones"]["shot_retired_rows"] == 1


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
    evidence.write_text(record[:-1] + ',"schema_version":5}\n')
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
            expected_plant_xml=PLANT_XML,
            snapshot_dir=None,
        )
    with pytest.raises(ValueError, match="expected update count"):
        module.consume(
            evidence,
            expected_updates=2,
            expected_source_commit=COMMIT,
            expected_run_namespace=NAMESPACE,
            expected_plant_xml=PLANT_XML,
            snapshot_dir=snapshots,
        )
