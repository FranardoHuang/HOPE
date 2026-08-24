from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
UID = 6907688916670928
SOURCE_COMMIT = "a" * 40
RUN_NAMESPACE = "mujoco-fullmdp-ledger-test-v2"
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
RUNTIME_STACK = {
    "schema_version": 1,
    "mujoco_warp": MUJOCO_WARP_RUNTIME,
    "rsl_rl": {
        "distribution": "rsl-rl-lib",
        "version": "3.1.2",
        "wheel_sha256": (
            "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
        ),
        "import_scope": "fresh_run_local_site",
    },
    "mjlab": {
        "schema_version": 1,
        "distribution": "mjlab",
        "version": "1.5.3",
        "import_scope": "verified_venv_distribution",
        "selected_tree_scope": "mjlab/**/*.py+mjlab/scene/scene.xml",
        "selected_file_count": 193,
        "selected_byte_count": 1399177,
        "selected_tree_sha256": (
            "88c9725d0416b4ac3e21f6752ad423c13ea3b8cfb9e23ca664f8aba146cec33d"
        ),
        "mjlab_tasks_entry_point_count": 0,
    },
}


def _plant_contract():
    name = "_ledger_test_mujoco_full_mdp_plant_contract"
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
    verification_receipt_sha256="c" * 64,
    owner_local_frame_sha256="d" * 64,
    final_augmented_mjb=_plant_contract().expected_plant_model_identity()[
        "runtime_attach"
    ]["final_augmented_mjb"],
)


def _run_identity(source_commit=SOURCE_COMMIT, run_namespace=RUN_NAMESPACE):
    return {
        "source_commit": source_commit,
        "run_namespace": run_namespace,
        "runtime_stack": copy.deepcopy(RUNTIME_STACK),
        "plant_model": _plant_contract().clone_plant_model_identity(PLANT_MODEL),
    }


TERMINATION_BITS = {
    "time_out": 1,
    "base_fell_tilt": 2,
    "base_too_low": 4,
    "joint_qdes_forbidden": 8,
    "robot_hit_table": 16,
}
EVENT_KEYS = {
    "scheduled_due_rows": "full_a_scheduled_due_event",
    "due_terminal_overlap_rows": "full_a_due_terminal_overlap_event",
    "reveal_rows": "full_a_reveal_event",
    "reveal_due_rows": "full_a_reveal_due_event",
    "reveal_deferred_rows": "full_a_reveal_deferred_event",
    "launch_rows": "full_a_launch_event",
    "missed_launch_rows": "full_a_missed_launch_event",
    "flight_terminal_rows": "full_a_flight_terminal_event",
    "shot_retired_rows": "full_a_shot_retired_event",
    "completed_action_epoch_rows": "full_a_completed_action_epoch_event",
    "selected_reset_rows": "full_a_selected_reset_event",
    "racket_contact_rows": "full_a_racket_contact_event",
    "selected_contact_rows": "full_a_selected_contact_event",
    "opposite_contact_rows": "full_a_opposite_contact_event",
    "edge_contact_rows": "full_a_edge_contact_event",
    "between_contact_rows": "full_a_between_contact_event",
    "invalid_contact_rows": "full_a_invalid_contact_event",
    "actual_hard_edge_rows": "full_a_actual_hard_edge_event",
    "qdes_guard_intervention_rows": "full_a_qdes_guard_intervention_event",
    "r03_present_rows": "full_a_r03_present_event",
    "r03_physically_valid_rows": "full_a_r03_physically_valid_event",
    "landing_crossing_rows": "full_a_landing_crossing_event",
    "r06_present_rows": "full_a_r06_present_event",
    "r06_eligible_rows": "full_a_r06_eligible_event",
    "r06_common_rows": "full_a_r06_common_event",
    "r07_present_rows": "full_a_r07_present_event",
    "r07_eligible_rows": "full_a_r07_eligible_event",
    "recovery_success_rows": "full_a_recovery_success_event",
    "recovery_failure_rows": "full_a_recovery_failure_event",
    "recovery_timeout_rows": "full_a_recovery_timeout_event",
    "recovery_completion_fault_rows": (
        "full_a_recovery_completion_fault_event"
    ),
}
STORAGE_WIDTHS = {
    "observations_policy": 215,
    "observations_critic": 231,
    "actions": 31,
    "values": 1,
    "actions_log_prob": 1,
    "mu": 31,
    "sigma": 31,
    "rewards": 1,
    "returns": 1,
    "advantages": 1,
}


def _load():
    path = LANE / "mujoco_full_mdp_update_ledger.py"
    name = "mujoco_full_mdp_update_ledger_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_fact_integrity_bit_and_counter_abi_is_exact():
    module = _load()
    assert module.FACT_INTEGRITY_CAUSES == (
        ("fact_integrity_r03_nonfinite_rows", 1),
        ("fact_integrity_r06_source_invalid_rows", 2),
        ("fact_integrity_r07_sequence_rows", 4),
        ("fact_integrity_r07_nonfinite_rows", 8),
    )
    assert module.FACT_INTEGRITY_COUNT_NAMES == (
        "fact_integrity_r03_nonfinite_rows",
        "fact_integrity_r06_source_invalid_rows",
        "fact_integrity_r07_sequence_rows",
        "fact_integrity_r07_nonfinite_rows",
        "fact_integrity_unknown_bits_rows",
    )
    assert module.FACT_INTEGRITY_KNOWN_MASK == 15
    assert set(module.FACT_INTEGRITY_COUNT_NAMES).issubset(
        module._MISC_NAMES
    )
    assert set(module.FACT_INTEGRITY_COUNT_NAMES).issubset(
        module._ZERO_FAULTS
    )


def test_reward_graph_contract_is_loaded_from_the_shared_ordered_authority():
    module = _load()
    contract = module._reward_contract_module()
    assert module.SCHEMA_VERSION == 9
    assert module.REWARD_TERM_NAMES == contract.MANAGER_NAMES
    assert module.REWARD_TERM_COUNT == contract.REWARD_TERM_COUNT
    assert module.REWARD_TERM_COUNT == len(module.REWARD_TERM_NAMES)


def test_rollout_storage_widths_are_loaded_from_semantic_v3_authority():
    module = _load()
    contract = module._portable_observation_module()
    assert contract.OBSERVATION_KIND_V3 == (
        "action_ball_full_mdp_semantic_observation_v3"
    )
    assert contract.ACTOR_WIDTH_V3 == 215
    assert contract.CRITIC_WIDTH_V3 == 231
    assert module.FULLMDP_ACTOR_OBSERVATION_WIDTH == contract.ACTOR_WIDTH_V3
    assert module.FULLMDP_CRITIC_OBSERVATION_WIDTH == contract.CRITIC_WIDTH_V3
    assert dict(module.STORAGE_FLOAT_WIDTHS) == STORAGE_WIDTHS


@pytest.mark.parametrize(
    "loader_name,cache_name,expected_error",
    (
        (
            "_plant_contract_module",
            "_hope_mujoco_full_mdp_plant_contract",
            "cached MuJoCo plant contract origin differs",
        ),
        (
            "_reward_contract_module",
            "_hope_mujoco_ledger_action_ball_full_mdp_reward_contract",
            "cached FullMDP reward contract origin differs",
        ),
        (
            "_portable_catalog_module",
            "_hope_mujoco_ledger_action_ball_full_mdp_portable_catalog",
            "cached FullMDP portable catalog origin differs",
        ),
        (
            "_portable_observation_module",
            "_hope_mujoco_ledger_action_ball_full_mdp_portable_observation",
            "cached FullMDP portable observation contract origin differs",
        ),
    ),
)
def test_local_contract_loaders_reject_cached_origin_drift(
    monkeypatch, loader_name, cache_name, expected_error
):
    module = _load()
    impostor = types.ModuleType(cache_name)
    impostor.__file__ = "/wrong/origin.py"
    monkeypatch.setitem(sys.modules, cache_name, impostor)

    with pytest.raises(RuntimeError) as caught:
        getattr(module, loader_name)()
    assert str(caught.value) == expected_error


def test_local_module_loader_drops_failed_partial_import(tmp_path):
    module = _load()
    source = tmp_path / "broken_contract.py"
    source.write_text("raise RuntimeError('broken contract')\n", encoding="utf-8")
    name = "_ledger_test_failed_pinned_local_module"
    sys.modules.pop(name, None)

    with pytest.raises(RuntimeError, match="broken contract"):
        module._load_pinned_local_module(
            source=source.resolve(),
            name=name,
            subject="test contract",
        )

    assert name not in sys.modules


def _ledger(module, *, steps=2, num_envs=3, run_identity=None):
    return module.FullMdpUpdateLedger(
        torch_module=torch,
        num_envs=num_envs,
        num_steps_per_env=steps,
        device=torch.device("cpu"),
        termination_bits=TERMINATION_BITS,
        action_slot=0,
        action_uid=UID,
        mount_normal_sign=1,
        family="forehand",
        initial_reset_generation=torch.zeros(num_envs, dtype=torch.long),
        run_identity=_run_identity() if run_identity is None else run_identity,
    )


@pytest.mark.parametrize(
    "bits",
    (
        {"time_out": 1, "bogus": 2},
        {**TERMINATION_BITS, "robot_hit_table": 32},
    ),
)
def test_constructor_rejects_nonexact_termination_bit_schema(bits):
    module = _load()
    with pytest.raises(ValueError, match="termination bit schema differs"):
        module.FullMdpUpdateLedger(
            torch_module=torch,
            num_envs=3,
            num_steps_per_env=2,
            device=torch.device("cpu"),
            termination_bits=bits,
            action_slot=0,
            action_uid=UID,
            mount_normal_sign=1,
            family="forehand",
            initial_reset_generation=torch.zeros(3, dtype=torch.long),
            run_identity=_run_identity(),
        )


@pytest.mark.parametrize(
    "source_commit,run_namespace",
    (
        ("a" * 39, RUN_NAMESPACE),
        ("A" * 40, RUN_NAMESPACE),
        (SOURCE_COMMIT, "too-short"),
        (SOURCE_COMMIT, "mujoco/fullmdp/invalid/namespace"),
    ),
)
def test_constructor_rejects_nonexact_run_identity(source_commit, run_namespace):
    module = _load()
    with pytest.raises(ValueError, match="run identity"):
        module.FullMdpUpdateLedger(
            torch_module=torch,
            num_envs=3,
            num_steps_per_env=2,
            device=torch.device("cpu"),
            termination_bits=TERMINATION_BITS,
            action_slot=0,
            action_uid=UID,
            mount_normal_sign=1,
            family="forehand",
            initial_reset_generation=torch.zeros(3, dtype=torch.long),
            run_identity=_run_identity(source_commit, run_namespace),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda identity: identity.pop("runtime_stack"),
        lambda identity: identity.__setitem__("runtime_stack", []),
        lambda identity: (
            identity.__setitem__(
                "mujoco_warp_runtime", identity.pop("runtime_stack")["mujoco_warp"]
            )
        ),
        lambda identity: identity["runtime_stack"].__setitem__(
            "schema_version", True
        ),
        lambda identity: identity["runtime_stack"].__setitem__("extra", 1),
        lambda identity: identity["runtime_stack"]["mujoco_warp"].__setitem__(
            "epa_horizon", 24
        ),
        lambda identity: identity["runtime_stack"]["rsl_rl"].__setitem__(
            "version", "3.1.1"
        ),
        lambda identity: identity["runtime_stack"]["mjlab"].pop(
            "selected_tree_sha256"
        ),
        lambda identity: identity["runtime_stack"]["mjlab"].__setitem__(
            "selected_file_count", 194
        ),
        lambda identity: identity.pop("plant_model"),
        lambda identity: identity["plant_model"]["runtime_attach"][
            "policy_clock"
        ].__setitem__(
            "step_dt", 0.002
        ),
        lambda identity: identity["plant_model"]["runtime_attach"][
            "final_augmented_mjb"
        ].__setitem__(
            "sha256", "0" * 64
        ),
        lambda identity: identity["plant_model"]["source_plant"].__setitem__(
            "verification_receipt_sha256", "not-a-digest"
        ),
    ),
)
def test_constructor_rejects_missing_or_nonmapping_runtime_identity(mutation):
    module = _load()
    identity = _run_identity()
    mutation(identity)
    with pytest.raises(ValueError, match="run identity|runtime identity"):
        _ledger(module, run_identity=identity)


def test_constructor_isolates_nested_runtime_identity_copy():
    module = _load()
    identity = _run_identity()
    ledger = _ledger(module, run_identity=identity)
    identity["runtime_stack"]["mujoco_warp"]["epa_horizon"] = 24
    identity["runtime_stack"]["rsl_rl"]["version"] = "mutated"
    identity["runtime_stack"]["mjlab"]["selected_tree_sha256"] = "0" * 64
    identity["plant_model"]["runtime_attach"]["final_augmented_mjb"][
        "sha256"
    ] = "0" * 64
    assert ledger.run_identity == _run_identity()


def _result(module, *, num_envs=3, event=None, bits=None, done=None,
            generation=None, time_outs=None, resolved=None, status=None,
            landing=None, opponent_bound=None, outcome=None, phase=None,
            fact_integrity_bits=None, paddle_playback=None):
    event = dict(event or {})
    if "scheduled_due_rows" not in event:
        public_due = event.get("reveal_due_rows", [False] * num_envs)
        terminal_overlap = event.get(
            "due_terminal_overlap_rows", [False] * num_envs
        )
        event["scheduled_due_rows"] = [
            bool(public_due[index] or terminal_overlap[index])
            for index in range(num_envs)
        ]
    extras = {
        key: torch.tensor(event.get(name, [False] * num_envs), dtype=torch.bool)
        for name, key in EVENT_KEYS.items()
    }
    terms = (
        torch.arange(
            1, module.REWARD_TERM_COUNT + 1, dtype=torch.float32
        ).repeat(num_envs, 1)
        / 100.0
    )
    extras.update(
        {
            "termination_bits": torch.tensor(bits or [0] * num_envs, dtype=torch.long),
            "time_outs": torch.tensor(time_outs or [False] * num_envs, dtype=torch.bool),
            "backend_resolved_table_contact": torch.tensor(
                resolved or [False] * num_envs, dtype=torch.bool
            ),
            "reward_terms": terms,
            "reset_generation": torch.tensor(
                generation or [0] * num_envs, dtype=torch.long
            ),
            "full_a_action_slot": torch.zeros(num_envs, dtype=torch.long),
            "full_a_action_uid": torch.full((num_envs,), UID, dtype=torch.long),
            "full_a_mount_normal_sign": torch.ones(num_envs, dtype=torch.int8),
            "full_a_contact_classification_status": torch.tensor(
                status or [0] * num_envs, dtype=torch.int8
            ),
            "full_a_outcome_code": torch.tensor(
                outcome or [0] * num_envs, dtype=torch.long
            ),
            "full_a_phase_before_reset": torch.tensor(
                phase or [2] * num_envs, dtype=torch.long
            ),
            "full_a_fact_integrity_fault_bits": torch.tensor(
                fact_integrity_bits or [0] * num_envs, dtype=torch.long
            ),
            "full_a_paddle_prior_playback": torch.tensor(
                paddle_playback or [False] * num_envs, dtype=torch.bool
            ),
            "full_a_landing_on_opponent": torch.tensor(
                landing or [False] * num_envs, dtype=torch.bool
            ),
            "full_a_landing_opponent_bound": torch.tensor(
                opponent_bound or [False] * num_envs, dtype=torch.bool
            ),
        }
    )
    return (
        {"policy": torch.zeros(num_envs, 229)},
        terms.sum(1),
        torch.tensor(done or [0] * num_envs, dtype=torch.long),
        extras,
    )


def _decode(prepared):
    return json.loads(prepared.payload.decode("utf-8"))


def _metrics():
    return {"value_function": 0.25, "surrogate": -0.125, "entropy": 1.5}


def _timings():
    return {
        "collection_seconds": 1.0,
        "learning_seconds": 2.0,
        "pre_ack_iteration_seconds": 3.25,
        "run_elapsed_pre_ack_seconds": 3.25,
    }


def _snapshot(update_index=0):
    return {
        "name": f"model_{update_index}.pt",
        "bytes": 1234,
        "sha256": "b" * 64,
    }


def _prepare(ledger, update_index, *, environment_steps, finite=True,
             policy_std=0.02):
    storage = {
        name: torch.ones(ledger.num_steps, ledger.num_envs, width)
        for name, width in STORAGE_WIDTHS.items()
    }
    dones = torch.zeros(
        ledger.num_steps, ledger.num_envs, 1, dtype=torch.uint8
    )
    if not finite:
        storage["advantages"][0, 0, 0] = torch.nan
    return ledger.prepare(
        update_index, environment_steps=environment_steps,
        storage_step=ledger.num_steps, storage_tensors=storage,
        storage_dones=dones,
        policy_std=(
            torch.full((ledger.num_envs, 31), policy_std)
            if type(policy_std) in (int, float)
            else policy_std
        ),
    )


def test_prepare_packs_exact_raw_update_schema_and_zero_denominators():
    module = _load()
    assert dict(module.EVENT_FIELDS) == EVENT_KEYS
    ledger = _ledger(module)
    ledger.ingest(
        _result(
            module,
            event={
                "flight_terminal_rows": [True, False, False],
                "landing_crossing_rows": [True, False, False],
                "r06_present_rows": [True, False, False],
                "r06_eligible_rows": [True, False, False],
                "selected_reset_rows": [False, True, True],
            },
            bits=[0, 1, 18],
            done=[0, 1, 1],
            generation=[0, 1, 1],
            time_outs=[False, True, False],
            outcome=[3, 0, 0],
            phase=[6, 0, 0],
            landing=[True, False, False],
            opponent_bound=[True, False, False],
        )
    )
    ledger.ingest(
        _result(
            module,
            event={
                "reveal_rows": [False, True, True],
                "reveal_due_rows": [False, True, True],
                "shot_retired_rows": [True, False, False],
                "recovery_success_rows": [True, False, False],
            },
            generation=[0, 1, 1],
            outcome=[3, 0, 0],
            phase=[8, 2, 2],
            landing=[True, False, False],
            opponent_bound=[True, False, False],
        )
    )
    prepared = _prepare(ledger, 0, environment_steps=2)
    record = _decode(prepared)

    assert record["schema_version"] == 9
    assert record["run_identity"] == _run_identity()
    assert record["run_identity"]["plant_model"]["runtime_attach"][
        "final_augmented_mjb"
    ] == _plant_contract().expected_plant_model_identity()["runtime_attach"][
        "final_augmented_mjb"
    ]
    assert record["update_index"] == 0
    assert record["num_envs"] == 3
    assert record["num_steps_per_env"] == 2
    assert record["transitions_delta"] == 6
    assert record["transitions_cumulative"] == 6
    assert len(record["extras_counts"]) == 31
    assert record["storage_finite"] == {
        name: True for name in STORAGE_WIDTHS
    }
    assert record["storage_domains"] == {
        "dones_binary": True, "sigma_positive": True,
    }
    assert record["extras_counts"]["reveal_rows"] == 2
    assert record["extras_counts"]["scheduled_due_rows"] == 2
    assert record["extras_counts"]["due_terminal_overlap_rows"] == 0
    assert record["extras_counts"]["reveal_due_rows"] == 2
    assert record["extras_counts"]["reveal_deferred_rows"] == 0
    assert record["extras_counts"]["shot_retired_rows"] == 1
    assert record["extras_counts"]["r07_present_rows"] == 0
    assert record["selected_reset_rows"] == 2
    assert record["gym_reset_rows"] == 2
    assert record["environment_steps_delta"] == 2
    assert record["environment_steps_cumulative"] == 2
    assert record["lifecycle_counts"]["reset_generation_rows"] == 2
    assert record["fact_integrity_counts"] == {
        name: 0 for name in module.FACT_INTEGRITY_COUNT_NAMES
    }
    assert record["classification_status_counts"] == {
        "0": 6, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
    }
    assert record["outcome_code_counts"] == {
        "0": 0, "1": 0, "2": 0, "3": 1, "4": 0, "5": 0, "6": 0
    }
    assert record["phase_counts"] == {
        "0": 2, "2": 2, "5": 0, "6": 1, "8": 1
    }
    per_row_reward = sum(range(1, module.REWARD_TERM_COUNT + 1)) / 100.0
    assert record["episodes"] == {
        "completed_count": 2,
        "return_sum": pytest.approx(2 * per_row_reward),
        "length_sum": 2,
    }
    assert record["rollout_policy_mean_std"] == pytest.approx(0.02)
    assert record["terminal_bit_counts"] == {
        "time_out": 1,
        "base_fell_tilt": 1,
        "base_too_low": 0,
        "joint_qdes_forbidden": 0,
        "robot_hit_table": 1,
    }
    reward = record["reward_graph"]
    assert reward["term_names"] == list(module.REWARD_TERM_NAMES)
    assert reward["term_count"] == module.REWARD_TERM_COUNT
    assert len(reward["term_sums"]) == module.REWARD_TERM_COUNT
    assert reward["term_sums"][0] == pytest.approx(0.06)
    assert reward["actual_reward_sum"] == pytest.approx(6 * per_row_reward)
    assert reward["reward_terms_finite_rows"] == 6
    assert reward["reward_terms_nonfinite_rows"] == 0
    assert reward["conservation_fault_rows"] == 0
    assert reward["playback_paddle_prior"] == {
        "term_names": list(module.PADDLE_PRIOR_TERM_NAMES),
        "row_count": 0,
        "finite_rows": [0] * module.PADDLE_PRIOR_TERM_COUNT,
        "kernel_sum": [0.0] * module.PADDLE_PRIOR_TERM_COUNT,
        "kernel_sumsq": [0.0] * module.PADDLE_PRIOR_TERM_COUNT,
        "domain_violation_rows": [0] * module.PADDLE_PRIOR_TERM_COUNT,
    }
    assert record["action_identity"] == {
        "action_slot": 0,
        "action_uid": UID,
        "mount_normal_sign": 1,
        "family": "forehand",
        "family_source": "runner_pinned_identity",
        "observed_rows": 6,
        "slot0_rows": 6,
        "uid_rows": 6,
        "mount_sign_rows": 6,
        "identity_rows": 6,
        "family_counts": {"forehand": 6, "backhand": 0},
    }
    assert "rate" not in prepared.payload.decode("utf-8")
    assert "not_produced" not in prepared.payload.decode("utf-8")


def test_playback_paddle_prior_reuses_reward_terms_for_non_gating_moments():
    module = _load()
    ledger = _ledger(module, steps=1, num_envs=4)
    result = _result(
        module,
        num_envs=4,
        paddle_playback=[True, True, False, True],
    )
    kernels = torch.tensor(
        [
            [1.0, 0.50, 0.25, 0.20],
            [0.50, 0.25, 0.20, 0.10],
            [0.75, 0.75, 0.75, 0.75],
            [0.00, 0.20, 0.10, 0.05],
        ],
        dtype=torch.float32,
    )
    max_payment = torch.tensor(
        module.PADDLE_PRIOR_WEIGHTS, dtype=torch.float32
    ) * module.FULLMDP_POLICY_STEP_S
    terms = result[3]["reward_terms"]
    terms[
        :,
        module.PADDLE_PRIOR_TERM_START : (
            module.PADDLE_PRIOR_TERM_START
            + module.PADDLE_PRIOR_TERM_COUNT
        ),
    ] = kernels * max_payment
    result = (result[0], terms.sum(1), result[2], result[3])

    ledger.ingest(result)
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    measured = record["reward_graph"]["playback_paddle_prior"]
    playback = terms[
        [0, 1, 3],
        module.PADDLE_PRIOR_TERM_START : (
            module.PADDLE_PRIOR_TERM_START
            + module.PADDLE_PRIOR_TERM_COUNT
        ),
    ].to(torch.float64) / torch.tensor(
        [
            weight * module.FULLMDP_POLICY_STEP_S
            for weight in module.PADDLE_PRIOR_WEIGHTS
        ],
        dtype=torch.float64,
    )
    assert measured["term_names"] == list(module.PADDLE_PRIOR_TERM_NAMES)
    assert measured["row_count"] == 3
    assert measured["finite_rows"] == [3, 3, 3, 3]
    assert measured["kernel_sum"] == pytest.approx(
        playback.sum(0).tolist()
    )
    assert measured["kernel_sumsq"] == pytest.approx(
        playback.square().sum(0).tolist()
    )
    assert measured["domain_violation_rows"] == [0, 0, 0, 0]


def test_playback_paddle_prior_domain_violations_are_visible_but_non_gating():
    module = _load()
    ledger = _ledger(module, steps=1, num_envs=2)
    result = _result(
        module,
        num_envs=2,
        paddle_playback=[True, True],
    )
    kernels = torch.tensor(
        [[-0.25, 1.25, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    max_payment = torch.tensor(
        module.PADDLE_PRIOR_WEIGHTS, dtype=torch.float32
    ) * module.FULLMDP_POLICY_STEP_S
    terms = result[3]["reward_terms"]
    terms[
        :,
        module.PADDLE_PRIOR_TERM_START : (
            module.PADDLE_PRIOR_TERM_START
            + module.PADDLE_PRIOR_TERM_COUNT
        ),
    ] = kernels * max_payment
    result = (result[0], terms.sum(1), result[2], result[3])

    ledger.ingest(result)
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    measured = record["reward_graph"]["playback_paddle_prior"]
    assert measured["finite_rows"] == [2, 2, 2, 2]
    assert measured["domain_violation_rows"] == [1, 1, 0, 0]


def test_prepare_does_not_gate_structurally_clean_zero_business_telemetry():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert set(record["extras_counts"].values()) == {0}
    assert "event_semantics_fault_rows" not in record["lifecycle_counts"]


@pytest.mark.parametrize(
    "mutation,expected",
    (
        (lambda module, result: result[3].pop("full_a_r06_present_event"), "full_a_r06_present_event"),
        (lambda module, result: result[3].pop("reward_terms"), "reward_terms"),
        (
            lambda module, result: result[3].__setitem__(
                "reward_terms",
                torch.zeros(3, module.REWARD_TERM_COUNT - 1),
            ),
            "reward_terms",
        ),
        (lambda module, result: result[3].pop("full_a_action_uid"), "full_a_action_uid"),
        (
            lambda module, result: result[3].pop(
                "full_a_paddle_prior_playback"
            ),
            "full_a_paddle_prior_playback",
        ),
        (
            lambda module, result: result[3].__setitem__(
                "full_a_paddle_prior_playback",
                torch.zeros(3, dtype=torch.long),
            ),
            "full_a_paddle_prior_playback",
        ),
        (
            lambda module, result: result[3].pop(
                "full_a_fact_integrity_fault_bits"
            ),
            "full_a_fact_integrity_fault_bits",
        ),
        (
            lambda module, result: result[3].__setitem__(
                "full_a_r07_present_event", torch.zeros(3, dtype=torch.long)
            ),
            "full_a_r07_present_event",
        ),
        (
            lambda module, result: result[3].__setitem__(
                "full_a_fact_integrity_fault_bits",
                torch.zeros(3, dtype=torch.bool),
            ),
            "full_a_fact_integrity_fault_bits",
        ),
        (
            lambda module, result: result[3].__setitem__(
                "full_a_fact_integrity_fault_bits",
                torch.zeros(2, dtype=torch.long),
            ),
            "full_a_fact_integrity_fault_bits",
        ),
    ),
)
def test_prepare_rejects_missing_or_wrong_schema_before_optimizer(mutation, expected):
    module = _load()
    ledger = _ledger(module, steps=1)
    result = _result(module)
    mutation(module, result)
    ledger.ingest(result)
    with pytest.raises(RuntimeError, match=expected):
        _prepare(ledger, 0, environment_steps=1)


@pytest.mark.parametrize("target", ("terms", "actual"))
def test_prepare_rejects_nonfinite_reward_rows(target):
    module = _load()
    ledger = _ledger(module, steps=1)
    result = _result(module)
    if target == "terms":
        result[3]["reward_terms"][1, 7] = torch.nan
    else:
        result[1][1] = torch.inf
    ledger.ingest(result)
    with pytest.raises(RuntimeError, match="nonfinite"):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_rejects_reward_conservation_fault():
    module = _load()
    ledger = _ledger(module, steps=1)
    result = _result(module)
    result[1][2] += 0.25
    ledger.ingest(result)
    with pytest.raises(RuntimeError, match="conservation_fault_rows"):
        _prepare(ledger, 0, environment_steps=1)


@pytest.mark.parametrize("field,value", (
    ("full_a_action_slot", 1), ("full_a_action_uid", UID + 1),
    ("full_a_mount_normal_sign", -1),
))
def test_prepare_rejects_action_identity_drift(field, value):
    module = _load()
    ledger = _ledger(module, steps=1)
    result = _result(module)
    result[3][field][0] = value
    ledger.ingest(result)
    with pytest.raises(RuntimeError, match="identity_or_finite_rows"):
        _prepare(ledger, 0, environment_steps=1)


@pytest.mark.parametrize(
    "mutation,expected",
    (
        (lambda result: result[2].fill_(2), "invalid_done_rows"),
        (
            lambda result: result[3]["time_outs"].fill_(True),
            "timeout_fault_rows",
        ),
        (
            lambda result: result[3]["backend_resolved_table_contact"].fill_(True),
            "done_explanation_fault_rows",
        ),
        (
            lambda result: result[3]["reset_generation"].fill_(1),
            "reset_generation_fault_rows",
        ),
        (
            lambda result: result[3]["full_a_outcome_code"].fill_(7),
            "outcome_unknown_rows",
        ),
        (
            lambda result: (
                result[3]["full_a_flight_terminal_event"].fill_(True),
                result[3]["full_a_outcome_code"].zero_(),
            ),
            "outcome_event_code_fault_rows",
        ),
        (
            lambda result: result[3]["full_a_phase_before_reset"].fill_(1),
            "phase_unknown_rows",
        ),
        (
            lambda result: result[3]["full_a_selected_reset_event"].fill_(True),
            "selected_reset_fault_rows",
        ),
        (
            lambda result: (
                result[2].fill_(1),
                result[3]["termination_bits"].fill_(2),
                result[3]["reset_generation"].fill_(1),
            ),
            "selected_reset_fault_rows",
        ),
    ),
)
def test_prepare_rejects_lifecycle_counterexamples(mutation, expected):
    module = _load()
    ledger = _ledger(module, steps=1)
    result = _result(module)
    mutation(result)
    ledger.ingest(result)
    with pytest.raises(RuntimeError, match=expected):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_rejects_named_missed_launch_fault_before_optimizer():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={"missed_launch_rows": [True, False, False]},
        phase=[5, 0, 0],
    ))
    with pytest.raises(RuntimeError, match="missed_launch_rows"):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_rejects_named_recovery_completion_fault_with_invalid_contact():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "racket_contact_rows": [True, False, False],
            "invalid_contact_rows": [True, False, False],
            "recovery_completion_fault_rows": [True, False, False],
        },
        status=[5, 0, 0],
        phase=[6, 0, 0],
    ))
    with pytest.raises(RuntimeError, match="recovery_completion_fault_rows"):
        _prepare(ledger, 0, environment_steps=1)


@pytest.mark.parametrize("cause_name,bit", (
    (name, bit) for name, bit in _load().FACT_INTEGRITY_CAUSES
))
def test_prepare_rejects_each_named_fact_integrity_cause_before_optimizer(
    cause_name, bit
):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module, fact_integrity_bits=[bit, 0, 0], phase=[5, 0, 0]
    ))
    with pytest.raises(RuntimeError, match=cause_name):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_rejects_unknown_fact_integrity_bit_before_optimizer():
    module = _load()
    ledger = _ledger(module, steps=1)
    unknown = 1 << module.FACT_INTEGRITY_KNOWN_MASK.bit_length()
    ledger.ingest(_result(
        module, fact_integrity_bits=[unknown, 0, 0], phase=[5, 0, 0]
    ))
    with pytest.raises(RuntimeError, match="fact_integrity_unknown_bits_rows"):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_counts_each_cause_once_when_packed_bits_are_combined():
    module = _load()
    ledger = _ledger(module, steps=1)
    causes = dict(module.FACT_INTEGRITY_CAUSES)
    ledger.ingest(_result(
        module,
        fact_integrity_bits=[
            causes["fact_integrity_r03_nonfinite_rows"]
            | causes["fact_integrity_r07_sequence_rows"],
            causes["fact_integrity_r06_source_invalid_rows"],
            causes["fact_integrity_r07_nonfinite_rows"],
        ],
        phase=[5, 5, 5],
    ))
    with pytest.raises(RuntimeError) as caught:
        _prepare(ledger, 0, environment_steps=1)
    message = str(caught.value)
    for name in causes:
        assert f"{name}=1" in message


def test_prepare_accepts_idle_retired_occupancy_and_both_due_verdicts():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "reveal_due_rows": [True, True, False],
            "reveal_rows": [True, False, False],
            "reveal_deferred_rows": [False, True, False],
        },
        phase=[2, 5, 6],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"]["reveal_due_rows"] == 2
    assert record["extras_counts"]["reveal_rows"] == 1
    assert record["extras_counts"]["reveal_deferred_rows"] == 1
    assert record["phase_counts"] == {
        "0": 0, "2": 1, "5": 1, "6": 1, "8": 0
    }


@pytest.mark.parametrize(
    "event_name",
    (
        "launch_rows",
        "r03_present_rows",
        "r07_present_rows",
        "completed_action_epoch_rows",
    ),
)
def test_prepare_records_producer_marginals_without_rebuilding_implications(
    event_name,
):
    """Only the two explicit opportunity partitions are ledger invariants."""

    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module, event={event_name: [True, False, False]}
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"][event_name] == 1
    assert "event_semantics_fault_rows" not in record["lifecycle_counts"]


@pytest.mark.parametrize(
    ("event", "fault_name"),
    (
        (
            {
                "scheduled_due_rows": [True, False, False],
                "due_terminal_overlap_rows": [True, False, False],
                "reveal_due_rows": [True, False, False],
                "reveal_rows": [True, False, False],
            },
            "scheduled_due_partition_fault_rows=1",
        ),
        (
            {"reveal_due_rows": [True, False, False]},
            "reveal_due_partition_fault_rows=1",
        ),
        (
            {"r03_physically_valid_rows": [True, False, False]},
            "r03_subset_fault_rows=1",
        ),
        (
            {"r06_eligible_rows": [True, False, False]},
            "r06_subset_fault_rows=1",
        ),
        (
            {
                "r06_present_rows": [True, False, False],
                "r06_eligible_rows": [True, False, False],
                "r06_common_rows": [True, True, False],
            },
            "r06_subset_fault_rows=1",
        ),
    ),
)
def test_prepare_rejects_broken_event_subset_or_opportunity_partition(
    event, fault_name
):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module, event=event))
    with pytest.raises(RuntimeError, match=fault_name):
        _prepare(ledger, 0, environment_steps=1)


def test_immediate_invalid_launch_is_semantically_legal_but_integrity_rejected():
    module = _load()
    ledger = _ledger(module, steps=1)
    source_invalid = dict(module.FACT_INTEGRITY_CAUSES)[
        "fact_integrity_r06_source_invalid_rows"
    ]
    ledger.ingest(_result(
        module,
        event={
            "launch_rows": [True, False, False],
            "racket_contact_rows": [True, False, False],
            "invalid_contact_rows": [True, False, False],
            "flight_terminal_rows": [True, False, False],
            "r06_present_rows": [True, False, False],
            "recovery_failure_rows": [True, False, False],
            "shot_retired_rows": [True, False, False],
        },
        status=[5, 0, 0], outcome=[6, 0, 0], phase=[8, 0, 0],
        fact_integrity_bits=[source_invalid, 0, 0],
    ))
    with pytest.raises(RuntimeError, match="r06_source_invalid") as caught:
        _prepare(ledger, 0, environment_steps=1)
    assert "event_semantics_fault_rows" not in str(caught.value)


@pytest.mark.parametrize(
    "event_name,status",
    (("opposite_contact_rows", 2), ("edge_contact_rows", 3),
     ("between_contact_rows", 4)),
)
def test_prepare_accepts_finite_source_valid_contact_business_results(
    event_name, status
):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "racket_contact_rows": [True, False, False],
            event_name: [True, False, False],
        },
        status=[status, 0, 0], phase=[5, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"][event_name] == 1
    assert set(record["fact_integrity_counts"].values()) == {0}


def test_prepare_accepts_scheduled_due_terminal_overlap_without_public_due():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "scheduled_due_rows": [True, False, False],
            "due_terminal_overlap_rows": [True, False, False],
            "selected_reset_rows": [True, False, False],
        },
        bits=[2, 0, 0],
        done=[1, 0, 0],
        generation=[1, 0, 0],
        phase=[0, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"]["scheduled_due_rows"] == 1
    assert record["extras_counts"]["due_terminal_overlap_rows"] == 1
    assert record["extras_counts"]["reveal_due_rows"] == 0
    assert "event_semantics_fault_rows" not in record["lifecycle_counts"]


def test_prepare_accepts_shot_retirement_without_gym_reset_or_generation_change():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "shot_retired_rows": [True, False, False],
            "recovery_success_rows": [True, False, False],
        },
        phase=[8, 0, 0],
        outcome=[3, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"]["shot_retired_rows"] == 1
    assert record["selected_reset_rows"] == 0
    assert record["gym_reset_rows"] == 0
    assert record["lifecycle_counts"]["reset_generation_rows"] == 0


def test_prepare_accepts_retirement_and_r07_before_same_boundary_reveal():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "scheduled_due_rows": [True, False, False],
            "reveal_due_rows": [True, False, False],
            "reveal_rows": [True, False, False],
            "r07_present_rows": [True, False, False],
            "r07_eligible_rows": [True, False, False],
            "recovery_timeout_rows": [True, False, False],
            "shot_retired_rows": [True, False, False],
        },
        phase=[2, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"]["reveal_rows"] == 1
    assert record["extras_counts"]["shot_retired_rows"] == 1
    assert record["extras_counts"]["r07_present_rows"] == 1
    assert record["phase_counts"] == {
        "0": 2, "2": 1, "5": 0, "6": 0, "8": 0
    }
    assert record["lifecycle_counts"]["reset_generation_rows"] == 0
    assert "event_semantics_fault_rows" not in record["lifecycle_counts"]


def test_prepare_accepts_gym_terminal_recovery_failure_without_shot_retirement():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "selected_reset_rows": [True, False, False],
            "recovery_failure_rows": [True, False, False],
        },
        bits=[2, 0, 0],
        done=[1, 0, 0],
        generation=[1, 0, 0],
        phase=[6, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"]["recovery_failure_rows"] == 1
    assert record["extras_counts"]["shot_retired_rows"] == 0
    assert record["selected_reset_rows"] == 1


def test_prepare_rejects_invalid_contact_failure_as_source_integrity_fault():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "racket_contact_rows": [True, False, False],
            "invalid_contact_rows": [True, False, False],
            "flight_terminal_rows": [True, False, False],
            "r06_present_rows": [True, False, False],
            "recovery_failure_rows": [True, False, False],
            "shot_retired_rows": [True, False, False],
        },
        status=[5, 0, 0],
        outcome=[6, 0, 0],
        phase=[8, 0, 0],
        fact_integrity_bits=[dict(module.FACT_INTEGRITY_CAUSES)[
            "fact_integrity_r06_source_invalid_rows"
        ], 0, 0],
    ))
    with pytest.raises(
        RuntimeError, match="fact_integrity_r06_source_invalid_rows"
    ):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_rejects_invalid_contact_with_gym_reset_before_optimizer():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "racket_contact_rows": [True, False, False],
            "invalid_contact_rows": [True, False, False],
            "flight_terminal_rows": [True, False, False],
            "r06_present_rows": [True, False, False],
            "selected_reset_rows": [True, False, False],
            "recovery_failure_rows": [True, False, False],
        },
        bits=[2, 0, 0],
        done=[1, 0, 0],
        generation=[1, 0, 0],
        status=[5, 0, 0],
        outcome=[6, 0, 0],
        phase=[6, 0, 0],
        fact_integrity_bits=[dict(module.FACT_INTEGRITY_CAUSES)[
            "fact_integrity_r06_source_invalid_rows"
        ], 0, 0],
    ))
    with pytest.raises(
        RuntimeError, match="fact_integrity_r06_source_invalid_rows"
    ):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_counts_joint_safety_telemetry_without_a_done_bit():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "actual_hard_edge_rows": [True, False, False],
            "qdes_guard_intervention_rows": [True, False, False],
        },
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["extras_counts"]["actual_hard_edge_rows"] == 1
    assert record["extras_counts"]["qdes_guard_intervention_rows"] == 1
    assert record["terminal_bit_counts"]["joint_qdes_forbidden"] == 0
    assert record["lifecycle_counts"]["gym_reset_rows"] == 0
    assert "event_semantics_fault_rows" not in record["lifecycle_counts"]


@pytest.mark.parametrize(
    "bits,resolved",
    (
        ([3, 0, 0], [False, False, False]),
        ([17, 0, 0], [True, False, False]),
    ),
)
def test_prepare_rejects_nonpure_timeout_bit(bits, resolved):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={"selected_reset_rows": [True, False, False]},
        bits=bits,
        done=[1, 0, 0],
        generation=[1, 0, 0],
        time_outs=[True, False, False],
        resolved=resolved,
    ))
    with pytest.raises(RuntimeError, match="timeout_fault_rows"):
        _prepare(ledger, 0, environment_steps=1)


def test_prepare_accepts_resolved_table_only_with_canonical_table_bit():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={"selected_reset_rows": [True, False, False]},
        bits=[16, 0, 0],
        done=[1, 0, 0],
        generation=[1, 0, 0],
        resolved=[True, False, False],
        phase=[0, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["terminal_bit_counts"]["robot_hit_table"] == 1
    assert record["lifecycle_counts"]["resolved_table_rows"] == 1
    assert record["lifecycle_counts"]["done_explanation_fault_rows"] == 0


def test_prepare_accepts_own_table_outcome_only_with_landing_crossing():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(
        module,
        event={
            "flight_terminal_rows": [True, False, False],
            "landing_crossing_rows": [True, False, False],
            "r06_present_rows": [True, False, False],
        },
        outcome=[4, 0, 0],
        phase=[6, 0, 0],
    ))
    record = _decode(_prepare(ledger, 0, environment_steps=1))
    assert record["outcome_code_counts"]["4"] == 1


def test_prepare_rejects_wrong_environment_step_boundary():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    with pytest.raises(RuntimeError, match="environment_steps=2"):
        _prepare(ledger, 0, environment_steps=2)


def test_prepare_packs_storage_health_in_the_single_copy_and_rejects_nonfinite():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    with pytest.raises(RuntimeError, match="storage is nonfinite"):
        _prepare(ledger, 0, environment_steps=1, finite=False)


@pytest.mark.parametrize(
    ("name", "v2_width"),
    (("observations_policy", 203), ("observations_critic", 219)),
)
def test_prepare_rejects_semantic_v2_rollout_observation_width_before_optimizer(
    name, v2_width
):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    storage = {
        field: torch.ones(1, ledger.num_envs, width)
        for field, width in STORAGE_WIDTHS.items()
    }
    storage[name] = torch.ones(1, ledger.num_envs, v2_width)
    with pytest.raises(RuntimeError, match="storage_tensors"):
        ledger.prepare(
            0,
            environment_steps=1,
            storage_step=1,
            storage_tensors=storage,
            storage_dones=torch.zeros(
                1, ledger.num_envs, 1, dtype=torch.uint8
            ),
            policy_std=torch.full((ledger.num_envs, 31), 0.02),
        )
    assert ledger._pending is None


@pytest.mark.parametrize("name", tuple(STORAGE_WIDTHS))
def test_prepare_rejects_every_nonfinite_rollout_storage_lane_before_optimizer(name):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    storage = {
        field: torch.ones(1, ledger.num_envs, width)
        for field, width in STORAGE_WIDTHS.items()
    }
    storage[name][0, 0, 0] = torch.inf
    with pytest.raises(RuntimeError, match="storage is nonfinite") as caught:
        ledger.prepare(
            0, environment_steps=1, storage_step=1,
            storage_tensors=storage,
            storage_dones=torch.zeros(
                1, ledger.num_envs, 1, dtype=torch.uint8
            ),
            policy_std=torch.full((ledger.num_envs, 31), 0.02),
        )
    assert name in str(caught.value)


def test_prepare_rejects_nonbinary_rollout_dones_before_optimizer():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    storage = {
        field: torch.ones(1, ledger.num_envs, width)
        for field, width in STORAGE_WIDTHS.items()
    }
    dones = torch.zeros(1, ledger.num_envs, 1, dtype=torch.uint8)
    dones[0, 0, 0] = 2
    with pytest.raises(RuntimeError, match="dones_binary"):
        ledger.prepare(
            0, environment_steps=1, storage_step=1,
            storage_tensors=storage, storage_dones=dones,
            policy_std=torch.full((ledger.num_envs, 31), 0.02),
        )


@pytest.mark.parametrize("value", (0.0, -1.0))
def test_prepare_rejects_nonpositive_finite_rollout_sigma_before_optimizer(value):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    storage = {
        field: torch.ones(1, ledger.num_envs, width)
        for field, width in STORAGE_WIDTHS.items()
    }
    storage["sigma"][0, 0, 0] = value
    with pytest.raises(RuntimeError, match="sigma_positive"):
        ledger.prepare(
            0, environment_steps=1, storage_step=1,
            storage_tensors=storage,
            storage_dones=torch.zeros(
                1, ledger.num_envs, 1, dtype=torch.uint8
            ),
            policy_std=torch.full((ledger.num_envs, 31), 0.02),
        )


@pytest.mark.parametrize("fault", ("scalar", "wrong_shape", "noncontiguous"))
def test_prepare_rejects_nonexact_or_noncontiguous_storage(fault):
    module = _load()
    ledger = _ledger(module, steps=2)
    ledger.ingest(_result(module))
    ledger.ingest(_result(module))
    shape = (ledger.num_steps, ledger.num_envs, 1)
    storage = {
        name: torch.ones(ledger.num_steps, ledger.num_envs, width)
        for name, width in STORAGE_WIDTHS.items()
    }
    if fault == "scalar":
        storage["rewards"] = torch.tensor(1.0)
    elif fault == "wrong_shape":
        storage["returns"] = torch.ones(shape[:-1])
    else:
        storage["advantages"] = torch.ones(
            ledger.num_steps, ledger.num_envs, 2
        )[:, :, :1]
        assert not storage["advantages"].is_contiguous()
    with pytest.raises(RuntimeError, match="storage_tensors"):
        ledger.prepare(
            0,
            environment_steps=2,
            storage_step=2,
            storage_tensors=storage,
            storage_dones=torch.zeros(2, ledger.num_envs, 1, dtype=torch.uint8),
            policy_std=torch.full((ledger.num_envs, 31), 0.02),
        )


def test_prepare_accepts_stock_expanded_stride_zero_policy_std():
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    policy_std = torch.full((1, 31), 0.03).expand(ledger.num_envs, 31)
    assert policy_std.stride()[0] == 0
    record = _decode(
        _prepare(ledger, 0, environment_steps=1, policy_std=policy_std)
    )
    assert record["rollout_policy_mean_std"] == pytest.approx(0.03)


def test_prepare_rejects_nonfinite_or_nonpositive_policy_std():
    module = _load()
    for value in (
        0.0,
        float("nan"),
        torch.tensor([0.0, 0.02]),
    ):
        ledger = _ledger(module, steps=1)
        ledger.ingest(_result(module))
        with pytest.raises(RuntimeError, match=r"policy[ _]std"):
            _prepare(ledger, 0, environment_steps=1, policy_std=value)


def test_landing_counts_are_crossing_events_not_sticky_state_occupancy():
    module = _load()
    ledger = _ledger(module)
    ledger.ingest(_result(
        module, landing=[True, False, False], opponent_bound=[True, False, False]
    ))
    ledger.ingest(_result(
        module,
        event={
            "flight_terminal_rows": [True, False, False],
            "landing_crossing_rows": [True, False, False],
            "r06_present_rows": [True, False, False],
        },
        outcome=[5, 0, 0], phase=[6, 2, 2],
        landing=[True, False, False], opponent_bound=[True, False, False],
    ))
    lifecycle = _decode(_prepare(ledger, 0, environment_steps=2))["lifecycle_counts"]
    assert lifecycle["landing_on_opponent_rows"] == 1
    assert lifecycle["landing_opponent_bound_rows"] == 1


def test_ack_is_post_optimizer_ordered_append_fsync_and_resets(monkeypatch, tmp_path):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    with pytest.raises(RuntimeError, match="update order"):
        _prepare(ledger, 1, environment_steps=1)
    prepared = _prepare(ledger, 0, environment_steps=1)
    with pytest.raises(RuntimeError, match="update order"):
        _prepare(ledger, 0, environment_steps=1)

    evidence = tmp_path / "updates.jsonl"
    evidence.touch()
    fd = os.open(evidence, os.O_WRONLY | os.O_APPEND)
    fsync_calls = []
    monkeypatch.setattr(module.os, "fsync", lambda seen_fd: fsync_calls.append(seen_fd))
    try:
        with pytest.raises(RuntimeError, match="ACK order"):
            ledger.ack(prepared, completed_updates=0, evidence_fd=fd,
                       optimizer_metrics=_metrics(), learning_rate=1e-3,
                       timings=_timings(), snapshot=_snapshot())
        impostor = module.PreparedUpdate(0, prepared.token, prepared.payload)
        with pytest.raises(RuntimeError, match="ACK order"):
            ledger.ack(impostor, completed_updates=1, evidence_fd=fd,
                       optimizer_metrics=_metrics(), learning_rate=1e-3,
                       timings=_timings(), snapshot=_snapshot())
        payload = ledger.ack(prepared, completed_updates=1, evidence_fd=fd,
                             optimizer_metrics=_metrics(), learning_rate=1e-3,
                             timings=_timings(), snapshot=_snapshot())
    finally:
        os.close(fd)
    assert evidence.read_bytes() == payload + b"\n"
    assert len(fsync_calls) == 1
    assert json.loads(payload)["optimizer_metrics"] == _metrics()
    assert json.loads(payload)["learning_rate"] == pytest.approx(1e-3)
    assert json.loads(payload)["timings"] == _timings()
    assert json.loads(payload)["rollout_policy_mean_std"] == pytest.approx(0.02)
    assert json.loads(payload)["snapshot"] == _snapshot()
    assert json.loads(payload)["prepared_update_sha256"] == hashlib.sha256(
        prepared.payload
    ).hexdigest()

    ledger.ingest(_result(module))
    assert _decode(_prepare(ledger, 1, environment_steps=2))["transitions_cumulative"] == 6


def test_prepare_makes_one_host_copy_and_payload_survives_device_mutation(tmp_path):
    module = _load()
    source = inspect.getsource(module.FullMdpUpdateLedger.prepare)
    assert source.count("torch.cat(") == 1
    assert source.count(".cpu()") == 1

    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module, event={
        "reveal_rows": [True, False, False],
        "reveal_due_rows": [True, False, False],
    }))
    prepared = _prepare(ledger, 0, environment_steps=1)
    original = bytes(prepared.payload)
    ledger._event.fill_(999.0)
    ledger._reward.fill_(999.0)
    evidence = tmp_path / "updates.jsonl"
    evidence.touch()
    fd = os.open(evidence, os.O_WRONLY | os.O_APPEND)
    try:
        ledger.ack(prepared, completed_updates=1, evidence_fd=fd,
                   optimizer_metrics=_metrics(), learning_rate=1e-3,
                   timings=_timings(), snapshot=None)
    finally:
        os.close(fd)
    persisted = json.loads(evidence.read_text(encoding="utf-8"))
    assert persisted["extras_counts"]["reveal_rows"] == 1
    assert json.loads(original)["extras_counts"]["reveal_rows"] == 1


def test_ack_rejects_bad_optimizer_metrics_without_mutating_ledger(tmp_path):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    prepared = _prepare(ledger, 0, environment_steps=1)
    evidence = tmp_path / "updates.jsonl"
    evidence.touch()
    fd = os.open(evidence, os.O_WRONLY | os.O_APPEND)
    try:
        with pytest.raises(RuntimeError, match="optimizer metrics"):
            ledger.ack(prepared, completed_updates=1, evidence_fd=fd,
                       optimizer_metrics={"surrogate": float("nan")},
                       learning_rate=1e-3, timings=_timings(), snapshot=None)
    finally:
        os.close(fd)
    assert evidence.read_bytes() == b""


@pytest.mark.parametrize(
    "snapshot",
    (
        {"name": "model_1.pt", "bytes": 1234, "sha256": "b" * 64},
        {"name": "model_0.pt", "bytes": 0, "sha256": "b" * 64},
        {"name": "model_0.pt", "bytes": 1234, "sha256": "not-a-sha"},
        {"name": "model_0.pt", "bytes": 1234, "sha256": "b" * 64, "extra": 1},
    ),
)
def test_ack_rejects_snapshot_receipt_not_bound_to_prepared_update(snapshot, tmp_path):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    prepared = _prepare(ledger, 0, environment_steps=1)
    evidence = tmp_path / "updates.jsonl"
    evidence.touch()
    fd = os.open(evidence, os.O_WRONLY | os.O_APPEND)
    try:
        with pytest.raises(RuntimeError, match="snapshot receipt"):
            ledger.ack(
                prepared,
                completed_updates=1,
                evidence_fd=fd,
                optimizer_metrics=_metrics(),
                learning_rate=1e-3,
                timings=_timings(),
                snapshot=snapshot,
            )
    finally:
        os.close(fd)
    assert evidence.read_bytes() == b""


def test_ack_rejects_nonmonotonic_wall_timing_without_writing(tmp_path):
    module = _load()
    ledger = _ledger(module, steps=1)
    ledger.ingest(_result(module))
    prepared = _prepare(ledger, 0, environment_steps=1)
    evidence = tmp_path / "updates.jsonl"
    evidence.touch()
    fd = os.open(evidence, os.O_WRONLY | os.O_APPEND)
    try:
        with pytest.raises(RuntimeError, match="timings"):
            ledger.ack(
                prepared,
                completed_updates=1,
                evidence_fd=fd,
                optimizer_metrics=_metrics(),
                learning_rate=1e-3,
                timings={
                    "collection_seconds": 2.0,
                    "learning_seconds": 2.0,
                    "pre_ack_iteration_seconds": 3.0,
                    "run_elapsed_pre_ack_seconds": 3.0,
                },
                snapshot=None,
            )
    finally:
        os.close(fd)
    assert evidence.read_bytes() == b""
