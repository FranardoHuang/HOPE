from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_n1_vendor_baseline_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("vendor_probe_gate_consumer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
MATERIALIZER = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_n1_vendor_probe_gate_receipt.py"
)
M_SPEC = importlib.util.spec_from_file_location(
    "vendor_probe_gate_consumer_materializer", MATERIALIZER
)
assert M_SPEC is not None and M_SPEC.loader is not None
M = importlib.util.module_from_spec(M_SPEC)
M_SPEC.loader.exec_module(M)


IDENTITY = {
    "action_id": "bh_loop_c",
    "scientific_argv_canonical_sha256": "a" * 64,
    L.VENDOR_CONTRACT_FIELD: "b" * 64,
}


def test_launcher_runtime_sources_feed_registry_bound_gate_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real launcher label set through the gate identity code."""

    checkout = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda root, commit, pin, **kwargs: (pin, root / pin["path"]),
    )
    runtime_sources = L._validate_runtime_sources(checkout, "a" * 40)
    assert "A3 vendor action registry" in runtime_sources
    assert (
        "vendor runtime training-contract identity manifest"
        not in runtime_sources
    )

    action = M._V._R.get_action_config("bh_loop_c")
    bundle_pin = dict(
        M._V._R.require_materialized_pin(
            action.contact_bundle,
            action_id=action.action_id,
            layer="contact bundle",
        )
    )
    identity_pin = dict(
        M._V._R.require_materialized_pin(
            action.required_identity_manifest,
            action_id=action.action_id,
            layer="required identity manifest",
        )
    )
    authority_pin = dict(
        M._V._R.require_materialized_pin(
            action.runtime_authority_receipt,
            action_id=action.action_id,
            layer="runtime authority receipt",
        )
    )
    contract_pin = dict(
        M._V._R.require_materialized_pin(
            action.runtime_contract,
            action_id=action.action_id,
            layer="runtime contract",
        )
    )
    payload = {
        "spec": {
            "source": {"checkout": str(checkout)},
            "action_id": action.action_id,
            "scope": action.scope,
            "seed": 0,
            "bundle": bundle_pin,
            "policy_contract_sha256": "8" * 64,
            "expected_effective_reward_recipe_sha256": "9" * 64,
            M._V.VENDOR_CONTRACT_FIELD: contract_pin["sha256"],
        },
        "runtime_sources": runtime_sources,
        "bundle": {
            "dynamic_ready": {
                "artifact": {"path": "ready.json", "sha256": "1" * 64},
                "nominal_hold_receipt": {
                    "path": "hold.json",
                    "sha256": "2" * 64,
                },
            },
            "motion": {"path": "motion.npz", "sha256": "3" * 64},
        },
        "vendor_runtime_authority": {
            "receipt_path": authority_pin["path"],
            "receipt_sha256": authority_pin["sha256"],
            "runtime_training_contract": {
                **contract_pin,
                "schema_version": 3,
            },
            "verified_vendor_runtime": {"action_id": action.action_id},
        },
        "training_argv": [
            "python",
            "train.py",
            M._V.STABLE_READY_PLANT_OVERRIDE,
            f"task.actor_obs_contract={M.ACTOR_OBS_CONTRACT}",
        ],
    }
    monkeypatch.setattr(
        M,
        "_contact_timing",
        lambda payload, checkout: {"fixture": "contact-timing"},
    )

    identity = M._scientific_identity(payload)

    assert identity["action_registry"] == runtime_sources[
        "A3 vendor action registry"
    ]
    assert identity["bundle"] == bundle_pin
    assert identity["required_identity"] == identity_pin
    assert identity["runtime_authority_receipt"] == authority_pin

    wrong_bundle = copy.deepcopy(payload)
    wrong_bundle["spec"]["bundle"]["sha256"] = "0" * 64
    with pytest.raises(M.ReceiptRefused, match="action-specific registry pin"):
        M._scientific_identity(wrong_bundle)

    wrong_authority = copy.deepcopy(payload)
    wrong_authority["vendor_runtime_authority"]["receipt_sha256"] = "0" * 64
    with pytest.raises(M.ReceiptRefused, match="action-specific registry pins"):
        M._scientific_identity(wrong_authority)


def _gate_module():
    return SimpleNamespace(
        EXPECTED_STAGES={
            "probe": {"num_envs": 4096, "max_iterations": 5, "save_interval": 1},
        },
        EXPECTED_CHECKPOINT_INDICES={
            "probe": (0, 1, 2, 3, 4),
        },
        ROLLOUT_STEPS_PER_UPDATE=24,
        POLICY_DT_S=0.02,
        PUSH_INTERVAL_RANGE_S=(1.0, 3.0),
        _scientific_identity=lambda payload: IDENTITY,
        ReceiptRefused=M.ReceiptRefused,
        _validate_abi=M._validate_abi,
        _validate_delay=M._validate_delay,
        _validate_std_lr=M._validate_std_lr,
        _validate_joint_safety=M._validate_joint_safety,
        _validate_behavior=M._validate_behavior,
        _validate_push_velocity=M._validate_push_velocity,
    )


def _checkpoint_normalizer_summary(role: str, index: int) -> dict:
    features = 194 if role == "actor" else 318
    return {
        "state_keys": ["_mean", "_std", "count"],
        "mean_key": "_mean",
        "scale_key": "_std",
        "count_key": "count",
        "feature_count": features,
        "count": float(100 + index),
        "tensor_count": 3,
        "element_count": features * 2 + 1,
        "all_finite": True,
    }


def _stage(name: str) -> dict:
    module = _gate_module()
    budget = module.EXPECTED_STAGES[name]
    updates = budget["max_iterations"]
    stage = {
        "stage": name,
        "namespace": f"/runs/{name}",
        "run_directory": f"/logs/{name}",
        "launch_claim": {
            "path": f"/runs/{name}/launch_claim.json",
            "file_sha256": "1" * 64,
            "launch_claim_sha256": "2" * 64,
        },
        "run_log": {"path": f"/runs/{name}/run.log", "sha256": "3" * 64},
        "source_commit": "9" * 40,
        "budget": budget,
        "checkpoints": [
            {
                "index": index,
                "path": f"/logs/{name}/model_{index}.pt",
                "sha256": "4" * 64,
                "embedded_iteration": index,
                "training_launch_claim_sha256": "2" * 64,
                "training_contract_sha256": "c" * 64,
                "tensor_count": 83,
                "element_count": 1_794_020,
                "all_finite": True,
                "actor_normalizer": _checkpoint_normalizer_summary(
                    "actor", index
                ),
                "critic_normalizer": _checkpoint_normalizer_summary(
                    "critic", index
                ),
            }
            for index in module.EXPECTED_CHECKPOINT_INDICES[name]
        ],
        "runtime_abi": {
            "event": "hope_rsl_rl_runtime_abi",
            "schema_version": 1,
            "runtime": {
                "distributions": [
                    {"name": "rsl-rl-lib", "version": "3.1.2"}
                ],
                "package_origin": "/venv/rsl_rl/__init__.py",
                "runner_module": "rsl_rl.runners.on_policy_runner",
                "runner_origin": "/venv/rsl_rl/runners/on_policy_runner.py",
            },
            "capabilities": {
                "empirical_normalization_preflight": True,
                "positive_realized_policy_std_guard": True,
                "normalizer_binding": {
                    "empirical_normalization": True,
                    "normalizers": {
                        role: {
                            "enabled": True,
                            "state_shapes": {"mean": [features]},
                            "semantic_buffers": {"mean": "mean"},
                        }
                        for role, features in (("actor", 194), ("critic", 318))
                    },
                },
                "policy_std_abi": {
                    "noise_std_type": "scalar",
                    "parameter_name": "std",
                    "parameter_shape": [31],
                    "parameter_count": 31,
                },
            },
        },
        "control_step_action_delay": {
            "event": "hope_control_step_action_delay_runtime",
            "schema_version": 1,
            "training_contract_sha256": "c" * 64,
            "active_action_term_names": ["joint_pos"],
            "delay_terms": [
                {
                    "term_name": "joint_pos",
                    "schema_version": 1,
                    "kind": "whole_body_tracking.policy_control_step_action_delay_receipt",
                    "num_envs": budget["num_envs"],
                    "initialized_env_count": budget["num_envs"],
                    "lag_histogram": {
                        "0": 1366,
                        "1": 1365,
                        "2": 1365,
                    },
                    "contract": {
                        "schema_version": 1,
                        "enabled": True,
                        "semantic_unit": "policy_control_step",
                        "sample_timing": "once_per_episode_reset",
                        "distribution": "discrete_uniform_inclusive",
                        "min_steps": 0,
                        "max_steps": 2,
                        "shared_across_all_31_joints": True,
                        "history_fill": "safe_default_or_action_specific_hold",
                    },
                }
            ]
        },
        "policy_std_lr_updates": [
            {
                "event": "hope_policy_std_update",
                "schema_version": 1,
                "ppo_update": update,
                "noise_std_type": "scalar",
                "parameter_name": "std",
                "parameter_shape": [31],
                "parameter_count": 31,
                "policy_std_min": 0.01,
                "policy_std_mean": 0.02,
                "policy_std_max": 0.03,
                "learning_rate": 1e-5,
                "learning_rate_at_floor": True,
            }
            for update in range(updates)
        ],
        "joint_safety": {
            "updates": [
                {
                    "event": "hope_joint_safety_diagnostic_compact_update",
                    "schema_version": 1,
                    "status": "diagnostic_compact_optimizer_committed_and_ledger_acknowledged",
                    "ppo_update": update,
                    "num_envs": budget["num_envs"],
                    "policy_step_count": 24,
                    "minimum_hard_gap_rad": 0.1,
                    "counter_totals": {
                        "actual_hard_edge_events": 0,
                        "qdes_events": 0,
                        "policy_crossing_events": 0,
                        "substep_crossing_events": 0,
                        "policy_steps": budget["num_envs"] * 24,
                        "complete_policy_steps": budget["num_envs"] * 24,
                    },
                }
                for update in range(updates)
            ],
            "aggregate_counter_totals": {
                "actual_hard_edge_events": 0,
                "complete_policy_steps": budget["num_envs"] * 24 * updates,
                "policy_steps": budget["num_envs"] * 24 * updates,
                "qdes_events": 0,
                "policy_crossing_events": 0,
                "substep_crossing_events": 0,
            },
            "minimum_hard_gap_rad": 0.1,
            "fatal_marker_count": 0,
        },
        "behavior": {
            "updates": [
                {
                    "event": "hope_exact_behavior_update",
                    "schema_version": 1,
                    "ppo_update": update,
                    "counters": {
                        # Deliberately poor quality telemetry: the integrated
                        # probe authorizes only hard safety/runtime integrity,
                        # not table/fall/reachability quality.
                        "physical_fall_count": 1000,
                        "pre_strike_physical_fall_count": 1000,
                        "post_strike_physical_fall_count": 0,
                        "terminal_reset_count": 1000,
                        "timeout_reset_count": 0,
                        "non_physical_terminal_reset_count": 0,
                        "ready_nonfinite_value_count": 0,
                        "strike_opportunity_count": 0,
                        "swing_start_count": 0,
                        "swing_outcome_count": 0,
                        "reference_guard_union_count": 0,
                        "reference_guard_reference_only_count": 0,
                        "reference_guard_reference_and_hard_count": 0,
                        "reference_guard_hard_without_snapshot_count": 0,
                        M._ENTRY_COUNT: 1,
                        M._ENTRY_NONFINITE: 0,
                        **{key: 0 for key in M._ENTRY_BUCKETS},
                        "table_guard_first_hit_total_count": 0,
                        **{
                            f"table_guard_first_hit_category_{category}_count": 0
                            for category in M._TABLE_ATTRIBUTION_CATEGORIES
                        },
                        **{
                            f"table_guard_first_hit_phase_{phase}_count": 0
                            for phase in M._TABLE_ATTRIBUTION_PHASES
                        },
                        "termination_reason_joint_actual_forbidden_count": 0,
                        "termination_reason_joint_qdes_forbidden_count": 0,
                        "termination_reason_robot_hit_table_count": 1000,
                    },
                }
                for update in range(updates)
            ],
        },
        "push_velocity_diagnostic": {
            "updates": [
                {
                    "event": "hope_push_velocity_diagnostic_update",
                    "schema_version": 1,
                    "ppo_update": update,
                    "counters": {
                        "event_call_count": 1 if update == updates - 1 else 0,
                        "env_application_count": (
                            1 if update == updates - 1 else 0
                        ),
                        "delta_nonfinite_element_count": 0,
                        "axes": {
                            axis: {
                                "observed_delta_min": (
                                    -0.05
                                    if update == updates - 1
                                    else None
                                ),
                                "observed_delta_max": (
                                    -0.01
                                    if update == updates - 1
                                    else None
                                ),
                                "below_range_count": 0,
                                "above_range_count": 0,
                            }
                            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
                        },
                    },
                }
                for update in range(updates)
            ],
        },
        "training_completion": {
            "cleanup_complete": True,
            "completed_ppo_updates": updates,
            "event": "hope_training_complete",
            "num_envs": budget["num_envs"],
            "schema_version": 1,
            "stage": name,
            "training_contract_sha256": "c" * 64,
            "training_launch_claim_sha256": "2" * 64,
            "vendor_runtime_training_contract_sha256": "b" * 64,
        },
    }
    sources = {
        label: {"path": pin["path"], "sha256": pin["sha256"]}
        for label, pin in L.INTEGRATED_PROBE_RUNTIME_SOURCE_PINS.items()
    }
    stage["push_timer_control_flow"] = {
        "runtime_sources": sources,
        "push_semantics": "velocity_only",
        "interval_range_s": [1.0, 3.0],
        "rollout_steps_per_update": 24,
        "policy_dt_s": 0.02,
        "duration_s": 2.4,
        "interval_lower_bound_s": 1.0,
        "duration_crosses_interval_lower_bound": True,
        "runtime_observation_required": True,
        "push_counter": {
            "kind": "runtime_observed_nonzero_v2",
            "event_call_count": 1,
            "minimum_event_call_count": 1,
            "environment_application_count": 1,
            "minimum_environment_application_count": 1,
        },
    }
    stage["joint_safety"] = module._validate_joint_safety(
        stage["joint_safety"]["updates"],
        [],
        updates=updates,
        num_envs=budget["num_envs"],
    )
    stage["behavior"] = module._validate_behavior(
        stage["behavior"]["updates"],
        stage=name,
        updates=updates,
        num_envs=budget["num_envs"],
    )
    stage["push_velocity_diagnostic"] = module._validate_push_velocity(
        stage["push_velocity_diagnostic"]["updates"],
        stage=name,
        updates=updates,
        num_envs=budget["num_envs"],
    )
    return stage


def _receipt() -> dict:
    receipt = {
        "schema_version": 2,
        "kind": L.VENDOR_PROBE_GATE_KIND,
        "verdict": "PASS",
        "producer": {
            "source": {"path": L.VENDOR_PROBE_GATE_PRODUCER_SOURCE, "sha256": "5" * 64},
            "gate_source_commit": "9" * 40,
            "algorithm": "exact_integrated_probe_v2",
            "self_reference_free": True,
        },
        "evidence_source_commit": "9" * 40,
        "scientific_identity": IDENTITY,
        "stages": {"probe": _stage("probe")},
        "acceptance": {
            "integrated_probe_exact_pass": True,
            "finite_checkpoints": True,
            "normalizer_checkpoint_persistence": True,
            "runtime_abi_exact": True,
            "control_step_delay_exact": True,
            "positive_policy_std_and_finite_lr": True,
            "zero_actual_hard_edge": True,
            "zero_qdes_edge": True,
            "zero_nonfinite": True,
            "push_timer_lower_bound_crossed": True,
            "velocity_only_push_observed_nonzero": True,
            "velocity_push_six_axis_extrema_finite_and_in_range": True,
            "natural_training_completion": True,
        },
        "successor_policy": {
            "required_gate_source_ancestor_commit": "9" * 40,
            "allowed_artifact_descendant_diff": {
                "exact_paths": ["configs/n1_vendor_probe_gate_20260731/pass.json", "configs/n1_vendor_launch_20260731/bh_loop_c.long.template.json"],
                "prefixes": ["docs/"],
            },
        },
        "authorization": {
            "vendor_n1_long_launch": True,
            "formal_evidence": False,
            "curriculum_promotion": False,
            "resume": False,
            "export": False,
            "judge": False,
            "deployment": False,
            "hardware": False,
        },
    }
    receipt["content_sha256"] = L.canonical_sha256(receipt)
    return receipt


def _reseal(receipt: dict) -> None:
    receipt["content_sha256"] = L.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )


def _install_validator_fixtures(monkeypatch: pytest.MonkeyPatch, receipt: dict) -> None:
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "configs/n1_vendor_probe_gate_20260731/pass.json", "sha256": "6" * 64},
            receipt,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (receipt["producer"]["source"], Path("/producer")),
    )
    monkeypatch.setattr(L, "_git_is_ancestor", lambda *args, **kwargs: True)
    module = _gate_module()
    module._stage_evidence = lambda namespace, run_dir, expected_stage: (
        receipt["stages"][expected_stage],
        IDENTITY,
    )
    monkeypatch.setattr(L, "_load_probe_gate_module", lambda checkout: module)
    monkeypatch.setattr(L, "_validate_probe_gate_descendant_policy", lambda *args, **kwargs: None)


def test_exact_pass_receipt_is_recomputed_and_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    assert (
        receipt["stages"]["probe"]["control_step_action_delay"][
            "training_contract_sha256"
        ]
        != receipt["scientific_identity"][L.VENDOR_CONTRACT_FIELD]
    )
    _install_validator_fixtures(monkeypatch, receipt)
    result = L._validate_vendor_probe_gate_receipt(
        tmp_path,
        "7" * 40,
        {"path": "configs/n1_vendor_probe_gate_20260731/pass.json", "sha256": "6" * 64},
        spec={},
        payload={},
    )
    assert result["authorization"]["vendor_n1_long_launch"] is True
    assert set(receipt["stages"]) == {"probe"}
    push = receipt["stages"]["probe"]["push_velocity_diagnostic"]
    assert push["aggregate"]["event_call_count"] == 1
    assert push["aggregate"]["env_application_count"] == 1
    assert set(push["axis_extrema"]) == {
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    }
    assert all(
        extrema["finite_and_in_range"] is True
        for extrema in push["axis_extrema"].values()
    )
    assert all(
        extrema["observed_delta_max"] < 0.0
        for extrema in push["axis_extrema"].values()
    )
    behavior = receipt["stages"]["probe"]["behavior"]
    assert behavior["reachability_and_failure_rates"][
        "within_telemetry_thresholds"
    ] is False
    assert behavior["reachability_and_failure_rates"]["telemetry_only"] is True
    assert behavior["table_guard_attribution"]["conserves"] is False
    assert behavior["table_guard_attribution"]["telemetry_only"] is True
    assert behavior["strike_window_entry_conservation"]["matches"] is False
    assert behavior["strike_window_entry_conservation"]["telemetry_only"] is True

    tampered = copy.deepcopy(receipt)
    tampered["stages"]["probe"]["joint_safety"]["aggregate_counter_totals"][
        "actual_hard_edge_events"
    ] = 1
    _reseal(tampered)
    _install_validator_fixtures(monkeypatch, tampered)
    with pytest.raises(L.LaunchRefused, match="joint-hard/qdes"):
        L._validate_vendor_probe_gate_receipt(
            tmp_path,
            "7" * 40,
            {"path": "configs/n1_vendor_probe_gate_20260731/pass.json", "sha256": "6" * 64},
            spec={},
            payload={},
        )

    incomplete = copy.deepcopy(receipt)
    incomplete["stages"]["probe"]["training_completion"][
        "cleanup_complete"
    ] = False
    _reseal(incomplete)
    _install_validator_fixtures(monkeypatch, incomplete)
    with pytest.raises(L.LaunchRefused, match="natural-completion"):
        L._validate_vendor_probe_gate_receipt(
            tmp_path,
            "7" * 40,
            {
                "path": "configs/n1_vendor_probe_gate_20260731/pass.json",
                "sha256": "6" * 64,
            },
            spec={},
            payload={},
        )


@pytest.mark.parametrize(
    "failure",
    (
        "checkpoint",
        "normalizer",
        "abi",
        "delay",
        "std_lr",
        "qdes",
        "nonfinite",
    ),
)
def test_integrated_probe_retains_core_rejection_gates(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    stage = receipt["stages"]["probe"]
    if failure == "checkpoint":
        stage["checkpoints"][0]["all_finite"] = False
    elif failure == "normalizer":
        stage["checkpoints"][0]["actor_normalizer"]["count"] = 0.0
    elif failure == "abi":
        stage["runtime_abi"]["capabilities"][
            "empirical_normalization_preflight"
        ] = False
    elif failure == "delay":
        stage["control_step_action_delay"]["delay_terms"][0]["lag_histogram"] = {
            "0": 4096,
            "1": 0,
            "2": 0,
        }
    elif failure == "std_lr":
        stage["policy_std_lr_updates"][0]["learning_rate"] = 0.0
    elif failure == "qdes":
        stage["joint_safety"]["aggregate_counter_totals"]["qdes_events"] = 1
    else:
        stage["push_velocity_diagnostic"]["updates"][-1]["counters"][
            "delta_nonfinite_element_count"
        ] = 1
        stage["push_velocity_diagnostic"]["aggregate"][
            "delta_nonfinite_element_count"
        ] = 1
    _reseal(receipt)
    _install_validator_fixtures(monkeypatch, receipt)

    with pytest.raises(L.LaunchRefused):
        L._validate_vendor_probe_gate_receipt(
            tmp_path,
            "7" * 40,
            {
                "path": "configs/n1_vendor_probe_gate_20260731/pass.json",
                "sha256": "6" * 64,
            },
            spec={},
            payload={},
        )


@pytest.mark.parametrize(
    "failure", ("zero_observation", "missing_extrema", "range")
)
def test_integrated_probe_push_front_door_rejects_invalid_observation(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    stage = receipt["stages"]["probe"]
    diagnostic = stage["push_velocity_diagnostic"]
    last = diagnostic["updates"][-1]["counters"]
    if failure == "zero_observation":
        last["event_call_count"] = 0
        last["env_application_count"] = 0
        for extrema in last["axes"].values():
            extrema["observed_delta_min"] = None
            extrema["observed_delta_max"] = None
        diagnostic["aggregate"]["event_call_count"] = 0
        diagnostic["aggregate"]["env_application_count"] = 0
        stage["push_timer_control_flow"]["push_counter"]["event_call_count"] = 0
        stage["push_timer_control_flow"]["push_counter"][
            "environment_application_count"
        ] = 0
    elif failure == "missing_extrema":
        last["axes"]["x"]["observed_delta_max"] = None
    else:
        last["axes"]["x"]["above_range_count"] = 1
        diagnostic["aggregate"]["above_range_count"] = 1
    _reseal(receipt)
    _install_validator_fixtures(monkeypatch, receipt)

    with pytest.raises(L.LaunchRefused, match="push|velocity"):
        L._validate_vendor_probe_gate_receipt(
            tmp_path,
            "7" * 40,
            {
                "path": "configs/n1_vendor_probe_gate_20260731/pass.json",
                "sha256": "6" * 64,
            },
            spec={},
            payload={},
        )


def test_artifact_descendant_rejects_launcher_or_producer_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = _receipt()["producer"]
    policy = _receipt()["successor_policy"]
    monkeypatch.setattr(L, "_git_is_ancestor", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        L._B,
        "_run_git",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "M\t" + L.LAUNCHER_SOURCE
                + "\nA\tconfigs/n1_vendor_probe_gate_20260731/pass.json\n"
            ),
            stderr="",
        ),
    )
    with pytest.raises(L.LaunchRefused, match="non-allowlisted"):
        L._validate_probe_gate_descendant_policy(
            tmp_path,
            "7" * 40,
            receipt_pin={
                "path": "configs/n1_vendor_probe_gate_20260731/pass.json",
                "sha256": "6" * 64,
            },
            spec={},
            producer=producer,
            successor_policy=policy,
        )


def test_receipt_v2_rejects_legacy_second_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    receipt["stages"]["push_evidence"] = copy.deepcopy(
        receipt["stages"]["probe"]
    )
    receipt["stages"]["push_evidence"]["stage"] = "push_evidence"
    _reseal(receipt)
    _install_validator_fixtures(monkeypatch, receipt)
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_vendor_probe_gate_receipt(
            tmp_path,
            "7" * 40,
            {
                "path": "configs/n1_vendor_probe_gate_20260731/pass.json",
                "sha256": "6" * 64,
            },
            spec={},
            payload={},
        )


def test_delay_histogram_must_exercise_all_vendor_lags() -> None:
    delay = _stage("probe")["control_step_action_delay"]
    delay["delay_terms"][0]["lag_histogram"] = {
        "0": 4096,
        "1": 0,
        "2": 0,
    }
    with pytest.raises(M.ReceiptRefused, match="histogram/contract"):
        M._validate_delay([delay], num_envs=4096)


def test_plan_and_internal_exec_both_revalidate_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = {
        "stage": "long",
        "source": {"checkout": str(tmp_path), "commit_sha": "7" * 40},
        L.VENDOR_CONTRACT_FIELD: "a" * 64,
        L.VENDOR_PROBE_GATE_FIELD: {"path": "configs/n1_vendor_probe_gate_20260731/pass.json", "sha256": "6" * 64},
        "bundle": {},
        "action_id": "bh_loop_c",
        "scope": "upper",
    }
    payload = {"spec": spec, "bundle": {}}
    plan = {"canonical_payload": payload, "launch_claim_sha256": "0" * 64}
    calls = []
    monkeypatch.setattr(L._B, "build_plan", lambda path: plan)
    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda *args, **kwargs: {
            "runtime_training_contract_sha256": "a" * 64
        },
    )
    monkeypatch.setattr(
        L, "_validate_actual_vendor_authority", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        L, "_validate_vendor_runtime_binding", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        L,
        "_validate_vendor_probe_gate_receipt",
        lambda *args, **kwargs: calls.append("gate") or {"pin": spec[L.VENDOR_PROBE_GATE_FIELD]},
    )
    L.build_plan(tmp_path / "spec.json")
    assert calls == ["gate"]

    payload[L.VENDOR_PROBE_GATE_FIELD] = {"pin": spec[L.VENDOR_PROBE_GATE_FIELD]}
    monkeypatch.setattr(L, "_load_internal_plan_for_vendor_binding", lambda *args: (plan, payload))
    monkeypatch.setattr(L, "_validate_spec_document", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        L, "_revalidate_integrated_probe_claim_sources", lambda payload: None
    )
    monkeypatch.setattr(L._B, "_verify_clean_source", lambda *args: {})
    monkeypatch.setattr(L._B, "_validate_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(L._B, "_internal_exec", lambda *args: 0)
    assert L._internal_exec(tmp_path / "claim.json", "0" * 64, 3) == 0
    assert calls == ["gate", "gate"]
