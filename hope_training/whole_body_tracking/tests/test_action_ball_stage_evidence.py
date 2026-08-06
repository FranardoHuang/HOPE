from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCER_PATH = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/scripts/action_ball_stage_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_stage_evidence_under_test", PRODUCER_PATH
)
assert SPEC is not None and SPEC.loader is not None
PRODUCER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRODUCER
SPEC.loader.exec_module(PRODUCER)

TABLE_PRODUCER_PATH = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/scripts/check_table_obstacle_scene.py"
)
TABLE_SPEC = importlib.util.spec_from_file_location(
    "check_table_obstacle_scene_stage_roundtrip_test", TABLE_PRODUCER_PATH
)
assert TABLE_SPEC is not None and TABLE_SPEC.loader is not None
TABLE_PRODUCER = importlib.util.module_from_spec(TABLE_SPEC)
sys.modules[TABLE_SPEC.name] = TABLE_PRODUCER
TABLE_SPEC.loader.exec_module(TABLE_PRODUCER)


def _write_canonical_json(path, value):
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _delay_contract():
    return {
        "schema_version": 1,
        "enabled": True,
        "semantic_unit": "policy_control_step",
        "sample_timing": "once_per_episode_reset",
        "distribution": "discrete_uniform_inclusive",
        "min_steps": 0,
        "max_steps": 2,
        "shared_across_all_31_joints": True,
        "history_fill": "safe_default_or_action_specific_hold",
    }


def _delay_runtime_record(*, contract_sha="a" * 64, num_envs=4):
    return {
        "event": "hope_control_step_action_delay_runtime",
        "schema_version": 1,
        "training_contract_sha256": contract_sha,
        "active_action_term_names": ["joint_pos", "aux"],
        "delay_terms": [
            {
                "term_name": "joint_pos",
                "schema_version": 1,
                "kind": (
                    "whole_body_tracking."
                    "policy_control_step_action_delay_receipt"
                ),
                "contract": _delay_contract(),
                "num_envs": num_envs,
                "initialized_env_count": num_envs,
                "lag_histogram": {"0": 1, "1": 1, "2": num_envs - 2},
            }
        ],
    }


def _write_delay_runtime_log(path, records):
    lines = ["ordinary trainer output"]
    lines.extend(
        PRODUCER.CONTROL_STEP_ACTION_DELAY_RUNTIME_PREFIX
        + json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for record in records
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_control_step_delay_runtime_receipt_binds_contract_and_histogram(tmp_path):
    log_path = tmp_path / "train.log"
    record = _delay_runtime_record()
    _write_delay_runtime_log(log_path, [record])

    evidence = PRODUCER._control_step_action_delay_runtime_evidence(
        log_path=log_path,
        training_contract={
            "schema_version": 3,
            "control_step_action_delay": _delay_contract(),
        },
        training_contract_sha256="a" * 64,
        expected_num_envs=4,
        expected_action_term_order=["joint_pos", "aux"],
    )

    assert evidence["status"] == "passed"
    assert evidence["runtime_log_line_number"] == 2
    assert evidence["active_action_term_names"] == ["joint_pos", "aux"]
    assert evidence["delay_terms"][0]["lag_histogram"] == {
        "0": 1,
        "1": 1,
        "2": 2,
    }


def test_disabled_control_step_delay_keeps_legacy_no_receipt_behavior(tmp_path):
    missing_log = tmp_path / "does-not-exist.log"
    assert PRODUCER._control_step_action_delay_runtime_evidence(
        log_path=missing_log,
        training_contract={"schema_version": 3},
        training_contract_sha256="a" * 64,
        expected_num_envs=4,
        expected_action_term_order=[],
    ) is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: None, "exactly one"),
        (
            lambda record: record.update(training_contract_sha256="b" * 64),
            "identity/contract SHA",
        ),
        (
            lambda record: record["delay_terms"][0].update(num_envs=3),
            "term/histogram",
        ),
        (
            lambda record: record["delay_terms"][0].update(
                lag_histogram={"0": 2, "1": 2, "2": 2}
            ),
            "term/histogram",
        ),
        (
            lambda record: record["delay_terms"][0]["contract"].update(
                sample_timing="every_control_step"
            ),
            "term/histogram",
        ),
        (
            lambda record: record.update(
                active_action_term_names=["aux", "joint_pos"]
            ),
            "term order",
        ),
    ),
)
def test_control_step_delay_runtime_receipt_rejects_missing_or_tampered(
    tmp_path, mutation, message
):
    log_path = tmp_path / "train.log"
    record = _delay_runtime_record()
    mutation(record)
    records = [] if message == "exactly one" else [record]
    _write_delay_runtime_log(log_path, records)

    with pytest.raises(PRODUCER.EvidenceError, match=message):
        PRODUCER._control_step_action_delay_runtime_evidence(
            log_path=log_path,
            training_contract={
                "schema_version": 3,
                "control_step_action_delay": _delay_contract(),
            },
            training_contract_sha256="a" * 64,
            expected_num_envs=4,
            expected_action_term_order=["joint_pos", "aux"],
        )


def test_control_step_delay_runtime_receipt_rejects_duplicate(tmp_path):
    log_path = tmp_path / "train.log"
    record = _delay_runtime_record()
    _write_delay_runtime_log(log_path, [record, record])
    with pytest.raises(PRODUCER.EvidenceError, match="exactly one"):
        PRODUCER._control_step_action_delay_runtime_evidence(
            log_path=log_path,
            training_contract={
                "schema_version": 3,
                "control_step_action_delay": _delay_contract(),
            },
            training_contract_sha256="a" * 64,
            expected_num_envs=4,
            expected_action_term_order=["joint_pos", "aux"],
        )


def _bindings():
    return [
        {
            "motion_id": action_id,
            "action_uid": index + 1,
            "motion_sha256": f"{index + 1:064x}",
            "scope": "upper",
            "mobility_mode": "no_move",
            "arm_width_spec": {
                arm: {
                    "initial": 0.01,
                    "maximum": 0.11,
                    "unit": "fixture",
                }
                for arm in PRODUCER.ARM_KEYS
            },
        }
        for index, action_id in enumerate(PRODUCER.ACTION_ORDER)
    ]


def _ledger(proposed: int, *, safe_nonreturn: int = 1):
    return {
        "proposed": proposed,
        "physics_invalid": 0,
        "solver_rejected": 0,
        "solver_admitted": proposed,
        "installed": proposed,
        "started": proposed,
        "closed": proposed,
        "legal_return": proposed - safe_nonreturn,
        "safe_nonreturn": safe_nonreturn,
        "table_hit": 0,
        "fall": 0,
        "collision": 0,
        "joint_qdes_limit": 0,
        "joint_actual_limit": 0,
        "infrastructure_invalid": 0,
        "physics_invalid_reasons": {},
        "solver_reject_reasons": {},
    }


def _formal_records():
    arms = (
        "time_to_contact_lower",
        "contact_y_upper",
        "incoming_speed_lower",
        "spin_direction_u_pos",
        "base_spawn_x_upper",
    )
    return [
        {
            "action_uid": binding["action_uid"],
            "profile_sha256": f"{index + 101:064x}",
            "mobility_mode": "no_move",
            "domain_epoch": index,
            "stratum": f"marginal:{arms[index]}",
            "selected_arm_key": arms[index],
            "selection_round": index + 1,
            "arm_levels": [
                0.25 if arm == arms[index] else 0.0
                for arm in PRODUCER.ARM_KEYS
            ],
            "rho": 0.0,
            "accepted_ack": {
                "decision": "accepted",
                "consumer_code_sha256": "a" * 64,
                "consumer_state_sha256": f"{index + 201:064x}",
                "consumer_checkpoint": {
                    "path": f"/fixture/model_{index}.pt",
                    "sha256": f"{index + 301:064x}",
                    "size_bytes": 1024,
                },
            },
            "windows": [
                {
                    "role": "frozen_canary",
                    "ledger": _ledger(320),
                    "raw_attempt_count": 320,
                    "raw_attempts_sha256": f"{index + 401:064x}",
                    "raw_nonfinite_count": 0,
                    "attempt_receipt_root_sha256": f"{index + 501:064x}",
                },
                {
                    "role": "frozen_heldout",
                    "ledger": _ledger(960),
                    "raw_attempt_count": 960,
                    "raw_attempts_sha256": f"{index + 601:064x}",
                    "raw_nonfinite_count": 0,
                    "attempt_receipt_root_sha256": f"{index + 701:064x}",
                },
            ],
        }
        for index, binding in enumerate(_bindings())
    ]


def _formal_records_with_raw_attempts():
    records = _formal_records()
    cursor = 1
    for record in records:
        record.update(
            {
                "policy_checkpoint_sha256": f"{cursor + 800:064x}",
                "policy_generation": cursor,
                "sampler_sha256": "b" * 64,
                "solver_sha256": "c" * 64,
                "policy_contract_sha256": "d" * 64,
            }
        )
        for window in record["windows"]:
            count = window["ledger"]["proposed"]
            role = window["role"]
            attempts = []
            samples = []
            births = []
            for offset in range(count):
                samples.append(f"{cursor:064x}")
                cursor += 1
                births.append(f"{cursor:064x}")
                cursor += 1
                attempts.append(
                    {
                        "sampling_stratum": "frontier",
                        "frontier_arm": record["selected_arm_key"],
                        "closed": True,
                        "terminal_signals": {
                            "infrastructure_invalid": False,
                            "joint_actual_limit": False,
                            "joint_qdes_limit": False,
                            "fall": False,
                            "table_hit": False,
                            "collision": False,
                            "legal_return": offset != 0,
                        },
                    }
                )
            window.update(
                {
                    "allocation": {
                        "role": role,
                        "proposal_count": count,
                        "seed_start": cursor,
                        "seed_end_exclusive": cursor + count,
                        "sample_start": cursor + count,
                        "sample_end_exclusive": cursor + 2 * count,
                        "birth_start": cursor + 2 * count,
                        "birth_end_exclusive": cursor + 3 * count,
                    },
                    "ordered_sample_receipt_sha256": samples,
                    "ordered_birth_receipt_sha256": births,
                    "raw_attempts": attempts,
                }
            )
            cursor += 3 * count
    return records


def _derive(records=None, *, interval_updates=10, max_iterations=50):
    return PRODUCER.derive_stage_metrics(
        stage="canary",
        max_iterations=max_iterations,
        interval_updates=interval_updates,
        action_bindings=_bindings(),
        formal_records=_formal_records() if records is None else records,
    )


def test_stage_metrics_are_derived_from_raw_ledgers():
    metrics, detailed, starvation = _derive()

    assert metrics["proposed_count"] == 5 * (320 + 960)
    assert metrics["solver_admitted_count"] == metrics["proposed_count"]
    assert metrics["attempt_count"] == metrics["proposed_count"]
    assert metrics["policy_return_failure_count"] == 10
    assert metrics["unsafe_count"] == 0
    assert metrics["table_hit_count"] == 0
    assert detailed["formal_request_count"] == 5
    assert len(detailed["per_action_axis_side"]) == 5
    assert starvation["sample_count"] == 5 * len(PRODUCER.NO_MOVE_ARMS)
    first = detailed["per_action_axis_side"][0]
    assert first["arm_physical_widths"]["time_to_contact_lower"]["width"] in (
        0.01,
        0.035,
    )


@pytest.mark.parametrize(
    ("records", "message"),
    (
        ([], "zero samples"),
        (_formal_records()[:-1], "omitted actions"),
    ),
)
def test_stage_metrics_refuse_zero_or_missing_action_coverage(records, message):
    with pytest.raises(PRODUCER.EvidenceError, match=message):
        _derive(records)


def test_smoke_metrics_also_refuse_zero_formal_samples():
    with pytest.raises(PRODUCER.EvidenceError, match="zero samples"):
        PRODUCER.derive_stage_metrics(
            stage="smoke",
            max_iterations=2,
            interval_updates=1,
            action_bindings=_bindings(),
            formal_records=[],
        )


def test_stage_metrics_refuse_missing_axis_side_identity():
    records = _formal_records()
    records[0]["selected_arm_key"] = "time_to_contact"

    with pytest.raises(PRODUCER.EvidenceError, match="missing or unknown"):
        _derive(records)


def test_center_domain_is_valid_and_retains_all_asymmetric_levels():
    records = [_formal_records()[0]]
    records[0].update(
        {
            "stratum": "center",
            "selected_arm_key": "",
            "arm_levels": [0.0] * len(PRODUCER.ARM_KEYS),
            "rho": 0.0,
        }
    )
    metrics, detailed, _starvation = PRODUCER.derive_stage_metrics(
        stage="smoke",
        max_iterations=2,
        interval_updates=1,
        action_bindings=_bindings(),
        formal_records=records,
    )

    assert metrics["attempt_count"] == 1280
    assert detailed["per_action_axis_side"][0]["axis"] == "center"
    assert detailed["per_action_axis_side"][0]["side"] == "not_applicable"


def test_no_move_domain_refuses_nonzero_base_travel_level():
    records = _formal_records()
    records[0]["arm_levels"][
        PRODUCER.ARM_KEYS.index("base_travel_x_upper")
    ] = 0.25

    with pytest.raises(PRODUCER.EvidenceError, match="base-travel level"):
        _derive(records)


def test_stage_metrics_refuse_interval_beyond_budget():
    with pytest.raises(PRODUCER.EvidenceError, match="interval exceeds"):
        _derive(interval_updates=51, max_iterations=50)


def test_stage_metrics_refuse_any_unsafe_outcome():
    records = _formal_records()
    ledger = records[0]["windows"][0]["ledger"]
    ledger["table_hit"] = 1
    ledger["legal_return"] -= 1

    with pytest.raises(PRODUCER.EvidenceError, match="unsafe/infrastructure"):
        _derive(records)


def _sealed_table_receipt(bindings):
    source_map = {
        name: "f" * 64
        for name in PRODUCER._ACTION_BALL_SOLVER_SOURCE_NAMES
    }
    # Solver profile v3: the payload seals a per-symbol semantic surface, and
    # the profile document publishes the same surface SHA.
    semantic_surface = {
        "kind": "whole_body_tracking.action_ball.solver_semantic_surface",
        "schema_version": 1,
        "sha256": "a" * 64,
    }
    solver_payload = {
        "kind": "fixture.solver",
        "semantic_surface": semantic_surface,
    }
    physics_payload = {"kind": "fixture.physics"}
    solver_sha = PRODUCER.canonical_sha256(solver_payload)
    physics_sha = PRODUCER.canonical_sha256(physics_payload)
    geometry = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": (
            "hope_training/whole_body_tracking/source/"
            "whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
            "racket_contact_geometry.py"
        ),
        "source_sha256": "f" * 64,
        "geometry_source_sha256": "e" * 64,
    }
    profile_document = {
        "solver_payload": solver_payload,
        "physics_payload": physics_payload,
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "solver_implementation_source_sha256": source_map,
        "solver_semantic_surface": {"sha256": semantic_surface["sha256"]},
    }
    manifest_document = {
        "solver_profile_sha256": solver_sha,
        "physics_profile_sha256": physics_sha,
        "racket_geometry_contract": geometry,
    }
    action_ids = [binding["motion_id"] for binding in bindings]
    action_uids = [binding["action_uid"] for binding in bindings]
    profile_id = "fixture_stage_n{}".format(len(bindings))
    # The stage fixture checks the future multi-action receipt schema, not the
    # current fixed-width N1 actor contract.  Temporarily name that future
    # contract without weakening production validation.
    original_actor_obs_contract = (
        TABLE_PRODUCER.action_set_contract.ACTOR_OBS_CONTRACT
    )
    TABLE_PRODUCER.action_set_contract.ACTOR_OBS_CONTRACT = (
        "fixture_content_derived_future_motion_intent_v1"
    )
    try:
        action_set_contract = TABLE_PRODUCER.action_set_contract.validate_contract(
            {
                "profile_id": profile_id,
                "expected_n": len(bindings),
                "scope": "upper",
                "mobility_mode": "no_move",
                "ordered_action_ids": action_ids,
                "ordered_action_uids": action_uids,
                "order_uid_digest_sha256": (
                    TABLE_PRODUCER.action_set_contract.order_uid_digest(
                        action_ids, action_uids
                    )
                ),
                "manifest_path": "fixture_manifest.json",
                "manifest_sha256": "a" * 64,
                "experiment_name": profile_id,
            },
            profile_id=profile_id,
            profile_policies={},
        )
    finally:
        TABLE_PRODUCER.action_set_contract.ACTOR_OBS_CONTRACT = (
            original_actor_obs_contract
        )
    document = {
        "schema_version": 4,
        "receipt_class": "isaac_action_ball_table_pose_obb_smoke_v4",
        "verdict": "PASS",
        "task_id": "HOPE-PingPong-ActionBall-AgibotA3-v0",
        "with_table": True,
        "scope": "upper",
        "mobility_mode": "no_move",
        "action_set_contract": action_set_contract,
        "manifest": {
            "path": "fixture_manifest.json",
            "sha256": "a" * 64,
        },
        "profile_contract": {
            "profile_pins": {
                "path": "fixture_profile.json",
                "sha256": "d" * 64,
            },
            "solver_profile_sha256": solver_sha,
            "physics_profile_sha256": physics_sha,
            "solver_implementation_sources": [
                {
                    "name": name,
                    "path": (
                        "hope_training/whole_body_tracking/source/"
                        "whole_body_tracking/whole_body_tracking/tasks/"
                        "tracking/mdp/{}".format(name)
                    ),
                    "sha256": source_map[name],
                }
                for name in sorted(
                    PRODUCER._ACTION_BALL_SOLVER_SOURCE_NAMES
                )
            ],
            "racket_geometry_contract": geometry,
        },
        "ordered_action_ids": action_ids,
        "motion_sha256": [binding["motion_sha256"] for binding in bindings],
        "runtime_contract": {
            "source_commit_sha": "b" * 40,
            "isaac_version": "4.5.0",
            "python_executable": "/opt/isaac/python.sh",
            "runtime_source": {
                "path": PRODUCER.TABLE_SMOKE_SOURCE,
                "sha256": "f" * 64,
            },
            "gpu_identity": {
                "physical_index": 1,
                "logical_index": 0,
                "cuda_visible_devices": "1",
                "gpu_uuid": "GPU-test",
                "gpu_name": "Fixture GPU",
                "driver_version": "fixture-driver",
                "nvml_verified": True,
            },
            "physics_steps": 100,
            "pose_obb_guard_pass": True,
            "full_action_ball_assembly": True,
            "all_five_table_components_with_pose_obb": True,
            "action_robot_body_contract_rows": 32 * len(bindings),
            "all_five_obstacles": True,
            "all_four_substeps": True,
            "positive_control_pass": True,
            "negative_control_pass": True,
            "zero_reset_leakage": True,
        },
        "actions": [
            {
                "motion_id": binding["motion_id"],
                "action_uid": binding["action_uid"],
                "scope": "upper",
                "robot_body_contract_count": 32,
                "motion_sha256": binding["motion_sha256"],
                "complete_cycle": True,
                "isaac_pose_obb_pass": True,
                "table_contact_count": 0,
                "fall_count": 0,
                "hard_limit_count": 0,
                "unsafe_count": 0,
                "verdict": "PASS",
            }
            for binding in bindings
        ],
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "training_authorization",
            "deployment_authorization",
            "hardware_authorization",
        ],
    }
    document["receipt_payload_sha256"] = PRODUCER.canonical_sha256(document)
    return (
        document,
        manifest_document,
        profile_document,
        action_set_contract,
    )


def _table_bindings(action_count):
    if action_count <= len(PRODUCER.ACTION_ORDER):
        action_ids = PRODUCER.ACTION_ORDER[:action_count]
    else:
        action_ids = tuple(
            "fixture_action_{:03d}".format(index)
            for index in range(action_count)
        )
    return [
        {
            "motion_id": action_id,
            "action_uid": index + 1,
            "motion_sha256": "{:064x}".format(index + 1),
            "scope": "upper",
            "mobility_mode": "no_move",
        }
        for index, action_id in enumerate(action_ids)
    ]


def _producer_generated_table_receipt(bindings):
    (
        _manual_document,
        manifest_document,
        profile_document,
        action_set_contract,
    ) = _sealed_table_receipt(bindings)
    geometry = manifest_document["racket_geometry_contract"]
    source_map = profile_document[
        "solver_implementation_source_sha256"
    ]

    def snapshot(repo_path, sha256):
        return TABLE_PRODUCER._FileSnapshot(
            path=Path("/fixture") / Path(repo_path).name,
            repo_path=repo_path,
            payload=b"",
            sha256=sha256,
            device=1,
            inode=1,
            size=0,
        )

    motions = tuple(
        TABLE_PRODUCER._MotionInput(
            motion_id=binding["motion_id"],
            action_uid=binding["action_uid"],
            family="backhand",
            strike_phase=0.5,
            mount_normal_sign=-1,
            reference_t_cycle_s=1.0,
            file=snapshot(
                "motions/{}.npz".format(binding["motion_id"]),
                binding["motion_sha256"],
            ),
        )
        for binding in bindings
    )
    inputs = TABLE_PRODUCER._FormalInputs(
        repo_root=Path("/fixture"),
        source=snapshot(PRODUCER.TABLE_SMOKE_SOURCE, "f" * 64),
        manifest=snapshot("fixture_manifest.json", "a" * 64),
        profile_pins=snapshot("fixture_profile.json", "d" * 64),
        solver_profile_sha256=profile_document[
            "solver_profile_sha256"
        ],
        physics_profile_sha256=profile_document[
            "physics_profile_sha256"
        ],
        solver_sources=tuple(
            (
                name,
                snapshot(
                    (
                        "hope_training/whole_body_tracking/source/"
                        "whole_body_tracking/whole_body_tracking/tasks/"
                        "tracking/mdp/{}".format(name)
                    ),
                    source_map[name],
                ),
            )
            for name in sorted(source_map)
        ),
        racket_geometry_contract=geometry,
        motions=motions,
        action_set_contract=action_set_contract,
    )
    runtime_origin = object()
    TABLE_PRODUCER._ISAAC_RUNTIME_ORIGIN = runtime_origin
    action_rows = tuple(
        TABLE_PRODUCER._RuntimeActionEvidence(
            motion_id=binding["motion_id"],
            action_uid=binding["action_uid"],
            motion_sha256=binding["motion_sha256"],
            frame_count=3,
            physics_steps=12,
            complete_cycle=True,
            table_contact_count=0,
            fall_count=0,
            hard_limit_count=0,
            unsafe_count=0,
        )
        for binding in bindings
    )
    evidence = TABLE_PRODUCER._RuntimeEvidence(
        origin=runtime_origin,
        source_commit_sha="b" * 40,
        isaac_version="isaaclab=fixture",
        python_executable=sys.executable,
        gpu_identity={
            "physical_index": 1,
            "logical_index": 0,
            "cuda_visible_devices": "1",
            "gpu_uuid": "GPU-fixture",
            "gpu_name": "Fixture GPU",
            "driver_version": "fixture-driver",
            "nvml_verified": True,
        },
        physics_steps=12 * len(action_rows),
        actions=action_rows,
        pose_obb_guard_pass=True,
        full_action_ball_assembly=True,
        all_five_table_components_with_pose_obb=True,
        all_five_obstacles=True,
        all_four_substeps=True,
        positive_control_pass=True,
        negative_control_pass=True,
        zero_reset_leakage=True,
    )
    return (
        TABLE_PRODUCER._build_formal_receipt(inputs, evidence),
        manifest_document,
        profile_document,
        action_set_contract,
    )


def _validate_table_fixture(
    document,
    bindings,
    manifest_document,
    profile_document,
    action_set_contract,
):
    PRODUCER._validate_table_rows(
        document,
        bindings,
        "a" * 64,
        manifest_relative="fixture_manifest.json",
        manifest_document=manifest_document,
        profile_relative="fixture_profile.json",
        profile_sha256="d" * 64,
        profile_document=profile_document,
        checkout=REPO_ROOT,
        source_commit="b" * 40,
        action_set_contract=action_set_contract,
    )


@pytest.mark.parametrize("action_count", (1, 5, 73))
def test_table_smoke_schema4_producer_roundtrips_into_stage_evidence(
    monkeypatch, action_count
):
    bindings = _table_bindings(action_count)
    (
        document,
        manifest_document,
        profile_document,
        action_set_contract,
    ) = _producer_generated_table_receipt(bindings)
    monkeypatch.setattr(
        PRODUCER,
        "_committed_file",
        lambda *_args, **_kwargs: {"sha256": "f" * 64},
    )

    TABLE_PRODUCER._validate_formal_receipt_document(document)
    assert document["schema_version"] == 4
    assert (
        document["receipt_class"]
        == "isaac_action_ball_table_pose_obb_smoke_v4"
    )
    _validate_table_fixture(
        document,
        bindings,
        manifest_document,
        profile_document,
        action_set_contract,
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "legacy_v3",
        "action_order",
        "action_set_contract",
        "mobility_mode",
        "action_uid",
        "robot_body_contract_count",
        "action_robot_body_contract_rows",
        "pose_obb_guard",
        "all_five_pose_obb",
        "legacy_filtered_contract",
        "pose_obb_action",
        "positive_control",
        "negative_control",
        "substep_latch",
        "reset_leakage",
    ),
)
def test_table_smoke_schema4_stage_rejects_identity_or_contact_control_tamper(
    monkeypatch, tamper
):
    bindings = _table_bindings(5)
    (
        document,
        manifest_document,
        profile_document,
        action_set_contract,
    ) = _producer_generated_table_receipt(bindings)
    monkeypatch.setattr(
        PRODUCER,
        "_committed_file",
        lambda *_args, **_kwargs: {"sha256": "f" * 64},
    )
    if tamper == "legacy_v3":
        document["schema_version"] = 3
        document["receipt_class"] = (
            "isaac_action_ball_table_filtered_smoke_v3"
        )
    elif tamper == "action_order":
        document["ordered_action_ids"] = list(
            reversed(document["ordered_action_ids"])
        )
    elif tamper == "action_set_contract":
        document["action_set_contract"]["expected_n"] += 1
    elif tamper == "mobility_mode":
        document["mobility_mode"] = "move"
    elif tamper == "action_uid":
        document["actions"][0]["action_uid"] += 1
    elif tamper == "robot_body_contract_count":
        document["actions"][0]["robot_body_contract_count"] = 31
    elif tamper == "action_robot_body_contract_rows":
        document["runtime_contract"]["action_robot_body_contract_rows"] -= 1
    elif tamper == "pose_obb_guard":
        document["runtime_contract"][
            "pose_obb_guard_pass"
        ] = False
    elif tamper == "all_five_pose_obb":
        document["runtime_contract"][
            "all_five_table_components_with_pose_obb"
        ] = False
    elif tamper == "legacy_filtered_contract":
        document["runtime_contract"][
            "all_five_table_sources_with_explicit_robot_body_filters"
        ] = document["runtime_contract"].pop(
            "all_five_table_components_with_pose_obb"
        )
    elif tamper == "pose_obb_action":
        document["actions"][0]["isaac_pose_obb_pass"] = False
    elif tamper == "positive_control":
        document["runtime_contract"]["positive_control_pass"] = False
    elif tamper == "negative_control":
        document["runtime_contract"]["negative_control_pass"] = False
    elif tamper == "substep_latch":
        document["runtime_contract"]["all_four_substeps"] = False
    else:
        document["runtime_contract"]["zero_reset_leakage"] = False
    document["receipt_payload_sha256"] = PRODUCER.canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "receipt_payload_sha256"
        }
    )

    with pytest.raises(PRODUCER.EvidenceError):
        _validate_table_fixture(
            document,
            bindings,
            manifest_document,
            profile_document,
            action_set_contract,
        )


def test_claimed_pass_cannot_hide_raw_table_contact(monkeypatch):
    bindings = _bindings()
    (
        document,
        manifest_document,
        profile_document,
        action_set_contract,
    ) = (
        _sealed_table_receipt(bindings)
    )
    monkeypatch.setattr(
        PRODUCER,
        "_committed_file",
        lambda *_args, **_kwargs: {"sha256": "f" * 64},
    )
    document["actions"][0]["table_contact_count"] = 1
    document["receipt_payload_sha256"] = PRODUCER.canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "receipt_payload_sha256"
        }
    )

    with pytest.raises(PRODUCER.EvidenceError, match="partial/unsafe"):
        PRODUCER._validate_table_rows(
            document,
            bindings,
            "a" * 64,
            manifest_relative="fixture_manifest.json",
            manifest_document=manifest_document,
            profile_relative="fixture_profile.json",
            profile_sha256="d" * 64,
            profile_document=profile_document,
                checkout=REPO_ROOT,
                source_commit=document["runtime_contract"]["source_commit_sha"],
                action_set_contract=action_set_contract,
            )


class _FiniteCount:
    def __init__(self, value):
        self._value = value

    def sum(self):
        return self

    def item(self):
        return self._value


class _FakeTensor:
    def __init__(self, values):
        self.values = tuple(values)

    def is_floating_point(self):
        return True

    def is_complex(self):
        return False

    def numel(self):
        return len(self.values)


class _FakeTorch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        return _FiniteCount(sum(math.isfinite(item) for item in value.values))


def _checkpoint(*, tensor_values=(1.0, 2.0)):
    return {
        "iter": 7,
        "model_state_dict": {"weight": _FakeTensor(tensor_values)},
        "optimizer_state_dict": {"state": {"moment": _FakeTensor((0.1,))}},
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": "c" * 64,
            "training_contract_lineage_exact": 1,
            "training_launch_claim_sha256": "d" * 64,
            "hope_exact_resume_state": {
                "schema_version": 3,
                "next_learning_iteration": 8,
                "tot_timesteps": 1024,
                "tot_time": 1.5,
                "algorithm_learning_rate": 3.0e-4,
                "python_random_state": ("state",),
                "numpy_random_state": {
                    "schema_version": 1,
                    "bit_generator": "MT19937",
                    "state_uint32": list(range(624)),
                    "position": 17,
                    "has_gauss": 0,
                    "cached_gaussian": 0.0,
                },
                "torch_random_state": _FakeTensor((1.0,)),
                "torch_cuda_random_states": [],
                "torch_cuda_device_count": 0,
                "environment_resume_state": {
                    "schema_version": 3,
                    "common_step_counter": 32,
                    "active_term_names": ["racket_target", "motion"],
                    "command_terms": {
                        "racket_target": {
                            "capture_mode": "explicit",
                            "term_type": "test.RacketTarget",
                            "exact_state": {"epoch": 1},
                        },
                        "motion": {
                            "capture_mode": "explicit",
                            "term_type": "test.Motion",
                            "exact_state": {"action_uid": 1},
                        },
                    },
                },
            },
        },
    }


def _audit_checkpoint(checkpoint):
    return PRODUCER.audit_checkpoint_object(
        checkpoint=checkpoint,
        checkpoint_path=Path("/tmp/model_7.pt"),
        checkpoint_sha256="e" * 64,
        training_contract_sha256="c" * 64,
        launch_claim_sha256="d" * 64,
        torch_module=_FakeTorch,
    )


def _checkpoint_with_curriculum():
    checkpoint = _checkpoint()
    records = _formal_records_with_raw_attempts()
    arm_catalog = {
        "schema_version": 3,
        "arm_keys": list(PRODUCER.ARM_KEYS),
    }
    arm_catalog_sha = PRODUCER.canonical_sha256(arm_catalog)
    scheduler_sha = "e" * 64
    profiles = []
    progress = []
    receipt_seq = 1
    for binding, record in zip(_bindings(), records):
        profile = {
            "action_uid": binding["action_uid"],
            "profile_sha256": record["profile_sha256"],
            "mobility": "no_move",
        }
        profiles.append(profile)
        formal_receipts = []
        for window in record["windows"]:
            match = PRODUCER._formal_window_match_document(
                record=record,
                window=window,
                arm_catalog_sha256=arm_catalog_sha,
                scheduler_contract_sha256=scheduler_sha,
            )
            evidence = {
                "schema_version": 4,
                **match,
                "seq": receipt_seq,
                "window_id": f"{receipt_seq + 5000:064x}",
            }
            receipt_seq += 1
            formal_receipts.append(
                {
                    "evidence": evidence,
                    "window_sha256": PRODUCER.canonical_sha256(evidence),
                    "certified": window["role"] == "frozen_heldout",
                }
            )
        selected = record["selected_arm_key"]
        selection_round = record["selection_round"]
        last_selected = [0] * len(PRODUCER.ARM_KEYS)
        last_selected[PRODUCER.ARM_KEYS.index(selected)] = selection_round
        status = [
            "disabled" if arm not in PRODUCER.NO_MOVE_ARMS else "probing"
            for arm in PRODUCER.ARM_KEYS
        ]
        progress.append(
            {
                "key": profile,
                "phase": "marginal",
                "arm_frontier_indices": [0] * len(PRODUCER.ARM_KEYS),
                "arm_status": status,
                "arm_probe_indices": [0] * len(PRODUCER.ARM_KEYS),
                "arm_epochs": [0] * len(PRODUCER.ARM_KEYS),
                "selected_arm_key": selected,
                "selection_round": selection_round,
                "last_selected_round": last_selected,
                "center_epoch": 0,
                "joint_epoch": 0,
                "joint_probe_index": 0,
                "joint_rho_index": 0,
                "center_failures": 0,
                "domain_release_epoch": 1,
                "pending_canary_window_sha256": None,
                "pending_release": None,
                "release_receipts": [],
                "formal_receipts": formal_receipts,
                "scheduler_receipts": [],
                "event_hash_chain_sha256": "f" * 64,
                "last_certified": None,
            }
        )
    curriculum = {
        "schema_version": 10,
        "contract_sha256": "1" * 64,
        "profile_order": profiles,
        "arm_catalog": arm_catalog,
        "arm_catalog_sha256": arm_catalog_sha,
        "scheduler_config": {},
        "scheduler_contract_sha256": scheduler_sha,
        "sampler_sha256": "b" * 64,
        "solver_sha256": "c" * 64,
        "policy_contract_sha256": "d" * 64,
        "config": {},
        "evaluator_authority_contract_sha256": "2" * 64,
        "evaluator_authority_state_owner_sha256": "3" * 64,
        "evaluator_authority_state": {},
        "evaluator_authority_state_sha256": PRODUCER.canonical_sha256({}),
        "drain_reset_authority_contract_sha256": "4" * 64,
        "drain_reset_launch_receipt_sha256": "5" * 64,
        "drain_reset_authority_state_owner_sha256": "6" * 64,
        "drain_reset_authority_state": {},
        "drain_reset_authority_state_sha256": PRODUCER.canonical_sha256({}),
        "next_barrier_serial": 1,
        "issued_global_pre_reset_barriers": [],
        "progress": progress,
    }
    curriculum["state_sha256"] = PRODUCER.canonical_sha256(curriculum)
    racket_state = {
        "schema_version": 6,
        "kind": "whole_body_tracking.RacketTargetCommand.action_ball",
        "manifest_sha256": "a" * 64,
        "hard_contract": {},
        "action_order": list(PRODUCER.ACTION_ORDER),
        "action_uids": [row["action_uid"] for row in _bindings()],
        "num_envs": 5,
        "solver": {},
        "physics": {},
        "curriculum": curriculum,
        "frozen_evaluation": {},
        "mutable_state": {},
        "broker": {},
        "pool": {},
        "ledger": {
            name: [0] * len(PRODUCER.ACTION_ORDER)
            for name in PRODUCER.ACTION_BALL_LEDGER_NAMES
        },
        "env_state": {},
        "last_rollout_step": 0,
    }
    racket_state["integrity_sha256"] = PRODUCER.canonical_sha256(racket_state)
    checkpoint["infos"]["hope_exact_resume_state"]["environment_resume_state"][
        "command_terms"
    ]["racket_target"]["exact_state"] = racket_state
    return checkpoint, records


def test_checkpoint_audit_requires_finite_exact_resume_state():
    audit = _audit_checkpoint(_checkpoint())

    assert audit["finite"] is True
    assert audit["exact_resume_structure_passed"] is True
    assert "exact_resume_passed" not in audit
    assert audit["nonfinite_floating_elements"] == 0
    assert audit["explicit_command_terms"] == ["racket_target", "motion"]
    assert audit["environment_resume_schema_version"] == 3
    assert audit["active_action_terms"] == []
    assert audit["explicit_action_terms"] == []


def test_checkpoint_audit_accepts_schema4_explicit_control_step_delay():
    checkpoint = _checkpoint()
    environment = checkpoint["infos"]["hope_exact_resume_state"][
        "environment_resume_state"
    ]
    environment.update(
        {
            "schema_version": 4,
            "active_action_term_names": ["joint_pos"],
            "action_terms": {
                "joint_pos": {
                    "capture_mode": "explicit_delay",
                    "term_type": "test.ClampedJointPositionAction",
                    "exact_state": {
                        "schema_version": 1,
                        "kind": (
                            "whole_body_tracking."
                            "policy_control_step_action_delay"
                        ),
                        "contract": {
                            "enabled": True,
                            "semantic_unit": "policy_control_step",
                            "sample_timing": "once_per_episode_reset",
                        },
                        "num_envs": 4,
                        "action_dim": 31,
                        "lag_steps": _FakeTensor((0.0, 1.0, 2.0, 0.0)),
                        "episode_initialized": _FakeTensor(
                            (1.0, 1.0, 1.0, 1.0)
                        ),
                        "history": _FakeTensor((0.0,) * 8),
                    },
                }
            },
        }
    )
    audit = _audit_checkpoint(checkpoint)
    assert audit["environment_resume_schema_version"] == 4
    assert audit["active_action_terms"] == ["joint_pos"]
    assert audit["explicit_action_terms"] == ["joint_pos"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda checkpoint: checkpoint["infos"]["hope_exact_resume_state"].update(
                {"next_learning_iteration": 7}
            ),
            "schema/iteration",
        ),
        (
            lambda checkpoint: checkpoint.update(
                {"optimizer_state_dict": {}}
            ),
            "optimizer state",
        ),
        (
            lambda checkpoint: checkpoint["infos"][
                "hope_exact_resume_state"
            ].update({"numpy_random_state": ("MT19937",)}),
            "NumPy RNG state",
        ),
    ),
)
def test_checkpoint_audit_refuses_structurally_bad_checkpoint(mutation, message):
    checkpoint = _checkpoint()
    mutation(checkpoint)

    with pytest.raises(PRODUCER.EvidenceError, match=message):
        _audit_checkpoint(checkpoint)


def test_checkpoint_audit_refuses_nonfinite_weights():
    with pytest.raises(PRODUCER.EvidenceError, match="non-finite"):
        _audit_checkpoint(_checkpoint(tensor_values=(1.0, float("nan"))))


def test_checkpoint_loader_decodes_the_pinned_snapshot_not_a_raced_path(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "model_7.pt"
    checkpoint_path.write_bytes(b"snapshot-A")
    loaded_checkpoint = {"decoded_from": "snapshot-A"}
    seen = {}

    class FakeTorch:
        @staticmethod
        def load(stream, *, map_location, weights_only):
            seen["is_bytes_io"] = type(stream).__name__ == "BytesIO"
            seen["raw"] = stream.read()
            seen["map_location"] = map_location
            seen["weights_only"] = weights_only
            checkpoint_path.write_bytes(b"raced-B")
            checkpoint_path.write_bytes(b"snapshot-A")
            return loaded_checkpoint

    def fake_audit(**kwargs):
        assert kwargs["checkpoint"] is loaded_checkpoint
        assert kwargs["checkpoint_sha256"] == hashlib.sha256(
            b"snapshot-A"
        ).hexdigest()
        return {"finite": True}

    monkeypatch.setattr(PRODUCER, "audit_checkpoint_object", fake_audit)
    audit, checkpoint = PRODUCER._load_checkpoint(
        checkpoint_path,
        training_contract_sha256="a" * 64,
        launch_claim_sha256="b" * 64,
        torch_module=FakeTorch(),
    )
    assert seen == {
        "is_bytes_io": True,
        "raw": b"snapshot-A",
        "map_location": "cpu",
        "weights_only": True,
    }
    assert audit == {"finite": True}
    assert checkpoint is loaded_checkpoint


def test_checkpoint_curriculum_drives_n5_x_28_starvation_and_raw_match():
    checkpoint, records = _checkpoint_with_curriculum()

    evidence = PRODUCER._derive_checkpoint_curriculum_evidence(
        checkpoint=checkpoint,
        action_bindings=_bindings(),
        formal_records=records,
        manifest_sha256="a" * 64,
    )

    assert evidence["domain_epoch_stale_count"] == 0
    assert evidence["formal_receipt_count"] == 10
    assert evidence["starvation"]["sample_count"] == 5 * 28
    assert evidence["starvation"]["coverage_count"] == 5
    assert evidence["starvation"]["uncovered_count"] == 5 * 27


def test_checkpoint_curriculum_refuses_unbacked_last_selected_round():
    checkpoint, records = _checkpoint_with_curriculum()
    racket = PRODUCER._action_ball_racket_state(checkpoint)
    progress = racket["curriculum"]["progress"][0]
    progress["last_selected_round"][
        PRODUCER.ARM_KEYS.index("contact_x_lower")
    ] = progress["selection_round"]
    unsigned_curriculum = dict(racket["curriculum"])
    unsigned_curriculum.pop("state_sha256")
    racket["curriculum"]["state_sha256"] = PRODUCER.canonical_sha256(
        unsigned_curriculum
    )
    unsigned_racket = dict(racket)
    unsigned_racket.pop("integrity_sha256")
    racket["integrity_sha256"] = PRODUCER.canonical_sha256(unsigned_racket)

    with pytest.raises(PRODUCER.EvidenceError, match="not backed"):
        PRODUCER._derive_checkpoint_curriculum_evidence(
            checkpoint=checkpoint,
            action_bindings=_bindings(),
            formal_records=records,
            manifest_sha256="a" * 64,
        )


def test_trainer_ledger_must_match_checkpoint_exact_state(tmp_path):
    checkpoint, _records = _checkpoint_with_curriculum()
    racket = PRODUCER._action_ball_racket_state(checkpoint)
    action_uids = [row["action_uid"] for row in _bindings()]
    ledger = {
        action: {
            "P": 2,
            "A": 2,
            "I": 2,
            "S": 2,
            "C": 2,
            "L": 1,
            "F": 1,
            "U_table": 0,
            "U_fall": 0,
            "U_collision": 0,
            "U_joint_qdes": 0,
            "U_joint_actual": 0,
            "X": 0,
        }
        for action in PRODUCER.ACTION_ORDER
    }
    for slot, action in enumerate(PRODUCER.ACTION_ORDER):
        for name in PRODUCER.ACTION_BALL_LEDGER_NAMES:
            racket["ledger"][name][slot] = ledger[action][name]
    racket["last_rollout_step"] = 3
    unsigned_racket = dict(racket)
    unsigned_racket.pop("integrity_sha256")
    racket["integrity_sha256"] = PRODUCER.canonical_sha256(unsigned_racket)
    event = {
        "event": "action_ball_training_ledger",
        "schema_version": 1,
        "step": 3,
        "manifest_sha256": "a" * 64,
        "status": "report_only_requires_frozen_checkpoint_evidence",
        "action_order": list(PRODUCER.ACTION_ORDER),
        "ledger": ledger,
        "solver_rejections": {str(uid): {} for uid in action_uids},
        "pool": {
            str(uid): {
                "requests": 2,
                "refill_calls": 1,
                "proposed": 2,
                "admitted": 2,
                "issued": 2,
                "discarded": 0,
                "pending": 0,
            }
            for uid in action_uids
        },
        "curriculum": {},
    }
    log_path = tmp_path / "train.log"
    log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = PRODUCER._trainer_ledger_evidence(
        log_path=log_path,
        checkpoint=checkpoint,
        action_bindings=_bindings(),
        manifest_sha256="a" * 64,
    )

    assert result["event_count"] == 1
    assert result["counter_invariants_passed"] is True


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ("plain trainer output\n", "contains no"),
        (
            '{"event":"action_ball_training_ledger","value":NaN}\n',
            "malformed/non-finite",
        ),
    ),
)
def test_trainer_ledger_refuses_absent_or_nonfinite_raw_event(
    tmp_path, payload, message
):
    checkpoint, _records = _checkpoint_with_curriculum()
    log_path = tmp_path / "train.log"
    log_path.write_text(payload, encoding="utf-8")

    with pytest.raises(PRODUCER.EvidenceError, match=message):
        PRODUCER._trainer_ledger_evidence(
            log_path=log_path,
            checkpoint=checkpoint,
            action_bindings=_bindings(),
            manifest_sha256="a" * 64,
        )


def test_schema10_scheduler_retains_overlapping_raw_safety_signals():
    def attempt(index, *, terminal, signals):
        return {
            "sample_receipt_sha256": f"{index + 1:064x}",
            "birth_receipt_sha256": f"{index + 11:064x}",
            "solver_admitted": True,
            "installed": True,
            "started": True,
            "closed": True,
            "terminal_outcome": terminal,
            "infrastructure_invalid": False,
            "in_new_band": False,
            "terminal_signals": {
                "infrastructure_invalid": False,
                "joint_actual_limit": signals.get(
                    "joint_actual_limit", False
                ),
                "joint_qdes_limit": signals.get(
                    "joint_qdes_limit", False
                ),
                "fall": signals.get("fall", False),
                "table_hit": signals.get("table_hit", False),
                "collision": signals.get("collision", False),
                "legal_return": signals.get("legal_return", False),
            },
        }

    ledger = PRODUCER._scheduler_attempt_ledger(
        [
            attempt(
                0,
                terminal="joint_actual_limit",
                signals={
                    "joint_actual_limit": True,
                    "table_hit": True,
                },
            ),
            attempt(
                1,
                terminal="legal_return",
                signals={"legal_return": True},
            ),
        ],
        label="fixture scheduler",
    )

    assert ledger["C"] == 2
    assert ledger["L"] == 1
    assert ledger["U_joint_actual"] == 1
    assert ledger["U_table"] == 1


def _exact_resume_evidence_fixture(tmp_path, monkeypatch):
    namespace = tmp_path / "namespace"
    namespace.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    rsl = tmp_path / "rsl"
    rsl.mkdir()
    checkpoint = rsl / "model_2.pt"
    checkpoint.write_bytes(b"source-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    claim_sha = "a" * 64
    verifier_sha = "b" * 64
    factory_sha = "c" * 64
    roundtrip_dir = rsl / (
        "exact_resume_roundtrip_" + claim_sha[:16]
    )
    roundtrip_dir.mkdir()
    roundtrip = roundtrip_dir / checkpoint.name
    roundtrip.write_bytes(b"roundtrip-checkpoint")
    roundtrip_sha = hashlib.sha256(roundtrip.read_bytes()).hexdigest()
    restore = {
        "factory_call_count": 2,
        "closed_runtime_count": 2,
        "load_optimizer": True,
        "fresh_strict_load_token_consumed": True,
        "roundtrip_save_api": "save_exact_resume_roundtrip",
        "roundtrip_save_receipt_sha256": "d" * 64,
        "source_construction_receipt_sha256": "e" * 64,
        "roundtrip_construction_receipt_sha256": "f" * 64,
        "runtime_inventory_live_verification_sha256": "0" * 64,
        "source_live_state_receipt_sha256": "1" * 64,
        "roundtrip_live_state_receipt_sha256": "1" * 64,
        "live_core_sha256": "2" * 64,
        "common_step_counter": 48,
        "common_step_counter_delta": 0,
    }
    state = {
        "source_core_sha256": "1" * 64,
        "roundtrip_core_sha256": "1" * 64,
        "source_exact_resume_sha256": "2" * 64,
        "roundtrip_exact_resume_sha256": "2" * 64,
        "model_state_sha256": "3" * 64,
        "optimizer_state_sha256": "4" * 64,
        "normalizer_state_sha256": "5" * 64,
    }
    receipt = {
        "schema_version": 1,
        "kind": "action_ball_exact_resume_verification_v1",
        "status": "passed",
        "source_commit_sha": "9" * 40,
        "launch_claim_sha256": claim_sha,
        "stage": "smoke",
        "namespace": str(namespace),
        "verifier": {
            "source_path": PRODUCER.EXACT_RESUME_VERIFIER_SOURCE,
            "source_sha256": verifier_sha,
            "runtime_factory_source_path": (
                PRODUCER.EXACT_RESUME_FACTORY_SOURCE
            ),
            "runtime_factory_source_sha256": factory_sha,
        },
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint.stat().st_size,
            "embedded_iteration": 2,
        },
        "roundtrip_checkpoint": {
            "path": str(roundtrip),
            "sha256": roundtrip_sha,
            "size_bytes": roundtrip.stat().st_size,
            "embedded_iteration": 2,
        },
        "runtime_bootstrap": {
            "content_sha256": "6" * 64,
            "lineage_payload_sha256": "7" * 64,
        },
        "restore": restore,
        "state": state,
        "natural_exit": True,
    }
    receipt["receipt_payload_sha256"] = PRODUCER.canonical_sha256(
        receipt
    )
    _write_canonical_json(
        namespace / "exact_resume_verification.json", receipt
    )

    def committed_file(checkout, commit, relative, label):
        digest = (
            verifier_sha
            if relative == PRODUCER.EXACT_RESUME_VERIFIER_SOURCE
            else factory_sha
        )
        return {
            "path": tmp_path / relative,
            "sha256": digest,
            "size_bytes": 1,
        }

    monkeypatch.setattr(PRODUCER, "_committed_file", committed_file)
    return {
        "namespace": namespace,
        "checkout": checkout,
        "checkpoint": checkpoint,
        "checkpoint_audit": {
            "sha256": checkpoint_sha,
            "embedded_iteration": 2,
        },
        "runtime_bootstrap": {
            "content_sha256": "6" * 64,
            "lineage_payload_sha256": "7" * 64,
        },
        "claim_sha": claim_sha,
        "claim_payload": {
            "runtime_code_sha256": {
                PRODUCER.EXACT_RESUME_VERIFIER_SOURCE: verifier_sha,
                PRODUCER.EXACT_RESUME_FACTORY_SOURCE: factory_sha,
            }
        },
    }


def test_exact_resume_pass_requires_real_roundtrip_receipt(
    tmp_path, monkeypatch
):
    fixture = _exact_resume_evidence_fixture(tmp_path, monkeypatch)
    result = PRODUCER._exact_resume_verification_evidence(
        namespace=fixture["namespace"],
        checkpoint_path=fixture["checkpoint"],
        checkpoint_audit=fixture["checkpoint_audit"],
        runtime_bootstrap=fixture["runtime_bootstrap"],
        checkout=fixture["checkout"],
        source_commit="9" * 40,
        claim_sha256=fixture["claim_sha"],
        stage="smoke",
        claim_payload=fixture["claim_payload"],
    )
    assert result["exact_resume_passed"] is True
    assert result["restore"]["common_step_counter_delta"] == 0
    assert (
        result["state"]["source_core_sha256"]
        == result["state"]["roundtrip_core_sha256"]
    )


def test_exact_resume_refuses_state_drift(tmp_path, monkeypatch):
    fixture = _exact_resume_evidence_fixture(tmp_path, monkeypatch)
    path = fixture["namespace"] / "exact_resume_verification.json"
    receipt = json.loads(path.read_text(encoding="ascii"))
    receipt["state"]["roundtrip_core_sha256"] = "8" * 64
    receipt["receipt_payload_sha256"] = PRODUCER.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_payload_sha256"
        }
    )
    _write_canonical_json(path, receipt)

    with pytest.raises(PRODUCER.EvidenceError, match="core"):
        PRODUCER._exact_resume_verification_evidence(
            namespace=fixture["namespace"],
            checkpoint_path=fixture["checkpoint"],
            checkpoint_audit=fixture["checkpoint_audit"],
            runtime_bootstrap=fixture["runtime_bootstrap"],
            checkout=fixture["checkout"],
            source_commit="9" * 40,
            claim_sha256=fixture["claim_sha"],
            stage="smoke",
            claim_payload=fixture["claim_payload"],
        )
