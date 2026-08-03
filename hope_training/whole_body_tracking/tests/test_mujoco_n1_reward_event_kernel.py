"""Unit contracts for the pure MuJoCo N1 reward/event kernel."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import n1_reward_event_kernel as kernel  # noqa: E402
from mujoco_native import observed_outcome_resolver as outcome_resolver  # noqa: E402
from mujoco_native import physical_ball_scene  # noqa: E402
from mujoco_native import selected_rubber_classifier as classifier  # noqa: E402


SOURCE = kernel.SourceBinding(
    source_id="unit-test-native-contact-ledger",
    source_sha256="a" * 64,
    event_contract_sha256="b" * 64,
)
PHYSICAL_CONTRACT = kernel.native_physical_event_facts_contract()
PHYSICAL_SOURCE = kernel.SourceBinding(
    source_id="mujoco_native/n1_ball_core.py",
    source_sha256="c" * 64,
    event_contract_sha256=PHYSICAL_CONTRACT["content_sha256"],
)


def _physical_sample(**changes):
    values = {
        "schema_version": 4,
        "kind": kernel.NATIVE_PHYSICAL_EVENT_FACTS_KIND,
        "source": {
            "source_id": PHYSICAL_SOURCE.source_id,
            "source_sha256": PHYSICAL_SOURCE.source_sha256,
            "event_contract_sha256": PHYSICAL_SOURCE.event_contract_sha256,
        },
        "policy_tick": 5,
        "racket_contact_edge_count_total": 1,
        "first_racket_contact_stamp": {
            "policy_tick": 4,
            "physics_substep": 3,
        },
        "outgoing_flight": {
            "policy_tick": 5,
            "physics_substep": 0,
            "time_s": 0.1,
            "position_w_m": [1.0, -0.665, 1.0],
            "linear_velocity_w_mps": [2.0, 0.0, 0.3],
            "spin_w_radps": [0.0, 1.0, 0.0],
            "semantic": (
                "first_contact_free_physics_substep_after_first_racket_contact"
            ),
        },
        "invalid_reasons": [],
        "selected_rubber_authority_available": False,
        "selected_rubber_action_lineage": None,
        "first_racket_contact_classification": None,
        "observed_outcome_authority_available": False,
        "observed_outcome_resolver_binding": None,
        "observed_outcome_question_binding": None,
        "observed_outcome_snapshot": None,
    }
    values.update(changes)
    return values


def _seal(values):
    payload = dict(values)
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _selected_physical_sample():
    lineage = _seal(
        {
            "schema_version": 1,
            "kind": classifier.ACTION_LINEAGE_KIND,
            "action_id": "take_061_unit04_bh",
            "action_uid": 7,
            "mount_normal_sign": 1,
            "motion_sha256": "1" * 64,
            "physics_sha256": "2" * 64,
            "geometry_source_sha256": "3" * 64,
            "action_manifest_repo_relative_path": "configs/action.json",
            "action_manifest_sha256": "4" * 64,
            "scene_binding_sha256": "5" * 64,
            "mujoco_backend_version": "unit-test",
            "classifier_binding_sha256": "6" * 64,
        }
    )
    classification = _seal(
        {
            "schema_version": 1,
            "kind": classifier.CLASSIFICATION_KIND,
            "policy_tick": 4,
            "physics_substep": 3,
            "status": classifier.STATUS_SELECTED,
            "ambiguity_reason": None,
            "observed_face_sign": 1,
            "selected_rubber": True,
            "ball_center_local_m": [0.0, 0.02, 0.0],
            "tangential_distance_from_face_center_m": 0.0,
            "safe_ball_center_tangential_radius_m": 0.04,
            "action_lineage_sha256": lineage["content_sha256"],
            "classifier_binding_sha256": lineage[
                "classifier_binding_sha256"
            ],
        }
    )
    return _physical_sample(
        selected_rubber_authority_available=True,
        selected_rubber_action_lineage=lineage,
        first_racket_contact_classification=classification,
    )


def _selected_resolved_physical_sample():
    selected = _selected_physical_sample()
    lineage = selected["selected_rubber_action_lineage"]
    table_scene = physical_ball_scene._load_table_scene_module()
    rows = table_scene.action_ball_policy_obstacle_geometry()
    geometry = table_scene.action_ball_policy_geometry_contract(rows)
    source_obstacle_rows = (
        rows["table_top"],
        rows["robot_keepout"],
        rows["net"],
        *rows["net_posts"],
    )
    obstacle_ids = {
        row["name"]: geom_id
        for geom_id, row in enumerate(source_obstacle_rows, start=10)
    }
    compiled_obstacles = {
        row["name"]: {
            "name": row["name"],
            "geom_id": geom_id,
            "body_id": 0,
            "primitive": "axis_aligned_box_full_extents_m",
            "center_mjcf_world_m": list(row["center_mjcf_world_m"]),
            "full_extents_m": list(row["full_extents_m"]),
        }
        for row, geom_id in zip(source_obstacle_rows, obstacle_ids.values())
    }
    scene = {
        "kind": "a3_mujoco_physical_ball_scene_binding_v1",
        "binding_sha256": lineage["scene_binding_sha256"],
        "assembled_xml_sha256": "8" * 64,
        "canonical_mjcf_sha256": "9" * 64,
        "table_geometry_contract_sha256": geometry["sha256"],
        "ball_contract_source": {"sha256": "a" * 64},
        "ball": {"radius_m": 0.02},
        "with_ball": True,
        "strict_pair_filter": True,
        "compiled_runtime": {
            "mujoco_version": lineage["mujoco_backend_version"],
            "model_timestep_s": 0.005,
            "ball_radius_m": 0.02,
            "obstacle_geom_ids": obstacle_ids,
            "obstacle_geometry": compiled_obstacles,
        },
    }
    binding = outcome_resolver.build_resolver_binding(
        scene_binding=scene,
        obstacle_rows=rows,
        plant_binding_sha256="7" * 64,
        policy_step_dt_s=0.02,
        control_decimation=4,
    )
    question = outcome_resolver.bind_question(
        resolver_binding=binding,
        question_source_sha256="b" * 64,
        landing_aim_xy_w_m=(2.3, -0.665),
        action_lineage_sha256=lineage["content_sha256"],
    )
    outgoing = selected["outgoing_flight"]
    snapshot = outcome_resolver.replay_trace(
        resolver_binding=binding,
        question_binding=question,
        expected_scene_binding=scene,
        expected_obstacle_rows=rows,
        expected_plant_binding_sha256="7" * 64,
        expected_policy_step_dt_s=0.02,
        expected_control_decimation=4,
        expected_resolver_source_sha256=(
            kernel.EXPECTED_OBSERVED_OUTCOME_RESOLVER_SOURCE_SHA256
        ),
        expected_question_source_sha256=question["question_source_sha256"],
        expected_landing_aim_xy_w_m=question["landing_aim_xy_w_m"],
        expected_action_lineage_sha256=question["action_lineage_sha256"],
        outgoing_flight={
            "policy_tick": outgoing["policy_tick"],
            "physics_substep": outgoing["physics_substep"],
            "time_s": outgoing["time_s"],
            "position_w_m": outgoing["position_w_m"],
        },
        samples=[
            {
                "policy_tick": 5,
                "physics_substep": 1,
                "time_s": 0.105,
                "ball_center_w_m": [2.0, -0.665, 1.1],
                "active_contact_labels": [],
            },
            {
                "policy_tick": 5,
                "physics_substep": 2,
                "time_s": 0.110,
                "ball_center_w_m": [2.3, -0.665, 0.78],
                "active_contact_labels": ["table"],
            },
            {
                "policy_tick": 5,
                "physics_substep": 3,
                "time_s": 0.115,
                "ball_center_w_m": [2.31, -0.665, 0.80],
                "active_contact_labels": [],
            },
        ],
    )
    return _physical_sample(
        **{
            **selected,
            "policy_tick": 6,
            "observed_outcome_authority_available": True,
            "observed_outcome_resolver_binding": binding,
            "observed_outcome_question_binding": question,
            "observed_outcome_snapshot": snapshot,
        }
    )


def _sample(**changes):
    values = {
        "source": SOURCE,
        "motion_mimic_eligible": True,
        "target_valid": True,
        "strike_window": True,
        "actual_contact": kernel.ContactEvidence(
            occurred=True,
            stamp=kernel.EventStamp(5, 1),
            selected_rubber=True,
        ),
        "outgoing_flight": kernel.OutgoingFlightEvidence(
            valid=True,
            stamp=kernel.EventStamp(5, 2),
            position_w_m=(1.0, 0.0, 1.0),
            linear_velocity_w_mps=(2.0, 0.0, 0.3),
            spin_w_radps=(0.0, 1.0, 0.0),
        ),
        "predicted_outcome": kernel.PredictedOutcomeEvidence(
            evaluated=True,
            predicted_net_clear=True,
            predicted_legal_landing=True,
        ),
        "observed_outcome": kernel.ObservedOutcomeEvidence(
            resolved=True,
            stamp=kernel.EventStamp(6, 0),
            observed_net_clear=True,
            observed_legal_landing=True,
        ),
        "swing_closure": kernel.SwingClosureEvidence(
            closed=True,
            stamp=kernel.EventStamp(6, 1),
            timeout=False,
        ),
    }
    values.update(changes)
    return kernel.N1RewardEventInput(**values)


def _absent_flight():
    return kernel.OutgoingFlightEvidence(False, None, None, None, None)


def _unevaluated_prediction():
    return kernel.PredictedOutcomeEvidence(False, None, None)


def _unresolved_outcome():
    return kernel.ObservedOutcomeEvidence(False, None, None, None)


def test_source_binding_is_exact_not_a_best_effort_label():
    sample = _sample()
    changed = replace(SOURCE, source_sha256="c" * 64)
    with pytest.raises(kernel.N1RewardEventKernelError, match="source binding"):
        kernel.evaluate_n1_reward_event(sample, expected_source=changed)


def test_native_physical_fact_contract_is_source_bound_and_reward_unauthorized():
    assert PHYSICAL_CONTRACT["reward_authorized"] is False
    sample = _physical_sample()
    canonical = kernel.validate_native_physical_event_facts(
        sample, expected_source=PHYSICAL_SOURCE
    )
    assert canonical == sample
    canonical["invalid_reasons"].append("caller_mutation")
    assert sample["invalid_reasons"] == []


def test_native_selected_rubber_classification_becomes_contact_evidence():
    sample = _selected_physical_sample()
    evidence = kernel.contact_evidence_from_native_facts(
        sample, expected_source=PHYSICAL_SOURCE
    )
    assert evidence == kernel.ContactEvidence(
        occurred=True,
        stamp=kernel.EventStamp(4, 3),
        selected_rubber=True,
    )


@pytest.mark.parametrize(
    ("changes", "expected_selected"),
    [
        ({}, True),
        (
            {
                "racket_contact_edge_count_total": 2,
                "invalid_reasons": ["racket_recontact"],
            },
            False,
        ),
        (
            {
                "invalid_reasons": [
                    "racket_contact_simultaneous_with_other"
                ]
            },
            False,
        ),
    ],
)
def test_native_selected_contact_bridge_requires_one_clean_generic_edge(
    changes, expected_selected
):
    sample = _selected_physical_sample()
    sample.update(changes)
    evidence = kernel.contact_evidence_from_native_facts(
        sample, expected_source=PHYSICAL_SOURCE
    )
    assert evidence.occurred is True
    assert evidence.selected_rubber is expected_selected


def test_native_observed_outcome_consumer_uses_sealed_resolver_facts():
    sample = _selected_resolved_physical_sample()
    resolver_sha = sample["observed_outcome_resolver_binding"][
        "content_sha256"
    ]
    question_sha = sample["observed_outcome_question_binding"][
        "content_sha256"
    ]
    flight = kernel.outgoing_flight_evidence_from_native_facts(
        sample,
        expected_source=PHYSICAL_SOURCE,
        expected_outcome_resolver_binding_sha256=resolver_sha,
        expected_outcome_question_binding_sha256=question_sha,
        expected_outcome_scene_binding_sha256="5" * 64,
        expected_outcome_plant_binding_sha256="7" * 64,
        expected_question_source_sha256="b" * 64,
        expected_question_landing_aim_xy_w_m=(2.3, -0.665),
    )
    observed = kernel.observed_outcome_evidence_from_native_facts(
        sample,
        expected_source=PHYSICAL_SOURCE,
        expected_outcome_resolver_binding_sha256=resolver_sha,
        expected_outcome_question_binding_sha256=question_sha,
        expected_outcome_scene_binding_sha256="5" * 64,
        expected_outcome_plant_binding_sha256="7" * 64,
        expected_question_source_sha256="b" * 64,
        expected_question_landing_aim_xy_w_m=(2.3, -0.665),
    )
    assert flight.valid is True
    assert observed == kernel.ObservedOutcomeEvidence(
        resolved=True,
        stamp=kernel.EventStamp(5, 2),
        observed_net_clear=True,
        observed_legal_landing=True,
    )

    with pytest.raises(
        kernel.N1RewardEventKernelError,
        match="authority or snapshot|external parent",
    ):
        kernel.observed_outcome_evidence_from_native_facts(
            sample,
            expected_source=PHYSICAL_SOURCE,
            expected_outcome_resolver_binding_sha256="0" * 64,
            expected_outcome_question_binding_sha256=question_sha,
            expected_outcome_scene_binding_sha256="5" * 64,
            expected_outcome_plant_binding_sha256="7" * 64,
            expected_question_source_sha256="b" * 64,
            expected_question_landing_aim_xy_w_m=(2.3, -0.665),
        )

    truncated = copy.deepcopy(sample)
    snapshot = truncated["observed_outcome_snapshot"]
    snapshot.pop("content_sha256")
    snapshot["transcript_samples"].pop()
    snapshot["sample_count"] = len(snapshot["transcript_samples"])
    snapshot["last_sample"] = snapshot["transcript_samples"][-1]
    snapshot["last_sample_stamp"] = snapshot["last_sample"]["stamp"]
    snapshot["trace_sha256"] = outcome_resolver._trace_sha256(
        snapshot["transcript_samples"]
    )
    truncated["observed_outcome_snapshot"] = outcome_resolver._seal(snapshot)
    with pytest.raises(
        kernel.N1RewardEventKernelError,
        match="does not reach native fact cutoff",
    ):
        kernel.observed_outcome_evidence_from_native_facts(
            truncated,
            expected_source=PHYSICAL_SOURCE,
            expected_outcome_resolver_binding_sha256=resolver_sha,
            expected_outcome_question_binding_sha256=question_sha,
            expected_outcome_scene_binding_sha256="5" * 64,
            expected_outcome_plant_binding_sha256="7" * 64,
            expected_question_source_sha256="b" * 64,
            expected_question_landing_aim_xy_w_m=(2.3, -0.665),
        )


def test_production_native_sources_match_external_kernel_literals():
    assert hashlib.sha256(Path(outcome_resolver.__file__).read_bytes()).hexdigest() == (
        kernel.EXPECTED_OBSERVED_OUTCOME_RESOLVER_SOURCE_SHA256
    )
    assert hashlib.sha256(
        (WBT_ROOT / "mujoco_native/n1_ball_core.py").read_bytes()
    ).hexdigest() == kernel.EXPECTED_N1_BALL_CORE_SOURCE_SHA256


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selected_rubber_authority_available": True}, "selected-rubber"),
        (
            {
                "outgoing_flight": {
                    **_physical_sample()["outgoing_flight"],
                    "policy_tick": 4,
                    "physics_substep": 3,
                }
            },
            "strictly after",
        ),
        (
            {
                "outgoing_flight": {
                    **_physical_sample()["outgoing_flight"],
                    "linear_velocity_w_mps": [float("nan"), 0.0, 0.0],
                }
            },
            "finite",
        ),
    ],
)
def test_native_physical_fact_contract_rejects_false_authority_and_bad_facts(
    changes, message
):
    with pytest.raises(kernel.N1RewardEventKernelError, match=message):
        kernel.validate_native_physical_event_facts(
            _physical_sample(**changes), expected_source=PHYSICAL_SOURCE
        )


@pytest.mark.parametrize(
    "bad_flight",
    [
        kernel.OutgoingFlightEvidence(
            True,
            kernel.EventStamp(5, 2),
            (float("nan"), 0.0, 1.0),
            (2.0, 0.0, 0.3),
            (0.0, 1.0, 0.0),
        ),
        kernel.OutgoingFlightEvidence(
            True,
            kernel.EventStamp(5, 2),
            (1.0, 0.0, 1.0),
            (2.0, float("inf"), 0.3),
            (0.0, 1.0, 0.0),
        ),
    ],
)
def test_valid_outgoing_flight_requires_finite_state_fields(bad_flight):
    with pytest.raises(kernel.N1RewardEventKernelError, match="finite"):
        kernel.evaluate_n1_reward_event(_sample(outgoing_flight=bad_flight), expected_source=SOURCE)


def test_event_order_and_actual_contact_precondition_are_fail_closed():
    unordered = _sample(
        outgoing_flight=kernel.OutgoingFlightEvidence(
            True,
            kernel.EventStamp(5, 1),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 0.3),
            (0.0, 1.0, 0.0),
        )
    )
    with pytest.raises(kernel.N1RewardEventKernelError, match="strictly after"):
        kernel.evaluate_n1_reward_event(unordered, expected_source=SOURCE)

    no_selected_rubber = _sample(
        actual_contact=kernel.ContactEvidence(True, kernel.EventStamp(5, 1), False)
    )
    with pytest.raises(kernel.N1RewardEventKernelError, match="selected-rubber"):
        kernel.evaluate_n1_reward_event(no_selected_rubber, expected_source=SOURCE)


def test_eligibility_conservation_tracks_separate_denominators():
    eligible = kernel.evaluate_n1_reward_event(_sample(), expected_source=SOURCE)
    assert eligible.motion_mimic_denominator
    assert eligible.contact_target_denominator
    assert eligible.closed_swing_denominator
    assert eligible.actual_contact_numerator
    assert eligible.achieved_outgoing_flight_denominator
    assert eligible.predicted_outcome_denominator
    assert eligible.predicted_net_clear_numerator
    assert eligible.predicted_legal_landing_numerator
    assert eligible.observed_outcome_denominator
    assert eligible.observed_net_clear_numerator
    assert eligible.observed_legal_landing_numerator
    assert eligible.actual_contact_numerator <= eligible.closed_swing_denominator
    assert eligible.predicted_legal_landing_numerator <= eligible.predicted_outcome_denominator
    assert eligible.observed_legal_landing_numerator <= eligible.observed_outcome_denominator
    assert eligible.observed_outcome_denominator <= eligible.achieved_outgoing_flight_denominator


def test_miss_timeout_never_pays_outcome_and_does_not_mutate_input():
    miss = _sample(
        actual_contact=kernel.ContactEvidence(False, None, False),
        outgoing_flight=_absent_flight(),
        predicted_outcome=_unevaluated_prediction(),
        observed_outcome=_unresolved_outcome(),
        swing_closure=kernel.SwingClosureEvidence(True, kernel.EventStamp(7, 0), True),
    )
    before = copy.deepcopy(miss)
    result = kernel.evaluate_n1_reward_event(miss, expected_source=SOURCE)
    assert miss == before
    assert result.closed_swing_denominator
    assert result.contact_target_denominator  # A target eligibility is deliberately separate.
    assert not result.actual_contact_numerator
    assert not result.actual_contact_pay_eligible
    assert not result.predicted_outcome_pay_eligible
    assert not result.observed_outcome_pay_eligible
    assert not result.achieved_outgoing_flight_denominator
