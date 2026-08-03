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
        "schema_version": 2,
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
            "position_w_m": [1.0, 0.0, 1.0],
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
