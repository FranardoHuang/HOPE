import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "classify_sparse_reward_milestones.py"
CONTRACT = ROOT / "configs" / "phase1_sparse_reward_eligibility_contract_20260715.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("sparse_reward_classifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _measurement(
    milestone=200,
    *,
    run_id="p1_long_no_replay_control_seed3_eligibility_successor",
    opportunity=50,
    capture=1,
    net=1,
    landing=1,
    legal=1,
    qdot_observed=1000,
    qdot_active=0,
    qdot_excess=2,
):
    per_family = {
        "strike_opportunity_count": opportunity,
        "virtual_capture_count": capture,
        "virtual_net_clear_count": net,
        "virtual_landing_valid_count": landing,
        "virtual_legal_return_count": legal,
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_commit": "1" * 40,
        "training_claim_sha256": "2" * 64,
        "milestone": milestone,
        "checkpoint_sha256": f"{milestone % 10:x}" * 64,
        "counter_window": {
            "start_update_exclusive": -1,
            "end_update_inclusive": milestone,
        },
        "action_families": {
            "forehand": deepcopy(per_family),
            "backhand": deepcopy(per_family),
        },
        "qdot": {
            "observed_sample_count": qdot_observed,
            "hinge_active_sample_count": qdot_active,
            "excess_sample_count": qdot_excess,
            "normalized_excess_square_sum": 0.25 if qdot_excess else 0.0,
        },
        "measurement_contract": {
            "virtual_outcome_semantics": "analytic_virtual_contact_phase_a",
            "physical_contact_phase_b_observed": False,
            "same_step_virtual_ledger": True,
            "runtime_qdot_limits_bound": True,
            "counter_window_complete": True,
            "counter_reset_at_window_start": True,
        },
    }


def _classify(tmp_path, *measurements):
    module = _module()
    contract = module.load_contract(CONTRACT)
    paths_and_values = []
    for index, measurement in enumerate(measurements):
        path = tmp_path / f"measurement-{index}.json"
        path.write_text(json.dumps(measurement), encoding="utf-8")
        paths_and_values.append((path, measurement))
    return module.classify(contract, paths_and_values)


def test_contract_freezes_required_thresholds_and_action_families():
    module = _module()
    contract = module.load_contract(CONTRACT)
    assert contract["thresholds"] == {
        "minimum_strike_opportunities_total": 100,
        "minimum_strike_opportunities_per_action": 50,
        "minimum_virtual_captures_per_action": 1,
        "consecutive_eligible_milestones": 2,
    }
    assert contract["runs"]["p1_long_no_replay_qdot_w5_seed3_eligibility_successor"] == {
        "required_action_families": ["forehand", "backhand"],
        "qdot_hinge_expected_active": True,
    }


def test_zero_strikes_continue_without_claiming_a_negative_reward_result(tmp_path):
    measurement = _measurement(
        opportunity=0, capture=0, net=0, landing=0, legal=0
    )
    receipt = _classify(tmp_path, measurement)
    assert receipt["latest"]["state"] == "NO_OPPORTUNITY_CONTINUE"
    assert receipt["latest"]["automatic_trainer_action"] == "CONTINUE_UNCHANGED"


@pytest.mark.parametrize(
    "change",
    [
        {"opportunity": 49, "capture": 1},
        {"opportunity": 50, "capture": 0, "net": 0, "landing": 0, "legal": 0},
    ],
)
def test_sparse_or_underpowered_hit_conditioned_rewards_are_censored(tmp_path, change):
    receipt = _classify(tmp_path, _measurement(**change))
    assert receipt["latest"]["state"] == "CENSORED_CONTINUE"
    assert receipt["latest"]["eligible_denominators_passed"] is False


def test_one_eligible_milestone_is_direction_only(tmp_path):
    receipt = _classify(tmp_path, _measurement())
    assert receipt["latest"]["state"] == "DIRECTION_ONLY"
    assert receipt["trainer_control"] == {
        "mode": "receipt_only",
        "automatic_stop_authorized": False,
        "automatic_restart_authorized": False,
        "automatic_promotion_authorized": False,
        "automatic_second_seed_authorized": False,
        "required_action_for_all_states": "CONTINUE_UNCHANGED",
    }


def test_two_consecutive_eligible_milestones_are_decision_eligible_but_do_not_stop(tmp_path):
    receipt = _classify(tmp_path, _measurement(200), _measurement(500))
    assert [item["state"] for item in receipt["classifications"]] == [
        "DIRECTION_ONLY",
        "DECISION_ELIGIBLE",
    ]
    assert receipt["latest"]["automatic_trainer_action"] == "CONTINUE_UNCHANGED"
    assert receipt["evidence_boundary"]["physical_contact_phase_b_measured"] is False


def test_nonconsecutive_eligible_milestones_remain_direction_only(tmp_path):
    receipt = _classify(tmp_path, _measurement(200), _measurement(1000))
    assert receipt["latest"]["state"] == "DIRECTION_ONLY"


def test_qdot_treatment_requires_observed_active_and_excess_samples(tmp_path):
    treatment = _measurement(
        run_id="p1_long_no_replay_qdot_w5_seed3_eligibility_successor",
        qdot_observed=1000,
        qdot_active=1000,
        qdot_excess=0,
    )
    censored = _classify(tmp_path, treatment)
    assert censored["latest"]["state"] == "CENSORED_CONTINUE"

    treatment["qdot"]["excess_sample_count"] = 3
    treatment["qdot"]["normalized_excess_square_sum"] = 0.2
    eligible = _classify(tmp_path, treatment)
    assert eligible["latest"]["state"] == "DIRECTION_ONLY"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["action_families"]["forehand"].update(
            virtual_capture_count=51
        ),
        lambda value: value["action_families"]["forehand"].update(
            virtual_legal_return_count=2
        ),
        lambda value: value["qdot"].update(hinge_active_sample_count=1),
        lambda value: value["measurement_contract"].update(
            physical_contact_phase_b_observed=True
        ),
        lambda value: value.update(unregistered_field=True),
    ],
)
def test_bad_counter_or_contract_evidence_is_measurement_invalid(tmp_path, mutate):
    measurement = _measurement()
    mutate(measurement)
    receipt = _classify(tmp_path, measurement)
    assert receipt["latest"]["state"] == "MEASUREMENT_INVALID"


def test_cli_writes_one_no_clobber_receipt_and_never_edits_measurement(tmp_path):
    measurement = tmp_path / "measurement.json"
    original = json.dumps(_measurement(), sort_keys=True)
    measurement.write_text(original, encoding="utf-8")
    output = tmp_path / "receipt.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--contract",
        str(CONTRACT),
        "--measurement",
        str(measurement),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["latest"]["state"] == "DIRECTION_ONLY"
    assert measurement.read_text(encoding="utf-8") == original
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True)
