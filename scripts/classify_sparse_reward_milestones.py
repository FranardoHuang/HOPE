#!/usr/bin/env python3
"""Classify sparse-reward milestone evidence without controlling a trainer.

The input is a set of immutable, cumulative milestone counter ledgers.  The
classifier validates denominators before interpreting any reward direction and
writes one no-clobber JSON receipt.  It deliberately has no process, queue,
checkpoint, SSH, or simulator control path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
STATES = (
    "NO_OPPORTUNITY_CONTINUE",
    "CENSORED_CONTINUE",
    "MEASUREMENT_INVALID",
    "DIRECTION_ONLY",
    "DECISION_ELIGIBLE",
)

_MEASUREMENT_KEYS = {
    "schema_version",
    "run_id",
    "source_commit",
    "training_claim_sha256",
    "milestone",
    "checkpoint_sha256",
    "counter_window",
    "action_families",
    "qdot",
    "measurement_contract",
}
_WINDOW_KEYS = {"start_update_exclusive", "end_update_inclusive"}
_ACTION_KEYS = {
    "strike_opportunity_count",
    "virtual_capture_count",
    "virtual_net_clear_count",
    "virtual_landing_valid_count",
    "virtual_legal_return_count",
}
_QDOT_KEYS = {
    "observed_sample_count",
    "hinge_active_sample_count",
    "excess_sample_count",
    "normalized_excess_square_sum",
}
_MEASUREMENT_CONTRACT_KEYS = {
    "virtual_outcome_semantics",
    "physical_contact_phase_b_observed",
    "same_step_virtual_ledger",
    "runtime_qdot_limits_bound",
    "counter_window_complete",
    "counter_reset_at_window_start",
}


class ContractError(ValueError):
    """The classifier contract itself is malformed and cannot be trusted."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _exact_keys(value: Any, expected: set[str], label: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label} must be an object"]
    actual = set(value)
    errors: list[str] = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{label} missing keys: {missing}")
    if extra:
        errors.append(f"{label} unexpected keys: {extra}")
    return errors


def load_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "contract_id",
        "milestones",
        "thresholds",
        "runs",
    }
    errors = _exact_keys(raw, required, "contract")
    if errors:
        raise ContractError("; ".join(errors))
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ContractError("contract schema_version must equal 1")
    if not isinstance(raw["contract_id"], str) or not raw["contract_id"]:
        raise ContractError("contract_id must be a non-empty string")
    milestones = raw["milestones"]
    if (
        not isinstance(milestones, list)
        or not milestones
        or any(type(item) is not int or item <= 0 for item in milestones)
        or milestones != sorted(set(milestones))
    ):
        raise ContractError("milestones must be unique positive integers in ascending order")

    thresholds = raw["thresholds"]
    threshold_keys = {
        "minimum_strike_opportunities_total",
        "minimum_strike_opportunities_per_action",
        "minimum_virtual_captures_per_action",
        "consecutive_eligible_milestones",
    }
    errors = _exact_keys(thresholds, threshold_keys, "contract.thresholds")
    if errors:
        raise ContractError("; ".join(errors))
    for key in threshold_keys:
        if type(thresholds[key]) is not int or thresholds[key] < 1:
            raise ContractError(f"contract.thresholds.{key} must be a positive integer")
    if thresholds["minimum_strike_opportunities_total"] < 100:
        raise ContractError("minimum_strike_opportunities_total must be at least 100")
    if thresholds["minimum_strike_opportunities_per_action"] < 50:
        raise ContractError("minimum_strike_opportunities_per_action must be at least 50")
    if thresholds["consecutive_eligible_milestones"] != 2:
        raise ContractError("consecutive_eligible_milestones must equal 2 in schema 1")

    runs = raw["runs"]
    if not isinstance(runs, Mapping) or not runs:
        raise ContractError("contract.runs must be a non-empty object")
    run_keys = {"required_action_families", "qdot_hinge_expected_active"}
    for run_id, run in runs.items():
        if not isinstance(run_id, str) or not run_id:
            raise ContractError("contract run ids must be non-empty strings")
        errors = _exact_keys(run, run_keys, f"contract.runs.{run_id}")
        if errors:
            raise ContractError("; ".join(errors))
        families = run["required_action_families"]
        if (
            not isinstance(families, list)
            or not families
            or any(not isinstance(item, str) or not item for item in families)
            or len(families) != len(set(families))
        ):
            raise ContractError(
                f"contract.runs.{run_id}.required_action_families must be unique non-empty strings"
            )
        if type(run["qdot_hinge_expected_active"]) is not bool:
            raise ContractError(
                f"contract.runs.{run_id}.qdot_hinge_expected_active must be boolean"
            )
    return dict(raw)


def _validate_measurement(
    measurement: Any,
    contract: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    errors = _exact_keys(measurement, _MEASUREMENT_KEYS, "measurement")
    if not isinstance(measurement, Mapping):
        return errors, None

    run_id = measurement.get("run_id")
    run_contract = contract["runs"].get(run_id)
    if run_contract is None:
        errors.append(f"run_id {run_id!r} is not registered in the contract")
    if measurement.get("schema_version") != SCHEMA_VERSION:
        errors.append("measurement.schema_version must equal 1")
    if not isinstance(run_id, str) or not run_id:
        errors.append("measurement.run_id must be a non-empty string")
    if not _is_git_commit(measurement.get("source_commit")):
        errors.append("measurement.source_commit must be a lowercase 40-hex commit")
    if not _is_sha256(measurement.get("training_claim_sha256")):
        errors.append("measurement.training_claim_sha256 must be lowercase 64-hex")
    if not _is_sha256(measurement.get("checkpoint_sha256")):
        errors.append("measurement.checkpoint_sha256 must be lowercase 64-hex")
    milestone = measurement.get("milestone")
    if type(milestone) is not int or milestone not in contract["milestones"]:
        errors.append("measurement.milestone must be registered in contract.milestones")

    window = measurement.get("counter_window")
    errors.extend(_exact_keys(window, _WINDOW_KEYS, "measurement.counter_window"))
    if isinstance(window, Mapping):
        start = window.get("start_update_exclusive")
        end = window.get("end_update_inclusive")
        if type(start) is not int or start < -1:
            errors.append("counter_window.start_update_exclusive must be an integer >= -1")
        if type(end) is not int or end < 0:
            errors.append("counter_window.end_update_inclusive must be an integer >= 0")
        if type(start) is int and type(end) is int and end <= start:
            errors.append("counter_window must have end_update_inclusive > start_update_exclusive")
        if type(milestone) is int and type(end) is int and end != milestone:
            errors.append("counter_window.end_update_inclusive must equal milestone")

    action_families = measurement.get("action_families")
    if not isinstance(action_families, Mapping):
        errors.append("measurement.action_families must be an object")
    elif run_contract is not None:
        expected_families = set(run_contract["required_action_families"])
        actual_families = set(action_families)
        if actual_families != expected_families:
            errors.append(
                "measurement.action_families must equal the registered family set: "
                f"expected={sorted(expected_families)} actual={sorted(actual_families)}"
            )
        for family, counters in action_families.items():
            errors.extend(
                _exact_keys(counters, _ACTION_KEYS, f"action_families.{family}")
            )
            if not isinstance(counters, Mapping):
                continue
            for key in _ACTION_KEYS:
                if key in counters and not _is_count(counters[key]):
                    errors.append(f"action_families.{family}.{key} must be a non-negative integer")
            if all(key in counters and _is_count(counters[key]) for key in _ACTION_KEYS):
                opportunity = counters["strike_opportunity_count"]
                capture = counters["virtual_capture_count"]
                net = counters["virtual_net_clear_count"]
                landing = counters["virtual_landing_valid_count"]
                legal = counters["virtual_legal_return_count"]
                if capture > opportunity:
                    errors.append(f"action_families.{family}: capture exceeds strike opportunity")
                if net > capture:
                    errors.append(f"action_families.{family}: net-clear exceeds capture")
                if landing > capture:
                    errors.append(f"action_families.{family}: landing-valid exceeds capture")
                if legal > net or legal > landing:
                    errors.append(
                        f"action_families.{family}: legal return exceeds net-clear or landing-valid"
                    )

    qdot = measurement.get("qdot")
    errors.extend(_exact_keys(qdot, _QDOT_KEYS, "measurement.qdot"))
    if isinstance(qdot, Mapping):
        for key in _QDOT_KEYS - {"normalized_excess_square_sum"}:
            if key in qdot and not _is_count(qdot[key]):
                errors.append(f"measurement.qdot.{key} must be a non-negative integer")
        excess_sum = qdot.get("normalized_excess_square_sum")
        if (
            isinstance(excess_sum, bool)
            or not isinstance(excess_sum, (int, float))
            or not math.isfinite(float(excess_sum))
            or float(excess_sum) < 0.0
        ):
            errors.append(
                "measurement.qdot.normalized_excess_square_sum must be finite and non-negative"
            )
        count_keys_valid = all(
            _is_count(qdot.get(key))
            for key in (
                "observed_sample_count",
                "hinge_active_sample_count",
                "excess_sample_count",
            )
        )
        if count_keys_valid:
            observed = qdot["observed_sample_count"]
            active = qdot["hinge_active_sample_count"]
            excess = qdot["excess_sample_count"]
            if active > observed:
                errors.append("measurement.qdot.hinge_active_sample_count exceeds observed")
            if excess > observed:
                errors.append("measurement.qdot.excess_sample_count exceeds observed")
            if excess == 0 and isinstance(excess_sum, (int, float)) and float(excess_sum) != 0.0:
                errors.append("qdot excess sum must be zero when excess_sample_count is zero")
            if excess > 0 and isinstance(excess_sum, (int, float)) and float(excess_sum) <= 0.0:
                errors.append("qdot excess sum must be positive when excess_sample_count is positive")
            if run_contract is not None:
                expected_active = run_contract["qdot_hinge_expected_active"]
                if expected_active and active != observed:
                    errors.append(
                        "qdot hinge is registered active, so hinge_active_sample_count must equal observed"
                    )
                if not expected_active and active != 0:
                    errors.append(
                        "qdot hinge is registered inactive, so hinge_active_sample_count must be zero"
                    )

    measurement_contract = measurement.get("measurement_contract")
    errors.extend(
        _exact_keys(
            measurement_contract,
            _MEASUREMENT_CONTRACT_KEYS,
            "measurement.measurement_contract",
        )
    )
    if isinstance(measurement_contract, Mapping):
        expected_values = {
            "virtual_outcome_semantics": "analytic_virtual_contact_phase_a",
            "physical_contact_phase_b_observed": False,
            "same_step_virtual_ledger": True,
            "runtime_qdot_limits_bound": True,
            "counter_window_complete": True,
            "counter_reset_at_window_start": True,
        }
        for key, expected in expected_values.items():
            if measurement_contract.get(key) != expected:
                errors.append(
                    f"measurement.measurement_contract.{key} must equal {expected!r}"
                )
    return errors, dict(measurement)


def _base_state(
    measurement: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    thresholds: Mapping[str, int],
) -> tuple[str, list[str], bool]:
    families = run_contract["required_action_families"]
    counters = measurement["action_families"]
    total_opportunities = sum(
        counters[family]["strike_opportunity_count"] for family in families
    )
    if total_opportunities == 0:
        return (
            "NO_OPPORTUNITY_CONTINUE",
            ["no exact-strike opportunity was observed in this milestone window"],
            False,
        )

    censor_reasons: list[str] = []
    if total_opportunities < thresholds["minimum_strike_opportunities_total"]:
        censor_reasons.append(
            "total strike opportunities below "
            f"{thresholds['minimum_strike_opportunities_total']}"
        )
    for family in families:
        family_counts = counters[family]
        if (
            family_counts["strike_opportunity_count"]
            < thresholds["minimum_strike_opportunities_per_action"]
        ):
            censor_reasons.append(
                f"{family} strike opportunities below "
                f"{thresholds['minimum_strike_opportunities_per_action']}"
            )
        if (
            family_counts["virtual_capture_count"]
            < thresholds["minimum_virtual_captures_per_action"]
        ):
            censor_reasons.append(
                f"{family} virtual captures below "
                f"{thresholds['minimum_virtual_captures_per_action']}; net/landing/legal-return "
                "rewards were not observed often enough"
            )

    qdot = measurement["qdot"]
    if qdot["observed_sample_count"] == 0:
        censor_reasons.append("qdot observer saw zero samples")
    if (
        run_contract["qdot_hinge_expected_active"]
        and qdot["excess_sample_count"] == 0
    ):
        censor_reasons.append(
            "qdot hinge was active but no above-margin sample occurred, so its gradient was censored"
        )
    if censor_reasons:
        return "CENSORED_CONTINUE", censor_reasons, False
    return (
        "DIRECTION_ONLY",
        ["all eligibility denominators passed for one milestone"],
        True,
    )


def classify(
    contract: Mapping[str, Any],
    measurements: Sequence[tuple[Path, Any]],
) -> dict[str, Any]:
    if not measurements:
        raise ValueError("at least one --measurement is required")

    validated: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for path, raw in measurements:
        errors, normalized = _validate_measurement(raw, contract)
        input_records.append(
            {
                "path": str(path),
                "file_sha256": _sha256_file(path),
                "content_sha256": _sha256_bytes(_canonical_json_bytes(raw)),
            }
        )
        validated.append(
            {
                "path": str(path),
                "measurement": normalized,
                "errors": errors,
            }
        )

    run_ids = {
        item["measurement"].get("run_id")
        for item in validated
        if item["measurement"] is not None
    }
    cross_errors: list[str] = []
    if len(run_ids) != 1:
        cross_errors.append("all measurement files must describe exactly one run_id")
    run_id = next(iter(run_ids)) if len(run_ids) == 1 else None

    milestones = [
        item["measurement"].get("milestone")
        for item in validated
        if item["measurement"] is not None
    ]
    valid_int_milestones = [value for value in milestones if type(value) is int]
    if len(valid_int_milestones) != len(set(valid_int_milestones)):
        cross_errors.append("duplicate milestone measurements are forbidden")

    identity_fields = ("source_commit", "training_claim_sha256")
    for field in identity_fields:
        values = {
            item["measurement"].get(field)
            for item in validated
            if item["measurement"] is not None
        }
        if len(values) != 1:
            cross_errors.append(f"all measurements must share one {field}")

    order = {milestone: index for index, milestone in enumerate(contract["milestones"])}
    validated.sort(
        key=lambda item: order.get(
            item["measurement"].get("milestone") if item["measurement"] else None,
            len(order) + 1,
        )
    )
    classifications: list[dict[str, Any]] = []
    previous_qualified = False
    previous_milestone: int | None = None
    for item in validated:
        measurement = item["measurement"]
        errors = list(item["errors"])
        errors.extend(cross_errors)
        milestone = measurement.get("milestone") if measurement else None
        if errors or measurement is None or run_id not in contract["runs"]:
            state = "MEASUREMENT_INVALID"
            reasons = sorted(set(errors or ["measurement is not an object"]))
            qualified = False
        else:
            state, reasons, qualified = _base_state(
                measurement,
                contract["runs"][run_id],
                contract["thresholds"],
            )
            if qualified:
                index = order[milestone]
                immediately_previous = (
                    index > 0 and previous_milestone == contract["milestones"][index - 1]
                )
                if previous_qualified and immediately_previous:
                    state = "DECISION_ELIGIBLE"
                    reasons = [
                        "eligibility denominators passed at two consecutive registered milestones"
                    ]
                else:
                    state = "DIRECTION_ONLY"
                    reasons = [
                        "eligibility denominators passed once; a consecutive eligible milestone is still required"
                    ]
        classifications.append(
            {
                "milestone": milestone,
                "state": state,
                "reasons": reasons,
                "eligible_denominators_passed": qualified,
                "automatic_trainer_action": "CONTINUE_UNCHANGED",
            }
        )
        previous_qualified = qualified
        previous_milestone = milestone if type(milestone) is int else None

    latest = classifications[-1]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "phase1_sparse_reward_milestone_classification_receipt",
        "contract": {
            "contract_id": contract["contract_id"],
            "milestones": list(contract["milestones"]),
            "thresholds": dict(contract["thresholds"]),
        },
        "run_id": run_id,
        "inputs": input_records,
        "classifications": classifications,
        "latest": dict(latest),
        "trainer_control": {
            "mode": "receipt_only",
            "automatic_stop_authorized": False,
            "automatic_restart_authorized": False,
            "automatic_promotion_authorized": False,
            "automatic_second_seed_authorized": False,
            "required_action_for_all_states": "CONTINUE_UNCHANGED",
        },
        "evidence_boundary": {
            "virtual_outcomes": "analytic_virtual_contact_phase_a_only",
            "physical_contact_phase_b_measured": False,
            "physical_hit_net_landing_or_legal_return_claim_authorized": False,
        },
    }
    if receipt["latest"]["state"] not in STATES:
        raise AssertionError("classifier emitted an unknown state")
    return receipt


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--measurement",
        type=Path,
        action="append",
        required=True,
        help="repeat once per cumulative milestone counter ledger",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    contract = load_contract(args.contract)
    measurements = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.measurement
    ]
    receipt = classify(contract, measurements)
    _write_no_clobber(args.output, _canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
