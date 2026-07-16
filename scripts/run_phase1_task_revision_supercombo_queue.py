#!/usr/bin/env python3
"""Validate and operate the task-revision successor continuation queue.

The reviewed rolling continuation harness still owns SSH, immutable claims,
same-Pod parent verification, process isolation, and no-clobber launch.  This
entry point adds the scientific contract that the superseded timing queue did
not know about: every cell must enable one complete same-ball task-revision
profile, an explicit preparation-time mixture with a real 0.5-second point
mass, and the exact per-update behavior ledger needed for honest pruning.

``validate`` and ``plan`` never contact a Pod.  A pending queue is expected to
validate while remaining activation-blocked.  After the source commit and
full-scene evidence are materialized, the same entry point can delegate the
reviewed no-clobber continuation operations without weakening their checks.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Mapping
import zlib


_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import run_phase1_rolling_timing_supercombo_queue as continuation  # noqa: E402
yaml = continuation.yaml


QUEUE_PATH = Path("configs/phase1_task_revision_supercombo_20260716.yaml")
CONFIRM = "SIM_ONLY_LAUNCH_ONE_TASK_REVISION_SUPERCOMBO_JOB"
FULL_SCENE_PROBE_CONFIRM = "SIM_ONLY_RUN_ONE_TASK_REVISION_FULL_SCENE_PROBE"
FINALIZE_FULL_SCENE_PROBE_CONFIRM = (
    "SIM_ONLY_FINALIZE_ONE_TASK_REVISION_FULL_SCENE_PROBE"
)
BEHAVIOR_INSPECT_CONFIRM = "SIM_ONLY_INSPECT_ONE_TASK_REVISION_BEHAVIOR_WINDOW"
BEHAVIOR_ATTEST_CONFIRM = "SIM_ONLY_ATTEST_ONE_TASK_REVISION_BEHAVIOR_WINDOW"
BEHAVIOR_STOP_CONFIRM = "SIM_ONLY_EXACT_STOP_ONE_TASK_REVISION_BEHAVIOR_FAILURE"
PORTFOLIO_INSPECT_CONFIRM = "SIM_ONLY_INSPECT_ONE_TASK_REVISION_PARENT_PORTFOLIO"
PORTFOLIO_ATTEST_CONFIRM = "SIM_ONLY_ATTEST_ONE_TASK_REVISION_PARENT_PORTFOLIO"
LOCAL_PROBE_FINALIZE_CONFIRM = "SIM_ONLY_LOCAL_FINALIZE_TASK_REVISION_PROBE"
PENDING_STATUS = "pending_source_commit_and_full_scene_probe"
ACTIVATED_STATUS = continuation.ACTIVATED_PREREGISTRATION_STATUS
EXPECTED_REVISION_KEYS = {
    "enabled",
    "profile",
    "initial_tts_range_s",
    "initial_tts_mixture",
    "position_std_m",
    "velocity_std_mps",
    "normal_std_rad",
    "tts_std_s",
}
EXPECTED_PROFILE_KEYS = {
    "policy_dt_s",
    "min_tts_s",
    "max_tts_s",
    "max_phase_rate_per_s",
    "max_phase_acceleration_per_s2",
    "max_deadline_revision_delta_s",
    "max_position_revision_delta_m",
    "max_velocity_revision_delta_mps",
    "max_normal_revision_delta_rad",
    "normal_unit_tolerance",
    "early_deadline_tolerance_s",
    "contract_version",
    "schema_version",
}
EXPECTED_COMPONENTS = (
    ("late_stress", (0.36, 0.49)),
    ("baseline_0p5", (0.5, 0.5)),
    ("fast_deploy", (0.5, 0.9)),
    ("broad_arrival", (0.9, 1.7)),
)
ALLOWED_NOISE = {
    (0.02, 0.10, 0.05, 0.08),
    (0.04, 0.20, 0.10, 0.16),
}
REQUIRED_OFFSETS = [200, 500, 1000, 2000]
REQUIRED_SINGLE_CLOCK_OVERRIDES = {
    "task.motion.hold_steps_range": "task.motion.hold_steps_range=[0,0]",
    "task.motion.stand_start_min_hold": "task.motion.stand_start_min_hold=0",
    "task.motion.post_swing_min_hold": "task.motion.post_swing_min_hold=0",
}
REQUIRED_TASK_IDENTITY_OVERRIDES = {
    "task.motion.clip_switch_prob": "task.motion.clip_switch_prob=0.0",
}
EXACT_EVENT_PREFIX = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
READY_METRICS = {
    "ready_tilt_rad_mean": "minimize",
    "ready_base_speed_xy_mps_mean": "minimize",
    "ready_station_offset_m_mean": "minimize",
    "ready_foot_contact_fraction_mean": "maximize",
    "ready_foot_slip_speed_mps_mean": "minimize",
}
REVISION_ACTIVATION_COUNTERS = frozenset(
    {
        "planner_revision_attempt_count",
        "planner_revision_accepted_count",
        "planner_revision_last_precontact_attempt_count",
        "planner_revision_last_precontact_accepted_count",
        "planner_revision_actor_visible_count",
        "planner_revision_last_precontact_actor_visible_count",
    }
)
PORTFOLIO_OFFSETS = frozenset({200, 500, 1000})
TRANSPORT_BLOCKED_JOB_IDS = frozenset(
    {
        "taskrev_p1_core_low_noise",
        "taskrev_p1_uncompensated_negative",
    }
)
TRANSPORT_BLOCKER = "governor_actor_transport_not_atomic"


class SuccessorQueueError(continuation.ContinuationQueueError):
    """The successor-specific scientific contract is incomplete or ambiguous."""


def _job_launch_authorized(job: Mapping[str, Any]) -> bool:
    return job.get("scientific_launch_authorized") is True


def _require_launchable_job(job: Mapping[str, Any]) -> None:
    if not _job_launch_authorized(job):
        raise SuccessorQueueError(
            f"{job.get('id')} is scientific NO-LAUNCH: "
            f"{job.get('scientific_blocker')}"
        )


def _launchable_continuation_queue(
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact control document with scientifically blocked cells omitted.

    The tracked 24-cell document remains the preregistration and validation truth.  The
    delegated legacy launcher, however, treats every blocked row as a global activation
    blocker.  Filtering only at this adapter boundary lets the 22 valid delay-zero cells run
    without ever relabelling the two known-bad delay rows as READY.  Original file provenance is
    retained so the superseded-queue deny gate and claim construction still bind the tracked
    control document.
    """

    value = dict(queue)
    value["jobs"] = [job for job in queue["jobs"] if _job_launch_authorized(job)]
    source_path = getattr(queue, "source_path", None)
    source_sha256 = getattr(queue, "source_sha256", None)
    if isinstance(source_path, Path) and isinstance(source_sha256, str):
        return continuation._LoadedContinuationQueue(
            value,
            source_path=source_path,
            source_sha256=source_sha256,
        )
    return value


def _hydra_literal(value: Any) -> str:
    """Render strict JSON data as one Hydra override value without quoted map keys.

    Hydra's override grammar accepts JSON-compatible values but not JSON's quoted mapping
    keys.  The tracked queue stores this canonical native form directly: validation, launch,
    immutable claim reconstruction and later attestation therefore all consume identical argv
    bytes, with no hidden transport rewrite.
    """

    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise SuccessorQueueError("Hydra transport forbids non-finite numbers")
        rendered = repr(value)
        if "e" in rendered.lower():
            mantissa, exponent = rendered.lower().split("e", 1)
            if "." not in mantissa:
                mantissa += ".0"
            rendered = f"{mantissa}e{int(exponent):+d}"
        return rendered
    if isinstance(value, str):
        return json.dumps(value, allow_nan=False, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_hydra_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        rows: list[str] = []
        for key in sorted(value):
            if (
                not isinstance(key, str)
                or not key
                or key[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
                or not all(
                    character
                    in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789"
                    for character in key
                )
            ):
                raise SuccessorQueueError(
                    f"Hydra transport cannot encode mapping key {key!r}"
                )
            rows.append(f"{key}: {_hydra_literal(value[key])}")
        return "{" + ", ".join(rows) + "}"
    raise SuccessorQueueError(
        f"Hydra transport cannot encode value of type {type(value).__name__}"
    )
def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuccessorQueueError(f"{label} must be a mapping")
    return value


def _compiled_overrides(job: Mapping[str, Any]) -> dict[str, str]:
    arguments = continuation._compile_recipe(job, str(job.get("id", "job")))
    result: dict[str, str] = {}
    for argument in arguments:
        key = continuation.lean._override_key(argument, f"{job['id']} recipe")
        if key in result:
            raise SuccessorQueueError(f"{job['id']} has duplicate final key {key!r}")
        result[key] = argument
    return result


def _override_hydra_mapping(argument: str, *, key: str, job_id: str) -> dict[str, Any]:
    prefix, separator, raw = argument.partition("=")
    if not separator or prefix.lstrip("+") != key:
        raise SuccessorQueueError(f"{job_id} does not provide one complete {key} override")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SuccessorQueueError(
            f"{job_id} {key} must be one canonical Hydra mapping: {exc}"
        ) from exc
    document = _mapping(value, f"{job_id}.{key}")
    canonical = _hydra_literal(document)
    if raw != canonical:
        raise SuccessorQueueError(
            f"{job_id} {key} is not the canonical Hydra mapping; quoted JSON keys and "
            "implicit transport rewrites are forbidden"
        )
    return document


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuccessorQueueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SuccessorQueueError(f"{label} must be finite")
    return result


def _validate_revision(job: Mapping[str, Any], overrides: Mapping[str, str]) -> dict[str, Any]:
    job_id = str(job["id"])
    if "task.planner_revision" not in overrides:
        raise SuccessorQueueError(f"{job_id} is missing task.planner_revision")
    revision = _override_hydra_mapping(
        overrides["task.planner_revision"], key="task.planner_revision", job_id=job_id
    )
    if set(revision) != EXPECTED_REVISION_KEYS or revision.get("enabled") is not True:
        raise SuccessorQueueError(
            f"{job_id} task.planner_revision must be enabled and contain exactly "
            f"{sorted(EXPECTED_REVISION_KEYS)}"
        )
    profile = _mapping(revision["profile"], f"{job_id}.profile")
    if set(profile) != EXPECTED_PROFILE_KEYS:
        raise SuccessorQueueError(f"{job_id} phase-governor profile is incomplete")
    if profile.get("contract_version") != "phase_governor_v1" or profile.get("schema_version") != 1:
        raise SuccessorQueueError(f"{job_id} phase-governor version changed")
    if _finite(profile["policy_dt_s"], f"{job_id}.policy_dt_s") != 0.02:
        raise SuccessorQueueError(f"{job_id} policy clock must remain 50 Hz")
    if _finite(profile["min_tts_s"], f"{job_id}.min_tts_s") != 0.02:
        raise SuccessorQueueError(
            f"{job_id} min_tts_s must equal one 50-Hz policy tick so revisions stay live "
            "through the final pre-contact tick"
        )
    if (
        _finite(
            profile["early_deadline_tolerance_s"],
            f"{job_id}.early_deadline_tolerance_s",
        )
        != 1.0e-6
    ):
        raise SuccessorQueueError(
            f"{job_id} early_deadline_tolerance_s must be exact 1e-6 for the "
            "float32 0.02-second policy grid"
        )
    if _finite(profile["max_tts_s"], f"{job_id}.max_tts_s") < 1.7:
        raise SuccessorQueueError(f"{job_id} governor excludes the broad-arrival stratum")

    initial_range = revision["initial_tts_range_s"]
    if initial_range != [0.36, 1.7]:
        raise SuccessorQueueError(
            f"{job_id} initial_tts_range_s must be [0.36, 1.7]; 0.5 is not the floor"
        )
    mixture = _mapping(revision["initial_tts_mixture"], f"{job_id}.initial_tts_mixture")
    if set(mixture) != {"contract_version", "components"}:
        raise SuccessorQueueError(f"{job_id} initial TTS mixture shape changed")
    if mixture.get("contract_version") != "initial_tts_mixture_v1":
        raise SuccessorQueueError(f"{job_id} initial TTS mixture version changed")
    components = mixture.get("components")
    if not isinstance(components, list) or len(components) != len(EXPECTED_COMPONENTS):
        raise SuccessorQueueError(f"{job_id} must contain all four preparation-time strata")
    weights: list[float] = []
    for index, ((expected_name, expected_range), component) in enumerate(
        zip(EXPECTED_COMPONENTS, components, strict=True)
    ):
        row = _mapping(component, f"{job_id}.components[{index}]")
        if set(row) != {"name", "range_s", "weight"}:
            raise SuccessorQueueError(f"{job_id} mixture component {index} shape changed")
        if row.get("name") != expected_name or row.get("range_s") != list(expected_range):
            raise SuccessorQueueError(
                f"{job_id} mixture component {index} must be {expected_name} {list(expected_range)}"
            )
        weight = _finite(row.get("weight"), f"{job_id}.{expected_name}.weight")
        if weight <= 0.0:
            raise SuccessorQueueError(f"{job_id}.{expected_name}.weight must be positive")
        weights.append(weight)
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise SuccessorQueueError(f"{job_id} initial TTS weights must sum to one")

    noise = tuple(
        _finite(revision[key], f"{job_id}.{key}")
        for key in ("position_std_m", "velocity_std_mps", "normal_std_rad", "tts_std_s")
    )
    if noise not in ALLOWED_NOISE:
        raise SuccessorQueueError(f"{job_id} must use one of the two registered planner-noise levels")
    return revision


def _validate_ledger(queue: Mapping[str, Any]) -> None:
    ledger = _mapping(queue.get("exact_behavior_ledger_contract"), "exact_behavior_ledger_contract")
    if ledger.get("event_prefix") != "HOPE_EXACT_BEHAVIOR_UPDATE_JSON=":
        raise SuccessorQueueError("exact behavior ledger must bind the per-update event prefix")
    if ledger.get("window_updates") != 100 or ledger.get("consume_once_per_update") is not True:
        raise SuccessorQueueError("exact behavior ledger must use consume-once 100-update windows")
    if ledger.get("missing_or_duplicate_update_action") != "continue_training_no_decision":
        raise SuccessorQueueError("incomplete exact windows must continue, never prune")
    required = set(ledger.get("required_counters", ()))
    minimum = {
        "swing_start_count",
        "swing_outcome_count",
        "swing_completion_count",
        "strike_opportunity_count",
        "virtual_legal_return_count",
        "pre_strike_physical_fall_count",
        "post_strike_physical_fall_count",
        "termination_reason_base_fell_tilt_count",
        "termination_reason_base_too_low_count",
        "ready_tilt_eligible_sample_count",
        "ready_tilt_rad_sum",
        "ready_base_speed_eligible_sample_count",
        "ready_base_speed_xy_mps_sum",
        "ready_station_offset_eligible_sample_count",
        "ready_station_offset_m_sum",
        "ready_foot_contact_eligible_sample_count",
        "ready_foot_contact_fraction_sum",
        "ready_foot_slip_eligible_sample_count",
        "ready_foot_slip_speed_mps_sum",
        "planner_initial_tts_sample_count",
        "planner_initial_tts_sub_0p5_count",
        "planner_initial_tts_exact_0p5_count",
        "planner_initial_tts_above_0p5_count",
        "planner_revision_attempt_count",
        "planner_revision_accepted_count",
        "planner_revision_rejected_count",
        "planner_revision_last_precontact_attempt_count",
        "planner_revision_last_precontact_accepted_count",
        "planner_revision_actor_visible_count",
        "planner_revision_last_precontact_actor_visible_count",
    }
    if not minimum.issubset(required):
        raise SuccessorQueueError(
            "exact behavior ledger is missing decision counters: "
            + ",".join(sorted(minimum - required))
        )

    pruning = _mapping(queue.get("pruning_contract"), "pruning_contract")
    if pruning.get("checkpoint_offsets_from_parent") != REQUIRED_OFFSETS:
        raise SuccessorQueueError("pruning checkpoints must remain +200/+500/+1000/+2000")
    if pruning.get("sparse_zero_without_positive_eligible_denominator_may_stop") is not False:
        raise SuccessorQueueError("sparse zero without eligible opportunities must never stop")
    if pruning.get("behavior_decision_requires_two_disjoint_complete_windows") is not True:
        raise SuccessorQueueError("behavior pruning requires two disjoint complete windows")
    for key in ("plus_200", "plus_500", "plus_1000", "plus_2000"):
        if not isinstance(pruning.get(key), dict):
            raise SuccessorQueueError(f"pruning_contract.{key} must be explicit")

    probe = _mapping(queue.get("full_scene_probe_contract"), "full_scene_probe_contract")
    expected_probe = {
        "representative_job_id": "taskrev_p1_core_high_noise",
        "generic_reviewed_result_required": True,
        "specialized_no_clobber_result": "task_revision_probe_result.json",
        "minimum_consecutive_exact_updates": 2,
        "initial_tts_all_four_components_must_be_observed": True,
        "initial_tts_sub_exact_and_above_0p5_must_all_be_observed": True,
        "planner_revision_attempt_accept_reject_identity_must_hold": True,
        "accepted_revision_must_be_observed": True,
        "accepted_last_precontact_revision_must_be_observed": True,
        "actor_visible_revision_must_be_observed": True,
        "actor_visible_last_precontact_revision_must_be_observed": True,
        "exact_behavior_ledger_must_be_complete": True,
        "generic_pass_without_specialized_pass_may_unlock_launch": False,
    }
    if probe != expected_probe:
        raise SuccessorQueueError("full-scene task-revision probe contract changed")
    receipt = _mapping(
        queue.get("behavior_decision_receipt_contract"),
        "behavior_decision_receipt_contract",
    )
    expected_receipt = {
        "directory": "behavior_milestones",
        "publish": "atomic_no_clobber",
        "exact_log_prefix_bound_by_inode_size_and_sha256": True,
        "checkpoint_receipt_required": True,
        "claim_binding_process_identity_required": True,
        "automatic_signal_authorized": False,
        "exact_stop_consumer": "reviewed_numeric_identity_v1",
        "manual_command": "exact-stop-behavior",
        "required_revalidation": [
            "job",
            "claim",
            "binding",
            "pid",
            "pgid",
            "starttime",
            "argv",
            "checkpoint_receipt",
            "behavior_receipt",
        ],
        "no_clobber_intent_before_signal": True,
        "exact_group_membership_required_before_term_or_kill": True,
    }
    if receipt != expected_receipt:
        raise SuccessorQueueError("behavior decision receipt contract changed")


def _validate_timestamp_pair(
    jobs: list[dict[str, Any]], overrides_by_id: Mapping[str, Mapping[str, str]]
) -> None:
    negatives = [job for job in jobs if job.get("timestamp_role") == "uncompensated_negative_control"]
    treatments = [job for job in jobs if job.get("timestamp_role") == "compensated_treatment"]
    if len(negatives) != 1 or len(treatments) != 1:
        raise SuccessorQueueError("queue must contain one explicit compensated/uncompensated matched pair")
    negative = negatives[0]
    treatment = treatments[0]
    if negative.get("matched_timestamp_job") != treatment["id"] or treatment.get(
        "matched_timestamp_job"
    ) != negative["id"]:
        raise SuccessorQueueError("timestamp matched-pair links are not reciprocal")
    if negative["warm_start"]["parent"] != treatment["warm_start"]["parent"]:
        raise SuccessorQueueError("timestamp matched pair must use the same immutable parent")
    left = dict(overrides_by_id[negative["id"]])
    right = dict(overrides_by_id[treatment["id"]])
    key = "task.racket.target_delay_tts_mode"
    left_mode = left.pop(key, None)
    right_mode = right.pop(key, None)
    if left_mode != "++task.racket.target_delay_tts_mode=uncompensated":
        raise SuccessorQueueError("timestamp negative control must remove elapsed-time compensation")
    if right_mode != "++task.racket.target_delay_tts_mode=source_timestamp_compensated":
        raise SuccessorQueueError("timestamp treatment must enable source timestamp compensation")
    if left != right:
        raise SuccessorQueueError("timestamp pair differs in more than elapsed-time compensation")
    delay_key = "task.racket.target_delay_steps"
    if (
        left.get(delay_key) != "task.racket.target_delay_steps=2"
        or right.get(delay_key) != "task.racket.target_delay_steps=2"
    ):
        raise SuccessorQueueError("timestamp matched pair must share target_delay_steps=2")
    for job in (negative, treatment):
        if _job_launch_authorized(job) or job.get("scientific_blocker") != TRANSPORT_BLOCKER:
            raise SuccessorQueueError(
                "timestamp delay pair must remain NO-LAUNCH until governor and actor "
                "consume one coupled transport tuple"
            )


def _ratio(counters: Mapping[str, int | float], numerator: str, denominator: str):
    denominator_value = int(counters.get(denominator, 0))
    if denominator_value <= 0:
        return None
    value = float(counters.get(numerator, 0)) / float(denominator_value)
    return value if math.isfinite(value) else None


def _derived_behavior(counters: Mapping[str, int | float]) -> dict[str, float | None]:
    """Recompute behavior values only after raw counters have been summed."""

    return {
        "swing_completion_rate": _ratio(
            counters, "swing_completion_count", "swing_outcome_count"
        ),
        "pre_strike_physical_fall_rate": _ratio(
            counters, "pre_strike_physical_fall_count", "swing_outcome_count"
        ),
        "post_strike_physical_fall_rate": _ratio(
            counters, "post_strike_physical_fall_count", "swing_outcome_count"
        ),
        "virtual_capture_per_strike": _ratio(
            counters, "virtual_capture_count", "strike_opportunity_count"
        ),
        "virtual_net_clear_per_capture": _ratio(
            counters, "virtual_net_clear_count", "virtual_capture_count"
        ),
        "virtual_landing_valid_per_capture": _ratio(
            counters, "virtual_landing_valid_count", "virtual_capture_count"
        ),
        "virtual_legal_return_per_strike": _ratio(
            counters, "virtual_legal_return_count", "strike_opportunity_count"
        ),
        "ready_tilt_rad_mean": _ratio(
            counters, "ready_tilt_rad_sum", "ready_tilt_eligible_sample_count"
        ),
        "ready_base_speed_xy_mps_mean": _ratio(
            counters,
            "ready_base_speed_xy_mps_sum",
            "ready_base_speed_eligible_sample_count",
        ),
        "ready_station_offset_m_mean": _ratio(
            counters,
            "ready_station_offset_m_sum",
            "ready_station_offset_eligible_sample_count",
        ),
        "ready_foot_contact_fraction_mean": _ratio(
            counters,
            "ready_foot_contact_fraction_sum",
            "ready_foot_contact_eligible_sample_count",
        ),
        "ready_foot_slip_speed_mps_mean": _ratio(
            counters,
            "ready_foot_slip_speed_mps_sum",
            "ready_foot_slip_eligible_sample_count",
        ),
    }


def _required_behavior_counters(queue: Mapping[str, Any]) -> set[str]:
    ledger = _mapping(
        queue.get("exact_behavior_ledger_contract"), "exact_behavior_ledger_contract"
    )
    counters = ledger.get("required_counters")
    if not isinstance(counters, list) or any(not isinstance(name, str) for name in counters):
        raise SuccessorQueueError("exact behavior required_counters must be a string list")
    return set(counters)


def _validate_counter_invariants(counters: Mapping[str, int | float], label: str) -> None:
    for name, value in counters.items():
        if name.endswith("_sum"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SuccessorQueueError(f"{label}.{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise SuccessorQueueError(f"{label}.{name} must be finite and non-negative")
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SuccessorQueueError(f"{label}.{name} must be a non-negative integer")

    outcome = int(counters["swing_outcome_count"])
    completion = int(counters["swing_completion_count"])
    physical = int(counters["physical_fall_count"])
    pre = int(counters["pre_strike_physical_fall_count"])
    post = int(counters["post_strike_physical_fall_count"])
    if not 0 <= completion <= outcome:
        raise SuccessorQueueError(f"{label} completion/outcome invariant failed")
    if pre + post != physical or physical > outcome:
        raise SuccessorQueueError(f"{label} physical-fall closeout invariant failed")

    strike = int(counters["strike_opportunity_count"])
    capture = int(counters["virtual_capture_count"])
    net = int(counters["virtual_net_clear_count"])
    landing = int(counters["virtual_landing_valid_count"])
    legal = int(counters["virtual_legal_return_count"])
    if not 0 <= legal <= landing <= capture <= strike or not 0 <= net <= capture:
        raise SuccessorQueueError(f"{label} sparse outcome denominator invariant failed")

    sample = int(counters["planner_initial_tts_sample_count"])
    components = sum(
        int(counters[f"planner_initial_tts_component_{index}_count"])
        for index in range(4)
    )
    strata = sum(
        int(counters[name])
        for name in (
            "planner_initial_tts_sub_0p5_count",
            "planner_initial_tts_exact_0p5_count",
            "planner_initial_tts_above_0p5_count",
        )
    )
    if sample != components or sample != strata:
        raise SuccessorQueueError(f"{label} initial-TTS mixture accounting invariant failed")
    # A producer that never registered one of these counters is itself a +200
    # mechanism-activation failure.  Keep the parser capable of carrying that
    # evidence to the registered two-window decision instead of turning it into
    # an ambiguous sparse-return failure.  The full-scene probe still requires
    # the complete counter set separately.
    missing_activation = REVISION_ACTIVATION_COUNTERS - set(counters)
    if not missing_activation:
        attempts = int(counters["planner_revision_attempt_count"])
        accepted = int(counters["planner_revision_accepted_count"])
        rejected = int(counters["planner_revision_rejected_count"])
        last_attempts = int(counters["planner_revision_last_precontact_attempt_count"])
        last_accepted = int(counters["planner_revision_last_precontact_accepted_count"])
        if attempts != accepted + rejected:
            raise SuccessorQueueError(f"{label} planner-revision accounting invariant failed")
        if not 0 <= last_accepted <= last_attempts <= attempts or last_accepted > accepted:
            raise SuccessorQueueError(f"{label} last-precontact revision invariant failed")
        actor_visible = int(counters["planner_revision_actor_visible_count"])
        last_actor_visible = int(
            counters["planner_revision_last_precontact_actor_visible_count"]
        )
        if not 0 <= actor_visible <= accepted or not 0 <= last_actor_visible <= last_accepted:
            raise SuccessorQueueError(f"{label} planner actor-visible invariant failed")
    contact_count = int(counters["ready_foot_contact_eligible_sample_count"])
    contact_sum = float(counters["ready_foot_contact_fraction_sum"])
    if contact_sum > float(contact_count) + 1.0e-9:
        raise SuccessorQueueError(f"{label} foot-contact fraction sum exceeds its denominator")


def parse_exact_behavior_log(
    raw: bytes, *, required_counters: set[str]
) -> dict[int, dict[str, Any]]:
    """Parse the append-only exact-event prefix and reject duplicates or provider drift."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SuccessorQueueError("behavior log prefix is not UTF-8") from exc
    records: dict[int, dict[str, Any]] = {}
    providers: set[str] = set()
    expected_keys = {
        "event",
        "schema_version",
        "ppo_update",
        "term",
        "counters",
        "derived",
        "window_aggregation",
    }
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith(EXACT_EVENT_PREFIX):
            continue
        try:
            record = json.loads(line[len(EXACT_EVENT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise SuccessorQueueError(
                f"exact behavior line {line_number} is malformed JSON"
            ) from exc
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise SuccessorQueueError(f"exact behavior line {line_number} schema changed")
        if (
            record.get("event") != "hope_exact_behavior_update"
            or record.get("schema_version") != 1
            or record.get("window_aggregation")
            != "sum_counters_then_recompute_derived"
        ):
            raise SuccessorQueueError(f"exact behavior line {line_number} contract changed")
        update = record.get("ppo_update")
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise SuccessorQueueError(f"exact behavior line {line_number} update is invalid")
        if update in records:
            raise SuccessorQueueError(f"duplicate exact behavior update {update}")
        provider = record.get("term")
        if not isinstance(provider, str) or not provider:
            raise SuccessorQueueError(f"exact behavior line {line_number} provider is invalid")
        providers.add(provider)
        counters = record.get("counters")
        if not isinstance(counters, dict):
            raise SuccessorQueueError(
                f"exact behavior update {update} counters must be a mapping"
            )
        missing = required_counters - set(counters)
        non_activation_missing = missing - REVISION_ACTIVATION_COUNTERS
        if non_activation_missing:
            raise SuccessorQueueError(
                "exact behavior update "
                f"{update} is missing counters: {sorted(non_activation_missing)}"
            )
        _validate_counter_invariants(counters, f"exact behavior update {update}")
        if record.get("derived") != _derived_behavior(counters):
            raise SuccessorQueueError(f"exact behavior update {update} derived values drifted")
        records[update] = record
    if not records:
        raise SuccessorQueueError("behavior log contains no exact behavior events")
    if providers != {"racket_target"}:
        raise SuccessorQueueError(
            f"exact behavior requires one racket_target provider, got {sorted(providers)}"
        )
    return records


def _sum_window(records: list[dict[str, Any]]) -> dict[str, int | float]:
    keys = set().union(*(record["counters"].keys() for record in records))
    return {
        key: math.fsum(float(record["counters"].get(key, 0)) for record in records)
        if key.endswith("_sum")
        else sum(int(record["counters"].get(key, 0)) for record in records)
        for key in keys
    }


def analyze_behavior_windows(
    records: Mapping[int, dict[str, Any]],
    *,
    milestone: int,
    required_counters: set[str],
    milestone_offset: int,
) -> dict[str, Any]:
    """Build two exact disjoint 100-update windows ending at one checkpoint."""

    first_ids = list(range(milestone - 199, milestone - 99))
    second_ids = list(range(milestone - 99, milestone + 1))
    if first_ids[0] < 0:
        raise SuccessorQueueError("milestone cannot contain two complete 100-update windows")
    required_ids = first_ids + second_ids
    missing = [update for update in required_ids if update not in records]
    if missing:
        raise SuccessorQueueError(
            "behavior log lacks the registered two-window checkpoint alignment: "
            f"missing={missing[:8]} count={len(missing)}"
        )
    windows = []
    for index, ids in enumerate((first_ids, second_ids), start=1):
        selected = [records[update] for update in ids]
        aggregate = _sum_window(selected)
        missing = required_counters - set(aggregate)
        if missing - REVISION_ACTIVATION_COUNTERS:
            raise SuccessorQueueError(
                f"behavior window {index} lost required counters: "
                + ",".join(sorted(missing - REVISION_ACTIVATION_COUNTERS))
            )
        _validate_counter_invariants(aggregate, f"behavior window {index}")
        windows.append(
            {
                "index": index,
                "first_update": ids[0],
                "last_update": ids[-1],
                "update_count": len(ids),
                "counters": aggregate,
                "derived": _derived_behavior(aggregate),
            }
        )

    reasons: list[str] = []
    activation_failures: list[str] = []
    if milestone_offset == 200:
        for window in windows:
            counters = window["counters"]
            for name in sorted(REVISION_ACTIVATION_COUNTERS):
                if name not in counters:
                    activation_failures.append(
                        f"window_{window['index']}:{name}:missing"
                    )
                elif int(counters[name]) <= 0:
                    activation_failures.append(f"window_{window['index']}:{name}:zero")

    if activation_failures:
        return {
            "schema_version": 1,
            "milestone": milestone,
            "milestone_offset_from_parent": milestone_offset,
            "window_alignment": "checkpoint_iteration_is_last_update_of_second_window",
            "provider": "racket_target",
            "windows": windows,
            "decision": "stop_revision_or_ledger_activation_absent",
            "decision_reasons": activation_failures,
            "sparse_zero_without_positive_eligible_denominator_may_stop": False,
            "sparse_outcome_used_for_activation_decision": False,
            "stop_execution": "manual_reviewed_exact_consumer_only",
        }
    stop = milestone_offset == 500
    for window in windows:
        derived = window["derived"]
        completion = derived["swing_completion_rate"]
        pre = derived["pre_strike_physical_fall_rate"]
        post = derived["post_strike_physical_fall_rate"]
        if completion is None or pre is None or post is None:
            stop = False
            reasons.append(f"window_{window['index']}_closeout_denominator_zero")
            continue
        if completion >= 0.40:
            stop = False
            reasons.append(f"window_{window['index']}_completion_not_below_0p40")
        if pre + post <= 0.10:
            stop = False
            reasons.append(f"window_{window['index']}_physical_fall_not_above_0p10")

    first = windows[0]["derived"]
    second = windows[1]["derived"]
    ready_missing = [name for name in READY_METRICS if first[name] is None or second[name] is None]
    if ready_missing:
        stop = False
        reasons.append("ready_denominator_zero:" + ",".join(sorted(ready_missing)))
    else:
        improvements = []
        for name, direction in READY_METRICS.items():
            before = float(first[name])
            after = float(second[name])
            # The frozen YAML says "improves no ready metric" and declares no
            # tolerance.  Any strict finite improvement therefore protects the
            # cell; hidden code-only tolerances would silently loosen the
            # preregistered stop rule.
            if direction == "minimize" and after < before:
                improvements.append(name)
            if direction == "maximize" and after > before:
                improvements.append(name)
        if improvements:
            stop = False
            reasons.append("ready_metric_improved:" + ",".join(sorted(improvements)))
    if milestone_offset != 500:
        reasons.append("milestone_requires_manual_pareto_or_terminal_review")
    decision = (
        "stop_clear_dense_collapse"
        if stop
        else "continue_training_no_automatic_stop"
    )
    return {
        "schema_version": 1,
        "milestone": milestone,
        "milestone_offset_from_parent": milestone_offset,
        "window_alignment": "checkpoint_iteration_is_last_update_of_second_window",
        "provider": "racket_target",
        "windows": windows,
        "decision": decision,
        "decision_reasons": reasons,
        "sparse_zero_without_positive_eligible_denominator_may_stop": False,
        "stop_execution": "manual_reviewed_exact_consumer_only",
    }


def _portfolio_metric_contract(
    queue: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return YAML-bound Pareto axes.

    A scalar metric entry means an exact (zero-tolerance) comparison.  A future
    queue may replace it with ``{name, tolerance}`` without changing the
    consumer.  This makes the tolerance an explicit property of the tracked
    YAML contract instead of an unrecorded command-line choice.
    """

    plus_1000 = _mapping(
        _mapping(queue.get("pruning_contract"), "pruning_contract").get("plus_1000"),
        "pruning_contract.plus_1000",
    )
    configured = _mapping(plus_1000.get("pareto_metrics"), "plus_1000.pareto_metrics")
    axes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for direction in ("maximize", "minimize"):
        values = configured.get(direction)
        if not isinstance(values, list) or not values:
            raise SuccessorQueueError(f"plus_1000.pareto_metrics.{direction} is empty")
        for value in values:
            if isinstance(value, str):
                name = value
                tolerance = 0.0
                source = "yaml_scalar_exact_zero"
            elif isinstance(value, dict) and set(value) == {"name", "tolerance"}:
                name = value["name"]
                tolerance = value["tolerance"]
                source = "yaml_explicit"
            else:
                raise SuccessorQueueError(
                    f"plus_1000.pareto_metrics.{direction} entry is malformed"
                )
            if not isinstance(name, str) or not name or name in seen:
                raise SuccessorQueueError("Pareto metric names must be unique strings")
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or float(tolerance) < 0.0
            ):
                raise SuccessorQueueError(f"Pareto tolerance for {name} is invalid")
            seen.add(name)
            axes.append(
                {
                    "name": name,
                    "direction": direction,
                    "tolerance": float(tolerance),
                    "tolerance_source": source,
                }
            )
    return axes


def _portfolio_metric_value(derived: Mapping[str, Any], name: str) -> float | None:
    aliases = {"swing_closeout_completion_rate": "swing_completion_rate"}
    value = derived.get(aliases.get(name, name))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuccessorQueueError(f"portfolio metric {name} is not numeric/null")
    result = float(value)
    if not math.isfinite(result):
        raise SuccessorQueueError(f"portfolio metric {name} is non-finite")
    return result


def _job_timing_roles(job: Mapping[str, Any]) -> dict[str, bool]:
    revision = _validate_revision(job, _compiled_overrides(job))
    components = revision["initial_tts_mixture"]["components"]
    weights = {str(row["name"]): float(row["weight"]) for row in components}
    maximum = max(weights.values())
    return {
        "fast_curriculum": weights["fast_deploy"] == maximum,
        "broad_curriculum": weights["broad_arrival"] == maximum,
    }


def _receipt_last_window(content: Mapping[str, Any]) -> dict[str, Any]:
    behavior = _mapping(content.get("behavior"), "portfolio behavior analysis")
    windows = behavior.get("windows")
    if not isinstance(windows, list) or len(windows) != 2:
        raise SuccessorQueueError("portfolio behavior receipt must bind two windows")
    window = _mapping(windows[1], "portfolio trailing behavior window")
    if window.get("update_count") != 100:
        raise SuccessorQueueError("portfolio trailing behavior window must contain 100 updates")
    derived = _mapping(window.get("derived"), "portfolio trailing derived metrics")
    counters = _mapping(window.get("counters"), "portfolio trailing raw counters")
    return {"behavior": behavior, "derived": derived, "counters": counters}


def _dominates(
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
    axes: list[dict[str, Any]],
) -> bool:
    """Tolerance-aware dominance; unknown metrics are protected, never imputed."""

    strictly_better = False
    for axis in axes:
        name = axis["name"]
        a = left[name]
        b = right[name]
        if a is None or b is None:
            return False
        tolerance = axis["tolerance"]
        if axis["direction"] == "maximize":
            if a < b - tolerance:
                return False
            strictly_better = strictly_better or a > b + tolerance
        else:
            if a > b + tolerance:
                return False
            strictly_better = strictly_better or a < b - tolerance
    return strictly_better


def analyze_parent_portfolio(
    queue: Mapping[str, Any],
    *,
    parent: str,
    milestone_offset: int,
    behavior_contents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select stop-authorized cells without destroying parent-level coverage."""

    if milestone_offset not in PORTFOLIO_OFFSETS:
        raise SuccessorQueueError("portfolio offset must be one of 200, 500, or 1000")
    parent_jobs = {
        job["id"]: job
        for job in queue["jobs"]
        if _job_launch_authorized(job) and job["warm_start"]["parent"] == parent
    }
    if not parent_jobs:
        raise SuccessorQueueError(f"unknown or scientifically blocked parent: {parent}")
    unknown = set(behavior_contents) - set(parent_jobs)
    if unknown:
        raise SuccessorQueueError(f"portfolio contains other-parent jobs: {sorted(unknown)}")

    axes = _portfolio_metric_contract(queue)
    rows: dict[str, dict[str, Any]] = {}
    for job_id, raw_content in behavior_contents.items():
        content = _mapping(raw_content, f"portfolio behavior content {job_id}")
        if content.get("job_id") != job_id:
            raise SuccessorQueueError("portfolio behavior job identity changed")
        trailing = _receipt_last_window(content)
        behavior = trailing["behavior"]
        if behavior.get("milestone_offset_from_parent") != milestone_offset:
            raise SuccessorQueueError("portfolio behavior offset differs from requested offset")
        metric_values = {
            axis["name"]: _portfolio_metric_value(trailing["derived"], axis["name"])
            for axis in axes
        }
        counters = trailing["counters"]
        roles = _job_timing_roles(parent_jobs[job_id])
        roles["exact_0p5_exposed"] = int(
            counters.get("planner_initial_tts_exact_0p5_count", 0)
        ) > 0
        rows[job_id] = {
            "job_id": job_id,
            "single_job_decision": behavior.get("decision"),
            "metrics": metric_values,
            "roles": roles,
        }

    included = sorted(rows)
    absent = sorted(set(parent_jobs) - set(rows))
    minimum = int(
        _mapping(
            _mapping(queue["pruning_contract"], "pruning_contract").get("plus_1000"),
            "plus_1000",
        ).get("minimum_survivors_per_parent", 2)
    )
    if minimum < 2:
        raise SuccessorQueueError("minimum_survivors_per_parent must be at least two")

    dominators = {
        job_id: sorted(
            other
            for other in included
            if other != job_id
            and _dominates(rows[other]["metrics"], rows[job_id]["metrics"], axes)
        )
        for job_id in included
    }
    if milestone_offset == 200:
        proposed = {
            job_id
            for job_id in included
            if rows[job_id]["single_job_decision"]
            == "stop_revision_or_ledger_activation_absent"
        }
    elif milestone_offset == 500:
        proposed = {
            job_id
            for job_id in included
            if rows[job_id]["single_job_decision"] == "stop_clear_dense_collapse"
        }
    else:
        proposed = {job_id for job_id in included if dominators[job_id]}

    survivors = set(included) - proposed

    def candidate_order(job_id: str) -> tuple[int, str]:
        return (len(dominators[job_id]), job_id)

    for job_id in sorted(proposed, key=candidate_order):
        if len(survivors) >= minimum:
            break
        survivors.add(job_id)

    def preserve_role(predicate) -> bool:
        if any(predicate(rows[job_id]["roles"]) for job_id in survivors):
            return True
        choices = [job_id for job_id in included if predicate(rows[job_id]["roles"])]
        if not choices:
            return False
        survivors.add(min(choices, key=candidate_order))
        return True

    exact_0p5 = preserve_role(lambda role: role["exact_0p5_exposed"])
    broad = preserve_role(lambda role: role["broad_curriculum"])
    coverage_satisfied = len(survivors) >= minimum and exact_0p5 and broad
    eliminated = sorted(set(included) - survivors) if coverage_satisfied else []
    survivors = set(included) - set(eliminated)
    decisions = {
        job_id: (
            "eliminate"
            if job_id in eliminated
            else "retain_portfolio_guard"
        )
        for job_id in included
    }
    return {
        "schema_version": 1,
        "parent": parent,
        "milestone_offset_from_parent": milestone_offset,
        "included_attested_job_ids": included,
        "unattested_job_ids": absent,
        "metric_contract": axes,
        "rows": [
            {**rows[job_id], "dominated_by": dominators[job_id]}
            for job_id in included
        ],
        "decisions": decisions,
        "eliminated_job_ids": eliminated,
        "survivor_job_ids": sorted(survivors),
        "minimum_survivors_per_parent": minimum,
        "minimum_survivors_satisfied": len(survivors) >= minimum,
        "exact_0p5_coverage_satisfied": exact_0p5,
        "broad_arrival_coverage_satisfied": broad,
        "portfolio_stop_authorized": bool(eliminated) and coverage_satisfied,
        "sparse_return_zero_used_for_elimination": False,
        "unknown_metric_imputed": False,
    }


def _stable_append_prefix(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read and re-read one immutable prefix while allowing concurrent appends."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise SuccessorQueueError(f"behavior log must be a non-empty regular file: {path}")
        size = before.st_size
        first = os.pread(fd, size, 0)
        second = os.pread(fd, size, 0)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    outside = os.lstat(path)
    if len(first) != size or first != second:
        raise SuccessorQueueError("behavior log prefix changed during stable read")
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or (
        before.st_dev,
        before.st_ino,
    ) != (outside.st_dev, outside.st_ino):
        raise SuccessorQueueError("behavior log file identity changed during read")
    return first, {
        "path": str(path),
        "prefix_size": size,
        "prefix_sha256": hashlib.sha256(first).hexdigest(),
        "file_size_after_read": after.st_size,
    }


def _load_runtime(source_checkout: str):
    path = (
        Path(source_checkout)
        / "hope_training/whole_body_tracking/scripts/lean_queue_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("task_revision_lean_runtime", path)
    if spec is None or spec.loader is None:
        raise SuccessorQueueError(f"cannot load exact runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _load_exact_process_group(source_checkout: str):
    path = (
        Path(source_checkout)
        / "hope_training/whole_body_tracking/scripts/exact_process_group.py"
    )
    spec = importlib.util.spec_from_file_location("task_revision_exact_process_group", path)
    if spec is None or spec.loader is None:
        raise SuccessorQueueError(f"cannot load exact process-group helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _verify_clean_source(source: Mapping[str, Any]) -> None:
    checkout = Path(str(source["checkout"]))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if head != source["commit"] or dirty:
        raise SuccessorQueueError("behavior consumer source is not clean at the queue commit")


def _job(queue: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    for job in queue["jobs"]:
        if job["id"] == job_id:
            return job
    raise SuccessorQueueError(f"unknown queue job: {job_id}")


def _validate_envelope(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(value) != {"schema_version", "content", "content_sha256"}:
        raise SuccessorQueueError(f"{label} envelope changed")
    content = value.get("content")
    if not isinstance(content, dict):
        raise SuccessorQueueError(f"{label} content must be a mapping")
    if continuation._canonical_sha256(content) != value.get("content_sha256"):
        raise SuccessorQueueError(f"{label} canonical digest mismatch")
    return content


def _milestone_receipt(
    runtime,
    runtime_path: Path,
    *,
    binding_path: Path,
    run_dir: Path,
    milestone: int,
    expected_claim_sha: str,
    expected_job_id: str,
    create_if_missing: bool,
) -> tuple[dict[str, Any], bytes, Path]:
    receipt_path = run_dir / "milestones" / f"model_{milestone}.json"
    if not receipt_path.exists():
        if not create_if_missing:
            raise SuccessorQueueError(
                "checkpoint milestone receipt is absent; inspect is read-only and cannot create it"
            )
        runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
        runtime.attest_milestone(
            binding_path,
            milestone,
            expected_claim_content_sha256=expected_claim_sha,
            expected_job_id=expected_job_id,
            expected_runtime_sha256=runtime_sha,
        )
    receipt, raw = runtime._read_regular_json(receipt_path, "milestone receipt")
    content = _validate_envelope(receipt, "milestone receipt")
    if (
        content.get("job_id") != expected_job_id
        or content.get("milestone") != milestone
        or content.get("claim_content_sha256") != expected_claim_sha
        or content.get("binding_path") != str(binding_path)
    ):
        raise SuccessorQueueError("milestone receipt differs from the registered job")
    return receipt, raw, receipt_path


def inspect_or_attest_behavior_local(
    queue: Mapping[str, Any], *, job_id: str, milestone: int, write_receipt: bool
) -> dict[str, Any]:
    """Inspect one bound log; optionally publish one no-clobber behavior decision receipt."""

    job = _job(queue, job_id)
    _require_launchable_job(job)
    slot = continuation._slots(queue)[job["resource"]["required_slot"]]
    absolute = continuation._absolute_schedule(
        job, continuation._parent_records_from_job_context(job)
    )
    if milestone not in absolute["milestones"]:
        raise SuccessorQueueError("behavior consumer milestone is not registered")
    milestone_offset = milestone - absolute["parent_iteration"]
    source = job["source"]
    _verify_clean_source(source)
    runtime, runtime_path = _load_runtime(source["checkout"])
    claim_spec = continuation._attestor_claim_spec(queue, job, slot)
    binding_path = Path(claim_spec["binding_path"])
    binding, binding_content, _claim, _claim_content = runtime._load_binding(binding_path)
    if (
        binding_content.get("job_id") != job_id
        or binding_content.get("claim_content_sha256") != claim_spec["content_sha256"]
        or binding_content.get("run_dir") != job["run_dir"]
        or binding_content.get("pod") != slot.pod
        or binding_content.get("gpu") != slot.gpu
    ):
        raise SuccessorQueueError("actual binding differs from the registered successor cell")
    process_state = runtime._verify_bound_process(
        binding_content, proc_root=Path("/proc"), getpgid=os.getpgid
    )
    run_dir = Path(job["run_dir"])
    milestone_receipt, milestone_raw, milestone_path = _milestone_receipt(
        runtime,
        runtime_path,
        binding_path=binding_path,
        run_dir=run_dir,
        milestone=milestone,
        expected_claim_sha=claim_spec["content_sha256"],
        expected_job_id=job_id,
        create_if_missing=write_receipt,
    )
    raw_log, log_evidence = _stable_append_prefix(run_dir / "run.log")
    records = parse_exact_behavior_log(
        raw_log, required_counters=_required_behavior_counters(queue)
    )
    analysis = analyze_behavior_windows(
        records,
        milestone=milestone,
        required_counters=_required_behavior_counters(queue),
        milestone_offset=milestone_offset,
    )
    receipt_content = {
        "schema_version": 1,
        "job_id": job_id,
        "formal_evidence_eligible": False,
        "binding_path": str(binding_path),
        "binding_content_sha256": binding["content_sha256"],
        "claim_content_sha256": claim_spec["content_sha256"],
        "process_identity": dict(binding_content["process"]),
        "process_state_at_behavior_attestation": process_state,
        "milestone_receipt": {
            "path": str(milestone_path),
            "file_sha256": hashlib.sha256(milestone_raw).hexdigest(),
            "content_sha256": milestone_receipt["content_sha256"],
        },
        "log_prefix": log_evidence,
        "behavior": analysis,
        "consumer_source": _consumer_source_evidence(),
        "automatic_retry": False,
    }
    receipt = {
        "schema_version": 1,
        "content": receipt_content,
        "content_sha256": continuation._canonical_sha256(receipt_content),
    }
    receipt_path = run_dir / "behavior_milestones" / f"model_{milestone}.json"
    if write_receipt:
        runtime._atomic_publish_json(receipt_path, receipt, "behavior milestone receipt")
    return {
        "mode": "attest-behavior" if write_receipt else "inspect-behavior",
        "read_only": not write_receipt,
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def _portfolio_receipt_path(
    queue: Mapping[str, Any], *, parent: str, milestone_offset: int
) -> Path:
    root = _mapping(queue.get("namespace_contract"), "namespace_contract").get("root")
    if not isinstance(root, str) or not root.startswith("/workspace/"):
        raise SuccessorQueueError("portfolio namespace root must be absolute /workspace")
    if not continuation.lean.SAFE_ID.fullmatch(parent):
        raise SuccessorQueueError("portfolio parent name is unsafe")
    return (
        Path(root)
        / "portfolio_decisions"
        / parent
        / f"offset_{milestone_offset}.json"
    )


def inspect_or_attest_parent_portfolio_local(
    queue: Mapping[str, Any],
    *,
    parent: str,
    milestone_offset: int,
    write_receipt: bool,
) -> dict[str, Any]:
    """Bind every same-parent behavior receipt currently attested on one Pod."""

    if milestone_offset not in PORTFOLIO_OFFSETS:
        raise SuccessorQueueError("portfolio offset must be one of 200, 500, or 1000")
    jobs = [
        job
        for job in queue["jobs"]
        if _job_launch_authorized(job) and job["warm_start"]["parent"] == parent
    ]
    if not jobs:
        raise SuccessorQueueError(f"no launchable jobs use parent {parent}")
    pods = {job["resource"]["required_slot"].split("/", 1)[0] for job in jobs}
    sources = {(job["source"]["checkout"], job["source"]["commit"]) for job in jobs}
    if len(pods) != 1 or len(sources) != 1:
        raise SuccessorQueueError("same-parent portfolio must remain on one Pod and source")
    _verify_clean_source(jobs[0]["source"])
    runtime, _runtime_path = _load_runtime(jobs[0]["source"]["checkout"])

    contents: dict[str, dict[str, Any]] = {}
    receipt_bindings: list[dict[str, Any]] = []
    for job in jobs:
        absolute = continuation._absolute_schedule(
            job, continuation._parent_records_from_job_context(job)
        )
        milestone = absolute["parent_iteration"] + milestone_offset
        if milestone not in absolute["milestones"]:
            raise SuccessorQueueError("portfolio offset is not registered for every parent cell")
        path = Path(job["run_dir"]) / "behavior_milestones" / f"model_{milestone}.json"
        if not path.exists():
            continue
        receipt, raw = runtime._read_regular_json(path, "portfolio behavior receipt")
        content = _validate_envelope(receipt, "portfolio behavior receipt")
        behavior = _mapping(content.get("behavior"), "portfolio behavior analysis")
        if (
            content.get("job_id") != job["id"]
            or behavior.get("milestone") != milestone
            or behavior.get("milestone_offset_from_parent") != milestone_offset
        ):
            raise SuccessorQueueError("portfolio behavior receipt identity changed")
        contents[job["id"]] = content
        receipt_bindings.append(
            {
                "job_id": job["id"],
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": receipt["content_sha256"],
            }
        )
    analysis = analyze_parent_portfolio(
        queue,
        parent=parent,
        milestone_offset=milestone_offset,
        behavior_contents=contents,
    )
    pruning_contract = _mapping(queue.get("pruning_contract"), "pruning_contract")
    content = {
        "schema_version": 1,
        "parent": parent,
        "pod": next(iter(pods)),
        "milestone_offset_from_parent": milestone_offset,
        "behavior_receipts": sorted(receipt_bindings, key=lambda row: row["job_id"]),
        "pruning_contract_sha256": continuation._canonical_sha256(pruning_contract),
        "analysis": analysis,
        "consumer_source": _consumer_source_evidence(),
        "automatic_signal_authorized": False,
        "automatic_retry": False,
    }
    receipt = {
        "schema_version": 1,
        "content": content,
        "content_sha256": continuation._canonical_sha256(content),
    }
    path = _portfolio_receipt_path(
        queue, parent=parent, milestone_offset=milestone_offset
    )
    if write_receipt:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not stat.S_ISDIR(path.parent.lstat().st_mode):
            raise SuccessorQueueError("portfolio decision parent is not a real directory")
        runtime._atomic_publish_json(path, receipt, "parent portfolio decision receipt")
    return {
        "mode": "attest-portfolio" if write_receipt else "inspect-portfolio",
        "read_only": not write_receipt,
        "receipt_path": str(path),
        "receipt": receipt,
    }


def run_pruning_cycle_local(
    queue: Mapping[str, Any],
    *,
    pod: str,
    milestone_offset: int,
    write_receipts: bool,
) -> dict[str, Any]:
    """In one Pod-local process, attest every ready checkpoint then each parent.

    This function never signals.  A live bound cell that has not reached the
    checkpoint keeps its entire parent portfolio waiting, while an exited cell
    with no checkpoint is explicitly recorded as an infrastructure exclusion.
    """

    if milestone_offset not in PORTFOLIO_OFFSETS:
        raise SuccessorQueueError("pruning-cycle offset must be 200, 500, or 1000")
    jobs = [
        job
        for job in queue["jobs"]
        if _job_launch_authorized(job)
        and job["resource"]["required_slot"].split("/", 1)[0] == pod
    ]
    if not jobs:
        raise SuccessorQueueError(f"no launchable jobs are assigned to {pod}")
    _verify_clean_source(jobs[0]["source"])
    runtime, _runtime_path = _load_runtime(jobs[0]["source"]["checkout"])
    rows: list[dict[str, Any]] = []
    parent_waiting: dict[str, bool] = {}
    for job in jobs:
        parent = job["warm_start"]["parent"]
        parent_waiting.setdefault(parent, False)
        slot = continuation._slots(queue)[job["resource"]["required_slot"]]
        claim_spec = continuation._attestor_claim_spec(queue, job, slot)
        binding_path = Path(claim_spec["binding_path"])
        if not binding_path.exists():
            rows.append(
                {
                    "job_id": job["id"],
                    "parent": parent,
                    "state": "unbound_never_launched_excluded",
                }
            )
            continue
        binding, binding_content, _claim, _claim_content = runtime._load_binding(
            binding_path
        )
        if (
            binding_content.get("job_id") != job["id"]
            or binding_content.get("claim_content_sha256")
            != claim_spec["content_sha256"]
            or binding_content.get("pod") != pod
            or binding_content.get("run_dir") != job["run_dir"]
            or binding.get("content_sha256") is None
        ):
            raise SuccessorQueueError("pruning-cycle binding differs from registered cell")
        process_state = runtime._verify_bound_process(
            binding_content, proc_root=Path("/proc"), getpgid=os.getpgid
        )
        absolute = continuation._absolute_schedule(
            job, continuation._parent_records_from_job_context(job)
        )
        milestone = absolute["parent_iteration"] + milestone_offset
        checkpoint = Path(str(binding_content["rsl_log_dir"])) / f"model_{milestone}.pt"
        checkpoint_ready = False
        try:
            info = checkpoint.lstat()
            checkpoint_ready = stat.S_ISREG(info.st_mode) and info.st_size > 0
            if not checkpoint_ready:
                raise SuccessorQueueError(
                    f"registered checkpoint is not a non-empty regular file: {checkpoint}"
                )
        except FileNotFoundError:
            checkpoint_ready = False
        behavior_path = (
            Path(job["run_dir"])
            / "behavior_milestones"
            / f"model_{milestone}.json"
        )
        if behavior_path.exists():
            state = "behavior_receipt_present"
        elif not checkpoint_ready:
            if process_state == "live":
                state = "waiting_for_checkpoint"
                parent_waiting[parent] = True
            else:
                state = "exited_before_checkpoint_excluded"
        elif not write_receipts:
            state = "checkpoint_ready_behavior_receipt_absent"
            parent_waiting[parent] = True
        else:
            inspect_or_attest_behavior_local(
                queue,
                job_id=job["id"],
                milestone=milestone,
                write_receipt=True,
            )
            state = "behavior_attested_now"
        rows.append(
            {
                "job_id": job["id"],
                "parent": parent,
                "milestone": milestone,
                "process_state": process_state,
                "checkpoint_ready": checkpoint_ready,
                "behavior_receipt_path": str(behavior_path),
                "state": state,
            }
        )

    portfolios: list[dict[str, Any]] = []
    for parent in sorted(parent_waiting):
        if parent_waiting[parent]:
            portfolios.append({"parent": parent, "state": "waiting_for_all_live_cells"})
            continue
        path = _portfolio_receipt_path(
            queue, parent=parent, milestone_offset=milestone_offset
        )
        if path.exists():
            receipt, _raw = runtime._read_regular_json(
                path, "existing parent portfolio decision receipt"
            )
            content = _validate_envelope(
                receipt, "existing parent portfolio decision receipt"
            )
            portfolios.append(
                {
                    "parent": parent,
                    "state": "portfolio_receipt_present",
                    "receipt_path": str(path),
                    "content_sha256": receipt["content_sha256"],
                    "eliminated_job_ids": _mapping(
                        content.get("analysis"), "existing portfolio analysis"
                    ).get("eliminated_job_ids", []),
                }
            )
            continue
        inspected = inspect_or_attest_parent_portfolio_local(
            queue,
            parent=parent,
            milestone_offset=milestone_offset,
            write_receipt=False,
        )
        analysis = _mapping(
            _validate_envelope(inspected["receipt"], "inspected portfolio").get(
                "analysis"
            ),
            "inspected portfolio analysis",
        )
        if not write_receipts or analysis.get("portfolio_stop_authorized") is not True:
            portfolios.append(
                {
                    "parent": parent,
                    "state": "no_elimination_receipt_not_published",
                    "analysis": analysis,
                }
            )
            continue
        attested = inspect_or_attest_parent_portfolio_local(
            queue,
            parent=parent,
            milestone_offset=milestone_offset,
            write_receipt=True,
        )
        portfolios.append(
            {
                "parent": parent,
                "state": "portfolio_attested_now",
                "receipt_path": attested["receipt_path"],
                "analysis": _validate_envelope(
                    attested["receipt"], "attested portfolio"
                )["analysis"],
            }
        )
    return {
        "mode": "attest-pruning-cycle" if write_receipts else "inspect-pruning-cycle",
        "pod": pod,
        "milestone_offset_from_parent": milestone_offset,
        "jobs": rows,
        "portfolios": portfolios,
        "ssh_signal_count": 0,
        "automatic_stop_authorized": False,
    }


def _validated_stop_inputs(
    queue: Mapping[str, Any], *, job_id: str, milestone: int
) -> dict[str, Any]:
    """Revalidate every immutable input required before an exact stop."""

    job = _job(queue, job_id)
    _require_launchable_job(job)
    slot = continuation._slots(queue)[job["resource"]["required_slot"]]
    absolute = continuation._absolute_schedule(
        job, continuation._parent_records_from_job_context(job)
    )
    if milestone not in absolute["milestones"]:
        raise SuccessorQueueError("exact stop milestone is not registered")
    milestone_offset = milestone - absolute["parent_iteration"]
    if milestone_offset not in PORTFOLIO_OFFSETS:
        raise SuccessorQueueError("exact stop requires a registered portfolio offset")
    _verify_clean_source(job["source"])
    runtime, runtime_path = _load_runtime(job["source"]["checkout"])
    claim_spec = continuation._attestor_claim_spec(queue, job, slot)
    binding_path = Path(claim_spec["binding_path"])
    binding, binding_content, claim, _claim_content = runtime._load_binding(binding_path)
    if (
        binding_content.get("job_id") != job_id
        or binding_content.get("claim_content_sha256") != claim_spec["content_sha256"]
        or binding_content.get("run_dir") != job["run_dir"]
        or binding_content.get("pod") != slot.pod
        or binding_content.get("gpu") != slot.gpu
    ):
        raise SuccessorQueueError("exact-stop binding differs from registered cell")
    process_state = runtime._verify_bound_process(
        binding_content, proc_root=Path("/proc"), getpgid=os.getpgid
    )
    run_dir = Path(job["run_dir"])
    behavior_path = run_dir / "behavior_milestones" / f"model_{milestone}.json"
    behavior, behavior_raw = runtime._read_regular_json(
        behavior_path, "behavior decision receipt"
    )
    behavior_content = _validate_envelope(behavior, "behavior decision receipt")
    behavior_analysis = _mapping(
        behavior_content.get("behavior"), "behavior decision analysis"
    )
    allowed_single_decision = {
        200: "stop_revision_or_ledger_activation_absent",
        500: "stop_clear_dense_collapse",
        # +1000 is deliberately a portfolio decision; the single-cell receipt
        # must remain neutral rather than inventing a scalar winner score.
        1000: "continue_training_no_automatic_stop",
    }[milestone_offset]
    if (
        behavior_content.get("job_id") != job_id
        or behavior_content.get("binding_content_sha256") != binding["content_sha256"]
        or behavior_content.get("claim_content_sha256") != claim_spec["content_sha256"]
        or behavior_content.get("process_identity") != binding_content.get("process")
        or behavior_analysis.get("milestone") != milestone
        or behavior_analysis.get("decision") != allowed_single_decision
        or behavior_analysis.get("stop_execution")
        != "manual_reviewed_exact_consumer_only"
    ):
        raise SuccessorQueueError("behavior receipt does not authorize this exact stop")

    parent = job["warm_start"]["parent"]
    portfolio_path = _portfolio_receipt_path(
        queue, parent=parent, milestone_offset=milestone_offset
    )
    portfolio, portfolio_raw = runtime._read_regular_json(
        portfolio_path, "parent portfolio decision receipt"
    )
    portfolio_content = _validate_envelope(
        portfolio, "parent portfolio decision receipt"
    )
    portfolio_analysis = _mapping(
        portfolio_content.get("analysis"), "parent portfolio analysis"
    )
    behavior_bindings = portfolio_content.get("behavior_receipts")
    if not isinstance(behavior_bindings, list):
        raise SuccessorQueueError("parent portfolio behavior bindings are malformed")
    bound_behavior = [
        row
        for row in behavior_bindings
        if isinstance(row, dict) and row.get("job_id") == job_id
    ]
    if (
        portfolio_content.get("parent") != parent
        or portfolio_content.get("pod") != slot.pod
        or portfolio_content.get("milestone_offset_from_parent") != milestone_offset
        or portfolio_content.get("pruning_contract_sha256")
        != continuation._canonical_sha256(
            _mapping(queue.get("pruning_contract"), "pruning_contract")
        )
        or portfolio_analysis.get("parent") != parent
        or portfolio_analysis.get("milestone_offset_from_parent") != milestone_offset
        or portfolio_analysis.get("portfolio_stop_authorized") is not True
        or portfolio_analysis.get("minimum_survivors_satisfied") is not True
        or portfolio_analysis.get("exact_0p5_coverage_satisfied") is not True
        or portfolio_analysis.get("broad_arrival_coverage_satisfied") is not True
        or job_id not in portfolio_analysis.get("eliminated_job_ids", [])
        or len(bound_behavior) != 1
        or bound_behavior[0].get("path") != str(behavior_path)
        or bound_behavior[0].get("file_sha256")
        != hashlib.sha256(behavior_raw).hexdigest()
        or bound_behavior[0].get("content_sha256") != behavior["content_sha256"]
    ):
        raise SuccessorQueueError(
            "parent portfolio receipt does not authorize this exact elimination"
        )
    milestone_info = _mapping(
        behavior_content.get("milestone_receipt"), "bound checkpoint receipt"
    )
    milestone_path = Path(str(milestone_info.get("path")))
    milestone_receipt, milestone_raw = runtime._read_regular_json(
        milestone_path, "bound checkpoint receipt"
    )
    milestone_content = _validate_envelope(
        milestone_receipt, "bound checkpoint receipt"
    )
    if (
        hashlib.sha256(milestone_raw).hexdigest() != milestone_info.get("file_sha256")
        or milestone_receipt.get("content_sha256")
        != milestone_info.get("content_sha256")
        or milestone_content.get("job_id") != job_id
        or milestone_content.get("milestone") != milestone
        or milestone_content.get("claim_content_sha256") != claim_spec["content_sha256"]
        or milestone_content.get("binding_path") != str(binding_path)
    ):
        raise SuccessorQueueError("checkpoint receipt changed after behavior attestation")
    return {
        "job": job,
        "slot": slot,
        "runtime": runtime,
        "runtime_path": runtime_path,
        "binding_path": binding_path,
        "binding": binding,
        "binding_content": binding_content,
        "claim": claim,
        "claim_spec": claim_spec,
        "process_state": process_state,
        "behavior_path": behavior_path,
        "behavior": behavior,
        "behavior_raw": behavior_raw,
        "portfolio_path": portfolio_path,
        "portfolio": portfolio,
        "portfolio_raw": portfolio_raw,
        "milestone_path": milestone_path,
        "milestone_receipt": milestone_receipt,
        "milestone_raw": milestone_raw,
    }


def _require_bound_leader_evidence(
    leader_evidence: Mapping[str, Any], process: Mapping[str, Any]
) -> None:
    """Require the post-intent proc read to equal the immutable binding."""

    leader = _mapping(leader_evidence.get("leader"), "exact-stop leader evidence")
    if (
        leader.get("pid") != process.get("pid")
        or leader.get("pgid") != process.get("pgid")
        or leader.get("starttime_ticks") != process.get("starttime_ticks")
    ):
        raise SuccessorQueueError(
            "exact-stop leader evidence differs from immutable binding"
        )


def exact_stop_behavior_local(
    queue: Mapping[str, Any], *, job_id: str, milestone: int
) -> dict[str, Any]:
    """Consume one stop decision and signal only its twice-bound numeric process group."""

    evidence = _validated_stop_inputs(queue, job_id=job_id, milestone=milestone)
    runtime = evidence["runtime"]
    binding_content = evidence["binding_content"]
    process = _mapping(binding_content.get("process"), "bound process")
    pid = process.get("pid")
    pgid = process.get("pgid")
    starttime = process.get("starttime_ticks")
    argv = process.get("argv")
    if (
        type(pid) is not int
        or pid < 1
        or pgid != pid
        or type(starttime) is not int
        or starttime < 1
        or not isinstance(argv, list)
        or argv != binding_content.get("training_argv")
    ):
        raise SuccessorQueueError("bound stop PID/PGID/starttime/argv is malformed")
    run_dir = Path(evidence["job"]["run_dir"])
    control = run_dir / "behavior_stops"
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not stat.S_ISDIR(control.lstat().st_mode):
        raise SuccessorQueueError("behavior stop control path is not a real directory")
    prefix = control / f"model_{milestone}"
    intent_path = Path(str(prefix) + ".intent.json")
    leader_path = Path(str(prefix) + ".leader.json")
    term_path = Path(str(prefix) + ".term.json")
    kill_path = Path(str(prefix) + ".kill.json")
    receipt_path = Path(str(prefix) + ".receipt.json")
    if any(path.exists() for path in (intent_path, leader_path, term_path, kill_path, receipt_path)):
        raise SuccessorQueueError("exact-stop transaction has already been consumed or attempted")
    intent_content = {
        "schema_version": 1,
        "job_id": job_id,
        "milestone": milestone,
        "binding_path": str(evidence["binding_path"]),
        "binding_content_sha256": evidence["binding"]["content_sha256"],
        "claim_content_sha256": evidence["claim_spec"]["content_sha256"],
        "behavior_receipt_path": str(evidence["behavior_path"]),
        "behavior_receipt_file_sha256": hashlib.sha256(
            evidence["behavior_raw"]
        ).hexdigest(),
        "behavior_receipt_content_sha256": evidence["behavior"]["content_sha256"],
        "portfolio_receipt_path": str(evidence["portfolio_path"]),
        "portfolio_receipt_file_sha256": hashlib.sha256(
            evidence["portfolio_raw"]
        ).hexdigest(),
        "portfolio_receipt_content_sha256": evidence["portfolio"]["content_sha256"],
        "checkpoint_receipt_path": str(evidence["milestone_path"]),
        "checkpoint_receipt_file_sha256": hashlib.sha256(
            evidence["milestone_raw"]
        ).hexdigest(),
        "process": dict(process),
        "signal_policy": "exact_group_TERM_then_bound_residual_KILL",
        "automatic_invocation": False,
    }
    intent = {
        "schema_version": 1,
        "content": intent_content,
        "content_sha256": continuation._canonical_sha256(intent_content),
    }
    runtime._atomic_publish_json(intent_path, intent, "exact-stop intent")
    signals: list[str] = []
    exact_group, exact_group_path = _load_exact_process_group(
        evidence["job"]["source"]["checkout"]
    )
    if evidence["process_state"] == "live":
        leader_evidence = exact_group.bind_leader(Path("/proc"), pid, pgid, leader_path)
        _require_bound_leader_evidence(leader_evidence, process)
        behavior_again, raw_again = runtime._read_regular_json(
            evidence["behavior_path"], "behavior decision receipt"
        )
        if raw_again != evidence["behavior_raw"] or behavior_again != evidence["behavior"]:
            raise SuccessorQueueError("behavior receipt changed before exact signal")
        portfolio_again, portfolio_raw_again = runtime._read_regular_json(
            evidence["portfolio_path"], "parent portfolio decision receipt"
        )
        if (
            portfolio_raw_again != evidence["portfolio_raw"]
            or portfolio_again != evidence["portfolio"]
        ):
            raise SuccessorQueueError("parent portfolio receipt changed before exact signal")
        # Close the initial-verify -> intent/receipt-read TOCTOU.  bind_leader
        # rechecks PID/PGID/starttime, while this final runtime read also binds
        # the immutable argv.  exact_process_group.term_group then performs one
        # more starttime/group check immediately around the signal.
        if runtime._verify_bound_process(
            binding_content, proc_root=Path("/proc"), getpgid=os.getpgid
        ) != "live":
            raise SuccessorQueueError("bound process exited before exact signal")
        exact_group.term_group(Path("/proc"), leader_path, term_path)
        signals.append("SIGTERM")
        deadline = time.monotonic() + 20.0
        residual = exact_group.verify_residual(Path("/proc"), term_path)
        while residual and time.monotonic() < deadline:
            time.sleep(0.25)
            residual = exact_group.verify_residual(Path("/proc"), term_path)
        if residual:
            exact_group.kill_residual(Path("/proc"), term_path, kill_path)
            signals.append("SIGKILL_bound_residual_after_20s")
            deadline = time.monotonic() + 10.0
            residual = exact_group.verify_residual(Path("/proc"), term_path)
            while residual and time.monotonic() < deadline:
                time.sleep(0.25)
                residual = exact_group.verify_residual(Path("/proc"), term_path)
        if residual:
            raise SuccessorQueueError("exact process group remains after reviewed stop")
    else:
        signals.append("natural_exit_before_exact_stop")

    group_files = []
    for path in (leader_path, term_path, kill_path):
        if path.exists():
            _value, raw = runtime._read_regular_json(path, "exact group evidence")
            group_files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    receipt_content = {
        "schema_version": 1,
        "job_id": job_id,
        "milestone": milestone,
        "intent_content_sha256": intent["content_sha256"],
        "process": dict(process),
        "signals": signals,
        "group_evidence": group_files,
        "exact_group_helper": {
            "path": str(exact_group_path),
            "sha256": hashlib.sha256(exact_group_path.read_bytes()).hexdigest(),
        },
        "terminal_group_empty": True,
        "automatic_retry": False,
    }
    receipt = {
        "schema_version": 1,
        "content": receipt_content,
        "content_sha256": continuation._canonical_sha256(receipt_content),
    }
    runtime._atomic_publish_json(receipt_path, receipt, "exact-stop receipt")
    return {
        "mode": "exact-stop-behavior",
        "job_id": job_id,
        "milestone": milestone,
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def validate_task_revision_probe_records(
    records: Mapping[int, dict[str, Any]], *, required_counters: set[str]
) -> dict[str, Any]:
    """Validate complete exact-ledger and mechanism activation for one probe."""

    update_ids = sorted(records)
    if len(update_ids) < 2 or update_ids != list(
        range(update_ids[0], update_ids[-1] + 1)
    ):
        raise SuccessorQueueError("full-scene probe must emit at least two consecutive ledger updates")
    selected = [records[update] for update in update_ids]
    aggregate = _sum_window(selected)
    if not required_counters.issubset(aggregate):
        raise SuccessorQueueError("full-scene probe exact ledger is incomplete")
    _validate_counter_invariants(aggregate, "full-scene probe aggregate")
    required_positive = [
        "planner_initial_tts_sample_count",
        "planner_initial_tts_sub_0p5_count",
        "planner_initial_tts_exact_0p5_count",
        "planner_initial_tts_above_0p5_count",
        *[f"planner_initial_tts_component_{index}_count" for index in range(4)],
        "planner_revision_attempt_count",
        "planner_revision_accepted_count",
        "planner_revision_last_precontact_attempt_count",
        "planner_revision_last_precontact_accepted_count",
        "planner_revision_actor_visible_count",
        "planner_revision_last_precontact_actor_visible_count",
    ]
    missing_activation = [name for name in required_positive if int(aggregate[name]) <= 0]
    if missing_activation:
        raise SuccessorQueueError(
            "full-scene probe did not activate required task-revision mechanisms: "
            + ",".join(missing_activation)
        )
    return {"exact_update_ids": update_ids, "aggregate_counters": aggregate}


def finalize_task_revision_probe_local(
    queue: Mapping[str, Any], *, job_id: str, pod: str, gpu: int, attempt_id: str
) -> dict[str, Any]:
    """Extend the reviewed generic probe result with revision/mixture/ledger evidence."""

    job = _job(queue, job_id)
    _require_launchable_job(job)
    slot = continuation.lean._slot_by_identity(queue, pod, gpu)
    claim, _argv, run_dir_text = continuation.lean._full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    run_dir = Path(run_dir_text)
    runtime, _runtime_path = _load_runtime(job["source"]["checkout"])
    generic_result, generic_raw = runtime._read_regular_json(
        run_dir / "probe_result.json", "generic full-scene probe result"
    )
    generic_content = _validate_envelope(generic_result, "generic full-scene probe result")
    if (
        generic_content.get("status") != "passed"
        or generic_content.get("unlock_authorized") is not True
        or generic_content.get("not_science") is not True
        or generic_content.get("attestable") is not False
        or generic_content.get("promotable") is not False
        or generic_content.get("run_dir") != str(run_dir)
        or generic_content.get("claim_path")
        != str(run_dir / "full_scene_probe_claim.json")
        or generic_content.get("claim_content_sha256") != claim["content_sha256"]
    ):
        raise SuccessorQueueError("generic full-scene probe did not pass")
    raw_log, log_evidence = _stable_append_prefix(run_dir / "run.log")
    records = parse_exact_behavior_log(
        raw_log, required_counters=_required_behavior_counters(queue)
    )
    probe_evidence = validate_task_revision_probe_records(
        records, required_counters=_required_behavior_counters(queue)
    )
    update_ids = probe_evidence["exact_update_ids"]
    aggregate = probe_evidence["aggregate_counters"]
    content = {
        "schema_version": 1,
        "status": "passed",
        "unlock_authorized": True,
        "job_id": job_id,
        "representative_job_id": job_id,
        "pod": pod,
        "gpu": gpu,
        "attempt_id": attempt_id,
        "claim_content_sha256": claim["content_sha256"],
        "generic_result_sha256": hashlib.sha256(generic_raw).hexdigest(),
        "generic_result_content_sha256": generic_result["content_sha256"],
        "log_prefix": log_evidence,
        "exact_update_ids": update_ids,
        "exact_behavior_ledger_schema_complete": True,
        "initial_tts_mixture_all_strata_observed": True,
        "planner_revision_attempt_accept_reject_accounting_exact": True,
        "planner_revision_accepted_observed": True,
        "planner_revision_last_precontact_accepted_observed": True,
        "planner_revision_actor_visible_observed": True,
        "planner_revision_last_precontact_actor_visible_observed": True,
        "aggregate_counters": aggregate,
        "formal_evidence_eligible": False,
        "consumer_source": _consumer_source_evidence(),
    }
    receipt = {
        "schema_version": 1,
        "content": content,
        "content_sha256": continuation._canonical_sha256(content),
    }
    path = run_dir / "task_revision_probe_result.json"
    runtime._atomic_publish_json(path, receipt, "task-revision probe result")
    return {"receipt_path": str(path), "receipt": receipt}


def validate_successor_contract(queue: Mapping[str, Any]) -> dict[str, Any]:
    status = queue.get("preregistration_status")
    if status not in {PENDING_STATUS, ACTIVATED_STATUS}:
        raise SuccessorQueueError(f"unsupported successor preregistration_status {status!r}")
    pending = status == PENDING_STATUS
    if pending and queue.get("launch_authorized") is not False:
        raise SuccessorQueueError("pending successor queue must have launch_authorized=false")
    if not pending and queue.get("launch_authorized") is not True:
        raise SuccessorQueueError("activated successor queue must have launch_authorized=true")

    namespace = _mapping(queue.get("namespace_contract"), "namespace_contract")
    if namespace.get("root") != "/workspace/codexschema/phase1_task_revision_supercombo_20260716":
        raise SuccessorQueueError("successor must use its new no-clobber run namespace")
    if pending and namespace.get("status") != PENDING_STATUS:
        raise SuccessorQueueError("run namespace must remain pending with the source/probe")
    if not pending and namespace.get("status") != "activated_no_clobber":
        raise SuccessorQueueError("activated run namespace must be marked activated_no_clobber")

    blocking = _mapping(queue.get("blocking_contract"), "blocking_contract")
    specialized = blocking.get("task_revision_full_scene_probe_evidence")
    if pending:
        if specialized != "PENDING_TASK_REVISION_PROBE":
            raise SuccessorQueueError(
                "pending queue must keep task-revision probe evidence pending"
            )
    elif _specialized_probe_blockers(queue):
        raise SuccessorQueueError(
            "activated queue lacks exact specialized probe evidence: "
            + "; ".join(_specialized_probe_blockers(queue))
        )

    execution = _mapping(
        queue.get("task_revision_execution_contract"),
        "task_revision_execution_contract",
    )
    if execution != {
        "actor_visible_revision_fields": [
            "target_position",
            "target_velocity",
            "signed_target_normal",
            "time_to_strike",
        ],
        "update_window": "every_policy_tick_through_last_pre_contact_tick",
        "policy_tick_s": 0.02,
        "same_ball_revision_is_not_a_new_task": True,
        "side_and_clip_are_task_invariant": True,
    }:
        raise SuccessorQueueError(
            "task revision execution contract must keep all four actor fields live through "
            "the final pre-contact policy tick"
        )

    transport = _mapping(
        queue.get("transport_atomicity_contract"),
        "transport_atomicity_contract",
    )
    if transport != {
        "status": "delay_zero_only_until_coupled_governor_actor_transport_exists",
        "launchable_target_delay_steps": [0],
        "blocked_job_ids": [
            "taskrev_p1_core_low_noise",
            "taskrev_p1_uncompensated_negative",
        ],
        "blocker": TRANSPORT_BLOCKER,
        "blocked_pair_is_not_counted_as_a_runtime_or_reward_failure": True,
    }:
        raise SuccessorQueueError(
            "transport atomicity contract must keep the two delay-2 cells NO-LAUNCH"
        )

    jobs = queue["jobs"]
    questions = [job.get("scientific_question") for job in jobs]
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise SuccessorQueueError("every cell must state one human-auditable scientific question")
    if len(set(questions)) != len(questions):
        raise SuccessorQueueError("all 24 scientific questions must be distinct")
    if len({job.get("seed") for job in jobs}) != 1:
        raise SuccessorQueueError("the funnel must use exactly one shared seed, not replicate failures")

    overrides_by_id: dict[str, dict[str, str]] = {}
    recipe_signatures: set[tuple[tuple[str, str], ...]] = set()
    noise_levels: set[tuple[float, float, float, float]] = set()
    for job in jobs:
        job_id = job["id"]
        if pending and (job.get("status") != continuation.lean.BLOCKED or not job.get("blocker")):
            raise SuccessorQueueError(f"pending job {job_id} must remain explicitly blocked")
        expected_transport_block = job_id in TRANSPORT_BLOCKED_JOB_IDS
        if expected_transport_block:
            if (
                _job_launch_authorized(job)
                or job.get("scientific_blocker") != TRANSPORT_BLOCKER
            ):
                raise SuccessorQueueError(
                    f"{job_id} must remain scientific NO-LAUNCH for coupled transport"
                )
            if not pending and (
                job.get("status") != continuation.lean.BLOCKED
                or job.get("blocker") != TRANSPORT_BLOCKER
            ):
                raise SuccessorQueueError(
                    f"activated {job_id} must remain blocked by {TRANSPORT_BLOCKER}"
                )
        elif (
            not _job_launch_authorized(job)
            or job.get("scientific_blocker") is not None
        ):
            raise SuccessorQueueError(
                f"{job_id} is a delay-zero cell and must remain scientifically launchable"
            )
        elif not pending and job.get("status") == continuation.lean.BLOCKED:
            raise SuccessorQueueError(
                f"activated launchable cell {job_id} may not remain globally blocked"
            )
        if job.get("formal_evidence_eligible") is not False:
            raise SuccessorQueueError(f"{job_id} must remain formal-ineligible")
        if job.get("behavior_ledger_contract_ref") != "exact_behavior_update_v1":
            raise SuccessorQueueError(f"{job_id} does not bind the exact behavior ledger")
        if job.get("milestones") != REQUIRED_OFFSETS:
            raise SuccessorQueueError(f"{job_id} checkpoints changed")
        overrides = _compiled_overrides(job)
        for key, expected in REQUIRED_SINGLE_CLOCK_OVERRIDES.items():
            if overrides.get(key) != expected:
                raise SuccessorQueueError(
                    f"{job_id} must disable the legacy hold clock with final override "
                    f"{expected!r}; got {overrides.get(key)!r}"
                )
        for key, expected in REQUIRED_TASK_IDENTITY_OVERRIDES.items():
            if overrides.get(key) != expected:
                raise SuccessorQueueError(
                    f"{job_id} must preserve same-task side/clip identity with final "
                    f"override {expected!r}; got {overrides.get(key)!r}"
                )
        delay = overrides.get("task.racket.target_delay_steps")
        timestamp_pair = job.get("timestamp_role") in {
            "compensated_treatment",
            "uncompensated_negative_control",
        }
        expected_delay = (
            "task.racket.target_delay_steps=2"
            if timestamp_pair
            else "task.racket.target_delay_steps=0"
        )
        if delay != expected_delay:
            raise SuccessorQueueError(
                f"{job_id} target_delay_steps must be {'2 for the matched delay pair' if timestamp_pair else '0 for same-tick actor delivery'}"
            )
        revision = _validate_revision(job, overrides)
        noise_levels.add(
            tuple(
                float(revision[key])
                for key in (
                    "position_std_m",
                    "velocity_std_mps",
                    "normal_std_rad",
                    "tts_std_s",
                )
            )
        )
        forbidden = [
            key
            for key in overrides
            if "lateral_perturb" in key
            or "external_force" in key
            or key == "task.motion.speed_scale_per_clip"
        ]
        if forbidden:
            raise SuccessorQueueError(
                f"{job_id} contains blocked torso-force or fake fixed-retiming keys: {forbidden}"
            )
        signature = tuple(sorted(overrides.items()))
        if signature in recipe_signatures:
            raise SuccessorQueueError(f"{job_id} duplicates another scientific cell recipe")
        recipe_signatures.add(signature)
        overrides_by_id[job_id] = overrides
    if noise_levels != ALLOWED_NOISE:
        raise SuccessorQueueError("portfolio must exercise both registered planner-noise levels")

    _validate_timestamp_pair(jobs, overrides_by_id)
    _validate_ledger(queue)
    return {
        "successor_contract_valid": True,
        "pending": pending,
        "distinct_scientific_cells": len(questions),
        "planner_noise_levels": len(noise_levels),
        "explicit_0p5_point_mass": True,
        "sub_0p5_is_stress_not_floor": True,
        "exact_behavior_ledger_bound": True,
        "formal_evidence_eligible": False,
        "launchable_job_count": len(jobs) - len(TRANSPORT_BLOCKED_JOB_IDS),
        "transport_blocked_job_count": len(TRANSPORT_BLOCKED_JOB_IDS),
        "transport_blocked_job_ids": sorted(TRANSPORT_BLOCKED_JOB_IDS),
    }


def load_queue(path: Path) -> dict[str, Any]:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = (key_node.tag, getattr(key_node, "value", None))
            if key in seen:
                raise SuccessorQueueError(
                    f"duplicate YAML key is forbidden: {getattr(key_node, 'value', None)!r}"
                )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
    )
    raw = continuation._stable_regular_bytes(path.resolve(), "task-revision queue YAML")
    try:
        yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SuccessorQueueError(f"task-revision queue YAML is invalid: {exc}") from exc
    queue = continuation.load_queue(path)
    validate_successor_contract(queue)
    continuation._bind_parent_context(queue)
    return queue


def _specialized_probe_blockers(queue: Mapping[str, Any]) -> list[str]:
    evidence = _mapping(queue.get("blocking_contract"), "blocking_contract").get(
        "task_revision_full_scene_probe_evidence"
    )
    if not isinstance(evidence, dict):
        return ["task_revision_full_scene_probe_evidence must be a pass mapping"]
    expected = {
        "status": "passed",
        "unlock_authorized": True,
        "representative_job_id": queue["full_scene_probe_contract"][
            "representative_job_id"
        ],
        "initial_tts_mixture_all_strata_observed": True,
        "planner_revision_attempt_accept_reject_accounting_exact": True,
        "planner_revision_accepted_observed": True,
        "planner_revision_last_precontact_accepted_observed": True,
        "planner_revision_actor_visible_observed": True,
        "planner_revision_last_precontact_actor_visible_observed": True,
        "exact_behavior_ledger_schema_complete": True,
    }
    blockers = [
        f"task_revision_full_scene_probe_evidence.{key} must be {value!r}"
        for key, value in expected.items()
        if evidence.get(key) != value
    ]
    receipt_path = evidence.get("receipt_path")
    if not isinstance(receipt_path, str) or not receipt_path.startswith("/workspace/"):
        blockers.append("task_revision_full_scene_probe_evidence.receipt_path must be absolute")
    for key in ("receipt_file_sha256", "receipt_content_sha256"):
        value = evidence.get(key)
        if not isinstance(value, str) or continuation.lean.SHA256.fullmatch(value) is None:
            blockers.append(f"task_revision_full_scene_probe_evidence.{key} must be SHA-256")
    return blockers


def successor_activation_blockers(queue: Mapping[str, Any]) -> list[str]:
    launchable = _launchable_continuation_queue(queue)
    return [
        *continuation.activation_blockers(launchable),
        *_specialized_probe_blockers(queue),
    ]


def cmd_validate(queue: Mapping[str, Any]) -> dict[str, Any]:
    result = continuation.validate_queue(queue)
    result.update(validate_successor_contract(queue))
    result["blockers"] = successor_activation_blockers(queue)
    result["activation_ready"] = not result["blockers"]
    result["mode"] = "validate_task_revision_successor"
    return result


def cmd_plan(queue: Mapping[str, Any]) -> dict[str, Any]:
    result = continuation.cmd_plan(queue)
    by_id = {job["id"]: job for job in queue["jobs"]}
    for row in result["jobs"]:
        job = by_id[row["job_id"]]
        row["human_name"] = job["human_name"]
        row["scientific_question"] = job["scientific_question"]
        row["formal_evidence_eligible"] = False
        row["scientific_launch_authorized"] = _job_launch_authorized(job)
        row["scientific_blocker"] = job.get("scientific_blocker")
    result.update(validate_successor_contract(queue))
    result["blockers"] = successor_activation_blockers(queue)
    result["activation_ready"] = not result["blockers"]
    result["mode"] = "plan_task_revision_successor"
    return result


def cmd_fill(
    queue: dict[str, Any], *, count: int, execute: bool, confirm: str | None
) -> dict[str, Any]:
    blockers = successor_activation_blockers(queue)
    if blockers:
        raise SuccessorQueueError("successor fill is blocked: " + "; ".join(blockers))
    delegated = continuation.CONFIRM if confirm == CONFIRM else confirm
    launchable = _launchable_continuation_queue(queue)
    return continuation.cmd_fill(
        launchable, count=count, execute=execute, confirm=delegated
    )


def cmd_prepare_source_assets(
    queue: dict[str, Any],
    *,
    job_id: str,
    pod: str,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    """Expose the reviewed ignored-asset hydrator through the successor entry point."""

    job = _job(queue, job_id)
    _require_launchable_job(job)
    return continuation.lean.cmd_prepare_source_assets(
        queue,
        job_id=job_id,
        pod=pod,
        execute=execute,
        confirm=confirm,
    )


def cmd_full_scene_probe(
    queue: dict[str, Any],
    *,
    job_id: str,
    pod: str,
    gpu: int,
    attempt_id: str,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    """Delegate one representative-scale probe to the reviewed generic harness."""

    if job_id != queue["full_scene_probe_contract"]["representative_job_id"]:
        raise SuccessorQueueError("full-scene probe must use the preregistered representative job")
    _require_launchable_job(_job(queue, job_id))
    if execute and confirm != FULL_SCENE_PROBE_CONFIRM:
        raise SuccessorQueueError(
            f"--execute requires --confirm {FULL_SCENE_PROBE_CONFIRM}"
        )
    delegated_confirm = (
        continuation.lean.FULL_SCENE_PROBE_CONFIRM if execute else None
    )
    result = continuation.lean.cmd_full_scene_probe(
        queue,
        job_id=job_id,
        pod=pod,
        gpu=gpu,
        attempt_id=attempt_id,
        execute=execute,
        confirm=delegated_confirm,
    )
    result["task_revision_specialized_result_required"] = True
    result["task_revision_specialized_result_path"] = (
        f"{result['run_dir']}/task_revision_probe_result.json"
    )
    result["generic_result_alone_may_unlock_successor"] = False
    return result


def _remote_task_revision_command(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    function: str,
    kwargs: Mapping[str, Any],
) -> list[str]:
    """Run reviewed consumer bytes with an in-memory SHA-bound queue, without remote writes."""

    allowed = {
        "finalize_task_revision_probe_local",
        "inspect_or_attest_behavior_local",
        "inspect_or_attest_parent_portfolio_local",
        "run_pruning_cycle_local",
        "exact_stop_behavior_local",
    }
    if function not in allowed:
        raise SuccessorQueueError("unsupported embedded remote consumer")
    source = str(job["source"]["checkout"]).rstrip("/")
    script_raw = Path(__file__).read_bytes()
    script_sha = hashlib.sha256(script_raw).hexdigest()
    queue_value = json.loads(
        json.dumps(queue, allow_nan=False, ensure_ascii=False, sort_keys=True)
    )
    request = {
        "schema_version": 1,
        "function": function,
        "kwargs": dict(kwargs),
        "queue": queue_value,
    }
    request_raw = json.dumps(
        request,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    encoded_script = base64.b64encode(zlib.compress(script_raw, 9)).decode()
    encoded_request = base64.b64encode(zlib.compress(request_raw, 9)).decode()
    remote_filename = f"{source}/scripts/run_phase1_task_revision_supercombo_queue.py"
    program = (
        "import base64,hashlib,json,pathlib,sys,zlib;"
        "raw=zlib.decompress(base64.b64decode(sys.argv[1],validate=True));"
        f"assert hashlib.sha256(raw).hexdigest()=={script_sha!r};"
        "req=json.loads(zlib.decompress(base64.b64decode(sys.argv[2],validate=True)));"
        "assert req['schema_version']==1;"
        f"ns={{'__name__':'embedded_task_revision_consumer','__file__':{remote_filename!r}}};"
        "exec(compile(raw,ns['__file__'],'exec'),ns);"
        f"ns['EMBEDDED_CONSUMER_SHA256']={script_sha!r};"
        "qraw=json.dumps(req['queue'],allow_nan=False,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode();"
        "queue=ns['continuation']._LoadedContinuationQueue(req['queue'],source_path=pathlib.Path('/embedded/task_revision_queue.yaml'),source_sha256=hashlib.sha256(qraw).hexdigest());"
        "ns['validate_successor_contract'](queue);"
        "ns['continuation']._bind_parent_context(queue);"
        "assert req['function'] in {'finalize_task_revision_probe_local','inspect_or_attest_behavior_local','inspect_or_attest_parent_portfolio_local','run_pruning_cycle_local','exact_stop_behavior_local'};"
        "result=ns[req['function']](queue,**req['kwargs']);"
        "print(json.dumps(result,allow_nan=False,ensure_ascii=False,sort_keys=True))"
    )
    return [
        continuation.lean.ISAAC_PYTHON,
        "-c",
        program,
        encoded_script,
        encoded_request,
    ]


def _consumer_source_evidence() -> dict[str, Any]:
    embedded = globals().get("EMBEDDED_CONSUMER_SHA256")
    if isinstance(embedded, str) and continuation.lean.SHA256.fullmatch(embedded):
        return {"mode": "embedded_sha_bound", "sha256": embedded}
    path = Path(__file__).resolve()
    return {
        "mode": "filesystem",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _last_marked_json(raw: str, marker: str) -> dict[str, Any]:
    token = marker + "\n"
    if token not in raw:
        raise SuccessorQueueError("remote task-revision consumer omitted its result marker")
    tail = raw.rsplit(token, 1)[1].strip()
    try:
        value = json.loads(tail)
    except json.JSONDecodeError as exc:
        raise SuccessorQueueError(
            "remote task-revision consumer returned malformed JSON"
        ) from exc
    return _mapping(value, "remote task-revision result")


def cmd_finalize_full_scene_probe(
    queue: dict[str, Any],
    *,
    job_id: str,
    pod: str,
    gpu: int,
    attempt_id: str,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    """In one SSH, run the reviewed finalizer then the stricter revision finalizer."""

    if job_id != queue["full_scene_probe_contract"]["representative_job_id"]:
        raise SuccessorQueueError("probe finalization must use the preregistered representative job")
    if execute and confirm != FINALIZE_FULL_SCENE_PROBE_CONFIRM:
        raise SuccessorQueueError(
            f"--execute requires --confirm {FINALIZE_FULL_SCENE_PROBE_CONFIRM}"
        )
    job = continuation.lean._job_by_id(queue, job_id)
    slot = continuation.lean._slot_by_identity(queue, pod, gpu)
    continuation.lean._require_bound_slot(
        job, slot, phase="task-revision probe finalization", include_preferred=True
    )
    claim, _argv, run_dir = continuation.lean._full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    generic = continuation.lean._finalize_full_scene_probe_script(
        job, slot.pod, run_dir, claim["content_sha256"]
    )
    marker = "TASK_REVISION_SPECIALIZED_PROBE_RESULT_JSON"
    specialized = _remote_task_revision_command(
        queue,
        job,
        function="finalize_task_revision_probe_local",
        kwargs={"job_id": job_id, "pod": pod, "gpu": gpu, "attempt_id": attempt_id},
    )
    remote = generic + f"\nprintf '%s\\n' {shlex.quote(marker)}\n" + shlex.join(
        specialized
    )
    result: dict[str, Any] = {
        "mode": "finalize-task-revision-full-scene-probe",
        "dry_run": not execute,
        "job_id": job_id,
        "resource": slot.name,
        "run_dir": run_dir,
        "generic_result_path": f"{run_dir}/probe_result.json",
        "specialized_result_path": f"{run_dir}/task_revision_probe_result.json",
        "claim_sha256": claim["content_sha256"],
        "one_ssh_transaction": True,
        "automatic_retry_authorized": False,
        "queue_status_mutated": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *continuation.lean._ssh_prefix(queue, slot.pod),
            f"bash -lc {shlex.quote(remote)}",
        ]
        return result
    raw = continuation.lean._run_ssh(
        queue,
        slot.pod,
        remote,
        timeout=240,
        phase=f"finalize-task-revision-probe:{job_id}:{attempt_id}",
    )
    terminal = _last_marked_json(raw, marker)
    receipt = _mapping(terminal.get("receipt"), "specialized probe receipt")
    content = _validate_envelope(receipt, "specialized probe receipt")
    if content.get("status") != "passed" or content.get("unlock_authorized") is not True:
        raise SuccessorQueueError("specialized task-revision probe did not pass")
    result["terminal_status"] = "passed"
    result["unlock_authorized"] = True
    result["terminal_result"] = terminal
    return result


def cmd_behavior(
    queue: dict[str, Any],
    *,
    job_id: str,
    milestone: int,
    execute: bool,
    confirm: str | None,
    write_receipt: bool,
) -> dict[str, Any]:
    """Inspect/attest one registered exact window with one selected-Pod SSH."""

    expected_confirm = BEHAVIOR_ATTEST_CONFIRM if write_receipt else BEHAVIOR_INSPECT_CONFIRM
    if execute and confirm != expected_confirm:
        raise SuccessorQueueError(f"--execute requires --confirm {expected_confirm}")
    job = _job(queue, job_id)
    _require_launchable_job(job)
    slot = continuation._slots(queue)[job["resource"]["required_slot"]]
    command = _remote_task_revision_command(
        queue,
        job,
        function="inspect_or_attest_behavior_local",
        kwargs={
            "job_id": job_id,
            "milestone": milestone,
            "write_receipt": write_receipt,
        },
    )
    remote = shlex.join(command)
    result: dict[str, Any] = {
        "mode": "attest-behavior" if write_receipt else "inspect-behavior",
        "dry_run": not execute,
        "job_id": job_id,
        "milestone": milestone,
        "resource": slot.name,
        "receipt_path": f"{job['run_dir']}/behavior_milestones/model_{milestone}.json",
        "automatic_stop_authorized": False,
        "automatic_retry_authorized": False,
        "one_ssh_transaction": True,
    }
    if not execute:
        result["activation_blockers"] = successor_activation_blockers(queue)
        result["ssh_argv"] = [
            *continuation.lean._ssh_prefix(queue, slot.pod),
            f"bash -lc {shlex.quote(remote)}",
        ]
        return result
    blockers = successor_activation_blockers(queue)
    if blockers:
        raise SuccessorQueueError("behavior consumer is activation-blocked: " + "; ".join(blockers))
    raw = continuation.lean._run_ssh(
        queue,
        slot.pod,
        remote,
        timeout=240,
        phase=f"{'attest' if write_receipt else 'inspect'}-behavior:{job_id}:{milestone}",
    )
    try:
        terminal = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SuccessorQueueError("remote behavior consumer returned malformed JSON") from exc
    result["terminal_result"] = terminal
    return result


def cmd_pruning_cycle(
    queue: dict[str, Any],
    *,
    pod: str,
    milestone_offset: int,
    execute: bool,
    confirm: str | None,
    write_receipts: bool,
) -> dict[str, Any]:
    """Run at most one receipt-only SSH per selected Pod; never signal trainers."""

    if milestone_offset not in PORTFOLIO_OFFSETS:
        raise SuccessorQueueError("pruning-cycle offset must be 200, 500, or 1000")
    selected = ["pod1", "pod2"] if pod == "all" else [pod]
    if any(name not in {"pod1", "pod2"} for name in selected):
        raise SuccessorQueueError("pruning-cycle pod must be all, pod1, or pod2")
    expected_confirm = (
        PORTFOLIO_ATTEST_CONFIRM if write_receipts else PORTFOLIO_INSPECT_CONFIRM
    )
    if execute and confirm != expected_confirm:
        raise SuccessorQueueError(f"--execute requires --confirm {expected_confirm}")
    commands: dict[str, list[str]] = {}
    for pod_name in selected:
        representative = next(
            job
            for job in queue["jobs"]
            if _job_launch_authorized(job)
            and job["resource"]["required_slot"].startswith(pod_name + "/")
        )
        commands[pod_name] = _remote_task_revision_command(
            queue,
            representative,
            function="run_pruning_cycle_local",
            kwargs={
                "pod": pod_name,
                "milestone_offset": milestone_offset,
                "write_receipts": write_receipts,
            },
        )
    result: dict[str, Any] = {
        "mode": "attest-pruning-cycle" if write_receipts else "inspect-pruning-cycle",
        "dry_run": not execute,
        "milestone_offset_from_parent": milestone_offset,
        "selected_pods": selected,
        "maximum_ssh_connections_per_pod": 1,
        "automatic_signal_authorized": False,
        "automatic_retry_authorized": False,
        "pods": {},
    }
    if not execute:
        result["activation_blockers"] = successor_activation_blockers(queue)
        result["pods"] = {
            pod_name: {
                "ssh_argv": [
                    *continuation.lean._ssh_prefix(queue, pod_name),
                    f"bash -lc {shlex.quote(shlex.join(command))}",
                ],
                "machine_checklist": [
                    "exact_claim_and_binding",
                    "checkpoint_exists_and_is_regular",
                    "behavior_receipt_absent_before_attest",
                    "two_complete_100_update_windows",
                    "same_parent_all_attested_portfolio",
                    "minimum_two_survivors",
                    "exact_0p5_exposed_survivor",
                    "broad_arrival_survivor",
                    "no_clobber_portfolio_receipt_only_if_elimination_exists",
                ],
            }
            for pod_name, command in commands.items()
        }
        return result
    blockers = successor_activation_blockers(queue)
    if blockers:
        raise SuccessorQueueError(
            "pruning cycle is activation-blocked: " + "; ".join(blockers)
        )
    for pod_name in selected:
        raw = continuation.lean._run_ssh(
            queue,
            pod_name,
            shlex.join(commands[pod_name]),
            timeout=900,
            phase=f"{'attest' if write_receipts else 'inspect'}-pruning-cycle:"
            f"{pod_name}:{milestone_offset}",
        )
        try:
            result["pods"][pod_name] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SuccessorQueueError(
                f"remote pruning-cycle consumer returned malformed JSON on {pod_name}"
            ) from exc
    return result


def cmd_exact_stop_behavior(
    queue: dict[str, Any],
    *,
    job_id: str,
    milestone: int,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    """Explicitly consume one stop receipt; never called by rolling automation implicitly."""

    if execute and confirm != BEHAVIOR_STOP_CONFIRM:
        raise SuccessorQueueError(f"--execute requires --confirm {BEHAVIOR_STOP_CONFIRM}")
    job = _job(queue, job_id)
    _require_launchable_job(job)
    slot = continuation._slots(queue)[job["resource"]["required_slot"]]
    command = _remote_task_revision_command(
        queue,
        job,
        function="exact_stop_behavior_local",
        kwargs={"job_id": job_id, "milestone": milestone},
    )
    remote = shlex.join(command)
    result: dict[str, Any] = {
        "mode": "exact-stop-behavior",
        "dry_run": not execute,
        "job_id": job_id,
        "milestone": milestone,
        "resource": slot.name,
        "automatic_invocation": False,
        "automatic_retry_authorized": False,
        "one_ssh_transaction": True,
    }
    if not execute:
        result["activation_blockers"] = successor_activation_blockers(queue)
        result["ssh_argv"] = [
            *continuation.lean._ssh_prefix(queue, slot.pod),
            f"bash -lc {shlex.quote(remote)}",
        ]
        return result
    blockers = successor_activation_blockers(queue)
    if blockers:
        raise SuccessorQueueError("exact stop is activation-blocked: " + "; ".join(blockers))
    raw = continuation.lean._run_ssh(
        queue,
        slot.pod,
        remote,
        timeout=300,
        phase=f"exact-stop-behavior:{job_id}:{milestone}",
    )
    try:
        result["terminal_result"] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SuccessorQueueError("remote exact-stop consumer returned malformed JSON") from exc
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("validate")
    sub.add_parser("plan")
    sub.add_parser("inspect-parents")
    fill = sub.add_parser("fill")
    fill.add_argument("--count", type=int, required=True)
    fill.add_argument("--execute", action="store_true")
    fill.add_argument("--confirm")
    prepare = sub.add_parser("prepare-source-assets")
    prepare.add_argument("--job-id", required=True)
    prepare.add_argument("--pod", required=True)
    prepare.add_argument("--execute", action="store_true")
    prepare.add_argument("--confirm")
    full_scene = sub.add_parser("full-scene-probe")
    full_scene.add_argument("--job-id", required=True)
    full_scene.add_argument("--pod", required=True)
    full_scene.add_argument("--gpu", required=True, type=int)
    full_scene.add_argument("--attempt-id", required=True)
    full_scene.add_argument("--execute", action="store_true")
    full_scene.add_argument("--confirm")
    finalize = sub.add_parser("finalize-full-scene-probe")
    finalize.add_argument("--job-id", required=True)
    finalize.add_argument("--pod", required=True)
    finalize.add_argument("--gpu", required=True, type=int)
    finalize.add_argument("--attempt-id", required=True)
    finalize.add_argument("--execute", action="store_true")
    finalize.add_argument("--confirm")
    for mode in ("inspect-behavior", "attest-behavior"):
        behavior = sub.add_parser(mode)
        behavior.add_argument("--job-id", required=True)
        behavior.add_argument("--milestone", required=True, type=int)
        behavior.add_argument("--execute", action="store_true")
        behavior.add_argument("--confirm")
    for mode in ("inspect-pruning-cycle", "attest-pruning-cycle"):
        pruning = sub.add_parser(mode)
        pruning.add_argument("--pod", choices=("all", "pod1", "pod2"), default="all")
        pruning.add_argument(
            "--milestone-offset", required=True, type=int, choices=sorted(PORTFOLIO_OFFSETS)
        )
        pruning.add_argument("--execute", action="store_true")
        pruning.add_argument("--confirm")
    stop = sub.add_parser("exact-stop-behavior")
    stop.add_argument("--job-id", required=True)
    stop.add_argument("--milestone", required=True, type=int)
    stop.add_argument("--execute", action="store_true")
    stop.add_argument("--confirm")
    local_finalize = sub.add_parser("_local-finalize-task-revision-probe")
    local_finalize.add_argument("--job-id", required=True)
    local_finalize.add_argument("--pod", required=True)
    local_finalize.add_argument("--gpu", required=True, type=int)
    local_finalize.add_argument("--attempt-id", required=True)
    local_finalize.add_argument("--confirm", required=True)
    for mode in ("_local-inspect-behavior", "_local-attest-behavior"):
        behavior = sub.add_parser(mode)
        behavior.add_argument("--job-id", required=True)
        behavior.add_argument("--milestone", required=True, type=int)
        behavior.add_argument("--confirm", required=True)
    for mode in ("_local-inspect-pruning-cycle", "_local-attest-pruning-cycle"):
        pruning = sub.add_parser(mode)
        pruning.add_argument("--pod", required=True, choices=("pod1", "pod2"))
        pruning.add_argument(
            "--milestone-offset", required=True, type=int, choices=sorted(PORTFOLIO_OFFSETS)
        )
        pruning.add_argument("--confirm", required=True)
    local_stop = sub.add_parser("_local-exact-stop-behavior")
    local_stop.add_argument("--job-id", required=True)
    local_stop.add_argument("--milestone", required=True, type=int)
    local_stop.add_argument("--confirm", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue_path = args.queue.resolve()
        queue = load_queue(queue_path)
        if args.mode == "validate":
            result = cmd_validate(queue)
        elif args.mode == "plan":
            result = cmd_plan(queue)
        elif args.mode == "inspect-parents":
            result = continuation.cmd_inspect_parents(queue)
        elif args.mode == "fill":
            result = cmd_fill(
                queue, count=args.count, execute=args.execute, confirm=args.confirm
            )
        elif args.mode == "prepare-source-assets":
            result = cmd_prepare_source_assets(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "full-scene-probe":
            result = cmd_full_scene_probe(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                gpu=args.gpu,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "finalize-full-scene-probe":
            result = cmd_finalize_full_scene_probe(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                gpu=args.gpu,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode in {"inspect-behavior", "attest-behavior"}:
            result = cmd_behavior(
                queue,
                job_id=args.job_id,
                milestone=args.milestone,
                execute=args.execute,
                confirm=args.confirm,
                write_receipt=args.mode == "attest-behavior",
            )
        elif args.mode in {"inspect-pruning-cycle", "attest-pruning-cycle"}:
            result = cmd_pruning_cycle(
                queue,
                pod=args.pod,
                milestone_offset=args.milestone_offset,
                execute=args.execute,
                confirm=args.confirm,
                write_receipts=args.mode == "attest-pruning-cycle",
            )
        elif args.mode == "exact-stop-behavior":
            result = cmd_exact_stop_behavior(
                queue,
                job_id=args.job_id,
                milestone=args.milestone,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "_local-finalize-task-revision-probe":
            if args.confirm != LOCAL_PROBE_FINALIZE_CONFIRM:
                raise SuccessorQueueError("local probe finalizer confirmation mismatch")
            result = finalize_task_revision_probe_local(
                queue,
                job_id=args.job_id,
                pod=args.pod,
                gpu=args.gpu,
                attempt_id=args.attempt_id,
            )
        elif args.mode in {"_local-inspect-behavior", "_local-attest-behavior"}:
            expected = (
                BEHAVIOR_ATTEST_CONFIRM
                if args.mode == "_local-attest-behavior"
                else BEHAVIOR_INSPECT_CONFIRM
            )
            if args.confirm != expected:
                raise SuccessorQueueError("local behavior consumer confirmation mismatch")
            result = inspect_or_attest_behavior_local(
                queue,
                job_id=args.job_id,
                milestone=args.milestone,
                write_receipt=args.mode == "_local-attest-behavior",
            )
        elif args.mode in {
            "_local-inspect-pruning-cycle",
            "_local-attest-pruning-cycle",
        }:
            expected = (
                PORTFOLIO_ATTEST_CONFIRM
                if args.mode == "_local-attest-pruning-cycle"
                else PORTFOLIO_INSPECT_CONFIRM
            )
            if args.confirm != expected:
                raise SuccessorQueueError("local pruning-cycle confirmation mismatch")
            result = run_pruning_cycle_local(
                queue,
                pod=args.pod,
                milestone_offset=args.milestone_offset,
                write_receipts=args.mode == "_local-attest-pruning-cycle",
            )
        elif args.mode == "_local-exact-stop-behavior":
            if args.confirm != BEHAVIOR_STOP_CONFIRM:
                raise SuccessorQueueError("local exact-stop confirmation mismatch")
            result = exact_stop_behavior_local(
                queue, job_id=args.job_id, milestone=args.milestone
            )
        else:  # pragma: no cover
            raise SuccessorQueueError(f"unsupported mode: {args.mode}")
    except (
        SuccessorQueueError,
        continuation.ContinuationQueueError,
        continuation.lean.QueueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
