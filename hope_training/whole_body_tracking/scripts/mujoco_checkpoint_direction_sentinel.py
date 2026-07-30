#!/usr/bin/env python3
"""CPU-only, no-clobber MuJoCo direction sentinel for one or more ONNX milestones.

This wrapper deliberately does not produce a policy success score.  It invokes
``mujoco_eval_onnx.py`` with the immutable BankExam paper, the in-memory Isaac-equivalent table
top, and the explicitly inexact post-step velocity proxy.  The proxy lets a direction-only trace
continue after the first known plant divergence; the embedded success-score authority permanently
suppresses pass/return numbers whenever plant parity is false.

Example (hourly callers must choose a fresh timestamped output directory):

  CUDA_VISIBLE_DEVICES='' python3 scripts/mujoco_checkpoint_direction_sentinel.py \
    --milestone model_10000=/abs/exported_model10000/policy.onnx \
    --milestone model_12000=/abs/exported_model12000/policy.onnx \
    --output-dir /abs/sentinel_20260728T1600Z -- \
    --mjcf /abs/a3_pingpong.xml \
    --motion-files /abs/a.npz /abs/b.npz /abs/c.npz /abs/d.npz \
    --target-source bank --exam-bank /abs/exam.npz \
    --exam-schedule-json /abs/exam_schedule.json

Exit 0 means every machine gate passed.  Exit 3 means artifacts were written but at least one stop
gate fired (the expected result while plant parity is false).  Exit 2 means invocation/evaluator
failure.  Existing output directories are always rejected before any evaluator starts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = 1
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_OUTPUT_NAMES = frozenset(
    {"direction_sentinel.json", "direction_sentinel.csv"}
)
# No reviewed MuJoCo-vs-PhysX robot/table sensor equivalence certificate exists yet.
ACCEPTED_MUJOCO_PHYSX_TABLE_SENSOR_PARITY_CERTIFICATE_SHA256 = None
FORBIDDEN_EVALUATOR_FLAGS = {
    "--onnx",
    "--expected-onnx-sha256",
    "--expected-evaluator-sha256",
    "--out-dir",
    "--noise-scales",
    "--allow-inexact-contract",
    "--allow-velocity-limit-proxy",
    "--with-table-obstacle",
    "--viewer",
    "--no-realtime",
}
SINGLETON_REQUIRED_EVALUATOR_FLAGS = (
    "--target-source",
    "--exam-bank",
    "--exam-schedule-json",
    "--mjcf",
    "--motion-files",
)
SUCCESS_SCORE_KEYS = {
    "returned",
    "exact_composite",
    "n_composite",
    "n_returned",
    "n_returned_and_recovered_to_next",
    "landed_ok",
    "signed_face_ok",
    "pos_pass",
    "vel_pass",
    "normal_pass",
    "composite_pass",
    "net_clear",
    "cf_signed_face_ok",
    "cf_landed_ok",
    "cf_net_clear",
}
SUCCESS_SCORE_PREFIXES = (
    "strike_composite",
    "strike_pos_pass",
    "strike_vel_pass",
    "strike_normal_pass",
    "composite_rate",
    "composite_",
    "pos_pass_",
    "vel_pass_",
    "nrm_pass_",
    "score_",
    "return_success",
    "return_and_recover",
    "recover_rate_given_return",
    "exact_composite_rate",
    "pos_fail",
    "vel_fail",
    "normal_fail",
    "pos_only_fail",
    "vel_only_fail",
    "pos_and_vel_fail",
    "signed_face_ok_rate",
    "physical_b_opponent_facing_rate",
    "landing_valid_rate",
    "in_bounds_rate",
    "net_clear_rate",
)


class SentinelError(RuntimeError):
    """Fail-closed input or artifact error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_milestone(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("milestone must be LABEL=/absolute/policy.onnx")
    label, raw_path = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(
            f"invalid milestone label {label!r}; use letters, digits, dot, underscore or hyphen"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("milestone ONNX path must be absolute")
    return label, path.resolve()


def require_flag_value(
    arguments: Sequence[str], flag: str, expected: Optional[str] = None
) -> str:
    if list(arguments).count(flag) != 1:
        raise SentinelError(f"evaluator arguments require {flag} exactly once")
    try:
        index = list(arguments).index(flag)
    except ValueError as exc:
        raise SentinelError(f"evaluator arguments require {flag}") from exc
    if index + 1 >= len(arguments) or str(arguments[index + 1]).startswith("--"):
        raise SentinelError(f"evaluator flag {flag} requires a value")
    value = str(arguments[index + 1])
    if expected is not None and value != expected:
        raise SentinelError(f"{flag} must be {expected!r}, got {value!r}")
    return value


def validate_evaluator_arguments(arguments: Sequence[str]) -> list[str]:
    arguments = list(arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    for token in arguments:
        option = token.split("=", 1)[0] if str(token).startswith("--") else ""
        if option and any(
            option == flag or flag.startswith(option)
            for flag in FORBIDDEN_EVALUATOR_FLAGS
        ):
            raise SentinelError(
                f"sentinel owns evaluator flag {token!r}; remove it from the passthrough"
            )
        if option and any(
            flag.startswith(option) and option != flag
            for flag in SINGLETON_REQUIRED_EVALUATOR_FLAGS
        ):
            raise SentinelError(
                f"abbreviated evaluator flag {token!r} is forbidden; spell it exactly"
            )
    for flag in SINGLETON_REQUIRED_EVALUATOR_FLAGS:
        if arguments.count(flag) != 1:
            raise SentinelError(
                f"evaluator arguments require {flag} exactly once using separate flag/value tokens"
            )
    require_flag_value(arguments, "--target-source", "bank")
    require_flag_value(arguments, "--exam-bank")
    require_flag_value(arguments, "--exam-schedule-json")
    require_flag_value(arguments, "--mjcf")
    require_flag_value(arguments, "--motion-files")
    return arguments


def reserve_output_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise SentinelError(f"no-clobber output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_milestone_names(labels: Sequence[str]) -> None:
    """Reject child names that alias another milestone or a top-level receipt."""

    aliases: dict[str, str] = {}
    reserved = {name.casefold() for name in RESERVED_OUTPUT_NAMES}
    for label in labels:
        alias = label.casefold()
        if alias in reserved:
            raise SentinelError(
                f"milestone label {label!r} is reserved for a top-level receipt"
            )
        if alias in aliases:
            raise SentinelError(
                f"milestone labels {aliases[alias]!r} and {label!r} alias the same "
                "case-insensitive filesystem namespace"
            )
        aliases[alias] = label


def reserve_milestone_directories(
    output_dir: Path, labels: Sequence[str]
) -> dict[str, Path]:
    """Atomically claim every empty child namespace before any evaluator starts."""

    validate_milestone_names(labels)
    result = {}
    for label in labels:
        path = output_dir / label
        path.mkdir(exist_ok=False)
        result[label] = path
    return result


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _canonical_contract_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_addressed_contract_error(
    value: Any, *, kind: str, label: str
) -> Optional[str]:
    if not isinstance(value, dict):
        return f"{label}: not an object"
    body = dict(value)
    declared = body.pop("sha256", "")
    if body.get("kind") != kind:
        return f"{label}: kind {body.get('kind')!r} != {kind!r}"
    if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
        return f"{label}: missing/invalid sha256"
    try:
        actual = _canonical_contract_sha256(body)
    except (TypeError, ValueError) as exc:
        return f"{label}: non-canonical payload ({exc})"
    if actual != declared:
        return f"{label}: internal sha256 mismatch"
    return None


def _authority_blockers(authority: dict[str, Any], table: dict[str, Any]) -> list[str]:
    blockers = []
    if not authority.get("formal_execution_contract_ok", False):
        blockers.append("formal_execution_contract_false")
    if not authority.get("evaluation_contract_exact_at_launch", False):
        blockers.append("evaluation_contract_inexact")
    if authority.get("velocity_limit_proxy_allowed", True):
        blockers.append("post_integration_velocity_proxy")
    if authority.get("implicit_effort_proxy_nonexact", True):
        blockers.append("implicit_effort_proxy_nonexact")
    if not table.get("available", False):
        blockers.append("robot_table_contact_sensor_unavailable")
    if not table.get("physx_sensor_semantics_exact", False):
        blockers.append("robot_table_contact_semantics_not_physx_exact")
    certificate = authority.get(
        "mujoco_physx_table_sensor_parity_certificate_sha256"
    )
    accepted_certificate = (
        ACCEPTED_MUJOCO_PHYSX_TABLE_SENSOR_PARITY_CERTIFICATE_SHA256
    )
    if (
        not isinstance(accepted_certificate, str)
        or not re.fullmatch(r"[0-9a-f]{64}", accepted_certificate)
        or certificate != accepted_certificate
    ):
        blockers.append("mujoco_physx_table_sensor_parity_uncertified")
    if not authority.get("table_hit_terminates_episode", False):
        blockers.append("robot_hit_table_not_fail_closed")
    return blockers


def _execution_contract_violations(summary: dict[str, Any]) -> list[str]:
    """Verify that authority/table facts are inseparable from the full execution receipt."""

    violations = []
    execution = summary.get("execution_contract")
    authority = summary.get("success_score_authority")
    table = summary.get("robot_table_contact_contract")
    for value, kind, label in (
        (execution, "hope_mujoco_bank_execution_contract", "execution_contract"),
        (authority, "hope_mujoco_success_score_authority", "success_score_authority"),
        (table, "hope_mujoco_robot_table_contact_binding", "robot_table_contact_contract"),
    ):
        error = _content_addressed_contract_error(value, kind=kind, label=label)
        if error:
            violations.append(error)
    if violations:
        return violations

    execution = dict(execution)
    authority = dict(authority)
    table = dict(table)
    if summary.get("execution_contract_sha256") != execution["sha256"]:
        violations.append("top-level execution_contract_sha256 disagrees with embedded contract")
    if summary.get("success_score_authority_sha256") != authority["sha256"]:
        violations.append("top-level authority sha256 disagrees with embedded contract")
    if summary.get("robot_table_contact_contract_sha256") != table["sha256"]:
        violations.append("top-level table-contact sha256 disagrees with embedded contract")
    nested_authority = execution.get("success_score_authority")
    nested_table = execution.get("robot_table_contact")
    nested_authority_error = _content_addressed_contract_error(
        nested_authority,
        kind="hope_mujoco_success_score_authority",
        label="execution_contract.success_score_authority",
    )
    nested_table_error = _content_addressed_contract_error(
        nested_table,
        kind="hope_mujoco_robot_table_contact_binding",
        label="execution_contract.robot_table_contact",
    )
    if nested_authority_error:
        violations.append(nested_authority_error)
    if nested_table_error:
        violations.append(nested_table_error)
    if (
        not isinstance(nested_authority, dict)
        or nested_authority != authority
    ):
        violations.append("authority is not bound inside the full execution contract")
    if not isinstance(nested_table, dict) or nested_table != table:
        violations.append("table-contact receipt is not bound inside the full execution contract")
    if authority.get("table_contact_contract_sha256") != table["sha256"]:
        violations.append("authority names a different table-contact receipt")
    protocol_raw = execution.get("protocol_semantics")
    if isinstance(protocol_raw, dict):
        protocol = dict(protocol_raw)
    else:
        protocol = {}
        violations.append("execution_contract.protocol_semantics is not an object")
    if (
        execution.get("velocity_limit_proxy_allowed")
        is not authority.get("velocity_limit_proxy_allowed")
        or execution.get("implicit_effort_proxy_nonexact")
        is not authority.get("implicit_effort_proxy_nonexact")
        or execution.get("robot_hit_table_terminates_episode")
        is not authority.get("table_hit_terminates_episode")
        or protocol.get("formal_bank_execution_metadata_validated", False)
        is not authority.get("formal_execution_contract_ok")
        or summary.get("evaluation_contract_exact")
        is not authority.get("evaluation_contract_exact_at_launch")
    ):
        violations.append("authority disagrees with execution/runtime launch facts")
    expected_blockers = _authority_blockers(authority, table)
    if authority.get("blockers") != expected_blockers:
        violations.append("authority blocker list contradicts its bound launch facts")
    expected_authorized = not expected_blockers
    if (
        authority.get("plant_parity_valid_at_launch") is not expected_authorized
        or authority.get("success_scores_authorized") is not expected_authorized
    ):
        violations.append("authority booleans contradict the fail-closed blocker derivation")
    if summary.get("success_scores_authorized") is not expected_authorized:
        violations.append("top-level success authority disagrees with embedded authority")
    results_raw = summary.get("results")
    if not isinstance(results_raw, list):
        violations.append("top-level results is not an array")
        results_raw = []
    for index, result in enumerate(results_raw):
        if not isinstance(result, dict):
            violations.append(f"results[{index}] is not an object")
            continue
        result_authority = result.get("success_score_authority") or {}
        if (
            not isinstance(result_authority, dict)
            or result_authority.get("sha256") != authority["sha256"]
        ):
            violations.append(f"results[{index}] carries a different success authority")
        if result.get("success_scores_authorized") is not expected_authorized:
            violations.append(f"results[{index}] success authority boolean disagrees")
    return violations


def _finite_float(value: Any, default: float = math.inf) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _finite_vector(value: Any, size: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == size
        and all(_finite_float(item) != math.inf for item in value)
    )


def _signed_direction_record_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    scalar_keys = (
        "step",
        "attempt_id",
        "signed_racket_velocity_along_target_mps",
        "racket_velocity_world_x_mps",
        "signed_face_dot",
        "signed_face_error_deg",
        "contact_closing_speed_along_target_face_normal_mps",
    )
    vector_keys = (
        "actual_racket_velocity_w_mps",
        "target_racket_velocity_w_mps",
        "actual_signed_face_normal_w",
        "target_face_normal_w",
        "incoming_ball_velocity_w_mps",
    )
    return bool(
        all(_finite_float(value.get(key)) != math.inf for key in scalar_keys)
        and all(_finite_vector(value.get(key), 3) for key in vector_keys)
        and isinstance(value.get("table_hit_same_step"), bool)
        and isinstance(value.get("absolute_fall_same_step"), bool)
        and value.get("actual_racket_velocity_source")
        == "final_physics_substep_after_mj_step_before_post_step_qvel_proxy"
        and value.get("contact_direction_formula")
        == (
            "dot(actual_racket_velocity - incoming_ball_velocity, "
            "target_face_normal)"
        )
    )


def _nonnegative_int(value: Any, *, maximum: Optional[int] = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return False
    return maximum is None or value <= maximum


def _finite_range(value: Any, *, minimum: float, maximum: float) -> bool:
    number = _finite_float(value)
    return minimum <= number <= maximum


def _direction_receipt_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    first = value.get("actor_first_action")
    raw = value.get("qdes_first_raw")
    applied = value.get("qdes_first_applied")
    actor_max_per_joint = value.get("actor_max_abs_per_joint")
    required_nonnegative_scalars = (
        "actor_first_action_abs_max",
        "actor_max_abs",
        "qvel_pre_step_peak_ratio",
        "qvel_post_step_raw_peak_ratio",
        "qvel_post_proxy_peak_ratio",
    )
    strikes = value.get("signed_direction_strikes")
    return bool(
        _finite_vector(first, 31)
        and _finite_vector(raw, 31)
        and _finite_vector(applied, 31)
        and _finite_vector(actor_max_per_joint, 31)
        and isinstance(strikes, list)
        and all(_signed_direction_record_complete(item) for item in strikes)
        and all(
            _finite_range(value.get(key), minimum=0.0, maximum=math.inf)
            for key in required_nonnegative_scalars
        )
        and _nonnegative_int(value.get("qdes_first_clamp_count"), maximum=31)
        and _finite_range(
            value.get("qdes_clamp_fraction"), minimum=0.0, maximum=1.0
        )
        and _nonnegative_int(value.get("qvel_proxy_control_steps"))
        and _nonnegative_int(value.get("table_hit_control_steps"))
        and _nonnegative_int(value.get("physical_fall_events"))
    )


def _unauthorized_score_violations(value: Any, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            label = str(key)
            child = f"{path}.{label}" if path else label
            scored = label in SUCCESS_SCORE_KEYS or label.startswith(SUCCESS_SCORE_PREFIXES)
            if scored and item is not None:
                violations.append(child)
            elif not scored:
                violations.extend(_unauthorized_score_violations(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(_unauthorized_score_violations(item, f"{path}[{index}]"))
    return violations


def _score_column(label: str) -> bool:
    return label in SUCCESS_SCORE_KEYS or label.startswith(SUCCESS_SCORE_PREFIXES)


def audit_evaluator_csv_artifacts(
    summary: dict[str, Any], *, summary_path: Path
) -> tuple[list[str], list[str]]:
    """Verify evaluator-owned CSV identities and blank unauthorized score cells."""

    integrity = []
    score_leaks = []
    artifacts = dict(summary.get("artifacts") or {})
    root = summary_path.resolve().parent
    for artifact_key in ("per_step_csv", "per_strike_csv", "per_attempt_csv"):
        receipt = artifacts.get(artifact_key)
        if not isinstance(receipt, dict):
            integrity.append(f"{artifact_key}: missing artifact receipt")
            continue
        path = Path(str(receipt.get("path", ""))).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            integrity.append(f"{artifact_key}: cannot resolve artifact ({exc})")
            continue
        if resolved.parent != root:
            integrity.append(f"{artifact_key}: artifact escapes milestone directory")
            continue
        declared = str(receipt.get("sha256", ""))
        actual = sha256_file(resolved)
        if not re.fullmatch(r"[0-9a-f]{64}", declared) or actual != declared:
            integrity.append(f"{artifact_key}: sha256 mismatch")
            continue
        if (
            artifact_key not in ("per_strike_csv", "per_attempt_csv")
            or summary.get("success_scores_authorized", False)
        ):
            continue
        with resolved.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                integrity.append(f"{artifact_key}: missing CSV header")
                continue
            score_columns = [name for name in reader.fieldnames if _score_column(name)]
            for row_index, row in enumerate(reader, start=2):
                for column in score_columns:
                    if row.get(column, "") not in ("", None):
                        score_leaks.append(
                            f"{artifact_key}:{row_index}:{column}"
                        )
    return integrity, score_leaks


def _gate(passed: bool, observed: Any, required: Any, reason: str) -> dict[str, Any]:
    return {
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "reason": str(reason),
    }


def build_stop_gates(
    summary: dict[str, Any],
    *,
    evaluator_exit_code: int,
    max_qdes_clamp_fraction: float,
    qvel_ratio_tolerance: float,
) -> dict[str, Any]:
    """Turn one evaluator summary into explicit, machine-readable stop decisions."""

    if not isinstance(summary, dict):
        raise SentinelError("evaluator summary root must be a JSON object")
    authority_raw = summary.get("success_score_authority")
    table_raw = summary.get("robot_table_contact_contract")
    results_raw = summary.get("results")
    authority = dict(authority_raw) if isinstance(authority_raw, dict) else {}
    table = dict(table_raw) if isinstance(table_raw, dict) else {}
    results = list(results_raw) if isinstance(results_raw, list) else []
    directions = [
        (
            dict(result.get("direction_diagnostics"))
            if isinstance(result, dict)
            and isinstance(result.get("direction_diagnostics"), dict)
            else {}
        )
        for result in results
    ]
    direction_complete = bool(directions) and all(
        _direction_receipt_complete(item) for item in directions
    )
    finite = direction_complete and bool(results) and all(
        bool(item.get("finite", False)) and _finite_numbers(item) for item in directions
    )
    qvel_raw_peak = max(
        (
            _finite_float(item.get("qvel_post_step_raw_peak_ratio"))
            for item in directions
        ),
        default=math.inf,
    )
    qvel_pre_peak = max(
        (_finite_float(item.get("qvel_pre_step_peak_ratio")) for item in directions),
        default=math.inf,
    )
    clamp_fraction = max(
        (_finite_float(item.get("qdes_clamp_fraction")) for item in directions),
        default=math.inf,
    )
    table_hits = sum(
        item["table_hit_control_steps"]
        for item in directions
        if _nonnegative_int(item.get("table_hit_control_steps"))
    )
    falls = sum(
        item["physical_fall_events"]
        for item in directions
        if _nonnegative_int(item.get("physical_fall_events"))
    )
    signed_samples = sum(
        len(item.get("signed_direction_strikes") or []) for item in directions
    )
    unauthorized_violations = (
        _unauthorized_score_violations(summary)
        if not authority.get("success_scores_authorized", False)
        else []
    )
    contract_violations = _execution_contract_violations(summary)
    gates = {
        "evaluator_completed": _gate(
            evaluator_exit_code == 0,
            evaluator_exit_code,
            0,
            "the evaluator must complete and publish a summary",
        ),
        "content_addressed_execution_contract": _gate(
            not contract_violations,
            contract_violations,
            [],
            "table, score authority and evaluator facts must be bound by one valid execution SHA",
        ),
        "direction_receipt_complete": _gate(
            direction_complete,
            direction_complete,
            True,
            "every milestone must expose actor/qdes/qvel/signed-direction/table/fall fields",
        ),
        "finite_direction_evidence": _gate(
            finite,
            finite,
            True,
            "actor/qdes/qvel/racket direction evidence must contain no NaN/Inf",
        ),
        "table_contact_sensor_available": _gate(
            bool(table.get("available", False)),
            table.get("available", False),
            True,
            "robot_hit_table requires the bound in-memory table top and racket geometry",
        ),
        "no_robot_table_hit": _gate(
            table_hits == 0,
            table_hits,
            0,
            "any force-qualified robot/table contact stops promotion",
        ),
        "no_physical_fall": _gate(
            falls == 0,
            falls,
            0,
            "any tilt/root-height fall stops promotion",
        ),
        "raw_qvel_within_bound": _gate(
            max(qvel_pre_peak, qvel_raw_peak) <= 1.0 + qvel_ratio_tolerance,
            {"pre_peak_ratio": qvel_pre_peak, "post_raw_peak_ratio": qvel_raw_peak},
            f"<= {1.0 + qvel_ratio_tolerance:g}",
            "post-proxy qvel is not evidence; only pre and raw post-substep ratios count",
        ),
        "qdes_clamp_fraction": _gate(
            clamp_fraction <= max_qdes_clamp_fraction,
            clamp_fraction,
            f"<= {max_qdes_clamp_fraction:g}",
            "large clamp dependence is a deployment-direction stop",
        ),
        "signed_contact_direction_observed": _gate(
            signed_samples > 0,
            signed_samples,
            "> 0",
            "at least one exact-frame signed racket/face/contact-direction sample is required",
        ),
        "plant_parity_valid": _gate(
            bool(authority.get("plant_parity_valid_at_launch", False)),
            {
                "valid": authority.get("plant_parity_valid_at_launch", False),
                "blockers": authority.get("blockers", []),
            },
            True,
            "MuJoCo success scores are forbidden while any plant-contract blocker exists",
        ),
        "success_score_authorized": _gate(
            bool(authority.get("success_scores_authorized", False)),
            authority.get("success_scores_authorized", False),
            True,
            "only the immutable execution-contract authority may permit a success score",
        ),
        "unauthorized_scores_suppressed": _gate(
            not unauthorized_violations,
            unauthorized_violations,
            [],
            "unauthorized pass/return/composite fields must be null, never numeric",
        ),
    }
    return {
        "stop_required": any(not item["pass"] for item in gates.values()),
        "gates": gates,
    }


def milestone_report(
    *,
    label: str,
    onnx: Path,
    summary_path: Path,
    evaluator_exit_code: int,
    command: Sequence[str],
    expected_onnx_sha256: str,
    expected_evaluator_sha256: str,
    max_qdes_clamp_fraction: float,
    qvel_ratio_tolerance: float,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise SentinelError(
            f"evaluator did not publish summary for {label}: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise SentinelError(
            f"evaluator summary root for {label} must be a JSON object"
        )
    decision = build_stop_gates(
        summary,
        evaluator_exit_code=evaluator_exit_code,
        max_qdes_clamp_fraction=max_qdes_clamp_fraction,
        qvel_ratio_tolerance=qvel_ratio_tolerance,
    )
    onnx_sha256_after = sha256_file(onnx)
    evaluator_source = Path(str(command[1])).expanduser().resolve()
    evaluator_sha256_after = sha256_file(evaluator_source)
    input_artifacts = (
        summary.get("input_artifacts")
        if isinstance(summary.get("input_artifacts"), dict)
        else {}
    )
    onnx_artifact = (
        input_artifacts.get("onnx")
        if isinstance(input_artifacts.get("onnx"), dict)
        else {}
    )
    evaluator_artifact = (
        input_artifacts.get("evaluator_source")
        if isinstance(input_artifacts.get("evaluator_source"), dict)
        else {}
    )
    onnx_identity_ok = bool(
        summary.get("onnx_sha256") == expected_onnx_sha256
        and onnx_artifact.get("sha256") == expected_onnx_sha256
        and onnx_sha256_after == expected_onnx_sha256
    )
    decision["gates"]["onnx_identity"] = _gate(
        onnx_identity_ok,
        {
            "summary": summary.get("onnx_sha256"),
            "artifact": onnx_artifact.get("sha256"),
            "path_after": onnx_sha256_after,
        },
        expected_onnx_sha256,
        "summary/artifact/path must all name the exact ONNX bytes hashed before spawn",
    )
    execution_contract = (
        summary.get("execution_contract")
        if isinstance(summary.get("execution_contract"), dict)
        else {}
    )
    evaluator_identity_ok = bool(
        execution_contract.get("evaluator_source_sha256")
        == expected_evaluator_sha256
        and evaluator_artifact.get("sha256") == expected_evaluator_sha256
        and evaluator_sha256_after == expected_evaluator_sha256
    )
    decision["gates"]["evaluator_identity"] = _gate(
        evaluator_identity_ok,
        {
            "execution_contract": execution_contract.get(
                "evaluator_source_sha256"
            ),
            "artifact": evaluator_artifact.get("sha256"),
            "path_after": evaluator_sha256_after,
        },
        expected_evaluator_sha256,
        "execution/artifact/path must all name the evaluator bytes hashed before spawn",
    )
    csv_integrity, csv_score_leaks = audit_evaluator_csv_artifacts(
        summary, summary_path=summary_path
    )
    decision["gates"]["csv_artifact_integrity"] = _gate(
        not csv_integrity,
        csv_integrity,
        [],
        "all evaluator CSVs must remain inside the milestone namespace and match their SHA",
    )
    decision["gates"]["unauthorized_csv_scores_suppressed"] = _gate(
        not csv_score_leaks,
        csv_score_leaks,
        [],
        "unauthorized pass/return/composite CSV cells must be blank",
    )
    decision["stop_required"] = any(
        not item["pass"] for item in decision["gates"].values()
    )
    return {
        "label": label,
        "onnx": str(onnx),
        "onnx_sha256": expected_onnx_sha256,
        "onnx_path_sha256_after": onnx_sha256_after,
        "evaluator_source_sha256": expected_evaluator_sha256,
        "evaluator_source_sha256_after": evaluator_sha256_after,
        "evaluator_summary": str(summary_path),
        "evaluator_summary_sha256": sha256_file(summary_path),
        "evaluator_exit_code": int(evaluator_exit_code),
        "command": list(command),
        "success_score_authority": summary.get("success_score_authority"),
        "robot_table_contact_contract": summary.get("robot_table_contact_contract"),
        "results": [
            {
                "mode": result.get("mode"),
                "direction_diagnostics": result.get("direction_diagnostics"),
                "term_breakdown": result.get("term_breakdown"),
                "robot_table_contact_steps": result.get("robot_table_contact_steps"),
                "fell": result.get("fell"),
            }
            for result in summary.get("results", [])
        ],
        **decision,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, milestones: Iterable[dict[str, Any]]) -> None:
    columns = [
        "label",
        "mode",
        "onnx_sha256",
        "actor_first_action_abs_max",
        "actor_max_abs",
        "qdes_first_clamp_count",
        "qdes_clamp_fraction",
        "qvel_pre_step_peak_ratio",
        "qvel_post_step_raw_peak_ratio",
        "qvel_post_proxy_peak_ratio",
        "qvel_proxy_control_steps",
        "signed_direction_strikes",
        "table_hit_control_steps",
        "physical_fall_events",
        "plant_parity_valid",
        "success_scores_authorized",
        "stop_required",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for milestone in milestones:
            authority = milestone.get("success_score_authority") or {}
            for result in milestone.get("results", []):
                direction = result.get("direction_diagnostics") or {}
                writer.writerow({
                    "label": milestone["label"],
                    "mode": result.get("mode"),
                    "onnx_sha256": milestone["onnx_sha256"],
                    "actor_first_action_abs_max": direction.get(
                        "actor_first_action_abs_max"
                    ),
                    "actor_max_abs": direction.get("actor_max_abs"),
                    "qdes_first_clamp_count": direction.get(
                        "qdes_first_clamp_count"
                    ),
                    "qdes_clamp_fraction": direction.get("qdes_clamp_fraction"),
                    "qvel_pre_step_peak_ratio": direction.get(
                        "qvel_pre_step_peak_ratio"
                    ),
                    "qvel_post_step_raw_peak_ratio": direction.get(
                        "qvel_post_step_raw_peak_ratio"
                    ),
                    "qvel_post_proxy_peak_ratio": direction.get(
                        "qvel_post_proxy_peak_ratio"
                    ),
                    "qvel_proxy_control_steps": direction.get(
                        "qvel_proxy_control_steps"
                    ),
                    "signed_direction_strikes": len(
                        direction.get("signed_direction_strikes") or []
                    ),
                    "table_hit_control_steps": direction.get(
                        "table_hit_control_steps"
                    ),
                    "physical_fall_events": direction.get("physical_fall_events"),
                    "plant_parity_valid": authority.get(
                        "plant_parity_valid_at_launch"
                    ),
                    "success_scores_authorized": authority.get(
                        "success_scores_authorized"
                    ),
                    "stop_required": milestone["stop_required"],
                })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    result.add_argument(
        "--milestone",
        action="append",
        type=parse_milestone,
        required=True,
        metavar="LABEL=/ABS/POLICY.ONNX",
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name("mujoco_eval_onnx.py"),
    )
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--max-qdes-clamp-fraction", type=float, default=0.25)
    result.add_argument("--qvel-ratio-tolerance", type=float, default=1e-9)
    result.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not math.isfinite(args.max_qdes_clamp_fraction) or not (
            0.0 <= args.max_qdes_clamp_fraction <= 1.0
        ):
            raise SentinelError("--max-qdes-clamp-fraction must be finite in [0,1]")
        if not math.isfinite(args.qvel_ratio_tolerance) or args.qvel_ratio_tolerance < 0.0:
            raise SentinelError("--qvel-ratio-tolerance must be finite and non-negative")
        evaluator = args.evaluator.expanduser().resolve()
        if not evaluator.is_file():
            raise SentinelError(f"evaluator not found: {evaluator}")
        evaluator_sha256_at_launch = sha256_file(evaluator)
        evaluator_args = validate_evaluator_arguments(args.evaluator_args)
        labels = [label for label, _path in args.milestone]
        validate_milestone_names(labels)
        for label, path in args.milestone:
            if not path.is_file():
                raise SentinelError(f"milestone ONNX not found for {label}: {path}")
        output_dir = reserve_output_directory(args.output_dir)
        milestone_directories = reserve_milestone_directories(output_dir, labels)

        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        reports = []
        evaluator_failures = 0
        for label, onnx in args.milestone:
            onnx_sha256_at_launch = sha256_file(onnx)
            milestone_dir = milestone_directories[label]
            command = [
                args.python,
                str(evaluator),
                "--onnx",
                str(onnx),
                "--expected-onnx-sha256",
                onnx_sha256_at_launch,
                "--expected-evaluator-sha256",
                evaluator_sha256_at_launch,
                "--out-dir",
                str(milestone_dir),
                "--noise-scales",
                "0.0",
                "--allow-inexact-contract",
                "--allow-velocity-limit-proxy",
                "--with-table-obstacle",
                "--no-realtime",
                *evaluator_args,
            ]
            completed = subprocess.run(
                command,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if milestone_dir.is_symlink() or not milestone_dir.is_dir():
                raise SentinelError(
                    f"reserved milestone namespace was replaced: {milestone_dir}"
                )
            with (milestone_dir / "sentinel_stdout.log").open(
                "x", encoding="utf-8"
            ) as handle:
                handle.write(completed.stdout)
            with (milestone_dir / "sentinel_stderr.log").open(
                "x", encoding="utf-8"
            ) as handle:
                handle.write(completed.stderr)
            summary_path = milestone_dir / "mujoco_sim2sim_summary.json"
            if completed.returncode != 0 or not summary_path.is_file():
                evaluator_failures += 1
                reports.append({
                    "label": label,
                    "onnx": str(onnx),
                    "onnx_sha256": onnx_sha256_at_launch,
                    "evaluator_source_sha256": evaluator_sha256_at_launch,
                    "evaluator_exit_code": int(completed.returncode),
                    "command": command,
                    "stop_required": True,
                    "gates": {
                        "evaluator_completed": _gate(
                            False,
                            completed.returncode,
                            0,
                            "evaluator failed before a complete machine-readable receipt",
                        )
                    },
                    "results": [],
                })
                continue
            try:
                reports.append(milestone_report(
                    label=label,
                    onnx=onnx,
                    summary_path=summary_path,
                    evaluator_exit_code=completed.returncode,
                    command=command,
                    expected_onnx_sha256=onnx_sha256_at_launch,
                    expected_evaluator_sha256=evaluator_sha256_at_launch,
                    max_qdes_clamp_fraction=args.max_qdes_clamp_fraction,
                    qvel_ratio_tolerance=args.qvel_ratio_tolerance,
                ))
            except (
                OSError,
                SentinelError,
                TypeError,
                ValueError,
                AttributeError,
                json.JSONDecodeError,
            ) as exc:
                evaluator_failures += 1
                reports.append({
                    "label": label,
                    "onnx": str(onnx),
                    "onnx_sha256": onnx_sha256_at_launch,
                    "evaluator_source_sha256": evaluator_sha256_at_launch,
                    "evaluator_exit_code": int(completed.returncode),
                    "command": command,
                    "stop_required": True,
                    "gates": {
                        "summary_valid": _gate(
                            False,
                            f"{type(exc).__name__}: {exc}",
                            "valid fail-closed evaluator summary",
                            "malformed evaluator evidence cannot authorize a milestone",
                        )
                    },
                    "results": [],
                })

        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hope_mujoco_checkpoint_direction_sentinel",
            "gpu_policy": {
                "cuda_visible_devices": "",
                "onnxruntime_provider": "CPUExecutionProvider (enforced by evaluator)",
                "renderer": "disabled",
            },
            "no_clobber": True,
            "evaluator_source": str(evaluator),
            "evaluator_source_sha256": evaluator_sha256_at_launch,
            "max_qdes_clamp_fraction": args.max_qdes_clamp_fraction,
            "qvel_ratio_tolerance": args.qvel_ratio_tolerance,
            "milestones": reports,
            "stop_required": any(item["stop_required"] for item in reports),
        }
        _write_json(output_dir / "direction_sentinel.json", report)
        _write_csv(output_dir / "direction_sentinel.csv", reports)
        if evaluator_failures:
            return 2
        return 3 if report["stop_required"] else 0
    except (
        OSError,
        SentinelError,
        TypeError,
        ValueError,
        AttributeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[direction-sentinel] FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
