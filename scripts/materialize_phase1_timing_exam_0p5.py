#!/usr/bin/env python3
"""Materialize and validate the immutable Phase-1 0.5 s timing exam.

The tracked JSON is a preregistration, not a paper: the exact signed-face K100
schedule is a private runtime artifact.  This tool refuses to invent question
ids.  It verifies the source schedule's file, semantic, and question-order
SHA-256 values before deriving one timing row for every scheduled question.

The tool is deliberately simulator-free.  It never starts a trainer, judge,
planner, runner, simulator, or hardware process.  ``materialize`` writes one
new paper with O_EXCL; the other commands are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


SPEC_ID = "phase1-timing-exam-0p5-k100-v1"
ARTIFACT_TYPE = "phase1-timing-exam-paper"
RESULT_ARTIFACT_TYPE = "phase1-timing-exam-result-ledger"
SOURCE_ARTIFACT_TYPE = "bank-exam-schedule"
SIDE_ORDER = ("forehand", "backhand")
VALIDATOR_REPO_PATH = "scripts/materialize_phase1_timing_exam_0p5.py"
MATERIALIZE_CONFIRMATION = "SIM_ONLY_MATERIALIZE_ONE_PHASE1_TIMING_EXAM_PAPER"
CONVERT_ISAAC_CONFIRMATION = "SIM_ONLY_CONVERT_ONE_ISAAC_TIMING_SCORECARD"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ISAAC_SCORECARD_SCHEMA = "hope.isaac-bank-exam.v1"
ISAAC_DIAGNOSTIC_REASONS = {
    "timing paper uniform-phase time laws are not TOPP/dynamics-certified",
    "Isaac analytic timing lane has no self-hit or illegal table/net-contact instrumentation",
    "fixed-question policy exam bypasses the production planner; infeasibility is unmeasured",
}

EXPECTED_SOURCE_SCHEDULE = {
    "path": (
        "/workspace/codexschema/phase1_signed_face_rescue_20260713/"
        "papers/signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json"
    ),
    "bytes": 20237,
    "file_sha256": "f2777dcd02080ba68b839c76ea9d3f14c938457c9bc01b5692fe86ae59157ec7",
    "semantic_sha256": "3ca4bdba7f4acbe6f211d90e95305fc4a9459c118e4220c9683060c0a6723365",
    "question_id_order_sha256": (
        "09f778f2afd7888069ac75aabee3bf19deda015acbbad843ccdb70ca39548bd0"
    ),
    "bank_sha256": "60e1a7ade72eaf64e17a1b83795125551f08c6699c8a3cc3c269500d8e6cd1ca",
    "bank_source_family_sha256": (
        "9603a1788eb17ce03598cdde4efff946039613cf61fcc686f90a385706dba9db"
    ),
    "scheduled_attempts": 100,
    "per_side": {"forehand": 50, "backhand": 50},
}

FROZEN_TIME_LAWS = [
    {
        "time_law_id": "v4rg-uniform-phase-forehand-0p5-v1",
        "side": "forehand",
        "native_contact_ticks": 66,
        "speed_scale": 2.64,
        "contact_tts_ticks": 25,
        "contact_tts_seconds": 0.5,
        "topp_or_dynamics_certified": False,
    },
    {
        "time_law_id": "v4rg-uniform-phase-backhand-0p5-v1",
        "side": "backhand",
        "native_contact_ticks": 45,
        "speed_scale": 1.8,
        "contact_tts_ticks": 25,
        "contact_tts_seconds": 0.5,
        "topp_or_dynamics_certified": False,
    },
]

FROZEN_PAPER = {
    "policy_rate_hz": 50,
    "baseline": {
        "human_name": "0.5 second zero-velocity ready-state timing baseline",
        "tts_seconds": 0.5,
        "tts_ticks": 25,
        "initial_state_id": "nominal-frame0-zero-velocity-v1",
        "expected_feasible": None,
        "feasibility_status": "hypothesis_not_certified",
        "scheduled_attempts": 100,
        "per_side": {"forehand": 50, "backhand": 50},
        "source_hold_steps_are_replaced": True,
        "source_attempt_seed_and_question_order_are_preserved": True,
    },
    "time_laws": FROZEN_TIME_LAWS,
    "tts_sweep_plan": {
        "tts_seconds": [0.9, 0.7, 0.5, 0.4, 0.33, 0.25],
        "screen_questions_per_side": 10,
        "screen_role": "diagnostic_only_no_stop_or_promote",
        "decision_questions_per_side": 50,
        "decision_rule": (
            "run 0.5 s and the two screen bins bracketing the first failure transition"
        ),
        "sub_0p5_role": "late_ball_or_infeasible_boundary_diagnostic",
        "formal_shortest_time_claim_authorized": False,
    },
}

FROZEN_SCORING = {
    "denominator": {
        "policy": "all_scheduled_attempts",
        "aggregate": 100,
        "forehand": 50,
        "backhand": 50,
        "missing_invalid_reset_or_infeasible_attempt_counts_as_failure": True,
        "censoring_allowed": False,
    },
    "per_side_pass_count": 31,
    "per_side_total": 50,
    "one_sided_wilson_confidence": 0.95,
    "one_sided_wilson_lower_bound_at_31_of_50": 0.5036944597,
    "composite": {
        "returned_required": True,
        "position_error_m_strict_lt": 0.075,
        "velocity_error_mps_strict_lt": 0.5,
        "signed_normal_error_deg_strict_lt": 15.0,
        "unsigned_or_oriented_plane_fallback_allowed": False,
    },
    "safety_zero_tolerance_fields": [
        "physical_fall",
        "self_hit",
        "illegal_table_or_net_contact",
        "reset_or_teleport",
        "deadline_shifted",
    ],
    "formal_gate_requires_evaluation_contract_exact": True,
}

FROZEN_EXECUTION = {
    "materializer_launches_no_evaluator": True,
    "output_must_not_exist": True,
    "atomic_no_replace": True,
    "isaac_diagnostic_evaluator_authorized": True,
    "isaac_diagnostic_requires_allow_inexact_contract": True,
    "trainer_authorized": False,
    "judge_authorized": False,
    "planner_or_runner_authorized": False,
    "stop_or_promote_authorized": False,
    "deployment_authorized": False,
    "real_robot_authorized": False,
}


class ContractError(ValueError):
    """The timing-paper contract is incomplete, ambiguous, or inconsistent."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not finite canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read strict JSON {label} {path}: {exc}") from exc


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ContractError(f"{label} must be one lowercase SHA-256")
    return value


def require_plain_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ContractError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def require_nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContractError(f"{label} must be a finite non-negative number")
    return result


def require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ContractError(f"{label} schema changed: {actual}")
    return value


def _content_sha(document: Mapping[str, Any], field: str) -> str:
    content = dict(document)
    content.pop(field, None)
    return canonical_sha256(content)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _verify_validator_binding(binding: Any, *, root: Path) -> None:
    require_exact_keys(binding, {"repo_path", "bytes", "sha256"}, "validator binding")
    if binding["repo_path"] != VALIDATOR_REPO_PATH:
        raise ContractError("validator binding path changed")
    size = require_plain_int(binding["bytes"], "validator binding bytes", minimum=1)
    require_sha(binding["sha256"], "validator binding sha256")
    path = (root / binding["repo_path"]).resolve()
    if path != Path(__file__).resolve() or not path.is_file():
        raise ContractError("validator binding does not resolve to this tool")
    if path.stat().st_size != size or sha256_file(path) != binding["sha256"]:
        raise ContractError("validator source bytes differ from the timing contract")


def validate_spec_document(
    value: Any,
    *,
    root: Path,
    verify_production_source: bool = True,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract_id",
        "status",
        "recorded_local_date",
        "human_owner",
        "executor",
        "purpose",
        "simulation_only",
        "real_robot_commands_forbidden",
        "source_schedule",
        "source_bindings",
        "paper",
        "scoring",
        "execution",
        "contract_content_sha256",
    }
    document = dict(require_exact_keys(value, expected_keys, "timing exam spec"))
    if document["schema_version"] != 1 or document["contract_id"] != SPEC_ID:
        raise ContractError("timing exam spec schema/id changed")
    if document["status"] != "preregistered_materialization_not_run":
        raise ContractError("tracked timing exam spec may not claim a runtime result")
    if document["recorded_local_date"] != "2026-07-16":
        raise ContractError("timing exam spec date changed")
    if document["human_owner"] != "Franco" or document["executor"] != "Codex":
        raise ContractError("timing exam ownership changed")
    if not isinstance(document["purpose"], str) or "0.5" not in document["purpose"]:
        raise ContractError("timing exam purpose must describe the 0.5 second gate")
    if document["simulation_only"] is not True or document["real_robot_commands_forbidden"] is not True:
        raise ContractError("timing exam spec must remain simulation-only")
    require_sha(document["contract_content_sha256"], "contract_content_sha256")
    if _content_sha(document, "contract_content_sha256") != document["contract_content_sha256"]:
        raise ContractError("timing exam contract_content_sha256 mismatch")

    source = require_exact_keys(
        document["source_schedule"], set(EXPECTED_SOURCE_SCHEDULE), "source_schedule"
    )
    require_sha(source["file_sha256"], "source schedule file SHA")
    require_sha(source["semantic_sha256"], "source schedule semantic SHA")
    require_sha(source["question_id_order_sha256"], "source question-order SHA")
    require_sha(source["bank_sha256"], "source bank SHA")
    require_sha(source["bank_source_family_sha256"], "source bank family SHA")
    require_plain_int(source["bytes"], "source schedule bytes", minimum=1)
    if source["scheduled_attempts"] != 100 or source["per_side"] != {"forehand": 50, "backhand": 50}:
        raise ContractError("source schedule must remain K100 with 50 questions per side")
    if not isinstance(source["path"], str) or not source["path"].startswith("/workspace/"):
        raise ContractError("source schedule path must remain an absolute runtime artifact path")
    if verify_production_source and dict(source) != EXPECTED_SOURCE_SCHEDULE:
        raise ContractError("source schedule binding differs from the accepted signed-face K100")

    bindings = require_exact_keys(document["source_bindings"], {"validator"}, "source_bindings")
    _verify_validator_binding(bindings["validator"], root=root)
    if document["paper"] != FROZEN_PAPER:
        raise ContractError("0.5 second paper or timing sweep contract changed")
    if document["scoring"] != FROZEN_SCORING:
        raise ContractError("all-attempt scoring or safety gate changed")
    if document["execution"] != FROZEN_EXECUTION:
        raise ContractError("timing execution authorization changed")
    return document


def load_spec(
    path: Path,
    *,
    root: Path,
    expected_file_sha256: str,
    verify_production_source: bool = True,
) -> dict[str, Any]:
    require_sha(expected_file_sha256, "expected spec file SHA")
    actual = sha256_file(path)
    if actual != expected_file_sha256:
        raise ContractError(f"spec file SHA mismatch: {actual} != {expected_file_sha256}")
    return validate_spec_document(
        load_json(path, "timing exam spec"),
        root=root,
        verify_production_source=verify_production_source,
    )


def _validate_source_schedule_document(
    value: Any,
    *,
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    keys = {
        "artifact_type",
        "bank_schema_version",
        "bank_sha256",
        "clip_order",
        "question_counts",
        "per_clip_quota",
        "schedule_seed",
        "hold_range",
        "hold_semantics",
        "no_wrap",
        "items",
        "schema_version",
        "schedule_sha256",
    }
    schedule = dict(require_exact_keys(value, keys, "source BankExam schedule"))
    if schedule["schema_version"] != 3 or schedule["bank_schema_version"] != 3:
        raise ContractError("source BankExam schedule schema changed")
    if schedule["artifact_type"] != SOURCE_ARTIFACT_TYPE:
        raise ContractError("source schedule artifact_type changed")
    if schedule["bank_sha256"] != source_contract["bank_sha256"]:
        raise ContractError("source schedule bank SHA mismatch")
    if schedule["clip_order"] != list(SIDE_ORDER):
        raise ContractError("source schedule clip order must be forehand/backhand")
    if (
        not isinstance(schedule["question_counts"], list)
        or len(schedule["question_counts"]) != 2
        or any(
            require_plain_int(count, f"question_counts[{index}]", minimum=50) < 50
            for index, count in enumerate(schedule["question_counts"])
        )
    ):
        raise ContractError("source schedule must expose at least 50 questions per side")
    if schedule["per_clip_quota"] != 50 or schedule["schedule_seed"] != 0:
        raise ContractError("source schedule quota/seed changed")
    if schedule["hold_range"] != [0, 100] or schedule["hold_semantics"] != "stand-policy-actions-then-raw-frame0-v1":
        raise ContractError("source schedule hold contract changed")
    if schedule["no_wrap"] is not True:
        raise ContractError("source schedule must remain no-wrap")
    require_sha(schedule["schedule_sha256"], "declared schedule semantic SHA")
    if canonical_sha256({key: item for key, item in schedule.items() if key != "schedule_sha256"}) != schedule["schedule_sha256"]:
        raise ContractError("source schedule declared semantic SHA mismatch")
    if schedule["schedule_sha256"] != source_contract["semantic_sha256"]:
        raise ContractError("source schedule semantic SHA differs from the timing spec")

    items = schedule["items"]
    if not isinstance(items, list) or len(items) != source_contract["scheduled_attempts"]:
        raise ContractError("source schedule item count differs from K100")
    item_keys = {
        "schedule_index",
        "clip",
        "bank_row",
        "question_id",
        "hold_steps",
        "attempt_seed",
        "repeat",
    }
    seen_ids: set[str] = set()
    side_counts: Counter[str] = Counter()
    for index, raw in enumerate(items):
        item = require_exact_keys(raw, item_keys, f"source schedule item {index}")
        if require_plain_int(item["schedule_index"], f"item {index} schedule_index") != index:
            raise ContractError("source schedule indices must be contiguous and ordered")
        clip = require_plain_int(item["clip"], f"item {index} clip", maximum=1)
        side = SIDE_ORDER[clip]
        bank_row = require_plain_int(item["bank_row"], f"item {index} bank_row")
        if bank_row >= schedule["question_counts"][clip]:
            raise ContractError(f"source schedule item {index} bank_row is out of range")
        question_id = item["question_id"]
        prefix = f"{side}:"
        if (
            not isinstance(question_id, str)
            or not question_id.startswith(prefix)
            or not SHA_RE.fullmatch(question_id[len(prefix) :])
        ):
            raise ContractError(f"source schedule item {index} question_id is invalid")
        if question_id in seen_ids:
            raise ContractError("source schedule question ids must be unique")
        seen_ids.add(question_id)
        side_counts[side] += 1
        require_plain_int(item["hold_steps"], f"item {index} hold_steps", maximum=100)
        require_plain_int(item["attempt_seed"], f"item {index} attempt_seed", maximum=(1 << 64) - 1)
        if item["repeat"] != 0:
            raise ContractError("source schedule repeat must remain zero")
    if dict(side_counts) != source_contract["per_side"]:
        raise ContractError("source schedule side counts differ from the 50/50 contract")
    order_sha = canonical_sha256([item["question_id"] for item in items])
    if order_sha != source_contract["question_id_order_sha256"]:
        raise ContractError("source schedule question-order SHA mismatch")
    return schedule


def load_source_schedule(path: Path, *, source_contract: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"source schedule is missing: {path}")
    if path.stat().st_size != source_contract["bytes"]:
        raise ContractError("source schedule byte count mismatch")
    actual = sha256_file(path)
    if actual != source_contract["file_sha256"]:
        raise ContractError("source schedule file SHA mismatch")
    return _validate_source_schedule_document(
        load_json(path, "source BankExam schedule"),
        source_contract=source_contract,
    )


def _time_law_by_side(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {item["side"]: item for item in spec["paper"]["time_laws"]}
    if set(result) != set(SIDE_ORDER):
        raise ContractError("timing paper must bind exactly one time law per side")
    return result


def build_paper(
    *,
    spec: Mapping[str, Any],
    spec_file_sha256: str,
    source_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    require_sha(spec_file_sha256, "spec file SHA")
    baseline = spec["paper"]["baseline"]
    laws = _time_law_by_side(spec)
    rows = []
    for source_item in source_schedule["items"]:
        side = SIDE_ORDER[source_item["clip"]]
        law = laws[side]
        rows.append(
            {
                "schedule_index": source_item["schedule_index"],
                "question_id": source_item["question_id"],
                "side": side,
                "initial_state_id": baseline["initial_state_id"],
                "tts_seconds": baseline["tts_seconds"],
                "tts_ticks": baseline["tts_ticks"],
                "time_law_id": law["time_law_id"],
                "expected_feasible": baseline["expected_feasible"],
                "feasibility_status": baseline["feasibility_status"],
                "bank_row": source_item["bank_row"],
                "attempt_seed": source_item["attempt_seed"],
                "repeat": source_item["repeat"],
                "source_hold_steps": source_item["hold_steps"],
                "source_hold_steps_replaced": True,
            }
        )
    document: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "contract_id": spec["contract_id"],
        "spec_file_sha256": spec_file_sha256,
        "spec_content_sha256": spec["contract_content_sha256"],
        "source_schedule": {
            "file_sha256": spec["source_schedule"]["file_sha256"],
            "semantic_sha256": spec["source_schedule"]["semantic_sha256"],
            "question_id_order_sha256": spec["source_schedule"]["question_id_order_sha256"],
            "bank_sha256": spec["source_schedule"]["bank_sha256"],
            "bank_source_family_sha256": spec["source_schedule"]["bank_source_family_sha256"],
        },
        "paper": spec["paper"],
        "scoring": spec["scoring"],
        "execution": spec["execution"],
        "rows": rows,
    }
    document["paper_semantic_sha256"] = _content_sha(document, "paper_semantic_sha256")
    return document


def validate_paper_document(
    value: Any,
    *,
    spec: Mapping[str, Any],
    spec_file_sha256: str,
    source_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "artifact_type",
        "contract_id",
        "spec_file_sha256",
        "spec_content_sha256",
        "source_schedule",
        "paper",
        "scoring",
        "execution",
        "rows",
        "paper_semantic_sha256",
    }
    paper = dict(require_exact_keys(value, keys, "timing exam paper"))
    require_sha(paper["paper_semantic_sha256"], "paper semantic SHA")
    if _content_sha(paper, "paper_semantic_sha256") != paper["paper_semantic_sha256"]:
        raise ContractError("timing paper semantic SHA mismatch")
    expected = build_paper(
        spec=spec,
        spec_file_sha256=spec_file_sha256,
        source_schedule=source_schedule,
    )
    if paper != expected:
        raise ContractError("timing paper differs from exact spec + source schedule derivation")
    return paper


def _write_exclusive(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise ContractError(f"output parent does not exist: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"refusing to replace existing output: {path}") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _publish_exclusive_validated(path: Path, payload: bytes, validate_staged) -> None:
    """Validate in an exclusive staging directory, then hard-link publish without replacement."""

    parent = path.parent.resolve(strict=True)
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to replace existing output: {path}")
    stage = Path(tempfile.mkdtemp(prefix=".timing-ledger.stage.", dir=parent))
    staged = stage / "result.json"
    try:
        _write_exclusive(staged, payload)
        validate_staged(staged)
        try:
            os.link(staged, path)
        except FileExistsError as exc:
            raise ContractError(f"refusing to replace existing output: {path}") from exc
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def write_paper_exclusive(path: Path, paper: Mapping[str, Any]) -> None:
    _write_exclusive(path, canonical_json_bytes(paper) + b"\n")


def _nullable_error(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return require_nonnegative_finite(value, label)


def validate_result_document(
    value: Any,
    *,
    paper: Mapping[str, Any],
    paper_file_sha256: str,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "artifact_type",
        "paper_file_sha256",
        "paper_semantic_sha256",
        "source_scorecard_file_sha256",
        "checkpoint_sha256",
        "checkpoint_hard_contract_sha256",
        "evaluator_source_sha256",
        "converter_source_sha256",
        "evaluation_execution_contract_sha256",
        "engine",
        "evaluation_contract_exact",
        "attempts",
        "result_content_sha256",
    }
    result = dict(require_exact_keys(value, keys, "timing exam result ledger"))
    if result["schema_version"] != 2 or result["artifact_type"] != RESULT_ARTIFACT_TYPE:
        raise ContractError("timing result ledger schema/type changed")
    require_sha(paper_file_sha256, "paper file SHA")
    if result["paper_file_sha256"] != paper_file_sha256:
        raise ContractError("result ledger paper file SHA mismatch")
    if result["paper_semantic_sha256"] != paper["paper_semantic_sha256"]:
        raise ContractError("result ledger paper semantic SHA mismatch")
    for field in (
        "source_scorecard_file_sha256",
        "checkpoint_sha256",
        "checkpoint_hard_contract_sha256",
        "evaluator_source_sha256",
        "converter_source_sha256",
        "evaluation_execution_contract_sha256",
        "result_content_sha256",
    ):
        require_sha(result[field], field)
    if _content_sha(result, "result_content_sha256") != result["result_content_sha256"]:
        raise ContractError("timing result ledger content SHA mismatch")
    if result["engine"] not in {"Isaac", "vendor_mujoco"}:
        raise ContractError("result engine must be Isaac or vendor_mujoco")
    if not isinstance(result["evaluation_contract_exact"], bool):
        raise ContractError("evaluation_contract_exact must be boolean")

    attempts = result["attempts"]
    rows = paper["rows"]
    if not isinstance(attempts, list) or len(attempts) != len(rows):
        raise ContractError("result ledger must contain one row for every scheduled attempt")
    attempt_keys = {
        "schedule_index",
        "question_id",
        "side",
        "initial_state_id",
        "tts_ticks",
        "time_law_id",
        "expected_feasible",
        "feasibility_status",
        "observation_valid",
        "returned",
        "position_error_m",
        "velocity_error_mps",
        "signed_normal_error_deg",
        "planner_infeasible",
        "physical_fall",
        "self_hit",
        "illegal_table_or_net_contact",
        "reset_or_teleport",
        "deadline_shifted",
    }
    for index, (raw, row) in enumerate(zip(attempts, rows)):
        attempt = require_exact_keys(raw, attempt_keys, f"result attempt {index}")
        require_plain_int(
            attempt["schedule_index"], f"result attempt {index} schedule_index"
        )
        require_plain_int(attempt["tts_ticks"], f"result attempt {index} tts_ticks")
        for field in (
            "question_id",
            "side",
            "initial_state_id",
            "time_law_id",
            "feasibility_status",
        ):
            if not isinstance(attempt[field], str):
                raise ContractError(f"result attempt {index} {field} must be a string")
        if attempt["expected_feasible"] is not None and not isinstance(
            attempt["expected_feasible"], bool
        ):
            raise ContractError(
                f"result attempt {index} expected_feasible must be boolean or null"
            )
        for field in (
            "schedule_index",
            "question_id",
            "side",
            "initial_state_id",
            "tts_ticks",
            "time_law_id",
            "expected_feasible",
            "feasibility_status",
        ):
            if attempt[field] != row[field]:
                raise ContractError(f"result attempt {index} {field} differs from paper")
        for field in (
            "observation_valid",
            "returned",
            "physical_fall",
            "reset_or_teleport",
            "deadline_shifted",
        ):
            if not isinstance(attempt[field], bool):
                raise ContractError(f"result attempt {index} {field} must be boolean")
        for field in ("self_hit", "illegal_table_or_net_contact"):
            if attempt[field] is not None and not isinstance(attempt[field], bool):
                raise ContractError(
                    f"result attempt {index} {field} must be boolean or null"
                )
        if attempt["planner_infeasible"] is not None and not isinstance(
            attempt["planner_infeasible"], bool
        ):
            raise ContractError(
                f"result attempt {index} planner_infeasible must be boolean or null"
            )
        position = _nullable_error(attempt["position_error_m"], f"attempt {index} position error")
        velocity = _nullable_error(attempt["velocity_error_mps"], f"attempt {index} velocity error")
        normal = _nullable_error(
            attempt["signed_normal_error_deg"], f"attempt {index} signed normal error"
        )
        if not attempt["observation_valid"]:
            if attempt["returned"] or any(value is not None for value in (position, velocity, normal)):
                raise ContractError(
                    f"result attempt {index} invalid observation must not claim return/errors"
                )
        elif any(value is None for value in (position, velocity, normal)):
            raise ContractError(f"result attempt {index} valid observation lacks finite error metrics")
    return result


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{label} must be boolean")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _validate_isaac_source_closure(sources: Any, *, root: Path) -> dict[str, Any]:
    expected = {
        "git_head",
        "evaluator_sha256",
        "adapter_sha256",
        "schedule_module_sha256",
        "isaac_scorer_sha256",
        "ball_physics_yaml_sha256",
        "timing_adapter_sha256",
    }
    value = dict(require_exact_keys(sources, expected, "Isaac evaluator source closure"))
    git_head = _require_text(value["git_head"], "Isaac evaluator git_head")
    if not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise ContractError("Isaac evaluator git_head must be one lowercase Git SHA")
    paths = {
        "evaluator_sha256": root
        / "hope_training/whole_body_tracking/scripts/isaac_bank_exam.py",
        "adapter_sha256": root
        / "hope_training/whole_body_tracking/scripts/isaac_bank_exam_adapter.py",
        "schedule_module_sha256": root
        / "hope_training/whole_body_tracking/scripts/bank_exam_schedule.py",
        "isaac_scorer_sha256": root
        / (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/virtual_ball.py"
        ),
        "ball_physics_yaml_sha256": root / "configs/ball_physics_venue.yaml",
        "timing_adapter_sha256": root
        / "hope_training/whole_body_tracking/scripts/isaac_timing_exam_adapter.py",
    }
    for field, path in paths.items():
        require_sha(value[field], f"Isaac evaluator sources.{field}")
        if not path.is_file() or sha256_file(path) != value[field]:
            raise ContractError(
                f"Isaac evaluator source closure differs from scorecard: {field}={path}"
            )
    return value


ISAAC_SCORECARD_TOP_KEYS = {
    "schema",
    "evaluation_contract_exact",
    "inexact_reasons",
    "simulator",
    "protocol",
    "noise_scale",
    "schedule",
    "schedule_sha256",
    "hold_semantics",
    "exam_bank",
    "checkpoint",
    "training_contract_sha256",
    "termination_contract_id",
    "ready_state_sha256",
    "cross_engine_instrumentation",
    "nominal_eval_profile",
    "sources",
    "timing_exam",
    "status",
    "summary",
    "attempts",
}

ISAAC_TIMING_ATTEMPT_KEYS = {
    "schedule_index",
    "env_id",
    "clip",
    "side",
    "bank_row",
    "question_id",
    "repeat",
    "hold_steps",
    "attempt_seed",
    "ready_state_sha256",
    "start_step",
    "end_step",
    "finalize_reason",
    "finalized",
    "censored",
    "physical_fall",
    "guard_reset",
    "reached_exact",
    "hit",
    "returned",
    "pos_error_m",
    "vel_error_mps",
    "normal_error_deg",
    "landing_x",
    "landing_y",
    "net_clear",
    "instrumentation",
    "timing_exam_enabled",
    "all_attempt_denominator_member",
    "eligible",
    "planner_infeasible",
    "infeasible",
    "planner_infeasible_source",
    "deadline_miss",
    "deadline_shifted",
    "deadline_step",
    "exact_strike_step",
    "initial_state_id",
    "tts_seconds",
    "tts_ticks",
    "time_law_id",
    "expected_feasible",
    "feasibility_status",
    "effective_hold_steps",
    "contact",
    "composite",
    "safety",
}


def _numeric_vector3(value: Any, label: str) -> tuple[float, float, float]:
    document = require_exact_keys(value, {"shape", "values"}, label)
    if document["shape"] != [3] or not isinstance(document["values"], list) or len(
        document["values"]
    ) != 3:
        raise ContractError(f"{label} must be one 3-vector")
    result = []
    for index, item in enumerate(document["values"]):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractError(f"{label}[{index}] must be finite")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ContractError(f"{label}[{index}] must be finite")
        result.append(numeric)
    return tuple(result)


def _signed_normal_error_from_instrumentation(
    value: Any, *, index: int
) -> tuple[float, Mapping[str, Any]]:
    keys = {
        "schema",
        "kind",
        "observation_phase",
        "coordinate_contract",
        "base",
        "racket",
        "target",
        "incoming_ball",
        "analytic_counterfactual",
        "physical_truth",
        "sha256",
    }
    document = dict(require_exact_keys(value, keys, f"Isaac attempt {index} instrumentation"))
    if (
        document["schema"] != "hope.cross-engine-state-instrumentation.v1"
        or document["kind"] != "isaac_question_state"
        or document["observation_phase"] != "exact_strike"
    ):
        raise ContractError(f"Isaac attempt {index} lacks exact-strike instrumentation")
    require_sha(document["sha256"], f"Isaac attempt {index} instrumentation SHA")
    payload = dict(document)
    payload.pop("sha256")
    if canonical_sha256(payload) != document["sha256"]:
        raise ContractError(f"Isaac attempt {index} instrumentation content SHA mismatch")
    racket = require_exact_keys(
        document["racket"],
        {
            "position_env_m",
            "linear_velocity_world_mps",
            "face_normal_signed_pre_orient_world",
            "face_normal_raw_plus_y_world",
            "analytic_face_normal_oriented_world",
        },
        f"Isaac attempt {index} racket instrumentation",
    )
    target = require_exact_keys(
        document["target"],
        {
            "racket_position_env_m",
            "racket_linear_velocity_world_mps",
            "face_normal_world",
        },
        f"Isaac attempt {index} target instrumentation",
    )
    signed = _numeric_vector3(
        racket["face_normal_signed_pre_orient_world"],
        f"Isaac attempt {index} signed racket normal",
    )
    target_normal = _numeric_vector3(
        target["face_normal_world"], f"Isaac attempt {index} target normal"
    )
    signed_norm = math.sqrt(sum(item * item for item in signed))
    target_norm = math.sqrt(sum(item * item for item in target_normal))
    if signed_norm <= 0.0 or target_norm <= 0.0:
        raise ContractError(f"Isaac attempt {index} has a zero face normal")
    cosine = sum(a * b for a, b in zip(signed, target_normal)) / (
        signed_norm * target_norm
    )
    error = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    analytic = require_exact_keys(
        document["analytic_counterfactual"],
        {
            "available",
            "capability",
            "capture_gate",
            "net_clear",
            "on_opponent",
            "landing_valid",
            "landing_xy_env_m",
        },
        f"Isaac attempt {index} analytic instrumentation",
    )
    if analytic["available"] is not True:
        raise ContractError(f"Isaac attempt {index} exact observation lacks analytic state")
    for field in ("capture_gate", "net_clear", "on_opponent", "landing_valid"):
        _require_bool(analytic[field], f"Isaac attempt {index} analytic {field}")
    return error, analytic


def validate_isaac_timing_scorecard(
    value: Any,
    *,
    root: Path,
    source_schedule: Mapping[str, Any],
    paper: Mapping[str, Any],
    paper_file_sha256: str,
    expected_checkpoint_sha256: str,
    expected_hard_contract_sha256: str,
) -> dict[str, Any]:
    """Validate one native Isaac scorecard before converting any row.

    The current Isaac lane deliberately lacks planner feasibility, self-hit, and illegal table/net
    observations.  Those fields remain null in result-ledger v2; this validator never coerces an
    unobserved safety value to false.
    """

    scorecard = dict(
        require_exact_keys(value, ISAAC_SCORECARD_TOP_KEYS, "Isaac timing scorecard")
    )
    if scorecard["schema"] != ISAAC_SCORECARD_SCHEMA or scorecard["status"] != "valid":
        raise ContractError("Isaac timing scorecard schema/status changed")
    if (
        scorecard["evaluation_contract_exact"] is not False
        or scorecard["simulator"] != "isaac"
        or scorecard["protocol"] != "single"
        or scorecard["noise_scale"] != 0.0
    ):
        raise ContractError("Isaac timing scorecard must remain the inexact zero-noise single lane")
    reasons = scorecard["inexact_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(item, str) for item in reasons)
        or not ISAAC_DIAGNOSTIC_REASONS.issubset(set(reasons))
    ):
        raise ContractError("Isaac timing scorecard omits a required diagnostic limitation")
    if scorecard["schedule"] != source_schedule:
        raise ContractError("Isaac scorecard schedule differs from the exact private schedule")
    if scorecard["schedule_sha256"] != paper["source_schedule"]["semantic_sha256"]:
        raise ContractError("Isaac scorecard schedule semantic SHA differs from paper")
    if scorecard["hold_semantics"] != "stand-policy-actions-then-raw-frame0-v1":
        raise ContractError("Isaac scorecard hold semantics changed")

    bank = require_exact_keys(
        scorecard["exam_bank"],
        {"path", "sha256", "source_family_sha256", "schema_version", "split"},
        "Isaac scorecard exam_bank",
    )
    _require_text(bank["path"], "Isaac scorecard exam bank path")
    if (
        bank["sha256"] != paper["source_schedule"]["bank_sha256"]
        or bank["source_family_sha256"]
        != paper["source_schedule"]["bank_source_family_sha256"]
        or bank["schema_version"] != 3
        or bank["split"] != "exam"
    ):
        raise ContractError("Isaac scorecard exam bank differs from timing paper")

    checkpoint = require_exact_keys(
        scorecard["checkpoint"], {"path", "sha256"}, "Isaac scorecard checkpoint"
    )
    _require_text(checkpoint["path"], "Isaac scorecard checkpoint path")
    require_sha(expected_checkpoint_sha256, "expected checkpoint SHA")
    require_sha(expected_hard_contract_sha256, "expected hard-contract SHA")
    if checkpoint["sha256"] != expected_checkpoint_sha256:
        raise ContractError("Isaac scorecard checkpoint SHA differs from supplied checkpoint")
    if scorecard["training_contract_sha256"] != expected_hard_contract_sha256:
        raise ContractError("Isaac scorecard hard-contract SHA differs from supplied contract")
    require_sha(scorecard["ready_state_sha256"], "Isaac scorecard ready-state SHA")
    _require_text(scorecard["termination_contract_id"], "Isaac termination contract id")
    sources = _validate_isaac_source_closure(scorecard["sources"], root=root)

    timing = require_exact_keys(
        scorecard["timing_exam"],
        {
            "enabled",
            "mode",
            "paper_binding",
            "source_schedule_file_sha256",
            "ready_state",
            "runtime",
            "summary",
            "all_scheduled_attempts_in_denominator",
            "formal_gate_authorized",
            "mujoco_retiming_status",
        },
        "Isaac scorecard timing_exam",
    )
    binding = require_exact_keys(
        timing["paper_binding"], {"path", "file_sha256", "semantic_sha256"},
        "Isaac scorecard timing paper binding",
    )
    if (
        timing["enabled"] is not True
        or timing["mode"] != "0.5_second_zero_velocity_frame0_diagnostic"
        or binding["file_sha256"] != paper_file_sha256
        or binding["semantic_sha256"] != paper["paper_semantic_sha256"]
        or timing["source_schedule_file_sha256"]
        != paper["source_schedule"]["file_sha256"]
        or timing["all_scheduled_attempts_in_denominator"] is not True
        or timing["formal_gate_authorized"] is not False
        or timing["mujoco_retiming_status"]
        != "blocked_not_implemented_or_verified_in_this_evaluator"
    ):
        raise ContractError("Isaac scorecard timing-paper execution binding changed")
    _require_text(binding["path"], "Isaac scorecard timing paper path")
    if not isinstance(timing["ready_state"], dict) or not isinstance(timing["runtime"], dict):
        raise ContractError("Isaac scorecard lacks timing ready/runtime evidence")
    if not isinstance(timing["summary"], dict) or timing["summary"].get(
        "formal_gate_pass"
    ) is not False:
        raise ContractError("Isaac timing summary may not claim a formal pass")

    attempts = scorecard["attempts"]
    if not isinstance(attempts, list) or len(attempts) != len(paper["rows"]):
        raise ContractError("Isaac timing scorecard must contain exactly one row per paper row")
    converted: list[dict[str, Any]] = []
    composite = paper["scoring"]["composite"]
    for index, (raw, paper_row, schedule_row) in enumerate(
        zip(attempts, paper["rows"], source_schedule["items"])
    ):
        attempt = require_exact_keys(raw, ISAAC_TIMING_ATTEMPT_KEYS, f"Isaac attempt {index}")
        for field in ("schedule_index", "env_id", "clip", "bank_row", "repeat", "hold_steps"):
            require_plain_int(attempt[field], f"Isaac attempt {index} {field}")
        require_plain_int(
            attempt["attempt_seed"], f"Isaac attempt {index} attempt_seed",
            maximum=(1 << 64) - 1,
        )
        expected_identity = {
            "schedule_index": paper_row["schedule_index"],
            "env_id": index,
            "clip": 0 if paper_row["side"] == "forehand" else 1,
            "side": paper_row["side"],
            "bank_row": paper_row["bank_row"],
            "question_id": paper_row["question_id"],
            "repeat": paper_row["repeat"],
            "hold_steps": paper_row["source_hold_steps"],
            "attempt_seed": paper_row["attempt_seed"],
        }
        mismatch = {
            key: (attempt.get(key), expected)
            for key, expected in expected_identity.items()
            if attempt.get(key) != expected
        }
        if mismatch or schedule_row["question_id"] != paper_row["question_id"]:
            raise ContractError(f"Isaac attempt {index} differs from paper/schedule: {mismatch}")
        for field in (
            "finalized",
            "censored",
            "physical_fall",
            "guard_reset",
            "reached_exact",
            "hit",
            "returned",
            "net_clear",
            "timing_exam_enabled",
            "all_attempt_denominator_member",
            "eligible",
            "deadline_miss",
            "deadline_shifted",
            "contact",
            "composite",
        ):
            _require_bool(attempt[field], f"Isaac attempt {index} {field}")
        if (
            attempt["finalized"] is not True
            or attempt["censored"] is not False
            or attempt["timing_exam_enabled"] is not True
            or attempt["all_attempt_denominator_member"] is not True
            or attempt["eligible"] is not True
            or attempt["planner_infeasible"] is not None
            or attempt["infeasible"] is not None
            or attempt["planner_infeasible_source"]
            != "unmeasured_fixed_question_exam_bypasses_planner"
            or attempt["effective_hold_steps"] != 0
            or attempt["deadline_step"] != paper_row["tts_ticks"]
            or attempt["initial_state_id"] != paper_row["initial_state_id"]
            or attempt["tts_seconds"] != 0.5
            or attempt["tts_ticks"] != paper_row["tts_ticks"]
            or attempt["time_law_id"] != paper_row["time_law_id"]
            or attempt["expected_feasible"] != paper_row["expected_feasible"]
            or attempt["feasibility_status"] != paper_row["feasibility_status"]
        ):
            raise ContractError(f"Isaac attempt {index} timing contract changed")
        _require_text(attempt["finalize_reason"], f"Isaac attempt {index} finalize_reason")
        if attempt["physical_fall"] != (
            attempt["finalize_reason"] == "physical_fall"
        ) or attempt["guard_reset"] != (
            attempt["finalize_reason"] == "guard_reset"
        ):
            raise ContractError(f"Isaac attempt {index} termination bookkeeping disagrees")
        require_plain_int(attempt["deadline_step"], f"Isaac attempt {index} deadline_step")
        require_plain_int(attempt["tts_ticks"], f"Isaac attempt {index} tts_ticks")
        exact_step = attempt["exact_strike_step"]
        if exact_step is not None:
            require_plain_int(exact_step, f"Isaac attempt {index} exact_strike_step")
        if attempt["reached_exact"] != (exact_step is not None):
            raise ContractError(f"Isaac attempt {index} exact-strike bookkeeping disagrees")
        if attempt["contact"] != attempt["hit"]:
            raise ContractError(f"Isaac attempt {index} contact/hit bookkeeping disagrees")
        if attempt["deadline_shifted"] != (
            attempt["reached_exact"] and exact_step != attempt["deadline_step"]
        ):
            raise ContractError(f"Isaac attempt {index} deadline-shift bookkeeping disagrees")
        expected_deadline_miss = (
            not attempt["reached_exact"]
            or (
                exact_step is not None
                and exact_step > attempt["deadline_step"]
            )
        )
        if attempt["deadline_miss"] != expected_deadline_miss:
            raise ContractError(f"Isaac attempt {index} deadline-miss bookkeeping disagrees")

        metrics = []
        for field in ("pos_error_m", "vel_error_mps", "normal_error_deg"):
            value = attempt[field]
            metrics.append(None if value is None else require_nonnegative_finite(value, f"Isaac attempt {index} {field}"))
        signed_normal_error = None
        analytic = None
        if attempt["reached_exact"]:
            if any(value is None for value in metrics) or not isinstance(attempt["instrumentation"], dict):
                raise ContractError(f"Isaac attempt {index} reached exact without complete observation")
            signed_normal_error, analytic = _signed_normal_error_from_instrumentation(
                attempt["instrumentation"], index=index
            )
            if not math.isclose(
                metrics[2], signed_normal_error, rel_tol=0.0, abs_tol=1e-5
            ):
                raise ContractError(
                    f"Isaac attempt {index} normal error is not the raw signed-face error"
                )
            if (
                attempt["hit"] != analytic["capture_gate"]
                or attempt["net_clear"] != analytic["net_clear"]
                or attempt["returned"]
                != (
                    analytic["capture_gate"]
                    and analytic["net_clear"]
                    and analytic["on_opponent"]
                )
            ):
                raise ContractError(
                    f"Isaac attempt {index} return bookkeeping differs from analytic instrumentation"
                )
        elif attempt["returned"] or any(value is not None for value in metrics):
            raise ContractError(f"Isaac attempt {index} claims outcome without exact observation")
        expected_composite = (
            attempt["reached_exact"]
            and not attempt["deadline_miss"]
            and not attempt["deadline_shifted"]
            and attempt["returned"]
            and metrics[0] < composite["position_error_m_strict_lt"]
            and metrics[1] < composite["velocity_error_mps_strict_lt"]
            and signed_normal_error < composite["signed_normal_error_deg_strict_lt"]
        )
        if attempt["composite"] != expected_composite:
            raise ContractError(f"Isaac attempt {index} composite is not recomputable")
        safety = require_exact_keys(
            attempt["safety"],
            {
                "physical_fall",
                "self_hit",
                "illegal_table_or_net_contact",
                "reset_or_teleport",
                "deadline_shifted",
                "complete",
            },
            f"Isaac attempt {index} safety",
        )
        if safety != {
            "physical_fall": attempt["physical_fall"],
            "self_hit": None,
            "illegal_table_or_net_contact": None,
            "reset_or_teleport": attempt["guard_reset"],
            "deadline_shifted": attempt["deadline_shifted"],
            "complete": False,
        }:
            raise ContractError(f"Isaac attempt {index} safety observation boundary changed")
        converted.append(
            {
                "schedule_index": paper_row["schedule_index"],
                "question_id": paper_row["question_id"],
                "side": paper_row["side"],
                "initial_state_id": paper_row["initial_state_id"],
                "tts_ticks": paper_row["tts_ticks"],
                "time_law_id": paper_row["time_law_id"],
                "expected_feasible": paper_row["expected_feasible"],
                "feasibility_status": paper_row["feasibility_status"],
                "observation_valid": attempt["reached_exact"],
                "returned": attempt["returned"],
                "position_error_m": metrics[0],
                "velocity_error_mps": metrics[1],
                "signed_normal_error_deg": signed_normal_error,
                "planner_infeasible": None,
                "physical_fall": attempt["physical_fall"],
                "self_hit": None,
                "illegal_table_or_net_contact": None,
                "reset_or_teleport": attempt["guard_reset"],
                "deadline_shifted": attempt["deadline_shifted"],
            }
        )
    scorecard["_validated_conversion"] = {
        "sources": sources,
        "attempts": converted,
    }
    return scorecard


def build_isaac_result_ledger(
    *,
    scorecard: Mapping[str, Any],
    scorecard_file_sha256: str,
    paper: Mapping[str, Any],
    paper_file_sha256: str,
) -> dict[str, Any]:
    require_sha(scorecard_file_sha256, "scorecard file SHA")
    converter_sha = sha256_file(Path(__file__))
    sources = scorecard["_validated_conversion"]["sources"]
    execution_binding = {
        "schema_version": 1,
        "lane": "isaac_0p5_uniform_phase_diagnostic",
        "source_scorecard_file_sha256": scorecard_file_sha256,
        "paper_file_sha256": paper_file_sha256,
        "paper_semantic_sha256": paper["paper_semantic_sha256"],
        "source_schedule_file_sha256": paper["source_schedule"]["file_sha256"],
        "checkpoint_sha256": scorecard["checkpoint"]["sha256"],
        "checkpoint_hard_contract_sha256": scorecard["training_contract_sha256"],
        "evaluator_sources": sources,
        "converter_source_sha256": converter_sha,
        "evaluation_contract_exact": False,
    }
    result: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": RESULT_ARTIFACT_TYPE,
        "paper_file_sha256": paper_file_sha256,
        "paper_semantic_sha256": paper["paper_semantic_sha256"],
        "source_scorecard_file_sha256": scorecard_file_sha256,
        "checkpoint_sha256": scorecard["checkpoint"]["sha256"],
        "checkpoint_hard_contract_sha256": scorecard["training_contract_sha256"],
        "evaluator_source_sha256": canonical_sha256(sources),
        "converter_source_sha256": converter_sha,
        "evaluation_execution_contract_sha256": canonical_sha256(execution_binding),
        "engine": "Isaac",
        "evaluation_contract_exact": False,
        "attempts": scorecard["_validated_conversion"]["attempts"],
    }
    result["result_content_sha256"] = _content_sha(result, "result_content_sha256")
    return result


def score_result(result: Mapping[str, Any], *, paper: Mapping[str, Any]) -> dict[str, Any]:
    scoring = paper["scoring"]
    composite = scoring["composite"]
    safety_fields = scoring["safety_zero_tolerance_fields"]
    side_success = Counter({side: 0 for side in SIDE_ORDER})
    side_observed = Counter({side: 0 for side in SIDE_ORDER})
    side_infeasible = Counter({side: 0 for side in SIDE_ORDER})
    safety_counts = Counter({field: 0 for field in safety_fields})
    safety_unknown = Counter({field: 0 for field in safety_fields})
    for attempt in result["attempts"]:
        side = attempt["side"]
        side_observed[side] += int(attempt["observation_valid"])
        side_infeasible[side] += int(attempt["planner_infeasible"] is True)
        for field in safety_fields:
            if attempt[field] is None:
                safety_unknown[field] += 1
            else:
                safety_counts[field] += int(attempt[field])
        metrics_finite = all(
            attempt[field] is not None
            for field in (
                "position_error_m",
                "velocity_error_mps",
                "signed_normal_error_deg",
            )
        )
        success = (
            attempt["observation_valid"]
            and attempt["planner_infeasible"] is not True
            and attempt["returned"]
            and metrics_finite
            and attempt["position_error_m"] < composite["position_error_m_strict_lt"]
            and attempt["velocity_error_mps"] < composite["velocity_error_mps_strict_lt"]
            and attempt["signed_normal_error_deg"]
            < composite["signed_normal_error_deg_strict_lt"]
        )
        side_success[side] += int(success)
    per_side = {
        side: {
            "scheduled": scoring["denominator"][side],
            "valid_observations": side_observed[side],
            "planner_infeasible": side_infeasible[side],
            "planner_feasibility_unknown": sum(
                1
                for attempt in result["attempts"]
                if attempt["side"] == side and attempt["planner_infeasible"] is None
            ),
            "composite_successes": side_success[side],
            "pass_count_required": scoring["per_side_pass_count"],
            "passes": side_success[side] >= scoring["per_side_pass_count"],
        }
        for side in SIDE_ORDER
    }
    performance_pass = all(item["passes"] for item in per_side.values())
    safety_observation_complete = all(count == 0 for count in safety_unknown.values())
    observed_safety_violation = any(count > 0 for count in safety_counts.values())
    safety_pass = safety_observation_complete and not observed_safety_violation
    feasibility_complete = all(
        attempt["planner_infeasible"] is not None for attempt in result["attempts"]
    )
    time_laws_certified = all(
        law["topp_or_dynamics_certified"] is True for law in paper["paper"]["time_laws"]
    )
    exact = result["evaluation_contract_exact"]
    return {
        "paper_semantic_sha256": paper["paper_semantic_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "engine": result["engine"],
        "evaluation_contract_exact": exact,
        "denominator_policy": "all_scheduled_attempts",
        "per_side": per_side,
        "safety_counts": dict(safety_counts),
        "safety_unknown_counts": dict(safety_unknown),
        "safety_observation_complete": safety_observation_complete,
        "performance_threshold_pass": performance_pass,
        "safety_pass": safety_pass,
        "planner_feasibility_observation_complete": feasibility_complete,
        "time_laws_dynamics_certified": time_laws_certified,
        "diagnostic_performance_pass": performance_pass and not observed_safety_violation,
        "formal_gate_pass": (
            performance_pass
            and safety_pass
            and feasibility_complete
            and time_laws_certified
            and exact
        ),
    }


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _common_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-spec-file-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_spec_parser = subparsers.add_parser("validate-spec")
    _common_spec_args(validate_spec_parser)

    materialize_parser = subparsers.add_parser("materialize")
    _common_spec_args(materialize_parser)
    materialize_parser.add_argument("--source-schedule", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--confirm", required=True)

    validate_paper_parser = subparsers.add_parser("validate-paper")
    _common_spec_args(validate_paper_parser)
    validate_paper_parser.add_argument("--source-schedule", type=Path, required=True)
    validate_paper_parser.add_argument("--paper", type=Path, required=True)
    validate_paper_parser.add_argument("--expected-paper-file-sha256", required=True)

    score_parser = subparsers.add_parser("score-result")
    _common_spec_args(score_parser)
    score_parser.add_argument("--source-schedule", type=Path, required=True)
    score_parser.add_argument("--paper", type=Path, required=True)
    score_parser.add_argument("--expected-paper-file-sha256", required=True)
    score_parser.add_argument("--result", type=Path, required=True)
    score_parser.add_argument("--expected-result-file-sha256", required=True)

    convert_parser = subparsers.add_parser("convert-isaac-scorecard")
    _common_spec_args(convert_parser)
    convert_parser.add_argument("--source-schedule", type=Path, required=True)
    convert_parser.add_argument("--paper", type=Path, required=True)
    convert_parser.add_argument("--expected-paper-file-sha256", required=True)
    convert_parser.add_argument("--scorecard", type=Path, required=True)
    convert_parser.add_argument("--expected-scorecard-file-sha256", required=True)
    convert_parser.add_argument("--checkpoint", type=Path, required=True)
    convert_parser.add_argument("--expected-checkpoint-file-sha256", required=True)
    convert_parser.add_argument("--checkpoint-hard-contract", type=Path, required=True)
    convert_parser.add_argument("--expected-checkpoint-hard-contract-file-sha256", required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--confirm", required=True)
    return parser


def _load_validated_paper(
    args: argparse.Namespace,
    *,
    spec: Mapping[str, Any],
    source_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    require_sha(args.expected_paper_file_sha256, "expected paper file SHA")
    actual = sha256_file(args.paper)
    if actual != args.expected_paper_file_sha256:
        raise ContractError("timing paper file SHA mismatch")
    return validate_paper_document(
        load_json(args.paper, "timing paper"),
        spec=spec,
        spec_file_sha256=args.expected_spec_file_sha256,
        source_schedule=source_schedule,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    try:
        spec = load_spec(
            args.spec,
            root=root,
            expected_file_sha256=args.expected_spec_file_sha256,
        )
        if args.command == "validate-spec":
            _print_json(
                {
                    "status": "pass_preregistered_materialization_not_run",
                    "contract_id": spec["contract_id"],
                    "spec_file_sha256": args.expected_spec_file_sha256,
                    "source_schedule_file_sha256": spec["source_schedule"]["file_sha256"],
                    "judge_authorized": False,
                    "isaac_diagnostic_evaluator_authorized": spec["execution"][
                        "isaac_diagnostic_evaluator_authorized"
                    ],
                    "real_robot_authorized": False,
                }
            )
            return 0

        source_schedule = load_source_schedule(
            args.source_schedule,
            source_contract=spec["source_schedule"],
        )
        if args.command == "materialize":
            if args.confirm != MATERIALIZE_CONFIRMATION:
                raise ContractError("materialize confirmation token mismatch")
            paper = build_paper(
                spec=spec,
                spec_file_sha256=args.expected_spec_file_sha256,
                source_schedule=source_schedule,
            )
            validate_paper_document(
                paper,
                spec=spec,
                spec_file_sha256=args.expected_spec_file_sha256,
                source_schedule=source_schedule,
            )
            write_paper_exclusive(args.output, paper)
            persisted = load_json(args.output, "persisted timing paper")
            validate_paper_document(
                persisted,
                spec=spec,
                spec_file_sha256=args.expected_spec_file_sha256,
                source_schedule=source_schedule,
            )
            _print_json(
                {
                    "status": "materialized_paper_only_no_judge",
                    "path": str(args.output.resolve()),
                    "bytes": args.output.stat().st_size,
                    "file_sha256": sha256_file(args.output),
                    "semantic_sha256": paper["paper_semantic_sha256"],
                    "question_id_order_sha256": spec["source_schedule"]["question_id_order_sha256"],
                    "scheduled_attempts": len(paper["rows"]),
                }
            )
            return 0

        paper = _load_validated_paper(
            args,
            spec=spec,
            source_schedule=source_schedule,
        )
        if args.command == "validate-paper":
            _print_json(
                {
                    "status": "pass_exact_timing_paper",
                    "file_sha256": args.expected_paper_file_sha256,
                    "semantic_sha256": paper["paper_semantic_sha256"],
                    "scheduled_attempts": len(paper["rows"]),
                    "judge_authorized": False,
                }
            )
            return 0

        if args.command == "convert-isaac-scorecard":
            if args.confirm != CONVERT_ISAAC_CONFIRMATION:
                raise ContractError("Isaac scorecard conversion confirmation token mismatch")
            for value, label in (
                (args.expected_scorecard_file_sha256, "expected scorecard file SHA"),
                (args.expected_checkpoint_file_sha256, "expected checkpoint file SHA"),
                (
                    args.expected_checkpoint_hard_contract_file_sha256,
                    "expected checkpoint hard-contract file SHA",
                ),
            ):
                require_sha(value, label)
            inputs = {
                "spec": (args.spec, args.expected_spec_file_sha256),
                "source_schedule": (
                    args.source_schedule,
                    spec["source_schedule"]["file_sha256"],
                ),
                "paper": (args.paper, args.expected_paper_file_sha256),
                "scorecard": (args.scorecard, args.expected_scorecard_file_sha256),
                "checkpoint": (args.checkpoint, args.expected_checkpoint_file_sha256),
                "hard_contract": (
                    args.checkpoint_hard_contract,
                    args.expected_checkpoint_hard_contract_file_sha256,
                ),
            }
            for label, (path, expected_sha) in inputs.items():
                if not path.is_file() or sha256_file(path) != expected_sha:
                    raise ContractError(f"{label} file is missing or SHA differs")
            scorecard = validate_isaac_timing_scorecard(
                load_json(args.scorecard, "Isaac timing scorecard"),
                root=root,
                source_schedule=source_schedule,
                paper=paper,
                paper_file_sha256=args.expected_paper_file_sha256,
                expected_checkpoint_sha256=args.expected_checkpoint_file_sha256,
                expected_hard_contract_sha256=(
                    args.expected_checkpoint_hard_contract_file_sha256
                ),
            )
            result = build_isaac_result_ledger(
                scorecard=scorecard,
                scorecard_file_sha256=args.expected_scorecard_file_sha256,
                paper=paper,
                paper_file_sha256=args.expected_paper_file_sha256,
            )
            validate_result_document(
                result,
                paper=paper,
                paper_file_sha256=args.expected_paper_file_sha256,
            )
            # Re-hash every external input immediately before publishing.  A mixed-generation
            # conversion leaves no output because _write_exclusive has not run yet.
            for label, (path, expected_sha) in inputs.items():
                if sha256_file(path) != expected_sha:
                    raise ContractError(f"{label} changed during conversion")
            if sha256_file(Path(__file__)) != result["converter_source_sha256"]:
                raise ContractError("converter source changed during conversion")
            _validate_isaac_source_closure(scorecard["sources"], root=root)
            def validate_staged_result(path: Path) -> None:
                staged = validate_result_document(
                    load_json(path, "staged Isaac timing result ledger"),
                    paper=paper,
                    paper_file_sha256=args.expected_paper_file_sha256,
                )
                if staged != result:
                    raise ContractError(
                        "staged Isaac timing result ledger differs after reload"
                    )

            _publish_exclusive_validated(
                args.output,
                canonical_json_bytes(result) + b"\n",
                validate_staged_result,
            )
            persisted = validate_result_document(
                load_json(args.output, "persisted Isaac timing result ledger"),
                paper=paper,
                paper_file_sha256=args.expected_paper_file_sha256,
            )
            if persisted != result:
                raise ContractError("persisted Isaac timing result ledger differs after reload")
            summary = score_result(result, paper=paper)
            _print_json(
                {
                    "status": "converted_inexact_isaac_timing_ledger_no_clobber",
                    "path": str(args.output.resolve()),
                    "file_sha256": sha256_file(args.output),
                    "source_scorecard_file_sha256": args.expected_scorecard_file_sha256,
                    "checkpoint_sha256": args.expected_checkpoint_file_sha256,
                    "checkpoint_hard_contract_sha256": (
                        args.expected_checkpoint_hard_contract_file_sha256
                    ),
                    "scheduled_attempts": len(result["attempts"]),
                    "safety_observation_complete": summary[
                        "safety_observation_complete"
                    ],
                    "formal_gate_pass": summary["formal_gate_pass"],
                }
            )
            return 0

        require_sha(args.expected_result_file_sha256, "expected result file SHA")
        actual_result_sha = sha256_file(args.result)
        if actual_result_sha != args.expected_result_file_sha256:
            raise ContractError("timing result file SHA mismatch")
        result = validate_result_document(
            load_json(args.result, "timing result ledger"),
            paper=paper,
            paper_file_sha256=args.expected_paper_file_sha256,
        )
        _print_json(score_result(result, paper=paper))
        return 0
    except (ContractError, OSError) as exc:
        print(f"FAIL CLOSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
