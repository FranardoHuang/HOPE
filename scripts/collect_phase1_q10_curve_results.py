#!/usr/bin/env python3
"""Validate copied Phase-1 q10 evidence and emit a content-addressed archive.

This tool is intentionally post-hoc and local.  It never connects to a Pod, opens a
checkpoint with torch, starts a judge, signals a process, or contains a real-robot path.
The expensive/runtime facts are supplied by an explicit checkpoint-audit sidecar and are
cross-checked against hardened curve-worker state, the frozen worker manifest, the human
judge report, and ``mujoco_sim2sim_summary.json``.

The archive is a direction-screen record only.  q50 papers, decision claims, incomplete
pairs, mixed immutable schedules, legacy worker states, and unbound audit sidecars fail
closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


Q10_SCHEDULE_K = 20
Q10_ATTEMPTS_PER_SIDE = 10
Q10_NOISE_SCALE = 0.0
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")
PAIR_KINDS = {"face_pair", "plant_pair", "seed_replication_pair"}


class ContractError(ValueError):
    """Copied evidence cannot support a q10 curve archive."""


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical finite JSON: {exc}") from None


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from None
    return digest.hexdigest()


def _require_keys(value: Mapping[str, Any], expected: set[str], *, where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{where}: keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{where}: expected 64 lowercase SHA-256 hex chars")
    return value


def _require_safe_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ContractError(f"{where}: unsafe or missing identifier {value!r}")
    return value


def _require_int(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def _require_bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{where}: expected boolean")
    return value


def _require_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where}: expected finite number")
    return result


def _resolve_file(base: Path, raw: Any, *, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{where}: path must be a non-empty string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{where}: missing/unreadable file {path}: {exc}") from None
    if not resolved.is_file():
        raise ContractError(f"{where}: not a regular file: {resolved}")
    return resolved


def _artifact_ref(base: Path, raw: Any, *, where: str) -> tuple[Path, str]:
    if not isinstance(raw, Mapping):
        raise ContractError(f"{where}: artifact reference must be an object")
    _require_keys(raw, {"path", "sha256"}, where=where)
    expected = _require_sha(raw.get("sha256"), where=f"{where}.sha256")
    path = _resolve_file(base, raw.get("path"), where=f"{where}.path")
    actual = sha256_file(path)
    if actual != expected:
        raise ContractError(
            f"{where}: copied bytes changed; declared={expected}, actual={actual}"
        )
    return path, actual


def checkpoint_iteration(job: Mapping[str, Any], *, where: str) -> int:
    checkpoint = Path(str(job.get("checkpoint", "")))
    match = CHECKPOINT.fullmatch(checkpoint.name)
    if not match:
        raise ContractError(f"{where}: checkpoint must be model_<iteration>.pt")
    return int(match.group(1))


def _validate_screen_policy(policy: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise ContractError(f"{where}: screen_policy is required")
    if policy.get("screen_only") is not True:
        raise ContractError(f"{where}: q10 must remain screen_only=true")
    if policy.get("stop_or_promote_allowed") is not False:
        raise ContractError(f"{where}: decision claims are forbidden")
    if policy.get("schedule_k") != Q10_SCHEDULE_K:
        raise ContractError(
            f"{where}: q50/non-q10 paper refused; schedule_k must be {Q10_SCHEDULE_K}"
        )
    if policy.get("attempts_per_side") != Q10_ATTEMPTS_PER_SIDE:
        raise ContractError(
            f"{where}: attempts_per_side must be {Q10_ATTEMPTS_PER_SIDE}"
        )
    if policy.get("seed", 0) != 0:
        raise ContractError(f"{where}: this archive contract requires exam seed 0")
    if str(policy.get("noise_scales", "0.0")) != "0.0":
        raise ContractError(f"{where}: this archive contract requires clean noise_scales=0.0")
    return dict(policy)


def validate_manifest(
    manifest: dict[str, Any], *, label: str, selected_barriers: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise ContractError(f"{label}: worker manifest schema_version must be 1")
    policy = _validate_screen_policy(manifest.get("screen_policy"), where=label)
    judge_sha = _require_sha(
        manifest.get("judge_script_sha256"), where=f"{label}.judge_script_sha256"
    )
    train_commit = _require_sha(
        manifest.get("expected_training_commit"),
        where=f"{label}.expected_training_commit",
    )
    if not isinstance(manifest.get("jobs"), list) or not manifest["jobs"]:
        raise ContractError(f"{label}: jobs must be a non-empty list")
    if not selected_barriers or len(set(selected_barriers)) != len(selected_barriers):
        raise ContractError(f"{label}: barrier_ids must be non-empty and unique")
    barriers = {
        _require_safe_id(value, where=f"{label}.barrier_ids")
        for value in selected_barriers
    }
    all_barriers: set[str] = set()
    selected: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for index, raw in enumerate(manifest["jobs"]):
        where = f"{label}.jobs[{index}]"
        if not isinstance(raw, dict):
            raise ContractError(f"{where}: job must be an object")
        job_id = _require_safe_id(raw.get("id"), where=f"{where}.id")
        if job_id in seen_ids:
            raise ContractError(f"{label}: duplicate job id {job_id}")
        seen_ids.add(job_id)
        barrier = _require_safe_id(raw.get("barrier_id"), where=f"{where}.barrier_id")
        all_barriers.add(barrier)
        iteration = checkpoint_iteration(raw, where=where)
        if not job_id.endswith(f"_{iteration}_clean_q10") or "q50" in job_id.lower():
            raise ContractError(f"{where}: id/checkpoint must identify a clean q10 milestone")
        run_dir = Path(str(raw.get("run_dir", "")))
        checkpoint = Path(str(raw.get("checkpoint", "")))
        if not run_dir.is_absolute() or not checkpoint.is_absolute() or checkpoint.parent != run_dir:
            raise ContractError(f"{where}: run/checkpoint must be an absolute direct pair")
        _require_int(raw.get("gpu"), where=f"{where}.gpu")
        if raw.get("screen_only") is not True:
            raise ContractError(f"{where}: screen_only must be true")
        exact = _require_bool(
            raw.get("expected_evaluation_contract_exact"), where=f"{where}.exact"
        )
        formal = _require_bool(raw.get("formal_target"), where=f"{where}.formal_target")
        if formal and not exact:
            raise ContractError(f"{where}: an inexact job cannot be formal")
        if not isinstance(raw.get("evaluation_role"), str) or not raw["evaluation_role"]:
            raise ContractError(f"{where}: evaluation_role is required")
        expected_args = ["--schedule-k", str(Q10_SCHEDULE_K)]
        if not exact:
            expected_args += ["--exam-extra", "--allow-inexact-contract"]
        if raw.get("extra_args") != expected_args:
            raise ContractError(f"{where}: judge args must be exactly {expected_args!r}")
        if raw.get("seed", 0) != policy.get("seed", 0):
            raise ContractError(f"{where}: exam seed contradicts the manifest")
        if str(raw.get("noise_scales", "0.0")) != str(policy.get("noise_scales", "0.0")):
            raise ContractError(f"{where}: noise scale contradicts the manifest")
        if barrier in barriers:
            selected[job_id] = raw
    missing = sorted(barriers - all_barriers)
    if missing:
        raise ContractError(f"{label}: requested barrier(s) do not exist: {missing}")
    if not selected:
        raise ContractError(f"{label}: selected barriers contain no jobs")
    return selected, {
        "screen_policy": policy,
        "judge_script_sha256": judge_sha,
        "training_commit": train_commit,
    }


def _validate_worker_command(state: Mapping[str, Any], job: Mapping[str, Any], *, where: str) -> None:
    command = state.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ContractError(f"{where}: command must be a string list")
    expected_tail = [
        "--gpu", str(job["gpu"]),
        "--seed", str(job.get("seed", 0)),
        "--noise-scales", str(job.get("noise_scales", "0.0")),
        "--hold-ref", str(job.get("hold_ref", "auto")),
        *job["extra_args"],
    ]
    if (
        len(command) < 4
        or command[0] != "bash"
        or command[2] != job["run_dir"]
        or command[3] != job["checkpoint"]
        or command[4:] != expected_tail
    ):
        raise ContractError(f"{where}: worker command differs from the frozen q10 job")


def validate_worker_state(
    state: dict[str, Any], *, manifest_sha: str, manifest_meta: Mapping[str, Any],
    job: dict[str, Any], state_sha: str, where: str,
) -> dict[str, Any]:
    job_id = job["id"]
    if state.get("id") != job_id:
        raise ContractError(f"{where}: state id does not match job {job_id}")
    if state.get("status") != "complete" or state.get("returncode") != 0:
        raise ContractError(f"{where}: worker state is not complete rc=0")
    for key in (
        "manifest_sha256", "job_spec_sha256", "job_contract_sha256",
        "checkpoint_sha256", "judge_script_sha256", "eval_commit", "training_commit",
    ):
        _require_sha(state.get(key), where=f"{where}.{key}")
    expected_job_sha = canonical_sha256(job)
    expected_contract_sha = canonical_sha256(
        {"screen_policy": manifest_meta["screen_policy"], "job": job}
    )
    expected = {
        "manifest_sha256": manifest_sha,
        "job_spec_sha256": expected_job_sha,
        "job_contract_sha256": expected_contract_sha,
        "judge_script_sha256": manifest_meta["judge_script_sha256"],
        "training_commit": manifest_meta["training_commit"],
        "run_dir": job["run_dir"],
        "checkpoint": job["checkpoint"],
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ContractError(f"{where}: hardened state binding mismatch: {mismatches}")
    _validate_worker_command(state, job, where=where)
    return {
        "state_sha256": state_sha,
        "manifest_sha256": manifest_sha,
        "job_spec_sha256": expected_job_sha,
        "job_contract_sha256": expected_contract_sha,
        "checkpoint_sha256": state["checkpoint_sha256"],
        "judge_script_sha256": state["judge_script_sha256"],
        "training_commit": state["training_commit"],
        "eval_commit": state["eval_commit"],
    }


def validate_checkpoint_audit(
    audit: dict[str, Any], *, job: Mapping[str, Any], state_binding: Mapping[str, Any],
    report_sha: str, scorecard_sha: str, audit_sha: str, where: str,
) -> dict[str, Any]:
    _require_keys(
        audit,
        {
            "schema_version", "audit_kind", "read_only", "real_robot_commands",
            "job_id", "manifest_sha256", "state_sha256", "job_spec_sha256",
            "job_contract_sha256", "judge_report_sha256", "scorecard_sha256",
            "checkpoint", "training_contract", "provenance", "evaluation",
        },
        where=where,
    )
    if audit.get("schema_version") != 1 or audit.get("audit_kind") != (
        "phase1_q10_checkpoint_audit"
    ):
        raise ContractError(f"{where}: unsupported explicit checkpoint audit schema")
    if audit.get("read_only") is not True or audit.get("real_robot_commands") is not False:
        raise ContractError(f"{where}: audit must attest read_only=true and real_robot_commands=false")
    expected_root = {
        "job_id": job["id"],
        "manifest_sha256": state_binding["manifest_sha256"],
        "state_sha256": state_binding["state_sha256"],
        "job_spec_sha256": state_binding["job_spec_sha256"],
        "job_contract_sha256": state_binding["job_contract_sha256"],
        "judge_report_sha256": report_sha,
        "scorecard_sha256": scorecard_sha,
    }
    mismatches = {
        key: (audit.get(key), value)
        for key, value in expected_root.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ContractError(f"{where}: audit/evidence binding mismatch: {mismatches}")

    checkpoint = audit.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ContractError(f"{where}.checkpoint: object required")
    _require_keys(
        checkpoint,
        {
            "path", "sha256", "filename_iteration", "embedded_iteration",
            "floating_tensor_count", "nonfinite_floating_elements",
            "all_floating_tensors_finite", "embedded_training_contract_sha256",
            "embedded_training_contract_lineage_exact",
        },
        where=f"{where}.checkpoint",
    )
    iteration = checkpoint_iteration(job, where=where)
    if checkpoint.get("path") != job["checkpoint"]:
        raise ContractError(f"{where}: audited checkpoint path differs from manifest")
    if checkpoint.get("sha256") != state_binding["checkpoint_sha256"]:
        raise ContractError(f"{where}: checkpoint SHA differs from hardened state")
    if checkpoint.get("filename_iteration") != iteration or checkpoint.get(
        "embedded_iteration"
    ) != iteration:
        raise ContractError(f"{where}: filename/embedded checkpoint iteration mismatch")
    _require_int(
        checkpoint.get("floating_tensor_count"),
        where=f"{where}.checkpoint.floating_tensor_count",
        minimum=1,
    )
    if checkpoint.get("nonfinite_floating_elements") != 0 or checkpoint.get(
        "all_floating_tensors_finite"
    ) is not True:
        raise ContractError(f"{where}: checkpoint is not finite")
    embedded_contract_sha = _require_sha(
        checkpoint.get("embedded_training_contract_sha256"),
        where=f"{where}.checkpoint.embedded_training_contract_sha256",
    )
    embedded_lineage = _require_bool(
        checkpoint.get("embedded_training_contract_lineage_exact"),
        where=f"{where}.checkpoint.embedded_training_contract_lineage_exact",
    )

    contract = audit.get("training_contract")
    if not isinstance(contract, Mapping):
        raise ContractError(f"{where}.training_contract: object required")
    _require_keys(
        contract,
        {"path", "sha256", "schema_version", "structure_validator", "binding_validator", "lineage_exact"},
        where=f"{where}.training_contract",
    )
    expected_contract_path = str(Path(job["run_dir"]) / "params" / "training_contract.json")
    contract_sha = _require_sha(
        contract.get("sha256"), where=f"{where}.training_contract.sha256"
    )
    if (
        contract.get("path") != expected_contract_path
        or contract.get("schema_version") != 3
        or contract.get("structure_validator") != "pass"
        or contract.get("binding_validator") != "pass"
        or not isinstance(contract.get("lineage_exact"), bool)
        or contract_sha != embedded_contract_sha
        or contract.get("lineage_exact") is not embedded_lineage
    ):
        raise ContractError(f"{where}: checkpoint-to-adjacent hard-contract binding failed")

    provenance = audit.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError(f"{where}.provenance: object required")
    _require_keys(
        provenance,
        {"training_commit", "eval_commit", "judge_script_sha256", "evaluator_source_sha256"},
        where=f"{where}.provenance",
    )
    for key in provenance:
        _require_sha(provenance[key], where=f"{where}.provenance.{key}")
    for key in ("training_commit", "eval_commit", "judge_script_sha256"):
        if provenance[key] != state_binding[key]:
            raise ContractError(f"{where}: audit provenance {key} differs from worker state")

    evaluation = audit.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ContractError(f"{where}.evaluation: object required")
    _require_keys(
        evaluation, {"immutable_schedule_sha256", "evaluation_contract_exact"},
        where=f"{where}.evaluation",
    )
    schedule_sha = _require_sha(
        evaluation.get("immutable_schedule_sha256"),
        where=f"{where}.evaluation.immutable_schedule_sha256",
    )
    exact = _require_bool(
        evaluation.get("evaluation_contract_exact"),
        where=f"{where}.evaluation.evaluation_contract_exact",
    )
    if exact is not job["expected_evaluation_contract_exact"]:
        raise ContractError(f"{where}: audit exactness differs from manifest role")
    return {
        "audit_sha256": audit_sha,
        "checkpoint_iteration": iteration,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_all_floating_tensors_finite": True,
        "training_contract_sha256": contract_sha,
        "checkpoint_contract_sha_matches_adjacent": True,
        "checkpoint_lineage_exact": embedded_lineage,
        "immutable_schedule_sha256": schedule_sha,
        "evaluation_contract_exact": exact,
        "evaluator_source_sha256": provenance["evaluator_source_sha256"],
    }


def _finite_rate(value: Any, *, where: str) -> float:
    result = _require_number(value, where=where)
    if result < 0.0 or result > 1.0:
        raise ContractError(f"{where}: rate must be in [0, 1]")
    return result


def _metric_group(raw: Any, *, attempts: int, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ContractError(f"{where}: venue metric group missing")
    n_attempts = _require_int(raw.get("n_attempts"), where=f"{where}.n_attempts")
    n_strikes = _require_int(raw.get("n_strikes"), where=f"{where}.n_strikes")
    contacted = _require_int(raw.get("contacted"), where=f"{where}.contacted")
    landed_ok = _require_int(raw.get("landed_ok"), where=f"{where}.landed_ok")
    if n_attempts != attempts or not (0 <= landed_ok <= contacted <= n_strikes <= attempts):
        raise ContractError(f"{where}: all-attempt count ordering/denominator is invalid")
    exact_rate = _finite_rate(
        raw.get("exact_reach_rate_per_attempt"), where=f"{where}.exact_reach_rate_per_attempt"
    )
    contact_rate = _finite_rate(
        raw.get("contact_rate_per_attempt"), where=f"{where}.contact_rate_per_attempt"
    )
    return_rate = _finite_rate(
        raw.get("return_success_rate_per_attempt"),
        where=f"{where}.return_success_rate_per_attempt",
    )
    expected = (n_strikes / attempts, contacted / attempts, landed_ok / attempts)
    observed = (exact_rate, contact_rate, return_rate)
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-12) for a, b in zip(observed, expected)):
        raise ContractError(f"{where}: rates are not computed on the full attempt denominator")
    return {
        "attempts": attempts,
        "reached_exact": n_strikes,
        "contacted": contacted,
        "returned": landed_ok,
        "exact_reach_rate": exact_rate,
        "contact_rate": contact_rate,
        "return_rate": return_rate,
    }


def _schedule_item_identity(item: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ContractError(f"{where}: schedule item must be an object")
    result = {
        "schedule_index": _require_int(item.get("schedule_index"), where=f"{where}.schedule_index"),
        "clip": _require_int(item.get("clip"), where=f"{where}.clip"),
        "bank_row": _require_int(item.get("bank_row"), where=f"{where}.bank_row"),
        "question_id": item.get("question_id"),
        "repeat": _require_int(item.get("repeat"), where=f"{where}.repeat"),
        "hold_steps": _require_int(item.get("hold_steps"), where=f"{where}.hold_steps"),
        "attempt_seed": _require_int(item.get("attempt_seed"), where=f"{where}.attempt_seed"),
    }
    if not isinstance(result["question_id"], str) or not result["question_id"]:
        raise ContractError(f"{where}.question_id: non-empty string required")
    if result["clip"] not in (0, 1) or result["repeat"] != 0:
        raise ContractError(f"{where}: q10 requires clip 0/1 and no repeated question")
    return result


def validate_scorecard(
    scorecard: dict[str, Any], *, job: Mapping[str, Any], scorecard_sha: str, where: str,
) -> dict[str, Any]:
    if scorecard.get("schema_version") != 3:
        raise ContractError(f"{where}: MuJoCo summary schema_version must be 3")
    exact = _require_bool(
        scorecard.get("evaluation_contract_exact"),
        where=f"{where}.evaluation_contract_exact",
    )
    if exact is not job["expected_evaluation_contract_exact"]:
        raise ContractError(f"{where}: summary exactness differs from manifest")
    arguments = scorecard.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ContractError(f"{where}: arguments object is required")
    if (
        arguments.get("target_source") != "bank"
        or arguments.get("exam_schedule_k") != Q10_SCHEDULE_K
        or arguments.get("exam_schedule_json") is not None
        or arguments.get("seed") != job.get("seed", 0)
        or arguments.get("qdes_clamp") is not True
        or arguments.get("hold_ref") != job.get("hold_ref", "auto")
        or arguments.get("exam_continuity_diagnostic") is not False
        or arguments.get("allow_inexact_contract") is not (not exact)
    ):
        raise ContractError(f"{where}: evaluator arguments differ from the clean q10 contract")
    noise = arguments.get("noise_scales")
    if not isinstance(noise, list) or len(noise) != 1 or _require_number(
        noise[0], where=f"{where}.arguments.noise_scales[0]"
    ) != Q10_NOISE_SCALE:
        raise ContractError(f"{where}: scorecard must contain only clean noise_scale=0.0")

    artifacts = scorecard.get("input_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError(f"{where}: input_artifacts object is required")
    exam_bank = artifacts.get("exam_bank")
    evaluator = artifacts.get("evaluator_source")
    if not isinstance(exam_bank, Mapping) or not isinstance(evaluator, Mapping):
        raise ContractError(f"{where}: exam_bank/evaluator_source artifact bindings are required")
    bank_sha = _require_sha(exam_bank.get("sha256"), where=f"{where}.exam_bank.sha256")
    evaluator_sha = _require_sha(
        evaluator.get("sha256"), where=f"{where}.evaluator_source.sha256"
    )

    schedule = scorecard.get("exam_schedule")
    if not isinstance(schedule, Mapping):
        raise ContractError(f"{where}: exam_schedule object is required")
    schedule_sha = _require_sha(schedule.get("sha256"), where=f"{where}.exam_schedule.sha256")
    if (
        schedule.get("schema_version") != 1
        or schedule.get("bank_sha256") != bank_sha
        or schedule.get("seed") != job.get("seed", 0)
        or schedule.get("size") != Q10_SCHEDULE_K
        or schedule.get("one_question_reset") is not True
        or schedule.get("shared_artifact") is not None
    ):
        raise ContractError(f"{where}: immutable q10 schedule metadata is inconsistent")
    items = schedule.get("items")
    if not isinstance(items, list) or len(items) != Q10_SCHEDULE_K:
        raise ContractError(f"{where}: immutable q10 schedule must contain 20 items")
    identities = [
        _schedule_item_identity(item, where=f"{where}.exam_schedule.items[{index}]")
        for index, item in enumerate(items)
    ]
    if [item["schedule_index"] for item in identities] != list(range(Q10_SCHEDULE_K)):
        raise ContractError(f"{where}: schedule indices are not contiguous 0..19")
    if len({item["question_id"] for item in identities}) != Q10_SCHEDULE_K:
        raise ContractError(f"{where}: immutable q10 schedule repeats a question id")
    if [item["clip"] for item in identities].count(0) != Q10_ATTEMPTS_PER_SIDE or [
        item["clip"] for item in identities
    ].count(1) != Q10_ATTEMPTS_PER_SIDE:
        raise ContractError(f"{where}: immutable q10 schedule is not balanced 10/side")

    results = scorecard.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        raise ContractError(f"{where}: scorecard must have exactly one clean-noise result")
    result = results[0]
    if _require_number(result.get("noise_scale"), where=f"{where}.result.noise_scale") != 0.0:
        raise ContractError(f"{where}: result noise_scale must be 0.0")
    if result.get("evaluation_contract_exact") is not exact:
        raise ContractError(f"{where}: result/top-level exactness disagree")
    runtime_schedule = result.get("exam_schedule")
    if not isinstance(runtime_schedule, Mapping):
        raise ContractError(f"{where}: runtime exam_schedule is required")
    if (
        runtime_schedule.get("sha256") != schedule_sha
        or runtime_schedule.get("size") != Q10_SCHEDULE_K
        or runtime_schedule.get("one_question_reset") is not True
    ):
        raise ContractError(f"{where}: runtime schedule differs from the immutable schedule")
    runtime_items = runtime_schedule.get("items")
    if not isinstance(runtime_items, list) or len(runtime_items) != Q10_SCHEDULE_K:
        raise ContractError(f"{where}: runtime attempt ledger is incomplete")
    runtime_identities = []
    for index, item in enumerate(runtime_items):
        identity = _schedule_item_identity(item, where=f"{where}.runtime.items[{index}]")
        if not isinstance(item, Mapping) or item.get("eligible") is not True or item.get(
            "censored"
        ) is not False:
            raise ContractError(f"{where}: runtime q10 item is ineligible or censored")
        if item.get("question_sequence_index") != identity["schedule_index"]:
            raise ContractError(f"{where}: question sequence index differs from schedule index")
        runtime_identities.append(identity)
    if runtime_identities != identities:
        raise ContractError(f"{where}: runtime question order differs from immutable schedule")
    if runtime_schedule.get("question_id_order") != [item["question_id"] for item in identities]:
        raise ContractError(f"{where}: runtime question_id_order is inconsistent")

    venue = result.get("venue")
    if not isinstance(venue, Mapping):
        raise ContractError(f"{where}: venue full-denominator metrics are required")
    metrics = {
        "forehand": _metric_group(
            venue.get("forehand"), attempts=Q10_ATTEMPTS_PER_SIDE,
            where=f"{where}.venue.forehand",
        ),
        "backhand": _metric_group(
            venue.get("backhand"), attempts=Q10_ATTEMPTS_PER_SIDE,
            where=f"{where}.venue.backhand",
        ),
        "aggregate": _metric_group(
            venue.get("all"), attempts=Q10_SCHEDULE_K, where=f"{where}.venue.all"
        ),
    }
    for key in ("attempts", "reached_exact", "contacted", "returned"):
        if metrics["aggregate"][key] != metrics["forehand"][key] + metrics["backhand"][key]:
            raise ContractError(f"{where}: aggregate {key} does not equal the two side counts")

    attempt_summary = result.get("attempts")
    if not isinstance(attempt_summary, Mapping) or attempt_summary.get("n_attempts") != Q10_SCHEDULE_K:
        raise ContractError(f"{where}: all-attempt summary denominator must be 20")
    per_clip = attempt_summary.get("per_clip")
    if not isinstance(per_clip, Mapping) or not all(
        isinstance(per_clip.get(side), Mapping)
        and per_clip[side].get("n_attempts") == Q10_ATTEMPTS_PER_SIDE
        for side in ("forehand", "backhand")
    ):
        raise ContractError(f"{where}: per-side attempt denominators must be 10/10")
    return {
        "scorecard_sha256": scorecard_sha,
        "evaluation_contract_exact": exact,
        "immutable_schedule_sha256": schedule_sha,
        "schedule_identity": identities,
        "question_id_order_sha256": canonical_sha256(
            [item["question_id"] for item in identities]
        ),
        "exam_bank_sha256": bank_sha,
        "evaluator_source_sha256": evaluator_sha,
        "metrics": metrics,
    }


def validate_judge_report(
    text: str, *, job: Mapping[str, Any], report_sha: str,
    score: Mapping[str, Any], where: str,
) -> dict[str, Any]:
    checkpoints = re.findall(r"^-\s*checkpoint:\s*`([^`]+)`", text, flags=re.MULTILINE)
    if checkpoints != [job["checkpoint"]]:
        raise ContractError(f"{where}: report checkpoint does not exactly bind the manifest job")
    exact_values = {
        value == "true"
        for value in re.findall(r"evaluation_contract_exact=(true|false)", text)
    }
    if exact_values != {score["evaluation_contract_exact"]}:
        raise ContractError(f"{where}: report exactness is absent or contradictory")
    schedules = re.findall(
        r"immutable_schedule:\s*K=(\d+)\s+seed=(\d+)\s+sha256=([0-9a-f]{64})",
        text,
    )
    if not schedules or any(
        int(k) != Q10_SCHEDULE_K or int(seed) != job.get("seed", 0)
        or digest != score["immutable_schedule_sha256"]
        for k, seed, digest in schedules
    ):
        raise ContractError(f"{where}: report immutable schedule differs from scorecard")

    table: dict[str, dict[str, float | int]] = {}
    labels = {"正手": "forehand", "反手": "backhand"}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        for chinese, side in labels.items():
            if cells[0] == f"{chinese} ns=0.0":
                if len(cells) < 5 or side in table:
                    raise ContractError(f"{where}: malformed/duplicate {side} metric row")
                try:
                    table[side] = {
                        "attempts": int(cells[1]),
                        "reached_exact": int(cells[2]),
                        "contact_rate": float(cells[3]),
                        "return_rate": float(cells[4]),
                    }
                except ValueError:
                    raise ContractError(f"{where}: non-numeric {side} metric row") from None
    if set(table) != {"forehand", "backhand"}:
        raise ContractError(f"{where}: report lacks one clean q10 row per side")
    tolerance = 5.1e-5
    for side in table:
        expected = score["metrics"][side]
        observed = table[side]
        if (
            observed["attempts"] != expected["attempts"]
            or observed["reached_exact"] != expected["reached_exact"]
            or not math.isclose(observed["contact_rate"], expected["contact_rate"], abs_tol=tolerance)
            or not math.isclose(observed["return_rate"], expected["return_rate"], abs_tol=tolerance)
        ):
            raise ContractError(f"{where}: {side} report metrics differ from scorecard")

    aggregate_match = re.search(
        r"全侧汇总\([^\n]*\):尝试数\s+ns=0\.0→(\d+);"
        r"接触率\s+ns=0\.0→([0-9.]+);"
        r"回球成功率\s+ns=0\.0→([0-9.]+)",
        text,
    )
    if aggregate_match is None:
        raise ContractError(f"{where}: report lacks the full-side all-attempt denominator line")
    aggregate = score["metrics"]["aggregate"]
    if (
        int(aggregate_match.group(1)) != aggregate["attempts"]
        or not math.isclose(float(aggregate_match.group(2)), aggregate["contact_rate"], abs_tol=tolerance)
        or not math.isclose(float(aggregate_match.group(3)), aggregate["return_rate"], abs_tol=tolerance)
    ):
        raise ContractError(f"{where}: report aggregate metrics differ from scorecard")
    return {"report_sha256": report_sha, "metrics_cross_checked": True}


def _validate_pair(pair: Mapping[str, Any], jobs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require_keys(pair, {"id", "kind", "members"}, where="pairs[]")
    pair_id = _require_safe_id(pair.get("id"), where="pairs[].id")
    kind = pair.get("kind")
    if kind not in PAIR_KINDS:
        raise ContractError(f"pair {pair_id}: unsupported kind {kind!r}")
    members = pair.get("members")
    if not isinstance(members, list) or len(members) != 2 or len(set(members)) != 2:
        raise ContractError(f"pair {pair_id}: exactly two distinct member refs are required")
    if any(not isinstance(ref, str) or ref not in jobs for ref in members):
        raise ContractError(f"pair {pair_id}: unknown/incomplete member refs {members!r}")
    left, right = (jobs[ref] for ref in members)
    same_keys = ("training_kind", "training_family")
    for key in same_keys:
        if not isinstance(left.get(key), str) or left.get(key) != right.get(key):
            raise ContractError(f"pair {pair_id}: {key} must match")
    if checkpoint_iteration(left, where=pair_id) != checkpoint_iteration(right, where=pair_id):
        raise ContractError(f"pair {pair_id}: checkpoint milestones differ")
    for job in (left, right):
        _require_int(job.get("training_seed"), where=f"pair {pair_id}.training_seed")
        _require_bool(job.get("zero_joint_friction"), where=f"pair {pair_id}.plant")
        if not isinstance(job.get("face_command_pairing"), str):
            raise ContractError(f"pair {pair_id}: face_command_pairing is required")
    if kind == "face_pair":
        if (
            left["training_seed"] != right["training_seed"]
            or left["zero_joint_friction"] is not right["zero_joint_friction"]
            or {left["face_command_pairing"], right["face_command_pairing"]}
            != {"legacy_signed_vs_A", "shared_plus_y"}
        ):
            raise ContractError(f"pair {pair_id}: invalid face-pair controls")
    elif kind == "plant_pair":
        if (
            left["training_seed"] != right["training_seed"]
            or left["face_command_pairing"] != right["face_command_pairing"]
            or {left["zero_joint_friction"], right["zero_joint_friction"]} != {False, True}
        ):
            raise ContractError(f"pair {pair_id}: invalid plant-pair controls")
    else:
        if (
            left["training_seed"] == right["training_seed"]
            or left["face_command_pairing"] != right["face_command_pairing"]
            or left["zero_joint_friction"] is not right["zero_joint_friction"]
            or left.get("cell") != right.get("cell")
        ):
            raise ContractError(f"pair {pair_id}: invalid seed-replication controls")
    return {"id": pair_id, "kind": kind, "members": members}


def validate_index(index_path: Path) -> tuple[dict[str, Any], str]:
    index_path = index_path.expanduser().resolve()
    index = load_json(index_path)
    _require_keys(
        index,
        {"schema_version", "archive_id", "screen_policy", "manifests", "pairs", "evidence"},
        where="archive index",
    )
    if index.get("schema_version") != 1:
        raise ContractError("archive index schema_version must be 1")
    archive_id = _require_safe_id(index.get("archive_id"), where="archive_id")
    claims = index.get("screen_policy")
    if claims != {
        "screen_only": True,
        "stop_or_promote_allowed": False,
        "q50_triggered": False,
        "decision_claim": None,
    }:
        raise ContractError("archive index contains a decision claim or weakened q10 policy")

    base = index_path.parent
    manifest_entries = index.get("manifests")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise ContractError("archive index requires at least one manifest")
    manifests: dict[str, dict[str, Any]] = {}
    selected_jobs: dict[str, dict[str, Any]] = {}
    manifest_records: list[dict[str, Any]] = []
    common_judge: set[str] = set()
    common_train: set[str] = set()
    for raw in manifest_entries:
        if not isinstance(raw, Mapping):
            raise ContractError("manifests[] must be an object")
        _require_keys(raw, {"id", "path", "sha256", "barrier_ids"}, where="manifests[]")
        manifest_id = _require_safe_id(raw.get("id"), where="manifests[].id")
        if manifest_id in manifests:
            raise ContractError(f"duplicate manifest id {manifest_id}")
        manifest_path, manifest_sha = _artifact_ref(
            base, {"path": raw.get("path"), "sha256": raw.get("sha256")},
            where=f"manifest {manifest_id}",
        )
        barriers = raw.get("barrier_ids")
        if not isinstance(barriers, list):
            raise ContractError(f"manifest {manifest_id}: barrier_ids must be a list")
        manifest = load_json(manifest_path)
        jobs, meta = validate_manifest(
            manifest, label=f"manifest {manifest_id}", selected_barriers=barriers
        )
        manifests[manifest_id] = {
            "path": manifest_path,
            "sha256": manifest_sha,
            "jobs": jobs,
            "meta": meta,
            "barrier_ids": list(barriers),
        }
        for job_id, job in jobs.items():
            ref = f"{manifest_id}:{job_id}"
            selected_jobs[ref] = job
        common_judge.add(meta["judge_script_sha256"])
        common_train.add(meta["training_commit"])
        manifest_records.append(
            {
                "id": manifest_id,
                "sha256": manifest_sha,
                "barrier_ids": list(barriers),
                "selected_job_ids": sorted(jobs),
            }
        )
    if len(common_judge) != 1 or len(common_train) != 1:
        raise ContractError("selected manifests mix judge or training commits")

    raw_pairs = index.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ContractError("archive index requires explicit complete comparison pairs")
    pairs = [_validate_pair(pair, selected_jobs) for pair in raw_pairs if isinstance(pair, Mapping)]
    if len(pairs) != len(raw_pairs) or len({pair["id"] for pair in pairs}) != len(pairs):
        raise ContractError("pairs must be unique objects")
    coverage = [member for pair in pairs for member in pair["members"]]
    if len(coverage) != len(set(coverage)) or set(coverage) != set(selected_jobs):
        missing = sorted(set(selected_jobs) - set(coverage))
        repeated = sorted({value for value in coverage if coverage.count(value) > 1})
        extra = sorted(set(coverage) - set(selected_jobs))
        raise ContractError(
            f"comparison pairs do not partition selected jobs; missing={missing}, "
            f"repeated={repeated}, extra={extra}"
        )

    evidence_rows = index.get("evidence")
    if not isinstance(evidence_rows, list) or not evidence_rows:
        raise ContractError("archive index evidence must be a non-empty list")
    evidence_by_ref: dict[str, Mapping[str, Any]] = {}
    for raw in evidence_rows:
        if not isinstance(raw, Mapping):
            raise ContractError("evidence[] must be an object")
        _require_keys(
            raw,
            {"manifest_id", "job_id", "worker_state", "judge_report", "scorecard", "checkpoint_audit"},
            where="evidence[]",
        )
        ref = f"{raw.get('manifest_id')}:{raw.get('job_id')}"
        if ref in evidence_by_ref:
            raise ContractError(f"duplicate evidence for {ref}")
        evidence_by_ref[ref] = raw
    if set(evidence_by_ref) != set(selected_jobs):
        raise ContractError(
            "evidence does not exactly cover selected paired jobs; "
            f"missing={sorted(set(selected_jobs) - set(evidence_by_ref))}, "
            f"extra={sorted(set(evidence_by_ref) - set(selected_jobs))}"
        )

    output_records = []
    schedule_identities: set[str] = set()
    schedule_shas: set[str] = set()
    bank_shas: set[str] = set()
    eval_commits: set[str] = set()
    evaluator_shas: set[str] = set()
    for ref in sorted(selected_jobs):
        raw = evidence_by_ref[ref]
        manifest_id, job_id = ref.split(":", 1)
        manifest_info = manifests[manifest_id]
        job = selected_jobs[ref]
        state_path, state_sha = _artifact_ref(
            base, raw.get("worker_state"), where=f"{ref}.worker_state"
        )
        report_path, report_sha = _artifact_ref(
            base, raw.get("judge_report"), where=f"{ref}.judge_report"
        )
        scorecard_path, scorecard_sha = _artifact_ref(
            base, raw.get("scorecard"), where=f"{ref}.scorecard"
        )
        audit_path, audit_sha = _artifact_ref(
            base, raw.get("checkpoint_audit"), where=f"{ref}.checkpoint_audit"
        )
        state_binding = validate_worker_state(
            load_json(state_path),
            manifest_sha=manifest_info["sha256"],
            manifest_meta=manifest_info["meta"],
            job=job,
            state_sha=state_sha,
            where=f"{ref}.worker_state",
        )
        score = validate_scorecard(
            load_json(scorecard_path), job=job, scorecard_sha=scorecard_sha,
            where=f"{ref}.scorecard",
        )
        audit = validate_checkpoint_audit(
            load_json(audit_path), job=job, state_binding=state_binding,
            report_sha=report_sha, scorecard_sha=scorecard_sha, audit_sha=audit_sha,
            where=f"{ref}.checkpoint_audit",
        )
        if (
            audit["immutable_schedule_sha256"] != score["immutable_schedule_sha256"]
            or audit["evaluation_contract_exact"] is not score["evaluation_contract_exact"]
            or audit["evaluator_source_sha256"] != score["evaluator_source_sha256"]
        ):
            raise ContractError(f"{ref}: audit sidecar differs from the scorecard")
        report = validate_judge_report(
            report_path.read_text(encoding="utf-8"), job=job, report_sha=report_sha,
            score=score, where=f"{ref}.judge_report",
        )
        schedule_shas.add(score["immutable_schedule_sha256"])
        bank_shas.add(score["exam_bank_sha256"])
        schedule_identities.add(canonical_sha256(score["schedule_identity"]))
        eval_commits.add(state_binding["eval_commit"])
        evaluator_shas.add(score["evaluator_source_sha256"])
        output_records.append(
            {
                "ref": ref,
                "manifest_id": manifest_id,
                "job_id": job_id,
                "barrier_id": job["barrier_id"],
                "training_kind": job["training_kind"],
                "training_family": job["training_family"],
                "training_seed": job["training_seed"],
                "face_command_pairing": job["face_command_pairing"],
                "zero_joint_friction": job["zero_joint_friction"],
                **({"cell": job["cell"]} if "cell" in job else {}),
                "evaluation_role": job["evaluation_role"],
                "formal_target": job["formal_target"],
                "screen_only": True,
                "state": state_binding,
                "judge_report": report,
                "scorecard_sha256": scorecard_sha,
                "checkpoint_audit": audit,
                "metrics": score["metrics"],
            }
        )
    if len(schedule_shas) != 1 or len(schedule_identities) != 1 or len(bank_shas) != 1:
        raise ContractError(
            "mixed immutable schedules/banks are forbidden; archive each paper separately"
        )
    if len(eval_commits) != 1 or len(evaluator_shas) != 1:
        raise ContractError("selected evidence mixes evaluation commits/evaluator source bytes")

    content = {
        "archive_id": archive_id,
        "archive_role": "phase1_q10_direction_screen_only",
        "decision_authority": {
            "screen_only": True,
            "stop_authorized": False,
            "promote_authorized": False,
            "q50_triggered": False,
            "decision_claim": None,
        },
        "input_index_sha256": sha256_file(index_path),
        "manifests": sorted(manifest_records, key=lambda row: row["id"]),
        "provenance": {
            "training_commit": next(iter(common_train)),
            "eval_commit": next(iter(eval_commits)),
            "judge_script_sha256": next(iter(common_judge)),
            "evaluator_source_sha256": next(iter(evaluator_shas)),
        },
        "immutable_exam": {
            "schedule_k": Q10_SCHEDULE_K,
            "attempts_per_side": Q10_ATTEMPTS_PER_SIDE,
            "seed": 0,
            "noise_scale": 0.0,
            "schedule_sha256": next(iter(schedule_shas)),
            "schedule_identity_sha256": next(iter(schedule_identities)),
            "exam_bank_sha256": next(iter(bank_shas)),
        },
        "pairs": pairs,
        "records": output_records,
    }
    digest = canonical_sha256(content)
    document = {
        "schema_version": 1,
        "artifact_kind": "phase1_q10_curve_archive",
        "content_sha256": digest,
        "content": content,
    }
    return document, digest


def write_archive(index_path: Path, output_dir: Path) -> Path:
    document, digest = validate_index(index_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_id = document["content"]["archive_id"]
    path = output_dir / f"{archive_id}_{digest}.json"
    rendered = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ContractError(f"content-addressed path already contains different bytes: {path}")
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    os.replace(tmp, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate and print the content digest without writing an archive",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            if args.output_dir is not None:
                raise ContractError("--validate-only cannot be combined with --output-dir")
            document, digest = validate_index(args.index)
            print(json.dumps({
                "status": "pass_q10_screen_only",
                "content_sha256": digest,
                "archive_id": document["content"]["archive_id"],
                "job_count": len(document["content"]["records"]),
                "pair_count": len(document["content"]["pairs"]),
            }, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise ContractError("--output-dir is required unless --validate-only is used")
        path = write_archive(args.index, args.output_dir)
        print(path)
        return 0
    except ContractError as exc:
        print(f"[phase1-q10-collector][FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
