#!/usr/bin/env python3
"""Fail-closed continuation runner for the rolling-timing super-combo queue.

This entry point is deliberately separate from :mod:`run_lean_training_queue`:
the generic runner remains fresh-only, while this runner permits exactly one
kind of continuation -- strict full-state resume from a same-Pod, SHA-bound
parent.  ``validate`` and ``plan`` never contact a Pod.  ``fill`` is a dry run
unless the simulation-only confirmation token is supplied explicitly.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys
from typing import Any
import zlib

import yaml


# The sibling module owns the already-reviewed SSH, source-asset, exact Hydra
# compose, process-claim, GPU-lock, and Kit boot primitives.  Import it without
# turning scripts/ into a Python package or altering the fresh-only validator.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import run_lean_training_queue as lean  # noqa: E402


class ContinuationQueueError(RuntimeError):
    """The rolling continuation contract or one launch preflight failed."""


class ContinuationLaunchBatchError(ContinuationQueueError):
    """One cross-Pod launch batch failed after every submitted future settled.

    ``result`` deliberately retains successful siblings from the failed batch
    (and successful earlier batches) so callers can audit the exact attempted
    set without guessing or replaying a no-clobber namespace.
    """

    def __init__(self, result: dict[str, Any]):
        self.result = result
        successful = [row["job_id"] for row in result["launched"]]
        failed = [row["job_id"] for row in result["failed"]]
        super().__init__(
            "cross-Pod launch batch failed after all submitted attempts settled; "
            f"successful={successful}; failed={failed}; "
            f"attempted_count={result['attempted_count']}"
        )


CONFIRM = "SIM_ONLY_LAUNCH_ONE_ROLLING_CONTINUATION_JOB"
QUEUE_PATH = Path("configs/phase1_rolling_timing_supercombo_20260716.yaml")
EXPECTED_JOBS = 24
EXPECTED_ROUNDS = 4
EXPECTED_SLOTS = tuple(
    f"{pod}/gpu{gpu}" for pod in ("pod1", "pod2") for gpu in (0, 1, 2)
)
EXPECTED_OFFSETS = [200, 500, 1000, 2000]
EXPECTED_ADDITIONAL_BUDGET = 2001
ACTIVATED_PREREGISTRATION_STATUS = "activated_demo_only_inexact"
PARENT_KEY_RE = re.compile(r"^pod([12])_[A-Za-z0-9_]+$")

# Every item below is generated once by this harness.  In particular, a YAML
# row may not retain a fresh-run ``checkpoint_path=null`` and then silently
# receive a second continuation checkpoint later on argv.
HARNESS_OWNED_KEYS = set(lean.HARNESS_OWNED_OVERRIDE_KEYS) | {
    "checkpoint_path",
    "checkpoint_tolerant",
    "checkpoint_allow_missing_contract",
    "checkpoint_allow_contract_mismatch",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContinuationQueueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContinuationQueueError(f"{label} must be a list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContinuationQueueError(f"{label} must be a non-empty string")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ContinuationQueueError(f"{label} must be one line")
    return value


def _sha256(value: Any, label: str, *, allow_pending: bool = False) -> str:
    text = _text(value, label)
    if allow_pending and "PENDING" in text.upper():
        return text
    if not lean.SHA256.fullmatch(text):
        raise ContinuationQueueError(f"{label} must be 64 lowercase hex characters")
    return text


def _workspace_path(value: Any, label: str, *, allow_pending: bool = False) -> str:
    text = _text(value, label)
    if allow_pending and "PENDING" in text.upper():
        return text
    parsed = PurePosixPath(text)
    if not parsed.is_absolute() or not text.startswith("/workspace/"):
        raise ContinuationQueueError(f"{label} must be an absolute /workspace path")
    if ".." in parsed.parts or str(parsed) != text.rstrip("/"):
        raise ContinuationQueueError(f"{label} must be normalized and contain no ..")
    return text.rstrip("/")


def _pending(value: Any) -> bool:
    return isinstance(value, str) and (
        "PENDING" in value.upper() or value.lower().startswith("blocked_")
    )


def _parent_pod(parent_name: str) -> str:
    match = PARENT_KEY_RE.fullmatch(parent_name)
    if match is None:
        raise ContinuationQueueError(
            f"warm-start parent {parent_name!r} must begin with pod1_ or pod2_"
        )
    return f"pod{match.group(1)}"


def _compile_recipe(job: Mapping[str, Any], label: str) -> list[str]:
    """Compile base+delta to one final Hydra value per key.

    A delta may override the corresponding base key once.  Duplicate keys
    within either layer remain an error, and generated keys are forbidden in
    both layers.  Overridden base entries are removed before deltas are added,
    so the executed argv contains each key exactly once.
    """

    recipe = _mapping(job.get("recipe"), f"{label}.recipe")
    base = _list(recipe.get("base"), f"{label}.recipe.base")
    delta = _list(recipe.get("delta"), f"{label}.recipe.delta")
    if not base:
        raise ContinuationQueueError(f"{label}.recipe.base must not be empty")

    def layer(arguments: list[Any], name: str) -> tuple[list[tuple[str, str]], set[str]]:
        rows: list[tuple[str, str]] = []
        keys: set[str] = set()
        for index, raw in enumerate(arguments):
            argument = _text(raw, f"{label}.recipe.{name}[{index}]")
            try:
                key = lean._override_key(argument, f"{label}.recipe.{name}[{index}]")
            except lean.QueueError as exc:
                raise ContinuationQueueError(str(exc)) from exc
            if key in keys:
                raise ContinuationQueueError(
                    f"{label}.recipe.{name} sets final Hydra key {key!r} more than once"
                )
            if key in HARNESS_OWNED_KEYS:
                raise ContinuationQueueError(
                    f"{label}.recipe may not inject harness-owned key {key!r}"
                )
            keys.add(key)
            rows.append((key, argument))
        return rows, keys

    base_rows, _base_keys = layer(base, "base")
    delta_rows, delta_keys = layer(delta, "delta")

    motion = _mapping(job.get("motion"), f"{label}.motion")
    bindings = _mapping(motion.get("bindings"), f"{label}.motion.bindings")
    generated = {
        lean._generated_override_key(key, f"{label}.motion binding")
        for key in bindings
    }
    bank = _mapping(job.get("bank"), f"{label}.bank")
    generated.add(
        lean._generated_override_key(bank.get("train_arg"), f"{label}.bank.train_arg")
    )
    collisions = generated.intersection(
        {key for key, _argument in [*base_rows, *delta_rows]}
    )
    if collisions:
        raise ContinuationQueueError(
            f"{label}.recipe duplicates generated final keys: {sorted(collisions)}"
        )

    compiled = [argument for key, argument in base_rows if key not in delta_keys]
    compiled.extend(argument for _key, argument in delta_rows)
    final_keys = [
        lean._override_key(argument, f"{label}.compiled recipe") for argument in compiled
    ]
    if len(final_keys) != len(set(final_keys)):
        raise ContinuationQueueError(f"{label}.compiled recipe contains duplicate final keys")
    return compiled


def _parent_records(queue: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    selection = _mapping(queue.get("parent_selection"), "parent_selection")
    records: dict[str, dict[str, Any]] = {}
    for key, value in selection.items():
        if PARENT_KEY_RE.fullmatch(key):
            records[key] = _mapping(value, f"parent_selection.{key}")
    if not records:
        raise ContinuationQueueError("parent_selection has no pod-bound parent records")
    return records


def _validate_parent_record(name: str, record: Mapping[str, Any]) -> None:
    prefix = f"parent_selection.{name}"
    for path_key in (
        "original_queue_claim_path",
        "original_run_binding_path",
        "selected_rsl_log_dir",
        "selected_checkpoint_path",
        "selected_hard_contract_path",
    ):
        _workspace_path(record.get(path_key), f"{prefix}.{path_key}", allow_pending=True)
    for digest_key in (
        "original_queue_claim_sha256",
        "original_run_binding_sha256",
        "selected_checkpoint_sha256",
        "selected_hard_contract_sha256",
    ):
        _sha256(record.get(digest_key), f"{prefix}.{digest_key}", allow_pending=True)
    embedded = record.get("selected_embedded_iteration")
    if type(embedded) is not int or embedded <= 0:
        raise ContinuationQueueError(
            f"{prefix}.selected_embedded_iteration must be a positive integer"
        )
    original_job_id = _text(record.get("original_job_id"), f"{prefix}.original_job_id")
    if not lean.SAFE_ID.fullmatch(original_job_id):
        raise ContinuationQueueError(f"{prefix}.original_job_id is unsafe")
    if any(
        _pending(record.get(key))
        for key in (
            "selected_rsl_log_dir",
            "selected_checkpoint_path",
            "selected_hard_contract_path",
        )
    ):
        return
    rsl = PurePosixPath(record["selected_rsl_log_dir"])
    checkpoint = PurePosixPath(record["selected_checkpoint_path"])
    hard = PurePosixPath(record["selected_hard_contract_path"])
    if checkpoint.parent != rsl:
        raise ContinuationQueueError(f"{prefix}.checkpoint must be inside selected_rsl_log_dir")
    expected_filename = f"model_{embedded}.pt"
    if checkpoint.name != expected_filename:
        raise ContinuationQueueError(
            f"{prefix}.checkpoint filename must be {expected_filename}"
        )
    if record.get("ranking_snapshot_checkpoint_filename") != expected_filename:
        raise ContinuationQueueError(
            f"{prefix}.ranking checkpoint filename differs from embedded iteration"
        )
    if hard != rsl / "params" / "training_contract.json":
        raise ContinuationQueueError(
            f"{prefix}.hard contract must be selected_rsl_log_dir/params/training_contract.json"
        )


def _validate_job(
    queue: Mapping[str, Any],
    job: dict[str, Any],
    index: int,
    parents: Mapping[str, dict[str, Any]],
) -> None:
    label = f"jobs[{index}]"
    job_id = _text(job.get("id"), f"{label}.id")
    if not lean.SAFE_ID.fullmatch(job_id):
        raise ContinuationQueueError(f"{label}.id is unsafe")
    _text(job.get("human_name"), f"{job_id}.human_name")
    action = _text(job.get("action"), f"{job_id}.action")
    if not lean.SAFE_ID.fullmatch(action):
        raise ContinuationQueueError(f"{job_id}.action is unsafe")
    if job.get("status") not in {lean.READY, lean.BLOCKED, *lean.TERMINAL}:
        raise ContinuationQueueError(f"{job_id}.status is unsupported")
    if job["status"] == lean.BLOCKED:
        _text(job.get("blocker"), f"{job_id}.blocker")
    elif job.get("blocker") not in (None, ""):
        raise ContinuationQueueError(f"{job_id}.blocker must be empty unless blocked")
    launch_round = job.get("launch_round")
    if type(launch_round) is not int or launch_round not in range(1, EXPECTED_ROUNDS + 1):
        raise ContinuationQueueError(f"{job_id}.launch_round must be 1..{EXPECTED_ROUNDS}")

    motion = _mapping(job.get("motion"), f"{job_id}.motion")
    bank = _mapping(job.get("bank"), f"{job_id}.bank")
    exam = _mapping(job.get("exam"), f"{job_id}.exam")
    if motion.get("action") != action or bank.get("action") != action or exam.get("action") != action:
        raise ContinuationQueueError(f"{job_id} motion/bank/exam actions must equal job action")
    bindings = _mapping(motion.get("bindings"), f"{job_id}.motion.bindings")
    if not bindings:
        raise ContinuationQueueError(f"{job_id}.motion.bindings must not be empty")
    for key, path in bindings.items():
        if not isinstance(key, str) or not lean.SAFE_ID.fullmatch(key):
            raise ContinuationQueueError(f"{job_id}.motion binding key is unsafe")
        _workspace_path(path, f"{job_id}.motion.{key}")
    _workspace_path(bank.get("train_path"), f"{job_id}.bank.train_path")
    _text(bank.get("train_arg"), f"{job_id}.bank.train_arg")
    _workspace_path(exam.get("path"), f"{job_id}.exam.path")
    _text(exam.get("family"), f"{job_id}.exam.family")

    source = _mapping(job.get("source"), f"{job_id}.source")
    source_checkout = _workspace_path(
        source.get("checkout"), f"{job_id}.source.checkout", allow_pending=True
    )
    commit = _text(source.get("commit"), f"{job_id}.source.commit")
    if not _pending(commit) and not lean.COMMIT.fullmatch(commit):
        raise ContinuationQueueError(f"{job_id}.source.commit must be a full Git commit")
    try:
        lean._validate_ignored_runtime_asset(source, f"{job_id}.source")
    except lean.QueueError as exc:
        raise ContinuationQueueError(str(exc)) from exc
    if job.get("runtime_binding") is not True:
        raise ContinuationQueueError(f"{job_id}.runtime_binding must be true")

    _compile_recipe(job, job_id)
    if type(job.get("seed")) is not int or job["seed"] < 0:
        raise ContinuationQueueError(f"{job_id}.seed must be a non-negative integer")
    budget = _mapping(job.get("budget"), f"{job_id}.budget")
    if type(budget.get("num_envs")) is not int or budget["num_envs"] <= 0:
        raise ContinuationQueueError(f"{job_id}.budget.num_envs must be positive")
    if budget.get("max_iterations") != EXPECTED_ADDITIONAL_BUDGET:
        raise ContinuationQueueError(
            f"{job_id}.budget.max_iterations must be additional budget "
            f"{EXPECTED_ADDITIONAL_BUDGET}"
        )
    if budget.get("save_interval") != 100:
        raise ContinuationQueueError(f"{job_id}.budget.save_interval must be 100")
    if budget.get("iteration_semantics") != "additional_updates_after_full_state_resume":
        raise ContinuationQueueError(f"{job_id}.budget iteration semantics changed")
    if job.get("milestones") != EXPECTED_OFFSETS:
        raise ContinuationQueueError(f"{job_id}.milestones must be {EXPECTED_OFFSETS}")
    if job.get("milestone_semantics") != "offsets_from_attested_parent":
        raise ContinuationQueueError(f"{job_id}.milestone semantics changed")

    resource = _mapping(job.get("resource"), f"{job_id}.resource")
    if set(resource) != {"policy", "required_slot"}:
        raise ContinuationQueueError(
            f"{job_id}.resource must contain only policy and required_slot"
        )
    if resource.get("policy") != "dispatch_gpu_round_robin":
        raise ContinuationQueueError(f"{job_id}.resource policy changed")
    required_slot = _text(resource.get("required_slot"), f"{job_id}.required_slot")
    if required_slot not in EXPECTED_SLOTS:
        raise ContinuationQueueError(f"{job_id}.required_slot is invalid")

    warm = _mapping(job.get("warm_start"), f"{job_id}.warm_start")
    parent_name = _text(warm.get("parent"), f"{job_id}.warm_start.parent")
    if parent_name not in parents:
        raise ContinuationQueueError(f"{job_id} references unknown parent {parent_name!r}")
    if _parent_pod(parent_name) != required_slot.split("/", 1)[0]:
        raise ContinuationQueueError(f"{job_id} parent and required_slot are on different Pods")
    if warm.get("checkpoint_path") != parents[parent_name].get("selected_checkpoint_path"):
        raise ContinuationQueueError(f"{job_id} warm_start checkpoint differs from parent selection")
    expected_warm = {
        "transfer_mode": "strict_full_state_preserve_optimizer",
        "checkpoint_tolerant": False,
        "allow_missing_contract": False,
        "allow_contract_mismatch": True,
        "descendant_exact_eligible": False,
    }
    for key, expected in expected_warm.items():
        if warm.get(key) != expected:
            raise ContinuationQueueError(f"{job_id}.warm_start.{key} must be {expected!r}")
    if job.get("formal_evidence_eligible") is not False:
        raise ContinuationQueueError(f"{job_id} must remain formal-ineligible")

    run_name = _text(job.get("run_name"), f"{job_id}.run_name")
    if not lean.SAFE_ID.fullmatch(run_name):
        raise ContinuationQueueError(f"{job_id}.run_name is unsafe")
    run_dir = _workspace_path(job.get("run_dir"), f"{job_id}.run_dir")
    if not _pending(source_checkout):
        source_path = PurePosixPath(source_checkout)
        run_path = PurePosixPath(run_dir)
        if run_path == source_path or source_path in run_path.parents:
            raise ContinuationQueueError(f"{job_id}.run_dir must remain outside source")


def load_queue(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContinuationQueueError(f"cannot read queue YAML {path}: {exc}") from exc
    queue = _mapping(raw, "queue")
    if queue.get("schema_version") != 1 or queue.get("simulation_only") is not True:
        raise ContinuationQueueError("queue must be schema_version=1 and simulation_only=true")
    if type(queue.get("launch_authorized")) is not bool:
        raise ContinuationQueueError("launch_authorized must be an explicit boolean")
    if queue.get("formal_evidence_eligible") is not False:
        raise ContinuationQueueError("rolling descendants must be formal-ineligible")
    ssh = _mapping(queue.get("ssh"), "ssh")
    _text(ssh.get("key"), "ssh.key")
    pods = _mapping(queue.get("pods"), "pods")
    if list(pods) != ["pod1", "pod2"] or queue.get("dispatch_pods") != ["pod1", "pod2"]:
        raise ContinuationQueueError("pods and dispatch_pods must be ordered pod1, pod2")
    for pod_name in ("pod1", "pod2"):
        pod = _mapping(pods.get(pod_name), pod_name)
        _text(pod.get("host"), f"{pod_name}.host")
        if type(pod.get("port")) is not int or pod["port"] <= 0:
            raise ContinuationQueueError(f"{pod_name}.port must be positive")
        if pod.get("gpus") != [0, 1, 2]:
            raise ContinuationQueueError(f"{pod_name}.gpus must be [0, 1, 2]")
        if pod.get("max_trainers_per_gpu") != 4:
            raise ContinuationQueueError(f"{pod_name} capacity must be exactly four per GPU")

    parents = _parent_records(queue)
    for name, record in parents.items():
        _validate_parent_record(name, record)

    jobs = _list(queue.get("jobs"), "jobs")
    if len(jobs) != EXPECTED_JOBS:
        raise ContinuationQueueError(f"queue must contain exactly {EXPECTED_JOBS} jobs")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    seen_dirs: set[str] = set()
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"jobs[{index}]")
        _validate_job(queue, job, index, parents)
        for value, seen, label in (
            (job["id"], seen_ids, "job id"),
            (job["run_name"], seen_names, "run_name"),
            (job["run_dir"], seen_dirs, "run_dir"),
        ):
            if value in seen:
                raise ContinuationQueueError(f"duplicate {label}: {value}")
            seen.add(value)

    expected_slot_set = set(EXPECTED_SLOTS)
    for launch_round in range(1, EXPECTED_ROUNDS + 1):
        round_jobs = [job for job in jobs if job["launch_round"] == launch_round]
        round_slots = [job["resource"]["required_slot"] for job in round_jobs]
        if len(round_jobs) != 6 or set(round_slots) != expected_slot_set or len(round_slots) != len(set(round_slots)):
            raise ContinuationQueueError(
                f"launch_round {launch_round} must contain each of the six GPUs exactly once"
            )
    pod_counts = {
        pod: sum(job["resource"]["required_slot"].startswith(pod + "/") for job in jobs)
        for pod in ("pod1", "pod2")
    }
    if pod_counts != {"pod1": 12, "pod2": 12}:
        raise ContinuationQueueError(f"queue must bind 12 jobs per Pod, got {pod_counts}")
    referenced_parents = {job["warm_start"]["parent"] for job in jobs}
    if len(referenced_parents) != 3:
        raise ContinuationQueueError(
            f"queue must reference exactly three unique parents, got {sorted(referenced_parents)}"
        )
    slot_counts = {
        slot: sum(job["resource"]["required_slot"] == slot for job in jobs)
        for slot in EXPECTED_SLOTS
    }
    if any(count != 4 for count in slot_counts.values()):
        raise ContinuationQueueError(f"each GPU must have four jobs, got {slot_counts}")

    blocking = _mapping(queue.get("blocking_contract"), "blocking_contract")
    bound_checkout = blocking.get("source_checkout")
    bound_commit = blocking.get("source_commit")
    if not _pending(bound_checkout):
        _workspace_path(bound_checkout, "blocking_contract.source_checkout")
        if any(job["source"]["checkout"] != bound_checkout for job in jobs):
            raise ContinuationQueueError(
                "job source checkout differs from blocking_contract.source_checkout"
            )
    if not _pending(bound_commit):
        if not isinstance(bound_commit, str) or not lean.COMMIT.fullmatch(bound_commit):
            raise ContinuationQueueError(
                "blocking_contract.source_commit must be a full Git commit"
            )
        if any(job["source"]["commit"] != bound_commit for job in jobs):
            raise ContinuationQueueError(
                "job source commit differs from blocking_contract.source_commit"
            )
    predecessor = _mapping(
        queue.get("predecessor_stop_contract"), "predecessor_stop_contract"
    )
    for pod in ("pod1", "pod2"):
        receipt = _mapping(predecessor.get(pod), f"predecessor_stop_contract.{pod}")
        receipt_path = receipt.get("stop_receipt_path")
        receipt_sha = receipt.get("stop_receipt_sha256")
        if (receipt_path is None) != (receipt_sha is None):
            raise ContinuationQueueError(
                f"predecessor_stop_contract.{pod} receipt path/SHA must be both set or both null"
            )
        if receipt_path is None:
            if receipt.get("reported_fatal_count") not in (None, 0):
                raise ContinuationQueueError(f"{pod} exact stop audit reports a fatal run")
            if "no_separate_receipt" in str(predecessor.get("evidence_state", "")):
                _text(
                    receipt.get("stop_audit_result"),
                    f"predecessor_stop_contract.{pod}.stop_audit_result",
                )
        else:
            _workspace_path(
                receipt_path,
                f"predecessor_stop_contract.{pod}.stop_receipt_path",
                allow_pending=True,
            )
            _sha256(
                receipt_sha,
                f"predecessor_stop_contract.{pod}.stop_receipt_sha256",
                allow_pending=True,
            )
    return queue


def activation_blockers(queue: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if queue.get("launch_authorized") is not True:
        blockers.append("launch_authorized is false")
    if queue.get("preregistration_status") != ACTIVATED_PREREGISTRATION_STATUS:
        blockers.append(
            "preregistration_status must be "
            f"{ACTIVATED_PREREGISTRATION_STATUS!r}, got "
            f"{queue.get('preregistration_status')!r}"
        )
    blocking = _mapping(queue.get("blocking_contract"), "blocking_contract")
    for key in ("source_checkout", "source_commit"):
        if _pending(blocking.get(key)):
            blockers.append(f"blocking_contract.{key} is pending")
    evidence = blocking.get("source_full_scene_probe_evidence")
    if not isinstance(evidence, dict):
        blockers.append("source_full_scene_probe_evidence must be a pass mapping")
    else:
        required_evidence = {
            "training_runtime_status": "passed_natural_exit_rc0",
            "first_iteration_observed": True,
            "tensor_nonfinite_count": 0,
            "fatal_count": 0,
            "training_contract_lineage_exact": 1,
            "process_group_naturally_empty": True,
        }
        for key, expected in required_evidence.items():
            if evidence.get(key) != expected:
                blockers.append(
                    f"source_full_scene_probe_evidence.{key} must be {expected!r}"
                )
        if type(evidence.get("checkpoint_iteration")) is not int or evidence[
            "checkpoint_iteration"
        ] <= 0:
            blockers.append(
                "source_full_scene_probe_evidence.checkpoint_iteration must be positive"
            )
        for key in ("checkpoint_sha256", "hard_contract_sha256"):
            value = evidence.get(key)
            if not isinstance(value, str) or not lean.SHA256.fullmatch(value):
                blockers.append(
                    f"source_full_scene_probe_evidence.{key} must be SHA-256"
                )
    harness = blocking.get("hotstart_harness")
    if not isinstance(harness, dict):
        blockers.append("hotstart_harness must be a reviewed pass mapping")
    else:
        _runner_raw, expected_runner_sha = _runner_payload()
        if harness.get("runner_script_sha256") != expected_runner_sha:
            blockers.append("hotstart_harness.runner_script_sha256 differs from runner bytes")
        if harness.get("reviewed_tests_passed") is not True:
            blockers.append("hotstart_harness.reviewed_tests_passed must be true")
        count = harness.get("reviewed_test_count")
        if type(count) is not int or count < 80:
            blockers.append("hotstart_harness.reviewed_test_count must be >= 80")
    predecessor = _mapping(queue.get("predecessor_stop_contract"), "predecessor_stop_contract")
    if _pending(predecessor.get("evidence_state")):
        blockers.append("predecessor stop evidence is pending")
    for pod in ("pod1", "pod2"):
        row = _mapping(predecessor.get(pod), f"predecessor_stop_contract.{pod}")
        for key in ("stop_receipt_path", "stop_receipt_sha256"):
            if _pending(row.get(key)):
                blockers.append(f"{pod} {key} is pending")
    selection = _mapping(queue.get("parent_selection"), "parent_selection")
    if _pending(selection.get("selection_state")):
        blockers.append("parent selection is pending")
    for name, record in _parent_records(queue).items():
        if record.get("selection_is_final") is not True:
            blockers.append(f"{name} selection_is_final is not true")
        if record.get("immutable_stop_receipt_still_required") is True:
            blockers.append(f"{name} still requires immutable stop receipt")
        for key in (
            "original_queue_claim_path",
            "original_queue_claim_sha256",
            "original_run_binding_path",
            "original_run_binding_sha256",
            "selected_checkpoint_path",
            "selected_checkpoint_sha256",
            "selected_hard_contract_path",
            "selected_hard_contract_sha256",
        ):
            if _pending(record.get(key)):
                blockers.append(f"{name}.{key} is pending")
    blocked_jobs = [job["id"] for job in queue["jobs"] if job["status"] == lean.BLOCKED]
    if blocked_jobs:
        blockers.append(f"{len(blocked_jobs)} jobs remain blocked")
    for job in queue["jobs"]:
        if _pending(job["source"]["checkout"]) or _pending(job["source"]["commit"]):
            blockers.append("one or more job sources remain pending")
            break
    # Preserve order while avoiding a 24-row repetition of one global blocker.
    return list(dict.fromkeys(blockers))


def _absolute_schedule(
    job: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    parent_name = job["warm_start"]["parent"]
    parent_iteration = parents[parent_name]["selected_embedded_iteration"]
    absolute_iteration_exclusive_bound = (
        parent_iteration + job["budget"]["max_iterations"]
    )
    milestones = [parent_iteration + offset for offset in job["milestones"]]
    if milestones[-1] >= absolute_iteration_exclusive_bound:
        raise ContinuationQueueError(
            f"{job['id']} terminal milestone is unreachable after absolute conversion"
        )
    if any(value % job["budget"]["save_interval"] for value in milestones):
        raise ContinuationQueueError(
            f"{job['id']} absolute milestones do not align with save_interval"
        )
    return {
        "parent": parent_name,
        "parent_iteration": parent_iteration,
        "additional_iterations": job["budget"]["max_iterations"],
        "absolute_iteration_exclusive_bound": absolute_iteration_exclusive_bound,
        "milestones": milestones,
    }


def validate_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    blockers = activation_blockers(queue)
    return {
        "mode": "validate",
        "schema_valid": True,
        "activation_ready": not blockers,
        "job_count": len(queue["jobs"]),
        "pod_job_counts": {pod: 12 for pod in ("pod1", "pod2")},
        "per_gpu_jobs": 4,
        "blockers": blockers,
    }


def cmd_plan(queue: Mapping[str, Any]) -> dict[str, Any]:
    parents = _parent_records(queue)
    blockers = activation_blockers(queue)
    rows = []
    for job in queue["jobs"]:
        absolute = _absolute_schedule(job, parents)
        rows.append(
            {
                "launch_round": job["launch_round"],
                "job_id": job["id"],
                "required_slot": job["resource"]["required_slot"],
                "status": job["status"],
                **absolute,
                "absolute_checkpoint_filenames": [
                    f"model_{iteration}.pt" for iteration in absolute["milestones"]
                ],
            }
        )
    return {
        "mode": "plan",
        "dry_run": True,
        "activation_ready": not blockers,
        "blockers": blockers,
        "jobs": rows,
    }


def _training_argv(
    queue: Mapping[str, Any], job: Mapping[str, Any], slot: lean.Slot
) -> list[str]:
    del queue  # the complete queue is bound through the claim's job material.
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{lean.WBT_RELATIVE}"
    run_dir = job["run_dir"].rstrip("/")
    absolute = _absolute_schedule(job, _parent_records_from_job_context(job))
    argv = [
        lean.ISAAC_PYTHON,
        f"{workdir}/{lean.ENTRYPOINT_RELATIVE}",
        *_compile_recipe(job, job["id"]),
    ]
    for key, path in job["motion"]["bindings"].items():
        argv.append(f"{key}={path}")
    argv.extend(
        [
            f"{job['bank']['train_arg']}={job['bank']['train_path']}",
            f"seed={job['seed']}",
            f"num_envs={job['budget']['num_envs']}",
            # RSL-RL interprets max_iterations as the number of updates to run
            # *after* loading a resumed checkpoint.  The human-facing plan and
            # milestone filenames remain absolute iterations.
            f"max_iterations={absolute['additional_iterations']}",
            f"algo.runner.save_interval={job['budget']['save_interval']}",
            f"run_name={job['run_name']}",
            "device=cuda:0",
            f"checkpoint_path={job['warm_start']['checkpoint_path']}",
            "checkpoint_tolerant=false",
            "checkpoint_allow_missing_contract=false",
            "checkpoint_allow_contract_mismatch=true",
            f"++training_queue_claim_path={run_dir}/queue_claim.json",
            f"++training_run_binding_path={run_dir}/run_binding.json",
        ]
    )
    final_keys = [
        lean._override_key(argument, f"{job['id']} final argv") for argument in argv[2:]
    ]
    if len(final_keys) != len(set(final_keys)):
        raise ContinuationQueueError(f"{job['id']} final training argv contains duplicate keys")
    if slot.name != job["resource"]["required_slot"]:
        raise ContinuationQueueError(
            f"{job['id']} requires {job['resource']['required_slot']}, got {slot.name}"
        )
    return argv


def _parent_records_from_job_context(job: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the one lightweight parent record cached during queue loading."""

    record = job.get("_continuation_parent_record")
    if not isinstance(record, dict):
        raise ContinuationQueueError("job lacks validated continuation parent context")
    return {job["warm_start"]["parent"]: record}


def _bind_parent_context(queue: dict[str, Any]) -> None:
    parents = _parent_records(queue)
    for job in queue["jobs"]:
        # Private in-memory material is never serialized into the claim's source
        # mapping; it prevents helpers from depending on a mutable global queue.
        job["_continuation_parent_record"] = parents[job["warm_start"]["parent"]]


def _launch_contract(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    slot: lean.Slot,
    *,
    runner_script_sha256: str | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if runner_script_sha256 is None:
        _runner_raw, runner_script_sha256 = _runner_payload()
    absolute = _absolute_schedule(job, _parent_records_from_job_context(job))
    argv_without_claim = _training_argv(queue, job, slot)
    parent = job["_continuation_parent_record"]
    content = {
        "schema_version": 1,
        "job_id": job["id"],
        "action": job["action"],
        "pod": slot.pod,
        "gpu": slot.gpu,
        "source": dict(job["source"]),
        "run_name": job["run_name"],
        "run_dir": job["run_dir"],
        "runtime_binding": True,
        "seed": job["seed"],
        "formal_evidence_eligible": False,
        "continuation": {
            "transfer_mode": "strict_full_state_preserve_optimizer",
            "parent_name": job["warm_start"]["parent"],
            "parent_original_queue_claim_path": parent[
                "original_queue_claim_path"
            ],
            "parent_original_queue_claim_sha256": parent[
                "original_queue_claim_sha256"
            ],
            "parent_original_run_binding_path": parent[
                "original_run_binding_path"
            ],
            "parent_original_run_binding_sha256": parent[
                "original_run_binding_sha256"
            ],
            "parent_rsl_log_dir": parent["selected_rsl_log_dir"],
            "parent_checkpoint_path": parent["selected_checkpoint_path"],
            "parent_checkpoint_sha256": parent["selected_checkpoint_sha256"],
            "parent_hard_contract_path": parent["selected_hard_contract_path"],
            "parent_hard_contract_sha256": parent["selected_hard_contract_sha256"],
            "parent_iteration": absolute["parent_iteration"],
            "allow_contract_mismatch": True,
            "descendant_exact_eligible": False,
            "continuation_runner_script_sha256": runner_script_sha256,
        },
        "budget": {
            "num_envs": job["budget"]["num_envs"],
            "additional_iterations": absolute["additional_iterations"],
            "trainer_max_iterations_arg": absolute["additional_iterations"],
            "absolute_iteration_exclusive_bound": absolute[
                "absolute_iteration_exclusive_bound"
            ],
            "save_interval": job["budget"]["save_interval"],
            "milestones": absolute["milestones"],
            "milestone_offsets": list(job["milestones"]),
        },
        "inputs": {
            "motion": {
                "action": job["motion"]["action"],
                "bindings": dict(job["motion"]["bindings"]),
            },
            "bank": {
                "action": job["bank"]["action"],
                "train_path": job["bank"]["train_path"],
                "train_arg": job["bank"]["train_arg"],
            },
            "exam": {
                "action": job["exam"]["action"],
                "path": job["exam"]["path"],
                "family": job["exam"]["family"],
            },
        },
        "training_argv_without_claim": argv_without_claim,
    }
    digest = _canonical_sha256(content)
    execution_argv = [
        *argv_without_claim,
        f"++training_launch_claim_sha256={digest}",
    ]
    claim = {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": execution_argv,
    }
    return claim, execution_argv, absolute


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _stable_regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ContinuationQueueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ContinuationQueueError(f"{label} must be a nonempty regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if _stat_signature(opened) != _stat_signature(before):
            raise ContinuationQueueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = path.lstat()
    if _stat_signature(before) != _stat_signature(after_fd) or _stat_signature(before) != _stat_signature(after_path):
        raise ContinuationQueueError(f"{label} changed while reading")
    return b"".join(chunks)


def _stable_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_regular_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationQueueError(f"{label} is not JSON") from exc
    return _mapping(value, label), raw


def _tensor_audit(value: Any, torch_module: Any) -> dict[str, int]:
    floating_tensors = 0
    floating_elements = 0
    nonfinite_elements = 0
    seen: set[int] = set()

    def visit(item: Any) -> None:
        nonlocal floating_tensors, floating_elements, nonfinite_elements
        if isinstance(item, torch_module.Tensor):
            if torch_module.is_floating_point(item) or torch_module.is_complex(item):
                floating_tensors += 1
                elements = int(item.numel())
                floating_elements += elements
                finite = int(torch_module.isfinite(item).sum().item())
                nonfinite_elements += elements - finite
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            for child in item:
                visit(child)

    visit(value)
    return {
        "floating_tensor_count": floating_tensors,
        "floating_elements": floating_elements,
        "nonfinite_floating_elements": nonfinite_elements,
    }


def _validate_checkpoint_payload(
    checkpoint: Any,
    *,
    expected_iteration: int,
    expected_hard_sha256: str,
    torch_module: Any,
) -> dict[str, int]:
    checkpoint = _mapping(checkpoint, "parent checkpoint")
    if checkpoint.get("iter") != expected_iteration:
        raise ContinuationQueueError("parent checkpoint embedded iteration mismatch")
    model = _mapping(checkpoint.get("model_state_dict"), "model_state_dict")
    if not model:
        raise ContinuationQueueError("parent model_state_dict must be nonempty")
    model_keys = tuple(model)
    actor_key_count = sum(key.startswith("actor.") for key in model_keys)
    critic_key_count = sum(key.startswith("critic.") for key in model_keys)
    if actor_key_count <= 0 or critic_key_count <= 0:
        raise ContinuationQueueError(
            "parent checkpoint must contain both actor.* and critic.* model keys"
        )
    optimizer = _mapping(checkpoint.get("optimizer_state_dict"), "optimizer_state_dict")
    state = _mapping(optimizer.get("state"), "optimizer_state_dict.state")
    groups = optimizer.get("param_groups")
    if not optimizer or not state:
        raise ContinuationQueueError("parent optimizer state must be nonempty")
    if not isinstance(groups, list) or not groups:
        raise ContinuationQueueError("parent optimizer param_groups must be nonempty")
    infos = _mapping(checkpoint.get("infos"), "parent checkpoint infos")
    if infos.get("training_contract_schema_version") != 3:
        raise ContinuationQueueError("parent checkpoint hard-contract schema binding is not 3")
    if infos.get("training_contract_sha256") != expected_hard_sha256:
        raise ContinuationQueueError("parent checkpoint hard-contract SHA binding mismatch")
    lineage = infos.get("training_contract_lineage_exact")
    if type(lineage) is not int or lineage not in (0, 1):
        raise ContinuationQueueError("parent checkpoint lineage flag must be 0 or 1")
    audit = _tensor_audit(checkpoint, torch_module)
    if audit["floating_tensor_count"] <= 0:
        raise ContinuationQueueError("parent checkpoint contains no floating tensors")
    if audit["nonfinite_floating_elements"] != 0:
        raise ContinuationQueueError("parent checkpoint contains non-finite tensors")
    return {
        **audit,
        "actor_model_key_count": actor_key_count,
        "critic_model_key_count": critic_key_count,
        "optimizer_resume_eligible": 1,
    }


def _validate_parent_spec(spec: Mapping[str, Any], *, torch_module: Any) -> dict[str, Any]:
    paths = {
        key: Path(_workspace_path(spec.get(key), f"parent.{key}"))
        for key in ("claim_path", "binding_path", "checkpoint_path", "hard_contract_path", "rsl_log_dir")
    }
    expected = {
        key: _sha256(spec.get(key), f"parent.{key}")
        for key in ("claim_sha256", "binding_sha256", "checkpoint_sha256", "hard_contract_sha256")
    }
    iteration = spec.get("embedded_iteration")
    if type(iteration) is not int or iteration <= 0:
        raise ContinuationQueueError("parent.embedded_iteration must be positive")
    if paths["checkpoint_path"].name != f"model_{iteration}.pt":
        raise ContinuationQueueError("parent checkpoint filename/iteration mismatch")
    if paths["checkpoint_path"].parent != paths["rsl_log_dir"]:
        raise ContinuationQueueError("parent checkpoint is outside bound RSL log dir")
    if paths["hard_contract_path"] != paths["rsl_log_dir"] / "params" / "training_contract.json":
        raise ContinuationQueueError("parent hard contract is outside bound RSL log dir")

    claim, claim_raw = _stable_json(paths["claim_path"], "parent queue claim")
    binding, binding_raw = _stable_json(paths["binding_path"], "parent run binding")
    hard, hard_raw = _stable_json(paths["hard_contract_path"], "parent hard contract")
    if hashlib.sha256(claim_raw).hexdigest() != expected["claim_sha256"]:
        raise ContinuationQueueError("parent queue claim SHA mismatch")
    if hashlib.sha256(binding_raw).hexdigest() != expected["binding_sha256"]:
        raise ContinuationQueueError("parent run binding SHA mismatch")
    if hashlib.sha256(hard_raw).hexdigest() != expected["hard_contract_sha256"]:
        raise ContinuationQueueError("parent hard contract SHA mismatch")
    if hard.get("schema_version") != 3:
        raise ContinuationQueueError("parent hard contract is not schema 3")

    if claim.get("schema_version") != 2:
        raise ContinuationQueueError("parent queue claim is not schema 2")
    claim_content = _mapping(claim.get("content"), "parent queue claim content")
    claim_digest = _sha256(claim.get("content_sha256"), "parent claim content digest")
    if _canonical_sha256(claim_content) != claim_digest:
        raise ContinuationQueueError("parent queue claim canonical digest mismatch")
    if binding.get("schema_version") != 1:
        raise ContinuationQueueError("parent run binding is not schema 1")
    binding_content = _mapping(binding.get("content"), "parent run binding content")
    binding_digest = _sha256(binding.get("content_sha256"), "parent binding digest")
    if _canonical_sha256(binding_content) != binding_digest:
        raise ContinuationQueueError("parent run binding canonical digest mismatch")
    if binding_content.get("claim_path") != str(paths["claim_path"]):
        raise ContinuationQueueError("parent binding points to a different claim")
    if binding_content.get("binding_path") != str(paths["binding_path"]):
        raise ContinuationQueueError("parent binding path does not self-bind")
    if binding_content.get("claim_content_sha256") != claim_digest:
        raise ContinuationQueueError("parent binding/claim digest mismatch")
    if binding_content.get("rsl_log_dir") != str(paths["rsl_log_dir"]):
        raise ContinuationQueueError("parent binding points to a different RSL log dir")
    if binding_content.get("job_id") != spec.get("original_job_id"):
        raise ContinuationQueueError("parent binding job id mismatch")

    checkpoint_before = paths["checkpoint_path"].lstat()
    if not stat.S_ISREG(checkpoint_before.st_mode) or checkpoint_before.st_size <= 0:
        raise ContinuationQueueError("parent checkpoint must be a nonempty regular file")
    try:
        checkpoint = torch_module.load(
            paths["checkpoint_path"], map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch_module.load(paths["checkpoint_path"], map_location="cpu")
    checkpoint_after_load = paths["checkpoint_path"].lstat()
    if _stat_signature(checkpoint_before) != _stat_signature(checkpoint_after_load):
        raise ContinuationQueueError("parent checkpoint changed while loading")
    audit = _validate_checkpoint_payload(
        checkpoint,
        expected_iteration=iteration,
        expected_hard_sha256=expected["hard_contract_sha256"],
        torch_module=torch_module,
    )
    checkpoint_raw = _stable_regular_bytes(paths["checkpoint_path"], "parent checkpoint")
    if hashlib.sha256(checkpoint_raw).hexdigest() != expected["checkpoint_sha256"]:
        raise ContinuationQueueError("parent checkpoint SHA mismatch")
    if _stat_signature(checkpoint_before) != _stat_signature(paths["checkpoint_path"].lstat()):
        raise ContinuationQueueError("parent checkpoint changed while hashing")
    return {
        "parent": spec["parent_name"],
        "embedded_iteration": iteration,
        "optimizer_state_entries": len(checkpoint["optimizer_state_dict"]["state"]),
        "optimizer_param_groups": len(checkpoint["optimizer_state_dict"]["param_groups"]),
        **audit,
    }


def _parent_validation_entry() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("one base64 parent specification is required")
    try:
        spec = json.loads(base64.b64decode(sys.argv[1], validate=True))
        import torch

        result = _validate_parent_spec(_mapping(spec, "parent spec"), torch_module=torch)
    except Exception as exc:  # one fail-closed remote diagnostic line
        print(f"ROLLING_PARENT_INVALID: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, sort_keys=True))


def _parent_spec(job: Mapping[str, Any]) -> dict[str, Any]:
    parent = job["_continuation_parent_record"]
    return {
        "parent_name": job["warm_start"]["parent"],
        "original_job_id": parent["original_job_id"],
        "claim_path": parent["original_queue_claim_path"],
        "claim_sha256": parent["original_queue_claim_sha256"],
        "binding_path": parent["original_run_binding_path"],
        "binding_sha256": parent["original_run_binding_sha256"],
        "rsl_log_dir": parent["selected_rsl_log_dir"],
        "checkpoint_path": parent["selected_checkpoint_path"],
        "checkpoint_sha256": parent["selected_checkpoint_sha256"],
        "hard_contract_path": parent["selected_hard_contract_path"],
        "hard_contract_sha256": parent["selected_hard_contract_sha256"],
        "embedded_iteration": parent["selected_embedded_iteration"],
    }


def _runner_payload() -> tuple[bytes, str]:
    raw = _stable_regular_bytes(Path(__file__).resolve(), "continuation runner script")
    return raw, hashlib.sha256(raw).hexdigest()


def _parent_validation_command(
    job: Mapping[str, Any], runner_raw: bytes, runner_sha256: str
) -> str:
    if hashlib.sha256(runner_raw).hexdigest() != runner_sha256:
        raise ContinuationQueueError("continuation runner changed before parent validation")
    remote_filename = (
        f"{job['source']['checkout'].rstrip('/')}/scripts/"
        "run_phase1_rolling_timing_supercombo_queue.py"
    )
    program = (
        "import base64,hashlib,sys,zlib;"
        "raw=zlib.decompress(base64.b64decode(sys.argv[1],validate=True));"
        f"assert hashlib.sha256(raw).hexdigest()=={runner_sha256!r};"
        f"ns={{'__name__':'embedded_rolling_parent_validator','__file__':{remote_filename!r}}};"
        "exec(compile(raw,ns['__file__'],'exec'),ns);"
        "sys.argv=['rolling-parent-validator',sys.argv[2]];"
        "ns['_parent_validation_entry']()"
    )
    # Stay comfortably below Linux's per-argv-string limit after the remote
    # shell receives the complete atomic launch transaction.
    encoded_runner = base64.b64encode(zlib.compress(runner_raw, level=9)).decode("ascii")
    encoded_spec = base64.b64encode(
        json.dumps(_parent_spec(job), sort_keys=True, separators=(",", ":")).encode()
    ).decode("ascii")
    return shlex.join(
        [lean.ISAAC_PYTHON, "-c", program, encoded_runner, encoded_spec]
    )


def _doctor_body(
    queue: Mapping[str, Any],
    job: Mapping[str, Any],
    slot: lean.Slot,
    argv: list[str],
    runner_raw: bytes,
    runner_sha256: str,
) -> str:
    try:
        source_doctor = lean._doctor_body(
            queue, job, slot, training_argv=argv
        )
    except lean.QueueError as exc:
        raise ContinuationQueueError(str(exc)) from exc
    return source_doctor + (
        "\n# same-Pod strict full-state parent attestation\n"
        + _parent_validation_command(job, runner_raw, runner_sha256)
        + " >&2\n"
    )


def _launch_script(
    queue: Mapping[str, Any], job: Mapping[str, Any], slot: lean.Slot
) -> str:
    runner_raw, runner_sha256 = _runner_payload()
    claim, argv, absolute = _launch_contract(
        queue,
        job,
        slot,
        runner_script_sha256=runner_sha256,
    )
    run_dir = job["run_dir"].rstrip("/")
    run_parent = str(PurePosixPath(run_dir).parent)
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{lean.WBT_RELATIVE}"
    launcher = f"{workdir}/{lean.KIT_LAUNCHER_RELATIVE}"
    claim_text = json.dumps(
        claim, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ) + "\n"
    launch = shlex.join([launcher, f"{run_dir}/run.log"]) + " " + (
        lean._child_env_command(argv, slot.gpu)
    ) + f" {lean.GPU_LAUNCH_LOCK_FD}>&-"
    first_iteration = absolute["parent_iteration"] + 1
    marker = (
        "Learning iteration "
        f"{first_iteration}/{absolute['absolute_iteration_exclusive_bound']}"
    )
    body = _doctor_body(
        queue, job, slot, argv, runner_raw, runner_sha256
    ) + f"""
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(lean.UNIQUE_NUMERIC_PID_AWK)})
test "$count" -lt {slot.capacity}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
mkdir {shlex.quote(run_dir + '/milestones')}
( set -o noclobber; printf %s {shlex.quote(claim_text)} > {shlex.quote(run_dir + '/queue_claim.json')} )
test ! -e {shlex.quote(run_dir + '/run_binding.json')}
export KIT_BOOT_MARKER={shlex.quote(marker)}
export KIT_BOOT_TIMEOUT_S={lean.KIT_BOOT_TIMEOUT_SECONDS}
{launch}
printf '%s\n' {shlex.quote(f"phase=first_iter expected_iteration={first_iteration} absolute_iteration_exclusive_bound={absolute['absolute_iteration_exclusive_bound']} parent_iteration={absolute['parent_iteration']}")} >> {shlex.quote(run_dir + '/run.log.launch')}
"""
    return lean._gpu_launch_lock_script(slot, body)


def _slots(queue: Mapping[str, Any]) -> dict[str, lean.Slot]:
    return {slot.name: slot for slot in lean.slots(queue)}


def _assign_current_round(
    queue: Mapping[str, Any],
    occupancy: Mapping[str, int],
    claims: Mapping[str, Mapping[str, Any]],
) -> list[tuple[dict[str, Any], lean.Slot]]:
    pending = [
        job
        for job in queue["jobs"]
        if job["status"] == lean.READY and job["id"] not in claims
    ]
    if not pending:
        return []
    current_round = min(job["launch_round"] for job in pending)
    slots = _slots(queue)
    assignments: list[tuple[dict[str, Any], lean.Slot]] = []
    for job in pending:
        if job["launch_round"] != current_round:
            continue
        slot = slots[job["resource"]["required_slot"]]
        if occupancy.get(slot.name) is None:
            raise ContinuationQueueError(f"live snapshot omitted {slot.name}")
        if occupancy[slot.name] < slot.capacity:
            assignments.append((job, slot))
    return assignments


def _validate_live_claim_slots(
    queue: Mapping[str, Any], claims: Mapping[str, Mapping[str, Any]]
) -> None:
    jobs = {job["id"]: job for job in queue["jobs"]}
    for job_id, claim in claims.items():
        job = jobs.get(job_id)
        if job is None:
            raise ContinuationQueueError(f"live claim references unknown job {job_id}")
        observed = f"{claim.get('pod')}/gpu{claim.get('gpu')}"
        expected = job["resource"]["required_slot"]
        if observed != expected:
            raise ContinuationQueueError(
                f"{job_id} live claim occupies {observed}, expected required_slot {expected}"
            )


def _dry_assignments(queue: Mapping[str, Any], count: int) -> list[tuple[dict[str, Any], lean.Slot]]:
    occupancy = {slot: 0 for slot in EXPECTED_SLOTS}
    claims: dict[str, dict[str, Any]] = {}
    result: list[tuple[dict[str, Any], lean.Slot]] = []
    while len(result) < count:
        assignments = _assign_current_round(queue, occupancy, claims)
        if not assignments:
            break
        for job, slot in assignments:
            if len(result) >= count:
                break
            result.append((job, slot))
            claims[job["id"]] = {"pod": slot.pod, "gpu": slot.gpu}
            occupancy[slot.name] += 1
    return result


def _cross_pod_launch_batch(
    assignments: list[tuple[dict[str, Any], lean.Slot]], remaining: int
) -> list[tuple[dict[str, Any], lean.Slot]]:
    """Select at most one ready job per Pod for one concurrent batch."""

    if remaining <= 0:
        return []
    result: list[tuple[dict[str, Any], lean.Slot]] = []
    selected_pods: set[str] = set()
    for job, slot in assignments:
        if slot.pod in selected_pods:
            continue
        result.append((job, slot))
        selected_pods.add(slot.pod)
        if len(result) >= remaining:
            break
    return result


def _unique_parent_jobs(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one representative job per referenced parent in YAML order."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in queue["jobs"]:
        parent_name = job["warm_start"]["parent"]
        if parent_name in seen:
            continue
        seen.add(parent_name)
        expected_pod = _parent_pod(parent_name)
        if not job["resource"]["required_slot"].startswith(expected_pod + "/"):
            raise ContinuationQueueError(
                f"{parent_name} representative job is not bound to {expected_pod}"
            )
        result.append(job)
    return result


def cmd_inspect_parents(queue: dict[str, Any]) -> dict[str, Any]:
    """Read and validate each unique parent, using at most one SSH per Pod.

    The command is intentionally available while the science queue remains
    blocked.  It does not run source hydration, Hydra, Kit, a trainer, or any
    remote write.  A timeout/error propagates once; there is no replay loop.
    """

    runner_raw, runner_sha256 = _runner_payload()
    representatives = _unique_parent_jobs(queue)
    by_pod: dict[str, list[dict[str, Any]]] = {"pod1": [], "pod2": []}
    for job in representatives:
        parent_name = job["warm_start"]["parent"]
        by_pod[_parent_pod(parent_name)].append(job)

    remote_by_pod: dict[str, str] = {}
    expected_by_pod: dict[str, list[str]] = {}
    for pod, jobs in by_pod.items():
        if not jobs:
            continue
        expected_by_pod[pod] = [job["warm_start"]["parent"] for job in jobs]
        remote_by_pod[pod] = "set -euo pipefail\n" + "\n".join(
            _parent_validation_command(job, runner_raw, runner_sha256)
            for job in jobs
        )

    def inspect(pod: str) -> tuple[str, str]:
        try:
            output = lean._run_ssh(
                queue,
                pod,
                remote_by_pod[pod],
                timeout=180,
                phase="rolling-continuation-inspect-parents",
            )
        except lean.QueueError as exc:
            raise ContinuationQueueError(str(exc)) from exc
        return pod, output

    # Parallel only across Pods.  Pod2's two parents share one connection and
    # are checked sequentially inside that read-only remote shell.
    pod_names = list(remote_by_pod)
    with ThreadPoolExecutor(max_workers=len(pod_names)) as pool:
        outputs = dict(pool.map(inspect, pod_names))

    results: list[dict[str, Any]] = []
    for pod in pod_names:
        lines = [line for line in outputs[pod].splitlines() if line.strip()]
        expected_names = expected_by_pod[pod]
        if len(lines) != len(expected_names):
            raise ContinuationQueueError(
                f"{pod} parent validator returned {len(lines)} rows, "
                f"expected {len(expected_names)}"
            )
        for expected_name, line in zip(expected_names, lines, strict=True):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContinuationQueueError(
                    f"{pod} parent validator returned malformed JSON"
                ) from exc
            result = _mapping(result, f"{pod} parent validator result")
            if result.get("parent") != expected_name:
                raise ContinuationQueueError(
                    f"{pod} parent validator returned {result.get('parent')!r}, "
                    f"expected {expected_name!r}"
                )
            expected_iteration = _parent_records(queue)[expected_name][
                "selected_embedded_iteration"
            ]
            if result.get("embedded_iteration") != expected_iteration:
                raise ContinuationQueueError(
                    f"{expected_name} validator iteration differs from queue parent"
                )
            for key in (
                "optimizer_state_entries",
                "optimizer_param_groups",
                "actor_model_key_count",
                "critic_model_key_count",
                "floating_tensor_count",
                "floating_elements",
            ):
                if type(result.get(key)) is not int or result[key] <= 0:
                    raise ContinuationQueueError(
                        f"{expected_name} validator {key} must be positive"
                    )
            if result.get("nonfinite_floating_elements") != 0:
                raise ContinuationQueueError(
                    f"{expected_name} validator found non-finite parent tensors"
                )
            if result.get("optimizer_resume_eligible") != 1:
                raise ContinuationQueueError(
                    f"{expected_name} parent is not optimizer-resume eligible"
                )
            results.append({"pod": pod, **result})
    return {
        "mode": "inspect-parents",
        "read_only": True,
        "runner_script_sha256": runner_sha256,
        "unique_parent_count": len(representatives),
        "ssh_connections": {pod: 1 for pod in pod_names},
        "parents": results,
    }


def cmd_fill(
    queue: dict[str, Any], *, count: int, execute: bool, confirm: str | None
) -> dict[str, Any]:
    if type(count) is not int or count <= 0:
        raise ContinuationQueueError("fill --count must be a positive integer")
    blockers = activation_blockers(queue)
    if blockers:
        raise ContinuationQueueError("fill is blocked: " + "; ".join(blockers))
    if execute and confirm != CONFIRM:
        raise ContinuationQueueError(f"--execute requires --confirm {CONFIRM}")
    if not execute:
        assignments = _dry_assignments(queue, count)
        return {
            "mode": "fill",
            "dry_run": True,
            "count_limit": count,
            "jobs": [
                {
                    "launch_round": job["launch_round"],
                    "job_id": job["id"],
                    "required_slot": slot.name,
                    **_absolute_schedule(job, _parent_records_from_job_context(job)),
                    "ssh_argv": [
                        *lean._ssh_prefix(queue, slot.pod),
                        f"bash -lc {shlex.quote(_launch_script(queue, job, slot))}",
                    ],
                }
                for job, slot in assignments
            ],
        }

    launched: list[dict[str, Any]] = []
    attempted_count = 0
    # Remote claims remain the durable no-clobber truth.  This local overlay is
    # additionally required within one invocation: a transiently incomplete
    # next snapshot must never cause an already-submitted job to be replayed.
    attempted_claims: dict[str, dict[str, Any]] = {}
    lean.GLOBAL_SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with lean.GLOBAL_SCHEDULER_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        while attempted_count < count:
            try:
                occupancy, claims = lean.live_snapshot(queue)
                _validate_live_claim_slots(queue, claims)
                effective = lean._effective_occupancy(queue, occupancy, claims)
            except lean.QueueError as exc:
                raise ContinuationQueueError(str(exc)) from exc
            scheduling_claims = dict(claims)
            for job_id, synthetic in attempted_claims.items():
                scheduling_claims.setdefault(job_id, synthetic)
            _validate_live_claim_slots(queue, scheduling_claims)
            assignments = _assign_current_round(
                queue, effective, scheduling_claims
            )
            if not assignments:
                break
            batch = _cross_pod_launch_batch(assignments, count - attempted_count)
            if not batch:
                break

            def launch_one(job: dict[str, Any], slot: lean.Slot) -> str:
                return lean._run_ssh(
                    queue,
                    slot.pod,
                    _launch_script(queue, job, slot),
                    timeout=lean.KIT_BOOT_TIMEOUT_SECONDS + 60,
                    phase=f"rolling-continuation-launch:{job['id']}",
                )

            # Concurrency exists only across Pods.  There is never more than
            # one future per Pod in a batch, and the next batch is not sampled
            # until both futures have settled.  The remote per-Pod host boot
            # lock remains an independent second serialization boundary.
            batch_failed: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                for job, slot in batch:
                    attempted_claims[job["id"]] = {
                        "pod": slot.pod,
                        "gpu": slot.gpu,
                        # Avoid double-counting an already-visible NVML PID;
                        # this overlay exists only for job-id exclusion.
                        "state": "launched",
                    }
                futures = [
                    (job, slot, pool.submit(launch_one, job, slot))
                    for job, slot in batch
                ]
                attempted_count += len(futures)
                for job, slot, future in futures:
                    try:
                        output = future.result()
                    except Exception as exc:
                        batch_failed.append(
                            {
                                "launch_round": job["launch_round"],
                                "job_id": job["id"],
                                "required_slot": slot.name,
                                "error_kind": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    else:
                        launched.append(
                            {
                                "launch_round": job["launch_round"],
                                "job_id": job["id"],
                                "required_slot": slot.name,
                                "remote_output": output,
                            }
                        )

            if batch_failed:
                # Never continue, retry, or replay after a failed submitted
                # launch.  The structured exception preserves successful
                # siblings and makes the exact attempted set explicit.
                raise ContinuationLaunchBatchError(
                    {
                        "mode": "fill",
                        "dry_run": False,
                        "count_limit": count,
                        "attempted_count": attempted_count,
                        "scheduler_lock": str(lean.GLOBAL_SCHEDULER_LOCK),
                        "launched": launched,
                        "failed": batch_failed,
                    }
                )
    if not launched:
        raise ContinuationQueueError("no unclaimed job in the current launch round fits its required slot")
    return {
        "mode": "fill",
        "dry_run": False,
        "count_limit": count,
        "attempted_count": attempted_count,
        "scheduler_lock": str(lean.GLOBAL_SCHEDULER_LOCK),
        "launched": launched,
    }


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue.resolve())
        _bind_parent_context(queue)
        if args.mode == "validate":
            result = validate_queue(queue)
        elif args.mode == "plan":
            result = cmd_plan(queue)
        elif args.mode == "inspect-parents":
            result = cmd_inspect_parents(queue)
        elif args.mode == "fill":
            result = cmd_fill(
                queue,
                count=args.count,
                execute=args.execute,
                confirm=args.confirm,
            )
        else:  # pragma: no cover - argparse owns the mode surface.
            raise ContinuationQueueError(f"unsupported mode: {args.mode}")
    except ContinuationLaunchBatchError as exc:
        print(
            "BATCH_RESULT="
            + json.dumps(
                exc.result,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (ContinuationQueueError, lean.QueueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
