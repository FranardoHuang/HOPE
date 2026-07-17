#!/usr/bin/env python3
"""Plan, probe, launch, and prune the task-entry ready-ledger successor.

The old 19 checkpoint receipts proved checkpoint integrity but could not rank
readiness: planner-owned preparation deliberately disabled every legacy hold,
while the old ledger sampled readiness only in a hold.  This queue starts a
new namespace and requires the task-entry denominators added in source commit
``d7c38fcf`` before any science cell can launch.

Every command is dry-run/read-only unless it has both ``--execute`` and the
exact command-specific confirmation token.  A missing probe or parent digest
keeps ``fill`` fail closed; no command retries itself or signals a robot.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys
from typing import Any, Mapping
import zlib

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import run_phase1_task_revision_supercombo_queue as taskrev  # noqa: E402

continuation = taskrev.continuation
lean = continuation.lean
yaml = continuation.yaml


class ReadySuccessorError(RuntimeError):
    """The ready-successor contract or one explicit operation failed."""


QUEUE_PATH = Path("configs/phase1_task_revision_ready_successor_20260717.yaml")
PENDING_STATUS = "pending_ready_ledger_full_scene_probe_and_parent_binding"
ACTIVATED_STATUS = "activated_ready_ledger_successor_inexact"
SOURCE_COMMIT = "d7c38fcf70e7e9420800437fd5b467168ae72580"
SOURCE_CHECKOUT = "/workspace/codexschema/nohope_main_d7c38fcf"
NAMESPACE_ROOT = "/workspace/codexschema/phase1_task_revision_ready_successor_20260717"
PARENT_NAME = "pod2_equal_reward_model5700"
PARENT_ITERATION = 5700
PARENT_BINDING_CONTENT_SHA256 = (
    "9fe7528abd43e3420b890ad789cfcc46c29a3038ac9ec86e291ac886801661f1"
)
OFFSETS = [200, 500, 1000]
ABSOLUTE_MILESTONES = [5900, 6200, 6700]
JOB_IDS = (
    "ready_baseline_qdot_minus5",
    "ready_baseline_qdot_zero",
    "ready_strong_qdot_minus5",
    "ready_strong_qdot_zero",
)
FIXED_ROUND_ONE = {
    "ready_baseline_qdot_minus5": "pod2/gpu0",
    "ready_baseline_qdot_zero": "pod2/gpu1",
    "ready_strong_qdot_minus5": "pod2/gpu2",
}
DYNAMIC_FOURTH = "ready_strong_qdot_zero"
PROBE_JOB = "ready_baseline_qdot_zero"
PROBE_GPU = 1
PARENT_INSPECT_CONFIRM = "SIM_ONLY_INSPECT_ONE_READY_SUCCESSOR_PARENT"
EXECUTED_PARENT_INSPECTOR_RUNNER_SHA256 = (
    "c8d731b5d954d178ab9c9e0071dd71cc2c37b80c4084c049b1e375d977431659"
)
EXECUTED_PARENT_INSPECTOR_PROGRAM_SHA256 = (
    "7ba57d95cb76ecd2fb81812f3d8e9505777aa8e9b0e74e629e5ac1f97b566c86"
)
PROBE_CONFIRM = "SIM_ONLY_RUN_ONE_READY_SUCCESSOR_FULL_SCENE_PROBE"
PROBE_FINALIZE_CONFIRM = "SIM_ONLY_FINALIZE_ONE_READY_SUCCESSOR_FULL_SCENE_PROBE"
FILL_CONFIRM = "SIM_ONLY_LAUNCH_READY_SUCCESSOR_CELLS"
BEHAVIOR_INSPECT_CONFIRM = "SIM_ONLY_INSPECT_ONE_READY_SUCCESSOR_BEHAVIOR"
BEHAVIOR_ATTEST_CONFIRM = "SIM_ONLY_ATTEST_ONE_READY_SUCCESSOR_BEHAVIOR"
PORTFOLIO_INSPECT_CONFIRM = "SIM_ONLY_INSPECT_READY_SUCCESSOR_PORTFOLIO"
PORTFOLIO_ATTEST_CONFIRM = "SIM_ONLY_ATTEST_READY_SUCCESSOR_PORTFOLIO"
EXACT_STOP_CONFIRM = "SIM_ONLY_EXACT_STOP_ONE_READY_SUCCESSOR_FAILURE"
LOCAL_PROBE_FINALIZE_CONFIRM = "SIM_ONLY_LOCAL_FINALIZE_READY_SUCCESSOR_PROBE"
LOCAL_BEHAVIOR_INSPECT_CONFIRM = "SIM_ONLY_LOCAL_INSPECT_READY_SUCCESSOR_BEHAVIOR"
LOCAL_BEHAVIOR_ATTEST_CONFIRM = "SIM_ONLY_LOCAL_ATTEST_READY_SUCCESSOR_BEHAVIOR"
LOCAL_PORTFOLIO_INSPECT_CONFIRM = "SIM_ONLY_LOCAL_INSPECT_READY_SUCCESSOR_PORTFOLIO"
LOCAL_PORTFOLIO_ATTEST_CONFIRM = "SIM_ONLY_LOCAL_ATTEST_READY_SUCCESSOR_PORTFOLIO"
LOCAL_EXACT_STOP_CONFIRM = "SIM_ONLY_LOCAL_EXACT_STOP_READY_SUCCESSOR_FAILURE"
PENDING_PREFIX = "PENDING_"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXACT_EVENT_PREFIX = "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
READY_COUNTERS = {
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
    "ready_phase_sample_count",
    "ready_planner_task_entry_sample_count",
    "ready_planner_legacy_hold_violation_count",
    "ready_foot_sensor_unavailable_sample_count",
    "ready_nonfinite_value_count",
}


PARENT_INSPECT_PROGRAM = r'''
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

spec = json.loads(base64.b64decode(sys.argv[1], validate=True))

def canonical(value):
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")).hexdigest()

def strict_json(raw, label):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    def reject_nonfinite(value):
        raise RuntimeError(f"non-finite JSON in {label}: {value}")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be a mapping")
    return value

def stable_bytes(path, label):
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
            raise RuntimeError(f"{label} must be a nonempty single-link regular file")
        chunks = []
        while True:
            row = os.read(fd, 1024 * 1024)
            if not row:
                break
            chunks.append(row)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    outside = path.lstat()
    signature = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns,
    )
    if signature(before) != signature(after) or signature(after) != signature(outside):
        raise RuntimeError(f"{label} changed during stable read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"{label} stable read was incomplete")
    return raw

def stable_json(path, label):
    raw = stable_bytes(path, label)
    return strict_json(raw, label), raw

def envelope(value, label):
    if set(value) != {"schema_version", "content", "content_sha256"}:
        raise RuntimeError(f"{label} envelope shape changed")
    content = value.get("content")
    if not isinstance(content, dict) or canonical(content) != value.get("content_sha256"):
        raise RuntimeError(f"{label} content digest mismatch")
    return content

claim_path = Path(spec["queue_claim_path"])
binding_path = Path(spec["run_binding_path"])
milestone_path = Path(spec["milestone_receipt_path"])
expected_checkpoint = Path(spec["checkpoint_path"])
expected_rsl = Path(spec["rsl_log_dir"])

claim, claim_raw = stable_json(claim_path, "parent queue claim")
if set(claim) != {"schema_version", "content", "content_sha256", "training_argv"}:
    raise RuntimeError("parent queue claim envelope shape changed")
claim_content = claim.get("content")
if (claim.get("schema_version") != 2 or not isinstance(claim_content, dict) or
        canonical(claim_content) != claim.get("content_sha256") or
        claim.get("content_sha256") != spec["expected_claim_content_sha256"]):
    raise RuntimeError("parent queue claim content differs from known digest")
if (claim_content.get("job_id") != spec["job_id"] or
        claim_content.get("pod") != "pod2" or
        claim_content.get("gpu") != 0 or
        claim_content.get("run_dir") != spec["run_dir"]):
    raise RuntimeError("parent queue claim identity differs")
argv = claim.get("training_argv")
if not isinstance(argv, list) or any(type(row) is not str for row in argv):
    raise RuntimeError("parent queue claim argv is malformed")

binding, binding_raw = stable_json(binding_path, "parent run binding")
binding_content = envelope(binding, "parent run binding")
if (binding_content.get("job_id") != spec["job_id"] or
        binding_content.get("pod") != "pod2" or
        binding_content.get("gpu") != 0 or
        binding_content.get("run_dir") != spec["run_dir"] or
        binding_content.get("claim_path") != str(claim_path) or
        binding_content.get("binding_path") != str(binding_path) or
        binding_content.get("claim_content_sha256") != spec["expected_claim_content_sha256"] or
        binding_content.get("rsl_log_dir") != str(expected_rsl)):
    raise RuntimeError("parent run binding identity differs")

milestone, milestone_raw = stable_json(milestone_path, "parent milestone receipt")
milestone_content = envelope(milestone, "parent milestone receipt")
checkpoint_binding = milestone_content.get("checkpoint")
hard_binding = milestone_content.get("hard_contract")
if (milestone_content.get("job_id") != spec["job_id"] or
        milestone_content.get("milestone") != spec["milestone"] or
        milestone_content.get("claim_content_sha256") != spec["expected_claim_content_sha256"] or
        milestone_content.get("binding_path") != str(binding_path) or
        milestone_content.get("binding_content_sha256") != binding["content_sha256"] or
        not isinstance(checkpoint_binding, dict) or
        not isinstance(hard_binding, dict)):
    raise RuntimeError("parent milestone receipt identity differs")
if checkpoint_binding.get("path") != str(expected_checkpoint):
    raise RuntimeError("parent milestone checkpoint path differs")
hard_path = Path(hard_binding.get("path", ""))
if hard_path != expected_rsl / "params" / "training_contract.json":
    raise RuntimeError("parent hard contract path is outside the bound RSL log dir")

checkpoint_raw = stable_bytes(expected_checkpoint, "parent checkpoint")
checkpoint_sha = hashlib.sha256(checkpoint_raw).hexdigest()
if checkpoint_sha != checkpoint_binding.get("sha256"):
    raise RuntimeError("parent checkpoint SHA differs from milestone receipt")
hard, hard_raw = stable_json(hard_path, "parent hard contract")
hard_sha = hashlib.sha256(hard_raw).hexdigest()
if hard_sha != hard_binding.get("sha256") or hard.get("schema_version") != 3:
    raise RuntimeError("parent hard contract differs from milestone receipt/schema3")

import torch
try:
    checkpoint = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(expected_checkpoint, map_location="cpu")
if not isinstance(checkpoint, dict) or checkpoint.get("iter") != spec["milestone"]:
    raise RuntimeError("parent checkpoint embedded iteration differs")
model = checkpoint.get("model_state_dict")
optimizer = checkpoint.get("optimizer_state_dict")
infos = checkpoint.get("infos")
if (not isinstance(model, dict) or not model or
        not isinstance(optimizer, dict) or not optimizer.get("state") or
        not optimizer.get("param_groups") or not isinstance(infos, dict) or
        infos.get("training_contract_schema_version") != 3 or
        infos.get("training_contract_sha256") != hard_sha or
        infos.get("training_launch_claim_sha256") != spec["expected_claim_content_sha256"] or
        infos.get("training_contract_lineage_exact") != 0 or
        hard_binding.get("lineage_exact") != 0):
    raise RuntimeError("parent checkpoint is not a full-state schema3 resume parent")
floating_tensors = 0
floating_elements = 0
nonfinite_elements = 0
seen = set()
def visit(value):
    global floating_tensors, floating_elements, nonfinite_elements
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value) or torch.is_complex(value):
            floating_tensors += 1
            floating_elements += value.numel()
            nonfinite_elements += value.numel() - int(torch.isfinite(value).sum().item())
        return
    if isinstance(value, dict):
        if id(value) in seen:
            return
        seen.add(id(value))
        for child in value.values():
            visit(child)
    elif isinstance(value, (list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        for child in value:
            visit(child)
visit(checkpoint)
if floating_tensors <= 0 or nonfinite_elements != 0:
    raise RuntimeError("parent checkpoint tensor audit failed")

result = {
    "schema_version": 1,
    "status": "passed",
    "read_only": True,
    "no_write": True,
    "no_signal": True,
    "job_id": spec["job_id"],
    "milestone": spec["milestone"],
    "expected_claim_content_sha256": spec["expected_claim_content_sha256"],
    "queue_claim": {
        "path": str(claim_path),
        "file_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "content_sha256": claim["content_sha256"],
    },
    "run_binding": {
        "path": str(binding_path),
        "file_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "content_sha256": binding["content_sha256"],
    },
    "milestone_receipt": {
        "path": str(milestone_path),
        "file_sha256": hashlib.sha256(milestone_raw).hexdigest(),
        "content_sha256": milestone["content_sha256"],
        "binding_content_sha256": milestone_content["binding_content_sha256"],
    },
    "checkpoint": {
        "path": str(expected_checkpoint),
        "file_sha256": checkpoint_sha,
        "embedded_iteration": checkpoint["iter"],
        "floating_tensor_count": floating_tensors,
        "floating_element_count": floating_elements,
        "nonfinite_element_count": nonfinite_elements,
        "optimizer_state_present": True,
        "training_launch_claim_sha256": infos["training_launch_claim_sha256"],
        "lineage_exact": infos["training_contract_lineage_exact"],
    },
    "hard_contract": {
        "path": str(hard_path),
        "file_sha256": hard_sha,
        "schema_version": hard["schema_version"],
        "lineage_exact": hard_binding["lineage_exact"],
    },
    "parent_selection_patch": {
        "original_queue_claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "original_run_binding_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "selected_checkpoint_sha256": checkpoint_sha,
        "selected_hard_contract_path": str(hard_path),
        "selected_hard_contract_sha256": hard_sha,
        "selection_is_final": True,
    },
}
print(json.dumps(result, allow_nan=False, ensure_ascii=False,
                 separators=(",", ":"), sort_keys=True))
'''.strip()
REQUIRED_READY_INVARIANTS = (
    "ready_phase_sample_count_positive",
    "ready_planner_task_entry_sample_count_equals_ready_phase_sample_count",
    "ready_tilt_eligible_sample_count_equals_ready_phase_sample_count",
    "ready_base_speed_eligible_sample_count_equals_ready_phase_sample_count",
    "ready_station_offset_eligible_sample_count_equals_ready_phase_sample_count",
    "ready_foot_contact_eligible_plus_unavailable_equals_ready_phase_sample_count",
    "ready_foot_slip_eligible_plus_unavailable_equals_ready_phase_sample_count",
    "ready_planner_legacy_hold_violation_count_equals_zero",
    "ready_nonfinite_value_count_equals_zero",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReadySuccessorError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReadySuccessorError(f"{label} must be a list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReadySuccessorError(f"{label} must be a non-empty string")
    return value


def _pending(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PENDING_PREFIX)


def _sha(value: Any, label: str, *, allow_pending: bool = False) -> str:
    value = _text(value, label)
    if allow_pending and _pending(value):
        return value
    if SHA256.fullmatch(value) is None:
        raise ReadySuccessorError(f"{label} must be a lowercase SHA-256")
    return value


def _workspace(value: Any, label: str, *, allow_pending: bool = False) -> str:
    value = _text(value, label)
    if allow_pending and _pending(value):
        return value
    path = PurePosixPath(value)
    if not path.is_absolute() or not value.startswith("/workspace/") or ".." in path.parts:
        raise ReadySuccessorError(f"{label} must be a safe absolute /workspace path")
    return value


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _runner_sha() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _required_counters(queue: Mapping[str, Any]) -> set[str]:
    ledger = _mapping(queue.get("exact_behavior_ledger_contract"), "exact ledger")
    return set(_list(ledger.get("required_counters"), "required_counters"))


def _job(queue: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    for job in queue["jobs"]:
        if job["id"] == job_id:
            return job
    raise ReadySuccessorError(f"unknown ready-successor job: {job_id}")


def _parent(queue: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(queue.get("parent_selection"), "parent_selection")
    return _mapping(selection.get(PARENT_NAME), f"parent_selection.{PARENT_NAME}")


def _parent_record_for_continuation(queue: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_parent(queue))


def _bind_parent_context(queue: Mapping[str, Any]) -> None:
    record = _parent_record_for_continuation(queue)
    for job in queue["jobs"]:
        job["_continuation_parent_record"] = record


def _runtime_job(job: Mapping[str, Any], slot_name: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(job))
    result["resource"] = {
        "policy": "dispatch_gpu_round_robin",
        "required_slot": slot_name,
    }
    return result


def _evidence_content_sha(value: Mapping[str, Any]) -> str:
    return _canonical_sha(
        {key: child for key, child in value.items() if key != "evidence_content_sha256"}
    )


def _validate_parent_integrity_evidence(
    queue: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "read_only",
        "no_write",
        "no_signal",
        "inspector_runner_source_sha256",
        "inspector_program_sha256",
        "job_id",
        "milestone",
        "expected_claim_content_sha256",
        "queue_claim",
        "run_binding",
        "milestone_receipt",
        "checkpoint",
        "hard_contract",
        "evidence_content_sha256",
    }
    if set(evidence) != expected_keys:
        raise ReadySuccessorError("parent_binding_evidence envelope changed")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != "ready_successor_parent_inspection_v1"
        or evidence.get("status") != "passed"
        or evidence.get("read_only") is not True
        or evidence.get("no_write") is not True
        or evidence.get("no_signal") is not True
        or evidence.get("job_id") != "taskrev_p2_equal_reward"
        or evidence.get("milestone") != PARENT_ITERATION
    ):
        raise ReadySuccessorError("parent_binding_evidence identity/status changed")
    if evidence.get("inspector_runner_source_sha256") != (
        EXECUTED_PARENT_INSPECTOR_RUNNER_SHA256
    ):
        raise ReadySuccessorError("parent evidence does not bind the executed inspector runner")
    if evidence.get("inspector_program_sha256") != EXECUTED_PARENT_INSPECTOR_PROGRAM_SHA256:
        raise ReadySuccessorError("parent evidence does not bind the executed inspector program")
    parent = _parent(queue)
    if evidence.get("expected_claim_content_sha256") != parent.get(
        "original_claim_content_sha256"
    ):
        raise ReadySuccessorError("parent evidence known claim digest changed")

    claim = _mapping(evidence.get("queue_claim"), "parent evidence queue_claim")
    binding = _mapping(evidence.get("run_binding"), "parent evidence run_binding")
    milestone = _mapping(
        evidence.get("milestone_receipt"), "parent evidence milestone_receipt"
    )
    checkpoint = _mapping(evidence.get("checkpoint"), "parent evidence checkpoint")
    hard = _mapping(evidence.get("hard_contract"), "parent evidence hard_contract")
    for value, label in (
        (claim, "queue_claim"),
        (binding, "run_binding"),
        (milestone, "milestone_receipt"),
    ):
        if set(value) != {"path", "file_sha256", "content_sha256"}:
            raise ReadySuccessorError(f"parent evidence {label} shape changed")
        _workspace(value.get("path"), f"parent evidence {label}.path")
        _sha(value.get("file_sha256"), f"parent evidence {label}.file_sha256")
        _sha(value.get("content_sha256"), f"parent evidence {label}.content_sha256")
    if set(checkpoint) != {
        "path",
        "file_sha256",
        "embedded_iteration",
        "floating_tensor_count",
        "floating_element_count",
        "nonfinite_element_count",
        "optimizer_state_present",
    }:
        raise ReadySuccessorError("parent evidence checkpoint shape changed")
    if set(hard) != {"path", "file_sha256", "schema_version"}:
        raise ReadySuccessorError("parent evidence hard-contract shape changed")
    _workspace(checkpoint.get("path"), "parent evidence checkpoint.path")
    _workspace(hard.get("path"), "parent evidence hard_contract.path")
    _sha(checkpoint.get("file_sha256"), "parent evidence checkpoint.file_sha256")
    _sha(hard.get("file_sha256"), "parent evidence hard_contract.file_sha256")
    if (
        claim.get("path") != parent.get("original_queue_claim_path")
        or claim.get("file_sha256") != parent.get("original_queue_claim_sha256")
        or claim.get("content_sha256") != parent.get("original_claim_content_sha256")
        or binding.get("path") != parent.get("original_run_binding_path")
        or binding.get("file_sha256") != parent.get("original_run_binding_sha256")
        or binding.get("content_sha256") != PARENT_BINDING_CONTENT_SHA256
        or milestone.get("path") != parent.get("milestone_receipt_path")
        or checkpoint.get("path") != parent.get("selected_checkpoint_path")
        or checkpoint.get("file_sha256") != parent.get("selected_checkpoint_sha256")
        or checkpoint.get("embedded_iteration") != PARENT_ITERATION
        or checkpoint.get("floating_tensor_count") != 74
        or checkpoint.get("floating_element_count") != 1_762_715
        or checkpoint.get("nonfinite_element_count") != 0
        or checkpoint.get("optimizer_state_present") is not True
        or hard.get("path") != parent.get("selected_hard_contract_path")
        or hard.get("file_sha256") != parent.get("selected_hard_contract_sha256")
        or hard.get("schema_version") != 3
    ):
        raise ReadySuccessorError("parent evidence differs from the exact selected parent")
    _sha(evidence.get("evidence_content_sha256"), "parent evidence content SHA")
    if evidence["evidence_content_sha256"] != _evidence_content_sha(evidence):
        raise ReadySuccessorError("parent_integrity_evidence canonical digest mismatch")


def _validate_parent_semantic_binding_evidence(
    queue: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "inspector_runner_source_sha256",
        "inspector_program_sha256",
        "inspection_content_sha256",
        "inspection",
        "evidence_content_sha256",
    }
    if set(evidence) != expected_keys:
        raise ReadySuccessorError("parent_binding_evidence v2 envelope changed")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != "ready_successor_parent_semantic_inspection_v2"
        or evidence.get("status") != "passed"
        or evidence.get("inspector_runner_source_sha256") != _runner_sha()
        or evidence.get("inspector_program_sha256")
        != hashlib.sha256(PARENT_INSPECT_PROGRAM.encode("utf-8")).hexdigest()
    ):
        raise ReadySuccessorError("parent_binding_evidence v2 source/status changed")
    inspection = _mapping(evidence.get("inspection"), "parent semantic inspection")
    _sha(evidence.get("inspection_content_sha256"), "parent inspection content SHA")
    if evidence["inspection_content_sha256"] != _canonical_sha(inspection):
        raise ReadySuccessorError("parent semantic inspection canonical digest mismatch")
    parent = _parent(queue)
    integrity = _mapping(
        _mapping(queue.get("blocking_contract"), "blocking_contract").get(
            "parent_integrity_evidence"
        ),
        "parent_integrity_evidence",
    )
    claim = _mapping(inspection.get("queue_claim"), "semantic inspection queue_claim")
    binding = _mapping(inspection.get("run_binding"), "semantic inspection run_binding")
    milestone = _mapping(
        inspection.get("milestone_receipt"), "semantic inspection milestone_receipt"
    )
    checkpoint = _mapping(inspection.get("checkpoint"), "semantic inspection checkpoint")
    hard = _mapping(inspection.get("hard_contract"), "semantic inspection hard_contract")
    selection_patch = _mapping(
        inspection.get("parent_selection_patch"), "parent semantic selection patch"
    )
    if set(claim) != {"path", "file_sha256", "content_sha256"}:
        raise ReadySuccessorError("parent semantic queue claim shape changed")
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ReadySuccessorError("parent semantic run binding shape changed")
    if set(milestone) != {
        "path",
        "file_sha256",
        "content_sha256",
        "binding_content_sha256",
    }:
        raise ReadySuccessorError("parent semantic milestone receipt shape changed")
    if set(checkpoint) != {
        "path",
        "file_sha256",
        "embedded_iteration",
        "floating_tensor_count",
        "floating_element_count",
        "nonfinite_element_count",
        "optimizer_state_present",
        "training_launch_claim_sha256",
        "lineage_exact",
    }:
        raise ReadySuccessorError("parent semantic checkpoint shape changed")
    if set(hard) != {"path", "file_sha256", "schema_version", "lineage_exact"}:
        raise ReadySuccessorError("parent semantic hard-contract shape changed")
    if set(selection_patch) != {
        "original_queue_claim_sha256",
        "original_run_binding_sha256",
        "selected_checkpoint_sha256",
        "selected_hard_contract_path",
        "selected_hard_contract_sha256",
        "selection_is_final",
    }:
        raise ReadySuccessorError("parent semantic selection patch shape changed")
    expected_milestone = _mapping(
        integrity.get("milestone_receipt"), "parent integrity milestone receipt"
    )
    if (
        inspection.get("schema_version") != 1
        or inspection.get("status") != "passed"
        or inspection.get("read_only") is not True
        or inspection.get("no_write") is not True
        or inspection.get("no_signal") is not True
        or inspection.get("job_id") != "taskrev_p2_equal_reward"
        or inspection.get("milestone") != PARENT_ITERATION
        or inspection.get("expected_claim_content_sha256")
        != parent.get("original_claim_content_sha256")
        or claim.get("path") != parent.get("original_queue_claim_path")
        or claim.get("file_sha256") != parent.get("original_queue_claim_sha256")
        or claim.get("content_sha256") != parent.get("original_claim_content_sha256")
        or binding.get("path") != parent.get("original_run_binding_path")
        or binding.get("file_sha256") != parent.get("original_run_binding_sha256")
        or binding.get("content_sha256") != PARENT_BINDING_CONTENT_SHA256
        or milestone.get("path") != parent.get("milestone_receipt_path")
        or milestone.get("file_sha256") != expected_milestone.get("file_sha256")
        or milestone.get("content_sha256") != expected_milestone.get("content_sha256")
        or milestone.get("binding_content_sha256") != binding.get("content_sha256")
        or checkpoint.get("path") != parent.get("selected_checkpoint_path")
        or checkpoint.get("file_sha256") != parent.get("selected_checkpoint_sha256")
        or checkpoint.get("embedded_iteration") != PARENT_ITERATION
        or checkpoint.get("floating_tensor_count") != 74
        or checkpoint.get("floating_element_count") != 1_762_715
        or checkpoint.get("training_launch_claim_sha256")
        != parent.get("original_claim_content_sha256")
        or checkpoint.get("lineage_exact") != 0
        or checkpoint.get("nonfinite_element_count") != 0
        or checkpoint.get("optimizer_state_present") is not True
        or hard.get("path") != parent.get("selected_hard_contract_path")
        or hard.get("file_sha256") != parent.get("selected_hard_contract_sha256")
        or hard.get("schema_version") != 3
        or hard.get("lineage_exact") != 0
        or selection_patch.get("original_queue_claim_sha256")
        != parent.get("original_queue_claim_sha256")
        or selection_patch.get("original_run_binding_sha256")
        != parent.get("original_run_binding_sha256")
        or selection_patch.get("selected_checkpoint_sha256")
        != parent.get("selected_checkpoint_sha256")
        or selection_patch.get("selected_hard_contract_path")
        != parent.get("selected_hard_contract_path")
        or selection_patch.get("selected_hard_contract_sha256")
        != parent.get("selected_hard_contract_sha256")
        or selection_patch.get("selection_is_final") is not True
    ):
        raise ReadySuccessorError(
            "parent semantic inspection does not cross-bind claim/binding/receipt/checkpoint/hard"
        )
    _sha(evidence.get("evidence_content_sha256"), "parent semantic evidence SHA")
    if evidence["evidence_content_sha256"] != _evidence_content_sha(evidence):
        raise ReadySuccessorError("parent_binding_evidence v2 canonical digest mismatch")


def _validate_probe_binding_evidence(
    queue: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "unlock_authorized",
        "producer_runner_source_sha256",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_file_base64",
        "receipt_content_sha256",
        "receipt_content",
        "evidence_content_sha256",
    }
    if set(evidence) != expected_keys:
        raise ReadySuccessorError("ready_full_scene_probe_evidence envelope changed")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != "ready_successor_specialized_probe_receipt_v1"
        or evidence.get("status") != "passed"
        or evidence.get("unlock_authorized") is not True
    ):
        raise ReadySuccessorError("ready probe evidence identity/status changed")
    producer_sha = _sha(
        evidence.get("producer_runner_source_sha256"),
        "ready probe producer runner SHA",
    )
    if producer_sha != _runner_sha():
        raise ReadySuccessorError("ready probe producer runner differs from executing runner")
    receipt_file_sha = _sha(
        evidence.get("receipt_file_sha256"), "ready probe receipt file SHA"
    )
    receipt_file_b64 = _text(
        evidence.get("receipt_file_base64"), "ready probe receipt file bytes"
    )
    try:
        receipt_file_raw = base64.b64decode(receipt_file_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ReadySuccessorError("ready probe receipt file bytes are not base64") from exc
    if not receipt_file_raw or hashlib.sha256(receipt_file_raw).hexdigest() != receipt_file_sha:
        raise ReadySuccessorError("ready probe receipt file SHA does not bind file bytes")
    try:
        receipt_file_value = json.loads(receipt_file_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadySuccessorError("ready probe receipt file bytes are not JSON") from exc
    receipt_file_content = _validate_envelope(
        _mapping(receipt_file_value, "ready probe receipt file"),
        "ready probe receipt file",
    )
    content_sha = _sha(
        evidence.get("receipt_content_sha256"), "ready probe receipt content SHA"
    )
    content = _mapping(evidence.get("receipt_content"), "ready probe receipt content")
    if _canonical_sha(content) != content_sha:
        raise ReadySuccessorError("ready probe receipt canonical digest mismatch")
    if receipt_file_content != content:
        raise ReadySuccessorError("ready probe receipt file content differs from evidence")
    attempt_id = _text(content.get("attempt_id"), "ready probe attempt_id")
    job = _job(queue, PROBE_JOB)
    slot = lean._slot_by_identity(queue, "pod2", PROBE_GPU)
    claim, _argv, run_dir = lean._full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    expected_path = f"{run_dir}/ready_successor_probe_result.json"
    consumer_source = _mapping(content.get("consumer_source"), "ready probe consumer_source")
    expected_content_keys = {
        "schema_version",
        "status",
        "unlock_authorized",
        "representative_job_id",
        "pod",
        "gpu",
        "attempt_id",
        "claim_content_sha256",
        "generic_result_file_sha256",
        "generic_result_content_sha256",
        "log_prefix",
        "exact_update_ids",
        "task_revision_probe_passed",
        *REQUIRED_READY_INVARIANTS,
        "aggregate_counters",
        "formal_evidence_eligible",
        "consumer_source",
        "automatic_retry",
    }
    if (
        evidence.get("receipt_path") != expected_path
        or content.get("schema_version") != 1
        or content.get("status") != "passed"
        or content.get("unlock_authorized") is not True
        or content.get("representative_job_id") != PROBE_JOB
        or content.get("pod") != "pod2"
        or content.get("gpu") != PROBE_GPU
        or content.get("claim_content_sha256") != claim.get("content_sha256")
        or content.get("task_revision_probe_passed") is not True
        or content.get("formal_evidence_eligible") is not False
        or content.get("automatic_retry") is not False
        or set(content) != expected_content_keys
        or consumer_source
        != {"mode": "embedded_sha_bound", "sha256": producer_sha}
    ):
        raise ReadySuccessorError("ready probe evidence does not bind receipt/job/claim/source")
    for key in ("generic_result_file_sha256", "generic_result_content_sha256"):
        _sha(content.get(key), f"ready probe receipt {key}")
    checks = _validate_ready_counter_shape(
        _mapping(content.get("aggregate_counters"), "ready probe aggregate counters"),
        "ready probe aggregate counters",
    )
    if content.get("task_revision_probe_passed") is not True or any(
        content.get(key) is not value for key, value in checks.items()
    ):
        raise ReadySuccessorError("ready probe receipt invariant binding changed")
    _sha(evidence.get("evidence_content_sha256"), "ready probe evidence content SHA")
    if evidence["evidence_content_sha256"] != _evidence_content_sha(evidence):
        raise ReadySuccessorError("ready probe evidence canonical digest mismatch")


def _activation_blockers(queue: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if queue.get("launch_authorized") is not True:
        blockers.append("launch_authorized is false")
    if queue.get("preregistration_status") != ACTIVATED_STATUS:
        blockers.append(f"preregistration_status is not {ACTIVATED_STATUS}")
    blocking = _mapping(queue.get("blocking_contract"), "blocking_contract")
    source_gate = _mapping(blocking.get("runner_source_gate"), "runner_source_gate")
    if source_gate.get("sha256") != _runner_sha():
        blockers.append("runner_source_gate.sha256 differs from runner bytes")
    if source_gate.get("reviewed_tests_passed") is not True:
        blockers.append("runner source tests are not marked passed")
    parent = _parent(queue)
    for key in (
        "original_queue_claim_sha256",
        "original_run_binding_sha256",
        "selected_checkpoint_sha256",
        "selected_hard_contract_sha256",
    ):
        if _pending(parent.get(key)):
            blockers.append(f"parent {key} is pending")
    if _pending(parent.get("selected_hard_contract_path")):
        blockers.append("parent selected_hard_contract_path is pending")
    if parent.get("selection_is_final") is not True:
        blockers.append("parent selection_is_final is not true")
    parent_evidence = blocking.get("parent_binding_evidence")
    if not isinstance(parent_evidence, dict):
        blockers.append("parent_binding_evidence is not a pass mapping")
    else:
        try:
            _validate_parent_semantic_binding_evidence(queue, parent_evidence)
        except ReadySuccessorError as exc:
            blockers.append(f"parent_binding_evidence invalid: {exc}")
    probe = blocking.get("ready_full_scene_probe_evidence")
    if not isinstance(probe, dict):
        blockers.append("ready_full_scene_probe_evidence is not a pass mapping")
    else:
        try:
            _validate_probe_binding_evidence(queue, probe)
        except ReadySuccessorError as exc:
            blockers.append(f"ready_full_scene_probe_evidence invalid: {exc}")
    if any(job.get("status") != "ready" or job.get("blocker") is not None for job in queue["jobs"]):
        blockers.append("one or more science jobs remain blocked")
    return list(dict.fromkeys(blockers))


def _validate_ready_counter_shape(counters: Mapping[str, Any], label: str) -> dict[str, bool]:
    missing = READY_COUNTERS - set(counters)
    if missing:
        raise ReadySuccessorError(f"{label} lacks ready counters: {sorted(missing)}")
    values: dict[str, int | float] = {}
    for key in READY_COUNTERS:
        value = counters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReadySuccessorError(f"{label}.{key} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ReadySuccessorError(f"{label}.{key} must be finite and non-negative")
        values[key] = value
    phase = int(values["ready_phase_sample_count"])
    unavailable = int(values["ready_foot_sensor_unavailable_sample_count"])
    checks = {
        "ready_phase_sample_count_positive": phase > 0,
        "ready_planner_task_entry_sample_count_equals_ready_phase_sample_count": int(
            values["ready_planner_task_entry_sample_count"]
        )
        == phase,
        "ready_tilt_eligible_sample_count_equals_ready_phase_sample_count": int(
            values["ready_tilt_eligible_sample_count"]
        )
        == phase,
        "ready_base_speed_eligible_sample_count_equals_ready_phase_sample_count": int(
            values["ready_base_speed_eligible_sample_count"]
        )
        == phase,
        "ready_station_offset_eligible_sample_count_equals_ready_phase_sample_count": int(
            values["ready_station_offset_eligible_sample_count"]
        )
        == phase,
        "ready_foot_contact_eligible_plus_unavailable_equals_ready_phase_sample_count": int(
            values["ready_foot_contact_eligible_sample_count"]
        )
        + unavailable
        == phase,
        "ready_foot_slip_eligible_plus_unavailable_equals_ready_phase_sample_count": int(
            values["ready_foot_slip_eligible_sample_count"]
        )
        + unavailable
        == phase,
        "ready_planner_legacy_hold_violation_count_equals_zero": int(
            values["ready_planner_legacy_hold_violation_count"]
        )
        == 0,
        "ready_nonfinite_value_count_equals_zero": int(
            values["ready_nonfinite_value_count"]
        )
        == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ReadySuccessorError(f"{label} violates ready invariants: {failed}")
    return checks


def _validate_queue(queue: dict[str, Any]) -> dict[str, Any]:
    if queue.get("schema_version") != 1 or queue.get("simulation_only") is not True:
        raise ReadySuccessorError("queue must be schema-1 simulation-only")
    if queue.get("formal_evidence_eligible") is not False:
        raise ReadySuccessorError("warm-start successor must remain formal-ineligible")
    if queue.get("preregistration_status") != PENDING_STATUS:
        raise ReadySuccessorError(
            "this source is probe/inspection-only and rejects activated queues"
        )
    pending = True
    if queue.get("launch_authorized") is not False:
        raise ReadySuccessorError("probe/inspection-only queue must remain NO-LAUNCH")
    namespace = _mapping(queue.get("namespace_contract"), "namespace_contract")
    if (
        namespace.get("root") != NAMESPACE_ROOT
        or namespace.get("no_clobber") is not True
        or namespace.get("old_task_revision_queue_or_receipt_reuse_forbidden") is not True
    ):
        raise ReadySuccessorError("fresh no-clobber namespace contract changed")
    if namespace.get("status") != queue["preregistration_status"] and not (
        not pending and namespace.get("status") == "activated_no_clobber"
    ):
        raise ReadySuccessorError("namespace status differs from queue activation")
    if queue.get("dispatch_pods") != ["pod2"] or list(_mapping(queue.get("pods"), "pods")) != [
        "pod2"
    ]:
        raise ReadySuccessorError("ready successor may dispatch only to Pod2")
    pod = queue["pods"]["pod2"]
    if (
        pod.get("host") != "162.43.172.181"
        or pod.get("port") != 13146
        or pod.get("gpus") != [0, 1, 2]
        or pod.get("max_trainers_per_gpu") != 4
    ):
        raise ReadySuccessorError("Pod2 resource contract changed")
    blocking = _mapping(queue.get("blocking_contract"), "blocking_contract")
    if blocking.get("source_checkout") != SOURCE_CHECKOUT or blocking.get("source_commit") != SOURCE_COMMIT:
        raise ReadySuccessorError("training source must remain exact d7c38fcf")
    if (
        blocking.get("activation_supported_by_this_runner") is not False
        or blocking.get("future_fill_requires_remote_stable_receipt_revalidation")
        is not True
    ):
        raise ReadySuccessorError("probe-only future-fill safety contract changed")
    source_gate = _mapping(blocking.get("runner_source_gate"), "runner_source_gate")
    if source_gate.get("path") != "scripts/run_phase1_task_revision_ready_successor_queue.py":
        raise ReadySuccessorError("runner source-gate path changed")
    runner_gate_sha = _sha(source_gate.get("sha256"), "runner_source_gate.sha256")
    if runner_gate_sha != _runner_sha():
        raise ReadySuccessorError(
            "runner_source_gate.sha256 differs from the executing runner bytes"
        )

    parent = _parent(queue)
    if parent.get("original_job_id") != "taskrev_p2_equal_reward":
        raise ReadySuccessorError("common parent job changed")
    if parent.get("original_claim_content_sha256") != (
        "e10d2c248d90daa3172ea80147a394dad64ce326eb4052889c25bfb9d3df420b"
    ):
        raise ReadySuccessorError("known parent claim content digest changed")
    if parent.get("selected_embedded_iteration") != PARENT_ITERATION:
        raise ReadySuccessorError("common parent must remain model_5700")
    expected_checkpoint = (
        "/workspace/codexschema/nohope_task_revision_b1f5a38/hope_training/whole_body_tracking/"
        "logs/rsl_rl/agibot_a3_hope_virtualball/"
        "2026-07-16_20-04-47_phase1_taskrev_p2_equal_reward_seed3_20260716/model_5700.pt"
    )
    if parent.get("selected_checkpoint_path") != expected_checkpoint:
        raise ReadySuccessorError("known parent checkpoint path changed")
    if parent.get("ranking_snapshot_checkpoint_filename") != "model_5700.pt":
        raise ReadySuccessorError("parent checkpoint filename binding changed")
    for key in (
        "original_queue_claim_path",
        "original_run_binding_path",
        "selected_rsl_log_dir",
        "selected_checkpoint_path",
    ):
        _workspace(parent.get(key), f"parent.{key}")
    _workspace(parent.get("selected_hard_contract_path"), "parent.hard", allow_pending=True)
    for key in (
        "original_queue_claim_sha256",
        "original_run_binding_sha256",
        "selected_checkpoint_sha256",
        "selected_hard_contract_sha256",
    ):
        _sha(parent.get(key), f"parent.{key}", allow_pending=True)
    parent_integrity_evidence = blocking.get("parent_integrity_evidence")
    if not isinstance(parent_integrity_evidence, dict):
        raise ReadySuccessorError("parent_integrity_evidence must remain a bound mapping")
    _validate_parent_integrity_evidence(queue, parent_integrity_evidence)
    parent_evidence = blocking.get("parent_binding_evidence")
    if isinstance(parent_evidence, dict):
        _validate_parent_semantic_binding_evidence(queue, parent_evidence)
    probe_evidence = blocking.get("ready_full_scene_probe_evidence")
    if isinstance(probe_evidence, dict):
        _validate_probe_binding_evidence(queue, probe_evidence)

    jobs = _list(queue.get("jobs"), "jobs")
    if tuple(job.get("id") for job in jobs) != JOB_IDS:
        raise ReadySuccessorError("queue must contain exactly the four registered cells in order")
    questions: set[str] = set()
    matrix: set[tuple[str, float]] = set()
    run_dirs: set[str] = set()
    for job in jobs:
        job_id = job["id"]
        if job.get("formal_evidence_eligible") is not False or job.get("seed") != 3:
            raise ReadySuccessorError(f"{job_id} formal/seed contract changed")
        if job.get("budget") != {
            "num_envs": 4096,
            "max_iterations": 1001,
            "save_interval": 100,
            "iteration_semantics": "additional_updates_after_full_state_resume",
        }:
            raise ReadySuccessorError(f"{job_id} must run 1001 additional updates at 4096 envs")
        if job.get("milestones") != OFFSETS:
            raise ReadySuccessorError(f"{job_id} milestones must remain +200/+500/+1000")
        if job.get("milestone_semantics") != "offsets_from_model_5700_parent":
            raise ReadySuccessorError(f"{job_id} milestone semantics changed")
        if job.get("warm_start", {}).get("parent") != PARENT_NAME or job["warm_start"].get(
            "checkpoint_path"
        ) != expected_checkpoint:
            raise ReadySuccessorError(f"{job_id} does not use the common model_5700 parent")
        expected_warm = {
            "transfer_mode": "strict_full_state_preserve_optimizer",
            "checkpoint_tolerant": False,
            "allow_missing_contract": False,
            "allow_contract_mismatch": True,
            "descendant_exact_eligible": False,
        }
        for key, value in expected_warm.items():
            if job["warm_start"].get(key) != value:
                raise ReadySuccessorError(f"{job_id}.warm_start.{key} changed")
        if job.get("source", {}).get("checkout") != SOURCE_CHECKOUT or job["source"].get(
            "commit"
        ) != SOURCE_COMMIT:
            raise ReadySuccessorError(f"{job_id} source differs from exact main d7c38fcf")
        if job.get("runtime_binding") is not True:
            raise ReadySuccessorError(f"{job_id} runtime_binding must be true")
        run_dir = _workspace(job.get("run_dir"), f"{job_id}.run_dir")
        if not run_dir.startswith(NAMESPACE_ROOT + "/runs/") or run_dir in run_dirs:
            raise ReadySuccessorError(f"{job_id} run_dir is not unique in the fresh namespace")
        run_dirs.add(run_dir)
        question = _text(job.get("scientific_question"), f"{job_id}.scientific_question")
        if question in questions:
            raise ReadySuccessorError("scientific questions must be distinct")
        questions.add(question)
        ready_role = job.get("ready_role")
        qdot = job.get("qdot_limit_hinge_weight")
        if ready_role not in {"baseline", "strong"} or qdot not in {-5.0, 0.0}:
            raise ReadySuccessorError(f"{job_id} is outside the registered 2x2 matrix")
        matrix.add((ready_role, float(qdot)))
        overrides = taskrev._compiled_overrides(job)
        taskrev._validate_revision(job, overrides)
        expected_ready = (-0.3, -1.0) if ready_role == "baseline" else (-0.6, -2.0)
        for key, expected in zip(
            ("task.rewards.foot_orientation_weight", "task.rewards.prestrike_upright_weight"),
            expected_ready,
            strict=True,
        ):
            raw = overrides.get(key)
            if raw != f"{key}={expected}":
                raise ReadySuccessorError(f"{job_id} {key} must be {expected}")
        key = "task.rewards.joint_velocity_limit_hinge_weight"
        if overrides.get(key) != f"{key}={float(qdot)}":
            raise ReadySuccessorError(f"{job_id} qdot-limit hinge weight differs")
        if any("lateral_perturb" in key or "external_force" in key for key in overrides):
            raise ReadySuccessorError(f"{job_id} must not smuggle the unlaunched random-push axis")
        resource = _mapping(job.get("resource"), f"{job_id}.resource")
        if job_id in FIXED_ROUND_ONE:
            if resource != {
                "policy": "dispatch_gpu_round_robin",
                "required_slot": FIXED_ROUND_ONE[job_id],
            } or job.get("launch_round") != 1:
                raise ReadySuccessorError(f"{job_id} must occupy its one-per-GPU first round")
        elif resource != {
            "policy": "dispatch_gpu_round_robin",
            "preferred_slot": "pod2/gpu0",
        } or job.get("launch_round") != 2:
            raise ReadySuccessorError("fourth cell must dynamically choose the least occupied Pod2 GPU")
        if pending:
            if job.get("status") != "blocked" or not job.get("blocker"):
                raise ReadySuccessorError(f"pending {job_id} must remain blocked")
        elif job.get("status") != "ready" or job.get("blocker") is not None:
            raise ReadySuccessorError(f"activated {job_id} must be ready")
    if matrix != {("baseline", -5.0), ("baseline", 0.0), ("strong", -5.0), ("strong", 0.0)}:
        raise ReadySuccessorError("ready x qdot matrix is incomplete")

    probe = _mapping(queue.get("full_scene_probe_contract"), "full_scene_probe_contract")
    if (
        probe.get("representative_job_id") != PROBE_JOB
        or probe.get("pod") != "pod2"
        or probe.get("gpu") != PROBE_GPU
        or probe.get("num_envs") != 4096
        or probe.get("additional_updates") != 2
        or probe.get("required_ready_invariants") != list(REQUIRED_READY_INVARIANTS)
    ):
        raise ReadySuccessorError("full-scene probe contract changed")
    ledger = _mapping(queue.get("exact_behavior_ledger_contract"), "exact ledger")
    if (
        ledger.get("event_prefix") != EXACT_EVENT_PREFIX
        or ledger.get("consume_once_per_update") is not True
        or ledger.get("window_updates") != 100
        or ledger.get("missing_or_duplicate_update_action") != "continue_training_no_decision"
    ):
        raise ReadySuccessorError("exact behavior ledger cadence changed")
    if not READY_COUNTERS.issubset(_required_counters(queue)):
        raise ReadySuccessorError("exact ledger omits task-entry ready counters")
    pruning = _mapping(queue.get("pruning_contract"), "pruning_contract")
    if (
        pruning.get("checkpoint_offsets_from_parent") != OFFSETS
        or pruning.get("behavior_decision_requires_two_disjoint_complete_windows") is not True
        or pruning.get("window_updates") != 100
        or pruning.get("sparse_zero_without_positive_eligible_denominator_may_stop") is not False
        or pruning.get("automatic_stop") is not False
    ):
        raise ReadySuccessorError("integer two-window pruning contract changed")
    _bind_parent_context(queue)
    return {
        "schema_valid": True,
        "pending": pending,
        "job_count": 4,
        "matrix": "ready baseline/strong x qdot-limit hinge -5/0",
        "parent_iteration": PARENT_ITERATION,
        "absolute_milestones": ABSOLUTE_MILESTONES,
        "formal_evidence_eligible": False,
        "activation_blockers": _activation_blockers(queue),
    }


def load_queue(path: Path) -> dict[str, Any]:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def unique(loader, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = (key_node.tag, getattr(key_node, "value", None))
            if key in seen:
                raise ReadySuccessorError(
                    f"duplicate YAML key is forbidden: {getattr(key_node, 'value', None)!r}"
                )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique)
    raw = path.resolve().read_bytes()
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReadySuccessorError(f"queue YAML is invalid: {exc}") from exc
    queue = _mapping(value, "queue")
    _validate_queue(queue)
    return queue


def cmd_validate(queue: dict[str, Any]) -> dict[str, Any]:
    result = _validate_queue(queue)
    return {
        "mode": "validate_ready_successor",
        **result,
        "activation_ready": not result["activation_blockers"],
        "automatic_retry": False,
    }


def _slot_plan(job: Mapping[str, Any]) -> dict[str, Any]:
    resource = job["resource"]
    if "required_slot" in resource:
        return {
            "slot_policy": "required",
            "planned_slot": resource["required_slot"],
        }
    return {
        "slot_policy": "least_occupied_pod2_at_launch",
        "preferred_slot": resource["preferred_slot"],
        "k100_on_gpu0_causes_fallback": True,
        "allowed_slots": ["pod2/gpu0", "pod2/gpu1", "pod2/gpu2"],
    }


def cmd_plan(queue: dict[str, Any]) -> dict[str, Any]:
    validated = _validate_queue(queue)
    rows = []
    for job in queue["jobs"]:
        rows.append(
            {
                "launch_round": job["launch_round"],
                "job_id": job["id"],
                "human_name": job["human_name"],
                "scientific_question": job["scientific_question"],
                "ready_role": job["ready_role"],
                "qdot_limit_hinge_weight": job["qdot_limit_hinge_weight"],
                "parent": f"{PARENT_NAME}@model_{PARENT_ITERATION}",
                "additional_updates": 1001,
                "absolute_milestones": ABSOLUTE_MILESTONES,
                "status": job["status"],
                **_slot_plan(job),
            }
        )
    return {
        "mode": "plan_ready_successor",
        "dry_run": True,
        "activation_ready": not validated["activation_blockers"],
        "activation_blockers": validated["activation_blockers"],
        "full_scene_probe": {
            "job_id": PROBE_JOB,
            "pod": "pod2",
            "gpu": PROBE_GPU,
            "num_envs": 4096,
            "additional_updates": 2,
            "specialized_receipt": "ready_successor_probe_result.json",
        },
        "jobs": rows,
        "formal_evidence_eligible": False,
        "automatic_retry": False,
    }


def _parent_inspect_spec(queue: Mapping[str, Any]) -> dict[str, Any]:
    parent = _parent(queue)
    return {
        "schema_version": 1,
        "job_id": "taskrev_p2_equal_reward",
        "milestone": PARENT_ITERATION,
        "run_dir": parent["original_run_dir"],
        "queue_claim_path": parent["original_queue_claim_path"],
        "run_binding_path": parent["original_run_binding_path"],
        "milestone_receipt_path": parent["milestone_receipt_path"],
        "rsl_log_dir": parent["selected_rsl_log_dir"],
        "checkpoint_path": parent["selected_checkpoint_path"],
        "expected_claim_content_sha256": parent["original_claim_content_sha256"],
    }


def cmd_inspect_parent(
    queue: dict[str, Any], *, execute: bool, confirm: str | None
) -> dict[str, Any]:
    """Read and hash the fixed model-5700 parent without writing or signalling."""

    _validate_queue(queue)

    if execute and confirm != PARENT_INSPECT_CONFIRM:
        raise ReadySuccessorError(
            f"--execute requires --confirm {PARENT_INSPECT_CONFIRM}"
        )
    spec = _parent_inspect_spec(queue)
    encoded = base64.b64encode(
        json.dumps(
            spec,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).decode("ascii")
    command = shlex.join(
        [lean.ISAAC_PYTHON, "-B", "-c", PARENT_INSPECT_PROGRAM, encoded]
    )
    result: dict[str, Any] = {
        "mode": "inspect-ready-successor-parent",
        "dry_run": not execute,
        "read_only": True,
        "no_write": True,
        "no_signal": True,
        "pod": "pod2",
        "job_id": spec["job_id"],
        "milestone": spec["milestone"],
        "expected_claim_content_sha256": spec["expected_claim_content_sha256"],
        "paths": {
            "queue_claim": spec["queue_claim_path"],
            "run_binding": spec["run_binding_path"],
            "milestone_receipt": spec["milestone_receipt_path"],
            "checkpoint": spec["checkpoint_path"],
            "rsl_log_dir": spec["rsl_log_dir"],
        },
        "inspector_program_sha256": hashlib.sha256(
            PARENT_INSPECT_PROGRAM.encode("utf-8")
        ).hexdigest(),
        "automatic_retry": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *lean._ssh_prefix(queue, "pod2"),
            command,
        ]
        return result
    raw = lean._run_ssh(
        queue,
        "pod2",
        command,
        timeout=180,
        phase="ready-successor-parent-inspect:model5700",
    )
    try:
        inspection = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReadySuccessorError("parent inspector returned malformed JSON") from exc
    inspection = _mapping(inspection, "parent inspection")
    if (
        inspection.get("status") != "passed"
        or inspection.get("read_only") is not True
        or inspection.get("no_write") is not True
        or inspection.get("no_signal") is not True
        or inspection.get("expected_claim_content_sha256")
        != spec["expected_claim_content_sha256"]
    ):
        raise ReadySuccessorError("parent inspector did not return a strict read-only pass")
    evidence = {
        "schema_version": 1,
        "kind": "ready_successor_parent_semantic_inspection_v2",
        "status": "passed",
        "inspector_runner_source_sha256": _runner_sha(),
        "inspector_program_sha256": hashlib.sha256(
            PARENT_INSPECT_PROGRAM.encode("utf-8")
        ).hexdigest(),
        "inspection_content_sha256": _canonical_sha(inspection),
        "inspection": inspection,
    }
    evidence["evidence_content_sha256"] = _evidence_content_sha(evidence)
    _validate_parent_semantic_binding_evidence(queue, evidence)
    return {
        **result,
        "inspection": inspection,
        "semantic_binding_evidence": evidence,
    }


def cmd_full_scene_probe(
    queue: dict[str, Any],
    *,
    attempt_id: str,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    _validate_queue(queue)
    if execute and confirm != PROBE_CONFIRM:
        raise ReadySuccessorError(f"--execute requires --confirm {PROBE_CONFIRM}")
    delegated = lean.FULL_SCENE_PROBE_CONFIRM if execute else None
    result = lean.cmd_full_scene_probe(
        queue,
        job_id=PROBE_JOB,
        pod="pod2",
        gpu=PROBE_GPU,
        attempt_id=attempt_id,
        execute=execute,
        confirm=delegated,
    )
    result.update(
        {
            "ready_specialized_result_required": True,
            "ready_specialized_result_path": (
                f"{result['run_dir']}/ready_successor_probe_result.json"
            ),
            "generic_result_alone_may_unlock_launch": False,
            "automatic_retry": False,
        }
    )
    return result


def _consumer_source_evidence() -> dict[str, Any]:
    embedded = globals().get("EMBEDDED_READY_SUCCESSOR_CONSUMER_SHA256")
    if isinstance(embedded, str) and SHA256.fullmatch(embedded):
        return {"mode": "embedded_sha_bound", "sha256": embedded}
    path = Path(__file__).resolve()
    return {"mode": "filesystem", "path": str(path), "sha256": _runner_sha()}


def _validate_envelope(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(value) != {"schema_version", "content", "content_sha256"}:
        raise ReadySuccessorError(f"{label} envelope changed")
    content = _mapping(value.get("content"), f"{label}.content")
    if _canonical_sha(content) != value.get("content_sha256"):
        raise ReadySuccessorError(f"{label} canonical digest mismatch")
    return content


def finalize_ready_probe_local(
    queue: Mapping[str, Any], *, attempt_id: str
) -> dict[str, Any]:
    job = _job(queue, PROBE_JOB)
    slot = lean._slot_by_identity(queue, "pod2", PROBE_GPU)
    claim, _argv, run_dir_text = lean._full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    run_dir = Path(run_dir_text)
    runtime, _runtime_path = taskrev._load_runtime(job["source"]["checkout"])
    generic, generic_raw = runtime._read_regular_json(
        run_dir / "probe_result.json", "generic full-scene probe result"
    )
    generic_content = _validate_envelope(generic, "generic full-scene probe result")
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
        raise ReadySuccessorError("generic full-scene probe did not pass")
    raw_log, log_evidence = taskrev._stable_append_prefix(run_dir / "run.log")
    required = _required_counters(queue)
    records = taskrev.parse_exact_behavior_log(raw_log, required_counters=required)
    revision_evidence = taskrev.validate_task_revision_probe_records(
        records, required_counters=required
    )
    aggregate = revision_evidence["aggregate_counters"]
    ready_checks = _validate_ready_counter_shape(aggregate, "full-scene probe aggregate")
    content = {
        "schema_version": 1,
        "status": "passed",
        "unlock_authorized": True,
        "representative_job_id": PROBE_JOB,
        "pod": "pod2",
        "gpu": PROBE_GPU,
        "attempt_id": attempt_id,
        "claim_content_sha256": claim["content_sha256"],
        "generic_result_file_sha256": hashlib.sha256(generic_raw).hexdigest(),
        "generic_result_content_sha256": generic["content_sha256"],
        "log_prefix": log_evidence,
        "exact_update_ids": revision_evidence["exact_update_ids"],
        "task_revision_probe_passed": True,
        **ready_checks,
        "aggregate_counters": aggregate,
        "formal_evidence_eligible": False,
        "consumer_source": _consumer_source_evidence(),
        "automatic_retry": False,
    }
    receipt = {
        "schema_version": 1,
        "content": content,
        "content_sha256": _canonical_sha(content),
    }
    path = run_dir / "ready_successor_probe_result.json"
    runtime._atomic_publish_json(path, receipt, "ready-successor probe result")
    published, published_raw = runtime._read_regular_json(
        path, "published ready-successor probe result"
    )
    published_content = _validate_envelope(
        published, "published ready-successor probe result"
    )
    if published_content != content:
        raise ReadySuccessorError("published ready probe receipt changed after publish")
    return {
        "receipt_path": str(path),
        "receipt_file_sha256": hashlib.sha256(published_raw).hexdigest(),
        "receipt_file_base64": base64.b64encode(published_raw).decode("ascii"),
        "receipt_content_sha256": published["content_sha256"],
        "receipt": published,
    }


def _embedded_remote_command(
    queue: Mapping[str, Any], *, function: str, kwargs: Mapping[str, Any]
) -> list[str]:
    if function != "finalize_ready_probe_local":
        raise ReadySuccessorError("unsupported embedded ready-successor consumer")
    raw = Path(__file__).resolve().read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    request = {
        "schema_version": 1,
        "function": function,
        "kwargs": dict(kwargs),
        "queue": json.loads(json.dumps(queue, allow_nan=False, sort_keys=True)),
    }
    encoded_script = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    encoded_request = base64.b64encode(
        zlib.compress(
            json.dumps(
                request,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
            9,
        )
    ).decode("ascii")
    remote_filename = f"{SOURCE_CHECKOUT}/scripts/run_phase1_task_revision_ready_successor_queue.py"
    program = (
        "import base64,hashlib,json,sys,zlib;"
        "raw=zlib.decompress(base64.b64decode(sys.argv[1],validate=True));"
        f"assert hashlib.sha256(raw).hexdigest()=={digest!r};"
        "req=json.loads(zlib.decompress(base64.b64decode(sys.argv[2],validate=True)));"
        f"ns={{'__name__':'embedded_ready_successor_consumer','__file__':{remote_filename!r}}};"
        "exec(compile(raw,ns['__file__'],'exec'),ns);"
        f"ns['EMBEDDED_READY_SUCCESSOR_CONSUMER_SHA256']={digest!r};"
        "queue=req['queue'];ns['_validate_queue'](queue);"
        "assert req['function']=='finalize_ready_probe_local';"
        "result=ns[req['function']](queue,**req['kwargs']);"
        "print(json.dumps(result,allow_nan=False,ensure_ascii=False,sort_keys=True))"
    )
    return [lean.ISAAC_PYTHON, "-B", "-c", program, encoded_script, encoded_request]


def cmd_finalize_full_scene_probe(
    queue: dict[str, Any],
    *,
    attempt_id: str,
    execute: bool,
    confirm: str | None,
) -> dict[str, Any]:
    _validate_queue(queue)
    if execute and confirm != PROBE_FINALIZE_CONFIRM:
        raise ReadySuccessorError(
            f"--execute requires --confirm {PROBE_FINALIZE_CONFIRM}"
        )
    job = _job(queue, PROBE_JOB)
    slot = lean._slot_by_identity(queue, "pod2", PROBE_GPU)
    claim, _argv, run_dir = lean._full_scene_probe_contract(
        queue, job, slot, attempt_id
    )
    generic = lean._finalize_full_scene_probe_script(
        job, "pod2", run_dir, claim["content_sha256"]
    )
    marker = "READY_SUCCESSOR_SPECIALIZED_PROBE_RESULT_JSON"
    specialized = _embedded_remote_command(
        queue,
        function="finalize_ready_probe_local",
        kwargs={"attempt_id": attempt_id},
    )
    remote = generic + f"\nprintf '%s\\n' {shlex.quote(marker)}\n" + shlex.join(
        specialized
    )
    result: dict[str, Any] = {
        "mode": "finalize-ready-successor-full-scene-probe",
        "dry_run": not execute,
        "job_id": PROBE_JOB,
        "resource": "pod2/gpu1",
        "run_dir": run_dir,
        "generic_result_path": f"{run_dir}/probe_result.json",
        "specialized_result_path": f"{run_dir}/ready_successor_probe_result.json",
        "claim_sha256": claim["content_sha256"],
        "one_ssh_transaction": True,
        "automatic_retry": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *lean._ssh_prefix(queue, "pod2"),
            f"bash -lc {shlex.quote(remote)}",
        ]
        return result
    raw = lean._run_ssh(
        queue,
        "pod2",
        remote,
        timeout=240,
        phase=f"finalize-ready-successor-probe:{attempt_id}",
    )
    token = marker + "\n"
    if token not in raw:
        raise ReadySuccessorError("remote finalizer omitted specialized result marker")
    try:
        terminal = json.loads(raw.rsplit(token, 1)[1].strip())
    except json.JSONDecodeError as exc:
        raise ReadySuccessorError("remote specialized probe result is malformed") from exc
    receipt = _mapping(terminal.get("receipt"), "specialized probe receipt")
    content = _validate_envelope(receipt, "specialized probe receipt")
    if content.get("status") != "passed" or content.get("unlock_authorized") is not True:
        raise ReadySuccessorError("specialized ready probe did not pass")
    producer_sha = _mapping(
        content.get("consumer_source"), "specialized probe consumer_source"
    ).get("sha256")
    evidence = {
        "schema_version": 1,
        "kind": "ready_successor_specialized_probe_receipt_v1",
        "status": "passed",
        "unlock_authorized": True,
        "producer_runner_source_sha256": producer_sha,
        "receipt_path": terminal.get("receipt_path"),
        "receipt_file_sha256": terminal.get("receipt_file_sha256"),
        "receipt_file_base64": terminal.get("receipt_file_base64"),
        "receipt_content_sha256": terminal.get("receipt_content_sha256"),
        "receipt_content": content,
    }
    evidence["evidence_content_sha256"] = _evidence_content_sha(evidence)
    _validate_probe_binding_evidence(queue, evidence)
    return {
        **result,
        "terminal_status": "passed",
        "terminal_result": terminal,
        "activation_evidence": evidence,
        "activation_still_forbidden_by_this_probe_only_runner": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("validate")
    sub.add_parser("plan")
    inspect_parent = sub.add_parser("inspect-parent")
    inspect_parent.add_argument("--execute", action="store_true")
    inspect_parent.add_argument("--confirm")
    probe = sub.add_parser("full-scene-probe")
    probe.add_argument("--attempt-id", required=True)
    probe.add_argument("--execute", action="store_true")
    probe.add_argument("--confirm")
    finalize = sub.add_parser("finalize-full-scene-probe")
    finalize.add_argument("--attempt-id", required=True)
    finalize.add_argument("--execute", action="store_true")
    finalize.add_argument("--confirm")
    local = sub.add_parser("_local-finalize-ready-probe")
    local.add_argument("--attempt-id", required=True)
    local.add_argument("--confirm", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue)
        if args.mode == "validate":
            result = cmd_validate(queue)
        elif args.mode == "plan":
            result = cmd_plan(queue)
        elif args.mode == "inspect-parent":
            result = cmd_inspect_parent(
                queue, execute=args.execute, confirm=args.confirm
            )
        elif args.mode == "full-scene-probe":
            result = cmd_full_scene_probe(
                queue,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "finalize-full-scene-probe":
            result = cmd_finalize_full_scene_probe(
                queue,
                attempt_id=args.attempt_id,
                execute=args.execute,
                confirm=args.confirm,
            )
        elif args.mode == "_local-finalize-ready-probe":
            if args.confirm != LOCAL_PROBE_FINALIZE_CONFIRM:
                raise ReadySuccessorError("local probe finalizer confirmation mismatch")
            result = finalize_ready_probe_local(queue, attempt_id=args.attempt_id)
        else:  # pragma: no cover
            raise ReadySuccessorError(f"unsupported mode: {args.mode}")
    except (
        ReadySuccessorError,
        taskrev.SuccessorQueueError,
        continuation.ContinuationQueueError,
        lean.QueueError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
