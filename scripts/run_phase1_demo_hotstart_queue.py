#!/usr/bin/env python3
"""Fail-closed Pod2 demo-only strict-resume queue.

This is deliberately separate from ``run_lean_training_queue.py``: the generic
queue remains fresh-only.  Here every row resumes a declared model checkpoint,
preserves the full optimizer state, opts into a hard-contract mismatch, and is
therefore permanently formal-ineligible.  ``plan`` is dry-run; parent
attestation and launch each require distinct simulation-only confirmation
tokens.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
from typing import Any

import yaml


HERE = Path(__file__).resolve()
GENERIC_PATH = HERE.with_name("run_lean_training_queue.py")
SPEC = importlib.util.spec_from_file_location("lean_queue_for_demo_hotstart", GENERIC_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the generic lean queue module")
Q = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = Q
SPEC.loader.exec_module(Q)


class DemoQueueError(RuntimeError):
    pass


PARENT_ATTEST_CONFIRM = "SIM_ONLY_ATTEST_DEMO_WARMSTART_PARENTS"
LAUNCH_CONFIRM = "SIM_ONLY_LAUNCH_ONE_DEMO_WARMSTART_JOB"
ATTEST_CONFIRM = "SIM_ONLY_ATTEST_ONE_DEMO_WARMSTART_MILESTONE"
EXPECTED_SOURCE = "2c2d70d6d0ccf7b0757aac4dd8e575c2e077607e"
EXPECTED_SOURCE_CHECKOUT = "/workspace/codexschema/nohope_p1_activation_successor_2c2d70d"
EXPECTED_MOTION_BINDINGS = {
    "motion_file": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_forehand_v4rg_cal.npz",
    "motion_file_2": "/workspace/codexschema/phase1_fresh_20260711/assets/v4rg_runtime_order_v3/hope_backhand_v4rg_cal.npz",
}
EXPECTED_BANK = "/workspace/codexschema/phase1_signed_face_rescue_20260713/assets/schema3_bank_rebind_v2/s1_v4rg_runtime_order_schema3_train_882fea4_rebound.npz"
EXPECTED_EXAM = "/workspace/codexschema/phase1_signed_face_rescue_20260713/papers/signed_face_exam_k100_v1/signed_face_exam_k100.schedule.json"
EXPECTED_PARENT_ITERATION = 3500
EXPECTED_MILESTONES = [3700, 4000, 4500, 5500, 7500]
EXPECTED_SLOTS = [
    "pod2/gpu0", "pod2/gpu1", "pod2/gpu0",
    "pod2/gpu1", "pod2/gpu0", "pod2/gpu1",
]


PARENT_PROGRAM = r'''import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import torch


class ParentError(RuntimeError):
    pass


def canonical_sha256(value):
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_no_symlink_components(path, label, *, leaf_may_be_missing=False):
    if not path.is_absolute() or not str(path).startswith("/workspace/"):
        raise ParentError(f"{label} must be an absolute /workspace path")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            item = current.lstat()
        except FileNotFoundError:
            if leaf_may_be_missing and index == len(parts) - 1:
                return
            raise ParentError(f"{label} component missing: {current}")
        if stat.S_ISLNK(item.st_mode):
            raise ParentError(f"{label} contains a symlink component: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(item.st_mode):
            raise ParentError(f"{label} parent is not a directory: {current}")


def safe_mkdirs(path, label):
    if not path.is_absolute() or not str(path).startswith("/workspace/"):
        raise ParentError(f"{label} must be an absolute /workspace path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            item = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            item = current.lstat()
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise ParentError(f"{label} is not a real directory: {current}")


def file_bytes(path, label):
    require_no_symlink_components(path, label)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ParentError(f"{label} missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ParentError(f"{label} must be a non-empty regular non-symlink file")
    payload = path.read_bytes()
    after = path.lstat()
    signature = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    if signature(before) != signature(after):
        raise ParentError(f"{label} changed while reading")
    return payload, signature(after)


def finite_audit(value):
    seen = set()
    tensors = floating = elements = nonfinite = 0
    def visit(item):
        nonlocal tensors, floating, elements, nonfinite
        if isinstance(item, torch.Tensor):
            tensors += 1
            if torch.is_floating_point(item) or torch.is_complex(item):
                floating += 1
                count = int(item.numel())
                elements += count
                nonfinite += count - int(torch.isfinite(item).sum().item())
            return
        if isinstance(item, dict):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                return
            seen.add(id(item))
            for child in item:
                visit(child)
    visit(value)
    return {
        "tensor_count": tensors,
        "floating_tensor_count": floating,
        "floating_elements": elements,
        "nonfinite_floating_elements": nonfinite,
    }


def audit_parent(name, item):
    checkpoint_path = Path(item["checkpoint_path"])
    hard_path = Path(item["hard_contract_path"])
    raw, checkpoint_signature = file_bytes(checkpoint_path, f"{name} checkpoint")
    hard_raw, hard_signature = file_bytes(hard_path, f"{name} hard contract")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ParentError(f"cannot load {name} checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ParentError(f"{name} checkpoint is not a mapping")
    if type(checkpoint.get("iter")) is not int or checkpoint["iter"] != item["iteration"]:
        raise ParentError(f"{name} embedded iteration mismatch")
    if "optimizer_state_dict" not in checkpoint:
        raise ParentError(f"{name} checkpoint lacks optimizer_state_dict")
    audit = finite_audit(checkpoint)
    if audit["floating_tensor_count"] <= 0 or audit["nonfinite_floating_elements"] != 0:
        raise ParentError(f"{name} checkpoint floating tensors are not finite")
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        raise ParentError(f"{name} checkpoint infos missing")
    hard_sha = hashlib.sha256(hard_raw).hexdigest()
    if infos.get("training_contract_sha256") != hard_sha:
        raise ParentError(f"{name} checkpoint/hard-contract SHA binding mismatch")
    if infos.get("training_contract_schema_version") != 3:
        raise ParentError(f"{name} parent hard-contract schema binding is not 3")
    lineage = infos.get("training_contract_lineage_exact")
    if type(lineage) is not int or lineage != 1:
        raise ParentError(f"{name} parent lineage is not exact")
    launch_claim_sha = infos.get("training_launch_claim_sha256")
    if (
        not isinstance(launch_claim_sha, str)
        or len(launch_claim_sha) != 64
        or any(character not in "0123456789abcdef" for character in launch_claim_sha)
    ):
        raise ParentError(f"{name} parent launch-claim SHA is missing or malformed")
    if checkpoint_signature != (
        checkpoint_path.lstat().st_dev, checkpoint_path.lstat().st_ino,
        checkpoint_path.lstat().st_size, checkpoint_path.lstat().st_mtime_ns,
    ):
        raise ParentError(f"{name} checkpoint changed while auditing")
    if hard_signature != (
        hard_path.lstat().st_dev, hard_path.lstat().st_ino,
        hard_path.lstat().st_size, hard_path.lstat().st_mtime_ns,
    ):
        raise ParentError(f"{name} hard contract changed while auditing")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "hard_contract_path": str(hard_path),
        "hard_contract_sha256": hard_sha,
        "embedded_iteration": checkpoint["iter"],
        "optimizer_state_dict_present": True,
        "parent_training_contract_lineage_exact": True,
        "training_launch_claim_sha256": launch_claim_sha,
        "finite_audit": audit,
    }


def main():
    if len(sys.argv) != 2:
        raise ParentError("one base64 JSON specification is required")
    spec = json.loads(base64.b64decode(sys.argv[1], validate=True))
    content = {
        "schema_version": 1,
        "purpose": "demo_only_strict_full_state_warm_start_parent_receipt",
        "source_commit": spec["source_commit"],
        "transfer_mode": "strict_full_state_preserve_optimizer",
        "descendant_exact_eligible": False,
        "parents": {
            name: audit_parent(name, item)
            for name, item in sorted(spec["parents"].items())
        },
    }
    receipt = {
        "schema_version": 1,
        "content": content,
        "content_sha256": canonical_sha256(content),
    }
    encoded = (
        json.dumps(
            receipt, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ) + "\n"
    ).encode("utf-8")
    output = Path(spec["receipt_path"])
    if spec["mode"] == "attest":
        safe_mkdirs(output.parent, "activation receipt parent")
        require_no_symlink_components(output, "activation receipt", leaf_may_be_missing=True)
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    elif spec["mode"] == "verify":
        existing, _signature = file_bytes(output, "activation receipt")
        if existing != encoded:
            raise ParentError("activation receipt content differs from current parents")
    else:
        raise ParentError("mode must be attest or verify")
    receipt_sha = hashlib.sha256(encoded).hexdigest()
    expected = spec.get("expected_receipt_sha256")
    if expected is not None and receipt_sha != expected:
        raise ParentError("activation receipt file SHA mismatch")
    print(json.dumps({
        "status": "DEMO_WARMSTART_PARENTS_OK",
        "mode": spec["mode"],
        "receipt_path": str(output),
        "receipt_file_sha256": receipt_sha,
        "receipt": receipt,
    }, sort_keys=True))


try:
    main()
except ParentError as exc:
    print(f"DEMO_WARMSTART_PARENT_ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
'''


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _values(job: dict[str, Any]) -> dict[str, str]:
    Q._compile_recipe_override_keys(job, job["id"])
    result: dict[str, str] = {}
    for raw in [*job["recipe"]["base"], *job["recipe"]["delta"]]:
        result[Q._override_key(raw, job["id"])] = raw.partition("=")[2]
    return result


def _require_sha(value: Any, label: str, *, pending: bool) -> str | None:
    if pending and value is None:
        return None
    if not isinstance(value, str) or Q.SHA256.fullmatch(value) is None:
        raise DemoQueueError(f"{label} must be a SHA-256")
    return value


def load_queue(path: Path) -> dict[str, Any]:
    try:
        queue = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DemoQueueError(f"cannot read queue: {exc}") from exc
    if not isinstance(queue, dict) or queue.get("schema_version") != 1:
        raise DemoQueueError("queue schema_version must be 1")
    if queue.get("simulation_only") is not True:
        raise DemoQueueError("simulation_only must be true")
    if queue.get("dispatch_pods") != ["pod2"]:
        raise DemoQueueError("demo queue must dispatch only to Pod2")
    if list(queue.get("pods", {})) != ["pod1", "pod2"]:
        raise DemoQueueError("pods must remain ordered pod1, pod2")
    if queue["pods"]["pod1"].get("max_trainers_per_gpu") != 4:
        raise DemoQueueError("Pod1 capacity declaration must remain 4")
    if queue["pods"]["pod2"].get("max_trainers_per_gpu") != 3:
        raise DemoQueueError("Pod2 capacity declaration must remain 3")

    activation = queue.get("activation_contract")
    if not isinstance(activation, dict):
        raise DemoQueueError("activation_contract is required")
    state = activation.get("state")
    if state not in {"pending_parent_receipt_and_gpu_release", "activated"}:
        raise DemoQueueError("invalid activation state")
    pending = state != "activated"
    if queue.get("launch_authorized") is not (not pending):
        raise DemoQueueError("launch_authorized must exactly follow activation state")
    receipt_path = Q._ready_workspace_path(
        activation.get("receipt_path"), "activation receipt path"
    )
    _require_sha(
        activation.get("receipt_file_sha256"),
        "activation receipt_file_sha256", pending=pending,
    )
    if activation.get("gpu_release_rule") != (
        "pod2_gpu0_and_gpu1_slots_free_after_existing_scaleout_model500_stop"
    ):
        raise DemoQueueError("activation GPU release rule changed")

    parents = queue.get("parents")
    if not isinstance(parents, dict) or list(parents) != ["qdot", "v1v2", "control"]:
        raise DemoQueueError("parents must be ordered qdot, v1v2, control")
    for name, parent in parents.items():
        if not isinstance(parent, dict):
            raise DemoQueueError(f"parent {name} must be a mapping")
        checkpoint = Q._ready_workspace_path(
            parent.get("checkpoint_path"), f"parent {name} checkpoint"
        )
        hard = Q._ready_workspace_path(
            parent.get("hard_contract_path"), f"parent {name} hard contract"
        )
        if PurePosixPath(checkpoint).name != "model_3500.pt":
            raise DemoQueueError(f"parent {name} must be model_3500.pt")
        if PurePosixPath(hard) != PurePosixPath(checkpoint).parent / "params/training_contract.json":
            raise DemoQueueError(f"parent {name} hard contract is not adjacent")
        if parent.get("embedded_iteration") != EXPECTED_PARENT_ITERATION:
            raise DemoQueueError(f"parent {name} iteration must be 3500")
        if parent.get("optimizer_state_dict_required") is not True:
            raise DemoQueueError(f"parent {name} must preserve optimizer state")
        _require_sha(parent.get("checkpoint_sha256"), f"parent {name} checkpoint SHA", pending=pending)
        _require_sha(parent.get("hard_contract_sha256"), f"parent {name} hard SHA", pending=pending)
        _require_sha(
            parent.get("training_launch_claim_sha256"),
            f"parent {name} launch claim SHA", pending=pending,
        )

    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 6:
        raise DemoQueueError("exactly six demo rows are required")
    if [job.get("resource", {}).get("required_slot") for job in jobs] != EXPECTED_SLOTS:
        raise DemoQueueError("jobs must round-robin Pod2 GPU0/GPU1")
    ids: set[str] = set()
    runs: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise DemoQueueError("each job must be a mapping")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not Q.SAFE_ID.fullmatch(job_id) or job_id in ids:
            raise DemoQueueError("job ids must be unique safe identifiers")
        ids.add(job_id)
        if job.get("status") != ("blocked" if pending else "ready"):
            raise DemoQueueError(f"{job_id} status does not match activation state")
        if pending and not isinstance(job.get("blocker"), str):
            raise DemoQueueError(f"{job_id} must explain its blocker")
        if not pending and job.get("blocker") not in (None, ""):
            raise DemoQueueError(f"{job_id} ready row still has a blocker")
        if job.get("runtime_binding") is not True:
            raise DemoQueueError(f"{job_id} requires runtime binding")
        if job.get("source", {}).get("commit") != EXPECTED_SOURCE:
            raise DemoQueueError(f"{job_id} source changed")
        if job["source"].get("checkout") != EXPECTED_SOURCE_CHECKOUT:
            raise DemoQueueError(f"{job_id} source checkout changed")
        if job.get("action") != "signed_face_v4rg_shared_face":
            raise DemoQueueError(f"{job_id} action changed")
        if job.get("motion", {}).get("bindings") != EXPECTED_MOTION_BINDINGS:
            raise DemoQueueError(f"{job_id} v4rg motion binding changed")
        if job.get("bank", {}).get("train_path") != EXPECTED_BANK:
            raise DemoQueueError(f"{job_id} schema-3 bank changed")
        if job.get("exam", {}).get("path") != EXPECTED_EXAM:
            raise DemoQueueError(f"{job_id} immutable exam changed")
        warm = job.get("warm_start")
        if not isinstance(warm, dict) or warm.get("parent") not in parents:
            raise DemoQueueError(f"{job_id} warm-start parent is invalid")
        if warm.get("transfer_mode") != "strict_full_state_preserve_optimizer":
            raise DemoQueueError(f"{job_id} transfer mode changed")
        if warm.get("checkpoint_tolerant") is not False:
            raise DemoQueueError(f"{job_id} checkpoint_tolerant must be false")
        if warm.get("allow_missing_contract") is not False:
            raise DemoQueueError(f"{job_id} allow_missing_contract must be false")
        if warm.get("allow_contract_mismatch") is not True:
            raise DemoQueueError(f"{job_id} must opt into the contract mismatch")
        if warm.get("descendant_exact_eligible") is not False:
            raise DemoQueueError(f"{job_id} descendants must be exact-ineligible")
        values = _values(job)
        parent_path = parents[warm["parent"]]["checkpoint_path"]
        required = {
            "checkpoint_path": parent_path,
            "checkpoint_tolerant": "false",
            "checkpoint_allow_missing_contract": "false",
            "checkpoint_allow_contract_mismatch": "true",
            "task.env.episode_length_s": "10.0",
            "task.rewards.racket_position_weight": "14.0",
            "task.rewards.racket_velocity_weight": "10.0",
            "task.rewards.racket_normal_weight": "5.0",
        }
        for key, expected in required.items():
            if values.get(key) != expected:
                raise DemoQueueError(f"{job_id} requires {key}={expected}")
        budget = job.get("budget")
        if budget != {"num_envs": 4096, "max_iterations": 5001, "save_interval": 100}:
            raise DemoQueueError(f"{job_id} budget changed")
        if job.get("milestones") != EXPECTED_MILESTONES:
            raise DemoQueueError(f"{job_id} absolute milestones changed")
        run_name = job.get("run_name")
        if not isinstance(run_name, str) or not Q.SAFE_ID.fullmatch(run_name) or run_name in runs:
            raise DemoQueueError("run names must be unique safe identifiers")
        runs.add(run_name)
        Q._ready_workspace_path(job.get("run_dir"), f"{job_id} run_dir")
    return queue


def _parent_spec(queue: dict[str, Any], *, mode: str) -> dict[str, Any]:
    activation = queue["activation_contract"]
    return {
        "mode": mode,
        "source_commit": EXPECTED_SOURCE,
        "receipt_path": activation["receipt_path"],
        "expected_receipt_sha256": (
            activation["receipt_file_sha256"] if mode == "verify" else None
        ),
        "parents": {
            name: {
                "checkpoint_path": parent["checkpoint_path"],
                "hard_contract_path": parent["hard_contract_path"],
                "iteration": parent["embedded_iteration"],
            }
            for name, parent in queue["parents"].items()
        },
    }


def _parent_remote(queue: dict[str, Any], *, mode: str) -> str:
    encoded = base64.b64encode(
        json.dumps(_parent_spec(queue, mode=mode), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return shlex.join([Q.ISAAC_PYTHON, "-c", PARENT_PROGRAM, encoded])


def _demo_claim(
    queue: dict[str, Any], job: dict[str, Any], slot: Any
) -> tuple[dict[str, Any], list[str]]:
    claim, _old_argv = Q._launch_contract(queue, job, slot)
    content = claim["content"]
    parent_name = job["warm_start"]["parent"]
    content["demo_warm_start"] = {
        **job["warm_start"],
        **queue["parents"][parent_name],
    }
    content["activation_receipt"] = {
        "path": queue["activation_contract"]["receipt_path"],
        "file_sha256": queue["activation_contract"]["receipt_file_sha256"],
    }
    content["formal_exact_eligible"] = False
    digest = _canonical_sha256(content)
    argv = [
        *content["training_argv_without_claim"],
        f"++training_launch_claim_sha256={digest}",
    ]
    return {
        "schema_version": 2,
        "content": content,
        "content_sha256": digest,
        "training_argv": argv,
    }, argv


def _launch_script(queue: dict[str, Any], job: dict[str, Any], slot: Any) -> str:
    source = job["source"]["checkout"].rstrip("/")
    workdir = f"{source}/{Q.WBT_RELATIVE}"
    run_dir = job["run_dir"].rstrip("/")
    run_parent = str(PurePosixPath(run_dir).parent)
    claim_document, argv = _demo_claim(queue, job, slot)
    claim = json.dumps(
        claim_document, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True,
    ) + "\n"
    launcher = f"{workdir}/{Q.KIT_LAUNCHER_RELATIVE}"
    launch = shlex.join([launcher, f"{run_dir}/run.log"]) + " " + (
        Q._child_env_command(argv, slot.gpu)
    ) + f" {Q.GPU_LAUNCH_LOCK_FD}>&-"
    body = Q._doctor_body(queue, job, slot, training_argv=argv) + f"""
count=$(nvidia-smi -i {slot.gpu} --query-compute-apps=pid --format=csv,noheader,nounits | awk {shlex.quote(Q.UNIQUE_NUMERIC_PID_AWK)})
test "$count" -lt {slot.capacity}
mkdir -p {shlex.quote(run_parent)}
mkdir {shlex.quote(run_dir)}
mkdir {shlex.quote(run_dir + '/milestones')}
( set -o noclobber; printf %s {shlex.quote(claim)} > {shlex.quote(run_dir + '/queue_claim.json')} )
export KIT_BOOT_MARKER={shlex.quote(Q.KIT_BOOT_MARKER)}
export KIT_BOOT_TIMEOUT_S={Q.KIT_BOOT_TIMEOUT_SECONDS}
{launch}
printf '%s\n' phase=first_iter demo_only=true exact_eligible=false >> {shlex.quote(run_dir + '/run.log.launch')}
"""
    return Q._gpu_launch_lock_script(slot, body)


def cmd_plan(queue: dict[str, Any]) -> dict[str, Any]:
    occupancy = {slot.name: 0 for slot in Q.slots(queue)}
    assignments = Q._assign(queue, occupancy)
    return {
        "mode": "plan",
        "dry_run": True,
        "launch_authorized": queue["launch_authorized"],
        "activation_state": queue["activation_contract"]["state"],
        "assignments": [
            {"job_id": job["id"], "resource": slot.name}
            for job, slot in assignments
        ],
        "blocked": [
            {"job_id": job["id"], "reason": job["blocker"]}
            for job in queue["jobs"] if job["status"] == "blocked"
        ],
        "parent_attest_command": f"--execute --confirm {PARENT_ATTEST_CONFIRM}",
    }


def cmd_parent_attest(
    queue: dict[str, Any], *, execute: bool, confirm: str | None
) -> dict[str, Any]:
    if execute and confirm != PARENT_ATTEST_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {PARENT_ATTEST_CONFIRM}")
    remote = _parent_remote(queue, mode="attest")
    result: dict[str, Any] = {
        "mode": "parent-attest", "dry_run": not execute,
        "receipt_path": queue["activation_contract"]["receipt_path"],
        "automatic_activation": False, "automatic_retry": False,
    }
    if not execute:
        result["ssh_argv"] = [
            *Q._ssh_prefix(queue, "pod2"), f"bash -lc {shlex.quote(remote)}"
        ]
        return result
    result["remote_result"] = json.loads(
        Q._run_ssh(queue, "pod2", remote, timeout=600, phase="demo-parent-attest")
    )
    return result


def _require_remote_activation(queue: dict[str, Any]) -> dict[str, Any]:
    if queue["activation_contract"]["state"] != "activated":
        raise DemoQueueError("activation state is not activated")
    output = Q._run_ssh(
        queue, "pod2", _parent_remote(queue, mode="verify"),
        timeout=600, phase="demo-parent-verify",
    )
    result = json.loads(output)
    if result.get("receipt_file_sha256") != queue["activation_contract"]["receipt_file_sha256"]:
        raise DemoQueueError("verified receipt SHA differs from activation contract")
    try:
        observed = result["receipt"]["content"]["parents"]
    except (KeyError, TypeError) as exc:
        raise DemoQueueError("verified receipt lacks parent content") from exc
    for name, expected in queue["parents"].items():
        current = observed.get(name)
        if not isinstance(current, dict):
            raise DemoQueueError(f"verified receipt lacks parent {name}")
        bound = {
            "checkpoint_path": expected["checkpoint_path"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "hard_contract_path": expected["hard_contract_path"],
            "hard_contract_sha256": expected["hard_contract_sha256"],
            "embedded_iteration": expected["embedded_iteration"],
            "optimizer_state_dict_present": True,
            "parent_training_contract_lineage_exact": True,
            "training_launch_claim_sha256": expected["training_launch_claim_sha256"],
        }
        for key, value in bound.items():
            if current.get(key) != value:
                raise DemoQueueError(
                    f"verified parent {name}.{key} differs from activated queue"
                )
    return result


def cmd_fill(
    queue: dict[str, Any], *, execute: bool, confirm: str | None, count: int
) -> dict[str, Any]:
    if queue["launch_authorized"] is not True:
        raise DemoQueueError("launch_authorized is false; fill is blocked")
    if count <= 0:
        raise DemoQueueError("count must be positive")
    if execute and confirm != LAUNCH_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {LAUNCH_CONFIRM}")
    if not execute:
        occupancy = {slot.name: 0 for slot in Q.slots(queue)}
        assignments = Q._assign(queue, occupancy)[:count]
        return {
            "mode": "fill", "dry_run": True,
            "jobs": [
                {"job_id": job["id"], "resource": slot.name,
                 "ssh_argv": [*Q._ssh_prefix(queue, slot.pod),
                              f"bash -lc {shlex.quote(_launch_script(queue, job, slot))}"]}
                for job, slot in assignments
            ],
        }
    activation = _require_remote_activation(queue)
    launched: list[dict[str, Any]] = []
    Q.GLOBAL_SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with Q.GLOBAL_SCHEDULER_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for _ in range(count):
            occupancy, claims = Q.live_snapshot(queue)
            effective = Q._effective_occupancy(queue, occupancy, claims)
            assignments = Q._assign(queue, effective, set(claims))
            if not assignments:
                break
            job, slot = assignments[0]
            output = Q._run_ssh(
                queue, slot.pod, _launch_script(queue, job, slot),
                timeout=Q.KIT_BOOT_TIMEOUT_SECONDS + 60,
                phase=f"demo-hotstart:{job['id']}",
            )
            launched.append({"job_id": job["id"], "resource": slot.name,
                             "remote_output": output})
    if not launched:
        raise DemoQueueError("no ready job fits an available GPU slot")
    return {"mode": "fill", "dry_run": False,
            "activation": activation, "launched": launched}


def cmd_attest_milestone(
    queue: dict[str, Any], *, job_id: str, milestone: int,
    execute: bool, confirm: str | None,
) -> dict[str, Any]:
    jobs = {job["id"]: job for job in queue["jobs"]}
    if job_id not in jobs or milestone not in EXPECTED_MILESTONES:
        raise DemoQueueError("unknown job or non-preregistered milestone")
    if execute and confirm != ATTEST_CONFIRM:
        raise DemoQueueError(f"--execute requires --confirm {ATTEST_CONFIRM}")
    job = jobs[job_id]
    slot_name = job["resource"]["required_slot"]
    slot = next(slot for slot in Q.slots(queue) if slot.name == slot_name)
    remote = Q._milestone_attestor_script(job, milestone)
    result: dict[str, Any] = {
        "mode": "attest-milestone", "dry_run": not execute,
        "job_id": job_id, "milestone": milestone,
        "expected_lineage_exact": 0,
    }
    if not execute:
        result["ssh_argv"] = [
            *Q._ssh_prefix(queue, slot.pod), f"bash -lc {shlex.quote(remote)}"
        ]
        return result
    _occupancy, claims = Q.live_snapshot(queue)
    claim = claims.get(job_id)
    if claim is None or claim.get("claim_schema_version") != 2:
        raise DemoQueueError("live schema-2 job claim is missing")
    expected, _argv = _demo_claim(queue, job, slot)
    if claim.get("claim_content_sha256") != expected["content_sha256"]:
        raise DemoQueueError("live claim differs from current demo queue")
    remote_output = Q._run_ssh(
        queue, slot.pod, remote, timeout=180,
        phase=f"demo-attest:{job_id}:{milestone}",
    )
    receipt = json.loads(remote_output)
    try:
        lineage = receipt["receipt"]["content"]["hard_contract"]["lineage_exact"]
    except (KeyError, TypeError) as exc:
        raise DemoQueueError("milestone attestor omitted lineage exactness") from exc
    if type(lineage) is not int or lineage != 0:
        raise DemoQueueError("demo warm-start descendant did not remain lineage_exact=0")
    result["remote_result"] = receipt
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    parent = sub.add_parser("parent-attest")
    parent.add_argument("--execute", action="store_true")
    parent.add_argument("--confirm")
    fill = sub.add_parser("fill")
    fill.add_argument("--count", type=int, default=1)
    fill.add_argument("--execute", action="store_true")
    fill.add_argument("--confirm")
    attest = sub.add_parser("attest-milestone")
    attest.add_argument("--job-id", required=True)
    attest.add_argument("--milestone", type=int, required=True)
    attest.add_argument("--execute", action="store_true")
    attest.add_argument("--confirm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = load_queue(args.queue.resolve())
        if args.command == "plan":
            result = cmd_plan(queue)
        elif args.command == "parent-attest":
            result = cmd_parent_attest(
                queue, execute=args.execute, confirm=args.confirm
            )
        elif args.command == "fill":
            result = cmd_fill(
                queue, execute=args.execute, confirm=args.confirm, count=args.count
            )
        elif args.command == "attest-milestone":
            result = cmd_attest_milestone(
                queue, job_id=args.job_id, milestone=args.milestone,
                execute=args.execute, confirm=args.confirm,
            )
        else:
            raise DemoQueueError(f"unsupported command: {args.command}")
    except (DemoQueueError, Q.QueueError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
