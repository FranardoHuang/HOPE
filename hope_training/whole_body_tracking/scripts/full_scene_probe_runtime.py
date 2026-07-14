#!/usr/bin/env python3
"""Supervise and terminally finalize one non-science full-scene probe.

The supervisor is deliberately passive: it starts one trainer in its own
already-isolated process group, observes the first-iteration marker, waits for
natural termination, and writes an immutable exit receipt.  It never sends a
signal.  The finalizer is a separate read-only pass; only a fully bound,
finite, clean exit can produce ``status=passed``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import lean_queue_runtime as queue_runtime


class FullSceneProbeError(RuntimeError):
    """The probe evidence is terminal but cannot pass."""


class FullSceneProbeNotReady(FullSceneProbeError):
    """The probe has not naturally reached a terminal state yet."""


PROBE_PURPOSE = queue_runtime.PROBE_PURPOSE
CLAIM_NAME = queue_runtime.PROBE_CLAIM_NAME
BINDING_NAME = queue_runtime.PROBE_BINDING_NAME
EXIT_NAME = "full_scene_probe_exit.json"
RESULT_NAME = "probe_result.json"
FIRST_ITERATION_MARKER = "Learning iteration"
PHASE_PREFIX = "[train.py] LEAN_QUEUE_PHASE "
SUPERVISOR_PHASE_PREFIX = "[full_scene_probe] PHASE "
REQUIRED_TRAINER_PHASES = (
    "scene_import_start",
    "scene_import_done",
    "hard_contract_written",
)
FATAL_PATTERNS = {
    "bad_alloc": r"bad_alloc",
    "cuda_out_of_memory": r"cuda out of memory",
    "fatal_python_error": r"fatal python error",
    "inf": r"(?<![a-z])(?:inf|infinity)(?![a-z])",
    "killed": r"(?<![a-z])killed(?![a-z])",
    "malloc": r"malloc\(\):",
    "nan": r"(?<![a-z])nan(?![a-z])",
    "out_of_memory_error": r"outofmemoryerror",
    "segmentation_fault": r"segmentation fault",
    "traceback": r"traceback \(most recent call last\)",
    "train_error": r"\[train\.py\] error during run:",
}


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_absolute(value: str | Path, label: str) -> Path:
    return queue_runtime._canonical_absolute_path(str(value), label)


def _load_envelope(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope, _raw = queue_runtime._read_regular_json(path, label)
    if envelope.get("schema_version") != 1:
        raise FullSceneProbeError(f"{label} schema_version must be 1")
    content = queue_runtime._require_mapping(envelope.get("content"), f"{label} content")
    digest = queue_runtime._require_sha256(
        envelope.get("content_sha256"), f"{label} digest"
    )
    if queue_runtime.canonical_sha256(content) != digest:
        raise FullSceneProbeError(f"{label} canonical digest mismatch")
    return envelope, content


def _publish_or_accept_identical(path: Path, value: Mapping[str, Any], label: str) -> bool:
    """Publish once, or accept only the byte-identical deterministic repeat."""

    payload = _canonical_payload(value)
    if path.exists():
        observed = queue_runtime._read_regular_bytes(path, label)
        if observed != payload:
            raise FullSceneProbeError(
                f"{label} already exists with different bytes; replacement is forbidden"
            )
        return True
    queue_runtime._atomic_publish_json(path, value, label)
    return False


def _probe_claim(
    run_dir: Path, *, expected_digest: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    claim_path = run_dir / CLAIM_NAME
    claim, content, digest = queue_runtime._validate_claim(claim_path)
    if expected_digest is not None and digest != queue_runtime._require_sha256(
        expected_digest, "expected probe claim digest"
    ):
        raise FullSceneProbeError(
            "selected queue row differs from the immutable probe claim"
        )
    if content.get("purpose") != PROBE_PURPOSE:
        raise FullSceneProbeError("claim is not a full-scene non-science probe")
    if (
        content.get("not_science") is not True
        or content.get("attestable") is not False
        or content.get("promotable") is not False
    ):
        raise FullSceneProbeError("probe science/publication flags are not fail-closed")
    if _canonical_absolute(content.get("run_dir"), "claimed run_dir") != run_dir:
        raise FullSceneProbeError("claim run_dir does not match selected probe directory")
    budget = queue_runtime._require_mapping(content.get("budget"), "probe budget")
    if budget.get("milestones") != [1]:
        raise FullSceneProbeError("probe claim must bind milestones=[1]")
    if content.get("expected_training_contract_lineage_exact") != 1:
        raise FullSceneProbeError("probe claim must require fresh exact lineage=1")
    prefix = content.get("supervisor_argv_prefix")
    if not isinstance(prefix, list) or not prefix or any(
        type(item) is not str or not item for item in prefix
    ) or prefix[-1] != "--":
        raise FullSceneProbeError("probe claim has an invalid supervisor argv prefix")
    source = queue_runtime._require_mapping(content.get("source"), "probe source")
    if not isinstance(source.get("ignored_runtime_asset"), dict):
        raise FullSceneProbeError("probe claim lacks ignored runtime asset closure")
    expected_receipt = _canonical_absolute(
        content.get("source_asset_receipt_path"), "claimed source asset receipt"
    )
    if expected_receipt.parent == run_dir or run_dir in expected_receipt.parents:
        raise FullSceneProbeError("source asset receipt must remain outside probe run_dir")
    return claim, content, digest


def _process_identity(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    return queue_runtime._process_identity(pid, proc_root=proc_root, getpgid=os.getpgid)


def _binding_digest_if_valid(binding_path: Path) -> str | None:
    try:
        binding, _content = _load_envelope(binding_path, "probe run binding")
    except (queue_runtime.LeanQueueRuntimeError, FullSceneProbeError):
        return None
    return str(binding["content_sha256"])


def supervise(
    *,
    run_dir: str | Path,
    log_path: str | Path,
    trainer_argv: list[str],
    poll_seconds: float = 0.2,
) -> dict[str, Any]:
    """Wait naturally for one child trainer and emit an immutable exit receipt."""

    run_dir_obj = _canonical_absolute(run_dir, "probe run_dir")
    log_path_obj = _canonical_absolute(log_path, "probe log path")
    if log_path_obj != run_dir_obj / "run.log":
        raise FullSceneProbeError("probe log path must equal run_dir/run.log")
    claim, claim_content, claim_digest = _probe_claim(run_dir_obj)
    if not trainer_argv or any(type(item) is not str or not item for item in trainer_argv):
        raise FullSceneProbeError("trainer argv must be a non-empty string list")
    if trainer_argv != claim["training_argv"]:
        raise FullSceneProbeError("supervisor trainer argv differs from immutable claim")
    supervisor_pid = os.getpid()
    if os.getpgrp() != supervisor_pid:
        raise FullSceneProbeError("probe supervisor must be the isolated PGID leader")
    supervisor_identity = _process_identity(supervisor_pid)
    expected_supervisor_argv = [
        *claim_content["supervisor_argv_prefix"],
        *claim["training_argv"],
    ]
    if supervisor_identity["argv"] != expected_supervisor_argv:
        raise FullSceneProbeError(
            "live supervisor argv differs from the claim-derived command"
        )

    try:
        child = subprocess.Popen(trainer_argv)  # noqa: S603 - exact immutable claim argv
    except OSError as exc:
        exit_content = {
            "schema_version": 1,
            "purpose": PROBE_PURPOSE,
            "claim_path": str(run_dir_obj / CLAIM_NAME),
            "claim_content_sha256": claim_digest,
            "binding_path": str(run_dir_obj / BINDING_NAME),
            "binding_content_sha256": None,
            "log_path": str(log_path_obj),
            "supervisor_process": supervisor_identity,
            "trainer_process": None,
            "first_iteration_observed": False,
            "supervision_error": None,
            "termination": {
                "kind": "spawn_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_receipt = {
            "schema_version": 1,
            "content": exit_content,
            "content_sha256": queue_runtime.canonical_sha256(exit_content),
        }
        queue_runtime._atomic_publish_json(
            run_dir_obj / EXIT_NAME, exit_receipt, "probe exit receipt"
        )
        return exit_receipt

    supervision_error = None
    try:
        trainer_identity = _process_identity(child.pid)
    except (queue_runtime.LeanQueueRuntimeError, OSError) as exc:
        trainer_identity = {
            "pid": child.pid,
            "pgid": supervisor_pid,
            "starttime_ticks": None,
            "argv": trainer_argv,
        }
        supervision_error = f"trainer_identity_capture:{type(exc).__name__}:{exc}"
    else:
        if trainer_identity["pgid"] != supervisor_pid:
            supervision_error = "probe trainer did not inherit the supervisor PGID"
        elif trainer_identity["argv"] != trainer_argv:
            supervision_error = "spawned trainer /proc argv differs from immutable claim"

    first_iteration = False
    while child.poll() is None:
        if not first_iteration and log_path_obj.exists():
            try:
                first_iteration = FIRST_ITERATION_MARKER in log_path_obj.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                first_iteration = False
            if first_iteration:
                print(
                    SUPERVISOR_PHASE_PREFIX
                    + json.dumps(
                        {"phase": "first_iteration_observed"},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
        time.sleep(poll_seconds)
    returncode = child.wait()
    if not first_iteration and log_path_obj.exists():
        first_iteration = FIRST_ITERATION_MARKER in log_path_obj.read_text(
            encoding="utf-8", errors="replace"
        )
        if first_iteration:
            print(
                SUPERVISOR_PHASE_PREFIX
                + json.dumps(
                    {"phase": "first_iteration_observed"},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

    if returncode < 0:
        termination = {"kind": "signal", "signal": -returncode}
    else:
        termination = {"kind": "normal_exit", "exit_code": returncode}
    exit_content = {
        "schema_version": 1,
        "purpose": PROBE_PURPOSE,
        "claim_path": str(run_dir_obj / CLAIM_NAME),
        "claim_content_sha256": claim_digest,
        "binding_path": str(run_dir_obj / BINDING_NAME),
        "binding_content_sha256": _binding_digest_if_valid(run_dir_obj / BINDING_NAME),
        "log_path": str(log_path_obj),
        "supervisor_process": supervisor_identity,
        "trainer_process": trainer_identity,
        "first_iteration_observed": first_iteration,
        "supervision_error": supervision_error,
        "termination": termination,
    }
    exit_receipt = {
        "schema_version": 1,
        "content": exit_content,
        "content_sha256": queue_runtime.canonical_sha256(exit_content),
    }
    queue_runtime._atomic_publish_json(
        run_dir_obj / EXIT_NAME, exit_receipt, "probe exit receipt"
    )
    return exit_receipt


def _require_naturally_absent(
    process: Mapping[str, Any],
    label: str,
    *,
    proc_root: Path,
    getpgid: Callable[[int], int],
) -> None:
    pid = queue_runtime._require_plain_int(process.get("pid"), f"{label} PID", minimum=1)
    expected_start = queue_runtime._require_plain_int(
        process.get("starttime_ticks"), f"{label} starttime", minimum=1
    )
    proc_dir = proc_root / str(pid)
    if not proc_dir.exists():
        return
    observed = queue_runtime._process_identity(
        pid, proc_root=proc_root, getpgid=getpgid
    )
    if observed["starttime_ticks"] != expected_start:
        raise FullSceneProbeError(f"{label} PID was reused after probe termination")
    raise FullSceneProbeNotReady(f"{label} is still live; natural exit is required")


def _stat_process_group(stat_text: str) -> int:
    close = stat_text.rfind(")")
    if close < 0:
        raise FullSceneProbeError("proc stat lacks a closing command parenthesis")
    fields = stat_text[close + 2 :].split()
    if len(fields) <= 2 or not fields[2].isdigit():
        raise FullSceneProbeError("proc stat lacks a numeric process-group id")
    pgid = int(fields[2])
    if pgid <= 0:
        raise FullSceneProbeError("proc process-group id must be positive")
    return pgid


def _require_process_group_empty(expected_pgid: int, *, proc_root: Path) -> None:
    """Reject a naturally exited leader that left any orphan in its old PGID."""

    if not proc_root.is_dir() or proc_root.is_symlink():
        raise FullSceneProbeError("procfs root is missing or a symlink")
    members: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise FullSceneProbeError("cannot enumerate procfs for probe PGID closure") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            before = (entry / "stat").read_text(encoding="utf-8")
            observed_pgid = _stat_process_group(before)
            after = (entry / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError) as exc:
            raise FullSceneProbeError(
                f"cannot read proc identity while proving PGID {expected_pgid} empty"
            ) from exc
        if queue_runtime._proc_starttime(before) != queue_runtime._proc_starttime(after):
            raise FullSceneProbeError("proc identity changed during PGID closure scan")
        if observed_pgid == expected_pgid:
            members.append(int(entry.name))
    if members:
        raise FullSceneProbeNotReady(
            f"probe process group {expected_pgid} still has live members: {sorted(members)}"
        )


def _parse_log(log_path: Path, expected_hard_sha: str) -> dict[str, Any]:
    raw = queue_runtime._read_regular_bytes(log_path, "probe run log")
    text = raw.decode("utf-8", "replace")
    phases: list[tuple[str, dict[str, Any]]] = []
    for line in text.splitlines():
        if PHASE_PREFIX in line:
            payload_text = line.split(PHASE_PREFIX, 1)[1]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise FullSceneProbeError("probe log has malformed trainer phase telemetry") from exc
            phase = payload.get("phase")
            if type(phase) is str:
                phases.append((phase, payload))
        if SUPERVISOR_PHASE_PREFIX in line:
            payload_text = line.split(SUPERVISOR_PHASE_PREFIX, 1)[1]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise FullSceneProbeError("probe log has malformed supervisor phase telemetry") from exc
            phase = payload.get("phase")
            if type(phase) is str:
                phases.append((phase, payload))
    phase_names = [name for name, _payload in phases]
    positions: list[int] = []
    for required in REQUIRED_TRAINER_PHASES:
        if phase_names.count(required) != 1:
            raise FullSceneProbeError(
                f"probe log must contain exactly one {required} phase"
            )
        positions.append(phase_names.index(required))
    if positions != sorted(positions):
        raise FullSceneProbeError("probe scene/contract phases are out of order")
    hard_payload = next(payload for name, payload in phases if name == "hard_contract_written")
    if hard_payload.get("sha256") != expected_hard_sha:
        raise FullSceneProbeError("hard_contract_written phase SHA differs from adjacent contract")
    first_phase_count = phase_names.count("first_iteration_observed")
    if first_phase_count != 1 or FIRST_ITERATION_MARKER not in text:
        raise FullSceneProbeError("probe log lacks one bound first-iteration observation")
    lowered = text.lower()
    fatal_hits = {
        name: len(re.findall(pattern, lowered))
        for name, pattern in FATAL_PATTERNS.items()
    }
    if any(fatal_hits.values()):
        raise FullSceneProbeError(f"probe log contains fatal markers: {fatal_hits}")
    return {
        "path": str(log_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "phase_sequence": phase_names,
        "fatal_hits": fatal_hits,
    }


def _asset_hashes_and_contract_binding(
    claim_content: Mapping[str, Any], hard_contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    inputs = queue_runtime._require_mapping(claim_content.get("inputs"), "probe inputs")
    motion = queue_runtime._require_mapping(inputs.get("motion"), "probe motion input")
    bindings = queue_runtime._require_mapping(motion.get("bindings"), "probe motion bindings")
    ordered_motion = sorted(
        bindings.items(), key=lambda item: (0 if item[0] == "motion_file" else 1, item[0])
    )
    hard_clips = hard_contract.get("motion_clips")
    if not isinstance(hard_clips, list) or len(hard_clips) != len(ordered_motion):
        raise FullSceneProbeError("hard contract motion clips differ from claimed inputs")
    evidence: list[dict[str, Any]] = []
    for index, ((argument, raw_path), hard_clip) in enumerate(zip(ordered_motion, hard_clips)):
        path = _canonical_absolute(raw_path, f"claimed motion {argument}")
        digest = hashlib.sha256(
            queue_runtime._read_regular_bytes(path, f"motion asset {argument}")
        ).hexdigest()
        if not isinstance(hard_clip, dict):
            raise FullSceneProbeError("hard contract motion clip entry is not a mapping")
        if hard_clip.get("index") != index or hard_clip.get("sha256") != digest:
            raise FullSceneProbeError("hard contract motion asset binding mismatch")
        evidence.append({"kind": "motion", "argument": argument, "path": str(path), "sha256": digest})
    bank = queue_runtime._require_mapping(inputs.get("bank"), "probe bank input")
    bank_path = _canonical_absolute(bank.get("train_path"), "claimed train bank")
    bank_digest = hashlib.sha256(
        queue_runtime._read_regular_bytes(bank_path, "train bank asset")
    ).hexdigest()
    hard_bank = queue_runtime._require_mapping(
        hard_contract.get("question_bank"), "hard contract question bank"
    )
    if hard_bank.get("sha256") != bank_digest:
        raise FullSceneProbeError("hard contract train-bank asset binding mismatch")
    evidence.append({"kind": "train_bank", "path": str(bank_path), "sha256": bank_digest})
    return evidence


def _validate_source_asset_receipt(
    claim_content: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    claimed_path = _canonical_absolute(
        claim_content.get("source_asset_receipt_path"),
        "claimed source asset receipt",
    )
    if receipt_path != claimed_path:
        raise FullSceneProbeError(
            "selected source asset receipt path differs from immutable claim"
        )
    receipt, content = _load_envelope(receipt_path, "source asset receipt")
    if content.get("schema_version") != 1:
        raise FullSceneProbeError("source asset receipt content schema must be 1")
    source = queue_runtime._require_mapping(claim_content.get("source"), "probe source")
    source_identity = {
        "checkout": source.get("checkout"),
        "commit": source.get("commit"),
    }
    asset_contract = queue_runtime._require_mapping(
        source.get("ignored_runtime_asset"), "ignored runtime asset contract"
    )
    expected = {
        "pod": claim_content.get("pod"),
        "source": source_identity,
        "ignored_runtime_asset": asset_contract,
        "ignored_runtime_asset_sha256": queue_runtime.canonical_sha256(asset_contract),
        "target_gitignored": True,
        "symlinks_present": False,
    }
    for key, value in expected.items():
        if content.get(key) != value:
            raise FullSceneProbeError(
                f"source asset receipt {key} differs from immutable probe claim"
            )
    target_path = _canonical_absolute(content.get("target_path"), "source asset target")
    expected_target = Path(str(source_identity["checkout"])) / str(
        asset_contract.get("target_relative_path")
    )
    if target_path != expected_target:
        raise FullSceneProbeError("source asset receipt target path mismatch")
    inventory = queue_runtime._require_mapping(
        content.get("inventory"), "source asset inventory"
    )
    expected_inventory = {
        "file_count": asset_contract.get("file_count"),
        "total_file_bytes": asset_contract.get("total_file_bytes"),
        "tree_content_sha256": asset_contract.get("tree_content_sha256"),
    }
    if inventory != expected_inventory:
        raise FullSceneProbeError("source asset receipt inventory mismatch")
    urdf = queue_runtime._require_mapping(
        content.get("urdf_reference_closure"), "source asset URDF closure"
    )
    unique = urdf.get("unique_mesh_references")
    if type(unique) is not int or unique <= 0 or urdf.get("resolved_regular_meshes") != unique:
        raise FullSceneProbeError("source asset receipt has incomplete URDF mesh closure")
    raw = queue_runtime._read_regular_bytes(receipt_path, "source asset receipt")
    return {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": receipt["content_sha256"],
        "ignored_runtime_asset_sha256": content["ignored_runtime_asset_sha256"],
        "inventory": inventory,
        "urdf_reference_closure": urdf,
    }


def _validate_terminal(
    run_dir: Path,
    *,
    expected_claim_digest: str,
    source_asset_receipt: Path,
    checkpoint_loader: Callable[[Path], Any] | None,
    torch_module: Any | None,
    proc_root: Path,
    getpgid: Callable[[int], int],
    source_verifier: Callable[[Path, str], Mapping[str, Any]],
) -> dict[str, Any]:
    claim, claim_content, claim_digest = _probe_claim(
        run_dir, expected_digest=expected_claim_digest
    )
    exit_receipt, exit_content = _load_envelope(run_dir / EXIT_NAME, "probe exit receipt")
    binding, binding_content, _bound_claim, _bound_claim_content = queue_runtime._load_binding(
        run_dir / BINDING_NAME
    )
    binding_digest = queue_runtime._require_sha256(
        binding.get("content_sha256"), "probe binding digest"
    )
    expected_exit = {
        "purpose": PROBE_PURPOSE,
        "claim_path": str(run_dir / CLAIM_NAME),
        "claim_content_sha256": claim_digest,
        "binding_path": str(run_dir / BINDING_NAME),
        "binding_content_sha256": binding_digest,
        "log_path": str(run_dir / "run.log"),
        "trainer_process": binding_content.get("process"),
        "supervisor_process": binding_content.get("supervisor_process"),
        "supervision_error": None,
    }
    for key, value in expected_exit.items():
        if exit_content.get(key) != value:
            raise FullSceneProbeError(f"probe exit receipt {key} differs from immutable binding")
    _require_naturally_absent(
        queue_runtime._require_mapping(exit_content.get("trainer_process"), "exit trainer"),
        "bound trainer",
        proc_root=proc_root,
        getpgid=getpgid,
    )
    _require_naturally_absent(
        queue_runtime._require_mapping(exit_content.get("supervisor_process"), "exit supervisor"),
        "bound supervisor",
        proc_root=proc_root,
        getpgid=getpgid,
    )
    supervisor_process = queue_runtime._require_mapping(
        exit_content.get("supervisor_process"), "exit supervisor"
    )
    isolated_pgid = queue_runtime._require_plain_int(
        supervisor_process.get("pgid"), "bound isolated PGID", minimum=1
    )
    if supervisor_process.get("pid") != isolated_pgid:
        raise FullSceneProbeError("bound supervisor is not the isolated PGID leader")
    _require_process_group_empty(isolated_pgid, proc_root=proc_root)
    termination = queue_runtime._require_mapping(
        exit_content.get("termination"), "probe termination"
    )
    if termination != {"kind": "normal_exit", "exit_code": 0}:
        raise FullSceneProbeError(f"probe trainer did not exit normally with rc=0: {termination}")
    if exit_content.get("first_iteration_observed") is not True:
        raise FullSceneProbeError("probe exit receipt did not observe the first iteration")

    source = queue_runtime._require_mapping(claim_content.get("source"), "probe source")
    source_path = _canonical_absolute(source.get("checkout"), "probe source checkout")
    source_commit = queue_runtime._require_text(source.get("commit"), "probe source commit")
    source_state = dict(source_verifier(source_path, source_commit))
    if source_state != {"head": source_commit, "clean": True}:
        raise FullSceneProbeError("terminal source verifier did not prove exact clean source")
    source_asset_evidence = _validate_source_asset_receipt(
        claim_content, source_asset_receipt
    )

    log_dir = _canonical_absolute(binding_content.get("rsl_log_dir"), "bound RSL log dir")
    hard_path = log_dir / "params/training_contract.json"
    hard_contract, hard_raw = queue_runtime._read_regular_json(
        hard_path, "hard training contract"
    )
    if hard_contract.get("schema_version") != queue_runtime.TRAINING_CONTRACT_SCHEMA_VERSION:
        raise FullSceneProbeError("probe hard contract is not schema 3")
    hard_sha = hashlib.sha256(hard_raw).hexdigest()
    log_evidence = _parse_log(run_dir / "run.log", hard_sha)
    asset_evidence = _asset_hashes_and_contract_binding(claim_content, hard_contract)

    checkpoint_path = log_dir / "model_1.pt"
    before = queue_runtime._checkpoint_stat(checkpoint_path)
    if checkpoint_loader is None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        if torch_module is None:
            raise FullSceneProbeError("injected checkpoint loader requires torch_module")
        checkpoint = checkpoint_loader(checkpoint_path)
    if queue_runtime._checkpoint_stat(checkpoint_path) != before:
        raise FullSceneProbeError("probe checkpoint changed while loading")
    checkpoint = queue_runtime._require_mapping(checkpoint, "probe checkpoint")
    if checkpoint.get("iter") != 1:
        raise FullSceneProbeError("model_1 filename iteration differs from embedded iteration")
    infos = queue_runtime._require_mapping(checkpoint.get("infos"), "probe checkpoint infos")
    if infos.get("training_contract_schema_version") != 3:
        raise FullSceneProbeError("probe checkpoint hard-contract schema binding mismatch")
    if infos.get("training_contract_sha256") != hard_sha:
        raise FullSceneProbeError("probe checkpoint hard-contract SHA binding mismatch")
    if infos.get("training_launch_claim_sha256") != claim_digest:
        raise FullSceneProbeError("probe checkpoint launch-claim binding mismatch")
    if infos.get("training_contract_lineage_exact") != 1:
        raise FullSceneProbeError("fresh full-scene probe checkpoint lineage must equal 1")
    tensor_audit = queue_runtime._tensor_finiteness(checkpoint, torch_module)
    if tensor_audit["floating_tensor_count"] <= 0:
        raise FullSceneProbeError("probe checkpoint contains no floating tensors")
    if tensor_audit["nonfinite_floating_elements"] != 0:
        raise FullSceneProbeError("probe checkpoint contains non-finite floating tensors")
    checkpoint_raw = queue_runtime._read_regular_bytes(checkpoint_path, "probe checkpoint")
    if queue_runtime._checkpoint_stat(checkpoint_path) != before:
        raise FullSceneProbeError("probe checkpoint changed while hashing")
    if queue_runtime._read_regular_bytes(hard_path, "hard training contract") != hard_raw:
        raise FullSceneProbeError("probe hard contract changed during finalization")

    return {
        "schema_version": 1,
        "purpose": PROBE_PURPOSE,
        "status": "passed",
        "unlock_authorized": True,
        "not_science": True,
        "attestable": False,
        "promotable": False,
        "run_dir": str(run_dir),
        "claim_path": str(run_dir / CLAIM_NAME),
        "claim_content_sha256": claim_digest,
        "binding_path": str(run_dir / BINDING_NAME),
        "binding_content_sha256": binding_digest,
        "exit_receipt_path": str(run_dir / EXIT_NAME),
        "exit_receipt_content_sha256": exit_receipt["content_sha256"],
        "source_state_at_finalization": source_state,
        "source_asset_receipt": source_asset_evidence,
        "isolated_process_group_empty": True,
        "log": log_evidence,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
            "filename_iteration": 1,
            "embedded_iteration": 1,
            **tensor_audit,
        },
        "hard_contract": {
            "path": str(hard_path),
            "schema_version": 3,
            "sha256": hard_sha,
        },
        "assets": asset_evidence,
    }


def _terminal_failure_evidence(run_dir: Path) -> dict[str, Any]:
    """Preserve immutable byte identities even when semantic validation fails."""

    evidence: dict[str, Any] = {}
    for key, name in (
        ("exit_receipt", EXIT_NAME),
        ("binding", BINDING_NAME),
        ("log", "run.log"),
    ):
        path = run_dir / name
        if not path.exists():
            evidence[key] = {"path": str(path), "present": False}
            continue
        try:
            raw = queue_runtime._read_regular_bytes(path, f"failed probe {key}")
        except queue_runtime.LeanQueueRuntimeError as exc:
            evidence[key] = {
                "path": str(path),
                "present": True,
                "read_error": str(exc),
            }
            continue
        item: dict[str, Any] = {
            "path": str(path),
            "present": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if key == "exit_receipt":
            try:
                envelope = json.loads(raw.decode("utf-8"))
                content = envelope.get("content", {})
                item["termination"] = content.get("termination")
                item["first_iteration_observed"] = content.get(
                    "first_iteration_observed"
                )
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                item["parseable_json"] = False
        evidence[key] = item
    return evidence


def finalize(
    run_dir: str | Path,
    *,
    expected_claim_digest: str,
    source_asset_receipt: str | Path,
    checkpoint_loader: Callable[[Path], Any] | None = None,
    torch_module: Any | None = None,
    proc_root: str | Path = "/proc",
    getpgid: Callable[[int], int] = os.getpgid,
    source_verifier: Callable[[Path, str], Mapping[str, Any]] = queue_runtime._verify_git_source,
) -> dict[str, Any]:
    """Write one deterministic pass/fail receipt after natural probe termination."""

    run_dir_obj = _canonical_absolute(run_dir, "probe run_dir")
    _claim, _content, claim_digest = _probe_claim(
        run_dir_obj, expected_digest=expected_claim_digest
    )
    source_asset_receipt_obj = _canonical_absolute(
        source_asset_receipt, "source asset receipt"
    )
    if not (run_dir_obj / EXIT_NAME).exists():
        raise FullSceneProbeNotReady("probe exit receipt is not present")
    try:
        content = _validate_terminal(
            run_dir_obj,
            expected_claim_digest=expected_claim_digest,
            source_asset_receipt=source_asset_receipt_obj,
            checkpoint_loader=checkpoint_loader,
            torch_module=torch_module,
            proc_root=Path(proc_root),
            getpgid=getpgid,
            source_verifier=source_verifier,
        )
    except FullSceneProbeNotReady:
        raise
    except Exception as exc:
        content = {
            "schema_version": 1,
            "purpose": PROBE_PURPOSE,
            "status": "failed",
            "unlock_authorized": False,
            "not_science": True,
            "attestable": False,
            "promotable": False,
            "run_dir": str(run_dir_obj),
            "claim_path": str(run_dir_obj / CLAIM_NAME),
            "claim_content_sha256": claim_digest,
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "automatic_retry_authorized": False,
            "terminal_evidence": _terminal_failure_evidence(run_dir_obj),
        }
    result = {
        "schema_version": 1,
        "content": content,
        "content_sha256": queue_runtime.canonical_sha256(content),
    }
    repeated = _publish_or_accept_identical(
        run_dir_obj / RESULT_NAME, result, "probe terminal result"
    )
    return {
        "result_path": str(run_dir_obj / RESULT_NAME),
        "result": result,
        "repeated_identical": repeated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    supervise_parser = commands.add_parser("supervise")
    supervise_parser.add_argument("--run-dir", type=Path, required=True)
    supervise_parser.add_argument("--log", type=Path, required=True)
    supervise_parser.add_argument("trainer_argv", nargs=argparse.REMAINDER)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    finalize_parser.add_argument("--expected-claim-sha256", required=True)
    finalize_parser.add_argument("--source-asset-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "supervise":
            trainer_argv = list(args.trainer_argv)
            if trainer_argv and trainer_argv[0] == "--":
                trainer_argv = trainer_argv[1:]
            result = supervise(
                run_dir=args.run_dir,
                log_path=args.log,
                trainer_argv=trainer_argv,
            )
            termination = result["content"]["termination"]
            if termination["kind"] == "signal":
                return 128 + int(termination["signal"])
            if termination["kind"] == "spawn_error":
                return 127
            return int(termination["exit_code"])
        result = finalize(
            args.run_dir,
            expected_claim_digest=args.expected_claim_sha256,
            source_asset_receipt=args.source_asset_receipt,
        )
    except FullSceneProbeNotReady as exc:
        print(f"NOT_READY: {exc}", file=sys.stderr)
        return 3
    except (FullSceneProbeError, queue_runtime.LeanQueueRuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    # Exit zero means the immutable terminal classification was published, not
    # that the probe passed.  Consumers must read ``status`` and
    # ``unlock_authorized``; a failed receipt remains a terminal, no-retry fact.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
