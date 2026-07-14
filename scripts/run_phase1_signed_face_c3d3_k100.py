#!/usr/bin/env python3
"""Run one attested C3/D3 pair on the immutable signed-face K100 paper.

``static-validate`` and ``source-plan`` are repository-only checks. ``plan``
validates two explicit generic-attestor requests but reads no checkpoint or
attestation runtime artifact. ``execute`` requires both completed attestation
claims, replays every runtime binding, creates one no-replace evaluation root,
and runs C3 then D3 on the same 100-question paper.  It never signals an
existing process and never authorizes L2, a second seed, stop/promote, Gate3,
deployment, or robot work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


MANIFEST_RELATIVE = Path("configs/phase1_signed_face_c3d3_k100_execution_20260714.json")
MANIFEST_ID = "phase1-signed-face-c3-d3-k100-paired-execution-20260714-v1"
REQUEST_STATUS = "exact_c3_d3_signed_k100_pair_execution_requested_not_started"
CELL_ORDER = ("C3", "D3")
CHECKPOINT_SHA_BY_CELL = {
    "C3": "6b3e2cb17a0cfcc299b3f32a13814f4c7d34292bdd4d3d7ef73f5cb2d3bd70e7",
    "D3": "44c6117c8a2449140da056a9a03bfcce4a9e87a971f2549b9528b424a80f85b8",
}
HARD_SHA_BY_CELL = {
    "C3": "d76dc9440b0a9f58da71a40425e2eb6a6f003097ebfcd50bf3fec8d4d4e5ef2c",
    "D3": "98f6468f19ba7b57e538db1fdccfff41726eed73aa30cc810f2edd35efc734f4",
}
CLAIM_SHA_BY_CELL = {
    "C3": "aa240e2fe2ef9cef5cc5d8676d5c42a77d0d7764bd8d21710dba4b0e04b8a6d8",
    "D3": "7a1970d217bb7fd5b36daeb2cbbd789237ee4a2606300c954e01acc714e556d2",
}
TERMINAL_SHA_BY_CELL = {
    "C3": "8c579386df56f9b49863c9b879f2e45109ba392f3e0a66d55a56d6a08b00e8ef",
    "D3": "ccb9933c057d15079015d13a2b0a97589902650842369aaaad025e35fa367f0e",
}
PAIR_SHA = "bb3cd749477861b1cd55f059ed3b23307784030dcad758db3a819c3c8a37bbde"
ROOT_CONFIRMATION = "ROOT_APPROVES_SIM_ONLY_C3_D3_SIGNED_K100_PAIRED_EXECUTION_V1"

REQUEST_AUTHORIZATION = {
    "auto_start": False,
    "trainer_started": False,
    "training_mutation_allowed": False,
    "c3d3_rerun_allowed": False,
    "paired_k100_judge_requested": True,
    "l2_training_authorized": False,
    "second_seed_authorized": False,
    "checkpoint_stop_or_promote_authorized": False,
    "formal_setting_adoption_authorized": False,
    "gate3_authorized": False,
    "deployment_authorized": False,
    "real_robot_authorized": False,
    "signals_to_existing_processes_allowed": False,
}

MANIFEST_AUTHORIZATION = {
    "auto_start": False,
    "trainer_started": False,
    "training_mutation_allowed": False,
    "c3d3_rerun_allowed": False,
    "paired_k100_judge_allowed_only_after_exact_attestations": True,
    "l2_training_authorized": False,
    "second_seed_authorized": False,
    "checkpoint_stop_or_promote_authorized": False,
    "formal_setting_adoption_authorized": False,
    "gate3_authorized": False,
    "deployment_authorized": False,
    "real_robot_authorized": False,
    "signals_to_existing_processes_allowed": False,
}

RUNTIME_FACE_CONTRACT = {
    "schema_version": 1,
    "achieved_and_target_frame": "mount_plusY_A",
    "external_frame": "physical_striking_face_B",
    "physical_B_to_raw_A": "raw_A=mount_normal_sign_per_clip[clip]*physical_B",
    "mount_normal_sign_per_clip": [1.0, -1.0],
    "signed_face_required": True,
    "physical_B_min_x_strict": 1.0e-6,
    "unit_normal_atol": 2.0e-4,
    "identity_gate": (
        "dot(achieved_raw_A,target_raw_A)>0_and_achieved_physical_B.x>1e-6_and_"
        "target_physical_B.x>1e-6_before_orient_normal"
    ),
    "wrong_orthogonal_or_non_opponent_facing_face": "contacted=false",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parent
ATTESTOR_PATH = SCRIPT_DIR / "attest_phase1_signed_face_k100_checkpoint.py"
ATTESTOR_SHA256 = "b42ecd08ea1516ab50cda0d47ee957b82e7c1e5b0c19fc3f7f588862ed7c5ec3"


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not ATTESTOR_PATH.is_file() or _bootstrap_sha256(ATTESTOR_PATH) != ATTESTOR_SHA256:
    raise RuntimeError("generic signed-K100 checkpoint attestor bytes changed")
_attestor_spec = importlib.util.spec_from_file_location("signed_k100_attestor_bound", ATTESTOR_PATH)
if _attestor_spec is None or _attestor_spec.loader is None:
    raise RuntimeError("cannot import the bound generic checkpoint attestor")
A = importlib.util.module_from_spec(_attestor_spec)
sys.modules[_attestor_spec.name] = A
_attestor_spec.loader.exec_module(A)

ContractError = A.ContractError
canonical_sha256 = A.canonical_sha256
exact_keys = A.exact_keys
read_json = A.read_json
require_absolute = A.require_absolute
require_exact = A.require_exact
require_plain_int = A.require_plain_int
require_sha = A.require_sha
sha256_file = A.sha256_file


def _repo_file(repo_root: Path, spec: Mapping[str, Any], label: str) -> Path:
    exact_keys(spec, {"path", "sha256"}, label)
    relative = A.require_repo_relative(spec["path"], f"{label}.path")
    unresolved = repo_root / relative
    path = unresolved.resolve()
    if path != unresolved or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be one regular repository file")
    require_sha(spec["sha256"], f"{label}.sha256")
    if sha256_file(path) != spec["sha256"]:
        raise ContractError(f"{label} bytes changed")
    return path


def load_manifest(path: Path, *, repo_root: Path) -> dict[str, Any]:
    expected = (repo_root / MANIFEST_RELATIVE).resolve()
    if path.resolve() != expected or path.is_symlink() or path != path.resolve():
        raise ContractError("paired execution manifest must be the canonical tracked file")
    manifest = read_json(path, "C3/D3 K100 execution manifest")
    exact_keys(
        manifest,
        {
            "schema_version", "manifest_id", "status", "recorded_local_date",
            "human_owner", "executor", "simulation_only", "real_robot_commands_forbidden",
            "source_bindings", "paired_l1_receipt", "cells", "paper", "execution",
            "authorization",
        },
        "execution manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["manifest_id"] != MANIFEST_ID
        or manifest["status"] != "source_reviewed_runtime_request_and_attestations_required"
        or manifest["recorded_local_date"] != "2026-07-14"
        or manifest["human_owner"] != "Franco"
        or manifest["executor"] != "Codex"
        or manifest["simulation_only"] is not True
        or manifest["real_robot_commands_forbidden"] is not True
    ):
        raise ContractError("execution manifest identity/status/safety changed")
    require_exact(manifest["authorization"], MANIFEST_AUTHORIZATION, "manifest authorization")
    sources = exact_keys(
        manifest["source_bindings"],
        {
            "runner", "checkpoint_attestor", "checkpoint_attestor_manifest",
            "c3d3_l1_manifest", "c3d3_l1_finalizer",
        },
        "source_bindings",
    )
    resolved = {name: _repo_file(repo_root, spec, f"source binding {name}") for name, spec in sources.items()}
    if resolved["runner"] != Path(__file__).resolve():
        raise ContractError("manifest runner does not resolve to this script")
    if resolved["checkpoint_attestor"] != ATTESTOR_PATH:
        raise ContractError("manifest checkpoint attestor path changed")
    attestor_manifest = A.load_manifest(
        resolved["checkpoint_attestor_manifest"], repo_root=repo_root
    )

    pair = exact_keys(
        manifest["paired_l1_receipt"],
        {"path", "sha256", "ordered_cells", "only_hard_contract_difference"},
        "paired_l1_receipt",
    )
    require_absolute(pair["path"], "paired receipt path")
    require_exact(pair["sha256"], PAIR_SHA, "paired receipt SHA")
    require_exact(pair["ordered_cells"], list(CELL_ORDER), "paired receipt order")
    require_exact(
        pair["only_hard_contract_difference"],
        "racket_guidance_reward.signed_face.weight",
        "paired receipt causal difference",
    )
    cells = exact_keys(manifest["cells"], set(CELL_ORDER), "cells")
    expected_weights = {"C3": 0.0, "D3": -0.4}
    for cell_id in CELL_ORDER:
        cell = exact_keys(
            cells[cell_id],
            {
                "human_name", "run_name", "face_guidance_weight", "checkpoint",
                "adjacent_hard_contract", "producer_claim_canonical_sha256", "terminal_receipt",
            },
            f"cells.{cell_id}",
        )
        if type(cell["human_name"]) is not str or type(cell["run_name"]) is not str:
            raise ContractError(f"{cell_id} identity must be strings")
        require_exact(cell["face_guidance_weight"], expected_weights[cell_id], f"{cell_id} weight")
        checkpoint = exact_keys(cell["checkpoint"], {"path", "sha256", "iteration"}, f"{cell_id} checkpoint")
        checkpoint_path = require_absolute(checkpoint["path"], f"{cell_id} checkpoint path")
        require_exact(checkpoint["sha256"], CHECKPOINT_SHA_BY_CELL[cell_id], f"{cell_id} checkpoint SHA")
        require_exact(checkpoint["iteration"], 24, f"{cell_id} checkpoint iteration")
        if checkpoint_path.name != "model_24.pt":
            raise ContractError(f"{cell_id} checkpoint filename changed")
        hard = exact_keys(cell["adjacent_hard_contract"], {"path", "sha256"}, f"{cell_id} hard contract")
        hard_path = require_absolute(hard["path"], f"{cell_id} hard path")
        if hard_path != checkpoint_path.parent / "params" / "training_contract.json":
            raise ContractError(f"{cell_id} hard contract is not checkpoint-adjacent")
        require_exact(hard["sha256"], HARD_SHA_BY_CELL[cell_id], f"{cell_id} hard SHA")
        require_exact(
            cell["producer_claim_canonical_sha256"], CLAIM_SHA_BY_CELL[cell_id],
            f"{cell_id} producer claim SHA",
        )
        terminal = exact_keys(cell["terminal_receipt"], {"path", "sha256"}, f"{cell_id} terminal receipt")
        require_absolute(terminal["path"], f"{cell_id} terminal receipt path")
        require_exact(terminal["sha256"], TERMINAL_SHA_BY_CELL[cell_id], f"{cell_id} terminal SHA")

    paper = exact_keys(
        manifest["paper"], {"schedule", "activation", "denominator", "mount_normal_sign_per_clip"}, "paper"
    )
    exact_keys(
        paper["schedule"],
        {"path", "bytes", "file_sha256", "semantic_sha256", "question_id_order_sha256"},
        "paper schedule",
    )
    exact_keys(
        paper["activation"],
        {"path", "bytes", "file_sha256", "content_sha256"},
        "paper activation",
    )
    for key, value in paper["schedule"].items():
        require_exact(value, attestor_manifest["paper"]["schedule"][key], f"paper schedule {key}")
    for key, value in paper["activation"].items():
        require_exact(value, attestor_manifest["paper"]["activation"][key], f"paper activation {key}")
    require_exact(paper["denominator"], {"aggregate": 100, "forehand": 50, "backhand": 50}, "paper denominator")
    require_exact(paper["mount_normal_sign_per_clip"], [1.0, -1.0], "paper face signs")
    execution = exact_keys(
        manifest["execution"],
        {
            "ordered_cells", "schedule_seed", "noise_scales", "hold_range", "qdes_clamp",
            "allow_inexact_contract", "one_question_reset", "same_schedule_activation_required",
            "output_root", "root_confirmation", "result_policy",
        },
        "execution",
    )
    require_exact(execution["ordered_cells"], list(CELL_ORDER), "execution order")
    require_exact(execution["schedule_seed"], 0, "schedule seed")
    require_exact(execution["noise_scales"], [0.0], "noise scales")
    require_exact(execution["hold_range"], [0, 100], "hold range")
    require_exact(execution["qdes_clamp"], True, "qdes clamp")
    require_exact(execution["allow_inexact_contract"], False, "inexact escape")
    require_exact(execution["one_question_reset"], True, "one-question reset")
    require_exact(execution["same_schedule_activation_required"], True, "paired paper")
    require_absolute(execution["output_root"], "execution output root")
    require_exact(execution["root_confirmation"], ROOT_CONFIRMATION, "root confirmation")
    require_exact(
        execution["result_policy"],
        "publish_exact_paired_counts_and_delta_without_authorizing_l2_or_any_training_action",
        "result policy",
    )
    manifest["_resolved_sources"] = resolved
    manifest["_attestor_manifest"] = attestor_manifest
    manifest["_repo_root"] = repo_root
    return manifest


def _file_spec(value: Any, label: str, *, content_sha: bool = False) -> dict[str, Any]:
    keys = {"path", "bytes", "file_sha256"}
    if content_sha:
        keys.add("content_sha256")
    spec = exact_keys(value, keys, label)
    require_absolute(spec["path"], f"{label}.path")
    require_plain_int(spec["bytes"], f"{label}.bytes", minimum=1)
    require_sha(spec["file_sha256"], f"{label}.file_sha256")
    if content_sha:
        require_sha(spec["content_sha256"], f"{label}.content_sha256")
    return spec


def load_request(
    path: Path, manifest_path: Path, manifest: Mapping[str, Any], *, read_attestor_requests: bool
) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ContractError("paired execution request must be a canonical absolute non-symlink file")
    request = read_json(path, "C3/D3 paired execution request")
    exact_keys(
        request,
        {
            "schema_version", "request_id", "status", "human_owner", "executor",
            "simulation_only", "real_robot_commands_forbidden", "manifest",
            "attestor_requests", "attestations", "env_yamls", "runtime", "output",
            "authorization",
        },
        "paired execution request",
    )
    if (
        request["schema_version"] != 1
        or request["status"] != REQUEST_STATUS
        or type(request["request_id"]) is not str
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{7,127}", request["request_id"])
        or request["human_owner"] != "Franco"
        or request["executor"] != "Codex"
        or request["simulation_only"] is not True
        or request["real_robot_commands_forbidden"] is not True
    ):
        raise ContractError("paired execution request identity/status/safety changed")
    require_exact(request["authorization"], REQUEST_AUTHORIZATION, "request authorization")
    manifest_spec = _file_spec(request["manifest"], "request manifest")
    if Path(manifest_spec["path"]) != manifest_path:
        raise ContractError("request points to a foreign execution manifest")
    if manifest_spec["file_sha256"] != sha256_file(manifest_path) or manifest_spec["bytes"] != manifest_path.stat().st_size:
        raise ContractError("request execution manifest bytes changed")

    attestor_requests = exact_keys(request["attestor_requests"], set(CELL_ORDER), "attestor_requests")
    attestations = exact_keys(request["attestations"], set(CELL_ORDER), "attestations")
    env_yamls = exact_keys(request["env_yamls"], set(CELL_ORDER), "env_yamls")
    loaded_attestor_requests: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_ORDER:
        request_spec = _file_spec(attestor_requests[cell_id], f"attestor_requests.{cell_id}", content_sha=True)
        request_path = Path(request_spec["path"])
        cell = manifest["cells"][cell_id]
        expected_attestation_root = Path(manifest["_attestor_manifest"]["output"]["root"]) / cell["checkpoint"]["sha256"]
        attestation = exact_keys(attestations[cell_id], {"claim", "evidence"}, f"attestations.{cell_id}")
        claim = _file_spec(attestation["claim"], f"attestations.{cell_id}.claim", content_sha=True)
        evidence = _file_spec(attestation["evidence"], f"attestations.{cell_id}.evidence", content_sha=True)
        if Path(claim["path"]) != expected_attestation_root / manifest["_attestor_manifest"]["output"]["claim_basename"]:
            raise ContractError(f"{cell_id} attestation claim path escaped checkpoint-SHA namespace")
        if Path(evidence["path"]) != expected_attestation_root / manifest["_attestor_manifest"]["output"]["evidence_basename"]:
            raise ContractError(f"{cell_id} attestation evidence path escaped checkpoint-SHA namespace")
        env = _file_spec(env_yamls[cell_id], f"env_yamls.{cell_id}")
        expected_env = Path(cell["checkpoint"]["path"]).parent / "params" / "env.yaml"
        if Path(env["path"]) != expected_env:
            raise ContractError(f"{cell_id} env.yaml is not checkpoint-adjacent")
        if read_attestor_requests:
            if (
                not request_path.is_file()
                or request_path.stat().st_size != request_spec["bytes"]
                or sha256_file(request_path) != request_spec["file_sha256"]
            ):
                raise ContractError(f"{cell_id} attestor request bytes changed")
            raw = read_json(request_path, f"{cell_id} attestor request")
            if canonical_sha256(raw) != request_spec["content_sha256"]:
                raise ContractError(f"{cell_id} attestor request canonical SHA changed")
            value = A.load_request(request_path, manifest["_attestor_manifest"], runtime=False)
            checkpoint = value["checkpoint"]
            hard = value["adjacent_hard_contract"]
            if (
                checkpoint["path"] != cell["checkpoint"]["path"]
                or checkpoint["sha256"] != cell["checkpoint"]["sha256"]
                or checkpoint["filename_iteration"] != 24
                or checkpoint["embedded_iteration"] != 24
                or checkpoint["training_contract_sha256"] != cell["adjacent_hard_contract"]["sha256"]
                or checkpoint["training_contract_lineage_exact"] != 1
                or checkpoint["producer_claim"]["canonical_sha256"] != cell["producer_claim_canonical_sha256"]
                or hard["path"] != cell["adjacent_hard_contract"]["path"]
                or hard["sha256"] != cell["adjacent_hard_contract"]["sha256"]
                or value["output"]["root"] != str(expected_attestation_root)
            ):
                raise ContractError(f"{cell_id} attestor request is not the frozen terminal checkpoint")
            loaded_attestor_requests[cell_id] = value

    runtime = exact_keys(
        request["runtime"],
        {"gpu_by_cell", "isaac_activation", "mjeval_activation", "kit_boot_lock"},
        "runtime",
    )
    gpu_by_cell = exact_keys(runtime["gpu_by_cell"], set(CELL_ORDER), "runtime.gpu_by_cell")
    gpus = []
    for cell_id in CELL_ORDER:
        gpu = require_plain_int(gpu_by_cell[cell_id], f"runtime.gpu_by_cell.{cell_id}")
        gpus.append(gpu)
    if len(set(gpus)) != 2:
        raise ContractError("C3/D3 evaluation GPU lanes must be distinct")
    for name in ("isaac_activation", "mjeval_activation"):
        activation = _file_spec(runtime[name], f"runtime.{name}")
        if Path(activation["path"]).name != "activate":
            raise ContractError(f"runtime.{name} must bind one venv activate script")
    require_absolute(runtime["kit_boot_lock"], "runtime.kit_boot_lock")
    if read_attestor_requests:
        isaac_python = Path(loaded_attestor_requests["C3"]["runtime"]["checkpoint_python"]["path"])
        eval_python = Path(loaded_attestor_requests["C3"]["runtime"]["evaluator_python"]["path"])
        for cell_id in CELL_ORDER:
            other = loaded_attestor_requests[cell_id]["runtime"]
            if (
                Path(other["checkpoint_python"]["path"]) != isaac_python
                or Path(other["evaluator_python"]["path"]) != eval_python
            ):
                raise ContractError("C3/D3 attestor requests must use the same two Python runtimes")
            if Path(loaded_attestor_requests[cell_id]["source_checkout"]["path"]) != manifest["_repo_root"]:
                raise ContractError(f"{cell_id} attestor request must use this independent eval worktree")
        require_exact(
            loaded_attestor_requests["C3"]["source_checkout"],
            loaded_attestor_requests["D3"]["source_checkout"],
            "paired independent eval source checkout",
        )
        if Path(runtime["isaac_activation"]["path"]).parent != isaac_python.parent:
            raise ContractError("Isaac activation does not match attested checkpoint Python")
        if Path(runtime["mjeval_activation"]["path"]).parent != eval_python.parent:
            raise ContractError("MuJoCo activation does not match attested evaluator Python")

    output = exact_keys(request["output"], {"root"}, "request output")
    output_root = require_absolute(output["root"], "request output root")
    if output_root != Path(manifest["execution"]["output_root"]):
        raise ContractError("request output root changed from the one-shot paired namespace")
    request["_attestor_requests"] = loaded_attestor_requests
    return request


def _validate_bound_json(spec: Mapping[str, Any], label: str) -> tuple[dict[str, Any], os.stat_result]:
    path = Path(spec["path"])
    stat = A.validate_file_binding(path, size=spec["bytes"], digest=spec["file_sha256"], label=label)
    value = read_json(path, label)
    return value, stat


def _validate_terminal_receipts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pair_path = Path(manifest["paired_l1_receipt"]["path"])
    if not pair_path.is_file() or pair_path.is_symlink() or sha256_file(pair_path) != PAIR_SHA:
        raise ContractError("paired L1 receipt missing or changed")
    pair = read_json(pair_path, "paired L1 receipt")
    expected_pair_fields = {
        "artifact_kind": "phase1_signed_face_c3_d3_l1_paired_provenance_result",
        "schema_version": 1,
        "ordered_cells": list(CELL_ORDER),
        "only_hard_contract_difference": "racket_guidance_reward.signed_face.weight",
        "hard_contract_sha256_by_cell": HARD_SHA_BY_CELL,
        "terminal_checkpoint_sha256_by_cell": CHECKPOINT_SHA_BY_CELL,
        "terminal_result_sha256_by_cell": TERMINAL_SHA_BY_CELL,
        "both_model_24_finite_iter24_lineage1": True,
        "checkpoint_binds_adjacent_hard_contract_and_outer_claim_source": True,
        "activation": False,
        "judge": False,
        "l2": False,
        "second_seed": False,
        "stop_or_promote": False,
        "same_immutable_signed_paper_still_required": True,
        "real_robot_commands_executed": False,
    }
    for key, expected in expected_pair_fields.items():
        require_exact(pair.get(key), expected, f"paired L1 receipt {key}")
    terminals = {}
    for cell_id in CELL_ORDER:
        cell = manifest["cells"][cell_id]
        terminal_spec = cell["terminal_receipt"]
        path = Path(terminal_spec["path"])
        if not path.is_file() or path.is_symlink() or sha256_file(path) != terminal_spec["sha256"]:
            raise ContractError(f"{cell_id} terminal receipt missing or changed")
        terminal = read_json(path, f"{cell_id} terminal receipt")
        checks = {
            "artifact_kind": "phase1_signed_face_c3_d3_l1_terminal_result",
            "schema_version": 1,
            "cell_id": cell_id,
            "run_name": cell["run_name"],
            "training_launch_claim_sha256": cell["producer_claim_canonical_sha256"],
            "hard_contract_path": cell["adjacent_hard_contract"]["path"],
            "hard_contract_sha256": cell["adjacent_hard_contract"]["sha256"],
            "terminal_checkpoint_path": cell["checkpoint"]["path"],
            "terminal_checkpoint_sha256": cell["checkpoint"]["sha256"],
            "exact_trainer_natural_exit_observed": True,
            "gpu_empty_terminal_barrier_observed": True,
            "activation": False,
            "judge": False,
            "l2": False,
            "second_seed": False,
            "stop_or_promote": False,
            "real_robot_commands_executed": False,
        }
        for key, expected in checks.items():
            require_exact(terminal.get(key), expected, f"{cell_id} terminal receipt {key}")
        audit = terminal.get("checkpoint_audit", {})
        for key, expected in {
            "iter": 24,
            "training_contract_schema_version": 3,
            "training_contract_sha256": cell["adjacent_hard_contract"]["sha256"],
            "training_contract_lineage_exact": 1,
            "training_launch_claim_sha256": cell["producer_claim_canonical_sha256"],
            "nonfinite_floating_elements": 0,
        }.items():
            require_exact(audit.get(key), expected, f"{cell_id} terminal checkpoint audit {key}")
        terminals[cell_id] = terminal
    return {"pair": pair, "terminals": terminals}


def _validate_attestation(
    cell_id: str, request: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    attestor_request = request["_attestor_requests"][cell_id]
    runtime = A.validate_runtime_request(attestor_request, manifest["_attestor_manifest"])
    specs = request["attestations"][cell_id]
    claim, _ = _validate_bound_json(specs["claim"], f"{cell_id} attestation claim")
    evidence, _ = _validate_bound_json(specs["evidence"], f"{cell_id} attestation evidence")
    exact_keys(claim, {"schema_version", "artifact_kind", "content_sha256", "content"}, f"{cell_id} claim wrapper")
    exact_keys(evidence, {"schema_version", "artifact_kind", "content_sha256", "content"}, f"{cell_id} evidence wrapper")
    if (
        claim["schema_version"] != 1
        or claim["artifact_kind"] != A.CLAIM_KIND
        or claim["content_sha256"] != specs["claim"]["content_sha256"]
        or claim["content_sha256"] != canonical_sha256(claim["content"])
        or evidence["schema_version"] != 1
        or evidence["artifact_kind"] != A.EVIDENCE_KIND
        or evidence["content_sha256"] != specs["evidence"]["content_sha256"]
        or evidence["content_sha256"] != canonical_sha256(evidence["content"])
    ):
        raise ContractError(f"{cell_id} attestation wrapper/content binding changed")
    cell = manifest["cells"][cell_id]
    request_spec = request["attestor_requests"][cell_id]
    evidence_content = evidence["content"]
    attestor_manifest_path = manifest["_resolved_sources"]["checkpoint_attestor_manifest"]
    attestor_runner_path = manifest["_resolved_sources"]["checkpoint_attestor"]
    evidence_expected = {
        "manifest_id": A.MANIFEST_ID,
        "manifest_sha256": sha256_file(attestor_manifest_path),
        "runner_sha256": sha256_file(attestor_runner_path),
        "request_id": attestor_request["request_id"],
        "request_file_sha256": request_spec["file_sha256"],
        "request_canonical_sha256": request_spec["content_sha256"],
        "status": "exact_checkpoint_inputs_attested_judge_not_started",
        "checkpoint": attestor_request["checkpoint"],
        "adjacent_hard_contract": attestor_request["adjacent_hard_contract"],
        "source_checkout": runtime["source_checkout"],
        "checkpoint_audit": runtime["checkpoint_audit"],
        "producer_claim": runtime["producer_claim"],
        "runtime": {
            "checkpoint_python": runtime["checkpoint_python"],
            "evaluator_python": runtime["evaluator_python"],
        },
        "mjcf": runtime["mjcf"],
        "plant_contract_sha256": runtime["plant_contract_sha256"],
        "plant_execution": manifest["_attestor_manifest"]["execution_semantics"]["plant_execution"],
        "paper": runtime["paper"],
        "receipt_correction": manifest["_attestor_manifest"]["receipt_correction"],
        "signed_face_contract": A.EXPECTED_FACE_CONTRACT,
        "evaluation_contract_exact": True,
        "signed_face_exact": True,
        "authorization": A.ATTESTATION_AUTHORIZATION,
    }
    require_exact(evidence_content, evidence_expected, f"{cell_id} complete generic evidence")
    claim_content = claim["content"]
    claim_expected = {
        "manifest_id": A.MANIFEST_ID,
        "request_id": attestor_request["request_id"],
        "checkpoint_sha256": cell["checkpoint"]["sha256"],
        "checkpoint_iteration": 24,
        "training_contract_sha256": cell["adjacent_hard_contract"]["sha256"],
        "producer_claim_canonical_sha256": cell["producer_claim_canonical_sha256"],
        "plant_contract_sha256": attestor_request["adjacent_hard_contract"]["plant_contract_sha256"],
        "mjcf_sha256": attestor_request["mjcf"]["sha256"],
        "schedule_file_sha256": manifest["paper"]["schedule"]["file_sha256"],
        "schedule_semantic_sha256": manifest["paper"]["schedule"]["semantic_sha256"],
        "schedule_question_id_order_sha256": manifest["paper"]["schedule"]["question_id_order_sha256"],
        "activation_file_sha256": manifest["paper"]["activation"]["file_sha256"],
        "activation_content_sha256": manifest["paper"]["activation"]["content_sha256"],
        "evidence_path": specs["evidence"]["path"],
        "evidence_file_sha256": specs["evidence"]["file_sha256"],
        "evidence_content_sha256": specs["evidence"]["content_sha256"],
        "status": "attested_not_executed_no_decision",
        "judge_started": False,
        "stop_or_promote_authorized": False,
        "real_robot_authorized": False,
    }
    require_exact(claim_content, claim_expected, f"{cell_id} complete generic claim")
    return {"runtime": runtime, "claim": claim, "evidence": evidence}


def _strict_csv_bool(value: str, label: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ContractError(f"{label} is not an exact CSV boolean")


def _finite_rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite rate")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractError(f"{label} must be in [0,1]")
    return result


def validate_runtime_face_contract(value: Any) -> None:
    require_exact(value, RUNTIME_FACE_CONTRACT, "runtime signed-face float contract")


def _report_from_log(log_path: Path, staged_run: Path) -> Path:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"^\[judge\] 报告: (/.+)$", text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ContractError("judge log does not contain exactly one report path")
    report = Path(matches[0])
    expected_parent = staged_run / "judge"
    if report.parent != expected_parent or not report.is_file() or report.is_symlink():
        raise ContractError("judge report escaped the independent evaluation run")
    return report


def validate_exam_result(
    *, cell_id: str, report: Path, staged_run: Path, manifest: Mapping[str, Any],
    attestor_request: Mapping[str, Any], schedule_document: Mapping[str, Any]
) -> dict[str, Any]:
    match = re.fullmatch(r"judge_report_model_24_(\d{8}_\d{6})\.md", report.name)
    if match is None or report.parent != staged_run / "judge":
        raise ContractError(f"{cell_id} report path/name changed")
    exam_dir = report.parent / f"model_24_{match.group(1)}" / "exam"
    summary_path = exam_dir / "mujoco_sim2sim_summary.json"
    attempts_path = exam_dir / "mujoco_sim2sim_attempts.csv"
    if not summary_path.is_file() or not attempts_path.is_file():
        raise ContractError(f"{cell_id} judge lacks summary or unconditional attempt ledger")
    summary = read_json(summary_path, f"{cell_id} K100 summary")
    schedule_module = A._load_module(
        manifest["_attestor_manifest"]["_resolved_sources"]["schedule_module"],
        f"c3d3_k100_schedule_{cell_id}",
    )
    artifact = schedule_module._artifact_from_document(schedule_document)
    expected_items = [
        {
            "schedule_index": item.schedule_index,
            "clip": item.clip,
            "bank_row": item.bank_row,
            "question_id": item.question_id,
            "repeat": item.repeat,
            "hold_steps": item.hold_steps,
            "attempt_seed": item.attempt_seed,
        }
        for item in artifact.items
    ]
    order = [item["question_id"] for item in expected_items]
    arguments = summary.get("arguments", {})
    input_artifacts = summary.get("input_artifacts", {})
    schedule_runtime = summary.get("exam_schedule", {})
    scorer = summary.get("virtual_return_scorer_contract", {})
    scorer_material = dict(scorer) if isinstance(scorer, dict) else {}
    scorer_sha = scorer_material.pop("sha256", None)
    scorer_source_binding = manifest["_attestor_manifest"]["source_bindings"]["signed_face_scorer"]
    scorer_source = {
        "repo_relative_path": scorer_source_binding["path"],
        "sha256": scorer_source_binding["sha256"],
    }
    validate_runtime_face_contract(scorer.get("signed_face_contract"))
    if (
        summary.get("schema_version") != 3
        or summary.get("evaluation_contract_exact") is not True
        or arguments.get("allow_inexact_contract") is not False
        or arguments.get("target_source") != "bank"
        or arguments.get("seed") != 0
        or arguments.get("noise_scales") != [0.0]
        or arguments.get("qdes_clamp") is not True
        or arguments.get("hold_ref") != "auto"
        or arguments.get("exam_continuity_diagnostic") is not False
        or arguments.get("exam_schedule_k") is not None
        or os.path.abspath(str(arguments.get("exam_schedule_json"))) != manifest["paper"]["schedule"]["path"]
        or os.path.abspath(str(arguments.get("exam_bank"))) != attestor_request["_actual_bank_path"]
        or input_artifacts.get("exam_bank", {}).get("sha256") != manifest["_attestor_manifest"]["paper"]["bank"]["sha256"]
        or input_artifacts.get("exam_schedule_artifact", {}).get("sha256") != manifest["paper"]["schedule"]["file_sha256"]
        or input_artifacts.get("exam_schedule_artifact", {}).get("schedule_sha256") != manifest["paper"]["schedule"]["semantic_sha256"]
        or summary.get("mjcf_sha256") != attestor_request["mjcf"]["sha256"]
        or scorer.get("schema_version") != 2
        or scorer.get("kind") != "hope_virtual_return_scorer_contract"
        or scorer.get("source") != scorer_source
        or scorer_sha != summary.get("virtual_return_scorer_contract_sha256")
        or scorer_sha != canonical_sha256(scorer_material)
    ):
        raise ContractError(f"{cell_id} summary does not reproduce the exact signed K100 contract")
    shared = schedule_runtime.get("shared_artifact")
    summary_items = schedule_runtime.get("items", [])
    summary_projection = [
        {key: item.get(key) for key in expected_items[0]} for item in summary_items
    ] if isinstance(summary_items, list) and all(isinstance(item, dict) for item in summary_items) else []
    if (
        schedule_runtime.get("sha256") != manifest["paper"]["schedule"]["semantic_sha256"]
        or schedule_runtime.get("bank_sha256") != manifest["_attestor_manifest"]["paper"]["bank"]["sha256"]
        or schedule_runtime.get("seed") != 0
        or schedule_runtime.get("size") != 100
        or schedule_runtime.get("one_question_reset") is not True
        or schedule_runtime.get("artifact_path") != manifest["paper"]["schedule"]["path"]
        or schedule_runtime.get("artifact_file_sha256") != manifest["paper"]["schedule"]["file_sha256"]
        or shared != schedule_document
        or summary_projection != expected_items
        or any(item.get("question_sequence_index") != index for index, item in enumerate(summary_items))
    ):
        raise ContractError(f"{cell_id} summary reorders or replaces the immutable schedule")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ContractError(f"{cell_id} summary must contain exactly one noise=0 result")
    result = results[0]
    result_items = result.get("exam_schedule", {}).get("items", [])
    result_projection = [
        {key: item.get(key) for key in expected_items[0]} for item in result_items
    ] if isinstance(result_items, list) and all(isinstance(item, dict) for item in result_items) else []
    if (
        result.get("noise_scale") != 0.0
        or result.get("evaluation_contract_exact") is not True
        or result.get("exam_schedule", {}).get("question_id_order") != order
        or result_projection != expected_items
        or len(result_items) != 100
    ):
        raise ContractError(f"{cell_id} result does not cover the same 100 questions")
    with attempts_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100:
        raise ContractError(f"{cell_id} attempt ledger denominator is not 100")
    raw = []
    for index, (row, expected, item) in enumerate(zip(rows, expected_items, result_items)):
        clip_name = "forehand" if expected["clip"] == 0 else "backhand"
        try:
            ledger_item = {
                "schedule_index": int(row["schedule_index"]),
                "clip": 0 if row["clip_name"] == "forehand" else 1,
                "bank_row": int(row["bank_row"]),
                "question_id": row["question_id"],
                "repeat": int(row["repeat"]),
                "hold_steps": int(row["hold_steps"]),
                "attempt_seed": int(row["attempt_seed"]),
            }
            values = {
                "eligible": _strict_csv_bool(row["eligible"], f"row {index} eligible"),
                "censored": _strict_csv_bool(row["censored"], f"row {index} censored"),
                "physical_fall": _strict_csv_bool(row["physical_fall"], f"row {index} fall"),
                "guard_reset": _strict_csv_bool(row["guard_reset"], f"row {index} reset"),
                "hit": _strict_csv_bool(row["hit"], f"row {index} hit"),
                "returned": _strict_csv_bool(row["returned"], f"row {index} returned"),
                "reached_exact": _strict_csv_bool(row["reached_exact"], f"row {index} reached"),
                "exact_composite": _strict_csv_bool(row["exact_composite"], f"row {index} composite"),
                "finalize_reason": row["finalize_reason"],
            }
            sequence = int(row["question_sequence_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"{cell_id} attempt ledger row {index} is malformed: {exc}") from exc
        if (
            ledger_item != expected
            or row["clip_name"] != clip_name
            or row.get("schedule_sha256") != manifest["paper"]["schedule"]["semantic_sha256"]
            or sequence != index
            or values["eligible"] is not True
            or values["censored"] is not False
            or not values["finalize_reason"]
            or any(item.get(key) != values[key] for key in ("eligible", "censored", "physical_fall", "guard_reset", "hit", "returned", "finalize_reason"))
        ):
            raise ContractError(f"{cell_id} attempt ledger row {index} is censored, reordered, or disagrees with summary")
        raw.append((clip_name, values))
    counts = {
        "aggregate": sum(values["returned"] for _, values in raw),
        "forehand": sum(values["returned"] for clip, values in raw if clip == "forehand"),
        "backhand": sum(values["returned"] for clip, values in raw if clip == "backhand"),
        "hits": sum(values["hit"] for _, values in raw),
        "physical_falls": sum(values["physical_fall"] for _, values in raw),
        "guard_resets": sum(values["guard_reset"] for _, values in raw),
    }
    attempts = result.get("attempts", {})
    venue = result.get("venue", {})
    for name, denominator in (("all", 100), ("forehand", 50), ("backhand", 50)):
        attempt_group = attempts if name == "all" else attempts.get("per_clip", {}).get(name, {})
        venue_group = venue.get(name, {})
        returned = counts["aggregate" if name == "all" else name]
        if (
            attempt_group.get("n_attempts") != denominator
            or venue_group.get("n_attempts") != denominator
            or venue_group.get("landed_ok") != returned
            or not math.isclose(
                _finite_rate(venue_group.get("return_success_rate_per_attempt"), f"{cell_id} venue {name}"),
                returned / denominator, rel_tol=0.0, abs_tol=1e-15,
            )
        ):
            raise ContractError(f"{cell_id} headline {name} disagrees with unconditional ledger")
    return {
        "cell_id": cell_id,
        "checkpoint_sha256": manifest["cells"][cell_id]["checkpoint"]["sha256"],
        "hard_contract_sha256": manifest["cells"][cell_id]["adjacent_hard_contract"]["sha256"],
        "report": {"path": str(report), "file_sha256": sha256_file(report)},
        "summary": {"path": str(summary_path), "file_sha256": sha256_file(summary_path)},
        "attempt_ledger": {"path": str(attempts_path), "file_sha256": sha256_file(attempts_path)},
        "schedule_semantic_sha256": manifest["paper"]["schedule"]["semantic_sha256"],
        "question_id_order_sha256": canonical_sha256(order),
        "mjcf_sha256": summary["mjcf_sha256"],
        "execution_contract_sha256": summary["execution_contract_sha256"],
        "ready_state_sha256": summary["ready_state_sha256"],
        "virtual_return_scorer_contract_sha256": summary["virtual_return_scorer_contract_sha256"],
        "signed_face_contract": RUNTIME_FACE_CONTRACT,
        "denominators": {"aggregate": 100, "forehand": 50, "backhand": 50},
        "returned_counts": counts,
        "returned_rates": {
            "aggregate": counts["aggregate"] / 100,
            "forehand": counts["forehand"] / 50,
            "backhand": counts["backhand"] / 50,
        },
    }
def _gpu_compute_pids(gpu_indices: Sequence[int]) -> dict[int, list[int]]:
    try:
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
        app_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot audit evaluation GPUs: {exc}") from exc
    uuid_to_index = {}
    for row in gpu_rows:
        index_text, uuid = [part.strip() for part in row.split(",", 1)]
        uuid_to_index[uuid] = int(index_text)
    result = {int(index): [] for index in gpu_indices}
    for row in app_rows:
        if not row.strip():
            continue
        uuid, pid_text = [part.strip() for part in row.split(",", 1)]
        index = uuid_to_index.get(uuid)
        if index in result:
            result[index].append(int(pid_text))
    return {index: sorted(pids) for index, pids in result.items()}


def _write_exclusive(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> None:
    payload = A.canonical_bytes(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
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


def _copy_exclusive(source: Path, target: Path, *, expected_size: int, expected_sha: str) -> None:
    data = source.read_bytes()
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ContractError(f"source bytes changed before independent staging: {source}")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _judge_command(
    *, judge: Path, staged_run: Path, checkpoint: str, gpu: int, schedule: str, bank: str
) -> list[str]:
    exam_extra = f"--exam-schedule-json {shlex.quote(schedule)}"
    command = [
        "bash", str(judge), str(staged_run), checkpoint,
        "--gpu", str(gpu), "--steps", "0", "--seed", "0",
        "--noise-scales", "0.0", "--hold-ref", "auto",
        "--exam-bank", bank, "--task", "HOPEPingPongVirtualBall",
        "--exam-extra", exam_extra,
    ]
    if any("allow-inexact" in part for part in command) or "--schedule-k" in command:
        raise ContractError("constructed judge command contains an inexact or re-paper escape")
    return command


def execute_pair(
    *, request_path: Path, request: dict[str, Any], manifest_path: Path,
    manifest: dict[str, Any], root_confirmation: str
) -> dict[str, Any]:
    if root_confirmation != ROOT_CONFIRMATION:
        raise ContractError("execute requires the exact simulation-only root confirmation")
    output_root = Path(request["output"]["root"])
    if A._lexists(output_root):
        raise ContractError(f"no-clobber: paired evaluation root exists: {output_root}")
    A.require_safe_output_ancestry(output_root.parent)
    request_stat, request_sha = A.validate_json_snapshot(
        request_path, {key: value for key, value in request.items() if not key.startswith("_")},
        "paired execution request snapshot",
    )
    manifest_document = {key: value for key, value in manifest.items() if not key.startswith("_")}
    manifest_stat, manifest_sha = A.validate_json_snapshot(
        manifest_path, manifest_document, "paired execution manifest snapshot"
    )
    _validate_terminal_receipts(manifest)
    attested = {
        cell_id: _validate_attestation(cell_id, request, manifest) for cell_id in CELL_ORDER
    }
    mjcf_shas = {attested[cell_id]["runtime"]["mjcf"]["sha256"] for cell_id in CELL_ORDER}
    plant_shas = {attested[cell_id]["runtime"]["plant_contract_sha256"] for cell_id in CELL_ORDER}
    if len(mjcf_shas) != 1 or len(plant_shas) != 1:
        raise ContractError("C3/D3 attestations do not share one MJCF/plant execution contract")
    A.validate_actual_paper(manifest["_attestor_manifest"])
    activation = read_json(Path(manifest["paper"]["activation"]["path"]), "actual K100 activation")
    bank_path = activation["content"].get("bank", {}).get("path")
    bank = require_absolute(bank_path, "actual K100 exam bank path")
    bank_spec = manifest["_attestor_manifest"]["paper"]["bank"]
    if not bank.is_file() or bank.is_symlink() or sha256_file(bank) != bank_spec["sha256"]:
        raise ContractError("actual activation-bound K100 bank is missing or changed")
    schedule_path = Path(manifest["paper"]["schedule"]["path"])
    schedule_document = read_json(schedule_path, "actual immutable schedule")
    for cell_id in CELL_ORDER:
        request["_attestor_requests"][cell_id]["_actual_bank_path"] = str(bank)
    for name in ("isaac_activation", "mjeval_activation"):
        spec = request["runtime"][name]
        A.validate_file_binding(
            Path(spec["path"]), size=spec["bytes"], digest=spec["file_sha256"], label=name
        )
    _, request_sha_after = A.validate_json_snapshot(
        request_path, {key: value for key, value in request.items() if not key.startswith("_")},
        "paired execution request snapshot", initial=request_stat,
    )
    _, manifest_sha_after = A.validate_json_snapshot(
        manifest_path, manifest_document, "paired execution manifest snapshot", initial=manifest_stat,
    )
    if request_sha_after != request_sha or manifest_sha_after != manifest_sha:
        raise ContractError("request or manifest changed during paired preflight")
    if A._lexists(output_root):
        raise ContractError("paired evaluation root appeared during preflight")
    A.require_safe_output_ancestry(output_root.parent)
    output_root.mkdir(parents=True, exist_ok=False)
    judge = manifest["_attestor_manifest"]["_resolved_sources"]["judge"]
    claim_content = {
        "manifest_id": MANIFEST_ID,
        "manifest_file_sha256": manifest_sha,
        "request_id": request["request_id"],
        "request_file_sha256": request_sha,
        "status": "paired_execution_claimed_not_yet_complete",
        "ordered_cells": list(CELL_ORDER),
        "checkpoint_sha256_by_cell": CHECKPOINT_SHA_BY_CELL,
        "attestation_claim_content_sha256_by_cell": {
            cell_id: request["attestations"][cell_id]["claim"]["content_sha256"]
            for cell_id in CELL_ORDER
        },
        "schedule_file_sha256": manifest["paper"]["schedule"]["file_sha256"],
        "schedule_semantic_sha256": manifest["paper"]["schedule"]["semantic_sha256"],
        "activation_file_sha256": manifest["paper"]["activation"]["file_sha256"],
        "activation_content_sha256": manifest["paper"]["activation"]["content_sha256"],
        "gpu_by_cell": request["runtime"]["gpu_by_cell"],
        "judge_source_sha256": sha256_file(judge),
        "authorization": REQUEST_AUTHORIZATION,
    }
    claim_document = A.content_document("phase1_signed_face_c3_d3_k100_pair_execution_claim", claim_content)
    _write_exclusive(output_root / "pair_execution_claim.json", claim_document)
    env = os.environ.copy()
    env.update(
        JUDGE_ISAAC_ENV=request["runtime"]["isaac_activation"]["path"],
        JUDGE_MJEVAL_ACT=request["runtime"]["mjeval_activation"]["path"],
        JUDGE_KIT_BOOT_LOCK=request["runtime"]["kit_boot_lock"],
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    results = {}
    for cell_id in CELL_ORDER:
        gpu = request["runtime"]["gpu_by_cell"][cell_id]
        snapshot = _gpu_compute_pids([gpu])
        if snapshot[gpu]:
            raise ContractError(f"{cell_id} evaluation GPU{gpu} is not empty: {snapshot[gpu]}")
        cell_root = output_root / cell_id
        staged_run = cell_root / "run" / "agibot_a3_hope_virtualball" / manifest["cells"][cell_id]["run_name"]
        params = staged_run / "params"
        params.mkdir(parents=True, exist_ok=False)
        env_spec = request["env_yamls"][cell_id]
        _copy_exclusive(
            Path(env_spec["path"]), params / "env.yaml",
            expected_size=env_spec["bytes"], expected_sha=env_spec["file_sha256"],
        )
        command = _judge_command(
            judge=judge, staged_run=staged_run,
            checkpoint=manifest["cells"][cell_id]["checkpoint"]["path"], gpu=gpu,
            schedule=str(schedule_path), bank=str(bank),
        )
        launch_claim = A.content_document(
            "phase1_signed_face_c3_d3_k100_cell_launch_claim",
            {
                "cell_id": cell_id,
                "command": command,
                "gpu_snapshot_before": {"gpu": gpu, "compute_pids": []},
                "pair_execution_claim_content_sha256": claim_document["content_sha256"],
                "attestation_claim_content_sha256": request["attestations"][cell_id]["claim"]["content_sha256"],
                "signals_to_existing_processes_allowed": False,
            },
        )
        _write_exclusive(cell_root / "launch_claim.json", launch_claim)
        log_path = cell_root / "judge.runner.log"
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb", buffering=0) as log:
            proc = subprocess.Popen(
                command, cwd=judge.parent.parent, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            running = A.content_document(
                "phase1_signed_face_c3_d3_k100_cell_running_state",
                {
                    "cell_id": cell_id, "pid": proc.pid, "pgid": proc.pid,
                    "command": command, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "signals_to_existing_processes_allowed": False,
                },
            )
            _write_exclusive(cell_root / "running.json", running)
            returncode = proc.wait()
        completed = A.content_document(
            "phase1_signed_face_c3_d3_k100_cell_exit_state",
            {
                "cell_id": cell_id, "pid": proc.pid, "pgid": proc.pid,
                "returncode": returncode,
                "status": "natural_exit_zero" if returncode == 0 else "natural_exit_nonzero",
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "log_file_sha256": sha256_file(log_path),
                "signals_sent": [],
            },
        )
        _write_exclusive(cell_root / "exit.json", completed)
        if returncode != 0:
            raise ContractError(f"{cell_id} judge failed rc={returncode}; preserved {cell_root}")
        report = _report_from_log(log_path, staged_run)
        result = validate_exam_result(
            cell_id=cell_id, report=report, staged_run=staged_run, manifest=manifest,
            attestor_request=request["_attestor_requests"][cell_id],
            schedule_document=schedule_document,
        )
        result["attestation_claim_content_sha256"] = request["attestations"][cell_id]["claim"]["content_sha256"]
        _write_exclusive(cell_root / "validated_result.json", result)
        # Re-run the generic runtime attestation checks after evaluation; judge is not allowed to
        # mutate checkpoint, hard contract, producer claim, MJCF, source, paper, or runtime.
        _validate_attestation(cell_id, request, manifest)
        results[cell_id] = result
    shared_fields = (
        "schedule_semantic_sha256", "question_id_order_sha256", "mjcf_sha256",
        "execution_contract_sha256", "ready_state_sha256",
        "virtual_return_scorer_contract_sha256", "signed_face_contract", "denominators",
    )
    for field in shared_fields:
        require_exact(results["C3"][field], results["D3"][field], f"paired result {field}")
    delta = {
        key: results["D3"]["returned_counts"][key] - results["C3"]["returned_counts"][key]
        for key in ("aggregate", "forehand", "backhand")
    }
    pair_content = {
        "manifest_id": MANIFEST_ID,
        "request_id": request["request_id"],
        "status": "paired_signed_k100_complete_decision_not_authorized",
        "pair_execution_claim_content_sha256": claim_document["content_sha256"],
        "paired_l1_receipt_sha256": PAIR_SHA,
        "ordered_cells": list(CELL_ORDER),
        "same_immutable_schedule_activation": True,
        "paper": {
            "schedule_file_sha256": manifest["paper"]["schedule"]["file_sha256"],
            "schedule_semantic_sha256": manifest["paper"]["schedule"]["semantic_sha256"],
            "question_id_order_sha256": manifest["paper"]["schedule"]["question_id_order_sha256"],
            "activation_file_sha256": manifest["paper"]["activation"]["file_sha256"],
            "activation_content_sha256": manifest["paper"]["activation"]["content_sha256"],
            "denominators": manifest["paper"]["denominator"],
        },
        "cells": results,
        "D3_minus_C3_returned_count": delta,
        "l2_training_authorized": False,
        "second_seed_authorized": False,
        "checkpoint_stop_or_promote_authorized": False,
        "formal_setting_adoption_authorized": False,
        "gate3_authorized": False,
        "deployment_authorized": False,
        "real_robot_authorized": False,
        "signals_sent": [],
        "next_action": "human reviews paired behavior and publishes a separate L2 decision contract",
    }
    pair_result = A.content_document("phase1_signed_face_c3_d3_k100_paired_behavior_result", pair_content)
    result_path = output_root / "paired_behavior_result.json"
    _write_exclusive(result_path, pair_result)
    return {
        "status": pair_content["status"],
        "result_path": str(result_path),
        "result_file_sha256": sha256_file(result_path),
        "result_content_sha256": pair_result["content_sha256"],
        "D3_minus_C3_returned_count": delta,
        "l2_training_authorized": False,
        "second_seed_authorized": False,
        "stop_or_promote_authorized": False,
        "real_robot_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--request", type=Path)
    value.add_argument("--root-confirm")
    value.add_argument("mode", choices=("static-validate", "source-plan", "plan", "execute"))
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = (
        args.manifest.resolve() if args.manifest is not None else (repo_root / MANIFEST_RELATIVE).resolve()
    )
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    if args.mode == "static-validate":
        print(json.dumps({
            "status": "source_reviewed_runtime_request_and_attestations_required",
            "manifest_id": MANIFEST_ID, "paired_l1_receipt_sha256": PAIR_SHA,
            "checkpoint_sha256_by_cell": CHECKPOINT_SHA_BY_CELL,
            "same_K100_required": True, "writes_or_launches_performed": False,
            "l2_training_authorized": False, "second_seed_authorized": False,
            "stop_or_promote_authorized": False, "real_robot_authorized": False,
        }, sort_keys=True))
        return 0
    if args.mode == "source-plan":
        print(json.dumps({
            "status": "source_plan_only_runtime_request_absent_no_launch",
            "ordered_cells": list(CELL_ORDER),
            "required_attestor_request_checkpoint_sha256": CHECKPOINT_SHA_BY_CELL,
            "required_terminal_receipt_sha256": TERMINAL_SHA_BY_CELL,
            "required_paired_l1_receipt_sha256": PAIR_SHA,
            "required_schedule_file_sha256": manifest["paper"]["schedule"]["file_sha256"],
            "required_activation_file_sha256": manifest["paper"]["activation"]["file_sha256"],
            "runtime_request_bound": False, "writes_or_launches_performed": False,
            "judge_started": False, "l2_training_authorized": False,
            "second_seed_authorized": False, "stop_or_promote_authorized": False,
            "real_robot_authorized": False,
        }, sort_keys=True))
        return 0
    if args.request is None:
        raise ContractError(f"{args.mode} requires --request")
    request_path = args.request.resolve()
    request = load_request(request_path, manifest_path, manifest, read_attestor_requests=True)
    if args.mode == "plan":
        print(json.dumps({
            "status": "exact_pair_request_valid_runtime_artifacts_not_read_no_writes",
            "request_id": request["request_id"], "ordered_cells": list(CELL_ORDER),
            "checkpoint_sha256_by_cell": CHECKPOINT_SHA_BY_CELL,
            "gpu_by_cell": request["runtime"]["gpu_by_cell"],
            "output_root": request["output"]["root"],
            "attestation_claims_required": True, "writes_or_launches_performed": False,
            "judge_started": False, "l2_training_authorized": False,
            "second_seed_authorized": False, "stop_or_promote_authorized": False,
            "real_robot_authorized": False,
        }, sort_keys=True))
        return 0
    result = execute_pair(
        request_path=request_path, request=request, manifest_path=manifest_path,
        manifest=manifest, root_confirmation=args.root_confirm or "",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[signed-face-c3-d3-k100][FATAL] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
